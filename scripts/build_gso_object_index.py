#!/usr/bin/env python3
"""Build the checked-in index of Google Scanned Objects on Gazebo Fuel (``website_object_library``).

Google Scanned Objects (GSO) are about 1,030 real household objects Google scanned and published on Gazebo Fuel under
the owner ``GoogleResearch``. This script lists every model through the Fuel REST API, reads each model's own
``metadata.pbtxt`` (author, copyright, license as written), downloads its mesh once to measure its bounding box in
metres and record its sha256, and records its first thumbnail's sha256. Nothing is invented: a model whose files
cannot be read is listed under ``unreadable`` with the reason, never as a record.

The index is the library sourcing's catalog: a job's object is matched to it before any paid model is made (owner
direction 2026-11-23). The license gate there admits only an allow-listed license; this script records each license
exactly as Fuel and the model's own metadata state it and decides nothing.

    python scripts/build_gso_object_index.py --output src/blueprint_pipeline/_catalog_data/object_library_gso.v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HOST = "fuel.gazebosim.org"
OWNER = "GoogleResearch"
API = f"https://{HOST}/1.0/{OWNER}/models"
SCHEMA_VERSION = "object_library_gso_index.v1"
PER_PAGE = 100
MAX_MESH_BYTES = 64 * 1024 * 1024
MAX_SMALL_BYTES = 2 * 1024 * 1024


def fetcher(timeout_seconds: float = 120.0) -> Callable[[str, int], bytes]:
    """GET one Fuel URL through the pipeline's outbound boundary, pinned to the Fuel host."""
    from blueprint_pipeline.safe_outbound_http import pinned_api_policy, request

    policy = pinned_api_policy(f"https://{HOST}", max_response_bytes=MAX_MESH_BYTES)

    def fetch(url: str, max_bytes: int) -> bytes:
        response = request(url, policy=policy, timeout_seconds=timeout_seconds, max_response_bytes=max_bytes,
                           headers={"User-Agent": "object-library-index/1"})
        if response.status != 200:
            raise ValueError(f"http_status_{response.status}")
        return response.body

    return fetch


def list_models(fetch: Callable[[str, int], bytes]) -> list[dict[str, Any]]:
    """Every model row the Fuel API lists for the owner, page by page, in the API's order."""
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = json.loads(fetch(f"{API}?per_page={PER_PAGE}&page={page}", MAX_SMALL_BYTES))
        if not isinstance(batch, list) or not batch:
            return rows
        rows.extend(row for row in batch if isinstance(row, dict))
        if len(batch) < PER_PAGE:
            return rows
        page += 1


def parse_pbtxt(text: str) -> dict[str, Any]:
    """The few fields of a GSO ``metadata.pbtxt`` the index keeps, as written: version, authors, legal, brand."""
    def field(block: str, name: str) -> str | None:
        found = re.search(rf'^\s*{name}:\s*"((?:[^"\\]|\\.)*)"', block, re.MULTILINE)
        return found.group(1).encode().decode("unicode_escape") if found else None

    def blocks(name: str) -> list[str]:
        return re.findall(rf"^{name}\s*\{{(.*?)^\}}", text, re.MULTILINE | re.DOTALL)

    version = re.search(r"^version:\s*(\d+)", text, re.MULTILINE)
    legal = blocks("legal")
    brand = next((field(row, "value") for row in blocks("annotations") if field(row, "key") == "brand"), None)
    return {"version": int(version.group(1)) if version else None,
            "authors": [name for name in (field(row, "name") for row in blocks("authors")) if name],
            "copyright": field(legal[0], "copyright") if legal else None,
            "license": field(legal[0], "license") if legal else None,
            "brand": brand}


def obj_bounds(data: bytes) -> dict[str, Any]:
    """The OBJ's axis-aligned extent (its own units), vertex and face counts. Refused when it holds no vertex."""
    low = [float("inf")] * 3
    high = [float("-inf")] * 3
    vertices = faces = 0
    for line in data.splitlines():
        if line.startswith(b"v "):
            parts = line.split()
            point = [float(value) for value in parts[1:4]]
            vertices += 1
            for axis in range(3):
                low[axis] = min(low[axis], point[axis])
                high[axis] = max(high[axis], point[axis])
        elif line.startswith(b"f "):
            faces += 1
    if not vertices or not faces:
        raise ValueError("obj_has_no_geometry")
    return {"minimum": low, "maximum": high, "vertices": vertices, "faces": faces}


def model_record(row: dict[str, Any], fetch: Callable[[str, int], bytes]) -> dict[str, Any]:
    """One model's record, read from its own files at its pinned version."""
    name = str(row["name"])
    quoted = urllib.parse.quote(name)  # a few names are not ASCII
    metadata = parse_pbtxt(fetch(f"{API}/{quoted}/tip/files/metadata.pbtxt", MAX_SMALL_BYTES).decode("utf-8"))
    version = metadata["version"]
    if not isinstance(version, int) or version < 1:
        raise ValueError("metadata_version_missing")
    files = f"{API}/{quoted}/{version}/files"
    mesh = fetch(f"{files}/meshes/model.obj", MAX_MESH_BYTES)
    bounds = obj_bounds(mesh)
    thumbnail = fetch(f"{files}/thumbnails/0.jpg", MAX_SMALL_BYTES)
    size = [round(bounds["maximum"][axis] - bounds["minimum"][axis], 4) for axis in range(3)]
    return {"name": name, "version": version, "title": str(row.get("description") or "").strip() or None,
            "categories": [str(item) for item in row.get("categories") or []],
            **({"brand": metadata["brand"]} if metadata["brand"] else {}),
            "authors": metadata["authors"], "copyright": metadata["copyright"],
            "license": {"fuel_license_name": row.get("license_name"), "fuel_license_url": row.get("license_url"),
                        "metadata_license": metadata["license"]},
            "mesh": {"sha256": hashlib.sha256(mesh).hexdigest(), "size_bytes": len(mesh),
                     "vertices": bounds["vertices"], "faces": bounds["faces"]},
            "thumbnail": {"sha256": hashlib.sha256(thumbnail).hexdigest(), "size_bytes": len(thumbnail)},
            "size_m": size}


def build_index(fetch: Callable[[str, int], bytes], *, workers: int = 8,
                cache: Path | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """The whole index: every listed model's record, or why it could not be read. ``cache`` (JSON lines) keeps each
    finished model so an interrupted run resumes."""
    done: dict[str, dict[str, Any]] = {}
    if cache is not None and cache.is_file():
        for line in cache.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if "record" in row:  # an unreadable model is tried again
                    done[row["name"]] = row
    listed = list_models(fetch)
    log(f"listed {len(listed)} models; {len(done)} cached")

    def one(row: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"name": row["name"], "record": model_record(row, fetch)}
        except Exception as exc:  # noqa: BLE001 - an unreadable model is listed with its reason
            return {"name": row["name"], "unreadable": f"{type(exc).__name__}:{str(exc)[:160]}"}

    todo = [row for row in listed if row.get("name") not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(one, todo), start=1):
            done[result["name"]] = result
            if cache is not None:
                with cache.open("a") as stream:
                    stream.write(json.dumps(result, sort_keys=True) + "\n")
            if index % 50 == 0:
                log(f"{index}/{len(todo)}")
    names = [str(row.get("name")) for row in listed]
    # Each record's rights (license as written, authors, copyright) are kept once in ``rights``, named by index.
    rights: list[dict[str, Any]] = []
    records = []
    for name in sorted(names):
        if "record" not in done.get(name, {}):
            continue
        record = dict(done[name]["record"])
        held = {key: record.pop(key) for key in ("license", "authors", "copyright")}
        if held not in rights:
            rights.append(held)
        record["rights"] = rights.index(held)
        record["mesh"] = {key: record["mesh"][key] for key in ("sha256", "size_bytes", "faces")}
        records.append(record)
    unreadable = [{"name": name, "reason": done[name]["unreadable"]} for name in sorted(names)
                  if "unreadable" in done.get(name, {})]
    return {"schema_version": SCHEMA_VERSION, "library": "google_scanned_objects",
            "library_title": "Google Scanned Objects", "publisher": "Google", "host": HOST, "owner": OWNER,
            "api": API, "mesh_url": API + "/{name}/{version}/files/meshes/model.obj",
            "thumbnail_url": API + "/{name}/{version}/files/thumbnails/0.jpg",
            "item_url": API + "/{name}/{version}/{name}.zip",
            "size_basis": "axis-aligned extent of meshes/model.obj, as stored (metres, Z up per the SDF)",
            "url_name": "each {name} is the model name percent-encoded (urllib.parse.quote)",
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "listed_count": len(listed), "record_count": len(records), "rights": rights, "records": records,
            "unreadable": unreadable}


def write_index(index: dict[str, Any], path: Path) -> None:
    """Compact JSON, one record per line, so a diff shows the records that changed."""
    head = {key: value for key, value in index.items() if key not in ("records", "unreadable", "rights")}
    head["rights"] = index["rights"]
    lines = ["{" + json.dumps(head, sort_keys=True)[1:-1] + ',"records":[']
    lines += [json.dumps(row, sort_keys=True, separators=(",", ":")) + ("," if i < len(index["records"]) - 1 else "")
              for i, row in enumerate(index["records"])]
    lines.append('],"unreadable":' + json.dumps(index["unreadable"], sort_keys=True, separators=(",", ":")) + "}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.loads(path.read_text())  # what was written reads back


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None, help="JSON lines of finished models, to resume")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    index = build_index(fetcher(), workers=args.workers, cache=args.cache,
                        log=lambda text: print(text, file=sys.stderr, flush=True))
    write_index(index, args.output)
    print(json.dumps({"records": index["record_count"], "unreadable": len(index["unreadable"]),
                      "bytes": args.output.stat().st_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
