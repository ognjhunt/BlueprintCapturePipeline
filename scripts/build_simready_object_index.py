#!/usr/bin/env python3
"""Build the checked-in index of NVIDIA SimReady packages (``website_object_library``).

SimReady Central (the public, read-only GitHub mirror ``NVIDIA-Omniverse/simready-central``) lists, for each reviewed
Hugging Face dataset, the URL of every package NVIDIA reviewed, pinned to the commit it was reviewed at
(``submissions/<namespace>/<dataset>``). This script reads only those pinned URLs, never a branch, and for each package
reads, from the package itself:

- ``com.nvidia.simready.packaging.json``: the package definition's ``license`` (as written), its id, and the sha256 of
  its bill of materials;
- ``.metadata/com.nvidia.simready.packaging.bom.json``, checked against that sha256: every file with its size and
  sha256, so a job later fetches exactly the reviewed bytes;
- the root USD (named by ``.metadata/com.nvidia.simready.root_usds.json`` when the bill does not make it plain),
  checked against the bill, and its ``SimReady_Metadata`` in the root layer's ``customLayerData``, read with ``pxr``
  (only the root layer is opened): the asset's own ``asset_license`` and ``author``, its category, mass and extents.

Both licenses are recorded as written; the license gate (``website_object_library``) admits a package only when both
are on its allow-list. A package whose files cannot be read is listed under ``unreadable`` with the reason, never as
a record. Nothing is invented.

    python scripts/build_simready_object_index.py --mirror <simready-central checkout> \\
        --output src/blueprint_pipeline/_catalog_data/object_library_simready.v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

SCHEMA_VERSION = "object_library_simready_index.v1"
HOST = "huggingface.co"
#: The reviewed datasets of objects whose dataset card states CC BY 4.0 (checked 2026-11-23). Robot and gripper
#: datasets are not job objects, and a dataset with no card license or another one is left out; each package's own two
#: licenses are still read and gated, whatever its card says.
DEFAULT_DATASETS = ("nvidia/simready-assets", "PalatialSim/palatial-nv-asset-library", "projectsim/simready-assets",
                    "M-XR/marso-simready", "ALLSIDES/simready-scanned-objects", "UCLA-VAIL/UrbanVerse-SimReady-Validated",
                    "ForgeCAD/simready-assets", "NdotLight/SimReady_Assets", "Vertiv/simready-assets")
URL = re.compile(r"^https://huggingface\.co/datasets/(?P<repo>[A-Za-z0-9._-]+/[A-Za-z0-9._-]+)/resolve/"
                 r"(?P<commit>[0-9a-f]{40})/(?P<asset>[^/]+)/com\.nvidia\.simready\.packaging\.json$")
PACKAGING = "com.nvidia.simready.packaging.json"
BOM = ".metadata/com.nvidia.simready.packaging.bom.json"
ROOTS = ".metadata/com.nvidia.simready.root_usds.json"
USD_SUFFIXES = (".usd", ".usda", ".usdc")
MAX_SMALL_BYTES = 4 * 1024 * 1024
MAX_ROOT_BYTES = 64 * 1024 * 1024
#: What a package is read for besides its metadata: v3 keeps any copyright, notice or disclaimer its root layer states
#: and any copyright notice or disclaimer in a notice or license file its bill names; v4 also keeps how such a file
#: asks to be credited (``attribution_request``), so the catalog can refuse a credit the README cannot give.
SURVEY = "notices.v4"
NOTICE_WORDS = re.compile(r"(?i)copyright|notice|disclaim|warrant|attribution|legal")
#: A file the bill names whose name reads as a notice or a license (an ATTRIBUTION.txt, a NOTICE, a LICENSE.txt). The
#: package-level ``.metadata/LICENSE.txt`` is not in the bill: for every NVIDIA package it is one file, the CC BY 4.0
#: legal code, with no notice of its own (checked 2026-11-23 through the Hugging Face tree at each pinned commit).
NOTICE_FILES = re.compile(r"(?i)^(notice|disclaimer|copyright|attribution|legal|licen[cs]e)")
#: A copyright notice line: "Copyright (c) ...", "Copyright 2020 ...", "© ..." (never the legal code's "copyright and
#: certain other rights"); its first sentence is the notice kept.
COPYRIGHT_LINE = re.compile(r"^\s*(?:Copyright\s*(?:\(c\)|\(C\)|©|\d{4})|©|\(c\)\s*\d{4})")
COPYRIGHT_SENTENCE = re.compile(r"^\s*(.+?\.)(?:\s|$)")
#: A disclaimer of warranties in a notice file; a license file's own disclaimer is the license text, not one beyond it.
DISCLAIMER_LINE = re.compile(r"(?i)disclaim|warrant")
LICENSE_FILE = re.compile(r"(?i)^licen[cs]e")
#: A file that is itself a request for how to be credited: every line of it is the request (but a contact line).
ATTRIBUTION_FILE = re.compile(r"(?i)^(attribution|credits?)\b")
#: In any other notice or license file, a line that asks for a credit.
ATTRIBUTION_LINE = re.compile(r"(?i)\b(attribution|credit)\s*:|please (credit|attribute)\b")
CONTACT_LINE = re.compile(r"(?i)^\s*(contact|e-?mail)\s*:")


def fetcher(timeout_seconds: float = 120.0) -> Callable[[str, int], bytes]:
    """GET one pinned URL through the pipeline's outbound boundary: huggingface.co, its same-origin redirects, and
    its content hosts for a file's bytes; a 429 waits and tries again."""
    from blueprint_pipeline.safe_outbound_http import OutboundHttpPolicy, request
    from blueprint_pipeline.website_library_asset import CONTENT_HOSTS

    policy = OutboundHttpPolicy(allowed_hosts=frozenset({HOST}), follow_same_origin_redirects=True,
                                redirect_hosts=CONTENT_HOSTS, max_response_bytes=MAX_ROOT_BYTES)

    def fetch(url: str, max_bytes: int) -> bytes:
        for attempt in range(12):
            try:
                response = request(url, policy=policy, timeout_seconds=timeout_seconds, max_response_bytes=max_bytes,
                                   headers={"User-Agent": "object-library-index/1"})
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 503) and attempt < 11:
                    time.sleep(min(60, 5 * (attempt + 1)))
                    continue
                raise
            if response.status != 200:
                raise ValueError(f"http_status_{response.status}")
            return response.body
        raise ValueError("rate_limited")

    return fetch


def submission_urls(path: Path) -> list[dict[str, str]]:
    """The pinned package URLs a submission file lists, parsed; a line that is not one is refused."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        found = URL.match(line)
        if found is None:
            raise ValueError(f"submission_line_not_pinned:{line[:120]}")
        rows.append({"url": line, **found.groupdict()})
    return rows


def root_from_bom(items: dict[str, dict[str, Any]]) -> str | None:
    """The root USD the bill makes plain: the one USD outside ``payloads/`` with a 256 px thumbnail beside it."""
    found = []
    for relative in items:
        path = PurePosixPath(relative)
        if (path.suffix in USD_SUFFIXES and "payloads" not in path.parts and ".thumbs" not in path.parts
                and str(path.parent / ".thumbs" / "256x256" / (path.name + ".png")) in items):
            found.append(relative)
    return found[0] if len(found) == 1 else None


def parse_extents(value: Any) -> list[float] | None:
    """``asset_extents`` as written ("(x, y, z)" or a list), as three positive numbers, else None."""
    if isinstance(value, str):
        value = [part for part in re.split(r"[\s,()\[\]]+", value) if part]
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return numbers if len(numbers) == 3 and all(n > 0 for n in numbers) else None


def root_metadata(data: bytes, suffix: str) -> dict[str, Any]:
    """``SimReady_Metadata`` of the root layer, and the layer's up axis and metres per unit. Payloads stay closed."""
    from pxr import Sdf
    with tempfile.TemporaryDirectory(prefix="simready-root-") as folder:
        path = Path(folder) / f"root{suffix}"
        path.write_bytes(data)
        layer = Sdf.Layer.OpenAsAnonymous(str(path))
        if layer is None:
            raise ValueError("root_usd_unreadable")
        layer_data = dict(layer.customLayerData)
        metadata = dict(layer_data.get("SimReady_Metadata") or {})
        root = layer.pseudoRoot
        up = str(root.GetInfo("upAxis")) if root.HasInfo("upAxis") else None
        metres = float(root.GetInfo("metersPerUnit")) if root.HasInfo("metersPerUnit") else None
        documentation = str(layer.documentation or "")
    keep = ("asset_license", "author", "asset_author", "asset_name", "category", "asset_type", "mass",
            "asset_extents", "qcode")
    # Any copyright, notice or disclaimer the package states about itself (CC BY 4.0 3(a)(1)(A)(ii) and (iv)), as
    # written, wherever the root layer holds one: SimReady_Metadata, the layer's own customLayerData, its
    # documentation. None found is recorded as none; nothing is supplied in its place.
    notices = {f"{where}.{key}": str(value) for where, data in (("SimReady_Metadata", metadata),
                                                               ("customLayerData", layer_data))
               for key, value in data.items() if NOTICE_WORDS.search(str(key))}
    if NOTICE_WORDS.search(documentation):
        notices["documentation"] = documentation[:1000]
    return {"metadata": {key: (value if isinstance(value, (str, int, float)) else str(value))
                         for key, value in metadata.items() if key in keep},
            "up_axis": up, "meters_per_unit": metres, "notices": notices}


def size_in_metres(extents: list[float] | None, up: str | None, metres: float | None) -> list[float] | None:
    """The extents in metres with Z up, when the layer states its unit and up axis; else None."""
    if extents is None or metres is None or not metres > 0 or up not in ("Y", "Z"):
        return None
    x, y, z = (value * metres for value in extents)
    return [round(v, 4) for v in ((x, y, z) if up == "Z" else (x, z, y))]


def package_record(row: dict[str, str], fetch: Callable[[str, int], bytes]) -> dict[str, Any]:
    """One package's record, read from its own files at its pinned commit."""
    base = row["url"][: -len(PACKAGING)]
    packaging = json.loads(fetch(row["url"], MAX_SMALL_BYTES))
    bom_hash = next((item.get("hash", {}).get("sha256") for item in packaging.get("metadata") or []
                     if isinstance(item, dict) and item.get("name") == PurePosixPath(BOM).name), None)
    bom_bytes = fetch(base + BOM, MAX_SMALL_BYTES)
    if not bom_hash or hashlib.sha256(bom_bytes).hexdigest() != bom_hash:
        raise ValueError("bom_digest_mismatch")
    items = {item["relative_path"]: {"size_bytes": int(item["size"]), "sha256": item["hash"]["sha256"]}
             for item in json.loads(bom_bytes)["items"]}
    root = root_from_bom(items)
    if root is None:
        entries = json.loads(fetch(base + ROOTS, MAX_SMALL_BYTES)).get("entries") or []
        root = entries[0] if len(entries) == 1 and entries[0] in items else None
    if root is None:
        raise ValueError("root_usd_not_named")
    if items[root]["size_bytes"] > MAX_ROOT_BYTES:
        raise ValueError("root_usd_too_large")
    data = fetch(base + root, MAX_ROOT_BYTES)
    if hashlib.sha256(data).hexdigest() != items[root]["sha256"]:
        raise ValueError("root_usd_digest_mismatch")
    read = root_metadata(data, PurePosixPath(root).suffix)
    meta = read["metadata"]
    thumbnail = str(PurePosixPath(root).parent / ".thumbs" / "256x256" / (PurePosixPath(root).name + ".png"))
    mass = meta.get("mass")
    # A notice file the bill names (an ATTRIBUTION.txt, a NOTICE) is read, checked against the bill, for a copyright
    # line or a disclaimer of its own; its other lines (a contact address) are not kept.
    notice_files = []
    for path in sorted(path for path in items if NOTICE_FILES.search(PurePosixPath(path).name)):
        if items[path]["size_bytes"] > MAX_SMALL_BYTES:
            continue
        text = fetch(base + path, MAX_SMALL_BYTES)
        if hashlib.sha256(text).hexdigest() != items[path]["sha256"]:
            raise ValueError("notice_file_digest_mismatch")
        lines = text.decode("utf-8", "replace").splitlines()
        license_file = bool(LICENSE_FILE.match(PurePosixPath(path).name))
        asks = ATTRIBUTION_FILE.match(PurePosixPath(path).name)
        notice_files.append({"path": path, **items[path],
                             "copyright_lines": [COPYRIGHT_SENTENCE.match(line).group(1) for line in lines
                                                 if COPYRIGHT_LINE.match(line) and COPYRIGHT_SENTENCE.match(line)],
                             "disclaimer_lines": [] if license_file else [line.strip() for line in lines
                                                                          if DISCLAIMER_LINE.search(line)],
                             # How the file asks to be credited, as written (a contact line is not a credit).
                             "attribution_lines": [line.strip() for line in lines if line.strip()
                                                   and (asks or ATTRIBUTION_LINE.search(line))
                                                   and not CONTACT_LINE.match(line)]})
    supplied = [line for row in notice_files for line in row["copyright_lines"]]
    request = list(dict.fromkeys(line for row in notice_files for line in row["attribution_lines"]))
    return {"repo": row["repo"], "commit": row["commit"], "asset": row["asset"], "root": root,
            "package_license": packaging.get("license"),
            "asset_license": meta.get("asset_license"), "author": meta.get("author") or meta.get("asset_author"),
            "asset_name": meta.get("asset_name"), "category": meta.get("category"), "asset_type": meta.get("asset_type"),
            "up_axis": read["up_axis"], "meters_per_unit": read["meters_per_unit"],
            "size_m": size_in_metres(parse_extents(meta.get("asset_extents")), read["up_axis"], read["meters_per_unit"]),
            "mass_kg": round(float(mass), 6) if isinstance(mass, (int, float)) and mass > 0 else None,
            "bom": {"sha256": bom_hash, "size_bytes": len(bom_bytes)},
            **({"thumbnail": items[thumbnail]} if thumbnail in items else {}),
            # A copyright notice the package supplies, as written (its USD first, then a notice file); null when it
            # supplies none.
            "copyright": next((value for key, value in sorted(read["notices"].items())
                               if "copyright" in key.lower()), supplied[0] if supplied else None),
            # A disclaimer of warranties the package supplies beyond the license text, as written; null otherwise.
            "disclaimer": next((value for key, value in sorted(read["notices"].items())
                                if re.search(r"(?i)disclaim|warrant", key)),
                               next((line for row in notice_files for line in row["disclaimer_lines"]), None)),
            **({"notices": read["notices"]} if read["notices"] else {}),
            **({"notice_files": notice_files} if notice_files else {}),
            # How the package asks to be credited, when a notice file asks (the catalog gives it or refuses the model).
            **({"attribution_request": request} if request else {}),
            "survey": SURVEY}


def build_index(submissions: list[Path], fetch: Callable[[str, int], bytes], *, workers: int = 6,
                cache: Path | None = None, mirror_commit: str | None = None,
                log: Callable[[str], None] = print) -> dict[str, Any]:
    """The whole index over the submission files. ``cache`` (JSON lines) keeps each finished package so an
    interrupted run resumes; an unreadable package is tried again."""
    done: dict[str, dict[str, Any]] = {}
    if cache is not None and cache.is_file():
        for line in cache.read_text().splitlines():
            if line.strip():
                value = json.loads(line)
                if "record" in value and value["record"].get("survey") == SURVEY:
                    done[value["url"]] = value
    rows = [row for path in submissions for row in submission_urls(path)]
    log(f"listed {len(rows)} packages; {len(done)} cached")

    def one(row: dict[str, str]) -> dict[str, Any]:
        try:
            return {"url": row["url"], "record": package_record(row, fetch)}
        except Exception as exc:  # noqa: BLE001 - an unreadable package is listed with its reason
            return {"url": row["url"], "unreadable": f"{type(exc).__name__}:{str(exc)[:160]}"}

    todo = [row for row in rows if row["url"] not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(one, todo), start=1):
            done[result["url"]] = result
            if cache is not None:
                with cache.open("a") as stream:
                    stream.write(json.dumps(result, sort_keys=True) + "\n")
            if index % 100 == 0:
                log(f"{index}/{len(todo)}")
    # Repeated values are kept once and named by index: each dataset at its commit, each set of rights.
    sources: list[dict[str, str]] = []
    rights: list[dict[str, Any]] = []
    records, unreadable = [], []
    for row in sorted(rows, key=lambda item: (item["repo"], item["asset"])):
        value = done.get(row["url"]) or {}
        if "record" not in value:
            unreadable.append({"url": row["url"], "reason": value.get("unreadable", "not_read")})
            continue
        record = dict(value["record"])
        # Kept compact: the size is already in metres with Z up (``size_basis``), and a name equal to the folder's
        # is said once.
        for key in ("up_axis", "meters_per_unit", "asset_type", "survey"):
            record.pop(key, None)
        if record.get("asset_name") in (None, record["asset"]):
            record.pop("asset_name", None)
        source = {"repo": record.pop("repo"), "commit": record.pop("commit")}
        held = {key: record.pop(key, None) for key in ("package_license", "asset_license", "author", "copyright",
                                                      "disclaimer")}
        for table, item in ((sources, source), (rights, held)):
            if item not in table:
                table.append(item)
        record.update(source=sources.index(source), rights=rights.index(held))
        records.append(record)
    return {"schema_version": SCHEMA_VERSION, "library": "nvidia_simready", "library_title": "NVIDIA SimReady",
            "host": HOST, "index_mirror": "https://github.com/NVIDIA-Omniverse/simready-central",
            **({"index_mirror_commit": mirror_commit} if mirror_commit else {}),
            "submission_files": [f"submissions/{path.parent.name}/{path.name}" for path in submissions],
            "item_url": "https://huggingface.co/datasets/{repo}/resolve/{commit}/{asset}/" + PACKAGING,
            "file_url": "https://huggingface.co/datasets/{repo}/resolve/{commit}/{asset}/{path}",
            "size_basis": "SimReady_Metadata asset_extents of the root layer, times its metersPerUnit, Z up",
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "listed_count": len(rows), "record_count": len(records), "sources": sources, "rights": rights,
            "records": records, "unreadable": unreadable}


#: Each part of the index stays under this, a margin under the release source manifest's leaf budget
#: (``scripts/release_source_manifest.MAX_LEAF_BYTES``, 1 MiB): no checked-in file may exceed it.
MAX_PART_BYTES = 900_000
PART_SCHEMA = "object_library_simready_part.v1"


def part_name(path: Path, number: int) -> str:
    return f"{path.stem}.part-{number:02d}.json"


def _part_text(number: int, rows: list[dict[str, Any]]) -> str:
    lines = [json.dumps({"schema_version": PART_SCHEMA, "part": number}, sort_keys=True)[:-1] + ',"records":[']
    lines += [json.dumps(row, sort_keys=True, separators=(",", ":")) + ("," if i < len(rows) - 1 else "")
              for i, row in enumerate(rows)]
    return "\n".join([*lines, "]}"]) + "\n"


def write_index(index: dict[str, Any], path: Path, *, max_part_bytes: int = MAX_PART_BYTES) -> list[dict[str, Any]]:
    """The index as a small file (``path``: every table, the unreadable packages, and each part with its sha256 and
    record count) and its records in parts beside it (``<stem>.part-NN.json``), in the index's order, each under
    ``max_part_bytes``; compact JSON, one record per line, so a diff shows the records that changed. Earlier parts are
    removed first. Returns the parts written."""
    path = Path(path)
    for stale in path.parent.glob(f"{path.stem}.part-*.json"):
        stale.unlink()
    parts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    def flush() -> None:
        text = _part_text(len(parts) + 1, rows)
        name = part_name(path, len(parts) + 1)
        (path.parent / name).write_text(text, encoding="utf-8")
        data = text.encode()
        parts.append({"file": name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
                      "record_count": len(rows)})

    for row in index["records"]:
        if rows and len(_part_text(len(parts) + 1, [*rows, row]).encode()) > max_part_bytes:
            flush()
            rows = []
        rows.append(row)
    if rows:
        flush()
    if any(part["size_bytes"] > max_part_bytes for part in parts):
        raise ValueError("index_part_over_budget")
    head = {key: value for key, value in index.items() if key not in ("records", "unreadable")}
    head.update(parts=parts, part_schema=PART_SCHEMA)
    text = json.dumps(head, sort_keys=True, separators=(",", ":"))[:-1] + ',"unreadable":' + json.dumps(
        index["unreadable"], sort_keys=True, separators=(",", ":")) + "}\n"
    path.write_text(text, encoding="utf-8")
    if len(text.encode()) > max_part_bytes:
        raise ValueError("index_file_over_budget")
    json.loads(path.read_text())
    return parts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mirror", type=Path, required=True, help="a checkout of NVIDIA-Omniverse/simready-central")
    parser.add_argument("--dataset", action="append", default=None, help="namespace/dataset (repeatable)")
    parser.add_argument("--mirror-commit", default=None, help="the mirror checkout's commit, recorded")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    files = [args.mirror / "submissions" / name for name in (args.dataset or DEFAULT_DATASETS)]
    index = build_index(files, fetcher(), workers=args.workers, cache=args.cache, mirror_commit=args.mirror_commit,
                        log=lambda text: print(text, file=sys.stderr, flush=True))
    parts = write_index(index, args.output)
    print(json.dumps({"records": index["record_count"], "unreadable": len(index["unreadable"]),
                      "index_bytes": args.output.stat().st_size,
                      "parts": [{key: part[key] for key in ("file", "size_bytes", "record_count")} for part in parts]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
