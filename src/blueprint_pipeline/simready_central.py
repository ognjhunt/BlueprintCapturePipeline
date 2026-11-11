"""Immutable SimReady Central selection and lossless package intake.

Intake verifies bytes, attribution and dependency closure. Producer test reports
remain producer evidence; only the existing native qualification lane can admit
an asset for a task. No package conversion or variant selection occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Callable, Mapping
from urllib.parse import quote, urlparse
from urllib.request import urlopen

from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "simready_central_intake.v1"
CATALOG = "https://raw.githubusercontent.com/NVIDIA-Omniverse/simready-central"
MANIFEST = "com.nvidia.simready.packaging.json"
BOM = ".metadata/com.nvidia.simready.packaging.bom.json"
ROOTS = ".metadata/com.nvidia.simready.root_usds.json"
SHA = re.compile(r"[0-9a-f]{40}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
Fetcher = Callable[[str, int], bytes]


class SimReadyCentralError(ValueError):
    """An intake predicate failed before qualification."""


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path(value: str) -> str:
    path = PurePosixPath(value)
    if (value in {"", ".", ".."} or path.is_absolute() or ".." in path.parts
            or str(path) != value or "\\" in value or ":" in value
            or any(ord(char) < 32 for char in value)):
        raise SimReadyCentralError(f"unsafe_package_path:{value}")
    return value


def _fetch(url: str, limit: int) -> bytes:
    with urlopen(url, timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise SimReadyCentralError("download_size_limit_exceeded")
    return data


def resolve_selection(*, catalog_revision: str, submission: str, asset: str,
                      fetch: Fetcher = _fetch) -> dict[str, Any]:
    """Select exactly one immutable HF package URL in a pinned catalog entry."""
    if not SHA.fullmatch(catalog_revision):
        raise SimReadyCentralError("catalog_revision_must_be_commit_sha")
    _path(submission)
    _path(asset)
    url = f"{CATALOG}/{catalog_revision}/submissions/{quote(submission, safe='/')}"
    index = fetch(url, 8 * 1024 * 1024)
    matches = []
    for line in index.decode("utf-8").splitlines():
        parsed = urlparse(line.strip())
        parts = parsed.path.strip("/").split("/")
        if (parsed.scheme != "https" or parsed.netloc != "huggingface.co"
                or parsed.query or parsed.fragment or len(parts) < 7
                or parts[0] != "datasets" or parts[3] != "resolve"):
            continue
        package_path = "/".join(parts[5:-1])
        if package_path != asset or parts[-1] != MANIFEST:
            continue
        if not SHA.fullmatch(parts[4]):
            raise SimReadyCentralError("dataset_revision_must_be_commit_sha")
        matches.append({"manifest_url": line.strip(), "dataset": "/".join(parts[1:3]),
                        "dataset_revision": parts[4], "package_path": _path(package_path)})
    if len(matches) != 1:
        raise SimReadyCentralError(f"catalog_selection_not_unique:{len(matches)}")
    return {"catalog_revision": catalog_revision, "catalog_submission": submission,
            "catalog_url": url, "catalog_sha256": _hash(index), **matches[0]}


def normalize_producer_results(payload: Any, *, source: str) -> list[dict[str, Any]]:
    """Preserve false/null/missing and source pointers in nested producer reports."""
    rows: list[dict[str, Any]] = []

    def visit(value: Any, pointer: str, inherited: dict[str, Any]) -> None:
        if isinstance(value, Mapping):
            context = {**inherited, **{key: value[key] for key in
                       ("timestamp", "profile", "profile_version", "asset") if key in value}}
            if "passed" in value or "success" in value:
                key = "passed" if "passed" in value else "success"
                outcome = value[key]
                rows.append({"source": source, "pointer": pointer, **context,
                             "result_field": key, "raw_result": outcome,
                             "status": "passed" if outcome is True else
                             "failed" if outcome is False else "unknown",
                             "producer_evidence_only": True,
                             "details": dict(value)})
            for key, child in value.items():
                visit(child, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"), context)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{pointer}/{index}", inherited)

    visit(payload, "", {})
    return rows


def verify_intake(receipt_path: str | Path) -> dict[str, Any]:
    """Reopen all original bytes before static validation or provider packaging."""
    receipt_path = Path(receipt_path).resolve()
    value = json.loads(receipt_path.read_text())
    if (value.get("schema_version") != SCHEMA_VERSION
            or value.get("receipt_digest") != canonical_digest(value, digest_field="receipt_digest")
            or value.get("status") != "intake_verified_pending_qualification"):
        raise SimReadyCentralError("intake_receipt_invalid")
    directory = _path(value["package_directory"])
    if directory != "packages/" + _path(value["selection"]["package_path"]):
        raise SimReadyCentralError("package_directory_identity_mismatch")
    root = receipt_path.parent / directory
    expected = set()
    for row in value["files"]:
        relative = _path(row["relative_path"])
        path = root / relative
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or not path.is_file() or path.stat().st_size != row["size_bytes"]
                or _hash(path.read_bytes()) != row["sha256"]):
            raise SimReadyCentralError(f"package_file_changed:{relative}")
        expected.add(relative)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != expected or any(p.is_symlink() for p in root.rglob("*")):
        raise SimReadyCentralError("package_inventory_changed")
    for name, digest in [("catalog.txt", value["selection"]["catalog_sha256"]),
                         ("source_tree.json", value["metadata_tree_sha256"])]:
        path = receipt_path.parent / name
        if path.is_symlink() or not path.is_file() or _hash(path.read_bytes()) != digest:
            raise SimReadyCentralError("package_source_evidence_changed")
    return value


def inspect_usd_dependencies(package: Path, files: list[dict[str, Any]]) -> dict[str, Any]:
    """Inspect every layer, including inactive variants, before composing a stage."""
    from pxr import UsdUtils

    root = package.resolve()
    known = {row["relative_path"] for row in files}
    dependencies = []
    blockers = []
    for relative in sorted(known):
        if Path(relative).suffix.lower() not in {".usd", ".usda", ".usdc"}:
            continue
        layer = root / relative
        try:
            groups = UsdUtils.ExtractExternalReferences(str(layer))
        except Exception as exc:
            raise SimReadyCentralError(f"usd_layer_unreadable:{relative}") from exc
        for reference in sorted({str(ref) for group in groups for ref in group}):
            target = (layer.parent / reference).resolve()
            safe = not urlparse(reference).scheme and not Path(reference).is_absolute()
            safe = safe and target.is_relative_to(root)
            target_relative = target.relative_to(root).as_posix() if safe else None
            found = safe and target_relative in known and target.is_file()
            dependencies.append({"layer": relative, "reference": reference,
                                 "resolved_relative_path": target_relative,
                                 "retained": bool(found)})
            if not found:
                blockers.append(f"usd_dependency_unretained:{relative}:{reference}")
    return {"dependencies": dependencies, "blockers": sorted(set(blockers)),
            "all_usd_layers_inspected": True,
            "native_material_runtime_dependencies_qualified": False}


def intake_package(*, catalog_revision: str, submission: str, asset: str,
                   output_root: str | Path, accepted_license: str,
                   max_bytes: int = 512 * 1024 * 1024,
                   fetch: Fetcher = _fetch) -> dict[str, Any]:
    selection = resolve_selection(catalog_revision=catalog_revision, submission=submission,
                                  asset=asset, fetch=fetch)
    destination = Path(output_root).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise SimReadyCentralError("intake_output_exists")
    base = selection["manifest_url"].rsplit("/", 1)[0] + "/"
    manifest_bytes = fetch(base + MANIFEST, 1024 * 1024)
    manifest = json.loads(manifest_bytes)
    if manifest.get("format_version") != "1.0" or not manifest.get("package_id"):
        raise SimReadyCentralError("package_manifest_unsupported")
    if not accepted_license or manifest.get("license") != accepted_license:
        raise SimReadyCentralError("package_license_not_accepted")
    tree_url = (f"https://huggingface.co/api/datasets/{selection['dataset']}/tree/"
                f"{selection['dataset_revision']}/{quote(asset, safe='/')}?recursive=true&limit=1000")
    tree_bytes = fetch(tree_url, 16 * 1024 * 1024)
    tree = json.loads(tree_bytes)
    # The endpoint caps pages at 1000 entries. Never treat a truncated page as a package.
    if not isinstance(tree, list) or len(tree) >= 1000:
        raise SimReadyCentralError("package_tree_requires_pagination")
    inventory: dict[str, dict[str, Any]] = {}
    casefolded = set()
    for row in tree:
        if row.get("type") == "directory":
            continue
        if row.get("type") != "file" or not row["path"].startswith(asset + "/"):
            raise SimReadyCentralError("package_tree_invalid")
        relative = _path(row["path"][len(asset) + 1:])
        if relative.casefold() in casefolded:
            raise SimReadyCentralError("package_duplicate_path")
        casefolded.add(relative.casefold())
        inventory[relative] = row
    total = sum(row["size"] for row in inventory.values())
    if not 0 < total <= max_bytes:
        raise SimReadyCentralError("package_size_limit_exceeded")
    if shutil.disk_usage(destination.parent).free < total + 128 * 1024 * 1024:
        raise SimReadyCentralError("package_disk_headroom_insufficient")
    bom_bytes = fetch(base + BOM, 8 * 1024 * 1024)
    declarations = [row for row in manifest.get("metadata", []) if row.get("name") == Path(BOM).name]
    if len(declarations) != 1 or declarations[0].get("hash", {}).get("sha256") != _hash(bom_bytes):
        raise SimReadyCentralError("package_bom_hash_mismatch")
    bom = json.loads(bom_bytes)
    declared: dict[str, dict[str, Any]] = {}
    for row in bom.get("items", []):
        relative = _path(row["relative_path"])
        digest = row.get("hash", {}).get("sha256", "")
        if relative in declared or not HASH.fullmatch(digest):
            raise SimReadyCentralError("package_bom_invalid")
        if relative not in inventory or row["size"] != inventory[relative]["size"]:
            raise SimReadyCentralError(f"package_bom_inventory_mismatch:{relative}")
        declared[relative] = row
    if not declared or any(p not in declared and p != MANIFEST and not p.startswith(".metadata/")
                           for p in inventory):
        raise SimReadyCentralError("package_content_unlisted_in_bom")
    if MANIFEST not in inventory or BOM not in inventory or ROOTS not in inventory:
        raise SimReadyCentralError("package_required_metadata_missing")
    package_directory = "packages/" + selection["package_path"]
    package = destination / package_directory
    package.mkdir(parents=True)
    files = []
    producer_results = []
    for relative, entry in sorted(inventory.items()):
        cached = {MANIFEST: manifest_bytes, BOM: bom_bytes}
        data = cached[relative] if relative in cached else fetch(base + quote(relative, safe='/'), entry["size"])
        sha = _hash(data)
        if len(data) != entry["size"]:
            raise SimReadyCentralError(f"package_size_mismatch:{relative}")
        if relative in declared and sha != declared[relative]["hash"]["sha256"]:
            raise SimReadyCentralError(f"package_content_hash_mismatch:{relative}")
        # HF's Git tree binds non-LFS metadata; LFS files bind their SHA256.
        if entry.get("lfs"):
            valid = sha == entry["lfs"]["oid"]
        else:
            git_oid = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            valid = git_oid == entry["oid"]
        if not valid:
            raise SimReadyCentralError(f"package_revision_hash_mismatch:{relative}")
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files.append({"relative_path": relative, "size_bytes": len(data), "sha256": sha,
                      "source_url": base + quote(relative, safe='/'),
                      "bom_verified": relative in declared})
        if relative.startswith(".metadata/") and relative.endswith(".json"):
            producer_results.extend(normalize_producer_results(json.loads(data), source=relative))
    roots = json.loads((package / ROOTS).read_text())["entries"]
    if not roots or any(_path(path) not in declared for path in roots):
        raise SimReadyCentralError("package_root_usd_invalid")
    license_files = [row for row in files if "license" in row["relative_path"].lower()]
    if not license_files:
        raise SimReadyCentralError("package_license_text_missing")
    dependency_audit = inspect_usd_dependencies(package, files)
    # Embedded metadata is also retained and normalized, without executing producer code.
    from pxr import Sdf
    for relative in roots:
        layer = Sdf.Layer.FindOrOpen(str(package / relative))
        if layer is None:
            raise SimReadyCentralError(f"usd_layer_unreadable:{relative}")
        producer_results.extend(normalize_producer_results(dict(layer.customLayerData), source=relative))
        def visit_prim(path: Any) -> None:
            obj = layer.GetObjectAtPath(path)
            if obj and obj.HasInfo("customData"):
                producer_results.extend(normalize_producer_results(
                    dict(obj.GetInfo("customData")), source=f"{relative}#{path}"))
        layer.Traverse("/", visit_prim)
    receipt = {"schema_version": SCHEMA_VERSION,
               "status": "intake_verified_pending_qualification", "development_only": True,
               "selection": selection, "package_id": manifest["package_id"],
               "package_directory": package_directory,
               "license": accepted_license, "license_files": license_files,
               "publisher_content_hash": manifest.get("content_hash"),
               "publisher_package_hash": manifest.get("package_hash"),
               "publisher_aggregate_hashes_independently_verified": False,
               "metadata_tree_sha256": _hash(tree_bytes), "files": files,
               "root_usds": roots, "producer_results": producer_results,
               "producer_reports_present": bool(producer_results),
               "dependency_audit": dependency_audit,
               "native_simulator_import_qualified": False,
               "task_qualified": False, "physical_equivalence_claimed": False,
               "transformations_performed": [], "receipt_digest": ""}
    if dependency_audit["blockers"]:
        receipt["status"] = "blocked_dependency_closure"
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    retained_catalog = fetch(selection["catalog_url"], 8 * 1024 * 1024)
    if _hash(retained_catalog) != selection["catalog_sha256"]:
        raise SimReadyCentralError("catalog_changed_during_intake")
    (destination / "catalog.txt").write_bytes(retained_catalog)
    (destination / "source_tree.json").write_bytes(tree_bytes)
    (destination / "intake.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-revision", required=True)
    parser.add_argument("--submission", default="nvidia/simready-assets")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--accepted-license", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-bytes", type=int, default=512 * 1024 * 1024)
    args = parser.parse_args(argv)
    receipt = intake_package(**vars(args))
    print(json.dumps({"status": receipt["status"], "package_id": receipt["package_id"],
                      "files": len(receipt["files"]), "receipt_digest": receipt["receipt_digest"],
                      "blockers": receipt["dependency_audit"]["blockers"]}, sort_keys=True))
    return 0 if receipt["status"] == "intake_verified_pending_qualification" else 2


if __name__ == "__main__":
    raise SystemExit(main())
