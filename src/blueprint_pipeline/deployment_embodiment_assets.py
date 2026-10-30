"""Fetch pinned, licensed robot source assets without inventing complete robots.

The existing native runtime consumes a complete digest-bound USD manifest. This
small preparer retains publisher source/notice bytes for the approved importer;
it does not substitute a nearby robot, fit tools or grant evaluation authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .safe_outbound_http import pinned_api_policy, request
from .immutable_directory_publication import (
    ImmutableDirectoryPublicationError,
    publish_staged_immutable_directory,
)

DATA = Path(__file__).parent / "_catalog_data" / "embodiment_asset_recipes.v1.json"
RECEIPT = "source-bundle.json"
MAX_FILES = 2500
MAX_BYTES = 512 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_RECEIPT_BYTES = 8 * 1024 * 1024
_REPOSITORY = re.compile(r"https://github.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


class AssetPreparationError(ValueError):
    """A source closure cannot be prepared faithfully and within its bounds."""


def _path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise AssetPreparationError("asset_source_path_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise AssetPreparationError("asset_source_path_invalid")
    return value


def _composition_module(recipe):
    if recipe["composition"].get("schema_version") == "iiwa14_robotiq2f85_sim_composition.v1":
        from . import deployment_embodiment_kuka_composition
        return deployment_embodiment_kuka_composition
    if recipe["composition"].get("schema_version") == "industrial_robotiq2f85_sim_composition.v1":
        from . import deployment_embodiment_industrial_composition
        return deployment_embodiment_industrial_composition
    if recipe["composition"].get("schema_version") == "ur_single_robotiq2f85_sim_composition.v1":
        from . import deployment_embodiment_ur_single_composition
        return deployment_embodiment_ur_single_composition
    if recipe["composition"].get("schema_version") == "fr3_robotiq2f85_sim_composition.v1":
        from . import deployment_embodiment_fr3_composition
        return deployment_embodiment_fr3_composition
    from . import deployment_embodiment_tool_composition
    return deployment_embodiment_tool_composition


def asset_recipes() -> dict[str, Any]:
    value = json.loads(DATA.read_text(encoding="utf-8"))
    if value.get("schema_version") != "embodiment_asset_recipes.v1" or value.get("claim_ceiling") != "source_asset_not_qualified":
        raise AssetPreparationError("asset_recipe_schema_invalid")
    identities: set[str] = set()
    for row in value["recipes"]:
        if row["embodiment_id"] in identities:
            raise AssetPreparationError("asset_recipe_duplicate_identity")
        identities.add(row["embodiment_id"])
        if row.get("composition") is not None:
            _composition_module(row).component_recipe(row)
        source = row["source"]
        if source is None:
            if row["status"] not in {"existing_runtime", "source_unavailable"}:
                raise AssetPreparationError("asset_recipe_source_missing")
            continue
        if (row["status"] != "source_available" or not _REPOSITORY.fullmatch(source["repository_url"])
                or not _REVISION.fullmatch(source["revision"])
                or source["format"] not in {"urdf", "xacro", "mjcf"}):
            raise AssetPreparationError("asset_recipe_source_invalid")
        if (type(row["arm_count"]) is not int or type(source.get("source_arm_count")) is not int
                or not 1 <= source["source_arm_count"] <= row["arm_count"] <= 2
                or not isinstance(source["include_paths"], list) or not 1 <= len(source["include_paths"]) <= 256
                or source["license_id"] not in {"MIT", "Apache-2.0", "BSD-3-Clause"}):
            raise AssetPreparationError("asset_recipe_source_invalid")
        for path in [source["entry_path"], source["license_path"], *source["include_paths"]]:
            _path(path)
        if source["license_url"] != (
                f'{source["repository_url"]}/blob/{source["revision"]}/{source["license_path"]}'):
            raise AssetPreparationError("asset_recipe_license_invalid")
        for alias, path in source["package_roots"].items():
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", alias):
                raise AssetPreparationError("asset_recipe_package_root_invalid")
            if path != ".":
                _path(path)
        if (not isinstance(source["import_arguments"], dict)
                or len(source["import_arguments"]) > 64
                or not all(isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                           and isinstance(value, str) and len(value) <= 1024
                           for key, value in source["import_arguments"].items())):
            raise AssetPreparationError("asset_recipe_import_arguments_invalid")
        if source["license_path"] not in source["include_paths"]:
            raise AssetPreparationError("asset_recipe_license_not_retained")
    value["recipes_digest"] = cross_runtime_canonical_digest(value, digest_field="recipes_digest")
    return value


def recipe_for(embodiment_id: str) -> dict[str, Any]:
    row = next((r for r in asset_recipes()["recipes"] if r["embodiment_id"] == embodiment_id), None)
    if row is None:
        raise AssetPreparationError("asset_recipe_unknown_embodiment")
    return deepcopy(row)


def _get(url: str, *, maximum: int) -> bytes:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc not in {"api.github.com", "raw.githubusercontent.com"}:
        raise AssetPreparationError("asset_source_origin_not_allowed")
    origin = "https://" + parsed.netloc
    # GitHub's REST API requires User-Agent even for public unauthenticated reads.
    headers = {"User-Agent": "Blueprint-Embodiment-Asset-Preparer/1"}
    if origin.endswith("api.github.com"):
        headers["Accept"] = "application/vnd.github+json"
    # Public publisher URLs are fixed by the recipe, not customer input. Use
    # the shared transport so cloud sessions retain their required HTTPS proxy;
    # the direct-IP transport cannot reach GitHub from those environments.
    return request(url, headers=headers, timeout_seconds=60,
                   policy=pinned_api_policy(origin, max_response_bytes=maximum),
                   max_response_bytes=maximum).body


def _selected(path: str, prefixes: list[str]) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes)


def _sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _git_blob_sha(value: bytes) -> str:
    # Git's existing object identity is SHA-1; the retained content additionally
    # carries SHA-256. This is compatibility, not a new security primitive.
    return hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value, usedforsecurity=False).hexdigest()


def _refuse_symlink_ancestors(path: Path) -> None:
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise AssetPreparationError("asset_bundle_symlink_refused")


def verify_source_bundle(directory: str | Path, *, embodiment_id: str | None = None) -> dict[str, Any]:
    from .deployment_embodiment_tool_composition import COMPOSITE_SCHEMA, verify_composite
    from .deployment_embodiment_fr3_composition import COMPOSITE_SCHEMA as FR3_SCHEMA, verify_composite as verify_fr3
    from .deployment_embodiment_ur_single_composition import COMPOSITE_SCHEMA as UR_SINGLE_SCHEMA, verify_composite as verify_ur_single
    from .deployment_embodiment_industrial_composition import COMPOSITE_SCHEMA as INDUSTRIAL_SCHEMA, verify_composite as verify_industrial
    from .deployment_embodiment_kuka_composition import COMPOSITE_SCHEMA as KUKA_SCHEMA, verify_composite as verify_kuka
    root = Path(directory)
    _refuse_symlink_ancestors(root / RECEIPT)
    if (root / RECEIPT).is_file() and (root / RECEIPT).stat().st_size <= MAX_RECEIPT_BYTES:
        value = json.loads((root / RECEIPT).read_text())
        if isinstance(value, dict) and value.get("schema_version") == KUKA_SCHEMA:
            return verify_kuka(root, embodiment_id=embodiment_id)
        if isinstance(value, dict) and value.get("schema_version") == INDUSTRIAL_SCHEMA:
            return verify_industrial(root, embodiment_id=embodiment_id)
        if isinstance(value, dict) and value.get("schema_version") == COMPOSITE_SCHEMA:
            return verify_composite(root, embodiment_id=embodiment_id)
        if isinstance(value, dict) and value.get("schema_version") == UR_SINGLE_SCHEMA:
            return verify_ur_single(root, embodiment_id=embodiment_id)
        if isinstance(value, dict) and value.get("schema_version") == FR3_SCHEMA:
            return verify_fr3(root, embodiment_id=embodiment_id)
    return _verify_source_recipe(directory, embodiment_id=embodiment_id)


def _verify_source_recipe(directory: str | Path, *, embodiment_id: str | None = None,
                          _recipe: dict | None = None) -> dict[str, Any]:
    """A retry reuses exactly retained bytes; no overwrite, new download or grant."""
    root = Path(directory)
    _refuse_symlink_ancestors(root / RECEIPT)
    receipt_path = root / RECEIPT
    if not receipt_path.is_file() or not 0 < receipt_path.stat().st_size <= MAX_RECEIPT_BYTES:
        raise AssetPreparationError("asset_bundle_receipt_missing_or_bound_exceeded")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (not isinstance(receipt, dict) or set(receipt) != {
            "schema_version", "claim_ceiling", "embodiment_id", "recipe_digest", "repository_url",
            "revision", "entry_path", "format", "source_arm_count", "files", "license",
            "composition_required", "missing", "execution_authorized", "bundle_digest",
        } or receipt.get("schema_version") != "embodiment_asset_source_bundle.v1"
            or receipt["claim_ceiling"] != "source_asset_not_qualified"
            or receipt["execution_authorized"] is not False
            or receipt.get("bundle_digest") != cross_runtime_canonical_digest(receipt, digest_field="bundle_digest")):
        raise AssetPreparationError("asset_bundle_receipt_changed")
    recipe = _recipe or recipe_for(embodiment_id or receipt["embodiment_id"])
    expected = recipe["source"]
    if (receipt["embodiment_id"] != recipe["embodiment_id"]
            or receipt["recipe_digest"] != cross_runtime_canonical_digest(recipe)
            or expected is None):
        raise AssetPreparationError("asset_bundle_recipe_changed")
    if (any(receipt[key] != expected[key] for key in (
            "repository_url", "revision", "entry_path", "format", "source_arm_count"))
            or receipt["composition_required"] != recipe["composition_required"]
            or receipt["missing"] != recipe["missing"]):
        raise AssetPreparationError("asset_bundle_source_binding_changed")
    files = receipt["files"]
    if (not isinstance(files, list) or not 0 < len(files) <= MAX_FILES
            or not isinstance(receipt["license"], dict)):
        raise AssetPreparationError("asset_bundle_file_inventory_invalid")
    source = root / "source"
    _refuse_symlink_ancestors(source)
    paths = set()
    total_bytes = 0
    for item in files:
        if (not isinstance(item, dict) or set(item) != {"path", "size_bytes", "git_blob_sha", "sha256"}
                or type(item["size_bytes"]) is not int or not 0 <= item["size_bytes"] <= MAX_FILE_BYTES
                or not isinstance(item["git_blob_sha"], str) or not _REVISION.fullmatch(item["git_blob_sha"])):
            raise AssetPreparationError("asset_bundle_file_inventory_invalid")
        name = _path(item["path"])
        if name in paths or not _selected(name, expected["include_paths"]):
            raise AssetPreparationError("asset_bundle_file_inventory_invalid")
        total_bytes += item["size_bytes"]
        if total_bytes > MAX_BYTES:
            raise AssetPreparationError("asset_source_bundle_bound_exceeded")
        path = source / name
        _refuse_symlink_ancestors(path)
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise AssetPreparationError("asset_bundle_file_changed")
        with path.open("rb") as stream:
            content = stream.read(item["size_bytes"] + 1)
        if (_sha(content) != item["sha256"] or _git_blob_sha(content) != item["git_blob_sha"]
                or len(content) != item["size_bytes"]):
            raise AssetPreparationError("asset_bundle_file_changed")
        paths.add(name)
    if expected["entry_path"] not in paths or expected["license_path"] not in paths:
        raise AssetPreparationError("asset_source_entry_or_license_missing")
    license_record = next(item for item in files if item["path"] == expected["license_path"])
    if receipt["license"] != {"identifier": expected["license_id"], "path": expected["license_path"],
                              "sha256": license_record["sha256"], "note": expected["license_note"]}:
        raise AssetPreparationError("asset_bundle_license_changed")
    retained = set()
    for path in source.rglob("*"):
        if path.is_symlink():
            raise AssetPreparationError("asset_bundle_symlink_refused")
        if path.is_file():
            retained.add(path.relative_to(source).as_posix())
            if len(retained) > MAX_FILES:
                raise AssetPreparationError("asset_source_bundle_bound_exceeded")
    if paths != retained:
        raise AssetPreparationError("asset_bundle_unrecorded_file")
    return receipt


def materialize_sources(embodiment_id: str, destination: str | Path, *, get=None) -> dict[str, Any]:
    """Download only the pinned source closure and notices, with real byte hashes.

    No repository code is executed. LFS pointers, submodules, symlinks, tree
    truncation, excessive bytes and changed retries fail closed. Large artifacts
    remain local/operator storage; no private geometry is published by this tool.
    """
    recipe = recipe_for(embodiment_id)
    if recipe.get("composition") is not None:
        return _composition_module(recipe).materialize_composite(recipe, Path(destination), get=get)
    return _materialize_source_recipe(recipe, Path(destination), get=get)


def _materialize_source_recipe(recipe: dict, destination: Path, *, get=None) -> dict[str, Any]:
    embodiment_id = recipe["embodiment_id"]
    source = recipe["source"]
    if source is None:
        raise AssetPreparationError(f'asset_source_{recipe["status"]}:{recipe["reason"]}')
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        return _verify_source_recipe(destination, embodiment_id=embodiment_id, _recipe=recipe)
    if not destination.parent.is_dir():
        raise AssetPreparationError("asset_destination_parent_invalid")
    _refuse_symlink_ancestors(destination.parent)
    fetch = get or _get
    repository = _REPOSITORY.fullmatch(source["repository_url"]).groups()
    project = "/".join(repository)
    tree_url = f'https://api.github.com/repos/{project}/git/trees/{source["revision"]}?recursive=1'
    tree = json.loads(fetch(tree_url, maximum=8 * 1024 * 1024))
    if tree.get("truncated") or not isinstance(tree.get("tree"), list):
        raise AssetPreparationError("asset_source_tree_incomplete")
    entries = []
    for item in tree["tree"]:
        path = _path(item["path"])
        if not _selected(path, source["include_paths"]) or item["type"] == "tree":
            continue
        if item["type"] != "blob" or item["mode"] not in {"100644", "100755"}:
            raise AssetPreparationError("asset_source_symlink_or_submodule_refused")
        if (type(item.get("size")) is not int or not 0 <= item["size"] <= MAX_FILE_BYTES
                or not isinstance(item.get("sha"), str) or not _REVISION.fullmatch(item["sha"])):
            raise AssetPreparationError("asset_source_file_bound_exceeded")
        entries.append(item)
    entries.sort(key=lambda item: item["path"])
    selected = {item["path"] for item in entries}
    if source["entry_path"] not in selected or source["license_path"] not in selected:
        raise AssetPreparationError("asset_source_entry_or_license_missing")
    if (not 0 < len(entries) <= MAX_FILES or len(selected) != len(entries)
            or sum(item["size"] for item in entries) > MAX_BYTES):
        raise AssetPreparationError("asset_source_bundle_bound_exceeded")
    with tempfile.TemporaryDirectory(prefix=".blueprint-assets-", dir=destination.parent) as scratch:
        staged = Path(scratch)
        (staged / "source").mkdir(parents=True)

        def retain(item):
            path = item["path"]
            url = f'https://raw.githubusercontent.com/{project}/{source["revision"]}/{quote(path, safe="/")}'
            content = fetch(url, maximum=min(MAX_FILE_BYTES, item["size"] + 1))
            if len(content) != item["size"]:
                raise AssetPreparationError("asset_source_size_changed")
            git_hash = _git_blob_sha(content)
            if git_hash != item["sha"]:
                raise AssetPreparationError("asset_source_git_blob_changed")
            if content.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise AssetPreparationError("asset_source_lfs_bytes_not_materialized")
            target = staged / "source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            return {"path": path, "size_bytes": len(content), "git_blob_sha": git_hash, "sha256": _sha(content)}

        # License bytes are retained first. This is a publisher license reference,
        # never a declaration that a particular use has been admitted.
        license_entry = next(item for item in entries if item["path"] == source["license_path"])
        license_record = retain(license_entry)
        with ThreadPoolExecutor(max_workers=4) as executor:
            files = [license_record, *executor.map(retain, [item for item in entries if item != license_entry])]
        files.sort(key=lambda item: item["path"])
        receipt = {
            "schema_version": "embodiment_asset_source_bundle.v1", "claim_ceiling": "source_asset_not_qualified",
            "embodiment_id": embodiment_id, "recipe_digest": cross_runtime_canonical_digest(recipe),
            "repository_url": source["repository_url"], "revision": source["revision"],
            "entry_path": source["entry_path"], "format": source["format"],
            "source_arm_count": source["source_arm_count"], "files": files,
            "license": {"identifier": source["license_id"], "path": source["license_path"],
                        "sha256": license_record["sha256"], "note": source["license_note"]},
            "composition_required": recipe["composition_required"], "missing": recipe["missing"],
            "execution_authorized": False,
        }
        receipt["bundle_digest"] = cross_runtime_canonical_digest(receipt, digest_field="bundle_digest")
        (staged / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        try:
            publish_staged_immutable_directory(
                staging=staged, destination=destination, manifest_name=RECEIPT,
                output_exists_code="asset_destination_appeared",
            )
        except ImmutableDirectoryPublicationError as exc:
            raise AssetPreparationError(str(exc)) from exc
    return receipt


def materialize_available_sources(destination: str | Path, *, get=None) -> dict[str, Any]:
    """Prepare every public recipe independently and retain each outcome.

    Complete bundles are verified and reused on retry. Missing public sources
    and individual preparation failures remain visible without preventing other
    robots from being prepared. This downloads sources only, never a paid job.
    """
    catalog = asset_recipes()
    root = Path(destination)
    _refuse_symlink_ancestors(root)
    root.mkdir(exist_ok=True)
    if not root.is_dir():
        raise AssetPreparationError("asset_destination_parent_invalid")
    results = []
    for recipe in catalog["recipes"]:
        identity = recipe["embodiment_id"]
        row: dict[str, Any] = {"embodiment_id": identity}
        if recipe["source"] is None:
            row.update(status=recipe["status"], reason=recipe["reason"])
        else:
            try:
                bundle = materialize_sources(identity, root / identity, get=get)
                row.update(status="source_prepared", bundle_digest=bundle["bundle_digest"],
                           file_count=len(bundle["files"]),
                           size_bytes=sum(item["size_bytes"] for item in bundle["files"]),
                           missing=bundle["missing"])
            except (ValueError, OSError, KeyError, HTTPException) as exc:
                row.update(status="preparation_failed", reason=str(exc)[:500])
        results.append(row)
    return {"schema_version": "embodiment_asset_source_batch.v1",
            "claim_ceiling": "source_asset_not_qualified", "execution_authorized": False,
            "recipes_digest": catalog["recipes_digest"], "results": results}


def source_description(recipe: dict, receipt: dict) -> dict:
    if recipe.get("composition") is None:
        return recipe["source"]
    return _composition_module(recipe).effective_description(recipe, receipt)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--embodiment", help="Exact catalog or compatibility embodiment ID")
    mode.add_argument("--all-available", action="store_true",
                      help="Prepare all public sources; report each unavailable or failed recipe")
    parser.add_argument("--destination", type=Path, help="New source bundle directory, or a verified retry")
    mode.add_argument("--list", action="store_true", help="Show exact asset availability and remaining work")
    args = parser.parse_args(argv)
    try:
        if args.list:
            result = asset_recipes()
        elif args.all_available:
            if not args.destination:
                parser.error("--all-available requires --destination")
            result = materialize_available_sources(args.destination)
        elif args.embodiment and args.destination:
            result = materialize_sources(args.embodiment, args.destination)
        elif args.embodiment:
            result = recipe_for(args.embodiment)
        else:
            parser.error("supply --list, --embodiment, or --all-available --destination")
        print(json.dumps(result, indent=2))
        if args.all_available and any(row["status"] == "preparation_failed" for row in result["results"]):
            return 1
    except (ValueError, OSError, KeyError, HTTPException) as exc:
        parser.exit(2, f"Robot asset preparation refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
