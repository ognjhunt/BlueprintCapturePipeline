"""Content-addressed cache for upstream-authored ParticleField runtime assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .particlefield_usd import (
    UPSTREAM_GSPLAT_CONVERTER_DISTRIBUTION,
    UPSTREAM_GSPLAT_CONVERTER_REVISION,
    UPSTREAM_GSPLAT_CONVERTER_VERSION,
)


CACHE_SCHEMA_VERSION = "particlefield_runtime_asset_cache.v1"
DEFAULT_CACHE_ROOT = Path(
    os.environ.get(
        "BLUEPRINT_PARTICLEFIELD_RUNTIME_ASSET_CACHE_ROOT",
        "/var/lib/blueprint/task-evaluation-inputs/particlefield-runtime-assets",
    )
)


def _identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _safe_member(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or ".." in pure.parts:
        raise ValueError("particlefield_runtime_cache_member_invalid")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError("particlefield_runtime_cache_member_invalid")
    return path


def _validate_particlefield_contract(path: Path) -> None:
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    fields = (
        [prim for prim in stage.Traverse() if prim.GetTypeName() == "ParticleField3DGaussianSplat"]
        if stage and stage.GetDefaultPrim()
        else []
    )
    if len(fields) != 1:
        raise ValueError("particlefield_runtime_cache_asset_invalid")
    field = fields[0]
    if (
        field.GetRelationship("material:binding").GetTargets()
        or field.GetAttribute("projectionModeHint").HasAuthoredValueOpinion()
        or field.GetAttribute("sortingModeHint").HasAuthoredValueOpinion()
    ):
        raise ValueError("particlefield_runtime_cache_asset_nonstandard")


def cache_entry_root(source_digest: str, *, cache_root: str | Path = DEFAULT_CACHE_ROOT) -> Path:
    if not source_digest.startswith("sha256:") or len(source_digest) != 71:
        raise ValueError("particlefield_runtime_cache_source_digest_invalid")
    return Path(cache_root).expanduser().resolve() / source_digest.removeprefix("sha256:")


def publish_particlefield_runtime_asset(
    *,
    source_digest: str,
    particlefield_path: str | Path,
    authoring_receipt_path: str | Path,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
) -> dict[str, Any]:
    """Publish one immutable upstream artifact; never overwrite an entry."""

    source = Path(particlefield_path).expanduser().resolve()
    receipt_source = Path(authoring_receipt_path).expanduser().resolve()
    asset_identity = _identity(source)
    _validate_particlefield_contract(source)
    receipt = json.loads(receipt_source.read_text(encoding="utf-8"))
    if (
        source.is_symlink()
        or receipt_source.is_symlink()
        or receipt.get("status") != "completed"
        or receipt.get("source_sha256") != source_digest
        or receipt.get("output_sha256") != asset_identity[0]
        or receipt.get("output_bytes") != asset_identity[1]
        or receipt.get("particlefield_authoring_implementation")
        != "nvidia_usd_convert_gsplat"
        or receipt.get("particlefield_emissive_material_binding_authored") is not False
        or receipt.get("particlefield_custom_render_hints_authored") is not False
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        raise ValueError("particlefield_runtime_cache_source_invalid")
    root = cache_entry_root(source_digest, cache_root=cache_root)
    if root.exists() or root.is_symlink():
        raise ValueError("particlefield_runtime_cache_entry_exists")
    root.mkdir(parents=True, mode=0o750)
    asset = root / "scene_appearance.usdc"
    authoring = root / "particlefield_authoring_receipt.v1.json"
    shutil.copyfile(source, asset)
    shutil.copyfile(receipt_source, authoring)
    asset.chmod(0o440)
    authoring.chmod(0o440)
    authoring_identity = _identity(authoring)
    manifest: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_configured_appearance_digest": source_digest,
        "particlefield": {
            "relative_path": asset.name,
            "digest": asset_identity[0],
            "size_bytes": asset_identity[1],
        },
        "authoring_receipt": {
            "relative_path": authoring.name,
            "digest": authoring_identity[0],
            "size_bytes": authoring_identity[1],
        },
        "upstream_converter": {
            "distribution": UPSTREAM_GSPLAT_CONVERTER_DISTRIBUTION,
            "version": UPSTREAM_GSPLAT_CONVERTER_VERSION,
            "source_revision": UPSTREAM_GSPLAT_CONVERTER_REVISION,
        },
        "immutable": True,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    manifest_path = root / "particlefield_runtime_asset_cache.v1.json"
    write_json(manifest_path, manifest)
    manifest_path.chmod(0o440)
    return {**manifest, "root": str(root), "manifest_path": str(manifest_path)}


def materialize_cached_particlefield(
    *,
    source_digest: str,
    output_root: str | Path,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
) -> dict[str, Any] | None:
    """Verify and hardlink one cached official field into an episode compile."""

    root = cache_entry_root(source_digest, cache_root=cache_root)
    manifest_path = root / "particlefield_runtime_asset_cache.v1.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != CACHE_SCHEMA_VERSION
        or manifest.get("source_configured_appearance_digest") != source_digest
        or manifest.get("immutable") is not True
        or manifest.get("upstream_converter")
        != {
            "distribution": UPSTREAM_GSPLAT_CONVERTER_DISTRIBUTION,
            "version": UPSTREAM_GSPLAT_CONVERTER_VERSION,
            "source_revision": UPSTREAM_GSPLAT_CONVERTER_REVISION,
        }
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
    ):
        raise ValueError("particlefield_runtime_cache_manifest_invalid")
    asset_record = manifest.get("particlefield")
    receipt_record = manifest.get("authoring_receipt")
    if not isinstance(asset_record, Mapping) or not isinstance(receipt_record, Mapping):
        raise ValueError("particlefield_runtime_cache_manifest_invalid")
    asset = _safe_member(root, str(asset_record.get("relative_path") or ""))
    receipt_path = _safe_member(root, str(receipt_record.get("relative_path") or ""))
    if _identity(asset) != (asset_record.get("digest"), asset_record.get("size_bytes")):
        raise ValueError("particlefield_runtime_cache_asset_invalid")
    _validate_particlefield_contract(asset)
    if _identity(receipt_path) != (
        receipt_record.get("digest"),
        receipt_record.get("size_bytes"),
    ):
        raise ValueError("particlefield_runtime_cache_receipt_invalid")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("source_sha256") != source_digest
        or receipt.get("output_sha256") != asset_record.get("digest")
        or receipt.get("output_bytes") != asset_record.get("size_bytes")
        or receipt.get("particlefield_authoring_implementation")
        != "nvidia_usd_convert_gsplat"
        or receipt.get("particlefield_emissive_material_binding_authored") is not False
        or receipt.get("particlefield_custom_render_hints_authored") is not False
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        raise ValueError("particlefield_runtime_cache_receipt_invalid")
    destination_root = Path(output_root).resolve()
    destination_root.mkdir(parents=True, mode=0o750)
    destination = destination_root / "scene_appearance.usdc"
    try:
        os.link(asset, destination)
    except OSError:
        shutil.copyfile(asset, destination)
    destination.chmod(0o440)
    derived_receipt = dict(receipt)
    derived_receipt.update(
        output=str(destination),
        cache_manifest_digest=manifest["manifest_digest"],
        cache_source_digest=source_digest,
        cache_reused=True,
    )
    derived_receipt["receipt_digest"] = canonical_digest(
        derived_receipt, digest_field="receipt_digest"
    )
    derived_receipt_path = destination_root / "particlefield_authoring_receipt.v1.json"
    write_json(derived_receipt_path, derived_receipt)
    derived_receipt_path.chmod(0o440)
    return {
        "asset_path": str(destination),
        "authoring_receipt_path": str(derived_receipt_path),
        "authoring_receipt": derived_receipt,
        "cache_manifest_digest": manifest["manifest_digest"],
    }


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_CACHE_ROOT",
    "cache_entry_root",
    "materialize_cached_particlefield",
    "publish_particlefield_runtime_asset",
]
