"""Sealed source-calibration and legacy retained-scene ZIP inventories."""

from __future__ import annotations

import json
from collections.abc import Mapping
from zipfile import ZipFile

from .decision_evidence_contracts import canonical_digest

MANIFEST_MEMBER = "provider_runtime/adp_retained_scene_gpu_render_manifest.json"
COMMON_ENTRIES = {
    "provider_runtime/run_adp_retained_scene_render_provider_runtime.sh",
    "provider_runtime/adp_retained_scene_render_provider_runner.mjs",
    MANIFEST_MEMBER,
    "provider_runtime/render_request.json",
    "provider_runtime/renderer/render_splat.mjs",
}
RETAINED_ENTRIES = COMMON_ENTRIES | {
    "provider_runtime/execution_authority.json",
    "provider_runtime/input/shared_deleted_source_layer.ply",
    "provider_runtime/input/shared_retained_scene.ply",
}
SOURCE_ROLES = {"images", "target_support", "scene_without_target"}
SOURCE_ENTRIES = COMMON_ENTRIES | {
    "provider_runtime/source_calibration_execution_authority.json",
    "provider_runtime/input/cameras.v1.json",
} | {f"provider_runtime/input/{role}.ply" for role in SOURCE_ROLES}


def read_render_manifest(archive: ZipFile, bundle_kind: str) -> dict:
    if bundle_kind != "adp_retained_scene_render":
        return {}
    try:
        value = json.loads(archive.read(MANIFEST_MEMBER).decode("utf-8"))
    except (KeyError, ValueError, UnicodeError):
        # The adapter separately records malformed/missing JSON members.
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def resolve_render_inventory(bundle_kind: str, manifest: Mapping, current: set) -> tuple[set, list[str]]:
    if (bundle_kind != "adp_retained_scene_render"
            or manifest.get("schema_version") != "adp009d_source_calibration_gpu_render_bundle.v1"
            or manifest.get("render_scope") != "source_calibration"):
        return current, []
    blockers = []
    layers = manifest.get("layers")
    if not isinstance(layers, Mapping) or set(layers) != SOURCE_ROLES:
        blockers.append("source_calibration_render_manifest_layers_invalid")
    if manifest.get("manifest_digest") != canonical_digest(manifest, digest_field="manifest_digest"):
        blockers.append("source_calibration_render_manifest_digest_invalid")
    return SOURCE_ENTRIES, blockers
