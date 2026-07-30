"""Fail-closed inspection helpers for frozen Ctrl-World replay bundles."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping
from typing import Any

from .ctrl_world_provider_bundle import CLIP_REVISION, SVD_REVISION
from .policy_ranking_successor_cosmos import canonical_sha256


CTRL_WORLD_REPLAY_BUNDLE_ENTRIES = frozenset(
    {
        "provider_runtime/wam_provider_runtime_runner.py",
        "provider_runtime/run_wam_provider_runtime.sh",
        "provider_runtime/successor_retained_control.py",
        "provider_runtime/wam_provider_runtime_manifest.json",
        "provider_runtime/wam_rollout_input_manifest.json",
        "provider_runtime/ctrl_world_replay/canary_manifest.json",
        "provider_runtime/ctrl_world_replay/annotation.json",
        "provider_runtime/ctrl_world_replay/view_0.mp4",
        "provider_runtime/ctrl_world_replay/view_1.mp4",
        "provider_runtime/ctrl_world_replay/view_2.mp4",
        "provider_runtime/ctrl_world_source/scripts/rollout_replay_traj.py",
    }
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def inspect_ctrl_world_archive_inputs(
    archive: zipfile.ZipFile,
    *,
    manifest: Mapping[str, Any],
    names: set[str],
) -> tuple[dict[str, str], list[str]]:
    """Hash frozen Ctrl-World inputs and validate the embedded source inventory."""

    blockers: list[str] = []
    canary_manifest = json.loads(
        archive.read("provider_runtime/ctrl_world_replay/canary_manifest.json").decode("utf-8")
    )
    computed_canary_sha256 = canonical_sha256(
        {key: value for key, value in canary_manifest.items() if key != "manifest_sha256"}
    )
    if canary_manifest.get("manifest_sha256") != computed_canary_sha256:
        blockers.append("successor_ctrl_world_canary_manifest_hash_invalid")
    view_manifest = canary_manifest.get("views")
    if not isinstance(view_manifest, list):
        view_manifest = []
    source_manifest = manifest.get("source_files")
    if not isinstance(source_manifest, list):
        source_manifest = []
    for row in source_manifest:
        if not isinstance(row, Mapping):
            blockers.append("successor_ctrl_world_source_manifest_invalid")
            continue
        relative = str(row.get("relative_path") or "")
        entry = f"provider_runtime/ctrl_world_source/{relative}"
        if entry not in names:
            blockers.append("successor_ctrl_world_source_file_missing")
            continue
        if hashlib.sha256(archive.read(entry)).hexdigest() != row.get("sha256"):
            blockers.append("successor_ctrl_world_source_file_hash_invalid")
    embedded_hashes = {
        "runtime_manifest_file_sha256": hashlib.sha256(
            archive.read("provider_runtime/wam_provider_runtime_manifest.json")
        ).hexdigest(),
        "rollout_manifest_file_sha256": hashlib.sha256(
            archive.read("provider_runtime/wam_rollout_input_manifest.json")
        ).hexdigest(),
        "canary_manifest_sha256": computed_canary_sha256,
        "annotation_sha256": hashlib.sha256(
            archive.read("provider_runtime/ctrl_world_replay/annotation.json")
        ).hexdigest(),
        "view_manifest_sha256": canonical_sha256(view_manifest),
        "source_manifest_sha256": canonical_sha256(source_manifest),
        "runner_sha256": hashlib.sha256(
            archive.read("provider_runtime/wam_provider_runtime_runner.py")
        ).hexdigest(),
        "entrypoint_sha256": hashlib.sha256(
            archive.read("provider_runtime/run_wam_provider_runtime.sh")
        ).hexdigest(),
    }
    return embedded_hashes, blockers


def ctrl_world_manifest_blockers(
    manifest: Mapping[str, Any],
    rollout_manifest: Mapping[str, Any],
    *,
    source_revision: str,
    checkpoint_repository: str,
    checkpoint_revision: str,
) -> list[str]:
    """Validate Ctrl-World identity and open-loop claim boundaries."""

    blockers: list[str] = []
    expected_values = {
        "runtime": (
            "ctrl_world_public_replay_runtime",
            "successor_ctrl_world_bundle_runtime_mismatch",
        ),
        "model_name": ("Ctrl-World", "successor_ctrl_world_bundle_attribution_mismatch"),
        "ctrl_world_source_revision": (
            source_revision,
            "successor_ctrl_world_bundle_source_mismatch",
        ),
        "checkpoint_repository": (
            checkpoint_repository,
            "successor_ctrl_world_bundle_checkpoint_repository_mismatch",
        ),
        "checkpoint_revision": (
            checkpoint_revision,
            "successor_ctrl_world_bundle_checkpoint_mismatch",
        ),
    }
    blockers.extend(
        blocker
        for key, (expected, blocker) in expected_values.items()
        if manifest.get(key) != expected
    )
    models = {
        str(row.get("name") or ""): row
        for row in manifest.get("models", [])
        if isinstance(row, Mapping)
    }
    if _mapping(models.get("stable_video_diffusion")).get("revision") != SVD_REVISION:
        blockers.append("successor_ctrl_world_bundle_svd_revision_mismatch")
    if _mapping(models.get("clip")).get("revision") != CLIP_REVISION:
        blockers.append("successor_ctrl_world_bundle_clip_revision_mismatch")
    rollout_expectations = {
        "arm_id": (
            "ctrl_world_public_replay_reduced_canary",
            "successor_ctrl_world_bundle_arm_mismatch",
        ),
        "physical_future_rgb_provided_to_model": (
            False,
            "successor_ctrl_world_bundle_future_rgb_boundary_invalid",
        ),
        "physical_outcome_labels_accessed": (
            False,
            "successor_ctrl_world_bundle_label_boundary_invalid",
        ),
        "closed_loop": (False, "successor_ctrl_world_bundle_closed_loop_boundary_invalid"),
    }
    blockers.extend(
        blocker
        for key, (expected, blocker) in rollout_expectations.items()
        if rollout_manifest.get(key) != expected
    )
    canary_settings = _mapping(manifest.get("canary_settings"))
    if any(
        canary_settings.get(key) != value
        for key, value in {
            "trajectory_id": "899",
            "start_index": 8,
            "interaction_count": 1,
        }.items()
    ):
        blockers.append("successor_ctrl_world_bundle_canary_settings_mismatch")
    return blockers


__all__ = [
    "CTRL_WORLD_REPLAY_BUNDLE_ENTRIES",
    "ctrl_world_manifest_blockers",
    "inspect_ctrl_world_archive_inputs",
]
