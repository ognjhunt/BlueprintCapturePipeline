"""Verify staged G1 shared-scene inputs before a local or container episode.

This is an offline host check. It does not start Isaac, a policy server, or SONIC.
The receipt deliberately makes no inference, scoring, or qualification claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from .native_g1_official_sonic_target_bridge import (
    PINNED_ACTION_PROVIDER_SHA256,
    require_pinned_sonic_source,
)
from .native_task_arena_runtime import validate_native_task_arena_runtime_plan


PINNED_POLICY_SERVER_SHA256 = "8bfe62960d5aa33333c6bfe5602eeffd3d479e0a0eba93f6d0f82f72b4a92f62"
INVENTORY_SCHEMA = "g1_humanoidarena_checkpoint_inventory.v1"


def _identity(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"g1_preflight_file_missing_or_symlink:{path.name}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _require_sha256(path: Path, expected: str, *, label: str) -> dict[str, Any]:
    if (
        not isinstance(expected, str)
        or len(expected) != 71
        or not expected.startswith("sha256:")
        or any(ch not in "0123456789abcdef" for ch in expected[7:])
    ):
        raise ValueError(f"g1_preflight_{label}_digest_invalid")
    digest, size = _identity(path)
    if digest != expected[7:]:
        raise ValueError(f"g1_preflight_{label}_identity_mismatch")
    return {"sha256": expected, "size_bytes": size}


def _candidate(inventory: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    rows = inventory.get("candidates")
    if inventory.get("schema_version") != INVENTORY_SCHEMA or not isinstance(rows, list):
        raise ValueError("g1_preflight_inventory_invalid")
    matches = [row for row in rows if isinstance(row, dict) and row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError("g1_preflight_candidate_invalid")
    candidate = matches[0]
    files = candidate.get("files")
    folder = PurePosixPath(str(candidate.get("subdirectory") or ""))
    if (
        not isinstance(files, list) or not files or not folder.parts
        or folder.is_absolute() or ".." in folder.parts
        or candidate.get("policy_role") not in {"manipulation", "movement_navigation"}
        or candidate.get("action_interface") != "humanoidarena_semantic_v3"
        or candidate.get("input_image_shape_hwc") != [480, 640, 3]
        or candidate.get("input_state_width") != 64
        or candidate.get("output_action_width") != 40
    ):
        raise ValueError("g1_preflight_candidate_contract_invalid")
    inventory_digest = "sha256:" + hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if candidate.get("inventory_digest") != inventory_digest:
        raise ValueError("g1_preflight_candidate_inventory_digest_invalid")
    return candidate


def preflight_g1_shared_scene_run(
    *,
    scene_plan_path: Path,
    bundle_root: Path,
    inventory_path: Path,
    candidate_id: str,
    checkpoint_root: Path,
    policy_server_source: Path,
    sonic_provider_source: Path,
    sonic_encoder: Path,
    sonic_encoder_sha256: str,
    sonic_decoder: Path,
    sonic_decoder_sha256: str,
) -> dict[str, Any]:
    """Return verified input identities; reject absent or changed bytes."""

    # A missing pxr import would skip the articulation check inside the shared
    # validator. This preflight requires it to make the receipt unambiguous.
    try:
        from pxr import Usd  # noqa: F401
    except ImportError as exc:
        raise ValueError("g1_preflight_pxr_unavailable") from exc

    plan = json.loads(scene_plan_path.read_text(encoding="utf-8"))
    validate_native_task_arena_runtime_plan(plan, bundle_root=bundle_root)
    if plan.get("robot", {}).get("robot_id") != "unitree_g1":
        raise ValueError("g1_preflight_scene_robot_invalid")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    candidate = _candidate(inventory, candidate_id)
    folder = PurePosixPath(candidate["subdirectory"])
    verified = []
    seen: set[str] = set()
    for row in candidate["files"]:
        if not isinstance(row, dict):
            raise ValueError("g1_preflight_file_inventory_invalid")
        relative = PurePosixPath(str(row.get("path") or ""))
        size = row.get("size_bytes")
        if (
            not relative.parts or relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() in seen
            or isinstance(size, bool) or not isinstance(size, int) or size <= 0
        ):
            raise ValueError("g1_preflight_file_inventory_invalid")
        seen.add(relative.as_posix())
        path = checkpoint_root.joinpath(*folder.parts, *relative.parts)
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent != Path("/")):
            raise ValueError("g1_preflight_checkpoint_symlink_forbidden")
        identity = _require_sha256(path, "sha256:" + str(row.get("sha256")), label="checkpoint")
        if identity["size_bytes"] != size:
            raise ValueError("g1_preflight_checkpoint_size_mismatch")
        verified.append({"relative_path": (folder / relative).as_posix(), **identity})

    server = _require_sha256(
        policy_server_source, "sha256:" + PINNED_POLICY_SERVER_SHA256,
        label="policy_server_source",
    )
    require_pinned_sonic_source(sonic_provider_source)
    sonic_source = _require_sha256(
        sonic_provider_source,
        "sha256:" + PINNED_ACTION_PROVIDER_SHA256,
        label="sonic_provider_source",
    )
    encoder = _require_sha256(sonic_encoder, sonic_encoder_sha256, label="sonic_encoder")
    decoder = _require_sha256(sonic_decoder, sonic_decoder_sha256, label="sonic_decoder")
    return {
        "schema_version": "native_g1_shared_scene_run_preflight.v1",
        "status": "staged_inputs_verified",
        "scene_plan_digest": plan["plan_digest"],
        "robot_id": "unitree_g1",
        "candidate_id": candidate_id,
        "policy_role": candidate["policy_role"],
        "candidate_inventory_digest": candidate["inventory_digest"],
        "inventory_file_sha256": "sha256:" + _identity(inventory_path)[0],
        "checkpoint_files": verified,
        "policy_server_source": server,
        "sonic_provider_source": sonic_source,
        "sonic_encoder": encoder,
        "sonic_decoder": decoder,
        "server_process_verified": False,
        "sonic_process_verified": False,
        "episode_executed": False,
        "task_scored": False,
        "evidence_level": "development_only",
    }
