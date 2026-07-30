"""Public, revision-pinned staging for the Ctrl-World diagnostic runtime."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .ctrl_world_joint_position_reference_wam import MODEL_FREEZE
from .policy_ranking_thesis import canonical_sha256, file_sha256


STAGE_SCHEMA_VERSION = "blueprint_ctrl_world_joint_position_asset_stage.v1"
MODEL_NAMES = ("ctrl_world", "stable_video_diffusion", "clip")
IDENTITY_NAME = ".blueprint_snapshot_identity.json"


def _snapshot_download(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return str(snapshot_download(**kwargs))


def _model_spec(name: str) -> dict[str, Any]:
    if name == "ctrl_world":
        freeze = MODEL_FREEZE["ctrl_world_checkpoint"]
        required = [
            {
                "relative_path": freeze["file"],
                "size_bytes": freeze["size_bytes"],
                "sha256": freeze["sha256"],
            }
        ]
    else:
        freeze = MODEL_FREEZE[name]
        required = freeze.get("required_files")
    if not isinstance(required, list) or not required:
        raise ValueError(f"ctrl_world_asset_stage_required_files_missing:{name}")
    return {
        "name": name,
        "repository": freeze["repository"],
        "revision": freeze["revision"],
        "required_files": required,
    }


def _safe_file(root: Path, relative_value: Any, *, reason: str) -> Path:
    relative = Path(str(relative_value or ""))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(reason)
    candidate = root.joinpath(*relative.parts)
    current = candidate
    while current != root:
        if current.is_symlink():
            raise ValueError(reason)
        current = current.parent
    path = candidate.resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError(reason)
    return path


def _identity(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("ctrl_world_asset_stage_identity_invalid")
    return dict(value)


def _reject_existing_symlink_components(path: Path, *, reason: str) -> None:
    current = path
    while True:
        if current.exists() and current.is_symlink():
            raise ValueError(reason)
        if current == current.parent:
            return
        current = current.parent


def stage_ctrl_world_runtime_assets(
    *,
    model_root: str | Path,
    output_dir: str | Path,
    downloader: Callable[..., str] = _snapshot_download,
) -> dict[str, Any]:
    """Stage exact public snapshots before any candidate-policy initialization."""

    root_candidate = Path(model_root).expanduser()
    _reject_existing_symlink_components(root_candidate, reason="ctrl_world_asset_stage_root_unsafe")
    root = root_candidate.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("ctrl_world_asset_stage_root_unsafe")
    snapshots: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    for name in MODEL_NAMES:
        spec = _model_spec(name)
        target = root / name / spec["revision"]
        _reject_existing_symlink_components(
            target, reason=f"ctrl_world_asset_stage_target_unsafe:{name}"
        )
        target.mkdir(parents=True, exist_ok=True)
        marker = target / IDENTITY_NAME
        expected_identity = {
            "repository": spec["repository"],
            "revision": spec["revision"],
        }
        if marker.exists() and (marker.is_symlink() or _identity(marker) != expected_identity):
            raise ValueError(f"ctrl_world_asset_stage_identity_mismatch:{name}")
        observed_root = (
            Path(
                downloader(
                    repo_id=spec["repository"],
                    revision=spec["revision"],
                    local_dir=target,
                    allow_patterns=[row["relative_path"] for row in spec["required_files"]],
                    token=False,
                    max_workers=8,
                )
            )
            .expanduser()
            .resolve()
        )
        if observed_root != target.resolve():
            raise ValueError(f"ctrl_world_asset_stage_download_root_mismatch:{name}")
        observed_files: list[dict[str, Any]] = []
        for row in spec["required_files"]:
            path = _safe_file(
                target,
                row.get("relative_path"),
                reason=f"ctrl_world_asset_stage_file_missing_or_unsafe:{name}",
            )
            observed = {
                "relative_path": path.relative_to(target).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            if observed != row:
                raise ValueError(f"ctrl_world_asset_stage_file_mismatch:{name}")
            observed_files.append(observed)
        write_json(marker, expected_identity)
        snapshots[name] = {
            **expected_identity,
            "root": str(target),
            "required_file_count": len(observed_files),
            "required_files_sha256": canonical_sha256(observed_files),
            "required_bytes": sum(row["size_bytes"] for row in observed_files),
            "public_unauthenticated_download": True,
        }
        paths[name] = str(target)

    result: dict[str, Any] = {
        "schema_version": STAGE_SCHEMA_VERSION,
        "status": "completed",
        "snapshots": snapshots,
        "paths": {
            "world_model_checkpoint": str(
                Path(paths["ctrl_world"]) / MODEL_FREEZE["ctrl_world_checkpoint"]["file"]
            ),
            "svd_model_root": paths["stable_video_diffusion"],
            "clip_model_root": paths["clip"],
        },
        "stage_completed_before_policy_load": True,
        "token_argument": False,
        "raw_credentials_recorded": False,
        "physical_outcome_labels_accessed": False,
        "rankings_or_policy_outcomes_accessed": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "ctrl_world_runtime_asset_stage.json", result)
    return result


__all__ = ["STAGE_SCHEMA_VERSION", "stage_ctrl_world_runtime_assets"]
