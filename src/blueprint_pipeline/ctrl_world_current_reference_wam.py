"""Request-bound Ctrl-World current-reference WAM contract.

This module is the provider-neutral seam between Blueprint's closed-loop
orchestrator and a pinned Ctrl-World runtime.  It deliberately does not own
policy inference, action conversion, reliability scoring, or evaluation.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .common import ensure_dir, write_json
from .ctrl_world_provider_bundle import (
    CLIP_REPOSITORY,
    CLIP_REVISION,
    CTRL_WORLD_CHECKPOINT_REPOSITORY,
    CTRL_WORLD_CHECKPOINT_REVISION,
    CTRL_WORLD_SOURCE_REPOSITORY,
    CTRL_WORLD_SOURCE_REVISION,
    SVD_REPOSITORY,
    SVD_REVISION,
)
from .droid_ctrl_world_closed_loop_adapter import (
    CTRL_WORLD_RELEASED_VIEW_ORDER,
    CTRL_WORLD_SELECTED_HISTORY_INDICES,
)
from .policy_ranking_thesis import canonical_sha256, file_sha256


REQUEST_SCHEMA_VERSION = "blueprint_ctrl_world_current_reference_request.v1"
STAGED_REQUEST_SCHEMA_VERSION = "blueprint_ctrl_world_current_reference_staged_request.v1"
RUNTIME_RESULT_SCHEMA_VERSION = "blueprint_ctrl_world_current_reference_runtime_result.v1"
ARM_ID = "blueprint_ctrl_world_current_reference"
HISTORY_FRAME_COUNT = 6
PREDICTED_FRAME_COUNT = 5
ACTION_CONDITIONING_SHAPE = (11, 7)

MODEL_FREEZE = {
    "ctrl_world_source": {
        "repository": CTRL_WORLD_SOURCE_REPOSITORY,
        "revision": CTRL_WORLD_SOURCE_REVISION,
    },
    "ctrl_world_checkpoint": {
        "repository": CTRL_WORLD_CHECKPOINT_REPOSITORY,
        "revision": CTRL_WORLD_CHECKPOINT_REVISION,
    },
    "stable_video_diffusion": {
        "repository": SVD_REPOSITORY,
        "revision": SVD_REVISION,
    },
    "clip": {"repository": CLIP_REPOSITORY, "revision": CLIP_REVISION},
}

_FORBIDDEN_KEYS = frozenset(
    {
        "policy",
        "policy_id",
        "policy_name",
        "candidate_policy",
        "candidate_policy_id",
        "physical_outcome",
        "success",
        "success_rate",
        "score",
    }
)


def _safe_regular_file(value: Any, *, reason: str) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise ValueError(reason)
    return path


def _find_forbidden_keys(value: Any, *, prefix: str = "") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            path = f"{prefix}.{key}" if prefix else key
            if key in _FORBIDDEN_KEYS:
                violations.append(path)
            violations.extend(_find_forbidden_keys(child, prefix=path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            violations.extend(_find_forbidden_keys(child, prefix=f"{prefix}[{index}]"))
    return violations


def validate_ctrl_world_current_reference_request(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one outcome-blind Ctrl-World request."""

    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("ctrl_world_current_reference_request_schema_invalid")
    leaked = _find_forbidden_keys(request)
    if leaked:
        raise ValueError(f"ctrl_world_current_reference_request_leakage:{leaked[0]}")
    query_index = request.get("query_index")
    if isinstance(query_index, bool) or not isinstance(query_index, int) or query_index < 0:
        raise ValueError("ctrl_world_current_reference_query_index_invalid")
    task_prompt = request.get("task_prompt")
    if not isinstance(task_prompt, str) or not task_prompt.strip():
        raise ValueError("ctrl_world_current_reference_task_prompt_missing")
    if tuple(request.get("view_order") or ()) != CTRL_WORLD_RELEASED_VIEW_ORDER:
        raise ValueError("ctrl_world_current_reference_view_order_invalid")
    if tuple(request.get("selected_history_indices") or ()) != (
        CTRL_WORLD_SELECTED_HISTORY_INDICES
    ):
        raise ValueError("ctrl_world_current_reference_history_indices_invalid")

    histories = request.get("selected_history_views")
    if not isinstance(histories, Mapping) or set(histories) != set(CTRL_WORLD_RELEASED_VIEW_ORDER):
        raise ValueError("ctrl_world_current_reference_history_views_invalid")
    normalized_histories: dict[str, list[dict[str, str]]] = {}
    for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER:
        rows = histories[view_id]
        if not isinstance(rows, list) or len(rows) != HISTORY_FRAME_COUNT:
            raise ValueError(f"ctrl_world_current_reference_history_count_invalid:{view_id}")
        normalized_rows: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"ctrl_world_current_reference_history_row_invalid:{view_id}")
            path = _safe_regular_file(
                row.get("path"),
                reason=f"ctrl_world_current_reference_history_file_invalid:{view_id}",
            )
            digest = file_sha256(path)
            if row.get("sha256") != digest:
                raise ValueError(f"ctrl_world_current_reference_history_hash_mismatch:{view_id}")
            try:
                with Image.open(path) as image:
                    if image.mode != "RGB" or image.size != (320, 192):
                        raise ValueError(
                            f"ctrl_world_current_reference_history_geometry_invalid:{view_id}"
                        )
                    image.verify()
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(
                    f"ctrl_world_current_reference_history_decode_failed:{view_id}"
                ) from exc
            normalized_rows.append({"path": str(path), "sha256": digest})
        normalized_histories[view_id] = normalized_rows

    current_views = request.get("current_views")
    if not isinstance(current_views, Mapping) or set(current_views) != set(
        CTRL_WORLD_RELEASED_VIEW_ORDER
    ):
        raise ValueError("ctrl_world_current_reference_current_views_invalid")
    normalized_current: dict[str, dict[str, str]] = {}
    for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER:
        row = current_views[view_id]
        if not isinstance(row, Mapping):
            raise ValueError(f"ctrl_world_current_reference_current_view_row_invalid:{view_id}")
        path = _safe_regular_file(
            row.get("path"),
            reason=f"ctrl_world_current_reference_current_view_file_invalid:{view_id}",
        )
        digest = file_sha256(path)
        if row.get("sha256") != digest:
            raise ValueError(f"ctrl_world_current_reference_current_view_hash_mismatch:{view_id}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (320, 192):
                raise ValueError(
                    f"ctrl_world_current_reference_current_view_geometry_invalid:{view_id}"
                )
            image.verify()
        normalized_current[view_id] = {"path": str(path), "sha256": digest}

    action = np.asarray(request.get("action_conditioning_7d"), dtype=np.float64)
    if action.shape != ACTION_CONDITIONING_SHAPE or not np.isfinite(action).all():
        raise ValueError("ctrl_world_current_reference_action_conditioning_invalid")
    if request.get("action_conditioning_shape") != list(ACTION_CONDITIONING_SHAPE):
        raise ValueError("ctrl_world_current_reference_action_shape_declaration_invalid")
    if request.get("predicted_frame_count") != PREDICTED_FRAME_COUNT:
        raise ValueError("ctrl_world_current_reference_predicted_frame_count_invalid")
    if request.get("executed_prefix_steps") != 8:
        raise ValueError("ctrl_world_current_reference_executed_prefix_steps_invalid")
    seconds = request.get("executed_prefix_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError("ctrl_world_current_reference_executed_prefix_seconds_invalid")
    if not math.isclose(float(seconds), 8 / 15, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("ctrl_world_current_reference_executed_prefix_seconds_invalid")
    if request.get("physical_future_observation_used") is not False:
        raise ValueError("ctrl_world_current_reference_physical_future_not_false")
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "query_index": query_index,
        "task_prompt": task_prompt.strip(),
        "view_order": list(CTRL_WORLD_RELEASED_VIEW_ORDER),
        "selected_history_views": normalized_histories,
        "current_views": normalized_current,
        "selected_history_indices": list(CTRL_WORLD_SELECTED_HISTORY_INDICES),
        "action_conditioning_7d": action,
        "action_conditioning_shape": list(ACTION_CONDITIONING_SHAPE),
        "predicted_frame_count": PREDICTED_FRAME_COUNT,
        "executed_prefix_steps": 8,
        "executed_prefix_seconds": 8 / 15,
        "physical_future_observation_used": False,
    }


def stage_ctrl_world_current_reference_request(
    request: Mapping[str, Any], *, output_dir: str | Path, seed: int
) -> dict[str, Any]:
    """Copy one validated request into a deterministic, provider-safe directory."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("ctrl_world_current_reference_seed_invalid")
    normalized = validate_ctrl_world_current_reference_request(request)
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"ctrl_world_current_reference_stage_exists:{target}")
    ensure_dir(target)
    staged_histories: dict[str, list[dict[str, str]]] = {}
    staged_current: dict[str, dict[str, str]] = {}
    for view_index, view_id in enumerate(CTRL_WORLD_RELEASED_VIEW_ORDER):
        staged_histories[view_id] = []
        for frame_index, row in enumerate(normalized["selected_history_views"][view_id]):
            source = Path(row["path"])
            suffix = source.suffix.lower() or ".png"
            relative = Path("history") / f"view_{view_index}" / f"frame_{frame_index:02d}{suffix}"
            destination = target / relative
            ensure_dir(destination.parent)
            shutil.copyfile(source, destination)
            digest = file_sha256(destination)
            if digest != row["sha256"]:
                raise RuntimeError("ctrl_world_current_reference_staged_history_hash_mismatch")
            staged_histories[view_id].append(
                {"relative_path": relative.as_posix(), "sha256": digest}
            )
        current_source = Path(normalized["current_views"][view_id]["path"])
        current_suffix = current_source.suffix.lower() or ".png"
        current_relative = Path("current") / f"view_{view_index}{current_suffix}"
        current_destination = target / current_relative
        ensure_dir(current_destination.parent)
        shutil.copyfile(current_source, current_destination)
        current_digest = file_sha256(current_destination)
        if current_digest != normalized["current_views"][view_id]["sha256"]:
            raise RuntimeError("ctrl_world_current_reference_staged_current_hash_mismatch")
        staged_current[view_id] = {
            "relative_path": current_relative.as_posix(),
            "sha256": current_digest,
        }
    action_path = target / "action_conditioning_11x7.npy"
    np.save(action_path, normalized["action_conditioning_7d"], allow_pickle=False)
    manifest: dict[str, Any] = {
        "schema_version": STAGED_REQUEST_SCHEMA_VERSION,
        "source_request_schema_version": REQUEST_SCHEMA_VERSION,
        "query_index": normalized["query_index"],
        "task_prompt": normalized["task_prompt"],
        "view_order": normalized["view_order"],
        "selected_history_indices": normalized["selected_history_indices"],
        "selected_history_views": staged_histories,
        "current_views": staged_current,
        "action_conditioning": {
            "relative_path": action_path.relative_to(target).as_posix(),
            "sha256": file_sha256(action_path),
            "shape": list(ACTION_CONDITIONING_SHAPE),
            "dtype": "float64",
        },
        "predicted_frame_count": PREDICTED_FRAME_COUNT,
        "executed_prefix_steps": 8,
        "executed_prefix_seconds": 8 / 15,
        "seed": seed,
        "model_freeze": MODEL_FREEZE,
        "physical_future_observation_used": False,
        "physical_outcome_labels_accessed": False,
        "policy_identity_in_provider_request": False,
        "recorded_action_trace_used": False,
    }
    manifest["request_sha256"] = canonical_sha256(manifest)
    manifest_path = target / "ctrl_world_current_reference_request.json"
    write_json(manifest_path, manifest)
    return {
        "status": "completed",
        "request_dir": str(target),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "request_sha256": manifest["request_sha256"],
        "action_conditioning_sha256": manifest["action_conditioning"]["sha256"],
    }


def validate_ctrl_world_current_reference_result(
    result: Mapping[str, Any], *, request_receipt: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    """Fail closed unless the runtime returns only attributable generated views."""

    if result.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION:
        raise ValueError("ctrl_world_current_reference_result_schema_invalid")
    if result.get("status") != "completed":
        raise ValueError("ctrl_world_current_reference_runtime_not_completed")
    if result.get("arm_id") != ARM_ID:
        raise ValueError("ctrl_world_current_reference_arm_identity_mismatch")
    if result.get("request_sha256") != request_receipt.get("request_sha256"):
        raise ValueError("ctrl_world_current_reference_result_request_mismatch")
    if result.get("seed") != seed:
        raise ValueError("ctrl_world_current_reference_result_seed_mismatch")
    if result.get("model_freeze") != MODEL_FREEZE:
        raise ValueError("ctrl_world_current_reference_result_model_freeze_mismatch")
    required_false = (
        "physical_future_observation_used",
        "physical_outcome_labels_accessed",
        "recorded_action_trace_used",
        "wam_to_wam_chaining",
    )
    for key in required_false:
        if result.get(key) is not False:
            raise ValueError(f"ctrl_world_current_reference_result_{key}_not_false")
    if result.get("same_frozen_wam_generated_all_views") is not True:
        raise ValueError("ctrl_world_current_reference_result_cross_view_identity_invalid")
    sequences = result.get("generated_view_frame_sequences")
    hashes = result.get("generated_view_frame_sha256")
    if not isinstance(sequences, Mapping) or set(sequences) != set(CTRL_WORLD_RELEASED_VIEW_ORDER):
        raise ValueError("ctrl_world_current_reference_result_view_sequences_invalid")
    if not isinstance(hashes, Mapping) or set(hashes) != set(CTRL_WORLD_RELEASED_VIEW_ORDER):
        raise ValueError("ctrl_world_current_reference_result_view_hashes_invalid")
    normalized_sequences: dict[str, list[str]] = {}
    normalized_hashes: dict[str, list[str]] = {}
    for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER:
        paths = sequences[view_id]
        digests = hashes[view_id]
        if not isinstance(paths, list) or len(paths) != PREDICTED_FRAME_COUNT:
            raise ValueError(f"ctrl_world_current_reference_result_frame_count_invalid:{view_id}")
        if not isinstance(digests, list) or len(digests) != PREDICTED_FRAME_COUNT:
            raise ValueError(f"ctrl_world_current_reference_result_hash_count_invalid:{view_id}")
        normalized_sequences[view_id] = []
        normalized_hashes[view_id] = []
        for path_value, expected_digest in zip(paths, digests, strict=True):
            path = _safe_regular_file(
                path_value, reason=f"ctrl_world_current_reference_generated_frame_invalid:{view_id}"
            )
            digest = file_sha256(path)
            if expected_digest != digest:
                raise ValueError(
                    f"ctrl_world_current_reference_generated_frame_hash_mismatch:{view_id}"
                )
            normalized_sequences[view_id].append(str(path))
            normalized_hashes[view_id].append(digest)
    validated = dict(result)
    validated["generated_view_frame_sequences"] = normalized_sequences
    validated["generated_view_frame_sha256"] = normalized_hashes
    validated["result_sha256"] = canonical_sha256(
        {key: value for key, value in validated.items() if key != "result_sha256"}
    )
    return validated


@dataclass(frozen=True)
class CallableCtrlWorldCurrentReferenceWamArm:
    """Bind a provider/runtime callable to Blueprint's model-neutral ``WamArm``."""

    runner: Callable[..., Mapping[str, Any]]
    seed: int
    arm_id: str = ARM_ID

    def predict(self, request: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
        root = Path(output_dir).expanduser().resolve()
        request_receipt = stage_ctrl_world_current_reference_request(
            request, output_dir=root / "ctrl_world_request", seed=self.seed
        )
        runtime_dir = root / "ctrl_world_runtime"
        ensure_dir(runtime_dir)
        result = self.runner(
            request_manifest_path=Path(request_receipt["manifest_path"]),
            output_dir=runtime_dir,
            seed=self.seed,
        )
        if not isinstance(result, Mapping):
            raise ValueError("ctrl_world_current_reference_runtime_result_not_mapping")
        validated = validate_ctrl_world_current_reference_result(
            result, request_receipt=request_receipt, seed=self.seed
        )
        evidence = {
            **validated,
            "request_receipt": dict(request_receipt),
            "blueprint_current_reference_not_exact_paper_reproduction": True,
        }
        write_json(root / "ctrl_world_current_reference_wam_result.json", evidence)
        return evidence


__all__ = [
    "ACTION_CONDITIONING_SHAPE",
    "ARM_ID",
    "CallableCtrlWorldCurrentReferenceWamArm",
    "MODEL_FREEZE",
    "PREDICTED_FRAME_COUNT",
    "REQUEST_SCHEMA_VERSION",
    "RUNTIME_RESULT_SCHEMA_VERSION",
    "STAGED_REQUEST_SCHEMA_VERSION",
    "stage_ctrl_world_current_reference_request",
    "validate_ctrl_world_current_reference_request",
    "validate_ctrl_world_current_reference_result",
]
