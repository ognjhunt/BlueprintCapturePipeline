"""Request-bound Ctrl-World WAM seam for absolute-joint-position policies.

The provider/runtime callable behind this module must invoke the pinned
Ctrl-World model directly.  This module owns only request staging, immutable
model identity, generated-frame validation, and camera-separated media.  It
does not own policy inference, action conversion, scoring, or task labels.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .common import ensure_dir, write_json
from .droid_ctrl_world_joint_position_closed_loop_adapter import (
    CTRL_WORLD_RELEASED_VIEW_ORDER,
    CTRL_WORLD_SELECTED_HISTORY_INDICES,
    REQUEST_SCHEMA_VERSION,
)
from .policy_ranking_thesis import canonical_sha256, file_sha256


STAGED_REQUEST_SCHEMA_VERSION = "blueprint_ctrl_world_joint_position_staged_request.v1"
RUNTIME_RESULT_SCHEMA_VERSION = "blueprint_ctrl_world_joint_position_runtime_result.v1"
ARM_ID = "blueprint_ctrl_world_joint_position_reference"
HISTORY_FRAME_COUNT = 6
PREDICTED_FRAME_COUNT = 5
ACTION_CONDITIONING_SHAPE = (11, 7)
GENERATED_VIDEO_FPS = 5.0

MODEL_FREEZE = {
    "ctrl_world_source": {
        "repository": "https://github.com/Robert-gyj/Ctrl-World",
        "revision": "99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d",
        "required_files": [
            {
                "relative_path": "config.py",
                "size_bytes": 11169,
                "sha256": "f051477791dd9d3dad954b88e4a1d9f228d8378677d6d9d4de5d7579cef65e64",
            },
            {
                "relative_path": "models/__init__.py",
                "size_bytes": 0,
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            },
            {
                "relative_path": "models/ctrl_world.py",
                "size_bytes": 9186,
                "sha256": "b737c3b9097cb12ae39ac8524d4b50ca13320afac7f4c0ef0676176b8d83fb98",
            },
            {
                "relative_path": "models/pipeline_ctrl_world.py",
                "size_bytes": 43910,
                "sha256": "e88b7b76ce150c74b50ef6fae80f5e8a09d68b015ee339dcbd4a1ab150aaaf47",
            },
            {
                "relative_path": "models/pipeline_stable_video_diffusion.py",
                "size_bytes": 32884,
                "sha256": "73aea1aa5d16c6b734cb8a93b96bc98b44287dde9052ec9c0e6d2a0ffcd8c7a1",
            },
            {
                "relative_path": "models/unet_spatio_temporal_condition.py",
                "size_bytes": 23958,
                "sha256": "20e50ce474cb51b94404ccca00f2ef4bdb3ff348766ac9ff86567956fdb6a3c7",
            },
            {
                "relative_path": "models/utils.py",
                "size_bytes": 6602,
                "sha256": "bd9af90afdf379b95c2dfc7c3a5f8f6b8c6f1edc92ef8b8b7b59d08868ecfae3",
            },
            {
                "relative_path": "dataset_meta_info/droid/stat.json",
                "size_bytes": 294,
                "sha256": "1e6fa202c87d6295f8b988dfd2764dec88796c910846cecdf684670fb818f208",
            },
        ],
    },
    "ctrl_world_checkpoint": {
        "repository": "yjguo/Ctrl-World",
        "revision": "8cf814693f411962dc866a2ddb5b785afd17a93a",
        "file": "checkpoint-10000.pt",
        "size_bytes": 9_281_040_326,
        "sha256": "ed17de48180d4e6f89fd33c53e9fb7a0196189c1a67d44c2c486a279a80ea8a8",
    },
    "stable_video_diffusion": {
        "repository": "stabilityai/stable-video-diffusion-img2vid",
        "revision": "9cf024d5bfa8f56622af86c884f26a52f6676f2e",
        "required_files": [
            {
                "relative_path": "model_index.json",
                "size_bytes": 498,
                "sha256": "a6130bbf546242f454c184649a012809c34ab0f398cf8e80f887627b7f7dfc02",
            },
            {
                "relative_path": "feature_extractor/preprocessor_config.json",
                "size_bytes": 518,
                "sha256": "4db495644e3e5bd8fcac52f70e7fc0b413c911086021acf73ac30e5911166e95",
            },
            {
                "relative_path": "image_encoder/config.json",
                "size_bytes": 687,
                "sha256": "9f3e1b6d9c091720471c14efd208d4a2666642833cdc28e44ecc78b35e0dce13",
            },
            {
                "relative_path": "image_encoder/model.safetensors",
                "size_bytes": 2528371296,
                "sha256": "ed1e5af7b4042ca30ec29999a4a5cfcac90b7fb610fd05ace834f2dcbb763eab",
            },
            {
                "relative_path": "scheduler/scheduler_config.json",
                "size_bytes": 533,
                "sha256": "59aa43afc33395efd40fe94c7369c0477b81698f4b65b63e3ae06f26269876d5",
            },
            {
                "relative_path": "unet/config.json",
                "size_bytes": 986,
                "sha256": "b69bc73c489ebe9b7ecbcbd786b8d04a1be872599669084b0588112ba50bb46c",
            },
            {
                "relative_path": "unet/diffusion_pytorch_model.safetensors",
                "size_bytes": 6098682464,
                "sha256": "98c5e6b99df6bef015b2681c0f8ab9d4c807b733be46c067d6c9966101698f58",
            },
            {
                "relative_path": "vae/config.json",
                "size_bytes": 609,
                "sha256": "aab2b8766d6db1bf742fa0e3bf217fae64e67a20e7e0fe78de84f79007100ffb",
            },
            {
                "relative_path": "vae/diffusion_pytorch_model.safetensors",
                "size_bytes": 391017740,
                "sha256": "9975042d7bee021bd53a72b1af14c8627d624f6547ec9abe661b68b962b88c49",
            },
        ],
    },
    "clip": {
        "repository": "openai/clip-vit-base-patch32",
        "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "required_files": [
            {
                "relative_path": "config.json",
                "size_bytes": 4186,
                "sha256": "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770",
            },
            {
                "relative_path": "merges.txt",
                "size_bytes": 524657,
                "sha256": "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85",
            },
            {
                "relative_path": "preprocessor_config.json",
                "size_bytes": 316,
                "sha256": "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
            },
            {
                "relative_path": "pytorch_model.bin",
                "size_bytes": 605247071,
                "sha256": "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
            },
            {
                "relative_path": "special_tokens_map.json",
                "size_bytes": 389,
                "sha256": "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf",
            },
            {
                "relative_path": "tokenizer_config.json",
                "size_bytes": 592,
                "sha256": "34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad",
            },
            {
                "relative_path": "vocab.json",
                "size_bytes": 862328,
                "sha256": "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
            },
        ],
    },
    "ctrl_world_state_stats": {
        "repository": "https://github.com/Robert-gyj/Ctrl-World",
        "revision": "99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d",
        "file": "dataset_meta_info/droid/stat.json",
        "sha256": "1e6fa202c87d6295f8b988dfd2764dec88796c910846cecdf684670fb818f208",
    },
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
        "ranking",
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


def validate_ctrl_world_joint_position_request(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one label-free, outcome-blind WAM request."""

    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("ctrl_world_joint_position_request_schema_invalid")
    leaked = _find_forbidden_keys(request)
    if leaked:
        raise ValueError(f"ctrl_world_joint_position_request_leakage:{leaked[0]}")
    query_index = request.get("query_index")
    if isinstance(query_index, bool) or not isinstance(query_index, int) or query_index < 0:
        raise ValueError("ctrl_world_joint_position_query_index_invalid")
    task_prompt = request.get("task_prompt")
    if not isinstance(task_prompt, str) or not task_prompt.strip():
        raise ValueError("ctrl_world_joint_position_task_prompt_missing")
    if tuple(request.get("view_order") or ()) != CTRL_WORLD_RELEASED_VIEW_ORDER:
        raise ValueError("ctrl_world_joint_position_view_order_invalid")
    if tuple(request.get("selected_history_indices") or ()) != (
        CTRL_WORLD_SELECTED_HISTORY_INDICES
    ):
        raise ValueError("ctrl_world_joint_position_history_indices_invalid")

    histories = request.get("selected_history_views")
    if not isinstance(histories, Mapping) or set(histories) != set(CTRL_WORLD_RELEASED_VIEW_ORDER):
        raise ValueError("ctrl_world_joint_position_history_views_invalid")
    normalized_histories: dict[str, list[dict[str, str]]] = {}
    for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER:
        rows = histories[view_id]
        if not isinstance(rows, list) or len(rows) != HISTORY_FRAME_COUNT:
            raise ValueError(f"ctrl_world_joint_position_history_count_invalid:{view_id}")
        normalized_rows: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"ctrl_world_joint_position_history_row_invalid:{view_id}")
            path = _safe_regular_file(
                row.get("path"),
                reason=f"ctrl_world_joint_position_history_file_invalid:{view_id}",
            )
            digest = file_sha256(path)
            if row.get("sha256") != digest:
                raise ValueError(f"ctrl_world_joint_position_history_hash_mismatch:{view_id}")
            try:
                with Image.open(path) as image:
                    if image.mode != "RGB" or image.size != (320, 192):
                        raise ValueError(
                            f"ctrl_world_joint_position_history_geometry_invalid:{view_id}"
                        )
                    image.verify()
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(
                    f"ctrl_world_joint_position_history_decode_failed:{view_id}"
                ) from exc
            normalized_rows.append({"path": str(path), "sha256": digest})
        normalized_histories[view_id] = normalized_rows

    current_rows = request.get("current_views")
    if not isinstance(current_rows, Mapping) or set(current_rows) != set(
        CTRL_WORLD_RELEASED_VIEW_ORDER
    ):
        raise ValueError("ctrl_world_joint_position_current_views_invalid")
    normalized_current: dict[str, dict[str, str]] = {}
    for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER:
        row = current_rows[view_id]
        if not isinstance(row, Mapping):
            raise ValueError(f"ctrl_world_joint_position_current_view_row_invalid:{view_id}")
        path = _safe_regular_file(
            row.get("path"),
            reason=f"ctrl_world_joint_position_current_view_file_invalid:{view_id}",
        )
        digest = file_sha256(path)
        if row.get("sha256") != digest:
            raise ValueError(f"ctrl_world_joint_position_current_view_hash_mismatch:{view_id}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (320, 192):
                raise ValueError(
                    f"ctrl_world_joint_position_current_view_geometry_invalid:{view_id}"
                )
            image.verify()
        normalized_current[view_id] = {"path": str(path), "sha256": digest}

    action = np.asarray(request.get("action_conditioning_7d"), dtype=np.float64)
    if action.shape != ACTION_CONDITIONING_SHAPE or not np.isfinite(action).all():
        raise ValueError("ctrl_world_joint_position_action_conditioning_invalid")
    if request.get("action_conditioning_shape") != list(ACTION_CONDITIONING_SHAPE):
        raise ValueError("ctrl_world_joint_position_action_shape_declaration_invalid")
    if request.get("predicted_frame_count") != PREDICTED_FRAME_COUNT:
        raise ValueError("ctrl_world_joint_position_predicted_frame_count_invalid")
    if request.get("executed_prefix_steps") != 8:
        raise ValueError("ctrl_world_joint_position_executed_prefix_steps_invalid")
    seconds = request.get("executed_prefix_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError("ctrl_world_joint_position_executed_prefix_seconds_invalid")
    if not math.isclose(float(seconds), 8 / 15, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("ctrl_world_joint_position_executed_prefix_seconds_invalid")
    if request.get("physical_future_observation_used") is not False:
        raise ValueError("ctrl_world_joint_position_physical_future_not_false")
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


def stage_ctrl_world_joint_position_request(
    request: Mapping[str, Any], *, output_dir: str | Path, seed: int
) -> dict[str, Any]:
    """Copy one validated request into a deterministic provider-safe directory."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("ctrl_world_joint_position_seed_invalid")
    normalized = validate_ctrl_world_joint_position_request(request)
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"ctrl_world_joint_position_stage_exists:{target}")
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
                raise RuntimeError("ctrl_world_joint_position_staged_history_hash_mismatch")
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
            raise RuntimeError("ctrl_world_joint_position_staged_current_hash_mismatch")
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
        "label_free": True,
    }
    manifest["request_sha256"] = canonical_sha256(manifest)
    manifest_path = target / "ctrl_world_joint_position_request.json"
    write_json(manifest_path, manifest)
    return {
        "status": "completed",
        "request_dir": str(target),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "request_sha256": manifest["request_sha256"],
        "action_conditioning_sha256": manifest["action_conditioning"]["sha256"],
    }


def validate_ctrl_world_joint_position_result(
    result: Mapping[str, Any], *, request_receipt: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    """Fail closed unless the runtime returns attributable generated-only views."""

    if result.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION:
        raise ValueError("ctrl_world_joint_position_result_schema_invalid")
    if result.get("status") != "completed":
        raise ValueError("ctrl_world_joint_position_runtime_not_completed")
    if result.get("arm_id") != ARM_ID:
        raise ValueError("ctrl_world_joint_position_arm_identity_mismatch")
    if result.get("request_sha256") != request_receipt.get("request_sha256"):
        raise ValueError("ctrl_world_joint_position_result_request_mismatch")
    if result.get("seed") != seed:
        raise ValueError("ctrl_world_joint_position_result_seed_mismatch")
    if result.get("model_freeze") != MODEL_FREEZE:
        raise ValueError("ctrl_world_joint_position_result_model_freeze_mismatch")
    if result.get("runtime_asset_admission_passed") is not True:
        raise ValueError("ctrl_world_joint_position_runtime_assets_unverified")
    for key in (
        "physical_future_observation_used",
        "physical_outcome_labels_accessed",
        "recorded_action_trace_used",
        "wam_to_wam_chaining",
    ):
        if result.get(key) is not False:
            raise ValueError(f"ctrl_world_joint_position_result_{key}_not_false")
    if result.get("same_frozen_wam_generated_all_views") is not True:
        raise ValueError("ctrl_world_joint_position_result_cross_view_identity_invalid")
    sequences = result.get("generated_view_frame_sequences")
    hashes = result.get("generated_view_frame_sha256")
    if not isinstance(sequences, Mapping) or set(sequences) != set(CTRL_WORLD_RELEASED_VIEW_ORDER):
        raise ValueError("ctrl_world_joint_position_result_view_sequences_invalid")
    if not isinstance(hashes, Mapping) or set(hashes) != set(CTRL_WORLD_RELEASED_VIEW_ORDER):
        raise ValueError("ctrl_world_joint_position_result_view_hashes_invalid")
    normalized_sequences: dict[str, list[str]] = {}
    normalized_hashes: dict[str, list[str]] = {}
    for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER:
        paths = sequences[view_id]
        digests = hashes[view_id]
        if not isinstance(paths, list) or len(paths) != PREDICTED_FRAME_COUNT:
            raise ValueError(f"ctrl_world_joint_position_result_frame_count_invalid:{view_id}")
        if not isinstance(digests, list) or len(digests) != PREDICTED_FRAME_COUNT:
            raise ValueError(f"ctrl_world_joint_position_result_hash_count_invalid:{view_id}")
        normalized_sequences[view_id] = []
        normalized_hashes[view_id] = []
        for path_value, expected_digest in zip(paths, digests, strict=True):
            path = _safe_regular_file(
                path_value, reason=f"ctrl_world_joint_position_generated_frame_invalid:{view_id}"
            )
            digest = file_sha256(path)
            if expected_digest != digest:
                raise ValueError(
                    f"ctrl_world_joint_position_generated_frame_hash_mismatch:{view_id}"
                )
            try:
                with Image.open(path) as image:
                    if image.mode != "RGB" or image.size != (320, 192):
                        raise ValueError(
                            f"ctrl_world_joint_position_generated_frame_geometry_invalid:{view_id}"
                        )
                    image.verify()
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(
                    f"ctrl_world_joint_position_generated_frame_decode_failed:{view_id}"
                ) from exc
            normalized_sequences[view_id].append(str(path))
            normalized_hashes[view_id].append(digest)
    validated = dict(result)
    validated["generated_view_frame_sequences"] = normalized_sequences
    validated["generated_view_frame_sha256"] = normalized_hashes
    validated["result_sha256"] = canonical_sha256(
        {key: value for key, value in validated.items() if key != "result_sha256"}
    )
    return validated


def _encode_view_video(paths: Sequence[str], output_path: Path) -> dict[str, Any]:
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        GENERATED_VIDEO_FPS,
        (320, 192),
    )
    if not writer.isOpened():
        raise RuntimeError("ctrl_world_joint_position_video_writer_open_failed")
    try:
        for path_value in paths:
            with Image.open(path_value) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError("ctrl_world_joint_position_video_write_failed")
    return {
        "path": str(output_path),
        "sha256": file_sha256(output_path),
        "frame_count": len(paths),
        "fps": GENERATED_VIDEO_FPS,
        "resolution": [320, 192],
    }


@dataclass(frozen=True)
class CallableCtrlWorldJointPositionReferenceWamArm:
    """Bind a provider/runtime callable to Blueprint's model-neutral WAM arm."""

    runner: Callable[..., Mapping[str, Any]]
    seed: int
    arm_id: str = ARM_ID

    def predict(self, request: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
        root = Path(output_dir).expanduser().resolve()
        request_receipt = stage_ctrl_world_joint_position_request(
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
            raise ValueError("ctrl_world_joint_position_runtime_result_not_mapping")
        validated = validate_ctrl_world_joint_position_result(
            result, request_receipt=request_receipt, seed=self.seed
        )
        videos: dict[str, str] = {}
        video_evidence: dict[str, Any] = {}
        generated_frames: dict[str, str] = {}
        for view_index, view_id in enumerate(CTRL_WORLD_RELEASED_VIEW_ORDER):
            evidence = _encode_view_video(
                validated["generated_view_frame_sequences"][view_id],
                root / f"ctrl_world_generated_view_{view_index}.mp4",
            )
            videos[view_id] = evidence["path"]
            video_evidence[view_id] = evidence
            generated_frames[view_id] = validated["generated_view_frame_sequences"][view_id][-1]
        evidence = {
            **validated,
            "request_receipt": dict(request_receipt),
            "generated_videos_by_view": videos,
            "generated_video_evidence_by_view": video_evidence,
            "generated_view_frames": generated_frames,
            "blueprint_joint_position_reference_not_exact_paper_reproduction": True,
        }
        write_json(root / "ctrl_world_joint_position_wam_result.json", evidence)
        return evidence


__all__ = [
    "ACTION_CONDITIONING_SHAPE",
    "ARM_ID",
    "CallableCtrlWorldJointPositionReferenceWamArm",
    "GENERATED_VIDEO_FPS",
    "MODEL_FREEZE",
    "PREDICTED_FRAME_COUNT",
    "RUNTIME_RESULT_SCHEMA_VERSION",
    "STAGED_REQUEST_SCHEMA_VERSION",
    "stage_ctrl_world_joint_position_request",
    "validate_ctrl_world_joint_position_request",
    "validate_ctrl_world_joint_position_result",
]
