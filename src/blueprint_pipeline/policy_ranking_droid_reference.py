"""Materialize NVIDIA's published Cosmos3 DROID forward-dynamics canary.

This module is intentionally provider-neutral and label-free.  It validates the
checked-in upstream DROID example, reconstructs the exact 15 Hz wrist/shoulder
``concat_view`` contract, and freezes one high-motion single-chunk canary plus a
valid no-motion control.  It never launches a provider or loads benchmark
outcome labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2  # type: ignore[import-not-found]
import numpy as np

from .common import ensure_dir, write_json
from .policy_ranking_successor_cosmos import (
    CHECKPOINT_REVISION,
    DROID_HORIZON,
    VLLM_IMAGE,
    VLLM_IMAGE_DIGEST,
    canonical_sha256,
    convert_droid_states_to_action_stream,
    droid_action_stream,
)
from .policy_ranking_thesis import file_sha256


SCHEMA_VERSION = "policy_ranking_cosmos3_official_droid_reference_canary.v1"
EXPERIMENT_ID = "policy_ranking_roboarena_droid_reference_confirmation_20260729"
COSMOS_REPOSITORY = "https://github.com/NVIDIA/cosmos"
COSMOS_REVISION = "0299468993d8bcd8f6a95b0d8427b1221fccfced"
COSMOS_FRAMEWORK_REPOSITORY = "https://github.com/NVIDIA/cosmos-framework"
COSMOS_FRAMEWORK_REVISION = "9726697a83315540c6885baefd2fe353d9c74920"
VLLM_OMNI_REPOSITORY = "https://github.com/vllm-project/vllm-omni"
VLLM_OMNI_CURRENT_REVISION = "1c6e7313394923000215a3299f4f79ede3873ecc"
COSMOS3_DROID_DATASET = "nvidia/Cosmos3-DROID"
COSMOS3_DROID_DATASET_REVISION = "5c11a20accb11497270a5247a7f1e66ad04c956c"
LICENSE_ID = "OpenMDW-1.1"

UPSTREAM_ASSET_SHA256: Mapping[str, str] = {
    "data/chunk-000/file-000.parquet": (
        "56e3defd9e75a101a7b812ad7ae263dde8ec5699b6b47db51ba6954f81be2593"
    ),
    "meta/episodes/chunk-000/file-000.parquet": (
        "23f8e342bec24cf1af5fc4c5f58d8b51097ce09c2aaaafe6d467328af6dc016e"
    ),
    "meta/info.json": "64e39f3dbedbd5ffade567007093d92d1827fcdb68c5d9d573ff3e80eef23cf6",
    "meta/tasks.parquet": "ee2c7ec4f086cf2025b9a1d169bf6ba1857fed2579604c8859b49741f5eb29d6",
    "videos/observation.image.exterior_image_1_left/chunk-000/file-000.mp4": (
        "6aac49551ff9b9fc7b8e9899df4fc050f5cd31164ed4ee3c014f332bcef3b9f1"
    ),
    "videos/observation.image.exterior_image_2_left/chunk-000/file-000.mp4": (
        "34f6e5dae8591809324f1df9bf166c6e304bd2cbdad643595a991f641cbb4fd7"
    ),
    "videos/observation.image.wrist_image_left/chunk-000/file-000.mp4": (
        "e9d641240f9efc344924c31715a44be1f66207f667dd63c292bbd943dfa816d0"
    ),
}

CHUNK_STARTS = (0, 16, 32, 48, 64)
VIEW_PATHS: Mapping[str, str] = {
    "wrist": "videos/observation.image.wrist_image_left/chunk-000/file-000.mp4",
    "left": "videos/observation.image.exterior_image_1_left/chunk-000/file-000.mp4",
    "right": "videos/observation.image.exterior_image_2_left/chunk-000/file-000.mp4",
}


def _validated_source(root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in UPSTREAM_ASSET_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"official_droid_asset_missing:{relative}")
        digest = file_sha256(path)
        if digest != expected:
            raise ValueError(f"official_droid_asset_sha256_mismatch:{relative}")
        observed[relative] = digest
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if int(info.get("fps") or 0) != 15 or int(info.get("total_frames") or 0) != 722:
        raise ValueError("official_droid_info_temporal_contract_invalid")
    for key in (
        "observation.image.wrist_image_left",
        "observation.image.exterior_image_1_left",
        "observation.image.exterior_image_2_left",
    ):
        feature = (info.get("features") or {}).get(key) or {}
        video = feature.get("info") or {}
        if feature.get("shape") != [360, 640, 3] or float(video.get("video.fps") or 0) != 15.0:
            raise ValueError(f"official_droid_camera_contract_invalid:{key}")
    return observed


def _read_rows(root: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment-specific gate
        raise RuntimeError("official_droid_reference_requires_pyarrow") from exc
    rows = parquet.read_table(root / "data/chunk-000/file-000.parquet").to_pylist()
    if len(rows) != 722:
        raise ValueError("official_droid_parquet_row_count_invalid")
    return rows


def _read_composites(root: Path, frame_count: int = 81) -> list[np.ndarray]:
    captures = {
        name: cv2.VideoCapture(str(root / relative)) for name, relative in VIEW_PATHS.items()
    }
    if not all(capture.isOpened() for capture in captures.values()):
        for capture in captures.values():
            capture.release()
        raise ValueError("official_droid_video_open_failed")
    frames: list[np.ndarray] = []
    try:
        for _ in range(frame_count):
            decoded: dict[str, np.ndarray] = {}
            for name, capture in captures.items():
                ok, frame = capture.read()
                if not ok or frame is None or frame.shape != (360, 640, 3):
                    raise ValueError(f"official_droid_video_decode_failed:{name}")
                decoded[name] = frame
            left = cv2.resize(decoded["left"], (320, 180), interpolation=cv2.INTER_AREA)
            right = cv2.resize(decoded["right"], (320, 180), interpolation=cv2.INTER_AREA)
            frames.append(np.vstack([decoded["wrist"], np.hstack([left, right])]))
    finally:
        for capture in captures.values():
            capture.release()
    return frames


def _motion_metrics(frames: Sequence[np.ndarray]) -> dict[str, float]:
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]).astype(
        np.float32
    )
    return {
        "temporal_absolute_difference_mean_gray_0_255": float(np.abs(np.diff(gray, axis=0)).mean()),
        "first_to_last_absolute_difference_mean_gray_0_255": float(
            np.abs(gray[-1] - gray[0]).mean()
        ),
    }


def _action_stream(rows: Sequence[Mapping[str, Any]], start: int) -> dict[str, Any]:
    states = [row["observation.state.cartesian_position"] for row in rows[start : start + 17]]
    gripper = [float(row["action.gripper_position"]) for row in rows[start : start + 16]]
    return convert_droid_states_to_action_stream(
        states,
        gripper,
        source_gripper_action_flipped=True,
    )


def _no_motion_stream(*, gripper_hold: float) -> dict[str, Any]:
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    row = [0.0, 0.0, 0.0, *identity, float(gripper_hold)]
    return droid_action_stream([row for _ in range(DROID_HORIZON)])


def build_official_droid_reference_canary(
    *, source_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Build the immutable, provider-free official DROID reference packet."""

    root = Path(source_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    ensure_dir(output)
    source_hashes = _validated_source(root)
    rows = _read_rows(root)
    composites = _read_composites(root)
    candidates: list[dict[str, Any]] = []
    for start in CHUNK_STARTS:
        metrics = _motion_metrics(composites[start : start + DROID_HORIZON + 1])
        action = _action_stream(rows, start)
        candidates.append(
            {
                "start_frame": start,
                "action_sha256": action["action_sha256"],
                "reference_motion": metrics,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            float(row["reference_motion"]["temporal_absolute_difference_mean_gray_0_255"]),
            -int(row["start_frame"]),
        ),
    )
    selected_start = int(selected["start_frame"])
    recorded = _action_stream(rows, selected_start)
    initial_gripper = 1.0 - float(rows[selected_start]["observation.state.gripper_position"])
    no_motion = _no_motion_stream(gripper_hold=initial_gripper)

    initial_path = output / "initial_observation.png"
    if not cv2.imwrite(str(initial_path), composites[selected_start]):
        raise ValueError("official_droid_initial_observation_write_failed")
    actions = {
        "schema_version": "policy_ranking_cosmos3_official_droid_reference_actions.v1",
        "recorded": recorded,
        "no_motion": no_motion,
    }
    write_json(output / "action_streams.json", actions)

    request_common = {
        "model": "nvidia/Cosmos3-Nano",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "endpoint": "/v1/videos",
        "prompt": " ",
        "num_frames": 17,
        "fps": 15,
        "size": "640x540",
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": 0,
        "extra_params": {
            "action_mode": "forward_dynamics",
            "domain_name": "droid_lerobot",
            "action_chunk_size": 16,
            "image_size": 480,
            "view_point": "concat_view",
            "guardrails": False,
        },
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "prospectively_frozen_before_provider_execution",
        "sources": {
            "cosmos": {"url": COSMOS_REPOSITORY, "revision": COSMOS_REVISION},
            "cosmos_framework": {
                "url": COSMOS_FRAMEWORK_REPOSITORY,
                "revision": COSMOS_FRAMEWORK_REVISION,
            },
            "vllm_omni_current": {
                "url": VLLM_OMNI_REPOSITORY,
                "revision": VLLM_OMNI_CURRENT_REVISION,
            },
            "dataset": {
                "id": COSMOS3_DROID_DATASET,
                "revision": COSMOS3_DROID_DATASET_REVISION,
            },
            "license": LICENSE_ID,
            "asset_sha256": source_hashes,
        },
        "selection": {
            "candidate_chunk_starts": list(CHUNK_STARTS),
            "rule": "maximum matched-real adjacent-frame motion; earliest start breaks ties",
            "selected_start_frame": selected_start,
            "candidate_metrics": candidates,
            "model_outputs_accessed": False,
            "policy_ranking_labels_accessed": False,
        },
        "provider_inputs": {
            "initial_observation_path": "initial_observation.png",
            "initial_observation_sha256": file_sha256(initial_path),
            "action_streams_path": "action_streams.json",
            "action_streams_sha256": canonical_sha256(actions),
            "physical_future_pixels_allowed_in_provider_input": False,
        },
        "request_contract": request_common,
        "requests": [
            {
                "name": "structured_recorded_action_canary",
                "action_sha256": recorded["action_sha256"],
                "maximum_requests": 1,
            },
            {
                "name": "no_motion_causal_followup",
                "action_sha256": no_motion["action_sha256"],
                "maximum_requests": 1,
                "admitted_only_after_structured_canary_passes": True,
            },
        ],
        "frozen_gates": {
            "structured_canary": {
                "provider_response_id_required": True,
                "terminal_status": "completed",
                "output_width": 640,
                "output_height": 540,
                "output_frames": 17,
                "output_fps": 15,
                "temporal_absolute_difference_mean_minimum_gray_0_255": 1.0,
                "first_to_last_absolute_difference_mean_minimum_gray_0_255": 3.0,
            },
            "causal_followup": {
                "recorded_must_exceed_no_motion_timing_correlation": True,
                "recorded_must_not_trigger_static_under_command": True,
                "session_reliability_required": True,
            },
        },
        "runtime": {
            "image": f"{VLLM_IMAGE}@{VLLM_IMAGE_DIGEST}",
            "maximum_concurrent_gpus": 1,
            "paid_execution_admitted": False,
            "provider_called": False,
        },
        "claim_boundary": (
            "A pass validates the pinned deployment on NVIDIA's published DROID sample only. "
            "It does not establish untouched-session generalization, policy ranking, captured-site "
            "transfer, or physical performance."
        ),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(output / "canary_manifest.json", manifest)
    return manifest


__all__ = [
    "CHUNK_STARTS",
    "COSMOS3_DROID_DATASET_REVISION",
    "COSMOS_FRAMEWORK_REVISION",
    "COSMOS_REVISION",
    "EXPERIMENT_ID",
    "SCHEMA_VERSION",
    "UPSTREAM_ASSET_SHA256",
    "build_official_droid_reference_canary",
]
