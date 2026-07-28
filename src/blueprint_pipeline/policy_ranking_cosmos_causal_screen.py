"""Label-blind causal screening for a frozen Cosmos action-condition matrix.

This screen deliberately cannot qualify the WAM from one DROID session.  It
measures whether same-seed videos respond to the frozen action controls and
whether scene motion is temporally aligned with the action supplied to each
video more strongly than with the other frozen action traces.  The result is a
screening observation; the independently published outcome labels and the
policy evaluator are never read.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np

from .common import write_json
from .policy_ranking_successor_cosmos import canonical_sha256, validate_smoke_inventory_manifest
from .policy_ranking_thesis import file_sha256


SCHEMA_VERSION = "policy_ranking_cosmos_causal_screen.v1"
CONDITIONS = ("recorded", "zero", "shuffled", "reversed", "policy_swapped")
ACTIVE_CONDITIONS = tuple(condition for condition in CONDITIONS if condition != "zero")
EXPECTED_SEEDS = (0, 1)
TEMPORAL_CORRELATION_MINIMUM = 0.10
TEMPORAL_EXCESS_MARGIN = 0.05
SAME_SEED_SCENE_DIFFERENCE_MINIMUM = 0.01
MINIMUM_INDEPENDENT_SESSIONS = 17
FOLLOWUP_EXPERIMENT_ID = "policy_ranking_cosmos3_followup_20260728"


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) < 3:
        return 0.0
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def action_intensity(actions: Sequence[Sequence[float]]) -> np.ndarray:
    """Reduce the documented 10D midtrain action to a 16-step motion signal."""

    value = np.asarray(actions, dtype=np.float64)
    if value.shape != (16, 10):
        raise ValueError(f"action_shape_invalid:{value.shape}")
    if not np.any(value):
        return np.zeros(16, dtype=np.float64)
    rotation_identity = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    translation = np.linalg.norm(value[:, :3], axis=1)
    rotation = np.linalg.norm(value[:, 3:9] - rotation_identity, axis=1)
    gripper = np.abs(np.diff(value[:, 9], prepend=value[0, 9]))
    return translation + rotation + gripper


def _decode_scene(path: Path) -> tuple[np.ndarray, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("video_open_failed")
    color_frames: list[np.ndarray] = []
    gray_frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            resized = cv2.resize(frame, (160, 136), interpolation=cv2.INTER_AREA)
            color_frames.append(resized.astype(np.float32) / 255.0)
            gray_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    if len(color_frames) != 17:
        raise ValueError(f"video_frame_count_invalid:{len(color_frames)}")
    return np.asarray(color_frames), np.asarray(gray_frames)


def camera_compensated_motion(gray_frames: np.ndarray) -> np.ndarray:
    """Measure residual dense flow after subtracting global median camera flow."""

    if gray_frames.shape[0] != 17:
        raise ValueError("gray_frame_count_invalid")
    values: list[float] = []
    for previous, current in zip(gray_frames[:-1], gray_frames[1:], strict=True):
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        global_flow = np.median(flow.reshape(-1, 2), axis=0)
        residual = flow - global_flow
        magnitude = np.linalg.norm(residual, axis=2)
        values.append(float(np.mean(magnitude)))
    return np.asarray(values, dtype=np.float64)


def build_causal_screen(
    *,
    runtime_output_root: str | Path,
    action_streams: Mapping[str, Any],
    smoke_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(runtime_output_root).resolve()
    conditions = action_streams.get("conditions")
    blockers: list[str] = []
    if action_streams.get("experiment_id") != FOLLOWUP_EXPERIMENT_ID:
        blockers.append("action_streams_experiment_invalid")
    if smoke_inventory.get("experiment_id") != FOLLOWUP_EXPERIMENT_ID:
        blockers.append("smoke_inventory_experiment_invalid")
    try:
        validate_smoke_inventory_manifest(smoke_inventory)
    except ValueError as exc:
        blockers.append(f"smoke_inventory_invalid:{type(exc).__name__}")
    inventory_rows = smoke_inventory.get("requests")
    expected_by_id = {
        str(row.get("request_id")): row
        for row in inventory_rows
        if isinstance(row, Mapping) and row.get("request_id")
    } if isinstance(inventory_rows, Sequence) else {}
    if len(expected_by_id) != 10:
        blockers.append("smoke_inventory_request_matrix_invalid")
    if not isinstance(conditions, Mapping) or set(conditions) != set(CONDITIONS):
        blockers.append("action_conditions_invalid")
        conditions = {}
    action_signals: dict[str, np.ndarray] = {}
    frozen_action_hashes = smoke_inventory.get("action_hashes")
    for condition in CONDITIONS:
        try:
            row = conditions[condition]
            action_sha256 = canonical_sha256(row["actions"])
            if not isinstance(frozen_action_hashes, Mapping) or (
                row.get("action_sha256") != action_sha256
                or frozen_action_hashes.get(condition) != action_sha256
            ):
                raise ValueError("action_hash_binding_invalid")
            action_signals[condition] = action_intensity(row["actions"])
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(f"action_signal_invalid:{condition}:{type(exc).__name__}")

    records: dict[tuple[str, int], dict[str, Any]] = {}
    observed_request_ids: set[str] = set()
    responses_root = root / "responses"
    for response_path in sorted(responses_root.glob("*.json")):
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
            condition = str(response["condition"])
            seed = int(response["seed"])
            request_id = str(response["request_id"])
            pair = (condition, seed)
            if request_id in observed_request_ids:
                raise ValueError("duplicate_request_id")
            if pair in records:
                raise ValueError("duplicate_condition_seed")
            expected_row = expected_by_id.get(request_id)
            if not isinstance(expected_row, Mapping):
                raise ValueError("request_id_not_in_frozen_inventory")
            expected_bindings = {
                "experiment_id": FOLLOWUP_EXPERIMENT_ID,
                "request_id": request_id,
                "condition": str(expected_row.get("condition")),
                "seed": int(expected_row.get("seed")),
                "action_sha256": str(expected_row.get("action_sha256")),
                "initial_observation_sha256": str(
                    smoke_inventory.get("initial_observation_sha256")
                ),
                "task_instruction": str(smoke_inventory.get("task_instruction")),
            }
            if response_path.stem != request_id or any(
                response.get(key) != value for key, value in expected_bindings.items()
            ):
                raise ValueError("response_frozen_binding_mismatch")
            if response.get("accepted_first_valid") is not True:
                raise ValueError("response_not_accepted_first_valid")
            if response.get("generated_media_valid") is not True:
                raise ValueError("response_generated_media_invalid")
            video_path = root / "videos" / f"{request_id}.mp4"
            color, gray = _decode_scene(video_path)
            video_sha256 = file_sha256(video_path)
            provider_response = response.get("response")
            if not isinstance(provider_response, Mapping) or (
                provider_response.get("output_sha256") != video_sha256
            ):
                raise ValueError("response_video_hash_mismatch")
            observed_request_ids.add(request_id)
            records[pair] = {
                "request_id": request_id,
                "video_path": video_path,
                "video_sha256": video_sha256,
                "color": color,
                "motion": camera_compensated_motion(gray),
            }
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            blockers.append(
                f"response_invalid:{response_path.name}:{type(exc).__name__}:{str(exc)[:120]}"
            )
    expected = {(condition, seed) for condition in CONDITIONS for seed in EXPECTED_SEEDS}
    if set(records) != expected:
        blockers.append(f"response_matrix_invalid:{len(records)}")
    if observed_request_ids != set(expected_by_id):
        blockers.append(f"response_request_inventory_mismatch:{len(observed_request_ids)}")

    rows: list[dict[str, Any]] = []
    same_seed_scene_rows: list[dict[str, Any]] = []
    if not blockers:
        for seed in EXPECTED_SEEDS:
            zero = records[("zero", seed)]["color"]
            for condition in ACTIVE_CONDITIONS:
                item = records[(condition, seed)]
                difference = np.mean(np.abs(item["color"] - zero), axis=(1, 2, 3))
                mean_difference = float(np.mean(difference))
                same_seed_scene_rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "against": "zero",
                        "mean_absolute_scene_difference": mean_difference,
                        "maximum_absolute_scene_difference": float(np.max(difference)),
                        "scene_response_pass": (
                            item["video_sha256"] != records[("zero", seed)]["video_sha256"]
                            and mean_difference >= SAME_SEED_SCENE_DIFFERENCE_MINIMUM
                        ),
                    }
                )
                own = _correlation(action_signals[condition], item["motion"])
                placebo = {
                    other: _correlation(action_signals[other], item["motion"])
                    for other in ACTIVE_CONDITIONS
                    if other != condition
                }
                strongest = max(placebo.values())
                excess = own - strongest
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "request_id": item["request_id"],
                        "video_sha256": item["video_sha256"],
                        "camera_compensated_motion_mean": float(np.mean(item["motion"])),
                        "own_action_temporal_correlation": own,
                        "placebo_action_correlations": placebo,
                        "strongest_placebo_correlation": strongest,
                        "excess_over_strongest_placebo": excess,
                        "temporal_placebo_rejection_pass": (
                            own >= TEMPORAL_CORRELATION_MINIMUM and excess >= TEMPORAL_EXCESS_MARGIN
                        ),
                    }
                )

    scene_response_pass = bool(same_seed_scene_rows) and all(
        row["scene_response_pass"] for row in same_seed_scene_rows
    )
    temporal_pass = bool(rows) and all(row["temporal_placebo_rejection_pass"] for row in rows)
    seed_robustness = {
        condition: all(
            row["temporal_placebo_rejection_pass"] for row in rows if row["condition"] == condition
        )
        and len([row for row in rows if row["condition"] == condition]) == 2
        for condition in ACTIVE_CONDITIONS
    }
    all_seed_robust = bool(seed_robustness) and all(seed_robustness.values())
    screen_passed = not blockers and scene_response_pass and temporal_pass and all_seed_robust
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if not blockers else "blocked",
        "role": "screening_not_confirmatory",
        "thresholds_frozen_before_paid_output": {
            "temporal_correlation_minimum": TEMPORAL_CORRELATION_MINIMUM,
            "excess_over_strongest_placebo_minimum": TEMPORAL_EXCESS_MARGIN,
            "same_seed_scene_difference_minimum": SAME_SEED_SCENE_DIFFERENCE_MINIMUM,
            "minimum_independent_sessions": MINIMUM_INDEPENDENT_SESSIONS,
        },
        "measurement": {
            "scene_motion": "dense_farneback_flow_minus_spatial_median_global_flow",
            "action_signal": "translation_norm_plus_rot6d_identity_deviation_plus_gripper_transition",
            "same_seed_control": "each_active_condition_against_zero_action_video",
        },
        "rows": rows,
        "same_seed_scene_rows": same_seed_scene_rows,
        "gates": {
            "nonidentical_scene_response": scene_response_pass,
            "directional_temporal_placebo_rejection": temporal_pass,
            "seed_robustness": all_seed_robust,
        },
        "seed_robustness_by_condition": seed_robustness,
        "screen_passed": screen_passed,
        "independent_session_count": 1,
        "confirmatory_power_sufficient": False,
        "powered_causal_gate_passed": False,
        "cosmos_wam_qualification": "inconclusive",
        "evaluator_eligible": False,
        "benchmark_labels_seen": False,
        "policy_identity_seen": False,
        "task_success_scored": False,
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "A one-session label-blind falsification screen cannot establish the frozen "
            "95-percent causal-validity gates, even if every descriptive screen passes."
        ),
    }
    result["report_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-output-root", required=True)
    parser.add_argument("--action-streams", required=True)
    parser.add_argument("--smoke-inventory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build_causal_screen(
        runtime_output_root=args.runtime_output_root,
        action_streams=json.loads(Path(args.action_streams).read_text(encoding="utf-8")),
        smoke_inventory=json.loads(Path(args.smoke_inventory).read_text(encoding="utf-8")),
    )
    write_json(Path(args.output), report)
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
