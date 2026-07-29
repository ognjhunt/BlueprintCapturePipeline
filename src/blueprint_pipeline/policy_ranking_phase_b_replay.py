"""Build a label-blind, real-trace Phase-B replay canary.

This module converts RoboArena joint-velocity episode records into the pinned
Cosmos DROID midtrain action representation using recorded Cartesian states.
It also emits valid no-motion, shuffled, reversed, temporally shifted, and real
policy-swapped controls. It never opens session metadata or outcome labels.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np

from .common import ensure_dir, write_json
from .policy_ranking_successor_cosmos import (
    DROID_HORIZON,
    build_action_controls,
    convert_droid_states_to_action_stream,
    validate_droid_action_stream,
)
from .policy_ranking_thesis import canonical_sha256, file_sha256
from .policy_ranking_wam_validity import load_restricted_roboarena_npz


SCHEMA_VERSION = "policy_ranking_phase_b_replay_canary.v1"
SHUFFLE_SEED = 20260728


def _policy_dir(session_dir: Path, policy_id: str) -> Path:
    matches = [
        path
        for path in session_dir.iterdir()
        if path.is_dir() and path.name.endswith(f"_{policy_id}")
    ]
    if len(matches) != 1:
        raise ValueError(f"policy_directory_resolution:{policy_id}:{len(matches)}")
    return matches[0]


def _action_stream(arrays: Mapping[str, np.ndarray], start: int) -> dict[str, Any]:
    stop = start + DROID_HORIZON
    states = arrays["cartesian_position"][start : stop + 1]
    gripper = arrays["action"][start:stop, 7]
    return convert_droid_states_to_action_stream(
        states,
        gripper,
        source_gripper_action_flipped=True,
    )


def _first_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("video_open_failed")
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError("video_first_frame_decode_failed")
    return frame


def compose_initial_observation(policy_dir: Path, output: Path) -> dict[str, Any]:
    """Compose upstream DROID wrist/left/right geometry from real first frames."""

    inputs: dict[str, Path] = {}
    for view in ("wrist", "left", "right"):
        matches = sorted(policy_dir.glob(f"*_video_{view}.mp4"))
        if len(matches) != 1:
            raise ValueError(f"video_resolution:{view}:{len(matches)}")
        inputs[view] = matches[0]
    wrist = cv2.resize(_first_frame(inputs["wrist"]), (640, 360), interpolation=cv2.INTER_AREA)
    left = cv2.resize(_first_frame(inputs["left"]), (320, 180), interpolation=cv2.INTER_AREA)
    right = cv2.resize(_first_frame(inputs["right"]), (320, 180), interpolation=cv2.INTER_AREA)
    composed = np.zeros((540, 640, 3), dtype=np.uint8)
    composed[:360, :] = wrist
    composed[360:, :320] = left
    composed[360:, 320:] = right
    ensure_dir(output.parent)
    if not cv2.imwrite(str(output), composed):
        raise ValueError("initial_observation_write_failed")
    return {
        "path": str(output.resolve()),
        "sha256": file_sha256(output),
        "shape": [540, 640, 3],
        "color_space": "BGR_encoded_as_PNG; decoder_yields_RGB_equivalent_pixels",
        "composition": "wrist_full_width_top; left_bottom_left; right_bottom_right",
        "source_video_sha256": {view: file_sha256(path) for view, path in inputs.items()},
    }


def materialize_initial_view_frames(policy_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Preserve reviewable per-camera first frames beside the native composite."""

    ensure_dir(output_dir)
    result: dict[str, Any] = {}
    for view in ("left", "right", "wrist"):
        matches = sorted(policy_dir.glob(f"*_video_{view}.mp4"))
        if len(matches) != 1:
            raise ValueError(f"video_resolution:{view}:{len(matches)}")
        source = matches[0]
        output = output_dir / f"initial_{view}.png"
        if not cv2.imwrite(str(output), _first_frame(source)):
            raise ValueError(f"initial_view_write_failed:{view}")
        result[view] = {
            "path": str(output.resolve()),
            "sha256": file_sha256(output),
            "source_video_sha256": file_sha256(source),
            "source_video_bytes": source.stat().st_size,
        }
    return result


def _validated_manifest(path: Path, *, digest_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(payload.get(digest_field) or "")
    canonical = canonical_sha256(
        {key: value for key, value in payload.items() if key != digest_field}
    )
    if not recorded or recorded != canonical:
        raise ValueError(f"{digest_field}_mismatch")
    return payload


def build_selected_replay_canary(
    *,
    high_motion_selection_path: str | Path,
    task_instruction_receipt_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Materialize the selected fairer canary without loading outcome fields."""

    selection_path = Path(high_motion_selection_path).resolve()
    receipt_path = Path(task_instruction_receipt_path).resolve()
    selection_manifest = _validated_manifest(selection_path, digest_field="manifest_sha256")
    task_receipt = _validated_manifest(receipt_path, digest_field="receipt_sha256")
    if selection_manifest.get("status") != "passed":
        raise ValueError("high_motion_selection_not_passed")
    access = selection_manifest.get("input_access")
    if not isinstance(access, Mapping) or any(
        access.get(field) is not False
        for field in (
            "metadata_yaml_opened",
            "outcome_labels_accessed",
            "video_pixels_opened",
            "generated_media_accessed",
            "evaluator_predictions_accessed",
        )
    ):
        raise ValueError("high_motion_selection_not_label_blind")
    if task_receipt.get("status") != "passed":
        raise ValueError("task_instruction_receipt_not_passed")
    task_access = task_receipt.get("access_contract")
    if not isinstance(task_access, Mapping) or any(
        task_access.get(field) is not False
        for field in (
            "yaml_document_deserialized",
            "outcome_fields_parsed",
            "outcome_values_returned",
        )
    ):
        raise ValueError("task_instruction_receipt_access_contract_invalid")

    selected = selection_manifest.get("selection")
    if not isinstance(selected, Mapping):
        raise ValueError("high_motion_selection_payload_missing")
    recorded = selected.get("recorded")
    swapped = selected.get("policy_swapped")
    if not isinstance(recorded, Mapping) or not isinstance(swapped, Mapping):
        raise ValueError("high_motion_selected_pair_missing")
    session_id = str(recorded.get("session_id_internal_only") or "")
    recorded_policy = str(recorded.get("policy_id_internal_only") or "")
    swapped_policy = str(swapped.get("policy_id_internal_only") or "")
    if (
        not session_id
        or session_id != str(swapped.get("session_id_internal_only") or "")
        or session_id != str(task_receipt.get("session_id_internal_only") or "")
        or not recorded_policy
        or not swapped_policy
        or recorded_policy == swapped_policy
    ):
        raise ValueError("selected_pair_or_task_session_binding_invalid")

    root = Path(dataset_root).resolve()
    session_dir = root / "evaluation_sessions" / session_id
    recorded_dir = _policy_dir(session_dir, recorded_policy)
    swapped_dir = _policy_dir(session_dir, swapped_policy)
    recorded_npz_files = sorted(recorded_dir.glob("*_npz_file.npz"))
    swapped_npz_files = sorted(swapped_dir.glob("*_npz_file.npz"))
    if len(recorded_npz_files) != 1 or len(swapped_npz_files) != 1:
        raise ValueError("selected_replay_npz_resolution_invalid")
    recorded_npz, swapped_npz = recorded_npz_files[0], swapped_npz_files[0]
    recorded_arrays = load_restricted_roboarena_npz(recorded_npz)
    swapped_arrays = load_restricted_roboarena_npz(swapped_npz)
    if len(recorded_arrays["action"]) < DROID_HORIZON + 1:
        raise ValueError("recorded_trace_too_short_for_temporal_shift")
    if len(swapped_arrays["action"]) < DROID_HORIZON:
        raise ValueError("policy_swapped_trace_too_short")

    recorded_stream = _action_stream(recorded_arrays, 0)
    swapped_stream = _action_stream(swapped_arrays, 0)
    if recorded_stream["action_sha256"] != recorded.get("action_stream", {}).get("action_sha256"):
        raise ValueError("recorded_action_does_not_match_selection")
    if swapped_stream["action_sha256"] != swapped.get("action_stream", {}).get("action_sha256"):
        raise ValueError("policy_swapped_action_does_not_match_selection")

    hold = 1.0 - float(recorded_arrays["gripper_position"][0, 0])
    controls = build_action_controls(
        recorded_stream,
        swapped_stream,
        observation_gripper_hold=hold,
        shuffle_seed=SHUFFLE_SEED,
    )
    shifted = _action_stream(recorded_arrays, 1)
    if shifted["action_sha256"] in {payload["action_sha256"] for payload in controls.values()}:
        raise ValueError("temporally_shifted_action_not_distinct")
    controls["shifted"] = validate_droid_action_stream(shifted)
    if len({payload["action_sha256"] for payload in controls.values()}) != len(controls):
        raise ValueError("action_controls_not_pairwise_distinct")

    out = Path(output_dir).resolve()
    ensure_dir(out)
    initial_views = materialize_initial_view_frames(recorded_dir, out / "initial_views")
    observation = compose_initial_observation(recorded_dir, out / "initial_observation.png")
    result: dict[str, Any] = {
        "schema_version": "policy_ranking_phase_b_selected_replay_canary.v2",
        "status": "passed",
        "high_motion_selection_manifest_sha256": selection_manifest["manifest_sha256"],
        "high_motion_selection_file_sha256": file_sha256(selection_path),
        "task_instruction_receipt_sha256": task_receipt["receipt_sha256"],
        "task_instruction_receipt_file_sha256": file_sha256(receipt_path),
        "task_instruction": task_receipt["task_instruction"],
        "task_instruction_sha256": task_receipt["task_instruction_sha256"],
        "session_id_internal_only": session_id,
        "recorded_policy_id_internal_only": recorded_policy,
        "swapped_policy_id_internal_only": swapped_policy,
        "source": {
            "recorded_npz_sha256": file_sha256(recorded_npz),
            "swapped_npz_sha256": file_sha256(swapped_npz),
            "roboarena_action_space": "joint_velocity_7_plus_binary_gripper_1",
            "cosmos_conversion": "recorded_cartesian_states_to_backward_framewise_rot6d_10d",
            "source_gripper_index": 7,
            "source_gripper_action_flipped_for_cosmos": True,
            "no_motion_gripper_hold_source": "one_minus_initial_observation_gripper_position",
        },
        "selected_action_metrics": {
            "recorded": recorded.get("metrics"),
            "policy_swapped": swapped.get("metrics"),
        },
        "initial_views": initial_views,
        "initial_observation": observation,
        "controls": controls,
        "control_action_sha256": {
            name: payload["action_sha256"] for name, payload in controls.items()
        },
        "conditioning_modes": {
            "native_cosmos": "three_camera_composite_first_frame_only",
            "oscar_matched": "individual_camera_first_frame_plus_camera_aligned_skeleton_video",
            "starter_video_supported_by_pinned_native_action_api": False,
        },
        "access_contract": {
            "selection_used_outcomes": False,
            "outcome_fields_parsed_for_task_prompt": False,
            "physical_future_pixels_in_provider_input": False,
            "policy_identity_in_provider_payload": False,
        },
        "claim_boundary": (
            "already_exposed_same_snapshot_diagnostic causal canary; not independent Phase B, "
            "policy ranking, task success, or physical deployment"
        ),
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def build_replay_canary(
    *, split_manifest_path: str | Path, dataset_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    split_path = Path(split_manifest_path).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    recorded_digest = str(split.get("manifest_sha256") or "")
    canonical = canonical_sha256(
        {key: value for key, value in split.items() if key != "manifest_sha256"}
    )
    if recorded_digest != canonical:
        raise ValueError("split_manifest_sha256_mismatch")
    sessions = split["selection"]["session_ids"]
    policies = split["required_policy_ids"]
    if len(sessions) < 1 or len(policies) < 2:
        raise ValueError("replay_canary_requires_one_session_and_two_policies")
    session_id = str(sessions[0])
    recorded_policy = str(policies[0])
    swapped_policy = str(policies[1])
    session_dir = Path(dataset_root).resolve() / "evaluation_sessions" / session_id
    recorded_dir = _policy_dir(session_dir, recorded_policy)
    swapped_dir = _policy_dir(session_dir, swapped_policy)
    recorded_npz = next(iter(sorted(recorded_dir.glob("*_npz_file.npz"))), None)
    swapped_npz = next(iter(sorted(swapped_dir.glob("*_npz_file.npz"))), None)
    if recorded_npz is None or swapped_npz is None:
        raise ValueError("replay_canary_npz_missing")
    recorded_arrays = load_restricted_roboarena_npz(recorded_npz)
    swapped_arrays = load_restricted_roboarena_npz(swapped_npz)
    if len(recorded_arrays["action"]) < DROID_HORIZON + 1:
        raise ValueError("recorded_trace_too_short_for_temporal_shift")
    if len(swapped_arrays["action"]) < DROID_HORIZON:
        raise ValueError("policy_swapped_trace_too_short")

    recorded = _action_stream(recorded_arrays, 0)
    swapped = _action_stream(swapped_arrays, 0)
    hold = 1.0 - float(recorded_arrays["gripper_position"][0, 0])
    controls = build_action_controls(
        recorded,
        swapped,
        observation_gripper_hold=hold,
        shuffle_seed=SHUFFLE_SEED,
    )
    shifted = _action_stream(recorded_arrays, 1)
    if shifted["action_sha256"] in {payload["action_sha256"] for payload in controls.values()}:
        raise ValueError("temporally_shifted_action_not_distinct")
    controls["shifted"] = validate_droid_action_stream(shifted)

    out = Path(output_dir).resolve()
    ensure_dir(out)
    observation = compose_initial_observation(recorded_dir, out / "initial_observation.png")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "split_manifest_sha256": recorded_digest,
        "selection_rule": "first frozen session; first policy as recorded; second policy as real swap; first 16-step chunk; one-step temporal shift",
        "session_id_internal_only": session_id,
        "recorded_policy_id_internal_only": recorded_policy,
        "swapped_policy_id_internal_only": swapped_policy,
        "source": {
            "recorded_npz_sha256": file_sha256(recorded_npz),
            "swapped_npz_sha256": file_sha256(swapped_npz),
            "roboarena_action_space": "joint_velocity_7_plus_binary_gripper_1",
            "cosmos_conversion": "recorded_cartesian_states_to_backward_framewise_rot6d_10d",
            "source_gripper_index": 7,
            "source_gripper_action_flipped_for_cosmos": True,
            "no_motion_gripper_hold_source": "one_minus_initial_observation_gripper_position",
        },
        "initial_observation": observation,
        "controls": controls,
        "control_action_sha256": {
            name: payload["action_sha256"] for name, payload in controls.items()
        },
        "all_control_hashes_distinct": len(
            {payload["action_sha256"] for payload in controls.values()}
        )
        == len(controls),
        "label_seal": {
            "metadata_yaml_opened": False,
            "outcome_labels_accessed": False,
            "task_instruction_accessed": False,
        },
        "claim_boundary": "label_free_action_and_observation_canary_only; no WAM output or ranking",
    }
    if not result["all_control_hashes_distinct"]:
        raise ValueError("action_controls_not_pairwise_distinct")
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def build_powered_replay_packet(
    *,
    powered_window_selection_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Materialize all powered, label-blind native-Cosmos replay inputs."""

    selection_path = Path(powered_window_selection_path).resolve()
    manifest = _validated_manifest(selection_path, digest_field="manifest_sha256")
    if manifest.get("schema_version") != "policy_ranking_phase_b_powered_window_selection.v1":
        raise ValueError("powered_window_selection_schema_invalid")
    if manifest.get("status") != "passed":
        raise ValueError("powered_window_selection_not_passed")
    access = manifest.get("input_access")
    if not isinstance(access, Mapping) or any(
        access.get(field) is not False
        for field in (
            "metadata_yaml_opened",
            "outcome_labels_accessed",
            "video_pixels_opened",
            "generated_media_accessed",
            "evaluator_predictions_accessed",
        )
    ):
        raise ValueError("powered_window_selection_not_label_blind")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("powered_window_selection_payload_missing")
    nested_digest = str(selection.get("selection_sha256") or "")
    if nested_digest != canonical_sha256(
        {key: value for key, value in selection.items() if key != "selection_sha256"}
    ):
        raise ValueError("powered_window_nested_digest_mismatch")

    root = Path(dataset_root).resolve()
    out = Path(output_dir).resolve()
    ensure_dir(out)
    packet_rows: list[dict[str, Any]] = []
    for session in selection.get("sessions", []):
        if not isinstance(session, Mapping):
            raise ValueError("powered_window_session_row_invalid")
        session_id = str(session.get("session_id_internal_only") or "")
        windows = session.get("windows")
        if not session_id or not isinstance(windows, Sequence):
            raise ValueError("powered_window_session_identity_invalid")
        session_dir = root / "evaluation_sessions" / session_id
        for window in windows:
            if not isinstance(window, Mapping):
                raise ValueError("powered_window_row_invalid")
            window_index = int(window.get("window_index", -1))
            recorded = window.get("recorded")
            swapped = window.get("policy_swapped")
            if not isinstance(recorded, Mapping) or not isinstance(swapped, Mapping):
                raise ValueError("powered_window_action_pair_missing")
            recorded_policy = str(recorded.get("policy_id_internal_only") or "")
            swapped_policy = str(swapped.get("policy_id_internal_only") or "")
            if not recorded_policy or not swapped_policy or recorded_policy == swapped_policy:
                raise ValueError("powered_window_policy_pair_invalid")
            recorded_dir = _policy_dir(session_dir, recorded_policy)
            swapped_dir = _policy_dir(session_dir, swapped_policy)
            recorded_npz_files = sorted(recorded_dir.glob("*_npz_file.npz"))
            swapped_npz_files = sorted(swapped_dir.glob("*_npz_file.npz"))
            if len(recorded_npz_files) != 1 or len(swapped_npz_files) != 1:
                raise ValueError("powered_window_npz_resolution_invalid")
            recorded_npz = recorded_npz_files[0]
            swapped_npz = swapped_npz_files[0]
            recorded_arrays = load_restricted_roboarena_npz(recorded_npz)
            swapped_arrays = load_restricted_roboarena_npz(swapped_npz)
            if len(recorded_arrays["action"]) < DROID_HORIZON + 1:
                raise ValueError("powered_recorded_trace_too_short_for_shift")
            if len(swapped_arrays["action"]) < DROID_HORIZON:
                raise ValueError("powered_swapped_trace_too_short")
            recorded_stream = _action_stream(recorded_arrays, 0)
            swapped_stream = _action_stream(swapped_arrays, 0)
            if recorded_stream["action_sha256"] != (recorded.get("action_stream") or {}).get(
                "action_sha256"
            ):
                raise ValueError("powered_recorded_action_selection_mismatch")
            if swapped_stream["action_sha256"] != (swapped.get("action_stream") or {}).get(
                "action_sha256"
            ):
                raise ValueError("powered_swapped_action_selection_mismatch")
            hold = 1.0 - float(recorded_arrays["gripper_position"][0, 0])
            controls = build_action_controls(
                recorded_stream,
                swapped_stream,
                observation_gripper_hold=hold,
                shuffle_seed=SHUFFLE_SEED,
            )
            shifted = _action_stream(recorded_arrays, 1)
            if shifted["action_sha256"] in {
                payload["action_sha256"] for payload in controls.values()
            }:
                raise ValueError("powered_temporally_shifted_action_not_distinct")
            controls["shifted"] = validate_droid_action_stream(shifted)
            if len({payload["action_sha256"] for payload in controls.values()}) != len(controls):
                raise ValueError("powered_action_controls_not_pairwise_distinct")

            window_dir = out / "sessions" / session_id / f"window_{window_index:02d}"
            observation = compose_initial_observation(
                recorded_dir,
                window_dir / "initial_observation.png",
            )
            initial_views = materialize_initial_view_frames(
                recorded_dir,
                window_dir / "initial_views",
            )
            packet_rows.append(
                {
                    "session_id_internal_only": session_id,
                    "window_index": window_index,
                    "recorded_policy_id_internal_only": recorded_policy,
                    "swapped_policy_id_internal_only": swapped_policy,
                    "recorded_npz_sha256": file_sha256(recorded_npz),
                    "swapped_npz_sha256": file_sha256(swapped_npz),
                    "initial_observation": observation,
                    "initial_views": initial_views,
                    "controls": controls,
                    "control_action_sha256": {
                        name: payload["action_sha256"] for name, payload in controls.items()
                    },
                    "shuffle_seed": SHUFFLE_SEED,
                }
            )

    session_count = len({row["session_id_internal_only"] for row in packet_rows})
    expected_rows = int(manifest["selected_session_count"]) * int(manifest["windows_per_session"])
    if len(packet_rows) != expected_rows:
        raise ValueError(f"powered_window_packet_incomplete:{len(packet_rows)}:{expected_rows}")
    result: dict[str, Any] = {
        "schema_version": "policy_ranking_phase_b_powered_replay_packet.v1",
        "status": "passed",
        "powered_window_selection_manifest_sha256": manifest["manifest_sha256"],
        "powered_window_selection_file_sha256": file_sha256(selection_path),
        "session_count": session_count,
        "window_count": len(packet_rows),
        "conditions_per_window": 6,
        "seeds_per_condition": 2,
        "scientific_request_count": len(packet_rows) * 6 * 2,
        "task_prompt": " ",
        "rows": packet_rows,
        "label_seal": {
            "metadata_yaml_opened": False,
            "outcome_labels_accessed": False,
            "task_instruction_accessed": False,
            "physical_future_pixels_used_as_provider_input": False,
        },
        "claim_boundary": (
            "label_blind same-snapshot disjoint-session open-loop causal input packet; "
            "not WAM output, policy ranking, live closed loop, or physical performance"
        ),
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest")
    parser.add_argument("--high-motion-selection")
    parser.add_argument("--task-instruction-receipt")
    parser.add_argument("--powered-window-selection")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    powered_mode = bool(args.powered_window_selection)
    selected_mode = bool(args.high_motion_selection or args.task_instruction_receipt)
    if powered_mode:
        if args.split_manifest or selected_mode:
            parser.error("--powered-window-selection cannot be combined with other selectors")
        report = build_powered_replay_packet(
            powered_window_selection_path=args.powered_window_selection,
            dataset_root=args.dataset_root,
            output_dir=output_dir,
        )
    elif selected_mode:
        if not args.high_motion_selection or not args.task_instruction_receipt:
            parser.error(
                "--high-motion-selection and --task-instruction-receipt are required together"
            )
        if args.split_manifest:
            parser.error("--split-manifest cannot be combined with selected-canary inputs")
        report = build_selected_replay_canary(
            high_motion_selection_path=args.high_motion_selection,
            task_instruction_receipt_path=args.task_instruction_receipt,
            dataset_root=args.dataset_root,
            output_dir=output_dir,
        )
    else:
        if not args.split_manifest:
            parser.error("--split-manifest is required for the legacy frozen-first-row mode")
        report = build_replay_canary(
            split_manifest_path=args.split_manifest,
            dataset_root=args.dataset_root,
            output_dir=output_dir,
        )
    filename = "powered_replay_packet.json" if powered_mode else "replay_canary.json"
    write_json(output_dir / filename, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
