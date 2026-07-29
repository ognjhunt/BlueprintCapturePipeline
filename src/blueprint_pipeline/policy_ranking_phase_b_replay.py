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
    if shifted["action_sha256"] in {
        payload["action_sha256"] for payload in controls.values()
    }:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_replay_canary(
        split_manifest_path=args.split_manifest,
        dataset_root=args.dataset_root,
        output_dir=output_dir,
    )
    write_json(output_dir / "replay_canary.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
