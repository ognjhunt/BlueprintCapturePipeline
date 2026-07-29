"""Build a label-blind high-motion first-chunk selection manifest.

The selector is reusable across frozen RoboArena/DROID snapshots.  It validates
the split digest, opens only the restricted numeric NPZ action records, converts
the first 17 Cartesian states into the pinned 16x10 Cosmos/DROID action
contract, and delegates the final choice to the prospectively frozen
action-only rule.  Session metadata, outcomes, video pixels, generated media,
and evaluator predictions are never opened.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .policy_ranking_label_free_chunk_selection import (
    select_first_frame_high_motion_pair,
    select_first_frame_windows_by_session,
)
from .policy_ranking_successor_cosmos import (
    DROID_HORIZON,
    convert_droid_states_to_action_stream,
)
from .policy_ranking_thesis import canonical_sha256, file_sha256
from .policy_ranking_wam_validity import load_restricted_roboarena_npz


SCHEMA_VERSION = "policy_ranking_phase_b_high_motion_selection.v1"
SUPPORTED_SPLIT_SCHEMAS = frozenset(
    {
        "policy_ranking_disjoint_session_candidate_split.v1",
        "policy_ranking_disjoint_session_candidate_split_amendment.v2",
        "policy_ranking_disjoint_session_candidate_split_amendment.v3",
    }
)


def _validated_split(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in SUPPORTED_SPLIT_SCHEMAS:
        raise ValueError("split_schema_version_invalid")
    recorded = str(payload.get("manifest_sha256") or "")
    canonical = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    if recorded != canonical:
        raise ValueError("split_manifest_sha256_mismatch")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("split_selection_invalid")
    if selection.get("metadata_yaml_opened") is not False:
        raise ValueError("split_not_label_sealed")
    sessions = selection.get("session_ids")
    policies = payload.get("required_policy_ids")
    if not isinstance(sessions, list) or not sessions or len(sessions) != len(set(sessions)):
        raise ValueError("split_session_ids_invalid")
    if not isinstance(policies, list) or len(policies) < 2 or len(policies) != len(set(policies)):
        raise ValueError("split_policy_ids_invalid")
    return payload


def _policy_directory(session_dir: Path, policy_id: str) -> Path:
    matches = sorted(
        candidate
        for candidate in session_dir.iterdir()
        if candidate.is_dir() and candidate.name.endswith(f"_{policy_id}")
    )
    if len(matches) != 1:
        raise ValueError(f"policy_directory_resolution:{policy_id}:{len(matches)}")
    return matches[0]


def _first_action_stream(npz_path: Path) -> dict[str, Any]:
    arrays = load_restricted_roboarena_npz(npz_path)
    if len(arrays["cartesian_position"]) < DROID_HORIZON + 1:
        raise ValueError("trace_too_short_for_first_action_chunk")
    return convert_droid_states_to_action_stream(
        arrays["cartesian_position"][: DROID_HORIZON + 1],
        arrays["action"][:DROID_HORIZON, 7],
        source_gripper_action_flipped=True,
    )


def build_high_motion_selection(
    *, split_manifest_path: str | Path, dataset_root: str | Path
) -> dict[str, Any]:
    """Select the strongest first chunk without opening labels or pixels."""

    split_path = Path(split_manifest_path).resolve()
    root = Path(dataset_root).resolve()
    split = _validated_split(split_path)
    session_ids = [str(value) for value in split["selection"]["session_ids"]]
    policy_ids = [str(value) for value in split["required_policy_ids"]]
    candidates: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for session_id in session_ids:
        session_dir = root / "evaluation_sessions" / session_id
        if not session_dir.is_dir():
            raise ValueError(f"session_directory_missing:{session_id}")
        for policy_id in policy_ids:
            policy_dir = _policy_directory(session_dir, policy_id)
            npz_files = sorted(policy_dir.glob("*_npz_file.npz"))
            if len(npz_files) != 1:
                raise ValueError(f"npz_resolution:{session_id}:{policy_id}:{len(npz_files)}")
            npz_path = npz_files[0]
            action_stream = _first_action_stream(npz_path)
            candidates.append(
                {
                    "session_id": session_id,
                    "policy_id": policy_id,
                    "start_index": 0,
                    "action_stream": action_stream,
                }
            )
            source_rows.append(
                {
                    "session_id_internal_only": session_id,
                    "policy_id_internal_only": policy_id,
                    "npz_path_relative_to_dataset_root": str(npz_path.relative_to(root)),
                    "npz_sha256": file_sha256(npz_path),
                    "npz_bytes": npz_path.stat().st_size,
                    "restricted_loader_used": True,
                    "numpy_allow_pickle_used": False,
                    "selected_chunk_start_index": 0,
                    "selected_chunk_action_sha256": action_stream["action_sha256"],
                }
            )

    expected_count = len(session_ids) * len(policy_ids)
    if len(candidates) != expected_count:
        raise ValueError("candidate_matrix_incomplete")
    selection = select_first_frame_high_motion_pair(candidates)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "dataset": split.get("dataset"),
        "split_manifest_sha256": split["manifest_sha256"],
        "split_manifest_file_sha256": file_sha256(split_path),
        "selected_session_count": len(session_ids),
        "required_policy_count": len(policy_ids),
        "candidate_count": len(candidates),
        "source_rows": source_rows,
        "selection": selection,
        "input_access": {
            "restricted_numeric_npz_actions_opened": True,
            "metadata_yaml_opened": False,
            "outcome_labels_accessed": False,
            "video_pixels_opened": False,
            "generated_media_accessed": False,
            "evaluator_predictions_accessed": False,
        },
        "claim_boundary": (
            "label_blind_first_frame_action_canary_selection_only; not WAM qualification, "
            "policy ranking, task success, or independent confirmation"
        ),
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def build_powered_window_selection(
    *,
    split_manifest_path: str | Path,
    dataset_root: str | Path,
    windows_per_session: int = 3,
) -> dict[str, Any]:
    """Freeze multiple label-blind first-frame windows for each session.

    This is the powered causal-screen input selector. It deliberately chooses
    separate policies at their recorded first frame instead of later temporal
    offsets, because the source review videos are 10 FPS while the upstream
    DROID action contract is 15 Hz. No guessed cross-rate frame mapping enters
    the scientific packet.
    """

    split_path = Path(split_manifest_path).resolve()
    root = Path(dataset_root).resolve()
    split = _validated_split(split_path)
    session_ids = [str(value) for value in split["selection"]["session_ids"]]
    policy_ids = [str(value) for value in split["required_policy_ids"]]
    candidates: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for session_id in session_ids:
        session_dir = root / "evaluation_sessions" / session_id
        if not session_dir.is_dir():
            raise ValueError(f"session_directory_missing:{session_id}")
        for policy_id in policy_ids:
            policy_dir = _policy_directory(session_dir, policy_id)
            npz_files = sorted(policy_dir.glob("*_npz_file.npz"))
            if len(npz_files) != 1:
                raise ValueError(f"npz_resolution:{session_id}:{policy_id}:{len(npz_files)}")
            npz_path = npz_files[0]
            action_stream = _first_action_stream(npz_path)
            candidates.append(
                {
                    "session_id": session_id,
                    "policy_id": policy_id,
                    "start_index": 0,
                    "action_stream": action_stream,
                }
            )
            source_rows.append(
                {
                    "session_id_internal_only": session_id,
                    "policy_id_internal_only": policy_id,
                    "npz_path_relative_to_dataset_root": str(npz_path.relative_to(root)),
                    "npz_sha256": file_sha256(npz_path),
                    "npz_bytes": npz_path.stat().st_size,
                    "restricted_loader_used": True,
                    "numpy_allow_pickle_used": False,
                    "selected_chunk_start_index": 0,
                    "selected_chunk_action_sha256": action_stream["action_sha256"],
                }
            )

    expected_count = len(session_ids) * len(policy_ids)
    if len(candidates) != expected_count:
        raise ValueError("candidate_matrix_incomplete")
    selection = select_first_frame_windows_by_session(
        candidates,
        windows_per_session=windows_per_session,
    )
    result: dict[str, Any] = {
        "schema_version": "policy_ranking_phase_b_powered_window_selection.v1",
        "status": "passed",
        "dataset": split.get("dataset"),
        "split_manifest_sha256": split["manifest_sha256"],
        "split_manifest_file_sha256": file_sha256(split_path),
        "selected_session_count": len(session_ids),
        "required_policy_count": len(policy_ids),
        "windows_per_session": windows_per_session,
        "candidate_count": len(candidates),
        "source_rows": source_rows,
        "selection": selection,
        "temporal_alignment": {
            "source_review_video_fps": 10,
            "wam_action_fps": 15,
            "selected_action_start_index": 0,
            "selected_video_frame_index": 0,
            "cross_rate_frame_mapping_invented": False,
        },
        "input_access": {
            "restricted_numeric_npz_actions_opened": True,
            "metadata_yaml_opened": False,
            "outcome_labels_accessed": False,
            "video_pixels_opened": False,
            "generated_media_accessed": False,
            "evaluator_predictions_accessed": False,
        },
        "claim_boundary": (
            "label_blind_powered_causal_input_selection_only; not WAM qualification, "
            "policy ranking, task success, or independent new-snapshot confirmation"
        ),
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--windows-per-session",
        type=int,
        help="Build the powered per-session selection instead of the legacy global pair.",
    )
    args = parser.parse_args(argv)
    if args.windows_per_session is None:
        report = build_high_motion_selection(
            split_manifest_path=args.split_manifest,
            dataset_root=args.dataset_root,
        )
    else:
        report = build_powered_window_selection(
            split_manifest_path=args.split_manifest,
            dataset_root=args.dataset_root,
            windows_per_session=args.windows_per_session,
        )
    write_json(Path(args.output), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "build_high_motion_selection",
    "build_powered_window_selection",
]
