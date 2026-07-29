"""Label-blind technical preflight for disjoint RoboArena session splits.

The preflight intentionally never opens ``metadata.yaml``. It validates only
the frozen split, policy-directory identities, media containers, and restricted
numeric NPZ action records. Outcome unsealing remains a separate later event.
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
from .policy_ranking_thesis import canonical_sha256, file_sha256
from .policy_ranking_wam_validity import load_restricted_roboarena_npz


SCHEMA_VERSION = "policy_ranking_disjoint_technical_preflight.v1"
REQUIRED_VIEWS = ("left", "right", "wrist")
MIN_ACTION_STEPS = 17


def _validated_split(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {
        "policy_ranking_disjoint_session_candidate_split.v1",
        "policy_ranking_disjoint_session_candidate_split_amendment.v2",
        "policy_ranking_disjoint_session_candidate_split_amendment.v3",
    }:
        raise ValueError("split_schema_version_invalid")
    recorded = str(payload.get("manifest_sha256") or "")
    canonical = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    if recorded != canonical:
        raise ValueError("split_manifest_sha256_mismatch")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping) or selection.get("metadata_yaml_opened") is not False:
        raise ValueError("split_not_label_sealed")
    sessions = selection.get("session_ids")
    policies = payload.get("required_policy_ids")
    if not isinstance(sessions, list) or not sessions or len(sessions) != len(set(sessions)):
        raise ValueError("split_session_ids_invalid")
    if not isinstance(policies, list) or not policies or len(policies) != len(set(policies)):
        raise ValueError("split_policy_ids_invalid")
    return payload


def _video_probe(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("video_open_failed")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok_first, first = capture.read()
        if frame_count > 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ok_last, last = capture.read()
    finally:
        capture.release()
    if (
        frame_count < 2
        or not np.isfinite(fps)
        or fps <= 0
        or width <= 0
        or height <= 0
        or not ok_first
        or first is None
        or not ok_last
        or last is None
    ):
        raise ValueError("video_probe_invalid")
    return {
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "first_frame_decoded": True,
        "last_frame_decoded": True,
    }


def _policy_directory(session_dir: Path, policy_id: str) -> Path:
    matches = sorted(
        candidate
        for candidate in session_dir.iterdir()
        if candidate.is_dir() and candidate.name.endswith(f"_{policy_id}")
    )
    if len(matches) != 1:
        raise ValueError(f"policy_directory_resolution:{len(matches)}")
    return matches[0]


def build_disjoint_technical_preflight(
    *, split_manifest_path: str | Path, dataset_root: str | Path
) -> dict[str, Any]:
    """Validate a frozen split without reading outcomes or task metadata."""

    split_path = Path(split_manifest_path).resolve()
    root = Path(dataset_root).resolve()
    split = _validated_split(split_path)
    session_root = root / "evaluation_sessions"
    sessions = [str(value) for value in split["selection"]["session_ids"]]
    policies = [str(value) for value in split["required_policy_ids"]]
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    view_counts = {view: 0 for view in REQUIRED_VIEWS}

    for session_id in sessions:
        session_dir = session_root / session_id
        if not session_dir.is_dir():
            blockers.append(f"session_directory_missing:{session_id}")
            continue
        for policy_id in policies:
            row: dict[str, Any] = {
                "session_id": session_id,
                "policy_id": policy_id,
                "status": "blocked",
                "blockers": [],
            }
            try:
                policy_dir = _policy_directory(session_dir, policy_id)
                npz_files = sorted(policy_dir.glob("*_npz_file.npz"))
                if len(npz_files) != 1:
                    raise ValueError(f"npz_resolution:{len(npz_files)}")
                arrays = load_restricted_roboarena_npz(npz_files[0])
                action_steps = int(arrays["action"].shape[0])
                if action_steps < MIN_ACTION_STEPS:
                    raise ValueError(f"insufficient_action_steps:{action_steps}")
                row["action"] = {
                    "sha256": file_sha256(npz_files[0]),
                    "bytes": npz_files[0].stat().st_size,
                    "step_count": action_steps,
                    "array_shapes": {
                        key: list(value.shape) for key, value in sorted(arrays.items())
                    },
                    "restricted_loader_used": True,
                    "numpy_allow_pickle_used": False,
                }
                videos: dict[str, Any] = {}
                for view in REQUIRED_VIEWS:
                    matches = sorted(policy_dir.glob(f"*_video_{view}.mp4"))
                    if len(matches) != 1:
                        row["blockers"].append(f"video_resolution:{view}:{len(matches)}")
                        continue
                    try:
                        videos[view] = _video_probe(matches[0])
                        view_counts[view] += 1
                    except Exception as exc:  # noqa: BLE001 - retain fail-closed reason
                        row["blockers"].append(f"video_invalid:{view}:{type(exc).__name__}")
                row["videos"] = videos
            except Exception as exc:  # noqa: BLE001 - retain fail-closed reason
                row["blockers"].append(f"technical_validation:{type(exc).__name__}:{exc}")
            if not row["blockers"]:
                row["status"] = "passed"
            else:
                blockers.extend(f"{session_id}:{policy_id}:{reason}" for reason in row["blockers"])
            rows.append(row)

    expected_rows = len(sessions) * len(policies)
    passed_rows = sum(row["status"] == "passed" for row in rows)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed_rows == expected_rows and not blockers else "blocked",
        "dataset": split["dataset"],
        "split_manifest_sha256": split["manifest_sha256"],
        "split_manifest_file_sha256": file_sha256(split_path),
        "selected_session_count": len(sessions),
        "required_policy_count": len(policies),
        "expected_row_count": expected_rows,
        "observed_row_count": len(rows),
        "passed_row_count": passed_rows,
        "view_counts": view_counts,
        "required_views": list(REQUIRED_VIEWS),
        "rows": rows,
        "blockers": blockers,
        "label_seal": {
            "metadata_yaml_opened": False,
            "outcome_labels_accessed": False,
            "task_instructions_accessed": False,
        },
        "claim_ceiling": {
            "technical_preflight_only": True,
            "same_published_snapshot": True,
            "independent_new_snapshot_confirmation": False,
            "policy_ranking_or_wam_qualification": False,
            "live_closed_loop_phase_b": False,
        },
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build_disjoint_technical_preflight(
        split_manifest_path=args.split_manifest,
        dataset_root=args.dataset_root,
    )
    write_json(Path(args.output), report)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
