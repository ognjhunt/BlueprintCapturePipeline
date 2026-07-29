from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.policy_ranking_phase_b_high_motion_selection import (
    build_high_motion_selection,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256


def _npz(path: Path, translation_per_step: float) -> None:
    path.parent.mkdir(parents=True)
    rows = []
    for index in range(17):
        rows.append(
            {
                "cartesian_position": [
                    index * translation_per_step,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "joint_position": [0.0] * 7,
                "gripper_position": [0.0],
                "action": [0.0] * 7 + [float(index >= 8)],
            }
        )
    np.savez(path, data=np.asarray(rows, dtype=object))


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    for session_id, motion in (("session-a", 0.001), ("session-b", 0.004)):
        for prefix, policy_id, scale in (
            ("A", "policy-one", 1.0),
            ("B", "policy-two", 0.5),
        ):
            directory = root / "evaluation_sessions" / session_id / f"{prefix}_{policy_id}"
            _npz(directory / f"{policy_id}_npz_file.npz", motion * scale)
        (root / "evaluation_sessions" / session_id / "metadata.yaml").write_text(
            "this: is deliberately not parsed\n", encoding="utf-8"
        )
    split = {
        "schema_version": "policy_ranking_disjoint_session_candidate_split_amendment.v2",
        "dataset": {"id": "fixture", "revision": "frozen", "license": "mit"},
        "required_policy_ids": ["policy-one", "policy-two"],
        "selection": {
            "metadata_yaml_opened": False,
            "session_ids": ["session-a", "session-b"],
        },
    }
    split["manifest_sha256"] = canonical_sha256(split)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    return root, split_path


def test_builds_complete_label_blind_selection_manifest(tmp_path: Path) -> None:
    root, split = _fixture(tmp_path)
    result = build_high_motion_selection(split_manifest_path=split, dataset_root=root)

    assert result["status"] == "passed"
    assert result["candidate_count"] == 4
    assert result["selection"]["recorded"]["session_id_internal_only"] == "session-b"
    assert result["selection"]["recorded"]["policy_id_internal_only"] == "policy-one"
    assert result["selection"]["policy_swapped"]["policy_id_internal_only"] == "policy-two"
    assert result["input_access"] == {
        "restricted_numeric_npz_actions_opened": True,
        "metadata_yaml_opened": False,
        "outcome_labels_accessed": False,
        "video_pixels_opened": False,
        "generated_media_accessed": False,
        "evaluator_predictions_accessed": False,
    }
    assert all(row["npz_sha256"] for row in result["source_rows"])
    assert result["manifest_sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "manifest_sha256"}
    )


def test_rejects_tampered_or_unsealed_split(tmp_path: Path) -> None:
    root, split_path = _fixture(tmp_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["selection"]["session_ids"].append("tampered")
    split_path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="split_manifest_sha256_mismatch"):
        build_high_motion_selection(split_manifest_path=split_path, dataset_root=root)

    split["selection"]["session_ids"].pop()
    split["selection"]["metadata_yaml_opened"] = True
    split["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in split.items() if key != "manifest_sha256"}
    )
    split_path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="split_not_label_sealed"):
        build_high_motion_selection(split_manifest_path=split_path, dataset_root=root)
