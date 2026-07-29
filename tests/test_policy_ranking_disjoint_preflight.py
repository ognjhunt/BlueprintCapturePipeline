from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from blueprint_pipeline.policy_ranking_disjoint_preflight import (
    build_disjoint_technical_preflight,
    main,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256


EXPERIMENT_DOCS = (
    Path(__file__).resolve().parents[1]
    / "docs/experiments/policy_ranking_roboarena_disjoint_reasoner_successor_20260728"
)


def _npz(path: Path, steps: int = 17) -> None:
    rows = [
        {
            "cartesian_position": [index / 100, 0, 0, 0, 0, 0],
            "joint_position": [0.0] * 7,
            "gripper_position": [0.0],
            "action": [0.0] * 8,
        }
        for index in range(steps)
    ]
    np.savez(path, data=np.asarray(rows, dtype=object))


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    for index in range(3):
        writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    writer.release()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    policy_dir = root / "evaluation_sessions" / "session-1" / "A_policy-1"
    policy_dir.mkdir(parents=True)
    (policy_dir.parent / "metadata.yaml").write_text("outcome: DO_NOT_OPEN\n", encoding="utf-8")
    _npz(policy_dir / "policy-1_npz_file.npz")
    for view in ("left", "right", "wrist"):
        _video(policy_dir / f"policy-1_video_{view}.mp4")
    split = {
        "schema_version": "policy_ranking_disjoint_session_candidate_split.v1",
        "dataset": {"id": "fixture", "revision": "abc", "license": "mit"},
        "required_policy_ids": ["policy-1"],
        "selection": {"metadata_yaml_opened": False, "session_ids": ["session-1"]},
    }
    split["manifest_sha256"] = canonical_sha256(split)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    return split_path, root


def test_disjoint_preflight_passes_without_opening_metadata(tmp_path: Path, monkeypatch) -> None:
    split_path, root = _fixture(tmp_path)
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args, **kwargs):
        assert path.name != "metadata.yaml"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    report = build_disjoint_technical_preflight(
        split_manifest_path=split_path, dataset_root=root
    )
    assert report["status"] == "passed"
    assert report["passed_row_count"] == 1
    assert report["view_counts"] == {"left": 1, "right": 1, "wrist": 1}
    assert report["label_seal"]["outcome_labels_accessed"] is False
    assert report["claim_ceiling"]["policy_ranking_or_wam_qualification"] is False


def test_disjoint_preflight_fails_closed_on_missing_required_view(tmp_path: Path) -> None:
    split_path, root = _fixture(tmp_path)
    (root / "evaluation_sessions/session-1/A_policy-1/policy-1_video_right.mp4").unlink()
    report = build_disjoint_technical_preflight(
        split_manifest_path=split_path, dataset_root=root
    )
    assert report["status"] == "blocked"
    assert report["passed_row_count"] == 0
    assert any("video_resolution:right:0" in blocker for blocker in report["blockers"])


def test_disjoint_preflight_rejects_tampered_split(tmp_path: Path) -> None:
    split_path, root = _fixture(tmp_path)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["selection"]["session_ids"].append("session-2")
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        build_disjoint_technical_preflight(split_manifest_path=split_path, dataset_root=root)
    except ValueError as exc:
        assert str(exc) == "split_manifest_sha256_mismatch"
    else:
        raise AssertionError("tampered split was accepted")


def test_disjoint_preflight_cli_writes_output(tmp_path: Path) -> None:
    split_path, root = _fixture(tmp_path)
    output = tmp_path / "nested" / "report.json"
    assert (
        main(
            [
                "--split-manifest",
                str(split_path),
                "--dataset-root",
                str(root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"


def test_disjoint_preflight_accepts_prospective_v2_split_amendment(tmp_path: Path) -> None:
    split_path, root = _fixture(tmp_path)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "policy_ranking_disjoint_session_candidate_split_amendment.v2"
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = canonical_sha256(payload)
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    report = build_disjoint_technical_preflight(
        split_manifest_path=split_path, dataset_root=root
    )
    assert report["status"] == "passed"


def test_committed_disjoint_preflight_artifact_digests_are_canonical() -> None:
    for filename in (
        "disjoint_session_candidate_split_amendment_v2.json",
        "disjoint_technical_preflight_summary_v1.json",
        "phase_b_native_cosmos_environment_v1.json",
        "phase_b_native_cosmos_canary_preparation_v1.json",
        "phase_b_open_loop_replay_protocol_v1.json",
    ):
        payload = json.loads((EXPERIMENT_DOCS / filename).read_text(encoding="utf-8"))
        recorded = payload.pop("manifest_sha256")
        assert recorded == canonical_sha256(payload)
