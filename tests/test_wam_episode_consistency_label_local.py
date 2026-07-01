from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import wam_episode_consistency_label_local as consistency_labeler


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_motion_video(path: Path) -> Path:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (96, 64),
    )
    if not writer.isOpened():
        pytest.skip("cv2 VideoWriter cannot create mp4")
    try:
        for index in range(8):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            frame[:, :] = (20 + index * 8, 40, 80)
            cv2.rectangle(
                frame,
                (8 + index * 8, 18),
                (34 + index * 8, 44),
                (240, 240, 240),
                thickness=-1,
            )
            writer.write(frame)
    finally:
        writer.release()
    return path


def _request(tmp_path: Path, video: Path) -> Path:
    request = tmp_path / "wam_episode_consistency_request.json"
    _write_json(
        request,
        {
            "schema_version": "wam_episode_consistency_request.v1",
            "status": "ready_for_external_episode_scorer",
            "rollouts": [
                {
                    "rollout_id": "rollout_1",
                    "scenario_eval_run_id": "run_1",
                    "policy_id": "policy",
                    "task_id": "open_refrigerator",
                    "model_candidate": "oscar_wam",
                    "generated_video_path": str(video),
                }
            ],
            "task_prompts": [
                {
                    "scenario_eval_run_id": "run_1",
                    "task_prompt": "open the refrigerator",
                }
            ],
            "source_trace_paths": {
                "robot_policy_wam_loop_trace_jsonl": str(tmp_path / "trace.jsonl")
            },
            "trace_summary": {
                "policy_call_count": 2,
                "wam_transition_count": 1,
            },
        },
    )
    return request


def test_local_wam_episode_consistency_blocks_without_gate(tmp_path: Path, monkeypatch) -> None:
    video = _write_motion_video(tmp_path / "rollout.mp4")
    monkeypatch.delenv(consistency_labeler.GATE_ENV, raising=False)

    result = consistency_labeler.build_local_wam_episode_consistency_labels(
        input_path=_request(tmp_path, video),
        output_path=tmp_path / "out.json",
    )

    assert result["status"] == "blocked"
    assert f"missing_env_{consistency_labeler.GATE_ENV}" in result["blockers"]
    assert result["raw_credentials_written_to_artifacts"] is False


def test_local_wam_episode_consistency_scores_motion_without_claim_upgrades(
    tmp_path: Path, monkeypatch
) -> None:
    video = _write_motion_video(tmp_path / "rollout.mp4")
    output = tmp_path / "wam_episode_consistency.command.json"
    monkeypatch.setenv(consistency_labeler.GATE_ENV, "true")

    result = consistency_labeler.build_local_wam_episode_consistency_labels(
        input_path=_request(tmp_path, video),
        output_path=output,
        max_rollouts=1,
        max_frames=4,
    )

    assert result["status"] == "completed"
    assert result["provider"] == "local_cv_wam_episode_consistency_judge"
    assert result["rollout_check_count"] == 1
    check = result["rollout_checks"][0]
    assert check["forward_consistent"] is True
    assert check["inverse_consistent"] is True
    assert check["visual_evidence_used"] is True
    assert check["action_trace_evidence_used"] is True
    assert check["local_cv_scorer_is_not_vlm_semantic_judge"] is True
    assert check["task_success_proven"] is False
    assert check["rank_fidelity_result_proven"] is False
    assert result["claim_boundary"]["local_cv_scorer_is_not_vlm_semantic_judge"] is True
    assert output.is_file()
