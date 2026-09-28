from __future__ import annotations

import hashlib
import json

from PIL import Image
import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError, ToolContext
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.episode_investigation import EpisodeEvidenceTools
from tests.test_episode_interpretation import _episode_root, _request


def evidence(tmp_path):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    manifest = data["manifest"]
    rows = [*manifest["policy_input_frames"], manifest["terminal_observation"]]
    for index, row in enumerate(rows):
        path = data["root"] / row["relative_path"]
        Image.new("RGB", (12, 8), (index * 40, 20, 30)).save(path)
        row.update(
            png_sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size, camera_id="wrist" if index % 2 == 0 else "torso",
            simulation_time_s=(index // 2) * 0.1,
        )
    manifest["required_camera_ids"] = ["wrist", "torso"]
    manifest["frame_manifest_digest"] = canonical_digest(manifest, digest_field="frame_manifest_digest")
    data["manifest_path"].write_text(json.dumps(manifest))
    request = _request(data)
    artifacts = request.input_receipt["artifacts"]
    admitted = frozenset(
        [record["sha256"] for record in artifacts.values() if isinstance(record, dict)]
        + [row["sha256"] for row in artifacts["lossless_frames"]]
    )
    return EpisodeEvidenceTools(request, admitted_digests=admitted), data, admitted


def test_event_interval_returns_same_timestamp_cameras_and_original_frame_refs(tmp_path):
    tools, _, _ = evidence(tmp_path)
    content = tools.interval(start_seconds=0.05, end_seconds=0.15)
    summary = json.loads(content[0]["text"])
    assert summary["returned_observation_count"] == 1
    assert summary["groups"][0]["camera_ids"] == ["torso", "wrist"]
    assert summary["groups"][0]["missing_required_camera_ids"] == []
    assert len([part for part in content if part["type"] == "input_image"]) == 2


def test_sampling_missing_cameras_and_absent_time_windows_are_explicit(tmp_path):
    tools, _, _ = evidence(tmp_path)
    summary = json.loads(tools.interval(start_seconds=0, end_seconds=1, max_observations=1)[0]["text"])
    assert summary["sampled"] is True and summary["matching_observation_count"] == 3
    terminal = json.loads(tools.interval(start_seconds=0.2, end_seconds=0.2)[0]["text"])
    assert terminal["groups"][0]["missing_required_camera_ids"] == ["torso"]
    absent = json.loads(tools.interval(start_seconds=5, end_seconds=6)[0]["text"])
    assert absent["no_recorded_frames_in_interval"] is True


def test_trace_window_preserves_the_drop_and_does_not_change_the_score(tmp_path):
    tools, data, _ = evidence(tmp_path)
    original = data["score_path"].read_bytes()
    window = tools.trace("state_trace", start_step=2, end_step=3)
    rows = window["fields"]["task_state_samples"]["rows"]
    assert [row["step_index"] for row in rows] == [2, 3]
    assert rows[0]["task_object_pose_world"][2] > rows[1]["task_object_pose_world"][2]
    assert data["score_path"].read_bytes() == original
    assert tools.context()["deterministic_score"]["task_succeeded"] is False


def test_source_change_and_cross_task_disclosure_are_rejected(tmp_path):
    tools, data, _ = evidence(tmp_path)
    context_tool = next(t for t in tools.tools() if t.tool_id == "read_episode_context")
    context = ToolContext("run", "task", "revision", "op", "authority", 1000)
    with pytest.raises(AgentExecutionError, match="task_disclosure_mismatch"):
        context_tool.invoke({}, context)
    data["state_path"].write_text('{"changed":true}')
    with pytest.raises(AgentExecutionError, match="artifact_changed"):
        tools.trace("state_trace", start_step=0, end_step=5)
