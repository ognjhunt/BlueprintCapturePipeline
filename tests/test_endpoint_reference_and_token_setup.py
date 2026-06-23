from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import g1_endpoint_reference_adapter as adapter
from blueprint_pipeline import policy_endpoint_token_setup as token_setup
from blueprint_pipeline.g1_field_run_capture import _first_string, _webapp_route_prefill


def test_policy_endpoint_token_setup_creates_reuses_and_prints_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    token_file = tmp_path / "secrets" / "team-token.txt"
    manifest_path = tmp_path / "token-manifest.json"
    generated = "2026-06-20T00:00:00+00:00"

    monkeypatch.setattr(token_setup.secrets, "token_urlsafe", lambda _n: "first-token")
    created = token_setup.create_team_policy_endpoint_token(
        token_file=token_file,
        write_manifest=manifest_path,
        generated_at=generated,
    )

    assert created["status"] == "created"
    assert created["generated_at"] == generated
    assert created["file_mode_octal"] == "0o600"
    assert created["raw_token_written_to_artifacts"] is False
    assert token_file.read_text(encoding="utf-8") == "first-token\n"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["token_file"] == str(
        token_file.resolve()
    )

    monkeypatch.setattr(token_setup.secrets, "token_urlsafe", lambda _n: "second-token")
    reused = token_setup.create_team_policy_endpoint_token(token_file=token_file)
    assert reused["status"] == "already_present"
    assert token_file.read_text(encoding="utf-8") == "first-token\n"

    rc = token_setup.main(
        [
            "--token-file",
            str(token_file),
            "--force",
            "--write-manifest",
            str(tmp_path / "forced-manifest.json"),
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert printed["status"] == "created"
    assert printed["raw_token_written_to_stdout"] is False
    assert token_file.read_text(encoding="utf-8") == "second-token\n"


def test_g1_endpoint_reference_adapter_actions_manifest_and_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert adapter._mapping({"a": 1}) == {"a": 1}
    assert adapter._mapping(["bad"]) == {}
    assert adapter._number(True, 3.0) == 3.0
    assert adapter._number("bad", 4.0) == 4.0
    assert adapter._target_pose({"route_task_state": {"target_pose": [1, "2"]}}) == [
        1.0,
        2.0,
        0.79,
    ]
    assert adapter._target_pose({"route_task_state": {"target_pose": "bad"}}) == [
        0.0,
        0.0,
        0.79,
    ]
    assert adapter._object_waypoint({"object_state": {"position": "bad"}}) == [
        0.54,
        -0.65,
        0.79,
    ]

    assert adapter.choose_action({"task_id": "inspect_target", "step_index": 1})[
        "action_type"
    ] == "inspect_look"
    assert adapter.choose_action({"task_id": "inspect_target", "step_index": 80})[
        "action_type"
    ] == "stop"
    assert adapter.choose_action(
        {
            "task_id": "approach_target",
            "route_task_state": {"target_error_m": 0.2},
        }
    )["report"] == "within_goal_tolerance"
    assert adapter.choose_action(
        {
            "task_id": "approach_target",
            "step_index": 5,
            "route_task_state": {"target_error_m": 1.0},
        }
    )["action_type"] == "base_velocity"
    approach_action = adapter.choose_action(
        {
            "task_id": "approach_target",
            "step_index": 90,
            "base_pose": {"position": [0, 0, 0.79], "yaw_rad": 0},
            "route_task_state": {"target_error_m": 1.0, "target_pose": [1, 0, 3]},
        }
    )
    assert approach_action["velocity_frame"] == "robot_base"
    assert approach_action["linear_velocity_mps"] == adapter.SAFE_APPROACH_SPEED_MPS
    assert abs(approach_action["lateral_velocity_mps"]) < 1e-9
    route_action = adapter.choose_action(
        {
            "task_id": "route_around_obstruction",
            "step_index": 240,
            "route_task_state": {"route_waypoints": [[1, 2], [3, 4, 0.5]]},
        }
    )
    assert route_action["waypoint"] == [3.0, 4.0, 0.5]
    assert route_action["max_speed_mps"] == adapter.SAFE_WAYPOINT_SPEED_MPS
    assert adapter.choose_action(
        {"task_id": "route_around_obstruction", "route_task_state": {"route_waypoints": "bad"}}
    )["action_type"] == "waypoint"
    contact_action = adapter.choose_action({"task_id": "contact_or_push_light_object"})
    assert contact_action["action_type"] == "manipulation_contact"
    assert contact_action["approach_speed_mps"] == adapter.SAFE_CONTACT_APPROACH_SPEED_MPS
    assert adapter.choose_action(
        {
            "task_id": "stop_at_goal_and_report",
            "step_index": 10,
            "route_task_state": {"target_error_m": 0.2},
        }
    )["action_type"] == "stop"
    stop_approach = adapter.choose_action(
        {
            "task_id": "stop_at_goal_and_report",
            "step_index": 10,
            "route_task_state": {"target_error_m": 2.0},
        }
    )
    assert stop_approach["action_type"] == "waypoint"
    assert stop_approach["max_speed_mps"] == adapter.SAFE_STOP_APPROACH_SPEED_MPS
    unknown = adapter.choose_action({"task_id": "unknown"})
    assert unknown["action_type"] == "waypoint"
    assert unknown["max_speed_mps"] == adapter.SAFE_WAYPOINT_SPEED_MPS

    manifest = adapter.adapter_manifest()
    assert manifest["policy_id"] == adapter.POLICY_ID
    assert manifest["claim_boundary"]["not_real_wam_vla"] is True
    response = adapter.build_response(
        {"observation": {"task_id": "inspect_target", "step_index": 1}}
    )
    assert response["adapter_metadata"]["raw_token_values_returned"] is False

    assert adapter.main(["--print-manifest"]) == 0
    assert json.loads(capsys.readouterr().out)["policy_id"] == adapter.POLICY_ID

    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"observation": {"task_id": "inspect_target"}})),
    )
    assert adapter.main([]) == 0
    assert json.loads(capsys.readouterr().out)["action"]["action_type"] == "inspect_look"

    monkeypatch.setattr(sys, "stdin", io.StringIO("[]"))
    with pytest.raises(SystemExit, match="stdin_json_must_be_object"):
        adapter.main([])


def test_g1_field_route_helpers_cover_empty_and_unaccepted_inputs() -> None:
    assert _first_string("", None, default="fallback") == "fallback"
    assert _webapp_route_prefill({"status": "blocked"}) == {}
