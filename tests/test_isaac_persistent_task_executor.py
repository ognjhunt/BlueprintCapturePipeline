from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline import isaac_persistent_task_completion_client as client
from blueprint_pipeline.isaac_persistent_task_executor_service import _computed
from blueprint_pipeline.isaac_runtime_task_backend import IsaacPersistentTaskBackend


@pytest.mark.parametrize(
    ("criterion", "before", "after", "expected"),
    [
        ({"comparison": "increase_at_least", "tolerance": 0.35}, 0.0, 0.36, True),
        ({"comparison": "decrease_at_least", "tolerance": 0.35}, 1.0, 0.7, False),
        ({"comparison": "absolute_change_at_least", "tolerance": 0.2}, 0.0, -0.2, True),
        ({"comparison": "within_tolerance", "target_value": 1.0, "tolerance": 0.1}, 0.0, 1.09, True),
        ({"comparison": "at_or_above", "target_value": 1.0, "tolerance": 0.05}, 0.0, 0.96, True),
        ({"comparison": "at_or_below", "target_value": 1.0, "tolerance": 0.05}, 0.0, 1.04, True),
    ],
)
def test_computed_uses_typed_task_criterion(criterion, before, after, expected):
    assert _computed(criterion, before, after) is expected


def test_completion_client_posts_attempt_bound_request(monkeypatch, tmp_path: Path):
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    request_payload = {
        "schema_version": "oscar_task_completion_evaluator_request.v1",
        "step_index": 3,
        "source_action_sha256": "a" * 64,
        "action": {"generated_robot_state": {"joint_positions": [0.1]}},
        "task_success_contract": {"criterion_id": "door-open"},
    }
    request_path.write_text(json.dumps(request_payload))

    result_payload = {
        "status": "completed",
        "passed": True,
        "simulator_session_id": "session-1",
        "stage_id": "stage-1",
        "runtime_result_id": "result-1",
        "source_action_sha256": "a" * 64,
        "articulation_prim_path": "/World/G1",
        "before_timestamp": "2026-07-10T00:00:00Z",
        "after_timestamp": "2026-07-10T00:00:01Z",
        "before_value": 0.0,
        "after_value": 0.4,
        "unit": "radian",
        "criterion_id": "door-open",
        "observable_transition": "door_joint_increases",
        "evaluator_attestation": {"verification_status": "verified"},
        "persistent_simulator_state_applied": True,
        "official_controller_action_applied": True,
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(result_payload).encode()

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://127.0.0.1:8765/apply-and-measure"
        assert timeout == 120
        assert json.loads(request.data) == request_payload
        return Response()

    monkeypatch.setattr(client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("BLUEPRINT_TASK_COMPLETION_INPUT", str(request_path))
    monkeypatch.setenv("BLUEPRINT_TASK_COMPLETION_OUTPUT", str(output_path))

    assert client.main() == 0
    assert json.loads(output_path.read_text()) == result_payload


def test_backend_applies_two_actions_to_one_persistent_stage_and_measures_transition(
    tmp_path: Path,
) -> None:
    class DC:
        task_value = 0.0
        robot_target = 0.0

        def get_articulation(self, path):
            return "robot" if path == "/World/G1" else "task"

        def find_articulation_dof(self, articulation, name):
            return f"{articulation}:{name}"

        def get_dof_position(self, dof):
            return self.task_value if dof.startswith("task:") else self.robot_target

        def set_dof_position_target(self, dof, value):
            assert dof.startswith("robot:")
            self.robot_target = value

    class App:
        def __init__(self, dc):
            self.dc = dc

        def update(self):
            self.dc.task_value += abs(self.dc.robot_target) * 0.05

    backend = IsaacPersistentTaskBackend.__new__(IsaacPersistentTaskBackend)
    backend.dc = DC()
    backend.app = App(backend.dc)
    backend.robot_handle = "robot"
    backend.evidence_dir = tmp_path
    backend.session_id = "persistent-session-1"
    backend.stage_id = "stage-1"
    class ReviewRenderer:
        def render(self, *, step_index, target_prim_path):
            return [
                {
                    "camera_role": "overview",
                    "frame_index": step_index,
                    "path": str(tmp_path / f"overview_{step_index:04d}.png"),
                    "sha256": "a" * 64,
                    "target_prim_path": target_prim_path,
                }
            ]

    backend.review_renderer = ReviewRenderer()
    contract = {
        "registered_criteria": [
            {
                "criterion_id": "microwave_door_open_angle",
                "observable_transition": "articulation_angle_rad",
                "articulation_prim_path": "/root/Microwave017/Microwave017_Door",
                "unit": "rad",
            }
        ]
    }

    def request(step, value):
        action = {"action_chunk": [value]}
        action_sha = hashlib.sha256(
            json.dumps(action, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "step_index": step,
            "action": action,
            "wam_output": {
                "generated_robot_state": {
                    "source_action_sha256": action_sha,
                    "joint_names": ["right_elbow_joint"],
                    "joint_positions": [value],
                }
            },
            "task_success_contract": contract,
            "physics_steps_per_action": 4,
        }

    first = backend.apply_and_measure(request(1, 0.5))
    second = backend.apply_and_measure(request(2, 1.0))
    assert first["simulator_session_id"] == second["simulator_session_id"]
    assert first["stage_id"] == second["stage_id"]
    assert first["runtime_result_id"] != second["runtime_result_id"]
    assert second["before_value"] == pytest.approx(first["after_value"])
    assert second["after_value"] > second["before_value"]
    assert first["source_action_sha256"] != second["source_action_sha256"]
    assert Path(first["evidence_artifacts"][0]["path"]).is_file()
    assert first["review_frames"][0]["camera_role"] == "overview"
