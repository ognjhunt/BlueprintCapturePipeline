from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from blueprint_pipeline.policy_wam_closed_loop import (
    ClosedLoopConfig,
    run_policy_wam_closed_loop,
)


class _Policy:
    policy_id = "hidden_candidate_a"

    def __init__(self) -> None:
        self.seen_steps: list[int] = []

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        self.seen_steps.append(int(observation["step"]))
        return np.full((10, 8), float(observation["step"] + 1))


class _Wam:
    arm_id = "oscar_purpose_built_wam"

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def predict(self, request: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
        self.requests.append(dict(request))
        frame = output_dir / "predicted.png"
        frame.write_bytes(f"prediction-{request['query_index']}".encode())
        return {"predicted_frame": frame, "step": request["query_index"] + 1}


class _Adapter:
    adapter_id = "camera_aligned_skeleton_v1"

    def __init__(self, *, leak_identity: bool = False, invalid_provenance: bool = False) -> None:
        self.leak_identity = leak_identity
        self.invalid_provenance = invalid_provenance

    def prepare_transition(self, **kwargs: Any) -> dict[str, Any]:
        request = {
            "query_index": kwargs["query_index"],
            "task_prompt": kwargs["task_prompt"],
            "skeleton_conditioning_sha256": "a" * 64,
        }
        if self.leak_identity:
            request["policy_id"] = "hidden_candidate_a"
        return {
            "wam_request": request,
            "policy_action": kwargs["policy_action"],
            "executed_prefix_steps": kwargs["executed_prefix_steps"],
        }

    def advance_policy_observation(self, **kwargs: Any) -> dict[str, Any]:
        prediction = kwargs["wam_prediction"]
        return {
            "observation": {
                "step": prediction["step"],
                "external": np.full((4, 4, 3), prediction["step"], dtype=np.uint8),
            },
            "provenance": {
                "visual_source": "physical_future" if self.invalid_provenance else "wam_prediction",
                "state_source": "commanded_prefix_kinematics",
                "physical_future_observation_used": False,
            },
        }


class _Gate:
    gate_id = "frozen_rollout_reliability_v1"

    def __init__(self, abstain_at: int | None = None) -> None:
        self.abstain_at = abstain_at

    def assess(self, **kwargs: Any) -> dict[str, Any]:
        index = kwargs["query_index"]
        return {
            "abstain": index == self.abstain_at,
            "reasons": ["static_under_command"] if index == self.abstain_at else [],
        }


class _Terminal:
    criterion_id = "fixture_terminal_step_v1"

    def __init__(self, terminal_step: int) -> None:
        self.terminal_step = terminal_step

    def assess(self, *, observation: dict[str, Any], query_index: int) -> dict[str, Any]:
        del query_index
        return {
            "terminal": observation["step"] >= self.terminal_step,
            "reason": "fixture_task_terminal",
        }


def _run(tmp_path: Path, **overrides: Any) -> tuple[dict[str, Any], _Policy, _Wam]:
    policy = _Policy()
    wam = _Wam()
    result = run_policy_wam_closed_loop(
        initial_observation={"step": 0, "external": np.zeros((4, 4, 3), dtype=np.uint8)},
        policy_client=policy,
        wam_arm=wam,
        transition_adapter=overrides.get("adapter", _Adapter()),
        reliability_gate=overrides.get("gate", _Gate()),
        terminal_criterion=overrides.get("terminal", _Terminal(3)),
        config=overrides.get(
            "config",
            ClosedLoopConfig(
                task_prompt="Pick up the bottle and place it in the bin.",
                executed_prefix_steps=8,
                max_policy_queries=5,
            ),
        ),
        output_dir=tmp_path,
    )
    return result, policy, wam


def test_same_policy_is_requeried_from_wam_predictions_until_terminal(tmp_path: Path) -> None:
    result, policy, wam = _run(tmp_path)

    assert result["status"] == "completed"
    assert result["terminal_reason"] == "fixture_task_terminal"
    assert result["policy_call_count"] == 3
    assert result["wam_call_count"] == 3
    assert policy.seen_steps == [0, 1, 2]
    assert [row["query_index"] for row in wam.requests] == [0, 1, 2]
    assert all("policy_id" not in row for row in wam.requests)
    assert result["executed_prefix_seconds_derived"] == pytest.approx(8 / 15)
    rows = [json.loads(line) for line in Path(result["trace_path"]).read_text().splitlines()]
    assert all(row["wam_arm_id"] == "oscar_purpose_built_wam" for row in rows)
    assert all(row["next_observation_provenance"]["visual_source"] == "wam_prediction" for row in rows)


def test_reliability_abstention_stops_before_policy_requery(tmp_path: Path) -> None:
    result, policy, wam = _run(tmp_path, gate=_Gate(abstain_at=1))

    assert result["status"] == "abstained"
    assert result["terminal_reason"] == "wam_reliability_gate_abstention"
    assert policy.seen_steps == [0, 1]
    assert len(wam.requests) == 2


def test_wam_request_fails_closed_on_policy_identity_leak(tmp_path: Path) -> None:
    result, _policy, wam = _run(tmp_path, adapter=_Adapter(leak_identity=True))

    assert result["status"] == "blocked"
    assert "wam_request_policy_or_outcome_leakage:policy_id" in result["blockers"][0]
    assert wam.requests == []


def test_physical_future_cannot_become_next_policy_observation(tmp_path: Path) -> None:
    result, policy, _wam = _run(tmp_path, adapter=_Adapter(invalid_provenance=True))

    assert result["status"] == "blocked"
    assert "next_policy_visual_not_attributable_to_wam_prediction" in result["blockers"][0]
    assert policy.seen_steps == [0]


@pytest.mark.parametrize("value", [0, 2.4, True])
def test_executed_prefix_steps_must_be_positive_integer(tmp_path: Path, value: Any) -> None:
    with pytest.raises(ValueError, match="executed_prefix_steps"):
        _run(
            tmp_path,
            config=ClosedLoopConfig(
                task_prompt="Move the object.",
                executed_prefix_steps=value,
                max_policy_queries=1,
            ),
        )
