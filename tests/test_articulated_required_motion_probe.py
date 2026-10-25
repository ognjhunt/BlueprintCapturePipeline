"""Hermetic control/readback coverage; these tests do not establish Isaac physics."""
from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline.articulated_required_motion_probe import (
    build_required_motion_probe_plan, execute_required_motion_probe,
    required_motion_plan_from_static_receipt, validate_required_motion_probe_plan,
    validate_required_motion_probe_result,
)
from blueprint_pipeline.articulated_native_diagnostic_runtime import _execute_native_required_motions
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def inputs():
    graph = {"joints": [
        {"joint_id": "panel_hinge", "joint_type": "revolute", "limits": [-0.2, 1.4],
         "reset_position": -0.2, "reset_tolerance": 0.001},
        {"joint_id": "carriage_slide", "joint_type": "prismatic", "limits": [-0.45, 0.05],
         "reset_position": 0.05, "reset_tolerance": 0.001},
        {"joint_id": "upper_slide", "joint_type": "prismatic", "limits": [0.1, 0.4],
         "reset_position": 0.1, "reset_tolerance": 0.001},
    ]}
    readback = {"requirements_digest": "sha256:" + "a" * 64, "motions": [
        {"motion_id": motion, **{key: joint[key] for key in ("joint_id", "joint_type", "limits")}}
        for motion, joint in zip(("z_panel", "a_carriage", "b_upper"), graph["joints"], strict=True)]}
    requirements = [{"motion_id": row["motion_id"], "source_frame_ids": [f"frame{index}"]}
                    for index, row in enumerate(readback["motions"])]
    frames = [{"frame_id": f"frame{index}", "timestamp_seconds": index * 2.5} for index in range(3)]
    return readback, graph, requirements, frames


def plan():
    readback, graph, requirements, frames = inputs()
    return build_required_motion_probe_plan(readback=readback, graph=graph,
                                           requirements=requirements, source_frames=frames)


def test_observed_order_holds_previous_mechanisms_and_resets_in_reverse_in_si_units():
    value = plan()
    assert value["motion_order_basis"] == "source_frame_chronology"
    assert [row["motion_id"] for row in value["motions"]] == ["z_panel", "a_carriage", "b_upper"]
    phases = {row["phase_id"]: row["positions"] for row in value["phases"]}
    assert phases["a_carriage_endpoint"] == {"panel_hinge": 1.4, "carriage_slide": -0.45, "upper_slide": 0.1}
    assert [row["phase_id"] for row in value["phases"][-3:]] == ["b_upper_reset", "a_carriage_reset", "z_panel_reset"]
    assert value["phases"][0]["positions"] == value["phases"][-1]["positions"]
    assert value["motions"][1]["position_tolerance"] == pytest.approx(0.015)
    assert value["task_success_proven"] is False


class Articulation:
    dof_names = ["upper_slide", "panel_hinge", "carriage_slide"]

    def __init__(self, stuck=()):
        self.positions = np.array([0.1, -0.2, 0.05])
        self.stuck = set(stuck)
        self.commands = []

    def get_joint_positions(self):
        return self.positions.copy()

    def get_joint_velocities(self):
        return np.zeros(3)

    def set_joint_positions(self, *_args, **_kwargs):
        pytest.fail("Motion capability must be exercised through the articulation controller")

    def apply_action(self, action):
        self.commands.append(action)
        for index, position in zip(action.joint_indices, action.joint_positions, strict=True):
            if self.dof_names[index] not in self.stuck:
                self.positions[index] = position


@pytest.mark.parametrize("stuck", [(), ("carriage_slide",), ("upper_slide",)])
def test_real_native_control_adapter_cannot_pass_with_only_the_hinge_moving(stuck):
    articulation = Articulation(stuck)
    frames = []
    result = _execute_native_required_motions(
        plan(), articulation=articulation, action_type=lambda **kwargs: SimpleNamespace(**kwargs),
        simulation=SimpleNamespace(step=lambda **kwargs: None), settle_steps=3,
        capture=lambda phase, row: frames.append(phase))
    assert result["status"] == ("failed" if stuck else "passed")
    assert articulation.commands and len(frames) == len(result["phase_rows"])
    assert result["robot_embodiment_used"] is False and result["learned_policy_queried"] is False
    if not stuck:
        assert len(result["phase_rows"]) == 10
        assert np.allclose(articulation.positions, [0.1, -0.2, 0.05])


def test_missing_or_nonfinite_readback_cannot_pass():
    value = plan()
    for state in ({}, {row["joint_id"]: [math.nan, 0] for row in value["motions"]}):
        result = execute_required_motion_probe(value, command=lambda positions: None, read_state=lambda: state)
        assert result["status"] == "failed"


def test_resealed_plan_cannot_omit_a_required_motion_or_accept_a_frozen_endpoint():
    value = plan()
    value["motions"][0]["endpoint_position"] = value["motions"][0]["reset_position"]
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    with pytest.raises(ValueError, match="motion_invalid"):
        validate_required_motion_probe_plan(value)
    readback, graph, _, _ = inputs()
    receipt = {"source_constraints": {"required_motion_readback": readback}, "articulation_graph": graph}
    derived = required_motion_plan_from_static_receipt(receipt)
    assert derived["motion_order_basis"] == "candidate_authoring_requirement_order"
    changed = deepcopy(receipt)
    changed["source_constraints"]["required_motion_readback"]["motions"].pop()
    changed["required_motion_probe_plan"] = derived
    with pytest.raises(ValueError, match="static_binding_changed"):
        required_motion_plan_from_static_receipt(changed)


def test_consumer_recalculates_actual_joint_errors_instead_of_accepting_pass_flags():
    value = plan()
    state = {}

    def command(positions):
        state.update({name: [coordinate, 0.0] for name, coordinate in positions.items()})

    result = execute_required_motion_probe(value, command=command, read_state=lambda: state)
    assert validate_required_motion_probe_result(result, value) == result
    result["phase_rows"][2]["joint_readback"]["panel_hinge"]["position"] = -0.2
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    with pytest.raises(ValueError, match="result_motion_failed"):
        validate_required_motion_probe_result(result, value)
