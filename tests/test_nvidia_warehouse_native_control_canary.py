from __future__ import annotations

import inspect
import json
from pathlib import Path

import blueprint_pipeline.nvidia_warehouse_native_control_canary as control
from blueprint_pipeline.nvidia_warehouse_native_control_canary import (
    CLAIM_LABEL,
    _contact_impulse_magnitude,
    _controller_joint_limits,
    _validated_spec,
    rank_controller_results,
    run_native_control_canary,
)

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = (
    ROOT
    / "docs"
    / "experiments"
    / "policy_ranking_thesis_20260726"
    / "nvidia_warehouse_native_control_spec_v1.json"
)


def _controller_rows(spec: dict) -> list[dict]:
    return [
        {
            "controller_id": row["controller_id"],
            "success": index < 2,
            "partial_progress": 1.0 if index < 2 else 0.5 - index * 0.05,
            "termination_reason": "success" if index < 2 else "timeout",
            "steps": 1000 + index * 10,
            "safety_violation": False,
        }
        for index, row in enumerate(spec["controllers"])
    ]


def test_frozen_native_control_spec_is_digest_bound_and_has_exact_cohort() -> None:
    spec = _validated_spec(SPEC)

    assert spec["claim_label"] == CLAIM_LABEL
    assert len(spec["controllers"]) == 5
    assert spec["positive_control"]["controller_id"].startswith("positive_control_")
    assert spec["trace_contract"]["silent_retry_forbidden"] is True
    assert spec["trace_contract"]["presentation_renders_feed_evaluation"] is False


def test_ranking_preserves_deterministic_exact_ties() -> None:
    rows = [
        {
            "controller_id": controller_id,
            "success": True,
            "partial_progress": 1.0,
            "termination_reason": "success",
            "steps": 100,
            "safety_violation": False,
        }
        for controller_id in ("b", "a")
    ]

    ranked = rank_controller_results(rows)

    assert [row["controller_id"] for row in ranked] == ["a", "b"]
    assert [row["rank"] for row in ranked] == [1, 1]


def test_contact_impulse_uses_physx_float3_magnitude() -> None:
    assert _contact_impulse_magnitude(np.asarray([3.0, 4.0, 0.0])) == 5.0


def test_controller_joint_limits_normalizes_public_isaac_tuple() -> None:
    class Controller:
        def get_joint_limits(self):
            return (
                np.asarray([[-2.0, -1.0, 0.0]]),
                np.asarray([[2.0, 1.0, 0.04]]),
            )

    assert _controller_joint_limits(Controller()).tolist() == [
        [-2.0, 2.0],
        [-1.0, 1.0],
        [0.0, 0.04],
    ]


def test_controller_joint_limits_accepts_view_shaped_array() -> None:
    class Controller:
        def get_joint_limits(self):
            return np.asarray([[[-2.0, 2.0], [0.0, 0.04]]])

    assert _controller_joint_limits(Controller()).shape == (2, 2)


def test_injected_native_backend_emits_result_envelope_and_integrity_index(
    tmp_path: Path,
) -> None:
    spec = _validated_spec(SPEC)
    rows = _controller_rows(spec)

    def backend(*, output_dir: Path, **_kwargs):
        output_dir.mkdir(parents=True)
        (output_dir / "trace.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
        return {
            "runtime_backend": "isaac_sim_6_physx",
            "hybrid_or_mujoco_backend_used": False,
            "scene_physics": {
                "meters_per_unit_valid": True,
                "up_axis_valid": True,
                "gravity_valid": True,
                "collision_inventory_valid": True,
                "dependency_closure_resolved": True,
                "settle_stable": True,
                "support_contact_proven": True,
                "initial_overlap_clear": True,
                "franka_articulation_valid": True,
                "franka_joint_limits_valid": True,
                "franka_controller_binding_valid": True,
                "franka_collision_behavior_valid": True,
            },
            "reset_evidence": {"cycle_count": 5, "within_tolerances": True},
            "positive_control": {"success": True},
            "controller_results": rows,
            "evidence_complete": True,
        }

    result = run_native_control_canary(
        spec_path=SPEC,
        assets_root=tmp_path,
        output_dir=tmp_path / "evidence",
        backend=backend,
    )

    envelope = json.loads((tmp_path / "evidence" / "decision_envelope.json").read_text())
    index = json.loads((tmp_path / "evidence" / "evidence_index.json").read_text())
    assert result["status"] == "passed"
    assert envelope["decision"] == "supported"
    assert "arkitscenes_collision_readiness" in envelope["explicitly_denied_claims"]
    assert index["file_count"] == 3


def test_positive_control_failure_blocks_five_controller_claim(tmp_path: Path) -> None:
    def backend(**_kwargs):
        return {
            "runtime_backend": "isaac_sim_6_physx",
            "hybrid_or_mujoco_backend_used": False,
            "scene_physics": {
                field: True
                for field in (
                    "meters_per_unit_valid",
                    "up_axis_valid",
                    "gravity_valid",
                    "collision_inventory_valid",
                    "dependency_closure_resolved",
                    "settle_stable",
                    "support_contact_proven",
                    "initial_overlap_clear",
                    "franka_articulation_valid",
                    "franka_joint_limits_valid",
                    "franka_controller_binding_valid",
                    "franka_collision_behavior_valid",
                )
            },
            "reset_evidence": {"cycle_count": 5, "within_tolerances": True},
            "positive_control": {"success": False},
            "controller_results": [],
            "evidence_complete": True,
        }

    result = run_native_control_canary(
        spec_path=SPEC,
        assets_root=tmp_path,
        output_dir=tmp_path / "failed",
        backend=backend,
    )

    assert result["status"] == "failed"
    assert result["blockers"] == ["native_positive_control_failed"]
    assert result["assessment"]["controller_results"] == []


def test_native_backend_enables_bundled_franka_compatibility_extension_first() -> None:
    source = inspect.getsource(control.isaac_sim_6_native_control_backend)
    enable = 'enable_extension("isaacsim.robot.manipulators.examples")'
    franka_import = "from isaacsim.robot.manipulators.examples.franka import Franka"
    controller_import = (
        "from isaacsim.robot.manipulators.examples.franka.controllers import "
        "PickPlaceController"
    )

    assert enable in source
    assert source.index(enable) < source.index(franka_import)
    assert source.index(enable) < source.index(controller_import)
    between = source[source.index(enable) : source.index(franka_import)]
    assert "simulation_app.update()" in between


def test_native_backend_uses_franka_compatible_numpy_articulation_adapter() -> None:
    source = inspect.getsource(control.isaac_sim_6_native_control_backend)

    assert 'backend="numpy"' in source
    assert 'device="cpu"' in source
    assert 'backend="torch"' not in source


def test_native_backend_reads_limits_from_public_articulation_controller() -> None:
    source = inspect.getsource(control.isaac_sim_6_native_control_backend)

    assert "_controller_joint_limits(articulation_controller)" in source
    assert "robot.get_dof_limits()" not in source
