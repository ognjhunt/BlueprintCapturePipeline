from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_robot_placement_agent_cli import (
    RobotPlacementNoMatch,
    _persist_images,
    _read_mapping,
    placement_no_match_verdict,
    run_robot_placement_cli,
)
from blueprint_pipeline.task_evaluation_robot_placement_agent import (
    validate_robot_placement_receipt,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_robot_placement_inventory import (
    build_candidate_inventory_checkpoint,
    validate_candidate_inventory_checkpoint,
)


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_cli_persists_digest_bound_preview_without_embedding_data_url(tmp_path) -> None:
    digest = "sha256:" + hashlib.sha256(_ONE_PIXEL_PNG).hexdigest()
    records = _persist_images(
        [
            {
                "label": "top_down",
                "digest": digest,
                "image_url": "data:image/png;base64,"
                + base64.b64encode(_ONE_PIXEL_PNG).decode("ascii"),
                "detail": "high",
            }
        ],
        output_dir=tmp_path,
        prefix="candidate-00",
    )

    assert records[0]["digest"] == digest
    assert records[0]["size_bytes"] == len(_ONE_PIXEL_PNG)
    assert records[0]["path"].endswith("candidate-00-00-top_down.png")


def test_cli_reads_only_mapping_bindings(tmp_path) -> None:
    path = tmp_path / "binding.json"
    path.write_text('{"schema_version":"fixture.v1"}\n', encoding="utf-8")

    assert _read_mapping(path, label="scene_binding") == {
        "schema_version": "fixture.v1"
    }


def test_paid_visual_review_cannot_disable_geometry_previews(tmp_path):
    import pytest
    from blueprint_pipeline.task_evaluation_robot_placement_agent_cli import run_robot_placement_cli
    with pytest.raises(ValueError,match='visual_previews_required'):
        run_robot_placement_cli(run_id='review',scene_collision_usd=tmp_path/'missing.usd',
            robot_asset_usd=tmp_path/'missing-robot.usd',target_position_world_m=[0.,0.,0.],
            scene_binding={},task_binding={},overview_image_paths=[],output_dir=tmp_path/'out',
            max_rounds=1,candidate_inventory_cap=24,max_input_tokens=12000,max_inference_cost_usd=.15,
            allow_live_invocation=True,tracing_disabled=True,render_geometry_previews=False)


def test_task_pose_requires_object_collision_input_before_output_or_model_call(tmp_path):
    with pytest.raises(ValueError, match="task_asset_collision_context_required"):
        run_robot_placement_cli(
            run_id="missing-actor", scene_collision_usd=tmp_path / "missing.usd",
            robot_asset_usd=tmp_path / "robot.usd", target_position_world_m=[0., 0., 0.],
            scene_binding={}, task_binding={"asset_scene_pose": {"position_world_m": [0., 0., 0.]}},
            overview_image_paths=[], output_dir=tmp_path / "out", max_rounds=1,
            candidate_inventory_cap=12, max_input_tokens=1000, max_inference_cost_usd=.01,
            allow_live_invocation=False, tracing_disabled=True, deterministic_selection=True,
            render_geometry_previews=False,
        )
    assert not (tmp_path / "out").exists()


def test_empty_search_retains_geometry_diagnostics_before_any_model_call(tmp_path, monkeypatch):
    import blueprint_pipeline.task_evaluation_robot_placement_agent_cli as module
    monkeypatch.setattr(module, "build_robot_placement_geometry_index", lambda **_: object())
    monkeypatch.setattr(module, "summarize_robot_placement_geometry", lambda *_a, **_k: {
        "geometry_summary_digest": "sha256:summary", "scene_collision_digest": "sha256:scene"
    })

    def empty(**kwargs):
        kwargs["diagnostics"].update({"proposal_count": 72, "support_or_reach_prefilter_rejected_count": 72})
        return []

    monkeypatch.setattr(module, "enumerate_robot_placement_geometry_candidates", empty)
    with pytest.raises(RobotPlacementNoMatch, match="candidate_inventory_empty") as no_match:
        run_robot_placement_cli(
            run_id="empty", scene_collision_usd=tmp_path / "scene.usd",
            robot_asset_usd=tmp_path / "robot.usd", target_position_world_m=[0., 0., 0.],
            scene_binding={}, task_binding={}, overview_image_paths=[], output_dir=tmp_path / "out",
            max_rounds=1, candidate_inventory_cap=12, max_input_tokens=12000,
            max_inference_cost_usd=.01, allow_live_invocation=False, tracing_disabled=True,
            deterministic_selection=True, render_geometry_previews=False,
        )
    record = json.loads((tmp_path / "out/task_evaluation_robot_placement_geometry_search.v1.json").read_text())
    assert record["status"] == "no_candidate"
    assert record["search_diagnostics"]["support_or_reach_prefilter_rejected_count"] == 72
    assert record["match_verdict"] == no_match.value.verdict
    assert record["match_verdict"]["reason_code"] == "no_supported_base_within_reach"
    assert record["record_digest"] == canonical_digest(record, digest_field="record_digest")
    assert record["native_placement_qualified"] is False
    assert record["policy_executed"] is False


def test_no_match_verdict_names_what_ruled_every_base_out():
    # 2026-10-26 dishwasher: every reachable floor spot was in the door's swing or in the cabinets.
    verdict = placement_no_match_verdict({
        "detailed_geometry_tested_count": 31, "geometry_passing_count": 0, "support_search_exhausted": True,
        "detailed_geometry_blocker_counts": {"robot_base_in_task_motion_path": 31,
                                             "robot_reset_bounds_overlap_scene_geometry": 12}},
        robot_id="franka_panda")
    assert verdict["status"] == "not_a_match" and verdict["reason_code"] == "robot_base_in_task_motion_path"
    assert verdict["summary"] == (
        "Not a match: none of the 31 reachable, supported franka_panda base positions passed "
        "(31 in the path of the task's moving part, 12 colliding with the scene in its starting pose).")
    reachable = placement_no_match_verdict(
        {"detailed_geometry_tested_count": 5, "geometry_passing_count": 2}, robot_id="franka_panda")
    assert reachable["reason_code"] == "task_path_unreachable"
    assert placement_no_match_verdict({}, robot_id="franka_panda")["reason_code"] == "candidate_inventory_rejected"


def test_cli_draws_trajectory_but_keeps_analytic_gate_point_scoped(
    tmp_path, monkeypatch
) -> None:
    import blueprint_pipeline.task_evaluation_robot_placement_agent_cli as module

    trajectory = {
        "phases": [
            {
                "phase_id": "precontact",
                "position_world_m": [2.79, -6.76, 0.818],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
            },
            {
                "phase_id": "push_contact",
                "position_world_m": [2.91, -6.76, 0.818],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
            },
        ]
    }
    proposal = {
        "candidate_id": "candidate",
        "support_surface_id": "surface",
        "pose": {
            "position_world_m": [3.0, -6.3, 0.75],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "geometry_gate_digest": "sha256:geometry",
        "shoulder_to_target_distance_m": 0.5,
        "trajectory_position_ik_gate_digest": "sha256:trajectory",
        "trajectory_minimum_manipulability": 0.2,
        "trajectory_position_ik_gate": {
            "trajectory_position_ik_gate_digest": "sha256:trajectory",
            "minimum_manipulability": 0.2,
        },
    }
    rendered_waypoints = []

    monkeypatch.setattr(module.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(module, "build_robot_placement_geometry_index", lambda **_: object())
    monkeypatch.setattr(module, "summarize_robot_placement_geometry", lambda *_a, **_k: {})
    def enumerate_candidates(**kwargs):
        assert kwargs["trajectory_waypoints_world_m"] == [
            [2.79, -6.76, 0.818],
            [2.91, -6.76, 0.818],
        ]
        assert kwargs["trajectory_phase_ids"] == ["precontact", "push_contact"]
        assert kwargs["trajectory_orientations_world_xyzw"] == [
            [0.0, 0.70710678, 0.0, 0.70710678],
            [0.0, 0.70710678, 0.0, 0.70710678],
        ]
        assert kwargs["geometry_worker_count"] == 4
        assert kwargs["trajectory_worker_count"] == 4
        return [proposal]

    monkeypatch.setattr(
        module, "enumerate_robot_placement_geometry_candidates", enumerate_candidates
    )

    def render(*, trajectory_waypoints_world_m=(), **_kwargs):
        rendered_waypoints.append(list(trajectory_waypoints_world_m))
        digest = "sha256:" + hashlib.sha256(_ONE_PIXEL_PNG).hexdigest()
        return [
            {
                "label": "top_down_xy",
                "digest": digest,
                "image_url": "data:image/png;base64,"
                + base64.b64encode(_ONE_PIXEL_PNG).decode("ascii"),
                "detail": "high",
            }
        ]

    def validate(
        *,
        index,
        proposal,
        target_position_world_m,
        robot_id,
        trajectory_waypoints_world_m=(),
        trajectory_phase_ids=(),
        trajectory_orientations_world_xyzw=(),
        trajectory_gate_override=None,
    ):
        assert index is not None
        assert proposal["candidate_id"] == "candidate"
        assert target_position_world_m == [3.0, -6.7, 0.8]
        assert robot_id == "franka_panda"
        if trajectory_gate_override is None:
            assert trajectory_waypoints_world_m == [
                [2.79, -6.76, 0.818],
                [2.91, -6.76, 0.818],
            ]
            assert trajectory_phase_ids == ["precontact", "push_contact"]
            assert trajectory_orientations_world_xyzw == [
                [0.0, 0.70710678, 0.0, 0.70710678],
                [0.0, 0.70710678, 0.0, 0.70710678],
            ]
        else:
            assert trajectory_gate_override == proposal[
                "trajectory_position_ik_gate"
            ]
        return {
            "status": "passed",
            "geometry_gate_digest": "sha256:geometry",
            "shoulder_to_target_distance_m": 0.5,
        }

    def run_agent(**kwargs):
        inventory = kwargs["scene_context"][
            "deterministic_geometry_passing_candidate_inventory"
        ]
        assert kwargs["scene_context"][
            "deterministic_geometry_passing_candidate_inventory_digest"
        ] == module.canonical_digest(
            {"trajectory_digest": None, "candidates": inventory}
        )
        assert kwargs["scene_context"]["model_must_select_exact_inventory_member"] is True
        kwargs["validate_candidate"](proposal)
        kwargs["render_candidate"](proposal, 0)
        assert kwargs["task_trajectory"] == trajectory
        return {"receipt_digest": "sha256:" + "a" * 64}

    monkeypatch.setattr(module, "render_robot_placement_geometry_previews", render)
    monkeypatch.setattr(module, "validate_robot_placement_geometry_candidate", validate)
    monkeypatch.setattr(module, "OpenAIAgentsSDKInvoker", lambda _config: object())
    monkeypatch.setattr(module, "run_task_evaluation_robot_placement_agent", run_agent)

    run_robot_placement_cli(
        run_id="trajectory-preview",
        scene_collision_usd=Path("scene.usda"),
        robot_asset_usd=Path("robot.usda"),
        target_position_world_m=[3.0, -6.7, 0.8],
        scene_binding={"scene": "839873"},
        task_binding={"task": "push"},
        overview_image_paths=[],
        output_dir=tmp_path / "output",
        max_rounds=2,
        candidate_inventory_cap=12,
        max_input_tokens=1_000,
        max_inference_cost_usd=1.0,
        allow_live_invocation=False,
        tracing_disabled=True,
        task_trajectory=trajectory,
    )

    assert rendered_waypoints == [
        [[2.79, -6.76, 0.818], [2.91, -6.76, 0.818]],
        [[2.79, -6.76, 0.818], [2.91, -6.76, 0.818]],
    ]
    checkpoint = json.loads(
        (tmp_path / "output" / "task_evaluation_robot_placement_candidate_inventory.v1.json")
        .read_text(encoding="utf-8")
    )
    assert checkpoint["status"] == "complete"
    assert checkpoint["checkpoint_digest"] == canonical_digest(
        checkpoint, digest_field="checkpoint_digest"
    )

    monkeypatch.setattr(
        module,
        "enumerate_robot_placement_geometry_candidates",
        lambda **_kwargs: pytest.fail("checkpoint reuse must not enumerate"),
    )
    import blueprint_pipeline.task_evaluation_robot_placement_inventory as inventory

    monkeypatch.setattr(inventory, "validate_robot_placement_geometry_candidate", validate)
    run_robot_placement_cli(
        run_id="trajectory-preview-resume",
        scene_collision_usd=Path("scene.usda"),
        robot_asset_usd=Path("robot.usda"),
        target_position_world_m=[3.0, -6.7, 0.8],
        scene_binding={"scene": "839873"},
        task_binding={"task": "push"},
        overview_image_paths=[],
        output_dir=tmp_path / "resumed-output",
        max_rounds=2,
        candidate_inventory_cap=12,
        max_input_tokens=1_000,
        max_inference_cost_usd=1.0,
        allow_live_invocation=False,
        tracing_disabled=True,
        task_trajectory=trajectory,
        candidate_inventory_checkpoint=checkpoint,
    )


def test_candidate_inventory_checkpoint_revalidates_before_reuse(monkeypatch) -> None:
    trajectory_gate = {
        "trajectory_position_ik_gate_digest": "sha256:trajectory",
        "minimum_manipulability": 0.2,
    }
    candidate = {
        "candidate_id": "candidate",
        "support_surface_id": "surface",
        "pose": {
            "position_world_m": [3.0, -6.3, 0.75],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "geometry_gate_digest": "sha256:geometry",
        "shoulder_to_target_distance_m": 0.5,
        "trajectory_position_ik_gate_digest": "sha256:trajectory",
        "trajectory_minimum_manipulability": 0.2,
        "trajectory_position_ik_gate": trajectory_gate,
    }
    checkpoint = build_candidate_inventory_checkpoint(
        robot_id="franka_panda",
        target_position_world_m=[3.0, -6.7, 0.8],
        maximum_candidates=48,
        trajectory_digest="sha256:plan",
        geometry_summary_digest="sha256:geometry-summary",
        candidates=[candidate],
    )
    validation_calls = []

    def validate(**kwargs):
        validation_calls.append(kwargs)
        return {
            "status": "passed",
            "geometry_gate_digest": "sha256:geometry",
            "shoulder_to_target_distance_m": 0.5,
        }

    import blueprint_pipeline.task_evaluation_robot_placement_inventory as inventory

    monkeypatch.setattr(inventory, "validate_robot_placement_geometry_candidate", validate)
    restored = validate_candidate_inventory_checkpoint(
        checkpoint=checkpoint,
        index=object(),
        robot_id="franka_panda",
        target_position_world_m=[3.0, -6.7, 0.8],
        maximum_candidates=48,
        trajectory_digest="sha256:plan",
        geometry_summary_digest="sha256:geometry-summary",
    )

    assert restored == [candidate]
    assert validation_calls[0]["trajectory_gate_override"] == trajectory_gate

    with pytest.raises(
        ValueError, match="robot_placement_candidate_inventory_checkpoint_invalid"
    ):
        validate_candidate_inventory_checkpoint(
            checkpoint=checkpoint,
            index=object(),
            robot_id="franka_panda",
            target_position_world_m=[3.1, -6.7, 0.8],
            maximum_candidates=48,
            trajectory_digest="sha256:plan",
            geometry_summary_digest="sha256:geometry-summary",
        )


def test_deterministic_selection_never_invokes_paid_model(tmp_path, monkeypatch) -> None:
    import blueprint_pipeline.task_evaluation_robot_placement_agent_cli as module

    candidate = {
        "candidate_id": "candidate-0001",
        "support_surface_id": "surface",
        "pose": {
            "position_world_m": [3.0, -6.3, 0.75],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "geometry_gate_digest": "sha256:geometry",
        "shoulder_to_target_distance_m": 0.5,
        "trajectory_position_ik_gate_digest": "sha256:trajectory",
        "trajectory_minimum_manipulability": 0.2,
        "trajectory_position_ik_gate": {"status": "passed"},
    }
    gate = {
        "schema_version": "task_evaluation_robot_placement_geometry_gate.v1",
        "candidate_id": candidate["candidate_id"],
        "declared_support_surface_id": "surface",
        "status": "passed",
        "blockers": [],
        "orientation_slew_feasibility": {"feasible": True, "blockers": []},
        "geometry_gate_digest": "",
    }
    gate["geometry_gate_digest"] = canonical_digest(
        gate, digest_field="geometry_gate_digest"
    )
    monkeypatch.setattr(module, "build_robot_placement_geometry_index", lambda **_: object())
    monkeypatch.setattr(module, "summarize_robot_placement_geometry", lambda *_a, **_k: {})
    monkeypatch.setattr(
        module, "enumerate_robot_placement_geometry_candidates", lambda **_: [candidate]
    )
    monkeypatch.setattr(module, "validate_robot_placement_geometry_candidate", lambda **_: gate)
    monkeypatch.setattr(module, "_reject_infeasible_orientation_slew", lambda **kwargs: kwargs["gate"])
    digest = "sha256:" + hashlib.sha256(_ONE_PIXEL_PNG).hexdigest()
    monkeypatch.setattr(
        module,
        "render_robot_placement_geometry_previews",
        lambda **_: [{
            "label": "top_down",
            "digest": digest,
            "image_url": "data:image/png;base64," + base64.b64encode(_ONE_PIXEL_PNG).decode("ascii"),
            "detail": "high",
        }],
    )
    monkeypatch.setattr(
        module,
        "OpenAIAgentsSDKInvoker",
        lambda *_a, **_k: pytest.fail("deterministic CPU path must not invoke OpenAI"),
    )

    receipt = run_robot_placement_cli(
        run_id="cpu-only",
        scene_collision_usd=Path("scene.usda"),
        robot_asset_usd=Path("robot.usda"),
        target_position_world_m=[3.0, -6.7, 0.8],
        scene_binding={"scene": "839873"},
        task_binding={"task": "push"},
        overview_image_paths=[],
        output_dir=tmp_path / "output",
        max_rounds=1,
        candidate_inventory_cap=12,
        max_input_tokens=1_000,
        max_inference_cost_usd=0.0,
        allow_live_invocation=False,
        tracing_disabled=True,
        deterministic_selection=True,
    )

    validated = validate_robot_placement_receipt(
        receipt,
        expected_scene_binding_digest=canonical_digest({"scene": "839873"}),
        expected_task_binding_digest=canonical_digest({"task": "push"}),
    )
    assert validated["selection_method"] == "deterministic_inventory_rank"
    assert validated["visual_review_completed"] is False
    assert validated["native_camera_visibility_required"] is True

    forged = json.loads(json.dumps(receipt))
    forged["claim_ceiling"] = "analytic_and_visual_robot_placement_candidate"
    forged["receipt_digest"] = canonical_digest(
        forged, digest_field="receipt_digest"
    )
    with pytest.raises(ValueError, match="robot_placement_receipt_invalid"):
        validate_robot_placement_receipt(forged)


def test_live_selection_persists_interruption_safe_inference_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import blueprint_pipeline.task_evaluation_robot_placement_agent_cli as module

    candidate = {
        "candidate_id": "candidate-0001",
        "support_surface_id": "surface",
        "pose": {
            "position_world_m": [3.0, -6.3, 0.75],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "geometry_gate_digest": "sha256:geometry",
        "shoulder_to_target_distance_m": 0.5,
        "trajectory_position_ik_gate_digest": "sha256:trajectory",
        "trajectory_minimum_manipulability": 0.2,
        "trajectory_position_ik_gate": {"status": "passed"},
    }
    gate = {
        "schema_version": "task_evaluation_robot_placement_geometry_gate.v1",
        "candidate_id": candidate["candidate_id"],
        "declared_support_surface_id": "surface",
        "status": "passed",
        "blockers": [],
        "orientation_slew_feasibility": {"feasible": True, "blockers": []},
        "geometry_gate_digest": "sha256:geometry",
    }
    monkeypatch.setattr(
        module, "build_robot_placement_geometry_index", lambda **_: object()
    )
    monkeypatch.setattr(
        module, "summarize_robot_placement_geometry", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        module,
        "enumerate_robot_placement_geometry_candidates",
        lambda **_: [candidate],
    )
    monkeypatch.setattr(
        module, "validate_robot_placement_geometry_candidate", lambda **_: gate
    )
    digest = "sha256:" + hashlib.sha256(_ONE_PIXEL_PNG).hexdigest()
    monkeypatch.setattr(
        module,
        "render_robot_placement_geometry_previews",
        lambda **_: [
            {
                "label": "top_down",
                "digest": digest,
                "image_url": "data:image/png;base64,"
                + base64.b64encode(_ONE_PIXEL_PNG).decode("ascii"),
                "detail": "high",
            }
        ],
    )

    audits = []

    class FakeAudit:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.manifest_writes = 0
            audits.append(self)

        def record_reservation(self, _value):
            return None

        def record_completion(self, _value):
            return None

        def write_manifest(self):
            self.manifest_writes += 1

    invokers = []

    class FakeInvoker:
        def __init__(self, _config):
            self.reservation_audit_configured = False
            invokers.append(self)

        def configure_reservation_audit(self, **kwargs):
            assert kwargs["restored_reserved_cost_usd"] == 0.0
            assert callable(kwargs["record_reservation"])
            assert callable(kwargs["record_completion"])
            self.reservation_audit_configured = True

    def run_agent(**kwargs):
        assert kwargs["invoker"].reservation_audit_configured is True
        return {"receipt_digest": "sha256:" + "b" * 64}

    monkeypatch.setattr(module, "InferenceReservationAudit", FakeAudit)
    monkeypatch.setattr(module, "OpenAIAgentsSDKInvoker", FakeInvoker)
    monkeypatch.setattr(module, "run_task_evaluation_robot_placement_agent", run_agent)

    run_robot_placement_cli(
        run_id="live-audited-selection",
        scene_collision_usd=Path("scene.usda"),
        robot_asset_usd=Path("robot.usda"),
        target_position_world_m=[3.0, -6.7, 0.8],
        scene_binding={"scene": "839873"},
        task_binding={"task": "push"},
        overview_image_paths=[],
        output_dir=tmp_path / "output",
        max_rounds=1,
        candidate_inventory_cap=12,
        max_input_tokens=1_000,
        max_inference_cost_usd=1.0,
        allow_live_invocation=True,
        tracing_disabled=True,
        record_inference_reservations=True,
    )

    assert len(invokers) == 1
    assert len(audits) == 1
    assert audits[0].kwargs["run_id"] == "live-audited-selection"
    assert audits[0].manifest_writes == 1
