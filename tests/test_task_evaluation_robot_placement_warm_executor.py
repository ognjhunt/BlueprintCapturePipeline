from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_robot_placement_trajectory import (
    placement_trajectory_from_native_plan,
)
from blueprint_pipeline.task_evaluation_robot_placement_warm_executor import (
    CONFIG_SCHEMA_VERSION,
    WarmNativePlacementExecutor,
    _run_control_search_on_warm_session,
)
from blueprint_pipeline.task_evaluation_control_search_funnel import (
    OUTCOME_SCHEMA_VERSION,
    build_control_search_sweep_result,
)


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sealed(value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _sweep_candidate(index: int) -> dict:
    reset = _sealed(
        {
            "schema_version": "task_evaluation_native_robot_reset_variant.v1",
            "robot_joint_reset_positions_rad": {
                f"joint_{joint}": float(joint) * 0.01 for joint in range(7)
            },
            "reset_variant_digest": "",
        },
        "reset_variant_digest",
    )
    entry = _sealed(
        {
            "schema_version": "task_evaluation_native_entry_trajectory_variant.v1",
            "joins_authored_phase_id": "precontact",
            "waypoints": [
                {
                    "waypoint_id": "entry-00",
                    "position_world_m": [2.8, -6.7, 1.0],
                    "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                }
            ],
            "entry_trajectory_variant_digest": "",
        },
        "entry_trajectory_variant_digest",
    )
    camera = _sealed(
        {
            "schema_version": "task_evaluation_native_camera_variant.v1",
            "cameras": [
                {"role": "external"},
                {"role": "wrist"},
                {"role": "overview"},
            ],
            "camera_variant_digest": "",
        },
        "camera_variant_digest",
    )
    return _sealed(
        {
            "schema_version": "task_evaluation_native_construction_candidate.v1",
            "candidate_id": f"curobo-{index:02d}",
            "deterministic_rank": index,
            "robot_base_pose_world": {
                "position_world_m": [2.8 + index * 0.01, -6.13, 0.752958],
                "orientation_xyzw": [0.0, 0.0, 0.608761429, -0.79335334],
            },
            "support_surface_id": "/Site/counter",
            "reset_variant": reset,
            "entry_trajectory_variant": entry,
            "camera_variant": camera,
            "maximum_incremental_cost_usd": 0.1,
            "maximum_runtime_seconds": 120.0,
            "addressed_feedback_codes": [],
            "candidate_digest": "",
        },
        "candidate_digest",
    )


def _sweep_inventory(count: int = 10) -> dict:
    return _sealed(
        {
            "schema_version": (
                "task_evaluation_native_construction_candidate_inventory.v1"
            ),
            "run_id": "scene-839873-control-search",
            "round_index": 0,
            "source_native_feedback_digest": "sha256:" + "0" * 64,
            "model_authored_candidates": False,
            "candidates": [_sweep_candidate(index) for index in range(count)],
            "inventory_digest": "",
        },
        "inventory_digest",
    )


def _native_plan() -> dict:
    plan = {
        "schema_version": "native_rigid_construction_phase_plan.v1",
        "task_kind": "rigid_pick_place",
        "manipulation_strategy": "planar_push",
        "phase_count": 1,
        "execution_parameters": {
            "arrival_tolerance_m": 0.02,
            "arrival_orientation_tolerance_rad": 0.08,
        },
        "phases": [
            {
                "phase_id": "precontact",
                "position_world_m": [2.79, -6.76, 0.818],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                "gripper_state": "open",
                "gate_ids": ["precontact_reachability"],
            }
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _config(tmp_path: Path, *, compiler_reference: bool = False) -> dict:
    controls = _write(tmp_path / "controls.json", {"schema_version": "fixture.v1"})
    profile = _write(tmp_path / "profile.json", {"schema_version": "fixture.v1"})
    reference = {"digest": "sha256:x"}
    if compiler_reference:
        reference = {
            "schema_version": "task_evaluation_diagnostic_native_arena_compiler_output.v1",
            "status": "completed_development_only",
            "droid_profile_reference": reference,
            "compiler_output_digest": "",
        }
        reference["compiler_output_digest"] = canonical_digest(
            reference, digest_field="compiler_output_digest"
        )
    profile_ref = _write(tmp_path / "profile-ref.json", reference)
    source = _write(tmp_path / "source.json", {"schema_version": "fixture.v1"})
    session = _write(
        tmp_path / "session.json",
        {"schema_version": "native_task_arena_warm_session.v1", "instance_id": 49104791},
    )
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "diagnostic_controls_input_path": str(controls),
        "droid_profile_path": str(profile),
        "droid_profile_reference_path": str(profile_ref),
        "runtime_source_packet_receipt_path": str(source),
        "warm_session_path": str(session),
        "implementation_commit": "a" * 40,
        "authorization_reference": "owner-authorized bounded placement loop",
        "authorized_by": "task-evaluation-owner",
        "authorized_on": "2026-08-29T00:00:00Z",
        "max_hourly_rate_usd": 0.8,
        "hard_cap_usd": 2.0,
        "hard_ttl_seconds": 3600,
    }


def test_trajectory_binding_rejects_changed_phase_position() -> None:
    import blueprint_pipeline.task_evaluation_robot_placement_warm_executor as module

    plan = _native_plan()
    trajectory = placement_trajectory_from_native_plan(plan)
    changed = json.loads(json.dumps(plan))
    changed["phases"][0]["position_world_m"][0] += 0.01
    changed["plan_digest"] = canonical_digest(changed, digest_field="plan_digest")

    assert not module._trajectory_content_matches(trajectory, changed)


def test_warm_control_search_runs_once_and_returns_only_ranked_shortlist(
    tmp_path: Path,
) -> None:
    inventory = _sweep_inventory()
    runtime_receipt = _write(
        tmp_path / "runtime-source.json",
        {"receipt_digest": "sha256:" + "3" * 64},
    )
    authority = {
        "schema_version": "task_evaluation_control_search_authority.v1",
        "enabled": True,
        "claim_ceiling": "development_only_control_search",
        "provider_allocations_performed": 0,
        "requested_vector_env_count": 8,
        "maximum_vector_env_count": 1_024,
        "seeds_per_candidate": 1,
        "shortlist_size": 8,
        "appearance_mode": "omitted",
        "camera_mode": "disabled",
        "full_fidelity_replay_required": True,
        "authority_digest": "",
    }
    authority["authority_digest"] = canonical_digest(
        authority, digest_field="authority_digest"
    )
    request = {
        "request_digest": "sha256:" + "2" * 64,
        "scenario": {"seed": 839873104},
        "assets": [
            {
                "semantic_role": "scene_collision",
                "source": {"sha256": "sha256:" + "4" * 64},
            },
            {
                "semantic_role": "task_object",
                "source": {"sha256": "sha256:" + "5" * 64},
            },
        ],
    }
    calls = []

    class FakeRunner:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def execute(self, *, plan, schedule, candidate_inventory):
            outcomes = []
            for wave in schedule["waves"]:
                for assignment in wave["assignments"]:
                    # Lower deterministic ranks are deliberately worse here;
                    # this proves the native replay inventory follows measured
                    # physics ranking, not source order.
                    rank = next(
                        row["deterministic_rank"]
                        for row in candidate_inventory["candidates"]
                        if row["candidate_id"] == assignment["candidate_id"]
                    )
                    outcome = {
                        "schema_version": OUTCOME_SCHEMA_VERSION,
                        **assignment,
                        "reset_readback_passed": True,
                        "forbidden_collision_peak_force_n": float(10 - rank),
                        "required_task_contact_coverage_fraction": 1.0,
                        "push_path_tracking_error_m": 0.01,
                        "destination_error_m": 0.02,
                        "support_stability_error_m": 0.001,
                        "task_displacement_m": 0.12,
                        "physics_steps": 220,
                        "measurement_authority": (
                            "isaac_lab_simulator_state_and_contact_sensors"
                        ),
                        "learned_grader_used": False,
                        "outcome_digest": "",
                    }
                    outcome.pop("assignment_index")
                    outcome["outcome_digest"] = canonical_digest(
                        outcome, digest_field="outcome_digest"
                    )
                    outcomes.append(outcome)
            return build_control_search_sweep_result(
                plan=plan,
                outcomes=outcomes,
                actual_vector_env_count=schedule["vector_env_count"],
                peak_gpu_memory_bytes=18_000_000_000,
            )

    execution = _run_control_search_on_warm_session(
        request=request,
        authority=authority,
        candidate_inventory=inventory,
        candidate_generator_context=SimpleNamespace(
            robot_configuration={"digest": "sha256:" + "6" * 64},
            task_trajectory={"digest": "sha256:" + "7" * 64},
        ),
        warm_session={
            "status": "ready",
            "continuing_spend": True,
            "remote_work_dir": "/workspace",
        },
        runtime_source_packet_receipt_path=runtime_receipt,
        implementation_commit="a" * 40,
        output_root=tmp_path / "feedback",
        remote_python_package_root=(
            "/workspace/adp_arena_provider_bundle/provider_runtime"
        ),
        sweep_runner_factory=FakeRunner,
    )

    assert len(calls) == 1
    assert execution["status"] == "completed_development_only"
    assert execution["provider_allocations_performed"] == 0
    assert len(execution["candidate_inventory"]["candidates"]) == 8
    assert execution["candidate_inventory"]["candidates"][0][
        "candidate_id"
    ] == "curobo-09"
    assert Path(execution["plan_path"]).is_file()
    assert Path(execution["schedule_path"]).is_file()
    assert Path(execution["result_path"]).is_file()


def test_one_agent_run_revises_multiple_poses_on_same_warm_instance(
    tmp_path, monkeypatch
) -> None:
    import blueprint_pipeline.task_evaluation_robot_placement_warm_executor as module

    plan = _native_plan()
    trajectory = placement_trajectory_from_native_plan(plan)
    recompiled_plan = {**plan, "scene_plan_digest": "sha256:new-robot-base"}
    recompiled_plan["plan_digest"] = canonical_digest(
        recompiled_plan, digest_field="plan_digest"
    )
    assert recompiled_plan["plan_digest"] != plan["plan_digest"]
    allocator_calls = []

    def fake_compile(
        *, output_root, droid_profile_reference, task_trajectory, **_kwargs
    ):
        assert droid_profile_reference == {"digest": "sha256:x"}
        assert task_trajectory == trajectory
        packet = Path(output_root) / "native-task-packet"
        packet.mkdir(parents=True)
        _write(packet / "native_task_arena_scene_plan.v1.json", {"fixture": True})
        receipt = _write(
            packet / "native_task_arena_packet_receipt.v1.json",
            {"receipt_digest": "sha256:packet"},
        )
        return {"packet_receipt_path": str(receipt)}

    def fake_build(*, job_dir, **_kwargs):
        root = Path(job_dir)
        root.mkdir(parents=True)
        _write(root / "native_task_arena_provider_bundle_receipt.v1.json", {})
        return {
            "schema_version": "native_task_arena_provider_bundle.v1",
            "execution_mode": "construction_canary",
            "implementation_commit": "a" * 40,
            "bundle_sha256": "sha256:bundle",
            "input_digest": "sha256:input",
        }

    def fake_authority(*, output_path, **_kwargs):
        _write(Path(output_path), {"schema_version": "fixture-authority.v1"})

    def fake_allocator(argv):
        argv = list(argv or [])
        allocator_calls.append(argv)
        adapter = Path(argv[argv.index("--adapter-output") + 1])
        job = Path(argv[argv.index("--adp-job-dir") + 1])
        execution = job / "immutable_execution"
        frame = execution / "construction_frames" / "external" / "precontact.png"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(_ONE_PIXEL_PNG)
        passed = len(allocator_calls) == 2
        native = {
            "schema_version": "native_task_arena_construction_result.v1",
            "status": "completed" if passed else "blocked",
            "construction_gate_qualified": passed,
            "phase_reached": "controls" if passed else "construction",
            "blockers": [] if passed else ["native_task_phase_ik_unreached:precontact"],
            "initial_readback": {
                "robot_root_pose_world": [
                    3.4,
                    -6.1,
                    0.7545,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            },
            "phase_results": [
                {
                    "phase_id": "precontact",
                    "steps": 64,
                    "target_position_world_m": [2.79, -6.76, 0.818],
                    "terminal_position_world_m": [2.90, -6.78, 0.80],
                    "terminal_position_error_m": 0.05563,
                    "terminal_orientation_error_rad": 1.037,
                    "target_reached": False,
                }
            ],
            "camera_gates": {},
            "result_digest": "",
        }
        native["result_digest"] = canonical_digest(native, digest_field="result_digest")
        native_path = _write(
            execution / "native_task_arena_construction_result.v1.json", native
        )
        _write(
            adapter,
            {
                "status": "completed" if passed else "blocked",
                "blockers": native["blockers"],
                "provider_instance_id": 49104791,
                "provider_allocations_performed": 0,
                "runtime_seconds": 1.0,
                "incremental_cost_upper_bound_usd": 0.001,
                "native_construction_result_path": str(native_path),
            },
        )
        return 0 if passed else 2

    monkeypatch.setattr(module, "compile_diagnostic_native_arena_packet", fake_compile)
    monkeypatch.setattr(module, "build_native_task_arena_construction_bundle", fake_build)
    monkeypatch.setattr(
        module, "materialize_native_task_arena_warm_attempt_authority", fake_authority
    )
    monkeypatch.setattr(
        module,
        "materialize_native_task_construction_phase_plan",
        lambda _scene: recompiled_plan,
    )
    executor = WarmNativePlacementExecutor(
        config=_config(tmp_path, compiler_reference=True),
        task_trajectory=trajectory,
        output_root=tmp_path / "rounds",
        allocator_main=fake_allocator,
    )

    rejected = executor(
        {"candidate_id": "pose-a"}, {"receipt_digest": "sha256:a"}, 0
    )
    passed = executor(
        {"candidate_id": "pose-b"}, {"receipt_digest": "sha256:b"}, 1
    )

    assert rejected["status"] == "rejected"
    assert rejected["feedback_images"][0]["label"] == "native_external_precontact"
    assert rejected["native_feedback"]["initial_robot_root_pose_world"] == [
        3.4,
        -6.1,
        0.7545,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert rejected["native_feedback"]["phase_results"][0] == {
        "phase_id": "precontact",
        "steps": 64,
        "target_position_world_m": [2.79, -6.76, 0.818],
        "terminal_position_world_m": [2.90, -6.78, 0.80],
        "terminal_position_error_m": 0.05563,
        "terminal_orientation_error_rad": 1.037,
        "target_reached": False,
    }
    assert passed["status"] == "passed"
    assert len(allocator_calls) == 2
    assert all("49104791" in call for call in allocator_calls)
    assert all(call.count("--execute") == 1 for call in allocator_calls)
    assert all(
        json.loads((tmp_path / "rounds" / f"round-{index:02d}" / "native_attempt.v1.json").read_text())["provider_allocations_performed"]
        == 0
        for index in range(2)
    )
