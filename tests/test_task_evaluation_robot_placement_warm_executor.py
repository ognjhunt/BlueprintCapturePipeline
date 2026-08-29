from __future__ import annotations

import base64
import json
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_robot_placement_trajectory import (
    placement_trajectory_from_native_plan,
)
from blueprint_pipeline.task_evaluation_robot_placement_warm_executor import (
    CONFIG_SCHEMA_VERSION,
    WarmNativePlacementExecutor,
)


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


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
