from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_configured_controls_autostart as autostart
from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as worker


COMMIT = "a" * 40


def _write(path: Path, payload: bytes = b"{}\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path.resolve()


def _release_template() -> bytes:
    value = {
        "schema_version": "task_evaluation_configured_controls_release_window_template.v1",
        "status": "authorized_for_dynamic_release",
        "team_namespace": "blueprint-adp",
        "expected_production_commit": COMMIT,
        "allowed_mutations": [
            "profile_publication",
            "catalog_synchronization",
            "standing_authorization",
        ],
        "provider_allowlist": ["vast"],
        "maximum_hard_cap_usd": 1.0,
        "valid_for_seconds": 3600,
        "released_by": "configured-controls-coordinator",
        "release_reference": "ADP-009D Day-28 configured-controls continuation",
        "provider_resource_allocation_allowed": False,
        "paid_request_allowed": False,
        "template_digest": "",
    }
    value["template_digest"] = autostart.canonical_digest(
        value, digest_field="template_digest"
    )
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _intent(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    names = (
        "robot_asset_usd_path",
        "robot_mount_interface_path",
        "scene_camera_calibration_path",
        "native_trajectory_plan_path",
        "cameras_path",
        "runtime_binding_path",
    )
    paths: dict[str, object] = {
        name: str(_write(tmp_path / "inputs" / f"{name}.json")) for name in names
    }
    paths["overview_image_paths"] = [
        str(_write(tmp_path / "inputs" / "overview.png", b"png"))
    ]
    phases: dict[str, dict[str, str]] = {}
    for phase in ("construction", "controls"):
        phase_names = [
            "release_window_template_path",
            "authorization_path",
            "launch_authority_path",
        ]
        if phase == "construction":
            phase_names.append("lineage_path")
        phases[phase] = {}
        for name in phase_names:
            payload = _release_template() if name == "release_window_template_path" else b"{}\n"
            phases[phase][name] = str(
                _write(tmp_path / "inputs" / phase / f"{name}.json", payload)
            )
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    destination = tmp_path / "intent.json"
    value = autostart.materialize_configured_controls_autostart_intent(
        expected_production_commit=COMMIT,
        submitted_by="configured-controls-autostart",
        team_namespace="blueprint-adp",
        scene_id="interiorgs-839873",
        task_id="scene-839873-mug-planar-push",
        target_position_world_m=[2.9, -6.7, 0.8],
        paths=paths,
        phases=phases,
        profile_dir=profile_dir,
        output_path=destination,
        openai_project_id="proj_test",
        openai_api_key_id="key_visual_review",
    )
    return destination, value


def test_intent_binds_every_fixed_cpu_and_paid_boundary_input(tmp_path: Path) -> None:
    path, value = _intent(tmp_path)

    assert path.stat().st_mode & 0o777 == 0o440
    assert value["paid_execution_requested"] is True
    assert value["placement"]["agent_model"] == "gpt-5.6-sol"
    assert value["placement"]["reasoning_effort"] == "high"
    assert value["provider_mutation_performed"] is False
    assert set(value["artifact_inventory"]) == {
        "robot_asset_usd_path",
        "robot_mount_interface_path",
        "scene_camera_calibration_path",
        "native_trajectory_plan_path",
        "cameras_path",
        "runtime_binding_path",
        "overview_image_paths.0",
        "phases.construction.release_window_template_path",
        "phases.construction.lineage_path",
        "phases.construction.authorization_path",
        "phases.construction.launch_authority_path",
        "phases.controls.release_window_template_path",
        "phases.controls.authorization_path",
        "phases.controls.launch_authority_path",
    }


def test_intent_rejects_changed_robot_or_authority_bytes(tmp_path: Path) -> None:
    _, value = _intent(tmp_path)
    robot = Path(value["paths"]["robot_asset_usd_path"])
    robot.chmod(0o640)
    robot.write_bytes(b"changed")

    with pytest.raises(
        autostart.TaskEvaluationConfiguredControlsAutostartError,
        match="configured_controls_autostart_inventory_invalid",
    ):
        autostart.validate_configured_controls_autostart_intent(value)


def test_explicit_prior_configuration_adoption_is_separate_and_exact(
    tmp_path: Path,
) -> None:
    _, automatic = _intent(tmp_path)
    adoption = {
        "mode": "explicit_terminal_adoption",
        "source_launch_id": "scene-839873-2deff449-r1",
        "source_launch_receipt_digest": "sha256:" + "1" * 64,
        "terminal_result_digest": "sha256:" + "2" * 64,
        "configured_scene_revision_digest": "sha256:" + "3" * 64,
        "publication_result_digest": "sha256:" + "4" * 64,
        "webapp_sync_result_digest": "sha256:" + "5" * 64,
        "provider_zero_receipt_digest": "sha256:" + "6" * 64,
    }
    adopted = autostart.materialize_configured_controls_autostart_intent(
        expected_production_commit=COMMIT,
        configuration_source_commit="2" * 40,
        configuration_adoption=adoption,
        submitted_by="configured-controls-adoption",
        team_namespace=str(automatic["team_namespace"]),
        scene_id=str(automatic["scene_id"]),
        task_id=str(automatic["task_id"]),
        target_position_world_m=automatic["target_position_world_m"],
        paths=automatic["paths"],
        phases=automatic["phases"],
        profile_dir=str(automatic["profile_dir"]),
        output_path=tmp_path / "adoption-intent.json",
        openai_project_id="proj_test",
        openai_api_key_id="key_visual_review",
    )

    assert automatic["configuration_adoption"] == {
        "mode": "same_commit_automatic"
    }
    assert adopted["configuration_source_commit"] == "2" * 40
    assert autostart.configured_controls_autostart_registry_name(
        team_namespace=str(automatic["team_namespace"]),
        scene_id=str(automatic["scene_id"]),
        task_id=str(automatic["task_id"]),
    ) != autostart.configured_controls_autostart_adoption_registry_name(
        team_namespace=str(adopted["team_namespace"]),
        scene_id=str(adopted["scene_id"]),
        task_id=str(adopted["task_id"]),
        source_launch_id=str(adoption["source_launch_id"]),
    )

    evidence = {
        "source_launch_id": adoption["source_launch_id"],
        "terminal": {"result_digest": adoption["terminal_result_digest"]},
        "receipt": {"receipt_digest": adoption["source_launch_receipt_digest"]},
        "revision": {
            "revision_digest": adoption["configured_scene_revision_digest"]
        },
        "publication": {"result_digest": adoption["publication_result_digest"]},
        "sync": {"sync_result_digest": adoption["webapp_sync_result_digest"]},
        "zero": {
            "provider_zero_receipt_digest": adoption[
                "provider_zero_receipt_digest"
            ]
        },
    }
    autostart._validate_configuration_adoption(adoption=adoption, **evidence)
    evidence["sync"] = {"sync_result_digest": "sha256:" + "9" * 64}
    with pytest.raises(
        autostart.TaskEvaluationConfiguredControlsAutostartError,
        match="configured_controls_autostart_adoption_evidence_invalid",
    ):
        autostart._validate_configuration_adoption(adoption=adoption, **evidence)


def test_worker_materializes_cpu_autostart_before_advancing_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_root = tmp_path / "launch-runs"
    run_root = launch_root / "scene-839873-launch"
    run_root.mkdir(parents=True)
    (run_root / "launch_profile.json").write_text(
        json.dumps(
            {
                "immutable_inputs": [
                    {"name": "configured_controls_autostart_intent"}
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_root / "launch_receipt.json").write_text(
        '{"status":"completed"}', encoding="utf-8"
    )
    (run_root / "webapp_sync_succeeded.json").write_text("{}", encoding="utf-8")
    (run_root / "post_teardown_provider_zero_receipt.json").write_text(
        "{}", encoding="utf-8"
    )
    plan_root = tmp_path / "plans"

    def materialize(**kwargs):
        plan_root.mkdir()
        (plan_root / "scene-839873-launch.json").write_text("{}", encoding="utf-8")
        return {
            "status": "agent_binding_accepted_plan_materialized",
            "selected_candidate_id": "candidate-0042",
            "plan_digest": "sha256:" + "1" * 64,
        }

    monkeypatch.setattr(autostart, "materialize_configured_controls_autostart", materialize)
    monkeypatch.setattr(
        worker,
        "advance_configured_controls_plan",
        lambda **kwargs: {
            "status": "episode_preparation_queued",
            "source_launch_id": Path(kwargs["plan_path"]).stem,
        },
    )
    report = worker.process_plans(
        plan_root=plan_root,
        launch_state_root=launch_root,
        progression_root=tmp_path / "progression",
        preparation_queue_root=tmp_path / "preparations",
        activation_queue_root=tmp_path / "activations",
    )

    assert report["status"] == "completed"
    assert [row["status"] for row in report["rows"]] == [
        "agent_binding_accepted_plan_materialized",
        "episode_preparation_queued",
    ]
    assert report["rows"][0]["selected_candidate_id"] == "candidate-0042"
    assert report["allocator_invoked"] is False
    assert report["provider_mutation_performed"] is False


def test_cpu_placement_retry_uses_fresh_attempt_then_reopens_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory_digest = "sha256:" + "4" * 64
    receipt = {
        "status": "accepted",
        "candidate_inventory_digest": inventory_digest,
    }
    inventory = {
        "candidate_inventory_digest": inventory_digest,
        "checkpoint_digest": "",
    }
    inventory["checkpoint_digest"] = autostart.canonical_digest(
        inventory, digest_field="checkpoint_digest"
    )
    calls: list[Path] = []

    def flaky_runner(*, output_dir: Path, **_kwargs):
        calls.append(output_dir)
        if len(calls) == 1:
            (output_dir / "partial-preview.png").write_bytes(b"partial")
            raise RuntimeError("transient_cpu_failure")
        (output_dir / "task_evaluation_robot_placement_receipt.v1.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        (
            output_dir
            / "task_evaluation_robot_placement_candidate_inventory.v1.json"
        ).write_text(json.dumps(inventory), encoding="utf-8")
        return receipt

    monkeypatch.setattr(
        autostart,
        "validate_robot_placement_receipt",
        lambda value, **_kwargs: dict(value),
    )
    with pytest.raises(RuntimeError, match="transient_cpu_failure"):
        autostart._placement_checkpoint(
            root=tmp_path / "binding",
            placement_runner=flaky_runner,
            runner_kwargs={},
            expected_scene_binding_digest="sha256:" + "1" * 64,
            expected_task_binding_digest="sha256:" + "2" * 64,
        )

    accepted, reopened_inventory, attempt = autostart._placement_checkpoint(
        root=tmp_path / "binding",
        placement_runner=flaky_runner,
        runner_kwargs={},
        expected_scene_binding_digest="sha256:" + "1" * 64,
        expected_task_binding_digest="sha256:" + "2" * 64,
    )
    assert accepted == receipt
    assert reopened_inventory == inventory
    assert attempt.name == "attempt_001"
    assert (tmp_path / "binding/placement-attempts/attempt_000/partial-preview.png").is_file()

    def forbidden_runner(**_kwargs):
        raise AssertionError("complete checkpoint must be reopened")

    reopened = autostart._placement_checkpoint(
        root=tmp_path / "binding",
        placement_runner=forbidden_runner,
        runner_kwargs={},
        expected_scene_binding_digest="sha256:" + "1" * 64,
        expected_task_binding_digest="sha256:" + "2" * 64,
    )
    assert reopened[0] == receipt


def test_cpu_placement_checkpoint_is_namespaced_by_exact_binding_and_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "binding"
    scene_digest = "sha256:" + "1" * 64
    stale_task_digest = "sha256:" + "2" * 64
    current_task_digest = "sha256:" + "3" * 64
    current_checkpoint_binding_digest = (
        autostart._cpu_placement_checkpoint_binding_digest(
            intent_digest="sha256:" + "4" * 64,
            scene_binding_digest=scene_digest,
            task_binding_digest=current_task_digest,
        )
    )

    def validate(value, *, expected_scene_binding_digest, expected_task_binding_digest):
        receipt = dict(value)
        if (
            receipt["scene_binding_digest"] != expected_scene_binding_digest
            or receipt["task_binding_digest"] != expected_task_binding_digest
        ):
            raise autostart.TaskEvaluationConfiguredControlsAutostartError(
                "robot_placement_task_binding_mismatch"
            )
        return receipt

    monkeypatch.setattr(autostart, "validate_robot_placement_receipt", validate)

    def runner_for(task_digest: str):
        def runner(*, output_dir: Path, **_kwargs):
            inventory = {
                "candidate_inventory_digest": "sha256:" + "5" * 64,
                "checkpoint_digest": "",
            }
            inventory["checkpoint_digest"] = autostart.canonical_digest(
                inventory, digest_field="checkpoint_digest"
            )
            receipt = {
                "scene_binding_digest": scene_digest,
                "task_binding_digest": task_digest,
                "candidate_inventory_digest": inventory["candidate_inventory_digest"],
            }
            (output_dir / "task_evaluation_robot_placement_receipt.v1.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            (
                output_dir
                / "task_evaluation_robot_placement_candidate_inventory.v1.json"
            ).write_text(json.dumps(inventory), encoding="utf-8")
            return receipt

        return runner

    autostart._placement_checkpoint(
        root=root,
        placement_runner=runner_for(stale_task_digest),
        runner_kwargs={},
        expected_scene_binding_digest=scene_digest,
        expected_task_binding_digest=stale_task_digest,
    )
    legacy_checkpoint = root / "cpu-placement-checkpoint.v1.json"
    legacy_bytes = legacy_checkpoint.read_bytes()

    current_root = autostart._cpu_placement_checkpoint_root(
        root=root,
        checkpoint_binding_digest=current_checkpoint_binding_digest,
    )
    current_checkpoint_kwargs = {
        "root": current_root,
        "runner_kwargs": {},
        "expected_scene_binding_digest": scene_digest,
        "expected_task_binding_digest": current_task_digest,
        "expected_checkpoint_binding_digest": current_checkpoint_binding_digest,
        "checkpoint_file_name": "cpu-placement-checkpoint.v2.json",
        "checkpoint_schema_version": (
            autostart._BOUND_CPU_PLACEMENT_CHECKPOINT_SCHEMA_VERSION
        ),
    }
    accepted, _inventory, attempt = autostart._placement_checkpoint(
        placement_runner=runner_for(current_task_digest),
        **current_checkpoint_kwargs,
    )
    assert accepted["task_binding_digest"] == current_task_digest
    assert attempt.name == "attempt_000"
    assert legacy_checkpoint.read_bytes() == legacy_bytes
    assert (root / "placement-attempts/attempt_000").is_dir()
    current_checkpoint = current_root / "cpu-placement-checkpoint.v2.json"
    assert current_checkpoint.is_file()
    reopened = autostart._placement_checkpoint(
        placement_runner=lambda **_kwargs: pytest.fail(
            "the exact current binding must reopen its completed checkpoint"
        ),
        **current_checkpoint_kwargs,
    )
    assert reopened[2] == attempt

    checkpoint = json.loads(current_checkpoint.read_text(encoding="utf-8"))
    checkpoint["checkpoint_binding_digest"] = "sha256:" + "9" * 64
    checkpoint["checkpoint_digest"] = autostart.canonical_digest(
        checkpoint, digest_field="checkpoint_digest"
    )
    current_checkpoint.chmod(0o640)
    current_checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(
        autostart.TaskEvaluationConfiguredControlsAutostartError,
        match="configured_controls_autostart_placement_checkpoint_invalid",
    ):
        autostart._placement_checkpoint(
            placement_runner=lambda **_kwargs: pytest.fail(
                "tampered same-key checkpoint must not create another attempt"
            ),
            **current_checkpoint_kwargs,
        )


def _camera_template() -> dict[str, object]:
    intrinsics = {
        "cx": 159.5,
        "cy": 89.5,
        "fx": 172.88839142740494,
        "fy": 172.88839142740494,
        "height": 180,
        "width": 320,
    }
    world = {
        "policy_input": True,
        "scoring_input": False,
        "pose_frame": "world",
        "parent_prim_path": "{ENV_REGEX_NS}",
        "optical_convention": "opencv",
        "frame_from_camera_matrix": [
            1.0, 0.0, 0.0, 100.0,
            0.0, 1.0, 0.0, 100.0,
            0.0, 0.0, 1.0, 100.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        "intrinsics": intrinsics,
    }
    return {
        "schema_version": "native_task_arena_packet_request.v1",
        "cameras": [
            {"role": "external", **world},
            {
                "role": "wrist",
                "policy_input": True,
                "scoring_input": False,
                "pose_frame": "robot_body",
                "parent_prim_path": (
                    "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
                ),
                "optical_convention": "opencv",
                "frame_from_camera_matrix": [
                    1.0, 0.0, 0.0, 0.011,
                    0.0, 1.0, 0.0, -0.031,
                    0.0, 0.0, 1.0, -0.074,
                    0.0, 0.0, 0.0, 1.0,
                ],
                "intrinsics": intrinsics,
            },
            {"role": "overview", **{**world, "policy_input": False}},
        ],
    }


def _trajectory() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "task_evaluation_robot_placement_trajectory.v1",
        "source_plan_schema_version": "native_rigid_construction_phase_plan.v1",
        "source_plan_digest": "sha256:" + "3" * 64,
        "task_kind": "rigid_pick_place",
        "manipulation_strategy": "planar_push",
        "arrival_tolerance_m": 0.02,
        "arrival_orientation_tolerance_rad": 0.08,
        "maximum_steps_per_phase": 64,
        "phases": [
            {
                "phase_id": "precontact",
                "position_world_m": [2.79, -6.76, 0.82],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                "gripper_state": "open",
                "gate_ids": ["precontact_reachability"],
            },
            {
                "phase_id": "push",
                "position_world_m": [3.03, -6.76, 0.82],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                "gripper_state": "closed",
                "gate_ids": ["push_path"],
            },
        ],
        "model_may_modify_trajectory": False,
        "native_ik_and_collision_readback_required_for_every_phase": True,
        "trajectory_digest": "",
    }
    value["trajectory_digest"] = autostart.canonical_digest(
        value, digest_field="trajectory_digest"
    )
    return value


def test_world_cameras_are_derived_after_exact_inventory_member_selection() -> None:
    template = _camera_template()
    wrist = template["cameras"][1]
    pose = {
        "position_world_m": [3.5442285, -6.7605156, 0.752958],
        "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
    }
    result = autostart._placement_aware_camera_candidates(
        camera_template=template,
        accepted_pose=pose,
        selected_candidate_id="geometry_0003_0.753_0.450_36",
        trajectory=_trajectory(),
        source_commit=COMMIT,
    )

    by_role = {row["role"]: row for row in result["cameras"]}
    assert by_role["wrist"] == wrist
    assert by_role["external"]["frame_from_camera_matrix"][3] == pose[
        "position_world_m"
    ][0]
    assert by_role["external"]["frame_from_camera_matrix"][7] == pose[
        "position_world_m"
    ][1]
    assert by_role["external"]["frame_from_camera_matrix"][11] == pytest.approx(
        pose["position_world_m"][2] + 1.35
    )
    assert by_role["external"]["frame_from_camera_matrix"][3] != 100.0
    assert result["camera_configuration_qualified"] is False
    assert result["native_observability_readback_required"] is True
    assert result["document_digest"] == autostart.canonical_digest(
        result, digest_field="document_digest"
    )


def test_different_selected_pose_cannot_reuse_world_camera_bytes() -> None:
    common = {
        "camera_template": _camera_template(),
        "selected_candidate_id": "candidate-1",
        "trajectory": _trajectory(),
        "source_commit": COMMIT,
    }
    first = autostart._placement_aware_camera_candidates(
        accepted_pose={
            "position_world_m": [3.5, -6.7, 0.75],
            "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
        },
        **common,
    )
    second = autostart._placement_aware_camera_candidates(
        accepted_pose={
            "position_world_m": [3.1, -6.4, 0.75],
            "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
        },
        **common,
    )

    assert first["document_digest"] != second["document_digest"]
    assert first["cameras"][0] != second["cameras"][0]


def test_vertical_task_trajectory_still_gets_world_cameras_from_base_relation() -> None:
    trajectory = _trajectory()
    trajectory["phases"][0]["position_world_m"] = [3.0, -6.7, 0.8]
    trajectory["phases"][1]["position_world_m"] = [3.0, -6.7, 1.2]
    trajectory["trajectory_digest"] = autostart.canonical_digest(
        trajectory, digest_field="trajectory_digest"
    )

    result = autostart._placement_aware_camera_candidates(
        camera_template=_camera_template(),
        accepted_pose={
            "position_world_m": [3.5, -6.7, 0.75],
            "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
        },
        selected_candidate_id="vertical-task-candidate",
        trajectory=trajectory,
        source_commit=COMMIT,
    )

    assert {row["role"] for row in result["cameras"]} == {
        "external",
        "wrist",
        "overview",
    }


def test_camera_candidate_materialization_is_immutable(tmp_path: Path) -> None:
    template_path = _write(
        tmp_path / "camera-template.json",
        (json.dumps(_camera_template()) + "\n").encode(),
    )
    kwargs = {
        "root": tmp_path,
        "camera_template_path": template_path,
        "accepted_pose": {
            "position_world_m": [3.5, -6.7, 0.75],
            "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
        },
        "selected_candidate_id": "candidate-1",
        "trajectory": _trajectory(),
        "source_commit": COMMIT,
    }
    path = autostart._materialize_placement_aware_cameras(**kwargs)
    assert path.stat().st_mode & 0o777 == 0o440
    assert autostart._materialize_placement_aware_cameras(**kwargs) == path

    with pytest.raises(
        autostart.TaskEvaluationConfiguredControlsAutostartError,
        match="configured_controls_autostart_camera_candidate_conflict",
    ):
        autostart._materialize_placement_aware_cameras(
            **{
                **kwargs,
                "accepted_pose": {
                    "position_world_m": [3.2, -6.7, 0.75],
                    "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
                },
            }
        )


def test_autostart_result_is_bound_to_its_intent(tmp_path: Path) -> None:
    """The result is validated against the intent digest on reopen, so sharing one
    filename across intents fails closed forever instead of deriving fresh."""

    first = autostart._autostart_result_path(
        root=tmp_path, intent_digest="sha256:" + "a" * 64
    )
    second = autostart._autostart_result_path(
        root=tmp_path, intent_digest="sha256:" + "b" * 64
    )
    assert first != second
    assert first.parent == tmp_path == second.parent
    assert first.name.startswith(autostart.RESULT_SCHEMA_VERSION)
    assert first.name.endswith(".json")

    with pytest.raises(
        autostart.TaskEvaluationConfiguredControlsAutostartError,
        match="configured_controls_autostart_result_binding_invalid",
    ):
        autostart._autostart_result_path(root=tmp_path, intent_digest="not-a-digest")


def test_camera_candidates_are_scoped_to_the_source_commit(tmp_path: Path) -> None:
    """The camera document embeds source_commit, so a shared filename makes every
    redeploy collide with its predecessor and block the lane permanently."""

    template_path = _write(
        tmp_path / "camera-template.json",
        (json.dumps(_camera_template()) + "\n").encode(),
    )
    kwargs = {
        "root": tmp_path,
        "camera_template_path": template_path,
        "accepted_pose": {
            "position_world_m": [3.5, -6.7, 0.75],
            "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
        },
        "selected_candidate_id": "candidate-1",
        "trajectory": _trajectory(),
        "source_commit": COMMIT,
    }
    first = autostart._materialize_placement_aware_cameras(**kwargs)
    successor = "b" * 40
    second = autostart._materialize_placement_aware_cameras(
        **{**kwargs, "source_commit": successor}
    )

    assert first != second
    assert first.is_file() and second.is_file()
    assert COMMIT[:12] in first.name
    assert successor[:12] in second.name


def test_native_feedback_universe_expands_cpu_inventory_into_bounded_exact_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset = {
        "joint_positions_rad": [0.0, -0.62, 0.1, -1.36, 1.35, 1.75, -0.72]
    }

    def orientation_gate(*, gate, **_kwargs):
        return {
            **dict(gate),
            "status": "passed",
            "orientation_slew_feasibility": {
                "feasible": True,
                "task_aware_reset": reset,
            },
        }

    monkeypatch.setattr(
        autostart, "_reject_infeasible_orientation_slew", orientation_gate
    )
    monkeypatch.setattr(
        autostart,
        "validate_robot_placement_trajectory_position_ik",
        lambda **_kwargs: {
            "status": "passed",
            "trajectory_position_ik_gate_digest": "sha256:" + "e" * 64,
        },
    )
    inventory = {
        "candidates": [
            {
                "schema_version": "task_evaluation_robot_placement_geometry_gate.v1",
                "status": "passed",
                "blockers": [],
                "candidate_id": "candidate-42",
                "pose": {
                    "position_world_m": [3.54, -6.76, 0.753],
                    "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
                },
                "support_surface_id": "/Site/counter",
                "geometry_gate_digest": "sha256:" + "f" * 64,
            }
        ]
    }

    universe = autostart._native_feedback_candidate_universe(
        run_id="scene-839873-native-feedback",
        inventory=inventory,
        trajectory=_trajectory(),
        camera_template=_camera_template(),
        source_commit=COMMIT,
        maximum_candidates=8,
    )

    assert [row["candidate_id"] for row in universe["candidates"]] == [
        "candidate-42--direct--uniform_seed",
        "candidate-42--overhead--contact_ramp",
        "candidate-42--radial_standoff--push_contact_dense",
        "candidate-42--direct--release_retreat_dense",
    ]
    assert {
        row["interaction_trajectory_variant"]["interaction_branch_id"]
        for row in universe["candidates"]
    } == {
        "uniform_seed",
        "contact_ramp",
        "push_contact_dense",
        "release_retreat_dense",
    }
    assert all(
        row["interaction_trajectory_variant"][
            "preserves_authored_tcp_endpoints"
        ]
        is True
        for row in universe["candidates"]
    )
    assert all(
        row["reset_variant"]["robot_joint_reset_positions_rad"]["panda_joint4"]
        == pytest.approx(-1.36)
        for row in universe["candidates"]
    )
    assert all(len(row["camera_variant"]["cameras"]) == 3 for row in universe["candidates"])
    assert universe["inventory_digest"] == autostart.canonical_digest(
        universe, digest_field="inventory_digest"
    )


def test_autostart_plan_binds_placement_aware_not_prelaunch_world_cameras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent_path, intent = _intent(tmp_path)
    camera_template_path = Path(intent["paths"]["cameras_path"])
    camera_template_path.write_text(
        json.dumps(_camera_template()) + "\n", encoding="utf-8"
    )
    intent["artifact_inventory"]["cameras_path"] = autostart._artifact(
        camera_template_path
    )
    intent["intent_digest"] = autostart.canonical_digest(
        intent, digest_field="intent_digest"
    )
    intent_path.chmod(0o640)
    intent_path.write_text(json.dumps(intent) + "\n", encoding="utf-8")
    intent_path.chmod(0o440)

    launch_root = tmp_path / "launch-runs"
    source_launch_id = "scene-839873-launch"
    run_root = launch_root / source_launch_id
    run_root.mkdir(parents=True)
    revision_path = run_root / "configured-scene-revision.json"
    revision_path.write_text("{}\n", encoding="utf-8")
    revision = {
        "source_commit": COMMIT,
        "team_namespace": intent["team_namespace"],
        "scene_identity": {"id": intent["scene_id"]},
        "task_template": {
            "identity": {"id": intent["task_id"]},
            "definition": {"digest": "sha256:" + "1" * 64},
        },
        "registration": {
            "robot_mount_interface": {"digest": "sha256:" + "2" * 64},
            "workspace_clearance": {"digest": "sha256:" + "3" * 64},
        },
        "geometry": {
            "configured_collision": {
                "digest": "sha256:" + "4" * 64,
            }
        },
        "revision_digest": "sha256:" + "5" * 64,
    }
    terminal = {
        "run_id": "scene-839873-configured",
        "configured_scene_revision_path": str(revision_path),
    }
    receipt = {"source_commit": COMMIT}
    profile = {
        "task_evaluation_run": {
            "run_mode": "scene_configuration",
            "team_namespace": intent["team_namespace"],
            "scene_id": intent["scene_id"],
            "task_id": intent["task_id"],
        }
    }
    collision_path = _write(tmp_path / "collision.usda", b"#usda 1.0\n")
    accepted_pose = {
        "position_world_m": [3.5442285, -6.7605156, 0.752958],
        "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
    }
    candidate_inventory_digest = "sha256:" + "6" * 64
    cpu_placement_receipt = {
        "candidate_inventory_digest": candidate_inventory_digest,
        "receipt_digest": "sha256:" + "8" * 64,
    }
    agent_placement_receipt = {
        "accepted_candidate_id": "candidate-42",
        "accepted_pose": accepted_pose,
        "candidate_inventory_digest": candidate_inventory_digest,
        "receipt_digest": "sha256:" + "9" * 64,
        "rounds": [
            {
                "proposal_usage": {
                    "model": "gpt-5.6-sol",
                    "input_tokens": 2_000,
                    "cached_tokens": 0,
                    "cache_write_tokens": 1_200,
                    "uncached_input_tokens": 800,
                    "output_tokens": 20,
                    "reasoning_tokens": 5,
                    "cache_hit_ratio": 0.0,
                    "uncached_input_cost_usd": 0.0032,
                    "cache_write_cost_usd": 0.006,
                    "cached_read_cost_usd": 0.0,
                    "output_cost_usd": 0.0004,
                    "estimated_total_cost_usd": 0.0096,
                    "estimated_cost_without_caching_usd": 0.0084,
                    "estimated_savings_usd": -0.0012,
                    "cost_status": "model_pricing_estimate_not_official_billing",
                    "provider_response_id": "resp_fixture",
                    "provider_request_id": "req_fixture",
                    "usage_receipt_digest": "sha256:" + "b" * 64,
                    "breakpoint_digests": ["sha256:" + "c" * 64],
                    "cache_policy": {
                        "status": "enabled",
                        "model_family": "gpt-5.6-sol",
                        "family": "task_aware_robot_placement_proposal",
                        "contract_version": "robot-placement-proposal-v2",
                        "stable_prefix_digest": "sha256:" + "c" * 64,
                        "policy_digest": "sha256:" + "d" * 64,
                        "privacy_scope": "task_evaluation_rights_admitted",
                        "processing_region": "default",
                        "decision_reason": "expected_cached_cost_lower",
                        "cache_key": "blueprint:cache:v1:" + "e" * 40,
                        "economics": {"stable_prefix_tokens": 1_200},
                    },
                }
            }
        ],
    }
    inventory = {
        "candidate_inventory_digest": candidate_inventory_digest,
        "checkpoint_digest": "sha256:" + "a" * 64,
    }
    openai_evidence = {
        name: autostart._artifact(_write(tmp_path / "evidence" / f"{name}.json"))
        for name in (
            "reservation",
            "completion",
            "exclusive_lock",
            "exclusive_lock_release",
            "inference_reservations",
        )
    }
    captured: dict[str, object] = {}

    from blueprint_pipeline import (
        task_evaluation_configured_controls_progression_worker as progression_worker,
    )

    monkeypatch.setattr(
        progression_worker,
        "_validate_source",
        lambda _root: (terminal, receipt, object()),
    )
    monkeypatch.setattr(
        autostart,
        "_profile_intent",
        lambda _root, **_kwargs: (profile, intent_path),
    )
    monkeypatch.setattr(
        autostart,
        "validate_configured_scene_revision",
        lambda _value: revision,
    )
    monkeypatch.setattr(autostart, "_configured_collision", lambda **_kwargs: collision_path)
    monkeypatch.setattr(
        autostart,
        "placement_trajectory_from_native_plan",
        lambda _value: _trajectory(),
    )
    placement_checkpoint_calls = 0
    events: list[object] = []

    class FakeOfficialCostGate:
        def reserve(self):
            events.append("official-cost-reserved")

        def complete(self, **kwargs):
            events.append(("official-cost-completed", kwargs))

    def agent_placement_runner(**kwargs):
        assert kwargs["record_inference_reservations"] is True
        events.append("agent-called")
        return agent_placement_receipt

    def placement_checkpoint(**kwargs):
        nonlocal placement_checkpoint_calls
        placement_checkpoint_calls += 1
        runner_kwargs = kwargs["runner_kwargs"]
        if placement_checkpoint_calls == 1:
            checkpoint_binding_digest = kwargs[
                "expected_checkpoint_binding_digest"
            ]
            captured["cpu_checkpoint_binding_digest"] = (
                checkpoint_binding_digest
            )
            assert kwargs["checkpoint_file_name"] == (
                "cpu-placement-checkpoint.v2.json"
            )
            assert kwargs["root"] == (
                tmp_path
                / "progression"
                / source_launch_id
                / "cpu-robot-binding"
                / "cpu-placement-checkpoints"
                / checkpoint_binding_digest.removeprefix("sha256:")
            )
            assert runner_kwargs["deterministic_selection"] is True
            assert runner_kwargs["allow_live_invocation"] is False
            assert "candidate_inventory_checkpoint" not in runner_kwargs
            return cpu_placement_receipt, inventory, tmp_path / "cpu-placement"
        assert placement_checkpoint_calls == 2
        assert runner_kwargs["deterministic_selection"] is False
        assert runner_kwargs["allow_live_invocation"] is True
        assert runner_kwargs["candidate_inventory_checkpoint"] == inventory
        receipt = kwargs["placement_runner"](
            output_dir=tmp_path / "agent-placement",
            **runner_kwargs,
        )
        return receipt, inventory, tmp_path / "agent-placement"

    monkeypatch.setattr(
        autostart,
        "_placement_checkpoint",
        placement_checkpoint,
    )
    monkeypatch.setattr(
        autostart,
        "_validated_agent_openai_evidence",
        lambda **_kwargs: openai_evidence,
    )
    native_universe_path = _write(
        tmp_path / "native-construction-universe.json", b"{}\n"
    )
    native_universe = {
        "inventory_digest": "sha256:" + "d" * 64,
        "candidates": [{"candidate_id": "candidate-42--direct"}],
    }
    monkeypatch.setattr(
        autostart,
        "_materialize_native_feedback_candidate_universe",
        lambda **_kwargs: (native_universe_path, native_universe),
    )

    def readiness_materializer(**kwargs):
        Path(kwargs["output_path"]).write_text("{}\n", encoding="utf-8")
        return {"status": "materialized"}

    def plan_materializer(**kwargs):
        captured.update(kwargs)
        return {
            "plan_path": str(tmp_path / "plan.json"),
            "plan_digest": "sha256:" + "7" * 64,
        }

    legacy_result_path = (
        tmp_path
        / "progression"
        / source_launch_id
        / "cpu-robot-binding"
        / "task_evaluation_configured_controls_autostart.v2.json"
    )
    legacy_result_bytes = b'{"status":"stale-prior-binding"}\n'
    _write(legacy_result_path, legacy_result_bytes)
    result = autostart.materialize_configured_controls_autostart(
        source_launch_id=source_launch_id,
        launch_state_root=launch_root,
        progression_root=tmp_path / "progression",
        plan_root=tmp_path / "plans",
        placement_runner=lambda **_kwargs: {},
        agent_placement_runner=agent_placement_runner,
        readiness_materializer=readiness_materializer,
        plan_materializer=plan_materializer,
        environment={},
        openai_gate_builder=lambda **_kwargs: FakeOfficialCostGate(),
        openai_scope_lock=lambda **_kwargs: nullcontext({}),
        require_inference_usage_webapp_sync=False,
    )

    assert result["intent_digest"] == intent["intent_digest"]
    assert result["cpu_placement_checkpoint_binding_digest"] == (
        captured["cpu_checkpoint_binding_digest"]
    )
    assert legacy_result_path.read_bytes() == legacy_result_bytes
    # The result is bound to the intent that produced it, so a successor intent
    # derives its own destination instead of failing closed on this one forever.
    assert autostart._autostart_result_path(
        root=legacy_result_path.parent,
        intent_digest=str(intent["intent_digest"]),
    ).is_file()
    assert not (
        legacy_result_path.parent
        / "task_evaluation_configured_controls_autostart.v3.json"
    ).exists()
    with pytest.raises(
        autostart.TaskEvaluationConfiguredControlsAutostartError,
        match="configured_controls_autostart_result_invalid",
    ):
        autostart._validate_result(
            result,
            expected_intent_digest=str(intent["intent_digest"]),
            expected_scene_binding_digest=str(result["scene_binding_digest"]),
            expected_task_binding_digest="sha256:" + "f" * 64,
            expected_cpu_checkpoint_binding_digest=str(
                result["cpu_placement_checkpoint_binding_digest"]
            ),
        )

    selected_path = Path(captured["bindings"]["cameras_path"])
    assert selected_path != camera_template_path
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    assert selected["selected_candidate_id"] == "candidate-42"
    assert selected["accepted_pose"] == accepted_pose
    assert selected["world_camera_positions_depend_on_selected_base"] is True
    assert result["selected_candidate_id"] == "candidate-42"
    assert events == [
        "official-cost-reserved",
        "agent-called",
        (
            "official-cost-completed",
            {
                "provider_call_performed": True,
                "runtime_result_digest": agent_placement_receipt[
                    "receipt_digest"
                ],
                "runtime_exception_type": None,
            },
        ),
    ]
