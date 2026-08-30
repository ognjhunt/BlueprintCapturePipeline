from __future__ import annotations

import json
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
    )
    return destination, value


def test_intent_binds_every_fixed_cpu_and_paid_boundary_input(tmp_path: Path) -> None:
    path, value = _intent(tmp_path)

    assert path.stat().st_mode & 0o777 == 0o440
    assert value["paid_execution_requested"] is False
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
            "status": "cpu_binding_accepted_plan_materialized",
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
        "cpu_binding_accepted_plan_materialized",
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
