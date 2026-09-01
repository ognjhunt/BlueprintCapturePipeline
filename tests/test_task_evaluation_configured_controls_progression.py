from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_configured_controls_progression import (
    TaskEvaluationConfiguredControlsProgressionError,
    build_authorized_webapp_launch_request,
    stage_configured_controls_activation,
    stage_configured_controls_episode_preparation,
    submit_authorized_progression_launch,
)
from tests.test_task_evaluation_configured_scene_revision import revision as revision_fixture


def _ref(index: int) -> dict[str, object]:
    return {
        "uri": f"s3://blueprint-production-inputs/progression-{index}.json",
        "digest": f"sha256:{index:064x}",
        "size_bytes": 1000 + index,
    }


def _configured() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    revision = revision_fixture()
    offering: dict[str, object] = {
        "schema_version": "task_evaluation_configured_scene_offering.v1",
        "status": "configured_controls_pending",
        "configuration_run_id": revision["configuration_run_id"],
        "team_namespace": revision["team_namespace"],
        "catalog_visibility": "team_only",
        "scene_identity": revision["scene_identity"],
        "task": {
            "identity": revision["task_template"]["identity"],
            "kind": "rigid_relocation",
            "strategy": "planar_push",
            "subject_identity": revision["replacement"]["identity"],
        },
        "presentation": revision["presentation"],
        "evaluation_preparation_binding": {
            "scene_mode": "reuse_configured_revision",
            "construction_mode": "reuse_configured_scene",
            "task_binding_mode": "reuse_configured_template",
            "configuration_source_commit": revision["source_commit"],
            "configured_scene_revision": _ref(40),
            "configured_scene_revision_digest": revision["revision_digest"],
            "configured_scene_bundle": revision["configured_scene_bundle"],
        },
        "proof_boundary": {
            "thumbnail_is_derived_appearance_evidence": True,
            "thumbnail_is_capture_or_physical_evidence": False,
            "configuration_is_policy_evaluation": False,
            "configuration_is_deployment_or_safety_approval": False,
        },
        "evaluation_admission": {
            "zero_action_required": True,
            "scripted_positive_required": True,
            "learned_policy_evaluation_admitted": False,
        },
        "offering_digest": "",
    }
    offering["offering_digest"] = canonical_digest(
        offering, digest_field="offering_digest"
    )
    publication: dict[str, object] = {
        "schema_version": "task_evaluation_scene_configuration_publication.v1",
        "status": "configured_scene_published",
        "configuration_run_id": revision["configuration_run_id"],
        "configured_scene_revision": {
            "role": "configured_scene_revision",
            "path": "/not-used-in-this-boundary/revision.json",
            "digest": _ref(40)["digest"],
            "size_bytes": _ref(40)["size_bytes"],
        },
        "configured_scene_revision_reference": _ref(40),
        "configured_scene_revision_digest": revision["revision_digest"],
        "configured_scene_bundle_reference": revision["configured_scene_bundle"],
        "task_thumbnail_reference": revision["presentation"]["task_thumbnail"],
        "task_thumbnail_selection": revision["presentation"]["selection"],
        "task_thumbnail_selection_receipt_reference": revision["presentation"][
            "selection_receipt"
        ],
        "configured_scene_offering": offering,
        "publication_receipt_digest": "sha256:" + "d" * 64,
        "full_byte_service_account_readback_passed": True,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "result_digest": "",
    }
    publication["result_digest"] = canonical_digest(
        publication, digest_field="result_digest"
    )
    finalization: dict[str, object] = {
        "schema_version": "task_evaluation_scene_construction_finalization.v1",
        "status": "completed",
        "queue_state": "completed",
        "finalization_performed": True,
        "result_digest": "",
    }
    finalization["result_digest"] = canonical_digest(
        finalization, digest_field="result_digest"
    )
    terminal: dict[str, object] = {
        "schema_version": "task_evaluation_scene_configuration_vast_result.v1",
        "status": "completed",
        "run_id": revision["configuration_run_id"],
        "source_commit": "a" * 40,
        "configuration_completed": True,
        "configured_scene_published": True,
        "configured_scene_revision_digest": revision["revision_digest"],
        "publication_result_digest": publication["result_digest"],
        "full_byte_service_account_readback_passed": True,
        "provider_mutations_performed": 1,
        "retry_cap": 0,
        "evaluation_episode_executed": False,
        "candidate_policy_queried": False,
        "continuing_spend_from_this_run": False,
        "scene_construction_queue_finalization": finalization,
        "blockers": [],
        "result_digest": "",
    }
    terminal["result_digest"] = canonical_digest(
        terminal, digest_field="result_digest"
    )
    return terminal, publication, revision


def _runtime() -> dict[str, object]:
    return {
        "runtime": {
            "identity": {"id": "native-arena", "version": "isaac-2026-1"},
            "oci_image": "registry.example/arena@sha256:" + "a" * 64,
            "entrypoint": ["/opt/blueprint/run-task-evaluation"],
            "health_protocol": _ref(50),
            "requirements": {
                "cpu_cores": 8,
                "memory_gib": 64,
                "gpu_count": 1,
                "disk_gib": 100,
            },
            "network": {"default": "deny", "allowlist": []},
            "secret_refs": [],
            "mounts": [
                {
                    "source": _ref(51),
                    "container_path": "/inputs",
                    "mode": "read_only",
                },
                {"container_path": "/outputs", "mode": "output"},
            ],
            "output_limit_bytes": 20_000_000_000,
        },
        "execution_adapter": {
            "kind": "native_task_arena",
            "version": "v1",
            "runtime_source_bundle": _ref(52),
        },
        "spend": {
            "maximum_hourly_rate_usd": 0.8,
            "hard_cap_usd": 2.25,
            "hard_ttl_seconds": 9000,
            "retry_cap": 0,
            "selected_provider": "vast",
            "provider_allowlist": ["vast"],
        },
    }


def _fake_materializer(**kwargs: object) -> dict[str, object]:
    root = Path(kwargs["output_root"])
    roles = {
        "robot_configuration",
        "robot_kinematics",
        "robot_joint_bounds",
        "robot_base_registration",
        "controller_configuration",
        "sensor_configuration",
    }
    files = {}
    for index, role in enumerate(sorted(roles), start=1):
        path = root / f"{role}.json"
        path.write_text(json.dumps({"schema_version": f"test.{role}.v1"}), encoding="utf-8")
        payload = path.read_bytes()
        import hashlib

        files[role] = {
            "path": str(path),
            "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    return {
        "status": "materialized_candidate_pending_native_construction_readback",
        "configured_scene_revision_digest": kwargs["configured_revision"][
            "revision_digest"
        ],
        "robot_identity": {
            "id": "franka-panda-robotiq-2f85",
            "version": "isaaclab-arena-droid-8b4a3a47",
        },
        "robot_base_qualified": False,
        "camera_configuration_qualified": False,
        "native_construction_readback_required": True,
        "candidate_policy_queried": False,
        "files": files,
    }


def _publisher(*, path: Path, object_name: str) -> dict[str, object]:
    import hashlib

    payload = path.read_bytes()
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    return {
        "uri": f"s3://blueprint-production-inputs/{object_name}",
        "digest": digest,
        "size_bytes": len(payload),
        "full_byte_service_account_readback_passed": True,
        "readback_digest": digest,
        "readback_size_bytes": len(payload),
    }


def _stage_preparation(**kwargs: object) -> dict[str, object]:
    request = kwargs["value"]
    return {
        "status": "queued_for_no_spend_preparation",
        "accepted": True,
        "preparation_id": request["preparation_id"],
        "request_digest": canonical_digest(request),
        "provider_mutation_performed_inside_http_request": False,
        "catalog_mutation_performed_inside_http_request": False,
        "paid_execution_requested": False,
        "receipt_digest": "sha256:" + "e" * 64,
    }


def _episode_progression(tmp_path: Path) -> dict[str, object]:
    terminal, publication, revision = _configured()
    return stage_configured_controls_episode_preparation(
        terminal_result=terminal,
        publication_result=publication,
        configured_revision=revision,
        expected_production_commit="b" * 40,
        robot_mount_interface_path=tmp_path / "mount.json",
        scene_camera_calibration_path=tmp_path / "calibration.json",
        base_pose_candidate={},
        cameras=[],
        runtime_binding=_runtime(),
        output_root=tmp_path / "readiness",
        publisher=_publisher,
        queue_root=tmp_path / "preparation-queue",
        submitted_by="configured-controls-progression",
        readiness_materializer=_fake_materializer,
        preparation_stager=_stage_preparation,
    )


def test_qualifying_configuration_queues_digest_bound_episode_preparation(tmp_path: Path) -> None:
    result = _episode_progression(tmp_path)

    assert result["status"] == "episode_preparation_queued"
    assert result["provider_mutation_performed"] is False
    assert result["native_construction_readback_required"] is True
    request = result["episode_preparation_request"]
    assert request["controller"]["kind"] == "deterministic_scripted"
    assert request["expected_production_commit"] == "b" * 40
    assert request["scene"]["configured_revision"] == _ref(40)
    assert request["spend"]["retry_cap"] == 0
    assert request["robot"]["base_registration"]["uri"].startswith("s3://")

    # A restarted worker reopens the sealed progression receipt and performs
    # neither a second publication nor a second queue mutation.
    replay = _episode_progression(tmp_path)
    assert replay == result


def test_diagnostic_or_unclosed_configuration_never_materializes_inputs(tmp_path: Path) -> None:
    terminal, publication, revision = _configured()
    terminal["continuing_spend_from_this_run"] = True
    terminal["result_digest"] = canonical_digest(terminal, digest_field="result_digest")
    called = False

    def materializer(**_: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(
        TaskEvaluationConfiguredControlsProgressionError,
        match="configured_controls_progression_qualifying_configuration_missing",
    ):
        stage_configured_controls_episode_preparation(
            terminal_result=terminal,
            publication_result=publication,
            configured_revision=revision,
            expected_production_commit="b" * 40,
            robot_mount_interface_path=tmp_path / "mount.json",
            scene_camera_calibration_path=tmp_path / "calibration.json",
            base_pose_candidate={},
            cameras=[],
            runtime_binding=_runtime(),
            output_root=tmp_path / "readiness",
            publisher=_publisher,
            queue_root=tmp_path / "queue",
            submitted_by="configured-controls-progression",
            readiness_materializer=materializer,
            preparation_stager=_stage_preparation,
        )
    assert called is False


def _preparation_result(progression: dict[str, object]) -> dict[str, object]:
    value = {
        "schema_version": "task_evaluation_launch_preparation_result.v1",
        "status": "queued_for_production_episode_compilation",
        "run_mode": "episode_evaluation",
        "configured_scene_revision_digest": progression[
            "configured_scene_revision_digest"
        ],
        "automatic_progression_required": True,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "result_digest": "",
    }
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def _authorization() -> dict[str, object]:
    return {
        "reference": "owner-authorized-controls-progression-20260828",
        "authorized_by": "blueprint-owner",
        "authorized_on": "2026-08-28T20:00:00+00:00",
        "standing_authorization_expires_at": "2026-08-28T22:30:00+00:00",
        "profile_revision": "r1",
    }


def test_construction_activation_is_queued_without_execution(tmp_path: Path) -> None:
    progression = _episode_progression(tmp_path)
    observed = {}

    def stage(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs["value"])
        return {
            "status": "queued_for_authority_gated_activation",
            "accepted": True,
            "activation_id": observed["activation_id"],
            "lane": observed["lane"],
            "provider_mutation_performed_inside_http_request": False,
            "paid_execution_requested": False,
            "receipt_digest": "sha256:" + "f" * 64,
        }

    result = stage_configured_controls_activation(
        progression=progression,
        preparation_result=_preparation_result(progression),
        release_window=_ref(60),
        lineage={
            "kind": "initial_project",
            "project_spend_reconciliation": _ref(61),
            "initial_provider_zero": _ref(62),
        },
        authorization=_authorization(),
        lane="native_task_arena_construction",
        queue_root=tmp_path / "activation-queue",
        submitted_by="configured-controls-progression",
        activation_stager=stage,
    )
    assert result["status"] == "construction_activation_queued"
    assert observed["lane"] == "native_task_arena_construction"
    assert result["activation_executed_provider"] is False


def _controls_lineage(tmp_path: Path, *, qualified: bool = True) -> tuple[dict[str, object], dict[str, Path]]:
    import hashlib

    tmp_path.mkdir(parents=True)

    values = {
        "prior_authority": {"schema_version": "test.authority.v1"},
        "prior_result": {"schema_version": "test.provider_result.v1"},
        "prior_launch_receipt": {"schema_version": "task_evaluation_launch_receipt.v1"},
        "prior_webapp_sync": {"schema_version": "task_evaluation_launch_webapp_sync_result.v1"},
        "prior_provider_zero": {"schema_version": "adp_paid_provider_zero.v1"},
        "prior_spend_reconciliation": {"schema_version": "test.spend.v1"},
        "construction_result": {
            "schema_version": "native_task_arena_construction_result.v1",
            "status": "completed",
            "construction_gate_qualified": qualified,
            "blockers": [] if qualified else ["reachability_failed"],
            "candidate_policy_queried": False,
            "result_digest": "",
        },
    }
    values["construction_result"]["result_digest"] = canonical_digest(
        values["construction_result"], digest_field="result_digest"
    )
    lineage: dict[str, object] = {"kind": "predecessor"}
    paths = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        payload = path.read_bytes()
        lineage[name] = {
            "uri": f"s3://blueprint-production-inputs/{name}.json",
            "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        paths[name] = path
    return lineage, paths


def test_controls_activation_requires_exact_qualified_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progression = _episode_progression(tmp_path)
    lineage, paths = _controls_lineage(tmp_path / "qualified", qualified=True)
    # The controls lane enum is supplied by prerequisite PR #1328. This focused
    # test isolates this module's additional predecessor check until that PR is
    # merged and this branch is rebased.
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_configured_controls_progression.validate_launch_activation_request",
        lambda value: value,
    )

    def stage(**kwargs: object) -> dict[str, object]:
        request = kwargs["value"]
        return {
            "status": "queued_for_authority_gated_activation",
            "accepted": True,
            "activation_id": request["activation_id"],
            "lane": request["lane"],
            "provider_mutation_performed_inside_http_request": False,
            "paid_execution_requested": False,
            "receipt_digest": "sha256:" + "f" * 64,
        }

    result = stage_configured_controls_activation(
        progression=progression,
        preparation_result=_preparation_result(progression),
        release_window=_ref(60),
        lineage=lineage,
        authorization=_authorization(),
        lane="native_task_arena_controls",
        queue_root=tmp_path / "activation-queue",
        submitted_by="configured-controls-progression",
        lineage_artifact_paths=paths,
        activation_stager=stage,
    )
    assert result["status"] == "controls_activation_queued"

    bad_lineage, bad_paths = _controls_lineage(tmp_path / "unqualified", qualified=False)
    with pytest.raises(
        TaskEvaluationConfiguredControlsProgressionError,
        match="configured_controls_progression_construction_not_qualified",
    ):
        stage_configured_controls_activation(
            progression=progression,
            preparation_result=_preparation_result(progression),
            release_window=_ref(60),
            lineage=bad_lineage,
            authorization=_authorization(),
            lane="native_task_arena_controls",
            queue_root=tmp_path / "activation-queue",
            submitted_by="configured-controls-progression",
            lineage_artifact_paths=bad_paths,
            activation_stager=stage,
        )


def _activation_result(state: dict[str, object], profile: dict[str, object]) -> dict[str, object]:
    value = {
        "schema_version": "task_evaluation_launch_activation_result.v1",
        "status": "profile_authority_materialized_no_execution",
        "activation_id": state["activation_request"]["activation_id"],
        "lane": state["lane"],
        "source_commit": state["expected_production_commit"],
        "profile_id": profile["profile_id"],
        "profile_digest": profile["profile_digest"],
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "blockers": [],
        "result_digest": "",
    }
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def _profile(state: dict[str, object]) -> dict[str, object]:
    value = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": "scene-839873-franka-construction-r1",
        "source_commit": state["expected_production_commit"],
        "allocator": {"max_spend_usd": 2.25},
        "profile_digest": "",
    }
    value["profile_digest"] = canonical_digest(value, digest_field="profile_digest")
    return value


def test_activation_only_submits_paid_launch_through_webapp(tmp_path: Path) -> None:
    progression = _episode_progression(tmp_path)

    def stage(**kwargs: object) -> dict[str, object]:
        request = kwargs["value"]
        return {
            "status": "queued_for_authority_gated_activation",
            "accepted": True,
            "activation_id": request["activation_id"],
            "lane": request["lane"],
            "provider_mutation_performed_inside_http_request": False,
            "paid_execution_requested": False,
            "receipt_digest": "sha256:" + "f" * 64,
        }

    state = stage_configured_controls_activation(
        progression=progression,
        preparation_result=_preparation_result(progression),
        release_window=_ref(60),
        lineage={
            "kind": "initial_project",
            "project_spend_reconciliation": _ref(61),
            "initial_provider_zero": _ref(62),
        },
        authorization=_authorization(),
        lane="native_task_arena_construction",
        queue_root=tmp_path / "activation-queue",
        submitted_by="configured-controls-progression",
        activation_stager=stage,
    )
    profile = _profile(state)
    activation = _activation_result(state, profile)
    authority = {
        "rights_scope": "configured-scene native controls qualification",
        "rights_evidence": _ref(70),
        "max_spend_usd": 2.25,
        "expires_at": "2026-08-28T22:30:00.000Z",
    }
    request = build_authorized_webapp_launch_request(
        activation_progression=state,
        activation_result=activation,
        profile=profile,
        launch_authority=authority,
    )
    assert request["progression"]["provider_mutation_performed_before_webapp_submission"] is False

    seen = {}

    def submit(outbound: dict[str, object]) -> dict[str, object]:
        seen.update(outbound)
        return {
            "status": "submitted",
            "launch_id": outbound["launch_id"],
            "provider_mutation_performed_inside_web_request": False,
        }

    receipt = submit_authorized_progression_launch(
        activation_progression=state,
        activation_result=activation,
        profile=profile,
        launch_authority=authority,
        submitter=submit,
    )
    assert receipt["status"] == "construction_launch_queued"
    assert receipt["submitted_through_webapp"] is True
    assert "progression" not in seen


def test_long_activation_uses_stable_bounded_webapp_launch_id(
    tmp_path: Path,
) -> None:
    progression = _episode_progression(tmp_path)
    state = {
        "schema_version": "task_evaluation_configured_controls_progression.v1",
        "status": "construction_activation_queued",
        "configured_scene_revision_digest": progression[
            "configured_scene_revision_digest"
        ],
        "expected_production_commit": "b" * 40,
        "lane": "native_task_arena_construction",
        "activation_request": {
            "activation_id": "corrective-scene-" + "x" * 220,
        },
        "progression_digest": "",
    }
    state["progression_digest"] = canonical_digest(
        state, digest_field="progression_digest"
    )
    profile = _profile(state)
    activation = _activation_result(state, profile)
    authority = {
        "rights_scope": "configured-scene native controls qualification",
        "rights_evidence": _ref(70),
        "max_spend_usd": 2.25,
        "expires_at": "2026-08-28T22:30:00.000Z",
    }

    first = build_authorized_webapp_launch_request(
        activation_progression=state,
        activation_result=activation,
        profile=profile,
        launch_authority=authority,
    )
    second = build_authorized_webapp_launch_request(
        activation_progression=state,
        activation_result=activation,
        profile=profile,
        launch_authority=authority,
    )

    assert first["launch_id"] == second["launch_id"]
    assert first["run_id"] == first["launch_id"]
    assert first["launch_id"].endswith("-launch")
    assert len(first["launch_id"]) <= 192
