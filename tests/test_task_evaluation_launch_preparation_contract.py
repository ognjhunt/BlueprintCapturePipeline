from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.task_evaluation_launch_preparation_contract import (
    SCHEMA_VERSION,
    TaskEvaluationLaunchPreparationContractError,
    launch_preparation_request_digest,
    validate_launch_preparation_request,
)
from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
    MAX_ATTEMPT_SPEND_USD,
    MAX_EXTERNAL_SERVICE_SPEND_USD,
    MAX_PROVIDER_COMPUTE_SPEND_USD,
    MIN_ARTIFIXER_SEMANTIC_TEACHER_SPEND_USD,
    MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD,
    MIN_CONTENT_AGENTS_SPEND_USD,
)


DIGESTS = [f"sha256:{index:064x}" for index in range(1, 48)]


def ref(index: int) -> dict[str, object]:
    return {
        "uri": f"s3://blueprint-production-inputs/object-{index}.json",
        "digest": DIGESTS[index],
        "size_bytes": 1000 + index,
    }


def configuration_spend() -> dict[str, object]:
    return {
        "hard_cap_usd": MAX_ATTEMPT_SPEND_USD,
        "hard_ttl_seconds": 27_000,
        "provider_compute_spend_cap_usd": MAX_PROVIDER_COMPUTE_SPEND_USD,
        "external_service_caps": {
            "openai": {
                "maximum_cost_usd": MAX_EXTERNAL_SERVICE_SPEND_USD,
                "maximum_requests": 32,
                "stage_max_cost_usd": {
                    "artifixer_semantic_teacher": (
                        MIN_ARTIFIXER_SEMANTIC_TEACHER_SPEND_USD
                    ),
                    "artifixer_visual_review": (
                        MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD
                    ),
                    "content_agents": MIN_CONTENT_AGENTS_SPEND_USD,
                },
            }
        },
    }


def request() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_mode": "episode_evaluation",
        "expected_production_commit": "a" * 40,
        "preparation_id": "team-a-scene-841007-zero-v1",
        "team_namespace": "team-a",
        "run_id": "run-scene-841007-zero-v1",
        "scene": {
            "mode": "reuse_configured_revision",
            "identity": {"id": "interiorgs-841007", "version": "v1"},
            "configured_revision": ref(31),
        },
        "construction": {
            "mode": "reuse_configured_scene",
        },
        "robot": {
            "identity": {"id": "franka-panda", "version": "isaac-2026-1"},
            "configuration": ref(11),
            "kinematics": ref(12),
            "joint_bounds": ref(13),
            "base_registration": ref(32),
            "controller_configuration": ref(14),
        },
        "controller": {
            "identity": {"id": "zero-action", "version": "v1"},
            "kind": "zero_action",
            "configuration": ref(15),
        },
        "task": {
            "identity": {"id": "planar-mug-push", "version": "v1"},
            "binding_mode": "reuse_configured_template",
            "kind": "rigid_relocation",
            "strategy": "planar_push",
            "configured_scene_revision_digest": DIGESTS[31],
            "subject": {
                "mode": "configured_scene_object",
                "identity": {"id": "scene-mug", "version": "v1"},
                "physics_authority": "configured_scene_revision",
            },
        },
        "sensors": {"configuration": ref(19)},
        "runtime": {
            "identity": {"id": "native-arena", "version": "v1"},
            "oci_image": "registry.example/arena@sha256:" + "a" * 64,
            "entrypoint": ["/opt/blueprint/run-task-evaluation"],
            "health_protocol": ref(20),
            "requirements": {
                "cpu_cores": 8,
                "memory_gib": 32,
                "gpu_count": 1,
                "disk_gib": 64,
            },
            "network": {"default": "deny", "allowlist": []},
            "secret_refs": [],
            "mounts": [
                {"source": ref(21), "container_path": "/inputs", "mode": "read_only"},
                {"container_path": "/outputs", "mode": "output"},
            ],
            "output_limit_bytes": 20_000_000_000,
        },
        "execution_adapter": {
            "kind": "native_task_arena",
            "version": "v1",
            "runtime_source_bundle": ref(23),
        },
        "publication": {
            "input_namespace": "team-a-scene-841007-v1",
            "service_account_readback_required": True,
        },
        "spend": {
            "maximum_hourly_rate_usd": 0.8,
            "hard_cap_usd": 0.75,
            "hard_ttl_seconds": 3300,
            "retry_cap": 0,
            "selected_provider": "vast",
            "provider_allowlist": ["vast"],
        },
    }


def test_accepts_scene_neutral_customer_contract_and_has_stable_digest() -> None:
    value = request()
    assert validate_launch_preparation_request(value) == value

    assert launch_preparation_request_digest(value) == launch_preparation_request_digest(
        copy.deepcopy(value)
    )


def test_accepts_production_construction_after_authenticated_submission() -> None:
    value = request()
    value["run_mode"] = "scene_configuration"
    value.pop("robot")
    value.pop("controller")
    value["scene"] = {
        "mode": "configure_source_scene",
        "identity": {"id": "interiorgs-841007", "version": "v1"},
        "source_manifest": ref(1),
        "appearance": {
            "kind": "interiorgs",
            "representation": ref(2),
            "renderer_qualification": ref(3),
        },
        "geometry": {
            "kind": "sage_derived",
            "collision": ref(4),
            "validation": ref(5),
        },
        "registration": {
            "metric_registration": ref(6),
            "support_plane": ref(7),
            "robot_mount_interface": ref(8),
            "workspace_clearance": ref(33),
            "camera_calibration": ref(9),
        },
        "rights": {
            "admission": ref(10),
            "evidence": [
                {"role": "publisher_terms", "artifact": ref(24)},
                {"role": "human_authority_record", "artifact": ref(25)},
            ],
            "source_bytes_redistributable": False,
            "provider_disclosure_scope": "derived_only",
        },
    }
    value["construction"] = {
        "mode": "production_recipe",
        "recipe": ref(29),
        "output_identity": {
            "id": "scene-source-object-native-packet",
            "version": "v1",
        },
    }
    value["task"]["subject"] = {
        "mode": "construct_from_scene_object",
        "identity": {"id": "source-object-replacement", "version": "v1"},
        "representation_kind": "simready_usd",
        "source_object": ref(30),
        "rights_admission": ref(10),
        "provider_disclosure_allowed": True,
    }
    value["task"].update(
        binding_mode="define_configuration_template",
        definition=ref(16),
        success_criteria=ref(17),
        execution=ref(18),
    )
    value["task"].pop("configured_scene_revision_digest")
    value["execution_adapter"]["kind"] = "scene_configuration_pipeline"
    value["spend"].update(configuration_spend())
    assert validate_launch_preparation_request(value) == value


def test_configuration_run_cannot_claim_an_episode_or_bind_a_controller() -> None:
    value = test_configuration_request()
    value["controller"] = request()["controller"]
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_request_invalid",
    ):
        validate_launch_preparation_request(value)


def test_configuration_forbids_ungraded_review_pause_override() -> None:
    value = test_configuration_request()
    value["appearance_review_override"] = {
        "mode": "paused_ungraded",
        "scope": "artifixer_appearance_only",
        "ungraded_publication_acknowledged": True,
        "review_provider_call_permitted": False,
        "warning_label": "Visual review paused - appearance ungraded",
    }
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_request_invalid",
    ):
        validate_launch_preparation_request(value)

    episode = request()
    episode["appearance_review_override"] = value["appearance_review_override"]
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_request_invalid",
    ):
        validate_launch_preparation_request(episode)


def test_configuration_external_service_caps_fit_total_authority() -> None:
    value = test_configuration_request()
    value["spend"]["hard_cap_usd"] = 7.0
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_scene_configuration_external_spend_invalid",
    ):
        validate_launch_preparation_request(value)


def test_pick_and_place_requires_a_distinct_qualified_destination_asset() -> None:
    value = request()
    value["task"]["strategy"] = "pick_and_place"
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_request_invalid",
    ):
        validate_launch_preparation_request(value)

    reference = {
        "uri": "s3://blueprint-production-inputs/task/destination.json",
        "digest": "sha256:" + "d" * 64,
        "size_bytes": 123,
    }
    value["task"]["destination"] = {
        "schema_version": "task_evaluation_rigid_destination_asset.v1",
        "identity": {"id": "document-tray", "version": "v1"},
        "relation": "inside",
        "visible_label": "blue document tray",
        "asset": reference,
        "rights_admission": reference,
        "static_qualification": reference,
        "native_import_qualification": reference,
        "geometry": reference,
        "placement_qualification": reference,
        "pose_world": {
            "position_world_m": [3.25, -6.76, 0.82],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "provider_disclosure_allowed": True,
    }
    assert validate_launch_preparation_request(value) == value

    qualification = copy.deepcopy(value)
    qualification["run_mode"] = "destination_qualification"
    qualification["task"]["destination"].pop("placement_qualification")
    qualification["task"]["destination"]["native_probe"] = {
        "schema_version": (
            "task_evaluation_rigid_destination_native_probe_configuration.v1"
        ),
        "placement_support_scene_prim_paths": ["/Root/Support"],
        "qualification_limits": {
            "maximum_penetration_m": 0.001,
            "minimum_support_contact_force_n": 0.01,
            "maximum_forbidden_contact_force_n": 0.1,
            "settle_translation_tolerance_m": 0.002,
            "settle_rotation_tolerance_rad": 0.01,
            "reset_translation_tolerance_m": 0.002,
            "reset_rotation_tolerance_rad": 0.01,
            "minimum_camera_pixels": {
                "external": 100,
                "wrist": 100,
                "overview": 100,
            },
        },
        "settle_sample_count": 3,
        "settle_steps_per_sample": 60,
    }
    assert validate_launch_preparation_request(qualification) == qualification

    value["task"]["destination"]["identity"] = value["task"]["subject"][
        "identity"
    ]
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_pick_place_destination_invalid",
    ):
        validate_launch_preparation_request(value)


def test_configuration_external_service_caps_keep_stage_and_total_bounds() -> None:
    value = test_configuration_request()
    value["spend"]["external_service_caps"]["openai"][
        "stage_max_cost_usd"
    ]["artifixer_visual_review"] = MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD - 0.01
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_scene_configuration_external_spend_invalid",
    ):
        validate_launch_preparation_request(value)

    value = test_configuration_request()
    value["spend"]["external_service_caps"]["openai"][
        "maximum_cost_usd"
    ] = MAX_EXTERNAL_SERVICE_SPEND_USD + 0.01
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match=(
            "launch_preparation_request_invalid:"
            "spend.external_service_caps.openai.maximum_cost_usd"
        ),
    ):
        validate_launch_preparation_request(value)


def test_configuration_requires_canonical_parent_runtime_authority() -> None:
    value = test_configuration_request()
    value["spend"]["hard_ttl_seconds"] = 9_000
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_scene_configuration_parent_runtime_budget_invalid",
    ):
        validate_launch_preparation_request(value)

    value = test_configuration_request()
    value["spend"]["provider_compute_spend_cap_usd"] = 5.59
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_scene_configuration_external_spend_invalid",
    ):
        validate_launch_preparation_request(value)


def test_episode_mode_preserves_prior_spend_and_ttl_ceilings() -> None:
    value = request()
    value["spend"]["hard_cap_usd"] = 5.01
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_episode_spend_invalid",
    ):
        validate_launch_preparation_request(value)

    value = request()
    value["spend"]["hard_ttl_seconds"] = 9_001
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_episode_spend_invalid",
    ):
        validate_launch_preparation_request(value)


def test_episode_evaluation_requires_configured_scene_robot_and_controller() -> None:
    value = request()
    value["scene"].pop("configured_revision")
    value.pop("robot")
    value.pop("controller")
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_request_invalid",
    ):
        validate_launch_preparation_request(value)


def test_construction_mode_cannot_hide_a_missing_or_prebuilt_subject() -> None:
    value = test_configuration_request()
    value["task"]["subject"] = request()["task"]["subject"]
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_request_invalid:task.subject.mode",
    ):
        validate_launch_preparation_request(value)


def test_selected_provider_must_be_allowed_and_have_an_admitted_adapter() -> None:
    value = request()
    value["spend"]["provider_allowlist"] = ["runpod"]
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_selected_provider_not_allowed",
    ):
        validate_launch_preparation_request(value)


def test_task_strategy_and_subject_are_explicit_and_digest_bound() -> None:
    value = request()
    original_digest = launch_preparation_request_digest(value)
    value["task"]["configured_scene_revision_digest"] = "sha256:" + "f" * 64
    assert launch_preparation_request_digest(value) != original_digest

    value = request()
    value["task"]["strategy"] = "articulated_open_close"
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_request_invalid:task.strategy",
    ):
        validate_launch_preparation_request(value)

    value = test_configuration_request()
    value["task"]["subject"]["provider_disclosure_allowed"] = False
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_request_invalid:task.subject",
    ):
        validate_launch_preparation_request(value)

    value = request()
    value["spend"]["selected_provider"] = "runpod"
    value["spend"]["provider_allowlist"] = ["runpod"]
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_execution_adapter_provider_unavailable",
    ):
        validate_launch_preparation_request(value)


@pytest.mark.parametrize(
    ("mutate", "blocker"),
    [
        (
            lambda value: value["scene"]["identity"].update(id="different-scene"),
            None,
        ),
        (
            lambda value: value["runtime"].update(oci_image="registry.example/arena:latest"),
            "launch_preparation_request_invalid:runtime.oci_image",
        ),
        (
            lambda value: value["runtime"]["network"].update(default="allow"),
            "launch_preparation_request_invalid:runtime.network.default",
        ),
        (
            lambda value: value["spend"].update(retry_cap=1),
            "launch_preparation_request_invalid:spend.retry_cap",
        ),
        (
            lambda value: value["execution_adapter"].update(kind="unknown_adapter"),
            "launch_preparation_execution_adapter_unavailable",
        ),
    ],
)
def test_request_contract_is_scene_neutral_and_fail_closed(mutate, blocker) -> None:
    value = request()
    original_digest = launch_preparation_request_digest(value)
    mutate(value)
    if blocker is None:
        assert launch_preparation_request_digest(value) != original_digest
        return
    with pytest.raises(TaskEvaluationLaunchPreparationContractError, match=blocker):
        validate_launch_preparation_request(value)


def test_rejects_rights_disclosure_conflict_and_host_paths() -> None:
    value = test_configuration_request()
    value["scene"]["rights"]["provider_disclosure_scope"] = "source_and_derived"
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_scene_disclosure_conflicts_with_rights",
    ):
        validate_launch_preparation_request(value)


def test_requires_publisher_terms_and_human_rights_authority_bytes() -> None:
    value = test_configuration_request()
    value["scene"]["rights"]["evidence"] = [
        {"role": "publisher_readme", "artifact": ref(24)},
        {"role": "upstream_license", "artifact": ref(25)},
    ]
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_request_invalid:scene",
    ):
        validate_launch_preparation_request(value)

    value = test_configuration_request()
    value["scene"]["source_manifest"]["uri"] = "/var/lib/blueprint/scene.json"
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_request_invalid:scene",
    ):
        validate_launch_preparation_request(value)


def test_configuration_request() -> dict[str, object]:
    value = request()
    value["run_mode"] = "scene_configuration"
    value.pop("robot")
    value.pop("controller")
    value["scene"] = {
        "mode": "configure_source_scene",
        "identity": {"id": "interiorgs-841007", "version": "v1"},
        "source_manifest": ref(1),
        "appearance": {
            "kind": "interiorgs",
            "representation": ref(2),
            "renderer_qualification": ref(3),
        },
        "geometry": {
            "kind": "sage_derived",
            "collision": ref(4),
            "validation": ref(5),
        },
        "registration": {
            "metric_registration": ref(6),
            "support_plane": ref(7),
            "robot_mount_interface": ref(8),
            "workspace_clearance": ref(33),
            "camera_calibration": ref(9),
        },
        "rights": {
            "admission": ref(10),
            "evidence": [
                {"role": "publisher_terms", "artifact": ref(24)},
                {"role": "human_authority_record", "artifact": ref(25)},
            ],
            "source_bytes_redistributable": False,
            "provider_disclosure_scope": "derived_only",
        },
    }
    value["construction"] = {
        "mode": "production_recipe",
        "recipe": ref(29),
        "output_identity": {"id": "configured-scene", "version": "v1"},
    }
    value["task"] = {
        "identity": {"id": "planar-mug-push", "version": "v1"},
        "binding_mode": "define_configuration_template",
        "kind": "rigid_relocation",
        "strategy": "planar_push",
        "subject": {
            "mode": "construct_from_scene_object",
            "identity": {"id": "scene-mug", "version": "v1"},
            "representation_kind": "simready_usd",
            "source_object": ref(30),
            "rights_admission": ref(10),
            "provider_disclosure_allowed": True,
        },
        "definition": ref(16),
        "success_criteria": ref(17),
        "execution": ref(18),
    }
    value["execution_adapter"]["kind"] = "scene_configuration_pipeline"
    value["spend"].update(configuration_spend())
    return value


def test_requires_one_bounded_output_mount_and_gpu() -> None:
    value = request()
    value["runtime"]["mounts"][1]["mode"] = "read_only"
    value["runtime"]["mounts"][1]["source"] = ref(22)
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_output_mount_count_invalid",
    ):
        validate_launch_preparation_request(value)

    value = request()
    value["runtime"]["requirements"]["gpu_count"] = 0
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_gpu_requirement_missing",
    ):
        validate_launch_preparation_request(value)


def _native_probe() -> dict[str, object]:
    return {
        "schema_version": (
            "task_evaluation_rigid_destination_native_probe_configuration.v1"
        ),
        "placement_support_scene_prim_paths": ["/Root/Support"],
        "qualification_limits": {
            "maximum_penetration_m": 0.001,
            "minimum_support_contact_force_n": 0.01,
            "maximum_forbidden_contact_force_n": 0.1,
            "settle_translation_tolerance_m": 0.002,
            "settle_rotation_tolerance_rad": 0.01,
            "reset_translation_tolerance_m": 0.002,
            "reset_rotation_tolerance_rad": 0.01,
            "minimum_camera_pixels": {"external": 100, "wrist": 100, "overview": 100},
        },
        "settle_sample_count": 3,
        "settle_steps_per_sample": 60,
    }


def _pending_destination() -> dict[str, object]:
    reference = {
        "uri": "s3://blueprint-production-inputs/task/destination.json",
        "digest": "sha256:" + "d" * 64,
        "size_bytes": 123,
    }
    return {
        "schema_version": "task_evaluation_rigid_destination_asset.v1",
        "identity": {"id": "document-tray", "version": "v1"},
        "relation": "inside",
        "visible_label": "blue document tray",
        "asset": reference,
        "rights_admission": reference,
        "static_qualification": reference,
        "native_probe": _native_probe(),
        "pose_world": {
            "position_world_m": [3.25, -6.76, 0.82],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "provider_disclosure_allowed": True,
    }


def test_scene_configuration_destination_defers_native_import_and_geometry_to_the_run() -> None:
    value = test_configuration_request()
    value["task"]["strategy"] = "pick_and_place"
    value["task"]["destination"] = _pending_destination()
    assert validate_launch_preparation_request(value) == value

    for field in ("native_import_qualification", "geometry"):
        prequalified = copy.deepcopy(value)
        prequalified["task"]["destination"][field] = ref(40)
        with pytest.raises(
            TaskEvaluationLaunchPreparationContractError,
            match="launch_preparation_scene_configuration_destination_prequalified_reference_forbidden",
        ):
            validate_launch_preparation_request(prequalified)

    unprobed = copy.deepcopy(value)
    unprobed["task"]["destination"].pop("native_probe")
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_request_invalid",
    ):
        validate_launch_preparation_request(unprobed)


def test_destination_qualification_still_requires_run_produced_qualifications() -> None:
    value = request()
    value["run_mode"] = "destination_qualification"
    value["task"]["strategy"] = "pick_and_place"
    value["task"]["destination"] = _pending_destination()
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_destination_qualification_reference_missing",
    ):
        validate_launch_preparation_request(value)
    value["task"]["destination"]["native_import_qualification"] = ref(40)
    value["task"]["destination"]["geometry"] = ref(41)
    assert validate_launch_preparation_request(value) == value
