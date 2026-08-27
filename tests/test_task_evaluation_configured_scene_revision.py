from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_configured_scene_revision import (
    SCHEMA_VERSION,
    TaskEvaluationConfiguredSceneRevisionError,
    validate_configured_scene_revision,
)
from blueprint_pipeline.task_evaluation_scene_configuration_disclosure import (
    PROVIDER_GPU,
    SCHEMA_VERSION as DISCLOSURE_SCHEMA_VERSION,
)


def ref(index: int) -> dict[str, object]:
    return {
        "uri": f"s3://blueprint-production-inputs/configured-scene/object-{index}.json",
        "digest": f"sha256:{index:064x}",
        "size_bytes": 1000 + index,
    }


def revision() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "configured",
        "configuration_run_id": "scene-839873-configuration-v1",
        "team_namespace": "blueprint-adp",
        "scene_identity": {"id": "interiorgs-839873", "version": "mug-v1"},
        "source_commit": "a" * 40,
        "source": {
            "manifest": ref(1),
            "rights_admission": ref(2),
            "rights_evidence": [
                {"role": "publisher_terms", "artifact": ref(25)},
                {"role": "human_authority_record", "artifact": ref(26)},
            ],
            "raw_source_sent_to_external_provider": False,
        },
        "appearance": {
            "observed_source": ref(3),
            "object_removal_result": ref(4),
            "configured_representation": ref(23),
            "appearance_truth_source": (
                "interiorgs_observed_plus_labeled_generated_edit"
            ),
        },
        "geometry": {
            "candidate_collision_source": ref(5),
            "object_excision_result": ref(6),
            "configured_collision": ref(24),
            "validation": ref(7),
            "observed_source_truth_claimed": False,
        },
        "replacement": {
            "identity": {"id": "scene-839873-mug-replacement", "version": "v1"},
            "source_object": ref(8),
            "asset": ref(9),
            "static_qualification": ref(10),
            "native_import_qualification": ref(11),
            "physics_authority": "qualified_replacement_asset",
        },
        "registration": {
            "metric": ref(12),
            "support_plane": ref(13),
            "robot_mount_interface": ref(14),
            "camera_calibration": ref(15),
            "workspace_clearance": ref(16),
        },
        "configured_scene_bundle": ref(17),
        "task_template": {
            "identity": {"id": "scene-839873-mug-planar-push", "version": "v1"},
            "definition": ref(18),
            "success_criteria": ref(19),
            "execution": ref(20),
        },
        "robot_team_interface": {
            "scene_construction_repeated_per_evaluation": False,
            "configuration_run_executed_episode": False,
            "configuration_run_purpose": (
                "build_and_publish_reusable_robot_neutral_scene"
            ),
            "episode_run_purpose": (
                "evaluate_one_robot_or_policy_against_configured_scene"
            ),
            "episode_packet_compiled_by_production": True,
            "team_supplied_components": [
                "robot_configuration",
                "kinematics_and_joint_bounds",
                "robot_to_scene_registration",
                "controller_or_policy",
                "camera_and_sensor_configuration",
                "task_binding",
                "episode_runtime",
            ],
            "configured_scene_components": [
                "appearance",
                "collision_geometry",
                "replacement_assets",
                "metric_registration",
                "support_plane",
                "robot_mount_interface",
                "workspace_clearance",
                "scene_camera_calibration",
                "rights_and_provenance",
                "task_templates",
                "configured_scene_bundle",
            ],
            "production_route": (
                "authenticated_webapp_to_task_evaluation_dispatcher"
            ),
        },
        "publication": {
            "bundle_manifest": ref(21),
            "receipt": ref(22),
            "full_byte_service_account_readback_passed": True,
        },
        "evaluation_admission": {
            "zero_action_required": True,
            "scripted_positive_required": True,
            "learned_policy_admitted": False,
        },
        "revision_digest": "",
    }
    value["revision_digest"] = canonical_digest(
        value, digest_field="revision_digest"
    )
    return value


def test_accepts_terminal_configuration_artifact_for_later_evaluations() -> None:
    value = revision()
    assert validate_configured_scene_revision(value) == value


def test_accepts_provider_disclosure_only_with_its_digest_bound_decision() -> None:
    value = revision()
    decision: dict[str, object] = {
        "schema_version": DISCLOSURE_SCHEMA_VERSION,
        "render_execution_site": PROVIDER_GPU,
        "source_appearance_bytes_to_provider": True,
        "rights_admission_permits_upload": True,
        "stage_configuration_requests_upload": True,
        "human_authority_accepts_provider_terms": True,
        "provider_training_authorized": False,
        "public_redistribution_authorized": False,
        "refusals": [],
        "decision_digest": "",
    }
    decision["decision_digest"] = canonical_digest(
        decision, digest_field="decision_digest"
    )
    value["source"]["raw_source_sent_to_external_provider"] = True
    value["source"]["provider_disclosure_decision"] = decision
    value["revision_digest"] = canonical_digest(
        value, digest_field="revision_digest"
    )

    assert validate_configured_scene_revision(value) == value


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["source"].update(
            raw_source_sent_to_external_provider=True
        ),
        lambda value: value["geometry"].update(observed_source_truth_claimed=True),
        lambda value: value["publication"].update(
            full_byte_service_account_readback_passed=False
        ),
        lambda value: value["evaluation_admission"].update(
            learned_policy_admitted=True
        ),
    ],
)
def test_fails_closed_on_configuration_claim_boundary_drift(mutation) -> None:
    value = revision()
    mutation(value)
    value["revision_digest"] = canonical_digest(
        value, digest_field="revision_digest"
    )
    with pytest.raises(
        TaskEvaluationConfiguredSceneRevisionError,
        match="configured_scene_revision_invalid",
    ):
        validate_configured_scene_revision(value)


def test_digest_binds_every_published_component() -> None:
    value = revision()
    mutated = copy.deepcopy(value)
    mutated["replacement"]["asset"]["digest"] = "sha256:" + "f" * 64
    with pytest.raises(
        TaskEvaluationConfiguredSceneRevisionError,
        match="configured_scene_revision_digest_invalid",
    ):
        validate_configured_scene_revision(mutated)


def test_rejects_revision_that_repeats_construction_for_each_evaluation() -> None:
    value = revision()
    value["robot_team_interface"][
        "scene_construction_repeated_per_evaluation"
    ] = True
    value["revision_digest"] = canonical_digest(
        value, digest_field="revision_digest"
    )
    with pytest.raises(
        TaskEvaluationConfiguredSceneRevisionError,
        match="configured_scene_revision_invalid",
    ):
        validate_configured_scene_revision(value)
