from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.articulated_public_scene_pipeline import (
    ArticulatedPublicScenePipelineError,
    compile_articulated_public_scene_state,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.registered_scene_components import (
    build_registered_scene_components,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _freeze() -> dict:
    return json.loads(
        (
            REPO_ROOT
            / "docs/arm_decision_proof_v1/manifests"
            / "second_scene_840796_scene_task_freeze.v1.json"
        ).read_text(encoding="utf-8")
    )


def _with_digest(value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _inputs() -> dict:
    freeze = _freeze()
    components = build_registered_scene_components(freeze)
    appearance, appearance_receipt = components["interiorgs_appearance_scene"]
    collision, collision_receipt = components["sage3d_collision_companion"]
    inpainting = _with_digest(
        {
            "schema_version": "adp009b_interiorgs_edit_input_receipt.v1",
            "status": "render_derived_input_packet_materialized",
            "scene": {
                "publisher_scene_id": "840796",
                "target_instance_id": "123",
            },
            "receipt_digest": "",
        },
        "receipt_digest",
    )
    aura = _with_digest(
        {
            "schema_version": "adp009b_aurafusion360_adapter_receipt.v1",
            "status": "prepared_unexecuted",
            "scene": {
                "publisher_scene_id": "840796",
                "target_instance_id": "123",
                "input_receipt_digest": inpainting["receipt_digest"],
            },
            "execution": {"aurafusion360_interiorgs_executed": False},
            "receipt_digest": "",
        },
        "receipt_digest",
    )
    source = _with_digest(
        {
            "schema_version": "articulated_source_asset.v1",
            "status": "materialized",
            "target": {
                "interiorgs_instance_id": "123",
                "semantic_label": "refrigerator",
            },
            "output_asset": {"sha256": "sha256:" + "a" * 64},
            "connected_component_count": 28,
            "receipt_digest": "",
        },
        "receipt_digest",
    )
    joint = _with_digest(
        {
            "schema_version": "usd_content_joint_agent_packet.v1",
            "status": "blocked_before_remote_execution",
            "source_asset": {
                "sha256": source["output_asset"]["sha256"],
                "source_receipt_digest": source["receipt_digest"],
                "connected_component_count": 28,
            },
            "execution_admission": {
                "external_disclosure_authorized": False,
                "paid_execution_authorized": False,
                "remote_execution_performed": False,
            },
            "packet_digest": "",
        },
        "packet_digest",
    )
    return {
        "freeze": freeze,
        "appearance_manifest": appearance,
        "appearance_receipt": appearance_receipt,
        "collision_manifest": collision,
        "collision_receipt": collision_receipt,
        "inpainting_input_receipt": inpainting,
        "aura_adapter_receipt": aura,
        "articulated_source_receipt": source,
        "joint_agent_packet": joint,
        "repository_commit": "a" * 40,
    }


def _authority(inputs: dict) -> dict:
    source = inputs["articulated_source_receipt"]
    value = {
        "schema_version": "public_scene_execution_authority.v1",
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": "goal-user-message-2026-08-08",
        "authorized_by": "fixture-owner",
        "authorized_on": "2026-08-08",
        "publisher_scene_id": inputs["freeze"]["scene"]["publisher_scene_id"],
        "target_instance_id": inputs["freeze"]["scene"]["target_instance_id"],
        "freeze_digest": inputs["freeze"]["freeze_digest"],
        "prior_rights_authority_digest": "sha256:" + "1" * 64,
        "interiorgs_terms_text_digest": "sha256:" + "2" * 64,
        "aura_adapter_receipt_digest": inputs["aura_adapter_receipt"][
            "receipt_digest"
        ],
        "joint_agent_source_receipt_digest": source["receipt_digest"],
        "joint_agent_source_asset_digest": source["output_asset"]["sha256"],
        "provider_scope": [
            "vast",
            "nvidia_nim",
            "nvidia_remote_renderer",
            "object_store",
        ],
        "purpose_scope": [
            "released_code_inpainting",
            "articulation_topology_inference",
            "native_simulator_qualification",
            "two_candidate_policy_evaluation",
        ],
        "hard_total_spend_cap_usd": 12.0,
        "maximum_single_resource_ttl_seconds": 14400,
        "remote_upload_authorized": True,
        "paid_compute_authorized": True,
        "derived_aura_adapter_upload_authorized": True,
        "sage_cc_by_nc_derived_asset_upload_authorized": True,
        "one_instance_at_a_time": True,
        "provider_zero_required_before_and_after": True,
        "teardown_required": True,
        "raw_interiorgs_downloaded_bytes_upload_authorized": False,
        "public_disclosure_authorized": False,
        "model_training_authorized": False,
        "commercial_use_authorized": False,
        "automatic_paid_retry_authorized": False,
        "retention_policy": "bounded_to_goal_then_provider_zero",
        "dataset_claim": (
            "internal_noncommercial_private_processing_authority_only;"
            "does_not_change_publisher_nonredistribution_terms"
        ),
        "authorization_digest": "",
    }
    value["authorization_digest"] = canonical_digest(
        value, digest_field="authorization_digest"
    )
    return value


def test_non_agent_pipeline_stops_at_exact_rights_boundary() -> None:
    result = compile_articulated_public_scene_state(**_inputs())

    assert result["status"] == "typed_abstention_before_simready_build"
    assert result["smallest_blocker"] == (
        "external_scene_derived_byte_disclosure_authority_missing"
    )
    assert result["frozen_candidates"] == ["pi05_droid", "groot_n17_droid"]
    assert result["stage_status"]["evaluation_authorized_removal_inputs_materialized"]
    assert result["stage_status"]["simready_replacement_materialized"] is False
    assert result["claim_boundary"][
        "same_non_agent_stage_contracts_as_interactive_rehearsal"
    ]
    assert result["run_digest"] == canonical_digest(result, digest_field="run_digest")


def test_bound_authority_removes_only_rights_and_budget_blockers() -> None:
    inputs = _inputs()
    inputs["execution_authority"] = _authority(inputs)

    result = compile_articulated_public_scene_state(**inputs)

    assert result["blockers"] == [
        "released_code_inpainting_execution_missing",
        "joint_agent_topology_execution_missing",
    ]
    assert result["execution_authority"]["hard_total_spend_cap_usd"] == 12.0


def test_authority_cannot_cross_join_another_scene() -> None:
    inputs = _inputs()
    authority = _authority(inputs)
    authority["publisher_scene_id"] = "840999"
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    inputs["execution_authority"] = authority

    with pytest.raises(
        ArticulatedPublicScenePipelineError,
        match="execution_authority_scene_join_invalid",
    ):
        compile_articulated_public_scene_state(**inputs)


@pytest.mark.parametrize(
    ("scene_id", "target_id"),
    [("840313", "160"), ("840796", "123")],
)
def test_pipeline_join_is_scene_and_task_object_neutral(
    scene_id: str, target_id: str
) -> None:
    inputs = _inputs()
    inputs["freeze"]["scene"].update(
        {
            "publisher_scene_id": scene_id,
            "target_instance_id": target_id,
            "interiorgs_folder": f"fixture_{scene_id}",
        }
    )
    inputs["freeze"]["freeze_digest"] = canonical_digest(
        inputs["freeze"], digest_field="freeze_digest"
    )
    components = build_registered_scene_components(inputs["freeze"])
    inputs["appearance_manifest"], inputs["appearance_receipt"] = components[
        "interiorgs_appearance_scene"
    ]
    inputs["collision_manifest"], inputs["collision_receipt"] = components[
        "sage3d_collision_companion"
    ]
    for key in ("inpainting_input_receipt", "aura_adapter_receipt"):
        inputs[key]["scene"]["publisher_scene_id"] = scene_id
        inputs[key]["scene"]["target_instance_id"] = target_id
    inputs["inpainting_input_receipt"] = _with_digest(
        inputs["inpainting_input_receipt"], "receipt_digest"
    )
    inputs["aura_adapter_receipt"]["scene"]["input_receipt_digest"] = inputs[
        "inpainting_input_receipt"
    ]["receipt_digest"]
    inputs["aura_adapter_receipt"] = _with_digest(
        inputs["aura_adapter_receipt"], "receipt_digest"
    )
    inputs["articulated_source_receipt"]["target"][
        "interiorgs_instance_id"
    ] = target_id
    inputs["articulated_source_receipt"] = _with_digest(
        inputs["articulated_source_receipt"], "receipt_digest"
    )
    inputs["joint_agent_packet"]["source_asset"]["source_receipt_digest"] = inputs[
        "articulated_source_receipt"
    ]["receipt_digest"]
    inputs["joint_agent_packet"] = _with_digest(
        inputs["joint_agent_packet"], "packet_digest"
    )

    result = compile_articulated_public_scene_state(**inputs)

    assert result["scene"]["publisher_scene_id"] == scene_id
    assert result["scene"]["target_instance_id"] == target_id


def test_pipeline_rejects_cross_scene_aura_receipt() -> None:
    inputs = _inputs()
    aura = copy.deepcopy(inputs["aura_adapter_receipt"])
    aura["scene"]["publisher_scene_id"] = "other"
    inputs["aura_adapter_receipt"] = _with_digest(aura, "receipt_digest")

    with pytest.raises(
        ArticulatedPublicScenePipelineError, match="aura_scene_join_invalid"
    ):
        compile_articulated_public_scene_state(**inputs)


def test_pipeline_rejects_joint_agent_source_substitution() -> None:
    inputs = _inputs()
    joint = copy.deepcopy(inputs["joint_agent_packet"])
    joint["source_asset"]["sha256"] = "sha256:" + "b" * 64
    inputs["joint_agent_packet"] = _with_digest(joint, "packet_digest")

    with pytest.raises(
        ArticulatedPublicScenePipelineError,
        match="joint_agent_source_asset_join_invalid",
    ):
        compile_articulated_public_scene_state(**inputs)
