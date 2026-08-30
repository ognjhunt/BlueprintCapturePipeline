from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_configured_scene_revision import (
    validate_configured_scene_revision,
)
from blueprint_pipeline.task_evaluation_scene_configuration_publication import (
    MAX_TASK_THUMBNAIL_SIZE_BYTES,
    TaskEvaluationSceneConfigurationPublicationError,
    _thumbnail_selection,
    publish_configured_scene_revision,
)
from blueprint_pipeline.task_evaluation_scene_configuration_disclosure import (
    CONTROL_PLANE,
    PROVIDER_GPU,
    SCHEMA_VERSION as DISCLOSURE_SCHEMA_VERSION,
)
from tests.test_task_evaluation_launch_preparation_contract import (
    test_configuration_request as configuration_request_fixture,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(role: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path),
        "digest": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _thumbnail_artifacts(root: Path) -> list[dict[str, object]]:
    thumbnail = root / "configured-task-thumbnail.png"
    thumbnail.write_bytes(b"exact-reviewed-frame")
    review = {
        "schema_version": "task_evaluation_artifixer_ai_visual_review.v1",
        "status": "accepted",
        "review_frame_count": 8,
        "task_thumbnail_is_exact_review_frame": True,
        "task_thumbnail_selection": {
            "camera_id": "camera-3",
            "frame_sha256": _sha256(thumbnail),
            "rationale": "The task surface and configured scene are both clear.",
        },
        "reviewer": {
            "kind": "ai",
            "identity": "artifixer-independent-vision-reviewer-v1",
            "runtime": "openai_agents_sdk",
            "model": "gpt-5.6-terra",
        },
        "receipt_digest": "",
    }
    review["receipt_digest"] = canonical_digest(
        review, digest_field="receipt_digest"
    )
    review_path = root / "appearance-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    return [
        _artifact("appearance_visual_review_receipt", review_path),
        _artifact("configured_task_thumbnail", thumbnail),
    ]


def _authorize_public_display(request: dict[str, object]) -> None:
    scene = request["scene"]
    task = request["task"]
    rights = scene["rights"]
    human_authority = next(
        row["artifact"]
        for row in rights["evidence"]
        if row["role"] == "human_authority_record"
    )
    authority = {
        "schema_version": (
            "task_evaluation_configured_scene_public_display_authorization.v1"
        ),
        "status": "authorized",
        "scope": "configured_scene_derived_listing",
        "scene_identity": dict(scene["identity"]),
        "task_identity": dict(task["identity"]),
        "subject_identity": dict(task["subject"]["identity"]),
        "rights_admission_digest": rights["admission"]["digest"],
        "human_authority_record_digest": human_authority["digest"],
        "public_slug": "interiorgs-841007-planar-mug-push",
        "title": "Planar Mug Push",
        "summary": "A robot-neutral configured scene for a planar mug push.",
        "category": "Rigid relocation",
        "allowed_fields": [
            "status",
            "scene_identity",
            "task_identity",
            "task_kind",
            "task_strategy",
            "public_title",
            "public_summary",
            "public_category",
            "thumbnail",
            "proof_boundary",
        ],
        "thumbnail_publication_authorized": True,
        "derived_metadata_publication_authorized": True,
        "private_artifact_uri_publication_authorized": False,
        "raw_media_publication_authorized": False,
        "authority_reference": "owner-public-display-authorization-20260828",
        "authorized_by": "blueprint-owner",
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    rights["public_display_authorization"] = authority


def test_thumbnail_size_ceiling_matches_private_website_delivery(
    tmp_path: Path,
) -> None:
    thumbnail = tmp_path / "thumbnail.png"
    review_path = tmp_path / "review.json"

    def seal(size: int) -> None:
        thumbnail.write_bytes(b"x" * size)
        review = {
            "schema_version": "task_evaluation_artifixer_ai_visual_review.v1",
            "status": "accepted",
            "review_frame_count": 8,
            "task_thumbnail_is_exact_review_frame": True,
            "task_thumbnail_selection": {
                "camera_id": "camera-3",
                "frame_sha256": _sha256(thumbnail),
                "rationale": "Upright task view.",
            },
            "reviewer": {
                "kind": "ai",
                "identity": "artifixer-independent-vision-reviewer-v1",
                "runtime": "openai_agents_sdk",
                "model": "gpt-5.6-terra",
            },
            "receipt_digest": "",
        }
        review["receipt_digest"] = canonical_digest(
            review, digest_field="receipt_digest"
        )
        review_path.write_text(json.dumps(review), encoding="utf-8")

    seal(MAX_TASK_THUMBNAIL_SIZE_BYTES)
    assert _thumbnail_selection(
        review_receipt_path=review_path, thumbnail_path=thumbnail
    )["frame_digest"] == _sha256(thumbnail)
    seal(MAX_TASK_THUMBNAIL_SIZE_BYTES + 1)
    with pytest.raises(
        TaskEvaluationSceneConfigurationPublicationError,
        match="scene_configuration_publication_thumbnail_selection_invalid",
    ):
        _thumbnail_selection(
            review_receipt_path=review_path, thumbnail_path=thumbnail
        )


def test_publication_refuses_diagnostic_stage_results(tmp_path: Path) -> None:
    with pytest.raises(
        TaskEvaluationSceneConfigurationPublicationError,
        match="scene_configuration_diagnostic_result_publication_forbidden",
    ):
        publish_configured_scene_revision(
            envelope={},
            stage_results=[
                {
                    "diagnostic_only": True,
                    "qualification_eligible": False,
                    "executed_inside_one_parent_provider_run": False,
                    "configured_revision_publication_permitted": False,
                    "offering_publication_permitted": False,
                }
            ],
            output_root=tmp_path,
            publisher=lambda **_kwargs: {},
        )


def _disclosure_decision(*, provider: bool) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": DISCLOSURE_SCHEMA_VERSION,
        "render_execution_site": PROVIDER_GPU if provider else CONTROL_PLANE,
        "source_appearance_bytes_to_provider": provider,
        "decision_digest": "",
    }
    value["decision_digest"] = canonical_digest(
        value, digest_field="decision_digest"
    )
    return value


def test_control_plane_publishes_reads_back_and_seals_robot_neutral_revision(
    tmp_path: Path,
) -> None:
    request = configuration_request_fixture()
    _authorize_public_display(request)
    artifacts = tmp_path / "provider-artifacts"
    artifacts.mkdir()
    roles = {
        "configured_appearance_without_source_object": "appearance.usdc",
        "appearance_removal_receipt": "appearance-receipt.json",
        "configured_collision_without_source_object": "collision.usda",
        "collision_excision_receipt": "collision-receipt.json",
        "statically_qualified_replacement_asset": "static-replacement.usda",
        "static_qualification_receipt": "static-receipt.json",
        "native_qualified_replacement_asset": "replacement.usda",
        "native_import_qualification_receipt": "native-receipt.json",
        "configured_scene_bundle_candidate_manifest": "bundle-candidate.json",
        "scene_assembly_receipt": "assembly-receipt.json",
    }
    rows = []
    for role, name in roles.items():
        path = artifacts / name
        path.write_bytes((role + "\n").encode())
        rows.append(_artifact(role, path))
    rows.extend(_thumbnail_artifacts(artifacts))
    stage_results = [{"output_artifacts": rows}]
    envelope = {
        "run_id": request["run_id"],
        "team_namespace": request["team_namespace"],
        "expected_production_commit": request["expected_production_commit"],
        "request": request,
        "recipe": {
            "scene_identity": request["scene"]["identity"],
            "task_identity": request["task"]["identity"],
            "subject_identity": request["task"]["subject"]["identity"],
            "provider_disclosure": {
                "raw_source_bytes_to_external_provider": False,
            },
        },
        "render_inputs_result": {
            "status": "derived_method_inputs_materialized",
            "raw_interiorgs_bytes_in_provider_packet": False,
            "disclosure_decision": _disclosure_decision(provider=False),
        },
        "provider_disclosure_receipt": {
            "raw_interiorgs_bytes_in_provider_bundle": False,
        },
    }
    output = tmp_path / "publication"
    output.mkdir()
    object_store = tmp_path / "object-store"
    object_store.mkdir()

    def publish(*, path: Path, object_name: str):
        destination = object_store / object_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        return {
            "uri": f"s3://blueprint-production-inputs/{object_name}",
            "digest": _sha256(path),
            "size_bytes": path.stat().st_size,
            "full_byte_service_account_readback_passed": True,
            "readback_digest": _sha256(destination),
            "readback_size_bytes": destination.stat().st_size,
        }

    result = publish_configured_scene_revision(
        envelope=envelope,
        stage_results=stage_results,
        output_root=output,
        publisher=publish,
    )

    revision_path = Path(result["configured_scene_revision"]["path"])
    revision = validate_configured_scene_revision(
        json.loads(revision_path.read_text())
    )
    assert revision["robot_team_interface"][
        "episode_packet_compiled_by_production"
    ] is True
    assert revision["robot_team_interface"][
        "configuration_run_executed_episode"
    ] is False
    assert result["full_byte_service_account_readback_passed"] is True
    assert result["configured_scene_revision_reference"]["uri"].startswith(
        "s3://blueprint-production-inputs/"
    )
    assert revision["presentation"]["task_thumbnail"]["digest"] == _sha256(
        artifacts / "configured-task-thumbnail.png"
    )
    assert revision["presentation"]["selection"]["camera_id"] == "camera-3"
    assert revision["presentation"]["selected_from_exact_reviewed_frame_count"] == 8
    offering = result["configured_scene_offering"]
    assert offering["status"] == "configured_controls_pending"
    assert offering["evaluation_admission"] == {
        "zero_action_required": True,
        "scripted_positive_required": True,
        "learned_policy_evaluation_admitted": False,
    }
    public_display = offering["public_display"]
    assert public_display["status"] == "authorized"
    assert public_display["configured_scene_revision_digest"] == revision[
        "revision_digest"
    ]
    assert public_display["task_thumbnail_digest"] == _sha256(
        artifacts / "configured-task-thumbnail.png"
    )
    assert public_display["projection_digest"] == canonical_digest(
        public_display, digest_field="projection_digest"
    )
    private_source = dict(offering)
    private_source.pop("public_display")
    assert public_display["source_offering_digest"] == canonical_digest(
        private_source, digest_field="offering_digest"
    )
    serialized_public = json.dumps(public_display).lower()
    assert request["team_namespace"].lower() not in serialized_public
    assert not any(
        marker in serialized_public
        for marker in ("s3://", "gs://", "/var/", "/private/", "api_key")
    )
    assert result["configured_scene_offering"]["evaluation_preparation_binding"][
        "configuration_source_commit"
    ] == revision["source_commit"]
    assert "source_commit" not in result["configured_scene_offering"][
        "evaluation_preparation_binding"
    ]
    assert result["configured_scene_offering"]["evaluation_preparation_binding"][
        "configured_scene_revision"
    ] == result["configured_scene_revision_reference"]
    assert result["provider_mutation_performed"] is False

    request["appearance_review_override"] = {
        "mode": "paused_ungraded",
        "scope": "artifixer_appearance_only",
        "ungraded_publication_acknowledged": True,
        "review_provider_call_permitted": False,
        "warning_label": "Visual review paused - appearance ungraded",
    }
    thumbnail_path = artifacts / "configured-task-thumbnail.png"
    pause_review = {
        "schema_version": "task_evaluation_artifixer_visual_review_pause_receipt.v1",
        "status": "visual_review_paused_ungraded",
        "decision": "not_reviewed",
        "visual_review_mode": "paused_ungraded",
        "publisher_instance_id": "104",
        "review_frame_count": 8,
        "frames": [
            {"camera_id": f"camera-{index}", "frame_sha256": _sha256(thumbnail_path)}
            for index in range(8)
        ],
        "all_review_frames_digest_bound": True,
        "ai_visual_review_completed": False,
        "human_review_completed": False,
        "semantic_object_absence_review_passed": False,
        "multiview_consistency_review_passed": False,
        "task_thumbnail_is_exact_review_frame": False,
        "task_thumbnail_is_exact_rendered_frame": True,
        "task_thumbnail_selection": {
            "camera_id": "camera-0",
            "frame_sha256": _sha256(thumbnail_path),
            "rationale": "Deterministic ungraded thumbnail.",
        },
        "selector": {
            "kind": "system",
            "identity": "deterministic_ungraded_thumbnail_selector",
            "runtime": "blueprint_pipeline",
            "model": "none",
        },
        "review_provider_call_performed": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "warning_label": "Visual review paused - appearance ungraded",
        "receipt_digest": "",
    }
    pause_review["receipt_digest"] = canonical_digest(
        pause_review, digest_field="receipt_digest"
    )
    pause_review_path = artifacts / "appearance-review.json"
    pause_review_path.write_text(json.dumps(pause_review), encoding="utf-8")
    pause_removal = {
        "schema_version": "task_evaluation_artifixer_object_removal_result.v1",
        "status": "completed_ungraded_generated_appearance_edit",
        "visual_review_mode": "paused_ungraded",
        "publisher_instance_id": "104",
        "visual_review_receipt_digest": pause_review["receipt_digest"],
        "warning_label": "Visual review paused - appearance ungraded",
        "result_digest": "",
    }
    pause_removal["result_digest"] = canonical_digest(
        pause_removal, digest_field="result_digest"
    )
    (artifacts / "appearance-receipt.json").write_text(
        json.dumps(pause_removal), encoding="utf-8"
    )
    paused_rows = [
        _artifact(str(row["role"]), Path(str(row["path"]))) for row in rows
    ]
    paused_output = tmp_path / "paused-publication"
    paused_output.mkdir()
    paused = publish_configured_scene_revision(
        envelope=envelope,
        stage_results=[{"output_artifacts": paused_rows}],
        output_root=paused_output,
        publisher=publish,
    )
    paused_revision = validate_configured_scene_revision(
        json.loads(Path(paused["configured_scene_revision"]["path"]).read_text())
    )
    assert paused_revision["presentation"]["appearance_review_status"] == (
        "paused_ungraded"
    )
    assert paused_revision["presentation"][
        "selected_from_exact_reviewed_frame_count"
    ] == 0
    assert paused["configured_scene_offering"]["status"] == (
        "configured_controls_pending"
    )
    assert paused["configured_scene_offering"]["proof_boundary"][
        "appearance_visual_review_completed"
    ] is False
    assert paused["configured_scene_offering"]["proof_boundary"][
        "appearance_warning_label"
    ] == "Visual review paused - appearance ungraded"


def test_revision_reports_provider_disclosure_truthfully(tmp_path: Path) -> None:
    request = configuration_request_fixture()
    artifacts = tmp_path / "provider-artifacts"
    artifacts.mkdir()
    names = {
        "configured_appearance_without_source_object": "appearance.usdc",
        "appearance_removal_receipt": "appearance-receipt.json",
        "configured_collision_without_source_object": "collision.usda",
        "collision_excision_receipt": "collision-receipt.json",
        "statically_qualified_replacement_asset": "static-replacement.usda",
        "static_qualification_receipt": "static-receipt.json",
        "native_qualified_replacement_asset": "replacement.usda",
        "native_import_qualification_receipt": "native-receipt.json",
        "configured_scene_bundle_candidate_manifest": "bundle-candidate.json",
        "scene_assembly_receipt": "assembly-receipt.json",
    }
    rows = []
    for role, name in names.items():
        path = artifacts / name
        path.write_bytes((role + "\n").encode())
        rows.append(_artifact(role, path))
    rows.extend(_thumbnail_artifacts(artifacts))
    decision = _disclosure_decision(provider=True)
    envelope = {
        "run_id": request["run_id"],
        "team_namespace": request["team_namespace"],
        "expected_production_commit": request["expected_production_commit"],
        "request": request,
        "recipe": {
            "scene_identity": request["scene"]["identity"],
            "task_identity": request["task"]["identity"],
            "subject_identity": request["task"]["subject"]["identity"],
        },
        "render_inputs_result": {
            "status": "derived_method_inputs_pending_provider_render",
            "raw_interiorgs_bytes_in_provider_packet": True,
            "disclosure_decision": decision,
        },
        "provider_disclosure_receipt": {
            "raw_interiorgs_bytes_in_provider_bundle": True,
        },
    }
    output = tmp_path / "publication"
    output.mkdir()
    object_store = tmp_path / "object-store"
    object_store.mkdir()

    def publish(*, path: Path, object_name: str):
        destination = object_store / object_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        return {
            "uri": f"s3://blueprint-production-inputs/{object_name}",
            "digest": _sha256(path),
            "size_bytes": path.stat().st_size,
            "full_byte_service_account_readback_passed": True,
            "readback_digest": _sha256(destination),
            "readback_size_bytes": destination.stat().st_size,
        }

    result = publish_configured_scene_revision(
        envelope=envelope,
        stage_results=[{"output_artifacts": rows}],
        output_root=output,
        publisher=publish,
    )
    revision = validate_configured_scene_revision(
        json.loads(Path(result["configured_scene_revision"]["path"]).read_text())
    )

    assert revision["source"]["raw_source_sent_to_external_provider"] is True
    assert "public_display" not in result["configured_scene_offering"]
