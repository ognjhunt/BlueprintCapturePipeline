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
    assert result["configured_scene_offering"]["status"] == "launch_ready"
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
