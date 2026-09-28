from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_execution import (
    PARENT_LAUNCH_SCHEMA_VERSION,
    execute_and_publish_scene_configuration,
)
from blueprint_pipeline.task_evaluation_scene_configuration_orchestrator import (
    CANONICAL_ALLOCATOR,
)
from blueprint_pipeline.task_evaluation_scene_configuration_provider_runtime import (
    RESULT_SCHEMA_VERSION as STAGE_CHAIN_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_scene_configuration_disclosure import (
    CONTROL_PLANE,
    SCHEMA_VERSION as DISCLOSURE_SCHEMA_VERSION,
)
from tests.test_task_evaluation_launch_preparation_contract import (
    test_configuration_request as configuration_request_fixture,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_one_canonical_parent_launch_publishes_one_reusable_revision(
    tmp_path: Path,
) -> None:
    request = configuration_request_fixture()
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
            "disclosure_decision": {
                "schema_version": DISCLOSURE_SCHEMA_VERSION,
                "render_execution_site": CONTROL_PLANE,
                "source_appearance_bytes_to_provider": False,
                "decision_digest": "",
            },
        },
        "provider_disclosure_receipt": {
            "raw_interiorgs_bytes_in_provider_bundle": False,
        },
    }
    envelope["render_inputs_result"]["disclosure_decision"][
        "decision_digest"
    ] = canonical_digest(
        envelope["render_inputs_result"]["disclosure_decision"],
        digest_field="decision_digest",
    )
    launch_calls = 0

    def launch(*, envelope, configurations, output_root):
        nonlocal launch_calls
        launch_calls += 1
        assert configurations == {}
        provider_artifacts = output_root / "artifacts"
        provider_artifacts.mkdir()
        names = {
            "configured_appearance_without_source_object": "appearance.usdc",
            "appearance_removal_receipt": "appearance-receipt.json",
            "configured_collision_without_source_object": "collision.usda",
            "collision_excision_receipt": "collision-receipt.json",
            "statically_qualified_replacement_asset": "static.usda",
            "static_qualification_receipt": "static-receipt.json",
            "native_qualified_replacement_asset": "replacement.usda",
            "native_import_qualification_receipt": "native-receipt.json",
            "configured_scene_bundle_candidate_manifest": "candidate.json",
            "scene_assembly_receipt": "assembly-receipt.json",
        }
        rows = []
        for role, name in names.items():
            path = provider_artifacts / name
            path.write_bytes((role + "\n").encode())
            rows.append(
                {
                    "role": role,
                    "path": str(path),
                    "digest": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        # Publication also binds the reviewed task thumbnail to the exact review
        # frame it was chosen from, so the thumbnail digest has to be the one the
        # receipt names.
        thumbnail_path = provider_artifacts / "configured_task_thumbnail.png"
        thumbnail_path.write_bytes(b"configured_task_thumbnail\n")
        review_receipt = {
            "schema_version": "task_evaluation_artifixer_ai_visual_review.v1",
            "status": "accepted",
            "review_frame_count": 8,
            "task_thumbnail_is_exact_review_frame": True,
            "reviewer": {
                "kind": "ai",
                "identity": "fixture-reviewer",
                "runtime": "fixture-runtime",
                "model": "fixture-model",
            },
            "task_thumbnail_selection": {
                "camera_id": "camera_0",
                "frame_sha256": _sha256(thumbnail_path),
                "rationale": "fixture selection",
            },
            "receipt_digest": "",
        }
        review_receipt["receipt_digest"] = canonical_digest(
            review_receipt, digest_field="receipt_digest"
        )
        review_path = provider_artifacts / "appearance_visual_review_receipt.v1.json"
        review_path.write_text(
            json.dumps(review_receipt, sort_keys=True), encoding="utf-8"
        )
        for role, path in (
            ("appearance_visual_review_receipt", review_path),
            ("configured_task_thumbnail", thumbnail_path),
        ):
            rows.append(
                {
                    "role": role,
                    "path": str(path),
                    "digest": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        stage_results = [
            {"output_artifacts": rows},
            *({"output_artifacts": []} for _ in range(5)),
        ]
        chain = {
            "schema_version": STAGE_CHAIN_SCHEMA_VERSION,
            "status": "completed",
            "run_id": envelope["run_id"],
            "stage_result_digests": ["sha256:" + f"{index:064x}" for index in range(6)],
            "stage_results": stage_results,
            "stage_count": 6,
            "executed_inside_one_parent_provider_run": True,
            "nested_provider_mutations_performed": 0,
            "nested_paid_execution_requested": False,
            "evaluation_episode_executed": False,
            "retry_cap": 0,
            "result_digest": "",
        }
        chain["result_digest"] = canonical_digest(
            chain, digest_field="result_digest"
        )
        result = {
            "schema_version": PARENT_LAUNCH_SCHEMA_VERSION,
            "status": "completed",
            "run_id": envelope["run_id"],
            "submitted_via_authenticated_webapp": True,
            "dispatched_by_task_evaluation_dispatcher": True,
            "orchestration_worker_executed": True,
            "canonical_allocator": CANONICAL_ALLOCATOR,
            "provider_mutations_performed": 1,
            "paid_execution_requested": True,
            "retry_cap": 0,
            "evaluation_episode_executed": False,
            "raw_secret_values_recorded": False,
            "paid_authority_digest": "sha256:" + "a" * 64,
            "billing_reconciliation_digest": "sha256:" + "b" * 64,
            "teardown_digest": "sha256:" + "c" * 64,
            "provider_zero_digest": "sha256:" + "d" * 64,
            "launch_receipt_digest": "sha256:" + "e" * 64,
            "stage_chain": chain,
            "launch_digest": "",
        }
        result["launch_digest"] = canonical_digest(
            result, digest_field="launch_digest"
        )
        return result

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

    output = tmp_path / "output"
    output.mkdir()
    result = execute_and_publish_scene_configuration(
        envelope=envelope,
        configurations={},
        output_root=output,
        parent_launch_executor=launch,
        publisher=publish,
    )

    assert launch_calls == 1
    assert result["provider_mutations_performed"] == 1
    assert result["evaluation_episode_executed"] is False
    assert result["full_byte_service_account_readback_passed"] is True
    assert result["configured_scene_revision_reference"]["uri"].startswith(
        "s3://blueprint-production-inputs/"
    )
