from __future__ import annotations

import hashlib
import json
import os
import pwd
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
    stage_launch_preparation_request,
)
from blueprint_pipeline.task_evaluation_launch_preparation_worker import (
    process_launch_preparation_queue,
)
from blueprint_pipeline.task_evaluation_scene_configuration_orchestrator import (
    CANONICAL_ALLOCATOR,
    PROVIDER_EXECUTION_SCHEMA_VERSION,
    STAGE_RESULT_SCHEMA_VERSION,
    process_scene_configuration_queue,
)
from tests.test_task_evaluation_configured_scene_revision import revision
from tests.test_task_evaluation_launch_preparation_worker import (
    fetcher,
    fake_scene_render_inputs,
    production_request_with_fetchable_bytes,
)


SERVICE_ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name


def staged_configuration(tmp_path: Path) -> tuple[dict, Path, Path]:
    value, payloads = production_request_with_fetchable_bytes()
    preparation_queue = tmp_path / "preparation-queue"
    construction_queue = tmp_path / "construction-queue"
    inputs = tmp_path / "inputs"
    stage_launch_preparation_request(
        value=value,
        queue_root=preparation_queue,
        submitted_by="blueprint-webapp",
    )
    prepared = process_launch_preparation_queue(
        queue_root=preparation_queue,
        input_root=inputs,
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        source_commit=value["expected_production_commit"],
        fetcher=fetcher(payloads),
        construction_queue_root=construction_queue,
        scene_render_input_materializer=fake_scene_render_inputs,
    )
    assert prepared["results"][0]["status"] == (
        "queued_for_production_scene_configuration"
    )
    return value, construction_queue, inputs


def executor(*, invalid_parent_governance: bool = False):
    observed: list[str] = []

    def execute(**kwargs):
        envelope = kwargs["envelope"]
        output_root = kwargs["output_root"]
        configurations = kwargs["configurations"]
        stage_results = []
        for stage in envelope["recipe"]["stage_sequence"]:
            configuration_path = configurations[stage["stage_id"]][1]
            stage_output = output_root / stage["stage_id"]
            stage_output.mkdir()
            observed.append(stage["stage_id"])
            artifact_path = stage_output / f"{stage['stage_id']}.json"
            role = f"{stage['capability']}_result"
            value = {
                "schema_version": "test_stage_artifact.v1",
                "stage_id": stage["stage_id"],
            }
            artifact_path.write_text(
                json.dumps(value, sort_keys=True), encoding="utf-8"
            )
            artifact_bytes = artifact_path.read_bytes()
            result = {
                "schema_version": STAGE_RESULT_SCHEMA_VERSION,
                "status": "completed",
                "stage_id": stage["stage_id"],
                "capability": stage["capability"],
                "execution_class": stage["execution_class"],
                "configuration_digest": "sha256:"
                + hashlib.sha256(configuration_path.read_bytes()).hexdigest(),
                "canonical_allocator": None,
                "provider_mutations_performed": 0,
                "paid_execution_requested": False,
                "executed_inside_parent_configuration_run": True,
                "retry_cap": 0,
                "raw_secret_values_recorded": False,
                "output_artifacts": [
                    {
                        "role": role,
                        "path": str(artifact_path),
                        "digest": "sha256:"
                        + hashlib.sha256(artifact_bytes).hexdigest(),
                        "size_bytes": len(artifact_bytes),
                    }
                ],
                "stage_result_digest": "",
            }
            result["stage_result_digest"] = canonical_digest(
                result, digest_field="stage_result_digest"
            )
            stage_results.append(result)
        configured = revision()
        configured["configuration_run_id"] = envelope["run_id"]
        configured["team_namespace"] = envelope["team_namespace"]
        configured["scene_identity"] = envelope["recipe"]["scene_identity"]
        configured["source_commit"] = envelope["expected_production_commit"]
        configured["task_template"]["identity"] = envelope["recipe"][
            "task_identity"
        ]
        configured["replacement"]["identity"] = envelope["recipe"][
            "subject_identity"
        ]
        configured["revision_digest"] = canonical_digest(
            configured, digest_field="revision_digest"
        )
        revision_path = output_root / "configured_scene_revision.v1.json"
        revision_path.write_text(
            json.dumps(configured, sort_keys=True), encoding="utf-8"
        )
        revision_bytes = revision_path.read_bytes()
        parent = {
            "schema_version": PROVIDER_EXECUTION_SCHEMA_VERSION,
            "status": "completed",
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
            "stage_results": stage_results,
            "configured_scene_revision": {
                "role": "configured_scene_revision",
                "path": str(revision_path),
                "digest": "sha256:"
                + hashlib.sha256(revision_bytes).hexdigest(),
                "size_bytes": len(revision_bytes),
            },
            "execution_digest": "",
        }
        if invalid_parent_governance:
            parent.pop("provider_zero_digest")
        parent["execution_digest"] = canonical_digest(
            parent, digest_field="execution_digest"
        )
        return parent

    return observed, execute


def test_automatically_executes_configuration_chain_and_emits_no_episode(
    tmp_path,
) -> None:
    value, queue, inputs = staged_configuration(tmp_path)
    observed, execute = executor()
    run = process_scene_configuration_queue(
        queue_root=queue,
        input_root=inputs,
        output_root=tmp_path / "outputs",
        source_commit=value["expected_production_commit"],
        configuration_run_executor=execute,
    )

    result = run["results"][0]
    assert result["status"] == "configured"
    assert result["stage_count"] == 6
    assert result["evaluation_episode_executed"] is False
    assert result["automatic_progression_performed"] is True
    assert observed == [f"stage-{index}" for index in range(1, 7)]
    assert len(list((queue / "completed").glob("*.json"))) == 1


def test_parent_run_without_provider_zero_blocks_configuration_result(
    tmp_path,
) -> None:
    value, queue, inputs = staged_configuration(tmp_path)
    observed, execute = executor(invalid_parent_governance=True)
    run = process_scene_configuration_queue(
        queue_root=queue,
        input_root=inputs,
        output_root=tmp_path / "outputs",
        source_commit=value["expected_production_commit"],
        configuration_run_executor=execute,
    )

    result = run["results"][0]
    assert result["status"] == "blocked"
    assert result["blockers"] == [
        "scene_configuration_parent_execution_governance_invalid:provider_zero_digest"
    ]
    assert observed == [f"stage-{index}" for index in range(1, 7)]
    assert len(list((queue / "blocked").glob("*.json"))) == 1
