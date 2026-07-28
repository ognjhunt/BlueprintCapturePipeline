from __future__ import annotations

import json
import zipfile

from blueprint_pipeline.policy_ranking_cosmos_reasoner_gpu_admission import (
    ADMISSION_SCHEMA,
    AUTHORIZATION_ID,
    PREFLIGHT_SCHEMA,
    build_admission,
    build_authorization_record,
    inspect_bundle,
)
from blueprint_pipeline.policy_ranking_evaluator_diagnostic import (
    COSMOS_MODEL,
    COSMOS_REVISION,
)
from blueprint_pipeline.policy_ranking_evaluator_diagnostic_cosmos_bundle import (
    PUBLIC_IMAGE,
    RECEIPT_SCHEMA_VERSION,
)
from blueprint_pipeline.policy_ranking_evaluator_diagnostic_cosmos_provider_runtime import (
    _extract_json_object,
    _validate_payload,
)
from blueprint_pipeline.policy_ranking_roboarena_calibration import file_sha256


COMMIT = "a" * 40


def _payload() -> dict:
    return {
        "preferred_episode": "A",
        "episode_a_progress_0_to_5": 4,
        "episode_b_progress_0_to_5": 2,
        "stable_success_a": False,
        "stable_success_b": False,
        "comparison_confidence": 0.7,
        "uncertainty": 0.3,
        "decisive_evidence": ["A advances the object farther"],
        "artifact_flags_a": [],
        "artifact_flags_b": ["frozen future"],
        "abstention_factors": [],
    }


def _bundle(tmp_path):
    path = tmp_path / "pilot.zip"
    runner = """
evaluator_runtime_result.json
nvidia/Cosmos3-Nano
post_unseal_diagnostic_only
"""
    entrypoint = """
write_missing_result() { :; }
evaluator_runner_process_exited_without_runtime_result
blocked_evaluator_process_exited_without_result
"""
    runtime = {
        "model": COSMOS_MODEL,
        "model_revision": COSMOS_REVISION,
        "public_image": PUBLIC_IMAGE,
    }
    inputs = {"pair_count": 7, "claim_class": "post_unseal_diagnostic_only"}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("provider_runtime/evaluator_provider_runtime_runner.py", runner)
        archive.writestr("provider_runtime/run_evaluator_provider_runtime.sh", entrypoint)
        archive.writestr(
            "provider_runtime/evaluator_provider_runtime_manifest.json",
            json.dumps(runtime),
        )
        archive.writestr(
            "provider_runtime/evaluator_input_manifest.json", json.dumps(inputs)
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "bundle_sha256": file_sha256(path),
        "pair_count": 7,
    }
    return path, receipt


def _preflight(now: float) -> dict:
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "verified",
        "provider_inventory_verified_zero": True,
        "provider_mutations_performed": 0,
        "observed_at_epoch": now,
        "selected_offer": {
            "gpu_name": "H100 SXM",
            "hourly_rate_usd": 2.25,
            "gpu_ram_mb": 81_559,
            "reliability": 0.99,
        },
    }


def test_reasoner_payload_extracts_after_reasoning_prefix():
    text = "analysis omitted\n```json\n" + json.dumps(_payload()) + "\n```"
    assert _validate_payload(_extract_json_object(text)) == _payload()


def test_reasoner_bundle_and_security_exception_admit_dry_run(tmp_path, monkeypatch):
    bundle_path, receipt = _bundle(tmp_path)
    inspection = inspect_bundle(bundle_path, receipt)
    assert inspection["status"] == "passed"
    authorization = build_authorization_record(source_commit=COMMIT)
    assert authorization["authorization_id"] == AUTHORIZATION_ID
    monkeypatch.setattr(
        "blueprint_pipeline.policy_ranking_cosmos_reasoner_gpu_admission.time.time",
        lambda: 1_001.0,
    )
    admission = build_admission(
        authorization=authorization,
        preflight=_preflight(1_000.0),
        bundle=inspection,
        expected_source_commit=COMMIT,
        execute=True,
    )
    assert admission["schema_version"] == ADMISSION_SCHEMA
    assert admission["status"] == "admitted"
    assert admission["blockers"] == []


def test_reasoner_admission_rejects_unacknowledged_rotation_metadata(tmp_path):
    bundle_path, receipt = _bundle(tmp_path)
    authorization = build_authorization_record(source_commit=COMMIT)
    authorization["task_security_exception"][
        "rotation_metadata_missing_acknowledged"
    ] = False
    admission = build_admission(
        authorization=authorization,
        preflight=_preflight(1_000.0),
        bundle=inspect_bundle(bundle_path, receipt),
        expected_source_commit=COMMIT,
        execute=False,
    )
    assert admission["status"] == "blocked"
    assert "cosmos_reasoner_vast_security_exception_invalid" in admission["blockers"]
