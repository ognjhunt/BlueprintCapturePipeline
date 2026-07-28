from __future__ import annotations

import json
import zipfile

from blueprint_pipeline.policy_ranking_cosmos_reasoner_gpu_admission import (
    ADMISSION_SCHEMA,
    AUTHORIZATION_ID,
    MAX_HOURLY_RATE_USD,
    PREFLIGHT_SCHEMA,
    TARGET_MAX_LIVE_MINUTES,
    TARGET_SPEND_USD,
    build_admission,
    inspect_bundle,
    load_external_authorization,
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
from blueprint_pipeline.policy_ranking_roboarena_calibration import (
    canonical_sha256,
    file_sha256,
)


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
        "source_commit": COMMIT,
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
        "source_commit": COMMIT,
    }
    return path, receipt


def _external_authorization(tmp_path, monkeypatch, *, source_commit=COMMIT):
    authority_root = tmp_path / "external-authorizations"
    authority_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        "blueprint_pipeline.policy_ranking_cosmos_reasoner_gpu_admission.EXTERNAL_AUTHORIZATION_ROOT",
        authority_root,
    )
    record = {
        "schema_version": "policy_ranking_cosmos_reasoner_compute_authorization.v1",
        "authorization_id": AUTHORIZATION_ID,
        "experiment_id": "policy_ranking_roboarena_full_stack_calibration_20260728",
        "source_commit": source_commit,
        "authorization_origin": "external_workspace_user_task_authority",
        "paid_mutation_authorized": True,
        "authorized_by": "workspace_user",
        "authorized_compute_cap_usd": 5.0,
        "reasoner_arm_total_cap_usd": 15.0,
        "maximum_provider_allocations": 1,
        "hard_ttl_seconds": 7200,
        "task_security_exception": {
            "existing_vast_key_use_explicitly_authorized": True,
            "rotation_metadata_missing_acknowledged": True,
            "provider_side_rotation_event_claimed": False,
            "key_exposure_evidence_found": False,
            "live_authenticated_validation_required": True,
        },
    }
    record["authorization_sha256"] = canonical_sha256(record)
    path = authority_root / "reasoner-pilot.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    path.chmod(0o600)
    return load_external_authorization(path)


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


def test_reasoner_session_window_stays_below_frozen_target_spend():
    projected = TARGET_MAX_LIVE_MINUTES / 60 * MAX_HOURLY_RATE_USD
    next_minute = (TARGET_MAX_LIVE_MINUTES + 1) / 60 * MAX_HOURLY_RATE_USD
    assert projected <= TARGET_SPEND_USD
    assert next_minute > TARGET_SPEND_USD


def test_reasoner_bundle_and_security_exception_admit_dry_run(tmp_path, monkeypatch):
    bundle_path, receipt = _bundle(tmp_path)
    inspection = inspect_bundle(
        bundle_path, receipt, expected_source_commit=COMMIT
    )
    assert inspection["status"] == "passed"
    authorization = _external_authorization(tmp_path, monkeypatch)
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


def test_reasoner_admission_rejects_unacknowledged_rotation_metadata(
    tmp_path, monkeypatch
):
    bundle_path, receipt = _bundle(tmp_path)
    authorization = _external_authorization(tmp_path, monkeypatch)
    authorization["task_security_exception"][
        "rotation_metadata_missing_acknowledged"
    ] = False
    admission = build_admission(
        authorization=authorization,
        preflight=_preflight(1_000.0),
        bundle=inspect_bundle(
            bundle_path, receipt, expected_source_commit=COMMIT
        ),
        expected_source_commit=COMMIT,
        execute=False,
    )
    assert admission["status"] == "blocked"
    assert "cosmos_reasoner_vast_security_exception_invalid" in admission["blockers"]


def test_reasoner_bundle_rejects_stale_source_commit(tmp_path):
    bundle_path, receipt = _bundle(tmp_path)
    inspection = inspect_bundle(
        bundle_path, receipt, expected_source_commit="b" * 40
    )
    assert inspection["status"] == "blocked"
    assert "cosmos_reasoner_bundle_source_commit_mismatch" in inspection["blockers"]
