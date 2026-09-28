"""G1 billing may close a failed episode without inventing policy results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import vast_official_billing_extractor as billing


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _record(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path), "size_bytes": len(payload),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }


def _case(root: Path) -> tuple[Path, dict]:
    instance_id = 52686067
    attempt = root / "attempts/attempt_001"
    provider = attempt / "vast_provider_run"
    adapter = _write(provider / "vast_provider_adapter_result.json", {
        "schema_version": "vast_provider_adapter_result.v1",
        "provider_bundle_kind": "native_g1_development_campaign",
        "vast_instance_ids": [instance_id], "status": "completed",
        "final_validation_status": "passed", "continuing_spend_from_this_run": False,
        "retained_owned": False, "raw_api_key_stored": False,
        "secret_values_in_artifact": False,
    })
    teardown = _write(provider / "vast_teardown_manifest.json", {
        "schema_version": "vast_teardown_manifest.v1", "status": "completed",
        "vast_instance_ids": [instance_id], "runner_gpu_teardown_completed": True,
        "continuing_spend_from_this_run": False, "retention_authorized": False,
        "raw_secret_values_recorded": False,
    })
    bundle = "sha256:" + "a" * 64
    artifact = {
        "schema_version": "task_evaluation_artifact_manifest.v1",
        "status": "completed", "blockers": [],
        "binding": {"bundle_sha256": bundle, "attempt_number": 1},
    }
    artifact["manifest_digest"] = canonical_digest(artifact, digest_field="manifest_digest")
    artifact_path = _write(attempt / "artifact_manifest.json", artifact)
    native = {
        "schema_version": "native_g1_provider_campaign_result.v1", "status": "blocked",
        "claim_ceiling": "development_only", "ranking_eligible": False,
        "physical_outcome_claimed": False, "candidate_policy_queried": False,
        "pairs": [], "policy_query_counts": {},
    }
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    native_path = _write(
        attempt / "immutable_execution/native_g1_provider_campaign_result.v1.json", native
    )
    watcher_path = _write(
        attempt / "independent_vast_watchdog/groot_oscar_runpod_canary_watchdog.json",
        {
            "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
            "provider": "vast", "status": "provider_terminal",
            "provider_absence_confirmed": True,
            "final_global_inventory": {"api_confirmed": True, "live_resource_count": 0},
            "raw_secret_values_recorded": False,
        },
    )
    cleanup = _write(
        attempt / "object_store_staging/wam_provider_object_store_cleanup.json",
        {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "status": "completed", "all_objects_absent": True,
            "all_ephemeral_objects_absent": True, "blockers": [],
            "raw_secret_values_recorded": False,
        },
    )
    watchdog_close = {
        "schema_version": "vast_independent_watchdog_handoff.v1",
        "status": "provider_terminal", "instance_ids": [instance_id],
        "provider_absence_confirmed": True, "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    result = {
        "schema_version": "native_g1_paid_campaign_result.v1", "status": "blocked",
        "attempt_number": 1, "attempt_root": str(attempt), "retry_cap": 0,
        "bundle_sha256": bundle, "continuing_spend_from_this_run": False,
        "raw_secret_values_recorded": False, "all_staged_objects_absent": True,
        "adapter_result_path": str(adapter), "teardown_manifest_path": str(teardown),
        "artifact_manifest_path": str(artifact_path),
        "native_control_result_path": str(native_path),
        "native_control_result_digest": native["result_digest"],
        "watchdog_receipt_path": str(watcher_path),
        "object_store_cleanup_path": str(cleanup),
        "independent_watchdog_handoff": {
            "status": "armed", "watchdog_armed_before_allocation": True,
        },
        "independent_watchdog": watchdog_close,
        "independent_watchdog_close": watchdog_close,
        "provider_closeout": {
            "provider_zero_confirmed": True, "warm_session_retained": False,
            "all_staged_objects_absent": True,
            "adapter_result": _record(adapter), "teardown_manifest": _record(teardown),
        },
    }
    path = _write(root / "adp_arena_vast_result.json", result)
    _write(attempt / "adp_arena_vast_result.json", result)
    return path, result


def _pre_native_case(root: Path) -> tuple[Path, dict]:
    path, result = _case(root)
    Path(result["native_control_result_path"]).unlink()
    artifact_path = Path(result["artifact_manifest_path"])
    artifact = json.loads(artifact_path.read_text())
    artifact["status"] = "blocked"
    artifact["blockers"] = ["task_evaluation_artifact_role_missing:provider_runtime_evidence"]
    artifact["manifest_digest"] = canonical_digest(artifact, digest_field="manifest_digest")
    _write(artifact_path, artifact)
    result.update({
        "native_control_result_path": None,
        "native_control_result_digest": None,
        "candidate_policy_queried": False,
        "scientific_attempt_started": False,
        "first_observation_reached": False,
    })
    _write(path, result)
    _write(Path(result["attempt_root"]) / "adp_arena_vast_result.json", result)
    return path, result


def _selected_case(root: Path, *, status="completed", pre_native=False) -> tuple[Path, dict]:
    path, result = (_pre_native_case if pre_native else _case)(root)
    result.update(schema_version="native_g1_team_paid_policy_result.v1", status=status)
    adapter_path = Path(result["adapter_result_path"])
    adapter = json.loads(adapter_path.read_text())
    adapter["provider_bundle_kind"] = "native_g1_team_policy"
    _write(adapter_path, adapter)
    result["provider_closeout"]["adapter_result"] = _record(adapter_path)
    if not pre_native:
        Path(result["native_control_result_path"]).unlink()
        native = {
            "schema_version": "native_g1_team_provider_result.v1",
            "status": "completed_development_only" if status == "completed" else "blocked",
            "claim_ceiling": "development_only", "public_redistribution_authorized": False,
            "candidate_policy_queried": True if status == "completed" else None,
        }
        native["result_digest"] = canonical_digest(native, digest_field="result_digest")
        native_path = _write(root / "attempts/attempt_001/immutable_execution/native_g1_team_provider_result.v1.json", native)
        result.update(native_control_result_path=str(native_path), native_control_result_digest=native["result_digest"])
    _write(path, result)
    _write(Path(result["attempt_root"]) / "adp_arena_vast_result.json", result)
    return path, result


@pytest.mark.parametrize("status,pre_native", [("completed", False), ("blocked", False), ("blocked", True)])
def test_selected_g1_financial_closeout_preserves_scientific_claim_ceiling(tmp_path, status, pre_native):
    path, _ = _selected_case(tmp_path / "run", status=status, pre_native=pre_native)
    evidence = billing._terminal_evidence(instance_id=52686067, terminal_result_path=path)
    assert evidence["financial_closeout_kind"] == "native_g1_team_paid_policy.v1"
    assert evidence["terminal_status"] == status
    assert evidence["policy_evaluation_qualified"] is False
    assert evidence["scientific_success_inferred"] is False
    assert ("native_result" in evidence) is not pre_native


def _selected_billing_source(root: Path, *, instance_id=52686067, label="blueprint-native-task-arena-g1-team-one") -> Path:
    from tests.test_vast_official_billing_extractor import _fixture, _charge, _refresh_response_binding
    fixture = _fixture(root)
    response = fixture["responses"][0]
    value = json.loads(response.read_text())
    value["results"] = [_charge(instance_id=instance_id, label=label, total=0.5, gpu=0.4, disk=0.1)]
    _write(response, value)
    _refresh_response_binding(fixture, 0)
    return fixture["receipt"]


def test_selected_g1_reconciliation_extracts_exact_posted_charge_and_reopens_proof(tmp_path):
    path, _ = _selected_case(tmp_path / "run")
    source = _selected_billing_source(tmp_path / "audit")
    output = tmp_path / "reconciliation.json"
    result = billing.materialize_vast_official_same_goal_reconciliation(
        provider_billing_source_receipt_path=source,
        expected_instances=[(52686067, "blueprint-native-task-arena-g1-team-one", path)],
        output_path=output,
    )
    assert result["official_total_usd"] == 0.5
    evidence = result["entries"][0]["terminal_execution_evidence"]
    assert evidence["financial_closeout_kind"] == "native_g1_team_paid_policy.v1"
    assert evidence["policy_evaluation_qualified"] is False
    assert billing.validate_vast_official_same_goal_reconciliation(output) == result
    Path(json.loads(path.read_text())["object_store_cleanup_path"]).write_text("{}")
    with pytest.raises(billing.VastOfficialBillingExtractionError):
        billing.validate_vast_official_same_goal_reconciliation(output)


@pytest.mark.parametrize("fault", ["wrong_native_schema", "public_claim", "native_status", "bundle_kind", "continuing_spend", "attempt_copy", "watchdog", "cleanup"])
def test_selected_financial_closeout_refuses_mismatched_or_live_evidence(tmp_path, fault):
    path, result = _selected_case(tmp_path / "run")
    target = Path(result["native_control_result_path"])
    value = json.loads(target.read_text())
    if fault in {"wrong_native_schema", "public_claim", "native_status"}:
        value[{"wrong_native_schema": "schema_version", "public_claim": "public_redistribution_authorized", "native_status": "status"}[fault]] = {
            "wrong_native_schema": "native_g1_provider_campaign_result.v1", "public_claim": True, "native_status": "blocked",
        }[fault]
        value["result_digest"] = canonical_digest(value, digest_field="result_digest")
        _write(target, value)
        result["native_control_result_digest"] = value["result_digest"]
        _write(path, result)
        _write(Path(result["attempt_root"]) / "adp_arena_vast_result.json", result)
    elif fault == "bundle_kind":
        target = Path(result["adapter_result_path"])
        value = json.loads(target.read_text())
        value["provider_bundle_kind"] = "native_g1_development_campaign"
        _write(target, value)
        result["provider_closeout"]["adapter_result"] = _record(target)
        _write(path, result)
        _write(Path(result["attempt_root"]) / "adp_arena_vast_result.json", result)
    elif fault == "continuing_spend":
        result["continuing_spend_from_this_run"] = True
        _write(path, result)
    elif fault == "attempt_copy":
        _write(Path(result["attempt_root"]) / "adp_arena_vast_result.json", {})
    else:
        target = Path(result["watchdog_receipt_path"] if fault == "watchdog" else result["object_store_cleanup_path"])
        value = json.loads(target.read_text())
        if fault == "watchdog":
            value["final_global_inventory"]["live_resource_count"] = 1
        else:
            value["all_objects_absent"] = False
        _write(target, value)
    with pytest.raises(billing.VastOfficialBillingExtractionError):
        billing._terminal_evidence(instance_id=52686067, terminal_result_path=path)


def test_blocked_g1_campaign_seals_financial_closeout_only(tmp_path: Path) -> None:
    result_path, _ = _case(tmp_path / "run")
    evidence = billing._terminal_evidence(
        instance_id=52686067, terminal_result_path=result_path
    )
    assert evidence["financial_closeout_kind"] == "native_g1_paid_campaign.v1"
    assert evidence["terminal_status"] == "blocked"
    assert evidence["provider_zero_verified"] is True
    assert evidence["policy_evaluation_qualified"] is False
    assert evidence["scientific_success_inferred"] is False
    assert evidence["native_result"]["sha256"] == _record(Path(
        json.loads(result_path.read_text())["native_control_result_path"]
    ))["sha256"]


def test_pre_native_g1_startup_failure_can_reconcile_provider_charge(tmp_path: Path) -> None:
    result_path, _ = _pre_native_case(tmp_path / "run")
    evidence = billing._terminal_evidence(
        instance_id=52686067, terminal_result_path=result_path
    )
    assert evidence["terminal_status"] == "blocked"
    assert evidence["provider_zero_verified"] is True
    assert evidence["policy_evaluation_qualified"] is False
    assert "native_result" not in evidence


@pytest.mark.parametrize("fault", ["claimed_query", "claimed_observation", "completed_artifacts"])
def test_pre_native_g1_billing_rejects_claimed_science_or_false_artifacts(
    tmp_path: Path, fault: str
) -> None:
    result_path, result = _pre_native_case(tmp_path / "run")
    if fault == "completed_artifacts":
        artifact_path = Path(result["artifact_manifest_path"])
        artifact = json.loads(artifact_path.read_text())
        artifact["status"] = "completed"
        artifact["blockers"] = []
        artifact["manifest_digest"] = canonical_digest(artifact, digest_field="manifest_digest")
        _write(artifact_path, artifact)
    else:
        result["candidate_policy_queried" if fault == "claimed_query" else "first_observation_reached"] = True
        _write(result_path, result)
        _write(Path(result["attempt_root"]) / "adp_arena_vast_result.json", result)
    with pytest.raises(billing.VastOfficialBillingExtractionError):
        billing._terminal_evidence(instance_id=52686067, terminal_result_path=result_path)


@pytest.mark.parametrize("fault", ["instance", "watchdog", "cleanup", "native", "attempt_copy"])
def test_g1_financial_closeout_rejects_broken_terminal_evidence(
    tmp_path: Path, fault: str
) -> None:
    result_path, result = _case(tmp_path / "run")
    if fault == "instance":
        path = Path(result["adapter_result_path"])
        value = json.loads(path.read_text())
        value["vast_instance_ids"] = [42]
        _write(path, value)
    elif fault == "watchdog":
        path = Path(result["watchdog_receipt_path"])
        value = json.loads(path.read_text())
        value["final_global_inventory"]["live_resource_count"] = 1
        _write(path, value)
    elif fault == "cleanup":
        path = Path(result["object_store_cleanup_path"])
        value = json.loads(path.read_text())
        value["all_objects_absent"] = False
        _write(path, value)
    elif fault == "native":
        path = Path(result["native_control_result_path"])
        value = json.loads(path.read_text())
        value["candidate_policy_queried"] = True
        _write(path, value)
    else:
        path = Path(result["attempt_root"]) / "adp_arena_vast_result.json"
        value = json.loads(path.read_text())
        value["status"] = "completed"
        _write(path, value)
    with pytest.raises(billing.VastOfficialBillingExtractionError):
        billing._terminal_evidence(instance_id=52686067, terminal_result_path=result_path)
