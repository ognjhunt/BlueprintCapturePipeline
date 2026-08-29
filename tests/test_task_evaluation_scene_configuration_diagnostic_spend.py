from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.project_spend_reconciliation import (
    materialize_project_spend_reconciliation,
    validate_project_spend_reconciliation,
)
from blueprint_pipeline.same_goal_spend_reconciliation import (
    materialize_same_goal_spend_reconciliation,
)
from blueprint_pipeline import task_evaluation_scene_configuration_diagnostic_spend as spend


def _write(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path.resolve()


def _digest(value: dict[str, object], field: str) -> dict[str, object]:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _sources(tmp_path: Path) -> dict[str, Path]:
    bundle = _write(tmp_path / "bundle.json", {"diagnostic_only": True})
    authority = _digest(
        {
            "schema_version": "task_evaluation_scene_configuration_paid_authority.v1",
            "run_id": "scene-839873-configuration-v1",
            "source_commit": "a" * 40,
            "bundle_sha256": "sha256:" + "b" * 64,
            "bundle_receipt": {
                "path": str(bundle),
                "size_bytes": bundle.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest(),
            },
            "resource_name": "blueprint-scene-diagnostic-resume-fixture",
            "diagnostic_only": True,
            "qualification_eligible": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "retry_cap": 0,
            "authority_digest": "",
        },
        "authority_digest",
    )
    authority_path = _write(tmp_path / "authority.json", authority)
    adapter = {
        "schema_version": "vast_provider_adapter_result.v1",
        "status": "blocked",
        "provider_bundle_kind": "task_evaluation_scene_configuration",
        "provider_create_attempted": True,
        "vast_instance_ids": [41234567],
        "estimated_cost_usd": 0.147,
        "continuing_spend_from_this_run": False,
        "final_validation_status": "passed",
        "retained_owned": False,
        "raw_api_key_stored": False,
        "secret_values_in_artifact": False,
        "raw_secret_values_recorded": False,
    }
    adapter_path = _write(tmp_path / "provider" / "adapter.json", adapter)
    teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "generated_at": "2026-08-28T12:05:00Z",
        "status": "completed",
        "vast_instance_ids": [41234567],
        "continuing_spend_from_this_run": False,
        "runner_gpu_teardown_completed": True,
        "retention_authorized": False,
        "raw_secret_values_recorded": False,
    }
    teardown_path = _write(tmp_path / "provider" / "teardown.json", teardown)
    authority_digest = authority["authority_digest"]
    result = _digest(
        {
            "schema_version": (
                "task_evaluation_scene_configuration_diagnostic_vast_result.v1"
            ),
            "status": "blocked_diagnostic_only",
            "run_id": authority["run_id"],
            "source_commit": authority["source_commit"],
            "bundle_sha256": authority["bundle_sha256"],
            "authority_digest": authority_digest,
            "authorization_consumption": {
                "status": "consumed",
                "authorization_digest": authority_digest,
            },
            "provider_adapter_result_path": str(adapter_path),
            "teardown_manifest_path": str(teardown_path),
            "provider_mutations_performed": 1,
            "retry_cap": 0,
            "independent_watchdog": {
                "status": "provider_terminal",
                "instance_ids": [41234567],
                "provider_absence_confirmed": True,
                "raw_secret_values_recorded": False,
            },
            "continuing_spend_from_this_run": False,
            "raw_secret_values_recorded": False,
            "diagnostic_only": True,
            "qualification_eligible": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "result_digest": "",
        },
        "result_digest",
    )
    result_path = _write(tmp_path / "job" / "terminal.json", result)
    zero = _digest(
        {
            "schema_version": "adp_paid_provider_zero.v1",
            "provider": "vast",
            "observed_at_utc": "2026-08-28T12:06:00Z",
            "api_command": [
                "blueprint_pipeline.gpu_render_providers.VastRenderProvider.billable_inventory",
                "name_prefix=",
            ],
            "api_confirmed": True,
            "global_live_resource_count": 0,
            "provider_zero": True,
            "inventory": [],
            "stderr_present": False,
            "raw_secret_values_recorded": False,
            "provider_zero_digest": "",
        },
        "provider_zero_digest",
    )
    zero_path = _write(tmp_path / "provider-zero.json", zero)
    billing = {
        "results": [
            {
                "source": "instance-41234567",
                "amount": 0.151,
                "items": [
                    {
                        "type": "gpu",
                        "description": "0.25 hours at $0.604/hour",
                        "amount": 0.151,
                    }
                ],
            }
        ]
    }
    billing_path = _write(tmp_path / "billing.json", billing)
    billing_source = _digest(
        {
            "schema_version": "blueprint.provider_billing_source_receipt.v1",
            "status": "reconciled",
            "sources": [
                {
                    "provider": "vast",
                    "retained_path": str(billing_path),
                    "response_digest": (
                        "sha256:" + hashlib.sha256(billing_path.read_bytes()).hexdigest()
                    ),
                    "response_size_bytes": billing_path.stat().st_size,
                }
            ],
            "receipt_digest": "",
        },
        "receipt_digest",
    )
    billing_source_path = _write(tmp_path / "billing-source.json", billing_source)
    return {
        "authority": authority_path,
        "result": result_path,
        "adapter": adapter_path,
        "teardown": teardown_path,
        "zero": zero_path,
        "billing": billing_path,
        "billing_source": billing_source_path,
    }


def _patch_authority_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        spend,
        "load_scene_configuration_provider_bundle_receipt",
        lambda *_args, **_kwargs: {"diagnostic_only": True},
    )
    def validate_historical(value: dict, **kwargs: object) -> dict:
        assert kwargs.get("historical_terminal_evidence") is True
        return value

    monkeypatch.setattr(
        spend,
        "validate_scene_configuration_paid_authority",
        validate_historical,
    )


def _rewrite_adapter(path: Path, **updates: object) -> None:
    adapter = json.loads(path.read_text(encoding="utf-8"))
    for key, value in updates.items():
        if value is _MISSING:
            adapter.pop(key, None)
        else:
            adapter[key] = value
    path.write_text(json.dumps(adapter), encoding="utf-8")


_MISSING = object()


def test_direct_diagnostic_enters_authoritative_project_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_authority_validation(monkeypatch)
    source = _sources(tmp_path)
    terminal_path = tmp_path / "diagnostic-terminal-evidence.json"
    terminal = spend.materialize_scene_configuration_diagnostic_terminal_evidence(
        attempt_authority_path=source["authority"],
        terminal_result_path=source["result"],
        provider_adapter_result_path=source["adapter"],
        teardown_manifest_path=source["teardown"],
        post_teardown_provider_zero_path=source["zero"],
        output_path=terminal_path,
    )
    assert terminal["attempt_id"] == "blueprint-scene-diagnostic-resume-fixture"
    assert terminal["provider_zero_scope"] == "global_vast_billable_inventory"
    assert terminal["raw_secret_values_recorded"] is False

    same_goal_path = tmp_path / "diagnostic-same-goal.json"
    same_goal = materialize_same_goal_spend_reconciliation(
        lane="task_evaluation_scene_configuration_diagnostic",
        terminal_result_paths=[terminal_path],
        teardown_manifest_paths=[source["teardown"]],
        provider_zero_paths=[source["zero"]],
        official_billing_response_paths=[source["billing"]],
        provider_billing_source_receipt_paths=[source["billing_source"]],
        output_path=same_goal_path,
    )
    assert same_goal["entries"][0]["cost_usd"] == 0.151
    assert same_goal["entries"][0]["authority_digest"] == terminal["authority_digest"]

    authorization_text = "Authorize the retained diagnostic charge as project spend."
    baseline = {
        "schema_version": "blueprint_project_spend_human_authorization.v1",
        "status": "authorized",
        "program_id": "arm-decision-proof-v1",
        "authorization_text": authorization_text,
        "authorization_text_sha256": (
            "sha256:" + hashlib.sha256(authorization_text.encode()).hexdigest()
        ),
        "opening_project_exposure_usd": 40.0,
        "aggregate_project_ceiling_usd": 100.0,
        "authorized_attempt": {
            "count": 1,
            "retry_cap": 0,
            "maximum_spend_usd": 1.0,
            "maximum_hourly_rate_usd": 0.8,
            "hard_ttl_seconds": 3600,
        },
        "maximum_bounded_exposure_after_full_attempt_reserve_usd": 41.0,
        "minimum_guaranteed_headroom_after_full_attempt_reserve_usd": 59.0,
        "production_standing_authorization": False,
        "launch_request": False,
        "provider_mutation_performed": False,
    }
    baseline_path = _write(tmp_path / "baseline.json", baseline)
    project_path = tmp_path / "project-spend.json"
    project = materialize_project_spend_reconciliation(
        baseline_authority_path=baseline_path,
        posted_reconciliation_paths=[same_goal_path],
        expected_coverage_ids=[terminal["attempt_id"]],
        completeness_reference="retained direct diagnostic inventory fixture",
        authorized_by="fixture-owner",
        authorized_on="2026-08-28T12:10:00Z",
        output_path=project_path,
    )
    assert project["total_cost_usd"] == 40.151
    validate_project_spend_reconciliation(project_path)


def test_diagnostic_terminal_evidence_reopens_exact_attempt_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_authority_validation(monkeypatch)
    source = _sources(tmp_path)
    terminal_path = tmp_path / "diagnostic-terminal-evidence.json"
    spend.materialize_scene_configuration_diagnostic_terminal_evidence(
        attempt_authority_path=source["authority"],
        terminal_result_path=source["result"],
        provider_adapter_result_path=source["adapter"],
        teardown_manifest_path=source["teardown"],
        post_teardown_provider_zero_path=source["zero"],
        output_path=terminal_path,
    )
    adapter = json.loads(source["adapter"].read_text(encoding="utf-8"))
    adapter["estimated_cost_usd"] = 0.001
    source["adapter"].write_text(json.dumps(adapter), encoding="utf-8")

    with pytest.raises(
        spend.SceneConfigurationDiagnosticSpendError,
        match="scene_configuration_diagnostic_spend_adapter_record_invalid",
    ):
        spend.validate_scene_configuration_diagnostic_terminal_evidence(terminal_path)


@pytest.mark.parametrize(
    "aggregate_value",
    [False, _MISSING],
    ids=["current-explicit-false", "legacy-v1-omission"],
)
def test_adapter_secret_proof_accepts_current_and_legacy_v1_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregate_value: object,
) -> None:
    _patch_authority_validation(monkeypatch)
    source = _sources(tmp_path)
    _rewrite_adapter(
        source["adapter"], raw_secret_values_recorded=aggregate_value
    )

    terminal = spend.materialize_scene_configuration_diagnostic_terminal_evidence(
        attempt_authority_path=source["authority"],
        terminal_result_path=source["result"],
        provider_adapter_result_path=source["adapter"],
        teardown_manifest_path=source["teardown"],
        post_teardown_provider_zero_path=source["zero"],
        output_path=tmp_path / "terminal.json",
    )

    assert terminal["raw_secret_values_recorded"] is False


@pytest.mark.parametrize(
    ("updates", "case"),
    [
        ({"raw_secret_values_recorded": None}, "explicit-null"),
        ({"raw_secret_values_recorded": True}, "explicit-true"),
        (
            {
                "raw_secret_values_recorded": _MISSING,
                "raw_api_key_stored": True,
            },
            "legacy-raw-key-not-proven",
        ),
        (
            {
                "raw_secret_values_recorded": _MISSING,
                "secret_values_in_artifact": True,
            },
            "legacy-artifact-secret-not-proven",
        ),
    ],
)
def test_adapter_secret_proof_rejects_contradictory_or_unproven_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, object],
    case: str,
) -> None:
    _patch_authority_validation(monkeypatch)
    source = _sources(tmp_path)
    _rewrite_adapter(source["adapter"], **updates)

    with pytest.raises(
        spend.SceneConfigurationDiagnosticSpendError,
        match="scene_configuration_diagnostic_spend_terminal_binding_invalid",
    ):
        spend.materialize_scene_configuration_diagnostic_terminal_evidence(
            attempt_authority_path=source["authority"],
            terminal_result_path=source["result"],
            provider_adapter_result_path=source["adapter"],
            teardown_manifest_path=source["teardown"],
            post_teardown_provider_zero_path=source["zero"],
            output_path=tmp_path / f"terminal-{case}.json",
        )


def test_diagnostic_terminal_evidence_rejects_non_vast_global_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_authority_validation(monkeypatch)
    source = _sources(tmp_path)
    zero = json.loads(source["zero"].read_text(encoding="utf-8"))
    zero["provider"] = "digitalocean"
    zero["provider_zero_digest"] = canonical_digest(
        zero, digest_field="provider_zero_digest"
    )
    source["zero"].write_text(json.dumps(zero), encoding="utf-8")

    with pytest.raises(
        spend.SceneConfigurationDiagnosticSpendError,
        match="scene_configuration_diagnostic_spend_terminal_binding_invalid",
    ):
        spend.materialize_scene_configuration_diagnostic_terminal_evidence(
            attempt_authority_path=source["authority"],
            terminal_result_path=source["result"],
            provider_adapter_result_path=source["adapter"],
            teardown_manifest_path=source["teardown"],
            post_teardown_provider_zero_path=source["zero"],
            output_path=tmp_path / "terminal.json",
        )


def test_diagnostic_terminal_evidence_rejects_secret_shaped_source_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_authority_validation(monkeypatch)
    source = _sources(tmp_path)
    authority = json.loads(source["authority"].read_text(encoding="utf-8"))
    authority["openai_api_key"] = "fixture-secret-material"
    authority["authority_digest"] = canonical_digest(
        authority, digest_field="authority_digest"
    )
    source["authority"].write_text(json.dumps(authority), encoding="utf-8")
    result = json.loads(source["result"].read_text(encoding="utf-8"))
    result["authority_digest"] = authority["authority_digest"]
    result["authorization_consumption"]["authorization_digest"] = authority[
        "authority_digest"
    ]
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    source["result"].write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(
        spend.SceneConfigurationDiagnosticSpendError,
        match="scene_configuration_diagnostic_spend_terminal_binding_invalid",
    ):
        spend.materialize_scene_configuration_diagnostic_terminal_evidence(
            attempt_authority_path=source["authority"],
            terminal_result_path=source["result"],
            provider_adapter_result_path=source["adapter"],
            teardown_manifest_path=source["teardown"],
            post_teardown_provider_zero_path=source["zero"],
            output_path=tmp_path / "terminal.json",
        )
