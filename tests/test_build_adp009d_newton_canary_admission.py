from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_physics_backend_comparison import (
    build_backend_profile,
    validate_newton_canary_admission,
)
from blueprint_pipeline.spend_admission_lock import build_spend_admission_lock
from scripts import build_adp009d_newton_canary_admission as builder


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    now = datetime.now(timezone.utc)
    inventory = [
        {
            "provider": provider,
            "status": "succeeded",
            "required": True,
            "credential_present": True,
            "row_count": 0,
            "blockers": [],
        }
        for provider in ("runpod", "vast", "digitalocean")
    ]
    spend_lock = build_spend_admission_lock(
        fleet_budget={"status": "passed", "total_spend_usd": 0.0, "blockers": []},
        billing_reconciliation={
            "status": "reconciled",
            "required": True,
            "billing_export_schema_version": "blueprint.provider_billing_export.v1",
            "billing_export_sha256": "sha256:" + "a" * 64,
            "billing_export_mode_octal": "0600",
            "generated_at": now.isoformat(),
            "currency": "USD",
            "scope": "blueprint_beta_100_user_cohort",
            "provider_totals_usd": {
                "runpod": 1.0,
                "vast": 1.0,
                "digitalocean": 1.0,
            },
            "blockers": [],
        },
        instances=[],
        reap_results=[],
        inventory_results=inventory,
        override_path=None,
        now=now,
    )
    guard = {
        "schema_version": "gpu_spend_guard.v1",
        "status": "passed",
        "generated_at": now.isoformat(),
        "provider_zero_verified": True,
        "live_instance_count": 0,
        "total_burn_per_hour_usd": 0.0,
        "blockers": [],
        "inventory_results": inventory,
        "instances": [],
    }
    spend_path = tmp_path / "spend-lock.json"
    guard_path = tmp_path / "provider-guard.json"
    spend_path.write_text(json.dumps(spend_lock), encoding="utf-8")
    guard_path.write_text(json.dumps(guard), encoding="utf-8")
    return spend_path, guard_path


def test_cli_materializes_validator_clean_no_mutation_admission(tmp_path: Path) -> None:
    spend_path, guard_path = _inputs(tmp_path)
    output = tmp_path / "newton-admission.json"

    exit_code = builder.main(
        [
            "--authorization-evidence-ref",
            "goal:scene-840920-newton-controls",
            "--spend-admission-lock",
            str(spend_path),
            "--provider-guard",
            str(guard_path),
            "--max-spend-usd",
            "2.0",
            "--hard-ttl-seconds",
            "5400",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    admission = json.loads(output.read_text(encoding="utf-8"))
    assert (
        validate_newton_canary_admission(admission, profile=build_backend_profile("newton")) == []
    )
    assert admission["controls_only"] is True
    assert admission["policy_query_allowed"] is False
    assert admission["candidate_outcome_access_allowed"] is False
    assert admission["max_spend_usd"] == 2.0
    assert admission["hard_ttl_seconds"] == 5400
    assert admission["provider_mutation_performed"] is False


def test_script_entrypoint_executes_the_same_no_mutation_builder(tmp_path: Path) -> None:
    spend_path, guard_path = _inputs(tmp_path)
    output = tmp_path / "newton-admission-subprocess.json"
    repo = Path(__file__).resolve().parents[1]

    completed = subprocess.run(  # nosec B603 - fixed interpreter and local script
        [
            sys.executable,
            str(repo / "scripts" / "build_adp009d_newton_canary_admission.py"),
            "--authorization-evidence-ref",
            "goal:scene-840920-newton-controls",
            "--spend-admission-lock",
            str(spend_path),
            "--provider-guard",
            str(guard_path),
            "--max-spend-usd",
            "2.0",
            "--hard-ttl-seconds",
            "5400",
            "--output",
            str(output),
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    admission = json.loads(output.read_text(encoding="utf-8"))
    assert admission["status"] == "passed"
    assert admission["provider_mutation_performed"] is False
    assert admission["policy_query_allowed"] is False


def test_cli_fails_closed_above_two_dollars_or_with_nonzero_provider(
    tmp_path: Path,
) -> None:
    spend_path, guard_path = _inputs(tmp_path)
    with pytest.raises(
        builder.ProductionProfileBuildError,
        match="spend_cap_invalid",
    ):
        builder.materialize_newton_canary_admission(
            authorization_evidence_ref="goal:scene-840920-newton-controls",
            spend_admission_lock_path=spend_path,
            provider_guard_path=guard_path,
            max_spend_usd=2.01,
            hard_ttl_seconds=5400,
            output_path=tmp_path / "over-cap.json",
        )

    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    guard["live_instance_count"] = 1
    guard["inventory_results"][1]["row_count"] = 1
    guard["instances"] = [{"provider": "vast", "id": "47", "live": True}]
    guard_path.write_text(json.dumps(guard), encoding="utf-8")
    with pytest.raises(
        builder.PhysicsBackendContractError,
        match="provider_inventory_precheck_invalid",
    ):
        builder.materialize_newton_canary_admission(
            authorization_evidence_ref="goal:scene-840920-newton-controls",
            spend_admission_lock_path=spend_path,
            provider_guard_path=guard_path,
            max_spend_usd=2.0,
            hard_ttl_seconds=5400,
            output_path=tmp_path / "nonzero.json",
        )

    assert not (tmp_path / "over-cap.json").exists()
    assert not (tmp_path / "nonzero.json").exists()
