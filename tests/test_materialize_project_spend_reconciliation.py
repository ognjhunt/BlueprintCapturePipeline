from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from blueprint_pipeline.decision_evidence_contracts import canonical_digest


ROOT = Path(__file__).resolve().parents[1]


def test_cli_materializes_a_no_spend_project_roll_forward(tmp_path: Path) -> None:
    authority: dict[str, object] = {
        "schema_version": "native_task_arena_paid_attempt_authority.v1",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "hard_attempt_spend_cap_usd": 0.75,
        "aggregate_goal_spend_before_attempt_usd": 40.333914,
        "authorized_on": "2026-08-25T14:30:00+00:00",
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    output = tmp_path / "project-spend.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/materialize_project_spend_reconciliation.py"),
            "--baseline-authority",
            str(authority_path),
            "--unposted-authority",
            str(authority_path),
            "--expected-coverage-id",
            str(authority["authorization_digest"]),
            "--completeness-reference",
            "retained queue inventory",
            "--authorized-by",
            "user",
            "--authorized-on",
            "2026-08-25T15:15:00+00:00",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    status = json.loads(completed.stdout)
    assert status["status"] == "materialized"
    assert status["total_cost_usd"] == 41.083914
    assert status["provider_mutation_performed"] is False
    assert output.is_file()


def test_cli_accepts_human_authorized_current_opening_without_fake_attempt(
    tmp_path: Path,
) -> None:
    text = "Adopt $43.197914 as the conservative opening project exposure."
    baseline = {
        "schema_version": "blueprint_project_spend_human_authorization.v1",
        "status": "authorized",
        "program_id": "arm-decision-proof-v1",
        "authorization_text": text,
        "authorization_text_sha256": (
            "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        ),
        "opening_project_exposure_usd": 43.197914,
        "aggregate_project_ceiling_usd": 50.0,
        "authorized_attempt": {
            "count": 1,
            "retry_cap": 0,
            "maximum_spend_usd": 0.75,
            "maximum_hourly_rate_usd": 0.8,
            "hard_ttl_seconds": 9000,
        },
        "maximum_bounded_exposure_after_full_attempt_reserve_usd": 43.947914,
        "minimum_guaranteed_headroom_after_full_attempt_reserve_usd": 6.052086,
        "production_standing_authorization": False,
        "launch_request": False,
        "provider_mutation_performed": False,
    }
    baseline_path = tmp_path / "human-authorization.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    output = tmp_path / "project-spend.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/materialize_project_spend_reconciliation.py"),
            "--baseline-authority",
            str(baseline_path),
            "--completeness-reference",
            str(baseline_path),
            "--authorized-by",
            "user",
            "--authorized-on",
            "2026-08-25T20:41:55Z",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    status = json.loads(completed.stdout)
    assert status["total_cost_usd"] == 43.197914
