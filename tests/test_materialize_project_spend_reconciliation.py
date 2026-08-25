from __future__ import annotations

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
