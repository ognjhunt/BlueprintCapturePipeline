from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.beta_policy_screening_status import build_worker_status, write_worker_status
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _ledger(root: Path, campaigns: dict) -> None:
    root.mkdir(exist_ok=True)
    (root / "attempts").mkdir(exist_ok=True)
    ledger = {"schema_version": "blueprint.beta_policy_screening_ledger.v1", "campaigns": campaigns}
    ledger["ledger_digest"] = canonical_digest(ledger, digest_field="ledger_digest")
    (root / "screening_ledger.v1.json").write_text(json.dumps(ledger))


def test_idle_is_healthy_and_ambiguous_work_remains_reconciliation(tmp_path: Path) -> None:
    root = tmp_path / "screening"
    _ledger(root, {})
    assert build_worker_status(root, now=1000)["status"] == "idle"
    _ledger(root, {"private-id": {"site_id": "private-site", "revision": 2, "published_revision": 1,
        "created_at": "1970-01-01T00:01:40+00:00", "updated_at": "1970-01-01T00:05:00+00:00",
        "candidates": {"candidate": {"status": "reconciliation_required", "blockers": ["provider_outcome_unknown"]}}}})
    status = build_worker_status(root, publication_after={"last_failure_category": "ack", "next_retry_at_epoch": 1060}, now=1000)
    assert status["candidate_counts"]["reconciliation_required"] == 1
    assert status["publication_pending"] == 1 and status["oldest_eligible_wait_seconds"] == 900
    assert status["last_publication_failure"]["category"] == "ack"
    assert status["performs_live_actions"] is False
    assert "private-id" not in json.dumps(status) and "private-site" not in json.dumps(status)


def test_invalid_ledger_never_becomes_idle(tmp_path: Path) -> None:
    _ledger(tmp_path, {})
    path = tmp_path / "screening_ledger.v1.json"
    data = json.loads(path.read_text())
    data["ledger_digest"] = "wrong"
    path.write_text(json.dumps(data))
    assert build_worker_status(tmp_path)["reason"] == "screening_ledger_invalid"


def test_retained_failure_cannot_inject_remote_text_into_safe_snapshot(tmp_path: Path) -> None:
    _ledger(tmp_path, {})
    (tmp_path / "worker_status.json").write_text(json.dumps({
        "schema_version": "blueprint.beta_policy_screening_worker_status.v1",
        "last_publication_failure": {"category": "auth", "owner": "private body", "next_action": "private body",
                                     "observed_at_epoch": 50, "next_retry_at_epoch": 100},
    }))
    result = write_worker_status(tmp_path, publication_after={"acknowledged": 1, "raw_remote_body": "private body"})
    assert result["last_successful_publication_at_epoch"]
    assert "private body" not in json.dumps(result)
    assert json.loads((tmp_path / "worker_status.json").read_text()) == result
