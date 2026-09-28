"""Repository digest renewal does not mint closure or extend its policy window."""
from __future__ import annotations

import copy
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("rebind", ROOT / "scripts/rebind_quality_gap_ledger_digests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path, monkeypatch):
    module = _module()
    source = tmp_path / "source.py"
    source.write_text("changed bytes")
    ledger = json.loads((ROOT / "docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json").read_text())
    ledger["gaps"] = ledger["gaps"][:1]
    criterion = ledger["gaps"][0]["criteria"][0]
    criterion["evidence_artifacts"] = [criterion["evidence_artifacts"][0]]
    criterion["evidence_artifacts"][0]["path"] = "source.py"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_baseline_criteria", lambda: {})
    monkeypatch.setattr(module, "EVIDENCE_OVERRIDES", {})
    monkeypatch.setattr(module, "EVIDENCE_EXTENSIONS", {})
    return module, ledger, source


def test_renewal_rehashes_actual_bytes_preserving_window_and_authority(tmp_path, monkeypatch):
    module, ledger, source = _fixture(tmp_path, monkeypatch)
    prior = copy.deepcopy(ledger)
    evaluated = datetime.now(timezone.utc) - timedelta(seconds=1)
    module.rebind(ledger, reevaluate_at=evaluated)
    policy = ledger["freshness_policy"]
    old = prior["freshness_policy"]
    assert datetime.fromisoformat(policy["fresh_until"]) - evaluated == (
        datetime.fromisoformat(old["fresh_until"]) - datetime.fromisoformat(old["evaluated_at"]))
    criterion = ledger["gaps"][0]["criteria"][0]
    artifact = criterion["evidence_artifacts"][0]
    assert artifact["sha256"] == module._sha256(source)
    assert artifact["freshness_evaluated_at"] == policy["evaluated_at"] == criterion["freshness"]["evaluated_at"]
    assert artifact["fresh_until"] == policy["fresh_until"] == criterion["freshness"]["fresh_until"]
    assert artifact["supports_closure"] is False
    assert artifact["commit"] is artifact["release_id"] is None
    assert criterion["command_result"] == prior["gaps"][0]["criteria"][0]["command_result"]
    assert ledger["closure_authority_policy"] == prior["closure_authority_policy"]
    assert criterion["derived_status"] in {"partial", "open"}


@pytest.mark.parametrize("field,value", [("commit", "a" * 40), ("supports_closure", True), ("release_id", "release")])
def test_cannot_renew_closure_evidence(tmp_path, monkeypatch, field, value):
    module, ledger, _ = _fixture(tmp_path, monkeypatch)
    ledger["gaps"][0]["criteria"][0]["evidence_artifacts"][0][field] = value
    with pytest.raises(ValueError, match="cannot_renew_closure"):
        module.rebind(ledger, reevaluate_at=datetime.now(timezone.utc))
