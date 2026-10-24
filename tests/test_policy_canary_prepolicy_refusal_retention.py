# Covers: src/blueprint_pipeline/team_evaluation_worker_retention.py
"""The exact native camera refusal remains blocked and survives parent sealing."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_evaluation_worker_retention import _validate_isolated_cell_result

FIXTURE = Path(__file__).parent / "fixtures/droid_constructor_replay/retained_blocked_gate_child.json"


def _child():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == "74b1953b449173a164fd5b3fa027a637b7bd63201e5f2e900705cb50bf450088"
    return json.loads(FIXTURE.read_text())


def _inputs(child):
    return {key: child[key] for key in ("candidate_ids", "task_success_contract", "task_success_contract_digest")}


def test_exact_archived_gate_refusal_is_a_prepolicy_failure():
    child = _child()
    assert _validate_isolated_cell_result(child=child, inputs=_inputs(child), index=0) is False
    assert child["preload_observation_gate"]["blockers"] == [
        "external:task_object:composition_support_background_occlusion_requires_review"]
    assert child["candidate_policy_queried"] is False


@pytest.mark.parametrize("mutation", [
    "gate_digest", "gate_passed", "gate_queried", "gate_loaded", "no_blockers",
    "gate_schema", "failure_type", "child_queried", "policy_loaded",
])
def test_unproven_or_inconsistent_gate_refusal_is_rejected(mutation):
    original = _child()
    child = deepcopy(original)
    gate = child["preload_observation_gate"]
    if mutation == "gate_passed":
        gate.update(status="passed", policy_observation_integrity_passed=True)
    elif mutation == "gate_queried":
        gate["candidate_policy_queried"] = True
    elif mutation == "gate_loaded":
        gate["candidate_policy_loaded"] = True
    elif mutation == "no_blockers":
        gate["blockers"] = []
    elif mutation == "gate_schema":
        gate["schema_version"] = "unknown"
    elif mutation == "failure_type":
        child["session_failure_type"] = "RuntimeError"
    elif mutation == "child_queried":
        child["candidate_policy_queried"] = True
    elif mutation == "policy_loaded":
        child["policy_loads"] = [{"candidate_id": "pi05_droid", "loaded_once": True}]
    gate["gate_digest"] = ("sha256:" + "0" * 64 if mutation == "gate_digest"
                           else canonical_digest(gate, digest_field="gate_digest"))
    child["result_digest"] = canonical_digest(child, digest_field="result_digest")
    with pytest.raises(RuntimeError, match="isolated_cell_result_invalid"):
        _validate_isolated_cell_result(child=child, inputs=_inputs(original), index=0)
