"""The native-versus-contract parity report reads two finished runs and states a plain verdict."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.policy_path_parity import compare_runs, main, plain_verdict
from tests.test_beta_policy_screening_coordinator import native_result, plan

CONTRACT = "pi05_droid_contract"


def _runs():
    value = plan()
    native = native_result(value)
    contract = deepcopy(native)
    for row in contract["episodes"]:
        row["candidate_id"] = CONTRACT
        row["episode"]["commanded_actions"] = [[0.1 * index for index in range(8)]]
    for row in native["episodes"]:
        row["episode"]["commanded_actions"] = [[0.1 * index for index in range(8)]]
    return native, contract


def _compare(native, contract):
    return compare_runs(native=native, contract=contract, native_candidate="candidate-000",
                        contract_candidate=CONTRACT)


def test_identical_runs_are_the_same():
    report = _compare(*_runs())
    assert report["verdict"] == "same" and report["tries_differing"] == []
    assert report["outcome_counts"]["native"] == report["outcome_counts"]["contract"] == {
        "success": 7, "failure": 3, "void": 0}
    assert report["report_digest"] == canonical_digest(report, digest_field="report_digest")
    assert plain_verdict(report).startswith("Same: all 10 tries agree")


def test_an_action_trace_difference_alone_keeps_the_outcomes_the_same():
    native, contract = _runs()
    row = contract["episodes"][4]
    row["episode"]["commanded_actions"] = [[0.1 * index + 1e-7 for index in range(8)]]
    row["returned_action_sequence_digest"] = canonical_digest({"value": row["episode"]["commanded_actions"]})
    report = _compare(native, contract)
    assert report["verdict"] == "same_outcomes"
    differing = next(item for item in report["tries"] if item["differences"])
    assert differing["actions"]["identical"] is False and differing["actions"]["max_abs_difference"] == pytest.approx(1e-7)


def test_a_different_outcome_is_named_per_try():
    native, contract = _runs()
    contract["episodes"][0]["episode"]["score"] = {"status": "scored", "task_succeeded": False}
    report = _compare(native, contract)
    assert report["verdict"] == "differs"
    assert report["tries"][0]["differences"][:2] == ["outcome", "reason"]
    assert plain_verdict(report).startswith("Differs: " + report["tries"][0]["cell_id"])


@pytest.mark.parametrize("change", ["scene", "seed", "missing_cell"])
def test_runs_on_another_scene_or_other_tries_are_not_comparable(change):
    native, contract = _runs()
    if change == "scene":
        contract["scene_revision_digest"] = "sha256:" + "9" * 64
    elif change == "seed":
        contract["episodes"][2]["seed"] += 1
    else:
        contract["episodes"].pop()
    assert _compare(native, contract)["verdict"] == "not_comparable"


def test_cli_writes_the_report(tmp_path, capsys):
    native, contract = _runs()
    (tmp_path / "n.json").write_text(json.dumps(native))
    (tmp_path / "c.json").write_text(json.dumps(contract))
    assert main(["--native", str(tmp_path / "n.json"), "--contract", str(tmp_path / "c.json"),
                 "--native-candidate", "candidate-000", "--contract-candidate", CONTRACT,
                 "--output", str(tmp_path / "report.json")]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["verdict"] == "same" and report["plain_verdict"].startswith("Same")
