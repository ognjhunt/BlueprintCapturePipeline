"""The native-versus-contract parity report reads two finished runs and states a plain verdict."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.policy_path_parity import compare_runs, main, plain_verdict, run_identity
from tests.test_beta_policy_screening_coordinator import plan
from tests.test_beta_policy_screening_progress import session_result

CONTRACT = "pi05_droid_contract"
DEPTHS = [7, 7, 7, 7, 7, 7, 7, 3, 2, 1]
COMMIT = "a" * 40


def _identities(**contract_changes):
    native = {"implementation_commit": COMMIT, "screening_harness_revision": 2,
              "runtime_identity_digest": canonical_digest({"value": "runtime"}), "native_runtime_identity_digest": None,
              "checkpoint_digest": canonical_digest({"value": "checkpoint"}), "open_loop_horizon": 8,
              "session_served": False}
    contract = {**native, "runtime_identity_digest": canonical_digest({"value": "served"}),
                "native_runtime_identity_digest": native["runtime_identity_digest"], "session_served": True,
                **contract_changes}
    return native, contract


def _runs():
    """Two graded runs (real scores and sealed traces), each try's settings and action trace recorded."""
    value = plan()
    native = session_result(value, value["candidates"][0], DEPTHS)
    native_identity, contract_identity = _identities()
    for row in native["episodes"]:
        row["runtime_identity_digest"] = native_identity["runtime_identity_digest"]
        row["episode"].update(commanded_actions=[[0.1 * index for index in range(8)]], max_policy_queries=45,
                              open_loop_horizon=8)
    contract = deepcopy(native)
    for row in contract["episodes"]:
        row["candidate_id"] = CONTRACT
        row["runtime_identity_digest"] = contract_identity["runtime_identity_digest"]
    return native, contract


def _compare(native, contract, identities=None):
    native_identity, contract_identity = identities or _identities()
    return compare_runs(native=native, contract=contract, native_candidate="candidate-000",
                        contract_candidate=CONTRACT, native_identity=native_identity,
                        contract_identity=contract_identity)


def test_identical_runs_are_the_same():
    report = _compare(*_runs())
    assert report["verdict"] == "same" and report["tries_differing"] == []
    assert report["outcome_counts"]["native"] == report["outcome_counts"]["contract"]
    assert all("not_graded" not in row["native"] and row["actions"]["compared"] for row in report["tries"])
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
    from tests.test_beta_policy_screening_progress import episode

    # The contract's first try got no further than the start: a real, sealed, graded failure.
    row = contract["episodes"][0]
    score, samples = episode(0)
    row["episode"].update(score=score, state_trace={"task_state_samples": samples})
    row["deterministic_score_digest"] = canonical_digest({"value": score})
    row["state_trace_digest"] = canonical_digest({"value": row["episode"]["state_trace"]})
    report = _compare(native, contract)
    assert report["verdict"] == "differs", report["not_comparable_reasons"]
    assert "outcome" in report["tries"][0]["differences"] and "deepest_stage" in report["tries"][0]["differences"]
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


def test_cli_without_bound_plans_is_not_comparable(tmp_path, capsys):
    native, contract = _runs()
    (tmp_path / "n.json").write_text(json.dumps(native))
    (tmp_path / "c.json").write_text(json.dumps(contract))
    (tmp_path / "plan.json").write_text(json.dumps(plan()))  # its policy spec path does not exist here
    assert main(["--native", str(tmp_path / "n.json"), "--contract", str(tmp_path / "c.json"),
                 "--native-plan", str(tmp_path / "plan.json"), "--contract-plan", str(tmp_path / "plan.json"),
                 "--native-candidate", "candidate-000", "--contract-candidate", CONTRACT,
                 "--output", str(tmp_path / "report.json")]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["verdict"] == "not_comparable" and "run_identity_missing" in report["not_comparable_reasons"]


@pytest.mark.parametrize("strip", ["actions", "grading", "both"])
def test_nothing_compared_is_never_same(strip):
    """Review of aefc7d09a, finding 3: missing actions or traces on both sides read as 'same'."""
    native, contract = _runs()
    for row in native["episodes"] + contract["episodes"]:
        if strip in {"actions", "both"}:
            row.pop("returned_action_sequence_digest", None)
        if strip in {"grading", "both"}:
            row["episode"].pop("state_trace", None)
    report = _compare(native, contract)
    assert report["verdict"] == "not_comparable"
    missing = report["not_comparable_reasons"]
    assert any(code.startswith("action_trace_missing:") for code in missing) == (strip != "grading")
    assert any(code.startswith("not_graded:") for code in missing) == (strip != "actions")
    assert plain_verdict(report).startswith("Not comparable: ")


@pytest.mark.parametrize(("identities", "reason"), [
    (_identities(implementation_commit="b" * 40), "implementation_commit_differs"),
    (_identities(screening_harness_revision=1), "screening_harness_revision_differs"),
    (_identities(checkpoint_digest=None), "checkpoint_digest_differs"),
    (_identities(open_loop_horizon=4), "open_loop_horizon_differs"),
    (_identities(native_runtime_identity_digest="sha256:" + "0" * 64), "runtime_identity_differs"),
    (_identities(session_served=False), "contract_run_was_not_served_over_the_contract"),
    ((_identities()[0] | {"session_served": True}, _identities()[1]), "native_run_was_served_over_the_contract"),
    ((None, _identities()[1]), "run_identity_missing"),
    (({**_identities()[0], "checkpoint_digest": None}, {**_identities()[1], "checkpoint_digest": None}),
     "checkpoint_digest_missing"),
])
def test_another_measurement_is_not_comparable(identities, reason):
    report = _compare(*_runs(), identities=identities)
    assert report["verdict"] == "not_comparable" and reason in report["not_comparable_reasons"]


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("max_policy_queries", 30, "query_limit_differs"), ("max_policy_queries", None, "query_limit_missing"),
    ("open_loop_horizon", 4, "horizon_differs")])
def test_a_try_run_under_other_settings_is_not_comparable(field, value, reason):
    native, contract = _runs()
    if value is None:
        contract["episodes"][3]["episode"].pop(field)
    else:
        contract["episodes"][3]["episode"][field] = value
    report = _compare(native, contract)
    assert report["verdict"] == "not_comparable"
    assert any(code.startswith(reason + ":") for code in report["not_comparable_reasons"])


def test_a_try_bound_to_another_runtime_is_not_comparable():
    native, contract = _runs()
    contract["episodes"][0]["runtime_identity_digest"] = native["episodes"][0]["runtime_identity_digest"]
    report = _compare(native, contract)
    assert any(code.startswith("episode_runtime_not_its_run:") for code in report["not_comparable_reasons"])


def test_run_identity_reads_the_bound_policy_spec(tmp_path):
    import hashlib

    from blueprint_pipeline.session_served_policy import SPEC_FIELD

    value = deepcopy(plan())
    spec = {"runtime_identity_digest": "sha256:" + "1" * 64, "checkpoint_digest": "sha256:" + "2" * 64,
            "open_loop_horizon": 8}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    value["candidates"][0]["execution_spec"]["policy_execution_spec"] = {
        "path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": 1}
    identity = run_identity(value, "candidate-000")
    assert identity["session_served"] is False and identity["checkpoint_digest"] == spec["checkpoint_digest"]
    assert identity["implementation_commit"] == COMMIT and identity["screening_harness_revision"] == 2
    path.write_text(json.dumps({**spec, SPEC_FIELD: {}}))
    with pytest.raises(ValueError, match="policy_spec_unbound"):
        run_identity(value, "candidate-000")
