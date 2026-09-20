"""Credit recovery waits for funding and preserves terminal provenance and caps."""
import copy
import json
from pathlib import Path
import time

import pytest

from blueprint_pipeline import task_evaluation_scene_capacity_recovery as recovery
from blueprint_pipeline import provider_credit_admission as credit
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_task_evaluation_scene_capacity_recovery import capacity_fixture, write


def credit_fixture(tmp_path, monkeypatch):
    intent, first, observed, config, evidence = capacity_fixture(tmp_path, monkeypatch)
    refs, values = observed["records"], observed["values"]
    result = {**values["result"], "blockers": ["provider_credit_insufficient"],
              "api_pretraining": None, "provider_runtime_output_zip_path": None}
    write(Path(refs["result"]["path"]), result, "result_digest")
    run = Path(refs["launch"]["path"]).parent
    teardown = write(run / "teardown.json", {"schema_version": "vast_teardown_manifest.v1",
        "status": "not_required_prelaunch_inventory_guard_blocked", "vast_instance_ids": [],
        "continuing_spend_from_this_run": False})
    launch = values["launch"]
    launch["terminal_evidence"] = {"artifacts": {"teardown_manifest_path": {
        "path": teardown["path"], "exists": True, "digest": teardown["sha256"]}}}
    write(Path(refs["launch"]["path"]), launch, "receipt_digest")
    zero = {**values["zero"], "receipt_digest": launch["receipt_digest"]}
    write(Path(refs["zero"]["path"]), zero, "provider_zero_receipt_digest")
    return intent, first, config, refs


def observe(first, config, refs):
    return recovery.observe_failure(attempt=first, link_path=Path(refs["link"]["path"]),
        preparation_path=Path(refs["preparation"]["path"]), factory_path=Path(refs["factory"]["path"]), config=config)


def test_credit_refusal_reopens_bound_teardown_and_rejects_tampering(tmp_path, monkeypatch):
    _, first, config, refs = credit_fixture(tmp_path, monkeypatch)
    observed = observe(first, config, refs)
    assert observed["recoverable"] and observed["kind"] == recovery.CREDIT_KIND
    launch = json.loads(Path(refs["launch"]["path"]).read_text())
    Path(launch["terminal_evidence"]["artifacts"]["teardown_manifest_path"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="credit_teardown_changed"):
        observe(first, config, refs)


@pytest.mark.parametrize("change", [{"provider_mutations_performed": 1}, {"provider_mutations_performed": False},
    {"api_pretraining": {"status": "completed"}}, {"blockers": ["provider_credit_insufficient", "execution_failed"]},
    {"continuing_spend_from_this_run": True}, {"allocation_created": True}])
def test_paid_or_ambiguous_failures_are_not_credit_retries(change):
    result = {"schema_version": "task_evaluation_scene_configuration_vast_result.v1", "status": "blocked",
              "blockers": ["provider_credit_insufficient"], "provider_mutations_performed": 0,
              "continuing_spend_from_this_run": False, "api_pretraining": None}
    assert recovery.credit_launch_failure(result)
    assert not recovery.credit_launch_failure({**result, **change})


def test_credit_waits_for_fresh_funding_without_reducing_compute_allowance(tmp_path, monkeypatch):
    _, first, config, refs = credit_fixture(tmp_path, monkeypatch)
    observed = observe(first, config, refs)
    observed["values"]["authority"]["provider_compute_spend_cap_usd"] = 6
    available = {"amount": 5.57}
    def balance():
        value = {"schema_version": "provider_credit_observation.v1", "provider": "vast",
                 "observed_at_epoch": time.time(), "status": "observed", "http_status": 200,
                 "credit_usd": available["amount"], "blockers": [], "provider_mutations_performed": 0}
        value["observation_digest"] = canonical_digest(value, digest_field="observation_digest")
        return value
    monkeypatch.setattr(credit, "observe_vast_credit", balance)
    monkeypatch.setenv(credit.RESERVE_ENV, "1")
    waiting = recovery.capacity_admission(observed, config, 103)
    assert waiting["status"] == "waiting_for_capacity"
    assert waiting["provider_credit_admission"]["blockers"] == ["provider_credit_insufficient"]
    available["amount"] = 7
    passed = recovery.capacity_admission(observed, config, 104)
    assert passed["status"] == "admitted"
    assert passed["provider_credit_admission"]["required_usd"] == 6
    available["amount"] = 0
    assert recovery.capacity_admission(observed, config, 105)["status"] == "waiting_for_capacity"


def test_closed_credit_failure_accepts_free_source_identity(tmp_path, monkeypatch):
    _, first, config, refs = credit_fixture(tmp_path, monkeypatch)
    free = {**first, "schema_version": "task_evaluation_scene_preparation_attempt.v1",
            "provider": "control_plane", "maximum_spend_usd": 0, "paid_authority_granted": False}
    free["attempt_digest"] = canonical_digest(free, digest_field="attempt_digest")
    factory = json.loads(Path(refs["factory"]["path"]).read_text())
    factory["attempt_digest"] = free["attempt_digest"]
    write(Path(refs["factory"]["path"]), factory, "factory_digest")
    assert observe(free, config, refs)["recoverable"]
    bad = copy.deepcopy(free)
    bad["maximum_spend_usd"] = 1
    with pytest.raises(ValueError, match="capacity_launch_binding_invalid"):
        observe(bad, config, refs)


@pytest.mark.parametrize("previous_recoveries", [0, 6])
def test_free_preparation_recovery_preserves_paid_ledger_and_own_retry_limit(tmp_path, monkeypatch, previous_recoveries):
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    from blueprint_pipeline import task_evaluation_scene_progression_recovery as ownership
    from tests.test_terminal_scene_attempt_settlement import _website_preparation
    from blueprint_pipeline.task_evaluation_public_scene_attempt_factory import record
    fx, row_id = _website_preparation(tmp_path, monkeypatch)
    directory = fx["directory"]
    main = json.loads((directory / "attempts" / (row_id + ".json")).read_text())
    source_path = directory / "preparation-attempts" / (fx["source"]["attempt_id"] + ".json")
    state = {"attempt": record(source_path), "attempt_id": fx["source"]["attempt_id"],
        "factory": fx["factory"], "binding_digest": fx["source"]["input_digest"],
        "recovery_predecessors": [{"kind": recovery.KIND}] * previous_recoveries}
    observed = {"recoverable": True, "kind": recovery.CREDIT_KIND, "result": record(fx["transition"]),
                "values": {"result": {"result_digest": "sha256:" + "1" * 64}, "configuration_attempt": main}}
    monkeypatch.setattr(recovery, "observe_failure", lambda **_: observed)
    monkeypatch.setattr(recovery, "capacity_admission", lambda *_: {"status": "admitted"})
    monkeypatch.setattr(recovery, "retain_failure", lambda **_: fx["transition"])
    def reconcile(**kwargs):
        assert kwargs["execution_attempt"] == main
        return {"failure": record(fx["transition"]), "ownership_reconciliation": record(fx["ownership"])}
    monkeypatch.setattr(ownership, "reconcile_ownership", reconcile)
    args = dict(directory=directory, intent=json.loads((directory / "intent.json").read_text()),
        state=state, attempt=fx["source"],
        link_path=tmp_path / "unused-link", preparation_path=tmp_path / "unused-request",
        config={"launch_execution_root": str(fx["launches"]), "launch_queue_root": str(fx["queue"])},
        release={"source_commit": fx["source"]["source_commit"], "runtime_digest": fx["source"]["runtime_digest"]},
        machinery={"maximum_preparation_spend_usd": 0}, output=tmp_path / "output", now=300)
    if previous_recoveries:
        with pytest.raises(ValueError, match="preallocation_capacity_recovery_cap_exhausted"):
            engine._recover_configuration_capacity(**args)
    else:
        assert engine._recover_configuration_capacity(**args)["phase"] == "capacity_recovery_reserved"
        new = json.loads(Path(state["attempt"]["path"]).read_text())
        assert new["maximum_spend_usd"] == 0 and new["paid_authority_granted"] is False
        assert new["attempt_id"] != fx["source"]["attempt_id"]
        assert len(list((directory / "preparation-attempts").glob("*.json"))) == 2
        assert len(list((directory / "attempts").glob("*.json"))) == 1
        assert state["recovery_predecessors"][0]["factory"] == fx["factory"]
        assert state["terminal_settlements"]["settled_rows"] == 1
