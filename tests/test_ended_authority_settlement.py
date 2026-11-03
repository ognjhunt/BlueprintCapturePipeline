"""An intent whose authority ended settles its current attempt when that attempt never allocated.

capture-447ae4f4, 2026-11-02: the current attempt's launch was refused before
allocation at 13:39Z (scene_configuration_provider_output_disk_budget_exceeded,
provider_mutations_performed 0, no instance), and the intent was revoked at
14:08Z. Progression stops at the authority gate and only retires an attempt
when it starts a successor, so the attempt's $6.01 hold stayed held for good
and blocked a re-issue, which needs every prior row cancelled or settled.

Every figure and identifier here is synthetic.
"""
import copy
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_terminal_scene_attempt_settlement as settlement
from blueprint_pipeline import task_evaluation_ended_authority_settlement as ended
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_retained_controls_evidence import validated_cancellation
from blueprint_pipeline.task_evaluation_sam31_prefix_adoption import record
from blueprint_pipeline.task_evaluation_scene_intake import (
    committed_spend_usd, reserve_scene_attempt, revoke_scene_intent, scene_intent_status,
)
from tests.test_artisan_proven_authoring_spend import (
    AGREEING, COMMIT, _artisan_runtime, _intent, _pre_gate_refusal, _request, _seal, _terminal_attempt, _write,
)

DISK_REFUSAL = "scene_configuration_provider_output_disk_budget_exceeded"
REVOKED_AT, EXPIRED_AT = 400, 1500  # the synthetic intent's window closes at 1000


@pytest.fixture
def site(tmp_path, monkeypatch):
    root = tmp_path / "intents"
    intent = _intent(root)
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(root))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS", "webapp")
    queue = tmp_path / "launches"
    for state in ("pending", "processing"):
        (queue / state).mkdir(parents=True)
    for state in ("pending", "processing", "waiting_external"):
        (tmp_path / "child-queue" / state).mkdir(parents=True)
    (tmp_path / "children").mkdir()
    config = {"factory_output_root": str(tmp_path / "factory-output"), "launch_execution_root": str(tmp_path / "launch-runs"),
              "launch_queue_root": str(queue), "provider_guard_path": str(tmp_path / "guard.json"),
              "ownership_roots": [str(tmp_path)], "child_execution_root": str(tmp_path / "children"),
              "child_queue_root": str(tmp_path / "child-queue")}
    return {"tmp": tmp_path, "root": root, "intent": intent, "directory": root / intent["intent_id"],
            "launches": tmp_path / "launch-runs", "queue": queue, "config": config}


def _guard(site, at, *, live=0):
    """The GPU spend guard's provider inventory, as reconcile_ownership reads it."""
    _write(site["tmp"] / "guard.json", {"schema_version": "gpu_spend_guard.v1",
        "generated_at": datetime.fromtimestamp(at, timezone.utc).isoformat(), "reap_mode": True,
        "provider_zero_verified": live == 0, "live_instance_count": live, "total_burn_per_hour_usd": 0,
        "reap_candidate_ids": [], "reap_results": [], "provider_zero": {
            "status": "verified" if live == 0 else "unverified", "global_live_instance_count": live,
            "global_total_burn_per_hour_usd": 0, "required_provider_ids": ["vast"]},
        "inventory_results": [{"provider": "vast", "status": "succeeded", "row_count": live, "required": True}]})


def _current_attempt(site, *, change=None):
    """The intent's current attempt: a $6.01 row whose only launch was refused before allocation."""
    from blueprint_pipeline.task_evaluation_scene_preparation_attempts import (
        create_preparation_attempt, preparation_attempt_path,
    )
    directory, intent = site["directory"], site["intent"]
    source = create_preparation_attempt(directory=directory, attempt_id="source-current", source_commit=COMMIT,
                                        runtime_digest="sha256:" + "e" * 64, input_digest="sha256:" + "f" * 64,
                                        now=101)
    request = _request("current", authoring=0.01, total=6.01)
    digest = canonical_digest(request)
    row_id = next(iter(settlement.dependent_row_ids(digest)))
    reserve_scene_attempt(queue_root=site["root"], intent_id=intent["intent_id"], attempt_id=row_id,
                          source_commit=COMMIT, runtime_digest="sha256:" + "e" * 64, input_digest=digest,
                          provider="vast", maximum_spend_usd=request["spend"]["hard_cap_usd"], now=102)
    row_path = directory / "attempts" / (row_id + ".json")
    link = {"schema_version": "task_evaluation_scene_preparation_link.v1", "intent_id": intent["intent_id"],
            "intent_digest": intent["intent_digest"], "expected_production_commit": COMMIT,
            "preparation_id": request["preparation_id"], "request_digest": digest, "result_filename": "r.json",
            "scene_id": "s", "task_id": "t", "team_namespace": "n"}
    _write(directory / "preparations" / (digest[7:] + ".json"), _seal(dict(link), "link_digest"))
    activation = _write(directory / "preparations" / (digest[7:] + ".activation.json"),
                        _seal({**link, "scene_configuration_attempt": record(row_path)}, "link_digest"))
    out = site["tmp"] / "factory-output" / intent["intent_id"] / source["attempt_id"]
    factory = record(_write(out / "factory.json", _seal({
        "schema_version": "website_scene_attempt_factory.v1", "attempt_digest": source["attempt_digest"],
        "intent_digest": intent["intent_digest"], "source_commit": COMMIT,
        "submission_request": record(_write(out / "request.json", request))}, "factory_digest")))
    run = site["launches"] / settlement.launch_id_for_preparation(request["preparation_id"])
    result = {"schema_version": "task_evaluation_scene_configuration_vast_result.v1", "status": "blocked",
              "run_id": request["run_id"], "source_commit": COMMIT, "retry_cap": 0, "provider_mutations_performed": 0,
              "continuing_spend_from_this_run": False, "blockers": [DISK_REFUSAL]}
    teardown = {"schema_version": "vast_teardown_manifest.v1", "status": "not_required_provider_adapter_never_invoked",
                "vast_instance_ids": [], "continuing_spend_from_this_run": False}
    if change == "allocated":
        result["provider_mutations_performed"] = 1
        teardown.update(status="completed", vast_instance_ids=[7])
    elif change == "instance_recorded":
        teardown["vast_instance_ids"] = [7]
    elif change == "api_preparation_ran":
        result["api_pretraining"] = {"status": "completed"}
    elif change == "authoring_output_without_proof":
        output = site["tmp"] / "runtime_output"
        _artisan_runtime(output, request, runs=[([0.005], 0.004, True)])  # ARTISAN ran past its model gate
        from blueprint_pipeline.task_evaluation_scene_configuration_output_archive import write_output_archive
        run.mkdir(parents=True)
        write_output_archive(output, run / "output.zip")
        result.update(provider_runtime_output_zip_path=str(run / "output.zip"),
                      provider_runtime_output_zip_sha256=record(run / "output.zip")["sha256"])
    result_path = _write(run / "allocator/result.json", _seal(result, "result_digest"))
    teardown_path = _write(run / "allocator/teardown.json", teardown)
    if change == "still_running":
        _write(run / "launch_started.json", {"launch_id": run.name})
    else:
        _write(run / "launch_receipt.json", {"launch_id": run.name, "source_commit": COMMIT, "status": "blocked",
            "terminal_evidence": {"result": {**settlement._file(result_path), "exists": True},
                                  "artifacts": {"teardown_manifest_path": {**settlement._file(teardown_path),
                                                                           "exists": True}}}})
    if change == "result_changed_after_receipt":
        _write(result_path, {**result, "provider_mutations_performed": 1})
    state = {"attempt_id": source["attempt_id"], "attempt": record(preparation_attempt_path(directory, "source-current")),
             "factory": factory, "activation_link": record(activation)}
    return state, row_path


def _end(site, how):
    if how == "revoked":
        intent = site["intent"]
        revoke_scene_intent(queue_root=site["root"], intent_id=intent["intent_id"], intent_digest=intent["intent_digest"],
                            owner=intent["request"]["owner"], now=REVOKED_AT)
        return REVOKED_AT + 100
    return EXPIRED_AT


def _settle(site, state, now):
    return ended.settle_ended_authority_attempt(directory=site["directory"], intent=site["intent"], state=state,
                                                     config=site["config"], now=now)


@pytest.mark.parametrize("how", ["revoked", "expired"])
def test_capture_447_current_attempt_settles_at_zero_once_authority_ends(site, how):
    # Two earlier attempts settled as capture-447's did, then the current one.
    _terminal_attempt(site, "pregate", _request("pregate", authoring=15.0, total=20.0), _pre_gate_refusal)
    _terminal_attempt(site, "parent", _request("parent", authoring=15.0, total=20.0),
                      lambda output, request: _artisan_runtime(output, request, runs=AGREEING))
    state, row_path = _current_attempt(site)
    assert committed_spend_usd(site["directory"], site["intent"]) == Decimal("29.77")  # 5 + 18.76 + 6.01
    now = _end(site, how)
    rows_before = sorted(path.name for path in (site["directory"] / "attempts").glob("*.json"))
    _guard(site, now)

    outcome = _settle(site, state, now)

    assert outcome["status"] == "settled" and {"attempt_id": row_path.stem, "status": "settled"} in outcome["rows"]
    row = json.loads(row_path.read_text())
    receipt = validated_cancellation(site["directory"], row)
    assert receipt["schema_version"] == settlement.SCHEMA
    retirement = json.loads(Path(receipt["retirement_record"]["path"]).read_text())
    assert retirement["schema_version"] == settlement.AUTHORITY_ENDED_SCHEMA
    assert retirement["provider_allocation_performed"] is False and retirement["new_execution_authorized"] is False
    assert (retirement["authority"]["revocation"] is not None) == (how == "revoked")
    assert settlement.budget_retained_hold(receipt) == {"basis": "preallocation_api_budget_upper_bound",
                                                        "retained_spend_usd": 0.0, "counts_as_attempt": False}
    assert committed_spend_usd(site["directory"], site["intent"]) == Decimal("23.76")
    # The re-issue's check: every prior paid row now carries a validated cancellation.
    assert all(validated_cancellation(site["directory"], json.loads(path.read_text())) is not None
               for path in (site["directory"] / "attempts").glob("*.json"))
    status = scene_intent_status(queue_root=site["root"], intent_id=site["intent"]["intent_id"], now=now)
    settled = {a["attempt_id"]: a for a in status["attempts"]}[row_path.stem]
    assert (settled["status"], settled["retained_spend_usd"], settled["retained_spend_basis"]) == (
        "cancelled_before_execution", 0.0, "preallocation_api_budget_upper_bound")
    # Accounting only: no row reserved, nothing queued.
    assert sorted(path.name for path in (site["directory"] / "attempts").glob("*.json")) == rows_before
    assert not [path for name in ("pending", "processing") for path in (site["queue"] / name).iterdir()]
    assert _settle(site, state, now + 60) is None


def test_live_authority_settles_nothing(site):
    state, row_path = _current_attempt(site)
    _guard(site, 500)
    with pytest.raises(ValueError, match="authority_not_ended"):
        _settle(site, state, 500)
    assert validated_cancellation(site["directory"], json.loads(row_path.read_text())) is None


@pytest.mark.parametrize("change", ["allocated", "instance_recorded", "api_preparation_ran", "still_running",
                                    "authoring_output_without_proof", "result_changed_after_receipt"])
def test_a_launch_that_may_have_spent_or_may_still_run_keeps_its_hold(site, change):
    state, row_path = _current_attempt(site, change=change)
    now = _end(site, "revoked")
    _guard(site, now)
    with pytest.raises(ValueError, match="ended_attempt_launch"):
        _settle(site, state, now)
    assert validated_cancellation(site["directory"], json.loads(row_path.read_text())) is None
    assert committed_spend_usd(site["directory"], site["intent"]) == Decimal("6.01")


def test_settlement_waits_for_the_global_ownership_reconciliation(site):
    state, row_path = _current_attempt(site)
    now = _end(site, "revoked")
    _guard(site, now, live=1)  # another job's instance is still live somewhere
    with pytest.raises(ValueError, match="recovery_provider_zero_required"):
        _settle(site, state, now)
    child = _write(site["tmp"] / "child-queue/processing/other-job.json", {"job": "other"})
    _guard(site, now + 10)
    with pytest.raises(ValueError, match="ownership_reconciliation_required"):
        _settle(site, state, now + 10)
    # A reconciliation that found live work is not repeated every tick.
    child.unlink()
    _guard(site, now + 70)
    assert _settle(site, state, now + 70)["status"] == "waiting_for_ownership_reconciliation"
    assert validated_cancellation(site["directory"], json.loads(row_path.read_text())) is None
    _guard(site, now + 10 + settlement.OWNERSHIP_RETRY_SECONDS)
    assert _settle(site, state, now + 10 + settlement.OWNERSHIP_RETRY_SECONDS)["status"] == "settled"
    assert validated_cancellation(site["directory"], json.loads(row_path.read_text())) is not None


def test_progression_settles_before_it_stops_at_the_authority_gate(site):
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    from blueprint_pipeline.task_evaluation_scene_progression_state import advance
    state, row_path = _current_attempt(site)
    directory, intent = site["directory"], site["intent"]
    advance(directory, intent, None, status="blocked", phase="configuration_capacity", state=copy.deepcopy(state),
            blockers=["preallocation_capacity_not_recovered"], now=300)
    now = _end(site, "revoked")
    _guard(site, now)
    progress = engine._advance_intent(directory, intent, site["config"], {"source_commit": COMMIT}, resolver=None,
                                      publisher=None, submitter=None, status_reader=None, activation_provisioner=None,
                                      now=now)
    assert (progress["status"], progress["phase"]) == ("blocked", "authority")
    assert progress["state"]["terminal_settlements"]["ended_authority"] == {
        "status": "settled", "retired_attempt_id": "source-current", "settled_rows": 1}
    assert validated_cancellation(directory, json.loads(row_path.read_text())) is not None
