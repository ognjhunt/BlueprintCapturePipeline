"""Terminal attempts settle their rows conservatively; live or tampered evidence fails closed."""

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_terminal_scene_attempt_settlement as settlement
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_sam31_prefix_adoption import record
from blueprint_pipeline.task_evaluation_retained_controls_evidence import validated_cancellation
from blueprint_pipeline.task_evaluation_scene_intake import (
    SceneIntakeError, reserve_scene_attempt, stage_scene_intent,
)
from tests.test_task_evaluation_scene_intake import request

COMMIT = "d" * 40
REQUEST_DIGEST = "sha256:" + "7" * 64


def _intent(root, *, spend=26.0, attempts=8):
    value = copy.deepcopy(request())
    value["execution"].update({"max_total_spend_usd": spend, "max_paid_attempts": attempts,
                               "allowed_providers": ["vast", "openai"]})
    return stage_scene_intent(value=value, queue_root=root, authenticated_client="webapp",
                              trusted_clients={"webapp"}, now=100)


def _reserve(root, intent, attempt_id, cost, *, provider="vast", input_digest="sha256:" + "f" * 64, now=101):
    return reserve_scene_attempt(queue_root=root, intent_id=intent["intent_id"], attempt_id=attempt_id,
        source_commit=COMMIT, runtime_digest="sha256:" + "e" * 64, input_digest=input_digest,
        provider=provider, maximum_spend_usd=cost, now=now)


def _seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def _fixture(tmp_path, *, launch_status="blocked", queue_pending=False, **intent_kwargs):
    root = tmp_path / "intents"
    intent = _intent(root, **intent_kwargs)
    directory = root / intent["intent_id"]
    source = _reserve(root, intent, "source-a1", 4.5)
    rows = settlement.dependent_row_ids(REQUEST_DIGEST)
    for row_id, phase in rows.items():
        cost = {"scene_configuration": 16.76, "construction": 0.45, "controls": 0.45, "placement": 2.56}[phase]
        _reserve(root, intent, row_id, cost, provider="openai" if phase == "placement" else "vast",
                 input_digest=REQUEST_DIGEST if phase == "scene_configuration" else "sha256:" + "f" * 64)
    preparation_id = f"scene-abc-{source['attempt_id']}-dddddddd-20260913t000000z-scene-configuration-preparation"
    link = _seal({"schema_version": "task_evaluation_scene_preparation_link.v1", "intent_id": intent["intent_id"],
        "intent_digest": intent["intent_digest"], "expected_production_commit": COMMIT,
        "preparation_id": preparation_id, "request_digest": REQUEST_DIGEST, "result_filename": "r.json",
        "scene_id": "s", "task_id": "t", "team_namespace": "n"}, "link_digest")
    _write(directory / "preparations" / (REQUEST_DIGEST[7:] + ".json"), link)
    transition = _write(tmp_path / "out" / "release-transition.json", _seal({
        "schema_version": "task_evaluation_scene_release_transition.v1", "attempt_digest": source["attempt_digest"],
        "observed_at_epoch": 200, "parent_envelope": {}, "parent_state": "materialized",
        "provider_allocation_performed": False}, "failure_digest"))
    ownership = _write(tmp_path / "out" / "ownership.json", _seal({
        "schema_version": "task_evaluation_scene_attempt_ownership.v1", "attempt_digest": source["attempt_digest"],
        "status": "closed_without_resource", "active_writer_count": 0, "unresolved_create_count": 0,
        "observed_at_epoch": 201, "provider_mutation_performed": False}, "ownership_digest"))
    launches = tmp_path / "launch-runs"
    queue = tmp_path / "launches"
    for state in ("pending", "processing"):
        (queue / state).mkdir(parents=True)
    launch_id = settlement.launch_id_for_preparation(preparation_id)
    if launch_status is not None:
        _write(launches / launch_id / "launch_receipt.json", {"launch_id": launch_id, "status": launch_status})
    if queue_pending:
        _write(queue / "pending" / (launch_id + "-abc.json"), {"launch_id": launch_id})
    launches.mkdir(exist_ok=True)
    return {"root": root, "intent": intent, "directory": directory, "source": source, "rows": rows,
            "transition": transition, "ownership": ownership, "launches": launches, "queue": queue,
            "launch_id": launch_id}


def _settle(fx, **kwargs):
    return settlement.settle_retired_attempt_rows(directory=fx["directory"], retired_attempt=fx["source"],
        retirement_record={"path": str(fx["transition"])}, ownership_record={"path": str(fx["ownership"])},
        launch_execution_root=fx["launches"], launch_queue_root=fx["queue"], **kwargs)


@pytest.mark.parametrize("preparation_id,expected", [
    ("scene-source-a1-scene-configuration-preparation", "scene-source-a1-scene-configuration-activation-auto-launch"),
    ("website-owner-commit-preparation", "website-owner-commit-activation-auto-launch"),
])
def test_settlement_uses_the_controllers_actual_launch_id(preparation_id, expected):
    assert settlement.launch_id_for_preparation(preparation_id) == expected


def _website_preparation(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_preparation_attempts import create_preparation_attempt
    fx = _fixture(tmp_path, launch_status=None)
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(fx["root"]))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS", "webapp")
    for path in (fx["directory"] / "attempts").glob("*.json"):
        path.unlink()
    source = create_preparation_attempt(directory=fx["directory"], attempt_id="source-website",
        source_commit=COMMIT, runtime_digest="sha256:" + "e" * 64,
        input_digest="sha256:" + "f" * 64, now=101)
    fx["source"] = source
    from tests.test_task_evaluation_launch_preparation_contract import test_configuration_request
    from blueprint_pipeline.task_evaluation_launch_preparation_contract import validate_launch_preparation_request
    request = test_configuration_request()
    request.update(preparation_id="website-owner-commit-preparation", expected_production_commit=COMMIT)
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import spend_block
    request["replacement_authoring_backend"] = "astra_cad_blender_v1"
    request["spend"] = spend_block("astra_cad_blender_v1")
    request = validate_launch_preparation_request(request)
    digest = canonical_digest(request)
    fx["rows"] = settlement.dependent_row_ids(digest)
    row_id = next(iter(fx["rows"]))
    _reserve(fx["root"], fx["intent"], row_id, 16.76, input_digest=digest)
    link_path = next((fx["directory"] / "preparations").glob("*.json"))
    link = json.loads(link_path.read_text())
    link.update(preparation_id=request["preparation_id"], request_digest=digest)
    _write(link_path, _seal(link, "link_digest"))
    for path, field in ((fx["transition"], "failure_digest"), (fx["ownership"], "ownership_digest")):
        value = json.loads(path.read_text())
        value["attempt_digest"] = source["attempt_digest"]
        _write(path, _seal(value, field))
    request_path = _write(tmp_path / "website-request.json", request)
    factory_path = _write(tmp_path / "factory.json", _seal({
        "schema_version": "website_scene_attempt_factory.v1", "attempt_digest": source["attempt_digest"],
        "intent_digest": fx["intent"]["intent_digest"], "source_commit": COMMIT,
        "submission_request": record(request_path)}, "factory_digest"))
    fx["factory"] = record(factory_path)
    fx["launch_id"] = settlement.launch_id_for_preparation(request["preparation_id"])
    return fx, row_id


def test_zero_cost_website_preparation_releases_only_unstarted_construction_hold(tmp_path, monkeypatch):
    fx, row_id = _website_preparation(tmp_path, monkeypatch)
    result = _settle(fx, source_factory=fx["factory"])
    assert result["rows"] == [{"attempt_id": row_id, "status": "settled"}]
    attempt = json.loads((fx["directory"] / "attempts" / (row_id + ".json")).read_text())
    receipt = validated_cancellation(fx["directory"], attempt)
    assert receipt["settled_spend"]["retained_spend_usd"] == 0
    assert receipt["settled_spend"]["counts_as_attempt"] is False
    assert _reserve(fx["root"], fx["intent"], "scene-configuration-successor", 16.76, now=300)["status"] == "reserved"
    assert _settle(fx, source_factory=fx["factory"])["rows"][0]["status"] == "already_released"


def test_website_preparation_does_not_release_pending_execution_or_changed_factory(tmp_path, monkeypatch):
    fx, _ = _website_preparation(tmp_path, monkeypatch)
    pending = _write(fx["queue"] / "pending" / (fx["launch_id"] + ".json"), {"launch_id": fx["launch_id"]})
    result = _settle(fx, source_factory=fx["factory"])
    assert result["rows"] == []
    assert result["skipped"][0]["reason"] == "launch_not_terminal"
    pending.unlink()
    Path(fx["factory"]["path"]).write_text("{}")
    with pytest.raises(ValueError):
        _settle(fx, source_factory=fx["factory"])


@pytest.mark.parametrize("activation_completed", [False, True])
def test_activated_website_preparation_uses_execution_reconciliation_on_release_change(tmp_path, monkeypatch, activation_completed):
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    from blueprint_pipeline import task_evaluation_scene_progression_recovery as recovery
    from blueprint_pipeline.task_evaluation_launch_preparation_queue import ENVELOPE_SCHEMA_VERSION
    fx, row_id = _website_preparation(tmp_path, monkeypatch)
    directory = fx["directory"]
    link_path = next((directory / "preparations").glob("*.json"))
    link = json.loads(link_path.read_text())
    paid_path = directory / "attempts" / (row_id + ".json")
    paid = json.loads(paid_path.read_text())
    link["scene_configuration_attempt"] = record(paid_path)
    activation_link = _write(directory / "activation.json", _seal(link, "link_digest"))
    factory = json.loads(Path(fx["factory"]["path"]).read_text())
    request = json.loads(Path(factory["submission_request"]["path"]).read_text())
    preparation_queue = tmp_path / "preparations"
    _write(preparation_queue / "materialized" / (request["preparation_id"] + "-x.json"), _seal({
        "schema_version": ENVELOPE_SCHEMA_VERSION, "request": request,
        "request_digest": canonical_digest(request)}, "envelope_digest"))
    output = tmp_path / "factory-output" / fx["intent"]["intent_id"] / fx["source"]["attempt_id"]
    _write(output / "submission-attempts" / "1.json", {})
    state = {"attempt_id": fx["source"]["attempt_id"],
        "attempt": record(directory / "preparation-attempts" / (fx["source"]["attempt_id"] + ".json")),
        "activation_link": record(activation_link),
        "factory": fx["factory"]}
    if activation_completed:
        state["activation"] = {"present": True}
    observed = []
    def reconcile(**kwargs):
        observed.append(kwargs)
        return {"failure": record(fx["transition"]),
                "ownership_reconciliation": record(fx["ownership"])}
    monkeypatch.setattr(recovery, "reconcile_ownership", reconcile)
    assert engine._release_successor(directory=directory, intent=fx["intent"], state=state,
        config={"activation_enabled": True, "factory_output_root": str(tmp_path / "factory-output"),
                "preparation_queue_root": str(preparation_queue), "launch_execution_root": str(fx["launches"]),
                "launch_queue_root": str(fx["queue"])}, release={"source_commit": "b" * 40}, now=300)
    assert observed[0]["execution_attempt"] == paid
    assert observed[0]["attempt"] == fx["source"]
    assert "attempt_id" not in state
    assert state["release_predecessors"][0]["factory"] == fx["factory"]
    assert validated_cancellation(directory, paid)["settled_spend"]["retained_spend_usd"] == 0


def test_settlement_releases_holds_and_attempt_slots(tmp_path: Path) -> None:
    """2026-09-13: 31 rows held $148.38 of a $150 cap while ~$4 was spent."""
    fx = _fixture(tmp_path)
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(fx["root"], fx["intent"], "source-a2", 4.5, now=300)
    outcome = _settle(fx)
    assert outcome["status"] == "settled"
    assert sorted(row["attempt_id"] for row in outcome["rows"] if row["status"] == "settled") == sorted(
        ["source-a1", *fx["rows"]])
    for row_id in ("source-a1", *fx["rows"]):
        attempt = json.loads((fx["directory"] / "attempts" / (row_id + ".json")).read_text())
        receipt = validated_cancellation(fx["directory"], attempt)
        assert receipt["schema_version"] == settlement.SCHEMA
        assert receipt["retired_attempt"]["attempt_id"] == "source-a1"
    successor = _reserve(fx["root"], fx["intent"], "source-a2", 4.5, now=300)
    assert successor["status"] == "reserved"
    again = _settle(fx)
    assert all(row["status"] == "already_released" for row in again["rows"])


def test_paid_attempts_are_source_rows_and_executed_ones_keep_counting(tmp_path: Path) -> None:
    """Dependent rows are holds of one attempt, never attempts; a settled executed attempt still counts."""
    fx = _fixture(tmp_path, spend=100.0, attempts=2)
    assert _reserve(fx["root"], fx["intent"], "source-a2", 4.5, now=300)["status"] == "reserved"
    with pytest.raises(SceneIntakeError, match="attempt_cap_exhausted"):
        _reserve(fx["root"], fx["intent"], "source-a3", 4.5, now=301)
    _settle(fx)  # source-a1 executed its preparation: settling it releases no attempt slot
    with pytest.raises(SceneIntakeError, match="attempt_cap_exhausted"):
        _reserve(fx["root"], fx["intent"], "source-a3", 4.5, now=302)


def test_settlement_keeps_unproven_spend_and_releases_only_never_started_rows(tmp_path: Path) -> None:
    """A stopped resource proves nothing more can be spent, not that nothing was (2026-09-13 audit)."""
    fx = _fixture(tmp_path)  # launch blocked: edits may have been paid, controls never became eligible
    _settle(fx)
    holds = {}
    for row_id in ("source-a1", *fx["rows"]):
        attempt = json.loads((fx["directory"] / "attempts" / (row_id + ".json")).read_text())
        receipt = validated_cancellation(fx["directory"], attempt)
        holds[fx["rows"].get(row_id, "source")] = (receipt["settled_spend"], settlement.retained_hold(receipt))
    for phase, (sealed, derived) in holds.items():
        assert sealed == derived, phase
    assert holds["source"][0] == {"basis": "retired_source_unreconciled", "retained_spend_usd": 4.5, "counts_as_attempt": True}
    assert holds["scene_configuration"][0] == {"basis": "terminal_launch_unreconciled", "retained_spend_usd": 16.76, "counts_as_attempt": False}
    for phase in ("construction", "controls", "placement"):
        assert holds[phase][0] == {"basis": "downstream_of_blocked_launch", "retained_spend_usd": 0.0, "counts_as_attempt": False}
    # 4.5 + 16.76 stay held: 4.5 more fits under the $26 cap, 5.0 more does not.
    assert _reserve(fx["root"], fx["intent"], "source-a2", 4.5, now=300)["status"] == "reserved"
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(fx["root"], fx["intent"], "source-a3", 0.75, now=301)


def test_completed_launch_keeps_every_dependent_hold_until_reconciled(tmp_path: Path) -> None:
    fx = _fixture(tmp_path, launch_status="completed")
    _settle(fx)
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(fx["root"], fx["intent"], "source-a2", 1.3, now=300)  # 24.72 held + 1.3 > 26
    assert _reserve(fx["root"], fx["intent"], "source-a2", 1.28, now=300)["status"] == "reserved"


def test_legacy_settlement_receipts_without_settled_spend_are_accounted_the_same_way(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    _settle(fx)
    for row_id in ("source-a1", *fx["rows"]):
        path = fx["directory"] / "cancelled-unstarted-controls" / (row_id + ".json")
        receipt = json.loads(path.read_text())
        del receipt["settled_spend"]
        del receipt["receipt_digest"]
        path.chmod(0o600)
        path.write_text(json.dumps(_seal(receipt, "receipt_digest"), sort_keys=True) + "\n")
    assert _reserve(fx["root"], fx["intent"], "source-a2", 4.5, now=300)["status"] == "reserved"
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(fx["root"], fx["intent"], "source-a3", 0.75, now=301)


def test_tampered_settled_spend_fails_closed(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    _settle(fx)
    path = fx["directory"] / "cancelled-unstarted-controls" / "source-a1.json"
    receipt = json.loads(path.read_text())
    receipt["settled_spend"] = {"basis": "launch_never_queued", "retained_spend_usd": 0.0, "counts_as_attempt": False}
    del receipt["receipt_digest"]
    path.chmod(0o600)
    path.write_text(json.dumps(_seal(receipt, "receipt_digest"), sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="terminal_settlement_settled_spend_invalid"):
        _reserve(fx["root"], fx["intent"], "source-a2", 4.5, now=300)


def test_dependent_rows_wait_for_a_terminal_launch(tmp_path: Path) -> None:
    fx = _fixture(tmp_path, launch_status="running")
    outcome = _settle(fx)
    assert [row["attempt_id"] for row in outcome["rows"]] == ["source-a1"]
    assert outcome["skipped"][0]["reason"] == "launch_not_terminal"
    _write(fx["launches"] / fx["launch_id"] / "launch_receipt.json", {"launch_id": fx["launch_id"], "status": "blocked"})
    outcome = _settle(fx)
    assert sorted(row["attempt_id"] for row in outcome["rows"] if row["status"] == "settled") == sorted(fx["rows"])


def test_never_queued_launch_settles_dependent_rows_but_a_queued_one_waits(tmp_path: Path) -> None:
    fx = _fixture(tmp_path, launch_status=None)
    outcome = _settle(fx)
    assert len([row for row in outcome["rows"] if row["status"] == "settled"]) == 5
    receipt = json.loads((fx["directory"] / "cancelled-unstarted-controls" / (next(iter(fx["rows"])) + ".json")).read_text())
    assert receipt["execution_terminal"] == {"launch_id": fx["launch_id"], "launch_receipt": None, "launch_never_queued": True}
    queued = _fixture(tmp_path / "second", launch_status=None, queue_pending=True)
    outcome = _settle(queued)
    assert [row["attempt_id"] for row in outcome["rows"]] == ["source-a1"]


def test_tampered_evidence_fails_closed_at_reservation(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    _settle(fx)
    ownership = json.loads(fx["ownership"].read_text())
    ownership["status"] = "unresolved"
    fx["ownership"].write_text(json.dumps(ownership, sort_keys=True) + "\n")
    attempt = json.loads((fx["directory"] / "attempts" / "source-a1.json").read_text())
    with pytest.raises(ValueError, match="terminal_settlement_ownership_record_changed"):
        validated_cancellation(fx["directory"], attempt)
    with pytest.raises(ValueError, match="terminal_settlement_ownership_record_changed"):
        _reserve(fx["root"], fx["intent"], "source-a3", 4.5, now=400)


def test_sweep_settles_lineage_and_tolerates_broken_entries(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    state = {"release_predecessors": [
        {"attempt": {"path": str(fx["directory"] / "attempts" / "source-a1.json")},
         "reconciliation": {"failure": {"path": str(fx["transition"])},
                            "ownership_reconciliation": {"path": str(fx["ownership"])}}},
        {"attempt": {"path": str(tmp_path / "missing.json")}, "reconciliation": {}},
        "garbage",
    ], "recovery_predecessors": [{"attempt": None, "evidence": None}]}
    config = {"launch_execution_root": str(fx["launches"]), "launch_queue_root": str(fx["queue"])}
    summary = settlement.sweep_retired_attempts(directory=fx["directory"], state=state, config=config)
    assert summary["settled_rows"] == 5
    assert len(summary["skipped"]) >= 2
    again = settlement.sweep_retired_attempts(directory=fx["directory"], state=state, config=config)
    assert again["settled_rows"] == 0 and again["already_released_rows"] == 5
    assert again["summary_digest"] != summary["summary_digest"]


def _website_preallocation_failure(tmp_path, monkeypatch, *, allocated=False,
        teardown_status="not_required_provider_adapter_never_invoked"):
    fx, row_id = _website_preparation(tmp_path, monkeypatch)
    factory = json.loads(Path(fx["factory"]["path"]).read_text())
    request = json.loads(Path(factory["submission_request"]["path"]).read_text())
    run = fx["launches"] / fx["launch_id"]
    result_path = _write(run / "result.json", {
        "schema_version": "task_evaluation_scene_configuration_vast_result.v1",
        "run_id": request["run_id"], "source_commit": COMMIT, "status": "blocked",
        "provider_mutations_performed": int(allocated), "continuing_spend_from_this_run": False,
        "provider_runtime_output_zip_path": None})
    teardown_path = _write(run / "teardown.json", {"schema_version": "vast_teardown_manifest.v1",
        "status": teardown_status, "vast_instance_ids": [],
        "continuing_spend_from_this_run": False})
    def ref(path):
        return {**settlement._file(path), "exists": True}
    _write(run / "launch_receipt.json", {"launch_id": fx["launch_id"], "source_commit": COMMIT,
        "status": "blocked", "terminal_evidence": {"result": ref(result_path),
            "artifacts": {"teardown_manifest_path": ref(teardown_path)}}})
    _settle(fx, source_factory=fx["factory"])
    attempt = json.loads((fx["directory"] / "attempts" / (row_id + ".json")).read_text())
    return fx, validated_cancellation(fx["directory"], attempt), result_path, teardown_path


@pytest.mark.parametrize("teardown_status", ["not_required_provider_adapter_never_invoked",
                                           "not_required_prelaunch_inventory_guard_blocked"])
def test_proven_no_allocation_releases_only_gpu_and_provider_authoring_budget(tmp_path, monkeypatch, teardown_status):
    fx, receipt, _, _ = _website_preallocation_failure(tmp_path, monkeypatch, teardown_status=teardown_status)
    assert receipt["settled_spend"]["retained_spend_usd"] == 16.76
    assert settlement.retained_hold(receipt)["retained_spend_usd"] == 16.76
    assert settlement.budget_retained_hold(receipt) == {
        "basis": "preallocation_api_budget_upper_bound", "retained_spend_usd": 5.76,
        "counts_as_attempt": False}
    assert _reserve(fx["root"], fx["intent"], "scene-configuration-successor", 15, now=300)["status"] == "reserved"
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        _reserve(fx["root"], fx["intent"], "scene-configuration-another", 5.25, now=301)


@pytest.mark.parametrize("changed", ["result", "teardown", "allocated"])
def test_preallocation_budget_reduction_requires_bound_nonallocation_evidence(tmp_path, monkeypatch, changed):
    _, receipt, result, teardown = _website_preallocation_failure(tmp_path, monkeypatch, allocated=changed == "allocated")
    if changed != "allocated":
        (result if changed == "result" else teardown).write_text("{}")
    assert settlement.budget_retained_hold(receipt)["retained_spend_usd"] == 16.76
