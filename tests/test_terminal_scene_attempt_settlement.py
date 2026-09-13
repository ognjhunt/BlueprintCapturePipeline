"""Terminal attempts release their holds; live or tampered evidence fails closed."""

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_terminal_scene_attempt_settlement as settlement
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
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


def test_settlement_releases_holds_and_attempt_slots(tmp_path: Path) -> None:
    """2026-09-13: 31 rows held $148.38 of a $150 cap while ~$4 was spent."""
    fx = _fixture(tmp_path, launch_status=None)
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


def test_attempt_count_ignores_settled_rows(tmp_path: Path) -> None:
    fx = _fixture(tmp_path, spend=100.0, attempts=5, launch_status=None)
    with pytest.raises(SceneIntakeError, match="attempt_cap_exhausted"):
        _reserve(fx["root"], fx["intent"], "source-a2", 4.5, now=300)
    _settle(fx)
    assert _reserve(fx["root"], fx["intent"], "source-a2", 4.5, now=300)["status"] == "reserved"


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
