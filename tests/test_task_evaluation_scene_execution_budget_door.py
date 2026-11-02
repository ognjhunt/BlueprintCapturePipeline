"""The door form of the owner budget grant: preview, then apply with the preview's digest."""
import json

import pytest

from blueprint_pipeline import task_evaluation_scene_execution_budget_door as door
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.task_evaluation_scene_execution_budget_evidence import effective_execution_budget
from blueprint_pipeline.task_evaluation_scene_intent_contracts import ROOT_ENV
from tests.test_task_evaluation_scene_intake import request, stage


def _request(record, **changes):
    owner = request()["owner"]
    body = {"kind": "extend-scene-execution-budget", "intent_id": record["intent_id"],
            "owner_user_id": owner["user_id"], "owner_organization_id": owner["organization_id"],
            "approved_by": owner["user_id"], "approval_evidence": "owner chose +$5 in session X",
            "max_total_spend_usd": 25, "max_paid_attempts": 16, "apply": False, "apply_ack": "",
            "ack": "extend-scene-execution-budget"}
    body.update(changes)
    return body


def test_preview_writes_nothing_and_apply_with_its_digest_records_one_grant(tmp_path, monkeypatch):
    record = stage(tmp_path)
    monkeypatch.setenv(ROOT_ENV, str(tmp_path))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS", "webapp")
    from blueprint_pipeline import task_evaluation_scene_execution_budget as budget
    monkeypatch.setattr(budget.time, "time", lambda: 200.0)  # inside the fixture intent's window
    directory = tmp_path / record["intent_id"]
    intent = intake._read(directory / "intent.json", "intent_digest")
    before = effective_execution_budget(directory, intent)
    preview = door.run(_request(record), door_caller="cloud-scene-runs", door_request_id="req-1")
    assert preview["door_outcome"] == "planned" and preview["current"] == before
    assert preview["requested"] == {"max_total_spend_usd": 25, "max_paid_attempts": 16}
    assert effective_execution_budget(directory, intent) == before
    with pytest.raises(ValueError, match="budget_apply_ack_mismatch"):
        door.run(_request(record, apply=True, apply_ack="sha256:" + "0" * 64),
                 door_caller="cloud-scene-runs", door_request_id="req-2")
    applied = door.run(_request(record, apply=True, apply_ack=preview["apply_ack"]),
                       door_caller="cloud-scene-runs", door_request_id="req-3")
    assert applied["door_outcome"] == "extended"
    after = effective_execution_budget(directory, intent)
    assert after["max_total_spend_usd"] == 25 and after["max_paid_attempts"] == 16
    grants = [json.loads(path.read_text()) for path in (directory / "execution-budget-extensions").glob("*.json")] \
        if (directory / "execution-budget-extensions").is_dir() else []
    references = json.dumps(grants) + json.dumps(applied, default=str)
    assert "channel=operator_door" in references and "door_request_id=req-3" in references


def test_an_approval_that_is_not_the_owners_is_refused(tmp_path, monkeypatch):
    record = stage(tmp_path)
    monkeypatch.setenv(ROOT_ENV, str(tmp_path))
    with pytest.raises(ValueError, match="owner_or_revocation_mismatch"):
        door.run(_request(record, approved_by="someone-else"), door_caller="cloud", door_request_id="r")


def test_main_reports_a_refusal_as_a_door_outcome(tmp_path, monkeypatch, capsys):
    record = stage(tmp_path)
    monkeypatch.setenv(ROOT_ENV, str(tmp_path))
    body = _request(record, approved_by="someone-else")
    assert door.main(["--request-json", json.dumps(body), "--door-caller", "cloud", "--door-request-id", "r"]) == 2
    assert json.loads(capsys.readouterr().out) == {"door_outcome": "refused", "code": "scene_execution_budget_owner_or_revocation_mismatch"}
