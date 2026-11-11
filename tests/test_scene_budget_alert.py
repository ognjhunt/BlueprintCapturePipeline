"""Owner decision 2026-11-10: page once per job per whole multiple of its grant; never stop anything."""
import copy
import json

from blueprint_pipeline import control_plane_capacity_controller as capacity
from blueprint_pipeline import control_plane_stall_alarm as stall
from blueprint_pipeline import scene_budget_alert as alert
from blueprint_pipeline import task_evaluation_scene_execution_budget as owner_budget
from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
from tests.test_task_evaluation_scene_intake import request as intake_request


def _stage(root):
    value = copy.deepcopy(intake_request())
    value["execution"].update({"max_total_spend_usd": 10, "max_paid_attempts": 8})
    receipt = stage_scene_intent(value=value, queue_root=root, authenticated_client="webapp",
                                 trusted_clients={"webapp"}, now=100)
    return json.loads((root / receipt["intent_id"] / "intent.json").read_text())


def _raise(root, intent, spend, now):
    owner_budget.extend_scene_execution_budget(
        queue_root=root, intent_id=intent["intent_id"], intent_digest=intent["intent_digest"],
        owner=intent["request"]["owner"], authenticated_client="webapp", trusted_clients={"webapp"},
        max_total_spend_usd=spend, max_paid_attempts=8, authorization_reference="owner_approval:test",
        ack=owner_budget.ACK, now=now)


def _tick(root, intent, now, memory):
    out = {}
    _, rows = alert.check_scene_budgets(root, [intent["intent_id"]], now=now, remembered=memory, memory_out=out)
    return rows, out


def test_a_page_once_per_whole_multiple_and_nothing_below_two(tmp_path):
    intent = _stage(tmp_path)
    _raise(tmp_path, intent, 19.99, now=150)
    rows, memory = _tick(tmp_path, intent, 1000, {})
    assert rows == []
    _raise(tmp_path, intent, 20, now=160)
    rows, memory = _tick(tmp_path, intent, 1000, memory)
    assert [(r["code"], r["targets"], r["multiple"], r["ceiling_usd"]) for r in rows] == [
        ("scene_budget_doubled", [intent["intent_id"] + "@2x"], 2, 20.0)]
    # Announced while a failed delivery can still be retried, then quiet: the controller would repeat it hourly.
    rows, memory = _tick(tmp_path, intent, 1000 + alert.ANNOUNCE_SECONDS - 1, memory)
    assert len(rows) == 1
    rows, memory = _tick(tmp_path, intent, 1000 + alert.ANNOUNCE_SECONDS, memory)
    assert rows == []
    _raise(tmp_path, intent, 31, now=170)
    rows, memory = _tick(tmp_path, intent, 5000, memory)
    assert rows[0]["targets"] == [intent["intent_id"] + "@3x"]
    rows, _ = _tick(tmp_path, intent, 5000 + alert.ANNOUNCE_SECONDS, memory)
    assert rows == []


def test_the_stall_tick_carries_it_as_a_page_that_gates_nothing(tmp_path):
    intent = _stage(tmp_path)
    _raise(tmp_path, intent, 25, now=150)
    config = stall.StallConfig(scene_intent_root=tmp_path, launch_queue_root=None, launch_state_root=None,
                               deploy_tool=None)
    first = stall.observe_stalls(config, now=1000, unit_reader=lambda units: {})
    [row] = [row for row in first["alerts"] if row["code"] == alert.CODE]
    assert capacity._severity(row["code"]) == "page"
    assert "2x" in stall.describe(row) and "nothing is stopped" in stall.describe(row)
    later = stall.observe_stalls(config, now=1000 + alert.ANNOUNCE_SECONDS, previous=first,
                                 unit_reader=lambda units: {})
    assert not [row for row in later["alerts"] if row["code"] == alert.CODE]
    # The intent itself is untouched by the alarm.
    assert json.loads((tmp_path / intent["intent_id"] / "intent.json").read_text()) == intent
