"""Several objects per job, stage 6: paid further objects, each inside the job's one $40 cap on its own row.

A further object that needs a generated model (ARTISAN, $12 per object) is reserved before the room is reconstructed,
in rank order, each reservation naming what the scored object still needs (``held_ahead_usd``) so the WebApp counts it
beside the $12 and a further object never takes the scored object's money (``website_build_budget.reserve_at_removal``).
Admitted, it leaves the scan with the job's object, is authored at compile under its own reservation and settled at
what it cost (``website_further_artisan``), and is composed beside the scored object. Refused, it and every paid one
ranked below it stay in the scan as filmed, named, under ``spending_cap_reached``: "the job reached its limit on paid
work". Its authoring failing releases its reservation and never touches the scored object. Without a host that can
author one (today's hosts), nothing is reserved and nothing changes.

Fixtures: ``tests.test_website_further_objects_removal``'s Atlas decision and handoff over the cup on its table.
Offline: the WebApp's ledger and the ARTISAN run are local fakes; nothing is paid.
"""
# Covers: src/blueprint_pipeline/website_build_budget.py
# Covers: src/blueprint_pipeline/website_further_artisan.py
# Covers: src/blueprint_pipeline/website_further_objects.py
# Covers: src/blueprint_pipeline/website_atlas_pipeline.py
# Covers: src/blueprint_pipeline/website_scene_handoff.py
# Covers: src/blueprint_pipeline/website_task_context.py
# Covers: src/blueprint_pipeline/website_company_readme.py
# Covers: src/blueprint_pipeline/website_build_plan_notes.py
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from blueprint_pipeline import paid_resource_admission as admission
from blueprint_pipeline import website_build_budget as budget
from blueprint_pipeline import website_further_artisan as artisan
from blueprint_pipeline import website_further_objects as further
from blueprint_pipeline import website_task_context
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.hard_stops import JOB_SPEND_CAP, webapp_refusal
from blueprint_pipeline.website_build_plan_notes import budget_root, plan_notes, sealed_budget
from blueprint_pipeline.website_job_spend_cap import SPENDING_CAP_REACHED, per_job_cap_usd
from tests import test_website_further_objects as fo
from tests import test_website_further_objects_removal as removal

pytest.importorskip("pxr")

#: The scene intake's allowance, held ahead while each further object is reserved.
ALLOWANCE = 20.0
NO_FREE_SOURCE = "further_object_needs_artisan:website_extra_object_no_free_source"


def _micros(usd):
    return math.ceil(usd * 1_000_000)


class Ledger:
    """The job's WebApp ledger as ``reserveWebsitePreparationSpend`` keeps it (WebApp stage 6): what the job spent
    (``spent``) and each row at its settlement, else its quote; a new quote, with the hold ahead it names, must fit under
    the cap (``checkWebsiteJobCeiling``), and the hold ahead is never held; a repeated binding answers its row.
    ``old``: a WebApp before stage 6, whose strict request schema refuses the field and writes nothing."""

    def __init__(self, *, spent=0.0, old=False):
        self.rows, self.spent, self.old, self.asked, self.settle_down = {}, spent, old, [], False

    def total_micros(self):
        return _micros(self.spent) + sum(_micros(row["settlement"]["actual_cost_usd"] if "settlement" in row
                                                 else row["admission"]["maximum_cost_usd"]) for row in self.rows.values())

    def __call__(self, *, task_context, operation, payload):
        self.asked.append((operation, copy.deepcopy(payload)))
        if operation == "preparation-spend":
            command, key = payload["spend"], payload["spend"]["allocation_binding_digest"][7:]
            if self.old and "held_ahead_usd" in command:
                raise ValueError("website_control_preparation-spend_http_400:website_scene_request_invalid")
            if key in self.rows:
                return {**self.rows[key]["admission"], "status": "already_reserved"}
            if (self.total_micros() + _micros(command["maximum_cost_usd"]) + _micros(command.get("held_ahead_usd", 0))
                    > _micros(float(per_job_cap_usd()))):
                raise webapp_refusal(f"website_control_preparation-spend_http_409:{JOB_SPEND_CAP}", JOB_SPEND_CAP)
            self.rows[key] = {"admission": {**command, "schema_version": "paid_lane_admission.v1",
                "status": "admitted", "blockers": [], "external_disclosure_allowed": True,
                "sponsorship_digest": "sha256:" + "5" * 64, "expires_at_epoch": 4_000_000_000}}
            return dict(self.rows[key]["admission"])
        assert operation == "preparation-settlement"
        if self.settle_down:
            raise ValueError("website_control_preparation-settlement_http_503:unclassified")
        command = payload["settlement"]
        row = self.rows[command["allocation_binding_digest"][7:]]
        assert (command["provider"], command["completed_request_count"]) == ("openai", 1)
        assert command["provider_charge_amount_usd"] <= row["admission"]["maximum_cost_usd"]
        row.setdefault("settlement", {**command, "actual_cost_usd": command["provider_charge_amount_usd"],
                                      "status": "settled"})
        return dict(row["settlement"])

    def reservations(self):
        return [payload["spend"] for operation, payload in self.asked if operation == "preparation-spend"]

    def settled(self):
        return sorted(row["settlement"]["actual_cost_usd"] for row in self.rows.values() if "settlement" in row)


class Runner:
    """The host's ARTISAN run for one further object, as ``website_further_artisan.installed_runner`` names its
    contract: a rigid body of the object's estimated size (here a plain carton), its spend kept in its meter as it goes.
    ``costs`` by target; ``fail`` the targets whose run stops after spending, with its meter complete or not."""

    def __init__(self, costs=None, fail=(), unread=()):
        self.costs, self.fail, self.unread, self.calls = dict(costs or {}), set(fail), set(unread), []

    def __call__(self, *, request, output_root, maximum_cost_usd):
        from blueprint_pipeline.website_parametric_object import author_asset, design
        target = request["target_id"]
        self.calls.append((request["rank"], target, maximum_cost_usd))
        output_root = Path(output_root)
        write_json(output_root / artisan.METER, {"schema_version": artisan.METER_SCHEMA,
                                                 "cost_usd": self.costs.get(target, 2.0),
                                                 "complete": target not in self.unread, "usage": {"calls": 3}})
        if target in self.fail:
            raise RuntimeError("artisan_framework_terminal_failure")
        receipt = author_asset(design("carton", request["dimensions_m"]), physics_bounds=request["physics_bounds"],
                               output_root=output_root)
        return output_root / receipt["asset"]["path"]


@pytest.fixture(autouse=True)
def _labels(monkeypatch):
    """The objects are labelled in plain object words, and each further object here needs a generated model."""
    monkeypatch.setitem(fo.LABELS, "artisan-1", "toy")
    for name in ("carton-1", "mug-1", "artisan-1"):
        monkeypatch.setitem(removal.SOURCING, name, ("generate", None))


@pytest.fixture
def paid(monkeypatch, tmp_path):
    """A host that can author a paid further object (``runner``), the job's ledger (``ledger``) on the real reservation
    path, and the scored object's needs as the Atlas job reads them: its intake's $20 allowance, nothing of the scan
    left. Returns a function that installs a ledger and a runner."""
    monkeypatch.setattr(website_task_context, "load_current_website_task_context",
                        lambda **kw: {"context_digest": fo.fx._task_context()["context_digest"]})
    monkeypatch.setattr(admission, "_account_monthly_limit_blockers", lambda **_: [])
    monkeypatch.setattr(admission, "_monthly_budget_blockers", lambda _resource_class: [])
    monkeypatch.delenv("BLUEPRINT_ARTISAN_AUTHORING_MAX_COST_USD", raising=False)
    monkeypatch.setenv("BLUEPRINT_PAID_SPEND_METER_ROOT", str(tmp_path / "meter"))
    held = {"value": ALLOWANCE}
    monkeypatch.setattr(budget, "scored_object_needs", lambda **_: {
        "scene_allowance_usd": held["value"], "scan_share_left_usd": 0.0, "held_ahead_usd": held["value"]})

    def install(ledger: Ledger, runner: Runner | None, *, held_ahead=ALLOWANCE):
        held["value"] = held_ahead
        monkeypatch.setattr(website_task_context, "preparation_webapp_request", ledger)
        monkeypatch.setattr(artisan, "installed_runner", lambda: runner)
        return ledger, runner
    return install


def _binding(context, name):
    return canonical_digest(budget.reservation_binding(task_context=context, row={"target_id": name,
                                                                                   "name": fo.LABELS[name]}))


# --- two paid further objects: reserved in order, authored, settled and composed ---------------------------------------

def test_two_paid_further_objects_are_reserved_in_order_authored_settled_and_composed(tmp_path, monkeypatch, paid):
    capture = tmp_path / "capture"
    names = ["carton-1", "artisan-1"]
    arguments = fo._job(capture / "pipeline" / "fixture", names)
    context = arguments["task_context"]
    # Nothing spent yet, $16 the scored object still needs: two $12 rows fit beside it, $40 in all.
    ledger, runner = paid(Ledger(), Runner({"carton-1": 3.25, "artisan-1": 4.5}), held_ahead=16.0)
    decided, consumed = removal._reviewed(capture, arguments, names, monkeypatch)
    # Both reserved before the room is reconstructed, in rank order, each its own $12 row with the hold ahead named.
    assert ledger.reservations() == [
        {"task_context_digest": context["context_digest"], "allocation_binding_digest": _binding(context, name),
         "resource_class": "openai_api_candidate", "provider": "openai", "maximum_cost_usd": 12.0, "request_count": 1,
         "held_ahead_usd": 16.0} for name in names]
    assert decided["further"] == [
        {"rank": 2, "target_id": "carton-1", "name": "carton", "editor_object_ids": ["e-carton-1"], "removed": True,
         "paid": {"binding_digest": _binding(context, "carton-1"), "maximum_cost_usd": 12.0}},
        {"rank": 3, "target_id": "artisan-1", "name": "toy", "editor_object_ids": ["e-artisan-1"], "removed": True,
         "paid": {"binding_digest": _binding(context, "artisan-1"), "maximum_cost_usd": 12.0}}]
    assert decided["kept_in_scan_target_ids"] == []
    # The scene editor erases all three with the words for several objects.
    assert [row["target_id"] for row in consumed["automation"]["objects"] if row["disposition"] == "remove"] == [
        "e-cup-1", "e-carton-1", "e-artisan-1"]
    record = sealed_budget(capture)
    assert [(row["rank"], row["status"]) for row in record["objects"]] == [
        (1, budget.FIRST), (2, budget.ADMITTED), (3, budget.ADMITTED)]
    assert record["held_ahead_usd"] == 16.0 and record["kept_at_cap"] == []
    # The handoff compiles: each authored with ARTISAN one at a time in rank order, each under what its row left.
    result = removal._handoff(capture, arguments, monkeypatch)
    assert result["status"] == "intake_ready", result["blockers"]  # the scored object's job
    assert runner.calls == [(2, "carton-1", 12.0), (3, "artisan-1", 12.0)]
    # Each row settled at what its authoring cost: the job's ledger counts $7.75 for the two, not $24.
    assert ledger.settled() == [3.25, 4.5] and ledger.total_micros() == 7_750_000
    value = json.loads(Path(result["preparation_path"]).read_text())
    composed = value["further_objects"]
    assert [(row["rank"], row["target_id"], row["prim_path"], row["source"]) for row in composed["objects"]] == [
        (2, "carton-1", "/World/Object_02", "artisan"), (3, "artisan-1", "/World/Object_03", "artisan")]
    assert [row["artisan_authoring"]["authoring_cost_usd"] for row in composed["objects"]] == [3.25, 4.5]
    assert composed["left_in_scan"] == [] and composed["flags"] == []
    assert [row["rank"] for row in value["intake_request"]["task"]["additional_objects"]] == [2, 3]
    assert value["subject"]["description"] == "cup"
    # A rerun authors nothing again and asks nothing more: each object's result and settlement are kept.
    asked = len(ledger.asked)
    assert removal._handoff(capture, arguments, monkeypatch)["status"] == "intake_ready"
    assert len(runner.calls) == 2 and len(ledger.asked) == asked
    # The package ships both and says what they are; the rule scores the cup alone.
    found = further.for_revision(capture, fo._revision(capture))
    published, unpacked = fo._published(tmp_path, fo._in_room(found), composed, decided)
    assert published["status"] == "published", published
    readme = fo._readme(unpacked)
    for sentence in (
            "Also built as their own objects, where the video shows them: carton and toy.",
            "The package also holds /World/Object_02 (carton): a 3D model Blueprint generated from frames of the "
            "video and sized from it: its shape and look are the generator's estimate, not a scan or a measurement.",
            "success_rule.json scores only the job's object, /World/Object."):
        assert sentence in readme, sentence
    assert "limit on paid work" not in readme and "Left in the scan" not in readme


# --- the cap refuses the second: it, and each paid one ranked below it, stay in the scan --------------------------------

def test_the_cap_refuses_the_second_and_it_and_every_lower_ranked_one_stay_in_the_scan(tmp_path, monkeypatch, paid):
    capture = tmp_path / "capture"
    names = ["carton-1", "mug-1", "artisan-1"]
    arguments = fo._job(capture / "pipeline" / "fixture", names)
    context = arguments["task_context"]
    # The scan holds $8 and the scored object needs $20: one $12 row fits, a second would take the scored one's money.
    ledger, runner = paid(Ledger(spent=8.0), Runner({"carton-1": 5.0}))
    decided, consumed = removal._reviewed(capture, arguments, names, monkeypatch)
    # Asked for the carton, then the mug box (refused); the toy, ranked below the refusal, is never asked.
    assert [spend["allocation_binding_digest"] for spend in ledger.reservations()] == [
        _binding(context, "carton-1"), _binding(context, "mug-1")]
    assert [(row["target_id"], row["removed"], row.get("kept_in_scan_reason")) for row in decided["further"]] == [
        ("carton-1", True, None), ("mug-1", False, SPENDING_CAP_REACHED), ("artisan-1", False, SPENDING_CAP_REACHED)]
    assert decided["kept_in_scan_target_ids"] == ["artisan-1", "mug-1"]
    assert [row["target_id"] for row in consumed["automation"]["objects"] if row["disposition"] == "remove"] == [
        "e-cup-1", "e-carton-1"]
    record = sealed_budget(capture)
    assert record["kept_at_cap"] == ["mug box", "toy"] and record["flags"] == [SPENDING_CAP_REACHED]
    refused, never = record["objects"][2], record["objects"][3]
    assert (refused["refused"], refused["refusal_code"]) == (True, JOB_SPEND_CAP)
    assert never["refused"] is False and "binding_digest" not in never
    # The first is built; the job's ledger never went past its $40, the scored object's $20 still free beside it.
    assert ledger.total_micros() + _micros(ALLOWANCE) == 40_000_000
    result = removal._handoff(capture, arguments, monkeypatch)
    assert result["status"] == "intake_ready", result["blockers"]
    assert result["build_plan"]["kept_at_spending_cap"] == ["mug box", "toy"]
    assert runner.calls == [(2, "carton-1", 12.0)] and ledger.settled() == [5.0]
    value = json.loads(Path(result["preparation_path"]).read_text())
    composed = value["further_objects"]
    assert [(row["target_id"], row["source"]) for row in composed["objects"]] == [("carton-1", "artisan")]
    assert composed["left_in_scan"] == [{"rank": 3, "target_id": "mug-1", "reason": SPENDING_CAP_REACHED},
                                        {"rank": 4, "target_id": "artisan-1", "reason": SPENDING_CAP_REACHED}]
    # The README says what is true of either limit on paid work, once for each, and never puts them anywhere else.
    from blueprint_pipeline import website_half_preparation as half_preparation
    told = half_preparation.further_objects(value, capture)["further_objects"]
    notes = plan_notes(decided, record, told)
    assert "Kept as filmed: mug box, the job reached its limit on paid work." in notes
    assert "Kept as filmed: toy, the job reached its limit on paid work." in notes
    assert not any("Left in the scan" in note or "keeps these in the scan" in note for note in notes)
    assert "Also built as its own object, where the video shows it: carton." in notes
    # The company's package says the same: its README reads the job's sealed decision and build budget.
    from tests.test_website_company_simulation import CAPTURE, SITE
    shipped = tmp_path / "download" / f"storage/source-bucket/scenes/{SITE}/captures/{CAPTURE}"
    write_json(budget_root(shipped) / "build_budget.json", record)
    published, unpacked = fo._published(tmp_path, fo._in_room(further.for_revision(capture, fo._revision(capture))),
                                        composed, decided)
    assert published["status"] == "published", published
    readme = fo._readme(unpacked)
    assert "Kept as filmed: mug box, the job reached its limit on paid work." in readme
    assert "Kept as filmed: toy, the job reached its limit on paid work." in readme
    assert "spending cap" not in readme and "$" not in readme.split("Kept as filmed", 1)[1].split(".", 1)[0]


# --- authoring a paid further object fails: its row is released, it is named, the scored object goes on ----------------

def test_a_paid_further_objects_authoring_failing_releases_its_reservation_and_never_touches_the_scored_one(
        tmp_path, monkeypatch, paid):
    capture = tmp_path / "capture"
    names = ["carton-1", "artisan-1"]
    arguments = fo._job(capture / "pipeline" / "fixture", names)
    ledger, runner = paid(Ledger(), Runner({"carton-1": 1.5, "artisan-1": 2.0}, fail={"carton-1"}), held_ahead=16.0)
    decided, _consumed = removal._reviewed(capture, arguments, names, monkeypatch)
    result = removal._handoff(capture, arguments, monkeypatch)
    assert result["status"] == "intake_ready", result["blockers"]
    # The carton's run stopped after spending $1.50: its row settles there, the other $10.50 released at once.
    assert ledger.settled() == [1.5, 2.0]
    value = json.loads(Path(result["preparation_path"]).read_text())
    composed = value["further_objects"]
    assert [row["target_id"] for row in composed["objects"]] == ["artisan-1"]  # the next one is still built
    assert composed["removed_not_built"] == [{"rank": 2, "target_id": "carton-1", "reason":
        "further_object_generation_failed:runner_failed:RuntimeError"}]
    assert composed["flags"] == [further.REMOVED_NOT_BUILT]
    assert value["subject"]["description"] == "cup" and value["status"] == "intake_ready"
    # A failed run is final: a rerun never asks the runner again for it, nor spends under its released row.
    removal._handoff(capture, arguments, monkeypatch)
    assert runner.calls == [(2, "carton-1", 12.0), (3, "artisan-1", 12.0)]
    from blueprint_pipeline import website_half_preparation as half_preparation
    told = half_preparation.further_objects(value, capture)["further_objects"]
    assert ("Taken out of the scan to be built as its own object, but it could not be built, so it is not in this "
            "simulation: carton, because Blueprint could not generate a model of it.") in plan_notes(decided, None, told)


def test_a_run_that_stopped_part_way_resumes_with_only_what_its_reservation_left(tmp_path, monkeypatch, paid):
    capture = tmp_path / "capture"
    names = ["carton-1"]
    arguments = fo._job(capture / "pipeline" / "fixture", names)
    ledger, runner = paid(Ledger(), Runner({"carton-1": 7.0}))
    decided, _consumed = removal._reviewed(capture, arguments, names, monkeypatch)
    # A run started and stopped (its host restarted) after spending $5, every call of it priced.
    work = artisan.authoring_root(budget_root(capture), decided["further"][0]["paid"]["binding_digest"])
    artisan._seal(work / artisan.STARTED, {"schema_version": "website_further_object_artisan_started.v1"})
    write_json(work / "run" / artisan.METER, {"schema_version": artisan.METER_SCHEMA, "cost_usd": 5.0,
                                              "complete": True, "usage": {"calls": 2}})
    assert removal._handoff(capture, arguments, monkeypatch)["status"] == "intake_ready"
    # Resumed under the same reservation with the $7 its frozen $12 cap left, and settled at the whole $7 it cost.
    assert runner.calls == [(2, "carton-1", 7.0)] and ledger.settled() == [7.0]
    assert len(ledger.reservations()) == 1  # never reserved again


def test_a_run_whose_spend_cannot_be_read_keeps_its_whole_hold(tmp_path, monkeypatch, paid):
    capture = tmp_path / "capture"
    names = ["carton-1"]
    arguments = fo._job(capture / "pipeline" / "fixture", names)
    ledger, _runner = paid(Ledger(), Runner(fail={"carton-1"}, unread={"carton-1"}))
    removal._reviewed(capture, arguments, names, monkeypatch)
    assert removal._handoff(capture, arguments, monkeypatch)["status"] == "intake_ready"
    # Unknown usage: the row keeps its $12 quote (over, never under).
    assert ledger.settled() == [] and ledger.total_micros() == 12_000_000


def test_a_settlement_the_webapp_cannot_take_now_keeps_the_hold_and_the_object_is_still_built(tmp_path, monkeypatch,
                                                                                               paid):
    capture = tmp_path / "capture"
    names = ["carton-1"]
    arguments = fo._job(capture / "pipeline" / "fixture", names)
    ledger, runner = paid(Ledger(), Runner({"carton-1": 3.0}))
    removal._reviewed(capture, arguments, names, monkeypatch)
    ledger.settle_down = True
    result = removal._handoff(capture, arguments, monkeypatch)
    composed = json.loads(Path(result["preparation_path"]).read_text())["further_objects"]
    assert [row["target_id"] for row in composed["objects"]] == ["carton-1"]
    assert ledger.settled() == [] and ledger.total_micros() == 12_000_000  # its whole quote: over, never under
    # The next pass settles it at what it cost, and authors nothing again.
    ledger.settle_down = False
    removal._handoff(capture, arguments, monkeypatch)
    assert ledger.settled() == [3.0] and len(runner.calls) == 1


# --- the hold ahead: a WebApp that cannot count it reserves nothing; one that can never lets an extra take it ----------

def test_a_webapp_that_cannot_count_the_hold_ahead_reserves_nothing_and_the_objects_stay_as_filmed(
        tmp_path, monkeypatch, paid):
    capture = tmp_path / "capture"
    names = ["carton-1", "artisan-1"]
    arguments = fo._job(capture / "pipeline" / "fixture", names)
    ledger, runner = paid(Ledger(old=True), Runner())
    decided, _consumed = removal._reviewed(capture, arguments, names, monkeypatch)
    # Asked once, refused before anything was written; nothing lower is asked; each stays as stage 5 left it.
    assert len(ledger.reservations()) == 1 and ledger.rows == {}
    assert [(row["removed"], row["kept_in_scan_reason"]) for row in decided["further"]] == [
        (False, NO_FREE_SOURCE), (False, NO_FREE_SOURCE)]
    record = sealed_budget(capture)
    assert [row["status"] for row in record["objects"][1:]] == [budget.UNRESERVED, budget.UNRESERVED]
    assert {row["reason"] for row in record["objects"][1:]} == {budget.HOLD_UNCONFIRMED}
    assert record["kept_at_cap"] == [] and "flags" not in record  # no limit is claimed
    assert not list((budget_root(capture) / "reservations").glob("*.json"))


def test_the_webapp_refuses_an_extra_that_would_take_the_scored_objects_allowance(tmp_path, paid):
    """The ledger's arithmetic is the WebApp's (``checkWebsiteJobCeiling`` with the hold ahead): without it the $12
    would fit beside the $8 the scan spent, and then the scored object's $20 intake would not."""
    context = fo.fx._task_context()
    ledger, _runner = paid(Ledger(spent=20.0), Runner())
    rows = [{"target_id": "cup-1", "name": "cup", "sourcing": "generate"},
            {"target_id": "carton-1", "name": "carton", "sourcing": "generate", "rank": 2}]
    record = budget.admit_extra_objects(rows, task_context=context, build_plan_digest="sha256:" + "a" * 64,
                                        root=tmp_path, routes={"carton-1": "generate"}, held_ahead_usd=ALLOWANCE)
    assert record["objects"][1]["status"] == budget.KEPT and record["kept_at_cap"] == ["carton"]
    assert ledger.rows == {} and ledger.total_micros() + _micros(ALLOWANCE) == 40_000_000


def test_the_hold_ahead_is_the_intakes_allowance_and_the_scans_share_left_read_once(tmp_path, monkeypatch):
    from blueprint_pipeline.website_task_context import RETAINED_GRANT_FILE
    context = {**fo.fx._task_context()}
    capture = tmp_path / "capture"
    grant = {"schema_version": "website_scene_sponsorship.v1", "max_total_spend_usd": 20.0,
             **{key: context[key] for key in ("request_id", "scene_id", "capture_id")}}
    grant["authority_digest"] = canonical_digest(grant, digest_field="authority_digest")
    write_json(capture / RETAINED_GRANT_FILE, grant)
    shares = iter([6.5, 1.0])
    monkeypatch.setattr(website_task_context, "load_atlas_configuration",
                        lambda *_a, **_k: {"remaining_cost_usd": next(shares)})
    source = "sha256:" + "b" * 64
    assert budget.scored_object_needs(capture_root=capture, task_context=context, source_digest=source) == {
        "scene_allowance_usd": 20.0, "scan_share_left_usd": 6.5, "held_ahead_usd": 26.5}
    root = budget_root(capture)
    frozen = budget._hold_ahead(root, capture_root=capture, task_context=context, source_digest=source,
                                now=lambda: 1.0)
    assert frozen == 21.0  # the second read: 20 + 1
    again = budget._hold_ahead(root, capture_root=capture, task_context=context, source_digest=source,
                               now=lambda: 2.0)
    assert again == 21.0  # frozen once: the same command each time it is asked
    # No grant, another job's, or no Atlas source: nothing to hold ahead, so nothing is reserved.
    assert budget.scored_object_needs(capture_root=tmp_path / "none", task_context=context, source_digest=source) is None
    assert budget.scored_object_needs(capture_root=capture, task_context={**context, "request_id": "other"},
                                      source_digest=source) is None
    assert budget.scored_object_needs(capture_root=capture, task_context=context, source_digest=None) is None


# --- a room package, and a decision the editor could not erase, settle every reservation -------------------------------

def test_a_room_package_and_an_erase_that_failed_release_each_reservation(tmp_path, monkeypatch, paid):
    capture = tmp_path / "capture"
    names = ["carton-1"]
    arguments = fo._job(capture / "pipeline" / "fixture", names)
    ledger, runner = paid(Ledger(), Runner())
    removal._reviewed(capture, arguments, names, monkeypatch)
    assert ledger.total_micros() == 12_000_000
    artisan.close_all(capture, task_context=arguments["task_context"])  # as the handoff's room package does
    assert ledger.settled() == [0.0] and ledger.total_micros() == 0 and runner.calls == []
    # Released, the object is never authored under it: compile names it not built.
    result = removal._handoff(capture, arguments, monkeypatch)
    assert result["status"] == "intake_ready" and runner.calls == []
    composed = json.loads(Path(result["preparation_path"]).read_text())["further_objects"]
    assert composed["removed_not_built"] == [{"rank": 2, "target_id": "carton-1",
                                              "reason": "further_object_generation_failed:reservation_settled"}]


def test_a_paid_object_the_editor_could_not_erase_stays_in_the_scan_and_its_row_is_released(tmp_path, monkeypatch, paid):
    from blueprint_pipeline import website_atlas_pipeline as atlas
    capture = tmp_path / "capture"
    names = ["carton-1"]
    arguments = fo._job(capture / "pipeline" / "fixture", names)
    ledger, runner = paid(Ledger(), Runner())
    removal._reviewed(capture, arguments, names, monkeypatch)
    assert atlas._erase_alone(capture / "pipeline" / "atlas", removal.FLOOR) is True
    result = removal._handoff(capture, arguments, monkeypatch)
    assert result["status"] == "intake_ready" and runner.calls == []
    assert ledger.settled() == [0.0]
    composed = json.loads(Path(result["preparation_path"]).read_text())["further_objects"]
    assert composed["left_in_scan"] == [{"rank": 2, "target_id": "carton-1",
                                         "reason": "further_object_not_erasable:atlas_removal_visibility_incomplete"}]


# --- without a host that can author one, or without a paid one, nothing changes -----------------------------------------

def test_without_a_runner_a_paid_further_object_stays_in_the_scan_and_nothing_is_asked(tmp_path, monkeypatch, paid):
    capture = tmp_path / "capture"
    names = ["carton-1", "artisan-1"]
    arguments = fo._job(capture / "pipeline" / "fixture", names)
    ledger, _runner = paid(Ledger(), None)  # today's hosts
    decided, _consumed = removal._reviewed(capture, arguments, names, monkeypatch)
    assert ledger.asked == [] and not budget_root(capture).exists()
    assert decided["further"] == [
        {"rank": 2, "target_id": "carton-1", "name": "carton", "editor_object_ids": ["e-carton-1"],
         "removed": False, "kept_in_scan_reason": NO_FREE_SOURCE},
        {"rank": 3, "target_id": "artisan-1", "name": "toy", "editor_object_ids": ["e-artisan-1"],
         "removed": False, "kept_in_scan_reason": NO_FREE_SOURCE}]


def test_the_live_host_has_no_runner():
    assert artisan.installed_runner() is None and artisan.installed() is False


def test_a_job_with_only_free_further_objects_is_byte_for_byte_as_stage_5_sealed_it(tmp_path, monkeypatch, paid):
    names = ["carton-1", "mug-1"]
    for name in names:
        monkeypatch.setitem(removal.SOURCING, name, ("parametric", "carton"))
    sealed = {}
    for label, runner in (("stage5", None), ("stage6", Runner())):
        capture = tmp_path / label
        arguments = fo._job(capture / "pipeline" / "fixture", names)
        ledger, _ = paid(Ledger(), runner)
        decided, consumed = removal._reviewed(capture, arguments, names, monkeypatch)
        assert ledger.asked == [] and not budget_root(capture).exists()
        result = removal._handoff(capture, arguments, monkeypatch)
        value = json.loads(Path(result["preparation_path"]).read_text())
        sealed[label] = (json.dumps({key: decided[key] for key in ("further", "kept_in_scan_target_ids", "built")},
                                    sort_keys=True), json.dumps(consumed, sort_keys=True),
                         canonical_digest(value["further_objects"]["objects"][0]["asset_receipt"]))
    assert sealed["stage5"] == sealed["stage6"]
