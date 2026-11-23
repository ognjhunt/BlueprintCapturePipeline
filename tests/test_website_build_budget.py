"""The build plan's dollar budget (owner decision 2026-11-23): the job's cap stays $40; the first object builds under
the scene intake's allowance as today; each extra object the plan generates reserves its own ARTISAN cap on the job's
ledger, in the plan's order, through the existing preparation reservation; free objects reserve nothing; at the cap the
refused object and every lower-ranked paid one are kept as filmed, named, and the job goes on."""
from __future__ import annotations

import json
import math

import pytest

from blueprint_pipeline import paid_resource_admission as admission
from blueprint_pipeline import website_build_budget as budget
from blueprint_pipeline import website_task_context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.hard_stops import JOB_SPEND_CAP, webapp_refusal
from blueprint_pipeline.website_build_plan_notes import (DECISION_SCHEMA, JOB_OBJECT_RECORD, KEPT_AT_CAP_NOTE,
                                                         REFUSAL_CODE_MEANS_ONE_OF, budget_root, build_plan_notes,
                                                         kept_at_cap, kept_disclosure, plan_notes)
from blueprint_pipeline.website_job_spend_cap import SPENDING_CAP_REACHED, per_job_cap_usd

CONTEXT = {"context_digest": "sha256:" + "1" * 64, "request_id": "req-1", "scene_id": "site-req-1",
           "capture_id": "walkthrough-req-1"}
PLAN_DIGEST = "sha256:" + "a" * 64


def _object(target, name, sourcing):
    return {"target_id": target, "name": name, "role": "moved_object", "variety": None, "rigidity": "rigid",
            "size_m": None, "mass_kg": None, "sourcing": sourcing,
            "parametric_shape": "open_bin" if sourcing == "parametric" else None,
            "reason": f"The job needs the {name}.", "editor_object_ids": []}


#: The plan's objects the build builds, most important first: four generated (paid), two free.
OBJECTS = [_object("tote", "blue tote", "generate"), _object("carton", "carton", "generate"),
           _object("bin", "open bin", "parametric"), _object("crate", "crate", "generate"),
           _object("mug", "mug", "library"), _object("drill", "drill", "generate")]


def _micros(usd):
    return math.ceil(usd * 1_000_000)


class Ledger:
    """The job's WebApp ledger as ``reserveWebsitePreparationSpend`` and ``settleWebsitePreparationSpend`` keep it:
    each row at its settlement, else its quote, beside the scene intake's allowance held in full; a new quote must fit
    under the cap (``checkWebsiteJobCeiling``) and the native rows' paid-request count under its limit
    (``preparationRequestLimit``), else the one typed refusal the WebApp sends for both; a repeated binding answers the
    row it holds (``already_reserved``)."""

    def __init__(self, *, scan=8.0, intake=20.0, request_limit=32):
        self.rows = {"scan": {"admission": {"maximum_cost_usd": scan, "atlas_authorization_digest": "sha256:atlas"}}}
        self.intake, self.request_limit, self.asked, self.down = intake, request_limit, [], False

    def requests(self):
        return sum(row["admission"]["request_count"] for row in self.rows.values()
                   if "atlas_authorization_digest" not in row["admission"])

    def total_micros(self):
        return _micros(self.intake) + sum(_micros(row["settlement"]["actual_cost_usd"] if "settlement" in row
                                                  else row["admission"]["maximum_cost_usd"]) for row in self.rows.values())

    def __call__(self, *, task_context, operation, payload):
        assert task_context == CONTEXT
        self.asked.append((operation, json.loads(json.dumps(payload))))
        if self.down:
            raise ValueError(f"website_control_{operation}_http_503:unclassified")
        if operation == "preparation-spend":
            command, key = payload["spend"], payload["spend"]["allocation_binding_digest"][7:]
            if key in self.rows:
                return {**self.rows[key]["admission"], "status": "already_reserved"}
            if (self.requests() + command["request_count"] > self.request_limit
                    or self.total_micros() + _micros(command["maximum_cost_usd"]) > _micros(float(per_job_cap_usd()))):
                raise webapp_refusal(f"website_control_preparation-spend_http_409:{JOB_SPEND_CAP}", JOB_SPEND_CAP)
            self.rows[key] = {"admission": {**command, "schema_version": "paid_lane_admission.v1",
                "status": "admitted", "blockers": [], "external_disclosure_allowed": True,
                "sponsorship_digest": "sha256:" + "5" * 64, "expires_at_epoch": 4_000_000_000}}
            return dict(self.rows[key]["admission"])
        assert operation == "preparation-settlement"
        command = payload["settlement"]
        row = self.rows[command["allocation_binding_digest"][7:]]
        assert (command["provider"], command["completed_request_count"]) == ("openai", row["admission"]["request_count"])
        assert command["task_context_digest"] == row["admission"]["task_context_digest"]
        assert command["provider_charge_amount_usd"] <= row["admission"]["maximum_cost_usd"]
        row.setdefault("settlement", {**command, "actual_cost_usd": command["provider_charge_amount_usd"],
                                      "status": "settled"})
        return dict(row["settlement"])

    def reservations(self):
        return [payload["spend"] for operation, payload in self.asked if operation == "preparation-spend"]


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    """The real reservation path (``reserve_website_preparation_spend`` and its paid chokepoint) onto ``Ledger``."""
    value = Ledger()
    monkeypatch.setattr(website_task_context, "preparation_webapp_request", value)
    monkeypatch.setattr(admission, "_account_monthly_limit_blockers", lambda: [])
    monkeypatch.setattr(admission, "_monthly_budget_blockers", lambda _resource_class: [])
    monkeypatch.delenv("BLUEPRINT_ARTISAN_AUTHORING_MAX_COST_USD", raising=False)
    monkeypatch.setenv("BLUEPRINT_PAID_SPEND_METER_ROOT", str(tmp_path / "meter"))
    return value


def _admit(root, objects=OBJECTS):
    return budget.admit_extra_objects(objects, task_context=CONTEXT, build_plan_digest=PLAN_DIGEST, root=root)


def _binding(name):
    row = next(row for row in OBJECTS if row["name"] == name)
    return canonical_digest(budget.reservation_binding(task_context=CONTEXT, build_plan_digest=PLAN_DIGEST, row=row))


def test_each_extra_paid_object_reserves_its_own_share_in_the_plans_order_until_the_cap(tmp_path, ledger):
    # The scan holds $8 and the scene intake's allowance (the first object's) $20: $12 is left under the $40.
    record = _admit(tmp_path)
    assert [(row["name"], row["status"]) for row in record["objects"]] == [
        ("blue tote", budget.FIRST), ("carton", budget.ADMITTED), ("open bin", budget.FREE),
        ("crate", budget.KEPT), ("mug", budget.FREE), ("drill", budget.KEPT)]
    # Only the paid extras were asked, one at a time in rank order, each through the one OpenAI gate at its frozen
    # $12 ARTISAN cap: never the first object, never a free one, never one ranked below a refusal.
    assert ledger.reservations() == [
        {"task_context_digest": CONTEXT["context_digest"], "allocation_binding_digest": _binding(name),
         "resource_class": "openai_api_candidate", "provider": "openai", "maximum_cost_usd": 12.0,
         "request_count": 1} for name in ("carton", "crate")]
    # Nothing spends past the cap: the carton's $12 fills it exactly.
    assert ledger.total_micros() == 40_000_000
    crate, drill = record["objects"][3], record["objects"][5]
    assert crate == {"rank": 3, "target_id": "crate", "name": "crate", "sourcing": "generate", "status": budget.KEPT,
                     "reason": SPENDING_CAP_REACHED, "refused": True, "binding_digest": _binding("crate"),
                     "refusal_code": JOB_SPEND_CAP, "refusal_code_means_one_of": ["job_dollar_cap", "paid_request_count"]}
    # The drill was never asked: it is kept under the crate's refusal, its code as received.
    assert drill["refused"] is False and "binding_digest" not in drill
    assert (drill["refusal_code"], drill["refusal_code_means_one_of"]) == (JOB_SPEND_CAP, list(REFUSAL_CODE_MEANS_ONE_OF))
    assert record["objects"][1]["maximum_cost_usd"] == 12.0
    assert record["objects"][1]["authoring_budget"]["scope"] == "whole_authoring_stage_cumulative_including_resume"
    assert record["kept_at_cap"] == ["crate", "drill"] and record["flags"] == [SPENDING_CAP_REACHED]
    assert record["digest"] == canonical_digest(record, digest_field="digest")
    assert json.loads((tmp_path / "build_budget.json").read_text()) == record
    # The refusal is the object's own record: the website's code exactly as received, and both limits it may mean.
    refused = json.loads((tmp_path / "reservations" / (_binding("crate")[7:] + ".json")).read_text())
    assert refused["status"] == budget.REFUSED and refused["refusal_code"] == "website_scene_preparation_budget_exhausted"
    assert refused["refusal_code_means_one_of"] == ["job_dollar_cap", "paid_request_count"]


def test_a_retry_or_a_resume_reuses_each_reservation_and_never_reserves_twice(tmp_path, ledger, monkeypatch):
    first = _admit(tmp_path)
    asked = len(ledger.asked)
    # The same run again, and a later one whose host now freezes a larger cap: each reuses what was written.
    monkeypatch.setenv("BLUEPRINT_ARTISAN_AUTHORING_MAX_COST_USD", "15")
    assert _admit(tmp_path) == first and _admit(tmp_path) == first
    assert len(ledger.asked) == asked
    # A shorter list (a build resumed part way) reads the same reservations.
    assert [row["status"] for row in _admit(tmp_path, OBJECTS[:2])["objects"]] == [budget.FIRST, budget.ADMITTED]
    assert len(ledger.asked) == asked and len([key for key in ledger.rows if key != "scan"]) == 1


def test_an_outage_is_raised_and_the_retry_asks_only_for_what_was_never_admitted(tmp_path, ledger, monkeypatch):
    original = Ledger.__call__

    def flaky(self, **kwargs):  # the carton is admitted, then the WebApp is unreachable for the crate
        if self.reservations():
            self.down = True
        return original(self, **kwargs)
    monkeypatch.setattr(Ledger, "__call__", flaky)
    with pytest.raises(ValueError, match="_http_503:"):
        _admit(tmp_path)
    assert not (tmp_path / "reservations" / (_binding("crate")[7:] + ".json")).exists()
    assert not (tmp_path / "build_budget.json").exists()
    monkeypatch.setattr(Ledger, "__call__", original)
    ledger.down = False
    record = _admit(tmp_path)
    # The carton's row was reused, the crate asked once more and refused at the cap.
    assert [spend["allocation_binding_digest"] for spend in ledger.reservations()] == [
        _binding("carton"), _binding("crate"), _binding("crate")]
    assert record["kept_at_cap"] == ["crate", "drill"]


def test_the_accounts_month_is_a_wait_not_the_jobs_cap(tmp_path, ledger, monkeypatch):
    """The account's monthly limit stays a wait (it is no job's cap): it is raised before the website is asked, and
    nothing is kept as filmed for it."""
    monkeypatch.setattr(admission, "_account_monthly_limit_blockers", lambda: ["account_monthly_spend_limit_reached"])
    with pytest.raises(admission.PaidResourceAdmissionBlocked, match="account_monthly_spend_limit_reached"):
        _admit(tmp_path)
    assert ledger.asked == [] and not list((tmp_path / "reservations").glob("*.json"))


def test_a_row_the_website_admitted_stays_reserved_when_the_local_recheck_then_refuses(tmp_path, ledger, monkeypatch):
    calls = []

    def month():  # clear at the precheck, reached by the time the admission is rechecked
        calls.append(1)
        return [] if len(calls) == 1 else ["account_monthly_spend_limit_reached"]
    monkeypatch.setattr(admission, "_account_monthly_limit_blockers", month)
    with pytest.raises(admission.PaidResourceAdmissionBlocked):
        _admit(tmp_path)
    written = json.loads((tmp_path / "reservations" / (_binding("carton")[7:] + ".json")).read_text())
    assert written["status"] == budget.ADMITTED and written["admission"]["status"] == "admitted"
    monkeypatch.setattr(admission, "_account_monthly_limit_blockers", lambda: [])
    record = _admit(tmp_path)
    # The retry never asked for the carton again: its row was already held.
    assert [spend["allocation_binding_digest"] for spend in ledger.reservations()] == [
        _binding("carton"), _binding("crate")]
    assert record["objects"][1]["status"] == budget.ADMITTED


def test_a_partly_spent_reservation_settles_at_its_cost_and_a_refusal_stays_final(tmp_path, ledger):
    record = _admit(tmp_path)
    carton = record["objects"][1]
    usage = {"calls": 14, "input_tokens": 120_000, "output_tokens": 9_000}
    receipt = budget.settle_object(binding_digest=carton["binding_digest"], task_context=CONTEXT, root=tmp_path,
                                   cost_usd=4.3712345, usage=usage)
    assert receipt["actual_cost_usd"] == 4.371234 and receipt["status"] == "settled"
    assert receipt["usage_receipt_digest"] == canonical_digest({"usage": usage})
    # The job's ledger now counts what the carton cost, not its $12 quote.
    assert ledger.total_micros() == _micros(8) + _micros(20) + _micros(4.371234)
    asked = len(ledger.asked)
    assert budget.settle_object(binding_digest=carton["binding_digest"], task_context=CONTEXT, root=tmp_path,
                                cost_usd=9.0, usage=usage) == receipt
    # The money it freed brings no refused object back.
    assert _admit(tmp_path)["kept_at_cap"] == ["crate", "drill"] and len(ledger.asked) == asked
    for binding in (_binding("crate"), _binding("drill")):
        with pytest.raises(ValueError, match=budget.SETTLEMENT_INVALID):
            budget.settle_object(binding_digest=binding, task_context=CONTEXT, root=tmp_path, cost_usd=1, usage={})


def test_a_settlement_never_passes_its_reservation(tmp_path, ledger):
    carton = _admit(tmp_path)["objects"][1]
    for cost in (-1, float("nan"), True, "4"):
        with pytest.raises(ValueError, match=budget.SETTLEMENT_INVALID):
            budget.settle_object(binding_digest=carton["binding_digest"], task_context=CONTEXT, root=tmp_path,
                                 cost_usd=cost, usage={})
    assert budget.settle_object(binding_digest=carton["binding_digest"], task_context=CONTEXT, root=tmp_path,
                                cost_usd=15.2, usage={})["actual_cost_usd"] == 12.0


def test_a_row_whose_record_was_lost_is_never_reserved_twice(tmp_path, ledger):
    """The website keys each row by the object's binding: asked again without the record, it answers the row it
    holds, which pays for nothing, so the same object never holds two rows (an edge every preparation lane shares)."""
    _admit(tmp_path / "first")
    with pytest.raises(admission.PaidResourceAdmissionBlocked, match="paid_resource_admission_not_admitted"):
        _admit(tmp_path / "lost")
    assert len([key for key in ledger.rows if key != "scan"]) == 1 and ledger.total_micros() == 40_000_000


def test_a_resumed_run_spends_only_what_its_reservation_left(tmp_path, ledger):
    entry = _admit(tmp_path)["objects"][1]
    assert budget.authoring_allocation(entry, None) == 12.0
    assert budget.authoring_allocation(entry, {"cost_usd": 5.0, "completed": False}) == 7.0
    assert budget.authoring_allocation(entry, {"cost_usd": 11.234, "completed": False}) == 0.76
    assert budget.authoring_allocation(entry, {"cost_usd": 11.234, "completed": True}) == 0.01
    with pytest.raises(ValueError, match=budget.OBJECTS_INVALID):
        budget.authoring_allocation(_admit(tmp_path)["objects"][3], None)


def test_the_first_object_and_free_objects_reserve_nothing(tmp_path, ledger):
    record = _admit(tmp_path, [OBJECTS[1], OBJECTS[2], OBJECTS[4]])
    assert [row["status"] for row in record["objects"]] == [budget.FIRST, budget.FREE, budget.FREE]
    assert ledger.asked == [] and record["kept_at_cap"] == [] and "flags" not in record


def test_a_paid_object_reserved_before_a_refusal_ranked_above_it_stays_admitted(tmp_path, ledger):
    """An admitted reservation holds its money and keeps it, whatever the order a later list gives."""
    _admit(tmp_path, OBJECTS[:2])  # the carton's $12 fills the cap
    record = _admit(tmp_path, [OBJECTS[0], OBJECTS[3], OBJECTS[1]])
    assert [(row["name"], row["status"]) for row in record["objects"]] == [
        ("blue tote", budget.FIRST), ("crate", budget.KEPT), ("carton", budget.ADMITTED)]


@pytest.mark.parametrize("objects, digest", [
    ([], PLAN_DIGEST), ([OBJECTS[0], OBJECTS[0]], PLAN_DIGEST),
    ([OBJECTS[0], {**OBJECTS[1], "sourcing": "keep_as_filmed"}], PLAN_DIGEST),
    ([{k: v for k, v in OBJECTS[0].items() if k != "name"}], PLAN_DIGEST), ([OBJECTS[0], "carton"], PLAN_DIGEST),
    (OBJECTS, "sha256:short"),
])
def test_objects_that_cannot_be_budgeted_are_refused_before_anything_is_asked(tmp_path, ledger, objects, digest):
    with pytest.raises(ValueError, match=budget.OBJECTS_INVALID):
        budget.admit_extra_objects(objects, task_context=CONTEXT, build_plan_digest=digest, root=tmp_path)
    assert ledger.asked == []


def test_a_tampered_reservation_fails_closed_and_asks_nothing(tmp_path, ledger):
    _admit(tmp_path)
    path = tmp_path / "reservations" / (_binding("carton")[7:] + ".json")
    path.write_text(json.dumps({**json.loads(path.read_text()), "maximum_cost_usd": 1.0}))
    asked = len(ledger.asked)
    with pytest.raises(ValueError, match=budget.RECORD_INVALID):
        _admit(tmp_path)
    assert len(ledger.asked) == asked


def _decided(capture_root, plan_objects, built):
    record = {"schema_version": DECISION_SCHEMA, "applied": False, "clean_plate_status": "objects_removed",
              "planner_remove_count": 2, "build_plan": {"digest": PLAN_DIGEST, "plan": {"objects": plan_objects}},
              "built": built, "kept_in_scan_target_ids": []}
    record["digest"] = canonical_digest(record, digest_field="digest")
    path = capture_root / "pipeline" / "atlas" / JOB_OBJECT_RECORD
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record))
    return record


def test_the_readme_names_each_object_the_cap_kept_out(tmp_path, ledger):
    decided = _decided(tmp_path, OBJECTS, {"target_id": "tote", "name": "blue tote", "sourcing": "generate",
                                           "editor_object_ids": ["e"]})
    record = _admit(budget_root(tmp_path))
    assert kept_at_cap(record, decided) == ["crate", "drill"]
    assert build_plan_notes(tmp_path)["plan_notes"] == [
        "What this simulation needs, as Blueprint's model read the job: blue tote, carton, open bin, crate, mug and "
        "drill.",
        "The plan takes this out of the scan to build as its own object: blue tote.",
        "The plan keeps these in the scan as filmed, not built on their own: carton, open bin and mug.",
        "Kept as filmed: crate, the job reached its limit on paid work.",
        "Kept as filmed: drill, the job reached its limit on paid work."]
    assert KEPT_AT_CAP_NOTE.format(name="crate") in build_plan_notes(tmp_path)["plan_notes"]
    # A budget made for another plan is never read; without one the notes are today's.
    assert kept_at_cap({**record, "build_plan_digest": "sha256:" + "b" * 64}, decided) == []
    assert plan_notes(decided, {**record, "build_plan_digest": "sha256:" + "b" * 64}) == plan_notes(decided)
    assert not any("limit on paid work" in note for note in plan_notes(decided))
    # A record that does not seal itself is not read.
    path = budget_root(tmp_path) / "build_budget.json"
    path.write_text(json.dumps({**record, "kept_at_cap": []}))
    assert not any("limit on paid work" in note for note in build_plan_notes(tmp_path)["plan_notes"])


def test_a_refusal_by_the_paid_request_count_reads_exactly_as_one_by_the_dollar_cap(tmp_path, monkeypatch):
    """The WebApp refuses its paid-request count with the dollar cap's code, so the Pipeline cannot tell them apart:
    every record keeps the code as received with both limits it may mean, and every sentence is true of either."""
    def kept_by(ledger, root):
        monkeypatch.setattr(website_task_context, "preparation_webapp_request", ledger)
        decided = _decided(root, OBJECTS, {"target_id": "tote", "name": "blue tote", "sourcing": "generate",
                                           "editor_object_ids": ["e"]})
        record = _admit(budget_root(root))
        kept = [{key: row[key] for key in ("name", "reason", "refused", "refusal_code", "refusal_code_means_one_of")}
                for row in record["objects"] if row["status"] == budget.KEPT]
        return kept, kept_disclosure(record, decided), build_plan_notes(root)["plan_notes"]
    monkeypatch.setattr(admission, "_account_monthly_limit_blockers", lambda: [])
    monkeypatch.setattr(admission, "_monthly_budget_blockers", lambda _resource_class: [])
    monkeypatch.delenv("BLUEPRINT_ARTISAN_AUTHORING_MAX_COST_USD", raising=False)
    # $28 of $40 held, so one $12 fits; and a job with $40 free whose request count allows one more call.
    dollars, counted = Ledger(), Ledger(scan=0.0, intake=0.0, request_limit=1)
    by_dollars, by_count = kept_by(dollars, tmp_path / "dollars"), kept_by(counted, tmp_path / "count")
    assert counted.total_micros() == 12_000_000  # dollars to spare: the count refused the crate
    assert by_dollars == by_count
    kept, disclosed, notes = by_count
    assert [row["name"] for row in kept] == ["crate", "drill"]
    assert {row["refusal_code"] for row in kept} == {"website_scene_preparation_budget_exhausted"}
    assert disclosed == {"kept_at_spending_cap": ["crate", "drill"], "reason": SPENDING_CAP_REACHED,
                         "refusal_code": "website_scene_preparation_budget_exhausted",
                         "refusal_code_means_one_of": ["job_dollar_cap", "paid_request_count"]}
    said = " ".join(notes)
    assert "Kept as filmed: crate, the job reached its limit on paid work." in notes
    assert "spending cap" not in said and "$" not in said and "request" not in said


def test_the_handoff_discloses_each_object_the_cap_kept_out(tmp_path, monkeypatch):
    from blueprint_pipeline import website_build_plan, website_build_plan_notes
    from blueprint_pipeline.task_family_native_runtime import CATALOG_PATH_ENV
    from tests.test_website_scene_handoff_navigation import CUPS, room_job
    monkeypatch.delenv(CATALOG_PATH_ENV, raising=False)
    decided = {"build_plan": {"digest": PLAN_DIGEST},
               "built": {"target_id": "cup-1", "name": "cup", "sourcing": "generate", "editor_object_ids": ["e"]},
               "kept_in_scan_target_ids": ["cup-2"]}
    kept = {"schema_version": "website_build_budget.v1", "build_plan_digest": PLAN_DIGEST, "objects": [
        {"rank": 0, "name": "cup", "status": budget.FIRST},
        {"rank": 1, "name": "second cup", "status": budget.KEPT, "reason": SPENDING_CAP_REACHED,
         "refusal_code": JOB_SPEND_CAP, "refusal_code_means_one_of": list(REFUSAL_CODE_MEANS_ONE_OF)}]}
    monkeypatch.setattr(website_build_plan, "sealed_decision", lambda _root: decided)
    monkeypatch.setattr(website_build_plan_notes, "sealed_budget", lambda _root: kept)
    _job, _context, planned = room_job(tmp_path, monkeypatch, removal=CUPS, family="rigid_pick_place",
                                       description="Move the cup.", all_targets=True)
    assert planned["build_plan"] == {"digest": PLAN_DIGEST, "built": "cup-1", "kept_in_scan": 1,
                                     "kept_at_spending_cap": ["second cup"], "reason": SPENDING_CAP_REACHED,
                                     "refusal_code": "website_scene_preparation_budget_exhausted",
                                     "refusal_code_means_one_of": ["job_dollar_cap", "paid_request_count"]}
