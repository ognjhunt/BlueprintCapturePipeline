"""The library step's pick when it cannot be asked, and the one-object route under a plan that names several objects
(the $0 stress run of round 2, 2026-11-24).

A withdrawal stops the step; money (the account's month) and an outage are waits that seal nothing; the job's own
cap is final and the rules decide. On the one-object route the plan's row for the object is used only where the plan
names it as the object moved. Offline, at $0 (fixture catalog, injected fetcher and answers).
"""
# Covers: src/blueprint_pipeline/website_object_library.py
# Covers: src/blueprint_pipeline/website_build_plan.py
# Covers: src/blueprint_pipeline/website_scene_handoff.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pxr")

from blueprint_pipeline import website_object_library as library  # noqa: E402
from blueprint_pipeline.hard_stops import JOB_SPEND_CAP, MoneyStop, PrivacyRightsStop, webapp_refusal  # noqa: E402
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionBlocked  # noqa: E402
from blueprint_pipeline.website_build_plan import apply_to_removal  # noqa: E402
from blueprint_pipeline.website_supplied_mesh import LIBRARY_SUBJECT, PARAMETRIC_SUBJECT  # noqa: E402
from tests.test_website_object_library import TASK, Fetcher, _asking, _index  # noqa: E402
from tests.test_website_object_library_single_object import ENTRY, REMOVAL, ROW, _sealed_plan  # noqa: E402

WITHDRAWN = "website_control_preparation-spend_http_409:source_revoked"


def _attach(capture: Path, ask, removal=REMOVAL, tmp_path: Path | None = None):
    return library.attach_library_subject(dict(removal), capture_root=capture, task_context=TASK,
                                          output_root=capture / "handoff" / "object_library",
                                          entries=library.catalog(_index(tmp_path or capture.parent)),
                                          fetch=Fetcher(), ask=ask)


def test_a_withdrawal_at_the_models_pick_stops_and_seals_no_fallback(tmp_path):
    """Control: the pick's reservation answers the site's withdrawal: the step stops, no deterministic stand-in is
    sealed in its place and nothing is downloaded."""
    capture = tmp_path / "capture"
    _sealed_plan(capture)
    fetch = Fetcher()
    ask, _ = _asking(PrivacyRightsStop(WITHDRAWN))
    with pytest.raises(PrivacyRightsStop):
        library.attach_library_subject(dict(REMOVAL), capture_root=capture, task_context=TASK,
                                       output_root=capture / "handoff" / "object_library",
                                       entries=library.catalog(_index(tmp_path)), fetch=fetch, ask=ask)
    out = capture / "handoff" / "object_library"
    assert not (out / library.RECORD).exists() and not (out / library.SUBJECT_RECORD).exists()
    assert not [url for url in fetch.urls if url.endswith(".obj")]  # thumbnails only, before the call


@pytest.mark.parametrize("refusal", [
    PaidResourceAdmissionBlocked(["account_monthly_spend_limit_reached"]),
    ValueError("website_control_preparation-spend_http_503:unclassified"),
    OSError("Connection reset by peer")])
def test_money_or_an_outage_at_the_models_pick_is_a_wait_that_seals_nothing(tmp_path, refusal):
    """The account's month reached at the pick (or the provider down) sealed the word-match fallback for good, under
    a defect's code: the model was never asked once the month cleared, and the fallback's stricter rule (every word
    and the size) could send an object a model would have matched to the paid builder. Now it is raised, as the
    build budget raises it, and nothing is sealed: the job's own retry asks the model."""
    from blueprint_pipeline.job_retry_policy import MONEY
    from blueprint_pipeline.website_preparation_retry import OUTAGE, classify
    from blueprint_pipeline.website_scene_handoff import DEFECT, blocker_code
    capture = tmp_path / "capture"
    _sealed_plan(capture, [{**ROW, "name": "stackable tote", "size_m": None}])  # no size: the fallback takes nothing
    with pytest.raises((ValueError, OSError)) as waited:
        _attach(capture, _asking(refusal)[0], tmp_path=tmp_path)
    code = blocker_code(waited.value)
    assert not code.startswith(DEFECT) and classify(code) in (MONEY, OUTAGE)
    if isinstance(refusal, PaidResourceAdmissionBlocked):
        assert isinstance(waited.value, MoneyStop) and "account_monthly_spend_limit_reached" in code
    out = capture / "handoff" / "object_library"
    assert not (out / library.RECORD).exists() and not (out / library.SUBJECT_RECORD).exists()
    ask_again, asked = _asking({"candidate_id": "c01", "reason": "The same tote."})
    again = _attach(capture, ask_again, tmp_path=tmp_path)
    assert asked and again[LIBRARY_SUBJECT]["attribution"]["title"] == "Sterilite Storage Tote, Blue"


def test_the_jobs_own_cap_at_the_pick_is_final_and_the_rules_decide(tmp_path):
    """Control: the job's $40 cap is no wait (nothing waits on it, owner 2026-11-19): the pick's rules decide and the
    answer is sealed, as for any refused call."""
    capture = tmp_path / "capture"
    _sealed_plan(capture)
    at_cap = webapp_refusal(f"website_control_preparation-spend_http_409:{JOB_SPEND_CAP}", JOB_SPEND_CAP)
    out = _attach(capture, _asking(at_cap)[0], tmp_path=tmp_path)
    subject = out[LIBRARY_SUBJECT]
    assert subject["pick"]["source"] == "deterministic" and subject["pick"]["model_fallback"].endswith(JOB_SPEND_CAP)
    assert (capture / "handoff" / "object_library" / library.RECORD).exists()


def test_a_refused_admission_is_recorded_under_its_own_code_not_as_a_defect():
    """``blocker_code``: the paid chokepoint's refusal is a money wall, never Blueprint's defect."""
    from blueprint_pipeline.website_scene_handoff import blocker_code
    code = blocker_code(PaidResourceAdmissionBlocked(["account_monthly_spend_limit_reached"]))
    assert code == "paid_resource_admission_blocked:account_monthly_spend_limit_reached"
    assert blocker_code(KeyError("x")).startswith("website_preparation_defect:")


# --- a plan that names several objects, on the one-object route ---------------------------------------------------

def _plan(capture, rows):
    _sealed_plan(capture, rows)
    return apply_to_removal(REMOVAL, capture)


def test_a_plan_naming_several_objects_builds_only_the_routes_own_object(tmp_path):
    """Control: the plan names three objects and takes none out alone; the removal has the one the route moves (the
    tote). The library looks up the tote, by the plan's row for it, and nothing else in the removal changes."""
    capture = tmp_path / "capture"
    rows = [{**ROW, "target_id": "carton", "name": "carton", "sourcing": "parametric", "parametric_shape": "carton"},
            ROW, {**ROW, "target_id": "crate", "name": "crate"}]
    removal = _plan(capture, rows)
    assert PARAMETRIC_SUBJECT not in removal  # the carton's shape is not the tote's
    ask, asked = _asking({"candidate_id": "c01", "reason": "The same tote."})
    out = _attach(capture, ask, removal, tmp_path)
    subject = out[LIBRARY_SUBJECT]
    assert subject["target_id"] == "tote" and subject["row_source"] == "plan"
    assert json.loads(asked[0]["binding"]["prompt"].split("\n", 1)[1])["object"]["name"] == "storage tote"
    assert out["entries"] == REMOVAL["entries"]


def test_a_plan_whose_row_for_the_routes_object_is_not_the_moved_object_is_flagged(tmp_path):
    """The plan names the tote as the destination and another object as the one moved; the removal's analysis moves
    the tote. The route looked the tote up by the plan's row (its role never read), as if the plan agreed. Now the
    row is read from the analysis and the disagreement recorded."""
    capture = tmp_path / "capture"
    rows = [{**ROW, "target_id": "carton", "name": "carton"},
            {**ROW, "role": "destination", "sourcing": "parametric", "parametric_shape": "tote"}]
    removal = _plan(capture, rows)
    assert PARAMETRIC_SUBJECT not in removal  # a destination row's shape is not the moved object's
    ask, asked = _asking({"candidate_id": "c01", "reason": "The same tote."})
    out = _attach(capture, ask, removal, tmp_path)
    subject = out[LIBRARY_SUBJECT]
    assert subject["row_source"] == "analysis" and subject["plan_role_disagrees"] == "destination"
    # The analysis names the object (its label), with no size: the model is asked by those words.
    assert json.loads(asked[0]["binding"]["prompt"].split("\n", 1)[1])["object"]["name"] == ENTRY["semantic_label"]


def test_a_plan_whose_row_for_the_routes_object_is_soft_keeps_todays_builder(tmp_path):
    """Control: the plan reads the tote as soft; the library step skips it with its reason."""
    capture = tmp_path / "capture"
    removal = _plan(capture, [{**ROW, "rigidity": "deformable", "sourcing": "keep_as_filmed"},
                              {**ROW, "target_id": "carton", "name": "carton"}])
    out = _attach(capture, _asking({"candidate_id": "c01", "reason": "x"})[0], removal, tmp_path)
    assert LIBRARY_SUBJECT not in out  # keep_as_filmed: today's builder, exactly
    assert json.loads((capture / "handoff/object_library" / library.SUBJECT_RECORD).read_text())["not_used"] == \
        library.KEEPS_AS_FILMED
