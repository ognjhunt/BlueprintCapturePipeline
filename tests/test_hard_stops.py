"""Privacy, rights and money walls are raised as types where they are guarded, and a job rechecks them.

Owner direction 2026-11-17: a website job's room step goes on, flagged, for anything that is not privacy,
rights or money. That needs the walls recognised by type, not by the words of a code.
"""
import json
from datetime import datetime, timedelta, timezone
from io import BytesIO
from urllib.error import HTTPError

import pytest

from blueprint_pipeline import hard_stops
from blueprint_pipeline import website_atlas_preview as preview
from blueprint_pipeline.hard_stops import HardStop, MoneyStop, PrivacyRightsStop
from blueprint_pipeline.job_retry_policy import MONEY

RECHECK = preview.AUTHORITY_RECHECK_SECONDS


def _job(**fields):
    return {"state": "processing", "reason": None, "revision": 1, "enqueued_at_epoch": 0, **fields}


# --- Where each wall is raised -------------------------------------------------------------------


@pytest.mark.parametrize("code,kind,final", [
    ("source_revoked", PrivacyRightsStop, True),
    ("website_capture_withdrawn", PrivacyRightsStop, True),
    ("capture_revoked_or_expired", PrivacyRightsStop, True),
    ("website_run_closed_by_owner", PrivacyRightsStop, True),
    ("website_atlas_source_consent_required", PrivacyRightsStop, False),
    ("website_atlas_planner_consent_required", PrivacyRightsStop, False),
    ("provider_terms_not_configured_or_changed", PrivacyRightsStop, False),
    ("consent_expired", PrivacyRightsStop, False),
    ("website_scene_preparation_budget_exhausted", MoneyStop, False),
    # The website's other answers are not walls and keep their plain type.
    ("website_scene_sponsorship_binding_invalid", None, False),
    ("website_atlas_grant_missing", None, False),
])
def test_the_websites_wall_answers_come_back_typed_with_their_text_unchanged(monkeypatch, code, kind, final):
    from blueprint_pipeline import website_task_context as module
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://tryblueprint.io/api/internal/pipeline/sync")
    monkeypatch.setattr(module, "load_pipeline_sync_token", lambda: "test-secret")

    def refuse(*args, **kwargs):
        raise HTTPError(args[0], 409, "request refused", {}, BytesIO(json.dumps({"code": code}).encode()))
    monkeypatch.setattr(module, "safe_request", refuse)
    with pytest.raises(ValueError) as error:
        module.website_webapp_request(capture_id="walkthrough-req1", operation="preparation-spend", payload={})
    assert str(error.value) == f"website_control_preparation-spend_http_409:{code}"
    if kind is None:
        assert not isinstance(error.value, HardStop) and hard_stops.wall(error.value) is None
    else:
        assert type(error.value) is kind and error.value.final is final


def _admission(**changes):
    from blueprint_pipeline import website_atlas_pose as pose
    return {"allocation_binding_digest": "sha256:binding", "provider": pose.PROVIDER,
            "resource_class": pose.RESOURCE_CLASS, "external_disclosure_allowed": True,
            "maximum_cost_usd": 5.0, "expires_at_epoch": 10_000_000, **changes}


@pytest.mark.parametrize("changes,kind,code", [
    ({"external_disclosure_allowed": False}, PrivacyRightsStop, "website_atlas_pose_disclosure_not_authorized"),
    ({"maximum_cost_usd": 1.0}, MoneyStop, "website_atlas_pose_budget_insufficient"),
    # A binding that does not match and an admission about to lapse are not walls.
    ({"provider": "other"}, None, "website_atlas_pose_spend_binding_missing"),
    ({"expires_at_epoch": 10}, None, "website_atlas_pose_admission_expired"),
])
def test_the_pose_purchase_refuses_disclosure_and_money_as_walls(changes, kind, code):
    from blueprint_pipeline import website_atlas_pose as pose
    with pytest.raises(ValueError) as error:
        pose._require_admission(_admission(**changes), request_digest="sha256:binding",
                                binding={"maximum_cost_usd": 2.0}, now=100)
    assert str(error.value) == code
    assert (type(error.value) is kind) if kind else hard_stops.wall(error.value) is None


def test_no_recorded_provider_retention_and_no_face_screen_are_privacy_walls(monkeypatch, tmp_path):
    from PIL import Image

    from blueprint_pipeline import atlas_privacy_screen
    from blueprint_pipeline import website_atlas_pose as pose
    monkeypatch.delenv(pose.ATLAS_PROVIDER_RETENTION_ENV, raising=False)
    with pytest.raises(PrivacyRightsStop, match="^website_atlas_provider_retention_decision_required$"):
        pose.atlas_provider_retention()
    monkeypatch.setattr(atlas_privacy_screen, "MODEL", tmp_path / "missing.onnx")
    with pytest.raises(PrivacyRightsStop, match="^atlas_privacy_screen_unavailable$"):
        atlas_privacy_screen.blur_faces(Image.new("RGB", (8, 8)))


def test_a_spend_receipt_that_does_not_allow_disclosure_is_a_privacy_wall_under_its_old_code(monkeypatch):
    from blueprint_pipeline import paid_resource_admission as admission
    from blueprint_pipeline import website_task_context as module
    monkeypatch.setattr(admission, "require_paid_resource_budget", lambda **_: None)
    command = {"task_context_digest": "sha256:task", "allocation_binding_digest": "sha256:binding",
               "resource_class": "openai_api_candidate", "provider": "openai", "maximum_cost_usd": 1.0,
               "request_count": 1}
    answer = {**command, "external_disclosure_allowed": False, "expires_at_epoch": 4_000_000_000}
    monkeypatch.setattr(module, "preparation_webapp_request", lambda **_: dict(answer))
    options = {"task_context": {"context_digest": "sha256:task", "request_id": "r", "scene_id": "s"},
                   "binding_digest": "sha256:binding", "maximum_cost_usd": 1.0, "request_count": 1,
                   "resource_class": "openai_api_candidate", "provider": "openai"}
    with pytest.raises(PrivacyRightsStop, match="^website_preparation_spend_receipt_invalid$"):
        module.reserve_website_preparation_spend(**options)
    # A receipt for another binding keeps its plain refusal.
    answer.update(external_disclosure_allowed=True, allocation_binding_digest="sha256:other")
    with pytest.raises(ValueError) as error:
        module.reserve_website_preparation_spend(**options)
    assert str(error.value) == "website_preparation_spend_receipt_invalid" and hard_stops.wall(error.value) is None


def test_a_lapsed_price_review_is_a_money_wall():
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    stale = (datetime.now(timezone.utc).date() - timedelta(days=400)).isoformat()
    with pytest.raises(MoneyStop, match="^atlas_runtime_price_review_expired$"):
        pipeline._require_current_price_review({"reviewed_on": stale})


def test_the_scene_review_refuses_a_pose_allowance_and_its_own_cap_as_money_walls(tmp_path):
    from types import SimpleNamespace

    from blueprint_pipeline import atlas_scene_planner as planner
    from tests.test_atlas_scene_planner import fixture
    path, survey, proposal, policy = fixture(tmp_path)
    with pytest.raises(MoneyStop, match="^atlas_runtime_budget_insufficient_for_pose_and_generation$"):
        planner.proposal_review(proposal, survey=survey, policy=policy, pose_budget=0)

    class Poor:
        config = SimpleNamespace(max_inference_cost_usd=0.0)

        def configure_reservation_audit(self, **_):
            raise AssertionError("a refused review reserved nothing")
    with pytest.raises(MoneyStop, match="^atlas_scene_planner_invoker_budget_exhausted$"):
        planner.review_survey(survey_path=path, task_context={
            "description": "test task", "context_digest": "sha256:" + "b" * 64, "request_id": "r", "scene_id": "s",
            "capture_id": "c"}, configuration={"policy": policy}, output_root=tmp_path / "plans", invoker=Poor(),
            release_blockers=lambda: ("c" * 40, []))


def test_the_account_and_cloud_ceilings_are_money_walls_and_other_admission_refusals_are_not():
    from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionBlocked
    assert hard_stops.wall(PaidResourceAdmissionBlocked(["account_monthly_spend_limit_reached"])) == (MONEY, False)
    assert hard_stops.wall(PaidResourceAdmissionBlocked(["gcp_monthly_budget_report_unavailable"])) == (MONEY, False)
    for blockers in (["paid_resource_admission_grant_missing"],
                     ["account_monthly_spend_limit_reached", "paid_resource_admission_class_mismatch"], []):
        assert hard_stops.wall(PaidResourceAdmissionBlocked(blockers)) is None


# --- Codes that only read like a wall -------------------------------------------------------------


def test_codes_that_only_read_like_money_or_privacy_are_not_typed(tmp_path):
    from blueprint_pipeline import atlas_scene_planner as planner
    from blueprint_pipeline import atlas_video_inputs
    from tests.test_atlas_scene_planner import fixture
    _, survey, proposal, policy = fixture(tmp_path)
    # A stale review of the selection's prices: a planning ceiling, not an admission.
    with pytest.raises(ValueError) as error:
        atlas_video_inputs.selection_frame_budget({})
    assert str(error.value) == "atlas_reviewed_selection_budget_required"
    assert hard_stops.wall(error.value) is None
    # A model answer that names an unsafe frame as required: a malformed answer, not a privacy wall.
    bad = {**proposal, "unsafe_frame_ids": ["1"]}
    with pytest.raises(ValueError) as error:
        planner.proposal_review(bad, survey=survey, policy=policy, pose_budget=10)
    assert str(error.value) == "atlas_scene_planner_coverage_or_privacy_invalid"
    assert hard_stops.wall(error.value) is None


@pytest.mark.parametrize("code,state,waits", [
    # Words alone decide them, exactly as before this change.
    ("atlas_scene_planner_coverage_or_privacy_invalid", "failed", False),
    ("atlas_observed_room_privacy_selection_invalid", "failed", False),
    ("atlas_scene_planner_context_budget_exceeded", "needs_input", True),
    ("atlas_reviewed_selection_budget_required", "needs_input", True),
    ("website_image_completion_budget_exhausted", "needs_input", True),
])
def test_an_untyped_code_behaves_as_before(code, state, waits):
    after = preview._failed_step(_job(), ValueError(code), 100)
    assert (after["state"], after["reason"]) == (state, code)
    assert after["retry_at_epoch"] == (100 + RECHECK if waits else None)


# --- What the job does with a wall ----------------------------------------------------------------


@pytest.mark.parametrize("error", [
    PrivacyRightsStop("website_control_atlas-configuration_http_409:website_atlas_source_consent_required"),
    PrivacyRightsStop("website_control_preparation-spend_http_409:consent_expired"),
    PrivacyRightsStop("website_atlas_pose_disclosure_not_authorized"),
    # Before this change these two ended the job at once by their words.
    PrivacyRightsStop("atlas_privacy_screen_unavailable"),
    PrivacyRightsStop("website_preparation_spend_receipt_invalid"),
    MoneyStop("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted"),
    MoneyStop("atlas_scene_planner_invoker_budget_exhausted"),
    MoneyStop("website_atlas_pose_budget_insufficient"),
    MoneyStop("atlas_runtime_price_review_expired"),
])
def test_a_wall_waits_and_is_asked_again_until_the_jobs_deadline(error):
    after = preview._failed_step(_job(), error, 100)
    assert after["state"] == "needs_input" and after["reason"] == preview._reason(error)
    assert after["retry_at_epoch"] == 100 + RECHECK
    late = preview._failed_step(_job(), error, preview.MAXIMUM_JOB_SECONDS - 60)
    assert late["retry_at_epoch"] == preview.MAXIMUM_JOB_SECONDS


def test_the_account_ceiling_waits_as_money_instead_of_spending_the_refusal_schedule():
    from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionBlocked
    after = preview._failed_step(_job(), PaidResourceAdmissionBlocked(["account_monthly_spend_limit_reached"]), 100)
    assert (after["state"], after["retry_at_epoch"]) == ("needs_input", 100 + RECHECK)
    assert "refusal_retries" not in after


@pytest.mark.parametrize("error,state,reason", [
    (PrivacyRightsStop("website_control_task-context_http_409:source_revoked", final=True), "withdrawn",
     "website_capture_withdrawn"),
    (PrivacyRightsStop("website_control_preparation-spend_http_409:website_run_closed_by_owner", final=True),
     "failed", "website_run_closed_by_owner"),
])
def test_a_withdrawn_or_owner_closed_job_stays_final(error, state, reason):
    from blueprint_pipeline.website_atlas_preview_store import SCHEMA
    after = preview._failed_step(_job(schema_version=SCHEMA), error, 100)
    assert (after["state"], after["reason"], after["retry_at_epoch"]) == (state, reason, None)


# --- Inside the background automation and in the foreground ---------------------------------------


@pytest.mark.parametrize("error", [
    MoneyStop("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted"),
    MoneyStop("atlas_scene_planner_invoker_budget_exhausted"),
    PrivacyRightsStop("atlas_privacy_screen_unavailable"),
    PrivacyRightsStop("website_control_preparation-spend_http_409:website_atlas_source_consent_required"),
    PrivacyRightsStop("website_control_preparation-spend_http_409:source_revoked", final=True),
])
def test_a_wall_inside_the_background_automation_is_raised_and_nothing_more_is_bought(tmp_path, monkeypatch, error):
    from blueprint_pipeline import atlas_background_automation as edits
    from blueprint_pipeline import atlas_depth_splat
    from blueprint_pipeline import atlas_scene_automation as worker
    from blueprint_pipeline import atlas_scene_measurements as sizes
    from tests.test_atlas_scene_automation import plan, view
    pose = {'digest': 'pose', 'binding': {'source_video_digest': 'video'},
            'operation': {'id': 'already-paid'}, 'depth_unit': 'estimated', 'scale_status': 'estimated'}
    monkeypatch.setattr(sizes, 'load_views', lambda *args: (pose, [view(tmp_path)]))
    monkeypatch.setattr(sizes, 'estimate_sizes', lambda **kw: {})

    def refused(**kw):
        raise error
    monkeypatch.setattr(edits, 'run_background', refused)
    monkeypatch.setattr(atlas_depth_splat, "write_depth_splat", lambda **_: pytest.fail("bought after a wall"))
    with pytest.raises(HardStop) as caught:
        worker.run_automation(pose_root=tmp_path, inputs={}, review={'automation': plan()}, output_root=tmp_path,
                              task_context={}, configuration={'policy': {}},
                              task_runner=lambda **_: pytest.fail("bought after a wall"),
                              pose_runner=lambda **_: pytest.fail("bought after a wall"), release_blockers=None)
    assert caught.value is error and not (tmp_path / 'result.json').exists()
    after = preview._failed_step(_job(schema_version="website_atlas_preview_job.v1"), caught.value, 100)
    if error.final:
        assert after["state"] == "withdrawn" and after["retry_at_epoch"] is None
    else:
        assert (after["state"], after["retry_at_epoch"]) == ("needs_input", 100 + RECHECK)


def test_a_wall_at_the_source_pose_purchase_is_raised_not_held_as_a_result(tmp_path, monkeypatch):
    """Before this change ``_source_pose`` turned any refusal into a held result that rested for good."""
    from blueprint_pipeline import atlas_source_pose_inputs
    from blueprint_pipeline import paid_resource_allocator as allocator
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    inputs = {"frames": [{"frame_id": "a"}, {"frame_id": "b"}], "binding": {"source_video_digest": "sha256:v"}}
    monkeypatch.setattr(atlas_source_pose_inputs, "prepare_source_pose", lambda root: (root, inputs))
    wall = MoneyStop("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted")

    def pose(**_):
        raise wall
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_pose", pose)
    policy = {"pose_input_cost_usd": .01, "rate_card_reference": "r", "reviewed_by": "o",
              "reviewed_on": datetime.now(timezone.utc).date().isoformat()}
    from blueprint_pipeline.website_atlas_pose import ATLAS_PRICE_REVIEW_ENV
    monkeypatch.setenv(ATLAS_PRICE_REVIEW_ENV, "{}")  # the step writes its quote here; restored after
    options = {"pose_root": tmp_path / "pose", "inputs": inputs, "review": {}, "policy": policy, "task_context": {"capture_id": "c"},
                   "check": lambda *_a, **_k: None, "experimental": False, "max_polls": 1}
    with pytest.raises(MoneyStop) as caught:
        pipeline._source_pose(**options)
    assert caught.value is wall
    # A refusal that is not a wall is still held as before.
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_pose",
                        lambda **_: (_ for _ in ()).throw(ValueError("website_atlas_pose_operation_failed")))
    assert pipeline._source_pose(**options) == {"status": "needs_input",
                                                 "reason": "website_atlas_pose_operation_failed"}


def test_a_foreground_wall_waits_is_asked_again_and_buys_nothing_until_it_lifts(tmp_path, monkeypatch):
    """The step runs again on each recheck; its own admission refuses again, so nothing is bought."""
    from types import SimpleNamespace

    from tests.test_website_atlas_preview import World, _enqueue
    from tests.test_website_atlas_preview import _job as stored
    capture = tmp_path / "storage/bucket/scenes/site-req1/captures/walkthrough-req1"
    (capture / "raw").mkdir(parents=True)
    (capture / "raw/walkthrough.mov").write_bytes(b"video")
    (capture / "pipeline").mkdir()
    monkeypatch.setenv(preview.ROUTING_ENV, "consented")
    site = SimpleNamespace(storage=tmp_path / "storage", capture=capture, pipeline=capture / "pipeline",
                           job=capture / "pipeline/atlas" / preview.JOB_FILE)
    _enqueue(site)
    bought = []
    refusal = "website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted"

    def admission_refuses(_kwargs):
        raise MoneyStop(refusal)

    def admitted(kwargs):
        bought.append(kwargs)
        return {"status": "processing", "stage": "plan"}
    world = World(monkeypatch, site, [admission_refuses, admission_refuses, admitted])
    assert world.advance() == "needs_input"
    job = stored(site)
    assert job["reason"] == "website_scene_preparation_budget_exhausted"
    assert job["retry_at_epoch"] == world.clock + RECHECK and job["provider_cleanup"] is None
    world.clock += 60
    assert world.advance() == "needs_input" and len(world.runs) == 1  # not before the recheck
    world.clock += RECHECK
    assert world.advance() == "needs_input" and len(world.runs) == 2 and bought == []
    world.clock += RECHECK
    assert world.advance() == "processing" and len(bought) == 1
    # The site saw one wait, then the job going on.
    assert [(row["state"], row["reason"]) for row in world.pushed] == [
        ("needs_input", "website_scene_preparation_budget_exhausted"), ("processing", None)]
    # A job whose deadline passes while it waits ends by itself.
    world.steps.append(admission_refuses)
    world.clock = stored(site)["next_attempt_at_epoch"]
    assert world.advance() == "needs_input"
    world.clock = stored(site)["enqueued_at_epoch"] + preview.MAXIMUM_JOB_SECONDS + 1
    assert world.advance() == "failed" and stored(site)["reason"] == "atlas_preview_expired"


# --- The site's frames never wait at the provider on a privacy or rights recheck -------------------


def _site(tmp_path, monkeypatch):
    from types import SimpleNamespace
    capture = tmp_path / "storage/bucket/scenes/site-req1/captures/walkthrough-req1"
    (capture / "raw").mkdir(parents=True)
    (capture / "raw/walkthrough.mov").write_bytes(b"video")
    (capture / "pipeline").mkdir()
    monkeypatch.setenv(preview.ROUTING_ENV, "consented")
    return SimpleNamespace(storage=tmp_path / "storage", capture=capture, pipeline=capture / "pipeline",
                           job=capture / "pipeline/atlas" / preview.JOB_FILE)


def _uploads(kwargs, name):
    root = kwargs["output_root"] / name / "atlas_pose"
    root.mkdir(parents=True, exist_ok=True)
    (root / "submission.json").write_text(json.dumps({"created_assets": [name + "-upload"]}))


PRIVACY_WALLS = [
    PrivacyRightsStop("website_control_preparation-spend_http_409:consent_expired"),
    PrivacyRightsStop("website_preparation_spend_receipt_invalid"),
    PrivacyRightsStop("website_atlas_pose_disclosure_not_authorized"),
    PrivacyRightsStop("atlas_privacy_screen_unavailable"),
    PrivacyRightsStop("website_atlas_provider_retention_decision_required"),
    PrivacyRightsStop("website_control_atlas-configuration_http_409:website_atlas_source_consent_required"),
    PrivacyRightsStop("website_control_preparation-spend_http_409:provider_terms_not_configured_or_changed"),
]


@pytest.mark.parametrize("error", PRIVACY_WALLS, ids=lambda error: preview._reason(error))
def test_a_privacy_or_rights_wall_deletes_provider_assets_at_once_and_a_passing_recheck_goes_on(
        tmp_path, monkeypatch, error):
    from tests.test_website_atlas_preview import World, _enqueue
    from tests.test_website_atlas_preview import _job as stored
    site = _site(tmp_path, monkeypatch)
    _enqueue(site)

    def walled(kwargs):
        _uploads(kwargs, "pose")
        raise error

    def admitted(kwargs):
        _uploads(kwargs, "again")  # the normal admitted path uploads what it needs again
        return {"status": "processing", "stage": "pose"}
    world = World(monkeypatch, site, [walled, admitted, ValueError("atlas_scene_subject_origin_uncertain")])
    assert world.advance() == "needs_input"
    job = stored(site)
    # Cleanup is armed and done in the same tick the wall is hit, as for an ended job.
    assert job["retry_at_epoch"] == world.clock + RECHECK and job[preview.WALL_CLEANUP] is True
    assert job["provider_cleanup"]["state"] == "complete" and world.deleted == ["pose-upload"]
    world.clock += RECHECK
    assert world.advance() == "processing" and len(world.runs) == 2
    job = stored(site)
    # Going on: cleanup is armed again for what the job uploads from now on; nothing is deleted twice.
    assert preview.WALL_CLEANUP not in job and job["provider_cleanup"]["state"] == "pending"
    assert job["provider_cleanup"]["deleted"] == ["pose-upload"] and world.deleted == ["pose-upload"]
    assert world.advance() == "needs_input"  # an ordinary stop: its retry keeps the job's assets
    assert stored(site)["provider_cleanup"]["state"] == "pending" and world.deleted == ["pose-upload"]


@pytest.mark.parametrize("error", [
    MoneyStop("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted"),
    MoneyStop("atlas_runtime_price_review_expired"),
], ids=lambda error: preview._reason(error))
def test_a_money_wall_keeps_the_jobs_assets_while_it_waits_as_before(tmp_path, monkeypatch, error):
    from tests.test_website_atlas_preview import World, _enqueue
    from tests.test_website_atlas_preview import _job as stored
    site = _site(tmp_path, monkeypatch)
    _enqueue(site)

    def walled(kwargs):
        _uploads(kwargs, "pose")
        raise error
    world = World(monkeypatch, site, [walled])
    assert world.advance() == "needs_input"
    job = stored(site)
    assert preview.WALL_CLEANUP not in job and job["provider_cleanup"] is None and world.deleted == []
    world.clock = job["enqueued_at_epoch"] + preview.MAXIMUM_JOB_SECONDS + 1
    assert world.advance() == "failed" and world.deleted == ["pose-upload"]


@pytest.mark.parametrize("error,state", [
    (PrivacyRightsStop("website_control_task-context_http_409:source_revoked", final=True), "withdrawn"),
    (PrivacyRightsStop("website_control_preparation-spend_http_409:website_run_closed_by_owner", final=True),
     "failed"),
])
def test_a_withdrawn_or_closed_job_is_cleaned_at_once_and_stays_final(tmp_path, monkeypatch, error, state):
    from tests.test_website_atlas_preview import World, _enqueue
    from tests.test_website_atlas_preview import _job as stored
    site = _site(tmp_path, monkeypatch)
    _enqueue(site)

    def walled(kwargs):
        _uploads(kwargs, "pose")
        raise error
    world = World(monkeypatch, site, [walled])
    assert world.advance() == state and world.deleted == ["pose-upload"]
    assert stored(site)["retry_at_epoch"] is None
    world.clock += preview.MAXIMUM_JOB_SECONDS
    assert world.advance() == state and len(world.runs) == 1
