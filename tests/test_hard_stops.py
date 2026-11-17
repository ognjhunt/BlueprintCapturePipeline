"""Privacy, rights and money walls are raised as types where they are guarded; what a job does is unchanged.

Owner direction 2026-11-17: a website job's room step goes on, flagged, for anything that is not privacy,
rights or money. That needs the walls recognised by type first. This change only names them: for every
typed code the job ends where main ends it. Three catches that used to take a wall and go on buying now
raise it (the last tests here and in the windowed re-pose and native splat tests).
"""
import ast
import json
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from blueprint_pipeline import hard_stops
from blueprint_pipeline import website_atlas_preview as preview
from blueprint_pipeline.hard_stops import MoneyStop, PrivacyRightsStop, is_wall
from blueprint_pipeline.job_retry_policy import MONEY, PRIVACY_RIGHTS

SOURCE = Path(hard_stops.__file__).parent

#: Every code a step raises as a wall, and its type. Codes the website returns are typed by
#: ``webapp_refusal`` and listed in ``hard_stops.WEBAPP_*``.
RAISED = {
    "website_atlas_pose_disclosure_not_authorized": PrivacyRightsStop,
    "website_atlas_provider_retention_decision_required": PrivacyRightsStop,
    "website_atlas_provider_retention_admission_mismatch": PrivacyRightsStop,
    "atlas_task_retention_admission_mismatch": PrivacyRightsStop,
    "atlas_privacy_screen_unavailable": PrivacyRightsStop,
    "website_atlas_pose_budget_insufficient": MoneyStop,
    "website_atlas_pose_spend_binding_missing": MoneyStop,
    "website_atlas_pose_admission_expired": MoneyStop,
    "website_atlas_reviewed_price_required": MoneyStop,
    "website_atlas_reviewed_price_invalid": MoneyStop,
    "website_atlas_reviewed_price_options_mismatch": MoneyStop,
    "website_atlas_reviewed_price_input_count_exceeded": MoneyStop,
    "atlas_task_reviewed_price_required": MoneyStop,
    "atlas_runtime_price_review_expired": MoneyStop,
    "atlas_reviewed_selection_budget_required": MoneyStop,
    "atlas_selection_budget_insufficient": MoneyStop,
    "atlas_selection_budget_below_accurate_inputs": MoneyStop,
    "atlas_runtime_budget_insufficient": MoneyStop,
    "atlas_runtime_budget_insufficient_for_pose_and_generation": MoneyStop,
    "atlas_scene_planner_invoker_budget_exhausted": MoneyStop,
}
#: Read like a wall and are not one (or not yet): raised as plain ValueError.
UNTYPED = [
    "website_image_completion_authorization_missing",  # nothing is bought; the piece is skipped, the room ships
    "website_image_completion_budget_missing", "website_image_completion_budget_exhausted",
    "website_image_completion_budget_exceeded",
    "atlas_scene_planner_token_budget_exceeded", "atlas_scene_planner_context_budget_exceeded",  # token limits
    "atlas_scene_planner_coverage_or_privacy_invalid",  # a model answer in the wrong form
    "atlas_observed_room_privacy_selection_invalid",
]


def _raises():
    """Every ``raise <callable>("<code>...")`` in the package: (code head, callable name, file:line)."""
    for path in sorted(SOURCE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call) or not node.exc.args:
                continue
            first, func = node.exc.args[0], node.exc.func
            text = (first.value if isinstance(first, ast.Constant) and isinstance(first.value, str)
                    else first.values[0].value if isinstance(first, ast.JoinedStr) and first.values
                    and isinstance(first.values[0], ast.Constant) else None)
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if isinstance(text, str):
                yield text.partition(":")[0], name, f"{path.name}:{node.lineno}"


def test_every_raise_of_a_wall_code_raises_its_type_and_near_misses_stay_plain():
    found = {}
    for code, name, where in _raises():
        found.setdefault(code, []).append((name, where))
    for code, kind in RAISED.items():
        assert found.get(code), code
        expected = {kind.__name__, "_money"} if kind is MoneyStop else {kind.__name__}
        assert {name for name, _ in found[code]} <= expected, (code, found[code])
    for code in UNTYPED:
        assert {name for name, _ in found.get(code, [])} <= {"ValueError"}, (code, found.get(code))


# --- Where each wall is raised -------------------------------------------------------------------


@pytest.mark.parametrize("code,kind", [
    *[(code, PrivacyRightsStop) for code in sorted(hard_stops.WEBAPP_PRIVACY_RIGHTS)],
    ("website_scene_preparation_budget_exhausted", MoneyStop),
    # The website's other answers are not walls and keep their plain type.
    ("website_scene_sponsorship_binding_invalid", None),
    ("website_atlas_grant_missing", None),
])
def test_the_websites_wall_answers_come_back_typed_with_their_text_unchanged(monkeypatch, code, kind):
    from blueprint_pipeline import website_task_context as module
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://tryblueprint.io/api/internal/pipeline/sync")
    monkeypatch.setattr(module, "load_pipeline_sync_token", lambda: "test-secret")

    def refuse(*args, **kwargs):
        raise HTTPError(args[0], 409, "request refused", {}, BytesIO(json.dumps({"code": code}).encode()))
    monkeypatch.setattr(module, "safe_request", refuse)
    with pytest.raises(ValueError) as error:
        module.website_webapp_request(capture_id="walkthrough-req1", operation="preparation-spend", payload={})
    assert str(error.value) == f"website_control_preparation-spend_http_409:{code}"
    assert (type(error.value) is kind) if kind else not is_wall(error.value)


def _admission(**changes):
    from blueprint_pipeline import website_atlas_pose as pose
    return {"allocation_binding_digest": "sha256:binding", "provider": pose.PROVIDER,
            "resource_class": pose.RESOURCE_CLASS, "external_disclosure_allowed": True,
            "maximum_cost_usd": 5.0, "expires_at_epoch": 10_000_000, **changes}


@pytest.mark.parametrize("changes,kind,code", [
    ({"external_disclosure_allowed": False}, PrivacyRightsStop, "website_atlas_pose_disclosure_not_authorized"),
    ({"maximum_cost_usd": 1.0}, MoneyStop, "website_atlas_pose_budget_insufficient"),
    ({"provider": "other"}, MoneyStop, "website_atlas_pose_spend_binding_missing"),
    ({"expires_at_epoch": 10}, MoneyStop, "website_atlas_pose_admission_expired"),
])
def test_the_pose_purchase_refuses_disclosure_and_admission_as_walls(changes, kind, code):
    from blueprint_pipeline import website_atlas_pose as pose
    with pytest.raises(kind, match=f"^{code}$"):
        pose._require_admission(_admission(**changes), request_digest="sha256:binding",
                                binding={"maximum_cost_usd": 2.0}, now=100)


def test_no_recorded_retention_no_face_screen_and_no_reviewed_price_are_walls(monkeypatch, tmp_path):
    from PIL import Image

    from blueprint_pipeline import atlas_privacy_screen
    from blueprint_pipeline import website_atlas_pose as pose
    monkeypatch.delenv(pose.ATLAS_PROVIDER_RETENTION_ENV, raising=False)
    with pytest.raises(PrivacyRightsStop, match="^website_atlas_provider_retention_decision_required$"):
        pose.atlas_provider_retention()
    monkeypatch.setattr(atlas_privacy_screen, "MODEL", tmp_path / "missing.onnx")
    with pytest.raises(PrivacyRightsStop, match="^atlas_privacy_screen_unavailable$"):
        atlas_privacy_screen.blur_faces(Image.new("RGB", (8, 8)))
    monkeypatch.delenv(pose.ATLAS_PRICE_REVIEW_ENV, raising=False)
    with pytest.raises(MoneyStop, match="^website_atlas_reviewed_price_required$"):
        pose.atlas_pose_spend_line(planned_input_count=2, target_resolution=[720, 1280])
    monkeypatch.setenv(pose.ATLAS_PRICE_REVIEW_ENV, "{")
    with pytest.raises(MoneyStop, match="^website_atlas_reviewed_price_invalid:json$"):
        pose.atlas_pose_spend_line(planned_input_count=2, target_resolution=[720, 1280])


@pytest.mark.parametrize("answer,kind", [
    ({"external_disclosure_allowed": False}, PrivacyRightsStop),
    ({"allocation_binding_digest": "sha256:other"}, MoneyStop),
    ({"expires_at_epoch": 10}, MoneyStop),
])
def test_a_spend_receipt_that_covers_nothing_is_a_wall_under_its_one_code(monkeypatch, answer, kind):
    from blueprint_pipeline import paid_resource_admission as admission
    from blueprint_pipeline import website_task_context as module
    monkeypatch.setattr(admission, "require_paid_resource_budget", lambda **_: None)
    command = {"task_context_digest": "sha256:task", "allocation_binding_digest": "sha256:binding",
               "resource_class": "openai_api_candidate", "provider": "openai", "maximum_cost_usd": 1.0,
               "request_count": 1}
    value = {**command, "external_disclosure_allowed": True, "expires_at_epoch": 4_000_000_000, **answer}
    monkeypatch.setattr(module, "preparation_webapp_request", lambda **_: dict(value))
    with pytest.raises(kind, match="^website_preparation_spend_receipt_invalid$"):
        module.reserve_website_preparation_spend(
            task_context={"context_digest": "sha256:task", "request_id": "r", "scene_id": "s"},
            binding_digest="sha256:binding", maximum_cost_usd=1.0, request_count=1,
            resource_class="openai_api_candidate", provider="openai")


def test_price_reviews_and_selection_dollars_are_money_walls(tmp_path):
    from blueprint_pipeline import atlas_video_inputs
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    stale = (datetime.now(timezone.utc).date() - timedelta(days=400)).isoformat()
    with pytest.raises(MoneyStop, match="^atlas_runtime_price_review_expired$"):
        pipeline._require_current_price_review({"reviewed_on": stale})
    with pytest.raises(MoneyStop, match="^atlas_reviewed_selection_budget_required$"):
        atlas_video_inputs.selection_frame_budget({})
    budget = {"maximum_pose_cost_usd": .01, "input_cost_usd": .01, "rate_card_reference": "r", "reviewed_by": "o",
              "reviewed_on": datetime.now(timezone.utc).date().isoformat()}
    with pytest.raises(MoneyStop, match="^atlas_selection_budget_insufficient$"):
        atlas_video_inputs.selection_frame_budget({"selection_budget": budget})


def test_the_scene_review_refuses_a_pose_allowance_and_its_own_cap_as_money_walls(tmp_path):
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


def test_is_wall_counts_the_two_types_an_admission_refusal_and_a_withdrawal_and_nothing_else():
    from blueprint_pipeline.atlas_preparation_job_control import _Withdrawn
    from blueprint_pipeline.deterministic_refusal import DeterministicRefusal
    from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionBlocked
    assert hard_stops.wall_kind(PrivacyRightsStop("x")) == PRIVACY_RIGHTS
    assert hard_stops.wall_kind(MoneyStop("x")) == MONEY
    assert hard_stops.wall_kind(_Withdrawn()) == PRIVACY_RIGHTS
    for blockers in (["account_monthly_spend_limit_reached"], ["paid_resource_admission_not_admitted"],
                     ["paid_resource_admission_has_blockers", "gcp_monthly_budget_report_unavailable"], []):
        assert hard_stops.wall_kind(PaidResourceAdmissionBlocked(blockers)) == MONEY
    for other in (ValueError("website_scene_preparation_budget_exhausted"), RuntimeError("x"), OSError("x"),
                  DeterministicRefusal("atlas_x", step="atlas_proposal_review", inputs={})):
        assert not is_wall(other)  # never by its words


# --- What the job does is what main does ------------------------------------------------------------


def _site(tmp_path, monkeypatch, name):
    capture = tmp_path / name / "storage/bucket/scenes/site-req1/captures/walkthrough-req1"
    (capture / "raw").mkdir(parents=True)
    (capture / "raw/walkthrough.mov").write_bytes(b"video")
    (capture / "pipeline").mkdir()
    monkeypatch.setenv(preview.ROUTING_ENV, "consented")
    return SimpleNamespace(storage=tmp_path / name / "storage", capture=capture, pipeline=capture / "pipeline",
                           job=capture / "pipeline/atlas" / preview.JOB_FILE)


def _run(tmp_path, monkeypatch, name, error):
    """One job through two ticks whose step raises ``error``: its record, what the website read, what was deleted."""
    from tests.test_website_atlas_preview import World, _enqueue
    site = _site(tmp_path, monkeypatch, name)
    with monkeypatch.context() as clock:
        clock.setattr(preview.time, "time", lambda: 1_800_000_000)  # intake stamps the clock
        _enqueue(site)

    def walled(kwargs):
        root = kwargs["output_root"] / "pose/atlas_pose"
        root.mkdir(parents=True, exist_ok=True)
        (root / "submission.json").write_text(json.dumps({"created_assets": ["upload"]}))
        raise error
    world = World(monkeypatch, site, [walled, walled])
    states = [world.advance()]
    job = json.loads(site.job.read_text())
    world.clock = max(job["next_attempt_at_epoch"], job.get("retry_at_epoch") or 0) + 1
    states.append(world.advance())
    job = json.loads(site.job.read_text())
    if isinstance(job.get("last_error"), dict):
        job["last_error"] = {**job["last_error"], "type": "-"}  # the retained diagnostic names the class
    pushed = [{k: v for k, v in row.items() if k not in {"updated_at_epoch", "preview_digest"}} for row in world.pushed]
    return states, job, pushed, \
        world.deleted, len(world.runs)


TYPED_EXAMPLES = [
    *[PrivacyRightsStop(f"website_control_preparation-spend_http_409:{code}")
      for code in sorted(hard_stops.WEBAPP_PRIVACY_RIGHTS)],
    MoneyStop("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted"),
    PrivacyRightsStop("website_preparation_spend_receipt_invalid"),
    MoneyStop("website_preparation_spend_receipt_invalid"),
    *[kind(code) for code, kind in RAISED.items()],
]


@pytest.mark.parametrize("error", TYPED_EXAMPLES, ids=lambda error: f"{type(error).__name__}:{error}")
def test_a_typed_wall_ends_the_job_exactly_where_the_same_code_did_before(tmp_path, monkeypatch, error):
    typed = _run(tmp_path, monkeypatch, "typed", error)
    plain = _run(tmp_path, monkeypatch, "plain", ValueError(str(error)))
    assert typed == plain


# --- Catches between a wall and the controller ------------------------------------------------------


def test_no_code_a_catch_lets_go_on_is_a_wall():
    """Catches that take only named codes and raise every other one: no wall code is among the names."""
    from blueprint_pipeline import atlas_splat_rebuild
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    walls = set(RAISED) | hard_stops.WEBAPP_PRIVACY_RIGHTS | hard_stops.WEBAPP_MONEY
    named = {*pipeline.COMPLETION_FLOOR_ERRORS,  # website_atlas_pipeline: the completion floor
             "atlas_scale_review_abstained",  # the scale review
             "atlas_task_operation_failed",  # the refined splat, the native forms, the task retry
             "website_atlas_pose_operation_failed",  # the pose resubmission
             "atlas_scene_planner_token_budget_exceeded", "atlas_scene_planner_context_budget_exceeded",
             "atlas_scene_planner_output_invalid", "atlas_scene_planner_output_truncated_retry_unavailable",
             "website_image_completion_cached_batch_missing", "website_gemini_requires_reconciliation"}
    assert not walls & named
    assert not any(code.startswith(atlas_splat_rebuild.STOP_PREFIXES) for code in walls)


@pytest.mark.parametrize("error", [MoneyStop("atlas_scene_planner_invoker_budget_exhausted"),
                                   PrivacyRightsStop("atlas_privacy_screen_unavailable")])
def test_the_background_automation_holds_a_wall_exactly_as_before(tmp_path, monkeypatch, error):
    """Typed, outcome unchanged: held as background input and nothing more bought; to be changed with the fallback."""
    from blueprint_pipeline import atlas_background_automation as edits
    from blueprint_pipeline import atlas_depth_splat
    from blueprint_pipeline import atlas_scene_automation as worker
    from blueprint_pipeline import atlas_scene_measurements as sizes
    from tests.test_atlas_scene_automation import plan, view
    pose = {'digest': 'pose', 'binding': {'source_video_digest': 'video'},
            'operation': {'id': 'already-paid'}, 'depth_unit': 'estimated', 'scale_status': 'estimated'}
    monkeypatch.setattr(sizes, 'load_views', lambda *args: (pose, [view(tmp_path)]))
    monkeypatch.setattr(sizes, 'estimate_sizes', lambda **kw: {})
    monkeypatch.setattr(atlas_depth_splat, "write_depth_splat", lambda **_: pytest.fail("bought after a wall"))
    results = []
    for raised in (error, ValueError(str(error))):
        def refused(*, _raised=raised, **kw):
            raise _raised
        monkeypatch.setattr(edits, 'run_background', refused)
        result = worker.run_automation(pose_root=tmp_path, inputs={}, review={'automation': plan()},
            output_root=tmp_path / type(raised).__name__, task_context={}, configuration={'policy': {}},
            task_runner=lambda **_: pytest.fail("bought after a wall"), pose_runner=None, release_blockers=None)
        results.append((result["status"], result["background"]["reason"]))
    assert results[0] == results[1] == ("needs_input", str(error))


def test_the_original_views_pose_raises_a_wall_instead_of_holding_it_as_a_result(tmp_path, monkeypatch):
    """A deliberate change: the windowed re-pose went on buying after a held pose result."""
    from blueprint_pipeline import atlas_source_pose_inputs
    from blueprint_pipeline import paid_resource_allocator as allocator
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    from blueprint_pipeline.website_atlas_pose import ATLAS_PRICE_REVIEW_ENV
    inputs = {"frames": [{"frame_id": "a"}, {"frame_id": "b"}], "binding": {"source_video_digest": "sha256:v"}}
    monkeypatch.setattr(atlas_source_pose_inputs, "prepare_source_pose", lambda root: (root, inputs))
    monkeypatch.setenv(ATLAS_PRICE_REVIEW_ENV, "{}")  # the step writes its quote here; restored after
    wall = MoneyStop("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted")

    def pose(**_):
        raise wall
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_pose", pose)
    policy = {"pose_input_cost_usd": .01, "rate_card_reference": "r", "reviewed_by": "o",
              "reviewed_on": datetime.now(timezone.utc).date().isoformat()}
    options = {"pose_root": tmp_path / "pose", "inputs": inputs, "review": {}, "policy": policy,
                   "task_context": {"capture_id": "c"}, "check": lambda *_a, **_k: None, "experimental": False, "max_polls": 1}
    with pytest.raises(MoneyStop) as caught:
        pipeline._source_pose(**options)
    assert caught.value is wall
    # Any other refusal is still held as before.
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_pose",
                        lambda **_: (_ for _ in ()).throw(ValueError("website_atlas_pose_operation_failed")))
    assert pipeline._source_pose(**options) == {"status": "needs_input",
                                                 "reason": "website_atlas_pose_operation_failed"}
