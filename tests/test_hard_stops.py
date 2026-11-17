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


WALL_AFTER_POSE = MoneyStop("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted")


def _paid_pose_then(monkeypatch, root, error):
    """The original views' pose is bought and collected; the reference masks after it (SAM, paid) raise ``error``."""
    from blueprint_pipeline import (
        atlas_context_fov,
        atlas_reference_masks,
        atlas_source_pose_inputs,
    )
    from blueprint_pipeline import atlas_scene_automation as automation
    from blueprint_pipeline import atlas_scene_measurements as measurements
    from blueprint_pipeline import paid_resource_allocator as allocator
    from blueprint_pipeline.website_atlas_pose import ATLAS_PRICE_REVIEW_ENV
    inputs = {"frames": [{"frame_id": "a"}, {"frame_id": "b"}], "binding": {"source_video_digest": "sha256:v"}}
    monkeypatch.setattr(atlas_source_pose_inputs, "prepare_source_pose", lambda pose_root: (pose_root, inputs))
    monkeypatch.setenv(ATLAS_PRICE_REVIEW_ENV, "{}")  # the step writes its quote here; restored after
    bought = []

    def pose(**kw):
        bought.append("pose")
        (kw["output_root"] / "atlas_posed_rgbd_result.json").write_text("{}")
        return {"status": "completed", "operation_id": "source"}
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_pose", pose)
    monkeypatch.setattr(allocator, "run_sponsored_website_atlas_task", lambda **_: pytest.fail("bought after a wall"))
    monkeypatch.setattr(measurements, "load_views", lambda *_: ({"digest": "sha256:pose", "operation": {"id": "s"}},
                                                                ["views"]))
    monkeypatch.setattr(atlas_context_fov, "verify_pose_context_fov", lambda **_: None)
    monkeypatch.setattr(automation, "measurement_plan", lambda review: {"scale_references": []})

    def masks(**_):
        raise error
    monkeypatch.setattr(atlas_reference_masks, "segment_scale_references", masks)
    return inputs, bought


def _policy():
    return {"pose_input_cost_usd": .01, "rate_card_reference": "r", "reviewed_by": "o",
            "reviewed_on": datetime.now(timezone.utc).date().isoformat()}


def test_a_wall_after_the_bought_pose_is_held_as_before_and_raised_only_for_the_windowed_repose(
        tmp_path, monkeypatch):
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    held = []
    for name, error in (("typed", WALL_AFTER_POSE), ("plain", ValueError(str(WALL_AFTER_POSE)))):
        root = tmp_path / name / "pose"
        root.mkdir(parents=True)
        inputs, bought = _paid_pose_then(monkeypatch, root, error)
        held.append(pipeline._source_pose(pose_root=root, inputs=inputs, review={}, policy=_policy(),
                                          task_context={"capture_id": "c"}, check=lambda *_a, **_k: None,
                                          experimental=False, max_polls=1))
        assert bought == ["pose"]
    assert held[0] == held[1] == {"status": "needs_input", "reason": str(WALL_AFTER_POSE)}
    # The windowed re-pose asks for the wall: it would go on to buy past a held result.
    root = tmp_path / "repose" / "pose"
    root.mkdir(parents=True)
    inputs, _ = _paid_pose_then(monkeypatch, root, WALL_AFTER_POSE)
    with pytest.raises(MoneyStop) as caught:
        pipeline._source_pose(pose_root=root, inputs=inputs, review={}, policy=_policy(),
                              task_context={"capture_id": "c"}, check=lambda *_a, **_k: None, experimental=False,
                              max_polls=1, raise_walls=True)
    assert caught.value is WALL_AFTER_POSE


def test_a_scene_kept_as_filmed_still_exports_its_room_when_a_wall_follows_the_bought_pose(tmp_path, monkeypatch):
    """``_preserved_scene`` (geometry-only and as-filmed jobs): the wall is held, nothing more is bought, the room
    is exported with the wall named, exactly as for the same code raised plain. The controller then archives it."""
    from blueprint_pipeline import atlas_depth_splat as splats
    from blueprint_pipeline import atlas_result_bundle as bundles
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    results = []
    for name, error in (("typed", WALL_AFTER_POSE), ("plain", ValueError(str(WALL_AFTER_POSE)))):
        out = tmp_path / name
        (out / "pose").mkdir(parents=True)
        (out / "atlas_scene_review.json").write_text("{}")
        _, bought = _paid_pose_then(monkeypatch, out / "pose", error)
        monkeypatch.setattr(splats, "load_posed_rgbd", lambda root: ({"digest": "posed"}, ["observed_depth"]))
        monkeypatch.setattr(splats, "write_depth_splat", lambda **_: {"digest": "splat",
                                                                     "splat_kind": "depth_fused_draft"})
        monkeypatch.setattr(bundles, "export_pose_bundle", lambda **kw: {"zip_path": "scene.zip"})
        review = {"subject_origin": "no_manipulated_subject", "uncaptured_regions": [],
                  "automation": {"objects": [{"disposition": "keep"}], "removal_instruction": ""}}
        result = pipeline._preserved_scene(pose_root=out / "pose", output_root=out, inputs={}, review=review,
            policy={**_policy(), "mesh_export_cost_usd": .5}, task_context={"capture_id": "c", "context_digest": "t"},
            source_digest="source", configuration=None, check=lambda *_a, **_k: None, experimental=False,
            max_polls=1, model="test")
        assert bought == ["pose"] and result["download"] == {"zip_path": "scene.zip"}
        assert str(WALL_AFTER_POSE) in result["blockers"] and not result["mesh_available"]
        results.append(json.loads(json.dumps(result).replace(str(out), "<job>")))
    assert results[0] == results[1]


def test_the_reviewed_background_path_still_delivers_when_a_wall_follows_the_bought_pose(tmp_path, monkeypatch):
    """The cleaned-background route keeps main's outcome: the source pose is held, the background is delivered."""
    from blueprint_pipeline import atlas_video_inputs as inputs_module
    from blueprint_pipeline import paid_resource_allocator as allocator
    from blueprint_pipeline import website_atlas_pipeline as pipeline
    from blueprint_pipeline import website_atlas_pose as pose_api
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    results = []
    for name, error in (("typed", WALL_AFTER_POSE), ("plain", ValueError(str(WALL_AFTER_POSE)))):
        root = tmp_path / name / "job"
        (root / "survey").mkdir(parents=True)
        video = tmp_path / name / "source.mov"
        video.write_bytes(b"source")
        survey = {"schema_version": "atlas_video_survey.v1", "source_video_digest": pipeline._sha256_file(video),
                  "display_rotation_degrees": 0, "heldout_pixels_included": False, "frames": []}
        survey["digest"] = canonical_digest(survey, digest_field="digest")
        (root / "survey/atlas_video_survey.json").write_text(json.dumps(survey))
        (root / "selection_binding.json").write_text(json.dumps({
            "survey_relative_path": "survey/atlas_video_survey.json",
            "review": {"automation": {"objects": [{"disposition": "remove"}]}}, "planner_receipt": {"model": "m"}}))
        config = {"authority": {"provider_retention": "approved"}, "remaining_cost_usd": 10,
                  "policy": {**_policy(), "maximum_review_cost_usd": 1, "rgbd_view_cost_usd": .05}}
        monkeypatch.setattr(pipeline, "load_configuration", lambda *_, _config=config, **kw: _config)
        monkeypatch.setattr(pose_api, "atlas_api_config", lambda: ("https://example.test", "non-secret"))
        monkeypatch.setattr(inputs_module, "prepare_video_inputs",
                            lambda **_: {"frames": [{"frame_id": "a"}, {"frame_id": "b"}]})
        monkeypatch.setattr(allocator, "run_sponsored_website_atlas_automation", lambda **kw: {
            "status": "completed", "measurements": {"status": "relative_sizes_only"},
            "background": {"status": "completed", "splat_manifest_digest": "splat", "mesh_available": True}})
        _, bought = _paid_pose_then(monkeypatch, root / "pose", error)
        result = pipeline.run_atlas_pipeline(source_video=video, output_root=root, checkpoint=lambda *_a, **_k: None,
                                             task_context={"capture_id": "c", "context_digest": "task"})
        assert bought == ["pose"] and result["source_pose"] == {"status": "needs_input",
                                                                "reason": str(WALL_AFTER_POSE)}
        assert result["splat_available"] is True
        results.append(json.loads(json.dumps(result).replace(str(tmp_path / name), "<run>")))
    assert results[0] == results[1]


