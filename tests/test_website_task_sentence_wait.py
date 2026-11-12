"""A job with no sentence buys no reading of the video; it waits for the site and starts by itself.

capture-98a79ce3 (2026-10-29) sent the website's placeholder "Your site task" as its description.
The removal analysis and two Atlas reviews were bought for it, and both readers said there was no
task to read. These fixtures keep that context's shape, with its identifiers replaced.
"""
from types import SimpleNamespace

import pytest

from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
from blueprint_pipeline import website_gemini_receipts
from blueprint_pipeline.website_task_context_contracts import (
    TASK_NOT_NAMED, UNNAMED_TASK_DESCRIPTION, task_named)

UNNAMED = {
    "schema_version": "website_site_task_context.v1", "purpose": "scene_preparation",
    "request_id": "capture-unnamed", "scene_id": "site-capture-unnamed",
    "capture_id": "walkthrough-capture-unnamed", "description": "Your site task", "task_family": "general_task",
    "confirmed": False, "confirmed_at": None,
    "unresolved": ["sceneStability", "taskShape", "objectVariety", "deploymentTimeline", "accessWindow"],
    "capture_rights": {"derived_scene_generation_allowed": True, "consent_status": "granted",
                       "consent_revoked": False}, "context_digest": "sha256:" + "4" * 64,
}
NAMED = {**UNNAMED, "task_family": "rigid_pick_and_place", "context_digest": "sha256:" + "c" * 64,
         "description": "Pick up the glass vase from the middle of the dining table and place it at the "
                        "other end of the same table."}


def test_the_placeholder_and_an_empty_description_name_no_job():
    assert UNNAMED_TASK_DESCRIPTION == "Your site task"
    assert not task_named(UNNAMED) and not task_named({**UNNAMED, "description": "  Your site task "})
    assert not task_named({**UNNAMED, "description": ""}) and not task_named(None)
    assert task_named(NAMED)


def test_removal_analysis_of_no_job_reserves_and_buys_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(website_gemini_receipts, "retained_gemini_call", lambda **_k: pytest.fail("must not reserve"))
    video = tmp_path / "walkthrough.mp4"
    video.write_bytes(b"original video")
    plan = analysis.analyze_removal_targets(video_path=video, output_root=tmp_path / "receipts", task_context=UNNAMED)
    assert plan["status"] == "blocked" and plan["blockers"] == [TASK_NOT_NAMED]
    assert not (tmp_path / "receipts").exists()


def test_removal_analysis_of_a_named_job_is_unchanged(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(website_gemini_receipts, "retained_gemini_call",
                        lambda **kwargs: seen.append(kwargs["task_context"]) or {"status": "completed"})
    video = tmp_path / "walkthrough.mp4"
    video.write_bytes(b"original video")
    monkeypatch.setattr(analysis, "_video_processing", lambda path, mode: ("static", 14.4))
    assert analysis.analyze_removal_targets(video_path=video, output_root=tmp_path / "receipts",
                                            task_context=NAMED) == {"status": "completed"}
    assert seen == [NAMED]


def _prepare(context, tmp_path, effects):
    from blueprint_pipeline.website_capture_preparation import run_website_capture_preparation
    from tests.test_qualification_coverage_edges import _descriptor
    capture = tmp_path / "capture"
    (capture / "pipeline").mkdir(parents=True)

    def prepare_media(**_):
        effects.append("media")
        raise RuntimeError("stop after media")

    return lambda: run_website_capture_preparation(
        descriptor=_descriptor(metadata={"site_task_context": context}), capture_root=capture,
        pipeline_dir=capture / "pipeline", source_video=None, derived_scene_generation_allowed=True, gates=[],
        prepare_media=prepare_media, check_privacy=lambda **_: SimpleNamespace(passed=True),
        make_gate=lambda _name, passed, _detail: SimpleNamespace(passed=passed))


def test_preparation_of_no_job_stops_before_any_paid_stage(tmp_path):
    from blueprint_pipeline.common import StageError
    effects = []
    with pytest.raises(StageError, match="website_scene_preparation: website_task_not_named"):
        _prepare(UNNAMED, tmp_path, effects)()
    assert effects == []


def test_preparation_of_a_named_job_reaches_the_media_stage(tmp_path):
    effects = []
    with pytest.raises(RuntimeError, match="stop after media"):
        _prepare(NAMED, tmp_path, effects)()
    assert effects == ["media"]


def test_the_wait_retries_until_the_grant_ends_and_never_on_the_refusal_schedule():
    from blueprint_pipeline import website_preparation_retry as retry
    kind = retry.classify("website_scene_preparation: website_task_not_named")
    assert kind == retry.AWAITING_JOB_SENTENCE
    # With a grant: at most an hour apart, never exhausted by a count.
    assert [retry.retry_delay(kind, n) for n in (1, 2, 30)] == [120, 240, 3600]
    # With no grant (no deadline): the bounded refusal schedule, as before.
    assert retry.retry_delay(kind, 1, deadline=False) == 900
    assert retry.classify("website_scene_preparation: website_atlas_job_failed") == retry.REFUSAL


def test_atlas_job_of_no_job_configures_and_buys_nothing(tmp_path):
    from blueprint_pipeline.website_atlas_pipeline import _run_atlas_pipeline
    video = tmp_path / "raw" / "walkthrough.mp4"
    video.parent.mkdir()
    video.write_bytes(b"original video")
    job = tmp_path / "pipeline" / "atlas"
    with pytest.raises(ValueError, match="^website_task_not_named$"):
        _run_atlas_pipeline(source_video=video, output_root=job, task_context=UNNAMED)
    assert not (job / "configuration.json").exists() and not (job / "scene_planner").exists()


def test_atlas_preview_waits_on_the_sentence_and_rechecks_it(monkeypatch):
    from blueprint_pipeline import website_atlas_preview as preview
    from blueprint_pipeline.website_preparation_outcome import NEW_REASONS_ENV
    monkeypatch.setenv(NEW_REASONS_ENV, "false")
    off = preview._failed_step({"state": "processing", "stage": "queued", "reason": None, "revision": 1,
                                "enqueued_at_epoch": 100, "next_attempt_at_epoch": 100, "retry_at_epoch": None,
                                "transient_failures": 0}, ValueError(TASK_NOT_NAMED), 1000)
    assert off["state"] == "needs_input" and off["reason"] == "task_brief_missing"  # until the website lists it
    monkeypatch.setenv(NEW_REASONS_ENV, "true")
    job = {"state": "processing", "stage": "queued", "reason": None, "revision": 1, "enqueued_at_epoch": 100,
           "next_attempt_at_epoch": 100, "retry_at_epoch": None, "transient_failures": 0}
    after = preview._failed_step(job, ValueError(TASK_NOT_NAMED), 1000)
    assert after["state"] == "needs_input" and after["reason"] == TASK_NOT_NAMED
    assert after["retry_at_epoch"] == 1000 + preview.AUTHORITY_RECHECK_SECONDS


def test_a_job_with_no_sentence_starts_nothing(tmp_path):
    """A job waiting for its sentence has started nothing: the check runs before any paid step."""
    import json
    import inspect
    from blueprint_pipeline import pubsub_handoff_scene_operations as operations
    from blueprint_pipeline.common import StageError
    from blueprint_pipeline.website_capture_preparation import require_task_named
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline/website_task_context.json").write_text(json.dumps(UNNAMED))
    with pytest.raises(StageError, match="website_task_not_named"):
        require_task_named(tmp_path)
    (tmp_path / "pipeline/website_task_context.json").write_text(json.dumps(NAMED))
    require_task_named(tmp_path)
    source = inspect.getsource(operations)
    assert source.index("require_task_named(capture_root)") < source.index("# Capture delivery prepares the site/task package.")


def test_the_wait_is_reported_as_waiting_on_the_site_only_when_the_website_accepts_it(monkeypatch):
    from blueprint_pipeline.website_preparation_outcome import NEW_REASONS_ENV, awaiting_job_sentence
    ledger = {"status": "failed_retryable", "capture_id": "walkthrough-capture-unnamed",
              "last_error": "website_scene_preparation: website_task_not_named", "source_payload_sha256": "a" * 64}
    monkeypatch.setenv(NEW_REASONS_ENV, "false")
    assert awaiting_job_sentence(ledger, task_context_digest=UNNAMED["context_digest"]) is None
    monkeypatch.setenv(NEW_REASONS_ENV, "true")
    outcome = awaiting_job_sentence(ledger, task_context_digest=UNNAMED["context_digest"])
    assert {key: value for key, value in outcome.items() if key != "evidence_digest"} == {
        "category": "insufficient_evidence", "reason": "website_task_not_named", "basis": "missing_evidence",
        "robot_id": None}
    assert outcome["evidence_digest"].startswith("sha256:") and len(outcome["evidence_digest"]) == 71
    assert awaiting_job_sentence({**ledger, "last_error": "website_scene_preparation: website_atlas_job_failed"},
                                 task_context_digest=UNNAMED["context_digest"]) is None
