"""A pulled Atlas preview job replays its next controller steps against this tree and buys nothing."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import atlas_preview_replay as replay
from blueprint_pipeline import website_atlas_pipeline as pipeline
from blueprint_pipeline import website_atlas_preview as preview
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

ROOT = Path(__file__).resolve().parents[1]


def _sha(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _tree(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A capture root as ``pull`` leaves it: the job, its allowance, task context and video."""
    monkeypatch.setenv(preview.ROUTING_ENV, "consented")
    capture = tmp_path / "storage/bucket/scenes/site-req1/captures/walkthrough-req1"
    (capture / "raw").mkdir(parents=True)
    (capture / "raw/walkthrough.mov").write_bytes(b"video")
    (capture / "pipeline").mkdir()
    context = {"schema_version": "website_site_task_context.v1", "request_id": "req1", "scene_id": "site-req1",
               "capture_id": "walkthrough-req1", "purpose": "scene_preparation", "confirmed": False,
               "confirmed_at": None, "description": "Move the blue box to the tray"}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    (capture / "pipeline/website_task_context.json").write_text(json.dumps(context))
    job = preview.enqueue_preview(
        descriptor=SimpleNamespace(site_submission_id="req1", metadata={}, capture_id="walkthrough-req1",
                                   scene_id="site-req1"),
        capture_root=capture, pipeline_dir=capture / "pipeline", source_video=lambda: capture / "raw/walkthrough.mov")
    assert job["state"] == "processing"
    source = _sha(b"video")
    policy = {"model": "gpt-6.1-sol", "maximum_review_cost_usd": .5, "pose_input_cost_usd": .01,
              "rgbd_view_cost_usd": .05, "rate_card_reference": "test fixture, not a rate card",
              "reviewed_by": "test", "reviewed_on": datetime.now(timezone.utc).date().isoformat()}
    configuration = {"schema_version": "website_atlas_configuration.v1", "source_video_digest": source,
                     "task_context_digest": context["context_digest"], "dispatch_authorized": False,
                     "authority": {"maximum_cost_usd": 5.0, "provider_retention": "account_does_not_retain"},
                     "policy": policy, "reserved_or_settled_cost_usd": 1.0, "remaining_cost_usd": 4.0}
    atlas = capture / "pipeline/atlas"
    (atlas / "configuration.json").write_text(json.dumps(configuration))
    survey = {"source_video_digest": source, "frames": [], "heldout_pixels_included": False}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    (atlas / "survey").mkdir()
    (atlas / "survey/atlas_video_survey.json").write_text(json.dumps(survey))
    return SimpleNamespace(storage=tmp_path / "storage", capture=capture, job=atlas / preview.JOB_FILE,
                           context=context, source=source, configuration=configuration,
                           replay_root=tmp_path / "replays")


def _run(site, **options):
    before = _tree(site.storage)
    report = replay.replay(site.job, replay_root=site.replay_root, overlay="relocated", **options)
    assert _tree(site.storage) == before  # the pulled evidence is byte-identical
    assert report["evidence"]["unchanged"] and report["new_paid_calls"] == 0
    assert json.loads(Path(report["report_path"]).read_text())["first_stop"] == report["first_stop"]
    stop = report["first_stop"]
    if stop["kind"] == "replay_unsupported":
        # The mirror cannot redirect retained absolute paths; the job's own stop is underneath.
        assert stop["code"] == "relocated_replay_cannot_resolve_retained_absolute_paths"
        report = {**report, "first_stop": stop["underlying"]}
    return report


def _retained_review(site, binding, maximum_cost_usd):
    """A completed Gemini receipt as the worker leaves it after paying for it."""
    request = {"binding": binding, "task_context_digest": site.context["context_digest"],
               "maximum_cost_usd": maximum_cost_usd, "source_video_digest": site.source}
    digest = canonical_digest(request)
    result = {"answer": "retained"}
    path = site.job.parent / "reviews" / f"{digest[7:]}.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"status": "completed", "request_digest": digest, "request": request,
                                "admission": {"status": "admitted"}, "result": result,
                                "result_digest": canonical_digest(result)}))
    return path


def _controller(monkeypatch, step):
    """Stands in for the Atlas pipeline's own stages; the boundaries it reaches are the real ones."""
    monkeypatch.setattr(pipeline, "run_atlas_pipeline", lambda **kwargs: step(**kwargs))


def _gemini_step(site, invoked):
    from blueprint_pipeline.website_gemini_receipts import retained_gemini_call

    def step(*, output_root, task_context, checkpoint, **_):
        def call(binding, cost):
            checkpoint("independent_review")
            return retained_gemini_call(output_root=output_root / "reviews", binding=binding,
                task_context=task_context, maximum_cost_usd=cost, preflight=lambda: None,
                invoke=lambda: invoked.append(binding) or {"answer": "new"}, source_video_digest=site.source)
        assert call({"kind": "retained"}, .25) == {"answer": "retained"}
        call({"kind": "next"}, .5)
        raise AssertionError("the replay continued past a new purchase")
    return step


def test_a_retained_receipt_is_adopted_and_the_next_purchase_is_reported_not_made(site, monkeypatch):
    retained = _retained_review(site, {"kind": "retained"}, .25)
    invoked = []
    _controller(monkeypatch, _gemini_step(site, invoked))
    report = _run(site)
    assert invoked == []  # neither the retained nor the new call reached a provider
    assert report["first_stop"]["kind"] == "would_buy" and report["first_stop"]["stage"] == "independent_review"
    assert report["first_stop"]["code"] == "google"
    [bought] = report["would_buy"]
    assert {key: bought[key] for key in ("boundary", "resource_class", "provider", "maximum_cost_usd",
                                         "request_count", "already_reserved", "stage")} == {
        "boundary": "webapp:preparation-spend", "resource_class": "evaluator_api", "provider": "google",
        "maximum_cost_usd": .5, "request_count": 1, "already_reserved": False, "stage": "independent_review"}
    assert bought["call_site"].startswith("website_gemini_receipts.retained_gemini_call:")
    assert bought["allowance"] == {"maximum_cost_usd": 5.0, "reserved_or_settled_cost_usd": 1.0,
                                   "remaining_cost_usd": 4.0}
    assert report["adopted"] == [{"path": str(retained), "record": "completed"}]
    assert report["steps_run"] == 0 and report["final_job"]["state"] == "processing"
    assert report["source_commit"] and report["job_digest"] == _sha(site.job.read_bytes())
    assert report["overlay"]["mode"] == "relocated" and not Path(report["overlay"]["copy"]).exists()
    assert {"configuration": str(site.job.parent / "configuration.json"),
            "task_context": str(site.capture / "pipeline/website_task_context.json")} == report["snapshot_sources"]


def test_a_purchase_over_the_snapshot_allowance_is_refused_as_webapp_refuses_it(site, monkeypatch, tmp_path):
    _retained_review(site, {"kind": "retained"}, .25)
    _controller(monkeypatch, _gemini_step(site, []))
    snapshot = tmp_path / "allowance.json"
    snapshot.write_text(json.dumps({**site.configuration, "reserved_or_settled_cost_usd": 4.6,
                                    "remaining_cost_usd": .4}))
    report = _run(site, configuration=str(snapshot))
    assert {key: report["first_stop"][key] for key in ("kind", "stage", "code", "exception")} == {
        "kind": "refusal", "stage": "independent_review", "code": "website_scene_preparation_budget_exhausted",
        "exception": "ValueError: website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted"}
    assert report["would_buy"] == [] and report["refused_spend"][0]["maximum_cost_usd"] == .5
    assert report["final_job"]["state"] == "failed" and report["snapshot_sources"]["configuration"] == str(snapshot)


def test_the_real_controller_reaches_its_first_purchase_the_scene_review(site):
    report = _run(site)
    assert report["first_stop"]["kind"] == "would_buy" and report["stages_reached"] == ["plan"]
    [bought] = report["would_buy"]
    assert (bought["provider"], bought["resource_class"], bought["maximum_cost_usd"], bought["request_count"]) == (
        "openai", "openai_api_candidate", .5, 1)
    assert bought["call_site"].startswith("atlas_scene_planner._bound_review_attempt:")
    # Had the job's retained selection been pulled, the review would have been adopted instead.
    assert str(site.job.parent / "selection_binding.json") in bought["unpulled_before"]
    assert str(site.job.parent / "selection_binding.json") in report["absent_paths_probed"]
    assert not (site.job.parent / "scene_planner").exists()


def test_the_real_controller_refuses_an_allowance_too_small_for_the_next_stage(site, tmp_path):
    snapshot = tmp_path / "allowance.json"
    snapshot.write_text(json.dumps({**site.configuration, "remaining_cost_usd": .9}))
    report = _run(site, configuration=str(snapshot))
    assert report["first_stop"]["kind"] == "refusal"
    assert report["first_stop"]["code"] == "atlas_runtime_budget_insufficient"
    assert report["first_stop"]["raised_at"].split(" > ")[-1].startswith("website_atlas_pipeline._run_atlas_pipeline:")
    assert report["would_buy"] == [] and report["steps"][-1]["state"] == "failed"


def test_a_failed_job_is_retried_in_the_overlay_and_a_missing_input_names_its_path(site):
    job = json.loads(site.job.read_text())
    site.job.write_text(json.dumps({**job, "state": "failed", "reason": "atlas_background_review_required"}))
    (site.capture / "raw/walkthrough.mov").unlink()
    report = _run(site)
    assert report["retried_in_overlay"] is True
    assert {key: report["first_stop"][key] for key in ("kind", "stage", "code")} == {
        "kind": "missing_input", "stage": "queued", "code": str(site.capture / "raw/walkthrough.mov")}
    assert report["first_stop"]["raised_at"].split(" > ")[-1].startswith("local_reconstruction_adapters._sha256_file:")
    assert json.loads(site.job.read_text())["state"] == "failed"  # the pulled job is untouched


def test_a_refusal_after_looking_for_an_unpulled_record_names_it(site, monkeypatch):
    def step(*, output_root, **_):
        (output_root / "work").mkdir()
        (output_root / "work/scratch.png").exists()  # the replay's own work in progress is not reported
        if not (output_root / "pose/outputs/depth-0.exr").is_file():
            raise ValueError("website_atlas_retained_output_changed")
    _controller(monkeypatch, step)
    report = _run(site)
    assert report["first_stop"]["code"] == "website_atlas_retained_output_changed"
    assert report["first_stop"]["unpulled_before"][-1] == str(site.job.parent / "pose/outputs/depth-0.exr")
    assert str(site.job.parent / "work/scratch.png") not in report["absent_paths_probed"]


def test_an_unstubbed_network_call_fails_loudly_even_where_the_controller_catches_errors(site, monkeypatch):
    import urllib.request

    def step(**_):
        try:
            urllib.request.urlopen("https://provider.example.test/v1/answer", timeout=5)
        except Exception:  # noqa: BLE001 - a stage that swallows its transport error
            pass
        return {"status": "processing", "stage": "unreachable"}
    _controller(monkeypatch, step)
    report = _run(site)
    assert report["first_stop"]["kind"] == "error"
    assert report["first_stop"]["code"] == "unstubbed_network_call:provider.example.test"
    assert report["network_attempts"][0]["target"] == "provider.example.test"
    assert report["final_job"]["stage"] == "queued"


def test_no_stub_outlives_the_replay(site, monkeypatch):
    import socket
    from blueprint_pipeline import paid_resource_admission, website_task_context
    from blueprint_pipeline import website_gemini_receipts  # noqa: F401 - binds an alias the replay swaps
    originals = (website_task_context.website_webapp_request, paid_resource_admission.require_paid_resource_admission,
                 socket.socket.connect, socket.getaddrinfo, Path.exists, preview._failed_step)
    _controller(monkeypatch, _gemini_step(site, []))
    _retained_review(site, {"kind": "retained"}, .25)
    _run(site)
    assert (website_task_context.website_webapp_request, paid_resource_admission.require_paid_resource_admission,
            socket.socket.connect, socket.getaddrinfo, Path.exists, preview._failed_step) == originals
    assert "connect" not in vars(socket.socket)


def test_the_inner_overlay_never_mounts_outside_a_private_namespace(site, tmp_path, monkeypatch):
    mounted = []
    monkeypatch.setattr(replay.subprocess, "run", lambda *a, **k: mounted.append(a))
    for outer in (replay._namespaces(), ""):
        with pytest.raises(SystemExit):
            replay.main(["--job", str(site.job), "--inner-overlay", str(tmp_path), "--outer-namespace", outer,
                         "--report", str(tmp_path / "inner.json")])
    assert mounted == [] and not (tmp_path / "inner.json").exists()


def test_the_command_runs_at_the_capture_roots_own_path_in_a_private_namespace(site, tmp_path):
    tmp_path.joinpath("probe").mkdir()
    if replay._namespace_prefix(tmp_path / "probe") is None:
        pytest.skip("this host permits no private mount namespace")
    snapshot = tmp_path / "allowance.json"
    snapshot.write_text(json.dumps({**site.configuration, "remaining_cost_usd": .9}))
    before = _tree(site.storage)
    completed = subprocess.run(
        [sys.executable, "-m", "blueprint_pipeline.atlas_preview_replay", "--job", str(site.job),
         "--configuration", str(snapshot), "--replay-root", str(site.replay_root), "--report", str(tmp_path / "r.json")],
        capture_output=True, text=True, cwd=ROOT, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "src")})
    report = json.loads((tmp_path / "r.json").read_text())
    assert completed.returncode == 3, completed.stderr
    assert report["overlay"]["mode"] == "bind" and report["evidence"]["unchanged"]
    assert report["first_stop"]["code"] == "atlas_runtime_budget_insufficient"
    assert _tree(site.storage) == before


def test_a_relocated_refusal_is_not_presented_as_the_jobs_stop(site, tmp_path):
    snapshot = tmp_path / "allowance.json"
    snapshot.write_text(json.dumps({**site.configuration, "remaining_cost_usd": .9}))
    report = replay.replay(site.job, replay_root=site.replay_root, overlay="relocated", configuration=str(snapshot))
    assert report["first_stop"]["kind"] == "replay_unsupported"
    assert report["first_stop"]["underlying"]["code"] == "atlas_runtime_budget_insufficient"
