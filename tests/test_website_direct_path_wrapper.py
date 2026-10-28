"""The website preparation wrapper must not restore removed reporting gates."""

from __future__ import annotations

import json

import pytest
from types import SimpleNamespace

from blueprint_pipeline import run_e2e
from blueprint_pipeline.common import PipelineError
from tests import test_website_preparation_direct_path as direct_path


@pytest.fixture
def website_capture(tmp_path, monkeypatch):
    return direct_path.website_capture.__wrapped__(tmp_path, monkeypatch)


def test_website_preparation_completes_through_real_wrapper_without_generic_reports(
    website_capture,
    monkeypatch,
):
    """Run the actual qualification lane and wrapper, with only external work faked.

    The shared fixture replaces source/task grant reads and provider media/scene
    preparation. Capture lane orchestration, supervisor (if called), required
    stage validation, and wrapper completion remain real. There are no network,
    inference, simulator, spend, or live WebApp calls in this rehearsal.
    """
    fx = website_capture
    context = SimpleNamespace(
        capture_root=fx.capture,
        storage_root=fx.storage,
        bucket="bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        raw_prefix_uri="gs://bucket/scenes/scene-1/captures/capture-1/raw",
        descriptor_uri=fx.uri,
        descriptor_path=fx.capture / "capture_descriptor.json",
        raw_complete_path=fx.capture / "raw/capture_upload_complete.json",
    )
    monkeypatch.setattr(run_e2e, "resolve_local_capture_context", lambda *_: context)
    monkeypatch.setattr(
        run_e2e,
        "build_capture_preflight_report",
        lambda *_: {
            "status": "passed",
            "missing_required_inputs": [],
        },
    )

    result = run_e2e.run_end_to_end(
        capture_root=str(fx.capture),
        provider="openai",
        pipeline_lane="qualification",
        allow_live_supervisor_agents=False,
        supervisor_inference_budget_usd=0.0,
    )

    assert result["pipeline_status"] == "completed"
    ledger = json.loads((fx.capture / "pipeline/run_e2e_stage_ledger.json").read_text())
    assert ledger["status"] == "completed"
    assert ledger["failed_stage"] is None
    assert ledger["stages"]["capture_pipeline"]["status"] == "completed"
    assert result["task_evaluation_supervisor"] == {
        "status": "skipped",
        "reason": "website_scene_preparation_uses_native_handoff",
        "native_execution_complete": False,
        "proof_state_mutated_by_agent": False,
    }
    assert ledger["stages"]["task_evaluation_supervisor"]["status"] == "skipped"
    assert (
        ledger["stages"]["task_evaluation_supervisor"]["detail"]
        == result["task_evaluation_supervisor"]["reason"]
    )
    assert not (fx.capture / "pipeline/task_evaluation_supervisor").exists()
    assert not any(
        (fx.capture / "pipeline" / name).exists()
        for name in (
            "canonical_site_package.json",
            "readiness_decision.json",
            "buyer_trust_score.json",
        )
    )


@pytest.mark.parametrize(
    "change", ["not_website", "native_execution_complete", "evaluation_requested"]
)
def test_wrapper_retains_required_supervisor_outside_website_preparation(
    website_capture,
    monkeypatch,
    change,
):
    fx = website_capture
    context = SimpleNamespace(
        capture_root=fx.capture,
        storage_root=fx.storage,
        bucket="bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        descriptor_uri=fx.uri,
        descriptor_path=fx.capture / "capture_descriptor.json",
        raw_prefix_uri="gs://bucket/scenes/scene-1/captures/capture-1/raw",
        raw_complete_path=fx.capture / "raw/capture_upload_complete.json",
    )
    monkeypatch.setattr(run_e2e, "resolve_local_capture_context", lambda *_: context)
    monkeypatch.setattr(
        run_e2e,
        "build_capture_preflight_report",
        lambda *_: {"status": "passed", "missing_required_inputs": []},
    )
    real_pipeline = run_e2e.run_capture_pipeline

    def pipeline(**kwargs):
        result = real_pipeline(**kwargs)
        if change == "not_website":
            result["results"][0].pop("website_scene_preparation")
        elif change == "native_execution_complete":
            result["results"][0]["native_execution_complete"] = True
        return result

    monkeypatch.setattr(run_e2e, "run_capture_pipeline", pipeline)
    # The real supervisor reads the retained descriptor successfully, then
    # refuses absent scientific inputs. The wrapper must retain that refusal
    # for a non-preparation/full-evaluation call rather than claim completion.
    with pytest.raises(
        PipelineError, match="required_stage_not_complete:task_evaluation_supervisor:blocked"
    ):
        run_e2e.run_end_to_end(
            capture_root=str(fx.capture),
            provider="openai",
            pipeline_lane="qualification",
            run_evaluation_prep=change == "evaluation_requested",
            allow_live_supervisor_agents=False,
            supervisor_inference_budget_usd=0.0,
        )
    ledger = json.loads((fx.capture / "pipeline/run_e2e_stage_ledger.json").read_text())
    assert ledger["failed_stage"] == "task_evaluation_supervisor"
