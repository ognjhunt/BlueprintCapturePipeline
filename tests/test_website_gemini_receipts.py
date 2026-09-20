"""One paid attempt per source/task; no live model calls."""
import fcntl
import json

import pytest

from blueprint_pipeline import website_gemini_receipts as module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def context():
    value = {"schema_version": "website_site_task_context.v1", "request_id": "req",
             "scene_id": "scene", "capture_id": "capture", "confirmed": True,
             "confirmed_at": "2026-09-19T00:00:00Z", "description": "Move the box"}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value


def setup_call(tmp_path, monkeypatch):
    calls = []
    def reserve(**kwargs):
        calls.append(("reserve", kwargs))
        return {"status": "admitted"}, object()
    monkeypatch.setattr(module, "reserve_website_preparation_spend", reserve)
    def invoke():
        calls.append(("invoke", None))
        return {"status": "completed", "targets": ["box"]}
    return dict(output_root=tmp_path, binding={"source": "video-one", "prompt": "task"},
                task_context=context(), maximum_cost_usd=1.04, preflight=lambda: None, invoke=invoke), calls


def test_completed_replay_needs_no_new_key_admission_or_call(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    first = module.retained_gemini_call(**args)
    args["preflight"] = lambda: pytest.fail("replay should not need a key")
    assert module.retained_gemini_call(**args) == first
    assert [row[0] for row in calls] == ["reserve", "invoke"]
    assert calls[0][1]["provider"] == "google"


def test_timeout_retains_uncertain_call_without_second_reservation(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    def timeout():
        raise TimeoutError("provider may have charged")
    args["invoke"] = timeout
    with pytest.raises(TimeoutError):
        module.retained_gemini_call(**args)
    with pytest.raises(ValueError, match="requires_reconciliation"):
        module.retained_gemini_call(**args)
    assert len(calls) == 1
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["status"] == "submitting"


def test_denied_cross_worker_reservation_cannot_invoke(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    def refuse(**kwargs):
        raise ValueError("already_reserved")
    monkeypatch.setattr(module, "reserve_website_preparation_spend", refuse)
    with pytest.raises(ValueError, match="already_reserved"):
        module.retained_gemini_call(**args)
    assert not calls
    assert not list(tmp_path.glob("*.json"))


def test_tampered_result_or_binding_cannot_be_reused(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    module.retained_gemini_call(**args)
    path = next(tmp_path.glob("*.json"))
    receipt = json.loads(path.read_text())
    receipt["result"]["targets"] = ["unrelated chair"]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt_invalid"):
        module.retained_gemini_call(**args)
    assert len(calls) == 2


def test_changed_source_requires_new_admission(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    module.retained_gemini_call(**args)
    args["binding"] = {"source": "video-two", "prompt": "task"}
    module.retained_gemini_call(**args)
    assert [row[0] for row in calls] == ["reserve", "invoke", "reserve", "invoke"]
    assert calls[0][1]["binding_digest"] != calls[2][1]["binding_digest"]


def test_concurrent_worker_cannot_enter_locked_request(tmp_path, monkeypatch):
    args, calls = setup_call(tmp_path, monkeypatch)
    module.retained_gemini_call(**args)
    with next(tmp_path.glob("*.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="in_progress"):
            module.retained_gemini_call(**args)
    assert len(calls) == 2


def test_quote_includes_full_thinking_window_and_refuses_unpriced_model():
    assert module.gemini_quote(model="gemini-3.8-flash", input_tokens=1_048_576) == 1.04
    assert module.gemini_quote(model="gemini-3.8-flash", input_tokens=24000) == .27
    with pytest.raises(ValueError, match="pricing_refresh"):
        module.gemini_quote(model="unpriced", input_tokens=1)


def test_video_entrypoint_reuses_analysis_before_credentials_or_sdk(tmp_path, monkeypatch):
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
    import sys
    from types import SimpleNamespace
    import google
    args, calls = setup_call(tmp_path / "receipts", monkeypatch)
    video = tmp_path / "video.mov"
    video.write_bytes(b"original video")
    monkeypatch.setenv(analysis.GATE_ENV, "true")
    monkeypatch.setattr(analysis, "_api_key", lambda: ("fixture", "fixture"))
    monkeypatch.setattr(google, "genai", SimpleNamespace(), raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", google.genai)
    monkeypatch.setattr(analysis, "_analyze_removal_targets", lambda **kwargs: {**args["invoke"](), "input_video_sha256": analysis.sha256_file(video)})
    kwargs = dict(video_path=video, task_context=context(), output_root=tmp_path / "receipts")
    assert analysis.analyze_removal_targets(**kwargs)["targets"] == ["box"]
    monkeypatch.delenv(analysis.GATE_ENV)
    monkeypatch.setattr(analysis, "_api_key", lambda: (None, None))
    assert analysis.analyze_removal_targets(**kwargs)["targets"] == ["box"]
    assert [row[0] for row in calls] == ["reserve", "invoke"]
    video.write_bytes(b"changed video")
    with pytest.raises(ValueError, match="runtime_not_configured"):
        analysis.analyze_removal_targets(**kwargs)
    assert len(calls) == 2
