"""Offline, explicitly synthetic explanation fixtures; no inference or real labels."""

import copy
import json
from dataclasses import replace

import pytest

from blueprint_pipeline.argus_shadow import ShadowError, binding, seal
from blueprint_pipeline.argus_video_explanation import (
    ADAPTER_ID, OFFLINE_REASON, default_video_profile,
    prepare_video_explanation,
)
from blueprint_pipeline.policy_canary_episode_interpretation_closeout import (
    _load_profile, materialize_policy_canary_episode_interpretations,
)
from tests.test_episode_interpretation import _episode_root, _request
from tests.test_policy_canary_episode_interpretation_closeout import _session


def _prepared(tmp_path):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    return prepare_video_explanation(
        _request(data), intended_task="Move cup to the marked target without dropping it",
        duration_s=1, evidence_kind="synthetic_fixture"), data


def _response(tmp_path, prepared, *, outcome="success", checks=True):
    labels = {"completion": {"task_completed": outcome, "reason": "SYNTHETIC model output"}}
    if checks:
        labels["criterion_evidence"] = {
            name: {"status": "satisfied", "finding": "SYNTHETIC assertion; not ground truth",
                   "evidence_role": "state_trace", "evidence_digest": prepared[
                       "model_input"]["artifacts"]["state_trace"]["logical_digest"]}
            for name in prepared["model_input"]["task_success_contract"]["criteria"]}
    path = tmp_path / "SYNTHETIC-imported-response.json"
    path.write_text(json.dumps({"parse_ok": True, "labels": labels}))
    return {"request_digest": prepared["request_digest"],
            "raw_response": binding(tmp_path, path.name),
            "inference_identity": {"model_requested": "openai/gpt-6.1-sol",
                                   "model_served": "gpt-6.1-sol", "provider": "openai",
                                   "generation_id": "SYNTHETIC_NO_GENERATION"}}


def _edit_raw(tmp_path, record, edit):
    path = tmp_path / record["raw_response"]["path"]
    raw = json.loads(path.read_text())
    edit(raw["labels"])
    path.write_text(json.dumps(raw))
    record["raw_response"] = binding(tmp_path, path.name)


def test_default_selects_sol_argus_and_cannot_invoke():
    profile, reason = _load_profile({})
    assert profile == default_video_profile()
    assert profile["interpreter_id"] == ADAPTER_ID
    assert profile["model"] == "gpt-6.1-sol"
    assert profile["paid_execution_authorized"] is False
    assert profile["live_processing_enabled"] is False
    assert profile["max_cost_usd"] == 0
    assert reason == OFFLINE_REASON


def test_profile_callers_cannot_change_pinned_prompt():
    profile = default_video_profile()
    original = copy.deepcopy(profile)
    profile["adapted_prompt"]["sha256"] = "sha256:" + "0" * 64
    assert default_video_profile() == original


def test_explicit_argus_profile_cannot_enable_processing(tmp_path):
    profile = default_video_profile()
    profile["live_processing_enabled"] = True
    profile = seal(profile, "profile_digest")
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    loaded, reason = _load_profile({"BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETER_PROFILE_FILE": str(path)})
    assert loaded is None and reason == "interpreter_profile_invalid"


def test_changed_native_trace_is_refused(tmp_path):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    request = _request(data)
    request.state_trace["task_state_samples"][0]["step_index"] = 999
    with pytest.raises(ShadowError, match="native_trace_changed"):
        prepare_video_explanation(request, intended_task="Move cup", duration_s=1,
                                  evidence_kind="synthetic_fixture")


def test_stale_video_bytes_are_refused_during_preparation(tmp_path):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    request = _request(data)
    data["video_path"].write_bytes(b"changed-after-original-input-receipt")
    with pytest.raises(ShadowError, match="artifact_changed"):
        prepare_video_explanation(request, intended_task="Move cup", duration_s=1,
                                  evidence_kind="synthetic_fixture")


def test_offline_preparation_never_reads_streamed_only_frames(tmp_path):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    request = _request(data)
    def forbidden(index):
        pytest.fail("Offline specification must not read remote frames")
    request = replace(request, frame_reader=forbidden)
    request.ordered_frame_paths[0].unlink()
    with pytest.raises(ShadowError, match="artifact_missing"):
        prepare_video_explanation(request, intended_task="Move cup", duration_s=1,
                                  evidence_kind="synthetic_fixture")


def test_default_closeout_never_constructs_provider_or_changes_scores(tmp_path, monkeypatch):
    data, session = _session(tmp_path, deterministic_success=False)
    original = copy.deepcopy(session["episodes"])
    def forbidden(*args, **kwargs):
        pytest.fail("Offline default must not construct a paid invoker")
    monkeypatch.setattr("blueprint_pipeline.policy_canary_episode_interpretation_closeout.OpenAIAgentsSDKInvoker", forbidden)
    result = materialize_policy_canary_episode_interpretations(
        run_root=tmp_path, evidence_root=data["root"], session_result=session,
        environment={"BLUEPRINT_ALLOW_LIVE_AGENTS_SDK": "true"})
    assert result["episode_interpretation"]["provider_call_count"] == 0
    assert result["episode_interpretation"]["score_overwrite_performed"] is False
    assert result["episode_interpretation"]["interpreter_profile_digest"] == default_video_profile()["profile_digest"]
    assert [r["episode"]["score"] for r in result["episodes"]] == [r["episode"]["score"] for r in original]
    plans = list((data["root"] / "episode_interpretation" / "plans").glob("*.json"))
    assert plans
    assert all(json.loads(path.read_text())["abstention_reason"] == OFFLINE_REASON for path in plans)
