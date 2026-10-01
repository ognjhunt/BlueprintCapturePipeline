"""Offline, explicitly synthetic explanation fixtures; no inference or real labels."""

import copy
import json
from dataclasses import replace

import pytest

from blueprint_pipeline.argus_shadow import ShadowError, binding, seal
from blueprint_pipeline.argus_video_explanation import (
    ADAPTER_ID, OFFLINE_REASON, REPO, default_video_profile,
    import_video_explanation, prepare_video_explanation,
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


def test_explicit_historical_profile_remains_explicit(tmp_path):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    source = REPO / "docs/arm_decision_proof_v1/manifests/policy_canary_episode_interpreter_profile.v1.json"
    value = json.loads(source.read_text())
    # Keep the historical profile bytes unchanged; the valid explicit legacy
    # fixture is sealed with the loader's existing digest convention.
    value["profile_digest"] = canonical_digest(value, digest_field="profile_digest")
    path = tmp_path / "explicit-legacy-profile.json"
    path.write_text(json.dumps(value))
    profile, reason = _load_profile({"BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETER_PROFILE_FILE": str(path)})
    assert reason is None
    assert profile["model"] == "gpt-6-luna"


def test_explicit_argus_profile_cannot_enable_processing(tmp_path):
    profile = default_video_profile()
    profile["live_processing_enabled"] = True
    profile = seal(profile, "profile_digest")
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    loaded, reason = _load_profile({"BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETER_PROFILE_FILE": str(path)})
    assert loaded is None and reason == "interpreter_profile_invalid"


def test_request_uses_exact_criteria_and_blinds_current_score(tmp_path):
    prepared, data = _prepared(tmp_path)
    model_input = prepared["model_input"]
    assert model_input["task_success_contract"] == data["contract"]
    assert "deterministic_score" not in model_input["artifacts"]
    assert "authoritative_result" not in model_input
    assert "candidate_policy_id" not in model_input
    assert "criterion_evidence" in prepared["prompt"]
    assert prepared["provider_disclosure_authorized"] is False
    assert prepared["real_corpus_admission"] is False


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


def test_apparent_success_cannot_change_authoritative_failure(tmp_path):
    prepared, data = _prepared(tmp_path)
    before = data["score_path"].read_bytes()
    receipt = import_video_explanation(prepared, _response(tmp_path, prepared), evidence_root=tmp_path)
    assert receipt["apparent_outcome"] == "appears_complete"
    assert receipt["authoritative_result"]["task_succeeded"] is False
    assert data["score_path"].read_bytes() == before
    assert receipt["model_calls"] == 0
    assert receipt["proof_boundary"]["ranking_or_promotion_effect"] == "none"
    assert receipt["proof_boundary"]["physical_validity_established"] is False
    assert receipt["calibrated_success_probability"] is None


@pytest.mark.parametrize("kind", ["generated_video", "simulator_recording", "physical_recording"])
def test_declared_video_kind_never_proves_physical_validity(tmp_path, kind):
    prepared, _ = _prepared(tmp_path)
    # Changing a declared kind is not real-episode admission or physical evidence.
    prepared["model_input"]["declared_evidence_kind"] = kind
    prepared = seal(prepared, "request_digest")
    receipt = import_video_explanation(prepared, _response(tmp_path, prepared), evidence_root=tmp_path)
    assert receipt["proof_boundary"]["physical_validity_established"] is False
    assert receipt["proof_boundary"]["real_corpus_admission"] is False


def test_missing_required_evidence_is_unclear(tmp_path):
    prepared, _ = _prepared(tmp_path)
    receipt = import_video_explanation(prepared, _response(tmp_path, prepared, checks=False), evidence_root=tmp_path)
    assert receipt["apparent_outcome"] == "unclear"
    assert receipt["missing_evidence_criteria"]


def test_unbound_criterion_claim_is_unclear(tmp_path):
    prepared, _ = _prepared(tmp_path)
    record = _response(tmp_path, prepared)
    _edit_raw(tmp_path, record, lambda labels: labels["criterion_evidence"]["motion"].update(
        evidence_digest="sha256:" + "0" * 64))
    receipt = import_video_explanation(prepared, record, evidence_root=tmp_path)
    assert receipt["apparent_outcome"] == "unclear"
    assert "motion" in receipt["missing_evidence_criteria"]


def test_acceptance_contract_is_not_observed_outcome_evidence(tmp_path):
    prepared, _ = _prepared(tmp_path)
    record = _response(tmp_path, prepared)
    def edit(labels):
        for check in labels["criterion_evidence"].values():
            check.update(evidence_role="task_success_contract", evidence_digest=prepared[
                "model_input"]["artifacts"]["task_success_contract"]["logical_digest"])
    _edit_raw(tmp_path, record, edit)
    receipt = import_video_explanation(prepared, record, evidence_root=tmp_path)
    assert receipt["apparent_outcome"] == "unclear"
    assert receipt["missing_evidence_criteria"]


@pytest.mark.parametrize("field,value", [
    ("finding", {"invented": "structured finding"}),
    ("evidence_digest", ["unhashable reference"]),
])
def test_malformed_criterion_support_abstains(tmp_path, field, value):
    prepared, _ = _prepared(tmp_path)
    record = _response(tmp_path, prepared)
    _edit_raw(tmp_path, record, lambda labels: labels["criterion_evidence"]["motion"].update({field: value}))
    receipt = import_video_explanation(prepared, record, evidence_root=tmp_path)
    assert receipt["apparent_outcome"] == "unclear"
    assert "motion" in receipt["missing_evidence_criteria"]


def test_malformed_explanation_is_refused(tmp_path):
    prepared, _ = _prepared(tmp_path)
    record = _response(tmp_path, prepared)
    _edit_raw(tmp_path, record, lambda labels: labels["completion"].update(reason={"not": "text"}))
    with pytest.raises(ShadowError, match="narrative_invalid"):
        import_video_explanation(prepared, record, evidence_root=tmp_path)


def test_success_with_violated_criterion_is_unclear(tmp_path):
    prepared, _ = _prepared(tmp_path)
    record = _response(tmp_path, prepared)
    _edit_raw(tmp_path, record, lambda labels: labels["criterion_evidence"]["motion"].update(status="violated"))
    receipt = import_video_explanation(prepared, record, evidence_root=tmp_path)
    assert receipt["apparent_outcome"] == "unclear"
    assert receipt["completion_criterion_contradiction"] is True


def test_undo_and_failure_taxonomy_are_advisory(tmp_path):
    prepared, _ = _prepared(tmp_path)
    record = _response(tmp_path, prepared, outcome="success_then_undone")
    def edit(labels):
        labels["completion"].update(goal_reached_at_s=.2, undone_at_s=.8)
        labels["performance_review"] = "SYNTHETIC apparent success was undone"
        labels["timeline_columns"] = ["start_s", "end_s", "action"]
        labels["timeline"] = [[.2, .8, "SYNTHETIC goal regression"]]
        labels["operator_mistakes"] = [{"category": "goal_undone", "severity": "high", "t_s": .8}]
        labels["data_issues"] = [{"category": "new_model_tag", "severity": "low", "t_s": None}]
    _edit_raw(tmp_path, record, edit)
    receipt = import_video_explanation(prepared, record, evidence_root=tmp_path)
    assert receipt["apparent_outcome"] == "appears_incomplete"
    assert receipt["events"]["undone_at_s"] == .8
    assert receipt["temporal_annotations"]["timeline"][0][:2] == [.2, .8]
    assert receipt["narrative"]["performance_review"] == "SYNTHETIC apparent success was undone"
    assert [o["taxonomy"] for o in receipt["observations"]] == ["goal_regression", "unclassified_observation"]
    assert all(o["severity_semantics"] == "training_data_impact_not_safety" for o in receipt["observations"])


def test_bad_explanation_timestamps_are_refused(tmp_path):
    prepared, _ = _prepared(tmp_path)
    record = _response(tmp_path, prepared)
    def edit(labels):
        labels["timeline_columns"] = ["start_s", "end_s"]
        labels["timeline"] = [[.8, .2]]
    _edit_raw(tmp_path, record, edit)
    with pytest.raises(ShadowError, match="annotation_interval_invalid"):
        import_video_explanation(prepared, record, evidence_root=tmp_path)


def test_other_model_or_changed_response_is_refused(tmp_path):
    prepared, _ = _prepared(tmp_path)
    record = _response(tmp_path, prepared)
    wrong = copy.deepcopy(record)
    wrong["inference_identity"]["model_served"] = "unapproved_model"
    with pytest.raises(ShadowError, match="model_provenance"):
        import_video_explanation(prepared, wrong, evidence_root=tmp_path)
    (tmp_path / record["raw_response"]["path"]).write_text("changed")
    with pytest.raises(ShadowError, match="artifact_changed"):
        import_video_explanation(prepared, record, evidence_root=tmp_path)


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
