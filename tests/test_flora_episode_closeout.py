"""ADP-009D research interpretation preserves sealed deterministic evidence."""

from __future__ import annotations

from dataclasses import replace
import copy
import json
from pathlib import Path
import shutil

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.episode_interpretation import (
    DeterministicFixtureInterpreter,
    EpisodeInterpretationError,
    validate_interpreter_adapter_metadata,
    validate_interpreter_prompt_identity,
)
from blueprint_pipeline import policy_canary_episode_interpretation_closeout as closeout
from tests.test_episode_interpretation import _output
from tests.test_policy_canary_episode_interpretation_closeout import (
    _FixtureRunner, _rights_for_all, _session,
)


class _ResearchInterpreter(DeterministicFixtureInterpreter):
    """No model calls; tests only the real profile/receipt/closeout boundary."""

    def __init__(self, output):
        super().__init__(output)
        self._identity = replace(
            self._identity, interpreter_id="flora_episode_interpreter_v1",
            provider_id="nvidia", runtime="flora_rewarder_batch_v1",
            model="nvidia/Flora-Rewarder", model_version="pinned-fixture-revision",
        )
        self.request = None
        self.batch_calls = []

    def run_batch(self, requests, *, rights_attestation_paths):
        assert all(path.is_file() for path in rights_attestation_paths.values())
        self.batch_calls.append([request.episode_id for request in requests])

    def prompt_identity(self):
        return {
            "prompt_contract_version": "flora_reward_prompts.v1",
            "prompt_digest": "sha256:" + "c" * 64,
        }

    def interpret(self, request):
        self.request = request
        return super().interpret(request)

    def adapter_metadata(self):
        return {
            "schema_version": "flora_episode_reward_evidence.v1",
            "input_bundle_digest": self.request.input_receipt["input_bundle_digest"],
            "interpreter_identity_digest": canonical_digest(self.identity.__dict__),
            "model_input_digest": "sha256:" + "d" * 64,
            "research_only": True,
            "learned_interpretation_only": True,
            "ranking_or_promotion_effect": "none",
            "rewards": {"terminal": 1.0, "dense": [0.0, 1.0], "rubric": [1.0]},
        }


def _profile(**changes):
    value = {
        "schema_version": closeout.FLORA_PROFILE_SCHEMA_VERSION,
        "status": "configured", "research_only": True,
        "fixture_only": True, **changes,
    }
    value["profile_digest"] = canonical_digest(value)
    return value


def test_configured_flora_uses_own_namespace_and_replay_preserves_scores(tmp_path, monkeypatch):
    data, result = _session(tmp_path, deterministic_success=False)
    original_score = data["score_path"].read_bytes()
    original_result = json.dumps(result, sort_keys=True)
    old = DeterministicFixtureInterpreter(_output(data))
    old_rights = tmp_path / "old-rights"
    _rights_for_all(data, result, old_rights, old)
    historical = closeout.materialize_policy_canary_episode_interpretations(
        run_root=tmp_path, evidence_root=data["root"], session_result=result,
        runner=_FixtureRunner(old), rights_root=old_rights, environment={},
    )
    interpreter = _ResearchInterpreter(_output(data))
    rights = tmp_path / "flora-rights"
    _rights_for_all(data, result, rights, interpreter)
    profile = _profile()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    output_roots = []

    def build(value, *, output_root=None):
        output_roots.append(output_root)
        return closeout.AdapterEpisodeInterpretationRunner(interpreter, profile=value)

    monkeypatch.setattr(closeout, "build_flora_episode_runner", build)
    arguments = dict(
        run_root=tmp_path, evidence_root=data["root"], session_result=historical,
        rights_root=rights,
        environment={"BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETER_PROFILE_FILE": str(profile_path)},
    )
    first = closeout.materialize_policy_canary_episode_interpretations(**arguments)
    second = closeout.materialize_policy_canary_episode_interpretations(**arguments)
    assert interpreter.call_count == 20
    assert len(interpreter.batch_calls) == 1
    assert len(interpreter.batch_calls[0]) == 20
    assert first["episode_interpretation"]["reused_receipt_count"] == 0
    assert second["episode_interpretation"]["reused_receipt_count"] == 20
    assert first["episode_interpretation"]["disagreement_count"] == 20
    assert first["episode_interpretation"]["interpreter_profile_digest"] == profile["profile_digest"]
    assert "flora" in output_roots[0].parts
    assert data["score_path"].read_bytes() == original_score
    assert json.dumps(result, sort_keys=True) == original_result
    assert all(row["episode"]["score"]["task_succeeded"] is False for row in first["episodes"])
    artifact = first["episodes"][0]["evidence_artifacts"]["episode_interpretation"]
    receipt = json.loads((data["root"] / artifact["relative_path"]).read_text())
    assert receipt["prompt_contract_version"] == "flora_reward_prompts.v1"
    assert receipt["adapter_metadata"]["rewards"]["terminal"] == 1.0
    assert "deterministic_score" not in json.dumps(receipt["adapter_metadata"])
    assert "task_succeeded" not in json.dumps(receipt["adapter_metadata"])
    assert "candidate_policy_id" not in json.dumps(receipt["adapter_metadata"])
    assert receipt["proof_boundary"]["ranking_or_promotion_effect"] == "none"

    # Same videos and model identity with changed sampling still require new inference.
    changed = _profile(sampling_seed=2)
    profile_path.write_text(json.dumps(changed))
    third = closeout.materialize_policy_canary_episode_interpretations(**arguments)
    assert interpreter.call_count == 40
    assert third["episode_interpretation"]["reused_receipt_count"] == 0


def test_invalid_flora_configuration_cannot_invoke_or_change_score(tmp_path, monkeypatch):
    data, result = _session(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile(research_only=False)))

    def refuse(*args, **kwargs):
        raise ValueError("flora_research_only_required")

    monkeypatch.setattr(closeout, "build_flora_episode_runner", refuse)
    output = closeout.materialize_policy_canary_episode_interpretations(
        run_root=tmp_path, evidence_root=data["root"], session_result=result,
        environment={"BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETER_PROFILE_FILE": str(profile_path)},
    )
    assert output["episode_interpretation"]["abstained_count"] == 20
    assert output["episode_interpretation"]["provider_call_count"] == 0
    assert [row["episode"]["score"] for row in output["episodes"]] == [
        row["episode"]["score"] for row in result["episodes"]
    ]


def test_failed_cohort_retains_attempts_and_never_reinvokes_on_replay(tmp_path):
    data, result = _session(tmp_path)
    interpreter = _ResearchInterpreter(_output(data))
    rights = tmp_path / "rights"
    _rights_for_all(data, result, rights, interpreter)

    def fail(requests, *, rights_attestation_paths):
        interpreter.batch_calls.append([request.episode_id for request in requests])
        raise RuntimeError("fixture_runtime_failure")

    interpreter.run_batch = fail
    runner = closeout.AdapterEpisodeInterpretationRunner(interpreter, profile=_profile())
    arguments = dict(run_root=tmp_path, evidence_root=data["root"], session_result=result,
                     runner=runner, rights_root=rights, environment={})
    first = closeout.materialize_policy_canary_episode_interpretations(**arguments)
    second = closeout.materialize_policy_canary_episode_interpretations(**arguments)
    assert len(interpreter.batch_calls) == 1
    assert interpreter.call_count == 0
    assert first["episode_interpretation"]["abstained_count"] == 20
    assert {row["abstention_reason"] for row in first["episode_interpretation"]["receipts"]} == {
        "interpreter_batch_execution_unavailable"
    }
    assert second["episode_interpretation"]["reused_receipt_count"] == 20
    assert sum(row["role"] == "episode_interpretation_attempt" for row in first["artifact_inventory"]) == 20


def test_cohort_excludes_missing_rights_before_runtime(tmp_path):
    data, result = _session(tmp_path)
    interpreter = _ResearchInterpreter(_output(data))
    rights = tmp_path / "rights"
    _rights_for_all(data, result, rights, interpreter)
    sorted(rights.glob("*.json"))[0].unlink()
    output = closeout.materialize_policy_canary_episode_interpretations(
        run_root=tmp_path, evidence_root=data["root"], session_result=result,
        runner=closeout.AdapterEpisodeInterpretationRunner(interpreter, profile=_profile()),
        rights_root=rights, environment={},
    )
    assert len(interpreter.batch_calls[0]) == 19
    assert interpreter.call_count == 19
    assert output["episode_interpretation"]["abstained_count"] == 1


def test_cohort_excludes_missing_review_video_before_runtime(tmp_path):
    data, result = _session(tmp_path)
    interpreter = _ResearchInterpreter(_output(data))
    rights = tmp_path / "rights"
    _rights_for_all(data, result, rights, interpreter)
    result["episodes"][0]["evidence_artifacts"].pop("review_video")
    output = closeout.materialize_policy_canary_episode_interpretations(
        run_root=tmp_path, evidence_root=data["root"], session_result=result,
        runner=closeout.AdapterEpisodeInterpretationRunner(interpreter, profile=_profile()),
        rights_root=rights, environment={},
    )
    assert len(interpreter.batch_calls[0]) == 19
    assert interpreter.call_count == 19
    assert "episode-0" not in interpreter.batch_calls[0]
    assert output["episode_interpretation"]["abstained_count"] == 1


def test_inferred_unclear_retains_adapter_evidence_and_prompt(tmp_path):
    data, result = _session(tmp_path)
    output = _output(data).model_copy(update={"episode_outcome": "unclear"})
    interpreter = _ResearchInterpreter(output)
    rights = tmp_path / "rights"
    _rights_for_all(data, result, rights, interpreter)
    runner = closeout.AdapterEpisodeInterpretationRunner(interpreter, profile=_profile())
    report = closeout.materialize_policy_canary_episode_interpretations(
        run_root=tmp_path, evidence_root=data["root"], session_result=result,
        runner=runner, rights_root=rights, environment={},
    )
    artifact = report["episodes"][0]["evidence_artifacts"]["episode_interpretation"]
    receipt = json.loads((data["root"] / artifact["relative_path"]).read_text())
    assert receipt["status"] == "abstained"
    assert receipt["deterministic_agreement"] == "abstains"
    assert receipt["adapter_metadata"]["research_only"] is True
    assert receipt["prompt_contract_version"] == "flora_reward_prompts.v1"


@pytest.mark.parametrize("field,value", [
    ("model_input_digest", "unpinned"),
    ("input_bundle_digest", "sha256:" + "a" * 64),
    ("research_only", False),
    ("ranking_or_promotion_effect", "winner"),
    ("rewards", {"terminal": float("nan")}),
])
def test_adapter_metadata_rejects_authority_or_identity_drift(tmp_path, field, value):
    data, result = _session(tmp_path)
    interpreter = _ResearchInterpreter(_output(data))
    rights = tmp_path / "rights"
    _rights_for_all(data, result, rights, interpreter)
    # Reuse the real request construction exercised by closeout, capturing its request.
    closeout.materialize_policy_canary_episode_interpretations(
        run_root=tmp_path, evidence_root=data["root"], session_result=result,
        runner=closeout.AdapterEpisodeInterpretationRunner(interpreter, profile=_profile()),
        rights_root=rights, environment={},
    )
    metadata = {**interpreter.adapter_metadata(), field: value}
    with pytest.raises(EpisodeInterpretationError, match="adapter_metadata_invalid"):
        validate_interpreter_adapter_metadata(
            metadata, input_bundle_digest=interpreter.request.input_receipt["input_bundle_digest"],
            interpreter_identity_digest=canonical_digest(interpreter.identity.__dict__),
        )


def test_invalid_prompt_identity_is_rejected_before_inference():
    with pytest.raises(EpisodeInterpretationError, match="prompt_identity_invalid"):
        validate_interpreter_prompt_identity({"prompt_contract_version": "flora", "prompt_digest": "unknown"})


def _real_flora_profile(tmp_path, contract):
    """Use the real adapter profile validator with a clearly synthetic fixture."""
    from blueprint_pipeline.flora_rewarder_runtime import FLORA_MODEL, FLORA_REVISION, MODES

    fixture = json.loads((Path(__file__).parent / "fixtures/flora_rewarder_output.v1.json").read_text())
    profile = {
        "schema_version": closeout.FLORA_PROFILE_SCHEMA_VERSION, "status": "configured",
        "interpreter_id": "flora_episode_interpreter_v1", "provider_id": "nvidia",
        "runtime": "flora_rewarder_batch_v1", "model": FLORA_MODEL,
        "model_version": FLORA_REVISION, "source_revision": FLORA_REVISION,
        "research_only": True, "modes": list(MODES), "execution_site": "local",
        "runtime_config": {
            "python_executable": "/fixture/python", "source_root": str(tmp_path / "source"),
            "model_path": str(tmp_path / "model"), "output_root": str(tmp_path / "raw"),
            "timeout_seconds": 60, "tensor_parallel_size": 1, "seed": 0,
            "rubric_num_samples": 9,
        },
        "task": {
            "instruction": fixture["records"]["terminal"]["task"],
            "task_success_contract_digest": contract["contract_digest"], "robot": "Franka arm",
            "cameras": ["Fixed review camera"], "rubric": fixture["rubric"],
        },
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    return profile


class _UpstreamFixtureRunner:
    """Actual upstream-shaped JSON; deliberately performs no model execution."""

    def __init__(self):
        self.calls = []

    def run_batch(self, *, profile, jobs, output_root):
        self.calls.append(copy.deepcopy(jobs))
        fixture = json.loads((Path(__file__).parent / "fixtures/flora_rewarder_output.v1.json").read_text())
        assert profile["task"]["rubric"] == fixture["rubric"]
        assert output_root.is_dir()
        records = {mode: [] for mode in fixture["records"]}
        for job in jobs:
            # Only admitted visual/task inputs reach the model runtime seam.
            assert set(job) == {"id", "task", "robot", "cameras", "rubric", "videos"}
            assert "pi05" not in json.dumps(job)
            assert "groot" not in json.dumps(job)
            for index, path in enumerate(job["videos"]):
                assert Path(path).is_file()
                assert Path(path).name == f"camera-{index}.mp4"
            for mode, row in fixture["records"].items():
                records[mode].append({**copy.deepcopy(row), "id": job["id"], "task": job["task"]})
        return {"records": records, "errors": [], "executions": [], "wall_time_seconds": 0.0}


def test_real_flora_adapter_integrates_with_closeout_and_replays_without_model(tmp_path):
    from blueprint_pipeline.flora_episode_interpretation import FloraEpisodeInterpreter

    data, result = _session(tmp_path, deterministic_success=False)
    score_bytes = data["score_path"].read_bytes()
    source_digest = result["result_digest"]
    profile = _real_flora_profile(tmp_path, data["contract"])
    upstream = _UpstreamFixtureRunner()
    interpreter = FloraEpisodeInterpreter.from_profile(profile, runner=upstream)
    rights = tmp_path / "rights"
    _rights_for_all(data, result, rights, interpreter)
    runner = closeout.AdapterEpisodeInterpretationRunner(interpreter, profile=profile)
    arguments = dict(run_root=tmp_path, evidence_root=data["root"], session_result=result,
                     runner=runner, rights_root=rights, environment={})
    first = closeout.materialize_policy_canary_episode_interpretations(**arguments)
    second = closeout.materialize_policy_canary_episode_interpretations(**arguments)
    assert len(upstream.calls) == 1
    assert len(upstream.calls[0]) == 20
    assert first["episode_interpretation"]["completed_count"] == 20
    assert first["episode_interpretation"]["disagreement_count"] == 20
    assert second["episode_interpretation"]["reused_receipt_count"] == 20
    assert data["score_path"].read_bytes() == score_bytes
    assert result["result_digest"] == source_digest
    for row in first["episodes"]:
        artifact = row["evidence_artifacts"]["episode_interpretation"]
        receipt = json.loads((data["root"] / artifact["relative_path"]).read_text())
        assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
        assert receipt["interpreter"] == interpreter.identity.__dict__
        assert receipt["prompt_digest"] == interpreter.prompt_identity()["prompt_digest"]
        assert receipt["prompt_contract_version"] == "flora_rewarder_prompts.v1"
        assert receipt["authoritative_deterministic_result"]["task_succeeded"] is False
        assert receipt["learned_interpretation"]["episode_outcome"] == "appears_complete"
        metadata = receipt["adapter_metadata"]
        assert metadata["profile_digest"] == profile["profile_digest"]
        assert metadata["runtime_kind"] == "hermetic_injected_runner"
        assert metadata["learned_model_executed"] is False
        assert metadata["inference_performed"] is False
        assert metadata["inference_mode_count"] == 0
        assert metadata["reference_scores_disclosed"] is False
        assert metadata["candidate_identity_disclosed"] is False
        assert metadata["state_and_contact_traces_disclosed"] is False
        assert metadata["ranking_or_promotion_effect"] == "none"
        assert set(metadata["rewards"]) == {"terminal", "dense", "rubric"}
        assert all(reward["apparent_pass"] for reward in metadata["rewards"].values())
        assert len(metadata["raw_artifacts"]) == 1
        raw = json.loads(Path(metadata["raw_artifacts"][0]["path"]).read_text())
        assert set(raw) == {"terminal", "dense", "rubric"}


@pytest.mark.parametrize("mutation", ["remove", "change"])
def test_closeout_preserves_invalidated_receipt_and_emits_new_abstention(tmp_path, mutation):
    from blueprint_pipeline.flora_episode_interpretation import FloraEpisodeInterpreter

    data, result = _session(tmp_path, deterministic_success=False)
    profile = _real_flora_profile(tmp_path, data["contract"])
    upstream = _UpstreamFixtureRunner()
    interpreter = FloraEpisodeInterpreter.from_profile(profile, runner=upstream)
    rights = tmp_path / "rights"
    _rights_for_all(data, result, rights, interpreter)
    runner = closeout.AdapterEpisodeInterpretationRunner(interpreter, profile=profile)
    arguments = dict(run_root=tmp_path, evidence_root=data["root"], session_result=result,
                     runner=runner, rights_root=rights, environment={})
    first = closeout.materialize_policy_canary_episode_interpretations(**arguments)
    original_path = data["root"] / first["episodes"][0]["evidence_artifacts"]["episode_interpretation"]["relative_path"]
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    artifact = Path(original["adapter_metadata"]["raw_artifacts"][0]["path"])
    if mutation == "remove":
        artifact.unlink()
    else:
        artifact.write_bytes(b"changed-retained-output")
    second = closeout.materialize_policy_canary_episode_interpretations(**arguments)
    assert len(upstream.calls) == 1
    assert original_path.read_bytes() == original_bytes
    assert second["episode_interpretation"]["completed_count"] == 19
    assert second["episode_interpretation"]["abstained_count"] == 1
    assert second["episode_interpretation"]["reused_receipt_count"] == 19
    derived_path = data["root"] / second["episodes"][0]["evidence_artifacts"]["episode_interpretation"]["relative_path"]
    assert derived_path != original_path
    derived = json.loads(derived_path.read_text())
    assert derived["abstention_reason"] == "retained_interpretation_evidence_invalid"
    assert derived["authoritative_deterministic_result"]["task_succeeded"] is False
    third = closeout.materialize_policy_canary_episode_interpretations(**arguments)
    assert third["episode_interpretation"]["abstained_count"] == 1
    assert original_path.read_bytes() == original_bytes
    assert len(upstream.calls) == 1


def test_closeout_replays_portable_raw_evidence_after_original_location_is_gone(tmp_path):
    from blueprint_pipeline.flora_episode_interpretation import FloraEpisodeInterpreter

    data, result = _session(tmp_path)
    profile = _real_flora_profile(tmp_path, data["contract"])
    upstream = _UpstreamFixtureRunner()
    interpreter = FloraEpisodeInterpreter.from_profile(profile, runner=upstream)
    rights = tmp_path / "rights"
    _rights_for_all(data, result, rights, interpreter)
    runner = closeout.AdapterEpisodeInterpretationRunner(interpreter, profile=profile)
    arguments = dict(run_root=tmp_path, evidence_root=data["root"], session_result=result,
                     runner=runner, rights_root=rights)
    closeout.materialize_policy_canary_episode_interpretations(**arguments, environment={})
    relocated = tmp_path / "downloaded_raw"
    shutil.copytree(tmp_path / "raw", relocated)
    shutil.rmtree(tmp_path / "raw")
    report = closeout.materialize_policy_canary_episode_interpretations(
        **arguments, environment={"BLUEPRINT_FLORA_RAW_ARTIFACT_ROOT": str(relocated)})
    assert report["episode_interpretation"]["completed_count"] == 20
    assert report["episode_interpretation"]["reused_receipt_count"] == 20
    assert len(upstream.calls) == 1
