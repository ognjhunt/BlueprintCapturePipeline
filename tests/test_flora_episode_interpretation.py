"""ADP-009D blinded visual grading never replaces simulator reference truth."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.episode_interpretation import (
    EpisodeInterpretationError,
    interpret_episode,
    materialize_episode_interpretation_rights,
)
from blueprint_pipeline.flora_episode_interpretation import (
    FLORA_MODEL,
    FLORA_REVISION,
    INTERPRETER_ID,
    PROFILE_SCHEMA,
    RUNTIME,
    FloraEpisodeInterpreter,
    normalize_flora_rewards,
    validate_flora_interpreter_profile,
    verify_flora_raw_artifacts,
)
from blueprint_pipeline.flora_rewarder_runtime import file_digest
from tests.test_episode_interpretation import _episode_root, _request


FIXTURE = Path(__file__).parent / "fixtures" / "flora_rewarder_output.v1.json"


def _golden():
    # Dense JSON was generated with the pinned upstream summarize/parse_dense;
    # terminal and rubric raw answers use the same released parsers. No inference.
    return json.loads(FIXTURE.read_text())


def _profile(request, tmp_path):
    value = {
        "schema_version": PROFILE_SCHEMA, "status": "configured",
        "interpreter_id": INTERPRETER_ID, "provider_id": "nvidia", "runtime": RUNTIME,
        "model": FLORA_MODEL, "model_version": FLORA_REVISION, "source_revision": FLORA_REVISION,
        "research_only": True, "modes": ["terminal", "dense", "rubric"], "execution_site": "local",
        "runtime_config": {"python_executable": "/usr/local/bin/python", "source_root": "/opt/blueprint/flora/source",
            "model_path": "/opt/blueprint/flora/model", "output_root": str(tmp_path / "outputs"),
            "timeout_seconds": 600, "tensor_parallel_size": 2, "seed": 0, "rubric_num_samples": 9},
        "task": {"instruction": "Move the cup to the green target.", "robot": "fixed robot arm",
            "cameras": ["scene camera"], "rubric": _golden()["rubric"],
            "task_success_contract_digest": request.task_success_contract["contract_digest"]},
    }
    value["profile_digest"] = canonical_digest(value, digest_field="profile_digest")
    return value


class _Runner:
    def __init__(self, mutate=None):
        self.jobs = []
        self.calls = 0
        self.mutate = mutate

    def run_batch(self, *, profile, jobs, output_root):
        self.calls += 1
        self.jobs.extend(copy.deepcopy(jobs))
        records = {mode: [] for mode in ("terminal", "dense", "rubric")}
        for job in jobs:
            rows = _golden()["records"]
            for mode, row in rows.items():
                row.update(id=job["id"], task=job["task"])
                records[mode].append(row)
        result = {"records": records, "executions": [{"mode": mode} for mode in records],
                  "wall_time_seconds": 0.01, "learned_model_executed": True}
        if self.mutate:
            self.mutate(result)
        return result


class _NativeArtifactRunner(_Runner):
    """Write production-shaped native files while remaining explicitly fake."""

    def run_batch(self, *, profile, jobs, output_root):
        result = super().run_batch(profile=profile, jobs=jobs, output_root=output_root)
        result["executions"] = []
        for mode, records in result["records"].items():
            output = output_root / (mode + ".jsonl")
            output.write_text("".join(json.dumps(row) + "\n" for row in records))
            log = output_root / (mode + ".log")
            log.write_text("Hermetic fake: no model was loaded.\n")
            result["executions"].append({"mode": mode, "returncode": 0, "timed_out": False,
                "output_path": str(output), "output_sha256": file_digest(output),
                "log_path": str(log), "log_sha256": file_digest(log)})
        return result


def _rights(request, interpreter, root):
    path = root / (request.episode_id + ".json")
    materialize_episode_interpretation_rights(
        episode_id=request.episode_id, input_bundle_digest=request.input_receipt["input_bundle_digest"],
        identity=interpreter.identity, allowed_artifact_roles=interpreter.disclosed_artifact_roles(request),
        external_disclosure_authorized=interpreter.identity.execution_site == "external_provider",
        accepted_by="fixture-owner", accepted_on="2026-10-20", authority_reference="hermetic-only",
        source_rights_admission_digest="sha256:" + "a" * 64, output_path=path)
    return path


def test_pinned_upstream_schema_preserves_three_distinct_reward_meanings():
    fixture = _golden()
    job = {"id": "fixture-episode", "task": "Move the cup to the green target.", "rubric": fixture["rubric"]}
    result = normalize_flora_rewards(fixture["records"], job=job, sample_count=9)
    assert result["terminal"]["score"] == 5
    assert result["dense"]["worst_score"] == 3
    assert result["rubric"]["total"] == 1.0
    assert "success_probability" not in json.dumps(result)


def test_batch_blinds_reference_identity_and_retains_non_authoritative_receipt(tmp_path):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    original_score = data["score_path"].read_bytes()
    request = _request(data)
    runner = _Runner()
    interpreter = FloraEpisodeInterpreter.from_profile(_profile(request, tmp_path), runner=runner)
    second_receipt = copy.deepcopy(dict(request.input_receipt))
    second_receipt.update(episode_id="episode-2", candidate_policy_id="pi05_droid")
    second_receipt["input_bundle_digest"] = canonical_digest(second_receipt, digest_field="input_bundle_digest")
    second = replace(request, episode_id="episode-2", candidate_policy_id="pi05_droid", input_receipt=second_receipt)
    requests = [request, second]
    rights = {r.episode_id: _rights(r, interpreter, tmp_path / "rights") for r in requests}
    interpreter.run_batch(requests, rights_attestation_paths=rights)
    assert runner.calls == 1
    serialized = json.dumps(runner.jobs)
    for forbidden in ("task_succeeded", "deterministic_score", "candidate_policy_id", "groot_n17_droid",
                      "pi05_droid", "contact_force_trace", "state_trace", "no_drop"):
        assert forbidden not in serialized
    assert all(Path(job["videos"][0]).name == "camera-0.mp4" for job in runner.jobs)
    assert all(Path(job["videos"][0]).read_bytes() == data["video_path"].read_bytes() for job in runner.jobs)
    digests = []
    for r in requests:
        receipt = interpret_episode(request=r, interpreter=interpreter, rights_attestation_path=rights[r.episode_id],
                                    output_path=tmp_path / (r.episode_id + ".receipt.json"))
        assert receipt["status"] == "completed"
        assert receipt["deterministic_agreement"] == "disagrees"
        assert receipt["authoritative_deterministic_result"]["task_succeeded"] is False
        metadata = interpreter.adapter_metadata()
        digests.append(metadata["model_input_digest"])
        assert metadata["inference_execution_kind"] == "hermetic_test"
        assert metadata["inference_performed"] is False
        assert metadata["learned_model_executed"] is False
        assert metadata["rewards"]["rubric"]["sample_count"] == 9
        assert metadata["confidence_calibrated"] is False
        assert "groot_n17_droid" not in json.dumps(metadata)
        assert "pi05_droid" not in json.dumps(metadata)
    assert digests[0] == digests[1]  # Reference identity does not contaminate the model input.
    assert runner.calls == 1  # Cached receipt materialization does not reload any model.
    assert data["score_path"].read_bytes() == original_score


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r["records"].pop("rubric"), "required_mode_missing"),
    (lambda r: r["records"]["terminal"][0].update(id="wrong-episode"), "batch_result_identity_invalid"),
    (lambda r: r["records"]["terminal"][0].update(task="different task"), "result_identity_mismatch"),
    (lambda r: r["records"]["terminal"][0].update(expected_score=float("nan")), "nonfinite_output"),
    (lambda r: r["records"]["dense"][0].update(invalid_clips=1), "dense_invalid_clips"),
    (lambda r: r["records"]["dense"][0]["clips"][0].update(valid=False), "normal_fallback_forbidden"),
    (lambda r: r["records"]["dense"][0]["clips"][0].update(output="unparseable"), "dense_raw_answer_invalid"),
    (lambda r: r["records"]["dense"][0].update(segments=[]), "segment_coverage_invalid"),
    (lambda r: r["records"]["rubric"][0]["samples"][0].update(answer="ANSWER: [2, 0.75]"), "numeric_value_invalid"),
    (lambda r: r["records"]["rubric"][0]["samples"][0].update(answer="ANSWER: [0.20, 0.75]"), "score_not_allowed"),
    (lambda r: r["records"]["rubric"][0].update(parsed_samples=8), "samples_missing_or_invalid"),
    (lambda r: r["records"]["rubric"][0].update(medoid_sample=2), "rubric_summary_invalid"),
])
def test_invalid_upstream_outputs_abstain_and_keep_raw_evidence(tmp_path, mutation, reason):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    request = _request(data)
    runner = _Runner(mutation)
    interpreter = FloraEpisodeInterpreter.from_profile(_profile(request, tmp_path), runner=runner)
    output = interpreter.interpret(request)
    assert output.episode_outcome == "unclear"
    metadata = interpreter.adapter_metadata()
    assert metadata["status"] == "abstained"
    assert reason in metadata["abstention_reason"]
    assert metadata["apparent_pass"] is None
    assert metadata["rewards"] == {}
    assert Path(metadata["raw_artifacts"][0]["path"]).is_file()
    assert interpreter.interpret(request).episode_outcome == "unclear"
    assert runner.calls == 1


def test_batch_requires_every_rights_attestation_before_runner(tmp_path):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    request = _request(data)
    runner = _Runner()
    interpreter = FloraEpisodeInterpreter.from_profile(_profile(request, tmp_path), runner=runner)
    with pytest.raises(EpisodeInterpretationError, match="rights_missing"):
        interpreter.run_batch([request], rights_attestation_paths={})
    assert runner.calls == 0


def test_cache_rechecks_video_and_retained_raw_output(tmp_path):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    request = _request(data)
    interpreter = FloraEpisodeInterpreter.from_profile(_profile(request, tmp_path), runner=_Runner())
    interpreter.interpret(request)
    raw_path = Path(interpreter.adapter_metadata()["raw_artifacts"][0]["path"])
    raw_path.write_text("tampered")
    with pytest.raises(EpisodeInterpretationError, match="raw_artifact_missing_or_changed"):
        interpreter.interpret(request)


@pytest.mark.parametrize("kind", ["episode_raw", "native_output", "runtime_log"])
@pytest.mark.parametrize("mutation", ["delete", "change"])
def test_raw_verifier_rejects_missing_or_changed_retained_files(tmp_path, kind, mutation):
    request = _request(_episode_root(tmp_path, no_drop=True, deterministic_success=False))
    interpreter = FloraEpisodeInterpreter.from_profile(_profile(request, tmp_path), runner=_NativeArtifactRunner())
    interpreter.interpret(request)
    metadata = interpreter.adapter_metadata()
    artifact = next(row for row in metadata["raw_artifacts"] if row["kind"] == kind)
    path = Path(artifact["path"])
    if mutation == "delete":
        path.unlink()
    else:
        path.write_text("changed retained evidence")
    with pytest.raises(EpisodeInterpretationError, match="raw_artifact_missing_or_changed"):
        verify_flora_raw_artifacts(metadata)


def test_raw_verifier_rechecks_native_record_even_if_file_digest_was_resealed(tmp_path):
    request = _request(_episode_root(tmp_path, no_drop=True, deterministic_success=False))
    interpreter = FloraEpisodeInterpreter.from_profile(_profile(request, tmp_path), runner=_NativeArtifactRunner())
    interpreter.interpret(request)
    metadata = interpreter.adapter_metadata()
    artifact = next(row for row in metadata["raw_artifacts"]
                    if row["kind"] == "native_output" and row["mode"] == "terminal")
    path = Path(artifact["path"])
    row = json.loads(path.read_text())
    row["score"] = 1
    path.write_text(json.dumps(row) + "\n")
    artifact["sha256"] = file_digest(path)
    metadata["runtime_execution"][0]["output_sha256"] = artifact["sha256"]
    with pytest.raises(EpisodeInterpretationError, match="native_raw_output_mismatch"):
        verify_flora_raw_artifacts(metadata)


def test_raw_verifier_rebases_portable_files_without_rewriting_sealed_metadata(tmp_path):
    request = _request(_episode_root(tmp_path, no_drop=True, deterministic_success=False))
    interpreter = FloraEpisodeInterpreter.from_profile(_profile(request, tmp_path), runner=_NativeArtifactRunner())
    interpreter.interpret(request)
    metadata = interpreter.adapter_metadata()
    sealed = copy.deepcopy(metadata)
    portable = tmp_path / "downloaded" / "raw"
    shutil.copytree(interpreter.output_root, portable)
    interpreter.output_root.rename(tmp_path / "original-output-moved")
    with pytest.raises(EpisodeInterpretationError, match="raw_artifact_missing_or_changed"):
        verify_flora_raw_artifacts(metadata)
    verify_flora_raw_artifacts(metadata, artifact_root=portable)
    assert metadata == sealed
    metadata["raw_artifacts"][0]["relative_path"] = "../escape.json"
    with pytest.raises(EpisodeInterpretationError, match="raw_artifact_binding_invalid"):
        verify_flora_raw_artifacts(metadata, artifact_root=portable)


def test_mutating_staged_input_cannot_mutate_retained_source_and_abstains(tmp_path):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    request = _request(data)
    original = data["video_path"].read_bytes()
    runner = _Runner()
    def mutate(_):
        staged = Path(runner.jobs[0]["videos"][0])
        staged.chmod(0o644)
        staged.write_bytes(b"changed only the staging copy")
    runner.mutate = mutate
    interpreter = FloraEpisodeInterpreter.from_profile(_profile(request, tmp_path), runner=runner)
    assert interpreter.interpret(request).episode_outcome == "unclear"
    assert data["video_path"].read_bytes() == original
    assert interpreter.adapter_metadata()["abstention_reason"] == "flora_staged_video_changed_during_inference"


def test_cloud_rights_identity_names_compute_recipient_separately_from_model_vendor(tmp_path):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    request = _request(data)
    profile = _profile(request, tmp_path)
    profile["execution_site"] = "external_provider"
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    with pytest.raises(EpisodeInterpretationError, match="compute_provider_missing"):
        validate_flora_interpreter_profile(profile)
    profile["compute_provider_id"] = "vast"
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    interpreter = FloraEpisodeInterpreter.from_profile(profile, runner=_Runner())
    assert interpreter.identity.provider_id == "vast"
    assert interpreter.identity.model == "nvidia/Flora-Rewarder"


@pytest.mark.parametrize("mutate,reason", [
    (lambda p: p.update(research_only=False), "profile_identity_invalid"),
    (lambda p: p.update(model_version="main"), "profile_identity_invalid"),
    (lambda p: p["runtime_config"].update(model_path="nvidia/Flora-Rewarder"), "path_must_be_absolute"),
    (lambda p: p["task"]["rubric"][0]["grading"][1].update(points=0.125), "not_exactly_renderable"),
])
def test_profile_fails_closed_on_mutable_or_nonresearch_configuration(tmp_path, mutate, reason):
    data = _episode_root(tmp_path, no_drop=True, deterministic_success=False)
    profile = _profile(_request(data), tmp_path)
    mutate(profile)
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    with pytest.raises(EpisodeInterpretationError, match=reason):
        validate_flora_interpreter_profile(profile)
