"""Retained-run comparisons protect references, blindness and replay boundaries."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.episode_interpretation import materialize_episode_interpretation_rights
from blueprint_pipeline import flora_episode_benchmark as benchmark
from blueprint_pipeline.flora_episode_interpretation import FloraEpisodeInterpreter
from tests.test_flora_episode_closeout import _real_flora_profile, _UpstreamFixtureRunner
from tests.test_policy_canary_episode_interpretation_closeout import _session


def _prepared(tmp_path: Path, *, success=False):
    data, source = _session(tmp_path, deterministic_success=success)
    source_path = tmp_path / "source_session.json"
    source_path.write_text(json.dumps(source))
    profile = _real_flora_profile(tmp_path, data["contract"])
    root = tmp_path / "staged"
    manifest = benchmark.prepare_flora_benchmark(session_result_path=source_path,
        evidence_root=data["root"], profile=profile, output_dir=root,
        episode_ids=["episode-0", "episode-10"])
    return data, source_path, source, root, manifest, profile


def _rights(root, profile, interpreter, tmp_path):
    dataset, requests = benchmark.load_flora_benchmark(root / "manifest.json", profile)
    rights = tmp_path / "rights"
    for request in requests:
        materialize_episode_interpretation_rights(
            episode_id=request.episode_id, input_bundle_digest=request.input_receipt["input_bundle_digest"],
            identity=interpreter.identity, allowed_artifact_roles=interpreter.disclosed_artifact_roles(request),
            external_disclosure_authorized=False, accepted_by="fixture-task-owner",
            accepted_on="2026-10-20T00:00:00Z", authority_reference="hermetic-test-only",
            source_rights_admission_digest="sha256:" + "a" * 64,
            output_path=rights / (request.input_receipt["input_bundle_digest"][7:] + ".json"))
    return rights


def test_prepare_copies_verifiable_inputs_and_preserves_source(tmp_path):
    data, source_path, source, root, manifest, profile = _prepared(tmp_path)
    assert json.loads(source_path.read_text()) == source
    assert manifest["source_episode_count"] == 20
    assert manifest["selected_episode_count"] == 2
    assert {row["candidate_id"] for row in manifest["episodes"]} == {"pi05_droid", "groot_n17_droid"}
    dataset, requests = benchmark.load_flora_benchmark(root / "manifest.json", profile)
    assert len(requests) == 2
    assert all(request.deterministic_score["task_succeeded"] is False for request in requests)
    assert not (data["root"] / "_flora").exists()
    assert dataset["source_scores_modified"] is False
    for request in requests:
        assert request.evidence_root != data["root"]
        assert request.review_video_paths[0].read_bytes() == data["video_path"].read_bytes()


def test_tampered_staged_video_cannot_reach_inference(tmp_path):
    _, _, _, root, manifest, profile = _prepared(tmp_path)
    row = manifest["episodes"][0]
    video = root / row["evidence_root"] / row["sources"]["review_video_paths"][0]
    video.write_bytes(b"different-video")
    with pytest.raises(ValueError, match="staged_video_digest_mismatch"):
        benchmark.load_flora_benchmark(root / "manifest.json", profile)


@pytest.mark.parametrize("field,value", [("candidate_id", "different-policy"), ("seed", 999)])
def test_resealed_manifest_cannot_reassign_source_episode(tmp_path, field, value):
    _, _, _, root, manifest, profile = _prepared(tmp_path)
    manifest["episodes"][0][field] = value
    manifest["dataset_digest"] = canonical_digest(manifest, digest_field="dataset_digest")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="episode_source_binding_invalid"):
        benchmark.load_flora_benchmark(root / "manifest.json", profile)


def test_resealed_manifest_cannot_change_reference_success(tmp_path):
    _, _, _, root, manifest, profile = _prepared(tmp_path)
    manifest["episodes"][0]["reference"]["task_succeeded"] = True
    manifest["dataset_digest"] = canonical_digest(manifest, digest_field="dataset_digest")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reference_score_mismatch"):
        benchmark.load_flora_benchmark(root / "manifest.json", profile)


def test_missing_rights_prevents_entire_batch_and_does_not_claim_runtime_ready(tmp_path):
    _, _, _, root, _, profile = _prepared(tmp_path)
    interpreter = FloraEpisodeInterpreter.from_profile(profile, runner=_UpstreamFixtureRunner())
    result = benchmark.run_flora_benchmark(manifest_path=root / "manifest.json", profile=profile,
        rights_root=tmp_path / "missing-rights", output_dir=tmp_path / "results", execute=True,
        interpreter=interpreter)
    assert result["status"] == "blocked"
    assert result["runtime_execution_verified"] is False
    assert result["inference_performed"] is False
    assert not interpreter.runner.calls
    assert not (tmp_path / "results/execution_attempt.json").exists()


def test_comparison_preserves_false_passes_and_replays_without_inference(tmp_path):
    _, source_path, source, root, _, profile = _prepared(tmp_path)
    interpreter = FloraEpisodeInterpreter.from_profile(profile, runner=_UpstreamFixtureRunner())
    rights = _rights(root, profile, interpreter, tmp_path)
    arguments = dict(manifest_path=root / "manifest.json", profile=profile,
                     rights_root=rights, output_dir=tmp_path / "results", execute=True, interpreter=interpreter)
    report = benchmark.run_flora_benchmark(**arguments)
    replay = benchmark.run_flora_benchmark(**arguments)
    assert report == replay
    assert len(interpreter.runner.calls) == 1
    assert len(interpreter.runner.calls[0]) == 2
    assert report["metrics"]["confusion_matrix"]["false_pass"] == 2
    assert report["metrics"]["false_pass_rate_among_classified_failures"]["rate"] == 1.0
    assert report["metrics"]["false_fail_rate_among_classified_successes"]["rate"] is None
    assert report["source_scores_modified"] is False
    assert report["deployment_accuracy_qualified"] is False
    assert report["live_episode_count"] == 0
    assert report["fixture_episode_count"] == 2
    assert report["unknown_episode_count"] == 0
    assert report["inference_performed"] is False
    assert report["runtime_execution_verified"] is False
    assert report["claim_ceiling"] == "hermetic_wiring_validation_only"
    assert json.loads(source_path.read_text()) == source
    assert (tmp_path / "results/benchmark_report.md").is_file()


def test_ambiguous_attempt_is_never_automatically_reexecuted(tmp_path):
    _, _, _, root, _, profile = _prepared(tmp_path)
    interpreter = FloraEpisodeInterpreter.from_profile(profile, runner=_UpstreamFixtureRunner())
    rights = _rights(root, profile, interpreter, tmp_path)
    calls = []

    def failure(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("fixture-worker-interrupted")

    interpreter.run_batch = failure
    arguments = dict(manifest_path=root / "manifest.json", profile=profile,
                     rights_root=rights, output_dir=tmp_path / "results", execute=True, interpreter=interpreter)
    with pytest.raises(RuntimeError, match="fixture-worker-interrupted"):
        benchmark.run_flora_benchmark(**arguments)
    with pytest.raises(ValueError, match="prior_execution_ambiguous"):
        benchmark.run_flora_benchmark(**arguments)
    assert len(calls) == 1


def test_absent_or_abstained_results_never_inflate_accuracy(tmp_path):
    _, _, _, _, dataset, profile = _prepared(tmp_path)
    report = benchmark.build_flora_benchmark_report(dataset=dataset, profile=profile,
                                                   receipts=[], elapsed_seconds=0.0)
    assert report["status"] == "partial"
    assert report["metrics"]["abstained_count"] == 2
    assert report["metrics"]["coverage"]["rate"] == 0.0
    assert report["metrics"]["pass_precision"]["rate"] is None
    assert report["metrics"]["agreement_among_classified"]["rate"] is None
    assert all(row["reference_success"] is False for row in report["episodes"])
    assert report["unknown_episode_count"] == 2
    assert report["inference_performed"] is None
    assert report["reference_labels_exposed_to_model"] is None


def _completed(tmp_path):
    _, _, _, root, dataset, profile = _prepared(tmp_path)
    interpreter = FloraEpisodeInterpreter.from_profile(profile, runner=_UpstreamFixtureRunner())
    rights = _rights(root, profile, interpreter, tmp_path)
    output_dir = tmp_path / "results"
    report = benchmark.run_flora_benchmark(manifest_path=root / "manifest.json", profile=profile,
        rights_root=rights, output_dir=output_dir, execute=True, interpreter=interpreter)
    receipts = [json.loads(path.read_text()) for path in sorted((output_dir / "receipts").glob("*.json"))]
    return dataset, profile, report, receipts


@pytest.mark.parametrize("path,value,error", [
    (("interpreter", "model_version"), "other-model", "interpreter_mismatch"),
    (("prompt_digest",), "sha256:" + "a" * 64, "interpreter_mismatch"),
    (("adapter_metadata", "profile_digest"), "sha256:" + "a" * 64, "metadata_mismatch"),
    (("adapter_metadata", "source_manifest_digest"), "sha256:" + "a" * 64, "metadata_mismatch"),
    (("adapter_metadata", "compute_provider_id"), "other-provider", "metadata_mismatch"),
    (("adapter_metadata", "generation_settings", "seed"), 12345, "metadata_mismatch"),
    (("adapter_metadata", "reference_scores_disclosed"), True, "metadata_mismatch"),
    (("adapter_metadata", "candidate_identity_disclosed"), True, "metadata_mismatch"),
    (("adapter_metadata", "model_input_digest"), "sha256:" + "a" * 64, "model_input_mismatch"),
    (("adapter_metadata", "learned_model_executed"), True, "fixture_execution_claim_invalid"),
    (("learned_interpretation", "episode_outcome"), "appears_incomplete", "reward_mismatch"),
    (("proof_boundary", "ranking_or_promotion_effect"), "promote", "proof_boundary_invalid"),
])
def test_resealed_receipt_cannot_change_profile_evidence_or_claims(tmp_path, path, value, error):
    dataset, profile, _, receipts = _completed(tmp_path)
    target = receipts[0]
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    receipts[0]["receipt_digest"] = canonical_digest(receipts[0], digest_field="receipt_digest")
    with pytest.raises(ValueError, match=error):
        benchmark.build_flora_benchmark_report(dataset=dataset, profile=profile,
                                               receipts=receipts, elapsed_seconds=0.0)


def test_live_flags_without_subprocess_evidence_are_unknown(tmp_path):
    dataset, profile, _, receipts = _completed(tmp_path)
    for receipt in receipts:
        receipt["adapter_metadata"].update(runtime_kind="live_subprocess", inference_execution_kind="real_model",
                                           learned_model_executed=True, inference_performed=True, inference_mode_count=3)
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    report = benchmark.build_flora_benchmark_report(dataset=dataset, profile=profile,
                                                    receipts=receipts, elapsed_seconds=0.0)
    assert report["live_episode_count"] == 0
    assert report["fixture_episode_count"] == 0
    assert report["unknown_episode_count"] == 2
    assert report["inference_performed"] is None
    assert report["runtime_execution_verified"] is False
    assert report["claim_ceiling"] == "unverified_execution_comparison"


def test_missing_reference_and_abstention_are_counted_independently(tmp_path):
    _, _, _, _, dataset, profile = _prepared(tmp_path)
    dataset["episodes"][0].pop("reference")
    dataset["dataset_digest"] = canonical_digest(dataset, digest_field="dataset_digest")
    report = benchmark.build_flora_benchmark_report(dataset=dataset, profile=profile,
                                                   receipts=[], elapsed_seconds=0.0)
    assert report["metrics"]["missing_reference_count"] == 1
    assert report["metrics"]["abstained_count"] == 2
    assert report["metrics"]["coverage"]["rate"] == 0.0


def test_blocked_preflight_snapshot_does_not_prevent_authorized_retry(tmp_path):
    _, _, _, root, _, profile = _prepared(tmp_path)
    interpreter = FloraEpisodeInterpreter.from_profile(profile, runner=_UpstreamFixtureRunner())
    rights = tmp_path / "rights"
    output_dir = tmp_path / "results"
    arguments = dict(manifest_path=root / "manifest.json", profile=profile,
                     rights_root=rights, output_dir=output_dir, execute=True, interpreter=interpreter)
    blocked = benchmark.run_flora_benchmark(**arguments)
    assert blocked["status"] == "blocked"
    first_path = output_dir / "input_preflights" / (blocked["preflight_digest"][7:] + ".json")
    first_bytes = first_path.read_bytes()
    _rights(root, profile, interpreter, tmp_path)
    report = benchmark.run_flora_benchmark(**arguments)
    assert report["status"] == "completed"
    assert len(interpreter.runner.calls) == 1
    assert first_path.read_bytes() == first_bytes
    snapshots = list((output_dir / "input_preflights").glob("*.json"))
    assert len(snapshots) == 2
    for path in snapshots:
        value = json.loads(path.read_text())
        assert value["preflight_digest"] == canonical_digest(value, digest_field="preflight_digest")
        assert path.stem == value["preflight_digest"][7:]


def test_report_markdown_escapes_untrusted_episode_and_run_text(tmp_path):
    dataset, profile, report, _ = _completed(tmp_path)
    report["run_id"] = "<img src=x onerror=alert(1)>"
    report["episodes"][0]["episode_id"] = "[misleading](https://example.invalid) |\n# injected `code`"
    rendered = benchmark._markdown(report)
    assert "<img" not in rendered
    assert "&lt;img" in rendered
    assert "[misleading](" not in rendered
    assert "\n# injected" not in rendered
    assert "&#124;" in rendered
    assert "&#96;code&#96;" in rendered


def _corrected_source(tmp_path, *, use_ingest=False):
    """Real correction producer/validator over synthetic retained media and traces."""
    from blueprint_pipeline.task_evaluation_policy_canary_rescore import rescore_policy_canary_result
    from tests.test_episode_interpretation import _episode_root
    from tests.test_policy_canary_episode_interpretation_closeout import _record
    from tests.test_task_evaluation_policy_canary_rescore import _fixture, _scorer_identity

    source_path, evidence, source = _fixture(tmp_path)
    media_fixture = tmp_path / "media-fixture"
    media_fixture.mkdir()
    media = _episode_root(media_fixture, no_drop=False, deterministic_success=True)
    media_records = {}
    for key, role in (("contact_path", "contact_force_trace"), ("manifest_path", "frame_manifest"),
                      ("video_path", "review_video")):
        target = evidence / media[key].name
        shutil.copyfile(media[key], target)
        media_records[role] = _record(target, evidence, role)
    for frame in (media["root"] / "media").glob("*.png"):
        target = evidence / "media" / frame.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(frame, target)
        source["artifact_inventory"].append(_record(target, evidence, "lossless_frame"))
    source["artifact_inventory"].extend(media_records.values())
    for row in source["episodes"]:
        state_path = evidence / row["evidence_artifacts"]["state_trace"]["relative_path"]
        state = json.loads(state_path.read_text())
        state["trace_digest"] = canonical_digest(state, digest_field="trace_digest")
        state_path.write_text(json.dumps(state))
        row["episode"]["state_trace"] = state
        row["state_trace_digest"] = canonical_digest({"value": state})
        # Reproduce the historical absent contract provenance; only the bound
        # derived correction may provide a current, complete scoring contract.
        score_path = evidence / row["evidence_artifacts"]["score_receipt"]["relative_path"]
        score = json.loads(score_path.read_text())
        score.pop("task_success_contract")
        score.pop("task_success_contract_digest")
        score["report_digest"] = canonical_digest(score, digest_field="report_digest")
        score_path.write_text(json.dumps(score))
        row["episode"]["score"] = score
        row["deterministic_score_digest"] = canonical_digest({"value": score})
        parity = row["episode"].pop("embodiment_parity_diagnostic")
        row["episode"]["receipt_digest"] = canonical_digest(row["episode"], digest_field="receipt_digest")
        row["episode"]["embodiment_parity_diagnostic"] = parity
        for key, path in (("state_trace", state_path), ("score_receipt", score_path)):
            row["evidence_artifacts"][key].update(_record(path, evidence, key))
        row["evidence_artifacts"].update(media_records)
    source["artifact_inventory_digest"] = canonical_digest({"value": source["artifact_inventory"]})
    source["result_digest"] = canonical_digest(source, digest_field="result_digest")
    source_path.write_text(json.dumps(source))
    correction = rescore_policy_canary_result(source_result_path=source_path, evidence_root=evidence,
        output_root=tmp_path / "corrections", expected_run_id=source["run_id"], scorer_identity=_scorer_identity())
    correction_root = tmp_path / "corrections" / correction["correction_id"]
    reference_path = correction_root / "score_correction.json"
    if use_ingest:
        payload = {"schema_version": "task_evaluation_policy_canary_score_correction_ingest.v2",
                   "source_binding": {"run_id": source["run_id"]}, "correction": correction,
                   "derived_rescore_receipts": [json.loads((correction_root / update["derived_rescore_receipt"]["relative_path"]).read_text())
                                                for update in correction["score_updates"]]}
        payload["ingest_digest"] = cross_runtime_canonical_digest(payload, digest_field="ingest_digest")
        reference_path = correction_root / "website_ingest.v2.json"
        reference_path.write_text(json.dumps(payload))
    profile = _real_flora_profile(tmp_path, correction["score_updates"][0]["new_score"]["task_success_contract"])
    return source_path, evidence, source, correction, reference_path, profile


@pytest.mark.parametrize("use_ingest", [False, True])
def test_corrected_reference_preserves_original_and_binds_current_comparison(tmp_path, use_ingest):
    source_path, evidence, source, correction, reference_path, profile = _corrected_source(tmp_path, use_ingest=use_ingest)
    source_bytes = source_path.read_bytes()
    original_scores = {path: path.read_bytes() for path in evidence.glob("*.score_receipt.json")}
    root = tmp_path / "staged"
    dataset = benchmark.prepare_flora_benchmark(session_result_path=source_path, evidence_root=evidence,
        profile=profile, output_dir=root, corrected_reference_path=reference_path)
    loaded, requests = benchmark.load_flora_benchmark(root / "manifest.json", profile)
    assert loaded == dataset
    assert dataset["corrected_reference"]["correction_digest"] == correction["correction_digest"]
    assert dataset["reference_kind"] == "corrected_retained_deterministic_simulator_score"
    assert all(row["original_reference"]["task_succeeded"] is False for row in dataset["episodes"])
    assert all(request.deterministic_score["task_succeeded"] is True for request in requests)
    assert len(requests) == 20
    interpreter = FloraEpisodeInterpreter.from_profile(profile, runner=_UpstreamFixtureRunner())
    rights = _rights(root, profile, interpreter, tmp_path)
    report = benchmark.run_flora_benchmark(manifest_path=root / "manifest.json", profile=profile,
        rights_root=rights, output_dir=tmp_path / "results", execute=True, interpreter=interpreter)
    assert report["metrics"]["confusion_matrix"]["true_pass"] == 20
    assert report["metrics"]["confusion_matrix"]["false_pass"] == 0
    assert report["source_result_digest"] == source["result_digest"]
    assert report["source_scores_modified"] is False
    assert report["corrected_reference"]["correction_digest"] == correction["correction_digest"]
    assert all(row["original_reference"]["task_succeeded"] is False for row in report["episodes"])
    assert source_path.read_bytes() == source_bytes
    assert all(path.read_bytes() == data for path, data in original_scores.items())


def test_correction_for_another_source_file_is_rejected_before_preparation(tmp_path):
    source_path, evidence, _, _, reference_path, profile = _corrected_source(tmp_path)
    payload = json.loads(reference_path.read_text())
    payload["source_result_file_sha256"] = "sha256:" + "0" * 64
    payload["correction_digest"] = cross_runtime_canonical_digest(payload, digest_field="correction_digest")
    reference_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="corrected_reference_source_mismatch"):
        benchmark.prepare_flora_benchmark(session_result_path=source_path, evidence_root=evidence,
            profile=profile, output_dir=tmp_path / "staged", corrected_reference_path=reference_path)


def test_correction_receipt_tamper_cannot_be_used_as_ground_truth(tmp_path):
    source_path, evidence, _, correction, reference_path, profile = _corrected_source(tmp_path)
    receipt = reference_path.parent / correction["score_updates"][0]["derived_rescore_receipt"]["relative_path"]
    payload = json.loads(receipt.read_text())
    payload["new_score"]["task_succeeded"] = False
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="correction_receipt_digest_mismatch"):
        benchmark.prepare_flora_benchmark(session_result_path=source_path, evidence_root=evidence,
            profile=profile, output_dir=tmp_path / "staged", corrected_reference_path=reference_path)


def test_resealed_manifest_cannot_change_selected_correction_reference(tmp_path):
    source_path, evidence, _, _, reference_path, profile = _corrected_source(tmp_path)
    root = tmp_path / "staged"
    dataset = benchmark.prepare_flora_benchmark(session_result_path=source_path, evidence_root=evidence,
        profile=profile, output_dir=root, corrected_reference_path=reference_path)
    dataset["episodes"][0]["reference"]["task_succeeded"] = False
    dataset["dataset_digest"] = canonical_digest(dataset, digest_field="dataset_digest")
    (root / "manifest.json").write_text(json.dumps(dataset))
    with pytest.raises(ValueError, match="corrected_episode_binding_invalid"):
        benchmark.load_flora_benchmark(root / "manifest.json", profile)


def test_multicamera_source_receipts_stage_and_blind_all_three_views(tmp_path):
    from tests.test_policy_canary_episode_interpretation_closeout import _record

    data, source = _session(tmp_path)
    cameras = ["external", "overview", "wrist"]
    frame_digest = json.loads(data["manifest_path"].read_text())["frame_manifest_digest"]
    videos = {}
    for camera in cameras:
        path = data["video_path"] if camera == "external" else data["root"] / f"{camera}.mp4"
        if camera != "external":
            path.write_bytes(data["video_path"].read_bytes() + camera.encode())
        videos[camera] = {**_record(path, data["root"], "review_video"), "camera_id": camera,
                          "derived_from_frame_manifest_digest": frame_digest}
    for row in source["episodes"]:
        row["episode"]["visual_evidence"] = {
            "schema_version": "adp_multicamera_episode_visual_evidence.v1", "status": "complete",
            "required_camera_ids": cameras, "videos": videos, "frame_manifest_digest": frame_digest,
        }
    source["result_digest"] = canonical_digest(source, digest_field="result_digest")
    source_path = tmp_path / "source_session.json"
    source_path.write_text(json.dumps(source))
    profile = _real_flora_profile(tmp_path, data["contract"])
    profile["task"]["cameras"] = ["External view", "Overhead view", "Wrist view"]
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    root = tmp_path / "staged"
    dataset = benchmark.prepare_flora_benchmark(session_result_path=source_path, evidence_root=data["root"],
        profile=profile, output_dir=root, episode_ids=["episode-0", "episode-10"])
    _, requests = benchmark.load_flora_benchmark(root / "manifest.json", profile)
    assert all(len(request.review_video_paths) == 3 for request in requests)
    assert all(row["review_video_camera_ids"] == cameras for row in dataset["episodes"])
    assert all(row["review_video_source"] == "episode_evidence_root" for row in dataset["episodes"])
    interpreter = FloraEpisodeInterpreter.from_profile(profile, runner=_UpstreamFixtureRunner())
    rights = _rights(root, profile, interpreter, tmp_path)
    report = benchmark.run_flora_benchmark(manifest_path=root / "manifest.json", profile=profile,
        rights_root=rights, output_dir=tmp_path / "results", execute=True, interpreter=interpreter)
    assert report["fixture_episode_count"] == 2
    assert all(len(job["videos"]) == 3 for job in interpreter.runner.calls[0])
    requests[0].review_video_paths[2].write_bytes(b"changed-wrist-view")
    with pytest.raises(ValueError, match="staged_video_digest_mismatch"):
        benchmark.load_flora_benchmark(root / "manifest.json", profile)


@pytest.mark.parametrize("mutation", ["remove", "change"])
def test_replay_reopens_raw_artifacts_and_refuses_missing_or_changed_evidence(tmp_path, mutation):
    _, profile, _, receipts = _completed(tmp_path)
    artifact = Path(receipts[0]["adapter_metadata"]["raw_artifacts"][0]["path"])
    if mutation == "remove":
        artifact.unlink()
    else:
        artifact.write_bytes(b"changed-native-evidence")
    interpreter = FloraEpisodeInterpreter.from_profile(profile, runner=_UpstreamFixtureRunner())
    with pytest.raises(ValueError, match="raw_artifact_missing_or_changed"):
        benchmark.run_flora_benchmark(manifest_path=tmp_path / "staged/manifest.json", profile=profile,
            rights_root=tmp_path / "rights", output_dir=tmp_path / "results", execute=True, interpreter=interpreter)
    assert interpreter.runner.calls == []


def test_portable_replay_rebases_raw_evidence_without_changing_sealed_report(tmp_path):
    _, profile, original_report, _ = _completed(tmp_path)
    portable = tmp_path / "downloaded_results"
    shutil.copytree(tmp_path / "results", portable)
    shutil.copytree(tmp_path / "raw", portable / "raw")
    shutil.rmtree(tmp_path / "raw")
    interpreter = FloraEpisodeInterpreter.from_profile(profile, runner=_UpstreamFixtureRunner())
    report = benchmark.run_flora_benchmark(manifest_path=tmp_path / "staged/manifest.json", profile=profile,
        rights_root=tmp_path / "rights", output_dir=portable, execute=True, interpreter=interpreter,
        raw_artifact_root=portable / "raw")
    assert report == original_report
    assert report["raw_artifacts_verified"] is True
    assert interpreter.runner.calls == []
