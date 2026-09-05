from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline import task_evaluation_sam31_preparation_paid_stages as paid
from blueprint_pipeline import task_evaluation_sam31_preparation_stages as stages


COMMIT = "a" * 40


def _write(path: Path, value: dict, *, digest_field: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if digest_field:
        value[digest_field] = canonical_digest(value, digest_field=digest_field)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return path


def _record(path: Path) -> dict:
    return {"path": str(path), "sha256": paid.sha(path), "size_bytes": path.stat().st_size}


def _job(tmp_path: Path, stage: str) -> tuple[dict, Path]:
    data = tmp_path / "data"
    repo = tmp_path / "repo"
    output = data / "execution" / stage
    for path in (data, repo, output):
        path.mkdir(parents=True, exist_ok=True)
    task = _write(data / "task.json", {
        "human_authority": {
            "accepted_by": "Nijel Hunt",
            "accepted_on": "2026-09-04",
            "authority_reference": "Scene 841757 production Task Evaluation goal",
            "private_derived_frame_disclosure_authorized": True,
            "provider_retention_terms_accepted": True,
            "provider_training_terms_accepted": True,
            "provider_training_authorized": False,
        }
    })
    secret = tmp_path / "hf-token"
    secret.write_text("fixture-secret-not-read-by-test")
    os.chmod(secret, 0o600)
    profile = {
        "source_commit": COMMIT,
        "approved_paid_input_roots": [str(tmp_path)],
        "paid_stages": {
            "sam31_tracking": {
                "source_profile": paid.SAM31_SOURCE_PROFILE,
                "hf_token_file": str(secret),
                "max_spend_usd": 1.0,
                "max_hourly_rate_usd": 0.5,
                "hard_ttl_seconds": 1800,
                "retry_cap": 0,
                "allowed_active_instance_ids": [],
                "aggregate_goal_spend_before_attempt_usd": 0.0,
                "aggregate_goal_spend_cap_usd": 4.0,
            },
            "contribution_sweep": {
                "max_spend_usd": 1.5,
                "max_hourly_rate_usd": 0.5,
                "hard_ttl_seconds": 3600,
                "retry_cap": 0,
                "allowed_active_instance_ids": [],
                "flashsplat_root": str(tmp_path / "flashsplat"),
                "dependency_wheelhouse_path": str(tmp_path / "wheelhouse"),
                "dependency_manifest_path": str(tmp_path / "wheelhouse.json"),
            },
        },
    }
    job = {
        "stage_id": stage,
        "child_id": "sam31-" + "b" * 64,
        "expected_source_commit": COMMIT,
        "server_profile": profile,
        "server_data_root": str(data),
        "repo_root": str(repo),
        "output_root": str(output),
        "plan": {"host_inputs": {"task_request": _record(task)}},
        "inputs": {},
        "resume_only": False,
    }
    return job, output


@pytest.mark.parametrize("fault", [None, "zero_missing", "trace_corrupt", "tracks_foreign", "commit_foreign", "bundle_foreign", "teardown_missing"])
def test_sam_tracking_builds_exact_derived_bundle_and_uses_canonical_allocator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault,
) -> None:
    job, output = _job(tmp_path, "sam31_tracking")
    run_request = _write(tmp_path / "data" / "run-request.json", {
        "frame_registry": [{"source_frame_id": "one"}, {"source_frame_id": "two"}],
    })
    provider_profile = _write(tmp_path / "data" / "provider-profile.json", {})
    job["inputs"].update(
        sam31_run_request=_record(run_request),
        sam31_provider_profile=_record(provider_profile),
    )

    def build_bundle(*, bundle_path, receipt_path, **_):
        bundle = Path(bundle_path)
        bundle.write_bytes(b"derived-sam-bundle")
        _write(Path(receipt_path), {
            "schema_version": "semantic_sam31_source_track_input_bundle_receipt.v1",
            "status": "completed",
            "source_track_run_request_digest": "sha256:" + "d" * 64,
            "bundle": _record(bundle),
        }, digest_field="receipt_digest")

    def build_request(*, output_path, source_profile, retry_cap, **_):
        assert source_profile == paid.SAM31_SOURCE_PROFILE
        assert retry_cap == 0
        _write(Path(output_path), {"schema_version": "semantic_sam31_gpu_canary_request.v1"}, digest_field="request_digest")

    def build_authority(*, output_path, allowed_active_instance_ids, **_):
        assert allowed_active_instance_ids == ()
        _write(Path(output_path), {"schema_version": "sam31_paid_attempt_authority.v1"})

    monkeypatch.setattr(
        "blueprint_pipeline.sam31_source_track_canary_worker.build_sam31_source_track_input_bundle",
        build_bundle,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_provider_launch_packet.materialize_sam31_gpu_canary_request",
        build_request,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_paid_attempt_authority.materialize_sam31_paid_attempt_authority",
        build_authority,
    )
    seen = []
    produced = {}

    def runner(argv, *, cwd, timeout):
        seen.append((argv, cwd, timeout))
        result_path = Path(argv[argv.index("--adapter-output") + 1])
        from blueprint_pipeline.sam31_paid_resource_allocator_lane import _write_terminal_result
        request_digest = json.loads(Path(argv[argv.index("--provider-launch-request") + 1]).read_text())["request_digest"]
        bound = _write(Path(argv[argv.index("--bound-request-out") + 1]), {
            "request_digest": request_digest,
        }, digest_field="bound_request_digest")
        bound_digest = json.loads(bound.read_text())["bound_request_digest"]
        canary = result_path.parent / "sam31_vast_source_track_canary"
        tracks = canary / "semantic_source_track_import_result.v1.json"
        produced["tracks"] = tracks
        _write(tracks, {"schema_version": "semantic_source_track_import_result.v1",
                        "status": "completed"}, digest_field="result_digest")
        zero = _write(canary / "provider_zero_verification.json", {
            "schema_version": "semantic_sam31_vast_provider_zero.v1",
            "status": "PASS", "provider": "vast", "api_confirmed": True,
            "scoped_live_resource_count": 0, "global_live_resource_count": 0,
            "request_digest": request_digest, "bound_request_digest": bound_digest,
        }, digest_field="provider_zero_digest")
        teardown = _write(canary / "teardown_receipt.json", {
            "status": "PASS", "provider_zero_verified": True, "instance_id": 123,
        }, digest_field="teardown_receipt_digest")
        trace = canary / "runtime.log"
        produced["trace"] = trace
        trace.write_text("completed native tracking")
        bundle = Path(argv[argv.index("--sam31-input-bundle") + 1])
        result = _write_terminal_result(result_path, {
            "schema_version": "semantic_sam31_vast_source_track_execution.v1",
            "status": "completed", "instance_id": 123,
            "provider_zero_verified": True, "retry_cap": 0, "blockers": [],
            "provider_mutations_performed": 2,
            "source_track_import_result_path": str(tracks),
            "source_track_import_result_digest": json.loads(tracks.read_text())["result_digest"],
            "source_commit_sha": COMMIT,
            "input_bundle_digest": paid.sha(bundle),
            "source_track_run_request_digest": "sha256:" + "d" * 64,
            "source_teardown_receipt_path": str(teardown),
            "provider_zero_digest": json.loads(zero.read_text())["provider_zero_digest"],
            "request_digest": request_digest, "bound_request_digest": bound_digest,
            "continuing_spend_from_this_run": False, "all_staged_objects_absent": True,
            "independent_watchdog": {"status": "provider_terminal"},
        }, extra_artifact_roots={"sam31_runtime": canary})
        if fault == "zero_missing":
            zero.unlink()
        elif fault == "trace_corrupt":
            trace.write_text("corrupted")
        elif fault == "tracks_foreign":
            _write(tracks, {"schema_version": "semantic_source_track_import_result.v1",
                           "status": "completed", "foreign": True}, digest_field="result_digest")
        elif fault == "teardown_missing":
            Path(result["teardown_manifest_path"]).unlink()
        elif fault in {"commit_foreign", "bundle_foreign"}:
            result["source_commit_sha" if fault == "commit_foreign" else "input_bundle_digest"] = "f" * 40
            _write(result_path, result, digest_field="execution_result_digest")
        return 0

    # Exercise the actual phase dispatcher with an absent artifact child directory.
    profile = dict(job["server_profile"], schema_version="task_evaluation_sam31_preparation_profile.v1",
                   repo_root=job["repo_root"], server_data_root=job["server_data_root"])
    profile_path = _write(tmp_path / "profile.json", profile, digest_field="profile_digest")
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SAM31_PREPARATION_PROFILE_FILE", str(profile_path))
    original = paid.execute_paid_stage
    monkeypatch.setattr(paid, "execute_paid_stage", lambda context: original(context, allocator_runner=runner))
    phase_job = {**job, "phase": "sam31_tracking", "job_digest": "sha256:" + "f" * 64,
                 "plan": {**job["plan"], "source_commit": COMMIT, "server_profile_sha256": paid.sha(profile_path)}}
    assert not (output / "artifacts").exists()
    result = stages.execute_stage(phase_job)
    assert result["status"] == ("failed" if fault else "completed")
    assert (output / "phase_execution_receipt.v1.json").is_file()
    if fault:
        return
    assert result["raw_source_uploaded"] is False
    assert result["candidate_policy_queried"] is False
    assert Path(result["artifacts"]["sam31_source_tracks"]["path"]) == produced["tracks"]
    argv, cwd, timeout = seen[0]
    assert argv[:4] == [paid.sys.executable, *paid.ALLOCATOR_PREFIX]
    assert argv[-1] == "--execute"
    assert argv[argv.index("--sam31-retry-cap") + 1] == "0"
    assert cwd == Path(job["repo_root"])
    assert timeout == 2700
    assert not any("original" in value or "3dgs_compressed" in value for value in argv)
    assert (output / "artifacts" / "prepared" / "sam31-input-bundle.zip").is_file()
    receipt_path = output / "phase_execution_receipt.v1.json"
    receipt_bytes = receipt_path.read_bytes()
    assert stages.execute_stage({**phase_job, "resume_only": True}) == result
    produced["trace"].write_text("corrupted after phase completion")
    with pytest.raises(paid.Sam31PreparationPaidStageError, match="replay_terminal_artifacts_changed"):
        stages.execute_stage({**phase_job, "resume_only": True})
    assert len(seen) == 1
    assert receipt_path.read_bytes() == receipt_bytes


def test_resumed_paid_stage_without_terminal_result_never_allocates_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, _ = _job(tmp_path, "sam31_tracking")
    run_request = _write(tmp_path / "data" / "run-request.json", {"frame_registry": [{}]})
    provider_profile = _write(tmp_path / "data" / "provider-profile.json", {})
    job["inputs"].update(
        sam31_run_request=_record(run_request),
        sam31_provider_profile=_record(provider_profile),
    )
    job["resume_only"] = True

    def build_bundle(*, bundle_path, receipt_path, **_):
        bundle = Path(bundle_path)
        bundle.write_bytes(b"derived")
        _write(Path(receipt_path), {"bundle": _record(bundle)}, digest_field="receipt_digest")

    monkeypatch.setattr(
        "blueprint_pipeline.sam31_source_track_canary_worker.build_sam31_source_track_input_bundle",
        build_bundle,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_provider_launch_packet.materialize_sam31_gpu_canary_request",
        lambda **kwargs: _write(Path(kwargs["output_path"]), {}),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_paid_attempt_authority.materialize_sam31_paid_attempt_authority",
        lambda **kwargs: _write(Path(kwargs["output_path"]), {}),
    )
    result = paid.execute_paid_stage(
        job,
        allocator_runner=lambda *_args, **_kwargs: pytest.fail("must not reallocate"),
    )
    assert result["status"] == "failed"
    assert result["blockers"] == ["sam31_paid_stage_started_without_terminal_reconciliation"]


@pytest.mark.parametrize("fault", [None, "teardown_missing", "array_corrupt", "bundle_foreign", "freeze_foreign"])
def test_contribution_sweep_requires_explicit_full_source_authority_and_returns_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault,
) -> None:
    job, output = _job(tmp_path, "contribution_sweep")
    frozen_avoidlist = _write(tmp_path / "data" / "frozen-avoidlist.json", {
        "schema_version": "vast_machine_avoidlist.v1", "machine_ids": [20166, 144209], "entries": [],
    })
    frozen_bytes = frozen_avoidlist.read_bytes()
    frozen_avoidlist.chmod(0o440)
    job["server_profile"]["paid_stages"]["contribution_sweep"].update(
        machine_avoidlist_path=str(frozen_avoidlist), machine_avoidlist=_record(frozen_avoidlist),
    )
    freeze = _write(tmp_path / "data" / "freeze.json", {
        "scene": {"publisher_scene_id": "841757", "target_instance_id": "115"},
        "segment_contribution_sweep": {
            "kind": "repair_supported_full_view_segment_contribution_sweep.v1"
        },
    }, digest_field="freeze_digest")
    source = tmp_path / "data" / "source-standard.ply"
    source.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n" + b"\0" * 4)
    from tests.test_sam31_contribution_disclosure import authorize_full_source
    terms = source.parent / "publisher-terms.txt"
    terms.write_text("Hermetic test-only full source processing permission.")
    job["inputs"]["interiorgs_terms"] = _record(terms)
    conversion = _write(source.parent / "conversion.json", {
        "schema_version": "standard_splat_conversion_receipt.v1",
        "status": "standard_splat_conversion_materialized", "repository": {"commit": COMMIT},
        "claim_ceiling": "local_format_conversion_only",
        "source": {**_record(source), "source_bytes_unchanged": True, "source_gaussian_count": 1,
                   "dataset": "hermetic", "revision": COMMIT},
        "output": {**_record(source), "gaussian_count": 1, "gaussian_count_preserved": True,
                   "standard_3dgs_schema_validated": True},
        "rights": {"conversion_execution_location": "local_only", "raw_private_upload_authorized": False,
                   "training_authorized": False, "terms_digest": paid.sha(terms)},
    }, digest_field="receipt_digest")
    authorize_full_source(job, source=source, original=source, receipt=conversion)
    frozen = json.loads(freeze.read_text())
    frozen["source_standard_splat"] = _record(source)
    _write(freeze, frozen, digest_field="freeze_digest")
    cameras = _write(tmp_path / "data" / "cameras.json", [])
    job["inputs"].update(
        segment_sweep_freeze=_record(freeze),
        standard_splat=_record(source),
        camera_contract=_record(cameras),
    )
    for name in ("flashsplat", "wheelhouse"):
        (tmp_path / name).mkdir()
    (tmp_path / "wheelhouse.json").write_text("{}")

    def build_bundle(*, job_dir, **_):
        root = Path(job_dir)
        root.mkdir()
        bundle = root / "adp_gaussian_excision_provider_runtime_bundle.zip"
        bundle.write_bytes(b"derived-gaussian-bundle")
        _write(root / "adp_gaussian_excision_bundle_receipt.json", {
            "schema_version": "adp009b_gaussian_excision_vast_bundle.v1",
            "status": "ready",
            "execution_authority_digest": json.loads((root.parent / "gaussian-execution-authority.json").read_text())["authorization_digest"],
            "standard_splat_sha256": paid.sha(source),
            "freeze_digest": json.loads(freeze.read_text())["freeze_digest"],
            "bundle_sha256": paid.sha(bundle),
            "bundle_path": str(bundle), "bundle_size_bytes": bundle.stat().st_size,
            "blueprint_commit": COMMIT,
            "hard_cap_usd": 1.5,
            "hard_ttl_seconds": 3600,
            "execution_purpose": "released_code_segment_contribution_sweep",
        })

    monkeypatch.setattr(
        "blueprint_pipeline.adp_gaussian_excision_vast.build_gaussian_excision_vast_bundle",
        build_bundle,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_gaussian_excision_vast.validate_gaussian_excision_paid_attempt_authority",
        lambda *args, **kwargs: None,
    )
    seen = []
    produced = {}

    def runner(argv, **_):
        seen.append(argv)
        mutable = Path(argv[argv.index("--adp-machine-avoidlist") + 1])
        assert mutable == output / "artifacts" / "provider_machine_avoidlist.json"
        assert mutable.read_bytes() == frozen_bytes
        assert mutable.stat().st_ino != frozen_avoidlist.stat().st_ino
        assert mutable.stat().st_mode & 0o777 == 0o600
        _write(mutable, {"schema_version": "vast_machine_avoidlist.v1",
                         "machine_ids": [20166, 144209, 999999], "entries": []})
        assert frozen_avoidlist.read_bytes() == frozen_bytes
        assert frozen_avoidlist.stat().st_mode & 0o777 == 0o440
        result_path = Path(argv[argv.index("--adapter-output") + 1])
        from blueprint_pipeline.task_evaluation_artifact_manifest import seal_lane_terminal_artifacts
        job_root = Path(argv[argv.index("--adp-job-dir") + 1])
        execution_root = job_root / "immutable_execution"
        execution_root.mkdir(parents=True)
        array = execution_root / "contribution_repetition_0.npz"
        array.write_bytes(b"retained-contribution-array")
        produced["array"] = array
        render = execution_root / "calibration_one.png"
        render.write_bytes(b"retained-calibration-render")
        def relative_record(path):
            return {"relative_path": path.name, "sha256": paid.sha(path), "size_bytes": path.stat().st_size}
        frozen_digest = json.loads(freeze.read_text())["freeze_digest"]
        manifest = _write(execution_root / "contributions.json", {
            "schema_version": "adp009b_gaussian_excision_contribution_evidence.v1",
            "freeze_digest": frozen_digest, "heldout_cameras_accessed_for_classification": False,
            "repetitions": [relative_record(array)], "calibration_renders": [relative_record(render)],
        }, digest_field="manifest_digest")
        produced["manifest"] = manifest
        execution = _write(execution_root / "result.json", {
            "schema_version": "adp009b_gaussian_excision_result.v1", "status": "completed",
            "freeze_digest": frozen_digest, "contribution_manifest": relative_record(manifest),
            "contribution_manifest_digest": json.loads(manifest.read_text())["manifest_digest"],
            "released_code_executed": True, "heldout_cameras_accessed_for_classification": False,
            "provider_zero_required_after_return": True, "depth_anything_3_used": False,
            "retry_cap": 0, "blockers": [],
        }, digest_field="result_digest")
        teardown = _write(job_root / "vast_provider_run" / "vast_teardown_manifest.json", {
            "schema_version": "vast_teardown_manifest.v1", "status": "completed",
            "continuing_spend_from_this_run": False,
        })
        _write(job_root / "vast_provider_run" / "vast_provider_adapter_result.json", {"status": "completed"})
        bundle_receipt = json.loads(Path(argv[argv.index("--adp-gaussian-excision-bundle-receipt") + 1]).read_text())
        terminal = seal_lane_terminal_artifacts({
            "schema_version": "adp009b_gaussian_excision_vast_run.v1", "status": "completed",
            "bundle_sha256": bundle_receipt["bundle_sha256"],
            "continuing_spend_from_this_run": False, "retry_cap": 0, "blockers": [],
            "provider_mutations_performed": 2, "all_staged_objects_absent": True,
            "independent_watchdog": {"status": "provider_terminal"},
            "execution_result_path": str(execution), "teardown_manifest_path": str(teardown),
        }, attempt_root=job_root, lane="adp_gaussian_excision",
            binding={"bundle_sha256": bundle_receipt["bundle_sha256"], "provider": "vast"})
        if fault == "teardown_missing":
            teardown.unlink()
        elif fault == "array_corrupt":
            array.write_bytes(b"corruption")
        elif fault == "bundle_foreign":
            terminal["bundle_sha256"] = "sha256:" + "f" * 64
        elif fault == "freeze_foreign":
            value = json.loads(execution.read_text())
            value["freeze_digest"] = "sha256:" + "f" * 64
            _write(execution, value, digest_field="result_digest")
        _write(result_path, terminal)
        return 0

    profile = dict(job["server_profile"], schema_version="task_evaluation_sam31_preparation_profile.v1",
                   repo_root=job["repo_root"], server_data_root=job["server_data_root"])
    profile_path = _write(tmp_path / "profile.json", profile, digest_field="profile_digest")
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SAM31_PREPARATION_PROFILE_FILE", str(profile_path))
    original = paid.execute_paid_stage
    monkeypatch.setattr(paid, "execute_paid_stage", lambda context: original(context, allocator_runner=runner))
    phase_job = {**job, "phase": "contribution_sweep", "job_digest": "sha256:" + "f" * 64,
                 "plan": {**job["plan"], "source_commit": COMMIT, "server_profile_sha256": paid.sha(profile_path)}}
    result = stages.execute_stage(phase_job)
    assert result["status"] == ("failed" if fault else "completed")
    if fault:
        return
    assert Path(result["artifacts"]["gaussian_contribution_evidence"]["path"]) == produced["manifest"]
    argv = seen[0]
    assert argv[argv.index("--adp-max-spend-usd") + 1] == "1.5"
    assert argv[argv.index("--adp-hard-ttl-seconds") + 1] == "3600"
    assert argv[-1] == "--execute"
    assert str(source) not in argv
    receipt_path = output / "phase_execution_receipt.v1.json"
    receipt_bytes = receipt_path.read_bytes()
    assert stages.execute_stage({**phase_job, "resume_only": True}) == result
    produced["array"].write_bytes(b"corrupted after phase completion")
    with pytest.raises(paid.Sam31PreparationPaidStageError, match="replay_terminal_artifacts_changed"):
        stages.execute_stage({**phase_job, "resume_only": True})
    assert len(seen) == 1
    assert receipt_path.read_bytes() == receipt_bytes


def test_paid_profile_cannot_expand_sam_spend_or_retry(tmp_path: Path) -> None:
    job, _ = _job(tmp_path, "sam31_tracking")
    job["server_profile"]["paid_stages"]["sam31_tracking"]["max_spend_usd"] = 1.01
    with pytest.raises(paid.Sam31PreparationPaidStageError, match="sam31_spend_cap_invalid"):
        paid.execute_paid_stage(job)


def test_closed_stage_dispatcher_reaches_paid_handler_and_seals_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _write(tmp_path / "profile.json", {
        "schema_version": "task_evaluation_sam31_preparation_profile.v1",
        "source_commit": COMMIT,
    }, digest_field="profile_digest")
    monkeypatch.setenv(
        "BLUEPRINT_TASK_EVALUATION_SAM31_PREPARATION_PROFILE_FILE", str(profile)
    )
    artifact = _write(tmp_path / "result.json", {"status": "complete"})
    seen = []

    def execute(job):
        seen.append(job)
        return {
            "status": "completed",
            "stage_id": "sam31_tracking",
            "artifacts": {"sam31_source_tracks": _record(artifact)},
            "candidate_policy_queried": False,
        }

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_sam31_preparation_paid_stages.execute_paid_stage",
        execute,
    )
    output = tmp_path / "stage-output"
    output.mkdir()
    result = stages.execute_stage({
        "phase": "sam31_tracking",
        "job_digest": "sha256:" + "f" * 64,
        "expected_source_commit": COMMIT,
        "plan": {"source_commit": COMMIT, "server_profile_sha256": paid.sha(profile)},
        "output_root": str(output),
        "request": {},
        "inputs": {},
    })
    assert result["status"] == "completed"
    assert seen[0]["stage_id"] == "sam31_tracking"
    receipt = json.loads((output / "phase_execution_receipt.v1.json").read_text())
    assert receipt["outcome"] == result
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


@pytest.mark.parametrize("defect", ["missing_record", "path_mismatch", "digest_drift"])
def test_contribution_refuses_unbound_or_changed_avoidlist_before_bundle_or_allocator(tmp_path, monkeypatch, defect):
    job, _output = _job(tmp_path, "contribution_sweep")
    source = _write(tmp_path / "data" / "avoidlist.json", {
        "schema_version": "vast_machine_avoidlist.v1", "machine_ids": [20166],
    })
    config = job["server_profile"]["paid_stages"]["contribution_sweep"]
    config.update(machine_avoidlist_path=str(source), machine_avoidlist=_record(source))
    if defect == "missing_record":
        config.pop("machine_avoidlist")
    elif defect == "path_mismatch":
        config["machine_avoidlist"]["path"] = str(source.parent / "other.json")
    else:
        source.write_bytes(source.read_bytes() + b" ")
    monkeypatch.setattr("blueprint_pipeline.adp_gaussian_excision_vast.build_gaussian_excision_vast_bundle",
                        lambda **kwargs: pytest.fail("must fail before building source bundle"))
    with pytest.raises(ValueError, match="avoidlist_binding_invalid|input_bytes_mismatch"):
        paid.execute_paid_stage(job, allocator_runner=lambda *_args, **_kwargs: pytest.fail("must not allocate"))


def test_contribution_resume_keeps_added_failures_without_changing_frozen_input(tmp_path):
    source = _write(tmp_path / "frozen.json", {
        "schema_version": "vast_machine_avoidlist.v1", "machine_ids": [20166],
    })
    source.chmod(0o440)
    config = {"machine_avoidlist_path": str(source), "machine_avoidlist": _record(source)}
    output = tmp_path / "attempt"
    output.mkdir()
    target = paid._contribution_machine_avoidlist(config=config, roots=(tmp_path,), output=output, resume_only=False)
    _write(target, {"schema_version": "vast_machine_avoidlist.v1", "machine_ids": [20166, 999999]})
    modified = target.read_bytes()
    assert paid._contribution_machine_avoidlist(config=config, roots=(tmp_path,), output=output,
                                               resume_only=True) == target
    assert target.read_bytes() == modified
    assert _record(source) == config["machine_avoidlist"]
    assert source.stat().st_mode & 0o777 == 0o440
    _write(target, {"schema_version": "vast_machine_avoidlist.v1", "machine_ids": [999999]})
    with pytest.raises(ValueError, match="attempt_dropped_exclusions"):
        paid._contribution_machine_avoidlist(config=config, roots=(tmp_path,), output=output, resume_only=True)


def test_contribution_rejects_source_aliasing_mutable_attempt_file(tmp_path):
    source = _write(tmp_path / "provider_machine_avoidlist.json", {
        "schema_version": "vast_machine_avoidlist.v1", "machine_ids": [20166],
    })
    with pytest.raises(ValueError, match="source_aliases_attempt"):
        paid._contribution_machine_avoidlist(
            config={"machine_avoidlist_path": str(source), "machine_avoidlist": _record(source)},
            roots=(tmp_path,), output=tmp_path, resume_only=False,
        )


def test_current_profile_cannot_drop_calibration_avoidlist_before_contribution(tmp_path):
    job, _output = _job(tmp_path, "contribution_sweep")
    source = _write(tmp_path / "data" / "avoidlist.json", {
        "schema_version": "vast_machine_avoidlist.v1", "machine_ids": [20166],
    })
    job["server_profile"]["calibrated_views"] = {"machine_avoidlist": _record(source)}
    with pytest.raises(ValueError, match="contribution_avoidlist_differs_from_calibration"):
        paid.execute_paid_stage(job, allocator_runner=lambda *_args, **_kwargs: pytest.fail("must not allocate"))
