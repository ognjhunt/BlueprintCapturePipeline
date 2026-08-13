from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline.public_scene_artifixer3d_vast import (
    DUAL_TARGET_PIPELINE_MODE,
    DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
    _materialize_raw_result,
    materialize_artifixer3d_paid_attempt_authority,
    recover_artifixer3d_local_closeout,
    validate_artifixer3d_bundle,
    validate_artifixer3d_paid_attempt_authority,
)
from blueprint_pipeline.public_scene_artifixer3d_bundle import (
    ArtiFixer3DBundleError,
    CHECKPOINT_REUSE_SCHEMA_VERSION,
    build_artifixer3d_bundle,
    materialize_artifixer3d_use_attestation,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_shell_script,
)
from tests.test_public_scene_artifixer3d_bundle import _repository, _source
from tests.test_public_scene_artifixer3d_dual_target_inputs import _dual_candidate
from tests.test_public_scene_artifixer3d_vast import _prior_chain


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, provider_path: str) -> dict[str, object]:
    return {
        "path": provider_path,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _absolute_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _reuse_source(
    tmp_path: Path,
    *,
    candidate: dict[str, object],
    steps: int = 10,
    corrupt_checkpoint: bool = False,
    terminal_status: str = "blocked",
) -> tuple[Path, Path, bytes]:
    task_ids = [str(task["task_id"]) for task in candidate["tasks"]]
    checkpoint_bytes = b"zero-closed-trained-checkpoint"
    checkpoint_sha = "sha256:" + hashlib.sha256(checkpoint_bytes).hexdigest()
    runtime_tasks: list[dict[str, object]] = []
    members: dict[str, bytes] = {}
    for index, task_id in enumerate(task_ids):
        member = f"tasks/{task_id}/artifixer3d/checkpoints/ckpt_{steps}.pt"
        members[member] = (
            b"tampered-checkpoint" if corrupt_checkpoint else checkpoint_bytes
        )
        runtime_tasks.append(
            {
                "task_id": task_id,
                "pipeline_mode": DUAL_TARGET_PIPELINE_MODE,
                "artifixer3d_checkpoint": {
                    "path": f"/provider/runtime_output/{member}",
                    "size_bytes": len(checkpoint_bytes),
                    "sha256": checkpoint_sha,
                },
            }
        )
    manifest_digest = "sha256:" + "1" * 64
    request_digest = "sha256:" + "2" * 64
    runtime: dict[str, object] = {
        "schema_version": "public_scene_artifixer3d_runtime_result.v1",
        "status": (
            "raw_artifixer3d_candidate_completed_requires_visual_and_multiview_review"
        ),
        "pipeline_mode": DUAL_TARGET_PIPELINE_MODE,
        "candidate_input_receipt_digest": candidate["receipt_digest"],
        "task_ids": task_ids,
        "manifest_digest": manifest_digest,
        "runtime_request_digest": request_digest,
        "artifixer3d_distillation_executed": True,
        "artifixer_direct_inference_executed": False,
        "semantic_editor_inference_executed": False,
        "artifixer3d_plus_inference_executed": False,
        "tasks": runtime_tasks,
    }
    runtime_path = tmp_path / "source/public_scene_artifixer3d_runtime_result.json"
    _write_json(runtime_path, runtime)
    members["public_scene_artifixer3d_runtime_result.json"] = runtime_path.read_bytes()
    provider_zip = tmp_path / "source/vast_provider_runtime_output.zip"
    with zipfile.ZipFile(provider_zip, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)

    authority: dict[str, object] = {
        "schema_version": "public_scene_artifixer3d_paid_attempt_authority.v1",
        "authorization_reference": "fixture-zero-closed-source",
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    authority_path = tmp_path / "source/attempt_authority.json"
    _write_json(authority_path, authority)
    attempt: dict[str, object] = {
        "schema_version": "public_scene_artifixer3d_vast_run.v1",
        "status": terminal_status,
        "execution_result_path": str(runtime_path.resolve()),
        "manifest_digest": manifest_digest,
        "runtime_request_digest": request_digest,
        "authorization_consumption": {
            "status": "consumed",
            "authorization_digest": authority["authorization_digest"],
        },
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
    }
    attempt_path = tmp_path / "source/attempt_result.json"
    _write_json(attempt_path, attempt)
    zero: dict[str, object] = {
        "schema_version": "artifixer3d_postblocked_provider_zero.v1",
        "attempt_authority_digest": authority["authorization_digest"],
        "attempt_terminal_status": terminal_status,
        "provider_zero_confirmed": True,
        "inventory": {"api_confirmed": True, "live_resource_count": 0},
        "attempt_authority": _absolute_record(authority_path),
        "attempt_result": _absolute_record(attempt_path),
        "continuing_spend_from_attempt": False,
        "all_staged_objects_absent": True,
        "receipt_digest": "",
    }
    zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
    zero_path = tmp_path / "source/provider_zero.json"
    _write_json(zero_path, zero)
    return provider_zip, zero_path, checkpoint_bytes


def test_raw_result_binds_only_physical_dual_target_review_frames(
    tmp_path: Path,
) -> None:
    execution_root = tmp_path / "immutable_execution"
    frame_rows: list[dict[str, object]] = []
    for index in range(8):
        relative = Path("tasks/task_a/artifixer3d_review_frames") / f"{index:05d}.png"
        path = execution_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"review-frame-{index}".encode())
        frame_rows.append(
            {
                "frame_index": index,
                "camera_id": f"task_a_camera_{index:05d}",
                **_record(path, provider_path=f"/provider/runtime_output/{relative}"),
            }
        )
    checkpoint_relative = Path("tasks/task_a/checkpoints/ckpt_30000.pt")
    checkpoint = execution_root / checkpoint_relative
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"dual-target-checkpoint")
    execution = {
        "tasks": [
            {
                "task_id": "task_a",
                "pipeline_mode": DUAL_TARGET_PIPELINE_MODE,
                "training_record_count": 16,
                "artifixer3d_review_frames": frame_rows,
                "artifixer3d_checkpoint": _record(
                    checkpoint,
                    provider_path=f"/provider/runtime_output/{checkpoint_relative}",
                ),
                "outside_support_invariance_status": (
                    "deferred_until_final_soft_composite"
                ),
                "outside_support_changed_pixels_total": None,
            }
        ]
    }
    raw = _materialize_raw_result(
        execution=execution,
        execution_root=execution_root,
        bundle={
            "pipeline_mode": DUAL_TARGET_PIPELINE_MODE,
            "task_ids": ["task_a"],
            "task_camera_counts": {"task_a": 8},
            "task_training_record_counts": {"task_a": 16},
            "bundle_sha256": "sha256:bundle",
            "manifest_digest": "sha256:manifest",
            "runtime_request_digest": "sha256:request",
            "replacement_object_count": 1,
        },
        closeout={"provider_zero_confirmed": True},
    )

    task = raw["tasks"][0]
    assert "final_candidate_frames" not in task
    assert len(task["artifixer3d_review_frames"]) == 8
    assert task["physical_camera_count"] == 8
    assert task["training_record_count"] == 16
    assert task["outside_support_invariance_status"] == (
        "deferred_until_final_soft_composite"
    )
    assert task["outside_support_invariance_proven"] is False
    assert raw["outside_exact_support_changed_pixels_total"] is None
    assert raw["outside_support_invariance_proven"] is False
    assert raw["appearance_repair_qualified"] is False


def test_local_closeout_recovery_rehashes_provider_bytes_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import blueprint_pipeline.public_scene_artifixer3d_vast as subject

    job = tmp_path / "vast_execute"
    execution_root = job / "immutable_execution"
    provider_run = job / "vast_provider_run"
    task_root = execution_root / "tasks/task_a"
    frame = task_root / "artifixer3d_review_frames/00000.png"
    checkpoint = task_root / "artifixer3d/checkpoints/ckpt_10.pt"
    frame.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")
    checkpoint.write_bytes(b"checkpoint")
    runtime: dict[str, object] = {
        "schema_version": "public_scene_artifixer3d_runtime_result.v1",
        "status": (
            "raw_artifixer3d_candidate_completed_requires_visual_and_multiview_review"
        ),
        "pipeline_mode": DUAL_TARGET_PIPELINE_MODE,
        "model_loaded": True,
        "provider_zero_required_after_return": True,
        "source_object_restoration_permitted": False,
        "artifixer_direct_inference_executed": False,
        "semantic_editor_inference_executed": False,
        "artifixer3d_distillation_executed": True,
        "artifixer3d_plus_inference_executed": False,
        "tasks": [
            {
                "task_id": "task_a",
                "pipeline_mode": DUAL_TARGET_PIPELINE_MODE,
                "training_record_count": 2,
                "artifixer3d_review_frames": [
                    {
                        "frame_index": 0,
                        "camera_id": "camera_0",
                        **_record(
                            frame,
                            provider_path=(
                                "/provider/runtime_output/tasks/task_a/"
                                "artifixer3d_review_frames/00000.png"
                            ),
                        ),
                    }
                ],
                "artifixer3d_checkpoint": _record(
                    checkpoint,
                    provider_path=(
                        "/provider/runtime_output/tasks/task_a/"
                        "artifixer3d/checkpoints/ckpt_10.pt"
                    ),
                ),
                "outside_support_invariance_status": (
                    "deferred_until_final_soft_composite"
                ),
                "outside_support_changed_pixels_total": None,
            }
        ],
    }
    runtime_path = execution_root / "public_scene_artifixer3d_runtime_result.json"
    _write_json(runtime_path, runtime)
    provider_run.mkdir(parents=True)
    provider_zip = provider_run / "vast_provider_runtime_output.zip"
    with zipfile.ZipFile(provider_zip, "w") as archive:
        archive.write(runtime_path, "public_scene_artifixer3d_runtime_result.json")
        archive.write(
            frame, "tasks/task_a/artifixer3d_review_frames/00000.png"
        )
        archive.write(
            checkpoint, "tasks/task_a/artifixer3d/checkpoints/ckpt_10.pt"
        )
    for path, value in (
        (
            provider_run / "vast_provider_adapter_result.json",
            {
                "status": "completed",
                "continuing_spend_from_this_run": False,
                "estimated_cost_usd": 0.1,
            },
        ),
        (
            provider_run / "vast_teardown_manifest.json",
            {"continuing_spend_from_this_run": False},
        ),
        (provider_run / "vast_final_validation.json", {"status": "completed"}),
        (
            job / "object_store_staging/wam_provider_object_store_cleanup.json",
            {"all_objects_absent": True, "signed_url_files_removed": True},
        ),
        (
            job
            / "independent_vast_watchdog/groot_oscar_runpod_canary_watchdog.json",
            {
                "status": "provider_terminal",
                "provider_absence_confirmed": True,
                "final_inventory": {
                    "api_confirmed": True,
                    "live_resource_count": 0,
                },
            },
        ),
    ):
        _write_json(path, value)
    authority: dict[str, object] = {
        "authorization_digest": "sha256:" + "a" * 64,
        "bundle_sha256": "sha256:bundle",
        "blueprint_commit": "fixture",
        "hard_attempt_spend_cap_usd": 1.0,
        "maximum_hourly_rate_usd": 0.5,
        "maximum_single_resource_ttl_seconds": 7200,
    }
    authority_path = tmp_path / "authority.json"
    _write_json(authority_path, authority)
    consumption_root = tmp_path / "consumed"
    consumption_root.mkdir()
    _write_json(
        consumption_root / f"artifixer3d-{'a' * 64}.json",
        {
            "schema_version": "artifixer3d_paid_attempt_consumption.v1",
            "authorization_digest": authority["authorization_digest"],
            "bundle_sha256": authority["bundle_sha256"],
            "blueprint_commit": authority["blueprint_commit"],
            "maximum_provider_allocations": 1,
        },
    )
    bundle = {
        "pipeline_mode": DUAL_TARGET_PIPELINE_MODE,
        "task_ids": ["task_a"],
        "task_camera_counts": {"task_a": 1},
        "task_training_record_counts": {"task_a": 2},
        "bundle_sha256": "sha256:bundle",
        "manifest_digest": "sha256:manifest",
        "runtime_request_digest": "sha256:request",
        "replacement_object_count": 1,
        "allowed_active_instance_ids": [],
    }
    monkeypatch.setattr(subject, "AUTHORIZATION_CONSUMPTION_ROOT", consumption_root)
    monkeypatch.setattr(subject, "validate_artifixer3d_bundle", lambda _path: bundle)
    monkeypatch.setattr(
        subject,
        "validate_artifixer3d_paid_attempt_authority",
        lambda value, **_kwargs: dict(value),
    )

    result = recover_artifixer3d_local_closeout(
        job_dir=job,
        bundle_receipt_path=tmp_path / "bundle.json",
        attempt_authority_path=authority_path,
    )

    assert result["status"] == "completed"
    assert result["local_receipt_recovered_after_provider_teardown"] is True
    assert result["retry_cap"] == 0
    assert Path(result["raw_result_path"]).is_file()
    assert Path(result["artifact_manifest_path"]).is_file()
    assert result["authorization_consumption"]["status"] == "consumed"


def test_sealed_dual_target_bundle_reopens_with_physical_and_training_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dual_root, _source_candidate, _teachers, _dual = _dual_candidate(
        tmp_path / "candidate", cameras_per_task=2
    )
    candidate = dual_root / "public_scene_artifixer3d_dual_target_inputs.v1.json"
    attestation = tmp_path / "attestation.json"
    materialize_artifixer3d_use_attestation(
        candidate_inputs_receipt_path=candidate,
        output_path=attestation,
        authorized_by="fixture_user",
    )
    artifixer_root = tmp_path / "artifixer"
    artifixer_root.mkdir()
    source, commit, tree = _source(artifixer_root)
    import blueprint_pipeline.public_scene_artifixer3d_bundle as subject

    monkeypatch.setattr(subject, "ARTIFIXER_COMMIT", commit)
    monkeypatch.setattr(subject, "ARTIFIXER_TREE", tree)
    monkeypatch.setattr(
        subject,
        "rehearse_provider_bundle_entrypoint",
        lambda **_kwargs: {
            "status": "passed",
            "provider_mutations_performed": 0,
            "paid_inference_performed": False,
            "gpu_runtime_started": False,
        },
    )
    output = tmp_path / "bundle"
    build_artifixer3d_bundle(
        candidate_inputs_receipt_path=candidate,
        use_attestation_path=attestation,
        artifixer_source_directory=source,
        output_root=output,
        repository_root=_repository(tmp_path),
        direct_editor_backend="none",
        semantic_editor_only=False,
        pipeline_mode=DUAL_TARGET_PIPELINE_MODE,
        artifixer3d_steps=10,
    )
    import blueprint_pipeline.public_scene_artifixer3d_vast as vast_subject

    parent_path = Path(_dual["execution_authority"]["path"])
    monkeypatch.setattr(
        vast_subject,
        "_validate_parent_execution_authority",
        lambda _attestation, *, publisher_scene_id: (
            parent_path,
            {
                "authority_digest": _dual["execution_authority"]["authority_digest"],
                "paid_compute": {
                    "hard_total_spend_cap_usd": 12.0,
                    "external_instance_allowlist": [47373597],
                },
            },
        ),
    )

    validated = validate_artifixer3d_bundle(
        output / "public_scene_artifixer3d_bundle_receipt.json"
    )
    assert validated["pipeline_mode"] == DUAL_TARGET_PIPELINE_MODE
    assert validated["direct_editor_backend"] == "none"
    assert validated["semantic_editor_only"] is False
    assert validated["task_camera_counts"] == {"task_1": 2}
    assert validated["task_training_record_counts"] == {"task_1": 4}


def test_dual_target_receipt_selects_paired_target_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dual_root, _source_candidate, _teachers, _dual = _dual_candidate(
        tmp_path / "candidate", cameras_per_task=2
    )
    candidate = dual_root / "public_scene_artifixer3d_dual_target_inputs.v1.json"
    attestation = tmp_path / "attestation.json"
    materialize_artifixer3d_use_attestation(
        candidate_inputs_receipt_path=candidate,
        output_path=attestation,
        authorized_by="fixture_user",
    )
    artifixer_root = tmp_path / "artifixer"
    artifixer_root.mkdir()
    source, commit, tree = _source(artifixer_root)
    import blueprint_pipeline.public_scene_artifixer3d_bundle as subject

    monkeypatch.setattr(subject, "ARTIFIXER_COMMIT", commit)
    monkeypatch.setattr(subject, "ARTIFIXER_TREE", tree)
    monkeypatch.setattr(
        subject,
        "rehearse_provider_bundle_entrypoint",
        lambda **_kwargs: {
            "status": "passed",
            "provider_mutations_performed": 0,
            "paid_inference_performed": False,
            "gpu_runtime_started": False,
        },
    )

    receipt = build_artifixer3d_bundle(
        candidate_inputs_receipt_path=candidate,
        use_attestation_path=attestation,
        artifixer_source_directory=source,
        output_root=tmp_path / "bundle",
        repository_root=_repository(tmp_path),
        artifixer3d_steps=10,
    )

    assert receipt["pipeline_mode"] == DUAL_TARGET_PIPELINE_MODE
    assert receipt["direct_editor_backend"] == "none"
    assert receipt["semantic_editor_only"] is False
    with zipfile.ZipFile(receipt["bundle"]["path"]) as archive:
        request = json.loads(
            archive.read("provider_runtime/artifixer3d_runtime_request.json")
        )
    assert request["pipeline_mode"] == DUAL_TARGET_PIPELINE_MODE
    assert request["direct_editor_backend"] == "none"
    assert "artifixer3d_plus" not in request
    assert request["phases"] == [
        "dual_target_input_validation",
        "artifixer3d_distillation",
        "artifixer3d_review_render",
        "native_appearance_export",
        "external_visual_and_multiview_review",
    ]


def test_render_only_bundle_seals_zero_closed_checkpoint_and_source_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dual_root, _source_candidate, _teachers, dual = _dual_candidate(
        tmp_path / "candidate", cameras_per_task=2
    )
    candidate = dual_root / "public_scene_artifixer3d_dual_target_inputs.v1.json"
    provider_zip, provider_zero, checkpoint_bytes = _reuse_source(
        tmp_path,
        candidate=dual,
    )
    attestation = tmp_path / "attestation.json"
    materialize_artifixer3d_use_attestation(
        candidate_inputs_receipt_path=candidate,
        output_path=attestation,
        authorized_by="fixture_user",
    )
    artifixer_root = tmp_path / "artifixer"
    artifixer_root.mkdir()
    source, commit, tree = _source(artifixer_root)
    import blueprint_pipeline.public_scene_artifixer3d_bundle as subject

    monkeypatch.setattr(subject, "ARTIFIXER_COMMIT", commit)
    monkeypatch.setattr(subject, "ARTIFIXER_TREE", tree)
    monkeypatch.setattr(
        subject,
        "rehearse_provider_bundle_entrypoint",
        lambda **_kwargs: {
            "status": "passed",
            "provider_mutations_performed": 0,
            "paid_inference_performed": False,
            "gpu_runtime_started": False,
        },
    )
    output = tmp_path / "bundle"
    receipt = build_artifixer3d_bundle(
        candidate_inputs_receipt_path=candidate,
        use_attestation_path=attestation,
        artifixer_source_directory=source,
        output_root=output,
        repository_root=_repository(tmp_path),
        direct_editor_backend="none",
        semantic_editor_only=False,
        pipeline_mode=DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
        artifixer3d_steps=10,
        reused_checkpoint_provider_output_zip_path=provider_zip,
        reused_checkpoint_source_provider_zero_path=provider_zero,
    )
    assert receipt["pipeline_mode"] == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
    assert str(receipt["checkpoint_reuse_digest"]).startswith("sha256:")
    bundle_path = Path(receipt["bundle"]["path"])
    with zipfile.ZipFile(bundle_path) as archive:
        request = json.loads(
            archive.read(
                "provider_runtime/artifixer3d_runtime_request.json"
            ).decode()
        )
        manifest = json.loads(
            archive.read(
                "provider_runtime/artifixer3d_bundle_manifest.json"
            ).decode()
        )
        reuse = request["artifixer3d"]["checkpoint_reuse"]
        checkpoint = reuse["checkpoints"][0]
        checkpoint_member = (
            "provider_runtime/input/" + checkpoint["checkpoint"]["relative_path"]
        )
        assert archive.read(checkpoint_member) == checkpoint_bytes
        assert checkpoint["checkpoint"]["size_bytes"] == len(checkpoint_bytes)
        assert checkpoint["checkpoint"]["sha256"] == (
            "sha256:" + hashlib.sha256(checkpoint_bytes).hexdigest()
        )
        assert reuse["schema_version"] == CHECKPOINT_REUSE_SCHEMA_VERSION
        assert reuse["source_attempt_terminal_status"] == "blocked"
        assert reuse["training_reexecution_permitted"] is False
        assert reuse["direct_inference_permitted"] is False
        assert reuse["artifixer3d_plus_permitted"] is False
        assert manifest["contains_model_weights"] is True
        assert manifest["contains_reused_private_derived_3dgrut_checkpoint"] is True
        assert manifest["contains_released_direct_model_weights"] is False
        assert not any(name.endswith("vast_provider_runtime_output.zip") for name in archive.namelist())

    import blueprint_pipeline.public_scene_artifixer3d_vast as vast_subject

    parent_path = Path(dual["execution_authority"]["path"])
    monkeypatch.setattr(
        vast_subject,
        "_validate_parent_execution_authority",
        lambda _attestation, *, publisher_scene_id: (
            parent_path,
            {
                "authority_digest": dual["execution_authority"]["authority_digest"],
                "paid_compute": {
                    "hard_total_spend_cap_usd": 12.0,
                    "external_instance_allowlist": [47373597],
                },
            },
        ),
    )
    validated = validate_artifixer3d_bundle(
        output / "public_scene_artifixer3d_bundle_receipt.json"
    )
    assert validated["pipeline_mode"] == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
    assert validated["checkpoint_reuse_digest"] == reuse["reuse_digest"]
    assert validated["reused_checkpoints"]["task_1"] == {
        "size_bytes": len(checkpoint_bytes),
        "sha256": "sha256:" + hashlib.sha256(checkpoint_bytes).hexdigest(),
        "source_provider_zip_member": (
            "tasks/task_1/artifixer3d/checkpoints/ckpt_10.pt"
        ),
    }
    prior_path, terminal_path = _prior_chain(tmp_path / "authority-chain")
    authority = materialize_artifixer3d_paid_attempt_authority(
        bundle_receipt_path=(
            output / "public_scene_artifixer3d_bundle_receipt.json"
        ),
        prior_aura_authority_path=prior_path,
        prior_terminal_result_path=terminal_path,
        authorization_reference="fixture-render-only-one-shot",
        authorized_by="fixture_user",
        authorized_on="2026-08-12",
        blueprint_commit=validated["blueprint_source_identity"]["commit"],
        max_hourly_rate_usd=1.5,
        hard_cap_usd=9.0,
        hard_ttl_seconds=21_600,
        output_path=tmp_path / "render_only_authority.json",
    )
    assert authority["checkpoint_reuse_digest"] == reuse["reuse_digest"]
    assert validate_artifixer3d_paid_attempt_authority(
        authority,
        prepared_bundle=validated,
        max_hourly_rate_usd=1.5,
        hard_cap_usd=9.0,
        hard_ttl_seconds=21_600,
        allowed_active_instance_ids=validated["allowed_active_instance_ids"],
    )["authorization_digest"] == authority["authorization_digest"]

    tampered = dict(authority)
    tampered["checkpoint_reuse_digest"] = "sha256:" + "f" * 64
    tampered["authorization_digest"] = canonical_digest(
        tampered, digest_field="authorization_digest"
    )
    with pytest.raises(
        ValueError, match="artifixer3d_paid_attempt_authority_invalid"
    ):
        validate_artifixer3d_paid_attempt_authority(
            tampered,
            prepared_bundle=validated,
            max_hourly_rate_usd=1.5,
            hard_cap_usd=9.0,
            hard_ttl_seconds=21_600,
            allowed_active_instance_ids=validated["allowed_active_instance_ids"],
        )


def test_render_only_bundle_rejects_checkpoint_bytes_not_matching_source_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dual_root, _source_candidate, _teachers, dual = _dual_candidate(
        tmp_path / "candidate", cameras_per_task=2
    )
    candidate = dual_root / "public_scene_artifixer3d_dual_target_inputs.v1.json"
    provider_zip, provider_zero, _checkpoint = _reuse_source(
        tmp_path,
        candidate=dual,
        corrupt_checkpoint=True,
    )
    attestation = tmp_path / "attestation.json"
    materialize_artifixer3d_use_attestation(
        candidate_inputs_receipt_path=candidate,
        output_path=attestation,
        authorized_by="fixture_user",
    )
    artifixer_root = tmp_path / "artifixer"
    artifixer_root.mkdir()
    source, commit, tree = _source(artifixer_root)
    import blueprint_pipeline.public_scene_artifixer3d_bundle as subject

    monkeypatch.setattr(subject, "ARTIFIXER_COMMIT", commit)
    monkeypatch.setattr(subject, "ARTIFIXER_TREE", tree)
    with pytest.raises(
        ArtiFixer3DBundleError,
        match="artifixer3d_checkpoint_reuse_checkpoint_(invalid|mismatch)",
    ):
        build_artifixer3d_bundle(
            candidate_inputs_receipt_path=candidate,
            use_attestation_path=attestation,
            artifixer_source_directory=source,
            output_root=tmp_path / "bundle",
            repository_root=_repository(tmp_path),
            direct_editor_backend="none",
            semantic_editor_only=False,
            pipeline_mode=DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
            artifixer3d_steps=10,
            reused_checkpoint_provider_output_zip_path=provider_zip,
            reused_checkpoint_source_provider_zero_path=provider_zero,
        )


def test_dual_target_raw_result_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="artifixer3d_runtime_task_outputs_invalid"):
        _materialize_raw_result(
            execution={
                "tasks": [
                    {
                        "task_id": "task_a",
                        "artifixer3d_review_frames": [],
                        "artifixer3d_checkpoint": None,
                        "outside_support_invariance_status": (
                            "deferred_until_final_soft_composite"
                        ),
                        "outside_support_changed_pixels_total": None,
                    }
                ]
            },
            execution_root=tmp_path,
            bundle={
                "pipeline_mode": DUAL_TARGET_PIPELINE_MODE,
                "task_ids": ["task_a"],
                "task_camera_counts": {"task_a": 0},
                "task_training_record_counts": {"task_a": 0},
                "bundle_sha256": "sha256:bundle",
                "manifest_digest": "sha256:manifest",
                "runtime_request_digest": "sha256:request",
                "replacement_object_count": 1,
            },
            closeout={},
        )


def test_render_only_raw_result_binds_reused_checkpoint_without_returning_it(
    tmp_path: Path,
) -> None:
    execution_root = tmp_path / "immutable_execution"
    frame_rows: list[dict[str, object]] = []
    for index in range(8):
        relative = Path("tasks/task_a/artifixer3d_review_frames") / f"{index:05d}.png"
        path = execution_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"review-frame-{index}".encode())
        frame_rows.append(
            {
                "frame_index": index,
                "camera_id": f"task_a_camera_{index:05d}",
                **_record(path, provider_path=f"/provider/runtime_output/{relative}"),
            }
        )
    checkpoint_identity = {
        "size_bytes": 708_038_581,
        "sha256": "sha256:" + "a" * 64,
    }
    reuse_digest = "sha256:" + "b" * 64
    execution = {
        "tasks": [
            {
                "task_id": "task_a",
                "pipeline_mode": DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
                "training_record_count": 16,
                "checkpoint_reused": True,
                "checkpoint_reuse_digest": reuse_digest,
                "training_executed": False,
                "direct_artifixer_executed": False,
                "artifixer3d_plus_executed": False,
                "artifixer3d_review_frames": frame_rows,
                "artifixer3d_checkpoint": {
                    "path": "/provider/provider_runtime/input/checkpoint_reuse/checkpoint_00000.pt",
                    **checkpoint_identity,
                },
                "outside_support_invariance_status": (
                    "deferred_until_final_soft_composite"
                ),
                "outside_support_changed_pixels_total": None,
            }
        ]
    }
    bundle = {
        "pipeline_mode": DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
        "task_ids": ["task_a"],
        "task_camera_counts": {"task_a": 8},
        "task_training_record_counts": {"task_a": 16},
        "bundle_sha256": "sha256:bundle",
        "manifest_digest": "sha256:manifest",
        "runtime_request_digest": "sha256:request",
        "replacement_object_count": 1,
        "checkpoint_reuse_digest": reuse_digest,
        "reused_checkpoints": {
            "task_a": {
                **checkpoint_identity,
                "source_provider_zip_member": (
                    "tasks/task_a/artifixer3d/checkpoints/ckpt_30000.pt"
                ),
            }
        },
    }

    raw = _materialize_raw_result(
        execution=execution,
        execution_root=execution_root,
        bundle=bundle,
        closeout={"provider_zero_confirmed": True},
    )

    reused = raw["tasks"][0]["artifixer3d_checkpoint"]
    assert reused == {
        **checkpoint_identity,
        "checkpoint_reused": True,
        "checkpoint_reuse_digest": reuse_digest,
        "source_provider_zip_member": (
            "tasks/task_a/artifixer3d/checkpoints/ckpt_30000.pt"
        ),
    }
    assert not any(execution_root.rglob("*.pt"))
    assert len(raw["tasks"][0]["artifixer3d_review_frames"]) == 8

    execution["tasks"][0]["artifixer3d_checkpoint"]["sha256"] = (
        "sha256:" + "c" * 64
    )
    with pytest.raises(
        ValueError, match="artifixer3d_runtime_checkpoint_reuse_mismatch"
    ):
        _materialize_raw_result(
            execution=execution,
            execution_root=execution_root,
            bundle=bundle,
            closeout={"provider_zero_confirmed": True},
        )


def _provider_bundle(
    path: Path,
    *,
    include_dual_candidate: bool,
    pipeline_mode: str = DUAL_TARGET_PIPELINE_MODE,
) -> None:
    members: dict[str, bytes] = {
        "provider_runtime/run_public_scene_artifixer3d.sh": b"#!/bin/sh\n",
        "provider_runtime/public_scene_artifixer3d_runner.py": b"# runner\n",
        "provider_runtime/artifixer3d_bundle_manifest.json": b"{}\n",
        "provider_runtime/artifixer3d_runtime_request.json": json.dumps(
            {"pipeline_mode": pipeline_mode}
        ).encode(),
        "provider_runtime/artifixer3d_use_attestation.json": b"{}\n",
    }
    candidate_name = (
        "public_scene_artifixer3d_dual_target_inputs.v1.json"
        if include_dual_candidate
        else "public_scene_artifixer3d_candidate_inputs.v3.json"
    )
    members[f"provider_runtime/input/{candidate_name}"] = b"{}\n"
    with zipfile.ZipFile(path, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)


def test_provider_preflight_requires_new_dual_target_candidate_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import blueprint_pipeline.vast_provider_adapter as subject

    monkeypatch.setattr(subject, "provider_runtime_contract_blockers", lambda **_kwargs: [])
    good = tmp_path / "good.zip"
    _provider_bundle(good, include_dual_candidate=True)
    passed = _blueprint_bundle_preflight(
        job_dir=tmp_path / "passed",
        generated_at="2026-08-12T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_artifixer3d",
        bundle_path=good,
        provider_bundle_url="https://object.invalid/bundle",
        provider_output_put_url="https://object.invalid/output",
    )
    assert passed["status"] == "passed"
    assert passed["missing_zip_entries"] == []

    stale = tmp_path / "stale.zip"
    _provider_bundle(stale, include_dual_candidate=False)
    blocked = _blueprint_bundle_preflight(
        job_dir=tmp_path / "blocked",
        generated_at="2026-08-12T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_artifixer3d",
        bundle_path=stale,
        provider_bundle_url="https://object.invalid/bundle",
        provider_output_put_url="https://object.invalid/output",
    )
    assert blocked["status"] == "blocked"
    assert blocked["missing_zip_entries"] == [
        "provider_runtime/input/public_scene_artifixer3d_dual_target_inputs.v1.json"
    ]


def test_provider_preflight_classifies_render_only_as_dual_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import blueprint_pipeline.vast_provider_adapter as subject

    monkeypatch.setattr(subject, "provider_runtime_contract_blockers", lambda **_kwargs: [])
    bundle = tmp_path / "render-only.zip"
    _provider_bundle(
        bundle,
        include_dual_candidate=True,
        pipeline_mode=DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
    )
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="2026-08-12T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_artifixer3d",
        bundle_path=bundle,
        provider_bundle_url="https://object.invalid/bundle",
        provider_output_put_url="https://object.invalid/output",
    )
    assert preflight["status"] == "passed"
    assert preflight["missing_zip_entries"] == []


def test_provider_output_allowlist_retains_raw_artifixer3d_review_frames() -> None:
    shell = _probe_shell_script(
        "https://heartbeat.invalid",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_artifixer3d",
    )
    normalized_review_predicate = (
        "'/artifixer3d_review_frames/' in '/' + relative"
    )
    assert normalized_review_predicate in shell
    assert "'/native_appearance/' in '/' + relative" in shell
    assert "parts[0] == 'tasks'" in shell
    assert "provider_runtime/input/checkpoint_reuse" not in shell
