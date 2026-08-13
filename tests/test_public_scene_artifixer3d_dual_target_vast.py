from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline.public_scene_artifixer3d_vast import (
    DUAL_TARGET_PIPELINE_MODE,
    _materialize_raw_result,
    validate_artifixer3d_bundle,
)
from blueprint_pipeline.public_scene_artifixer3d_bundle import (
    build_artifixer3d_bundle,
    materialize_artifixer3d_use_attestation,
)
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_shell_script,
)
from tests.test_public_scene_artifixer3d_bundle import _repository, _source
from tests.test_public_scene_artifixer3d_dual_target_inputs import _dual_candidate


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, provider_path: str) -> dict[str, object]:
    return {
        "path": provider_path,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


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


def _provider_bundle(path: Path, *, include_dual_candidate: bool) -> None:
    members: dict[str, bytes] = {
        "provider_runtime/run_public_scene_artifixer3d.sh": b"#!/bin/sh\n",
        "provider_runtime/public_scene_artifixer3d_runner.py": b"# runner\n",
        "provider_runtime/artifixer3d_bundle_manifest.json": b"{}\n",
        "provider_runtime/artifixer3d_runtime_request.json": json.dumps(
            {"pipeline_mode": DUAL_TARGET_PIPELINE_MODE}
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
    assert "parts[0] == 'tasks'" in shell
