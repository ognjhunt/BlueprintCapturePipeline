from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_aura_adapter import SCHEMA_VERSION as ADAPTER_SCHEMA
from blueprint_pipeline.public_scene_aura_execution import (
    REQUIRED_STAGES,
    AuraExecutionReceiptError,
    materialize_aura_execution_receipt,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record(path: Path, root: Path) -> dict:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    evidence = tmp_path / "evidence"
    immutable = evidence / "run/immutable_execution"
    repo.mkdir()
    immutable.mkdir(parents=True)
    source = {
        "repository": "https://github.com/kkennethwu/AuraFusion360_official",
        "commit": "a" * 40,
        "tree": "b" * 40,
        "tracked_files_clean": True,
        "source_modified_for_adapter": False,
    }
    adapter = {
        "schema_version": ADAPTER_SCHEMA,
        "status": "prepared_unexecuted",
        "scene": {
            "publisher_scene_id": "840313",
            "target_instance_id": "ins160",
            "target_semantic_label": "canned_beverage",
            "camera_count": 2,
            "input_receipt_digest": "sha256:" + "c" * 64,
            "reference_camera_id": "view_b",
            "reference_camera_index": 1,
        },
        "artifacts": [
            {"relative_path": "data/Other-360/840313_ins160/images/view_a.png"},
            {"relative_path": "data/Other-360/840313_ins160/images/view_b.png"},
        ],
        "source": source,
        "receipt_digest": "",
    }
    adapter["receipt_digest"] = canonical_digest(adapter, digest_field="receipt_digest")
    adapter_path = repo / "adapter.json"
    _write(adapter_path, adapter)

    workflow = []
    for stage in REQUIRED_STAGES:
        log = immutable / f"aura-{stage}.log"
        log.write_text(f"executed {stage}\n", encoding="utf-8")
        workflow.append(
            {
                "stage": stage,
                "command": ["python", f"{stage}.py"],
                "returncode": 0,
                "timed_out": False,
                "runtime_seconds": 1.0,
                "stdout_stderr_sha256": _sha256(log),
                "log": log.name,
            }
        )
    ply = immutable / "artifacts/final.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nend_header\n")
    frames = []
    for name in ("view_a.png", "view_b.png"):
        frame = immutable / "artifacts/final_frames" / name
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"png-" + name.encode())
        frames.append(_record(frame, immutable))
    runtime = {
        "schema_version": "adp_aura_interiorgs_result.v1",
        "status": "completed",
        "scene_id": "840313",
        "target_instance_id": "ins160",
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "source_identity_before": {"matches": True, "changed_files": []},
        "source_identity_after": {"matches": True, "changed_files": []},
        "source_modified": False,
        "workflow": workflow,
        "reference_generation_executed": True,
        "training_executed": True,
        "removal_executed": True,
        "inpaint_init_executed": True,
        "inpaint_finetune_executed": True,
        "final_point_cloud": {**_record(ply, immutable), "vertex_count": 3},
        "final_frames": frames,
        "depth_anything3_used": False,
        "blockers": [],
    }
    runtime_path = immutable / "adp_aura_interiorgs_result.json"
    _write(runtime_path, runtime)

    teardown_path = evidence / "run/vast_provider_run/vast_teardown_manifest.json"
    _write(teardown_path, {"continuing_spend_from_this_run": False})
    run = {
        "schema_version": "adp_aura_interiorgs_vast_run.v1",
        "status": "completed",
        "execution_result_path": str(runtime_path),
        "teardown_manifest_path": str(teardown_path),
        "estimated_cost_usd": 0.5,
        "hard_cap_usd": 6.0,
        "hard_ttl_seconds": 14_400,
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "blockers": [],
    }
    run_path = evidence / "run/adp_aura_interiorgs_vast_result.json"
    _write(run_path, run)
    return {
        "repo": repo,
        "evidence": evidence,
        "adapter": adapter_path,
        "runtime": runtime_path,
        "run": run_path,
        "ply": ply,
        "teardown": teardown_path,
    }


def _materialize(paths: dict[str, Path]) -> dict:
    return materialize_aura_execution_receipt(
        adapter_receipt_path=paths["adapter"],
        runtime_result_path=paths["runtime"],
        run_result_path=paths["run"],
        evidence_root=paths["evidence"],
        repo_root=paths["repo"],
        receipt_output=paths["repo"] / "receipt.json",
    )


def test_aura_execution_receipt_hashes_observed_files_without_self_admission(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    receipt = _materialize(paths)

    assert receipt["status"] == "executed_candidate"
    assert receipt["execution"]["required_stages"] == list(REQUIRED_STAGES)
    assert len(receipt["execution"]["stage_logs"]) == len(REQUIRED_STAGES)
    assert receipt["execution"]["final_point_cloud"]["sha256"] == _sha256(paths["ply"])
    assert receipt["provider_run"]["continuing_spend_from_this_run"] is False
    assert receipt["claim_boundary"]["released_method_execution_completed"] is True
    assert receipt["claim_boundary"]["successful_inpainting_admitted"] is False
    assert receipt["blockers"] == ["aurafusion360_stage_localization_evidence_missing"]
    assert receipt["quality"]["runtime_reference_camera_binding_valid"] is True
    assert receipt["quality"]["intermediate_stage_artifacts_retained"] is False
    assert [row["camera_id"] for row in receipt["execution"]["final_frames"]] == [
        "view_a",
        "view_b",
    ]
    assert canonical_digest(receipt, digest_field="receipt_digest") == receipt["receipt_digest"]


def test_aura_execution_receipt_verifies_stage_localization_frames(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    runtime = json.loads(paths["runtime"].read_text())
    frame_sets = {}
    for role in ("inpaint_init_renders", "sdedit_images"):
        records = []
        for index in range(2):
            frame = paths["runtime"].parent / "artifacts/intermediate_frames" / role / f"{index:05d}.png"
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(f"{role}-{index}".encode())
            records.append(_record(frame, paths["runtime"].parent))
        frame_sets[role] = records
    runtime["intermediate_frame_sets"] = frame_sets
    runtime["stage_localization_evidence_retained"] = True
    _write(paths["runtime"], runtime)

    receipt = _materialize(paths)

    assert receipt["quality"]["intermediate_stage_artifacts_retained"] is True
    assert receipt["blockers"] == ["aurafusion360_interiorgs_quality_admission_missing"]
    assert set(receipt["execution"]["intermediate_frame_sets"]) == set(frame_sets)


def test_aura_execution_retains_runtime_reference_camera_index_mismatch(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    adapter = json.loads(paths["adapter"].read_text())
    adapter["scene"]["reference_camera_index"] = 0
    adapter["receipt_digest"] = canonical_digest(adapter, digest_field="receipt_digest")
    _write(paths["adapter"], adapter)

    receipt = _materialize(paths)

    assert receipt["status"] == "executed_candidate"
    assert receipt["quality"]["runtime_reference_camera_binding_observed"] is True
    assert receipt["quality"]["runtime_reference_camera_binding_valid"] is False
    assert receipt["quality"]["configured_reference_index"] == 0
    assert receipt["quality"]["expected_runtime_sorted_reference_index"] == 1
    assert receipt["blockers"] == [
        "aurafusion360_runtime_reference_camera_binding_mismatch"
    ]


def test_aura_execution_receipt_accepts_legacy_v1_frame_path_field(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    runtime = json.loads(paths["runtime"].read_text())
    for record in runtime["final_frames"]:
        record["path"] = record.pop("relative_path")
    _write(paths["runtime"], runtime)

    receipt = _materialize(paths)

    assert len(receipt["execution"]["final_frames"]) == 2
    assert all(
        "relative_path" in record
        for record in receipt["execution"]["final_frames"]
    )


def test_aura_execution_receipt_rejects_ambiguous_frame_path_fields(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    runtime = json.loads(paths["runtime"].read_text())
    runtime["final_frames"][0]["path"] = runtime["final_frames"][0][
        "relative_path"
    ]
    _write(paths["runtime"], runtime)

    with pytest.raises(
        AuraExecutionReceiptError, match="aurafusion360_final_frame_changed"
    ):
        _materialize(paths)


def test_aura_execution_receipt_rejects_changed_final_point_cloud(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["ply"].write_bytes(paths["ply"].read_bytes() + b"changed")

    with pytest.raises(
        AuraExecutionReceiptError, match="aurafusion360_final_point_cloud_changed"
    ):
        _materialize(paths)


def test_aura_execution_receipt_rejects_caller_asserted_provider_zero(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    run = json.loads(paths["run"].read_text())
    run["continuing_spend_from_this_run"] = True
    _write(paths["run"], run)

    with pytest.raises(
        AuraExecutionReceiptError, match="aurafusion360_provider_zero_not_proven"
    ):
        _materialize(paths)


def test_aura_execution_receipt_rejects_missing_stage_log(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths["runtime"].parent / "aura-sdedit.log").unlink()

    with pytest.raises(
        AuraExecutionReceiptError, match="aurafusion360_stage_log_changed"
    ):
        _materialize(paths)
