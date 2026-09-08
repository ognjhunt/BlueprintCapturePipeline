from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import artifixer_completed_training_reuse as reuse
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_artifixer_warm_checkpoint import (
    materialize_artifixer_post_training_checkpoint,
)
from tests.test_task_evaluation_scene_configuration_artifixer_warm_checkpoint import _fixture


def sealed(path, value, field="receipt_digest"):
    value[field] = canonical_digest(value, digest_field=field)
    path.write_text(json.dumps(value))
    return value


def setup_source(tmp_path):
    source = tmp_path / "source-launch"
    source.mkdir()
    job = source / "allocator/scene-configuration-job"
    (job / "vast_provider_run").mkdir(parents=True)
    sealed(
        source / "post_teardown_provider_zero_receipt.json",
        {
            "launch_id": source.name,
            "status": "provider_zero_confirmed",
            "provider_zero_verified": True,
            "continuing_spend_from_this_run": False,
            "blockers": [],
        },
        "provider_zero_receipt_digest",
    )
    (source / "launch_receipt.json").write_text(json.dumps({"status": "blocked"}))
    f = _fixture(tmp_path)
    tuning = {"artifixer3d_steps": 30000, "random_seed": 1, "transition_radius_pixels": 3}
    f["bindings"]["artifixer_tuning"] = tuning
    runtime = json.loads(f["runtime_path"].read_text())
    runtime["artifixer3d_distillation_executed"] = True
    runtime["tasks"][0]["native_appearance"] = {
        "geometry_protection": {
            "mode": "freeze_declared_appearance_initialization",
            "status": "qualified",
            "exact_source_appearance_prefix_match": True,
            "exact_full_density_tensor_match": True,
            "exact_position_tensor_match": True,
            "exact_rotation_tensor_match": True,
            "exact_scale_tensor_match": True,
            "initialization_receipt_digest": "sha256:" + "1" * 64,
            "blockers": [],
        }
    }
    f["runtime_path"].write_text(json.dumps(runtime))
    checkpoint_root = tmp_path / "checkpoint"
    materialize_artifixer_post_training_checkpoint(
        source_diagnostic_checkpoint=f["source"],
        bindings=f["bindings"],
        runtime_result_path=f["runtime_path"],
        review_frames=f["frames"],
        native_appearance_path=f["native"],
        output_root=checkpoint_root,
    )
    with zipfile.ZipFile(job / "vast_provider_run/vast_provider_runtime_output.zip", "w") as z:
        for p in checkpoint_root.rglob("*"):
            if p.is_file():
                z.write(p, reuse.PREFIX + str(p.relative_to(checkpoint_root)))
    record = {"path": "/old/image.png", "sha256": "sha256:" + "2" * 64, "size_bytes": 10}
    task = {
        "task_id": "remove-source-object-104",
        "transforms": record,
        "camera_index": record,
        "frames": [
            {
                "frame_index": i,
                "camera_id": f"camera-{i}",
                "input_retained_frame": record,
                "input_exact_repair_mask": record,
            }
            for i in range(8)
        ],
    }
    candidate = {
        "publisher_scene_id": "scene",
        "shared_retained_scene": record,
        "appearance_initialization": {
            "geometry_mode": "freeze_declared_appearance_initialization",
            "parameter_partition": {"frozen_source_count": 10},
        },
        "tasks": [task],
    }
    teacher = {
        "task_id": task["task_id"],
        "frames": [{"camera_id": f"camera-{i}", "teacher": record} for i in range(8)],
    }
    teacher_path = tmp_path / "teacher.json"
    teacher_path.write_text(json.dumps(teacher))
    prepared = {"candidate": candidate, "teacher_receipt_path": str(teacher_path)}
    stage = {"configuration_sha256": f["bindings"]["configuration_sha256"]}
    state = {"state": prepared, "source_commit": "a" * 40}
    state["state_digest"] = canonical_digest(state, digest_field="state_digest")
    cap = {"logical_root": str(tmp_path), "state_path": "output/pretraining_state.json"}
    cap["capsule_digest"] = canonical_digest(cap, digest_field="capsule_digest")
    capsule_path = job / "api_pretraining_capsule.zip"
    with zipfile.ZipFile(capsule_path, "w") as z:
        z.writestr("capsule_manifest.json", json.dumps(cap))
        z.writestr(cap["state_path"], json.dumps(state))
        z.writestr("output/stage_production_input.v1.json", json.dumps(stage))
        z.writestr("teacher.json", json.dumps(teacher))
    sealed(
        job / "api_pretraining_receipt.json",
        {
            "capsule_sha256": "sha256:" + hashlib.sha256(capsule_path.read_bytes()).hexdigest(),
            "capsule_digest": cap["capsule_digest"],
        },
    )
    return {
        "source_launch_root": source,
        "prepared": prepared,
        "stage_input": stage,
        "tuning": tuning,
        "output_root": tmp_path / "adopted",
    }


def test_closed_training_reuses_exact_frames_and_native_bytes_without_qualification(tmp_path):
    args = setup_source(tmp_path)
    ref = reuse.stage_completed_training(**args)
    result = reuse.hydrate_completed_training(
        reference=ref,
        candidate=args["prepared"]["candidate"],
        teacher_receipt_path=Path(args["prepared"]["teacher_receipt_path"]),
        tuning=args["tuning"],
        configuration_sha256=args["stage_input"]["configuration_sha256"],
    )
    assert len(result["review_frames"]) == 8
    assert Path(result["native_appearance_path"]).read_bytes() == b"native-usdz"
    assert result["reuse_receipt"]["new_training_executed"] is False
    assert result["reuse_receipt"]["new_independent_review_required"] is True
    assert result["checkpoint"]["qualification_eligible"] is False


@pytest.mark.parametrize("changed", ["initialization", "camera", "mask", "teacher", "seed"])
def test_training_reuse_refuses_any_changed_training_input(tmp_path, changed):
    args = setup_source(tmp_path)
    args["prepared"] = copy.deepcopy(args["prepared"])
    c = args["prepared"]["candidate"]
    digest = "sha256:" + "f" * 64
    if changed == "initialization":
        c["shared_retained_scene"]["sha256"] = digest
    elif changed == "camera":
        c["tasks"][0]["transforms"]["sha256"] = digest
    elif changed == "mask":
        c["tasks"][0]["frames"][0]["input_exact_repair_mask"]["sha256"] = digest
    elif changed == "seed":
        args["tuning"]["random_seed"] += 1
    else:
        p = Path(args["prepared"]["teacher_receipt_path"])
        d = json.loads(p.read_text())
        d["frames"][0]["teacher"]["sha256"] = digest
        p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="training_inputs_changed"):
        reuse.stage_completed_training(**args)
    assert not args["output_root"].exists()


def test_open_provider_lineage_cannot_be_reused(tmp_path):
    args = setup_source(tmp_path)
    p = args["source_launch_root"] / "post_teardown_provider_zero_receipt.json"
    d = json.loads(p.read_text())
    d["provider_zero_verified"] = False
    sealed(p, d, "provider_zero_receipt_digest")
    with pytest.raises(ValueError, match="source_not_closed"):
        reuse.stage_completed_training(**args)


def test_changed_copied_frame_is_refused_at_consumption(tmp_path):
    args = setup_source(tmp_path)
    ref = reuse.stage_completed_training(**args)
    frame = Path(ref["checkpoint_root"]) / "review/frames/00000.png"
    frame.chmod(0o640)
    frame.write_bytes(b"changed")
    with pytest.raises(ValueError):
        reuse.hydrate_completed_training(
            reference=ref,
            candidate=args["prepared"]["candidate"],
            teacher_receipt_path=Path(args["prepared"]["teacher_receipt_path"]),
            tuning=args["tuning"],
            configuration_sha256=args["stage_input"]["configuration_sha256"],
        )


def test_real_final_review_reuses_only_identical_multimodal_input(tmp_path):
    from blueprint_pipeline import task_evaluation_artifixer_ai_visual_review as review
    from tests.test_task_evaluation_artifixer_ai_visual_review import _inputs

    final_path, execution_path = _inputs(tmp_path)
    final = json.loads(final_path.read_text())
    final["review_phase"] = "post_training"
    sealed(final_path, final)
    payload, _, _, _ = review.build_artifixer_ai_visual_review_input(
        final_composite_receipt_path=final_path
    )
    execution = json.loads(execution_path.read_text())
    execution.update(
        {
            "review_phase": "post_training",
            "review_policy": review.FINAL_REVIEW_POLICY,
            "review_prompt_sha256": "sha256:" + hashlib.sha256(review._PROMPT.encode()).hexdigest(),
            "input_digest": canonical_digest({"input": payload}),
            "final_composite_receipt_digest": final["receipt_digest"],
            "usage": {"provider_response_id": "fixture-response-only"},
        }
    )
    source = tmp_path / "review-source"
    (source / "review-live").mkdir(parents=True)
    (source / "review_input.json").write_bytes(final_path.read_bytes())
    sealed(
        source / "review-live" / (review.EXECUTION_SCHEMA_VERSION + ".json"),
        execution,
        "execution_digest",
    )
    ref = reuse.stage_completed_review(source_root=source, output_root=tmp_path / "staged-review")
    final["post_training_binding_digest"] = "sha256:" + "a" * 64
    sealed(final_path, final)
    output = tmp_path / "review-output"
    output.mkdir()
    result = reuse.reuse_completed_review(
        reference=ref,
        current_input_path=final_path,
        output_root=output,
        publisher_instance_id="104",
        minimum_frame_count=2,
    )
    assert result["review"]["decision"] == "accepted"
    assert result["review"]["new_model_call_performed"] is False
    rebound = json.loads(Path(result["review"]["execution_receipt"]["path"]).read_text())
    assert rebound["source_execution_digest"] == execution["execution_digest"]
    assert rebound["final_composite_receipt_digest"] == final["receipt_digest"]
    Path(final["tasks"][0]["frames"][0]["final_frame"]["path"]).write_bytes(b"changed-image")
    row = final["tasks"][0]["frames"][0]["final_frame"]
    row["size_bytes"] = len(b"changed-image")
    row["sha256"] = "sha256:" + hashlib.sha256(b"changed-image").hexdigest()
    sealed(final_path, final)
    with pytest.raises(ValueError, match="review_inputs_changed"):
        reuse.reuse_completed_review(
            reference=ref,
            current_input_path=final_path,
            output_root=output,
            publisher_instance_id="104",
            minimum_frame_count=2,
        )


def test_reused_training_round_bypasses_runtime_and_keeps_complete_frame_records(tmp_path):
    from blueprint_pipeline import task_evaluation_scene_configuration_artifixer_driver as driver

    args = setup_source(tmp_path)
    reference = reuse.stage_completed_training(**args)
    result = driver._run_artifixer_training_round(
        round_root=tmp_path / "round",
        teacher_receipt_path=Path(args["prepared"]["teacher_receipt_path"]),
        candidate=args["prepared"]["candidate"],
        candidate_path=tmp_path / "unused",
        package_root=tmp_path,
        stage_input=args["stage_input"],
        tuning=args["tuning"],
        configuration={},
        environment={},
        runner=lambda *a, **k: pytest.fail("must not execute training"),
        semantic_token="",
        source_semantic_checkpoint={},
        post_training_checkpoint_root=None,
        post_training_checkpoint_output=tmp_path / "unused-checkpoint",
        completed_training_reuse=reference,
    )
    assert result["completed_training_reused"] is True
    assert len(result["review_frames"]) == 8
    for frame in result["review_frames"]:
        record = frame["final_frame"]
        assert record["size_bytes"] == Path(record["path"]).stat().st_size
