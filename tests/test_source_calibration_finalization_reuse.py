"""A failed CPU mask boundary reuses exact closed pixels without another GPU."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from blueprint_pipeline import public_scene_inpainting_inputs as inputs
from blueprint_pipeline import public_scene_inpainting_preparation as preparation
from blueprint_pipeline import sam31_source_calibration_stage as stage
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.source_calibration_render_return import record
from tests.test_source_calibration_finalization_reentry import _closed_job, _no_allocation


def _seal(path, value, field):
    value[field] = canonical_digest(value, digest_field=field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value


def _case(tmp_path, monkeypatch):
    from tests import test_public_scene_inpainting_preparation as fixture
    original = fixture._write_v2_fixture
    def lost_policy(root, **kwargs):
        paths = original(root, **kwargs)
        request_path = paths["repo"] / "request.json"
        request = json.loads(request_path.read_text())
        request.pop("request_digest", None)
        request["mask_policy"].pop("maximum_image_fraction", None)
        request["mask_policy"]["dilation_pixels"] = 0
        request_path.write_text(json.dumps(inputs.build_public_scene_inpainting_input_request(request)))
        return paths
    monkeypatch.setattr(fixture, "_write_v2_fixture", lost_policy)
    old_job, old, original_root = _closed_job(tmp_path, monkeypatch)
    old_raw = Path(old["preparation_path"]).read_bytes()
    intent_path = tmp_path / "intent.json"
    intent = {"intent_id": "scene-fixture", "request": {"scope": "hermetic"}}
    intent["intent_digest"] = cross_runtime_canonical_digest(intent, digest_field="intent_digest")
    intent_path.write_text(json.dumps(intent))
    task_path = Path(old_job["plan"]["host_inputs"]["task_request"]["path"])
    task = json.loads(task_path.read_text())
    task["scene_intent_authority"] = {"intent": record(intent_path), "intent_digest": intent["intent_digest"]}
    task_path.write_text(json.dumps(task))
    plan = {"source_commit": old["repository"]["commit"],
        "host_inputs": {"task_request": record(task_path)},
        "task_identity": {"id": "fixture-task"}, "scene_identity": {"id": "fixture-scene"},
        "publisher_scene_id": "fixture", "claim_boundary": {"evaluation_authorized": False},
        "rendering": old["render_options"], "camera_policy": old["context"]["request"]["camera_policy"],
        "mask_policy": {"authority": "publisher_target_obb_plus_contained_gaussians",
            "minimum_contained_gaussians": 16, "dilation_pixels": 8, "maximum_image_fraction": .85,
            "visual_contribution_threshold_8bit": 8, "minimum_visible_target_fraction": .01}}
    plan_path = tmp_path / "original-plan.json"
    plan = _seal(plan_path, plan, "plan_digest")
    queue = tmp_path / "queue"
    child = "sam31-original"
    parent = "sha256:" + "a" * 64
    parent_queue = tmp_path / "owned-parent-queue"
    input_root = tmp_path / "owned-parent-inputs"
    (parent_queue / "blocked").mkdir(parents=True)
    input_root.mkdir()
    (parent_queue / "blocked" / ("fixture-parent-" + parent[7:] + ".json")).write_text("{}")
    monkeypatch.delenv("BLUEPRINT_TASK_EVALUATION_SCENE_PROGRESSION_CONFIG", raising=False)
    job_path = queue / "failed" / (child + ".json")
    saved_job = {"child_id": child, "parent_request_digest": parent, "parent_preparation_id": "fixture-parent",
        "phase": "calibrated_views",
        "expected_source_commit": old["repository"]["commit"], "plan_ref": record(plan_path)}
    saved_job = _seal(job_path, saved_job, "job_digest")
    result_path = queue / "results" / (child + ".json")
    _seal(result_path, {"child_id": child, "status": "failed", "job_digest": saved_job["job_digest"],
        "blocker": "edit_input_mask_invalid:source-00"}, "result_digest")
    # Only retained parent admission and source-owner packaging are fixture
    # edges. Real prepared-input, render, closure, frame, mask and reuse gates run.
    from blueprint_pipeline import task_evaluation_sam31_job_admission as admission
    from blueprint_pipeline import task_evaluation_sam31_prefix_evidence as science
    def validated(job, **kwargs):
        assert kwargs["validation_purpose"] == "retained_offline_replay"
        assert kwargs["parent_queue"] == parent_queue
        assert kwargs["input_root"] == input_root
        return {}, json.loads(Path(job["plan_ref"]["path"]).read_text())
    monkeypatch.setattr(admission, "_validated_job", validated)
    monkeypatch.setattr(science, "source_science", lambda host, commit: ({"task": "same"}, {}, {"source": "same"}))
    execution_root = tmp_path / "executions"
    old_output = execution_root / parent[7:] / child / "artifacts"
    old_output.parent.mkdir(parents=True)
    original_root.rename(old_output)
    # The fixture execution checkpoint moved with its folder; retain an exact
    # checkpoint copy at the original path referenced by its closed receipt.
    shutil.copytree(old_output, original_root)
    closed = json.loads((old_output / "source_calibration_closed_return.v1.json").read_text())
    closed["execution_closure"]["provider_execution"] = record(old_output / "allocator_result.json")
    _seal(old_output / "source_calibration_closed_return.v1.json", closed, "return_digest")
    current_request = deepcopy(old["context"]["request"])
    current_request.pop("request_digest")
    current_request["mask_policy"].update(plan["mask_policy"])
    current_root = Path(old["context"]["paths"]["data"]) / "successor"
    current_root.mkdir()
    current_request_path = current_root / "corrected-request.json"
    current_request_path.write_text(json.dumps(inputs.build_public_scene_inpainting_input_request(current_request)))
    current_repo = tmp_path / "corrected-release"
    shutil.copytree(old["context"]["paths"]["repo"], current_repo)
    (current_repo / "cpu-policy-fix.txt").write_text("different CPU release; identical renderer bytes")
    subprocess.run(["git", "-C", str(current_repo), "add", "cpu-policy-fix.txt"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(current_repo), "commit", "-qm", "fixture CPU policy forwarding fix"],
                   check=True, capture_output=True)
    current = inputs.prepare_public_scene_inpainting_inputs(request_path=current_request_path,
        repo_root=current_repo, data_root=old["context"]["paths"]["data"],
        output_root=current_root / "views")
    current_plan_path = tmp_path / "current-plan.json"
    current_plan = _seal(current_plan_path, {**plan, "source_commit": current["repository"]["commit"]}, "plan_digest")
    current_job = {**old_job, "child_id": "sam31-successor", "queue_root": str(queue),
        "parent_queue_root": str(parent_queue), "preparation_input_root": str(input_root),
        "retained_execution_root": str(execution_root), "plan": current_plan, "plan_ref": record(current_plan_path),
        "expected_source_commit": current["repository"]["commit"], "repo_root": str(current_repo),
        "output_root": str(current_root), "resume_only": True}
    current_job["server_profile"]["approved_paid_input_roots"] = [str(tmp_path)]
    prepared_outcome = {"prepared_inputs": record(Path(current["preparation_path"])),
                        "calibrated_view_request": current["request_file"]}
    (current_root / "cpu_preparation_outcome.json").write_text(json.dumps(prepared_outcome))
    return current_job, old, current, old_raw, job_path, result_path, old_output


def test_production_successor_reuses_closed_return_and_preserves_failed_history(tmp_path, monkeypatch):
    job, old, current, old_raw, failed, result, old_output = _case(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in (failed, result, old_output / "allocator_result.json")}
    with pytest.raises(ValueError, match="edit_input_mask_invalid"):
        preparation.finalize_public_scene_inpainting_inputs(preparation_path=old["preparation_path"],
            returned_group_path=old_output / "source_calibration_closed_return.v1.json")
    outcome = stage.execute_source_calibration_stage(job, allocator_runner=_no_allocation)
    assert outcome["status"] == "completed" and outcome["provider_mutation_performed"] is False
    assert outcome["retained_gpu_render_reused"] is True
    assert not (Path(job["output_root"]) / "allocator_started.json").exists()
    assert Path(old["preparation_path"]).read_bytes() == old_raw
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert not (Path(old["preparation_path"]).parent / "public_scene_interiorgs_edit_input_receipt.v2.json").exists()
    receipt = json.loads(Path(outcome["artifacts"]["calibrated_view_receipt"]["path"]).read_text())
    assert receipt["mask_policy"]["maximum_image_fraction"] == .85
    assert receipt["mask_policy"]["dilation_pixels"] == 8
    assert receipt["repository"]["commit"] != old["repository"]["commit"]
    assert receipt["source_calibration_render"]["original_render_commit"] == old["repository"]["commit"]
    binding = outcome["artifacts"]["source_calibration_retained_render_binding"]["path"]
    binding_value = json.loads(Path(binding).read_text())
    assert binding_value["parent_queue_root"] == job["parent_queue_root"]
    assert binding_value["input_root"] == job["preparation_input_root"]
    assert preparation.adopt_finalized_public_scene_inpainting_inputs(preparation_path=current["preparation_path"],
        returned_group_path=old_output / "source_calibration_closed_return.v1.json", retained_render_binding_path=binding) == receipt


@pytest.mark.parametrize("defect", ["plan_mask", "frame", "ambiguous", "current_request"])
def test_reuse_refuses_changed_frozen_policy_frames_or_ambiguous_history(tmp_path, monkeypatch, defect):
    job, old, current, _, failed, _, old_output = _case(tmp_path, monkeypatch)
    if defect == "plan_mask":
        job["plan"] = deepcopy(job["plan"])
        job["plan"]["mask_policy"]["maximum_image_fraction"] = .86
    elif defect == "frame":
        closed = json.loads((old_output / "source_calibration_closed_return.v1.json").read_text())
        manifest = Path(closed["render_groups"]["images"]["path"])
        (manifest.parent / "frames/source-00.png").write_bytes(b"changed")
    elif defect == "ambiguous":
        # A second REUSABLE same-intent closure (its own sealed mask-failure result) is ambiguous.
        other = json.loads(failed.read_text())
        other["child_id"] = "sam31-other"
        _seal(failed.parent / "sam31-other.json", other, "job_digest")
        other_job = json.loads((failed.parent / "sam31-other.json").read_text())
        saved = json.loads((failed.parent.parent / "results" / failed.name).read_text())
        _seal(failed.parent.parent / "results" / "sam31-other.json",
              {**{k: v for k, v in saved.items() if k != "result_digest"},
               "child_id": "sam31-other", "job_digest": other_job["job_digest"]}, "result_digest")
    else:
        Path(current["request_file"]["path"]).write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        stage.execute_source_calibration_stage(job, allocator_runner=_no_allocation)
    assert not (Path(current["preparation_path"]).parent / "public_scene_interiorgs_edit_input_receipt.v2.json").exists()


def test_unreadable_foreign_failed_history_is_skipped_not_fatal(tmp_path, monkeypatch):
    """Scene 840938, 2026-09-12: a retired scene's failed calibrated-views job whose plan file was
    no longer retained failed every fresh child of the new intent before any rental."""
    job, old, current, old_raw, failed, result, old_output = _case(tmp_path, monkeypatch)
    queue = Path(job["queue_root"])
    gone = {"path": str(tmp_path / "retired-scene" / "plan.json"), "sha256": "sha256:" + "0" * 64, "size_bytes": 1}
    (queue / "failed" / "sam31-foreign-plan-gone.json").write_text(json.dumps(
        {"phase": "calibrated_views", "child_id": "sam31-foreign-plan-gone", "plan_ref": gone}))
    (queue / "failed" / "sam31-foreign-not-json.json").write_text("not json")
    (queue / "failed" / "sam31-other-phase.json").write_text(json.dumps({"phase": "sam31_tracking", "child_id": "x"}))
    outcome = stage.execute_source_calibration_stage(job, allocator_runner=_no_allocation)
    assert outcome["status"] == "completed" and outcome["retained_gpu_render_reused"] is True
    assert failed.exists() and result.exists()  # this intent's own history untouched


def test_own_failure_without_closed_return_is_not_a_candidate(tmp_path, monkeypatch):
    """Scene 840938, 2026-09-12 19:08: the second fresh child found this intent's first failed
    child (failed before allocation) as a candidate and refused the whole stage with
    calibration_reuse_cpu_mask_failure_required instead of rendering fresh."""
    from blueprint_pipeline.source_calibration_finalization_reuse import select_retained_render
    job, old, current, old_raw, failed, result, old_output = _case(tmp_path, monkeypatch)
    queue = Path(job["queue_root"])
    # A sibling same-intent failure that never allocated (no closed return, no mask blocker).
    sibling = json.loads(failed.read_text())
    sibling_id = "sam31-own-pre-allocation-failure"
    sibling_path = queue / "failed" / (sibling_id + ".json")
    _seal(sibling_path, {**{k: v for k, v in sibling.items() if k != "job_digest"}, "child_id": sibling_id}, "job_digest")
    sibling_job = json.loads(sibling_path.read_text())
    _seal(queue / "results" / (sibling_id + ".json"), {"child_id": sibling_id, "status": "failed",
        "job_digest": sibling_job["job_digest"], "blocker": "scene_configuration_submission_input_file_invalid"},
        "result_digest")
    outcome = stage.execute_source_calibration_stage(job, allocator_runner=_no_allocation)
    assert outcome["status"] == "completed" and outcome["retained_gpu_render_reused"] is True
    # With only the non-reusable failure in history, selection yields no candidate at all.
    failed.unlink()
    result.unlink()
    prepared_path = Path(json.loads((Path(job["output_root"]) / "cpu_preparation_outcome.json").read_text())["prepared_inputs"]["path"])
    assert select_retained_render(job=job, prepared_path=prepared_path, output_root=tmp_path / "fresh-out") is None

