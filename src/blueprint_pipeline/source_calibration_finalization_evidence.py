"""Read-only proof of a closed calibration render and corrected CPU mask inputs."""
from __future__ import annotations

from pathlib import Path

from .source_calibration_render_return import (
    ROLES, record, require_source_calibration_closure, verify_source_calibration_return,
)
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, require, sha

SCHEMA = "source_calibration_retained_finalization_binding.v1"
FILENAME = SCHEMA + ".json"
RENDERER_FILES = ("tools/splat_render/render_splat.mjs", "tools/splat_render/src/render_entry.mjs",
                  "tools/splat_render/package.json", "tools/splat_render/package-lock.json",
                  "scripts/run_adp_retained_scene_render_provider_runtime.sh",
                  "scripts/adp_retained_scene_render_provider_runner.mjs",
                  "scripts/source_calibration_camera_recovery.mjs")


def _ref(value):
    return checked_file(value["path"], value)


def _intent(plan):
    task = read(_ref(plan["host_inputs"]["task_request"]))
    authority = task.get("scene_intent_authority", {})
    intent = read(_ref(authority["intent"]))
    from .decision_evidence_contracts import cross_runtime_canonical_digest
    require(intent.get("intent_digest") == authority.get("intent_digest")
            == cross_runtime_canonical_digest(intent, digest_field="intent_digest"),
            "calibration_reuse_owner_intent_invalid")
    return authority["intent_digest"]


def _original(job_path, *, queue_root, execution_root, approved_roots, parent_queue_root, input_root):
    from .task_evaluation_sam31_job_admission import _validated_job
    job = read(job_path, digest_field="job_digest")
    queue = Path(queue_root)
    require(job.get("phase") == "calibrated_views" and job_path == queue / "failed" / (job["child_id"] + ".json"),
            "calibration_reuse_failed_child_required")
    require(not any((queue / state / job_path.name).exists()
                    for state in ("pending", "processing", "waiting_external", "completed")),
            "calibration_reuse_child_ownership_ambiguous")
    result_path = queue / "results" / job_path.name
    result = read(result_path, digest_field="result_digest")
    require(result.get("status") == "failed" and result.get("job_digest") == job["job_digest"]
            and result.get("child_id") == job["child_id"]
            and str(result.get("blocker", "")).startswith("edit_input_mask_invalid:"),
            "calibration_reuse_cpu_mask_failure_required")
    _, plan = _validated_job(job, parent_queue=Path(parent_queue_root),
        input_root=Path(input_root), source_commit=job["expected_source_commit"],
        approved_roots=approved_roots, validation_purpose="retained_offline_replay")
    root = Path(execution_root) / job["parent_request_digest"][7:] / job["child_id"] / "artifacts"
    cpu = read(root / "cpu_preparation_outcome.json")
    from .public_scene_inpainting_preparation import validate_prepared_inputs
    prepared = validate_prepared_inputs(_ref(cpu["prepared_inputs"]))
    closed = root / "source_calibration_closed_return.v1.json"
    groups = verify_source_calibration_return(prepared, closed)
    closure = require_source_calibration_closure(prepared, closed)
    require(closure["execution_closure"]["provider_execution"] == record(root / "allocator_result.json"),
            "calibration_reuse_execution_checkpoint_changed")
    require(prepared["request_file"] == cpu["calibrated_view_request"],
            "calibration_reuse_request_checkpoint_changed")
    return job, plan, prepared, closed, groups, result_path


def _same_render(old, current, old_plan, current_plan):
    from .task_evaluation_sam31_prefix_evidence import camera_science, source_science, task_science
    for key in ("task_identity", "scene_identity", "publisher_scene_id", "rendering", "mask_policy", "claim_boundary"):
        require(old_plan.get(key) == current_plan.get(key), "calibration_reuse_frozen_plan_changed:" + key)
    require(camera_science(old_plan["camera_policy"]) == camera_science(current_plan["camera_policy"]),
            "calibration_reuse_frozen_camera_changed")
    require(_intent(old_plan) == _intent(current_plan), "calibration_reuse_owner_intent_changed")
    old_task, _, old_source = source_science(old_plan["host_inputs"], old_plan["source_commit"])
    current_task, _, current_source = source_science(current_plan["host_inputs"], current_plan["source_commit"])
    require(task_science(old_task) == task_science(current_task) and old_source == current_source,
            "calibration_reuse_source_or_task_changed")
    old_mask = old["context"]["request"]["mask_policy"]
    expected_mask = {**old_mask, **old_plan["mask_policy"]}
    require(old_mask != expected_mask and current["context"]["request"]["mask_policy"] == expected_mask,
            "calibration_reuse_frozen_mask_policy_required")
    # Camera recovery ran on these thresholds; no after-the-fact camera reselection.
    from .source_calibration_camera_resolution import visibility_gate
    require(visibility_gate(old) == visibility_gate(current), "calibration_reuse_visibility_gate_changed")
    for key in ("cameras", "render_options"):
        require(old[key] == current[key], "calibration_reuse_render_inputs_changed:" + key)
    require(old["camera_file"]["sha256"] == current["camera_file"]["sha256"],
            "calibration_reuse_camera_bytes_changed")
    for role in ROLES:
        require(all(old["layers"][role][key] == current["layers"][role][key]
                    for key in ("sha256", "size_bytes", "retained_gaussian_count")),
                "calibration_reuse_source_layer_changed:" + role)
    for key in ("corners", "target_center", "target_count", "scene_gaussian_count", "source_adapter"):
        require(old["context"][key] == current["context"][key], "calibration_reuse_geometry_changed:" + key)
    before, after = (Path(value["context"]["paths"]["repo"]) for value in (old, current))
    require(all(sha(before / name) == sha(after / name) for name in RENDERER_FILES),
            "calibration_reuse_renderer_changed")


def validate_retained_render_binding(path, *, current_prepared):
    value = read(path, digest_field="binding_digest")
    require(value.get("schema_version") == SCHEMA and value.get("status") == "closed_render_reused"
            and value.get("provider_mutation_performed") is False and value.get("historical_records_modified") is False,
            "calibration_reuse_binding_invalid")
    require(record(Path(current_prepared["preparation_path"])) == value["current_prepared_inputs"],
            "calibration_reuse_current_preparation_changed")
    job, old_plan, original, closed, groups, result_path = _original(_ref(value["original_failed_job"]),
        queue_root=value["queue_root"], execution_root=value["execution_root"],
        approved_roots=tuple(Path(root) for root in value["approved_roots"]),
        parent_queue_root=value["parent_queue_root"], input_root=value["input_root"])
    current_plan = read(_ref(value["current_plan"]), digest_field="plan_digest")
    require(record(closed) == value["original_closed_return"]
            and record(result_path) == value["original_failed_result"]
            and record(Path(original["preparation_path"])) == value["original_prepared_inputs"],
            "calibration_reuse_original_evidence_changed")
    _same_render(original, current_prepared, old_plan, current_plan)
    require(value["intent_digest"] == _intent(current_plan), "calibration_reuse_binding_intent_changed")
    return original, groups, require_source_calibration_closure(original, closed), value

