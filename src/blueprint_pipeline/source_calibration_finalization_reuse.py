"""Reuse closed calibration pixels after a frozen mask-policy propagation defect.

Only CPU finalization is repeated. Original jobs, failures, requests, manifests,
billing and frames stay immutable; a separate binding names the corrected input.
"""
from __future__ import annotations

from pathlib import Path

from .decision_evidence_contracts import canonical_digest, canonical_json
from .source_calibration_finalization_evidence import (
    SCHEMA, FILENAME, RENDERER_FILES as RENDERER_FILES, _ref, _intent, _original, _same_render,
    validate_retained_render_binding as validate_retained_render_binding,
)
from .source_calibration_render_return import record
from .task_evaluation_scene_configuration_submission_inputs import read, require



def _write(path, value):
    raw = canonical_json(value) + "\n"
    if path.exists():
        require(path.read_text() == raw, "calibration_reuse_receipt_conflict")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(raw)



def select_retained_render(*, job, prepared_path, output_root):
    """Find the one failed child of this intent; ambiguity never chooses nicer pixels."""
    from . import task_evaluation_sam31_preparation_execution as execution
    plan = job["plan"]
    if "scene_intent_authority" not in read(_ref(plan["host_inputs"]["task_request"])):
        return None
    intent = _intent(plan)
    queue = Path(job.get("queue_root", execution.DEFAULT_QUEUE))
    execution_root = Path(job.get("retained_execution_root", execution.DEFAULT_EXECUTION_ROOT))
    roots = tuple(Path(root) for root in job["server_profile"].get("approved_paid_input_roots", execution.DEFAULT_APPROVED_ROOTS))
    candidates = []
    for path in sorted((queue / "failed").glob("*.json")):
        previous = read(path)
        if previous.get("phase") != "calibrated_views" or previous.get("child_id") == job.get("child_id"):
            continue
        previous_plan = read(_ref(previous["plan_ref"]), digest_field="plan_digest")
        previous_task = read(_ref(previous_plan["host_inputs"]["task_request"]))
        if previous_task.get("scene_intent_authority", {}).get("intent_digest") == intent:
            require(_intent(previous_plan) == intent, "calibration_reuse_owner_intent_changed")
            candidates.append(path)
    require(len(candidates) <= 1, "calibration_reuse_candidates_ambiguous")
    if not candidates:
        return None
    # Production resolves the operator's actual route once and seals it into
    # the binding. Offline agents pass these same explicit roots without any
    # provider/configuration environment inheritance.
    parent_queue, input_root = execution.configured_parent_route(read(candidates[0]),
        Path(job.get("parent_queue_root", execution.DEFAULT_PARENT_QUEUE)),
        Path(job.get("preparation_input_root", execution.DEFAULT_INPUT_ROOT)))
    return bind_retained_render(original_job_path=candidates[0], job=job,
        prepared_path=prepared_path, output_root=output_root, queue_root=queue,
        execution_root=execution_root, approved_roots=roots,
        parent_queue_root=parent_queue, input_root=input_root)


def bind_retained_render(*, original_job_path, job, prepared_path, output_root,
                         queue_root, execution_root, approved_roots, parent_queue_root, input_root):
    from .public_scene_inpainting_preparation import validate_prepared_inputs
    queue, roots = Path(queue_root), tuple(Path(root) for root in approved_roots)
    plan = job["plan"]
    original_job, _, original, closed, _, result_path = _original(Path(original_job_path), queue_root=queue,
        execution_root=execution_root, approved_roots=roots,
        parent_queue_root=parent_queue_root, input_root=input_root)
    path = Path(output_root) / FILENAME
    prepared = validate_prepared_inputs(prepared_path)
    value = {"schema_version": SCHEMA, "status": "closed_render_reused", "intent_digest": _intent(plan),
        "queue_root": str(queue), "execution_root": str(execution_root), "approved_roots": [str(root) for root in roots],
        "parent_queue_root": str(parent_queue_root), "input_root": str(input_root),
        "original_failed_job": record(Path(original_job_path)), "original_failed_result": record(result_path),
        "original_prepared_inputs": record(Path(original["preparation_path"])),
        "original_closed_return": record(closed), "current_prepared_inputs": record(Path(prepared_path)),
        "current_plan": dict(job["plan_ref"]), "provider_mutation_performed": False,
        "historical_records_modified": False}
    value["binding_digest"] = canonical_digest(value, digest_field="binding_digest")
    # Validate everything before publishing any binding.
    _same_render(original, prepared, read(_ref(original_job["plan_ref"])), plan)
    _write(path, value)
    validate_retained_render_binding(path, current_prepared=prepared)
    return path


def replay_retained_calibration(*, job, plan, job_path, run_root, queue_root, approved_roots,
                                parent_queue_root, input_root, execution_root=None):
    """Run only preparation and finalization against a closed historical return."""
    from . import task_evaluation_sam31_preparation_execution as execution
    from .task_evaluation_sam31_profile_registry import resolve_sam31_profile
    from .task_evaluation_sam31_preparation_cpu_stages import execute_cpu_stage
    from .sam31_source_calibration_stage import finalize_retained_source_calibration
    profile = read(resolve_sam31_profile(plan), digest_field="profile_digest")
    execution_root = Path(execution_root or execution.DEFAULT_EXECUTION_ROOT)
    # Refuse before doing even CPU preparation unless the saved GPU return is
    # closed and its exact parent/child/failure can be reopened read-only.
    _original(Path(job_path), queue_root=queue_root, execution_root=execution_root,
              approved_roots=tuple(Path(root) for root in approved_roots),
              parent_queue_root=parent_queue_root, input_root=input_root)
    context = {**job, "plan": plan, "stage_id": "calibrated_views", "server_profile": profile,
        "output_root": str(Path(run_root) / "cpu"), "repo_root": profile["repo_root"],
        "server_data_root": profile["server_data_root"], "runtime_root": profile["runtime_root"]}
    prepared = execute_cpu_stage(context, prepare_hardware_render=True)
    binding = bind_retained_render(original_job_path=Path(job_path), job=context,
        prepared_path=prepared["prepared_inputs"]["path"], output_root=Path(run_root),
        queue_root=queue_root, execution_root=execution_root, approved_roots=approved_roots,
        parent_queue_root=parent_queue_root, input_root=input_root)
    return finalize_retained_source_calibration(job=context, prepared_outcome=prepared,binding_path=binding)
