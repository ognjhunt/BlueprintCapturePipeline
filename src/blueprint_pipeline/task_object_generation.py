"""Create task objects and variants through the existing Astra/CAD/Blender lane.

ADP-030/040: generated objects remain separate candidates, never replacements
for the captured canonical evidence. All objects share the caller's bounded
invoker; one failed object does not discard successful siblings.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_object_astra_authoring import (
    AuthoringRequest, GeneratedObjectSpecification, execute_asset_authoring,
    file_record, validate_request,
)


def build_generated_object_requests(*, context_request: Mapping[str, Any],
                                    specifications: Sequence[Mapping[str, Any]],
                                    output_root: Path) -> list[AuthoringRequest]:
    context = validate_request(dict(context_request))
    specs = [GeneratedObjectSpecification.model_validate(value) for value in specifications]
    ids = [spec.object_id for spec in specs]
    if not ids or len(ids) != len(set(ids)) or context.object_id in ids:
        raise ValueError("generated_object_identities_must_be_unique")
    if any(spec.variant_of not in {None, context.object_id} for spec in specs):
        raise ValueError("generated_variant_reference_unbound")
    if any(any(value <= 0 for value in spec.dimensions_m) for spec in specs):
        raise ValueError("generated_object_dimensions_invalid")
    output_root.mkdir(parents=True, exist_ok=True)
    requests = []
    for spec in specs:
        specification = {"schema_version": "task_object_generation_specification.v1",
                         "context_request_digest": context.request_digest,
                         "claim_ceiling": "development_only", "captured_object": False,
                         "specification": spec.model_dump(mode="json")}
        path = output_root / (spec.object_id + ".specification.json")
        encoded = canonical_json(specification) + "\n"
        if path.exists() and path.read_text() != encoded:
            raise ValueError("generated_object_specification_changed_use_new_identity")
        path.write_text(encoded)
        record = file_record(path)
        value = context.model_dump(mode="json")
        value.update(object_id=spec.object_id, owner_description=spec.description, role="task_object",
                     dimensions_m=list(spec.dimensions_m), dimension_authority="estimated",
                     dimension_source_digest=record["sha256"],
                     dimension_uncertainty_m=[dimension * .25 for dimension in spec.dimensions_m],
                     generated_specification=spec.model_dump(mode="json"))
        value["source_frames"] = [{**frame.model_dump(mode="json"), "role": "task_context",
            "description": "Task/family context only; this generated object was not observed. " + frame.description}
            for frame in context.source_frames if frame.role in {"observed_source", "task_context", "native_scene"}]
        value["construction_constraints"] = canonical_json({
            "confirmed_task_constraints": context.construction_constraints,
            "generated_specification": spec.model_dump(mode="json"),
            "context_request_digest": context.request_digest,
            "canonical_object_unchanged": True,
            "new_object_is_generated_not_captured": True,
            "native_qualification_required_before_evaluation": True,
        })
        physical = value["physical_review_input"]
        physical.update(object_id=spec.object_id, object_description=spec.description,
                        material_description=spec.material_description, appearance=spec.appearance,
                        measured=dict.fromkeys(("mass_kg", "static_friction", "dynamic_friction", "restitution")),
                        proposed=None)
        physical["optical_material"] = {"name": spec.material_description, "transmission": 0.0, "opacity": 1.0}
        physical["evidence"] = [{"evidence_id": "generated_design", "uri": str(path.resolve()),
            "sha256": record["sha256"][7:], "kind": "primary_reference",
            "excerpt": "Generated design specification for the stated task, not an observed physical object."}]
        physical["dimensions"] = {axis: {"value": dimension, "basis": "estimated",
            "interval": {"lower": dimension * .75, "upper": dimension * 1.25},
            "rationale": "Nominal generated geometry for this scenario.",
            "uncertainty": "Design estimate with a 25% interval; not physical metrology.",
            "evidence_ids": ["generated_design"]}
            for axis, dimension in zip(("x_m", "y_m", "z_m"), spec.dimensions_m, strict=True)}
        # Normalize optional defaults before sealing, just like the ordinary
        # authoring lane. No model call or spending permission is created here.
        value = AuthoringRequest.model_validate(value).model_dump(mode="json")
        value["request_digest"] = canonical_digest(value, digest_field="request_digest")
        request = validate_request(value)
        (output_root / (spec.object_id + ".request.json")).write_text(canonical_json(value) + "\n")
        requests.append(request)
    return requests



def _retained_candidate(root: Path, request: AuthoringRequest) -> dict[str, Any] | None:
    """Adopt only intact, request-bound outputs; never infer success from a folder."""
    results = [root / "result.json", *sorted(root.glob("attempt-*/result.json"))]
    for path in reversed(results):
        if not path.is_file():
            continue
        result = json.loads(path.read_text())
        if (result.get("request_digest") != request.request_digest
                or result.get("object_id") != request.object_id
                or result.get("status") != "candidate_authored_pending_native_qualification"
                or result.get("result_digest") != canonical_digest(result, digest_field="result_digest")):
            raise ValueError("generated_object_retained_result_changed")
        references = []

        def visit(value):
            if isinstance(value, dict):
                if {"path", "sha256", "size_bytes"} <= value.keys():
                    references.append(value)
                else:
                    for nested in value.values():
                        visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        visit(result)
        if not references or not isinstance(result.get("asset"), dict):
            raise ValueError("generated_object_retained_artifacts_missing")
        for record in references:
            if file_record(Path(record["path"])) != {key: record[key] for key in ("path", "sha256", "size_bytes")}:
                raise ValueError("generated_object_retained_artifact_changed")
        for frame in request.source_frames:
            if file_record(Path(frame.path))["sha256"] != frame.sha256:
                raise ValueError("generated_object_context_changed")
        return result
    return None


def execute_generated_object_batch(*, requests: Sequence[AuthoringRequest], output_root: Path,
                                   invoker, mac_executor, blender_runner, blender_executable: str,
                                   executor=execute_asset_authoring) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("generated_object_batch_in_progress") from exc
        return _execute_generated_object_batch(requests=requests, output_root=output_root, invoker=invoker,
            mac_executor=mac_executor, blender_runner=blender_runner, blender_executable=blender_executable,
            executor=executor)

def _execute_generated_object_batch(*, requests: Sequence[AuthoringRequest], output_root: Path,
                                   invoker, mac_executor, blender_runner, blender_executable: str,
                                   executor=execute_asset_authoring) -> dict[str, Any]:
    ids = [request.object_id for request in requests]
    if not ids or len(ids) != len(set(ids)) or any(request.generated_specification is None for request in requests):
        raise ValueError("generated_object_batch_invalid")
    # Validate the entire plan before any of its calls can spend.
    for request in requests:
        validate_request(request.model_dump(mode="json"))
    output_root.mkdir(parents=True, exist_ok=True)
    plan = {request.object_id: request.request_digest for request in requests}
    plan_path = output_root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("generated_object_batch_plan_changed")
    plan_path.write_text(canonical_json(plan) + "\n")
    objects = []
    for request in requests:
        try:
            object_root = output_root / request.object_id
            result = _retained_candidate(object_root, request)
            if result is None:
                object_root.mkdir(exist_ok=True)
                # Failed attempt inputs/receipts remain immutable. A later
                # explicit batch call gets a new root and the same shared cap.
                index = 1
                while (object_root / f"attempt-{index:04d}").exists():
                    index += 1
                attempt = object_root / f"attempt-{index:04d}"
                attempt.mkdir()
                result = executor(request_value=request.model_dump(mode="json"),
                                  output_root=attempt, invoker=invoker,
                                  mac_executor=mac_executor, blender_runner=blender_runner,
                                  blender_executable=blender_executable)
            if (result.get("status") != "candidate_authored_pending_native_qualification"
                    or result.get("object_id") != request.object_id
                    or result.get("request_digest") != request.request_digest):
                raise ValueError("generated_object_result_binding_invalid")
            objects.append({"object_id": request.object_id, "status": "candidate_authored",
                            "request_digest": request.request_digest, "result": result})
        except Exception as exc:
            objects.append({"object_id": request.object_id, "status": "held",
                            "request_digest": request.request_digest, "blocker": str(exc)[:2000]})
        value = {"schema_version": "task_object_generation_batch.v1", "objects": objects,
                 "claim_ceiling": "development_only", "native_qualification_required": True,
                 "evaluation_ready": False}
        value["digest"] = canonical_digest(value, digest_field="digest")
        temporary = output_root / "batch.tmp"
        temporary.write_text(canonical_json(value) + "\n")
        temporary.replace(output_root / "batch.json")
    return value
