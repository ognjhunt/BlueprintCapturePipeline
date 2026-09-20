"""Bind captured frame derivatives and prepared backgrounds before native spend.

ADP-030/040, day 28. Website preparation already removed the selected object
before reconstruction. Its authoring inputs are captured frame derivatives,
not renders of an object-present reconstructed scene.
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .website_native_background import (
    PREFIX, ADAPTER_ID, APPEARANCE_ADAPTER_ID, _runtime,
    appearance_configuration_refusal, collision_configuration_refusal,
)
from .website_object_observations import SCHEMA as OBSERVATIONS_SCHEMA
from .task_evaluation_scene_configuration_disclosure import WEBSITE_INPUT_STATUS as INPUT_STATUS



def _require(condition, code):
    if not condition:
        raise ValueError("website_native_inputs_" + code)


def validate_website_native_inputs(*, envelope, configurations, require_render_inputs):
    from .task_evaluation_scene_configuration_source_preflight import _reference

    request = envelope.get("request", {})
    recipe = envelope.get("recipe", {})
    stages = recipe.get("stage_sequence", [])
    _require(len(stages) == 6, "six_stages_required")
    first, second, third = (configurations[s["stage_id"]] for s in stages[:3])
    _require(stages[0]["adapter"]["id"] == APPEARANCE_ADAPTER_ID
             and stages[1]["adapter"]["id"] == ADAPTER_ID
             and appearance_configuration_refusal(first, envelope) is None
             and collision_configuration_refusal(second, envelope) is None
             and first["runtime_inputs_digest"] == second["runtime_inputs_digest"], "configuration_binding_invalid")

    def reference(contract, expected=None):
        row, path = _reference(envelope, contract)
        # Every transported derivative is part of the authenticated request.
        bound = request
        for part in contract.split("."):
            bound = bound[int(part)] if isinstance(bound, list) and part.isdigit() else bound.get(part, {})
        _require(all(bound.get(k) == row.get(k) for k in ("uri", "digest", "size_bytes")), "request_binding_invalid")
        if expected:
            _require(all(row[k] == expected[k] for k in ("digest", "size_bytes")), "artifact_binding_invalid")
        return row, path

    runtime_row, runtime_path = reference(PREFIX + ".runtime_inputs")
    runtime = _runtime(runtime_path)
    authoring = runtime["object_authoring"]
    _require(runtime_row["digest"] == first["runtime_inputs_digest"]
             and third == authoring["configuration"]
             and recipe["subject_identity"] == third["replacement_identity"], "object_configuration_changed")
    manifest_row, manifest_path = reference("scene.source_manifest")
    preparation = json.loads(manifest_path.read_text())
    _require(preparation.get("digest") == canonical_digest(preparation, digest_field="digest")
             and preparation["digest"] == runtime["preparation_digest"]
             and recipe.get("source_manifest_digest") == manifest_row["digest"]
             and "website_scene_processing_rights_required" not in preparation.get("blockers", []),
             "preparation_binding_invalid")
    reference("scene.geometry.collision", runtime["collision"])
    _, normalization_path = reference("scene.geometry.validation")
    normalization = json.loads(normalization_path.read_text())
    _require(normalization.get("normalization_digest") == canonical_digest(normalization, digest_field="normalization_digest")
             and normalization.get("source_digest") == preparation["binding"]["collision_mesh_digest"]
             and normalization["output"]["sha256"] == runtime["collision"]["digest"], "collision_normalization_invalid")
    appearance = runtime["appearance"]
    reference(PREFIX + ".appearance", appearance)
    receipt = appearance.get("receipt", {})
    _require(appearance.get("status") == "native_appearance_authored"
             and receipt.get("digest") == canonical_digest(receipt, digest_field="digest")
             and receipt.get("binding", {}).get("preparation_digest") == preparation["digest"]
             and receipt.get("binding", {}).get("source_digest") == preparation["binding"]["splat_digest"]
             and receipt.get("renderer_qualified") is False
             and receipt.get("physical_measurement_proven") is False, "appearance_binding_invalid")
    observed_row, observed_path = reference(PREFIX + ".observations", authoring["observation_manifest"])
    observed = json.loads(observed_path.read_text())
    _require(observed.get("schema_version") == OBSERVATIONS_SCHEMA
             and observed.get("digest") == canonical_digest(observed, digest_field="digest")
             and observed.get("preparation_digest") == preparation["digest"]
             and observed.get("source_geometry_digest") == preparation["binding"]["source_geometry_digest"]
             and observed.get("target_id") == third["source_object_identity"]
             and observed.get("complete_object_geometry") is False
             and observed.get("physical_measurement_proven") is False, "observations_binding_invalid")
    reference(PREFIX + ".candidate", observed["candidate"])
    _require(all(observed["candidate"][k] == authoring["source_candidate"][k] for k in ("digest", "size_bytes")),
             "candidate_binding_invalid")
    frame_rows = observed.get("frames", [])
    supplied = request["scene"]["website_native_inputs"]["frames"]
    _require(frame_rows and len(frame_rows) == len(supplied)
             and len({row["frame_id"] for row in frame_rows}) == len(frame_rows), "frames_invalid")
    for index, row in enumerate(frame_rows):
        reference(PREFIX + f".frames.{index}", row["image"])
    rights = request["scene"].get("rights", {})
    _require(rights.get("provider_disclosure_scope") == "derived_only", "disclosure_scope_invalid")
    rights_row, rights_path = reference("scene.rights.admission")
    admission = json.loads(rights_path.read_text())
    disclosure = admission.get("provider_disclosure", {})
    _require(recipe.get("rights_admission_digest") == rights_row["digest"]
             and admission.get("schema_version") == "website_native_rights_admission.v1"
             and admission.get("digest") == canonical_digest(admission, digest_field="digest")
             and admission.get("preparation_digest") == preparation["digest"]
             and admission.get("task_context_digest") == preparation["binding"]["task_context_digest"]
             and admission.get("owner") == preparation["intake_request"]["owner"]
             and admission.get("consent") == preparation["intake_request"]["consent"]
             and admission.get("execution_authority") == preparation["intake_request"]["execution"]
             and disclosure.get("captured_frame_derivatives_allowed") is True
             and disclosure.get("prepared_background_allowed") is True
             and disclosure.get("provider_training_allowed") is False
             and disclosure.get("public_redistribution_allowed") is False,
             "capture_derivative_disclosure_not_admitted")
    binding = {"runtime_inputs_digest": runtime_row["digest"], "source_manifest_digest": manifest_row["digest"],
               "observation_manifest_digest": observed_row["digest"], "rights_admission_digest": rights_row["digest"],
               "captured_frame_count": len(frame_rows)}
    if require_render_inputs:
        render = envelope.get("render_inputs_result", {})
        _require(render.get("status") == INPUT_STATUS and render.get("website_binding") == binding
                 and render.get("result_digest") == canonical_digest(render, digest_field="result_digest"),
                 "method_input_binding_invalid")
    return binding


def materialize_website_inputs(*, envelope, stage_one_configuration, output_root):
    from .task_evaluation_scene_configuration_builtin_adapters import _materialized_reference

    _, runtime_path = _materialized_reference(envelope, contract_path=PREFIX + ".runtime_inputs")
    runtime = _runtime(runtime_path)
    # The caller validates all six immutable configurations before bundling.
    # Use the same bound first/second/third inputs at method-input preparation.
    configurations = {}
    for index, stage in enumerate(envelope["recipe"]["stage_sequence"]):
        contract = f"construction.recipe.stage_sequence.{index}.configuration"
        row, path = _materialized_reference({"materialized_references": envelope["stage_configuration_references"]},
                                            contract_path=contract)
        _require(all(row[k] == stage["configuration"][k] for k in ("uri", "digest", "size_bytes")),
                 "configuration_reference_changed")
        configurations[stage["stage_id"]] = json.loads(path.read_text())
    _require(configurations[envelope["recipe"]["stage_sequence"][0]["stage_id"]] == stage_one_configuration,
             "stage_one_changed")
    binding = validate_website_native_inputs(envelope=envelope, configurations=configurations, require_render_inputs=False)
    result = {"schema_version": "task_evaluation_scene_configuration_render_inputs.v1", "status": INPUT_STATUS,
              "run_id": envelope["request"]["run_id"], "input_kind": "website_capture_derivatives", "website_binding": binding,
              "source_appearance_digest": runtime["appearance"]["digest"],
              "raw_interiorgs_bytes_in_provider_packet": False, "raw_capture_video_in_provider_packet": False,
              "captured_frame_derivatives_in_provider_packet": True, "prepared_background_in_provider_packet": True,
              "derived_frames": [], "derived_frame_count": 0, "renderer_qualified": False,
              "physical_truth_claimed": False, "provider_render_required": False,
              "provider_mutation_performed": False, "paid_execution_requested": False}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    write_json(Path(output_root) / "task_evaluation_scene_configuration_render_inputs.v1.json", result)
    return result
