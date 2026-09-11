"""No-cost validation of completed Astra CAD, Blender, and review artifacts."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_object_astra_authoring import (
    AppearanceReview, AssetAuthoringError, file_record, validate_geometry_readback,
)


def verified(record):
    path = Path(record["path"])
    actual = file_record(path)
    if any(actual[key] != record.get(key) for key in ("sha256", "size_bytes")):
        raise AssetAuthoringError("astra_retained_artifact_changed")
    return path


def completed_cad(prior_root: Path, request) -> tuple[dict, dict]:
    path = prior_root / "cad_result.json"
    result = json.loads(path.read_text())
    candidate = (verified(result["retained_candidate_receipt"]) if "retained_candidate_receipt" in result
                 else prior_root / "cad/candidate-receipt.json")
    receipt = json.loads(candidate.read_text())
    step, stl = (verified(result[name]) for name in ("step", "stl"))
    readback = result.get("readback") or {}
    expected = [value * 1000 for value in request.dimensions_m]
    measured = readback.get("measured_dimensions_mm")
    tolerance = request.maximum_export_error_m * 1000
    if (result.get("passed") is not True or result.get("source_unchanged_after") is not True
            or receipt.get("passed") is not True or receipt.get("source_unchanged_after") is not True
            or receipt.get("readback") != readback or readback.get("passed") is not True
            or readback.get("valid") is not True or readback.get("solid_count") != 1
            or readback.get("expected_dimensions_mm") != expected
            or readback.get("absolute_tolerance_mm") != tolerance
            or not isinstance(measured, list) or len(measured) != 3
            or any(not math.isfinite(value) or abs(value - expected[index]) > tolerance for index, value in enumerate(measured))
            or not math.isfinite(readback.get("volume_mm3", float("nan"))) or readback["volume_mm3"] <= 0):
        raise AssetAuthoringError("astra_retained_cad_readback_invalid")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise AssetAuthoringError("astra_retained_cad_inventory_missing")
    for relative, digest in artifacts.items():
        artifact = candidate.parent / relative
        if not artifact.resolve().is_relative_to(candidate.parent.resolve()):
            raise AssetAuthoringError("astra_retained_cad_inventory_path_invalid")
        if artifact.is_symlink() or not artifact.is_file():
            raise AssetAuthoringError("astra_retained_cad_inventory_changed")
        with artifact.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed != digest.removeprefix("sha256:"):
            raise AssetAuthoringError("astra_retained_cad_inventory_changed")
    if not all(path.name in artifacts for path in (step, stl)):
        raise AssetAuthoringError("astra_retained_cad_exports_unbound")
    result["retained_candidate_receipt"] = file_record(candidate)
    return result, {"schema_version": "astra_completed_cad_adoption.v1", "source_result": file_record(path),
        "candidate_receipt": file_record(candidate), "readback_digest": canonical_digest(readback),
        "stl_sha256": file_record(stl)["sha256"], "step_sha256": file_record(step)["sha256"],
        "cad_execution_repeated": False}


def completed_blender(prior_root: Path, request, *, round_index: int, program, cad) -> dict:
    attempt = prior_root / f"appearance-{round_index:02d}"
    measurement_path = attempt / "geometry_readback.json"
    measurement = json.loads(measurement_path.read_text())
    physical_input = json.loads((prior_root / "physical_review_input.json").read_text())
    validate_geometry_readback(request, measurement, physical_input["appearance"])
    if (attempt / "asset_program.py").read_text() != program.program:
        raise AssetAuthoringError("astra_retained_blender_program_changed")
    receipt = json.loads((attempt / "final_visual_mesh_receipt.json").read_text())
    records = {name: file_record(attempt / name) for name in (
        "asset_program.py", "geometry_readback.json", "candidate.stl", "candidate.usdc", "candidate.blend",
        "final_visual_mesh.json", "final_visual_mesh_receipt.json", "perspective.png", "top.png", "side.png")}
    if (receipt.get("schema_version") != "final_visual_mesh_receipt.v1"
            or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
            or receipt.get("source_cad_stl_sha256") != cad["stl"]["sha256"]
            or records["candidate.stl"]["sha256"] != cad["stl"]["sha256"]
            or receipt.get("author_program_sha256") != records["asset_program.py"]["sha256"]
            or receipt.get("candidate_usd_sha256") != records["candidate.usdc"]["sha256"]
            or receipt.get("mesh_sha256") != records["final_visual_mesh.json"]["sha256"]):
        raise AssetAuthoringError("astra_retained_blender_receipt_invalid")
    return {"schema_version": "astra_completed_blender_adoption.v1", "round_index": round_index,
        "records": records, "program_digest": canonical_digest(program.model_dump(mode="json")),
        "cad_stl_sha256": cad["stl"]["sha256"], "blender_execution_repeated": False}


def completed_visual_review(prior_root: Path, request_value: dict, budget_root: Path,
                            *, round_index: int, execution: dict) -> tuple[AppearanceReview, dict]:
    path = prior_root / f"appearance-{round_index:02d}/independent_visual_review_{round_index}.json"
    phase = json.loads(path.read_text())
    output = AppearanceReview.model_validate(phase["output"])
    digest = canonical_digest(output.model_dump(mode="json"))
    references = phase.get("references") or []
    source = request_value["source_frames"]
    if (phase.get("request_digest") != request_value["request_digest"] or phase.get("model") != "gpt-6-astra"
            or phase.get("provider") != "openai" or references[:len(source)] != source
            or len(references) != len(source) + 3):
        raise AssetAuthoringError("astra_retained_visual_review_inputs_changed")
    for reference, name in zip(references[len(source):], ("perspective.png", "top.png", "side.png"), strict=True):
        if (reference.get("role") != "prior_candidate" or reference.get("sha256") != execution["records"][name]["sha256"]
                or file_record(Path(reference["path"]))["sha256"] != reference["sha256"]):
            raise AssetAuthoringError("astra_retained_visual_review_images_changed")
    matches = []
    for candidate in (budget_root / "inference_reservations/completed").glob("*.json"):
        row = json.loads(candidate.read_text())
        if (row.get("run_id") == request_value["run_id"] and row.get("model") == "gpt-6-astra" and row.get("provider") == "openai"
                and row.get("capability") == request_value["object_id"] + f"_independent_visual_review_{round_index}"
                and row.get("structured_output_digest") == digest
                and row.get("inference_completion_digest") == canonical_digest(row, digest_field="inference_completion_digest")):
            matches.append(candidate)
    if len(matches) != 1:
        raise AssetAuthoringError("astra_retained_visual_review_completion_missing")
    return output, {"schema_version": "astra_completed_visual_review_adoption.v1", "output_digest": digest,
        "source_phase": file_record(path), "completed_provider_response": file_record(matches[0]),
        "round_index": round_index, "source_frames": source,
        "render_sha256": [execution["records"][name]["sha256"] for name in ("perspective.png", "top.png", "side.png")],
        "new_provider_call": False}


def completed_authoring(prior_root: Path, request_value: dict) -> dict[str, Any]:
    path = prior_root / "result.json"
    result = json.loads(path.read_text())
    if (result.get("schema_version") != "task_object_astra_authoring_result.v1"
            or result.get("status") != "candidate_authored_pending_native_qualification"
            or result.get("request_digest") != request_value["request_digest"]
            or result.get("object_id") != request_value["object_id"] or result.get("model") != "gpt-6-astra"
            or result.get("claim_ceiling") != "development_only"
            or any(result.get(key) is not False for key in ("native_import_qualified", "scene_placement_qualified", "physical_equivalence_proven"))
            or result.get("result_digest") != canonical_digest(result, digest_field="result_digest")):
        raise AssetAuthoringError("astra_retained_authoring_result_invalid")
    def records(value):
        if isinstance(value, dict):
            if {"path", "sha256", "size_bytes"} <= set(value):
                verified(value)
            for item in value.values():
                records(item)
        elif isinstance(value, list):
            for item in value:
                records(item)
    records(result)
    return result
