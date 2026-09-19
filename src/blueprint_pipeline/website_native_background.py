"""Reuse the website's prepared background in native construction, ADP-030/day 28.

The subject was removed before reconstruction. This stage passes the background
through byte-for-byte and supplies the separate observed object to authoring;
it never pretends to have excised a prim from the reconstructed background.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .website_object_observations import REFERENCE_ROLE, _record, validate_observation_handoff

SCHEMA = "website_prepared_collision.v1"
ADAPTER_ID = "website_prepared_collision"
PREFIX = "scene.website_native_inputs"
APPEARANCE_ADAPTER_ID = "website_prepared_appearance"


def collision_configuration_refusal(configuration: Mapping[str, Any], envelope: Mapping[str, Any]) -> str | None:
    if (configuration.get("schema_version") != SCHEMA
            or configuration.get("operation") != "reuse_background_without_excision"
            or configuration.get("claim_ceiling") != "development_only"):
        return "website_background_operation"
    for name, contract in (("runtime_inputs_digest", PREFIX + ".runtime_inputs"),
                           ("collision_source_digest", "scene.geometry.collision")):
        rows = [r for r in envelope.get("materialized_references", []) if r.get("contract_path") == contract]
        if len(rows) != 1 or rows[0].get("digest") != configuration.get(name):
            return name
    return None


def appearance_configuration_refusal(configuration: Mapping[str, Any], envelope: Mapping[str, Any]) -> str | None:
    if (configuration.get("schema_version") != "website_prepared_appearance.v1"
            or configuration.get("operation") != "reuse_background_without_removal"
            or configuration.get("claim_ceiling") != "development_only"):
        return "website_background_operation"
    for name, contract in (("runtime_inputs_digest", PREFIX + ".runtime_inputs"),
                           ("appearance_source_digest", PREFIX + ".appearance")):
        rows = [r for r in envelope.get("materialized_references", []) if r.get("contract_path") == contract]
        if len(rows) != 1 or rows[0].get("digest") != configuration.get(name):
            return name
    return None


def _runtime(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (value.get("schema_version") != "website_scene_runtime_inputs.v1"
            or value.get("digest") != canonical_digest(value, digest_field="digest")
            or value.get("claim_ceiling") != "development_only"
            or value.get("collision_excision_required") is not False
            or value.get("appearance_removal_required") is not False
            or value.get("subject", {}).get("geometry_origin") != "removed_before_reconstruction"
            or value.get("coordinate_frame") != {"up_axis": "Z", "unit": "estimated_meters",
                                                 "physical_scale_measured": False}):
        raise ValueError("website_native_background_inputs_invalid")
    return value


def prepare_collision_stage(runtime_inputs_path: Path) -> dict[str, Any]:
    """List the immutable files the existing object-store stager must transport."""
    value = _runtime(runtime_inputs_path)
    authoring = value["object_authoring"]
    manifest = Path(authoring["observation_manifest"]["path"])
    _, frames = validate_observation_handoff(manifest, configuration=authoring["configuration"])
    records = {PREFIX + ".runtime_inputs": _record(runtime_inputs_path),
               "scene.geometry.collision": value["collision"],
               PREFIX + ".observations": authoring["observation_manifest"],
               PREFIX + ".candidate": authoring["source_candidate"],
               **{PREFIX + f".frames.{i}": _record(frame) for i, frame in enumerate(frames)}}
    references = []
    for contract, row in records.items():
        checked = _record(Path(row["path"]))
        if any(row[key] != checked[key] for key in ("digest", "size_bytes")):
            raise ValueError("website_native_background_artifact_changed")
        references.append({"contract_path": contract, **checked})
    return {"stage": {"stage_id": "stage-2", "capability": "collision_object_excision",
                       "execution_class": "no_spend", "adapter": {"id": ADAPTER_ID, "version": "v1"}},
            "configuration": {"schema_version": SCHEMA, "operation": "reuse_background_without_excision",
                              "claim_ceiling": "development_only",
                              "runtime_inputs_digest": _record(runtime_inputs_path)["digest"],
                              "collision_source_digest": value["collision"]["digest"]},
            "references": references}


def prepare_appearance_stage(runtime_inputs_path: Path) -> dict[str, Any]:
    value = _runtime(runtime_inputs_path)
    appearance = value["appearance"]
    if appearance.get("status") != "native_appearance_authored":
        raise ValueError("website_native_appearance_pending")
    source = _record(Path(appearance["path"]))
    if any(source[key] != appearance[key] for key in ("digest", "size_bytes")):
        raise ValueError("website_native_appearance_changed")
    runtime = _record(runtime_inputs_path)
    return {"stage": {"stage_id": "stage-1", "capability": "observed_appearance_object_removal",
                       "execution_class": "no_spend", "adapter": {"id": APPEARANCE_ADAPTER_ID, "version": "v1"}},
            "configuration": {"schema_version": "website_prepared_appearance.v1",
                              "operation": "reuse_background_without_removal", "claim_ceiling": "development_only",
                              "runtime_inputs_digest": runtime["digest"], "appearance_source_digest": source["digest"]},
            "references": [{"contract_path": PREFIX + ".runtime_inputs", **runtime},
                           {"contract_path": PREFIX + ".appearance", **source}]}


def execute_prepared_appearance(*, envelope, stage, configuration, configuration_path,
                                dependency_results, output_root, provider_runtime_artifacts=()):
    from .task_evaluation_scene_configuration_builtin_adapters import (
        _copy_artifact, _materialized_reference, _stage_result,
    )
    if (appearance_configuration_refusal(configuration, envelope) or dependency_results
            or stage.get("adapter", {}).get("id") != APPEARANCE_ADAPTER_ID
            or stage.get("execution_class") != "no_spend"):
        raise ValueError("website_prepared_appearance_configuration_invalid")
    _, runtime_path = _materialized_reference(envelope, contract_path=PREFIX + ".runtime_inputs")
    value = _runtime(runtime_path)
    appearance = value["appearance"]
    row, source = _materialized_reference(envelope, contract_path=PREFIX + ".appearance")
    receipt = appearance.get("receipt", {})
    if (appearance.get("status") != "native_appearance_authored"
            or receipt.get("digest") != canonical_digest(receipt, digest_field="digest")
            or receipt.get("binding", {}).get("preparation_digest") != value["preparation_digest"]
            or receipt.get("appearance_removal_performed") is not False
            or receipt.get("renderer_qualified") is not False
            or receipt.get("physical_measurement_proven") is not False
            or envelope.get("recipe", {}).get("subject_identity") != value["object_authoring"]["configuration"]["replacement_identity"]
            or any(row[key] != appearance[key] or row[key] != receipt["artifact"][key]
                   for key in ("digest", "size_bytes"))):
        raise ValueError("website_prepared_appearance_source_mismatch")
    output_root.mkdir(parents=True, exist_ok=True)
    appearance_artifact = _copy_artifact(source, output_root / "background_appearance.usdc")
    receipt_path = output_root / "appearance_authoring.json"
    write_json(receipt_path, receipt)
    return _stage_result(stage=stage, configuration_path=configuration_path, output_artifacts=[
        {"role": "configured_appearance_without_source_object", **appearance_artifact},
        {"role": "website_background_appearance_receipt", **_record(receipt_path)}])


def execute_prepared_collision(*, envelope, stage, configuration, configuration_path,
                               dependency_results, output_root, provider_runtime_artifacts=()):
    from .task_evaluation_scene_configuration_builtin_adapters import (
        _copy_artifact, _materialized_reference, _stage_result,
    )

    if (collision_configuration_refusal(configuration, envelope)
            or stage.get("adapter", {}).get("id") != ADAPTER_ID
            or stage.get("execution_class") != "no_spend"):
        raise ValueError("website_prepared_collision_configuration_invalid")
    _, inputs_path = _materialized_reference(envelope, contract_path=PREFIX + ".runtime_inputs")
    value = _runtime(inputs_path)
    authoring = value["object_authoring"]
    identity = envelope.get("recipe", {}).get("subject_identity")
    if identity != authoring["configuration"]["replacement_identity"]:
        raise ValueError("website_prepared_collision_subject_mismatch")

    def source(contract, expected):
        row, path = _materialized_reference(envelope, contract_path=contract)
        if any(row[key] != expected[key] for key in ("digest", "size_bytes")):
            raise ValueError("website_prepared_collision_source_mismatch")
        return path

    collision = source("scene.geometry.collision", value["collision"])
    manifest = source(PREFIX + ".observations", authoring["observation_manifest"])
    observed = json.loads(manifest.read_text())
    output_root.mkdir(parents=True, exist_ok=True)
    observed_root = output_root / "observations"
    observed_root.mkdir(exist_ok=True)
    # Retain portable manifest bytes. Resolve files through the admitted envelope,
    # never through the control-plane machine's original absolute paths.
    copy_rows = [(PREFIX + ".candidate", observed["candidate"])] + [
        (PREFIX + f".frames.{i}", row["image"]) for i, row in enumerate(observed["frames"])]
    for contract, record in copy_rows:
        name = Path(record["path"])
        if name.is_absolute() or len(name.parts) != 1 or name.name in {"", ".", ".."}:
            raise ValueError("website_prepared_collision_observation_path_invalid")
        _copy_artifact(source(contract, record), observed_root / name)
    observation_artifact = _copy_artifact(manifest, observed_root / "observations.json")
    validate_observation_handoff(Path(observation_artifact["path"]), configuration=authoring["configuration"])
    candidate = _record(observed_root / observed["candidate"]["path"])
    if any(candidate[key] != authoring["source_candidate"][key] for key in ("digest", "size_bytes")):
        raise ValueError("website_prepared_collision_candidate_mismatch")
    background = _copy_artifact(collision, output_root / "background_collision.usda")
    receipt = {"schema_version": "website_prepared_background_result.v1", "status": "prepared_background_reused",
               "runtime_inputs_digest": configuration["runtime_inputs_digest"],
               "preparation_digest": value["preparation_digest"], "collision_digest": background["digest"],
               "background_bytes_unchanged": True, "source_prim_excision_performed": False,
               "source_object_candidate_separate": True, "claim_ceiling": "development_only",
               "physical_measurement_proven": False}
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    receipt_path = output_root / "background_reuse.json"
    write_json(receipt_path, receipt)
    return _stage_result(stage=stage, configuration_path=configuration_path, output_artifacts=[
        {"role": "configured_collision_without_source_object", **background},
        {"role": "source_object_candidate_mesh", **candidate},
        {"role": REFERENCE_ROLE, **observation_artifact},
        {"role": "website_background_reuse_receipt", **_record(receipt_path)}])
