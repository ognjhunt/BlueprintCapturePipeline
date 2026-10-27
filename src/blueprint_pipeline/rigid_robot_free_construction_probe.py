"""Bind construction's rigid physics trace to the subsequently published scene.

The native-import GPU session runs this probe without another allocation. A
revision does not exist then: the trace binds exact construction document bytes
and asset bytes instead. Publication retention verifies those same documents
and the exact final physics plan before adding the final revision binding.
The original trace remains inside the immutable native-import receipt.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .rigid_robot_free_probe_plan import physics_plan, probe_context
from .task_evaluation_rigid_relocation_native_adapter import (
    DEFINITION_CONTRACT_PATH, SUCCESS_CONTRACT_PATH, EXECUTION_CONTRACT_PATH,
    SUPPORT_PLANE_CONTRACT_PATH, SOURCE_OBJECT_CONTRACT_PATH,
    STATIC_QUALIFICATION_CONTRACT_PATH, NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH,
    _adapt_rigid_relocation_documents,
)

CONSTRUCTION_SCHEMA = "rigid_robot_free_construction_binding.v1"
TRACE_FIELD = "rigid_robot_free_success_probe"
SOURCE_ALIASES = {
    DEFINITION_CONTRACT_PATH: "task.definition",
    SUCCESS_CONTRACT_PATH: "task.success_criteria",
    EXECUTION_CONTRACT_PATH: "task.execution",
    SUPPORT_PLANE_CONTRACT_PATH: "scene.registration.support_plane",
    SOURCE_OBJECT_CONTRACT_PATH: "task.subject.source_object",
}
DESTINATION_SOURCE_ALIASES = {"task.destination." + name: "task.destination." + name
    for name in ("asset", "rights_admission", "static_qualification")}


def _read_record(record: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("rigid_construction_probe_source_changed")
    payload = path.read_bytes()
    if len(payload) != record.get("size_bytes") or (
        "sha256:" + hashlib.sha256(payload).hexdigest() != record.get("digest")
    ):
        raise ValueError("rigid_construction_probe_source_changed")
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise ValueError("rigid_construction_probe_source_invalid")
    return result


def prepare_construction_probe(*, envelope: Mapping[str, Any], asset_path: Path,
                               static_path: Path, native_import: Mapping[str, Any],
                               destination_asset_path: Path | None = None) -> dict[str, Any]:
    """Derive the same rigid task from verified construction inputs, without a revision."""
    from .rigid_robot_free_probe_isaac import prepare_probe_scene
    from .task_evaluation_scene_rigid_robot_free_readiness import scored_rigid_task_spec

    rows = envelope.get("materialized_references") or []
    documents, source_digests = {}, {}
    for contract_path, alias in SOURCE_ALIASES.items():
        matches = [row for row in rows if isinstance(row, Mapping) and row.get("contract_path") == alias]
        if len(matches) != 1 or matches[0].get("full_byte_service_account_readback_passed") is not True:
            raise ValueError("rigid_construction_probe_source_missing:" + alias)
        row = matches[0]
        documents[contract_path] = _read_record(row, Path(row["materialized_path"]))
        source_digests[contract_path] = row["digest"]
    static_payload = static_path.read_bytes()
    documents[STATIC_QUALIFICATION_CONTRACT_PATH] = json.loads(static_payload)
    source_digests[STATIC_QUALIFICATION_CONTRACT_PATH] = "sha256:" + hashlib.sha256(static_payload).hexdigest()
    documents[NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH] = dict(native_import)
    template = documents[DEFINITION_CONTRACT_PATH]
    destination = envelope["request"]["task"].get("destination")
    identity = template["task_identity"]
    adapted = _adapt_rigid_relocation_documents(
        documents=documents, bindings=[], task={"identity": identity,
            "subject": {"identity": template["object_identity"]}, "strategy": template["strategy"]},
        replacement_identity=template["object_identity"], authority_digest=canonical_digest(source_digests),
        diagnostic=True,
    )
    bound_spec = adapted["native_task_definition"]["task_spec"]
    destination_probe, destination_geometry_digest = None, None
    if destination is not None:
        from .rigid_destination_task_binding import bind_destination_task_spec, destination_binding_from_geometry
        from .task_evaluation_rigid_destination_geometry import derive_rigid_destination_geometry

        local_destination = {}
        for contract_path, alias in DESTINATION_SOURCE_ALIASES.items():
            matches = [row for row in rows if isinstance(row, Mapping) and row.get("contract_path") == alias]
            if len(matches) != 1 or matches[0].get("full_byte_service_account_readback_passed") is not True:
                raise ValueError("rigid_construction_probe_source_missing:" + alias)
            row, path = matches[0], Path(matches[0]["materialized_path"])
            payload = path.read_bytes()
            if path.is_symlink() or len(payload) != row.get("size_bytes") or (
                    "sha256:" + hashlib.sha256(payload).hexdigest() != row.get("digest")):
                raise ValueError("rigid_construction_probe_source_changed")
            source_digests[contract_path] = row["digest"]
            local_destination[contract_path] = path
        static = json.loads(local_destination["task.destination.static_qualification"].read_text())
        simready_rows = [row for row in rows if isinstance(row, Mapping) and row.get("contract_path")
            == "construction.recipe.supplemental_destination.simready_result"]
        if len(simready_rows) != 1 or simready_rows[0].get("full_byte_service_account_readback_passed") is not True:
            raise ValueError("rigid_construction_probe_destination_simready_missing")
        simready_row = simready_rows[0]
        simready = _read_record(simready_row, Path(simready_row["materialized_path"]))
        geometry = derive_rigid_destination_geometry(subject_identity=template["object_identity"],
            destination_identity=destination["identity"], relation=destination["relation"],
            pose_world=destination["pose_world"],
            subject_static_qualification=documents[STATIC_QUALIFICATION_CONTRACT_PATH],
            subject_static_qualification_digest=source_digests[STATIC_QUALIFICATION_CONTRACT_PATH],
            subject_scoring_transform=bound_spec["interaction_affordance"]["asset_root_from_scoring_frame"],
            destination_static_qualification=static,
            destination_static_qualification_digest=source_digests["task.destination.static_qualification"],
            destination_simready_result=simready,
            qualification_limits=destination["native_probe"]["qualification_limits"])
        metadata = destination_binding_from_geometry(destination, geometry)
        bound_spec = bind_destination_task_spec(bound_spec, metadata,
            source_support=documents[SUPPORT_PLANE_CONTRACT_PATH],
            support_plane_digest=source_digests[SUPPORT_PLANE_CONTRACT_PATH])
        destination_geometry_digest = geometry["geometry_digest"]
        destination_probe = {"asset_path": str(destination_asset_path or local_destination["task.destination.asset"]),
            "asset_digest": source_digests["task.destination.asset"], "pose_world": metadata["destination_pose_world"],
            "intended_support_prim_paths": geometry["intended_support_prim_paths"],
            "collision_prim_paths": static["observed_structure"]["collision_prim_paths"]}
    spec = scored_rigid_task_spec(adapted=adapted, task_spec=bound_spec,
        site_id=envelope["request"]["scene"]["identity"]["id"], task_id=identity["id"],
        team_namespace=envelope["team_namespace"])
    task = {"task_spec": spec, "task_object_pose_world": adapted["native_task_definition"]["task_object_pose_world"],
        "replacement_asset_digest": native_import["asset_digest"],
        **({"destination_probe": destination_probe} if destination_probe is not None else {})}
    prepared = prepare_probe_scene(task=task, replacement_asset=asset_path,
        support=documents[SUPPORT_PLANE_CONTRACT_PATH], destination=template.get("destination_support_plane"),
        revision_digest=None)
    prepared["binding"].update(construction_binding={"schema_version": CONSTRUCTION_SCHEMA,
        "source_document_digests": source_digests,
        **({"destination_geometry_digest": destination_geometry_digest} if destination_geometry_digest is not None else {}),
        "configuration_run_id": envelope["run_id"]})
    return prepared


def published_construction_probe(*, revision: Mapping[str, Any], documents: Mapping[str, Path],
                                 task: Mapping[str, Any]) -> dict[str, Any] | None:
    """Verify immutable provenance and project a raw trace into its final revision."""
    native = json.loads(documents[NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH].read_text())
    raw = native.get(TRACE_FIELD)
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or raw.get("trace_digest") != canonical_digest(dict(raw), digest_field="trace_digest"):
        raise ValueError("rigid_construction_probe_trace_digest_invalid")
    if raw.get("status") != "recorded":
        return dict(raw)
    recorded = raw.get("binding") or {}
    construction = recorded.get("construction_binding") or {}
    expected_sources = set(SOURCE_ALIASES) | {STATIC_QUALIFICATION_CONTRACT_PATH}
    if revision["task_template"].get("destination") is not None:
        expected_sources |= set(DESTINATION_SOURCE_ALIASES)
        geometry = json.loads(Path(documents["task.destination.geometry"]).read_text())
        if (construction.get("destination_geometry_digest") != geometry.get("geometry_digest")
                or recorded.get("destination_asset_digest") != task["destination_probe"]["asset_digest"]):
            raise ValueError("rigid_construction_probe_destination_geometry_mismatch")
    if (construction.get("schema_version") != CONSTRUCTION_SCHEMA
            or construction.get("configuration_run_id") != revision["configuration_run_id"]
            or set(construction.get("source_document_digests") or {}) != expected_sources
            or recorded.get("replacement_asset_digest") != task["replacement_asset_digest"]
            or raw.get("plan") != physics_plan(probe_context(task["task_spec"]))):
        raise ValueError("rigid_construction_probe_binding_mismatch")
    for contract_path, digest in construction["source_document_digests"].items():
        if "sha256:" + hashlib.sha256(documents[contract_path].read_bytes()).hexdigest() != digest:
            raise ValueError("rigid_construction_probe_document_mismatch:" + contract_path)
    trace = json.loads(json.dumps(raw))
    trace["construction_trace_digest"] = raw["trace_digest"]
    trace["binding"].update(configured_scene_revision_digest=revision["revision_digest"],
        task_spec_digest=canonical_digest(task["task_spec"]))
    trace["trace_digest"] = canonical_digest(trace, digest_field="trace_digest")
    return trace
