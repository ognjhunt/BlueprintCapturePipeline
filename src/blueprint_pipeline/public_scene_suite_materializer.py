"""Evidence-derived materialization for the bounded ADP-009A public suite.

This is intentionally not a dataset manager.  It opens an explicit, allowlisted
set of local files, derives their byte identities and scene joins, and emits two
component manifests plus a truthful ten-role index.  Missing method smoke or
other role evidence stays blocked; a caller cannot assert admission.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .public_scene_suite_index import (
    REQUIRED_ROLE_PROJECTS,
    build_public_scene_suite_index_receipt,
)
from .scene_placement.interiorgs_index import load_interiorgs_labels


COMPONENT_SCHEMA_VERSION = "public_scene_component_manifest.v1"
COMPONENT_RECEIPT_SCHEMA_VERSION = "public_scene_component_admission_receipt.v1"
REQUEST_SCHEMA_VERSION = "public_scene_suite_materialization_request.v1"
PROGRAM_ID = "arm-decision-proof-v1"
ADP_ITEM = "ADP-009A"
CLAIM_CEILING = "development_only"


class PublicSceneSuiteMaterializationError(ValueError):
    """The request or materialized evidence violates a fail-closed check."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicSceneSuiteMaterializationError(f"not_json_object:{path.name}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file_record(path: Path, *, root: Path, publisher_path: str, role: str) -> dict[str, Any]:
    _require_under(path, (root,))
    if not path.is_file() or path.stat().st_size <= 0:
        raise PublicSceneSuiteMaterializationError(f"missing_or_empty:{publisher_path}")
    return {
        "role": role,
        "publisher_path": publisher_path,
        "external_relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _require_under(path: Path, roots: Sequence[Path]) -> Path:
    resolved = path.expanduser().resolve()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise PublicSceneSuiteMaterializationError(f"path_outside_approved_roots:{resolved}")
    return resolved


def _rooted(root: Path, value: str) -> Path:
    if not value or Path(value).is_absolute():
        raise PublicSceneSuiteMaterializationError("paths_must_be_nonempty_and_relative")
    return _require_under(root / value, (root,))


def _aabb_for_instance(labels_path: Path, instance_id: str) -> tuple[Any, list[float], list[float]]:
    matches = [row for row in load_interiorgs_labels(labels_path) if row.id == instance_id]
    if len(matches) != 1:
        raise PublicSceneSuiteMaterializationError(
            f"target_instance_identity_not_unique:{instance_id}"
        )
    row = matches[0]
    return row, [float(value) for value in row.bbox_min], [float(value) for value in row.bbox_max]


def _box_iou(a_min: Sequence[float], a_max: Sequence[float], b_min: Sequence[float], b_max: Sequence[float]) -> tuple[float, float]:
    overlap = [max(0.0, min(a_max[i], b_max[i]) - max(a_min[i], b_min[i])) for i in range(3)]
    intersection = overlap[0] * overlap[1] * overlap[2]
    a_volume = max(0.0, a_max[0] - a_min[0]) * max(0.0, a_max[1] - a_min[1]) * max(0.0, a_max[2] - a_min[2])
    b_volume = max(0.0, b_max[0] - b_min[0]) * max(0.0, b_max[1] - b_min[1]) * max(0.0, b_max[2] - b_min[2])
    union = a_volume + b_volume - intersection
    return (intersection / union if union else 0.0, intersection / a_volume if a_volume else 0.0)


def _inspect_usd(path: Path, prim_paths: Sequence[str]) -> dict[str, Any]:
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    except ImportError as exc:  # pragma: no cover - production dependency boundary
        raise PublicSceneSuiteMaterializationError("openusd_runtime_missing") from exc

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise PublicSceneSuiteMaterializationError(f"usd_stage_open_failed:{path.name}")
    _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
    if unresolved:
        raise PublicSceneSuiteMaterializationError(f"usd_unresolved_dependencies:{path.name}")
    prims: dict[str, Any] = {}
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise PublicSceneSuiteMaterializationError(f"usd_prim_missing:{prim_path}")
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        prims[prim_path] = {
            "type_name": prim.GetTypeName(),
            "collision_api": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
            "world_aabb_min_m": [float(value) for value in box.GetMin()],
            "world_aabb_max_m": [float(value) for value in box.GetMax()],
        }
    return {
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "unresolved_dependency_count": 0,
        "prims": prims,
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _remote_head(repository: str) -> str:
    completed = subprocess.run(
        ["git", "ls-remote", repository, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = completed.stdout.strip().split()
    if len(fields) != 2 or fields[1] != "HEAD" or len(fields[0]) != 40:
        raise PublicSceneSuiteMaterializationError("method_official_remote_head_unresolved")
    return fields[0]


def _submodule_revisions(repo: Path) -> dict[str, str]:
    output = _git(repo, "submodule", "status", "--recursive")
    observed: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.strip().split()
        if len(fields) < 2:
            raise PublicSceneSuiteMaterializationError("method_submodule_status_invalid")
        revision = fields[0]
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise PublicSceneSuiteMaterializationError("method_submodule_not_clean_or_materialized")
        observed[fields[1]] = revision
    return dict(sorted(observed.items()))


def _method_component(
    *, role: str, project: str, spec: Mapping[str, Any], method_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = _rooted(method_root, str(spec["local_path"]))
    head = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    dirty = bool(_git(repo, "status", "--porcelain"))
    expected = str(spec["commit"])
    expected_tree = str(spec.get("tree") or "")
    blockers: list[str] = []
    if head != expected:
        blockers.append("method_source_revision_mismatch")
    if expected_tree and tree != expected_tree:
        blockers.append("method_source_tree_mismatch")
    if dirty:
        blockers.append("method_source_tree_dirty")
    remote_head = None
    if spec.get("verify_official_remote_head") is True:
        remote_head = _remote_head(str(spec["repository"]))
        if remote_head != expected:
            blockers.append("method_official_remote_head_advanced")

    expected_submodules = {
        str(path): str(revision)
        for path, revision in dict(spec.get("submodules") or {}).items()
    }
    observed_submodules = _submodule_revisions(repo)
    if observed_submodules != dict(sorted(expected_submodules.items())):
        blockers.append("method_submodule_revision_mismatch")

    source_files: list[dict[str, Any]] = []
    source_paths = [str(spec["source_license_path"])] + [
        str(path) for path in spec.get("dependency_files", [])
    ]
    for relative in source_paths:
        path = _rooted(repo, relative)
        source_files.append(
            _file_record(
                path,
                root=method_root,
                publisher_path=path.relative_to(repo).as_posix(),
                role="source_license" if relative == str(spec["source_license_path"]) else "dependency_declaration",
            )
        )

    documented_inputs = spec.get("documented_inputs") or {}
    for relative, expected_strings in documented_inputs.items():
        path = _rooted(repo, str(relative))
        text = path.read_text(encoding="utf-8")
        missing = [str(token) for token in expected_strings if str(token) not in text]
        if missing:
            raise PublicSceneSuiteMaterializationError(
                f"method_documented_input_changed:{role}:{relative}"
            )
        source_files.append(
            _file_record(
                path,
                root=method_root,
                publisher_path=path.relative_to(repo).as_posix(),
                role="author_workflow_declaration",
            )
        )
    deduplicated_files = {
        (record["external_relative_path"], record["sha256"]): record for record in source_files
    }
    source_files = list(deduplicated_files.values())
    smoke_path = spec.get("author_smoke_receipt")
    if not smoke_path:
        blockers.append(str(spec["smallest_blocker"]))
    else:
        receipt_path = _rooted(method_root, str(smoke_path))
        if not receipt_path.is_file():
            blockers.append(str(spec["smallest_blocker"]))
    manifest = {
        "schema_version": COMPONENT_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "adp_item": ADP_ITEM,
        "component_id": f"adp009a-{role}",
        "role": role,
        "source_project_id": project,
        "publisher_identity": {
            "repository": str(spec["repository"]),
            "revision": head,
            "repository_tree": tree,
            "official_remote_head": remote_head,
            "submodules": observed_submodules,
        },
        "materialized_artifacts": source_files,
        "rights": {
            "source_license": str(spec["source_license"]),
            "source_license_file_hashed": True,
            "checkpoint_or_author_data_rights_established": False,
        },
        "observed_evidence": {
            "source_checkout_opened": True,
            "source_tree_clean": not dirty,
            "source_tree_matches_expected": not expected_tree or tree == expected_tree,
            "official_remote_head_verified": remote_head == expected if remote_head else False,
            "submodule_revisions_verified": observed_submodules == dict(sorted(expected_submodules.items())),
            "dependency_declarations_hashed": True,
            "author_workflow_declarations_hashed": bool(documented_inputs),
            "author_method_executed": False,
            "author_smoke_receipt_bound": False,
        },
        "claim_ceiling": CLAIM_CEILING,
        "claim_boundaries": {
            "method_source_checkout_is_not_execution": True,
            "inpainting_result": False,
            "metric_geometry_qualified": False,
            "physical_evidence": False,
        },
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    receipt = _component_receipt(manifest, blockers=blockers, checks={
        "exact_source_revision": head == expected,
        "exact_source_tree": not expected_tree or tree == expected_tree,
        "clean_source_tree": not dirty,
        "official_remote_head": remote_head == expected if remote_head else False,
        "submodule_revisions": observed_submodules == dict(sorted(expected_submodules.items())),
        "source_license_and_dependency_files_hashed": True,
        "unchanged_author_smoke_executed": False,
    })
    return manifest, receipt


def _component_receipt(
    manifest: Mapping[str, Any], *, blockers: Sequence[str], checks: Mapping[str, bool]
) -> dict[str, Any]:
    normalized = sorted(set(str(item) for item in blockers if str(item)))
    receipt: dict[str, Any] = {
        "schema_version": COMPONENT_RECEIPT_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "adp_item": ADP_ITEM,
        "component_id": manifest["component_id"],
        "role": manifest["role"],
        "component_manifest_digest": manifest["manifest_digest"],
        "status": "admitted" if not normalized else "blocked",
        "blockers": normalized,
        "checks": dict(sorted(checks.items())),
        "artifact_bytes_opened": bool(manifest.get("materialized_artifacts")),
        "claim_ceiling": CLAIM_CEILING,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def _scene_components(
    *, request: Mapping[str, Any], repo_root: Path, data_root: Path
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    scene = request["scene"]
    scene_id = str(scene["publisher_scene_id"])
    folder = str(scene["interiorgs_folder"])
    if folder.rsplit("_", 1)[-1] != scene_id:
        raise PublicSceneSuiteMaterializationError("interiorgs_scene_id_mismatch")
    if Path(str(scene["sage_usdz_path"])).stem != scene_id:
        raise PublicSceneSuiteMaterializationError("sage_usdz_scene_id_mismatch")
    collision_name = Path(str(scene["sage_collision_path"])).name
    if collision_name != f"{scene_id}_collision.usd":
        raise PublicSceneSuiteMaterializationError("sage_collision_scene_id_mismatch")

    labels_path = _rooted(data_root, str(scene["labels_path"]))
    structure_path = _rooted(data_root, str(scene["structure_path"]))
    splat_path = _rooted(data_root, str(scene["splat_path"]))
    usdz_path = _rooted(data_root, str(scene["sage_usdz_path"]))
    collision_path = _rooted(data_root, str(scene["sage_collision_path"]))
    survey_path = _rooted(data_root, str(scene["survey_path"]))
    rights_path = _rooted(repo_root, str(scene["interiorgs_rights_authority_path"]))
    terms_path = _rooted(data_root, str(scene["interiorgs_terms_path"]))

    igs_revision = str(scene["interiorgs_revision"])
    sage_usdz_revision = str(scene["sage_usdz_revision"])
    sage_collision_revision = str(scene["sage_collision_revision"])
    artifacts = [
        _file_record(splat_path, root=data_root, publisher_path=f"{folder}/3dgs_compressed.ply", role="appearance_3dgs"),
        _file_record(labels_path, root=data_root, publisher_path=f"{folder}/labels.json", role="semantic_metadata"),
        _file_record(structure_path, root=data_root, publisher_path=f"{folder}/structure.json", role="scene_structure"),
        _file_record(usdz_path, root=data_root, publisher_path=f"InteriorGS_usdz/{scene_id}.usdz", role="publisher_simulation_representation"),
        _file_record(collision_path, root=data_root, publisher_path=f"Collision_Mesh/{scene_id}/{scene_id}_collision.usd", role="static_collision_geometry"),
    ]
    survey = _read_json(survey_path)
    if survey.get("scene_id") != scene_id:
        raise PublicSceneSuiteMaterializationError("survey_scene_id_mismatch")
    target = survey.get("target_closeup")
    if not isinstance(target, Mapping) or str(target.get("target_ins_id")) != str(scene["target_instance_id"]):
        raise PublicSceneSuiteMaterializationError("survey_target_identity_mismatch")
    if int(target.get("camera_count", 0)) < 4:
        raise PublicSceneSuiteMaterializationError("target_camera_coverage_insufficient")

    target_row, target_min, target_max = _aabb_for_instance(labels_path, str(scene["target_instance_id"]))
    support_row, support_min, support_max = _aabb_for_instance(labels_path, str(scene["support_instance_id"]))
    if target_row.label != str(scene["target_label"]):
        raise PublicSceneSuiteMaterializationError("target_semantic_identity_mismatch")
    if support_row.label != str(scene["support_label"]):
        raise PublicSceneSuiteMaterializationError("support_semantic_identity_mismatch")

    usdz = _inspect_usd(usdz_path, ())
    collision = _inspect_usd(collision_path, (str(scene["target_collision_prim"]), str(scene["support_collision_prim"])))
    for inspected in (usdz, collision):
        if inspected["up_axis"] != "Z":
            raise PublicSceneSuiteMaterializationError("usd_up_axis_not_z")
        if abs(inspected["meters_per_unit"] - 1.0) > 1e-9:
            raise PublicSceneSuiteMaterializationError("usd_units_not_meters")
    target_prim = collision["prims"][str(scene["target_collision_prim"])]
    support_prim = collision["prims"][str(scene["support_collision_prim"])]
    if target_prim["type_name"] != "Mesh" or not target_prim["collision_api"]:
        raise PublicSceneSuiteMaterializationError("target_collider_not_separately_removable_mesh")
    if support_prim["type_name"] != "Mesh" or not support_prim["collision_api"]:
        raise PublicSceneSuiteMaterializationError("support_collider_not_independent_mesh")
    target_iou, target_coverage = _box_iou(
        target_min, target_max,
        target_prim["world_aabb_min_m"], target_prim["world_aabb_max_m"],
    )
    support_iou, support_coverage = _box_iou(
        support_min, support_max,
        support_prim["world_aabb_min_m"], support_prim["world_aabb_max_m"],
    )
    if target_iou < float(scene["minimum_target_collider_iou"]):
        raise PublicSceneSuiteMaterializationError("target_collider_identity_below_threshold")
    if support_iou < float(scene["minimum_support_collider_iou"]):
        raise PublicSceneSuiteMaterializationError("support_collider_identity_below_threshold")

    rights = _read_json(rights_path)
    if rights.get("revision") != igs_revision:
        raise PublicSceneSuiteMaterializationError("interiorgs_rights_revision_mismatch")
    if rights.get("agent_accepted_terms") is not False:
        raise PublicSceneSuiteMaterializationError("interiorgs_rights_agent_acceptance_invalid")
    terms_sha = _sha256_file(terms_path)
    if rights.get("terms_text_sha256") != terms_sha:
        raise PublicSceneSuiteMaterializationError("interiorgs_terms_digest_mismatch")

    mapping = {
        "publisher_scene_id": scene_id,
        "interiorgs_folder": folder,
        "sage_usdz_scene_id": scene_id,
        "sage_collision_scene_id": scene_id,
    }
    coordinate = {
        "units": "meters",
        "handedness": "right_handed",
        "up_axis": "Z",
        "source_axes": ["right", "back", "up"],
        "origin": "publisher_scene_origin",
        "normalization_history": "publisher_bytes_preserved_without_blueprint_normalization",
        "T_source_world": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "T_world_source": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    }
    target_binding = {
        "interiorgs_instance_id": target_row.id,
        "semantic_label": target_row.label,
        "obb_aabb_min_m": target_min,
        "obb_aabb_max_m": target_max,
        "collision_prim_path": str(scene["target_collision_prim"]),
        "obb_collision_aabb_iou": round(target_iou, 6),
        "target_coverage_fraction": round(target_coverage, 6),
        "support_instance_id": support_row.id,
        "support_semantic_label": support_row.label,
        "support_collision_prim_path": str(scene["support_collision_prim"]),
        "support_obb_collision_aabb_iou": round(support_iou, 6),
        "support_coverage_fraction": round(support_coverage, 6),
        "separately_removable": True,
    }
    common_observed = {
        "room_count": len(survey.get("rooms", [])),
        "room_survey_camera_count": int(survey.get("camera_count", 0)),
        "target_closeup_camera_count": int(target["camera_count"]),
        "viewpoint_survey_digest": survey.get("survey_digest"),
        "method_outcomes_observed_before_selection": False,
        "raw_bytes_inspected": True,
    }
    common_boundaries = {
        "synthetic_appearance_and_semantics_only": True,
        "static_collision_proxy_only": True,
        "measurement_authoritative_local_surface_truth": False,
        "generated_geometry_is_metric_truth": False,
        "physical_evidence": False,
    }
    interior_manifest: dict[str, Any] = {
        "schema_version": COMPONENT_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "adp_item": ADP_ITEM,
        "component_id": f"adp009a-interiorgs-{scene_id}",
        "role": "interiorgs_appearance_scene",
        "source_project_id": "InteriorGS",
        "publisher_identity": {"repository": "spatialverse/InteriorGS", "revision": igs_revision},
        "scene_mapping": mapping,
        "materialized_artifacts": artifacts[:3],
        "rights": {
            "authority_record_digest": canonical_digest(rights, digest_field="record_digest"),
            "terms_artifact_sha256": terms_sha,
            "license": "custom_InteriorGS_terms",
            "allowed_use_ceiling": "internal_noncommercial_research_only",
            "redistribution_allowed": False,
            "attribution_required": True,
        },
        "coordinate_frame": coordinate,
        "target_binding": target_binding,
        "observed_evidence": common_observed,
        "claim_ceiling": CLAIM_CEILING,
        "claim_boundaries": common_boundaries,
    }
    interior_manifest["manifest_digest"] = canonical_digest(interior_manifest, digest_field="manifest_digest")
    sage_manifest: dict[str, Any] = {
        "schema_version": COMPONENT_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "adp_item": ADP_ITEM,
        "component_id": f"adp009a-sage3d-{scene_id}",
        "role": "sage3d_collision_companion",
        "source_project_id": "SAGE-3D",
        "publisher_identity": {
            "repository": "spatialverse/SAGE-3D_Collision_Mesh",
            "revision": sage_collision_revision,
            "matched_appearance_repository": "spatialverse/SAGE-3D_InteriorGS_usdz",
            "matched_appearance_revision": sage_usdz_revision,
        },
        "scene_mapping": mapping,
        "materialized_artifacts": artifacts[3:],
        "rights": {
            "license": "CC-BY-NC-4.0",
            "allowed_use_ceiling": "noncommercial_only",
            "redistribution_allowed_under_license": True,
            "blueprint_raw_byte_redistribution": False,
            "attribution_required": True,
        },
        "coordinate_frame": coordinate,
        "target_binding": target_binding,
        "usd_inspection": {"appearance_stage": usdz, "collision_stage": collision},
        "observed_evidence": common_observed,
        "claim_ceiling": CLAIM_CEILING,
        "claim_boundaries": common_boundaries,
    }
    sage_manifest["manifest_digest"] = canonical_digest(sage_manifest, digest_field="manifest_digest")
    checks = {
        "exact_scene_id_join": True,
        "materialized_files_hashed": True,
        "rights_authority_bound": True,
        "meters_and_z_up_verified": True,
        "inverse_transform_present": True,
        "target_collider_identity_verified": True,
        "support_collider_identity_verified": True,
        "usd_dependencies_resolved": True,
        "target_camera_coverage_verified": True,
    }
    return [
        (interior_manifest, _component_receipt(interior_manifest, blockers=(), checks=checks)),
        (sage_manifest, _component_receipt(sage_manifest, blockers=(), checks=checks)),
    ]


def _blocked_component(role: str, project: str, blocker: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest: dict[str, Any] = {
        "schema_version": COMPONENT_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "adp_item": ADP_ITEM,
        "component_id": f"adp009a-{role}",
        "role": role,
        "source_project_id": project,
        "publisher_identity": {"revision": "not_materialized"},
        "materialized_artifacts": [],
        "rights": {"established": False},
        "observed_evidence": {"required_receipt_present": False},
        "claim_ceiling": CLAIM_CEILING,
        "claim_boundaries": {"qualified": False, "physical_evidence": False},
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    return manifest, _component_receipt(manifest, blockers=(blocker,), checks={"required_receipt_present": False})


def _simready_control_component(
    *, spec: Mapping[str, Any], request: Mapping[str, Any], repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = _rooted(repo_root, str(spec["receipt_path"]))
    receipt = _read_json(receipt_path)
    supplied_digest = receipt.get("receipt_digest")
    if supplied_digest != canonical_digest(receipt, digest_field="receipt_digest"):
        raise PublicSceneSuiteMaterializationError("simready_control_receipt_digest_mismatch")
    if receipt.get("schema_version") != "adp009a_parametric_simready_receipt.v1":
        raise PublicSceneSuiteMaterializationError("simready_control_receipt_schema_invalid")
    if receipt.get("status") != "statically_validated":
        raise PublicSceneSuiteMaterializationError("simready_control_static_validation_missing")
    validation = receipt.get("simready_foundation_validation")
    if not isinstance(validation, Mapping) or validation.get("passed") is not True:
        raise PublicSceneSuiteMaterializationError("simready_foundation_profile_pass_missing")
    scene = request["scene"]
    if receipt.get("source_scene_id") != str(scene["publisher_scene_id"]):
        raise PublicSceneSuiteMaterializationError("simready_control_scene_mismatch")
    if receipt.get("source_instance_id") != str(scene["target_instance_id"]):
        raise PublicSceneSuiteMaterializationError("simready_control_target_mismatch")

    usd = receipt.get("usd")
    if not isinstance(usd, Mapping):
        raise PublicSceneSuiteMaterializationError("simready_control_usd_record_missing")
    usd_path = _rooted(repo_root, str(usd["relative_path"]))
    if _sha256_file(usd_path) != usd.get("sha256") or usd_path.stat().st_size != usd.get("size_bytes"):
        raise PublicSceneSuiteMaterializationError("simready_control_usd_bytes_changed")
    blocker = str(spec["smallest_blocker"])
    if receipt.get("blockers") != [blocker]:
        raise PublicSceneSuiteMaterializationError("simready_control_blocker_mismatch")

    artifacts = [
        _file_record(
            usd_path,
            root=repo_root,
            publisher_path=usd_path.relative_to(repo_root).as_posix(),
            role="cad_derived_simready_usd",
        ),
        _file_record(
            receipt_path,
            root=repo_root,
            publisher_path=receipt_path.relative_to(repo_root).as_posix(),
            role="materialization_and_static_validation_receipt",
        ),
    ]
    source_skill = receipt["cad_evidence"]["source_skill"]
    manifest: dict[str, Any] = {
        "schema_version": COMPONENT_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "adp_item": ADP_ITEM,
        "component_id": str(receipt["control_id"]),
        "role": "exact_simready_object",
        "source_project_id": "SimReady",
        "publisher_identity": {
            "cad_skill_repository": source_skill["repository"],
            "cad_skill_revision": source_skill["commit"],
            "simready_foundation_repository": validation["repository"],
            "revision": validation["commit"],
            "repository_tree": validation["tree"],
        },
        "scene_mapping": {
            "publisher_scene_id": receipt["source_scene_id"],
            "interiorgs_instance_id": receipt["source_instance_id"],
        },
        "materialized_artifacts": artifacts,
        "rights": {
            "cad_skill_license": source_skill["license"],
            "simready_foundation_license": validation["license"],
            "allowed_use_ceiling": "internal_noncommercial_research_only",
            "blueprint_asset_redistribution": False,
        },
        "observed_evidence": {
            "cad_step_and_mesh_opened": True,
            "usd_readback_passed": receipt["checks"]["usd_readback_passed"],
            "simready_foundation_profile": validation["profile"],
            "simready_foundation_profile_version": validation["profile_version"],
            "simready_foundation_profile_passed": True,
            "isaac_dynamic_probes_executed": False,
        },
        "claim_ceiling": CLAIM_CEILING,
        "claim_boundaries": {
            "static_profile_is_not_isaac_runtime": True,
            "physics_values_are_authoring_priors": True,
            "measurement_authoritative_geometry": False,
            "physical_evidence": False,
            "inpainting_result": False,
        },
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    return manifest, _component_receipt(
        manifest,
        blockers=(blocker,),
        checks={
            "cad_materialization_receipt_digest_verified": True,
            "usd_bytes_verified": True,
            "simready_foundation_profile_passed": True,
            "isaac_dynamic_probes_passed": False,
        },
    )


def _content_agents_component(
    *, spec: Mapping[str, Any], repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = _rooted(repo_root, str(spec["receipt_path"]))
    receipt = _read_json(receipt_path)
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        raise PublicSceneSuiteMaterializationError(
            "content_agents_preflight_receipt_digest_mismatch"
        )
    if receipt.get("schema_version") != "adp009a_usd_content_agents_preflight_receipt.v1":
        raise PublicSceneSuiteMaterializationError(
            "content_agents_preflight_receipt_schema_invalid"
        )
    if receipt.get("status") != "prepared_static_validation_passed":
        raise PublicSceneSuiteMaterializationError(
            "content_agents_preflight_status_invalid"
        )
    agents = receipt.get("agents")
    runtime = receipt.get("runtime")
    source = receipt.get("source")
    if not isinstance(agents, Mapping) or not isinstance(runtime, Mapping) or not isinstance(source, Mapping):
        raise PublicSceneSuiteMaterializationError(
            "content_agents_preflight_evidence_missing"
        )
    for name in ("material", "texture", "physics"):
        agent = agents.get(name)
        if (
            not isinstance(agent, Mapping)
            or agent.get("dry_run_executed") is not True
            or agent.get("full_agent_executed") is not False
        ):
            raise PublicSceneSuiteMaterializationError(
                f"content_agents_dry_run_evidence_invalid:{name}"
            )
    validation = agents.get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("executed") is not True
        or validation.get("dry_run") is not False
        or validation.get("verdict") != "pass"
    ):
        raise PublicSceneSuiteMaterializationError(
            "content_agents_validation_execution_missing"
        )
    joint = agents.get("joint")
    if (
        not isinstance(joint, Mapping)
        or joint.get("applicable") is not False
        or joint.get("executed") is not False
    ):
        raise PublicSceneSuiteMaterializationError(
            "content_agents_joint_applicability_invalid"
        )
    if (
        runtime.get("paid_resource_allocated") is not False
        or runtime.get("model_or_remote_renderer_called") is not False
    ):
        raise PublicSceneSuiteMaterializationError(
            "content_agents_preflight_runtime_claim_invalid"
        )
    blocker = str(spec["smallest_blocker"])
    manifest: dict[str, Any] = {
        "schema_version": COMPONENT_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "adp_item": ADP_ITEM,
        "component_id": "adp009a-usd-content-agents-candidate",
        "role": "usd_content_agents_candidate",
        "source_project_id": REQUIRED_ROLE_PROJECTS["usd_content_agents_candidate"],
        "publisher_identity": {
            "repository": source.get("repository"),
            "revision": source.get("commit"),
            "repository_tree": source.get("tree"),
            "version": source.get("version"),
            "container_image_digest": runtime.get("image_digest"),
            "container_platform": runtime.get("platform"),
        },
        "materialized_artifacts": [
            _file_record(
                receipt_path,
                root=repo_root,
                publisher_path=receipt_path.relative_to(repo_root).as_posix(),
                role="content_agents_preflight_receipt",
            )
        ],
        "rights": {
            "source_license": source.get("license"),
            "model_or_remote_backend_rights_exercised": False,
        },
        "observed_evidence": {
            "material_agent_dry_run": True,
            "texture_agent_dry_run": True,
            "physics_agent_dry_run": True,
            "validation_agent_static_check_executed": True,
            "validation_agent_static_check_passed": True,
            "joint_agent_inapplicable_single_rigid_body": True,
            "material_agent_full_execution": False,
            "texture_agent_full_execution": False,
            "physics_agent_full_execution": False,
            "paid_resource_allocated": False,
        },
        "claim_ceiling": CLAIM_CEILING,
        "claim_boundaries": {
            "dry_runs_are_not_agent_execution": True,
            "static_validation_is_not_dynamic_simulation": True,
            "content_agents_candidate_complete": False,
            "inpainting_result": False,
            "physical_evidence": False,
        },
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    return manifest, _component_receipt(
        manifest,
        blockers=(blocker,),
        checks={
            "preflight_receipt_digest_valid": True,
            "source_and_container_bound": True,
            "three_native_dry_runs_executed": True,
            "validation_agent_static_check_passed": True,
            "joint_agent_inapplicability_recorded": True,
            "full_material_texture_physics_execution": False,
        },
    )


def materialize_public_scene_suite(
    *, request_path: Path, repo_root: Path, data_root: Path, method_root: Path, output_root: Path
) -> dict[str, Any]:
    roots = tuple(path.expanduser().resolve() for path in (repo_root, data_root, method_root, output_root))
    repo_root, data_root, method_root, output_root = roots
    request_path = _require_under(request_path, (repo_root,))
    _require_under(output_root, (repo_root,))
    request = _read_json(request_path)
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise PublicSceneSuiteMaterializationError("materialization_request_schema_invalid")
    forbidden = {"status", "admitted", "matrix_complete"}.intersection(request)
    if forbidden:
        raise PublicSceneSuiteMaterializationError("caller_asserted_admission_forbidden")

    pairs = _scene_components(request=request, repo_root=repo_root, data_root=data_root)
    by_role = {manifest["role"]: (manifest, receipt) for manifest, receipt in pairs}
    for role, spec in request["methods"].items():
        project = REQUIRED_ROLE_PROJECTS[role]
        by_role[role] = _method_component(role=role, project=project, spec=spec, method_root=method_root)
    for role, blocker in request["missing_roles"].items():
        by_role[role] = _blocked_component(role, REQUIRED_ROLE_PROJECTS[role], str(blocker))
    simready_control = request.get("simready_control")
    if simready_control is not None:
        if not isinstance(simready_control, Mapping):
            raise PublicSceneSuiteMaterializationError("simready_control_spec_invalid")
        if "exact_simready_object" in request["missing_roles"]:
            raise PublicSceneSuiteMaterializationError("simready_control_role_duplicated")
        by_role["exact_simready_object"] = _simready_control_component(
            spec=simready_control,
            request=request,
            repo_root=repo_root,
        )
    content_agents = request.get("content_agents")
    if content_agents is not None:
        if not isinstance(content_agents, Mapping):
            raise PublicSceneSuiteMaterializationError("content_agents_spec_invalid")
        if "usd_content_agents_candidate" in request["missing_roles"]:
            raise PublicSceneSuiteMaterializationError("content_agents_role_duplicated")
        by_role["usd_content_agents_candidate"] = _content_agents_component(
            spec=content_agents,
            repo_root=repo_root,
        )
    if set(by_role) != set(REQUIRED_ROLE_PROJECTS):
        raise PublicSceneSuiteMaterializationError("request_does_not_cover_exact_ten_roles")

    output_root.mkdir(parents=True, exist_ok=True)
    components: list[dict[str, Any]] = []
    for role in REQUIRED_ROLE_PROJECTS:
        manifest, receipt = by_role[role]
        manifest_path = output_root / f"{role}.component_manifest.json"
        receipt_path = output_root / f"{role}.component_receipt.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        revision = manifest["publisher_identity"].get("revision")
        if isinstance(revision, str) and len(revision) == 40:
            exact_revision = {"kind": "git_commit", "value": revision}
        else:
            exact_revision = {"kind": "content_digest", "value": manifest["manifest_digest"]}
        artifacts = manifest.get("materialized_artifacts", [])
        artifact_digest = (
            canonical_digest({"artifacts": artifacts})
            if artifacts
            else manifest["manifest_digest"]
        )
        components.append({
            "role": role,
            "source_project_id": REQUIRED_ROLE_PROJECTS[role],
            "component_manifest_digest": manifest["manifest_digest"],
            "component_admission_receipt_digest": receipt["receipt_digest"],
            "exact_revision": exact_revision,
            "exact_artifact_digest": artifact_digest,
            "status": receipt["status"],
            "blockers": receipt["blockers"],
        })
    index: dict[str, Any] = {
        "schema_version": "public_scene_suite_index.v1",
        "program_id": PROGRAM_ID,
        "adp_item": "ADP-009",
        "index_id": str(request["index_id"]),
        "components": components,
        "claim_ceiling": CLAIM_CEILING,
        "claim_boundaries": {
            "exact_public_suite_binding": True,
            "public_scene_software_qualified": False,
            "metric_geometry_qualified": False,
            "task_physics_qualified": False,
            "partner_capture_qualified": False,
            "prospective_validation": False,
            "physical_evidence": False,
            "digital_twin": False,
            "deployment_readiness": False,
            "physical_safety": False,
            "customer_value": False,
            "general_sim_to_real_fidelity": False,
        },
    }
    index["index_digest"] = canonical_digest(index, digest_field="index_digest")
    evaluated_on = dt.date.fromisoformat(str(request["evaluated_on"]))
    index_receipt = build_public_scene_suite_index_receipt(index, evaluated_on=evaluated_on)
    (output_root / "public_scene_suite_index.v1.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "public_scene_suite_index_receipt.v1.json").write_text(
        json.dumps(index_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": index_receipt["status"],
        "admitted_role_count": index_receipt["admitted_role_count"],
        "blocked_roles": index_receipt["blocked_roles"],
        "index_digest": index["index_digest"],
        "index_receipt_digest": index_receipt["receipt_digest"],
        "output_root": str(output_root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = materialize_public_scene_suite(
        request_path=args.request,
        repo_root=args.repo_root,
        data_root=args.data_root,
        method_root=args.method_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
