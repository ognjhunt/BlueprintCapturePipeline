"""Materialize scene-neutral InteriorGS/SAGE component admission records.

The original ADP-009A suite materializer couples an InteriorGS target to a
pick-and-place support object.  Articulated tasks do not have that relation.
This module keeps the downstream component contract while deriving it from a
validated, outcome-blind scene/task freeze.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .second_scene_freeze import validate_second_scene_freeze


COMPONENT_SCHEMA = "public_scene_component_manifest.v1"
RECEIPT_SCHEMA = "public_scene_component_admission_receipt.v1"
PROGRAM_ID = "arm-decision-proof-v1"
CLAIM_CEILING = "development_only_public_dataset_component"


class RegisteredSceneComponentError(ValueError):
    """The frozen scene cannot produce an admitted component contract."""


def _artifact_by_role(freeze: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = freeze.get("source_artifacts")
    if not isinstance(rows, list):
        raise RegisteredSceneComponentError("registered_scene_source_artifacts_invalid")
    return {str(row.get("role")): dict(row) for row in rows if isinstance(row, Mapping)}


def _component_receipt(manifest: Mapping[str, Any]) -> dict[str, Any]:
    coordinate = manifest.get("coordinate_frame")
    qualification_status = (
        str(coordinate.get("qualification_status") or "legacy_verified")
        if isinstance(coordinate, Mapping)
        else "unavailable"
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "program_id": PROGRAM_ID,
        "adp_item": str(manifest["adp_item"]),
        "component_id": str(manifest["component_id"]),
        "role": str(manifest["role"]),
        "component_manifest_digest": str(manifest["manifest_digest"]),
        "status": "admitted",
        "blockers": [],
        "checks": {
            "exact_scene_id_join": True,
            "freeze_digest_verified": True,
            "materialized_files_hashed": True,
            "rights_authority_bound": True,
            "coordinate_frame_status_bound": qualification_status
            in {
                "legacy_verified",
                "provider_declared_not_independently_validated",
                "independently_qualified",
            },
            "coordinate_frame_qualification_status": qualification_status,
            "inverse_transform_present": True,
            "target_binding_verified": True,
        },
        "artifact_bytes_opened": bool(manifest.get("materialized_artifacts")),
        "claim_ceiling": CLAIM_CEILING,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def build_registered_scene_components(
    freeze: Mapping[str, Any],
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Build component records from an already validated scene freeze.

    This pure builder deliberately has no task-kind, support-surface, or object
    category branch.  The validating wrapper below owns byte verification.
    """

    scene = freeze.get("scene")
    rights = freeze.get("rights")
    registration = freeze.get("registration")
    topology = freeze.get("topology_and_visibility")
    if not all(isinstance(value, Mapping) for value in (scene, rights, registration, topology)):
        raise RegisteredSceneComponentError("registered_scene_freeze_sections_invalid")
    scene_id = str(scene.get("publisher_scene_id") or "")
    folder = str(scene.get("interiorgs_folder") or "")
    target_id = str(scene.get("target_instance_id") or "")
    target_label = str(scene.get("target_semantic_label") or "")
    collision_prim = str(scene.get("target_sage_collision_prim_path") or "")
    if not scene_id or not folder or not target_id or not target_label or not collision_prim:
        raise RegisteredSceneComponentError("registered_scene_identity_invalid")
    if folder.rsplit("_", 1)[-1] != scene_id:
        raise RegisteredSceneComponentError("registered_scene_folder_mismatch")

    sources = _artifact_by_role(freeze)
    required = {
        "interiorgs_splat",
        "interiorgs_labels",
        "interiorgs_structure",
        "sage_collision",
    }
    if set(sources) != required:
        raise RegisteredSceneComponentError("registered_scene_source_roles_invalid")

    def component_artifact(source_role: str, role: str) -> dict[str, Any]:
        source = sources[source_role]
        return {
            "role": role,
            "publisher_path": str(source["relative_path"]),
            "external_relative_path": str(source["relative_path"]),
            "size_bytes": int(source["size_bytes"]),
            "sha256": str(source["sha256"]),
            "source_repository": str(source["repository"]),
            "source_revision": str(source["revision"]),
        }

    coordinate = {
        "units": "meters",
        "handedness": str(registration["handedness"]),
        "up_axis": str(registration["up_axis"]),
        "origin": "publisher_scene_origin",
        "normalization_history": "publisher_bytes_preserved_without_blueprint_normalization",
        "T_source_world": registration["interiorgs_to_sage_transform"],
        "T_world_source": registration["inverse_transform"],
        "round_trip_max_error_m": float(registration["round_trip_max_error_m"]),
        "qualification_status": str(
            registration.get("qualification_status") or "legacy_verified"
        ),
    }
    target_binding = {
        "interiorgs_instance_id": target_id,
        "semantic_label": target_label,
        "obb_aabb_min_m": list(scene["target_world_aabb_min_m"]),
        "obb_aabb_max_m": list(scene["target_world_aabb_max_m"]),
        "collision_prim_path": collision_prim,
        "obb_collision_aabb_iou": float(scene["target_sage_aabb_iou"]),
        "separately_removable": True,
        "task_neutral_binding": True,
    }
    mapping = {
        "publisher_scene_id": scene_id,
        "interiorgs_folder": folder,
        "sage_collision_scene_id": scene_id,
    }
    observed = {
        "freeze_digest": str(freeze["freeze_digest"]),
        "room_count": int(topology["publisher_room_count"]),
        "room_survey_camera_count": int(topology["room_survey_camera_count"]),
        "target_closeup_camera_count": int(topology["target_closeup_camera_count"]),
        "source_splat_count": int(topology["source_splat_count"]),
        "retained_splat_count": int(topology["retained_splat_count"]),
        "global_decimation_applied": bool(topology["global_decimation_applied"]),
        "method_outcomes_observed_before_selection": False,
        "raw_bytes_inspected": True,
    }
    boundaries = {
        "synthetic_appearance_and_semantics_only": True,
        "publisher_collision_is_static_source_geometry": True,
        "source_joint_topology_claimed": False,
        "measurement_authoritative_local_surface_truth": False,
        "generated_geometry_is_metric_truth": False,
        "physical_evidence": False,
    }
    common = {
        "schema_version": COMPONENT_SCHEMA,
        "program_id": PROGRAM_ID,
        "adp_item": str(freeze.get("adp_item") or "ADP-009D"),
        "scene_mapping": mapping,
        "coordinate_frame": coordinate,
        "target_binding": target_binding,
        "observed_evidence": observed,
        "claim_ceiling": CLAIM_CEILING,
        "claim_boundaries": boundaries,
    }
    appearance: dict[str, Any] = {
        **common,
        "component_id": f"public-scene-interiorgs-{scene_id}-{target_id}",
        "role": "interiorgs_appearance_scene",
        "source_project_id": "InteriorGS",
        "publisher_identity": {
            "repository": str(rights["interiorgs_repository"]),
            "revision": str(rights["interiorgs_revision"]),
        },
        "materialized_artifacts": [
            component_artifact("interiorgs_splat", "appearance_3dgs"),
            component_artifact("interiorgs_labels", "semantic_metadata"),
            component_artifact("interiorgs_structure", "scene_structure"),
        ],
        "rights": {
            "license": str(rights["interiorgs_license"]),
            "declared_use_scope": str(rights["declared_use_scope"]),
            "redistribution_allowed": bool(rights["raw_dataset_redistribution_allowed"]),
            "external_provider_upload_authorized": bool(
                rights["external_provider_upload_authorized"]
            ),
            "commercial_use_allowed": bool(rights["commercial_use_allowed"]),
            "authority_record": dict(freeze["rights_authority_record"]),
            "disclosure_rule": str(rights["disclosure_rule"]),
        },
    }
    appearance["manifest_digest"] = canonical_digest(
        appearance, digest_field="manifest_digest"
    )
    collision: dict[str, Any] = {
        **common,
        "component_id": f"public-scene-sage3d-{scene_id}-{target_id}",
        "role": "sage3d_collision_companion",
        "source_project_id": "SAGE-3D",
        "publisher_identity": {
            "repository": str(rights["sage_collision_repository"]),
            "revision": str(rights["sage_collision_revision"]),
        },
        "materialized_artifacts": [
            component_artifact("sage_collision", "static_collision_geometry")
        ],
        "rights": {
            "license": str(rights["sage_collision_license"]),
            "declared_use_scope": str(rights["declared_use_scope"]),
            "blueprint_raw_byte_redistribution": False,
            "external_provider_upload_authorized": False,
            "commercial_use_allowed": False,
            "attribution_required": bool(rights["required_attribution_and_citation"]),
        },
    }
    collision["manifest_digest"] = canonical_digest(
        collision, digest_field="manifest_digest"
    )
    return {
        "interiorgs_appearance_scene": (appearance, _component_receipt(appearance)),
        "sage3d_collision_companion": (collision, _component_receipt(collision)),
    }


def materialize_registered_scene_components(
    *,
    freeze_path: str | Path,
    repo_root: str | Path,
    data_root: str | Path,
    output_root: str | Path,
) -> dict[str, Path]:
    """Verify source bytes and write the two admitted component records."""

    repo = Path(repo_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    freeze_file = Path(freeze_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if repo != freeze_file and repo not in freeze_file.parents:
        raise RegisteredSceneComponentError("registered_scene_freeze_outside_repo")
    if repo != output and repo not in output.parents:
        raise RegisteredSceneComponentError("registered_scene_output_outside_repo")
    parsed = json.loads(freeze_file.read_text(encoding="utf-8"))
    freeze = validate_second_scene_freeze(
        parsed, repo_root=repo, data_root=data, verify_files=True
    )
    components = build_registered_scene_components(freeze)
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for role, (manifest, receipt) in components.items():
        manifest_path = output / f"{role}.component_manifest.json"
        receipt_path = output / f"{role}.component_receipt.json"
        manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
        written[f"{role}_manifest"] = manifest_path
        written[f"{role}_receipt"] = receipt_path
    return written


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    written = materialize_registered_scene_components(
        freeze_path=args.freeze,
        repo_root=args.repo_root,
        data_root=args.data_root,
        output_root=args.output_root,
    )
    print(canonical_json({key: str(value) for key, value in written.items()}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
