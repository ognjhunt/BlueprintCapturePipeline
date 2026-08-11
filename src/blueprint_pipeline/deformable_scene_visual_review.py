"""Digest-bound visual review of deformable scene-selection closeups.

This seam records what a reviewer can and cannot see in retained reconnaissance
frames.  It never promotes a virtual closeup to source capture or evaluation
media.  Collision topology may independently support an engineered receptacle
design basis, but it cannot fill hidden appearance or material observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


REQUEST_SCHEMA_VERSION = "adp_deformable_scene_visual_review_request.v2"
RECEIPT_SCHEMA_VERSION = "adp_deformable_scene_visual_review.v2"
SUPPORTED_REQUEST_SCHEMA_VERSIONS = {
    "adp_deformable_scene_visual_review_request.v1",
    REQUEST_SCHEMA_VERSION,
}
SUPPORTED_TOPOLOGY_SCHEMA_VERSIONS = {
    "interiorgs_sage_collision_component_topology.v1",
    "interiorgs_sage_collision_component_topology.v2",
}

TARGET_KINDS = {"movable_deformable", "destination_receptacle"}
MATERIAL_CLASSES = {"towel_or_cloth", "sponge", "cable_or_hose", "not_applicable"}
REST_STATES = {"rolled", "folded", "flat", "coiled", "not_applicable"}
SUPPORT_RELATIONS = {
    "direct_rigid_surface",
    "observed_deformable_stack",
    "observed_container_contents",
    "not_applicable",
}


class DeformableSceneVisualReviewError(ValueError):
    """Stable, sorted visual-review failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _clone(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise DeformableSceneVisualReviewError([error]) from exc
    if not isinstance(result, dict):
        raise DeformableSceneVisualReviewError([error])
    return result


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _required_bool(
    value: Mapping[str, Any], key: str, *, target_id: str, errors: list[str]
) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        errors.append(f"visual_review_{key}_invalid:{target_id}")
        return False
    return result


def _camera_index(render_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = render_manifest.get("cameras")
    if not isinstance(rows, list):
        raise DeformableSceneVisualReviewError(["visual_review_render_cameras_invalid"])
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise DeformableSceneVisualReviewError(["visual_review_render_cameras_invalid"])
        camera_id = _string(row.get("id"))
        if not camera_id or camera_id in result:
            raise DeformableSceneVisualReviewError(["visual_review_render_camera_identity_invalid"])
        result[camera_id] = row
    return result


def _topology_index(collision_topology: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if collision_topology.get(
        "schema_version"
    ) not in SUPPORTED_TOPOLOGY_SCHEMA_VERSIONS or collision_topology.get(
        "receipt_digest"
    ) != canonical_digest(collision_topology, digest_field="receipt_digest"):
        raise DeformableSceneVisualReviewError(["visual_review_collision_topology_invalid"])
    rows = collision_topology.get("targets")
    if not isinstance(rows, list):
        raise DeformableSceneVisualReviewError(["visual_review_collision_topology_invalid"])
    return {
        _string(row.get("interiorgs_instance_id")): row
        for row in rows
        if isinstance(row, Mapping) and _string(row.get("interiorgs_instance_id"))
    }


def materialize_deformable_scene_visual_review(
    request: Mapping[str, Any],
    *,
    render_manifest: Mapping[str, Any],
    collision_topology: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate cited frames and derive bounded selection roles."""

    payload = _clone(request, error="visual_review_request_not_json")
    forbidden = {"status", "admitted", "qualified", "review_digest"}.intersection(payload)
    errors: list[str] = []
    if forbidden:
        errors.append("visual_review_caller_asserted_outcome_forbidden")
    if payload.get("schema_version") not in SUPPORTED_REQUEST_SCHEMA_VERSIONS:
        errors.append("visual_review_request_schema_invalid")
    scene_id = _string(payload.get("scene_id"))
    if not scene_id:
        errors.append("visual_review_scene_id_invalid")
    for field in ("reviewer_id", "reviewed_at"):
        if not _string(payload.get(field)):
            errors.append(f"visual_review_{field}_invalid")
    if payload.get("learned_policy_outcomes_inspected") is not False:
        errors.append("visual_review_policy_outcome_leakage")
    if payload.get("reconnaissance_only") is not True:
        errors.append("visual_review_reconnaissance_boundary_invalid")

    render_digest = _string(render_manifest.get("render_manifest_digest"))
    if (
        not render_digest
        or render_digest != canonical_digest(render_manifest, digest_field="render_manifest_digest")
        or payload.get("render_manifest_digest") != render_digest
    ):
        errors.append("visual_review_render_manifest_invalid")
    cameras = _camera_index(render_manifest)
    topology = _topology_index(collision_topology)
    if payload.get("collision_topology_receipt_digest") != collision_topology.get("receipt_digest"):
        errors.append("visual_review_collision_topology_join_invalid")

    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        errors.append("visual_review_targets_invalid")
        raw_targets = []
    target_ids: set[str] = set()
    normalized_targets: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, Mapping):
            errors.append(f"visual_review_target_invalid:{index}")
            continue
        target_id = _string(raw.get("target_id"))
        if not target_id or target_id in target_ids:
            errors.append(f"visual_review_target_id_invalid:{target_id or index}")
        target_ids.add(target_id)
        instance_id = _string(raw.get("publisher_instance_id"))
        target_kind = _string(raw.get("target_kind"))
        if not instance_id:
            errors.append(f"visual_review_instance_id_invalid:{target_id}")
        if target_kind not in TARGET_KINDS:
            errors.append(f"visual_review_target_kind_invalid:{target_id}")
        material_class = _string(raw.get("material_class"))
        rest_state = _string(raw.get("rest_state"))
        support_relation = _string(raw.get("support_relation"))
        if material_class not in MATERIAL_CLASSES:
            errors.append(f"visual_review_material_class_invalid:{target_id}")
        if rest_state not in REST_STATES:
            errors.append(f"visual_review_rest_state_invalid:{target_id}")
        if support_relation not in SUPPORT_RELATIONS:
            errors.append(f"visual_review_support_relation_invalid:{target_id}")

        cited_frames = raw.get("cited_frames")
        normalized_frames: list[dict[str, Any]] = []
        if not isinstance(cited_frames, list) or not cited_frames:
            errors.append(f"visual_review_cited_frames_invalid:{target_id}")
            cited_frames = []
        for frame in cited_frames:
            if not isinstance(frame, Mapping):
                errors.append(f"visual_review_cited_frames_invalid:{target_id}")
                continue
            camera_id = _string(frame.get("camera_id"))
            retained = cameras.get(camera_id)
            if (
                retained is None
                or retained.get("nonblank") is not True
                or frame.get("sha256") != retained.get("digest")
                or frame.get("size_bytes") != retained.get("bytes")
            ):
                errors.append(f"visual_review_frame_identity_invalid:{camera_id}")
                continue
            normalized_frames.append(
                {
                    "camera_id": camera_id,
                    "size_bytes": int(retained["bytes"]),
                    "sha256": str(retained["digest"]),
                }
            )

        material_supported = _required_bool(
            raw, "material_class_supported_by_observation", target_id=target_id, errors=errors
        )
        rigid_exterior = _required_bool(
            raw, "rigid_exterior_observed", target_id=target_id, errors=errors
        )
        open_rim = _required_bool(raw, "open_rim_observed", target_id=target_id, errors=errors)
        occupied = _required_bool(raw, "interior_occupied", target_id=target_id, errors=errors)
        complete_interior = _required_bool(
            raw,
            "complete_interior_appearance_observed",
            target_id=target_id,
            errors=errors,
        )
        topology_row = topology.get(instance_id)
        collision_identity = bool(
            topology_row and topology_row.get("component_collision_identity_passed") is True
        )
        opening_probe = topology_row.get("opening_probe") if topology_row else None
        open_collision_cavity = bool(
            isinstance(opening_probe, Mapping)
            and opening_probe.get("open_collision_cavity_passed") is True
        )

        if target_kind == "movable_deformable":
            selection_role = (
                "selected_movable_design_basis"
                if material_supported and collision_identity
                else "rejected_movable_candidate"
            )
            source_destination_admitted = False
            engineered_twin_basis_admitted = False
        else:
            source_destination_admitted = bool(
                rigid_exterior
                and open_rim
                and not occupied
                and complete_interior
                and collision_identity
                and open_collision_cavity
            )
            engineered_twin_basis_admitted = bool(
                rigid_exterior and open_rim and collision_identity
            )
            selection_role = (
                "source_destination"
                if source_destination_admitted
                else "engineered_twin_design_basis"
                if engineered_twin_basis_admitted
                else "rejected_destination_candidate"
            )
        normalized_targets.append(
            {
                "target_id": target_id,
                "publisher_instance_id": instance_id,
                "target_kind": target_kind,
                "material_class": material_class,
                "material_class_supported_by_observation": material_supported,
                "rest_state": rest_state,
                "support_relation": support_relation,
                "rigid_exterior_observed": rigid_exterior,
                "open_rim_observed": open_rim,
                "interior_occupied": occupied,
                "complete_interior_appearance_observed": complete_interior,
                "collision_component_identity_passed": collision_identity,
                "open_collision_cavity_passed": open_collision_cavity,
                "source_destination_admitted": source_destination_admitted,
                "engineered_twin_design_basis_admitted": engineered_twin_basis_admitted,
                "selection_role": selection_role,
                "cited_frames": sorted(normalized_frames, key=lambda row: row["camera_id"]),
                "review_notes": _string(raw.get("review_notes")),
            }
        )

    selected_movable = [
        row
        for row in normalized_targets
        if row["selection_role"] == "selected_movable_design_basis"
    ]
    destination_bases = [
        row
        for row in normalized_targets
        if row["selection_role"] in {"source_destination", "engineered_twin_design_basis"}
    ]
    if len(selected_movable) != 1:
        errors.append("visual_review_selected_movable_not_exactly_one")
    if len(destination_bases) != 1:
        errors.append("visual_review_destination_basis_not_exactly_one")
    if errors:
        raise DeformableSceneVisualReviewError(errors)

    normalized_targets.sort(key=lambda row: row["target_id"])
    result: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "scene_id": scene_id,
        "reviewer_id": _string(payload["reviewer_id"]),
        "reviewed_at": _string(payload["reviewed_at"]),
        "learned_policy_outcomes_inspected": False,
        "reconnaissance_only": True,
        "render_manifest_digest": render_digest,
        "collision_topology_receipt_digest": collision_topology["receipt_digest"],
        "targets": normalized_targets,
        "selected_movable_instance_id": selected_movable[0]["publisher_instance_id"],
        "selected_destination_design_basis_instance_id": destination_bases[0][
            "publisher_instance_id"
        ],
        "source_destination_is_occupied": destination_bases[0]["interior_occupied"],
        "source_destination_complete_interior_appearance_observed": destination_bases[0][
            "complete_interior_appearance_observed"
        ],
        "composition_required": not destination_bases[0]["source_destination_admitted"],
        "claim_boundary": {
            "virtual_closeup_recovers_missing_source_observations": False,
            "collision_cavity_establishes_hidden_appearance": False,
            "engineered_twin_hidden_geometry_is_source_truth": False,
            "review_is_evaluation_policy_media": False,
            "physical_material_equivalence_proven": False,
        },
        "review_digest": "",
    }
    result["review_digest"] = canonical_digest(result, digest_field="review_digest")
    return result


def _read_json(path: Path, *, error: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeformableSceneVisualReviewError([error]) from exc
    if not isinstance(value, dict):
        raise DeformableSceneVisualReviewError([error])
    return value


def _resolved_under(path: str | Path, roots: Sequence[str | Path]) -> Path:
    resolved = Path(path).expanduser().resolve()
    approved = [Path(root).expanduser().resolve() for root in roots]
    if not approved or not any(resolved == root or root in resolved.parents for root in approved):
        raise DeformableSceneVisualReviewError(
            [f"visual_review_path_outside_approved_roots:{resolved}"]
        )
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--render-manifest", required=True)
    parser.add_argument("--collision-topology", required=True)
    parser.add_argument("--approved-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    request_path = _resolved_under(args.request, args.approved_root)
    render_path = _resolved_under(args.render_manifest, args.approved_root)
    topology_path = _resolved_under(args.collision_topology, args.approved_root)
    output_path = _resolved_under(args.out, args.approved_root)
    result = materialize_deformable_scene_visual_review(
        _read_json(request_path, error="visual_review_request_invalid"),
        render_manifest=_read_json(render_path, error="visual_review_render_manifest_invalid"),
        collision_topology=_read_json(
            topology_path, error="visual_review_collision_topology_invalid"
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output_path),
                "review_digest": result["review_digest"],
                "composition_required": result["composition_required"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "DeformableSceneVisualReviewError",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "materialize_deformable_scene_visual_review",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
