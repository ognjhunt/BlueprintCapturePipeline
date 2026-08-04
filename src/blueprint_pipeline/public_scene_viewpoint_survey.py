"""Deterministic room-wide camera survey for ADP-009A scene selection.

This is selection evidence only. It extends the existing InteriorGS scene-index
layer with fixed room-centre overview cameras and a room-scoped publisher object
inventory; it does not render, edit, or admit a scene.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .decision_evidence_contracts import canonical_digest
from .scene_placement.interiorgs_index import (
    load_interiorgs_labels,
    load_interiorgs_structure,
    point_in_polygon,
)
from .scene_placement.perception_views import generate_view_ring


SCHEMA_VERSION = "adp009a_room_viewpoint_survey.v1"


def _resolved_under(path: str | Path, approved_roots: Sequence[str | Path]) -> Path:
    resolved = Path(path).expanduser().resolve()
    roots = [Path(root).expanduser().resolve() for root in approved_roots]
    if not roots or not any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError(f"path_outside_approved_roots:{resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _distance_to_segment(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _representative_point(polygon: list[tuple[float, float]]) -> tuple[float, float]:
    if len(polygon) < 3:
        raise ValueError("room_profile_has_fewer_than_three_points")
    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)
    best: tuple[float, float, float] | None = None
    best_key: tuple[float, float, float] | None = None
    # A fixed 41x41 lattice is deterministic and finds an interior point well
    # away from walls even for concave publisher room profiles.
    for ix in range(1, 41):
        x = min_x + (max_x - min_x) * ix / 41.0
        for iy in range(1, 41):
            y = min_y + (max_y - min_y) * iy / 41.0
            if not point_in_polygon((x, y), polygon):
                continue
            clearance = min(
                _distance_to_segment((x, y), polygon[index - 1], polygon[index])
                for index in range(len(polygon))
            )
            candidate = (clearance, -x, -y)
            if best_key is None or candidate > best_key:
                best = (clearance, x, y)
                best_key = candidate
    if best is None:
        raise ValueError("room_profile_has_no_interior_lattice_point")
    return best[1], best[2]


def build_room_viewpoint_survey(
    *,
    structure_path: str | Path,
    labels_path: str | Path,
    scene_id: str,
    approved_roots: Sequence[str | Path],
    target_ins_id: str | None = None,
) -> dict[str, Any]:
    structure_source = _resolved_under(structure_path, approved_roots)
    labels_source = _resolved_under(labels_path, approved_roots)
    structure = load_interiorgs_structure(structure_source)
    objects = load_interiorgs_labels(labels_source)
    if not structure.rooms:
        raise ValueError("structure_rooms_missing")
    cameras: list[dict[str, Any]] = []
    room_records: list[dict[str, Any]] = []
    assigned_object_ids: set[str] = set()
    for room_index, polygon in enumerate(structure.rooms):
        center_x, center_y = _representative_point(polygon)
        diagonal = math.hypot(
            max(point[0] for point in polygon) - min(point[0] for point in polygon),
            max(point[1] for point in polygon) - min(point[1] for point in polygon),
        )
        look_distance = min(2.0, max(0.75, diagonal * 0.35))
        room_id = f"room_{room_index:02d}"
        room_objects = sorted(
            (
                {
                    "ins_id": item.id,
                    "label": item.extra.get("raw_label", item.label),
                    "centroid_world_m": [round(value, 9) for value in item.centroid],
                }
                for item in objects
                if point_in_polygon((item.centroid[0], item.centroid[1]), polygon)
            ),
            key=lambda row: (str(row["label"]), str(row["ins_id"])),
        )
        assigned_object_ids.update(str(row["ins_id"]) for row in room_objects)
        room_records.append(
            {
                "room_id": room_id,
                "representative_point_xy_m": [round(center_x, 9), round(center_y, 9)],
                "profile_vertex_count": len(polygon),
                "publisher_objects": room_objects,
                "publisher_object_count": len(room_objects),
            }
        )
        for azimuth_deg in (0, 90, 180, 270):
            azimuth = math.radians(azimuth_deg)
            cameras.append(
                {
                    "id": f"{room_id}_yaw_{azimuth_deg:03d}",
                    "room_id": room_id,
                    "spec": {
                        "pos": [round(center_x, 9), round(center_y, 9), 1.35],
                        "target": [
                            round(center_x + look_distance * math.cos(azimuth), 9),
                            round(center_y + look_distance * math.sin(azimuth), 9),
                            1.05,
                        ],
                        "fov": 70.0,
                        "up": [0.0, 0.0, 1.0],
                    },
                }
            )
    target_closeup: dict[str, Any] | None = None
    if target_ins_id is not None:
        matches = [item for item in objects if item.id == str(target_ins_id)]
        if len(matches) != 1:
            raise ValueError(f"target_instance_not_exactly_one:{target_ins_id}")
        target = matches[0]
        target_room_index = structure.room_index_of_point(
            (target.centroid[0], target.centroid[1])
        )
        if target_room_index is None:
            raise ValueError(f"target_instance_not_in_room:{target_ins_id}")
        radius = max(0.75, 2.5 * max(target.size()[0], target.size()[1]))
        planned = generate_view_ring(
            target.centroid,
            radius,
            n_azimuths=8,
            elevations_deg=(25.0,),
            vfov_deg=55.0,
            width=1024,
            height=768,
        )
        target_cameras: list[dict[str, Any]] = []
        for index, camera in enumerate(planned):
            eye = camera["eye"]
            if structure.room_index_of_point((float(eye[0]), float(eye[1]))) != target_room_index:
                continue
            target_cameras.append(
                {
                    "id": f"target_{target.id}_view_{index:02d}",
                    "spec": {
                        "pos": [round(float(value), 9) for value in eye],
                        "target": [round(float(value), 9) for value in camera["target"]],
                        "fov": 55.0,
                        "up": [round(float(value), 9) for value in camera["up"]],
                    },
                }
            )
        if len(target_cameras) < 4:
            raise ValueError(f"target_closeup_viewpoint_coverage_insufficient:{target_ins_id}")
        target_closeup = {
            "target_ins_id": target.id,
            "target_label": target.extra.get("raw_label", target.label),
            "target_room_id": f"room_{target_room_index:02d}",
            "target_centroid_world_m": [round(value, 9) for value in target.centroid],
            "planner": "scene_placement.perception_views.generate_view_ring",
            "planner_parameters": {
                "radius_m": round(radius, 9),
                "azimuth_count_before_room_filter": 8,
                "elevation_degrees": 25.0,
                "vertical_fov_degrees": 55.0,
                "same_room_filter": True,
            },
            "cameras": target_cameras,
            "camera_count": len(target_cameras),
        }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009A",
        "scene_id": str(scene_id),
        "purpose": "selection_reconnaissance_only",
        "method_input": False,
        "method_outcome": False,
        "source_files": {
            "structure": {
                "path": structure_source.name,
                "size_bytes": structure_source.stat().st_size,
                "sha256": _sha256(structure_source),
            },
            "labels": {
                "path": labels_source.name,
                "size_bytes": labels_source.stat().st_size,
                "sha256": _sha256(labels_source),
            },
        },
        "rooms": room_records,
        "cameras": cameras,
        "camera_count": len(cameras),
        "publisher_object_count": len(objects),
        "room_assigned_object_count": len(assigned_object_ids),
        "unassigned_object_ids": sorted(
            (item.id for item in objects if item.id not in assigned_object_ids),
            key=lambda value: (len(value), value),
        ),
        "target_closeup": target_closeup,
        "claim_boundary": {
            "publisher_objects_are_selection_metadata": True,
            "camera_plan_only_renderer_not_executed": True,
            "survey_previews_are_not_method_inputs": True,
            "survey_render_quality_is_not_qualified": True,
            "survey_does_not_establish_visibility": True,
            "survey_does_not_establish_collision_identity": True,
            "survey_does_not_recover_uncaptured_geometry": True,
        },
        "survey_digest": "",
    }
    result["survey_digest"] = canonical_digest(result, digest_field="survey_digest")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--approved-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--renderer-cameras-out")
    parser.add_argument("--target-ins-id")
    parser.add_argument("--target-cameras-out")
    args = parser.parse_args(argv)
    output = _resolved_under(args.out, args.approved_root)
    survey = build_room_viewpoint_survey(
        structure_path=args.structure,
        labels_path=args.labels,
        scene_id=args.scene_id,
        approved_roots=args.approved_root,
        target_ins_id=args.target_ins_id,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(survey, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    camera_output = None
    if args.renderer_cameras_out:
        camera_output = _resolved_under(args.renderer_cameras_out, args.approved_root)
        camera_output.parent.mkdir(parents=True, exist_ok=True)
        camera_output.write_text(
            json.dumps(
                [{"id": row["id"], "spec": row["spec"]} for row in survey["cameras"]],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    target_camera_output = None
    if args.target_cameras_out:
        target_closeup = survey.get("target_closeup")
        if not isinstance(target_closeup, dict):
            raise ValueError("target_cameras_out_requires_target_ins_id")
        target_camera_output = _resolved_under(args.target_cameras_out, args.approved_root)
        target_camera_output.parent.mkdir(parents=True, exist_ok=True)
        target_camera_output.write_text(
            json.dumps(target_closeup["cameras"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "renderer_cameras_output": str(camera_output) if camera_output else None,
                "target_cameras_output": (
                    str(target_camera_output) if target_camera_output else None
                ),
                "survey_digest": survey["survey_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
