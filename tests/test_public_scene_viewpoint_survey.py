from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.public_scene_viewpoint_survey import (
    build_room_viewpoint_survey,
)


def _box(ins_id: str, label: str, center: tuple[float, float, float]) -> dict:
    cx, cy, cz = center
    corners = []
    for x in (cx - 0.1, cx + 0.1):
        for y in (cy - 0.1, cy + 0.1):
            for z in (cz - 0.1, cz + 0.1):
                corners.append({"x": x, "y": y, "z": z})
    return {"ins_id": ins_id, "label": label, "bounding_box": corners}


def _sources(root: Path) -> tuple[Path, Path]:
    structure = root / "structure.json"
    labels = root / "labels.json"
    structure.write_text(
        json.dumps(
            {
                "rooms": [
                    {"profile": [[0, 0], [2, 0], [2, 2], [0, 2]]},
                    {"profile": [[3, 0], [5, 0], [5, 2], [3, 2]]},
                ],
                "walls": [],
                "holes": [],
            }
        ),
        encoding="utf-8",
    )
    labels.write_text(
        json.dumps(
            [
                _box("10", "cup", (1.0, 1.0, 0.8)),
                _box("20", "ornament", (4.0, 1.0, 0.8)),
                _box("30", "outside", (8.0, 8.0, 0.8)),
            ]
        ),
        encoding="utf-8",
    )
    return structure, labels


def test_room_survey_uses_interiorgs_rooms_and_assigns_publisher_objects(
    tmp_path: Path,
) -> None:
    structure, labels = _sources(tmp_path)

    observed = build_room_viewpoint_survey(
        structure_path=structure,
        labels_path=labels,
        scene_id="scene-1",
        approved_roots=[tmp_path],
    )

    assert observed["camera_count"] == 8
    assert observed["publisher_object_count"] == 3
    assert observed["articulated_open_close_inventory"]["candidate_count"] == 0
    assert observed["room_assigned_object_count"] == 2
    assert observed["unassigned_object_ids"] == ["30"]
    assert observed["rooms"][0]["publisher_objects"][0]["ins_id"] == "10"
    assert observed["rooms"][1]["publisher_objects"][0]["ins_id"] == "20"
    assert observed["source_files"]["labels"]["sha256"].startswith("sha256:")
    assert observed["survey_digest"].startswith("sha256:")
    assert observed["claim_boundary"]["camera_plan_only_renderer_not_executed"]
    assert observed["claim_boundary"]["survey_previews_are_not_method_inputs"]
    assert observed["claim_boundary"]["survey_render_quality_is_not_qualified"]
    assert observed["claim_boundary"]["survey_does_not_recover_uncaptured_geometry"]


def test_room_survey_digest_changes_when_observed_labels_change(tmp_path: Path) -> None:
    structure, labels = _sources(tmp_path)
    first = build_room_viewpoint_survey(
        structure_path=structure,
        labels_path=labels,
        scene_id="scene-1",
        approved_roots=[tmp_path],
    )
    payload = json.loads(labels.read_text(encoding="utf-8"))
    payload[0]["label"] = "mug"
    labels.write_text(json.dumps(payload), encoding="utf-8")

    second = build_room_viewpoint_survey(
        structure_path=structure,
        labels_path=labels,
        scene_id="scene-1",
        approved_roots=[tmp_path],
    )

    assert first["source_files"]["labels"]["sha256"] != second["source_files"]["labels"]["sha256"]
    assert first["survey_digest"] != second["survey_digest"]


def test_room_survey_derives_same_room_target_closeups_from_existing_planner(
    tmp_path: Path,
) -> None:
    structure, labels = _sources(tmp_path)

    observed = build_room_viewpoint_survey(
        structure_path=structure,
        labels_path=labels,
        scene_id="scene-1",
        approved_roots=[tmp_path],
        target_ins_id="10",
    )

    closeup = observed["target_closeup"]
    assert closeup["target_ins_id"] == "10"
    assert closeup["target_room_id"] == "room_00"
    assert closeup["camera_count"] >= 4
    assert closeup["planner"] == "scene_placement.perception_views.generate_view_ring"


def test_room_survey_rejects_unknown_target_instance(tmp_path: Path) -> None:
    structure, labels = _sources(tmp_path)

    with pytest.raises(ValueError, match="target_instance_not_exactly_one"):
        build_room_viewpoint_survey(
            structure_path=structure,
            labels_path=labels,
            scene_id="scene-1",
            approved_roots=[tmp_path],
            target_ins_id="missing",
        )


def test_room_survey_admits_wall_bound_door_from_adjacent_room(tmp_path: Path) -> None:
    structure, labels = _sources(tmp_path)
    payload = json.loads(labels.read_text(encoding="utf-8"))
    payload.append(_box("40", "door", (-0.12, 1.0, 1.0)))
    labels.write_text(json.dumps(payload), encoding="utf-8")

    observed = build_room_viewpoint_survey(
        structure_path=structure,
        labels_path=labels,
        scene_id="scene-1",
        approved_roots=[tmp_path],
        target_ins_id="40",
    )

    closeup = observed["target_closeup"]
    assert closeup["room_resolution"] == "within_publisher_room_boundary_tolerance"
    assert closeup["target_room_ids"] == ["room_00"]
    assert closeup["camera_count"] >= 4
    assert {camera["room_id"] for camera in closeup["cameras"]} == {"room_00"}


def test_room_survey_rejects_source_outside_approved_roots(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    approved_root = tmp_path / "approved"
    approved_root.mkdir()
    structure, labels = _sources(source_root)

    with pytest.raises(ValueError, match="path_outside_approved_roots"):
        build_room_viewpoint_survey(
            structure_path=structure,
            labels_path=labels,
            scene_id="scene-1",
            approved_roots=[approved_root],
        )
