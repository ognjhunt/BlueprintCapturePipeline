from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.sam31_camera_geometry import select_geometry_aware_camera_policy
from blueprint_pipeline.scene_placement.interiorgs_index import InteriorGSSceneSpatialIndex
from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import sha


def geometry_fixture(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    lower, upper = [-0.95, -0.2, 0.28], [-0.65, 0.2, 0.32]
    def box(identity, label, low, high):
        return {"ins_id": identity, "label": label, "bounding_box": [dict(zip("xyz", c, strict=True))
            for c in itertools.product(*zip(low, high, strict=True))]}
    labels = tmp_path / "labels.json"
    structure = tmp_path / "structure.json"
    identity = tmp_path / "identity.json"
    labels.write_text(json.dumps([
        box("focus", "generic rigid object", lower, upper),
        box("support", "support", [-1., -.5, 0.], [.5, .5, .27]),
        box("screen", "opaque obstruction", [-.35, -.12, 0.], [-.25, .12, 1.5]),
    ]))
    structure.write_text(json.dumps({"rooms": [
        {"profile": [[-1., -2.], [2., -2.], [2., 2.], [-1., 2.]]},
        {"profile": [[-3., -2.], [-1.1, -2.], [-1.1, 2.], [-3., 2.]]}],
        "walls": [{"location": [[-1.05, -2.], [-1.05, 2.]], "thickness": .1, "height": 2.8}]}))
    receipt = {"schema_version": "interiorgs_sage_collision_identity.v1",
        "whole_object_collision_identity_passed": True,
        "target": {"interiorgs_instance_id": "focus", "world_aabb_min_m": lower, "world_aabb_max_m": upper},
        "source_files": {"interiorgs_labels": {"sha256": sha(labels)}}, "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    identity.write_text(json.dumps(receipt))
    return {"labels_path": labels, "structure_path": structure, "collision_identity_path": identity,
            "target_instance_id": "focus", "source_min": lower, "source_max": upper}


def test_replaces_wall_and_other_room_views_with_distinct_clear_primary_and_reserve(tmp_path):
    args = geometry_fixture(tmp_path)
    policy = select_geometry_aware_camera_policy(**args)
    assert policy == select_geometry_aware_camera_policy(**args)
    primary, reserve = policy["views"], policy["replacement_views"]
    assert [r["camera_id"] for r in primary] == [f"source-{i:02d}" for i in range(1, 17)]
    assert [r["camera_id"] for r in reserve] == [f"reserve-{i:02d}" for i in range(1, 17)]
    assert len({tuple(r["position_offset_m"]) for r in primary + reserve}) == 32
    assert len({round(r["position_offset_m"][2], 3) for r in primary}) >= 3
    assert len({round(math.sqrt(sum(v * v for v in r["position_offset_m"])), 3)
                for r in primary}) >= 3
    screen = policy["geometry_screen"]
    by_id = {r["candidate_id"]: r for r in screen["candidates"]}
    for key in screen["selected_candidate_ids"] + screen["replacement_candidate_ids"]:
        row = by_id[key]
        assert row["status"] == "screened_candidate"
        assert row["camera_room_index"] == screen["target_room_index"]
        assert not row["containing_obstacle_ids"] and not row["occluding_obstacle_ids"]
    legacy = {r["legacy_camera_id"]: r for r in screen["candidates"] if r["legacy_camera_id"]}
    assert "camera_outside_target_room" in legacy["source-07"]["rejection_reasons"]
    assert "camera_outside_target_room" in legacy["source-08"]["rejection_reasons"]
    assert any("target_sight_line_intersects_observed_bounds" in r["rejection_reasons"]
               for r in screen["candidates"])
    assert all(value is False for value in screen["claim_boundary"].values())
    assert screen["source_files"]["structure"]["sha256"] == sha(args["structure_path"])


def test_room_missing_blocks_before_render(tmp_path):
    args = geometry_fixture(tmp_path)
    args["structure_path"].write_text('{"rooms": []}')
    with pytest.raises(ValueError, match="target_room_missing"):
        select_geometry_aware_camera_policy(**args)


def test_fully_occluded_target_blocks_instead_of_repeating_views(tmp_path):
    args = geometry_fixture(tmp_path)
    labels = json.loads(args["labels_path"].read_text())
    labels[2]["bounding_box"] = [dict(zip("xyz", c, strict=True)) for c in
        itertools.product((-1., -.6), (-.3, .3), (.325, .36))]
    args["labels_path"].write_text(json.dumps(labels))
    receipt = json.loads(args["collision_identity_path"].read_text())
    receipt["source_files"]["interiorgs_labels"]["sha256"] = sha(args["labels_path"])
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    args["collision_identity_path"].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="insufficient_clear_candidates"):
        select_geometry_aware_camera_policy(**args)


def test_source_identity_drift_is_rejected(tmp_path):
    args = geometry_fixture(tmp_path)
    args["labels_path"].write_text(args["labels_path"].read_text() + " ")
    with pytest.raises(ValueError, match="geometry_identity_invalid"):
        select_geometry_aware_camera_policy(**args)


def test_all_selected_cameras_use_existing_room_topology(tmp_path):
    args = geometry_fixture(tmp_path)
    policy = select_geometry_aware_camera_policy(**args)
    index = InteriorGSSceneSpatialIndex(args["labels_path"], args["structure_path"])
    target = index.object_by_instance("focus")
    for row in policy["views"] + policy["replacement_views"]:
        position = [target.centroid[i] + row["position_offset_m"][i] for i in range(3)]
        assert index.structure.room_index_of_point(position[:2]) == 0
