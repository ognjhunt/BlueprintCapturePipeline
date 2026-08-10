from __future__ import annotations

import json
from pathlib import Path

import pytest
from pxr import Usd, UsdPhysics

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.simready_graph_asset import (
    SimReadyGraphAssetError,
    author_simready_graph_asset,
    validate_simready_graph_asset_spec,
)


def _source_receipt(tmp_path: Path) -> Path:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    import hashlib

    receipt = {
        "schema_version": "articulated_source_asset.v1",
        "status": "materialized",
        "source_collision_prim_path": "/Root/source",
        "target": {"interiorgs_instance_id": "fixture_source"},
        "output_asset": {
            "relative_path": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    path = tmp_path / "source_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def _task_freeze(tmp_path: Path, spec: dict) -> Path:
    value = {
        "schema_version": "dual_task_task_freeze.v1",
        "task_id": spec["task_id"],
        "task_kind": spec["task_kind"],
        "source_object": {"instance_id": spec["source_object_instance_id"]},
        "removal_plan": {"replacement_asset_id": spec["asset_id"]},
        "articulation_graph": spec["articulation_graph"],
        "task_freeze_digest": "",
    }
    value["task_freeze_digest"] = canonical_digest(
        value, digest_field="task_freeze_digest"
    )
    spec["task_freeze_digest"] = value["task_freeze_digest"]
    spec.pop("spec_digest", None)
    normalized = validate_simready_graph_asset_spec(spec)
    spec.clear()
    spec.update(normalized)
    path = tmp_path / "task_freeze.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _graph(*, target: bool = True) -> dict:
    role = "target" if target else "locked"
    return {
        "schema_version": "adp_articulation_graph.v1",
        "links": [
            {"link_id": "root", "is_root": True, "semantic_role": "root"},
            {"link_id": "child", "is_root": False, "semantic_role": "child"},
        ],
        "joints": [
            {
                "joint_id": "hinge",
                "parent_link_id": "root",
                "child_link_id": "child",
                "joint_type": "revolute",
                "role": role,
                "axis": [1.0, 0.0, 0.0],
                "limits": [0.0, 1.2],
                "reset_position": 0.0,
                "reset_tolerance": 0.001,
                "drive": {
                    "drive_type": "force",
                    "stiffness": 0.0 if target else 100.0,
                    "damping": 2.0,
                    "maximum_force": 50.0,
                },
                "dependency": None,
            }
        ],
        "collision_pairs": [
            {"link_a": "root", "link_b": "child", "collision_enabled": True}
        ],
        "success_predicate": {
            "combination": "all",
            "joint_intervals": {"hinge": [0.7, 0.9]} if target else {},
        },
    }


def _spec(source_receipt: Path, *, target: bool = True) -> dict:
    source = json.loads(source_receipt.read_text())

    def link(link_id: str, provenance: str) -> dict:
        return {
            "link_id": link_id,
            "mass_kg": 2.0,
            "center_of_mass_m": [0.0, 0.0, 0.0],
            "diagonal_inertia_kg_m2": [0.1, 0.1, 0.1],
            "friction": 0.6,
            "restitution": 0.05,
            "physics_provenance": "authored_estimate_unqualified",
            "rest_pose": {
                "translation_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "geometry": [
                {
                    "geometry_id": f"{link_id}_shape",
                    "kind": "box",
                    "size_m": [0.2, 0.1, 0.05],
                    "translation_m": [0.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "display_color_rgb": [0.4, 0.5, 0.6],
                    "provenance": provenance,
                }
            ],
        }
    value = {
        "schema_version": "simready_graph_asset_spec.v1",
        "asset_id": "fixture_asset",
        "task_id": "fixture_task",
        "source_object_instance_id": "fixture_source",
        "task_kind": "articulated_interaction" if target else "rigid_object_manipulation",
        "task_freeze_digest": "sha256:" + "a" * 64,
        "source_asset_receipt_digest": source["receipt_digest"],
        "root_body_mode": "fixed" if target else "dynamic",
        "world_pose": {
            "translation_m": [1.0, 2.0, 3.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "articulation_graph": _graph(target=target),
        "links": [
            link("root", "observed_bounds_derived_candidate"),
            link("child", "generated_candidate"),
        ],
        "joint_frames": [
            {
                "joint_id": "hinge",
                "parent_position_m": [0.1, 0.0, 0.0],
                "child_position_m": [-0.1, 0.0, 0.0],
                "parent_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "child_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        ],
        "appearance_materially_qualified": False,
        "physical_equivalence_claimed": False,
    }
    return validate_simready_graph_asset_spec(value)


@pytest.mark.parametrize("target", [True, False])
def test_general_graph_compiler_authors_articulated_and_rigid_subjects(
    tmp_path: Path, target: bool
) -> None:
    source = _source_receipt(tmp_path)
    spec = _spec(source, target=target)
    task_freeze = _task_freeze(tmp_path, spec)
    receipt = author_simready_graph_asset(
        spec=spec,
        task_freeze_receipt_path=task_freeze,
        source_asset_receipt_path=source,
        destination=tmp_path / f"asset_{target}.usda",
    )

    stage = Usd.Stage.Open(receipt["output_usd"]["path"])
    assert stage.GetDefaultPrim().GetPath().pathString == "/Asset"
    assert stage.GetPrimAtPath("/Asset/joints/hinge").IsA(UsdPhysics.RevoluteJoint)
    assert set(receipt["link_paths"]) == {"root", "child"}
    assert receipt["claim_boundary"]["native_simulator_import_qualified"] is False
    assert receipt["claim_boundary"]["generated_geometry_is_observed_truth"] is False


def test_source_receipt_and_asset_bytes_are_verified(tmp_path: Path) -> None:
    source = _source_receipt(tmp_path)
    spec = _spec(source)
    task_freeze = _task_freeze(tmp_path, spec)
    (tmp_path / "source.usda").write_text("changed", encoding="utf-8")

    with pytest.raises(SimReadyGraphAssetError) as caught:
        author_simready_graph_asset(
            spec=spec,
            task_freeze_receipt_path=task_freeze,
            source_asset_receipt_path=source,
            destination=tmp_path / "asset.usda",
        )
    assert "graph_asset_source_asset_bytes_changed" in caught.value.codes


def test_task_freeze_identity_and_digest_are_verified(tmp_path: Path) -> None:
    source = _source_receipt(tmp_path)
    spec = _spec(source)
    task_freeze = _task_freeze(tmp_path, spec)
    value = json.loads(task_freeze.read_text())
    value["removal_plan"]["replacement_asset_id"] = "swapped_asset"
    task_freeze.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(SimReadyGraphAssetError) as caught:
        author_simready_graph_asset(
            spec=spec,
            task_freeze_receipt_path=task_freeze,
            source_asset_receipt_path=source,
            destination=tmp_path / "asset.usda",
        )
    assert "graph_asset_task_freeze_invalid" in caught.value.codes


def test_spec_rejects_missing_joint_frames_and_self_qualification(tmp_path: Path) -> None:
    source = _source_receipt(tmp_path)
    spec = _spec(source)
    spec.pop("spec_digest")
    spec["joint_frames"] = []
    spec["appearance_materially_qualified"] = True

    with pytest.raises(SimReadyGraphAssetError) as caught:
        validate_simready_graph_asset_spec(spec)
    assert set(caught.value.codes) >= {
        "graph_asset_joint_frame_set_mismatch",
        "graph_asset_appearance_self_qualification_forbidden",
    }


def test_joint_frames_must_author_the_graph_axis(tmp_path: Path) -> None:
    source = _source_receipt(tmp_path)
    spec = _spec(source)
    spec.pop("spec_digest")
    spec["articulation_graph"]["joints"][0]["axis"] = [0.0, 0.0, 1.0]

    with pytest.raises(SimReadyGraphAssetError) as caught:
        validate_simready_graph_asset_spec(spec)
    assert set(caught.value.codes) >= {
        "graph_asset_joint_frame_axis_mismatch:hinge:parent",
        "graph_asset_joint_frame_axis_mismatch:hinge:child",
    }
