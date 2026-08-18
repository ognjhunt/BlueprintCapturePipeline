from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.simready_graph_asset import (
    SimReadyGraphAssetError,
    author_simready_graph_asset,
    validate_simready_graph_asset_spec,
)
from blueprint_pipeline.simready_graph_asset_static_qualification import (
    qualify_simready_graph_asset_static,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_INPUTS_CLI = REPO_ROOT / "scripts/materialize_paired_target_native_inputs.py"


def _run_native_inputs_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(NATIVE_INPUTS_CLI), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
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


def _measured_derivation(axis=(1.0, 0.0, 0.0)) -> dict:
    """A derivation receipt corroborating `axis`, shaped like the real one.

    Authoring a commanded joint now requires the scan to agree with the spec,
    so every fixture that authors one carries its measurement.
    """

    payload = {
        "schema_version": "measured_articulation_derivation.v1",
        "status": "derived_from_measurement",
        "source_vertex_count": 19020,
        "stage_up_axis_index": 2,
        "front_plate": {
            "axis_index": 1,
            "outward_sign": -1,
            "plane_m": -0.24,
            "plate_vertex_count": 872,
            "shell_vertex_count": 1850,
            "facing_proposed_by": "test_scene_context",
            "facing_is_proposal_not_measurement": True,
        },
        "forward_shell": {"vertex_count": 1850, "front_m": -0.301},
        "target_joint": {
            "axis": list(axis),
            "pivot_asset_m": [-0.299, -0.301, 0.435],
            "sign_rule": "commanded_travel_must_increase_clearance_from_parent",
        },
        "claim_boundary": {
            "physics_typed_by_hand": False,
            "axis_sign_is_derived_not_input": True,
        },
        "derivation_digest": "",
    }
    payload["derivation_digest"] = canonical_digest(
        payload, digest_field="derivation_digest"
    )
    return payload


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
        measured_derivation=_measured_derivation(),
    )

    stage = Usd.Stage.Open(receipt["output_usd"]["path"])
    assert stage.GetDefaultPrim().GetPath().pathString == "/Asset"
    assert stage.GetPrimAtPath("/Asset/joints/hinge").IsA(UsdPhysics.RevoluteJoint)
    assert set(receipt["link_paths"]) == {"root", "child"}
    assert receipt["claim_boundary"]["native_simulator_import_qualified"] is False
    assert receipt["claim_boundary"]["generated_geometry_is_observed_truth"] is False
    collider = stage.GetPrimAtPath("/Asset/links/root/geometry/root_shape")
    imageable = UsdGeom.Imageable(collider)
    assert imageable.ComputePurpose() == UsdGeom.Tokens.guide
    assert imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
    assert collider.GetCustomDataByKey("blueprint:collisionGeometryOnly") is True


def test_static_readback_qualifies_authored_structure_but_retains_claim_blockers(
    tmp_path: Path,
) -> None:
    source = _source_receipt(tmp_path)
    spec = _spec(source)
    task_freeze = _task_freeze(tmp_path, spec)
    receipt_path = tmp_path / "asset.receipt.json"
    author_simready_graph_asset(
        spec=spec,
        task_freeze_receipt_path=task_freeze,
        source_asset_receipt_path=source,
        destination=tmp_path / "asset.usda",
        measured_derivation=_measured_derivation(),
        receipt_path=receipt_path,
    )

    qualification = qualify_simready_graph_asset_static(
        spec=spec,
        authoring_receipt_path=receipt_path,
        output_path=tmp_path / "static_qualification.json",
    )

    assert qualification["status"] == "authored_structure_statically_qualified"
    assert qualification["authored_structure_statically_qualified"] is True
    assert qualification["structural_findings"] == []
    assert set(qualification["contract_blockers"]) >= {
        "visual_material_artifact_unbound",
        "texture_artifact_unbound",
        "collision_approximation_contract_unbound",
        "native_simulator_import_unexecuted",
        "joint_physics_behavior_unexecuted",
    }
    assert qualification["claim_boundary"]["native_simulator_import_qualified"] is False
    assert (tmp_path / "static_qualification.json").is_file()


def test_native_inputs_cli_authors_and_statically_qualifies_exact_graph_bytes(
    tmp_path: Path,
) -> None:
    """The production CLI joins spec, freeze, source bytes, USD and receipts."""

    source_receipt = _source_receipt(tmp_path)
    source_asset = tmp_path / "source.usda"
    source_before = source_asset.read_bytes()
    spec = _spec(source_receipt)
    task_freeze = _task_freeze(tmp_path, spec)
    measured_path = tmp_path / "measured_derivation.json"
    measured_path.write_text(json.dumps(_measured_derivation()), encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    authored_usd = tmp_path / "authored.usda"
    authored_receipt = tmp_path / "authored.receipt.json"

    authored_process = _run_native_inputs_cli(
        "simready-graph-asset",
        "--spec",
        str(spec_path),
        "--task-freeze",
        str(task_freeze),
        "--source-asset-receipt",
        str(source_receipt),
        "--measured-derivation",
        str(measured_path),
        "--output-usd",
        str(authored_usd),
        "--output-receipt",
        str(authored_receipt),
    )

    assert authored_process.returncode == 0, authored_process.stderr + authored_process.stdout
    authored_summary = json.loads(authored_process.stdout)
    assert authored_summary["step"] == "simready-graph-asset"
    assert authored_summary["provider_mutation_performed"] is False
    authored = json.loads(authored_receipt.read_text())
    assert authored["status"] == "simready_candidate_authored"
    assert authored["receipt_digest"] == canonical_digest(
        authored, digest_field="receipt_digest"
    )
    assert authored["output_usd"] == {
        "path": str(authored_usd.resolve()),
        "size_bytes": authored_usd.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(authored_usd.read_bytes()).hexdigest(),
        "default_prim": "/Asset",
        "meters_per_unit": 1.0,
        "up_axis": "Z",
    }
    assert source_asset.read_bytes() == source_before

    qualification_path = tmp_path / "static-qualification.json"
    qualification_process = _run_native_inputs_cli(
        "simready-static-qualification",
        "--spec",
        str(spec_path),
        "--authoring-receipt",
        str(authored_receipt),
        "--output",
        str(qualification_path),
    )

    assert qualification_process.returncode == 0, (
        qualification_process.stderr + qualification_process.stdout
    )
    qualification_summary = json.loads(qualification_process.stdout)
    assert qualification_summary["step"] == "simready-static-qualification"
    assert qualification_summary["provider_mutation_performed"] is False
    qualification = json.loads(qualification_path.read_text())
    assert qualification["status"] == "authored_structure_statically_qualified"
    assert qualification["receipt_digest"] == canonical_digest(
        qualification, digest_field="receipt_digest"
    )
    assert qualification["replacement_usd"] == {
        "path": str(authored_usd.resolve()),
        "size_bytes": authored_usd.stat().st_size,
        "sha256": authored["output_usd"]["sha256"],
    }
    assert qualification["registered_replacement_asset"] is None
    assert qualification["claim_boundary"]["native_simulator_import_qualified"] is False


def test_native_inputs_cli_refuses_drifted_source_and_invalid_registered_bytes(
    tmp_path: Path,
) -> None:
    """Hostile bytes produce typed no-provider refusals and no output artifact."""

    source_receipt = _source_receipt(tmp_path)
    spec = _spec(source_receipt)
    task_freeze = _task_freeze(tmp_path, spec)
    measured_path = tmp_path / "measured_derivation.json"
    measured_path.write_text(json.dumps(_measured_derivation()), encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    (tmp_path / "source.usda").write_text("changed", encoding="utf-8")
    authored_usd = tmp_path / "must-not-exist.usda"
    authored_receipt = tmp_path / "must-not-exist.receipt.json"

    refused = _run_native_inputs_cli(
        "simready-graph-asset",
        "--spec",
        str(spec_path),
        "--task-freeze",
        str(task_freeze),
        "--source-asset-receipt",
        str(source_receipt),
        "--measured-derivation",
        str(measured_path),
        "--output-usd",
        str(authored_usd),
        "--output-receipt",
        str(authored_receipt),
    )

    assert refused.returncode == 2
    refusal = json.loads(refused.stdout)
    assert refusal["provider_mutation_performed"] is False
    assert "graph_asset_source_asset_bytes_changed" in refusal["blockers"][0]
    assert not authored_usd.exists()
    assert not authored_receipt.exists()

    # Restore the exact source and create a valid authoring result, then prove
    # registered mode validates the receipt rather than treating a path as an
    # assertion from the caller.
    (tmp_path / "source.usda").write_text("#usda 1.0\n", encoding="utf-8")
    assert _run_native_inputs_cli(
        "simready-graph-asset",
        "--spec",
        str(spec_path),
        "--task-freeze",
        str(task_freeze),
        "--source-asset-receipt",
        str(source_receipt),
        "--measured-derivation",
        str(measured_path),
        "--output-usd",
        str(authored_usd),
        "--output-receipt",
        str(authored_receipt),
    ).returncode == 0
    invalid_registered = tmp_path / "invalid-registered.json"
    invalid_registered.write_text("{}\n", encoding="utf-8")
    static_output = tmp_path / "must-not-exist-static.json"
    refused_registered = _run_native_inputs_cli(
        "simready-static-qualification",
        "--spec",
        str(spec_path),
        "--authoring-receipt",
        str(authored_receipt),
        "--registered-asset-receipt",
        str(invalid_registered),
        "--output",
        str(static_output),
    )

    assert refused_registered.returncode == 2
    registered_refusal = json.loads(refused_registered.stdout)
    assert registered_refusal["provider_mutation_performed"] is False
    assert "graph_asset_static_registered_replacement_invalid" in registered_refusal[
        "blockers"
    ][0]
    assert not static_output.exists()


def test_static_readback_qualifies_exact_registered_visual_and_graph_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_receipt(tmp_path)
    spec = _spec(source)
    task_freeze = _task_freeze(tmp_path, spec)
    receipt_path = tmp_path / "asset.receipt.json"
    author_simready_graph_asset(
        spec=spec,
        task_freeze_receipt_path=task_freeze,
        source_asset_receipt_path=source,
        destination=tmp_path / "asset.usda",
        measured_derivation=_measured_derivation(),
        receipt_path=receipt_path,
    )
    final_usd = tmp_path / "registered.usda"
    stage = Usd.Stage.Open(str(tmp_path / "asset.usda"), load=Usd.Stage.LoadAll)
    assert stage.Export(str(final_usd))
    stage = Usd.Stage.Open(str(final_usd), load=Usd.Stage.LoadAll)
    root = stage.GetDefaultPrim()
    xform = UsdGeom.Xformable(root)
    xform.ClearXformOpOrder()
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslateOnly(Gf.Vec3d(1.0, 2.0, 3.0))
    xform.AddTransformOp(
        UsdGeom.XformOp.PrecisionDouble, "assetFrameRegistration"
    ).Set(matrix)
    root.SetCustomDataByKey("blueprint:assetFrameRegistrationDigest", "sha256:" + "d" * 64)
    root.SetCustomDataByKey("blueprint:identityOrientationAssumed", False)
    visual = UsdGeom.Mesh.Define(stage, "/Asset/links/root/visuals/body").GetPrim()
    visual.SetCustomDataByKey(
        "blueprint:agentAuthoredDisplayColorRgba", Gf.Vec4d(0.2, 0.3, 0.4, 1.0)
    )
    material = UsdShade.Material.Define(stage, "/Asset/visual_materials/body")
    UsdShade.MaterialBindingAPI.Apply(visual).Bind(material)
    stage.GetRootLayer().Save()

    def record(path: Path) -> dict:
        import hashlib

        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    composition_path = tmp_path / "composition.json"
    composition_path.write_text(json.dumps({"visual_mesh_count": 1}), encoding="utf-8")
    registration_path = tmp_path / "registration.json"
    registration_path.write_text("{}", encoding="utf-8")
    registered = {
        "scene_id": "scene",
        "task_id": spec["task_id"],
        "asset_id": spec["asset_id"],
        "task_freeze_digest": spec["task_freeze_digest"],
        "output_usd": record(final_usd),
        "visual_composition_receipt": {
            **record(composition_path),
            "receipt_digest": "sha256:" + "c" * 64,
        },
        "frame_registration": {
            **record(registration_path),
            "registration_digest": "sha256:" + "d" * 64,
        },
        "T_observed_world_axes_from_asset_local_axes": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "source_root_translation_preserved": [1.0, 2.0, 3.0],
        "receipt_digest": "sha256:" + "e" * 64,
    }
    registered_path = tmp_path / "registered.json"
    registered_path.write_text(json.dumps(registered), encoding="utf-8")
    monkeypatch.setattr(
        "blueprint_pipeline.simready_graph_asset_static_qualification."
        "validate_registered_replacement_asset",
        lambda value: value,
    )
    qualification = qualify_simready_graph_asset_static(
        spec=spec,
        authoring_receipt_path=receipt_path,
        registered_replacement_asset_receipt_path=registered_path,
    )

    assert qualification["status"] == "authored_structure_statically_qualified"
    assert qualification["replacement_usd"]["sha256"] == record(final_usd)["sha256"]
    assert qualification["registered_visual_readback"] == {
        "render_visible_visual_mesh_count": 1,
        "bound_material_visual_mesh_count": 1,
        "agent_authored_color_visual_mesh_count": 1,
        "expected_visual_mesh_count": 1,
        "asset_frame_registration_digest": "sha256:" + "d" * 64,
    }
    assert "visual_material_artifact_unbound" not in qualification["contract_blockers"]


def test_registered_static_readback_rejects_wrong_final_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_receipt(tmp_path)
    spec = _spec(source)
    task_freeze = _task_freeze(tmp_path, spec)
    receipt_path = tmp_path / "asset.receipt.json"
    author_simready_graph_asset(
        spec=spec,
        task_freeze_receipt_path=task_freeze,
        source_asset_receipt_path=source,
        destination=tmp_path / "asset.usda",
        measured_derivation=_measured_derivation(),
        receipt_path=receipt_path,
    )
    final_usd = tmp_path / "registered.usda"
    stage = Usd.Stage.Open(str(tmp_path / "asset.usda"), load=Usd.Stage.LoadAll)
    assert stage.Export(str(final_usd))
    stage = Usd.Stage.Open(str(final_usd), load=Usd.Stage.LoadAll)
    root = stage.GetDefaultPrim()
    UsdGeom.Xformable(root).ClearXformOpOrder()
    UsdGeom.Xformable(root).AddTransformOp(
        UsdGeom.XformOp.PrecisionDouble, "assetFrameRegistration"
    ).Set(Gf.Matrix4d(1.0))
    root.SetCustomDataByKey("blueprint:assetFrameRegistrationDigest", "sha256:" + "d" * 64)
    root.SetCustomDataByKey("blueprint:identityOrientationAssumed", False)
    stage.GetRootLayer().Save()
    import hashlib

    final_record = {
        "path": str(final_usd),
        "size_bytes": final_usd.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(final_usd.read_bytes()).hexdigest(),
    }
    registered_path = tmp_path / "registered.json"
    registered_path.write_text("{}", encoding="utf-8")
    composition_path = tmp_path / "composition.json"
    composition_path.write_text(json.dumps({"visual_mesh_count": 0}), encoding="utf-8")
    monkeypatch.setattr(
        "blueprint_pipeline.simready_graph_asset_static_qualification."
        "validate_registered_replacement_asset",
        lambda value: {
            "task_id": spec["task_id"],
            "asset_id": spec["asset_id"],
            "task_freeze_digest": spec["task_freeze_digest"],
            "output_usd": final_record,
            "T_observed_world_axes_from_asset_local_axes": [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "source_root_translation_preserved": [0.0, 0.0, 0.0],
            "frame_registration": {"registration_digest": "sha256:" + "d" * 64},
            "visual_composition_receipt": {"path": str(composition_path)},
            "receipt_digest": "sha256:" + "e" * 64,
        },
    )
    result = qualify_simready_graph_asset_static(
        spec=spec,
        authoring_receipt_path=receipt_path,
        registered_replacement_asset_receipt_path=registered_path,
    )
    assert "graph_asset_static_registered_world_pose_mismatch" in result["structural_findings"]


@pytest.mark.parametrize(
    ("declared_type", "expected_usd_type", "expected_implementation"),
    [
        ("acceleration", "acceleration", "usd_acceleration_drive"),
        ("none", "force", "passive_force_damper"),
    ],
)
def test_compiler_retains_declared_drive_semantics_and_box_xform_order(
    tmp_path: Path,
    declared_type: str,
    expected_usd_type: str,
    expected_implementation: str,
) -> None:
    source = _source_receipt(tmp_path)
    spec = _spec(source, target=False)
    spec.pop("spec_digest")
    drive = spec["articulation_graph"]["joints"][0]["drive"]
    drive["drive_type"] = declared_type
    if declared_type == "none":
        drive.update({"stiffness": 0.0, "damping": 0.25, "maximum_force": 0.0})
    spec["links"][0]["geometry"][0]["translation_m"] = [0.1, 0.2, 0.3]
    spec["links"][0]["center_of_mass_m"] = [0.1, 0.2, 0.3]
    spec = validate_simready_graph_asset_spec(spec)
    task_freeze = _task_freeze(tmp_path, spec)
    receipt_path = tmp_path / "asset.receipt.json"
    receipt = author_simready_graph_asset(
        spec=spec,
        task_freeze_receipt_path=task_freeze,
        source_asset_receipt_path=source,
        destination=tmp_path / "asset.usda",
        measured_derivation=_measured_derivation(),
        receipt_path=receipt_path,
    )

    stage = Usd.Stage.Open(receipt["output_usd"]["path"])
    joint = stage.GetPrimAtPath("/Asset/joints/hinge")
    drive_api = UsdPhysics.DriveAPI.Get(joint, "angular")
    assert drive_api.GetTypeAttr().Get() == expected_usd_type
    assert joint.GetCustomDataByKey("blueprint:declaredDriveType") == declared_type
    assert (
        joint.GetCustomDataByKey("blueprint:driveImplementation")
        == expected_implementation
    )
    assert receipt["joint_drive_implementations"]["hinge"] == {
        "declared_drive_type": declared_type,
        "usd_drive_authored": True,
        "usd_drive_type": expected_usd_type,
        "implementation": expected_implementation,
    }
    cube = stage.GetPrimAtPath("/Asset/links/root/geometry/root_shape")
    assert [str(op.GetOpName()) for op in UsdGeom.Xformable(cube).GetOrderedXformOps()] == [
        "xformOp:translate",
        "xformOp:orient",
        "xformOp:scale",
    ]
    qualification = qualify_simready_graph_asset_static(
        spec=spec,
        authoring_receipt_path=receipt_path,
    )
    assert qualification["authored_structure_statically_qualified"] is True


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
        measured_derivation=_measured_derivation(),
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
        measured_derivation=_measured_derivation(),
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


@pytest.mark.parametrize(
    ("task", "expected_task_id", "expected_pair_count"),
    [
        ("a", "task_a_washer_door_open", 15),
        ("b", "task_b_notebook_relocation", 1),
    ],
)
def test_third_scene_checked_in_graph_assets_retain_static_claim_boundary(
    task: str, expected_task_id: str, expected_pair_count: int
) -> None:
    manifest_root = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "arm_decision_proof_v1"
        / "manifests"
    )
    prefix = f"third_scene_840920_task_{task}_simready_graph_asset"
    spec = json.loads((manifest_root / f"{prefix}_spec.v1.json").read_text())
    receipt = json.loads((manifest_root / f"{prefix}_receipt.v1.json").read_text())
    qualification = json.loads(
        (manifest_root / f"{prefix}_static_qualification.v1.json").read_text()
    )

    normalized = validate_simready_graph_asset_spec(spec)
    assert normalized["spec_digest"] == spec["spec_digest"]
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    assert qualification["receipt_digest"] == canonical_digest(
        qualification, digest_field="receipt_digest"
    )
    assert receipt["task_id"] == qualification["task_id"] == expected_task_id
    assert receipt["spec_digest"] == qualification["spec_digest"] == spec["spec_digest"]
    assert (
        receipt["output_usd"]["sha256"]
        == qualification["replacement_usd"]["sha256"]
    )
    assert qualification["authored_structure_statically_qualified"] is True
    assert qualification["structural_findings"] == []
    assert (
        qualification["collision_pair_readback"]["complete_pair_count"]
        == expected_pair_count
    )
    assert qualification["claim_boundary"]["native_simulator_import_qualified"] is False
    assert qualification["claim_boundary"]["appearance_materially_qualified"] is False
    assert qualification["claim_boundary"]["physical_equivalence_proven"] is False


def test_a_typed_axis_the_scan_contradicts_cannot_become_an_asset(tmp_path: Path) -> None:
    """Scene 840920's exact bug, now unreachable.

    The spec carried ``+Z`` while the geometry demanded ``-Z``; it authored
    cleanly, sealed into a freeze, and cost two paid runs reading a jammed
    6.01 degrees. Authoring now requires the scan to agree, so the same spec
    is refused before an asset exists.
    """

    source = _source_receipt(tmp_path)
    spec = _spec(source)
    task_freeze = _task_freeze(tmp_path, spec)
    # Spec says +X; the scan measured -X.
    with pytest.raises(SimReadyGraphAssetError) as caught:
        author_simready_graph_asset(
            spec=spec,
            task_freeze_receipt_path=task_freeze,
            source_asset_receipt_path=source,
            destination=tmp_path / "contradicted.usda",
            measured_derivation=_measured_derivation(axis=(-1.0, 0.0, 0.0)),
        )
    assert "graph_asset_target_axis_contradicts_measurement" in caught.value.codes


def test_a_commanded_joint_cannot_be_authored_without_measurement(tmp_path: Path) -> None:
    """Doctrine 5b in code: no measurement, no physical claim."""

    source = _source_receipt(tmp_path)
    spec = _spec(source)
    task_freeze = _task_freeze(tmp_path, spec)
    with pytest.raises(SimReadyGraphAssetError) as caught:
        author_simready_graph_asset(
            spec=spec,
            task_freeze_receipt_path=task_freeze,
            source_asset_receipt_path=source,
            destination=tmp_path / "unmeasured.usda",
        )
    assert "graph_asset_measured_derivation_required" in caught.value.codes


def test_a_tampered_derivation_cannot_launder_a_typed_axis(tmp_path: Path) -> None:
    """Editing the receipt to agree must break its self-digest."""

    source = _source_receipt(tmp_path)
    spec = _spec(source)
    task_freeze = _task_freeze(tmp_path, spec)
    forged = _measured_derivation(axis=(-1.0, 0.0, 0.0))
    forged["target_joint"]["axis"] = [1.0, 0.0, 0.0]  # now "agrees" with the spec
    with pytest.raises(SimReadyGraphAssetError) as caught:
        author_simready_graph_asset(
            spec=spec,
            task_freeze_receipt_path=task_freeze,
            source_asset_receipt_path=source,
            destination=tmp_path / "forged.usda",
            measured_derivation=forged,
        )
    assert "graph_asset_measured_derivation_digest_invalid" in caught.value.codes


def test_the_authored_receipt_carries_its_measurement_proof(tmp_path: Path) -> None:
    source = _source_receipt(tmp_path)
    spec = _spec(source)
    task_freeze = _task_freeze(tmp_path, spec)
    derivation = _measured_derivation()
    receipt = author_simready_graph_asset(
        spec=spec,
        task_freeze_receipt_path=task_freeze,
        source_asset_receipt_path=source,
        destination=tmp_path / "proved.usda",
        measured_derivation=derivation,
    )
    binding = receipt["measured_axis_binding"]
    assert binding["measured_target_axis_required"] is True
    assert binding["measured_derivation_digest"] == derivation["derivation_digest"]
    assert binding["measured_target_axis"] == [1.0, 0.0, 0.0]
    assert binding["facing_proposed_by"] == "test_scene_context"


def test_a_graph_with_no_commanded_joint_needs_no_measurement(tmp_path: Path) -> None:
    """Nothing is commanded, so no physical claim is made to corroborate."""

    source = _source_receipt(tmp_path)
    spec = _spec(source, target=False)
    task_freeze = _task_freeze(tmp_path, spec)
    receipt = author_simready_graph_asset(
        spec=spec,
        task_freeze_receipt_path=task_freeze,
        source_asset_receipt_path=source,
        destination=tmp_path / "locked_only.usda",
    )
    assert receipt["measured_axis_binding"] == {"measured_target_axis_required": False}
