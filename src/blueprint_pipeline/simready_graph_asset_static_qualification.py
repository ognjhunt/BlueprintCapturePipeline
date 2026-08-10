"""Static, task-neutral readback for graph-authored SimReady candidates.

This gate proves only that an exact USD retains the structure and authored
values of its digest-bound graph spec.  It deliberately cannot qualify native
simulator import, dynamics, contact behavior, appearance, or physical
equivalence.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json
from .simready_graph_asset import (
    RECEIPT_SCHEMA as ASSET_RECEIPT_SCHEMA,
    SimReadyGraphAssetError,
    validate_simready_graph_asset_spec,
)


SCHEMA_VERSION = "simready_graph_asset_static_qualification.v1"


class SimReadyGraphAssetStaticQualificationError(ValueError):
    """Stable input/binding errors that prevent a static qualification run."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimReadyGraphAssetStaticQualificationError(
            ["graph_asset_static_authoring_receipt_unreadable"]
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != ASSET_RECEIPT_SCHEMA
        or value.get("status") != "simready_candidate_authored"
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
    ):
        raise SimReadyGraphAssetStaticQualificationError(
            ["graph_asset_static_authoring_receipt_invalid"]
        )
    return value


def _numbers(value: Any) -> list[float]:
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return []


def _close(actual: Any, expected: Any, *, tolerance: float = 1e-5) -> bool:
    if isinstance(expected, (list, tuple)):
        values = _numbers(actual)
        return len(values) == len(expected) and all(
            math.isclose(value, float(wanted), abs_tol=tolerance, rel_tol=0.0)
            for value, wanted in zip(values, expected, strict=True)
        )
    try:
        return math.isclose(
            float(actual), float(expected), abs_tol=tolerance, rel_tol=0.0
        )
    except (TypeError, ValueError):
        return False


def _quat_xyzw(value: Any) -> list[float]:
    if value is None:
        return []
    imaginary = value.GetImaginary()
    return [
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
        float(value.GetReal()),
    ]


def _xform_ops(prim: Any) -> tuple[list[str], dict[str, Any]]:
    from pxr import UsdGeom

    ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
    return [str(op.GetOpName()) for op in ops], {
        str(op.GetOpName()): op.Get() for op in ops
    }


def _rotation_matrix(xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(item) for item in xyzw)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _geometry_aabb(geometry: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(geometry["translation_m"], dtype=np.float64)
    rotation = _rotation_matrix(geometry["orientation_xyzw"])
    if geometry["kind"] == "box":
        extent = np.abs(rotation) @ (
            np.asarray(geometry["size_m"], dtype=np.float64) / 2.0
        )
    else:
        half_height = float(geometry["height_m"]) / 2.0
        radius = float(geometry["radius_m"])
        axis = rotation[:, 0]
        extent = np.abs(axis) * half_height + radius * np.sqrt(
            np.maximum(0.0, 1.0 - axis * axis)
        )
    return center - extent, center + extent


def _custom(prim: Any, key: str) -> Any:
    return prim.GetCustomDataByKey(key) if prim and prim.IsValid() else None


def qualify_simready_graph_asset_static(
    *,
    spec: Mapping[str, Any],
    authoring_receipt_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Re-open and compare every authored graph field against its sealed spec."""

    try:
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SimReadyGraphAssetStaticQualificationError(
            ["graph_asset_static_openusd_runtime_missing"]
        ) from exc

    try:
        admitted = validate_simready_graph_asset_spec(spec)
    except SimReadyGraphAssetError as exc:
        raise SimReadyGraphAssetStaticQualificationError(exc.codes) from exc
    receipt_path = Path(authoring_receipt_path).expanduser().resolve()
    receipt = _load_receipt(receipt_path)
    if (
        receipt.get("asset_id") != admitted["asset_id"]
        or receipt.get("task_id") != admitted["task_id"]
        or receipt.get("task_freeze_digest") != admitted["task_freeze_digest"]
        or receipt.get("spec_digest") != admitted["spec_digest"]
    ):
        raise SimReadyGraphAssetStaticQualificationError(
            ["graph_asset_static_authoring_receipt_binding_mismatch"]
        )
    output_usd = receipt.get("output_usd") or {}
    usd_path = Path(str(output_usd.get("path") or "")).expanduser().resolve()
    if (
        not usd_path.is_file()
        or usd_path.is_symlink()
        or usd_path.stat().st_size != output_usd.get("size_bytes")
        or _sha256(usd_path) != output_usd.get("sha256")
    ):
        raise SimReadyGraphAssetStaticQualificationError(
            ["graph_asset_static_usd_bytes_changed"]
        )
    stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise SimReadyGraphAssetStaticQualificationError(
            ["graph_asset_static_usd_unreadable"]
        )

    findings: list[str] = []
    if (
        float(UsdGeom.GetStageMetersPerUnit(stage)) != 1.0
        or str(UsdGeom.GetStageUpAxis(stage)).upper() != "Z"
        or str(stage.GetDefaultPrim().GetPath()) != "/Asset"
    ):
        findings.append("graph_asset_static_stage_frame_mismatch")
    root = stage.GetPrimAtPath("/Asset")
    if (
        not root.IsValid()
        or not root.HasAPI(UsdPhysics.ArticulationRootAPI)
        or _custom(root, "blueprint:assetId") != admitted["asset_id"]
        or _custom(root, "blueprint:specDigest") != admitted["spec_digest"]
    ):
        findings.append("graph_asset_static_root_binding_mismatch")
    root_order, root_ops = _xform_ops(root)
    if root_order != ["xformOp:translate", "xformOp:orient"]:
        findings.append("graph_asset_static_root_xform_order_mismatch")
    if not _close(
        root_ops.get("xformOp:translate"), admitted["world_pose"]["translation_m"]
    ) or not _close(
        _quat_xyzw(root_ops.get("xformOp:orient")),
        admitted["world_pose"]["orientation_xyzw"],
    ):
        findings.append("graph_asset_static_world_pose_mismatch")

    graph_links = {
        row["link_id"]: row for row in admitted["articulation_graph"]["links"]
    }
    link_rows: list[dict[str, Any]] = []
    expected_link_paths = {f"/Asset/links/{link_id}" for link_id in graph_links}
    actual_link_paths = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetParent().GetPath() == root.GetPath().AppendChild("links")
    }
    if actual_link_paths != expected_link_paths:
        findings.append("graph_asset_static_link_set_mismatch")
    for link in admitted["links"]:
        link_id = link["link_id"]
        prim = stage.GetPrimAtPath(f"/Asset/links/{link_id}")
        link_findings: list[str] = []
        if (
            not prim.IsValid()
            or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or not prim.HasAPI(UsdPhysics.MassAPI)
            or _custom(prim, "blueprint:semanticRole")
            != graph_links[link_id]["semantic_role"]
        ):
            link_findings.append("apis_or_semantic_role_mismatch")
        order, ops = _xform_ops(prim)
        if order != ["xformOp:translate", "xformOp:orient"]:
            link_findings.append("xform_order_mismatch")
        if not _close(
            ops.get("xformOp:translate"), link["rest_pose"]["translation_m"]
        ) or not _close(
            _quat_xyzw(ops.get("xformOp:orient")),
            link["rest_pose"]["orientation_xyzw"],
        ):
            link_findings.append("rest_pose_mismatch")
        rigid = UsdPhysics.RigidBodyAPI(prim)
        expected_kinematic = bool(
            graph_links[link_id]["is_root"]
            and admitted["root_body_mode"] == "fixed"
        )
        if bool(rigid.GetKinematicEnabledAttr().Get() or False) != expected_kinematic:
            link_findings.append("root_body_mode_mismatch")
        mass = UsdPhysics.MassAPI(prim)
        if not _close(mass.GetMassAttr().Get(), link["mass_kg"]):
            link_findings.append("mass_mismatch")
        if not _close(
            mass.GetCenterOfMassAttr().Get(), link["center_of_mass_m"]
        ):
            link_findings.append("center_of_mass_mismatch")
        if not _close(
            mass.GetDiagonalInertiaAttr().Get(), link["diagonal_inertia_kg_m2"]
        ):
            link_findings.append("diagonal_inertia_mismatch")
        inertia = [float(value) for value in link["diagonal_inertia_kg_m2"]]
        if any(
            inertia[index] > sum(inertia) - inertia[index] + 1e-9
            for index in range(3)
        ):
            link_findings.append("inertia_triangle_inequality_invalid")

        expected_geometry_paths = {
            f"/Asset/links/{link_id}/geometry/{row['geometry_id']}"
            for row in link["geometry"]
        }
        actual_geometry_paths = {
            str(child.GetPath())
            for child in stage.GetPrimAtPath(
                f"/Asset/links/{link_id}/geometry"
            ).GetChildren()
        }
        if actual_geometry_paths != expected_geometry_paths:
            link_findings.append("geometry_set_mismatch")
        bounds = [_geometry_aabb(row) for row in link["geometry"]]
        lower = np.min(np.stack([row[0] for row in bounds]), axis=0)
        upper = np.max(np.stack([row[1] for row in bounds]), axis=0)
        center_of_mass = np.asarray(link["center_of_mass_m"], dtype=np.float64)
        if np.any(center_of_mass < lower - 1e-6) or np.any(
            center_of_mass > upper + 1e-6
        ):
            link_findings.append("center_of_mass_outside_collision_bounds")
        for geometry in link["geometry"]:
            geometry_id = geometry["geometry_id"]
            geometry_prim = stage.GetPrimAtPath(
                f"/Asset/links/{link_id}/geometry/{geometry_id}"
            )
            geometry_findings: list[str] = []
            expected_type = "Cube" if geometry["kind"] == "box" else "Cylinder"
            if (
                not geometry_prim.IsValid()
                or geometry_prim.GetTypeName() != expected_type
                or not geometry_prim.HasAPI(UsdPhysics.CollisionAPI)
                or _custom(geometry_prim, "blueprint:geometryProvenance")
                != geometry["provenance"]
            ):
                geometry_findings.append("type_collision_or_provenance_mismatch")
            geometry_order, geometry_ops = _xform_ops(geometry_prim)
            expected_order = ["xformOp:translate", "xformOp:orient"]
            if geometry["kind"] == "box":
                expected_order.append("xformOp:scale")
            if geometry_order != expected_order:
                geometry_findings.append("xform_order_mismatch")
            if not _close(
                geometry_ops.get("xformOp:translate"), geometry["translation_m"]
            ) or not _close(
                _quat_xyzw(geometry_ops.get("xformOp:orient")),
                geometry["orientation_xyzw"],
            ):
                geometry_findings.append("pose_mismatch")
            if geometry["kind"] == "box":
                cube = UsdGeom.Cube(geometry_prim)
                if not _close(cube.GetSizeAttr().Get(), 1.0) or not _close(
                    geometry_ops.get("xformOp:scale"), geometry["size_m"]
                ):
                    geometry_findings.append("dimensions_mismatch")
            else:
                cylinder = UsdGeom.Cylinder(geometry_prim)
                if (
                    str(cylinder.GetAxisAttr().Get()).upper() != "X"
                    or not _close(cylinder.GetRadiusAttr().Get(), geometry["radius_m"])
                    or not _close(cylinder.GetHeightAttr().Get(), geometry["height_m"])
                ):
                    geometry_findings.append("dimensions_mismatch")
            color = UsdGeom.Gprim(geometry_prim).GetDisplayColorAttr().Get()
            if not color or not _close(color[0], geometry["display_color_rgb"]):
                geometry_findings.append("display_color_mismatch")
            bound_material, _ = UsdShade.MaterialBindingAPI(
                geometry_prim
            ).ComputeBoundMaterial(materialPurpose="physics")
            expected_material_path = f"/Asset/materials/{link_id}"
            if (
                not bound_material
                or str(bound_material.GetPath()) != expected_material_path
            ):
                geometry_findings.append("physics_material_binding_mismatch")
            if geometry_findings:
                findings.extend(
                    f"graph_asset_static_geometry_{item}:{link_id}:{geometry_id}"
                    for item in geometry_findings
                )
        material_prim = stage.GetPrimAtPath(f"/Asset/materials/{link_id}")
        material_api = UsdPhysics.MaterialAPI(material_prim)
        if (
            not material_prim.IsValid()
            or not material_prim.HasAPI(UsdPhysics.MaterialAPI)
            or not _close(material_api.GetStaticFrictionAttr().Get(), link["friction"])
            or not _close(material_api.GetDynamicFrictionAttr().Get(), link["friction"])
            or not _close(material_api.GetRestitutionAttr().Get(), link["restitution"])
        ):
            link_findings.append("physics_material_values_mismatch")
        findings.extend(
            f"graph_asset_static_link_{item}:{link_id}" for item in link_findings
        )
        link_rows.append(
            {
                "link_id": link_id,
                "finding_codes": sorted(link_findings),
                "center_of_mass_inside_collision_aabb": (
                    "center_of_mass_outside_collision_bounds" not in link_findings
                ),
                "inertia_triangle_inequality_passed": (
                    "inertia_triangle_inequality_invalid" not in link_findings
                ),
            }
        )

    graph_joints = {
        row["joint_id"]: row for row in admitted["articulation_graph"]["joints"]
    }
    frames = {row["joint_id"]: row for row in admitted["joint_frames"]}
    expected_joint_paths = {f"/Asset/joints/{joint_id}" for joint_id in graph_joints}
    actual_joint_paths = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Joint)
    }
    if actual_joint_paths != expected_joint_paths:
        findings.append("graph_asset_static_joint_set_mismatch")
    joint_rows: list[dict[str, Any]] = []
    expected_types = {
        "revolute": "PhysicsRevoluteJoint",
        "continuous": "PhysicsRevoluteJoint",
        "prismatic": "PhysicsPrismaticJoint",
        "fixed": "PhysicsFixedJoint",
    }
    for joint_id, joint in graph_joints.items():
        prim = stage.GetPrimAtPath(f"/Asset/joints/{joint_id}")
        joint_findings: list[str] = []
        if not prim.IsValid() or prim.GetTypeName() != expected_types[joint["joint_type"]]:
            joint_findings.append("type_mismatch")
        typed = UsdPhysics.Joint(prim)
        expected_bodies = (
            f"/Asset/links/{joint['parent_link_id']}",
            f"/Asset/links/{joint['child_link_id']}",
        )
        actual_bodies = (
            str(typed.GetBody0Rel().GetTargets()[0])
            if len(typed.GetBody0Rel().GetTargets()) == 1
            else "",
            str(typed.GetBody1Rel().GetTargets()[0])
            if len(typed.GetBody1Rel().GetTargets()) == 1
            else "",
        )
        if actual_bodies != expected_bodies:
            joint_findings.append("body_relationship_mismatch")
        frame = frames[joint_id]
        if (
            not _close(typed.GetLocalPos0Attr().Get(), frame["parent_position_m"])
            or not _close(typed.GetLocalPos1Attr().Get(), frame["child_position_m"])
            or not _close(
                _quat_xyzw(typed.GetLocalRot0Attr().Get()),
                frame["parent_orientation_xyzw"],
            )
            or not _close(
                _quat_xyzw(typed.GetLocalRot1Attr().Get()),
                frame["child_orientation_xyzw"],
            )
        ):
            joint_findings.append("frame_mismatch")
        if (
            _custom(prim, "blueprint:jointRole") != joint["role"]
            or not _close(_custom(prim, "blueprint:graphAxis"), joint["axis"])
            or not _close(
                _custom(prim, "blueprint:resetPosition"), joint["reset_position"]
            )
        ):
            joint_findings.append("graph_metadata_mismatch")
        if joint["joint_type"] != "fixed":
            expected_axis = "X"
            actual_axis = (
                UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
                if joint["joint_type"] in {"revolute", "continuous"}
                else UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get()
            )
            if str(actual_axis).upper() != expected_axis:
                joint_findings.append("usd_axis_mismatch")
        if joint["joint_type"] == "revolute":
            revolute = UsdPhysics.RevoluteJoint(prim)
            if not _close(
                revolute.GetLowerLimitAttr().Get(), math.degrees(joint["limits"][0])
            ) or not _close(
                revolute.GetUpperLimitAttr().Get(), math.degrees(joint["limits"][1])
            ):
                joint_findings.append("limits_mismatch")
        elif joint["joint_type"] == "prismatic":
            prismatic = UsdPhysics.PrismaticJoint(prim)
            if not _close(
                prismatic.GetLowerLimitAttr().Get(), joint["limits"][0]
            ) or not _close(
                prismatic.GetUpperLimitAttr().Get(), joint["limits"][1]
            ):
                joint_findings.append("limits_mismatch")
        drive = joint["drive"]
        drive_name = "linear" if joint["joint_type"] == "prismatic" else "angular"
        authored_drive = UsdPhysics.DriveAPI.Get(prim, drive_name)
        drive_expected = joint["joint_type"] != "fixed" and (
            drive["stiffness"] > 0.0 or drive["damping"] > 0.0
        )
        expected_drive_type = (
            drive["drive_type"]
            if drive["drive_type"] in {"force", "acceleration"}
            else "force"
        )
        if _custom(prim, "blueprint:declaredDriveType") != drive["drive_type"]:
            joint_findings.append("declared_drive_type_metadata_mismatch")
        if drive_expected:
            expected_target = joint["reset_position"]
            if joint["joint_type"] != "prismatic":
                expected_target = math.degrees(expected_target)
            if (
                not authored_drive
                or str(authored_drive.GetTypeAttr().Get()) != expected_drive_type
                or not _close(
                    authored_drive.GetStiffnessAttr().Get(), drive["stiffness"]
                )
                or not _close(authored_drive.GetDampingAttr().Get(), drive["damping"])
                or not _close(
                    authored_drive.GetTargetPositionAttr().Get(), expected_target
                )
                or (
                    drive["maximum_force"] > 0.0
                    and not _close(
                        authored_drive.GetMaxForceAttr().Get(), drive["maximum_force"]
                    )
                )
            ):
                joint_findings.append("drive_mismatch")
            expected_implementation = (
                "passive_force_damper"
                if drive["drive_type"] == "none"
                else f"usd_{expected_drive_type}_drive"
            )
            if (
                _custom(prim, "blueprint:driveImplementation")
                != expected_implementation
            ):
                joint_findings.append("drive_implementation_metadata_mismatch")
        elif authored_drive:
            joint_findings.append("unexpected_drive")
        findings.extend(
            f"graph_asset_static_joint_{item}:{joint_id}"
            for item in joint_findings
        )
        joint_rows.append(
            {"joint_id": joint_id, "finding_codes": sorted(joint_findings)}
        )

    filtered_pairs: set[tuple[str, str]] = set()
    for link_id in graph_links:
        prim = stage.GetPrimAtPath(f"/Asset/links/{link_id}")
        if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            continue
        for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
            target_link = target.name
            filtered_pairs.add(tuple(sorted((link_id, target_link))))
    expected_filtered = {
        tuple(sorted((row["link_a"], row["link_b"])))
        for row in admitted["articulation_graph"]["collision_pairs"]
        if not row["collision_enabled"]
    }
    if filtered_pairs != expected_filtered:
        findings.append("graph_asset_static_collision_filter_mismatch")

    all_pairs = set(itertools.combinations(sorted(graph_links), 2))
    declared_pairs = {
        tuple(sorted((row["link_a"], row["link_b"])))
        for row in admitted["articulation_graph"]["collision_pairs"]
    }
    contract_blockers: list[str] = []
    if declared_pairs != all_pairs:
        contract_blockers.append(
            "collision_pair_matrix_incomplete:"
            f"declared={len(declared_pairs)}:required={len(all_pairs)}"
        )
    contract_blockers.extend(
        [
            "visual_material_artifact_unbound",
            "texture_artifact_unbound",
            "collision_approximation_contract_unbound",
            "native_simulator_import_unexecuted",
            "joint_physics_behavior_unexecuted",
        ]
    )
    findings = sorted(set(findings))
    receipt_value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "authored_structure_statically_qualified"
            if not findings
            else "blocked_authored_structure_mismatch"
        ),
        "asset_id": admitted["asset_id"],
        "task_id": admitted["task_id"],
        "task_freeze_digest": admitted["task_freeze_digest"],
        "spec_digest": admitted["spec_digest"],
        "authoring_receipt": {
            "path": str(receipt_path),
            "sha256": _sha256(receipt_path),
            "receipt_digest": receipt["receipt_digest"],
        },
        "replacement_usd": {
            "path": str(usd_path),
            "size_bytes": usd_path.stat().st_size,
            "sha256": _sha256(usd_path),
        },
        "authored_structure_statically_qualified": not findings,
        "structural_findings": findings,
        "contract_blockers": sorted(set(contract_blockers)),
        "link_readback": link_rows,
        "joint_readback": joint_rows,
        "collision_pair_readback": {
            "declared_pair_count": len(declared_pairs),
            "complete_pair_count": len(all_pairs),
            "filtered_pairs": [list(pair) for pair in sorted(filtered_pairs)],
        },
        "claim_boundary": {
            "authored_usd_structure_only": True,
            "native_simulator_import_qualified": False,
            "joint_physics_behavior_qualified": False,
            "appearance_materially_qualified": False,
            "collision_approximation_qualified": False,
            "contact_or_support_qualified": False,
            "physical_equivalence_proven": False,
        },
        "receipt_digest": "",
    }
    receipt_value["receipt_digest"] = canonical_digest(
        receipt_value, digest_field="receipt_digest"
    )
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(canonical_json(receipt_value) + "\n", encoding="utf-8")
    return receipt_value


__all__ = [
    "SCHEMA_VERSION",
    "SimReadyGraphAssetStaticQualificationError",
    "qualify_simready_graph_asset_static",
]
