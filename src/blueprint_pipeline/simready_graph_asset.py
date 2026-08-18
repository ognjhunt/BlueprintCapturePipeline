"""Author a task-neutral SimReady candidate from a sealed joint-graph asset spec.

The compiler never infers links or mechanisms from a semantic class.  All
geometry, joint frames, physics values, and observed-versus-generated labels
come from a digest-bound data manifest.  Its receipt proves deterministic USD
authoring and source joins only; native simulator behavior and physical
equivalence remain downstream qualifications.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .articulation_graph_contract import validate_articulation_graph
from .decision_evidence_contracts import canonical_digest, canonical_json
from .measured_articulation_derivation import (
    SCHEMA_VERSION as MEASURED_DERIVATION_SCHEMA,
)


SPEC_SCHEMA = "simready_graph_asset_spec.v1"
RECEIPT_SCHEMA = "simready_graph_asset_receipt.v1"
GEOMETRY_KINDS = frozenset({"box", "cylinder"})
PROVENANCE_LABELS = frozenset(
    {
        "observed_exterior_derived_candidate",
        "observed_bounds_derived_candidate",
        "generated_candidate",
    }
)


class SimReadyGraphAssetError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _number(value: Any, *, positive: bool = False, nonnegative: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    if positive and result <= 0.0:
        return None
    if nonnegative and result < 0.0:
        return None
    return result


def _vector(value: Any, size: int, *, positive: bool = False) -> list[float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != size
    ):
        return None
    values = [_number(item, positive=positive) for item in value]
    if any(item is None for item in values):
        return None
    return [float(item) for item in values]


def _quaternion(value: Any) -> list[float] | None:
    values = _vector(value, 4)
    if values is None:
        return None
    norm = math.sqrt(sum(component * component for component in values))
    if abs(norm - 1.0) > 1e-6:
        return None
    return values


def _rotated_x_axis(xyzw: Sequence[float]) -> list[float]:
    x, y, z, w = (float(component) for component in xyzw)
    return [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ]


def _load_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimReadyGraphAssetError([code]) from exc
    if not isinstance(value, dict):
        raise SimReadyGraphAssetError([code])
    return value


def validate_simready_graph_asset_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SimReadyGraphAssetError(["graph_asset_spec_not_json"]) from exc
    supplied = payload.pop("spec_digest", None)
    errors: list[str] = []
    if payload.get("schema_version") != SPEC_SCHEMA:
        errors.append("graph_asset_spec_schema_invalid")
    for field in ("asset_id", "task_id", "source_object_instance_id"):
        if not str(payload.get(field) or ""):
            errors.append(f"graph_asset_{field}_missing")
    if payload.get("task_kind") not in {
        "articulated_interaction",
        "rigid_object_manipulation",
    }:
        errors.append("graph_asset_task_kind_invalid")
    for field in ("task_freeze_digest", "source_asset_receipt_digest"):
        if not _is_digest(payload.get(field)):
            errors.append(f"graph_asset_{field}_invalid")
    if payload.get("root_body_mode") not in {"fixed", "dynamic"}:
        errors.append("graph_asset_root_body_mode_invalid")
    world_pose = payload.get("world_pose")
    if not isinstance(world_pose, Mapping) or _vector(
        world_pose.get("translation_m"), 3
    ) is None or _quaternion(world_pose.get("orientation_xyzw")) is None:
        errors.append("graph_asset_world_pose_invalid")
    try:
        graph = validate_articulation_graph(
            payload.get("articulation_graph") or {},
            require_target_joint=payload.get("task_kind") == "articulated_interaction",
        )
    except ValueError as exc:
        graph = {}
        errors.extend(str(exc).split(";"))
    link_ids = {row["link_id"] for row in graph.get("links", [])}
    links = payload.get("links")
    if not isinstance(links, list) or not links:
        errors.append("graph_asset_links_missing")
        links = []
    asset_link_ids = [str(row.get("link_id") or "") for row in links if isinstance(row, Mapping)]
    if len(asset_link_ids) != len(links) or set(asset_link_ids) != link_ids:
        errors.append("graph_asset_link_set_mismatch")
    geometry_ids: set[str] = set()
    normalized_links: list[dict[str, Any]] = []
    for raw in links:
        if not isinstance(raw, Mapping):
            continue
        link_id = str(raw.get("link_id") or "")
        mass = _number(raw.get("mass_kg"), positive=True)
        com = _vector(raw.get("center_of_mass_m"), 3)
        inertia = _vector(raw.get("diagonal_inertia_kg_m2"), 3, positive=True)
        friction = _number(raw.get("friction"), nonnegative=True)
        restitution = _number(raw.get("restitution"), nonnegative=True)
        physics_provenance = str(raw.get("physics_provenance") or "")
        rest_pose = raw.get("rest_pose")
        rest_translation = (
            _vector(rest_pose.get("translation_m"), 3)
            if isinstance(rest_pose, Mapping)
            else None
        )
        rest_orientation = (
            _quaternion(rest_pose.get("orientation_xyzw"))
            if isinstance(rest_pose, Mapping)
            else None
        )
        if (
            mass is None
            or com is None
            or inertia is None
            or friction is None
            or restitution is None
            or restitution > 1.0
            or physics_provenance
            not in {"authored_estimate_unqualified", "measured", "provider_declared"}
            or rest_translation is None
            or rest_orientation is None
        ):
            errors.append(f"graph_asset_link_physics_invalid:{link_id}")
        geometry = raw.get("geometry")
        if not isinstance(geometry, list) or not geometry:
            errors.append(f"graph_asset_link_geometry_missing:{link_id}")
            geometry = []
        normalized_geometry: list[dict[str, Any]] = []
        for index, item in enumerate(geometry):
            if not isinstance(item, Mapping):
                errors.append(f"graph_asset_geometry_invalid:{link_id}:{index}")
                continue
            geometry_id = str(item.get("geometry_id") or "")
            kind = str(item.get("kind") or "")
            translation = _vector(item.get("translation_m"), 3)
            orientation = _quaternion(item.get("orientation_xyzw"))
            color = _vector(item.get("display_color_rgb"), 3)
            provenance = str(item.get("provenance") or "")
            dimensions: dict[str, Any]
            if kind == "box":
                size = _vector(item.get("size_m"), 3, positive=True)
                dimensions = {"size_m": size}
                valid_dimensions = size is not None
            elif kind == "cylinder":
                radius = _number(item.get("radius_m"), positive=True)
                height = _number(item.get("height_m"), positive=True)
                dimensions = {"radius_m": radius, "height_m": height}
                valid_dimensions = radius is not None and height is not None
            else:
                dimensions = {}
                valid_dimensions = False
            if (
                not geometry_id
                or geometry_id in geometry_ids
                or kind not in GEOMETRY_KINDS
                or translation is None
                or orientation is None
                or color is None
                or any(not 0.0 <= component <= 1.0 for component in color)
                or provenance not in PROVENANCE_LABELS
                or not valid_dimensions
            ):
                errors.append(f"graph_asset_geometry_invalid:{link_id}:{index}")
            geometry_ids.add(geometry_id)
            normalized_geometry.append(
                {
                    "geometry_id": geometry_id,
                    "kind": kind,
                    "translation_m": translation or [0.0, 0.0, 0.0],
                    "orientation_xyzw": orientation or [0.0, 0.0, 0.0, 1.0],
                    "display_color_rgb": color or [0.5, 0.5, 0.5],
                    "provenance": provenance,
                    **dimensions,
                }
            )
        normalized_links.append(
            {
                "link_id": link_id,
                "mass_kg": mass or 0.0,
                "center_of_mass_m": com or [0.0, 0.0, 0.0],
                "diagonal_inertia_kg_m2": inertia or [0.0, 0.0, 0.0],
                "friction": friction or 0.0,
                "restitution": restitution or 0.0,
                "physics_provenance": physics_provenance,
                "rest_pose": {
                    "translation_m": rest_translation or [0.0, 0.0, 0.0],
                    "orientation_xyzw": rest_orientation or [0.0, 0.0, 0.0, 1.0],
                },
                "geometry": normalized_geometry,
            }
        )
    frames = payload.get("joint_frames")
    if not isinstance(frames, list):
        errors.append("graph_asset_joint_frames_missing")
        frames = []
    joint_ids = {row["joint_id"] for row in graph.get("joints", [])}
    frame_ids = [str(row.get("joint_id") or "") for row in frames if isinstance(row, Mapping)]
    if len(frame_ids) != len(frames) or set(frame_ids) != joint_ids:
        errors.append("graph_asset_joint_frame_set_mismatch")
    normalized_frames: list[dict[str, Any]] = []
    joint_by_id = {row["joint_id"]: row for row in graph.get("joints", [])}
    for raw in frames:
        if not isinstance(raw, Mapping):
            continue
        joint_id = str(raw.get("joint_id") or "")
        parent_position = _vector(raw.get("parent_position_m"), 3)
        child_position = _vector(raw.get("child_position_m"), 3)
        parent_orientation = _quaternion(raw.get("parent_orientation_xyzw"))
        child_orientation = _quaternion(raw.get("child_orientation_xyzw"))
        if any(
            value is None
            for value in (
                parent_position,
                child_position,
                parent_orientation,
                child_orientation,
            )
        ):
            errors.append(f"graph_asset_joint_frame_invalid:{joint_id}")
        elif joint_id in joint_by_id and joint_by_id[joint_id]["joint_type"] != "fixed":
            axis = joint_by_id[joint_id]["axis"]
            axis_norm = math.sqrt(sum(component * component for component in axis))
            expected_axis = [component / axis_norm for component in axis]
            for role, orientation in (
                ("parent", parent_orientation),
                ("child", child_orientation),
            ):
                authored_axis = _rotated_x_axis(orientation)
                if sum(
                    authored_axis[index] * expected_axis[index] for index in range(3)
                ) < 1.0 - 1e-6:
                    errors.append(
                        f"graph_asset_joint_frame_axis_mismatch:{joint_id}:{role}"
                    )
        normalized_frames.append(
            {
                "joint_id": joint_id,
                "parent_position_m": parent_position or [0.0, 0.0, 0.0],
                "child_position_m": child_position or [0.0, 0.0, 0.0],
                "parent_orientation_xyzw": parent_orientation or [0.0, 0.0, 0.0, 1.0],
                "child_orientation_xyzw": child_orientation or [0.0, 0.0, 0.0, 1.0],
            }
        )
    if payload.get("appearance_materially_qualified") is not False:
        errors.append("graph_asset_appearance_self_qualification_forbidden")
    if payload.get("physical_equivalence_claimed") is not False:
        errors.append("graph_asset_physical_equivalence_forbidden")
    if errors:
        raise SimReadyGraphAssetError(errors)
    normalized = {
        **payload,
        "articulation_graph": graph,
        "links": normalized_links,
        "joint_frames": normalized_frames,
    }
    expected = canonical_digest(normalized, digest_field="spec_digest")
    if supplied is not None and supplied != expected:
        raise SimReadyGraphAssetError(["graph_asset_spec_digest_mismatch"])
    normalized["spec_digest"] = expected
    return normalized


def _verify_source_asset(receipt_path: Path, expected_digest: str) -> dict[str, Any]:
    receipt = _load_json(receipt_path, "graph_asset_source_receipt_unreadable")
    if (
        receipt.get("schema_version") != "articulated_source_asset.v1"
        or receipt.get("status") != "materialized"
        or receipt.get("receipt_digest") != expected_digest
        or canonical_digest(receipt, digest_field="receipt_digest") != expected_digest
    ):
        raise SimReadyGraphAssetError(["graph_asset_source_receipt_invalid"])
    output = receipt.get("output_asset") or {}
    asset_path = receipt_path.parent / str(output.get("relative_path") or "")
    if (
        not asset_path.is_file()
        or asset_path.is_symlink()
        or _sha256(asset_path) != output.get("sha256")
        or asset_path.stat().st_size != output.get("size_bytes")
    ):
        raise SimReadyGraphAssetError(["graph_asset_source_asset_bytes_changed"])
    return receipt


def _verify_task_freeze(
    receipt_path: Path, *, admitted: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _load_json(receipt_path, "graph_asset_task_freeze_unreadable")
    digest = admitted["task_freeze_digest"]
    graph = receipt.get("articulation_graph") or {}
    try:
        normalized_graph = validate_articulation_graph(
            graph,
            require_target_joint=admitted["task_kind"] == "articulated_interaction",
        )
    except ValueError:
        normalized_graph = {}
    removal = receipt.get("removal_plan") or {}
    if (
        receipt.get("schema_version") != "dual_task_task_freeze.v1"
        or receipt.get("task_freeze_digest") != digest
        or canonical_digest(receipt, digest_field="task_freeze_digest") != digest
        or receipt.get("task_id") != admitted["task_id"]
        or receipt.get("task_kind") != admitted["task_kind"]
        or str((receipt.get("source_object") or {}).get("instance_id") or "")
        != admitted["source_object_instance_id"]
        or removal.get("replacement_asset_id") != admitted["asset_id"]
        or canonical_json(normalized_graph)
        != canonical_json(admitted["articulation_graph"])
    ):
        raise SimReadyGraphAssetError(["graph_asset_task_freeze_invalid"])
    return receipt


def _gf_quat(xyzw: Sequence[float]):
    from pxr import Gf

    return Gf.Quatf(float(xyzw[3]), Gf.Vec3f(*[float(value) for value in xyzw[:3]]))


def _require_measured_target_axis(
    spec: Mapping[str, Any], measured_derivation: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Refuse any target joint the scan does not corroborate.

    Evidence ladder 5b makes measurement the author of physical claims, and a
    doctrine nothing enforces is a preference.  Scene 840920's door was sealed
    with a hand-typed ``+Z`` where the geometry demands ``-Z``; two paid runs
    read the jammed 6.01 degrees.  A spec may still *carry* an axis -- it is
    the document the freeze seals -- but authoring now requires a derivation
    receipt computed from the digest-bound scan, and the two must agree.  A
    typed axis that measurement contradicts cannot become an asset.

    Only the target joint is gated: it is the joint the task commands and the
    one whose sign the derivation computes from clearance.
    """

    joints = ((spec.get("articulation_graph") or {}).get("joints")) or []
    targets = [
        j for j in joints if isinstance(j, Mapping) and j.get("role") == "target"
    ]
    if not targets:
        # A graph with no commanded joint asserts no physical claim to check.
        return {"measured_target_axis_required": False}
    if measured_derivation is None:
        raise SimReadyGraphAssetError(["graph_asset_measured_derivation_required"])
    if not isinstance(measured_derivation, Mapping):
        raise SimReadyGraphAssetError(["graph_asset_measured_derivation_invalid"])

    errors: list[str] = []
    if measured_derivation.get("schema_version") != MEASURED_DERIVATION_SCHEMA:
        errors.append("graph_asset_measured_derivation_schema_invalid")
    if measured_derivation.get("status") != "derived_from_measurement":
        errors.append("graph_asset_measured_derivation_status_invalid")
    if measured_derivation.get("derivation_digest") != canonical_digest(
        measured_derivation, digest_field="derivation_digest"
    ):
        errors.append("graph_asset_measured_derivation_digest_invalid")
    if errors:
        raise SimReadyGraphAssetError(errors)

    derived = ((measured_derivation.get("target_joint") or {}).get("axis")) or []
    derived_axis = [float(v) for v in derived] if len(derived) == 3 else None
    if derived_axis is None:
        raise SimReadyGraphAssetError(["graph_asset_measured_derivation_axis_invalid"])
    for joint in targets:
        axis = joint.get("axis") or []
        if len(axis) != 3:
            raise SimReadyGraphAssetError(["graph_asset_target_axis_invalid"])
        typed = [float(v) for v in axis]
        if any(abs(a - b) > 1e-6 for a, b in zip(typed, derived_axis)):
            raise SimReadyGraphAssetError(
                ["graph_asset_target_axis_contradicts_measurement"]
            )
    return {
        "measured_target_axis_required": True,
        "measured_derivation_digest": str(
            measured_derivation.get("derivation_digest") or ""
        ),
        "measured_target_axis": derived_axis,
        "facing_proposed_by": str(
            (measured_derivation.get("front_plate") or {}).get(
                "facing_proposed_by"
            )
            or ""
        ),
    }


def author_simready_graph_asset(
    *,
    spec: Mapping[str, Any],
    task_freeze_receipt_path: str | Path,
    source_asset_receipt_path: str | Path,
    destination: str | Path,
    measured_derivation: Mapping[str, Any] | None = None,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    measured_axis_binding = _require_measured_target_axis(spec, measured_derivation)
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:
        raise SimReadyGraphAssetError(["graph_asset_openusd_runtime_missing"]) from exc

    admitted = validate_simready_graph_asset_spec(spec)
    task_receipt = _verify_task_freeze(
        Path(task_freeze_receipt_path).expanduser().resolve(), admitted=admitted
    )
    source_receipt_path = Path(source_asset_receipt_path).expanduser().resolve()
    source_receipt = _verify_source_asset(
        source_receipt_path, admitted["source_asset_receipt_digest"]
    )
    if (
        str((source_receipt.get("target") or {}).get("interiorgs_instance_id") or "")
        != admitted["source_object_instance_id"]
    ):
        raise SimReadyGraphAssetError(["graph_asset_source_object_identity_mismatch"])
    output = Path(destination).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise SimReadyGraphAssetError(["graph_asset_destination_exists"])
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    if stage is None:
        raise SimReadyGraphAssetError(["graph_asset_stage_create_failed"])
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    world = admitted["world_pose"]
    root.AddTranslateOp().Set(Gf.Vec3d(*world["translation_m"]))
    root.AddOrientOp().Set(_gf_quat(world["orientation_xyzw"]))
    root.GetPrim().SetCustomDataByKey("blueprint:assetId", admitted["asset_id"])
    root.GetPrim().SetCustomDataByKey("blueprint:specDigest", admitted["spec_digest"])

    graph_links = {row["link_id"]: row for row in admitted["articulation_graph"]["links"]}
    link_paths: dict[str, str] = {}
    for link in admitted["links"]:
        link_id = link["link_id"]
        path = f"/Asset/links/{link_id}"
        link_paths[link_id] = path
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*link["rest_pose"]["translation_m"]))
        xform.AddOrientOp().Set(_gf_quat(link["rest_pose"]["orientation_xyzw"]))
        body = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        if graph_links[link_id]["is_root"] and admitted["root_body_mode"] == "fixed":
            body.CreateKinematicEnabledAttr(True)
        mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
        mass_api.CreateMassAttr(float(link["mass_kg"]))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*link["center_of_mass_m"]))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*link["diagonal_inertia_kg_m2"]))
        xform.GetPrim().SetCustomDataByKey(
            "blueprint:semanticRole", graph_links[link_id]["semantic_role"]
        )
        material = UsdShade.Material.Define(stage, f"/Asset/materials/{link_id}")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(float(link["friction"]))
        physics_material.CreateDynamicFrictionAttr(float(link["friction"]))
        physics_material.CreateRestitutionAttr(float(link["restitution"]))
        for geometry in link["geometry"]:
            geometry_path = f"{path}/geometry/{geometry['geometry_id']}"
            if geometry["kind"] == "box":
                primitive = UsdGeom.Cube.Define(stage, geometry_path)
                primitive.CreateSizeAttr(1.0)
            else:
                primitive = UsdGeom.Cylinder.Define(stage, geometry_path)
                primitive.CreateAxisAttr("X")
                primitive.CreateRadiusAttr(float(geometry["radius_m"]))
                primitive.CreateHeightAttr(float(geometry["height_m"]))
            # Keep the conventional translate/orient/scale order.  Authoring a
            # box scale first makes the later translation part of the scaled
            # coordinate system, silently moving off-origin geometry (and its
            # collider) by a dimension-dependent amount.
            primitive.AddTranslateOp().Set(Gf.Vec3f(*geometry["translation_m"]))
            primitive.AddOrientOp().Set(_gf_quat(geometry["orientation_xyzw"]))
            if geometry["kind"] == "box":
                primitive.AddScaleOp().Set(Gf.Vec3f(*geometry["size_m"]))
            primitive.CreateDisplayColorAttr(
                [Gf.Vec3f(*geometry["display_color_rgb"])]
            )
            # Collision geometry is an internal simulator aid, never the
            # candidate's appearance geometry.  Keeping it render-visible can
            # silently enlarge an authored asset's depth silhouette and make a
            # replacement-coverage claim unsafe.  Agent-authored visual meshes
            # must live in a distinct render/default-purpose scope.
            primitive.CreatePurposeAttr(UsdGeom.Tokens.guide)
            primitive.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(primitive.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(primitive.GetPrim()).Bind(
                material, materialPurpose="physics"
            )
            primitive.GetPrim().SetCustomDataByKey(
                "blueprint:geometryProvenance", geometry["provenance"]
            )
            primitive.GetPrim().SetCustomDataByKey(
                "blueprint:collisionGeometryOnly", True
            )

    frame_by_id = {row["joint_id"]: row for row in admitted["joint_frames"]}
    joint_paths: dict[str, str] = {}
    dependencies: list[dict[str, Any]] = []
    joint_drive_implementations: dict[str, dict[str, Any]] = {}
    for joint in admitted["articulation_graph"]["joints"]:
        joint_id = joint["joint_id"]
        path = f"/Asset/joints/{joint_id}"
        joint_paths[joint_id] = path
        if joint["joint_type"] in {"revolute", "continuous"}:
            typed_joint = UsdPhysics.RevoluteJoint.Define(stage, path)
            typed_joint.CreateAxisAttr("X")
            if joint["joint_type"] == "revolute":
                typed_joint.CreateLowerLimitAttr(math.degrees(joint["limits"][0]))
                typed_joint.CreateUpperLimitAttr(math.degrees(joint["limits"][1]))
        elif joint["joint_type"] == "prismatic":
            typed_joint = UsdPhysics.PrismaticJoint.Define(stage, path)
            typed_joint.CreateAxisAttr("X")
            typed_joint.CreateLowerLimitAttr(float(joint["limits"][0]))
            typed_joint.CreateUpperLimitAttr(float(joint["limits"][1]))
        else:
            typed_joint = UsdPhysics.FixedJoint.Define(stage, path)
        typed_joint.CreateBody0Rel().SetTargets(
            [Sdf.Path(link_paths[joint["parent_link_id"]])]
        )
        typed_joint.CreateBody1Rel().SetTargets(
            [Sdf.Path(link_paths[joint["child_link_id"]])]
        )
        frame = frame_by_id[joint_id]
        typed_joint.CreateLocalPos0Attr(Gf.Vec3f(*frame["parent_position_m"]))
        typed_joint.CreateLocalPos1Attr(Gf.Vec3f(*frame["child_position_m"]))
        typed_joint.CreateLocalRot0Attr(_gf_quat(frame["parent_orientation_xyzw"]))
        typed_joint.CreateLocalRot1Attr(_gf_quat(frame["child_orientation_xyzw"]))
        prim = typed_joint.GetPrim()
        prim.SetCustomDataByKey("blueprint:jointRole", joint["role"])
        prim.SetCustomDataByKey("blueprint:graphAxis", Gf.Vec3d(*joint["axis"]))
        prim.SetCustomDataByKey("blueprint:resetPosition", joint["reset_position"])
        drive = joint["drive"]
        prim.SetCustomDataByKey(
            "blueprint:declaredDriveType", drive["drive_type"]
        )
        drive_implementation = "none"
        usd_drive_type: str | None = None
        if joint["joint_type"] != "fixed" and (
            drive["stiffness"] > 0.0 or drive["damping"] > 0.0
        ):
            name = "linear" if joint["joint_type"] == "prismatic" else "angular"
            authored = UsdPhysics.DriveAPI.Apply(prim, name)
            # USD has no "none" drive token.  A graph-declared non-actuated
            # joint may still carry passive damping, represented as a
            # zero-stiffness force damper.  Retain that translation explicitly
            # instead of silently rewriting the declared drive type.
            authored_type = (
                drive["drive_type"]
                if drive["drive_type"] in {"force", "acceleration"}
                else "force"
            )
            authored.CreateTypeAttr().Set(authored_type)
            usd_drive_type = authored_type
            drive_implementation = (
                "passive_force_damper"
                if drive["drive_type"] == "none"
                else f"usd_{authored_type}_drive"
            )
            prim.SetCustomDataByKey(
                "blueprint:driveImplementation",
                drive_implementation,
            )
            authored.CreateStiffnessAttr().Set(float(drive["stiffness"]))
            authored.CreateDampingAttr().Set(float(drive["damping"]))
            target = joint["reset_position"]
            if joint["joint_type"] != "prismatic":
                target = math.degrees(target)
            authored.CreateTargetPositionAttr().Set(float(target))
            if drive["maximum_force"] > 0.0:
                authored.CreateMaxForceAttr().Set(float(drive["maximum_force"]))
        joint_drive_implementations[joint_id] = {
            "declared_drive_type": drive["drive_type"],
            "usd_drive_authored": usd_drive_type is not None,
            "usd_drive_type": usd_drive_type,
            "implementation": drive_implementation,
        }
        if joint["dependency"] is not None:
            dependencies.append(
                {
                    "joint_id": joint_id,
                    **joint["dependency"],
                    "native_dependency_controller_required": True,
                }
            )
            prim.SetCustomDataByKey(
                "blueprint:dependency", canonical_json(joint["dependency"])
            )
    for pair in admitted["articulation_graph"]["collision_pairs"]:
        if pair["collision_enabled"]:
            continue
        source_prim = stage.GetPrimAtPath(link_paths[pair["link_a"]])
        filtered = UsdPhysics.FilteredPairsAPI.Apply(source_prim)
        filtered.CreateFilteredPairsRel().AddTarget(
            Sdf.Path(link_paths[pair["link_b"]])
        )
    stage.GetRootLayer().documentation = (
        "Blueprint graph-authored SimReady candidate; native behavior and physical "
        "equivalence are not qualified by authoring"
    )
    stage.GetRootLayer().Save()

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "simready_candidate_authored",
        "measured_axis_binding": measured_axis_binding,
        "asset_id": admitted["asset_id"],
        "task_id": admitted["task_id"],
        "task_freeze_digest": admitted["task_freeze_digest"],
        "task_freeze_schema": task_receipt["schema_version"],
        "spec_digest": admitted["spec_digest"],
        "source_asset_receipt": {
            "receipt_digest": source_receipt["receipt_digest"],
            "source_collision_prim_path": source_receipt["source_collision_prim_path"],
            "source_asset_sha256": source_receipt["output_asset"]["sha256"],
        },
        "output_usd": {
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "default_prim": "/Asset",
            "meters_per_unit": 1.0,
            "up_axis": "Z",
        },
        "root_body_mode": admitted["root_body_mode"],
        "link_paths": link_paths,
        "joint_paths": joint_paths,
        "articulation_graph_digest": canonical_digest(
            admitted["articulation_graph"], digest_field="graph_digest"
        ),
        "dependencies": dependencies,
        "joint_drive_implementations": joint_drive_implementations,
        "provenance_counts": {
            label: sum(
                geometry["provenance"] == label
                for link in admitted["links"]
                for geometry in link["geometry"]
            )
            for label in sorted(PROVENANCE_LABELS)
        },
        "physics_authored": {
            "mass_com_inertia": True,
            "friction_restitution_materials": True,
            "primitive_collision": True,
            "joint_limits_drives_and_reset_metadata": True,
            "dependency_controller_required": bool(dependencies),
            "physics_values_are_authored_estimates": any(
                link["physics_provenance"] == "authored_estimate_unqualified"
                for link in admitted["links"]
            ),
        },
        "claim_boundary": {
            "simready_candidate_authored": True,
            "native_simulator_import_qualified": False,
            "joint_physics_behavior_qualified": False,
            "appearance_materially_qualified": False,
            "physical_equivalence_proven": False,
            "generated_geometry_is_observed_truth": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    target_receipt = (
        Path(receipt_path).expanduser().resolve()
        if receipt_path is not None
        else output.with_suffix(".receipt.json")
    )
    target_receipt.parent.mkdir(parents=True, exist_ok=True)
    target_receipt.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


__all__ = [
    "RECEIPT_SCHEMA",
    "SPEC_SCHEMA",
    "SimReadyGraphAssetError",
    "author_simready_graph_asset",
    "validate_simready_graph_asset_spec",
]
