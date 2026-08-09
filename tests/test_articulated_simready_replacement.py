from __future__ import annotations

import math
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.articulated_simready_replacement import (
    ArticulatedSimReadyReplacementError,
    validate_articulated_replacement_topology,
)


UPPER_INTERVAL = (0.939981249, 1.631869998)
LOWER_INTERVAL = (0.0, 0.939981249)
HINGE_WORLD = (1.617248144, 1.829218141)
T_ASSET_WORLD = (-1.9742142, -1.4792181, -2e-09)
HINGE_ASSET = (HINGE_WORLD[0] + T_ASSET_WORLD[0], HINGE_WORLD[1] + T_ASSET_WORLD[1])
DOOR_HALF_X = 0.35696606
DOOR_FRONT_Y = 0.34999996


def _contract() -> dict:
    return {
        "task_joint_id": "refrigerator_upper_door_hinge",
        "hinge_origin_world_m": [HINGE_WORLD[0], HINGE_WORLD[1], 1.2859256235],
        "T_asset_world": [
            [1.0, 0.0, 0.0, T_ASSET_WORLD[0]],
            [0.0, 1.0, 0.0, T_ASSET_WORLD[1]],
            [0.0, 0.0, 1.0, T_ASSET_WORLD[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "task_axis_world": [0.0, 0.0, 1.0],
        "task_axis_absolute_dot_minimum": 0.99,
        "task_moving_z_interval_m": list(UPPER_INTERVAL),
        "task_z_overlap_minimum": 0.85,
        "task_limits_rad": [0.0, math.pi / 2.0],
        "limits_tolerance_rad": 1e-6,
        "pivot_xy_tolerance_m": 0.02,
        "minimum_assembly_joint_count": 1,
        "maximum_assembly_joint_count": 4,
        "required_articulation_root_count": 1,
    }


def _mesh_box(
    stage: Usd.Stage,
    path: str,
    *,
    center: tuple[float, float, float],
    half: tuple[float, float, float],
) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    cx, cy, cz = center
    hx, hy, hz = half
    points = [
        Gf.Vec3f(cx + sx * hx, cy + sy * hy, cz + sz * hz)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    mesh.CreateFaceVertexIndicesAttr(
        [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
    )
    return mesh


def _define_link(
    stage: Usd.Stage,
    path: str,
    *,
    center: tuple[float, float, float],
    half: tuple[float, float, float],
) -> Usd.Prim:
    xform = UsdGeom.Xform.Define(stage, path)
    _mesh_box(stage, f"{path}/geom", center=center, half=half)
    return xform.GetPrim()


def _define_revolute(
    stage: Usd.Stage,
    path: str,
    *,
    body0: str,
    body1: str,
    pivot: tuple[float, float, float],
    axis: str = "Z",
    lower_deg: float = 0.0,
    upper_deg: float = 90.0,
) -> UsdPhysics.RevoluteJoint:
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*pivot))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*pivot))
    joint.CreateAxisAttr().Set(axis)
    joint.CreateLowerLimitAttr().Set(lower_deg)
    joint.CreateUpperLimitAttr().Set(upper_deg)
    return joint


def _author_topology(
    path: Path,
    *,
    upper_axis: str = "Z",
    upper_lower_deg: float = 0.0,
    upper_upper_deg: float = 90.0,
    upper_pivot: tuple[float, float, float] | None = None,
    articulation_roots: tuple[str, ...] = ("/Asset",),
    include_lower_joint: bool = True,
    extra_joint_count: int = 0,
) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    upper_center_z = (UPPER_INTERVAL[0] + UPPER_INTERVAL[1]) / 2.0
    lower_center_z = (LOWER_INTERVAL[0] + LOWER_INTERVAL[1]) / 2.0
    _define_link(
        stage,
        "/Asset/cabinet",
        center=(0.0, 0.0, 0.8),
        half=(DOOR_HALF_X, 0.3, 0.8),
    )
    _define_link(
        stage,
        "/Asset/upper_door",
        center=(0.0, DOOR_FRONT_Y - 0.02, upper_center_z),
        half=(DOOR_HALF_X, 0.02, (UPPER_INTERVAL[1] - UPPER_INTERVAL[0]) / 2.0),
    )
    _define_link(
        stage,
        "/Asset/lower_door",
        center=(0.0, DOOR_FRONT_Y - 0.02, lower_center_z),
        half=(DOOR_HALF_X, 0.02, (LOWER_INTERVAL[1] - LOWER_INTERVAL[0]) / 2.0),
    )
    for root in articulation_roots:
        prim = stage.GetPrimAtPath(root)
        assert prim.IsValid()
        UsdPhysics.ArticulationRootAPI.Apply(prim)
    pivot = upper_pivot or (HINGE_ASSET[0], HINGE_ASSET[1], (UPPER_INTERVAL[0] + UPPER_INTERVAL[1]) / 2.0)
    _define_revolute(
        stage,
        "/Asset/joints/upper_hinge",
        body0="/Asset/cabinet",
        body1="/Asset/upper_door",
        pivot=pivot,
        axis=upper_axis,
        lower_deg=upper_lower_deg,
        upper_deg=upper_upper_deg,
    )
    if include_lower_joint:
        _define_revolute(
            stage,
            "/Asset/joints/lower_hinge",
            body0="/Asset/cabinet",
            body1="/Asset/lower_door",
            pivot=(HINGE_ASSET[0], HINGE_ASSET[1], lower_center_z),
        )
    for index in range(extra_joint_count):
        _define_link(
            stage,
            f"/Asset/extra_{index}",
            center=(0.0, -0.2, 0.2 + 0.1 * index),
            half=(0.05, 0.05, 0.05),
        )
        _define_revolute(
            stage,
            f"/Asset/joints/extra_{index}",
            body0="/Asset/cabinet",
            body1=f"/Asset/extra_{index}",
            pivot=(0.0, -0.2, 0.2 + 0.1 * index),
        )
    stage.GetRootLayer().Save()
    return path


def test_topology_validator_accepts_two_door_refrigerator(tmp_path: Path) -> None:
    asset = _author_topology(tmp_path / "topology.usda")

    receipt = validate_articulated_replacement_topology(
        replacement_usd_path=asset,
        contract=_contract(),
    )

    assert receipt["schema_version"] == "articulated_replacement_topology_validation.v1"
    assert receipt["status"] == "topology_statically_admitted"
    assert receipt["assembly_joint_count"] == 2
    assert receipt["task_joint_prim_path"] == "/Asset/joints/upper_hinge"
    assert receipt["task_joint_id"] == "refrigerator_upper_door_hinge"
    assert receipt["non_task_joint_prim_paths"] == ["/Asset/joints/lower_hinge"]
    assert receipt["articulation_root_prim_paths"] == ["/Asset"]
    assert receipt["claim_boundary"]["native_simulator_qualified"] is False
    assert receipt["claim_boundary"]["physical_equivalence_proven"] is False
    assert receipt["replacement_usd_sha256"].startswith("sha256:")
    assert receipt["receipt_digest"].startswith("sha256:")


def test_topology_validator_rejects_wrong_axis(tmp_path: Path) -> None:
    asset = _author_topology(tmp_path / "topology.usda", upper_axis="X")

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_topology(
            replacement_usd_path=asset,
            contract=_contract(),
        )

    assert any(
        error.startswith("articulated_replacement_exactly_one_task_joint_not_resolved")
        for error in excinfo.value.errors
    )


def test_topology_validator_rejects_wrong_limits(tmp_path: Path) -> None:
    asset = _author_topology(tmp_path / "topology.usda", upper_upper_deg=120.0)

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_topology(
            replacement_usd_path=asset,
            contract=_contract(),
        )

    assert any(
        "task_joint_limits_mismatch" in error for error in excinfo.value.errors
    )


def test_topology_validator_rejects_pivot_outside_tolerance(tmp_path: Path) -> None:
    asset = _author_topology(
        tmp_path / "topology.usda",
        upper_pivot=(HINGE_ASSET[0] + 0.08, HINGE_ASSET[1], 1.2859),
    )

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_topology(
            replacement_usd_path=asset,
            contract=_contract(),
        )

    assert any("task_joint_pivot_outside_tolerance" in error for error in excinfo.value.errors)


def test_topology_validator_rejects_joint_count_above_scope(tmp_path: Path) -> None:
    asset = _author_topology(tmp_path / "topology.usda", extra_joint_count=3)

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_topology(
            replacement_usd_path=asset,
            contract=_contract(),
        )

    assert any(
        "assembly_joint_count_outside_preregistered_bounds" in error
        for error in excinfo.value.errors
    )


def test_topology_validator_rejects_multiple_articulation_roots(tmp_path: Path) -> None:
    asset = _author_topology(
        tmp_path / "topology.usda",
        articulation_roots=("/Asset", "/Asset/upper_door"),
    )

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_topology(
            replacement_usd_path=asset,
            contract=_contract(),
        )

    assert any(
        "articulation_root_count_mismatch" in error for error in excinfo.value.errors
    )


from blueprint_pipeline.articulated_simready_replacement import (  # noqa: E402
    GENERATED_PROVENANCE_VALUE,
    HANDLE_ROLE_VALUE,
    OBSERVED_PROVENANCE_VALUE,
    PROVENANCE_ATTRIBUTE,
    TASK_CONTACT_ROLE_ATTRIBUTE,
    validate_articulated_replacement_physics,
)


def _physics_contract() -> dict:
    return {
        "link_collider_envelopes_m": {
            "/Asset/upper_door": {
                "aabb_min": [-0.36, 0.28, UPPER_INTERVAL[0] - 0.005],
                "aabb_max": [0.36, 0.46, UPPER_INTERVAL[1] + 0.005],
            },
            "/Asset/lower_door": {
                "aabb_min": [-0.36, 0.28, LOWER_INTERVAL[0] - 0.005],
                "aabb_max": [0.36, 0.46, LOWER_INTERVAL[1] + 0.005],
            },
            "/Asset/cabinet": {
                "aabb_min": [-0.36, -0.36, -0.005],
                "aabb_max": [0.36, 0.305, 1.64],
            },
        },
        "task_door_link": "/Asset/upper_door",
        "support_link": "/Asset/cabinet",
        "handle_minimum_protrusion_m": 0.005,
        "maximum_reset_pairwise_overlap_m": 0.002,
        "support_z_tolerance_m": 0.01,
        "mass_range_kg": [0.5, 400.0],
        "friction_range": [0.05, 2.0],
        "restitution_range": [0.0, 0.5],
        "required_generated_interior_links": ["/Asset/cabinet"],
    }


def _apply_physics(
    path: Path,
    *,
    skip_mass_link: str | None = None,
    door_collider_crosses_seam: bool = False,
    floating_body: bool = False,
    omit_handle: bool = False,
    omit_interior: bool = False,
    untagged_geometry: bool = False,
    cabinet_lifted: bool = False,
) -> Path:
    stage = Usd.Stage.Open(str(path))
    from pxr import UsdShade

    shade_material = UsdShade.Material.Define(
        stage, "/Asset/physics_materials/painted_steel"
    )
    material = UsdPhysics.MaterialAPI.Apply(shade_material.GetPrim())
    material.CreateStaticFrictionAttr().Set(0.6)
    material.CreateDynamicFrictionAttr().Set(0.5)
    material.CreateRestitutionAttr().Set(0.05)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/Asset")).Bind(
        shade_material, materialPurpose="physics"
    )
    links = {
        "/Asset/cabinet": 60.0,
        "/Asset/upper_door": 8.0,
        "/Asset/lower_door": 10.0,
    }
    for link_path, mass in links.items():
        prim = stage.GetPrimAtPath(link_path)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        if link_path != skip_mass_link:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr().Set(mass)
            geom = stage.GetPrimAtPath(f"{link_path}/geom")
            from pxr import Gf as _Gf
            bound = UsdGeom.Mesh(geom).ComputeExtent(
                UsdGeom.Mesh(geom).GetPointsAttr().Get()
            )
            center = (_Gf.Vec3f(bound[0]) + _Gf.Vec3f(bound[1])) / 2.0
            mass_api.CreateCenterOfMassAttr().Set(center)
            mass_api.CreateDiagonalInertiaAttr().Set(_Gf.Vec3f(1.0, 1.0, 1.0))
        geom_prim = stage.GetPrimAtPath(f"{link_path}/geom")
        UsdPhysics.CollisionAPI.Apply(geom_prim)
        geom_prim.CreateAttribute(PROVENANCE_ATTRIBUTE, Sdf.ValueTypeNames.String).Set(
            OBSERVED_PROVENANCE_VALUE
        )

    if door_collider_crosses_seam:
        crossing = _mesh_box(
            stage,
            "/Asset/upper_door/seam_crossing",
            center=(0.0, DOOR_FRONT_Y - 0.02, UPPER_INTERVAL[0] - 0.05),
            half=(0.1, 0.02, 0.1),
        )
        UsdPhysics.CollisionAPI.Apply(crossing.GetPrim())
        crossing.GetPrim().CreateAttribute(
            PROVENANCE_ATTRIBUTE, Sdf.ValueTypeNames.String
        ).Set(OBSERVED_PROVENANCE_VALUE)
    if floating_body:
        floating = _define_link(
            stage,
            "/Asset/floating_piece",
            center=(0.2, 0.0, 1.9),
            half=(0.03, 0.03, 0.03),
        )
        UsdPhysics.RigidBodyAPI.Apply(floating)
        mass_api = UsdPhysics.MassAPI.Apply(floating)
        mass_api.CreateMassAttr().Set(0.5)
        geom_prim = stage.GetPrimAtPath("/Asset/floating_piece/geom")
        UsdPhysics.CollisionAPI.Apply(geom_prim)
        geom_prim.CreateAttribute(PROVENANCE_ATTRIBUTE, Sdf.ValueTypeNames.String).Set(
            OBSERVED_PROVENANCE_VALUE
        )
    if not omit_handle:
        handle = _mesh_box(
            stage,
            "/Asset/upper_door/handle",
            center=(0.25, DOOR_FRONT_Y + 0.02, 1.06),
            half=(0.02, 0.02, 0.10),
        )
        UsdPhysics.CollisionAPI.Apply(handle.GetPrim())
        handle.GetPrim().CreateAttribute(
            TASK_CONTACT_ROLE_ATTRIBUTE, Sdf.ValueTypeNames.String
        ).Set(HANDLE_ROLE_VALUE)
        handle.GetPrim().CreateAttribute(
            PROVENANCE_ATTRIBUTE, Sdf.ValueTypeNames.String
        ).Set(OBSERVED_PROVENANCE_VALUE)
    if not omit_interior:
        interior = _mesh_box(
            stage,
            "/Asset/cabinet/generated_interior",
            center=(0.0, 0.05, 0.8),
            half=(0.3, 0.2, 0.75),
        )
        interior.GetPrim().CreateAttribute(
            PROVENANCE_ATTRIBUTE, Sdf.ValueTypeNames.String
        ).Set(GENERATED_PROVENANCE_VALUE)
    if untagged_geometry:
        _mesh_box(
            stage,
            "/Asset/cabinet/untagged_extra",
            center=(0.0, -0.1, 0.4),
            half=(0.02, 0.02, 0.02),
        )
    if cabinet_lifted:
        cabinet = UsdGeom.Xformable(stage.GetPrimAtPath("/Asset/cabinet"))
        cabinet.AddTranslateOp().Set((0.0, 0.0, 0.15))
    stage.GetRootLayer().Save()
    return path


def test_physics_validator_accepts_fully_authored_asset(tmp_path: Path) -> None:
    asset = _apply_physics(_author_topology(tmp_path / "asset.usda"))

    receipt = validate_articulated_replacement_physics(
        replacement_usd_path=asset,
        contract=_physics_contract(),
    )

    assert receipt["schema_version"] == "articulated_replacement_physics_validation.v1"
    assert receipt["status"] == "physics_statically_admitted"
    assert receipt["handle_prim_paths"] == ["/Asset/upper_door/handle"]
    assert receipt["generated_interior_prim_paths"] == [
        "/Asset/cabinet/generated_interior"
    ]
    assert receipt["claim_boundary"]["generated_geometry_is_observed_site_truth"] is False
    assert receipt["claim_boundary"]["native_simulator_qualified"] is False


def test_physics_validator_rejects_missing_mass(tmp_path: Path) -> None:
    asset = _apply_physics(
        _author_topology(tmp_path / "asset.usda"), skip_mass_link="/Asset/upper_door"
    )

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_physics(
            replacement_usd_path=asset, contract=_physics_contract()
        )

    assert any("link_mass_missing" in error for error in excinfo.value.errors)


def test_physics_validator_rejects_door_collider_crossing_seam(tmp_path: Path) -> None:
    asset = _apply_physics(
        _author_topology(tmp_path / "asset.usda"), door_collider_crosses_seam=True
    )

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_physics(
            replacement_usd_path=asset, contract=_physics_contract()
        )

    assert any(
        "collider_outside_link_envelope" in error for error in excinfo.value.errors
    )


def test_physics_validator_rejects_floating_component(tmp_path: Path) -> None:
    asset = _apply_physics(
        _author_topology(tmp_path / "asset.usda"), floating_body=True
    )

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_physics(
            replacement_usd_path=asset, contract=_physics_contract()
        )

    assert any(
        "floating_rigid_body_outside_joint_graph" in error
        for error in excinfo.value.errors
    )


def test_physics_validator_rejects_missing_handle(tmp_path: Path) -> None:
    asset = _apply_physics(
        _author_topology(tmp_path / "asset.usda"), omit_handle=True
    )

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_physics(
            replacement_usd_path=asset, contract=_physics_contract()
        )

    assert any(
        "task_door_handle_contact_missing" in error for error in excinfo.value.errors
    )


def test_physics_validator_rejects_missing_generated_interior(tmp_path: Path) -> None:
    asset = _apply_physics(
        _author_topology(tmp_path / "asset.usda"), omit_interior=True
    )

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_physics(
            replacement_usd_path=asset, contract=_physics_contract()
        )

    assert any(
        "generated_interior_missing" in error for error in excinfo.value.errors
    )


def test_physics_validator_rejects_untagged_geometry(tmp_path: Path) -> None:
    asset = _apply_physics(
        _author_topology(tmp_path / "asset.usda"), untagged_geometry=True
    )

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_physics(
            replacement_usd_path=asset, contract=_physics_contract()
        )

    assert any(
        "geometry_provenance_untagged" in error for error in excinfo.value.errors
    )


def test_physics_validator_rejects_unsupported_cabinet(tmp_path: Path) -> None:
    asset = _apply_physics(
        _author_topology(tmp_path / "asset.usda"), cabinet_lifted=True
    )

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_physics(
            replacement_usd_path=asset, contract=_physics_contract()
        )

    assert any("support_contact_not_grounded" in error for error in excinfo.value.errors)
