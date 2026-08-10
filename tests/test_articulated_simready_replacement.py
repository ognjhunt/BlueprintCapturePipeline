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


from blueprint_pipeline.articulated_simready_replacement import (  # noqa: E402
    author_articulated_simready_replacement,
)


def _rigged_topology_fixture(path: Path, *, include_handle_component: bool = True) -> Path:
    """Mimic an owned-core rigged asset: joints + links, no masses/colliders."""

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(asset.GetPrim())
    upper_z = (UPPER_INTERVAL[0] + UPPER_INTERVAL[1]) / 2.0
    lower_z = (LOWER_INTERVAL[0] + LOWER_INTERVAL[1]) / 2.0
    _define_link(
        stage, "/Asset/link_cabinet", center=(0.0, 0.0, 0.8), half=(DOOR_HALF_X, 0.3, 0.8)
    )
    _define_link(
        stage,
        "/Asset/link_upper_door",
        center=(0.0, DOOR_FRONT_Y - 0.025, upper_z),
        half=(DOOR_HALF_X, 0.025, (UPPER_INTERVAL[1] - UPPER_INTERVAL[0]) / 2.0 - 0.002),
    )
    _define_link(
        stage,
        "/Asset/link_lower_door",
        center=(0.0, DOOR_FRONT_Y - 0.025, lower_z),
        half=(DOOR_HALF_X, 0.025, (LOWER_INTERVAL[1] - LOWER_INTERVAL[0]) / 2.0 - 0.002),
    )
    if include_handle_component:
        _mesh_box(
            stage,
            "/Asset/link_upper_door/component_handle",
            center=(0.25, DOOR_FRONT_Y + 0.02, 1.05),
            half=(0.02, 0.018, 0.09),
        )
    _define_revolute(
        stage,
        "/Asset/joints/joint_0",
        body0="/Asset/link_cabinet",
        body1="/Asset/link_upper_door",
        pivot=(HINGE_ASSET[0], HINGE_ASSET[1], upper_z),
    )
    _define_revolute(
        stage,
        "/Asset/joints/joint_1",
        body0="/Asset/link_cabinet",
        body1="/Asset/link_lower_door",
        pivot=(HINGE_ASSET[0], HINGE_ASSET[1], lower_z),
    )
    stage.GetRootLayer().Save()
    return path


def _physics_template() -> dict:
    contract = {
        key: value
        for key, value in _physics_contract().items()
        if key
        not in {
            "link_collider_envelopes_m",
            "task_door_link",
            "support_link",
            "required_generated_interior_links",
        }
    }
    contract["task_door_envelope_m"] = {
        "aabb_min": [-0.36, 0.28, UPPER_INTERVAL[0] - 0.005],
        "aabb_max": [0.36, 0.46, UPPER_INTERVAL[1] + 0.005],
    }
    contract["non_task_door_envelope_m"] = {
        "aabb_min": [-0.36, 0.28, LOWER_INTERVAL[0] - 0.005],
        "aabb_max": [0.36, 0.46, LOWER_INTERVAL[1] + 0.005],
    }
    contract["support_envelope_m"] = {
        "aabb_min": [-0.36, -0.36, -0.005],
        "aabb_max": [0.36, 0.305, 1.64],
    }
    return contract


def _authoring_arguments(tmp_path: Path, rigged: Path) -> dict:
    return {
        "rigged_topology_usd_path": rigged,
        "output_usd_path": tmp_path / "simready_candidate.usda",
        "topology_contract": _contract(),
        "physics_contract_template": _physics_template(),
        "authoring_spec": {
            "support_link_mass_kg": 62.0,
            "door_link_mass_kg": 9.0,
            "other_link_mass_kg": 2.0,
            "static_friction": 0.6,
            "dynamic_friction": 0.5,
            "restitution": 0.05,
            "handle": {
                "minimum_protrusion_m": 0.012,
                "generated_center_asset_m": [0.25, DOOR_FRONT_Y + 0.02, 1.05],
                "generated_half_extents_m": [0.02, 0.018, 0.09],
            },
            "generated_interior_inset_m": 0.04,
            "fixed_base": True,
        },
    }


def test_authoring_produces_statically_admitted_candidate(tmp_path: Path) -> None:
    rigged = _rigged_topology_fixture(tmp_path / "rigged.usda")

    receipt = author_articulated_simready_replacement(
        **_authoring_arguments(tmp_path, rigged)
    )

    assert receipt["schema_version"] == "articulated_simready_authoring.v1"
    assert receipt["status"] == "simready_candidate_statically_admitted"
    assert receipt["claim_boundary"]["physical_equivalence_proven"] is False
    assert receipt["claim_boundary"]["native_simulator_qualified"] is False
    assert receipt["topology_validation"]["status"] == "topology_statically_admitted"
    assert receipt["physics_validation"]["status"] == "physics_statically_admitted"
    assert receipt["task_joint_prim_path"]
    assert receipt["output_usd_sha256"].startswith("sha256:")


def test_authoring_generates_labeled_handle_when_source_lacks_one(
    tmp_path: Path,
) -> None:
    rigged = _rigged_topology_fixture(
        tmp_path / "rigged.usda", include_handle_component=False
    )

    receipt = author_articulated_simready_replacement(
        **_authoring_arguments(tmp_path, rigged)
    )

    assert receipt["status"] == "simready_candidate_statically_admitted"
    assert receipt["handle"]["source"] == "generated_parametric_candidate"
    handle_paths = receipt["physics_validation"]["handle_prim_paths"]
    assert len(handle_paths) == 1
    stage = Usd.Stage.Open(str(receipt["output_usd_path"]))
    prim = stage.GetPrimAtPath(handle_paths[0])
    assert prim.GetAttribute(PROVENANCE_ATTRIBUTE).Get() == GENERATED_PROVENANCE_VALUE


def test_authoring_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    rigged = _rigged_topology_fixture(tmp_path / "rigged.usda")

    first = author_articulated_simready_replacement(
        **{
            **_authoring_arguments(tmp_path / "a", rigged),
            "output_usd_path": tmp_path / "a/out.usda",
        }
    )
    second = author_articulated_simready_replacement(
        **{
            **_authoring_arguments(tmp_path / "b", rigged),
            "output_usd_path": tmp_path / "b/out.usda",
        }
    )

    assert first["output_usd_sha256"] == second["output_usd_sha256"]


from blueprint_pipeline.articulated_simready_replacement import (  # noqa: E402
    derive_articulated_topology_from_source,
)


def _source_like_fixture(path: Path) -> Path:
    """One mesh + component subsets shaped like the real 28-component source."""

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/source_mesh")
    points: list[Gf.Vec3f] = []
    face_counts: list[int] = []
    face_indices: list[int] = []
    subset_faces: dict[str, list[int]] = {}

    def _add_box(name: str, center, half) -> None:
        base = len(points)
        cx, cy, cz = center
        hx, hy, hz = half
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    points.append(Gf.Vec3f(cx + sx * hx, cy + sy * hy, cz + sz * hz))
        quads = [
            (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
            (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3),
        ]
        rows = []
        for quad in quads:
            rows.append(len(face_counts))
            face_counts.append(4)
            face_indices.extend(base + corner for corner in quad)
        subset_faces[name] = rows

    upper_mid = (UPPER_INTERVAL[0] + UPPER_INTERVAL[1]) / 2.0
    lower_mid = (LOWER_INTERVAL[0] + LOWER_INTERVAL[1]) / 2.0
    _add_box("component_shell", (0.0, -0.086, 0.827), (DOOR_HALF_X, 0.264, 0.805))
    _add_box(
        "component_upper_slab",
        (0.0, 0.2525, upper_mid),
        (DOOR_HALF_X, 0.0535, (UPPER_INTERVAL[1] - UPPER_INTERVAL[0]) / 2.0 - 0.01),
    )
    _add_box(
        "component_lower_slab",
        (0.0, 0.2525, lower_mid - 0.02),
        (DOOR_HALF_X, 0.0535, (LOWER_INTERVAL[1] - LOWER_INTERVAL[0]) / 2.0 - 0.03),
    )
    _add_box("component_upper_handle", (0.12, 0.3275, 1.0225), (0.1225, 0.0215, 0.0185))
    _add_box("component_lower_handle", (0.12, 0.3275, 0.8155), (0.1225, 0.0215, 0.0185))
    _add_box("component_hinge_upper", (-0.3505, 0.1895, 1.009), (0.0035, 0.0025, 0.01))
    _add_box("component_foot", (-0.319, -0.316, 0.011), (0.02, 0.016, 0.01))

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    for name, rows in subset_faces.items():
        subset = UsdGeom.Subset.Define(stage, f"/Asset/source_mesh/{name}")
        subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
        subset.CreateIndicesAttr().Set(rows)
        subset.CreateFamilyNameAttr().Set("blueprint_connected_components")
    stage.GetRootLayer().Save()
    return path


def test_derived_topology_partitions_and_passes_full_authoring_chain(
    tmp_path: Path,
) -> None:
    source = _source_like_fixture(tmp_path / "source.usda")

    derived = derive_articulated_topology_from_source(
        source_asset_usd_path=source,
        output_usd_path=tmp_path / "topology.usda",
        seam_z_m=UPPER_INTERVAL[0],
        hinge_pivot_asset_xy_m=[HINGE_ASSET[0], HINGE_ASSET[1]],
        joint_limits_rad=[0.0, math.pi / 2.0],
        door_back_plane_y_m=0.178,
        door_face_plane_y_m=0.306,
    )

    assert derived["schema_version"] == "articulated_topology_from_source.v1"
    assert derived["construction_path"] == (
        "deterministic_parametric_from_frozen_observations"
    )
    partition = {
        row["component"]: (row["link"], row["role"])
        for row in derived["component_partition"]
    }
    assert partition["component_shell"] == ("cabinet", "cabinet_shell")
    assert partition["component_upper_slab"] == ("upper_door", "door_slab")
    assert partition["component_lower_slab"] == ("lower_door", "door_slab")
    assert partition["component_upper_handle"] == ("upper_door", "handle_hardware")
    assert partition["component_lower_handle"] == ("lower_door", "handle_hardware")
    assert partition["component_hinge_upper"] == ("upper_door", "hinge_hardware")
    assert partition["component_foot"] == ("cabinet", "cabinet_shell")
    assert derived["claim_boundary"]["joint_agent_output"] is False

    template = _physics_template()
    template["maximum_reset_pairwise_overlap_m"] = 0.006
    template["task_door_envelope_m"] = {
        "aabb_min": [-0.36, 0.17, UPPER_INTERVAL[0] - 0.005],
        "aabb_max": [0.36, 0.36, UPPER_INTERVAL[1] + 0.005],
    }
    template["non_task_door_envelope_m"] = {
        "aabb_min": [-0.36, 0.17, LOWER_INTERVAL[0] - 0.005],
        "aabb_max": [0.36, 0.36, LOWER_INTERVAL[1] + 0.005],
    }
    receipt = author_articulated_simready_replacement(
        rigged_topology_usd_path=tmp_path / "topology.usda",
        output_usd_path=tmp_path / "candidate.usda",
        topology_contract=_contract(),
        physics_contract_template=template,
        authoring_spec=_authoring_arguments(tmp_path, source)["authoring_spec"],
    )

    assert receipt["status"] == "simready_candidate_statically_admitted"
    assert receipt["handle"]["source"] == "observed_source_component"
    assert receipt["topology_validation"]["status"] == "topology_statically_admitted"
    assert receipt["physics_validation"]["status"] == "physics_statically_admitted"


def test_agent_enriched_asset_must_keep_blueprint_authored_link_masses(
    tmp_path: Path,
) -> None:
    """A SimReady pass may add priors; it may not overwrite authored masses.

    The 840796 Content Agents run preserved the authored 62/11/11 kg link
    masses and added per-component MassAPI with zero values. Zero component
    masses are tolerable only while the rigid-body links keep their authored
    values, so the physics gate must catch a link whose mass the agent moved.
    """

    asset = _apply_physics(_author_topology(tmp_path / "asset.usda"))
    stage = Usd.Stage.Open(str(asset))
    for path, mass in (
        ("/Asset/cabinet", 60.0),
        ("/Asset/upper_door", 8.0),
        ("/Asset/lower_door", 10.0),
    ):
        assert UsdPhysics.MassAPI(stage.GetPrimAtPath(path)).GetMassAttr().Get() == mass

    # An agent that rewrites a link mass outside the admitted range is caught.
    UsdPhysics.MassAPI(stage.GetPrimAtPath("/Asset/upper_door")).GetMassAttr().Set(0.0)
    stage.GetRootLayer().Save()

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_physics(
            replacement_usd_path=asset, contract=_physics_contract()
        )

    assert any(
        "link_mass_missing_or_out_of_range" in error for error in excinfo.value.errors
    )


def test_a_commanded_joint_without_a_drive_is_rejected(tmp_path: Path) -> None:
    """A position target on an undriven joint does nothing at all.

    Isaac proved this the expensive way: every other readback passed - axis,
    limits, locked joint, contact, reset, determinism - while the commanded
    door stayed at 0.0 degrees through all twelve steps, because the joint
    carried no DriveAPI. NVIDIA documents that the Joint Agent authors topology
    and not drives, so nothing upstream would have supplied one.
    """

    asset = _apply_physics(_author_topology(tmp_path / "asset.usda"))
    contract = dict(_physics_contract())
    contract["require_task_joint_drive"] = True
    contract["task_joint_prim_path"] = "/Asset/joints/upper_hinge"

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_physics(
            replacement_usd_path=asset, contract=contract
        )

    assert any(
        "task_joint_missing_drive" in error for error in excinfo.value.errors
    )


def test_a_commanded_joint_with_a_drive_is_admitted(tmp_path: Path) -> None:
    asset = _apply_physics(_author_topology(tmp_path / "asset.usda"))
    stage = Usd.Stage.Open(str(asset))
    drive = UsdPhysics.DriveAPI.Apply(
        stage.GetPrimAtPath("/Asset/joints/upper_hinge"), "angular"
    )
    drive.CreateStiffnessAttr().Set(400.0)
    drive.CreateDampingAttr().Set(40.0)
    drive.CreateMaxForceAttr().Set(200.0)
    stage.GetRootLayer().Save()
    contract = dict(_physics_contract())
    contract["require_task_joint_drive"] = True
    contract["task_joint_prim_path"] = "/Asset/joints/upper_hinge"

    receipt = validate_articulated_replacement_physics(
        replacement_usd_path=asset, contract=contract
    )

    assert receipt["status"] == "physics_statically_admitted"
    assert receipt["task_joint_drive"]["present"] is True
    assert receipt["task_joint_drive"]["stiffness"] == 400.0


def _drive_contract(stiffness: float, damping: float) -> dict:
    return {
        "require_task_joint_drive": True,
        "task_joint_prim_path": "/Asset/joints/upper_hinge",
        "dynamics_profile_object_class": "household_refrigerator_door",
        "dynamics_lever_arm_m": 0.495,
        "dynamics_nominal_open_angle_degrees": 50.0,
        "dynamics_nominal_sweep_duration_s": 2.0,
        "dynamics_breakaway_torque_n_m": 12.0,
        "dynamics_breakaway_angular_width_degrees": 5.0,
    }


def _driven(tmp_path: Path, stiffness: float, damping: float) -> Path:
    asset = _apply_physics(_author_topology(tmp_path / "asset.usda"))
    stage = Usd.Stage.Open(str(asset))
    drive = UsdPhysics.DriveAPI.Apply(
        stage.GetPrimAtPath("/Asset/joints/upper_hinge"), "angular"
    )
    drive.CreateStiffnessAttr().Set(stiffness)
    drive.CreateDampingAttr().Set(damping)
    stage.GetRootLayer().Save()
    return asset


def test_a_declared_object_class_forces_the_drive_through_its_measured_band(
    tmp_path: Path,
) -> None:
    """Declaring the class is what makes the research step non-optional.

    Without this the band exists but nothing obliges an asset to meet it, and
    the twin that shipped three times too stiff would ship again.
    """

    asset = _driven(tmp_path, 0.0, 14.0)
    contract = {**_physics_contract(), **_drive_contract(0.0, 14.0)}

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_physics(
            replacement_usd_path=asset, contract=contract
        )

    assert any(
        "dynamics_outside_measured_band" in error for error in excinfo.value.errors
    )


def test_an_in_band_drive_records_the_citation_it_was_judged_against(
    tmp_path: Path,
) -> None:
    asset = _driven(tmp_path, 0.0, 3.0)
    contract = {**_physics_contract(), **_drive_contract(0.0, 3.0)}

    receipt = validate_articulated_replacement_physics(
        replacement_usd_path=asset, contract=contract
    )

    realism = receipt["dynamics_realism"]
    assert realism["within_measured_band"] is True
    assert "BioRob 2010" in realism["reference_profile"]["measurement_source"]


def test_an_unresearched_object_class_blocks_rather_than_waving_through(
    tmp_path: Path,
) -> None:
    """A new class must send someone to measure, not silently skip the check."""

    asset = _driven(tmp_path, 0.0, 3.0)
    contract = {**_physics_contract(), **_drive_contract(0.0, 3.0)}
    contract["dynamics_profile_object_class"] = "dishwasher_door"

    with pytest.raises(ArticulatedSimReadyReplacementError) as excinfo:
        validate_articulated_replacement_physics(
            replacement_usd_path=asset, contract=contract
        )

    assert any("profile_not_researched" in error for error in excinfo.value.errors)


def test_omitting_the_object_class_leaves_the_existing_contract_unchanged(
    tmp_path: Path,
) -> None:
    """Assets that never declared a class must not start failing."""

    asset = _driven(tmp_path, 0.0, 14.0)
    contract = {**_physics_contract(), **_drive_contract(0.0, 14.0)}
    contract.pop("dynamics_profile_object_class")

    receipt = validate_articulated_replacement_physics(
        replacement_usd_path=asset, contract=contract
    )

    assert receipt["status"] == "physics_statically_admitted"
    assert receipt["dynamics_realism"]["required"] is False
