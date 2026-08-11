from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import native_deformable_asset_preparation as preparation
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_deformable_asset_preparation import (
    DEFORMABLE_BODY_CFG,
    DEFORMABLE_BODY_SCHEMAS,
    DEFORMABLE_MATERIAL_API,
    DEFORMABLE_MATERIAL_CFG,
    DEFORMABLE_PHYSICS_BINDING_API,
    NativeDeformableAssetPreparationError,
    PINNED_NATIVE_CALL_CONTRACT,
    build_native_deformable_asset_source_package,
    execute_native_deformable_asset_preparation,
    materialize_native_deformable_asset_preparation_plan,
    verify_native_deformable_asset_preparation_return,
)
from blueprint_pipeline.native_task_entity_asset_authoring_bundle import (
    DEFORMABLE_AUTHORING_API,
    DEFORMABLE_COOKING_API,
)
from blueprint_pipeline.native_task_runtime_source_packet import (
    ISAACLAB_COMMIT,
    ISAACLAB_REPOSITORY,
    ISAACLAB_TREE,
)


def _source_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    from tests.test_external_simready_deformable_asset import _inspect, _write_fixture

    paths = _write_fixture(
        tmp_path,
        observed_dimensions=(0.37, 0.119, 0.113),
        standard_schemas=False,
        nonempty_tetmesh=False,
        include_default_dome_light=True,
        static_rigid_contract=True,
    )
    receipt = _inspect(paths)
    receipt_path = _write_receipt(tmp_path / "inspection_receipt.json", receipt)
    return paths["usd"], paths["root"] / "textures", receipt_path, receipt


def _write_receipt(path: Path, receipt: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _physics() -> dict[str, Any]:
    return {
        "body_properties": {
            "deformable_enabled": True,
            "kinematic_enabled": False,
            "self_collision": True,
            "solver_position_iteration_count": 28,
            "vertex_velocity_damping": 0.18,
            "contact_offset": 0.003,
            "rest_offset": 0.001,
        },
        "cooking_properties": {
            "collision_simplification": True,
            "collision_simplification_remeshing": True,
            "collision_simplification_remeshing_resolution": 22,
            "simulation_hexahedral_resolution": 22,
        },
        "material_properties": {
            "density": 220.0,
            "static_friction": 2.2,
            "dynamic_friction": 2.2,
            "poissons_ratio": 0.42,
            "youngs_modulus": 180_000.0,
            "elasticity_damping": 0.005,
        },
    }


def _plan(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    source, textures, receipt_path, receipt = _source_fixture(tmp_path)
    plan = materialize_native_deformable_asset_preparation_plan(
        preparation_id="840873-user-towel-volume-prep",
        inspection_receipt_path=receipt_path,
        expected_inspection_receipt_digest=receipt["receipt_digest"],
        source_usd_path=source,
        source_texture_root=textures,
        target_metric_dimensions_m=[0.37, 0.119, 0.113],
        physics_configuration=_physics(),
    )
    return plan, source, textures


def _readback(plan: dict[str, Any]) -> dict[str, Any]:
    required = plan["required_native_readback"]
    return {
        "stage_metadata": copy.deepcopy(required["stage_metadata"]),
        "visual_mesh": {
            "prim_path": required["visual_mesh"]["prim_path"],
            "point_count": required["visual_mesh"]["point_count"],
            "triangle_count": required["visual_mesh"]["triangle_count"],
            "source_face_topology_sha256": "sha256:" + "a" * 64,
            "output_face_topology_sha256": "sha256:" + "a" * 64,
            "dimensions_m": copy.deepcopy(required["visual_mesh"]["dimensions_m"]),
            "authored_scale_xyz": [1.0, 1.0, 1.0],
            "metric_scale_baked_into_points": True,
            "source_xform_flattened": True,
            "source_world_bounds_center_m": copy.deepcopy(
                required["visual_mesh"]["source_world_bounds_center_m"]
            ),
            "recentered_before_scale": True,
            "aabb_center_m": [0.0, 0.0, 0.0],
            "authored_pivot_m": [0.0, 0.0, 0.0],
            "placement_origin_semantics": required["visual_mesh"]["placement_origin_semantics"],
            "point_positions_sha256": "sha256:" + "d" * 64,
            "closed_volume_m3": required["visual_mesh"]["closed_volume_m3"],
        },
        "authoring_root_prim_path": required["authoring_root_prim_path"],
        "deformable_schema_prim_path": required["deformable_schema_prim_path"],
        "body_api_schemas": copy.deepcopy(required["body_api_schemas"]),
        "physics_material": copy.deepcopy(required["physics_material"]),
        "mass_properties": copy.deepcopy(required["mass_properties"]),
        "physics_material_binding": copy.deepcopy(required["physics_material_binding"]),
        "material_binding": copy.deepcopy(required["material_binding"]),
        "simulation_topology": {
            "node_count": 901,
            "element_count": 700,
            "topology_sha256": "sha256:" + "b" * 64,
        },
        "collision_topology": {
            "node_count": 811,
            "element_count": 610,
            "topology_sha256": "sha256:" + "c" * 64,
        },
        "physics_configuration": copy.deepcopy(required["physics_configuration"]),
        "texture_inventory": copy.deepcopy(required["texture_inventory"]),
        "experimental_api_schemas": [],
        "empty_tet_mesh_prim_paths": [],
        "guide_prim_paths": [],
        "light_prim_paths": [],
        "source_provider_prim_paths": [],
        "source_provider_attributes": [],
    }


class _FakeStageAPI:
    def __init__(
        self,
        plan: dict[str, Any],
        events: list[tuple[str, Any]],
        *,
        stage_context_succeeds: bool = True,
    ) -> None:
        self.plan = plan
        self.events = events
        self.stage_context_succeeds = stage_context_succeeds

    def create_clean_stage(self, **kwargs: Any) -> dict[str, Any]:
        self.events.append(("create", kwargs))
        return {"output_path": kwargs["output_path"]}

    def copy_surface_mesh_baking_points(self, **kwargs: Any) -> None:
        self.events.append(("surface", kwargs))

    def copy_bound_material_network(self, **kwargs: Any) -> None:
        self.events.append(("material", kwargs))

    def activate_and_verify_current_stage(self, *, stage: object) -> bool:
        self.events.append(("activate_current_stage", {"stage": stage}))
        return self.stage_context_succeeds

    def record_native_configuration(self, **kwargs: Any) -> None:
        self.events.append(("record_native_configuration", kwargs))

    def release_current_stage(self, *, stage: object) -> None:
        self.events.append(("release_current_stage", {"stage": stage}))

    def save_stage(self, *, stage: object) -> None:
        assert isinstance(stage, dict)
        self.events.append(("save", {}))
        Path(stage["output_path"]).write_bytes(b"#usda 1.0\nfixture-clean-stage\n")

    def readback_prepared_stage(self, **kwargs: Any) -> dict[str, Any]:
        self.events.append(("readback", kwargs))
        return _readback(self.plan)


def _execute(
    tmp_path: Path,
    plan: dict[str, Any],
    *,
    material_succeeds: bool = True,
    authoring_returns_none: bool = True,
    binding_succeeds: bool = True,
    stage_context_succeeds: bool = True,
    events_out: list[tuple[str, Any]] | None = None,
) -> tuple[dict[str, Any], Path, list[tuple[str, Any]]]:
    package_root = tmp_path / "source-package"
    build_native_deformable_asset_source_package(
        output_dir=package_root,
        plan=plan,
        expected_plan_digest=plan["plan_digest"],
    )
    events: list[tuple[str, Any]] = events_out if events_out is not None else []

    class MaterialCfg:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            events.append((DEFORMABLE_MATERIAL_CFG, kwargs))

    class BodyCfg:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            events.append((DEFORMABLE_BODY_CFG, kwargs))

    def material(prim_path: str, cfg: MaterialCfg) -> object:
        events.append((DEFORMABLE_MATERIAL_API, {"prim_path": prim_path, "cfg": cfg}))
        return object() if material_succeeds else False

    def define(prim_path: str, cfg: BodyCfg, stage: object | None = None) -> object | None:
        events.append(
            (
                DEFORMABLE_AUTHORING_API,
                {"prim_path": prim_path, "cfg": cfg, "stage": stage},
            )
        )
        return None if authoring_returns_none else object()

    def bind_physics_material(
        prim_path: str,
        material_path: str,
        stage: object | None = None,
        stronger_than_descendants: bool = True,
    ) -> object | None:
        events.append(
            (
                DEFORMABLE_PHYSICS_BINDING_API,
                {
                    "prim_path": prim_path,
                    "material_path": material_path,
                    "stage": stage,
                    "stronger_than_descendants": stronger_than_descendants,
                },
            )
        )
        # The pinned @apply_nested wrapper discards the decorated function's
        # boolean and returns None even when the binding succeeds.  A non-None
        # value therefore represents a signature/return-contract drift here.
        return None if binding_succeeds else object()

    output_root = tmp_path / "native-output"
    result = execute_native_deformable_asset_preparation(
        plan=plan,
        expected_plan_digest=plan["plan_digest"],
        package_root=package_root,
        output_root=output_root,
        stage_api=_FakeStageAPI(
            plan,
            events,
            stage_context_succeeds=stage_context_succeeds,
        ),
        native_api_registry={
            DEFORMABLE_MATERIAL_CFG: MaterialCfg,
            DEFORMABLE_MATERIAL_API: material,
            DEFORMABLE_BODY_CFG: BodyCfg,
            DEFORMABLE_AUTHORING_API: define,
            DEFORMABLE_PHYSICS_BINDING_API: bind_physics_material,
        },
    )
    return result, output_root, events


def test_plan_bakes_metric_scale_and_rebuilds_only_allowlisted_content(
    tmp_path: Path,
) -> None:
    plan, _, _ = _plan(tmp_path)

    rebuild = plan["clean_stage_rebuild"]
    assert rebuild["strategy"] == "allowlisted_surface_and_bound_material_reconstruction"
    assert rebuild["point_bake_scale_xyz"] == pytest.approx([0.37 / 0.2, 0.119 / 0.1, 0.113 / 0.08])
    assert rebuild["recenter_source_world_bounds_to_output_origin"] is True
    assert rebuild["source_world_bounds_center_m"] == pytest.approx([0.0, 0.0, 0.0])
    assert rebuild["output_authored_pivot_m"] == [0.0, 0.0, 0.0]
    assert rebuild["point_bake_scale_determinant"] == pytest.approx(
        (0.37 / 0.2) * (0.119 / 0.1) * (0.113 / 0.08)
    )
    assert rebuild["expected_baked_closed_volume_m3"] == pytest.approx(
        0.0016 * rebuild["point_bake_scale_determinant"]
    )
    assert rebuild["expected_mass_kg"] == pytest.approx(
        rebuild["expected_baked_closed_volume_m3"] * 220.0
    )
    assert rebuild["bake_metric_scale_into_points"] is True
    assert rebuild["flatten_source_xform_to_points"] is True
    assert rebuild["authored_visual_scale_xyz_after_bake"] == [1.0, 1.0, 1.0]
    assert rebuild["copy_source_prim_subtree"] is False
    assert rebuild["copy_source_api_schemas"] is False
    assert rebuild["copy_empty_source_tet_meshes"] is False
    assert rebuild["copy_guides"] is False
    assert rebuild["copy_lights"] is False
    exclusions = plan["source_content_exclusions"]
    assert {
        (row["prim_path"], row["schema"]) for row in exclusions["experimental_api_schemas"]
    }.issuperset(
        {
            ("/root/Towel", "OmniPhysicsDeformableBodyAPI"),
            ("/root/Towel", "PhysxAutoDeformableBodyAPI"),
            ("/root/Towel", "PhysxBaseDeformableBodyAPI"),
        }
    )
    assert exclusions["empty_tet_mesh_prim_paths"] == [
        "/root/Towel/CollisionMesh",
        "/root/Towel/SimulationMesh",
    ]
    assert exclusions["guide_prim_paths"] == []
    assert exclusions["light_prim_paths"] == ["/root/DomeLight"]
    assert plan["native_runtime"]["source_repository"] == ISAACLAB_REPOSITORY
    assert plan["native_runtime"]["source_revision"] == ISAACLAB_COMMIT
    assert plan["native_runtime"]["source_tree"] == ISAACLAB_TREE
    assert plan["native_runtime"]["required_api_symbols"] == [
        DEFORMABLE_MATERIAL_CFG,
        DEFORMABLE_MATERIAL_API,
        DEFORMABLE_BODY_CFG,
        DEFORMABLE_AUTHORING_API,
        DEFORMABLE_PHYSICS_BINDING_API,
    ]
    assert plan["native_runtime"]["executed_api_symbols"] == [
        DEFORMABLE_MATERIAL_API,
        DEFORMABLE_AUTHORING_API,
        DEFORMABLE_PHYSICS_BINDING_API,
    ]
    assert (
        plan["native_runtime"]["embedded_cooking_contract"]["direct_cooking_call_forbidden"] is True
    )
    assert (
        plan["native_runtime"]["embedded_cooking_contract"][
            "legacy_external_cooking_symbol_not_required"
        ]
        == DEFORMABLE_COOKING_API
    )
    assert plan["native_runtime"]["pinned_source_call_contract"] == (PINNED_NATIVE_CALL_CONTRACT)
    assert PINNED_NATIVE_CALL_CONTRACT["material_spawn"]["parameters"] == [
        "prim_path",
        "cfg",
    ]
    assert PINNED_NATIVE_CALL_CONTRACT["material_spawn"]["stage_keyword_supported"] is False
    assert (
        PINNED_NATIVE_CALL_CONTRACT["deformable_authoring"]["direct_duplicate_cook_forbidden"]
        is True
    )
    assert PINNED_NATIVE_CALL_CONTRACT["deformable_authoring"]["embedded_cooking_owner"] == (
        DEFORMABLE_AUTHORING_API
    )
    assert PINNED_NATIVE_CALL_CONTRACT["deformable_authoring"]["parameters"] == [
        "prim_path",
        "cfg",
        "stage",
        "deformable_type",
        "sim_mesh_prim_path",
    ]
    assert PINNED_NATIVE_CALL_CONTRACT["deformable_authoring"]["explicit_success_return"] is None
    assert PINNED_NATIVE_CALL_CONTRACT["physics_material_binding"] == {
        "symbol": DEFORMABLE_PHYSICS_BINDING_API,
        "source_relative_path": "source/isaaclab/isaaclab/sim/utils/prims.py",
        "source_git_blob_sha1": "d0f0e8d9042a531ce617645cdc158fa4ac81f754",
        "parameters": [
            "prim_path",
            "material_path",
            "stage",
            "stronger_than_descendants",
        ],
        "decorator_return": None,
        "readback_required": True,
        "material_purpose": "physics",
    }
    assert PINNED_NATIVE_CALL_CONTRACT["configuration_sources"][DEFORMABLE_MATERIAL_CFG][
        "allowed_fields"
    ] == [
        "density",
        "dynamic_friction",
        "elasticity_damping",
        "poissons_ratio",
        "static_friction",
        "youngs_modulus",
    ]
    assert plan["required_native_readback"]["body_api_schemas"] == sorted(DEFORMABLE_BODY_SCHEMAS)
    assert plan["required_native_readback"]["authoring_root_prim_path"] == "/Deformable"
    assert plan["required_native_readback"]["deformable_schema_prim_path"] == (
        "/Deformable/Visuals/Surface"
    )
    assert plan["required_native_readback"]["physics_material_binding"] == {
        "prim_path": "/Deformable/Visuals/Surface",
        "material_prim_path": "/Deformable/PhysicsMaterial",
        "material_purpose": "physics",
        "binding_strength": "strongerThanDescendants",
    }
    assert plan["claim_boundary"]["native_worker_executed"] is False
    assert plan["claim_boundary"]["native_cook_qualified"] is False
    assert plan["claim_boundary"]["native_simulator_qualified"] is False


def test_plan_consumes_receipt_emitted_by_static_external_inspector(
    tmp_path: Path,
) -> None:
    from tests.test_external_simready_deformable_asset import _inspect, _write_fixture

    paths = _write_fixture(
        tmp_path,
        observed_dimensions=(0.37, 0.12, 0.11),
        standard_schemas=False,
        nonempty_tetmesh=False,
        include_default_dome_light=True,
        static_rigid_contract=True,
    )
    receipt = _inspect(paths)
    receipt_path = tmp_path / "inspection-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")

    plan = materialize_native_deformable_asset_preparation_plan(
        preparation_id="inspector-contract-integration",
        inspection_receipt_path=receipt_path,
        expected_inspection_receipt_digest=receipt["receipt_digest"],
        source_usd_path=paths["usd"],
        source_texture_root=paths["root"] / "textures",
        target_metric_dimensions_m=[0.37, 0.12, 0.11],
        physics_configuration=_physics(),
    )

    assert plan["inspection_receipt_digest"] == receipt["receipt_digest"]
    assert plan["clean_stage_rebuild"]["point_bake_scale_xyz"] == pytest.approx(
        receipt["openusd_inspection"]["dimension_alignment"]["required_bake_scale_xyz"]
    )
    assert [row["relative_path"] for row in plan["textures"]] == [
        "base.png",
        "metallic.png",
        "normal.png",
        "roughness.png",
    ]
    assert "env.hdr" not in {row["relative_path"] for row in plan["textures"]}
    assert plan["source_content_exclusions"]["empty_tet_mesh_prim_paths"] == [
        "/root/Towel/CollisionMesh",
        "/root/Towel/SimulationMesh",
    ]
    assert plan["source_content_exclusions"]["light_prim_paths"] == ["/root/DomeLight"]


def test_nonzero_source_world_center_is_recentered_and_bound_to_pose_semantics(
    tmp_path: Path,
) -> None:
    from tests.test_external_simready_deformable_asset import (
        _inspect,
        _rebuild_archive,
        _write_fixture,
    )

    paths = _write_fixture(tmp_path, observed_dimensions=(0.37, 0.119, 0.113))
    usd_text = paths["usd"].read_text(encoding="utf-8")
    marker = "    {\n        rel material:binding:physics = </root/Towel/Physics/PhysicsMaterial>"
    replacement = (
        "    {\n        double3 xformOp:translate = (0.25, -0.5, 1.0)\n"
        '        uniform token[] xformOpOrder = ["xformOp:translate"]\n'
        "        rel material:binding:physics = </root/Towel/Physics/PhysicsMaterial>"
    )
    assert marker in usd_text
    paths["usd"].write_text(usd_text.replace(marker, replacement, 1), encoding="utf-8")
    _rebuild_archive(paths)
    receipt = _inspect(paths)
    receipt_path = _write_receipt(tmp_path / "translated-inspection.json", receipt)

    plan = materialize_native_deformable_asset_preparation_plan(
        preparation_id="translated-origin",
        inspection_receipt_path=receipt_path,
        expected_inspection_receipt_digest=receipt["receipt_digest"],
        source_usd_path=paths["usd"],
        source_texture_root=paths["root"] / "textures",
        target_metric_dimensions_m=[0.37, 0.119, 0.113],
        physics_configuration=_physics(),
    )

    assert plan["source_surface_mesh"]["world_bounds_center_m"] == pytest.approx([0.25, -0.5, 1.0])
    assert plan["clean_stage_rebuild"]["source_world_bounds_center_m"] == pytest.approx(
        [0.25, -0.5, 1.0]
    )
    assert plan["clean_stage_rebuild"]["recenter_source_world_bounds_to_output_origin"] is True
    required = plan["required_native_readback"]["visual_mesh"]
    assert required["aabb_center_m"] == [0.0, 0.0, 0.0]
    assert required["authored_pivot_m"] == [0.0, 0.0, 0.0]
    assert required["placement_origin_semantics"] == (
        "body_pose_translation_is_replacement_aabb_center"
    )


def test_non_z_up_source_fails_until_explicit_axis_conversion_exists(tmp_path: Path) -> None:
    from tests.test_external_simready_deformable_asset import (
        _inspect,
        _rebuild_archive,
        _write_fixture,
    )

    paths = _write_fixture(tmp_path, observed_dimensions=(0.37, 0.119, 0.113))
    usd_text = paths["usd"].read_text(encoding="utf-8")
    assert 'upAxis = "Z"' in usd_text
    paths["usd"].write_text(
        usd_text.replace('upAxis = "Z"', 'upAxis = "Y"', 1),
        encoding="utf-8",
    )
    _rebuild_archive(paths)
    receipt = _inspect(paths)
    receipt_path = _write_receipt(tmp_path / "y-up-inspection.json", receipt)

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_source_non_z_up_requires_explicit_conversion",
    ):
        materialize_native_deformable_asset_preparation_plan(
            preparation_id="y-up",
            inspection_receipt_path=receipt_path,
            expected_inspection_receipt_digest=receipt["receipt_digest"],
            source_usd_path=paths["usd"],
            source_texture_root=paths["root"] / "textures",
            target_metric_dimensions_m=[0.37, 0.119, 0.113],
            physics_configuration=_physics(),
        )


def test_source_package_preserves_exact_usd_and_nested_texture_bytes(tmp_path: Path) -> None:
    plan, source, textures = _plan(tmp_path)
    output = tmp_path / "package"

    receipt = build_native_deformable_asset_source_package(
        output_dir=output,
        plan=plan,
        expected_plan_digest=plan["plan_digest"],
    )

    assert (output / "source/asset.usd").read_bytes() == source.read_bytes()
    for row in plan["textures"]:
        assert (output / row["package_path"]).read_bytes() == (
            textures / row["relative_path"]
        ).read_bytes()
    assert receipt["claim_boundary"]["native_worker_executed"] is False
    assert receipt["claim_boundary"]["native_cook_qualified"] is False
    assert receipt["claim_boundary"]["native_simulator_qualified"] is False
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    assert {row["role"] for row in receipt["files"]} == {
        "source_usd",
        "texture",
        "preparation_plan",
    }


def test_injected_native_worker_calls_all_pinned_apis_and_verifier_stays_bounded(
    tmp_path: Path,
) -> None:
    plan, _, _ = _plan(tmp_path)

    worker_return, output_root, events = _execute(tmp_path, plan)
    verification = verify_native_deformable_asset_preparation_return(
        plan=plan,
        expected_plan_digest=plan["plan_digest"],
        worker_return=worker_return,
        output_root=output_root,
    )

    names = [event[0] for event in events]
    assert names == [
        "create",
        "surface",
        "material",
        "activate_current_stage",
        DEFORMABLE_MATERIAL_CFG,
        DEFORMABLE_MATERIAL_API,
        DEFORMABLE_BODY_CFG,
        DEFORMABLE_AUTHORING_API,
        DEFORMABLE_PHYSICS_BINDING_API,
        "record_native_configuration",
        "save",
        "release_current_stage",
        "readback",
    ]
    surface_call = events[1][1]
    assert surface_call["bake_scale_xyz"] == plan["clean_stage_rebuild"]["point_bake_scale_xyz"]
    assert surface_call["flatten_source_xform"] is True
    assert surface_call["recenter_to_output_origin"] is True
    assert surface_call["source_world_bounds_center_m"] == pytest.approx([0.0, 0.0, 0.0])
    assert ".source_snapshot" in surface_call["source_usd_path"].parts
    assert not surface_call["source_usd_path"].exists()
    material_call = events[2][1]
    assert set(material_call["source_texture_paths"]) == {
        "base.png",
        "metallic.png",
        "normal.png",
        "roughness.png",
    }
    assert material_call["output_texture_asset_paths"] == {
        name: f"textures/{name}"
        for name in ("base.png", "metallic.png", "normal.png", "roughness.png")
    }
    assert events[3][1]["stage"] is events[7][1]["stage"]
    assert events[3][1]["stage"] is events[8][1]["stage"]
    assert events[4][1] == _physics()["material_properties"]
    assert events[5][1]["cfg"].kwargs == _physics()["material_properties"]
    assert events[6][1] == {
        **_physics()["body_properties"],
        **_physics()["cooking_properties"],
    }
    assert events[7][1]["cfg"].kwargs == events[6][1]
    assert events[8][1] == {
        "prim_path": "/Deformable/Visuals/Surface",
        "material_path": "/Deformable/PhysicsMaterial",
        "stage": events[7][1]["stage"],
        "stronger_than_descendants": True,
    }
    assert events[9][1] == {
        "stage": events[7][1]["stage"],
        "body_and_cooking_properties": {
            **_physics()["body_properties"],
            **_physics()["cooking_properties"],
        },
        "material_properties": _physics()["material_properties"],
    }
    assert DEFORMABLE_COOKING_API not in names
    assert worker_return["readback"]["empty_tet_mesh_prim_paths"] == []
    assert verification["status"] == ("worker_payload_verified_pending_trusted_execution_join")
    assert verification["readback_contract_satisfied"] is True
    assert verification["claim_boundary"] == {
        "worker_payload_and_returned_bytes_structurally_verified": True,
        "trusted_native_execution_join_present": False,
        "native_cook_qualified": False,
        "native_simulator_qualified": False,
        "visual_alignment_qualified": False,
        "physical_material_equivalence": False,
    }


@pytest.mark.parametrize(
    ("execution_kwargs", "error"),
    [
        (
            {"material_succeeds": False},
            "native_deformable_material_api_return_invalid",
        ),
        (
            {"authoring_returns_none": False},
            "native_deformable_authoring_api_return_contract_invalid",
        ),
        (
            {"binding_succeeds": False},
            "native_deformable_physics_material_binding_return_contract_invalid",
        ),
        (
            {"stage_context_succeeds": False},
            "native_deformable_material_spawn_stage_context_invalid",
        ),
    ],
)
def test_native_api_returns_and_physics_binding_fail_closed(
    tmp_path: Path,
    execution_kwargs: dict[str, bool],
    error: str,
) -> None:
    plan, _, _ = _plan(tmp_path)
    with pytest.raises(NativeDeformableAssetPreparationError, match=error):
        _execute(tmp_path, plan, **execution_kwargs)


def test_native_api_failure_releases_acquired_current_stage_context(tmp_path: Path) -> None:
    plan, _, _ = _plan(tmp_path)
    events: list[tuple[str, Any]] = []

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_material_api_return_invalid",
    ):
        _execute(
            tmp_path,
            plan,
            material_succeeds=False,
            events_out=events,
        )

    names = [event[0] for event in events]
    assert "save" not in names
    assert names[-1] == "release_current_stage"


def test_unsupported_damping_scale_cannot_enter_pinned_deformable_material_cfg(
    tmp_path: Path,
) -> None:
    source, textures, receipt_path, receipt = _source_fixture(tmp_path)
    physics = _physics()
    physics["material_properties"]["damping_scale"] = 1.0

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_physics_material_cfg_fields_unsupported",
    ):
        materialize_native_deformable_asset_preparation_plan(
            preparation_id="unsupported-material-field",
            inspection_receipt_path=receipt_path,
            expected_inspection_receipt_digest=receipt["receipt_digest"],
            source_usd_path=source,
            source_texture_root=textures,
            target_metric_dimensions_m=[0.37, 0.119, 0.113],
            physics_configuration=physics,
        )


def test_public_boundary_requires_exact_receipt_path_and_independent_digest() -> None:
    signature = inspect.signature(materialize_native_deformable_asset_preparation_plan)
    assert tuple(signature.parameters) == (
        "preparation_id",
        "inspection_receipt_path",
        "expected_inspection_receipt_digest",
        "source_usd_path",
        "source_texture_root",
        "target_metric_dimensions_m",
        "physics_configuration",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert tuple(inspect.signature(build_native_deformable_asset_source_package).parameters) == (
        "output_dir",
        "plan",
        "expected_plan_digest",
    )
    assert tuple(inspect.signature(execute_native_deformable_asset_preparation).parameters) == (
        "plan",
        "expected_plan_digest",
        "package_root",
        "output_root",
        "stage_api",
        "native_api_registry",
    )


def test_caller_rehashed_receipt_cannot_self_promote_claims(tmp_path: Path) -> None:
    source, textures, receipt_path, receipt = _source_fixture(tmp_path)
    expected_digest = receipt["receipt_digest"]
    receipt["claim_ceiling"]["native_simulator_qualified"] = True
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_inspection_receipt_expected_digest_mismatch",
    ):
        materialize_native_deformable_asset_preparation_plan(
            preparation_id="fixture",
            inspection_receipt_path=receipt_path,
            expected_inspection_receipt_digest=expected_digest,
            source_usd_path=source,
            source_texture_root=textures,
            target_metric_dimensions_m=[0.37, 0.119, 0.113],
            physics_configuration=_physics(),
        )


def test_observation_substitution_fails_external_inspector_replay(tmp_path: Path) -> None:
    source, textures, receipt_path, receipt = _source_fixture(tmp_path)
    observation_path = Path(receipt["input_paths"]["observation"])
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    observation["dimensions_m"] = [9.0, 9.0, 9.0]
    observation_path.write_text(json.dumps(observation, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match=(
            "native_deformable_inspection_replay_failed:"
            "external_simready_deformable_source_topology_dimensions_mismatch"
        ),
    ):
        materialize_native_deformable_asset_preparation_plan(
            preparation_id="observation-substitution",
            inspection_receipt_path=receipt_path,
            expected_inspection_receipt_digest=receipt["receipt_digest"],
            source_usd_path=source,
            source_texture_root=textures,
            target_metric_dimensions_m=[0.37, 0.119, 0.113],
            physics_configuration=_physics(),
        )


def test_source_bytes_cannot_change_between_inspection_plan_and_package(tmp_path: Path) -> None:
    plan, source, _ = _plan(tmp_path)
    source.write_bytes(source.read_bytes() + b"tampered")

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_source_changed_after_plan",
    ):
        build_native_deformable_asset_source_package(
            output_dir=tmp_path / "package",
            plan=plan,
            expected_plan_digest=plan["plan_digest"],
        )


def test_recomputed_plan_digest_cannot_remove_a_pinned_native_api_call(
    tmp_path: Path,
) -> None:
    plan, _, _ = _plan(tmp_path)
    expected_plan_digest = plan["plan_digest"]
    plan["native_runtime"]["api_calls_in_order"].pop()
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_preparation_expected_plan_digest_mismatch",
    ):
        build_native_deformable_asset_source_package(
            output_dir=tmp_path / "package",
            plan=plan,
            expected_plan_digest=expected_plan_digest,
        )


def test_recomputed_plan_cannot_change_scale_or_claim_even_with_matching_new_digest(
    tmp_path: Path,
) -> None:
    plan, _, _ = _plan(tmp_path)
    plan["clean_stage_rebuild"]["point_bake_scale_xyz"] = [9.0, 9.0, 9.0]
    plan["claim_boundary"]["physical_material_equivalence"] = True
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    with pytest.raises(NativeDeformableAssetPreparationError) as caught:
        build_native_deformable_asset_source_package(
            output_dir=tmp_path / "forged-package",
            plan=plan,
            expected_plan_digest=plan["plan_digest"],
        )
    assert {
        "native_deformable_preparation_rebuild_contract_invalid",
        "native_deformable_preparation_claim_boundary_invalid",
    }.issubset(caught.value.errors)


def test_exact_source_usd_path_cannot_be_a_symlink(tmp_path: Path) -> None:
    source, textures, receipt_path, receipt = _source_fixture(tmp_path)
    alias = source.with_name("asset-alias.usd")
    alias.symlink_to(source.name)

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match=(
            "native_deformable_inspection_replay_failed:"
            "external_simready_deformable_expanded_symlink_forbidden"
        ),
    ):
        materialize_native_deformable_asset_preparation_plan(
            preparation_id="symlink-fixture",
            inspection_receipt_path=receipt_path,
            expected_inspection_receipt_digest=receipt["receipt_digest"],
            source_usd_path=alias,
            source_texture_root=textures,
            target_metric_dimensions_m=[0.37, 0.119, 0.113],
            physics_configuration=_physics(),
        )


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda result: result["readback"].__setitem__(
                "body_api_schemas", ["provider:experimental"]
            ),
            "native_deformable_return_body_schemas_mismatch",
        ),
        (
            lambda result: result["readback"].__setitem__(
                "deformable_schema_prim_path", "/Deformable"
            ),
            "native_deformable_return_schema_prim_mismatch",
        ),
        (
            lambda result: result["readback"]["physics_material_binding"].__setitem__(
                "material_purpose", "allPurpose"
            ),
            "native_deformable_return_physics_material_binding_mismatch",
        ),
        (
            lambda result: result["readback"]["empty_tet_mesh_prim_paths"].append(
                "/Deformable/EmptySimulationMesh"
            ),
            "native_deformable_return_forbidden_empty_tet_mesh_prim_paths_present",
        ),
        (
            lambda result: result["readback"]["simulation_topology"].__setitem__(
                "element_count", 0
            ),
            "native_deformable_return_simulation_element_count_invalid",
        ),
        (
            lambda result: result["api_calls"].reverse(),
            "native_deformable_return_api_calls_mismatch",
        ),
        (
            lambda result: result.__setitem__("native_cook_qualified", True),
            "native_deformable_return_fields_invalid",
        ),
    ],
)
def test_worker_return_cannot_self_promote_or_omit_native_readback(
    tmp_path: Path,
    mutate: Any,
    error: str,
) -> None:
    plan, _, _ = _plan(tmp_path)
    worker_return, output_root, _ = _execute(tmp_path, plan)
    mutate(worker_return)
    worker_return["worker_result_digest"] = canonical_digest(
        worker_return, digest_field="worker_result_digest"
    )

    with pytest.raises(NativeDeformableAssetPreparationError, match=error):
        verify_native_deformable_asset_preparation_return(
            plan=plan,
            expected_plan_digest=plan["plan_digest"],
            worker_return=worker_return,
            output_root=output_root,
        )


def test_worker_return_is_bound_to_exact_output_bytes(tmp_path: Path) -> None:
    plan, _, _ = _plan(tmp_path)
    worker_return, output_root, _ = _execute(tmp_path, plan)
    output_usd = output_root / "prepared/deformable.usda"
    output_usd.write_bytes(output_usd.read_bytes() + b"tampered")

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_return_runtime_usd_identity_mismatch",
    ):
        verify_native_deformable_asset_preparation_return(
            plan=plan,
            expected_plan_digest=plan["plan_digest"],
            worker_return=worker_return,
            output_root=output_root,
        )


def test_worker_return_rejects_uninventoried_output_members(tmp_path: Path) -> None:
    plan, _, _ = _plan(tmp_path)
    worker_return, output_root, _ = _execute(tmp_path, plan)
    (output_root / "prepared/provider-light.usda").write_text(
        '#usda 1.0\ndef DomeLight "Unadmitted" {}\n', encoding="utf-8"
    )

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_return_output_member_set_invalid",
    ):
        verify_native_deformable_asset_preparation_return(
            plan=plan,
            expected_plan_digest=plan["plan_digest"],
            worker_return=worker_return,
            output_root=output_root,
        )


def test_worker_output_inventory_enforces_resource_count_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _, _ = _plan(tmp_path)
    worker_return, output_root, _ = _execute(tmp_path, plan)
    monkeypatch.setattr(preparation, "_MAX_OUTPUT_FILE_COUNT", 5)

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_return_output_resource_limit_exceeded",
    ):
        verify_native_deformable_asset_preparation_return(
            plan=plan,
            expected_plan_digest=plan["plan_digest"],
            worker_return=worker_return,
            output_root=output_root,
        )


def test_persisted_nested_claim_injection_is_rejected(tmp_path: Path) -> None:
    plan, _, _ = _plan(tmp_path)
    worker_return, output_root, _ = _execute(tmp_path, plan)
    worker_return["output_artifacts"]["runtime_usd"]["native_cook_qualified"] = True
    worker_return["readback"]["visual_mesh"]["physically_equivalent"] = True
    worker_return["worker_result_digest"] = canonical_digest(
        worker_return, digest_field="worker_result_digest"
    )
    (output_root / "worker_return.json").write_text(
        json.dumps(worker_return, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(NativeDeformableAssetPreparationError) as caught:
        verify_native_deformable_asset_preparation_return(
            plan=plan,
            expected_plan_digest=plan["plan_digest"],
            worker_return=worker_return,
            output_root=output_root,
        )
    assert {
        "native_deformable_return_runtime_usd_fields_invalid",
        "native_deformable_return_visual_mesh_fields_invalid",
    }.issubset(caught.value.errors)


def test_immutable_file_read_is_anchored_when_an_ancestor_is_swapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = tmp_path / "immutable-artifact-swap"
    trusted_parent = container / "trusted"
    trusted_leaf = trusted_parent / "receipts"
    trusted_leaf.mkdir(parents=True)
    artifact_path = trusted_leaf / "receipt.json"
    trusted_content = b'{"authority":"trusted-snapshot"}\n'
    artifact_path.write_bytes(trusted_content)
    replacement_parent = container / "replacement"
    replacement_leaf = replacement_parent / "receipts"
    replacement_leaf.mkdir(parents=True)
    (replacement_leaf / artifact_path.name).write_bytes(b'{"authority":"substituted"}\n')
    displaced_parent = container / "displaced"
    real_open = os.open
    swapped = False

    def swap_after_final_open(
        path: str | bytes | int,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == artifact_path.name and dir_fd is not None and not swapped:
            trusted_parent.rename(displaced_parent)
            replacement_parent.rename(trusted_parent)
            swapped = True
        return descriptor

    monkeypatch.setattr(preparation.os, "open", swap_after_final_open)
    snapshot = preparation._read_regular_file_once(
        artifact_path,
        maximum_size=1024,
        expected_digest=f"sha256:{hashlib.sha256(trusted_content).hexdigest()}",
        expected_size=len(trusted_content),
        error="immutable_artifact_snapshot_invalid",
    )

    assert swapped is True
    assert snapshot == trusted_content
    assert artifact_path.read_bytes() == b'{"authority":"substituted"}\n'


def test_immutable_file_read_rejects_parent_component_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = tmp_path / "parent-component-swap"
    trusted_parent = container / "trusted"
    receipt_directory = trusted_parent / "receipts"
    receipt_directory.mkdir(parents=True)
    artifact_path = receipt_directory / "receipt.json"
    trusted_content = b'{"authority":"trusted-snapshot"}\n'
    artifact_path.write_bytes(trusted_content)
    replacement_directory = trusted_parent / "replacement-receipts"
    replacement_directory.mkdir()
    (replacement_directory / artifact_path.name).write_bytes(b'{"authority":"substituted"}\n')
    displaced_directory = trusted_parent / "displaced-receipts"
    real_open = os.open
    swapped = False

    def swap_opened_parent_component(
        path: str | bytes | int,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "receipts" and dir_fd is not None and not swapped:
            receipt_directory.rename(displaced_directory)
            replacement_directory.rename(receipt_directory)
            swapped = True
        return descriptor

    monkeypatch.setattr(preparation.os, "open", swap_opened_parent_component)
    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="immutable_artifact_snapshot_invalid",
    ):
        preparation._read_regular_file_once(
            artifact_path,
            maximum_size=1024,
            expected_digest=f"sha256:{hashlib.sha256(trusted_content).hexdigest()}",
            expected_size=len(trusted_content),
            error="immutable_artifact_snapshot_invalid",
        )

    assert swapped is True


def test_output_snapshot_counts_empty_directories_against_resource_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _, _ = _plan(tmp_path)
    worker_return, output_root, _ = _execute(tmp_path, plan)
    (output_root / "empty-a").mkdir()
    (output_root / "empty-b").mkdir()
    (output_root / "empty-c").mkdir()
    monkeypatch.setattr(preparation, "_MAX_OUTPUT_DIRECTORY_COUNT", 4)

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_return_output_resource_limit_exceeded",
    ):
        verify_native_deformable_asset_preparation_return(
            plan=plan,
            expected_plan_digest=plan["plan_digest"],
            worker_return=worker_return,
            output_root=output_root,
        )


def test_output_snapshot_rejects_a_bounded_uninventoried_empty_directory(
    tmp_path: Path,
) -> None:
    plan, _, _ = _plan(tmp_path)
    worker_return, output_root, _ = _execute(tmp_path, plan)
    (output_root / "uninventoried-empty").mkdir()

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_return_output_member_set_invalid",
    ):
        verify_native_deformable_asset_preparation_return(
            plan=plan,
            expected_plan_digest=plan["plan_digest"],
            worker_return=worker_return,
            output_root=output_root,
        )


def test_output_snapshot_does_not_retain_an_oversized_worker_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _, _ = _plan(tmp_path)
    worker_return, output_root, _ = _execute(tmp_path, plan)
    persisted_size = (output_root / "worker_return.json").stat().st_size
    monkeypatch.setattr(preparation, "_MAX_WORKER_RETURN_BYTES", persisted_size - 1)

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_return_persisted_payload_invalid",
    ):
        verify_native_deformable_asset_preparation_return(
            plan=plan,
            expected_plan_digest=plan["plan_digest"],
            worker_return=worker_return,
            output_root=output_root,
        )


def test_output_snapshot_rejects_cross_file_torn_in_place_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _, _ = _plan(tmp_path)
    worker_return, output_root, _ = _execute(tmp_path, plan)
    runtime_path = output_root / "prepared/deformable.usda"
    worker_return_path = output_root / "worker_return.json"
    valid_runtime = runtime_path.read_bytes()
    valid_worker_return = worker_return_path.read_bytes()
    invalid_worker_return = b"[" + valid_worker_return[1:]
    assert len(invalid_worker_return) == len(valid_worker_return)
    worker_return_path.write_bytes(invalid_worker_return)
    real_snapshot = preparation._snapshot_open_regular_file
    mutation_performed = False

    def tear_after_runtime_read(**kwargs: Any) -> Any:
        nonlocal mutation_performed
        snapshot = real_snapshot(**kwargs)
        if kwargs["name"] == "deformable.usda" and not mutation_performed:
            runtime_path.write_bytes(b"X" * len(valid_runtime))
            worker_return_path.write_bytes(valid_worker_return)
            mutation_performed = True
        return snapshot

    monkeypatch.setattr(
        preparation,
        "_snapshot_open_regular_file",
        tear_after_runtime_read,
    )
    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_return_output_member_set_invalid",
    ):
        verify_native_deformable_asset_preparation_return(
            plan=plan,
            expected_plan_digest=plan["plan_digest"],
            worker_return=worker_return,
            output_root=output_root,
        )

    assert mutation_performed is True
    assert runtime_path.read_bytes() != valid_runtime
    assert worker_return_path.read_bytes() == valid_worker_return


def test_output_snapshot_descriptor_cap_reserves_process_rlimit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _, _ = _plan(tmp_path)
    worker_return, output_root, _ = _execute(tmp_path, plan)
    monkeypatch.setattr(
        preparation.resource,
        "getrlimit",
        lambda _limit: (preparation._OUTPUT_SNAPSHOT_DESCRIPTOR_RESERVE + 4, 4096),
    )

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_return_output_resource_limit_exceeded",
    ):
        verify_native_deformable_asset_preparation_return(
            plan=plan,
            expected_plan_digest=plan["plan_digest"],
            worker_return=worker_return,
            output_root=output_root,
        )


def test_output_verification_uses_one_anchored_snapshot_when_parent_is_swapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _, _ = _plan(tmp_path)
    worker_return, generated_output, _ = _execute(tmp_path, plan)
    container = tmp_path / "output-parent-swap"
    trusted_parent = container / "trusted"
    trusted_parent.mkdir(parents=True)
    output_root = trusted_parent / "native-output"
    generated_output.rename(output_root)
    replacement_parent = container / "replacement"
    shutil.copytree(trusted_parent, replacement_parent)
    (replacement_parent / "native-output/prepared/deformable.usda").write_bytes(
        b"#usda 1.0\nsubstituted-output\n"
    )
    displaced_parent = container / "displaced"
    real_open = os.open
    swapped = False

    def swap_after_output_root_open(
        path: str | bytes | int,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "native-output" and dir_fd is not None and not swapped:
            trusted_parent.rename(displaced_parent)
            replacement_parent.rename(trusted_parent)
            swapped = True
        return descriptor

    monkeypatch.setattr(preparation.os, "open", swap_after_output_root_open)
    verification = verify_native_deformable_asset_preparation_return(
        plan=plan,
        expected_plan_digest=plan["plan_digest"],
        worker_return=worker_return,
        output_root=output_root,
    )

    assert swapped is True
    assert verification["status"] == "worker_payload_verified_pending_trusted_execution_join"
    assert output_root.joinpath("prepared/deformable.usda").read_bytes() == (
        b"#usda 1.0\nsubstituted-output\n"
    )


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "relative_path"),
    [
        ("_MAX_OUTPUT_ENTRY_COUNT", 1, "one/two.bin"),
        ("_MAX_OUTPUT_DEPTH", 1, "one/two.bin"),
        ("_MAX_OUTPUT_RELATIVE_PATH_BYTES", 8, "long-name.bin"),
        ("_MAX_OUTPUT_TOTAL_BYTES", 3, "four.bin"),
    ],
)
def test_output_snapshot_enforces_each_non_file_resource_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: int,
    relative_path: str,
) -> None:
    output = tmp_path / f"bounded-{limit_name}"
    artifact = output / relative_path
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"four")
    monkeypatch.setattr(preparation, limit_name, limit_value)

    with pytest.raises(
        NativeDeformableAssetPreparationError,
        match="native_deformable_return_output_resource_limit_exceeded",
    ):
        preparation._snapshot_output_tree(
            output,
            retained_content_limits={},
            invalid_error="native_deformable_return_output_member_set_invalid",
            resource_error="native_deformable_return_output_resource_limit_exceeded",
        )
