from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.external_simready_deformable_asset import (
    CLAIM_CEILING,
    OBSERVATION_SCHEMA_VERSION,
    PENDING_STATUS,
    REJECTED_STATUS,
    SOURCE_TOPOLOGY_SCHEMA_VERSION,
    ExternalSimreadyDeformableAssetError,
    inspect_external_simready_deformable_asset,
    verify_external_simready_deformable_asset_inspection,
)
import blueprint_pipeline.external_simready_deformable_asset as asset_module


pytest.importorskip("pxr")

_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk/w8AAusB9Y9ZrroAAAAASUVORK5CYII="
)
_HDR_1X1 = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n" + bytes((128, 128, 128, 129))


def _mesh_usda(*, empty_surface: bool) -> str:
    if empty_surface:
        return """
                point3f[] points = []
                int[] faceVertexCounts = []
                int[] faceVertexIndices = []
        """
    points = [
        (-0.1, -0.05, -0.04),
        (0.1, -0.05, -0.04),
        (0.1, 0.05, -0.04),
        (-0.1, 0.05, -0.04),
        (-0.1, -0.05, 0.04),
        (0.1, -0.05, 0.04),
        (0.1, 0.05, 0.04),
        (-0.1, 0.05, 0.04),
    ]
    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    point_text = ", ".join(f"({x}, {y}, {z})" for x, y, z in points)
    index_text = ", ".join(str(index) for face in faces for index in face)
    count_text = ", ".join("3" for _ in faces)
    return f"""
                point3f[] points = [{point_text}]
                int[] faceVertexCounts = [{count_text}]
                int[] faceVertexIndices = [{index_text}]
        """


def _fixture_usda(
    *,
    standard_schemas: bool = False,
    nonempty_tetmesh: bool = False,
    include_default_dome_light: bool = True,
    static_rigid_contract: bool = True,
    empty_surface: bool = False,
    schema_decoy_attribute: bool = False,
    unrelated_standard_decoy: bool = False,
    body_scoped_tet_decoy: bool = False,
    include_visual_material_binding: bool = True,
    include_physics_material_binding: bool = True,
    meters_per_unit: float = 1.0,
) -> str:
    body_schemas = [
        "OmniPhysicsDeformableBodyAPI",
        "PhysxBaseDeformableBodyAPI",
        "PhysxAutoDeformableBodyAPI",
        "MaterialBindingAPI",
    ]
    material_schemas = ["OmniPhysicsDeformableMaterialAPI"]
    if standard_schemas:
        body_schemas.append("PhysxDeformableBodyAPI")
        material_schemas.append("PhysxDeformableBodyMaterialAPI")
    body_schema_text = ", ".join(f'"{schema}"' for schema in body_schemas)
    material_schema_text = ", ".join(f'"{schema}"' for schema in material_schemas)
    rigid_marker = (
        '    custom string lightwheelusd:assetFormat = "static_rigid_usd"\n'
        if static_rigid_contract
        else ""
    )
    decoy = "        custom bool PhysxDeformableBodyAPI = true\n" if schema_decoy_attribute else ""
    physics_binding = (
        "        rel material:binding:physics = </root/Towel/Physics/PhysicsMaterial>\n"
        if include_physics_material_binding
        else ""
    )
    visual_binding = (
        "                rel material:binding = </root/Looks/Material>\n"
        if include_visual_material_binding
        else ""
    )
    tet_topology = (
        """
            point3f[] points = [(0, 0, 0), (0.02, 0, 0), (0, 0.02, 0), (0, 0, 0.02)]
            int4[] tetVertexIndices = [(0, 1, 2, 3)]
        """
        if nonempty_tetmesh
        else ""
    )
    body_tet_decoy = (
        """
        def TetMesh "UnrelatedBodyScopedTet"
        {
            point3f[] points = [(0, 0, 0), (0.02, 0, 0), (0, 0.02, 0), (0, 0, 0.02)]
            int4[] tetVertexIndices = [(0, 1, 2, 3)]
        }
"""
        if body_scoped_tet_decoy
        else ""
    )
    dome_light = (
        """
    def DomeLight "DomeLight"
    {
        float inputs:intensity = 1000
        asset inputs:texture:file = @textures/env.hdr@
    }
        """
        if include_default_dome_light
        else ""
    )
    mesh = _mesh_usda(empty_surface=empty_surface)
    standard_decoy = (
        """
    def Xform "UnrelatedDecoy" (
        apiSchemas = ["PhysxDeformableBodyAPI"]
    )
    {
        def Material "UnrelatedPhysicsMaterial" (
            apiSchemas = ["PhysxDeformableBodyMaterialAPI"]
        )
        {
        }
        def TetMesh "UnrelatedTet"
        {
            point3f[] points = [(0, 0, 0), (0.02, 0, 0), (0, 0.02, 0), (0, 0, 0.02)]
            int4[] tetVertexIndices = [(0, 1, 2, 3)]
        }
    }
"""
        if unrelated_standard_decoy
        else ""
    )
    return f'''#usda 1.0
(
    defaultPrim = "root"
    doc = """Blender v4.2.3 LTS

Generated from Composed Stage of root layer /provider/run/source_raw.usdz
"""
    metersPerUnit = {meters_per_unit}
    upAxis = "Z"
    startTimeCode = 1
    endTimeCode = 250
    timeCodesPerSecond = 24
)

def Xform "root"
{{
{rigid_marker}    custom string lightwheelusd:hierarchyContract = "autopipeline_rigid_v1"

    def Scope "Looks"
    {{
        def Material "Material"
        {{
            token outputs:surface.connect = </root/Looks/Material/Preview.outputs:surface>
            def Shader "Preview"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </root/Looks/Material/Base.outputs:rgb>
                float inputs:metallic.connect = </root/Looks/Material/Metallic.outputs:r>
                float inputs:roughness.connect = </root/Looks/Material/Roughness.outputs:r>
                normal3f inputs:normal.connect = </root/Looks/Material/Normal.outputs:rgb>
                token outputs:surface
            }}
            def Shader "Base"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @textures/base.png@
                float3 outputs:rgb
            }}
            def Shader "Metallic"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @textures/metallic.png@
                float outputs:r
            }}
            def Shader "Roughness"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @textures/roughness.png@
                float outputs:r
            }}
            def Shader "Normal"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @textures/normal.png@
                float3 outputs:rgb
            }}
        }}
    }}

    def Xform "Towel" (
        apiSchemas = [{body_schema_text}]
    )
    {{
{decoy}{physics_binding}
        custom bool omniphysics:deformableBodyEnabled = true
        custom float omniphysics:mass = 0.352
        custom rel physxDeformableBody:cookingSourceMesh = </root/Towel/Visuals/TowelMesh>
        custom bool physxDeformableBody:selfCollision = true
        custom uint physxDeformableBody:solverPositionIterationCount = 28

        def Xform "Visuals"
        {{
            def Mesh "TowelMesh" (
                apiSchemas = ["MaterialBindingAPI"]
            )
            {{
{mesh}{visual_binding}
                bool doubleSided = true
                uniform token subdivisionScheme = "none"
            }}
        }}

        def TetMesh "SimulationMesh" (
            apiSchemas = ["OmniPhysicsVolumeDeformableSimAPI"]
        )
        {{
{tet_topology}        }}

        def TetMesh "CollisionMesh" (
            apiSchemas = ["PhysxCollisionAPI", "PhysicsCollisionAPI"]
        )
        {{
            custom float physxCollision:contactOffset = 0.003
            custom float physxCollision:restOffset = 0.001
{tet_topology}        }}
{body_tet_decoy}

        def Scope "Physics"
        {{
            def Material "PhysicsMaterial" (
                apiSchemas = [{material_schema_text}]
            )
            {{
                custom float omniphysics:density = 220
                custom float omniphysics:dynamicFriction = 2.2
                custom float omniphysics:poissonsRatio = 0.42
                custom float omniphysics:staticFriction = 2.2
                custom float omniphysics:youngsModulus = 180000
            }}
        }}
    }}
{standard_decoy}{dome_light}}}
'''


def _write_fixture(
    tmp_path: Path,
    *,
    observed_dimensions: tuple[float, float, float] = (0.37, 0.12, 0.11),
    **usda_options: bool,
) -> dict[str, Path]:
    root = tmp_path / "asset"
    textures = root / "textures"
    textures.mkdir(parents=True)
    usd = root / "rolled_towel.usda"
    usd.write_text(_fixture_usda(**usda_options), encoding="utf-8")
    for name in ("base.png", "metallic.png", "roughness.png", "normal.png"):
        (textures / name).write_bytes(_PNG_1X1)
    (textures / "env.hdr").write_bytes(_HDR_1X1)

    archive = tmp_path / "rolled_towel.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(root).as_posix())

    topology = tmp_path / "source_topology.json"
    topology_receipt = {
        "schema_version": SOURCE_TOPOLOGY_SCHEMA_VERSION,
        "receipt_digest": "",
        "all_component_collision_identities_passed": True,
        "coordinate_frame": {
            "meters_per_unit": 1.0,
            "up_axis": "Z",
        },
        "targets": [
            {
                "interiorgs_instance_id": "79",
                "semantic_label": "towel",
                "component_collision_identity_passed": True,
                "best_component": {
                    "collision_api_applied": True,
                    "geometry_digest": f"sha256:{'2' * 64}",
                    "prim_path": "/Root/Collision",
                    "world_aabb_size_m": list(observed_dimensions),
                },
            }
        ],
    }
    topology_receipt["receipt_digest"] = canonical_digest(
        topology_receipt, digest_field="receipt_digest"
    )
    topology_bytes = (
        json.dumps(topology_receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    topology.write_bytes(topology_bytes)

    observation = tmp_path / "observation.json"
    observation.write_text(
        json.dumps(
            {
                "schema_version": OBSERVATION_SCHEMA_VERSION,
                "entity_id": "inserted_deformable",
                "source_topology_receipt_relative_path": topology.name,
                "source_topology_receipt_file_sha256": (
                    f"sha256:{hashlib.sha256(topology_bytes).hexdigest()}"
                ),
                "source_topology_receipt_digest": topology_receipt["receipt_digest"],
                "source_instance_id": "79",
                "source_component_geometry_digest": f"sha256:{'2' * 64}",
                "source_semantic_label": "towel",
                "units": "m",
                "dimensions_m": list(observed_dimensions),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "root": root,
        "usd": usd,
        "archive": archive,
        "observation": observation,
        "topology": topology,
    }


def _inspect(paths: dict[str, Path]) -> dict:
    return inspect_external_simready_deformable_asset(
        archive_path=paths["archive"],
        expanded_root=paths["root"],
        usd_path=paths["usd"],
        observation_path=paths["observation"],
    )


def _rebuild_archive(paths: dict[str, Path]) -> None:
    with zipfile.ZipFile(paths["archive"], "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(paths["root"].rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(paths["root"]).as_posix())


def test_public_boundary_accepts_only_paths_and_no_qualification_override() -> None:
    signature = inspect.signature(inspect_external_simready_deformable_asset)
    assert tuple(signature.parameters) == (
        "archive_path",
        "expanded_root",
        "usd_path",
        "observation_path",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    replay_signature = inspect.signature(verify_external_simready_deformable_asset_inspection)
    assert tuple(replay_signature.parameters) == (
        "archive_path",
        "expanded_root",
        "usd_path",
        "observation_path",
        "expected_receipt_digest",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in replay_signature.parameters.values()
    )


def test_external_candidate_retains_source_and_requires_clean_physx_conversion(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    receipt = _inspect(paths)

    assert receipt["status"] == PENDING_STATUS
    assert receipt["source_package"]["archive_expanded_identity_verified"] is True
    assert receipt["source_package"]["source_bytes_mutated_by_inspector"] is False
    assert receipt["source_package"]["inspection_bound_to_snapshot_not_live_path"] is True
    assert receipt["source_package"]["source_retention_rights_evaluated"] is False
    assert (
        receipt["source_package"]["source_may_be_retained_as_immutable_conversion_input"] is False
    )
    assert receipt["source_package"]["expanded_file_count"] == 6

    stage = receipt["openusd_inspection"]
    assert len(stage["openusd_version"]) == 3
    assert stage["root_layer_sha256"] == next(
        row["sha256"]
        for row in receipt["source_package"]["expanded_files"]
        if row["relative_path"] == "rolled_towel.usda"
    )
    assert stage["default_prim_path"] == "/root"
    assert stage["up_axis"] == "Z"
    assert stage["meters_per_unit"] == 1.0
    assert "Blender v4.2.3 LTS" in stage["documentation"]
    assert stage["stage_metadata"]["defaultPrim"] == "root"
    assert len(stage["dependencies"]) == 5
    assert {row["prim_path"] for row in stage["materials"]} == {
        "/root/Looks/Material",
        "/root/Towel/Physics/PhysicsMaterial",
    }
    base_shader = next(row for row in stage["shaders"] if row["prim_path"].endswith("/Base"))
    base_file = next(
        row for row in base_shader["authored_properties"] if row["name"] == "inputs:file"
    )
    assert base_file["value"]["authored_path"] == "textures/base.png"
    assert stage["source_metadata"]["generator_identity_inferred"] is False
    assert {row["name"] for row in stage["source_metadata"]["provider_authored_attributes"]} == {
        "lightwheelusd:assetFormat",
        "lightwheelusd:hierarchyContract",
    }
    assert (
        stage["precomposition_dependency_preflight"]["composition_confined_before_stage_open"]
        is True
    )
    binding = stage["deformable_entity_binding"]
    assert binding["selected_body_prim_path"] == "/root/Towel"
    assert binding["selected_surface_mesh_prim_path"] == "/root/Towel/Visuals/TowelMesh"
    assert binding["cooking_source_relationship_join_valid"] is True
    assert binding["visual_material_binding"]["material_prim_path"] == "/root/Looks/Material"
    assert (
        binding["physics_material_binding"]["material_prim_path"]
        == "/root/Towel/Physics/PhysicsMaterial"
    )

    surface = stage["surface_mesh"]
    assert surface["selection_method"] == "authored_body_scoped_cooking_source_relationship"
    assert surface["closed_oriented_manifold"] is True
    assert surface["vertex_count"] == 8
    assert surface["face_count"] == 12
    assert surface["edge_count"] == 18
    assert surface["boundary_edge_count"] == 0
    assert surface["nonmanifold_edge_count"] == 0
    assert surface["world_bounds"]["dimensions"] == pytest.approx((0.2, 0.1, 0.08))
    assert surface["absolute_volume_m3"] == pytest.approx(0.0016)

    alignment = stage["dimension_alignment"]
    assert alignment["within_one_percent"] is False
    assert alignment["required_bake_scale_xyz"] == pytest.approx((1.85, 1.2, 1.375))

    raw_schemas = {
        schema for row in stage["raw_schema_bindings"] for schema in row["raw_authored_schemas"]
    }
    assert "OmniPhysicsDeformableBodyAPI" in raw_schemas
    assert "PhysxDeformableBodyAPI" not in raw_schemas
    attrs = {(row["prim_path"], row["name"]): row["value"] for row in stage["physics_attributes"]}
    assert attrs[("/root/Towel/Physics/PhysicsMaterial", "omniphysics:youngsModulus")] == 180000.0
    assert attrs[
        ("/root/Towel/Physics/PhysicsMaterial", "omniphysics:poissonsRatio")
    ] == pytest.approx(0.42)
    assert attrs[("/root/Towel", "physxDeformableBody:selfCollision")] is True

    assert all(row["point_count"] == 0 for row in stage["tetmeshes"])
    assert all(row["tetrahedron_count"] == 0 for row in stage["tetmeshes"])
    assert stage["compatibility"]["static_rigid_contract_authored"] is True
    assert stage["dome_lights_inside_default_prim"][0]["prim_path"] == "/root/DomeLight"
    assert {
        "external_simready_deformable_pinned_physx_body_schema_missing",
        "external_simready_deformable_pinned_physx_material_schema_missing",
        "external_simready_deformable_cooked_tetmesh_topology_missing",
        "external_simready_deformable_default_prim_dome_light_requires_exclusion",
        "external_simready_deformable_static_rigid_contract_conflict",
        "external_simready_deformable_frozen_dimensions_require_baked_scale",
        "external_simready_deformable_native_qualification_missing",
        "external_simready_deformable_rights_and_provider_output_terms_unresolved",
    }.issubset(receipt["blockers"])

    conversion = receipt["pinned_physx_conversion"]
    assert conversion["derived_runtime_usd_must_be_separate"] is True
    assert conversion["source_usd_must_remain_immutable"] is True
    assert conversion["loader_literal_schema_predicate_line"] == 544
    assert conversion["authoring_api"].endswith(":define_deformable_body_properties")
    assert conversion["cooking_api"].endswith(":add_physx_deformable_body")

    claims = receipt["claim_ceiling"]
    assert claims["maximum_claim"] == CLAIM_CEILING
    assert claims["standard_physx_runtime_asset"] is False
    assert claims["simready_asset_admitted"] is False
    assert claims["native_simulator_qualified"] is False
    assert claims["physically_equivalent_real_material"] is False
    assert claims["rights_and_provider_output_terms"] == "not_evaluated"
    assert receipt["receipt_integrity"] == {
        "canonical_self_digest_only": True,
        "authenticity_or_origin_proven_by_self_digest": False,
        "downstream_replay_with_independently_frozen_expected_digest_required": True,
    }
    assert receipt["receipt_digest"] == canonical_digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )


def test_literal_standard_schemas_and_nonempty_tets_are_observed_without_qualification(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(
        tmp_path,
        observed_dimensions=(0.2, 0.1, 0.08),
        standard_schemas=True,
        nonempty_tetmesh=True,
        include_default_dome_light=False,
        static_rigid_contract=False,
    )
    receipt = _inspect(paths)
    compatibility = receipt["openusd_inspection"]["compatibility"]
    assert compatibility == {
        "literal_physx_deformable_body_api_authored": True,
        "literal_physx_deformable_material_api_authored": True,
        "nonempty_tetmesh_authored": True,
        "palatial_shell_intent_authored": False,
        "static_rigid_contract_authored": False,
        "evidence_scoped_to_selected_body_mesh_and_material": True,
    }
    assert receipt["status"] == PENDING_STATUS
    assert receipt["blockers"] == [
        "external_simready_deformable_native_qualification_missing",
        "external_simready_deformable_rights_and_provider_output_terms_unresolved",
    ]
    assert receipt["claim_ceiling"]["native_simulator_qualified"] is False


def test_schema_named_attribute_cannot_impersonate_applied_api(tmp_path: Path) -> None:
    paths = _write_fixture(
        tmp_path,
        schema_decoy_attribute=True,
        include_default_dome_light=False,
        static_rigid_contract=False,
    )
    receipt = _inspect(paths)
    assert (
        receipt["openusd_inspection"]["compatibility"]["literal_physx_deformable_body_api_authored"]
        is False
    )
    assert "external_simready_deformable_pinned_physx_body_schema_missing" in receipt["blockers"]


def test_empty_surface_topology_is_retained_as_rejected_receipt(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path, empty_surface=True)
    receipt = _inspect(paths)
    assert receipt["status"] == REJECTED_STATUS
    assert receipt["openusd_inspection"]["surface_mesh"]["closed_oriented_manifold"] is False
    assert "external_simready_deformable_closed_surface_topology_invalid" in receipt["blockers"]
    assert receipt["claim_ceiling"]["geometry_and_material_candidate_inspected"] is False


def test_archive_traversal_is_rejected_before_identity_comparison(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    with zipfile.ZipFile(paths["archive"], "w") as archive:
        archive.writestr("../escape.usda", b"#usda 1.0\n")
    with pytest.raises(ExternalSimreadyDeformableAssetError) as caught:
        _inspect(paths)
    assert caught.value.errors == ("external_simready_deformable_archive_member_path_invalid",)


def test_expanded_tamper_is_rejected_against_archive_bytes(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    (paths["root"] / "textures" / "normal.png").write_bytes(_PNG_1X1 + b"tampered")
    with pytest.raises(ExternalSimreadyDeformableAssetError) as caught:
        _inspect(paths)
    assert caught.value.errors == (
        "external_simready_deformable_archive_expanded_identity_mismatch",
    )


def test_expanded_symlink_is_rejected_without_following(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    target = tmp_path / "outside.png"
    target.write_bytes(_PNG_1X1)
    texture = paths["root"] / "textures" / "base.png"
    texture.unlink()
    texture.symlink_to(target)
    with pytest.raises(ExternalSimreadyDeformableAssetError) as caught:
        _inspect(paths)
    assert caught.value.errors == ("external_simready_deformable_expanded_symlink_forbidden",)


def test_nofollow_unavailable_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_fixture(tmp_path)
    monkeypatch.delattr(asset_module.os, "O_NOFOLLOW")
    with pytest.raises(ExternalSimreadyDeformableAssetError) as caught:
        _inspect(paths)
    assert caught.value.errors == ("external_simready_deformable_nofollow_unavailable",)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS system alias contract")
def test_trusted_macos_var_alias_is_canonicalized_before_nofollow_walk(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    if not str(tmp_path).startswith("/private/var/"):
        pytest.skip("pytest temp root is not under the macOS /var alias")

    def alias(path: Path) -> str:
        return str(path).replace("/private/var/", "/var/", 1)

    receipt = inspect_external_simready_deformable_asset(
        archive_path=alias(paths["archive"]),
        expanded_root=alias(paths["root"]),
        usd_path=alias(paths["usd"]),
        observation_path=alias(paths["observation"]),
    )
    assert receipt["input_paths"]["archive"].startswith("/private/var/")
    assert receipt["input_paths"]["source_topology_receipt"].startswith("/private/var/")


def test_openusd_parses_snapshot_even_if_source_changes_after_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_fixture(tmp_path)
    original = asset_module._inspect_stage_snapshot

    def mutate_source_after_snapshot(*args, **kwargs):
        paths["usd"].write_text("not usd anymore", encoding="utf-8")
        return original(*args, **kwargs)

    monkeypatch.setattr(asset_module, "_inspect_stage_snapshot", mutate_source_after_snapshot)
    receipt = _inspect(paths)
    assert "Blender v4.2.3 LTS" in receipt["openusd_inspection"]["documentation"]
    assert paths["usd"].read_text(encoding="utf-8") == "not usd anymore"
    assert receipt["source_package"]["source_bytes_mutated_by_inspector"] is False
    assert receipt["source_package"]["inspection_bound_to_snapshot_not_live_path"] is True


def test_asset_path_without_authored_path_uses_path_field() -> None:
    value = SimpleNamespace(path="textures/base.png", resolvedPath="/tmp/root/textures/base.png")

    assert asset_module._asset_path_parts(value) == (
        "textures/base.png",
        "/tmp/root/textures/base.png",
    )
    assert asset_module._json_value(value) == {
        "authored_path": "textures/base.png",
        "resolved_path_recorded": False,
    }


def test_file_identity_drift_during_nofollow_read_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_fixture(tmp_path)
    real_fstat = os.fstat
    call_counts: dict[int, int] = {}

    def drifting_fstat(descriptor: int):
        result = real_fstat(descriptor)
        call_counts[descriptor] = call_counts.get(descriptor, 0) + 1
        if call_counts[descriptor] < 2:
            return result
        return SimpleNamespace(
            st_dev=result.st_dev,
            st_ino=result.st_ino,
            st_mode=result.st_mode,
            st_size=result.st_size,
            st_mtime_ns=result.st_mtime_ns + 1,
            st_ctime_ns=result.st_ctime_ns,
        )

    monkeypatch.setattr(asset_module.os, "fstat", drifting_fstat)
    with pytest.raises(ExternalSimreadyDeformableAssetError) as caught:
        _inspect(paths)
    assert caught.value.errors == ("external_simready_deformable_archive_changed_while_reading",)


def test_usd_asset_dependency_outside_snapshot_root_fails_closed(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(_PNG_1X1)
    usd_text = paths["usd"].read_text(encoding="utf-8")
    paths["usd"].write_text(
        usd_text.replace("@textures/base.png@", f"@{outside}@"), encoding="utf-8"
    )
    _rebuild_archive(paths)
    with pytest.raises(ExternalSimreadyDeformableAssetError) as caught:
        _inspect(paths)
    assert caught.value.errors == ("external_simready_deformable_usd_dependency_outside_root",)


def test_observation_duplicate_keys_and_non_finite_dimensions_fail_closed(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    paths["observation"].write_text(
        f'{{"schema_version":"{OBSERVATION_SCHEMA_VERSION}",'
        f'"schema_version":"{OBSERVATION_SCHEMA_VERSION}",'
        '"entity_id":"x","source_topology_receipt_relative_path":"source.json",'
        f'"source_topology_receipt_file_sha256":"sha256:{"1" * 64}",'
        f'"source_topology_receipt_digest":"sha256:{"1" * 64}",'
        '"source_instance_id":"79",'
        f'"source_component_geometry_digest":"sha256:{"1" * 64}",'
        '"source_semantic_label":"towel",'
        '"units":"m","dimensions_m":[NaN,1,1]}',
        encoding="utf-8",
    )
    with pytest.raises(ExternalSimreadyDeformableAssetError) as caught:
        _inspect(paths)
    assert caught.value.errors == ("external_simready_deformable_observation_invalid",)


def test_usd_entrypoint_must_be_inside_snapshotted_root(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    outside = tmp_path / "outside.usda"
    outside.write_text("#usda 1.0\n", encoding="utf-8")
    paths["usd"] = outside
    with pytest.raises(ExternalSimreadyDeformableAssetError) as caught:
        _inspect(paths)
    assert caught.value.errors == ("external_simready_deformable_usd_outside_root",)


def test_unrelated_schema_material_and_tet_decoys_cannot_clear_body_scoped_blockers(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(
        tmp_path,
        observed_dimensions=(0.2, 0.1, 0.08),
        include_default_dome_light=False,
        static_rigid_contract=False,
        unrelated_standard_decoy=True,
        body_scoped_tet_decoy=True,
    )
    receipt = _inspect(paths)
    compatibility = receipt["openusd_inspection"]["compatibility"]
    assert compatibility["literal_physx_deformable_body_api_authored"] is False
    assert compatibility["literal_physx_deformable_material_api_authored"] is False
    assert compatibility["nonempty_tetmesh_authored"] is False
    assert {
        "external_simready_deformable_pinned_physx_body_schema_missing",
        "external_simready_deformable_pinned_physx_material_schema_missing",
        "external_simready_deformable_cooked_tetmesh_topology_missing",
    }.issubset(receipt["blockers"])
    assert receipt["openusd_inspection"]["deformable_entity_binding"]["body_scoped_tetmeshes"] == [
        row
        for row in receipt["openusd_inspection"]["tetmeshes"]
        if row["prim_path"].startswith("/root/Towel/")
    ]


def test_missing_selected_visual_material_binding_rejects_geometry_material_claim(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path, include_visual_material_binding=False)
    receipt = _inspect(paths)
    assert receipt["status"] == REJECTED_STATUS
    assert receipt["claim_ceiling"]["geometry_and_material_candidate_inspected"] is False
    assert (
        "external_simready_deformable_visual_material_binding_missing_or_ambiguous"
        in receipt["blockers"]
    )


def test_external_composition_arc_is_rejected_before_composed_stage_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_fixture(tmp_path)
    outside = tmp_path / "outside.usda"
    outside.write_text('#usda 1.0\ndef Xform "outside" {}\n', encoding="utf-8")
    text = paths["usd"].read_text(encoding="utf-8")
    paths["usd"].write_text(
        text.replace(
            'def Xform "root"\n{',
            f'def Xform "root" (\n    prepend references = @{outside}@\n)\n{{',
        ),
        encoding="utf-8",
    )
    _rebuild_archive(paths)
    composed_open_called = False

    def reject_composed_open(*_args, **_kwargs):
        nonlocal composed_open_called
        composed_open_called = True
        raise AssertionError("composed stage must not open")

    monkeypatch.setattr(asset_module, "_open_composed_stage", reject_composed_open)
    with pytest.raises(ExternalSimreadyDeformableAssetError) as caught:
        _inspect(paths)
    assert caught.value.errors == ("external_simready_deformable_usd_dependency_outside_root",)
    assert composed_open_called is False


def test_replay_verifier_rejects_asset_substitution_against_frozen_digest(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    original = _inspect(paths)
    with pytest.raises(ExternalSimreadyDeformableAssetError) as invalid_digest:
        verify_external_simready_deformable_asset_inspection(
            archive_path=paths["archive"],
            expanded_root=paths["root"],
            usd_path=paths["usd"],
            observation_path=paths["observation"],
            expected_receipt_digest="caller-supplied-claim",
        )
    assert invalid_digest.value.errors == (
        "external_simready_deformable_expected_receipt_digest_invalid",
    )
    replay = verify_external_simready_deformable_asset_inspection(
        archive_path=paths["archive"],
        expanded_root=paths["root"],
        usd_path=paths["usd"],
        observation_path=paths["observation"],
        expected_receipt_digest=original["receipt_digest"],
    )
    assert replay == original

    (paths["root"] / "textures" / "base.png").write_bytes(_PNG_1X1 + b"replacement")
    _rebuild_archive(paths)
    with pytest.raises(ExternalSimreadyDeformableAssetError) as caught:
        verify_external_simready_deformable_asset_inspection(
            archive_path=paths["archive"],
            expanded_root=paths["root"],
            usd_path=paths["usd"],
            observation_path=paths["observation"],
            expected_receipt_digest=original["receipt_digest"],
        )
    assert caught.value.errors == ("external_simready_deformable_receipt_replay_mismatch",)


def test_observation_cannot_substitute_source_geometry_or_dimensions(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    observation = json.loads(paths["observation"].read_text(encoding="utf-8"))
    observation["source_component_geometry_digest"] = f"sha256:{'3' * 64}"
    observation["dimensions_m"] = [0.36, 0.12, 0.11]
    paths["observation"].write_text(json.dumps(observation), encoding="utf-8")
    with pytest.raises(ExternalSimreadyDeformableAssetError) as caught:
        _inspect(paths)
    assert {
        "external_simready_deformable_source_topology_geometry_digest_mismatch",
        "external_simready_deformable_source_topology_dimensions_mismatch",
    }.issubset(caught.value.errors)


def test_source_topology_symlink_is_rejected_without_following(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    outside = tmp_path / "outside_topology.json"
    outside.write_bytes(paths["topology"].read_bytes())
    paths["topology"].unlink()
    paths["topology"].symlink_to(outside)
    with pytest.raises(ExternalSimreadyDeformableAssetError) as caught:
        _inspect(paths)
    assert caught.value.errors == (
        "external_simready_deformable_source_topology_receipt_file_invalid",
    )


def test_coherent_source_observation_rewrite_still_fails_frozen_receipt_replay(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    frozen = _inspect(paths)
    topology = json.loads(paths["topology"].read_text(encoding="utf-8"))
    rewritten_geometry_digest = f"sha256:{'4' * 64}"
    topology["targets"][0]["best_component"]["geometry_digest"] = rewritten_geometry_digest
    topology["receipt_digest"] = ""
    topology["receipt_digest"] = canonical_digest(topology, digest_field="receipt_digest")
    topology_bytes = (
        json.dumps(topology, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    paths["topology"].write_bytes(topology_bytes)
    observation = json.loads(paths["observation"].read_text(encoding="utf-8"))
    observation["source_topology_receipt_file_sha256"] = (
        f"sha256:{hashlib.sha256(topology_bytes).hexdigest()}"
    )
    observation["source_topology_receipt_digest"] = topology["receipt_digest"]
    observation["source_component_geometry_digest"] = rewritten_geometry_digest
    paths["observation"].write_text(json.dumps(observation), encoding="utf-8")
    assert _inspect(paths)["receipt_digest"] != frozen["receipt_digest"]
    with pytest.raises(ExternalSimreadyDeformableAssetError) as caught:
        verify_external_simready_deformable_asset_inspection(
            archive_path=paths["archive"],
            expanded_root=paths["root"],
            usd_path=paths["usd"],
            observation_path=paths["observation"],
            expected_receipt_digest=frozen["receipt_digest"],
        )
    assert caught.value.errors == ("external_simready_deformable_receipt_replay_mismatch",)


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "expected_error"),
    [
        (
            "_MAX_ARCHIVE_MEMBERS",
            5,
            "external_simready_deformable_archive_member_count_exceeded",
        ),
        (
            "_MAX_EXPANDED_FILES",
            5,
            "external_simready_deformable_expanded_file_count_exceeded",
        ),
        (
            "_MAX_EXPANDED_ENTRIES",
            5,
            "external_simready_deformable_expanded_entry_count_exceeded",
        ),
        (
            "_MAX_MESH_POINTS",
            4,
            "external_simready_deformable_surface_point_count_exceeded",
        ),
    ],
)
def test_resource_limits_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: int,
    expected_error: str,
) -> None:
    paths = _write_fixture(tmp_path)
    monkeypatch.setattr(asset_module, limit_name, limit_value)
    with pytest.raises(ExternalSimreadyDeformableAssetError) as caught:
        _inspect(paths)
    assert caught.value.errors == (expected_error,)


def test_stage_units_are_applied_before_metric_dimension_join(tmp_path: Path) -> None:
    paths = _write_fixture(
        tmp_path,
        observed_dimensions=(0.002, 0.001, 0.0008),
        meters_per_unit=0.01,
    )
    receipt = _inspect(paths)
    stage = receipt["openusd_inspection"]
    assert stage["surface_mesh"]["world_bounds"]["dimensions"] == pytest.approx(
        (0.002, 0.001, 0.0008)
    )
    assert stage["dimension_alignment"]["within_one_percent"] is True
