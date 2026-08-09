from __future__ import annotations

from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from blueprint_pipeline.simready_render_appearance import (
    SimReadyRenderAppearanceError,
    author_minimal_render_appearance,
    derive_masked_observed_color,
    inspect_simready_render_appearance,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _appearance_spec() -> dict:
    observation = {
        "schema_version": "simready_masked_color_observation.v1",
        "status": "masked_observed_color_measured",
        "image_sha256": "sha256:" + "1" * 64,
        "mask_sha256": "sha256:" + "2" * 64,
        "mask_threshold_8bit": 127,
        "masked_pixel_count": 100,
        "estimator": "per_channel_median_srgb_v1",
        "median_srgb": [0.72, 0.61, 0.60],
        "claim_boundary": {
            "masked_color_is_not_texture_equivalence": True,
            "appearance_is_lighting_dependent": True,
        },
        "receipt_digest": "",
    }
    observation["receipt_digest"] = canonical_digest(
        observation, digest_field="receipt_digest"
    )
    return {
        "observed_exterior_reference_sha256": "sha256:" + "1" * 64,
        "observed_exterior_base_color_rgb": [0.72, 0.61, 0.60],
        "observed_exterior_color_observation": observation,
        "observed_exterior_roughness": 0.32,
        "observed_exterior_metallic": 0.15,
        "generated_surface_base_color_rgb": [0.88, 0.88, 0.86],
        "generated_surface_roughness": 0.55,
        "generated_surface_metallic": 0.0,
        "generated_surface_claim": "generated_candidate_unobserved",
    }


def _physics_only_asset(path: Path, *, articulated: bool) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    root_path = "/Asset" if articulated else "/canned_beverage"
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    exterior_parent = (
        UsdGeom.Xform.Define(stage, f"{root_path}/upper_door").GetPrim()
        if articulated
        else root.GetPrim()
    )
    UsdGeom.Cube.Define(stage, f"{exterior_parent.GetPath()}/exterior")
    if articulated:
        interior = UsdGeom.Cube.Define(
            stage, f"{root_path}/generated_interior"
        ).GetPrim()
        interior.CreateAttribute(
            "blueprint:articulatedReplacement:provenance",
            Sdf.ValueTypeNames.String,
        ).Set("generated_candidate_geometry")
    physics = UsdShade.Material.Define(stage, f"{root_path}/materials/contact")
    UsdPhysics.MaterialAPI.Apply(physics.GetPrim()).CreateStaticFrictionAttr(0.5)
    UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(
        physics, materialPurpose="physics"
    )
    assert stage.GetRootLayer().Save()
    return path


@pytest.mark.parametrize("articulated", [False, True])
def test_physics_material_never_counts_as_policy_render_appearance(
    tmp_path: Path, articulated: bool
) -> None:
    source = _physics_only_asset(
        tmp_path / ("articulated.usda" if articulated else "rigid.usda"),
        articulated=articulated,
    )
    with pytest.raises(
        SimReadyRenderAppearanceError, match="simready_render_material_unbound"
    ):
        inspect_simready_render_appearance(source)


@pytest.mark.parametrize("articulated", [False, True])
def test_minimal_material_authoring_covers_rigid_and_articulated_fixtures(
    tmp_path: Path, articulated: bool
) -> None:
    source = _physics_only_asset(
        tmp_path / ("articulated.usda" if articulated else "rigid.usda"),
        articulated=articulated,
    )
    output = tmp_path / ("articulated_materialized.usda" if articulated else "rigid_materialized.usda")
    receipt = author_minimal_render_appearance(
        source_asset_path=source,
        output_asset_path=output,
        appearance_spec=_appearance_spec(),
    )
    assert receipt["status"] == "minimal_render_appearance_authored"
    assert receipt["static_validation"]["default_neutral_renderer_fallback_required"] is False
    assert receipt["native_material_render_readback_required"] is True
    rows = receipt["static_validation"]["material_rows"]
    assert all(row["surface_shader_ids"] == ["UsdPreviewSurface"] for row in rows)
    generated = receipt["binding"]["generated_surface_prim_paths"]
    assert bool(generated) is articulated
    stage = Usd.Stage.Open(str(output))
    root = stage.GetDefaultPrim()
    physics_material, _relationship = UsdShade.MaterialBindingAPI(
        root
    ).ComputeBoundMaterial(materialPurpose="physics")
    assert physics_material.GetPrim().HasAPI(UsdPhysics.MaterialAPI)
    if articulated:
        generated_row = next(row for row in rows if row["prim_path"] in generated)
        assert generated_row["material_path"].endswith("/generated_unobserved")


def test_minimal_material_authoring_preserves_source_bytes(tmp_path: Path) -> None:
    source = _physics_only_asset(tmp_path / "source.usda", articulated=True)
    source_bytes = source.read_bytes()
    with pytest.raises(
        SimReadyRenderAppearanceError,
        match="simready_render_appearance_in_place_mutation_forbidden",
    ):
        author_minimal_render_appearance(
            source_asset_path=source,
            output_asset_path=source,
            appearance_spec=_appearance_spec(),
        )
    assert source.read_bytes() == source_bytes


def test_observed_color_is_measured_from_exact_masked_pixels(tmp_path: Path) -> None:
    from PIL import Image

    image = tmp_path / "observed.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(image)
    Image.fromarray(
        __import__("numpy").array([[255, 255], [0, 0]], dtype="uint8")
    ).save(mask)
    receipt = derive_masked_observed_color(image_path=image, mask_path=mask)
    assert receipt["median_srgb"] == [1.0, 0.0, 0.0]
    assert receipt["masked_pixel_count"] == 2
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )
