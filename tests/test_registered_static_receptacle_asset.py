from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
from PIL import Image
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import blueprint_pipeline.registered_static_receptacle_asset as receptacle_asset_module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.registered_static_receptacle_asset import (
    CANDIDATE_FILENAME,
    RECEIPT_FILENAME,
    VISUAL_BASIS_FILENAME,
    RegisteredStaticReceptacleAssetError,
    build_registered_static_receptacle_asset,
)
from blueprint_pipeline.sage_collision_component_topology import (
    inspect_sage_collision_component_topology,
)
from blueprint_pipeline.semantic_review_attestation import (
    TRUSTED_PUBLIC_KEY_SHA256_ENV,
    canonical_semantic_authority_selection_bytes,
    canonical_semantic_review_attestation_bytes,
    materialize_semantic_authority_selection,
    materialize_semantic_review_attestation,
    materialize_semantic_review_payload,
    semantic_frame_evidence_digest,
    semantic_review_signature_message,
)


SEMANTIC_AUTHORITY_KEY = Ed25519PrivateKey.from_private_bytes(b"\x27" * 32)


def _semantic_public_key_bytes() -> bytes:
    return SEMANTIC_AUTHORITY_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


@pytest.fixture(autouse=True)
def _configured_semantic_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        TRUSTED_PUBLIC_KEY_SHA256_ENV,
        "sha256:" + hashlib.sha256(_semantic_public_key_bytes()).hexdigest(),
    )


def _fixture(
    tmp_path: Path,
    *,
    missing_walls: bool = False,
    slatted_cap: bool = False,
    internal_planes: bool = False,
    sparse_walls: bool = False,
    lattice_aligned_posts: bool = False,
    interior_vertical_panel: bool = False,
    floor_wall_gap: bool = False,
) -> tuple[Path, Path, dict]:
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            [
                {
                    "ins_id": "bin",
                    "label": "storage_basket",
                    "bounding_box": [
                        {"x": x, "y": y, "z": z}
                        for z in (0.0, 1.0)
                        for x, y in (
                            (0.0, 0.0),
                            (0.0, 1.0),
                            (1.0, 1.0),
                            (1.0, 0.0),
                        )
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    collision = tmp_path / "collision.usda"
    stage = Usd.Stage.CreateNew(str(collision))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    mesh = UsdGeom.Mesh.Define(stage, "/Root/OpenBin")
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    def add_box(minimum: tuple[float, float, float], maximum: tuple[float, float, float]) -> None:
        start = len(points)
        x0, y0, z0 = minimum
        x1, y1, z1 = maximum
        points.extend(
            [
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y1, z0),
                (x0, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            ]
        )
        faces.extend(
            tuple(start + value for value in row)
            for row in (
                (0, 3, 2, 1),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (3, 0, 4, 7),
            )
        )

    def add_vertical_plane(
        *,
        axis: str,
        coordinate: float,
        transverse_min: float,
        transverse_max: float,
    ) -> None:
        start = len(points)
        if axis == "x":
            points.extend(
                [
                    (coordinate, transverse_min, 0.1),
                    (coordinate, transverse_max, 0.1),
                    (coordinate, transverse_max, 1.0),
                    (coordinate, transverse_min, 1.0),
                ]
            )
        else:
            points.extend(
                [
                    (transverse_min, coordinate, 0.1),
                    (transverse_max, coordinate, 0.1),
                    (transverse_max, coordinate, 1.0),
                    (transverse_min, coordinate, 1.0),
                ]
            )
        faces.append((start, start + 1, start + 2, start + 3))
        faces.append((0, start, start + 1))

    add_box((0.0, 0.0, 0.0), (1.0, 1.0, 0.1))
    if internal_planes:
        add_vertical_plane(axis="x", coordinate=0.4, transverse_min=0.0, transverse_max=1.0)
        add_vertical_plane(axis="x", coordinate=0.6, transverse_min=0.0, transverse_max=1.0)
        add_vertical_plane(axis="y", coordinate=0.4, transverse_min=0.0, transverse_max=1.0)
        add_vertical_plane(axis="y", coordinate=0.6, transverse_min=0.0, transverse_max=1.0)
    elif sparse_walls:
        add_vertical_plane(axis="x", coordinate=0.0, transverse_min=0.49, transverse_max=0.51)
        add_vertical_plane(axis="x", coordinate=1.0, transverse_min=0.49, transverse_max=0.51)
        add_vertical_plane(axis="y", coordinate=0.0, transverse_min=0.49, transverse_max=0.51)
        add_vertical_plane(axis="y", coordinate=1.0, transverse_min=0.49, transverse_max=0.51)
    elif lattice_aligned_posts:
        for center in (0.18, 0.34, 0.5, 0.66, 0.82):
            add_vertical_plane(
                axis="x",
                coordinate=0.0,
                transverse_min=center - 0.005,
                transverse_max=center + 0.005,
            )
            add_vertical_plane(
                axis="x",
                coordinate=1.0,
                transverse_min=center - 0.005,
                transverse_max=center + 0.005,
            )
            add_vertical_plane(
                axis="y",
                coordinate=0.0,
                transverse_min=center - 0.005,
                transverse_max=center + 0.005,
            )
            add_vertical_plane(
                axis="y",
                coordinate=1.0,
                transverse_min=center - 0.005,
                transverse_max=center + 0.005,
            )
    else:
        wall_bottom = 0.19 if floor_wall_gap else 0.1
        add_box((0.0, 0.0, wall_bottom), (0.1, 1.0, 1.0))
        add_box((0.1, 0.0, wall_bottom), (0.9, 0.1, 1.0))
        faces.extend([(0, 8, 9), (1, 16, 17)])
        if not missing_walls:
            right_start = len(points)
            add_box((0.9, 0.0, wall_bottom), (1.0, 1.0, 1.0))
            back_start = len(points)
            add_box((0.1, 0.9, wall_bottom), (0.9, 1.0, 1.0))
            faces.extend(
                [
                    (2, right_start, right_start + 1),
                    (3, back_start, back_start + 1),
                ]
            )
        if interior_vertical_panel:
            add_vertical_plane(
                axis="x",
                coordinate=0.45,
                transverse_min=0.42,
                transverse_max=0.46,
            )
    if slatted_cap:
        for index in range(8):
            start = len(points)
            x_min = 0.102 + index * 0.1
            add_box((x_min, 0.1, 0.9), (x_min + 0.096, 0.9, 1.0))
            faces.append((0, start, start + 1))
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([value for face in faces for value in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    topology = inspect_sage_collision_component_topology(
        labels_path=labels,
        target_instance_ids=["bin"],
        opening_probe_instance_ids=["bin"],
        sage_collision_usd_path=collision,
    )
    if not any(
        (
            missing_walls,
            slatted_cap,
            internal_planes,
            sparse_walls,
            lattice_aligned_posts,
            interior_vertical_panel,
            floor_wall_gap,
        )
    ):
        assert topology["targets"][0]["opening_probe"]["open_collision_cavity_passed"]
    return labels, collision, topology


def _rights() -> dict:
    return {
        "source_revision": "fixture-source-revision",
        "license_id": "CC-BY-NC-4.0",
        "license_reference": "https://example.invalid/license",
        "license_sha256": "sha256:" + "a" * 64,
        "attribution": "Fixture attribution",
        "derived_processing_authority_id": "fixture-derived-authority",
        "provider_terms_id": "fixture-provider-terms",
        "output_rights_id": "fixture-output-rights",
        "raw_source_private_upload_permitted": False,
        "derived_asset_private_upload_permitted": True,
        "raw_redistribution_permitted": False,
        "provider_retention_permitted": False,
        "provider_training_permitted": False,
    }


def _visual_review(
    topology: dict,
    *,
    render_manifest_digest: str,
    frame_sha256: str,
    frame_size_bytes: int,
    target_instance_id: str = "bin",
) -> dict:
    result = {
        "schema_version": "adp_deformable_scene_visual_review.v1",
        "scene_id": "fixture-scene",
        "reviewer_id": "fixture-reviewer",
        "reviewed_at": "2026-08-10T00:00:00Z",
        "learned_policy_outcomes_inspected": False,
        "reconnaissance_only": True,
        "render_manifest_digest": render_manifest_digest,
        "collision_topology_receipt_digest": topology["receipt_digest"],
        "targets": [
            {
                "target_id": "basket_fixture",
                "publisher_instance_id": target_instance_id,
                "target_kind": "destination_receptacle",
                "material_class": "not_applicable",
                "material_class_supported_by_observation": False,
                "rest_state": "not_applicable",
                "support_relation": "observed_container_contents",
                "rigid_exterior_observed": True,
                "open_rim_observed": True,
                "interior_occupied": True,
                "complete_interior_appearance_observed": False,
                "collision_component_identity_passed": True,
                "open_collision_cavity_passed": bool(
                    topology["targets"][0]["opening_probe"]["open_collision_cavity_passed"]
                ),
                "source_destination_admitted": False,
                "engineered_twin_design_basis_admitted": True,
                "selection_role": "engineered_twin_design_basis",
                "cited_frames": [
                    {
                        "camera_id": "fixture_basket_closeup",
                        "sha256": frame_sha256,
                        "size_bytes": frame_size_bytes,
                    }
                ],
                "review_notes": "Fixture engineered-twin basis.",
            }
        ],
        "selected_movable_instance_id": "fixture-cloth",
        "selected_destination_design_basis_instance_id": target_instance_id,
        "source_destination_is_occupied": True,
        "source_destination_complete_interior_appearance_observed": False,
        "composition_required": True,
        "claim_boundary": {
            "virtual_closeup_recovers_missing_source_observations": False,
            "collision_cavity_establishes_hidden_appearance": False,
            "engineered_twin_hidden_geometry_is_source_truth": False,
            "review_is_evaluation_policy_media": False,
            "physical_material_equivalence_proven": False,
        },
        "review_digest": "",
    }
    result["review_digest"] = canonical_digest(result, digest_field="review_digest")
    return result


def _visual_inputs(
    tmp_path: Path,
    topology: dict,
    *,
    target_instance_id: str = "bin",
    solid_color: bool = False,
    near_solid: bool = False,
) -> dict:
    root = tmp_path / "visual_basis"
    frames = root / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    frame = frames / "fixture_basket_closeup.png"
    image = Image.new("RGB", (4, 3))
    if solid_color:
        image.putdata([(90, 70, 40)] * 12)
    elif near_solid:
        image.putdata([(90, 70, 40)] * 11 + [(99, 79, 49)])
    else:
        image.putdata([(20 + index * 12, 40 + index * 7, 60 + index * 4) for index in range(12)])
    image.save(frame, format="PNG")
    frame_bytes = frame.read_bytes()
    frame_sha256 = "sha256:" + hashlib.sha256(frame_bytes).hexdigest()
    camera = {
        "id": "fixture_basket_closeup",
        "path": str(frame),
        "bytes": len(frame_bytes),
        "nonblank": True,
        "digest": frame_sha256,
    }
    calibration = {
        "id": camera["id"],
        "pose_convention": "world_position_target_up_look_at_z_up",
        "position_world_m": [1.0, 2.0, 3.0],
        "target_world_m": [0.0, 0.0, 0.0],
        "up_world": [0.0, 0.0, 1.0],
        "intrinsics": {
            "model": "pinhole_centered_square_pixels",
            "fx": 4.2,
            "fy": 4.2,
            "cx": 2.0,
            "cy": 1.5,
            "width": 4,
            "height": 3,
            "vertical_fov_deg": 45.0,
        },
    }
    manifest = {
        "schema_version": "splat_scene_render.v1",
        "source_digest": "sha256:" + "1" * 64,
        "renderer_identity": {
            "name": "fixture_renderer",
            "harness_sha256": "sha256:" + "2" * 64,
            "entry_sha256": "sha256:" + "3" * 64,
            "width": 4,
            "height": 3,
            "pixel_ratio": 1,
            "supersampling": 1,
            "alpha": False,
            "background_rgb_hex": "0x000000",
            "output_format": "lossless_png",
            "color_space": "srgb",
            "node_version": "v1.0.0",
            "dependency_versions": {"renderer": "1.0.0"},
        },
        "render": {"status": "completed", "returncode": 0},
        "appearance_fidelity": {
            "source_splat_count": 100,
            "retained_splat_count": 100,
            "appearance_fidelity_qualified": False,
            "evaluation_input_authorized": False,
        },
        "cameras": [camera],
        "camera_calibration": [calibration],
        "render_manifest_digest": "",
    }
    manifest["render_manifest_digest"] = canonical_digest(
        manifest, digest_field="render_manifest_digest"
    )
    review = _visual_review(
        topology,
        render_manifest_digest=manifest["render_manifest_digest"],
        frame_sha256=frame_sha256,
        frame_size_bytes=len(frame_bytes),
        target_instance_id=target_instance_id,
    )
    manifest_path = root / "manifest.json"
    review_path = root / "review.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    review_path.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with Image.open(frame) as decoded:
        decoded_rgb = decoded.convert("RGB").tobytes()
    cited_frames_digest = semantic_frame_evidence_digest(
        [
            {
                "target_id": "basket_fixture",
                "camera_id": camera["id"],
                "sha256": frame_sha256,
                "size_bytes": len(frame_bytes),
                "decoded_rgb_sha256": ("sha256:" + hashlib.sha256(decoded_rgb).hexdigest()),
            }
        ]
    )
    basket = review["targets"][0]
    payload = materialize_semantic_review_payload(
        attestation_id="fixture-receptacle-semantic-review-001",
        selection_id="fixture-receptacle-semantic-authority-freeze-001",
        authority_id="fixture-semantic-review-authority",
        authority_key_id="fixture-semantic-key-2026-08",
        scene_id=review["scene_id"],
        target_id=basket["target_id"],
        source_instance_id=target_instance_id,
        semantic_role=basket["target_kind"],
        visual_review_digest=review["review_digest"],
        render_manifest_digest=manifest["render_manifest_digest"],
        collision_topology_receipt_digest=topology["receipt_digest"],
        cited_frames_digest=cited_frames_digest,
        learned_policy_outcomes_inspected=False,
        semantic_assertions={
            "rigid_exterior_observed": True,
            "open_rim_observed": True,
            "interior_occupied": True,
            "complete_interior_appearance_observed": False,
            "source_destination_admitted": False,
            "engineered_twin_design_basis_admitted": True,
            "selection_role": "engineered_twin_design_basis",
        },
    )
    public_key = _semantic_public_key_bytes()
    attestation = materialize_semantic_review_attestation(
        payload=payload,
        public_key_base64=base64.b64encode(public_key).decode("ascii"),
        signature_base64=base64.b64encode(
            SEMANTIC_AUTHORITY_KEY.sign(semantic_review_signature_message(payload))
        ).decode("ascii"),
    )
    selection = materialize_semantic_authority_selection(attestation=attestation)
    attestation_path = root / "semantic-attestation.json"
    selection_path = root / "semantic-selection.json"
    attestation_path.write_bytes(canonical_semantic_review_attestation_bytes(attestation))
    selection_path.write_bytes(canonical_semantic_authority_selection_bytes(selection))
    return {
        "visual_review_receipt_path": review_path,
        "render_manifest_path": manifest_path,
        "frame_root": frames,
        "semantic_review_attestation_path": attestation_path,
        "semantic_authority_selection_path": selection_path,
    }


def _authoring() -> dict:
    return {
        "source_repository": "https://example.invalid/BlueprintCapturePipeline",
        "source_revision": "b" * 40,
        "source_tree": "c" * 40,
        "package_name": "blueprint-pipeline-fixture",
        "package_version": "1.0.0",
    }


def _physics() -> dict:
    return {
        "static_friction": 0.6,
        "dynamic_friction": 0.5,
        "restitution": 0.0,
        "contact_offset_m": 0.002,
        "rest_offset_m": 0.0,
        "diagnostic_display_color_rgb": [0.7, 0.55, 0.35],
    }


def test_checked_in_840873_topology_v2_binds_source_obstructions_truthfully() -> None:
    receipt_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "arm_decision_proof_v1"
        / "deformable_scene"
        / "840873_sage_component_topology.v2.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == ("interiorgs_sage_collision_component_topology.v2")
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    basket = next(row for row in receipt["targets"] if row["interiorgs_instance_id"] == "87")
    assert basket["best_component"]["aabb_iou"] >= 0.85
    assert basket["best_component"]["target_coverage_fraction"] >= 0.85
    assert basket["best_component"]["component_coverage_fraction"] >= 0.85
    opening = basket["opening_probe"]
    assert opening["center_connected_floor_rectangle_cell_count"] == 81
    assert opening["center_connected_floor_rectangle_fraction"] == pytest.approx(1.0)
    assert opening["side_wall_probe"]["all_four_sides_passed"] is True
    assert all(
        side["hit_fraction"] == pytest.approx(1.0)
        for side in opening["side_wall_probe"]["sides"].values()
    )
    assert opening["overhead_clearance_probe"] == {
        "clear_of_above_floor_projected_geometry": True,
        "intersections": [],
        "projected_intersection_area_m2": 0.0,
        "projected_intersection_triangle_count": 0,
    }
    projected = opening["side_projected_coverage"]
    assert projected["all_four_sides_passed"] is False
    assert projected["sides"]["x_max"]["passed"] is True
    assert projected["sides"]["x_min"]["maximum_uncovered_aperture_m"] == pytest.approx(0.002283894)
    assert projected["sides"]["y_max"]["maximum_uncovered_aperture_m"] == pytest.approx(0.002124516)
    assert projected["sides"]["y_min"]["maximum_uncovered_aperture_m"] == pytest.approx(0.000577983)
    assert opening["open_prism_obstruction_probe"]["clear_of_non_floor_geometry"] is False
    assert opening["open_prism_obstruction_probe"]["obstruction_triangle_count"] > 0
    assert opening["open_collision_cavity_passed"] is False


def _build(tmp_path: Path, output_name: str = "asset") -> dict:
    labels, collision, topology = _fixture(tmp_path)
    return build_registered_static_receptacle_asset(
        labels_path=labels,
        sage_collision_usd_path=collision,
        topology_receipt=topology,
        **_visual_inputs(tmp_path, topology),
        target_instance_id="bin",
        entity_id="destination",
        asset_id="asset:destination",
        reference_world_pose={
            "position_world_m": [1.0, 2.0, 0.5],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        rights=_rights(),
        authoring_identity=_authoring(),
        physics_configuration=_physics(),
        simulator_name="Isaac Sim",
        simulator_version="6.0.1",
        output_root=tmp_path / output_name,
    )


def test_builds_independent_open_static_receptacle_candidate(tmp_path: Path) -> None:
    result = _build(tmp_path)
    root = Path(result["output_root"])

    assert (root / CANDIDATE_FILENAME).is_file()
    assert (root / RECEIPT_FILENAME).is_file()
    assert (root / VISUAL_BASIS_FILENAME).is_file()
    candidate = result["candidate"]
    assert candidate["asset_class"] == "rigid_receptacle"
    assert candidate["claims"]["native_simulator_qualified"] is False
    assert candidate["receptacle_configuration"]["geometry"]["open_interior"]
    wall_clearances = candidate["receptacle_configuration"]["geometry"]["wall_clearances_m"]
    assert set(wall_clearances) == {"x_min", "x_max", "y_min", "y_max"}
    assert candidate["receptacle_configuration"]["geometry"]["wall_thickness_m"] == pytest.approx(
        min(wall_clearances.values())
    )
    derived = result["receipt"]["derived_geometry"]
    assert derived["source_opening_to_outer_boundary_clearances_m"] != wall_clearances
    assert derived["authored_twin_wall_thicknesses_m"] == wall_clearances
    assert (
        derived["authored_twin_dimension_rule"]["source_clearances_used_as_authored_thickness"]
        is False
    )
    assert derived["authored_twin_floor_thickness_m"] == pytest.approx(
        candidate["receptacle_configuration"]["geometry"]["floor_thickness_m"]
    )
    assert result["receipt"]["geometry_conversion"]["runtime_usd_readback_within_tolerance"] is True
    assert result["receipt"]["candidate_file"]["path"] == CANDIDATE_FILENAME
    assert result["receipt"]["candidate_file"]["sha256"].startswith("sha256:")
    assert result["receipt"]["visual_design_basis_file"]["path"] == (VISUAL_BASIS_FILENAME)
    assert result["receipt"]["semantic_review_attestation_digest"].startswith("sha256:")
    assert result["receipt"]["semantic_authority_selection_digest"].startswith("sha256:")
    assert result["receipt"]["semantic_authority"]["authority_id"] == (
        "fixture-semantic-review-authority"
    )
    retained_visual_basis = json.loads((root / VISUAL_BASIS_FILENAME).read_text())
    assert (
        retained_visual_basis["semantic_authority"]["signed_attestation"]["attestation_digest"]
        == result["receipt"]["semantic_review_attestation_digest"]
    )
    assert (
        retained_visual_basis["semantic_authority"]["frozen_selection_contract"]["selection_digest"]
        == result["receipt"]["semantic_authority_selection_digest"]
    )
    assert result["receipt"]["claim_boundary"] == {
        "source_collision_component_exactly_replayed": False,
        "source_collision_component_deterministically_quantized": True,
        "source_collision_component_used_as_runtime_geometry": False,
        "source_collision_cavity_clear": True,
        "source_collision_enclosure_qualified": True,
        "source_collision_obstruction_triangle_count": 0,
        "intermediate_and_obj_quantization_resolution_m": 1.0e-9,
        "runtime_usd_point_type": "point3f",
        "runtime_usd_native_readback_qualified": False,
        "source_bytes_copied_to_output": False,
        "engineered_twin_not_source_scene_truth": True,
        "source_wall_thickness_observed": False,
        "source_floor_thickness_observed": False,
        "authored_twin_wall_and_floor_thicknesses_bound": True,
        "authored_twin_dimensions_derived_from_registered_outer_bounds": True,
        "source_opening_clearances_used_as_material_thickness": False,
        "source_interior_appearance_observed": False,
        "visual_semantics_authority_signed": True,
        "signed_visual_semantics_are_native_qualification": False,
        "native_simulator_qualified": False,
        "physical_equivalence_proven": False,
    }

    stage = Usd.Stage.Open(str(root / "runtime_asset.usda"))
    assert stage is not None
    mesh = stage.GetPrimAtPath("/Asset/Geometry")
    assert mesh.HasAPI(UsdPhysics.CollisionAPI)
    geometry = UsdGeom.Mesh(mesh)
    assert len(geometry.GetPointsAttr().Get()) == 40
    assert len(geometry.GetFaceVertexCountsAttr().Get()) == 30


def test_materialization_is_deterministic_and_never_overwrites(tmp_path: Path) -> None:
    labels, collision, topology = _fixture(tmp_path)
    arguments = {
        "labels_path": labels,
        "sage_collision_usd_path": collision,
        "topology_receipt": topology,
        **_visual_inputs(tmp_path, topology),
        "target_instance_id": "bin",
        "entity_id": "destination",
        "asset_id": "asset:destination",
        "reference_world_pose": {
            "position_world_m": [1.0, 2.0, 0.5],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "rights": _rights(),
        "authoring_identity": _authoring(),
        "physics_configuration": _physics(),
        "simulator_name": "Isaac Sim",
        "simulator_version": "6.0.1",
    }
    first = build_registered_static_receptacle_asset(**arguments, output_root=tmp_path / "first")
    second = build_registered_static_receptacle_asset(**arguments, output_root=tmp_path / "second")
    assert first["candidate"]["candidate_digest"] == second["candidate"]["candidate_digest"]
    assert first["receipt"]["receipt_digest"] == second["receipt"]["receipt_digest"]
    with pytest.raises(RegisteredStaticReceptacleAssetError) as caught:
        build_registered_static_receptacle_asset(**arguments, output_root=tmp_path / "first")
    assert caught.value.errors == ("receptacle_asset_output_exists",)


def test_rehashing_tampered_topology_cannot_authorize_output(tmp_path: Path) -> None:
    labels, collision, topology = _fixture(tmp_path)
    tampered = copy.deepcopy(topology)
    tampered["targets"][0]["opening_probe"]["cavity_depth_m"] = 0.5
    tampered["receipt_digest"] = canonical_digest(tampered, digest_field="receipt_digest")

    with pytest.raises(RegisteredStaticReceptacleAssetError) as caught:
        build_registered_static_receptacle_asset(
            labels_path=labels,
            sage_collision_usd_path=collision,
            topology_receipt=tampered,
            **_visual_inputs(tmp_path, topology),
            target_instance_id="bin",
            entity_id="destination",
            asset_id="asset:destination",
            reference_world_pose={
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            rights=_rights(),
            authoring_identity=_authoring(),
            physics_configuration=_physics(),
            simulator_name="Isaac Sim",
            simulator_version="6.0.1",
            output_root=tmp_path / "rejected",
        )
    assert caught.value.errors == ("receptacle_asset_topology_replay_mismatch",)
    assert not (tmp_path / "rejected").exists()


def test_caller_weakened_topology_thresholds_cannot_authorize_output(
    tmp_path: Path,
) -> None:
    labels, collision, _ = _fixture(tmp_path)
    weak = inspect_sage_collision_component_topology(
        labels_path=labels,
        target_instance_ids=["bin"],
        opening_probe_instance_ids=["bin"],
        sage_collision_usd_path=collision,
        minimum_component_iou=0.0001,
        opening_grid_size=3,
        opening_margin_fraction=0.49,
    )

    with pytest.raises(RegisteredStaticReceptacleAssetError) as caught:
        build_registered_static_receptacle_asset(
            labels_path=labels,
            sage_collision_usd_path=collision,
            topology_receipt=weak,
            **_visual_inputs(tmp_path, weak),
            target_instance_id="bin",
            entity_id="destination",
            asset_id="asset:destination",
            reference_world_pose={
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            rights=_rights(),
            authoring_identity=_authoring(),
            physics_configuration=_physics(),
            simulator_name="Isaac Sim",
            simulator_version="6.0.1",
            output_root=tmp_path / "weak-threshold-output",
        )
    assert caught.value.errors == ("receptacle_asset_topology_thresholds_not_frozen",)
    assert not (tmp_path / "weak-threshold-output").exists()


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        ({"entity_id": 7}, "receptacle_asset_identifier_invalid"),
        ({"asset_id": 7}, "receptacle_asset_identifier_invalid"),
        ({"target_instance_id": 7}, "receptacle_asset_target_invalid"),
        ({"simulator_name": 7}, "receptacle_asset_simulator_identity_invalid"),
        ({"simulator_version": 7}, "receptacle_asset_simulator_identity_invalid"),
        ({"output_root": 7}, "receptacle_asset_output_root_invalid"),
    ],
)
def test_non_string_identifiers_return_stable_typed_errors(
    tmp_path: Path,
    override: dict,
    expected_error: str,
) -> None:
    labels, collision, topology = _fixture(tmp_path)
    arguments = {
        "labels_path": labels,
        "sage_collision_usd_path": collision,
        "topology_receipt": topology,
        **_visual_inputs(tmp_path, topology),
        "target_instance_id": "bin",
        "entity_id": "destination",
        "asset_id": "asset:destination",
        "reference_world_pose": {
            "position_world_m": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "rights": _rights(),
        "authoring_identity": _authoring(),
        "physics_configuration": _physics(),
        "simulator_name": "Isaac Sim",
        "simulator_version": "6.0.1",
        "output_root": tmp_path / f"typed-{expected_error}-{next(iter(override))}",
    }
    arguments.update(override)

    with pytest.raises(RegisteredStaticReceptacleAssetError) as caught:
        build_registered_static_receptacle_asset(**arguments)
    assert caught.value.errors == (expected_error,)


@pytest.mark.parametrize(
    "variant",
    [
        "missing_walls",
        "slatted_cap",
        "internal_planes",
        "sparse_walls",
        "lattice_aligned_posts",
        "floor_wall_gap",
    ],
)
def test_source_enclosure_defects_never_become_authored_twin_geometry(
    tmp_path: Path,
    variant: str,
) -> None:
    labels, collision, topology = _fixture(
        tmp_path,
        missing_walls=variant == "missing_walls",
        slatted_cap=variant == "slatted_cap",
        internal_planes=variant == "internal_planes",
        sparse_walls=variant == "sparse_walls",
        lattice_aligned_posts=variant == "lattice_aligned_posts",
        interior_vertical_panel=variant == "interior_vertical_panel",
        floor_wall_gap=variant == "floor_wall_gap",
    )
    opening = topology["targets"][0]["opening_probe"]
    assert opening["open_collision_cavity_passed"] is False
    if variant == "missing_walls":
        sides = opening["side_wall_probe"]["sides"]
        assert sides["x_max"]["passed"] is False
        assert sides["y_max"]["passed"] is False
    elif variant == "slatted_cap":
        assert opening["overhead_clearance_probe"]["projected_intersection_triangle_count"] > 0
        assert (
            opening["overhead_clearance_probe"]["clear_of_above_floor_projected_geometry"] is False
        )
    elif variant == "internal_planes":
        assert all(
            side["hit_count"] == 0
            and any(
                ray["hit_distance_m"] is not None and ray["qualified_wall_hit"] is False
                for ray in side["rays"]
            )
            for side in opening["side_wall_probe"]["sides"].values()
        )
    elif variant == "sparse_walls":
        assert all(
            side["hit_fraction"] == pytest.approx(0.2) and side["passed"] is False
            for side in opening["side_wall_probe"]["sides"].values()
        )
    elif variant == "lattice_aligned_posts":
        assert opening["side_wall_probe"]["all_four_sides_passed"] is True
        projected = opening["side_projected_coverage"]
        assert projected["all_four_sides_passed"] is False
        assert all(
            side["coverage_fraction"] == pytest.approx(0.0625) and side["passed"] is False
            for side in projected["sides"].values()
        )
    elif variant == "floor_wall_gap":
        assert opening["side_wall_probe"]["all_four_sides_passed"] is True
        projected = opening["side_projected_coverage"]
        assert projected["all_four_sides_passed"] is False
        assert all(
            side["coverage_fraction"] < 0.99
            and side["maximum_uncovered_aperture_m"] > 0.03
            and side["passed"] is False
            for side in projected["sides"].values()
        )

    output = tmp_path / f"composed-{variant}"
    result = build_registered_static_receptacle_asset(
        labels_path=labels,
        sage_collision_usd_path=collision,
        topology_receipt=topology,
        **_visual_inputs(tmp_path, topology),
        target_instance_id="bin",
        entity_id="destination",
        asset_id="asset:destination",
        reference_world_pose={
            "position_world_m": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        rights=_rights(),
        authoring_identity=_authoring(),
        physics_configuration=_physics(),
        simulator_name="Isaac Sim",
        simulator_version="6.0.1",
        output_root=output,
    )
    assert output.is_dir()
    assert result["receipt"]["claim_boundary"]["source_collision_cavity_clear"] is False
    assert (
        result["receipt"]["claim_boundary"]["source_collision_component_used_as_runtime_geometry"]
        is False
    )
    assert (
        result["receipt"]["claim_boundary"][
            "authored_twin_dimensions_derived_from_registered_outer_bounds"
        ]
        is True
    )


def test_engineered_empty_twin_records_and_excludes_source_internal_geometry(
    tmp_path: Path,
) -> None:
    labels, collision, topology = _fixture(tmp_path, interior_vertical_panel=True)
    opening = topology["targets"][0]["opening_probe"]
    assert opening["open_collision_cavity_passed"] is False
    assert opening["side_wall_probe"]["all_four_sides_passed"] is True
    assert opening["side_projected_coverage"]["all_four_sides_passed"] is True
    assert opening["overhead_clearance_probe"]["clear_of_above_floor_projected_geometry"]
    assert opening["open_prism_obstruction_probe"]["obstruction_triangle_count"] > 0

    result = build_registered_static_receptacle_asset(
        labels_path=labels,
        sage_collision_usd_path=collision,
        topology_receipt=topology,
        **_visual_inputs(tmp_path, topology),
        target_instance_id="bin",
        entity_id="destination",
        asset_id="asset:destination",
        reference_world_pose={
            "position_world_m": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        rights=_rights(),
        authoring_identity=_authoring(),
        physics_configuration=_physics(),
        simulator_name="Isaac Sim",
        simulator_version="6.0.1",
        output_root=tmp_path / "engineered-empty-twin",
    )
    claims = result["receipt"]["claim_boundary"]
    assert claims["source_collision_cavity_clear"] is False
    assert claims["source_collision_obstruction_triangle_count"] > 0
    assert claims["source_collision_component_used_as_runtime_geometry"] is False
    assert claims["engineered_twin_not_source_scene_truth"] is True


def test_concurrent_empty_destination_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels, collision, topology = _fixture(tmp_path)
    destination = tmp_path / "racing-writer-output"
    original_mkdir = os.mkdir

    def racing_mkdir(path: str | os.PathLike[str], mode: int = 0o777) -> None:
        if Path(path) == destination:
            original_mkdir(path, mode)
            raise FileExistsError(path)
        original_mkdir(path, mode)

    monkeypatch.setattr(receptacle_asset_module.os, "mkdir", racing_mkdir)
    with pytest.raises(RegisteredStaticReceptacleAssetError) as caught:
        build_registered_static_receptacle_asset(
            labels_path=labels,
            sage_collision_usd_path=collision,
            topology_receipt=topology,
            **_visual_inputs(tmp_path, topology),
            target_instance_id="bin",
            entity_id="destination",
            asset_id="asset:destination",
            reference_world_pose={
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            rights=_rights(),
            authoring_identity=_authoring(),
            physics_configuration=_physics(),
            simulator_name="Isaac Sim",
            simulator_version="6.0.1",
            output_root=destination,
        )
    assert caught.value.errors == ("receptacle_asset_output_exists",)
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_rejects_metadata_smuggling_and_source_symlink(tmp_path: Path) -> None:
    labels, collision, topology = _fixture(tmp_path)
    rights = _rights()
    rights["attribution"] = "ply\n" + "0 0 0\n" * 400
    with pytest.raises(RegisteredStaticReceptacleAssetError) as smuggled:
        build_registered_static_receptacle_asset(
            labels_path=labels,
            sage_collision_usd_path=collision,
            topology_receipt=topology,
            **_visual_inputs(tmp_path, topology),
            target_instance_id="bin",
            entity_id="destination",
            asset_id="asset:destination",
            reference_world_pose={
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            rights=rights,
            authoring_identity=_authoring(),
            physics_configuration=_physics(),
            simulator_name="Isaac Sim",
            simulator_version="6.0.1",
            output_root=tmp_path / "smuggled",
        )
    assert smuggled.value.errors == ("receptacle_asset_rights_invalid:attribution",)

    linked = tmp_path / "linked.usda"
    linked.symlink_to(collision)
    with pytest.raises(RegisteredStaticReceptacleAssetError) as linked_error:
        build_registered_static_receptacle_asset(
            labels_path=labels,
            sage_collision_usd_path=linked,
            topology_receipt=topology,
            **_visual_inputs(tmp_path, topology),
            target_instance_id="bin",
            entity_id="destination",
            asset_id="asset:destination",
            reference_world_pose={
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            rights=_rights(),
            authoring_identity=_authoring(),
            physics_configuration=_physics(),
            simulator_name="Isaac Sim",
            simulator_version="6.0.1",
            output_root=tmp_path / "linked-output",
        )
    assert linked_error.value.errors == ("receptacle_asset_topology_source_unreadable",)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_output = build_registered_static_receptacle_asset(
        labels_path=labels,
        sage_collision_usd_path=collision,
        topology_receipt=topology,
        **_visual_inputs(tmp_path, topology),
        target_instance_id="bin",
        entity_id="destination",
        asset_id="asset:destination",
        reference_world_pose={
            "position_world_m": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        rights=_rights(),
        authoring_identity=_authoring(),
        physics_configuration=_physics(),
        simulator_name="Isaac Sim",
        simulator_version="6.0.1",
        output_root=linked_parent / "asset",
    )
    assert Path(linked_output["output_root"]) == (real_parent / "asset").resolve()
    assert (linked_parent / "asset" / RECEIPT_FILENAME).is_file()


def test_unsigned_visual_review_cannot_authorize_asset_build(tmp_path: Path) -> None:
    labels, collision, topology = _fixture(tmp_path)
    visual = _visual_inputs(tmp_path, topology)
    Path(visual["semantic_review_attestation_path"]).unlink()

    with pytest.raises(RegisteredStaticReceptacleAssetError) as error:
        build_registered_static_receptacle_asset(
            labels_path=labels,
            sage_collision_usd_path=collision,
            topology_receipt=topology,
            **visual,
            target_instance_id="bin",
            entity_id="destination",
            asset_id="asset:destination",
            reference_world_pose={
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            rights=_rights(),
            authoring_identity=_authoring(),
            physics_configuration=_physics(),
            simulator_name="Isaac Sim",
            simulator_version="6.0.1",
            output_root=tmp_path / "unsigned-output",
        )

    assert error.value.errors == (
        "receptacle_asset_visual_design_basis_unqualified:"
        "visual_basis_semantic_authority_unqualified:"
        "semantic_review_attestation_file_open_failed",
    )


def test_self_rehashed_visual_review_cannot_authorize_asset_build(tmp_path: Path) -> None:
    labels, collision, topology = _fixture(tmp_path)
    visual = _visual_inputs(tmp_path, topology)
    review_path = Path(visual["visual_review_receipt_path"])
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["reviewer_id"] = "self-rehashed-attacker"
    review["review_digest"] = canonical_digest(review, digest_field="review_digest")
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RegisteredStaticReceptacleAssetError) as error:
        build_registered_static_receptacle_asset(
            labels_path=labels,
            sage_collision_usd_path=collision,
            topology_receipt=topology,
            **visual,
            target_instance_id="bin",
            entity_id="destination",
            asset_id="asset:destination",
            reference_world_pose={
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            rights=_rights(),
            authoring_identity=_authoring(),
            physics_configuration=_physics(),
            simulator_name="Isaac Sim",
            simulator_version="6.0.1",
            output_root=tmp_path / "self-rehashed-output",
        )

    assert error.value.errors == (
        "receptacle_asset_visual_design_basis_unqualified:"
        "visual_basis_semantic_authority_join_invalid",
    )


def test_signed_solid_color_fixture_cannot_authorize_asset_build(tmp_path: Path) -> None:
    labels, collision, topology = _fixture(tmp_path)

    with pytest.raises(RegisteredStaticReceptacleAssetError) as error:
        build_registered_static_receptacle_asset(
            labels_path=labels,
            sage_collision_usd_path=collision,
            topology_receipt=topology,
            **_visual_inputs(tmp_path, topology, solid_color=True),
            target_instance_id="bin",
            entity_id="destination",
            asset_id="asset:destination",
            reference_world_pose={
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            rights=_rights(),
            authoring_identity=_authoring(),
            physics_configuration=_physics(),
            simulator_name="Isaac Sim",
            simulator_version="6.0.1",
            output_root=tmp_path / "solid-output",
        )

    assert error.value.errors == (
        "receptacle_asset_visual_design_basis_unqualified:visual_basis_frame_visually_trivial",
    )


def test_signed_one_pixel_outlier_fixture_cannot_authorize_asset_build(
    tmp_path: Path,
) -> None:
    labels, collision, topology = _fixture(tmp_path)

    with pytest.raises(RegisteredStaticReceptacleAssetError) as error:
        build_registered_static_receptacle_asset(
            labels_path=labels,
            sage_collision_usd_path=collision,
            topology_receipt=topology,
            **_visual_inputs(tmp_path, topology, near_solid=True),
            target_instance_id="bin",
            entity_id="destination",
            asset_id="asset:destination",
            reference_world_pose={
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            rights=_rights(),
            authoring_identity=_authoring(),
            physics_configuration=_physics(),
            simulator_name="Isaac Sim",
            simulator_version="6.0.1",
            output_root=tmp_path / "one-pixel-outlier-output",
        )

    assert error.value.errors == (
        "receptacle_asset_visual_design_basis_unqualified:visual_basis_frame_visually_trivial",
    )


@pytest.mark.parametrize("field", ["static_friction", "contact_offset_m"])
def test_public_physics_boundary_rejects_numeric_strings(tmp_path: Path, field: str) -> None:
    labels, collision, topology = _fixture(tmp_path)
    physics = _physics()
    physics[field] = str(physics[field])

    with pytest.raises(RegisteredStaticReceptacleAssetError) as error:
        build_registered_static_receptacle_asset(
            labels_path=labels,
            sage_collision_usd_path=collision,
            topology_receipt=topology,
            **_visual_inputs(tmp_path, topology),
            target_instance_id="bin",
            entity_id="destination",
            asset_id="asset:destination",
            reference_world_pose={
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            rights=_rights(),
            authoring_identity=_authoring(),
            physics_configuration=physics,
            simulator_name="Isaac Sim",
            simulator_version="6.0.1",
            output_root=tmp_path / f"numeric-string-{field}",
        )

    assert error.value.errors == (f"receptacle_asset_{field}_invalid",)
