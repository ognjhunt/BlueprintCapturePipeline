from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.dynamic_splat_scene import (
    DynamicSplatSceneError,
    build_dynamic_splat_render_request,
    build_dynamic_splat_scene,
    render_dynamic_splat_frame,
    validate_dynamic_splat_scene,
)
from blueprint_pipeline.gaussian_object_partition import partition_gaussian_object
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.scene_placement.semantic_gaussian_lifting import (
    canonical_json_digest,
)
from blueprint_pipeline.simready_asset_lane import (
    build_simready_asset_request,
    compose_simready_scene_binding,
    generate_simready_asset_draft,
)
from blueprint_pipeline.task_site_measurement_routing import validate_site_evidence_profile


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _f_dc(rgb: tuple[float, float, float]) -> np.ndarray:
    sh_c0 = 0.28209479177387814
    return np.asarray([(channel - 0.5) / sh_c0 for channel in rgb], dtype=np.float32)


def _partition(tmp_path: Path) -> dict:
    background = np.asarray(
        [[x, y, 3.0] for y in np.linspace(-0.8, 0.8, 25) for x in np.linspace(-1.1, 1.1, 35)],
        dtype=np.float32,
    )
    mug = np.asarray(
        [[x, y, 2.0] for y in np.linspace(-0.18, 0.18, 9) for x in np.linspace(-0.14, 0.14, 7)],
        dtype=np.float32,
    )
    xyz = np.concatenate([background, mug], axis=0)
    mug_ids = list(range(background.shape[0], xyz.shape[0]))
    colors = np.repeat(_f_dc((0.10, 0.20, 0.82))[None, :], xyz.shape[0], axis=0)
    colors[mug_ids] = _f_dc((0.94, 0.06, 0.04))
    splat = SplatData(
        count=xyz.shape[0],
        xyz=xyz,
        opacity=np.full(xyz.shape[0], 8.0, dtype=np.float32),
        f_dc=colors,
        scales=np.full((xyz.shape[0], 3), np.log(0.05), dtype=np.float32),
        quats=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (xyz.shape[0], 1)),
        properties=(),
    )
    source = write_standard_3dgs_ply(splat, tmp_path / "scene.ply")
    selection = {
        "schema_version": "gaussian_object_selection.v1",
        "selection_id": "selection-counter-mug-fixture",
        "object_id": "counter-mug",
        "source_splat_digest": _sha256(source),
        "source_gaussian_count": splat.count,
        "selected_gaussian_ids": mug_ids,
        "selected_gaussian_ids_digest": canonical_json_digest(mug_ids),
        "method": {
            "method_id": "fixture.exact_known_rows",
            "method_version": "1",
            "method_output_digest": canonical_json_digest({"mug_ids": mug_ids}),
        },
        "claim_ceiling": "candidate_object_gaussian_membership",
        "semantic_completeness_validated": False,
        "physics_authority_granted": False,
    }
    return partition_gaussian_object(source, selection, output_dir=tmp_path / "partition")


def _simready() -> dict:
    vertices = [
        [-0.14, -0.18, -0.05], [0.14, -0.18, -0.05],
        [0.14, 0.18, -0.05], [-0.14, 0.18, -0.05],
        [-0.14, -0.18, 0.05], [0.14, -0.18, 0.05],
        [0.14, 0.18, 0.05], [-0.14, 0.18, 0.05],
    ]
    faces = [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ]
    request = build_simready_asset_request(
        request_id="simready-counter-mug",
        object_id="counter-mug",
        bundle_id="fixture-capture",
        bundle_hash="sha256:" + "a" * 64,
        source_references={
            "segmentation_record_id": "fixture-segmentation",
            "mesh_record_id": "fixture-mesh",
            "splat_record_id": "fixture-splat",
            "provenance_record_id": "fixture-provenance",
        },
        density_class="ceramic_glass",
    )
    return generate_simready_asset_draft(
        request,
        vertices=vertices,
        faces=faces,
        generated_on="2026-08-02",
    )


def _scene(tmp_path: Path) -> tuple[dict, dict]:
    partition = _partition(tmp_path)
    initial = partition["object_frame"]["T_world_object_at_extraction"]
    moved = copy.deepcopy(initial)
    moved[0][3] += 0.60
    scene = build_dynamic_splat_scene(
        partition,
        _simready(),
        frames=[
            {"frame_id": "initial", "T_world_body": initial},
            {"frame_id": "moved", "T_world_body": moved},
        ],
    )
    return scene, partition


def _site() -> dict:
    return validate_site_evidence_profile(
        {
            "schema_version": "site_evidence_profile.v1",
            "profile_id": "fixture-site",
            "bundle_id": "fixture-capture",
            "bundle_hash": "sha256:" + "a" * 64,
            "provenance_record_id": "fixture-provenance",
            "rights": {"commercial_evaluation_allowed": True},
            "privacy": {"external_processing_allowed": False},
            "coordinate_system": {"metric_scale_verified": True},
            "evidence": {
                "gaussian_splat_appearance": {
                    "available": True,
                    "validated": True,
                    "record_id": "fixture-splat",
                }
            },
            "limitations": {"known_missing_regions": [], "forbidden_claims": []},
        }
    )


def test_scene_binds_one_visual_and_one_collider_to_one_pose(tmp_path: Path) -> None:
    scene, partition = _scene(tmp_path)
    row = scene["objects"][0]
    assert row["appearance"]["visual_instance_count"] == 1
    assert row["physics"]["collider_instance_count"] == 1
    assert row["pose_binding"]["appearance_and_collider_share_body_pose"] is True
    assert scene["background"]["digest"] == partition["artifacts"]["background"]["digest"]
    assert scene["render_invariants"]["object_gaussians_absent_from_static_background"] is True

    initial = build_dynamic_splat_render_request(scene, frame_id="initial")
    moved = build_dynamic_splat_render_request(scene, frame_id="moved")
    schema = json.loads(
        (
            Path(__file__).parents[1] / "docs/schemas/dynamic_splat_scene.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(scene, schema)
    jsonschema.validate(initial, schema)
    jsonschema.validate(moved, schema)
    initial_transform = np.asarray(initial["objects"][0]["T_world_object"])
    moved_transform = np.asarray(moved["objects"][0]["T_world_object"])
    np.testing.assert_allclose(moved_transform[:3, 3] - initial_transform[:3, 3], [0.6, 0.0, 0.0])
    assert initial["expected_object_ids"] == ["counter-mug"]


def test_legacy_scene_binding_is_static_only_until_partition_is_supplied(
    tmp_path: Path,
) -> None:
    manifest = _simready()
    static_only = compose_simready_scene_binding(_site(), [manifest])
    assert static_only["object_slots"][0]["appearance_source"] == (
        "scene_gaussian_splat_static_only"
    )
    assert static_only["object_slots"][0][
        "dynamic_object_renderable_without_duplicate"
    ] is False

    partition = _partition(tmp_path)
    dynamic = compose_simready_scene_binding(
        _site(),
        [manifest],
        gaussian_object_partitions=[partition],
    )
    slot = dynamic["object_slots"][0]
    assert slot["appearance_source"] == "movable_object_gaussian_partition"
    assert slot["object_absent_from_static_background"] is True
    assert slot["dynamic_object_renderable_without_duplicate"] is True
    assert slot["gaussian_object_partition_digest"] == (
        partition["gaussian_object_partition_digest"]
    )


def test_duplicate_object_or_pose_channel_fails_closed(tmp_path: Path) -> None:
    scene, _ = _scene(tmp_path)
    duplicate = copy.deepcopy(scene)
    duplicate.pop("dynamic_splat_scene_digest")
    duplicate["objects"].append(copy.deepcopy(duplicate["objects"][0]))
    with pytest.raises(DynamicSplatSceneError, match="duplicate_object_id|duplicate_pose_channel"):
        validate_dynamic_splat_scene(duplicate)

    incomplete = copy.deepcopy(scene)
    incomplete.pop("dynamic_splat_scene_digest")
    incomplete["frames"][0]["body_poses"] = {}
    with pytest.raises(DynamicSplatSceneError, match="pose_channels_incomplete"):
        validate_dynamic_splat_scene(incomplete)


def test_renderer_blocks_if_partition_asset_is_tampered(tmp_path: Path) -> None:
    scene, _ = _scene(tmp_path)
    Path(scene["objects"][0]["appearance"]["path"]).write_bytes(b"tampered")
    result = render_dynamic_splat_frame(
        scene,
        frame_id="initial",
        cameras=[{"id": "camera", "spec": {}}],
        output_dir=tmp_path / "render",
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["dynamic_splat_render_asset_digest_mismatch"]
    schema = json.loads(
        (
            Path(__file__).parents[1] / "docs/schemas/dynamic_splat_scene.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(result, schema)


@pytest.mark.slow
def test_real_spark_render_moves_mug_without_leaving_a_duplicate(tmp_path: Path) -> None:
    scene, _ = _scene(tmp_path)
    camera = {
        "id": "robot-camera",
        "spec": {
            "pose": {"T_world_camera_opencv": np.eye(4).tolist()},
            "intrinsics": {
                "fx": 250.0,
                "fy": 250.0,
                "cx": 160.0,
                "cy": 120.0,
                "width": 320,
                "height": 240,
            },
        },
    }
    initial = render_dynamic_splat_frame(
        scene,
        frame_id="initial",
        cameras=[camera],
        output_dir=tmp_path / "initial-render",
        width=320,
        height=240,
    )
    moved = render_dynamic_splat_frame(
        scene,
        frame_id="moved",
        cameras=[camera],
        output_dir=tmp_path / "moved-render",
        width=320,
        height=240,
    )
    assert initial["status"] == "completed", initial
    assert moved["status"] == "completed", moved
    assert initial["proof_boundary"]["exactly_one_declared_visual_instance_rendered"] is True
    assert moved["proof_boundary"]["exactly_one_declared_visual_instance_rendered"] is True

    initial_png = Path(initial["render_result"]["cameras"][0]["path"])
    moved_png = Path(moved["render_result"]["cameras"][0]["path"])
    initial_rgb = np.asarray(Image.open(initial_png).convert("RGB"), dtype=np.int16)
    moved_rgb = np.asarray(Image.open(moved_png).convert("RGB"), dtype=np.int16)
    initial_red = (initial_rgb[..., 0] > initial_rgb[..., 1] + 45) & (
        initial_rgb[..., 0] > initial_rgb[..., 2] + 45
    )
    moved_red = (moved_rgb[..., 0] > moved_rgb[..., 1] + 45) & (
        moved_rgb[..., 0] > moved_rgb[..., 2] + 45
    )
    assert int(initial_red.sum()) > 150
    assert int(moved_red.sum()) > 150
    initial_x = float(np.where(initial_red)[1].mean())
    moved_x = float(np.where(moved_red)[1].mean())
    assert 145.0 < initial_x < 175.0
    assert moved_x > initial_x + 55.0
    # A moved render may contain anti-aliased edge residue, but not a second mug
    # at the old location.
    assert int(moved_red[:, 140:181].sum()) < int(initial_red[:, 140:181].sum()) * 0.08
