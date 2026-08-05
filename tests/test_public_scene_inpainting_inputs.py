from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.public_scene_inpainting_inputs import (
    PublicSceneInpaintingInputError,
    _inside_obb,
    build_public_scene_inpainting_input_request,
    materialize_public_scene_inpainting_inputs,
)


def _request() -> dict:
    views = [
        ("approach_wide", [0.0, 1.4, 0.55]),
        ("approach_close", [0.0, 0.72, 0.28]),
        ("left_translate", [-0.55, 0.95, 0.38]),
        ("right_translate", [0.65, 1.05, 0.48]),
        ("raised_left", [-0.42, 1.12, 0.88]),
        ("low_right", [0.38, 0.78, 0.10]),
    ]
    return {
        "schema_version": "adp009b_interiorgs_edit_input_request.v1",
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "frozen_before_render": True,
        "method_outcomes_observed_before_freeze": False,
        "scene": {
            "publisher_scene_id": "123456",
            "target_instance_id": "10",
            "target_semantic_label": "canned_beverage",
            "component_manifest_path": "manifest.json",
            "component_receipt_path": "receipt.json",
        },
        "rendering": {
            "renderer": "reference_spark_renderer_exact_camera",
            "graphics_backend": "swiftshader",
            "width": 1024,
            "height": 1024,
            "vertical_fov_deg": 50.0,
            "warmup_ms": 1,
            "settle_frames": 1,
            "settle_ms": 1,
            "timeout_seconds": 10,
        },
        "camera_policy": {
            "generator": "translated_target_coverage_v1",
            "orbit_only_forbidden": True,
            "views": [
                {
                    "camera_id": camera_id,
                    "position_offset_m": offset,
                    "target_offset_m": [0.0, 0.0, 0.0],
                }
                for camera_id, offset in views
            ],
        },
        "mask_policy": {
            "authority": "publisher_target_obb_plus_contained_gaussians",
            "minimum_contained_gaussians": 16,
            "dilation_pixels": 2,
            "support_threshold_8bit": 24,
            "minimum_support_inside_final_fraction": 0.99,
        },
    }


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, root: Path, role: str) -> dict:
    return {
        "role": role,
        "external_relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    harness = repo / "tools" / "splat_render" / "render_splat.mjs"
    harness.parent.mkdir(parents=True)
    harness.write_text("// fixture harness\n", encoding="utf-8")
    (harness.parent / "src").mkdir()
    (harness.parent / "src" / "render_entry.mjs").write_text(
        "// fixture render entry\n", encoding="utf-8"
    )
    source = data / "scene" / "source.ply"
    inside = np.asarray(
        [[x, y, z] for x in np.linspace(-0.08, 0.08, 4) for y in (-0.05, 0.05) for z in (0.03, 0.1, 0.17)],
        dtype=np.float32,
    )
    outside = np.asarray([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], dtype=np.float32)
    points = np.vstack([inside, outside])
    count = len(points)
    write_standard_3dgs_ply(
        SplatData(
            count=count,
            xyz=points,
            opacity=np.full(count, 8.0, dtype=np.float32),
            f_dc=np.full((count, 3), 0.4, dtype=np.float32),
            scales=np.full((count, 3), -3.0, dtype=np.float32),
            quats=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
            properties=(),
        ),
        source,
    )
    corners = [
        {"x": x, "y": y, "z": z}
        for x, y, z in (
            (-0.1, -0.1, 0.0), (0.1, -0.1, 0.0), (0.1, 0.1, 0.0), (-0.1, 0.1, 0.0),
            (-0.1, -0.1, 0.2), (0.1, -0.1, 0.2), (0.1, 0.1, 0.2), (-0.1, 0.1, 0.2),
        )
    ]
    labels = data / "scene" / "labels.json"
    labels.write_text(json.dumps([{"ins_id": "10", "label": "canned_beverage", "bounding_box": corners}]))
    structure = data / "scene" / "structure.json"
    structure.write_text(json.dumps({"rooms": []}))
    manifest = {
        "schema_version": "public_scene_component_manifest.v1",
        "scene_mapping": {"publisher_scene_id": "123456"},
        "target_binding": {"interiorgs_instance_id": "10", "semantic_label": "canned_beverage"},
        "materialized_artifacts": [
            _artifact(source, data, "appearance_3dgs"),
            _artifact(labels, data, "semantic_metadata"),
            _artifact(structure, data, "scene_structure"),
        ],
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    (repo / "manifest.json").write_text(json.dumps(manifest))
    receipt = {
        "schema_version": "public_scene_component_admission_receipt.v1",
        "status": "admitted",
        "component_manifest_digest": manifest["manifest_digest"],
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (repo / "receipt.json").write_text(json.dumps(receipt))
    request = build_public_scene_inpainting_input_request(_request())
    (repo / "request.json").write_text(json.dumps(request))
    return {"repo": repo, "data": data, "source": source, "output": data / "output"}


def _fake_render(**kwargs) -> dict:
    output = kwargs["output"]
    cameras = kwargs["cameras"]
    output.mkdir(parents=True, exist_ok=True)
    for row in cameras:
        size = (row["intrinsics"]["width"], row["intrinsics"]["height"])
        pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        if output.name == "images":
            pixels[:, :, 0] = np.arange(size[0], dtype=np.uint16)[None, :] % 255
            pixels[:, :, 1] = 80
        else:
            pixels[size[1] // 2 - 8 : size[1] // 2 + 8, size[0] // 2 - 8 : size[0] // 2 + 8] = 255
        Image.fromarray(pixels, mode="RGB").save(output / f"{row['camera_id']}.png")
    return {"command": ["fake-observed-render"], "result": {"status": "completed"}}


def test_request_forbids_caller_outcomes_and_orbit_only_camera_set() -> None:
    request = _request()
    request["status"] = "admitted"
    with pytest.raises(PublicSceneInpaintingInputError, match="caller_asserted_outcome"):
        build_public_scene_inpainting_input_request(request)
    request = _request()
    for row in request["camera_policy"]["views"]:
        row["position_offset_m"][2] = 0.4
        radius = 1.0
        row["position_offset_m"][0] = radius
        row["position_offset_m"][1] = 0.0
    with pytest.raises(PublicSceneInpaintingInputError, match="translation_baselines"):
        build_public_scene_inpainting_input_request(request)


def test_oriented_box_membership_does_not_collapse_to_aabb() -> None:
    angle = np.deg2rad(45.0)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    local = np.asarray(list(__import__("itertools").product((-1.0, 1.0), (-0.2, 0.2), (0.0, 0.3))))
    corners = local @ rotation.T
    points = np.asarray([[0.0, 0.0, 0.1], [0.8, 0.0, 0.1]])
    assert _inside_obb(points, corners).tolist() == [True, False]


def test_materializer_hashes_real_inputs_and_emits_truthful_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_fixture(tmp_path)

    def fake_convert(src, dst, **_kwargs):
        shutil.copy2(src, dst)
        return {"status": "completed", "command": ["copy-fixture"]}

    monkeypatch.setattr("blueprint_pipeline.public_scene_inpainting_inputs.convert_to_standard_ply", fake_convert)
    monkeypatch.setattr("blueprint_pipeline.public_scene_inpainting_inputs._render_harness", _fake_render)
    receipt = materialize_public_scene_inpainting_inputs(
        request_path=paths["repo"] / "request.json",
        repo_root=paths["repo"],
        data_root=paths["data"],
        output_root=paths["output"],
    )
    assert receipt["status"] == "render_derived_input_packet_materialized"
    assert receipt["scene"]["target_gaussian_count"] == 24
    assert receipt["camera_policy"]["orbit_only"] is False
    assert len(receipt["derived_artifacts"]["images"]) == 6
    assert len(receipt["derived_artifacts"]["masks"]) == 6
    assert receipt["method_execution"]["inpaint360gs_executed"] is False
    assert receipt["proof_boundaries"]["hidden_background_truth_available"] is False
    assert canonical_digest(receipt, digest_field="receipt_digest") == receipt["receipt_digest"]


def test_materializer_rejects_changed_source_bytes(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    paths["source"].write_bytes(paths["source"].read_bytes() + b"changed")
    with pytest.raises(PublicSceneInpaintingInputError, match="splat_bytes_changed"):
        materialize_public_scene_inpainting_inputs(
            request_path=paths["repo"] / "request.json",
            repo_root=paths["repo"],
            data_root=paths["data"],
            output_root=paths["output"],
        )
