from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.public_scene_inpainting_inputs import (
    PublicSceneInpaintingInputError,
    _inside_obb,
    _publisher_obb,
    build_public_scene_inpainting_input_request,
    materialize_public_scene_inpainting_inputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    return {"repo": repo, "data": data, "source": source, "output": data / "output"}


def _fake_render(**kwargs) -> dict:
    output = kwargs["output"]
    cameras = kwargs["cameras"]
    output.mkdir(parents=True, exist_ok=True)
    for row in cameras:
        size = (row["intrinsics"]["width"], row["intrinsics"]["height"])
        pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        if output.name in {"images", "scene_without_target"}:
            pixels[:, :, 0] = np.arange(size[0], dtype=np.uint16)[None, :] % 255
            pixels[:, :, 1] = 80
            if output.name == "scene_without_target":
                pixels[:, :, 1] = 48
        else:
            pixels[size[1] // 2 - 8 : size[1] // 2 + 8, size[0] // 2 - 8 : size[0] // 2 + 8] = 255
        Image.fromarray(pixels, mode="RGB").save(output / f"{row['camera_id']}.png")
    return {"command": ["fake-observed-render"], "result": {"status": "completed"}}


def _write_v2_fixture(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    source_dir = data / "public_scene"
    source_dir.mkdir()
    task = json.loads(
        (
            REPO_ROOT
            / "docs/arm_decision_proof_v1/manifests"
            / "third_scene_840920_task_a_freeze.v1.json"
        ).read_text(encoding="utf-8")
    )
    scene = json.loads(
        (
            REPO_ROOT
            / "docs/arm_decision_proof_v1/manifests"
            / "third_scene_840920_dual_task_scene_freeze.v1.json"
        ).read_text(encoding="utf-8")
    )
    bounds = task["source_object"]["observed_bounds_world_m"]
    lower = np.asarray(bounds["minimum"], dtype=np.float32)
    upper = np.asarray(bounds["maximum"], dtype=np.float32)
    points = np.asarray(
        [
            [x, y, z]
            for x in np.linspace(lower[0] + 0.1, upper[0] - 0.1, 4)
            for y in (lower[1] + 0.1, upper[1] - 0.1)
            for z in (lower[2] + 0.1, (lower[2] + upper[2]) / 2, upper[2] - 0.1)
        ],
        dtype=np.float32,
    )
    standard = source_dir / "scene_standard.ply"
    write_standard_3dgs_ply(
        SplatData(
            count=len(points),
            xyz=points,
            opacity=np.full(len(points), 8.0, dtype=np.float32),
            f_dc=np.full((len(points), 3), 0.4, dtype=np.float32),
            scales=np.full((len(points), 3), -3.0, dtype=np.float32),
            quats=np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (len(points), 1),
            ),
            properties=(),
        ),
        standard,
    )
    corners = [
        {"x": x, "y": y, "z": z}
        for x, y, z in __import__("itertools").product(
            (bounds["minimum"][0], bounds["maximum"][0]),
            (bounds["minimum"][1], bounds["maximum"][1]),
            (bounds["minimum"][2], bounds["maximum"][2]),
        )
    ]
    labels = source_dir / "labels.json"
    labels.write_text(
        json.dumps(
            [
                {
                    "ins_id": task["source_object"]["instance_id"],
                    "label": task["source_object"]["semantic_label"],
                    "bounding_box": corners,
                }
            ]
        ),
        encoding="utf-8",
    )
    structure = source_dir / "structure.json"
    structure.write_text(json.dumps({"rooms": []}), encoding="utf-8")
    interiorgs = scene["source_components"]["interiorgs"]
    interiorgs["sha256"] = "sha256:" + "1" * 64
    interiorgs["size_bytes"] = 123
    interiorgs["supporting_files"] = {
        "labels": {"sha256": _digest(labels), "size_bytes": labels.stat().st_size},
        "structure": {
            "sha256": _digest(structure),
            "size_bytes": structure.stat().st_size,
        },
    }
    sage_digest = "sha256:" + "2" * 64
    scene["source_components"]["sage_collision"]["sha256"] = sage_digest
    scene["scene_freeze_digest"] = canonical_digest(
        scene, digest_field="scene_freeze_digest"
    )
    task["scene_freeze_digest"] = scene["scene_freeze_digest"]
    task["task_freeze_digest"] = canonical_digest(
        task, digest_field="task_freeze_digest"
    )
    (repo / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    (repo / "task.json").write_text(json.dumps(task), encoding="utf-8")
    conversion = {
        "schema_version": "standard_splat_conversion_receipt.v1",
        "status": "standard_splat_conversion_materialized",
        "source": {"sha256": interiorgs["sha256"], "size_bytes": 123},
        "output": {
            "sha256": _digest(standard),
            "size_bytes": standard.stat().st_size,
            "gaussian_count": len(points),
            "gaussian_count_preserved": True,
            "standard_3dgs_schema_validated": True,
        },
        "raw_source_uploaded": False,
        "gaussian_ownership_claimed": False,
    }
    conversion["receipt_digest"] = canonical_digest(
        conversion, digest_field="receipt_digest"
    )
    (repo / "conversion.json").write_text(json.dumps(conversion), encoding="utf-8")
    frame = {
        "schema_version": "interiorgs_sage_shared_frame_candidate.v1",
        "source_digests": {
            "interiorgs_labels": _digest(labels),
            "sage_collision_usd": sage_digest,
        },
        "correspondences": [
            {
                "interiorgs_instance_id": task["source_object"]["instance_id"],
                "semantic_label": task["source_object"]["semantic_label"],
                "sage_prim_path": task["removal_plan"]["source_collider_prim_path"],
                "identity_receipt_digest": task["source_object"][
                    "collision_identity_receipt_digest"
                ],
            }
        ],
        "shared_frame_status": "provider_declared_not_independently_validated",
    }
    frame["receipt_digest"] = canonical_digest(frame, digest_field="receipt_digest")
    (source_dir / "frame.json").write_text(json.dumps(frame), encoding="utf-8")
    request = _request()
    request.update(
        {
            "schema_version": "public_scene_interiorgs_edit_input_request.v2",
            "adp_item": "ADP-009D",
            "scene": {
                "source_adapter": "dual_task_freeze_and_standard_splat_v1",
                "scene_freeze_path": "scene.json",
                "task_freeze_path": "task.json",
                "standard_splat_conversion_receipt_path": "conversion.json",
                "standard_splat_path": "public_scene/scene_standard.ply",
                "labels_path": "public_scene/labels.json",
                "structure_path": "public_scene/structure.json",
                "registered_frame_receipt_path": "public_scene/frame.json",
            },
        }
    )
    request["rendering"].update(
        {
            "supersampling": 1,
            "color_space": "srgb",
            "alpha_mode": "opaque_rgb",
            "background_rgb": 0,
            "exposure_mode": "renderer_default_unmodified",
        }
    )
    request["mask_policy"]["maximum_image_fraction"] = 0.84
    for view in request["camera_policy"]["views"]:
        view["position_offset_m"] = [
            float(value) * 2.0 for value in view["position_offset_m"]
        ]
    (repo / "request.json").write_text(
        json.dumps(build_public_scene_inpainting_input_request(request)),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True
    )
    return {
        "repo": repo,
        "data": data,
        "standard": standard,
        "output": data / "output",
    }


def _fake_sealed_render(**kwargs) -> dict:
    output = Path(kwargs["output_dir"])
    frames = output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for row in kwargs["cameras"]:
        width = int(row["intrinsics"]["width"])
        height = int(row["intrinsics"]["height"])
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        if output.name in {"images", "scene_without_target"}:
            pixels[:, :, 0] = np.arange(width, dtype=np.uint16)[None, :] % 255
            pixels[:, :, 1] = 80 if output.name == "images" else 48
        else:
            pixels[height // 2 - 8 : height // 2 + 8, width // 2 - 8 : width // 2 + 8] = 255
        Image.fromarray(pixels, mode="RGB").save(
            frames / f"{row['camera_id']}.png"
        )
    digest = "sha256:" + hashlib.sha256(output.name.encode()).hexdigest()
    return {
        "sealed_camera_render_manifest_digest": digest,
        "render_settings": {
            "dimensions": {
                "width": kwargs["cameras"][0]["intrinsics"]["width"],
                "height": kwargs["cameras"][0]["intrinsics"]["height"],
            },
            "supersampling": 1,
            "color_space": "srgb",
            "alpha_mode": "opaque_rgb",
            "background_rgb": "#000000",
            "exposure": {"mode": "renderer_default_unmodified", "ev": None},
        },
        "renderer_identity": {"repository_revision": "a" * 40},
    }


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


def test_mask_fraction_is_frozen_per_object_scale_with_legacy_default() -> None:
    legacy = build_public_scene_inpainting_input_request(_request())
    assert "maximum_image_fraction" not in legacy["mask_policy"]

    articulated = _request()
    articulated["mask_policy"]["maximum_image_fraction"] = 0.65
    built = build_public_scene_inpainting_input_request(articulated)
    assert built["mask_policy"]["maximum_image_fraction"] == 0.65

    invalid = _request()
    invalid["mask_policy"]["maximum_image_fraction"] = 0.95
    with pytest.raises(PublicSceneInpaintingInputError) as caught:
        build_public_scene_inpainting_input_request(invalid)
    assert "edit_input_mask_maximum_image_fraction_invalid" in caught.value.codes


def test_request_validates_occlusion_aware_contribution_gate() -> None:
    request = _request()
    request["mask_policy"].update(
        {
            "visual_contribution_threshold_8bit": 12,
            "minimum_visible_target_fraction": 0.05,
        }
    )
    built = build_public_scene_inpainting_input_request(request)
    assert built["mask_policy"]["minimum_visible_target_fraction"] == 0.05

    request["mask_policy"]["minimum_visible_target_fraction"] = 0.0
    with pytest.raises(PublicSceneInpaintingInputError) as caught:
        build_public_scene_inpainting_input_request(request)
    assert "edit_input_minimum_visible_target_fraction_invalid" in caught.value.codes


def test_oriented_box_membership_does_not_collapse_to_aabb() -> None:
    angle = np.deg2rad(45.0)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    local = np.asarray(list(__import__("itertools").product((-1.0, 1.0), (-0.2, 0.2), (0.0, 0.3))))
    corners = local @ rotation.T
    points = np.asarray([[0.0, 0.0, 0.1], [0.8, 0.0, 0.1]])
    assert _inside_obb(points, corners).tolist() == [True, False]


def test_publisher_semantic_text_and_stable_identifier_share_one_join() -> None:
    labels = [
        {
            "ins_id": "385",
            "label": "Notebook computer",
            "bounding_box": [
                {"x": x, "y": y, "z": z}
                for x, y, z in __import__("itertools").product((0, 1), repeat=3)
            ],
        }
    ]
    assert _publisher_obb(labels, "385", "notebook_computer").shape == (8, 3)


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
        receipt_output=paths["repo"] / "retained" / "receipt.json",
    )
    assert receipt["status"] == "render_derived_input_packet_materialized"
    assert receipt["scene"]["target_gaussian_count"] == 24
    assert receipt["camera_policy"]["orbit_only"] is False
    assert len(receipt["derived_artifacts"]["images"]) == 6
    assert len(receipt["derived_artifacts"]["masks"]) == 6
    assert receipt["proof_boundaries"]["source_target_obb_visual_contribution_measured"] is True
    assert all(
        row["visible_target_contribution_fraction"] == 1.0
        for row in receipt["derived_artifacts"]["masks"]
    )
    assert receipt["method_execution"]["inpaint360gs_executed"] is False
    assert receipt["proof_boundaries"]["hidden_background_truth_available"] is False
    assert receipt["repository"]["tracked_files_clean"] is True
    assert len(receipt["repository"]["commit"]) == 40
    assert receipt["executed_commands"]["decode"] == ["copy-fixture"]
    assert json.loads((paths["repo"] / "retained" / "receipt.json").read_text()) == receipt
    assert canonical_digest(receipt, digest_field="receipt_digest") == receipt["receipt_digest"]


def test_materializer_rejects_target_only_projection_hidden_by_scene_occluder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_fixture(tmp_path)

    def fake_convert(src, dst, **_kwargs):
        shutil.copy2(src, dst)
        return {"status": "completed", "command": ["copy-fixture"]}

    def occluded_render(**kwargs):
        result = _fake_render(**kwargs)
        if kwargs["output"].name == "scene_without_target":
            for camera in kwargs["cameras"]:
                shutil.copy2(
                    kwargs["output"].parent / "images" / f"{camera['camera_id']}.png",
                    kwargs["output"] / f"{camera['camera_id']}.png",
                )
        return result

    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_inpainting_inputs.convert_to_standard_ply",
        fake_convert,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_inpainting_inputs._render_harness",
        occluded_render,
    )
    with pytest.raises(
        PublicSceneInpaintingInputError, match="target_occluded_or_unrenderable"
    ):
        materialize_public_scene_inpainting_inputs(
            request_path=paths["repo"] / "request.json",
            repo_root=paths["repo"],
            data_root=paths["data"],
            output_root=paths["output"],
        )
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


def test_dual_task_adapter_reuses_qualified_standard_splat_and_sealed_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_v2_fixture(tmp_path)
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_inpainting_inputs.render_splat_at_exact_cameras",
        _fake_sealed_render,
    )
    receipt = materialize_public_scene_inpainting_inputs(
        request_path=paths["repo"] / "request.json",
        repo_root=paths["repo"],
        data_root=paths["data"],
        output_root=paths["output"],
    )

    assert receipt["schema_version"] == "public_scene_interiorgs_edit_input_receipt.v2"
    assert receipt["adp_item"] == "ADP-009D"
    assert receipt["source_admission"]["adapter"] == (
        "dual_task_freeze_and_standard_splat_v1"
    )
    assert receipt["scene"]["task_id"] == "task_a_washer_door_open"
    assert receipt["renderer"]["authorization_class"] == "method_input"
    assert set(receipt["renderer"]["render_manifest_digests"]) == {
        "images",
        "target_support",
        "scene_without_target",
    }
    assert receipt["proof_boundaries"]["gaussian_ownership_qualified"] is False
    assert receipt["proof_boundaries"][
        "mask_is_calibrated_candidate_not_owned_gaussian_classification"
    ] is True
    assert receipt["executed_commands"]["decode"][0] == (
        "reuse-standard-splat-conversion-receipt"
    )


def test_dual_task_adapter_rejects_changed_standard_splat_bytes(tmp_path: Path) -> None:
    paths = _write_v2_fixture(tmp_path)
    paths["standard"].write_bytes(paths["standard"].read_bytes() + b"changed")

    with pytest.raises(
        PublicSceneInpaintingInputError, match="standard_splat_bytes_changed"
    ):
        materialize_public_scene_inpainting_inputs(
            request_path=paths["repo"] / "request.json",
            repo_root=paths["repo"],
            data_root=paths["data"],
            output_root=paths["output"],
        )
