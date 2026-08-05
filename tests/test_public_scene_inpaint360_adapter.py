from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.public_scene_inpaint360_adapter import (
    INPAINT360_COMMIT,
    Inpaint360AdapterError,
    materialize_inpaint360_adapter,
)


def _record(path: Path, root: Path) -> dict:
    import hashlib

    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    repo, data, method = tmp_path / "repo", tmp_path / "data", tmp_path / "method"
    repo.mkdir()
    data.mkdir()
    method.mkdir()
    subprocess.run(["git", "init", "-q", str(method)], check=True)
    subprocess.run(["git", "-C", str(method), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(method), "config", "user.name", "Test"], check=True)
    (method / "README.md").write_text("fixture")
    subprocess.run(["git", "-C", str(method), "add", "."], check=True)
    subprocess.run(["git", "-C", str(method), "commit", "-qm", "fixture"], check=True)
    input_root = data / "input"
    (input_root / "images").mkdir(parents=True)
    (input_root / "masks").mkdir()
    count = 20
    splat_path = write_standard_3dgs_ply(
        SplatData(
            count=count,
            xyz=np.linspace(0.0, 1.0, count * 3, dtype=np.float32).reshape(count, 3),
            opacity=np.full(count, 5.0, np.float32),
            f_dc=np.full((count, 3), 0.2, np.float32),
            scales=np.full((count, 3), -3.0, np.float32),
            quats=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32), (count, 1)),
            properties=(),
            sh_rest=np.zeros((count, 45), np.float32),
        ),
        input_root / "scene_standard.ply",
    )
    cameras = []
    images, masks = [], []
    for index in range(8):
        camera_id = f"view_{index:02d}"
        image = input_root / "images" / f"{camera_id}.png"
        mask = input_root / "masks" / f"{camera_id}.png"
        Image.new("RGB", (1024, 1024), (20 + index, 30, 40)).save(image)
        pixels = np.zeros((1024, 1024), np.uint8)
        pixels[400:600, 450:550] = 255
        Image.fromarray(pixels, mode="L").save(mask)
        pose = np.eye(4)
        angle = 0.05 * index
        pose[:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
        pose[:3, 3] = [index * 0.1, 1.0, 0.5]
        cameras.append(
            {
                "camera_id": camera_id,
                "T_world_camera_opencv": pose.tolist(),
                "intrinsics": {
                    "model": "PINHOLE", "fx": 900.0, "fy": 900.0,
                    "cx": 512.0, "cy": 512.0, "width": 1024, "height": 1024,
                },
            }
        )
        images.append({"camera_id": camera_id, **_record(image, input_root)})
        masks.append({"camera_id": camera_id, **_record(mask, input_root)})
    cameras_path = input_root / "cameras.v1.json"
    cameras_path.write_text(json.dumps(cameras))
    receipt = {
        "schema_version": "adp009b_interiorgs_edit_input_receipt.v1",
        "status": "render_derived_input_packet_materialized",
        "scene": {
            "publisher_scene_id": "840313",
            "target_instance_id": "160",
            "target_obb_corners_m": [
                [-0.1, -0.2, -0.3],
                [0.1, -0.2, -0.3],
                [0.1, 0.2, -0.3],
                [-0.1, 0.2, -0.3],
                [-0.1, -0.2, 0.3],
                [0.1, -0.2, 0.3],
                [0.1, 0.2, 0.3],
                [-0.1, 0.2, 0.3],
            ],
        },
        "proof_boundaries": {"inpainting_result": False},
        "derived_artifacts": {
            "cameras": _record(cameras_path, input_root),
            "standard_splat": _record(splat_path, input_root),
            "images": images,
            "masks": masks,
        },
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = repo / "input_receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    return {"repo": repo, "data": data, "method": method, "input": input_root, "receipt": receipt_path}


def test_adapter_stages_exact_colmap_masks_and_unexecuted_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    real_run = subprocess.run

    def patched_run(command, **kwargs):
        if command[:4] == ["git", "-C", str(paths["method"]), "rev-parse"] and command[-1] == "HEAD":
            return subprocess.CompletedProcess(command, 0, INPAINT360_COMMIT + "\n", "")
        return real_run(command, **kwargs)

    monkeypatch.setattr("blueprint_pipeline.public_scene_inpaint360_adapter.subprocess.run", patched_run)
    receipt = materialize_inpaint360_adapter(
        input_receipt_path=paths["receipt"], input_root=paths["input"],
        repo_root=paths["repo"], data_root=paths["data"], method_root=paths["method"],
        output_root=paths["data"] / "adapter",
        receipt_output=paths["repo"] / "retained" / "adapter_receipt.json",
    )
    output = paths["data"] / "adapter"
    assert receipt["status"] == "prepared_unexecuted"
    assert receipt["execution"]["author_method_executed"] is False
    assert receipt["source"]["source_modified_for_adapter"] is False
    assert receipt["adapter_repository"]["tracked_files_clean"] is True
    assert (output / "source" / "sparse" / "0" / "cameras.txt").is_file()
    assert len([line for line in (output / "source" / "sparse" / "0" / "images.txt").read_text().splitlines() if ".png" in line]) == 8
    image_lines = [
        line for line in (output / "source" / "sparse" / "0" / "images.txt").read_text().splitlines()
        if ".png" in line
    ]
    source_cameras = json.loads((paths["input"] / "cameras.v1.json").read_text())
    for line, source_camera in zip(image_lines, source_cameras, strict=True):
        fields = line.split()
        qw, qx, qy, qz = map(float, fields[1:5])
        tx, ty, tz = map(float, fields[5:8])
        rotation = np.asarray(
            [
                [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
                [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qw * qx],
                [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx**2 - 2 * qy**2],
            ]
        )
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = [tx, ty, tz]
        np.testing.assert_allclose(
            np.linalg.inv(world_to_camera),
            np.asarray(source_camera["T_world_camera_opencv"]),
            atol=1e-9,
        )
    mask = np.asarray(Image.open(output / "source" / "raw_hqsam" / "view_00.png"))
    assert set(np.unique(mask)) == {0, 1}
    assert (output / "vanilla_3dgs" / "point_cloud" / "iteration_30000" / "point_cloud.ply").is_file()
    assert (output / "vanilla_3dgs" / "cfg_args").read_text() == "Namespace()\n"
    removal_config = json.loads(
        (
            output / "config/object_removal/blueprint/840313.json"
        ).read_text()
    )
    inpaint_config = json.loads(
        (
            output / "config/object_inpaint/blueprint/840313.json"
        ).read_text()
    )
    assert removal_config["select_obj_id"] == [1]
    assert removal_config["target_id"] == [1]
    assert removal_config["surrounding_ids"] == []
    assert removal_config["target_object_radius"] == pytest.approx(
        np.sqrt(0.1**2 + 0.2**2 + 0.3**2)
    )
    assert inpaint_config["select_obj_id"] == [1]
    assert inpaint_config["target_id"] == [1]
    assert inpaint_config["surrounding_ids"] == []
    assert inpaint_config["target_object_radius"] == removal_config["target_object_radius"]
    assert inpaint_config["images"] == "images_inpaint_unseen_virtual"
    assert receipt["adapter"]["paired_config_contract"].startswith(
        "config/object_removal/"
    )
    assert receipt["adapter"]["target_object_radius_derivation"] == (
        "max_distance_from_metric_obb_center"
    )
    assert receipt["adapter"]["target_obb_corners_m"] == receipt["scene"][
        "target_obb_corners_m"
    ]
    assert receipt["adapter"]["target_removal_volume_contract"] == (
        "gaussian_center_inside_exact_publisher_obb"
    )
    assert canonical_digest(receipt, digest_field="receipt_digest") == receipt["receipt_digest"]
    assert json.loads((paths["repo"] / "retained" / "adapter_receipt.json").read_text()) == receipt


def test_adapter_rejects_mutated_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_inpaint360_adapter.INPAINT360_COMMIT",
        subprocess.run(["git", "-C", str(paths["method"]), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
    )
    image = paths["input"] / "images" / "view_00.png"
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(Inpaint360AdapterError, match="image_bytes_changed"):
        materialize_inpaint360_adapter(
            input_receipt_path=paths["receipt"], input_root=paths["input"],
            repo_root=paths["repo"], data_root=paths["data"], method_root=paths["method"],
            output_root=paths["data"] / "adapter",
        )


def test_adapter_rejects_missing_metric_target_obb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_inpaint360_adapter.INPAINT360_COMMIT",
        subprocess.run(
            ["git", "-C", str(paths["method"]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    receipt = json.loads(paths["receipt"].read_text())
    del receipt["scene"]["target_obb_corners_m"]
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    paths["receipt"].write_text(json.dumps(receipt))
    subprocess.run(["git", "-C", str(paths["repo"]), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(paths["repo"]), "commit", "-qm", "missing obb"],
        check=True,
    )
    with pytest.raises(Inpaint360AdapterError, match="target_metric_obb_missing"):
        materialize_inpaint360_adapter(
            input_receipt_path=paths["receipt"],
            input_root=paths["input"],
            repo_root=paths["repo"],
            data_root=paths["data"],
            method_root=paths["method"],
            output_root=paths["data"] / "adapter",
        )
