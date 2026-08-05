from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.gaussian_splat_decode import SplatData, read_standard_3dgs_ply, write_standard_3dgs_ply
from blueprint_pipeline import public_scene_infusion_adapter as adapter


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, root: Path, **extra: object) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
        **extra,
    }


def _git_repo(path: Path) -> tuple[str, str]:
    path.mkdir(parents=True)
    (path / "LICENSE").write_text("test fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "LICENSE"], check=True)
    subprocess.run(
        [
            "git", "-C", str(path), "-c", "user.name=Fixture", "-c",
            "user.email=fixture@example.invalid", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD^{tree}"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return commit, tree


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    input_root = data / "inputs"
    input_root.mkdir(parents=True)
    repo.mkdir()
    infusion = tmp_path / "Infusion"
    lama = tmp_path / "Inpaint360GS"
    infusion_commit, infusion_tree = _git_repo(infusion)
    lama_commit, lama_tree = _git_repo(lama)
    monkeypatch.setattr(adapter, "INFUSION_COMMIT", infusion_commit)
    monkeypatch.setattr(adapter, "INFUSION_TREE", infusion_tree)
    monkeypatch.setattr(adapter, "INPAINT360_COMMIT", lama_commit)
    monkeypatch.setattr(adapter, "INPAINT360_TREE", lama_tree)
    monkeypatch.setattr(
        adapter,
        "_adapter_repository_identity",
        lambda _repo: {
            "commit": "a" * 40,
            "tree": "b" * 40,
            "tracked_files_clean": True,
            "implementation_files": [],
        },
    )

    xyz = np.asarray(
        [
            [0.1, 0.1, 0.1], [0.2, 0.2, 0.2],
            [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    splat_path = write_standard_3dgs_ply(
        SplatData(
            count=len(xyz),
            xyz=xyz,
            opacity=np.zeros(len(xyz), dtype=np.float32),
            f_dc=np.zeros((len(xyz), 3), dtype=np.float32),
            scales=np.full((len(xyz), 3), -2.0, dtype=np.float32),
            quats=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(xyz), 1)),
            properties=(),
            sh_rest=np.arange(len(xyz) * 45, dtype=np.float32).reshape(len(xyz), 45),
        ),
        input_root / "scene_standard.ply",
    )
    cameras = []
    image_records = []
    mask_records = []
    for camera_id, pixels in (("wide", 4), ("close", 9)):
        pose = np.eye(4, dtype=float)
        pose[2, 3] = 3.0
        cameras.append(
            {
                "camera_id": camera_id,
                "T_world_camera_opencv": pose.tolist(),
                "intrinsics": {"model": "PINHOLE", "width": 8, "height": 8, "fx": 8.0, "fy": 8.0, "cx": 4.0, "cy": 4.0},
            }
        )
        image_path = input_root / "images" / f"{camera_id}.png"
        mask_path = input_root / "masks" / f"{camera_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), 64, dtype=np.uint8), mode="RGB").save(image_path)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask.reshape(-1)[:pixels] = 255
        Image.fromarray(mask, mode="L").save(mask_path)
        image_records.append(_record(image_path, input_root, camera_id=camera_id))
        mask_records.append(
            _record(
                mask_path,
                input_root,
                camera_id=camera_id,
                masked_pixel_count=pixels,
                gaussian_support_inside_fraction=1.0,
            )
        )
    camera_path = input_root / "cameras.v1.json"
    camera_path.write_text(canonical_json(cameras) + "\n", encoding="utf-8")
    input_receipt = {
        "schema_version": "adp009b_interiorgs_edit_input_receipt.v1",
        "status": "render_derived_input_packet_materialized",
        "scene": {
            "publisher_scene_id": "fixture",
            "target_instance_id": "1",
            "target_semantic_label": "can",
            "target_gaussian_count": 2,
            "target_obb_corners_m": [
                [0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],
                [0, 0, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.5], [0, 0.5, 0.5],
            ],
        },
        "camera_policy": {"camera_count": 2},
        "derived_artifacts": {
            "cameras": _record(camera_path, input_root),
            "standard_splat": _record(splat_path, input_root),
            "images": image_records,
            "masks": mask_records,
        },
    }
    input_receipt["receipt_digest"] = canonical_digest(input_receipt)
    input_receipt_path = repo / "input_receipt.json"
    input_receipt_path.write_text(canonical_json(input_receipt) + "\n", encoding="utf-8")

    archive = data / "methods" / "big-lama.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"fixture checkpoint archive")
    monkeypatch.setattr(adapter, "BIG_LAMA_SHA256", _sha(archive))
    prerequisite = {
        "schema_version": "public_scene_method_prerequisite_receipt.v1",
        "methods": {
            "infusion_primary_adapter": {
                "checkpoint_rights_established": False,
                "remote_snapshots": [
                    {"publisher": {"revision": adapter.INFUSION_CHECKPOINT_REVISION}}
                ],
            },
            "inpaint360_author_smoke": {
                "artifacts": [
                    {
                        "artifact_id": "big_lama_author_linked_archive",
                        "relative_path": archive.relative_to(data).as_posix(),
                        "size_bytes": archive.stat().st_size,
                        "rights_established": True,
                    }
                ]
            },
        },
    }
    prerequisite["receipt_digest"] = canonical_digest(prerequisite)
    prerequisite_path = data / "methods" / "prerequisite.json"
    prerequisite_path.write_text(canonical_json(prerequisite) + "\n", encoding="utf-8")
    return {
        "repo": repo,
        "data": data,
        "input_root": input_root,
        "input_receipt": input_receipt_path,
        "prerequisite": prerequisite_path,
        "infusion": infusion,
        "lama": lama,
        "output": data / "output",
    }


def _run(paths: dict[str, Path]) -> dict[str, object]:
    return adapter.materialize_infusion_adapter(
        input_receipt_path=paths["input_receipt"],
        input_root=paths["input_root"],
        prerequisite_receipt_path=paths["prerequisite"],
        repo_root=paths["repo"],
        data_root=paths["data"],
        infusion_root=paths["infusion"],
        lama_source_root=paths["lama"],
        output_root=paths["output"],
        receipt_output=paths["repo"] / "retained.json",
    )


def test_materializes_blocked_packet_from_observed_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = _run(paths)
    assert receipt["status"] == "blocked"
    assert receipt["smallest_blocker"] == "infusion_checkpoint_license_missing"
    assert receipt["single_view_selection"]["selected_camera_id"] == "close"
    assert receipt["single_view_selection"]["method_outcomes_observed_before_selection"] is False
    assert receipt["execution"]["infusion_executed"] is False
    background = read_standard_3dgs_ply(
        paths["output"] / "incomplete_model/point_cloud/iteration_30000/point_cloud.ply"
    )
    assert background.count == 4
    assert background.sh_rest.shape == (4, 45)
    assert (paths["output"] / "lama_input/close_mask.png").is_file()


def test_rejects_caller_asserted_checkpoint_rights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    value = json.loads(paths["prerequisite"].read_text(encoding="utf-8"))
    value["methods"]["infusion_primary_adapter"]["checkpoint_rights_established"] = True
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    paths["prerequisite"].write_text(canonical_json(value) + "\n", encoding="utf-8")
    with pytest.raises(adapter.InFusionAdapterError, match="infusion_checkpoint_rights_state_unexpected"):
        _run(paths)


def test_rejects_mutated_render_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    (paths["input_root"] / "images/close.png").write_bytes(b"changed")
    with pytest.raises(adapter.InFusionAdapterError, match="infusion_input_artifact_bytes_changed"):
        _run(paths)
