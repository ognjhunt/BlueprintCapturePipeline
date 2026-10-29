"""Pinned public MolmoAct2 YAM inputs; no allocation, download or activation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MODEL_ID = "allenai/MolmoAct2-BimanualYAM"
MODEL_REVISION = "8dcbed66f2380e4393189c303ea72488eb9e63c2"
SOURCE_REVISION = "6070080a20321b4f498ab30f28e1d09ac465edb7"
NORM_TAG = "yam_dual_molmoact2"
NORM_SHA256 = "6357a264ac820c5b775c884bd303ed13ffa18a36def28409312c842139879f4b"
INTERFACE_ID = "molmoact2_yam_joint14.v1"


def release_inventory() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "_catalog_data/molmoact2_yam_release.v1.json").read_text())


def verify_checkpoint_directory(directory: str | Path) -> dict[str, Any]:
    """Hash all local inference inputs before executing checkpoint Python code.

    Large weight hashes are publisher LFS declarations until this function has
    read their full local bytes. A model card alone is never an execution receipt.
    The staged directory must contain real files, not mutable cache symlinks.
    """
    root = Path(directory).resolve(strict=True)
    manifest = release_inventory()
    required = {row["path"]: row for row in manifest["files"] if row["required_for_inference"]}
    for name, row in required.items():
        path = root / name
        if (not path.is_file() or path.is_symlink() or root not in path.resolve().parents
                or path.stat().st_size != row["bytes"]):
            raise ValueError(f"molmoact2_checkpoint_file_invalid:{name}")
        expected = row["publisher_lfs_sha256"] or row["locally_verified_sha256"]
        if not expected:
            raise ValueError(f"molmoact2_checkpoint_digest_missing:{name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"molmoact2_checkpoint_digest_mismatch:{name}")
    expected_code = {name for name in required if name.endswith(".py")}
    if {p.relative_to(root).as_posix() for p in root.rglob("*.py")} != expected_code:
        raise ValueError("molmoact2_checkpoint_extra_code")
    expected_shards = {name for name in required if name.endswith(".safetensors")}
    if ({p.name for p in root.glob("*.safetensors")} != expected_shards
            or set(json.loads((root / "model.safetensors.index.json").read_text())["weight_map"].values())
            != expected_shards):
        raise ValueError("molmoact2_checkpoint_selection_ambiguous")
    return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "source_revision": SOURCE_REVISION,
            "norm_stats_sha256": NORM_SHA256, "checkpoint_bytes_verified": True,
            "files_verified": len(required), "license": manifest["license"],
            "license_evidence_url": manifest["license_evidence_url"],
            "inference_performed": False}


def publisher_camera_reference() -> dict[str, Any]:
    """Publisher's ManiSkill reference, not an Isaac/site calibration claim.

    Poses use Sapien's optical convention and wxyz quaternions in the specified
    publisher frames. Frame conversion is required before applying them to Isaac.
    Current I2RT CAD includes overhead D405; this profile simulates D435i optics.
    """
    wrist = {"position_m": [0.0, 0.09, 0.06],
             "orientation_wxyz": [0.612372429196013, -0.35355339154618404,
                                   -0.3535533966987049, -0.612372438120441],
             "horizontal_fov_deg": 87.0, "camera_model": "D405"}
    return {"source_revision": SOURCE_REVISION, "source_path": "sim_eval/robots/bimanual_yam.py",
            "width": 640, "height": 360, "pose_convention": "sapien_camera_wxyz",
            "calibration_status": "publisher_simulation_reference_not_site_measurement",
            "current_cad_discrepancy": "overhead_D405_body_with_simulated_D435i_optics",
            "top_cam": {"mount_link": "bimanual_base", "position_m": [0.15, 0.0, 0.8],
                        "orientation_wxyz": [0.7660444431189782, 0.0, 0.6427876096865391, 0.0],
                        "horizontal_fov_deg": 69.4, "camera_model": "D435i"},
            "left_cam": {**wrist, "mount_link": "left_link_6"},
            "right_cam": {**wrist, "mount_link": "right_link_6"}}
