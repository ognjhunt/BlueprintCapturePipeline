"""Prepare AuraFusion360's distinct ADP-009B InteriorGS challenger packet.

AuraFusion360 does not edit the publisher PLY in place.  This adapter derives
the author's COLMAP-style training dataset from Blueprint's already frozen
render packet, preregisters one Big-LaMa-generated reference view, and emits
the author's native workflow commands.  It never marks staging as execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .gaussian_splat_decode import read_standard_3dgs_ply
from .method_input_scope import evaluate_multiview_mask_fraction_scope

SCHEMA_VERSION = "adp009b_aurafusion360_adapter_receipt.v1"
AURA_REPOSITORY = "https://github.com/kkennethwu/AuraFusion360_official"
AURA_COMMIT = "f23b26c44ba84608306ba952510533ebf4c7877d"
AURA_TREE = "cc8447c66448b29bb4d39fec29c031df63d4b179"
AURA_SUBMODULES = {
    "submodules/diff-surfel-rasterization": "e0ed0207b3e0669960cfad70852200a4a5847f61",
    "submodules/diff-surfel-rasterization/third_party/glm": (
        "5c46b9c07008ae65cb81ab79cd677ecc1934b903"
    ),
    "submodules/simple-knn": "86710c2d4b46680c02301765dd79e465819c8f19",
}
LAMA_REPOSITORY = "https://github.com/advimman/lama"
LAMA_COMMIT = "786f5936b27fb3dacd2b1ad799e4de968ea697e7"
LAMA_TREE = "25f9902ca0c2ec4bf6c31c2b4427f0a4f05f2fd1"
BIG_LAMA_SIZE = 381_428_720
BIG_LAMA_SHA256 = "sha256:d7161bba4d68b438f9fa7f09dcb750a223804c300c68d214a5e0be16251fba8d"
SH_C0 = 0.28209479177387814
# Empirical Blueprint admission ceiling: four times the largest target-mask
# fraction in the previously qualified small-object rehearsal. This is a
# conservative spend gate, not a claim about AuraFusion360's training set.
AURA_QUALIFIED_ANCHOR_MAX_MASK_FRACTION = 0.02533
AURA_MAXIMUM_ADMITTED_MASK_FRACTION = 4.0 * AURA_QUALIFIED_ANCHOR_MAX_MASK_FRACTION


class AuraAdapterError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _under(path: str | Path, root: Path, code: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise AuraAdapterError([code])
    return resolved


def _read_object(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuraAdapterError([code]) from exc
    if not isinstance(value, dict):
        raise AuraAdapterError([code])
    return value


def _git(method_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(method_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise AuraAdapterError(["aurafusion360_source_identity_unavailable"]) from exc


def _source_identity(method_root: Path) -> dict[str, Any]:
    commit = _git(method_root, "rev-parse", "HEAD")
    tree = _git(method_root, "rev-parse", "HEAD^{tree}")
    if commit != AURA_COMMIT or tree != AURA_TREE:
        raise AuraAdapterError(["aurafusion360_source_revision_mismatch"])
    if _git(method_root, "status", "--porcelain", "--untracked-files=no"):
        raise AuraAdapterError(["aurafusion360_source_tracked_files_dirty"])
    observed: dict[str, str] = {}
    for line in _git(method_root, "submodule", "status", "--recursive").splitlines():
        fields = line.strip().split()
        if len(fields) >= 2:
            observed[fields[1]] = fields[0].lstrip("-+")
    if observed != AURA_SUBMODULES:
        raise AuraAdapterError(["aurafusion360_submodule_identity_mismatch"])
    return {
        "repository": AURA_REPOSITORY,
        "commit": commit,
        "tree": tree,
        "submodules": [
            {"path": path, "commit": revision}
            for path, revision in sorted(observed.items())
        ],
        "tracked_files_clean": True,
        "source_modified_for_adapter": False,
    }


def _lama_identity(lama_root: Path, archive: Path) -> dict[str, Any]:
    if _git(lama_root, "rev-parse", "HEAD") != LAMA_COMMIT:
        raise AuraAdapterError(["aurafusion360_reference_lama_revision_mismatch"])
    if _git(lama_root, "rev-parse", "HEAD^{tree}") != LAMA_TREE:
        raise AuraAdapterError(["aurafusion360_reference_lama_tree_mismatch"])
    if _git(lama_root, "status", "--porcelain", "--untracked-files=no"):
        raise AuraAdapterError(["aurafusion360_reference_lama_tracked_files_dirty"])
    if (
        not archive.is_file()
        or archive.stat().st_size != BIG_LAMA_SIZE
        or _sha256(archive) != BIG_LAMA_SHA256
    ):
        raise AuraAdapterError(["aurafusion360_reference_lama_checkpoint_changed"])
    return {
        "repository": LAMA_REPOSITORY,
        "commit": LAMA_COMMIT,
        "tree": LAMA_TREE,
        "license": "Apache-2.0",
        "checkpoint_archive": {
            "path": str(archive),
            "size_bytes": BIG_LAMA_SIZE,
            "sha256": BIG_LAMA_SHA256,
        },
        "tracked_files_clean": True,
    }


def _rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    matrix = np.asarray(
        [
            [rxx - ryy - rzz, 0.0, 0.0, 0.0],
            [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    values, vectors = np.linalg.eigh(matrix)
    qvec = vectors[[3, 0, 1, 2], int(np.argmax(values))]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def _write_colmap(scene_root: Path, cameras: Sequence[Mapping[str, Any]]) -> list[Path]:
    sparse = scene_root / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    first = cameras[0]["intrinsics"]
    cameras_path = sparse / "cameras.txt"
    cameras_path.write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {first['width']} {first['height']} "
        f"{first['fx']:.17g} {first['fy']:.17g} "
        f"{first['cx']:.17g} {first['cy']:.17g}\n",
        encoding="utf-8",
    )
    lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    for index, row in enumerate(cameras, start=1):
        pose = np.asarray(row["T_world_camera_opencv"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise AuraAdapterError(["aurafusion360_camera_pose_invalid"])
        world_to_camera = np.linalg.inv(pose)
        qvec = _rotmat_to_qvec(world_to_camera[:3, :3])
        tvec = world_to_camera[:3, 3]
        values = " ".join(f"{value:.17g}" for value in (*qvec, *tvec))
        lines.extend([f"{index} {values} 1 {row['camera_id']}.png", ""])
    images_path = sparse / "images.txt"
    images_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    points_path = sparse / "points3D.txt"
    points_path.write_text("# empty: points3D.ply is materialized directly\n", encoding="utf-8")
    return [cameras_path, images_path, points_path]


def _write_points_ply(standard_splat: Path, output: Path) -> Path:
    splat = read_standard_3dgs_ply(standard_splat)
    rgb = np.clip(0.5 + SH_C0 * np.asarray(splat.f_dc), 0.0, 1.0)
    rows = np.empty(
        splat.count,
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    rows["x"], rows["y"], rows["z"] = splat.xyz.T
    rows["nx"], rows["ny"], rows["nz"] = 0.0, 0.0, 0.0
    colors = np.rint(rgb * 255.0).astype(np.uint8)
    rows["red"], rows["green"], rows["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {splat.count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(rows.tobytes())
    return output


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != _sha256(source):
            raise AuraAdapterError(["aurafusion360_existing_output_conflict"])
        return
    shutil.copyfile(source, destination)


def _write_config(path: Path, rows: Sequence[tuple[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"{key} = {json.dumps(value)}" for key, value in rows) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise AuraAdapterError(["aurafusion360_existing_output_conflict"])
    path.write_text(text, encoding="utf-8")
    return path


def materialize_aura_adapter(
    *,
    input_receipt_path: str | Path,
    input_root: str | Path,
    method_prerequisite_receipt_path: str | Path,
    author_smoke_receipt_path: str | Path,
    repo_root: str | Path,
    data_root: str | Path,
    method_root: str | Path,
    lama_root: str | Path,
    big_lama_archive: str | Path,
    output_root: str | Path,
    receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    input_dir = _under(input_root, data, "aurafusion360_input_root_outside_data")
    output = _under(output_root, data, "aurafusion360_output_root_outside_data")
    input_receipt = _under(
        input_receipt_path, repo, "aurafusion360_input_receipt_outside_repo"
    )
    prerequisite_path = _under(
        method_prerequisite_receipt_path,
        data,
        "aurafusion360_prerequisite_receipt_outside_data",
    )
    smoke_path = _under(
        author_smoke_receipt_path, data, "aurafusion360_author_smoke_outside_data"
    )
    retained = (
        _under(receipt_output, repo, "aurafusion360_receipt_output_outside_repo")
        if receipt_output is not None
        else None
    )

    source = _source_identity(Path(method_root).expanduser().resolve())
    reference_generator = _lama_identity(
        Path(lama_root).expanduser().resolve(),
        _under(big_lama_archive, data, "aurafusion360_lama_checkpoint_outside_data"),
    )
    frozen = _read_object(input_receipt, "aurafusion360_input_receipt_invalid")
    if canonical_digest(frozen, digest_field="receipt_digest") != frozen.get("receipt_digest"):
        raise AuraAdapterError(["aurafusion360_input_receipt_digest_mismatch"])
    if frozen.get("status") != "render_derived_input_packet_materialized":
        raise AuraAdapterError(["aurafusion360_input_packet_not_materialized"])
    scene = frozen.get("scene") or {}
    if not all(
        str(scene.get(key) or "").strip()
        for key in ("publisher_scene_id", "target_instance_id", "target_semantic_label")
    ):
        raise AuraAdapterError(["aurafusion360_scene_or_target_mismatch"])
    if frozen.get("proof_boundaries", {}).get("inpainting_result") is not False:
        raise AuraAdapterError(["aurafusion360_input_claim_boundary_invalid"])

    smoke = _read_object(smoke_path, "aurafusion360_author_smoke_invalid")
    smoke_requirements = {
        "status": "completed",
        "source_commit": AURA_COMMIT,
        "source_tree": AURA_TREE,
        "author_source_modified": False,
        "training_executed": True,
        "render_executed": True,
        "removal_executed": True,
        "sam2_masks_executed": True,
        "inpaint_init_executed": True,
    }
    if any(smoke.get(key) != value for key, value in smoke_requirements.items()):
        raise AuraAdapterError(["aurafusion360_unchanged_author_smoke_not_proven"])

    prerequisite = _read_object(
        prerequisite_path, "aurafusion360_method_prerequisite_receipt_invalid"
    )
    aura_prereq = (prerequisite.get("methods") or {}).get(
        "aurafusion360_quality_challenger"
    ) or {}
    required_ids = {
        "aurafusion360_sam2_hiera_large",
        "aurafusion360_marigold_depth_v1_0",
        "aurafusion360_marigold_agdd_v1_0",
        "aurafusion360_sd2_inpainting_exact_checkpoint",
    }
    observed_ids = {
        str(item.get("artifact_id"))
        for item in aura_prereq.get("remote_snapshots", [])
        if item.get("rights_established") is True
    }
    if not required_ids.issubset(observed_ids) or aura_prereq.get(
        "checkpoint_rights_established"
    ) is not True:
        raise AuraAdapterError(["aurafusion360_checkpoint_rights_incomplete"])

    derived = frozen.get("derived_artifacts") or {}
    cameras_record = derived.get("cameras") or {}
    standard_record = derived.get("standard_splat") or {}
    cameras_path = _under(
        input_dir / str(cameras_record.get("relative_path")),
        input_dir,
        "aurafusion360_camera_path_invalid",
    )
    standard_splat = _under(
        input_dir / str(standard_record.get("relative_path")),
        input_dir,
        "aurafusion360_splat_path_invalid",
    )
    for path, record, code in (
        (cameras_path, cameras_record, "aurafusion360_camera_bytes_changed"),
        (standard_splat, standard_record, "aurafusion360_splat_bytes_changed"),
    ):
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise AuraAdapterError([code])
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    if not isinstance(cameras, list) or len(cameras) != 8:
        raise AuraAdapterError(["aurafusion360_camera_count_invalid"])

    image_records = {str(row["camera_id"]): row for row in derived.get("images", [])}
    mask_records = {str(row["camera_id"]): row for row in derived.get("masks", [])}
    method_scope = evaluate_multiview_mask_fraction_scope(
        method_id="aurafusion360",
        profile_id="aurafusion360_blueprint_small_object_anchor_v1",
        cameras=cameras,
        mask_records=list(mask_records.values()),
        maximum_mask_fraction=AURA_MAXIMUM_ADMITTED_MASK_FRACTION,
        profile_basis=(
            "four_times_qualified_840313_small_object_rehearsal_maximum_mask_fraction"
        ),
    )
    max_mask = max(mask_records.values(), key=lambda row: int(row["masked_pixel_count"]))
    reference_camera_id = str(max_mask["camera_id"])
    if reference_camera_id not in image_records:
        raise AuraAdapterError(["aurafusion360_reference_camera_missing"])
    camera_ids = [str(camera.get("camera_id") or "") for camera in cameras]
    if len(set(camera_ids)) != len(camera_ids) or reference_camera_id not in camera_ids:
        raise AuraAdapterError(["aurafusion360_camera_artifact_join_missing"])
    # Aura's released COLMAP reader sorts CameraInfo rows by image_name before
    # inpaint.py consumes reference_index. Input JSON order is not authoritative.
    runtime_camera_ids = sorted(camera_ids)
    reference_index = runtime_camera_ids.index(reference_camera_id)
    scene_id = str(scene.get("publisher_scene_id") or "")
    target_instance_id = str(scene.get("target_instance_id") or "")
    if not scene_id or not target_instance_id or any(
        token in value for value in (scene_id, target_instance_id) for token in ("/", "..")
    ):
        raise AuraAdapterError(["aurafusion360_scene_or_target_identity_invalid"])
    scene_slug = f"{scene_id}_ins{target_instance_id.removeprefix('ins')}"

    scene_root = output / "data" / "Other-360" / scene_slug
    artifacts: list[dict[str, Any]] = []
    for directory in ("images", "object_masks", "test_images", "reference"):
        (scene_root / directory).mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        camera_id = str(camera["camera_id"])
        if camera_id not in image_records or camera_id not in mask_records:
            raise AuraAdapterError(["aurafusion360_camera_artifact_join_missing"])
        source_image = input_dir / image_records[camera_id]["relative_path"]
        source_mask = input_dir / mask_records[camera_id]["relative_path"]
        for path, record, code in (
            (source_image, image_records[camera_id], "aurafusion360_image_bytes_changed"),
            (source_mask, mask_records[camera_id], "aurafusion360_mask_bytes_changed"),
        ):
            if (
                not path.is_file()
                or path.stat().st_size != record["size_bytes"]
                or _sha256(path) != record["sha256"]
            ):
                raise AuraAdapterError([f"{code}:{camera_id}"])
        image_out = scene_root / "images" / f"{camera_id}.png"
        mask_out = scene_root / "object_masks" / f"{camera_id}.png"
        _copy_exact(source_image, image_out)
        pixels = np.asarray(Image.open(source_mask).convert("L"))
        if not np.any(pixels > 0):
            raise AuraAdapterError([f"aurafusion360_mask_empty:{camera_id}"])
        _copy_exact(source_mask, mask_out)
        artifacts.extend([_record(image_out, output), _record(mask_out, output)])

    artifacts.extend(_record(path, output) for path in _write_colmap(scene_root, cameras))
    points = _write_points_ply(
        standard_splat, scene_root / "sparse" / "0" / "points3D.ply"
    )
    artifacts.append(_record(points, output))

    reference_image = scene_root / "images" / f"{reference_camera_id}.png"
    reference_mask = scene_root / "object_masks" / f"{reference_camera_id}.png"
    lama_input_image = output / "reference_lama_input" / f"{reference_camera_id}.png"
    lama_input_mask = (
        output / "reference_lama_input" / f"{reference_camera_id}_mask.png"
    )
    _copy_exact(reference_image, lama_input_image)
    _copy_exact(reference_mask, lama_input_mask)
    artifacts.extend([_record(lama_input_image, output), _record(lama_input_mask, output)])

    configs = output / "configs" / "Other-360" / scene_slug
    source_path = f"./data/Other-360/{scene_slug}"
    model_path = f"./output/Other-360/{scene_slug}"
    config_paths = [
        _write_config(
            configs / "train.config",
            [
                ("source_path", source_path), ("model_path", model_path),
                ("resolution", 1), ("dilate_mask_kernel_size", 10),
                ("dilate_mask_iter", 1), ("optimize_is_masked_iter", 20000),
                ("is_masked_lr", 0.1), ("is_masked_3d_lr", 0.0),
                ("other_depth_lr", 0.0),
            ],
        ),
        _write_config(
            configs / "remove.config",
            [
                ("source_path", source_path), ("model_path", model_path),
                ("iteration", 30000), ("resolution", 1),
                ("skip_train", False), ("skip_test", True),
                ("skip_mesh", True), ("render_path", False),
                ("removal_thresh", 0.6), ("unseen_thresh", 0.0),
                ("outlier_factor", 1.0), ("aggreagte_threshold", 0.6),
            ],
        ),
        _write_config(
            configs / "inpaint.config",
            [
                ("source_path", source_path), ("model_path", model_path),
                ("iteration", 30000), ("resolution", 1),
                ("skip_train", False), ("skip_test", False),
                ("skip_mesh", True), ("render_path", False),
                ("dilate_mask_kernel_size", 5), ("dilate_mask_iter", 3),
                ("finetune_iteration", -1), ("reference_index", reference_index),
                ("dilate_iter", 5), ("kernel_size", 3),
                ("optimize_iter", 8), ("delta", 0.3),
                ("infer_iter", 4), ("densify_until_iter", 1000),
            ],
        ),
        _write_config(
            configs / "sdedit.config",
            [
                ("dataset", "Other-360"), ("scene", scene_slug),
                ("script", "sdedit"), ("use_ddim_inversion", True),
                ("strength", 0.85), ("eta", 1.0), ("scale", 2.5),
            ],
        ),
    ]
    artifacts.extend(_record(path, output) for path in config_paths)

    commands = {
        "reference_generation": [
            "python", "bin/predict.py", "model.path=./big-lama",
            "indir=./reference_lama_input", "outdir=./reference_lama_output",
        ],
        "reference_publish": {
            "from": f"reference_lama_output/{reference_camera_id}_mask.png",
            "to": f"data/Other-360/{scene_slug}/reference/{reference_camera_id}.png",
            "required_nonzero_mask_pixels": int(max_mask["masked_pixel_count"]),
        },
        "author_workflow": [
            ["python", "train.py", "--config", f"configs/Other-360/{scene_slug}/train.config"],
            ["python", "render.py", "-s", source_path, "-m", model_path,
             "--skip_mesh", "--iteration", "30000", "--resolution", "1"],
            ["python", "remove.py", "--config", f"configs/Other-360/{scene_slug}/remove.config"],
            ["python", "utils/sam2_utils.py", "--dataset", "Other-360", "--scene", scene_slug],
            ["python", "inpaint.py", "--config", f"configs/Other-360/{scene_slug}/inpaint.config"],
            ["python", "utils/LeftRefill/sdedit_utils.py", "--config", f"configs/Other-360/{scene_slug}/sdedit.config"],
            ["python", "inpaint.py", "--config", f"configs/Other-360/{scene_slug}/inpaint.config",
             "--images", "inpaint", "--finetune_iteration", "10000"],
        ],
    }
    spec_path = output / "aurafusion360_interiorgs_execution_spec.json"
    spec_path.write_text(canonical_json(commands) + "\n", encoding="utf-8")
    artifacts.append(_record(spec_path, output))

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "status": "prepared_unexecuted",
        "scene": {
            "publisher_scene_id": scene_id,
            "target_instance_id": f"ins{target_instance_id.removeprefix('ins')}",
            "target_semantic_label": scene.get("target_semantic_label"),
            "input_receipt_digest": frozen["receipt_digest"],
            "camera_count": len(cameras),
            "source_resolution": [2048, 1536],
            "method_resolution_divisor": 1,
            "reference_camera_id": reference_camera_id,
            "reference_camera_selection": "maximum_frozen_target_mask_pixel_count",
            "reference_masked_pixel_count": int(max_mask["masked_pixel_count"]),
            "reference_camera_index": reference_index,
            "runtime_camera_order": runtime_camera_ids,
            "runtime_camera_order_derivation": (
                "released_aura_colmap_reader_sorted_by_image_name"
            ),
            "scene_slug": scene_slug,
        },
        "source": source,
        "reference_generator": reference_generator,
        "method_prerequisite_receipt": {
            "path": str(prerequisite_path),
            "sha256": _sha256(prerequisite_path),
            "size_bytes": prerequisite_path.stat().st_size,
        },
        "unchanged_author_smoke": {
            "path": str(smoke_path),
            "sha256": _sha256(smoke_path),
            "size_bytes": smoke_path.stat().st_size,
            "completed": True,
        },
        "adapter": {
            "dataset_family": "Other-360",
            "profile_basis": "publisher_Other-360_kitchen_defaults_frozen_before_scene_outcome",
            "camera_contract": "COLMAP_PINHOLE_text_from_T_world_camera_opencv",
            "publisher_splat_use": "XYZ_and_SH0_color_seed_only_then_Aura_retraining",
            "publisher_ply_edited_in_place": False,
            "world_frame_preserved_in_seed_and_cameras": True,
            "reference_contract": "Big-LaMa_single_view_candidate_then_Aura_native_pipeline",
            "full_source_resolution_preserved": True,
            "trajectory_media_contract": {
                "generated": False,
                "retained": False,
                "reason": "unretained_240_frame_trajectory_is_not_evaluation_evidence",
            },
            "method_scope_admission": method_scope,
        },
        "artifacts": sorted(artifacts, key=lambda row: row["relative_path"]),
        "commands": commands,
        "execution": {
            "reference_generation_executed": False,
            "aurafusion360_interiorgs_executed": False,
            "result_ply_observed": False,
            "quality_metrics_observed": False,
        },
        "claim_boundary": {
            "prepared_packet_is_not_method_execution": True,
            "author_smoke_is_not_interiorgs_execution": True,
            "inpainting_result": False,
            "hidden_background_truth_available": False,
            "source_object_removed_from_appearance": False,
            "source_collider_removed": False,
            "simready_replacement_inserted": False,
            "output_claim_ceiling_after_execution": "visual_candidate_only",
        },
        "blockers": sorted(
            {
                "aurafusion360_interiorgs_gpu_execution_missing",
                *method_scope["blockers"],
            }
        ),
        "raw_secret_values_recorded": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output.mkdir(parents=True, exist_ok=True)
    output_receipt = output / "adp009b_aurafusion360_adapter_receipt.v1.json"
    output_receipt.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    if retained is not None:
        retained.parent.mkdir(parents=True, exist_ok=True)
        retained.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-receipt", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--method-prerequisite-receipt", required=True)
    parser.add_argument("--author-smoke-receipt", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--lama-root", required=True)
    parser.add_argument("--big-lama-archive", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--receipt-output")
    args = parser.parse_args()
    receipt = materialize_aura_adapter(
        input_receipt_path=args.input_receipt,
        input_root=args.input_root,
        method_prerequisite_receipt_path=args.method_prerequisite_receipt,
        author_smoke_receipt_path=args.author_smoke_receipt,
        repo_root=args.repo_root,
        data_root=args.data_root,
        method_root=args.method_root,
        lama_root=args.lama_root,
        big_lama_archive=args.big_lama_archive,
        output_root=args.output_root,
        receipt_output=args.receipt_output,
    )
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
