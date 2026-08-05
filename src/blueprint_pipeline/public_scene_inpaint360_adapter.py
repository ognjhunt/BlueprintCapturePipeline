"""Stage a digest-bound InteriorGS packet for unchanged Inpaint360GS source.

The adapter translates Blueprint's exact rendered cameras and binary target
masks into the COLMAP and instance-mask layout consumed by Inpaint360GS. It
also seeds the author's vanilla-Gaussian checkpoint layout with the decoded
publisher splat. Staging is not method execution or an inpainting result.
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

SCHEMA_VERSION = "adp009b_inpaint360_adapter_receipt.v1"
INPAINT360_COMMIT = "d54c893285c6cb27788e05cce607e7d3cca6388a"
SH_C0 = 0.28209479177387814


class Inpaint360AdapterError(ValueError):
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
        raise Inpaint360AdapterError([code])
    return resolved


def _read_object(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Inpaint360AdapterError([code]) from exc
    if not isinstance(value, dict):
        raise Inpaint360AdapterError([code])
    return value


def _git_identity(method_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(method_root), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise Inpaint360AdapterError(["inpaint360_source_identity_unavailable"]) from exc

    commit = run("rev-parse", "HEAD")
    if commit != INPAINT360_COMMIT:
        raise Inpaint360AdapterError(["inpaint360_source_revision_mismatch"])
    if run("status", "--porcelain", "--untracked-files=no"):
        raise Inpaint360AdapterError(["inpaint360_source_tracked_files_dirty"])
    return {
        "repository": "https://github.com/dfki-av/Inpaint360GS",
        "commit": commit,
        "tree": run("rev-parse", "HEAD^{tree}"),
        "tracked_files_clean": True,
        "source_modified_for_adapter": False,
    }


def _adapter_repository_identity(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise Inpaint360AdapterError(["inpaint360_adapter_repository_identity_unavailable"]) from exc

    if run("status", "--porcelain", "--untracked-files=no"):
        raise Inpaint360AdapterError(["inpaint360_adapter_repository_tracked_files_dirty"])
    return {
        "commit": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
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


def _write_colmap(source: Path, cameras: Sequence[Mapping[str, Any]]) -> list[Path]:
    sparse = source / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    first = cameras[0]["intrinsics"]
    camera_text = (
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {first['width']} {first['height']} "
        f"{first['fx']:.17g} {first['fy']:.17g} {first['cx']:.17g} {first['cy']:.17g}\n"
    )
    cameras_path = sparse / "cameras.txt"
    cameras_path.write_text(camera_text, encoding="utf-8")
    lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    for index, row in enumerate(cameras, start=1):
        pose = np.asarray(row["T_world_camera_opencv"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise Inpaint360AdapterError(["inpaint360_camera_pose_invalid"])
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
            raise Inpaint360AdapterError(["inpaint360_existing_output_conflict"])
        return
    shutil.copyfile(source, destination)


def materialize_inpaint360_adapter(
    *,
    input_receipt_path: str | Path,
    input_root: str | Path,
    repo_root: str | Path,
    data_root: str | Path,
    method_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    input_dir = _under(input_root, data, "inpaint360_input_root_outside_data")
    output = _under(output_root, data, "inpaint360_output_root_outside_data")
    receipt_path = _under(input_receipt_path, repo, "inpaint360_input_receipt_outside_repo")
    receipt = _read_object(receipt_path, "inpaint360_input_receipt_invalid")
    if canonical_digest(receipt, digest_field="receipt_digest") != receipt.get("receipt_digest"):
        raise Inpaint360AdapterError(["inpaint360_input_receipt_digest_mismatch"])
    if receipt.get("status") != "render_derived_input_packet_materialized":
        raise Inpaint360AdapterError(["inpaint360_input_packet_not_materialized"])
    if receipt.get("proof_boundaries", {}).get("inpainting_result") is not False:
        raise Inpaint360AdapterError(["inpaint360_input_claim_boundary_invalid"])
    method = Path(method_root).expanduser().resolve()
    source_identity = _git_identity(method)
    adapter_repository = _adapter_repository_identity(repo)

    derived = receipt.get("derived_artifacts") or {}
    cameras_record = derived.get("cameras") or {}
    standard_record = derived.get("standard_splat") or {}
    cameras_path = _under(input_dir / str(cameras_record.get("relative_path")), input_dir, "inpaint360_camera_path_invalid")
    standard_splat = _under(input_dir / str(standard_record.get("relative_path")), input_dir, "inpaint360_splat_path_invalid")
    for path, record, code in (
        (cameras_path, cameras_record, "inpaint360_camera_bytes_changed"),
        (standard_splat, standard_record, "inpaint360_splat_bytes_changed"),
    ):
        if not path.is_file() or path.stat().st_size != record.get("size_bytes") or _sha256(path) != record.get("sha256"):
            raise Inpaint360AdapterError([code])
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    if not isinstance(cameras, list) or len(cameras) != 8:
        raise Inpaint360AdapterError(["inpaint360_camera_count_invalid"])

    source_dir = output / "source"
    vanilla_dir = output / "vanilla_3dgs"
    model_dir = output / "inpaint360_model"
    artifacts: list[dict[str, Any]] = []
    image_records = {row["camera_id"]: row for row in derived.get("images", [])}
    mask_records = {row["camera_id"]: row for row in derived.get("masks", [])}
    for camera in cameras:
        camera_id = str(camera["camera_id"])
        if camera_id not in image_records or camera_id not in mask_records:
            raise Inpaint360AdapterError(["inpaint360_camera_artifact_join_missing"])
        source_image = input_dir / image_records[camera_id]["relative_path"]
        source_mask = input_dir / mask_records[camera_id]["relative_path"]
        for path, record, code in (
            (source_image, image_records[camera_id], "inpaint360_image_bytes_changed"),
            (source_mask, mask_records[camera_id], "inpaint360_mask_bytes_changed"),
        ):
            if not path.is_file() or path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
                raise Inpaint360AdapterError([f"{code}:{camera_id}"])
        image_out = source_dir / "images" / f"{camera_id}.png"
        _copy_exact(source_image, image_out)
        mask_pixels = np.asarray(Image.open(source_mask).convert("L"))
        instance_pixels = (mask_pixels > 0).astype(np.uint8)
        if int(instance_pixels.sum()) == 0:
            raise Inpaint360AdapterError([f"inpaint360_mask_empty:{camera_id}"])
        mask_out = source_dir / "raw_hqsam" / f"{camera_id}.png"
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        if mask_out.exists():
            existing = np.asarray(Image.open(mask_out).convert("L"))
            if not np.array_equal(existing, instance_pixels):
                raise Inpaint360AdapterError(["inpaint360_existing_output_conflict"])
        else:
            Image.fromarray(instance_pixels, mode="L").save(mask_out, format="PNG", optimize=False)
        artifacts.extend([_record(image_out, output), _record(mask_out, output)])

    artifacts.extend(_record(path, output) for path in _write_colmap(source_dir, cameras))
    points_ply = source_dir / "sparse" / "0" / "points3D.ply"
    if not points_ply.exists():
        _write_points_ply(standard_splat, points_ply)
    artifacts.append(_record(points_ply, output))
    checkpoint = vanilla_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    _copy_exact(standard_splat, checkpoint)
    artifacts.append(_record(checkpoint, output))
    vanilla_cfg = vanilla_dir / "cfg_args"
    vanilla_cfg.write_text("Namespace()\n", encoding="utf-8")
    artifacts.append(_record(vanilla_cfg, output))

    distill_config = output / "config" / "distill.json"
    distill_config.parent.mkdir(parents=True, exist_ok=True)
    distill_config.write_text(
        canonical_json(
            {
                "iterations": 2000,
                "train_distill": True,
                "reg3d_interval": 50,
                "reg3d_k": 5,
                "reg3d_lambda_val": 2,
                "reg3d_max_points": 300000,
                "reg3d_sample_size": 1000,
            }
        ) + "\n",
        encoding="utf-8",
    )
    removal_config = output / "config" / "removal.json"
    removal_config.write_text(
        canonical_json({"removal_thresh": 0.7, "select_obj_id": [1], "target_id": [1]}) + "\n",
        encoding="utf-8",
    )
    artifacts.extend([_record(distill_config, output), _record(removal_config, output)])

    commands = [
        [
            "python", "seg/mask_associate.py", "--source_path", str(source_dir),
            "--model_path", str(vanilla_dir), "--images", "images", "--iteration", "30000",
            "--mask_generator", "hqsam", "--resolution", "1",
        ],
        [
            "python", "seg/distillation.py", "--source_path", str(source_dir),
            "--model_path", str(model_dir), "--vanilla_3dgs_path", str(vanilla_dir),
            "--images", "images", "--object_path", "associated_hqsam", "--eval",
            "--config_file", str(distill_config), "--save_iterations", "2000",
        ],
        [
            "python", "edit_object_removal.py", "--source_path", str(source_dir),
            "--model_path", str(model_dir), "--images", "images", "--iteration", "2000",
            "--config_file", str(removal_config), "--skip_test",
        ],
    ]
    adapter_receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "status": "prepared_unexecuted",
        "input_receipt_digest": receipt["receipt_digest"],
        "scene": receipt["scene"],
        "source": source_identity,
        "adapter_repository": adapter_repository,
        "adapter": {
            "camera_contract": "COLMAP_PINHOLE_text_from_T_world_camera_opencv",
            "mask_contract": "raw_hqsam_uint8_instance_id_1",
            "vanilla_gaussian_iteration": 30000,
            "target_method_instance_id": 1,
            "upstream_cfg_args_materialized": True,
            "staged_artifacts": artifacts,
        },
        "prepared_commands": {"working_directory": str(method), "commands": commands},
        "execution": {
            "author_method_executed": False,
            "gpu_allocated": False,
            "source_modified": False,
        },
        "proof_boundaries": {
            "prepared_job_is_not_method_execution": True,
            "prepared_job_is_not_inpainting_result": True,
            "rendered_frames_have_no_hidden_background_truth": True,
            "publisher_splat_is_not_metric_surface_truth": True,
        },
        "smallest_blocker": "inpaint360_unchanged_author_smoke_and_cuda_runtime_required",
        "claim_ceiling": "inpaint360_interiorgs_adapter_prepared_unexecuted",
    }
    adapter_receipt["receipt_digest"] = canonical_digest(
        adapter_receipt, digest_field="receipt_digest"
    )
    receipt_out = output / "adp009b_inpaint360_adapter_receipt.v1.json"
    receipt_out.write_text(canonical_json(adapter_receipt) + "\n", encoding="utf-8")
    return adapter_receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-receipt", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    receipt = materialize_inpaint360_adapter(
        input_receipt_path=args.input_receipt,
        input_root=args.input_root,
        repo_root=args.repo_root,
        data_root=args.data_root,
        method_root=args.method_root,
        output_root=args.output_root,
    )
    print(canonical_json({"status": receipt["status"], "receipt_digest": receipt["receipt_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
