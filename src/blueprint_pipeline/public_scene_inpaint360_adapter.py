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
import re
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


def _tracked_repository_file(repo: Path, path: Path, code: str) -> str:
    """Return a clean-repository-relative path only for a tracked binding."""

    try:
        relative = path.relative_to(repo).as_posix()
    except ValueError as exc:
        raise Inpaint360AdapterError([code]) from exc
    try:
        subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", relative],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise Inpaint360AdapterError([code]) from exc
    return relative


def _external_input_receipt_binding(
    *,
    receipt_path: Path,
    repo: Path,
    data: Path,
    binding_path: str | Path | None,
) -> dict[str, Any]:
    """Authorize an evidence-root receipt through a tracked digest binding.

    Production render packets are deliberately retained outside the source
    checkout.  This prevents a caller from substituting an arbitrary local
    receipt merely because it is self-digesting: the source checkout must
    contain a clean, tracked binding to the exact evidence-root file.
    """

    if binding_path is None:
        raise Inpaint360AdapterError(
            ["inpaint360_external_input_receipt_binding_required"]
        )
    binding_file = _under(
        binding_path,
        repo,
        "inpaint360_input_receipt_binding_outside_repo",
    )
    relative_binding_path = _tracked_repository_file(
        repo,
        binding_file,
        "inpaint360_input_receipt_binding_not_tracked",
    )
    binding = _read_object(
        binding_file, "inpaint360_input_receipt_binding_invalid"
    )
    if (
        not isinstance(binding.get("schema_version"), str)
        or not binding["schema_version"]
        or canonical_digest(binding, digest_field="binding_digest")
        != binding.get("binding_digest")
    ):
        raise Inpaint360AdapterError(
            ["inpaint360_input_receipt_binding_digest_mismatch"]
        )
    record = binding.get("render_input_receipt")
    if not isinstance(record, Mapping):
        raise Inpaint360AdapterError(["inpaint360_input_receipt_binding_invalid"])
    relative_path = record.get("path")
    expected_file_sha256 = record.get("file_sha256")
    expected_receipt_digest = record.get("receipt_digest")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or not isinstance(expected_file_sha256, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_file_sha256)
        or not isinstance(expected_receipt_digest, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_receipt_digest)
    ):
        raise Inpaint360AdapterError(["inpaint360_input_receipt_binding_invalid"])
    expected_receipt_path = _under(
        data / relative_path,
        data,
        "inpaint360_input_receipt_binding_path_invalid",
    )
    if expected_receipt_path != receipt_path:
        raise Inpaint360AdapterError(
            ["inpaint360_input_receipt_binding_path_mismatch"]
        )
    if not receipt_path.is_file() or _sha256(receipt_path) != expected_file_sha256:
        raise Inpaint360AdapterError(
            ["inpaint360_input_receipt_binding_file_bytes_changed"]
        )
    return {
        "schema_version": binding["schema_version"],
        "binding_path": relative_binding_path,
        "binding_digest": binding["binding_digest"],
        "input_receipt_file_sha256": expected_file_sha256,
        "input_receipt_receipt_digest": expected_receipt_digest,
        "task_id": binding.get("task_id"),
        "task_freeze_digest": binding.get("task_freeze_digest"),
    }


def _resolve_input_artifact(
    record: Mapping[str, Any],
    *,
    input_root: Path,
    data_root: Path,
    code: str,
) -> tuple[Path, str]:
    """Resolve one receipt-bound artifact without assuming a single root.

    A task packet commonly owns cameras, source frames, and masks, while a
    shared scene conversion owns the standard splat.  The receipt's digest is
    authoritative; a record may optionally declare ``input_root`` or
    ``data_root``.  Legacy records without that field remain valid only when
    exactly one permitted root contains the exact recorded bytes.
    """

    if not isinstance(record, Mapping):
        raise Inpaint360AdapterError([f"{code}_record_invalid"])
    relative_path = record.get("relative_path")
    expected_size = record.get("size_bytes")
    expected_sha256 = record.get("sha256")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
        or not isinstance(expected_sha256, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_sha256)
    ):
        raise Inpaint360AdapterError([f"{code}_record_invalid"])
    declared_root = record.get("root")
    roots = {
        "input_root": input_root,
        "data_root": data_root,
    }
    if declared_root is not None:
        if declared_root not in roots:
            raise Inpaint360AdapterError([f"{code}_root_invalid"])
        candidates = [(str(declared_root), roots[str(declared_root)])]
    else:
        candidates = list(roots.items())
    matching: list[tuple[Path, str]] = []
    for root_name, root in candidates:
        candidate = _under(root / relative_path, root, f"{code}_path_invalid")
        if (
            candidate.is_file()
            and candidate.stat().st_size == expected_size
            and _sha256(candidate) == expected_sha256
        ):
            matching.append((candidate, root_name))
    if not matching:
        raise Inpaint360AdapterError([f"{code}_bytes_changed"])
    if len(matching) != 1:
        raise Inpaint360AdapterError([f"{code}_root_ambiguous"])
    return matching[0]


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


def _target_object_radius_m(scene: Mapping[str, Any]) -> float:
    """Derive the publisher target's bounding-sphere radius from its metric OBB."""

    corners = np.asarray(scene.get("target_obb_corners_m"), dtype=np.float64)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise Inpaint360AdapterError(["inpaint360_target_metric_obb_missing"])
    center = np.mean(corners, axis=0)
    radius = float(np.max(np.linalg.norm(corners - center, axis=1)))
    if not np.isfinite(radius) or radius <= 0.0:
        raise Inpaint360AdapterError(["inpaint360_target_metric_obb_degenerate"])
    return radius


def _method_config_id(scene: Mapping[str, Any]) -> tuple[str, str | None]:
    """Return a safe, task-bound upstream config stem.

    Old single-target packets did not retain a task ID and continue to use the
    publisher scene ID as their config stem.  New packets bind scene, target,
    and task so two removals from one scene cannot silently share an upstream
    config file.
    """

    scene_id = str(scene.get("publisher_scene_id") or "")
    target_instance_id = str(scene.get("target_instance_id") or "")
    task_value = scene.get("task_id")
    task_id = str(task_value or "") or None
    if not scene_id.isdigit() or not target_instance_id.isdigit():
        raise Inpaint360AdapterError(["inpaint360_scene_target_identity_invalid"])
    if task_id is None:
        return scene_id, None
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", task_id):
        raise Inpaint360AdapterError(["inpaint360_task_id_invalid"])
    return f"{scene_id}__{target_instance_id}__{task_id}", task_id


def materialize_inpaint360_adapter(
    *,
    input_receipt_path: str | Path,
    input_receipt_binding_path: str | Path | None = None,
    input_root: str | Path,
    repo_root: str | Path,
    data_root: str | Path,
    method_root: str | Path,
    output_root: str | Path,
    receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    input_dir = _under(input_root, data, "inpaint360_input_root_outside_data")
    output = _under(output_root, data, "inpaint360_output_root_outside_data")
    retained_receipt = (
        _under(receipt_output, repo, "inpaint360_receipt_output_outside_repo")
        if receipt_output is not None
        else None
    )
    requested_receipt_path = Path(input_receipt_path).expanduser().resolve()
    if requested_receipt_path == repo or repo in requested_receipt_path.parents:
        receipt_path = requested_receipt_path
        input_receipt_binding = None
    else:
        receipt_path = _under(
            requested_receipt_path,
            data,
            "inpaint360_input_receipt_outside_data",
        )
        input_receipt_binding = _external_input_receipt_binding(
            receipt_path=receipt_path,
            repo=repo,
            data=data,
            binding_path=input_receipt_binding_path,
        )
    receipt = _read_object(receipt_path, "inpaint360_input_receipt_invalid")
    if canonical_digest(receipt, digest_field="receipt_digest") != receipt.get("receipt_digest"):
        raise Inpaint360AdapterError(["inpaint360_input_receipt_digest_mismatch"])
    if input_receipt_binding is not None:
        if receipt["receipt_digest"] != input_receipt_binding["input_receipt_receipt_digest"]:
            raise Inpaint360AdapterError(
                ["inpaint360_input_receipt_binding_receipt_digest_mismatch"]
            )
        scene = receipt.get("scene") or {}
        task_id = str(scene.get("task_id") or "") or None
        if input_receipt_binding["task_id"] != task_id:
            raise Inpaint360AdapterError(
                ["inpaint360_input_receipt_binding_task_identity_mismatch"]
            )
        expected_freeze = input_receipt_binding["task_freeze_digest"]
        actual_freeze = (receipt.get("source_admission") or {}).get(
            "task_freeze_digest"
        )
        if expected_freeze is not None and expected_freeze != actual_freeze:
            raise Inpaint360AdapterError(
                ["inpaint360_input_receipt_binding_freeze_digest_mismatch"]
            )
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
    cameras_path, cameras_root = _resolve_input_artifact(
        cameras_record,
        input_root=input_dir,
        data_root=data,
        code="inpaint360_camera",
    )
    standard_splat, standard_splat_root = _resolve_input_artifact(
        standard_record,
        input_root=input_dir,
        data_root=data,
        code="inpaint360_splat",
    )
    input_artifact_bindings: list[dict[str, Any]] = [
        {
            "role": "cameras",
            "root": cameras_root,
            "relative_path": cameras_record["relative_path"],
            "sha256": cameras_record["sha256"],
            "size_bytes": cameras_record["size_bytes"],
        },
        {
            "role": "standard_splat",
            "root": standard_splat_root,
            "relative_path": standard_record["relative_path"],
            "sha256": standard_record["sha256"],
            "size_bytes": standard_record["size_bytes"],
        },
    ]
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    if not isinstance(cameras, list) or len(cameras) != 8:
        raise Inpaint360AdapterError(["inpaint360_camera_count_invalid"])

    scene = receipt.get("scene") or {}
    method_config_id, task_id = _method_config_id(scene)
    target_object_radius_m = _target_object_radius_m(scene)
    target_obb_corners_m = [
        [float(value) for value in row]
        for row in scene["target_obb_corners_m"]
    ]
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
        source_image, source_image_root = _resolve_input_artifact(
            image_records[camera_id],
            input_root=input_dir,
            data_root=data,
            code="inpaint360_image",
        )
        source_mask, source_mask_root = _resolve_input_artifact(
            mask_records[camera_id],
            input_root=input_dir,
            data_root=data,
            code="inpaint360_mask",
        )
        input_artifact_bindings.extend(
            [
                {
                    "role": "image",
                    "camera_id": camera_id,
                    "root": source_image_root,
                    "relative_path": image_records[camera_id]["relative_path"],
                    "sha256": image_records[camera_id]["sha256"],
                    "size_bytes": image_records[camera_id]["size_bytes"],
                },
                {
                    "role": "mask",
                    "camera_id": camera_id,
                    "root": source_mask_root,
                    "relative_path": mask_records[camera_id]["relative_path"],
                    "sha256": mask_records[camera_id]["sha256"],
                    "size_bytes": mask_records[camera_id]["size_bytes"],
                },
            ]
        )
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
    removal_config = (
        output
        / "config"
        / "object_removal"
        / "blueprint"
        / f"{method_config_id}.json"
    )
    removal_config.parent.mkdir(parents=True, exist_ok=True)
    removal_config.write_text(
        canonical_json(
            {
                "removal_thresh": 0.7,
                "select_obj_id": [1],
                "surrounding_ids": [],
                "target_object_radius": target_object_radius_m,
                "target_id": [1],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    inpaint_config = (
        output
        / "config"
        / "object_inpaint"
        / "blueprint"
        / f"{method_config_id}.json"
    )
    inpaint_config.parent.mkdir(parents=True, exist_ok=True)
    inpaint_config.write_text(
        canonical_json(
            {
                "finetune_iteration": 5000,
                "images": "images_inpaint_unseen_virtual",
                "lambda_dssim": 0.8,
                "lambda_lpips": 0.0005,
                "object_path": "inpaint_2d_unseen_mask_virtual",
                "opacity_init": 0.1,
                "removal_thresh": 0.7,
                "select_obj_id": [1],
                "surrounding_ids": [],
                "target_object_radius": target_object_radius_m,
                "target_id": [1],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.extend(
        [
            _record(distill_config, output),
            _record(removal_config, output),
            _record(inpaint_config, output),
        ]
    )

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
        "input_receipt_binding": input_receipt_binding,
        "scene": receipt["scene"],
        "source": source_identity,
        "adapter_repository": adapter_repository,
        "adapter": {
            "camera_contract": "COLMAP_PINHOLE_text_from_T_world_camera_opencv",
            "mask_contract": "raw_hqsam_uint8_instance_id_1",
            "vanilla_gaussian_iteration": 30000,
            "method_config_id": method_config_id,
            "task_id": task_id,
            "target_method_instance_id": 1,
            "target_object_radius_m": target_object_radius_m,
            "target_object_radius_derivation": "max_distance_from_metric_obb_center",
            "target_obb_corners_m": target_obb_corners_m,
            "target_removal_volume_contract": "gaussian_center_inside_exact_publisher_obb",
            "paired_config_contract": (
                f"config/object_removal/blueprint/{method_config_id}.json_to_"
                f"config/object_inpaint/blueprint/{method_config_id}.json"
            ),
            "input_artifact_bindings": input_artifact_bindings,
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
    if retained_receipt is not None:
        retained_receipt.parent.mkdir(parents=True, exist_ok=True)
        retained_receipt.write_text(canonical_json(adapter_receipt) + "\n", encoding="utf-8")
    return adapter_receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-receipt", required=True)
    parser.add_argument("--input-receipt-binding")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--receipt-output")
    args = parser.parse_args(argv)
    receipt = materialize_inpaint360_adapter(
        input_receipt_path=args.input_receipt,
        input_receipt_binding_path=args.input_receipt_binding,
        input_root=args.input_root,
        repo_root=args.repo_root,
        data_root=args.data_root,
        method_root=args.method_root,
        output_root=args.output_root,
        receipt_output=args.receipt_output,
    )
    print(canonical_json({"status": receipt["status"], "receipt_digest": receipt["receipt_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
