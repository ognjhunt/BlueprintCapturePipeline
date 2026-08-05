"""Materialize the frozen InteriorGS-to-InFusion execution packet.

This adapter derives every scene input from the retained ADP-009B render packet,
selects one view by a preregistered coverage rule, and removes OBB-contained
publisher Gaussians without observing completion outcomes.  It stays blocked
until the exact InFusion checkpoint has a rights authority and materialized
bytes; caller assertions cannot turn the packet into method execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .gaussian_splat_decode import SplatData, read_standard_3dgs_ply, write_standard_3dgs_ply

SCHEMA_VERSION = "adp009b_infusion_adapter_receipt.v1"
INFUSION_COMMIT = "788da7f40cad4314831a053b7419df277d7814c4"
INFUSION_TREE = "39f437f353114f7958b28a3a5b816d37ed2936a5"
INPAINT360_COMMIT = "d54c893285c6cb27788e05cce607e7d3cca6388a"
INPAINT360_TREE = "671626f4825cbf3d7c1ca37cc97a153d45e49b1c"
BIG_LAMA_SHA256 = "sha256:d7161bba4d68b438f9fa7f09dcb750a223804c300c68d214a5e0be16251fba8d"
INFUSION_CHECKPOINT_REVISION = "83c7eb648bb8e01ae27d4dfbc12638ca564bfcf2"
SH_C0 = 0.28209479177387814


class InFusionAdapterError(ValueError):
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
        raise InFusionAdapterError([code])
    return resolved


def _object(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InFusionAdapterError([code]) from exc
    if not isinstance(value, dict):
        raise InFusionAdapterError([code])
    return value


def _verify_digest(value: Mapping[str, Any], field: str, code: str) -> None:
    if value.get(field) != canonical_digest(value, digest_field=field):
        raise InFusionAdapterError([code])


def _git_identity(root: Path, *, commit: str, tree: str, label: str) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise InFusionAdapterError([f"{label}_source_identity_unavailable"]) from exc

    observed_commit = run("rev-parse", "HEAD")
    observed_tree = run("rev-parse", "HEAD^{tree}")
    if observed_commit != commit or observed_tree != tree:
        raise InFusionAdapterError([f"{label}_source_identity_mismatch"])
    if run("status", "--porcelain", "--untracked-files=no"):
        raise InFusionAdapterError([f"{label}_source_tracked_files_dirty"])
    return {
        "commit": observed_commit,
        "tree": observed_tree,
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
            raise InFusionAdapterError(["infusion_adapter_repository_identity_unavailable"]) from exc

    if run("status", "--porcelain", "--untracked-files=no"):
        raise InFusionAdapterError(["infusion_adapter_repository_tracked_files_dirty"])
    implementation_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("public_scene_infusion_compose.py").resolve(),
    ]
    if any(repo not in path.parents or not path.is_file() for path in implementation_paths):
        raise InFusionAdapterError(["infusion_adapter_implementation_missing"])
    return {
        "commit": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "tracked_files_clean": True,
        "implementation_files": [_record(path, repo) for path in implementation_paths],
    }


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != _sha256(source):
            raise InFusionAdapterError(["infusion_existing_output_conflict"])
        return
    shutil.copyfile(source, destination)


def _write_exact(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != payload:
            raise InFusionAdapterError(["infusion_existing_output_conflict"])
        return
    path.write_text(payload, encoding="utf-8")


def _atomic_splat(splat: SplatData, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".ply", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        write_standard_3dgs_ply(splat, temporary)
        if destination.exists():
            if not destination.is_file() or _sha256(destination) != _sha256(temporary):
                raise InFusionAdapterError(["infusion_existing_output_conflict"])
            return
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _obb_mask(points: np.ndarray, corners: Sequence[Sequence[float]]) -> np.ndarray:
    box = np.asarray(corners, dtype=np.float64)
    if box.shape != (8, 3) or not np.isfinite(box).all():
        raise InFusionAdapterError(["infusion_target_obb_invalid"])
    origin = box[0]
    vectors = box[1:] - origin
    candidates = sorted(vectors, key=lambda row: float(np.linalg.norm(row)))
    basis: np.ndarray | None = None
    for first_index, first in enumerate(candidates):
        for second_index, second in enumerate(candidates[first_index + 1 :], start=first_index + 1):
            for third in candidates[second_index + 1 :]:
                trial = np.column_stack((first, second, third))
                if abs(float(np.linalg.det(trial))) > 1e-10:
                    coordinates = np.linalg.solve(trial, vectors.T).T
                    if np.all((coordinates >= -1e-6) & (coordinates <= 1.0 + 1e-6)):
                        basis = trial
                        break
            if basis is not None:
                break
        if basis is not None:
            break
    if basis is None:
        raise InFusionAdapterError(["infusion_target_obb_basis_unresolved"])
    local = np.linalg.solve(basis, (points.astype(np.float64) - origin).T).T
    return np.all((local >= -1e-6) & (local <= 1.0 + 1e-6), axis=1)


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
    result = vectors[[3, 0, 1, 2], int(np.argmax(values))]
    if result[0] < 0:
        result *= -1
    return result


def _write_colmap(source: Path, cameras: Sequence[Mapping[str, Any]]) -> list[Path]:
    sparse = source / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    first = cameras[0]["intrinsics"]
    camera_path = sparse / "cameras.txt"
    _write_exact(
        camera_path,
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {first['width']} {first['height']} {first['fx']:.17g} "
        f"{first['fy']:.17g} {first['cx']:.17g} {first['cy']:.17g}\n",
    )
    lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    for index, row in enumerate(cameras, start=1):
        pose = np.asarray(row["T_world_camera_opencv"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise InFusionAdapterError(["infusion_camera_pose_invalid"])
        world_to_camera = np.linalg.inv(pose)
        values = (*_rotmat_to_qvec(world_to_camera[:3, :3]), *world_to_camera[:3, 3])
        lines.extend(
            [f"{index} {' '.join(f'{value:.17g}' for value in values)} 1 {row['camera_id']}.png", ""]
        )
    images_path = sparse / "images.txt"
    points_path = sparse / "points3D.txt"
    _write_exact(images_path, "\n".join(lines) + "\n")
    _write_exact(points_path, "# empty: points3D.ply is derived from the target-removed publisher PLY\n")
    return [camera_path, images_path, points_path]


def _write_rgb_points(splat: SplatData, destination: Path) -> None:
    colors = np.rint(np.clip(0.5 + SH_C0 * splat.f_dc, 0.0, 1.0) * 255.0).astype(np.uint8)
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
    rows["red"], rows["green"], rows["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {splat.count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload_digest = hashlib.sha256(header + rows.tobytes()).hexdigest()
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != "sha256:" + payload_digest:
            raise InFusionAdapterError(["infusion_existing_output_conflict"])
        return
    with destination.open("wb") as stream:
        stream.write(header)
        stream.write(rows.tobytes())


def _select_camera(masks: Sequence[Mapping[str, Any]]) -> str:
    eligible = [
        row for row in masks
        if row.get("gaussian_support_inside_fraction") == 1.0
        and isinstance(row.get("masked_pixel_count"), int)
        and row["masked_pixel_count"] > 0
    ]
    if not eligible:
        raise InFusionAdapterError(["infusion_single_view_selection_unavailable"])
    ordered = sorted(
        eligible,
        key=lambda row: (-row["masked_pixel_count"], str(row["camera_id"])),
    )
    return str(ordered[0]["camera_id"])


def materialize_infusion_adapter(
    *,
    input_receipt_path: str | Path,
    input_root: str | Path,
    prerequisite_receipt_path: str | Path,
    repo_root: str | Path,
    data_root: str | Path,
    infusion_root: str | Path,
    lama_source_root: str | Path,
    output_root: str | Path,
    receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    input_dir = _under(input_root, data, "infusion_input_root_outside_data")
    output = _under(output_root, data, "infusion_output_root_outside_data")
    input_receipt_path = _under(input_receipt_path, repo, "infusion_input_receipt_outside_repo")
    prereq_path = _under(prerequisite_receipt_path, data, "infusion_prerequisite_receipt_outside_data")
    retained = (
        _under(receipt_output, repo, "infusion_retained_receipt_outside_repo")
        if receipt_output is not None
        else None
    )
    input_receipt = _object(input_receipt_path, "infusion_input_receipt_invalid")
    _verify_digest(input_receipt, "receipt_digest", "infusion_input_receipt_digest_mismatch")
    if input_receipt.get("status") != "render_derived_input_packet_materialized":
        raise InFusionAdapterError(["infusion_input_packet_not_materialized"])
    prereq = _object(prereq_path, "infusion_prerequisite_receipt_invalid")
    _verify_digest(prereq, "receipt_digest", "infusion_prerequisite_receipt_digest_mismatch")

    infusion_identity = _git_identity(
        Path(infusion_root).expanduser().resolve(),
        commit=INFUSION_COMMIT,
        tree=INFUSION_TREE,
        label="infusion",
    )
    lama_identity = _git_identity(
        Path(lama_source_root).expanduser().resolve(),
        commit=INPAINT360_COMMIT,
        tree=INPAINT360_TREE,
        label="lama",
    )
    adapter_identity = _adapter_repository_identity(repo)
    methods = prereq.get("methods") or {}
    infusion_prereq = methods.get("infusion_primary_adapter") or {}
    snapshots = infusion_prereq.get("remote_snapshots") or []
    snapshot = next(
        (
            row for row in snapshots
            if (row.get("publisher") or {}).get("revision") == INFUSION_CHECKPOINT_REVISION
        ),
        None,
    )
    if snapshot is None:
        raise InFusionAdapterError(["infusion_checkpoint_snapshot_missing"])
    if infusion_prereq.get("checkpoint_rights_established") is not False:
        raise InFusionAdapterError(["infusion_checkpoint_rights_state_unexpected"])

    inpaint_prereq = methods.get("inpaint360_author_smoke") or {}
    big_lama = next(
        (row for row in inpaint_prereq.get("artifacts", []) if row.get("artifact_id") == "big_lama_author_linked_archive"),
        None,
    )
    if big_lama is None or big_lama.get("rights_established") is not True:
        raise InFusionAdapterError(["infusion_color_completer_rights_missing"])
    archive = _under(data / str(big_lama.get("relative_path")), data, "infusion_lama_archive_outside_data")
    if (
        not archive.is_file() or archive.is_symlink()
        or _sha256(archive) != BIG_LAMA_SHA256
        or archive.stat().st_size != big_lama.get("size_bytes")
    ):
        raise InFusionAdapterError(["infusion_lama_archive_bytes_changed"])

    derived = input_receipt.get("derived_artifacts") or {}
    records = [derived.get("cameras"), derived.get("standard_splat")]
    records += list(derived.get("images") or []) + list(derived.get("masks") or [])
    for record in records:
        if not isinstance(record, Mapping):
            raise InFusionAdapterError(["infusion_input_artifact_record_missing"])
        path = _under(input_dir / str(record.get("relative_path")), input_dir, "infusion_input_artifact_outside_root")
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != record.get("size_bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise InFusionAdapterError([f"infusion_input_artifact_bytes_changed:{record.get('relative_path')}"])

    camera_rows = json.loads((input_dir / str(derived["cameras"]["relative_path"])).read_text(encoding="utf-8"))
    if not isinstance(camera_rows, list) or len(camera_rows) != input_receipt.get("camera_policy", {}).get("camera_count"):
        raise InFusionAdapterError(["infusion_camera_manifest_invalid"])
    selected_camera = _select_camera(derived["masks"])
    image_by_id = {str(row["camera_id"]): row for row in derived["images"]}
    mask_by_id = {str(row["camera_id"]): row for row in derived["masks"]}
    camera_by_id = {str(row["camera_id"]): row for row in camera_rows}
    if set(image_by_id) != set(mask_by_id) or set(image_by_id) != set(camera_by_id):
        raise InFusionAdapterError(["infusion_camera_artifact_join_mismatch"])

    standard_path = input_dir / str(derived["standard_splat"]["relative_path"])
    splat = read_standard_3dgs_ply(standard_path)
    if splat.sh_rest is None or splat.sh_rest.shape[1] != 45:
        raise InFusionAdapterError(["infusion_publisher_sh_degree_not_three"])
    selected_rows = _obb_mask(splat.xyz, input_receipt.get("scene", {}).get("target_obb_corners_m") or [])
    selected_count = int(np.count_nonzero(selected_rows))
    if selected_count != input_receipt.get("scene", {}).get("target_gaussian_count"):
        raise InFusionAdapterError(["infusion_target_partition_count_mismatch"])
    keep = ~selected_rows
    background = SplatData(
        count=int(np.count_nonzero(keep)),
        xyz=splat.xyz[keep].copy(),
        opacity=splat.opacity[keep].copy(),
        f_dc=splat.f_dc[keep].copy(),
        scales=splat.scales[keep].copy(),
        quats=splat.quats[keep].copy(),
        properties=splat.properties,
        sh_rest=splat.sh_rest[keep].copy(),
    )
    model_ply = output / "incomplete_model" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    _atomic_splat(background, model_ply)
    _write_exact(output / "incomplete_model" / "cfg_args", "Namespace()\n")

    source = output / "source"
    staged: list[Path] = [model_ply, output / "incomplete_model" / "cfg_args"]
    for camera_id in sorted(image_by_id):
        image = input_dir / str(image_by_id[camera_id]["relative_path"])
        mask = input_dir / str(mask_by_id[camera_id]["relative_path"])
        image_out = source / "images" / f"{camera_id}.png"
        mask_out = source / "masks" / f"{camera_id}.png"
        _copy_exact(image, image_out)
        _copy_exact(mask, mask_out)
        staged.extend((image_out, mask_out))
    staged.extend(_write_colmap(source, camera_rows))
    points_path = source / "sparse" / "0" / "points3D.ply"
    _write_rgb_points(background, points_path)
    staged.append(points_path)

    selected_image = source / "images" / f"{selected_camera}.png"
    selected_mask = source / "masks" / f"{selected_camera}.png"
    lama_image = output / "lama_input" / f"{selected_camera}.png"
    lama_mask = output / "lama_input" / f"{selected_camera}_mask.png"
    _copy_exact(selected_image, lama_image)
    _copy_exact(selected_mask, lama_mask)
    staged.extend((lama_image, lama_mask))
    camera_record = output / "selected_camera.v1.json"
    _write_exact(camera_record, canonical_json(camera_by_id[selected_camera]) + "\n")
    staged.append(camera_record)
    with Image.open(lama_mask) as mask_image:
        if mask_image.mode != "L" or mask_image.size != (
            camera_by_id[selected_camera]["intrinsics"]["width"],
            camera_by_id[selected_camera]["intrinsics"]["height"],
        ):
            raise InFusionAdapterError(["infusion_selected_mask_contract_invalid"])

    infusion = Path(infusion_root).expanduser().resolve()
    lama = Path(lama_source_root).expanduser().resolve() / "LaMa"
    commands = {
        "extract_color_checkpoint": [
            "python", "-m", "zipfile", "-e", str(archive), str(output / "runtime" / "big-lama")
        ],
        "complete_selected_rgb": [
            "python", str(lama / "bin" / "predict.py"),
            f"model.path={output / 'runtime' / 'big-lama' / 'big-lama'}",
            f"indir={output / 'lama_input'}",
            f"outdir={output / 'lama_output'}",
        ],
        "render_incomplete_depth": [
            "python", str(infusion / "gaussian_splatting" / "render.py"),
            "-s", str(source), "-m", str(output / "incomplete_model"), "-u", "nothing",
            "--iteration", "30000", "--sh_degree", "3", "--skip_test",
        ],
        "run_infusion_depth_completion": [
            "python", str(infusion / "depth_inpainting" / "run" / "run_inference_inpainting.py"),
            "--input_rgb_path", str(output / "lama_output" / f"{selected_camera}.png"),
            "--input_mask", str(selected_mask),
            "--input_depth_path", str(output / "incomplete_model" / "train" / "ours_30000" / "depth_dis" / f"{selected_camera}.npy"),
            "--model_path", str(output / "runtime" / "Infusion"),
            "--output_dir", str(output / "infusion_output"),
            "--denoise_steps", "20",
            "--intri", str(output / "incomplete_model" / "train" / "ours_30000" / "intri" / f"{selected_camera}.npy"),
            "--c2w", str(output / "incomplete_model" / "train" / "ours_30000" / "c2w" / f"{selected_camera}.npy"),
            "--use_mask", "--blend",
        ],
        "compose_preserving_publisher_sh": [
            "python", "-m", "blueprint_pipeline.public_scene_infusion_compose",
            "--original-ply", str(model_ply),
            "--supplement-ply", str(output / "infusion_output" / f"{selected_camera}_mask.ply"),
            "--output-ply", str(output / "composed" / "point_cloud.ply"),
            "--data-root", str(data),
            "--receipt-output", str(output / "composed" / "composition_receipt.v1.json"),
        ],
    }
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "status": "blocked",
        "smallest_blocker": "infusion_checkpoint_license_missing",
        "blockers": ["infusion_checkpoint_license_missing", "infusion_checkpoint_bytes_not_materialized"],
        "source": {"infusion": infusion_identity, "color_completer_source": lama_identity},
        "adapter_repository": adapter_identity,
        "checkpoint": {
            "repository": "Johanan0528/Infusion",
            "revision": INFUSION_CHECKPOINT_REVISION,
            "declared_license": None,
            "rights_established": False,
            "materialized": False,
        },
        "color_completer": {
            "method": "Big-LaMa",
            "archive": _record(archive, data),
            "license": "Apache-2.0",
            "rights_established": True,
        },
        "input_receipt_digest": input_receipt["receipt_digest"],
        "prerequisite_receipt_digest": prereq["receipt_digest"],
        "scene": {
            **input_receipt["scene"],
            "source_gaussian_count": splat.count,
            "target_removed_gaussian_count": selected_count,
            "background_gaussian_count": background.count,
            "publisher_sh_degree": 3,
        },
        "single_view_selection": {
            "rule": "maximum_masked_pixel_count_among_full_gaussian_support_views_then_camera_id",
            "selected_camera_id": selected_camera,
            "masked_pixel_count": mask_by_id[selected_camera]["masked_pixel_count"],
            "method_outcomes_observed_before_selection": False,
        },
        "staged_artifacts": [_record(path, output) for path in sorted(set(staged))],
        "prepared_commands": commands,
        "execution": {
            "color_completion_executed": False,
            "infusion_executed": False,
            "composition_executed": False,
            "gpu_allocated_for_packet": False,
        },
        "claim_ceiling": "infusion_interiorgs_execution_packet_blocked_unexecuted",
        "proof_boundaries": {
            "prepared_packet_is_not_inpainting": True,
            "target_obb_partition_is_not_semantic_completeness": True,
            "generated_hidden_geometry_is_visual_candidate_only": True,
            "render_derived_inputs_are_not_independent_observation_truth": True,
        },
    }
    receipt["receipt_digest"] = canonical_digest(receipt)
    payload = canonical_json(receipt) + "\n"
    external_receipt = output / "adp009b_infusion_adapter_receipt.v1.json"
    _write_exact(external_receipt, payload)
    if retained is not None:
        _write_exact(retained, payload)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-receipt", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--prerequisite-receipt", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--infusion-root", required=True)
    parser.add_argument("--lama-source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--receipt-output")
    args = parser.parse_args(argv)
    receipt = materialize_infusion_adapter(
        input_receipt_path=args.input_receipt,
        input_root=args.input_root,
        prerequisite_receipt_path=args.prerequisite_receipt,
        repo_root=args.repo_root,
        data_root=args.data_root,
        infusion_root=args.infusion_root,
        lama_source_root=args.lama_source_root,
        output_root=args.output_root,
        receipt_output=args.receipt_output,
    )
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
