"""Seal a reusable, exact-mask AuraFusion360 provider bundle.

The 3-D Gaussian PLY that survives shared-scene excision is not an Aura 2-D
Gaussian checkpoint.  This module therefore stages a derived COLMAP scene
from the sealed retained frames, fits one *shared* 2-DGS, removes once using
the union of exact residual masks, and prepares one no-finetune Aura inpaint
initialization per task.  Each task starts from that one shared removal PLY;
only its own exact masks are nonzero during the task-specific initialization.

The materializer is deliberately not a provider launcher.  Its sole mutable
output is an immutable ZIP plus a no-mutation rehearsal receipt.  The ZIP
contains private derived frames/masks/PLY only, never raw InteriorGS bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .gaussian_splat_decode import read_standard_3dgs_ply
from .provider_bundle_rehearsal import rehearse_provider_bundle_entrypoint
from .public_scene_aura_exact_residual_preflight import SCHEMA_VERSION as PREFLIGHT_SCHEMA
from .public_scene_aura_residual_backend_admission import (
    AURA_COMMIT,
    AURA_REPOSITORY,
    AURA_TREE,
)
from .adp_aura_author_smoke_vast import (
    WONDERWORLD_MARIGOLD_LICENSE_SHA256,
    WONDERWORLD_MARIGOLD_RUNTIME_FILES,
    WONDERWORLD_SOURCE_COMMIT,
    WONDERWORLD_SOURCE_REPOSITORY,
    WONDERWORLD_SOURCE_TREE,
    _RUNTIME_MODELS,
)


SCHEMA_VERSION = "public_scene_aura_exact_residual_bundle.v1"
RUNTIME_REQUEST_SCHEMA = "public_scene_aura_exact_residual_runtime_request.v1"
ENTRYPOINT = "provider_runtime/run_public_scene_aura_exact_residual.sh"
AURA_RUNTIME_RUNNER = "provider_runtime/public_scene_aura_exact_residual_runner.py"
DEFAULT_IMAGE = (
    "docker.io/nvidia/cuda@"
    "sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749"
)
LAMA_REPOSITORY = "https://github.com/advimman/lama"
LAMA_COMMIT = "786f5936b27fb3dacd2b1ad799e4de968ea697e7"
LAMA_TREE = "25f9902ca0c2ec4bf6c31c2b4427f0a4f05f2fd1"
SH_C0 = 0.28209479177387814
MARIGOLD_RUNTIME_MODELS = tuple(
    dict(model)
    for model in _RUNTIME_MODELS
    if str(model.get("repository") or "").startswith("prs-eth/marigold")
)


class AuraExactResidualBundleError(ValueError):
    """Stable failures before a private-derived provider upload is possible."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    result["relative_path" if root is not None else "path"] = (
        path.relative_to(root).as_posix() if root is not None else str(path)
    )
    return result


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuraExactResidualBundleError([code]) from exc
    if not isinstance(value, dict):
        raise AuraExactResidualBundleError([code])
    return value


def _file(value: str | Path, *, code: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise AuraExactResidualBundleError([code])
    return path


def _bound_file(record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise AuraExactResidualBundleError([code])
    path = _file(record.get("path") or "", code=code)
    if path.stat().st_size != record.get("size_bytes") or _sha256(path) != record.get("sha256"):
        raise AuraExactResidualBundleError([code])
    return path


def _safe_slug(value: Any, *, code: str) -> str:
    text = str(value or "")
    if not text or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in text):
        raise AuraExactResidualBundleError([code])
    return text


def _link_or_copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise AuraExactResidualBundleError(["aura_exact_residual_bundle_source_file_invalid"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if _sha256(source) != _sha256(destination):
        raise AuraExactResidualBundleError(["aura_exact_residual_bundle_copy_digest_mismatch"])


def _copy_release_tree(source: Path, destination: Path) -> list[dict[str, Any]]:
    """Copy only the clean release's Git-tracked files without indirection."""

    files: list[dict[str, Any]] = []
    try:
        completed = subprocess.run(
            ["git", "-C", str(source), "ls-files", "--recurse-submodules", "-z"],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise AuraExactResidualBundleError(["aura_exact_residual_release_index_unreadable"]) from exc
    names = [name for name in completed.stdout.decode("utf-8").split("\0") if name]
    if not names or len(names) != len(set(names)):
        raise AuraExactResidualBundleError(["aura_exact_residual_release_tree_empty"])
    for name in sorted(names):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
            raise AuraExactResidualBundleError(["aura_exact_residual_release_member_invalid"])
        item = source / relative
        target = destination / relative
        if item.is_symlink():
            raise AuraExactResidualBundleError(["aura_exact_residual_release_symlink_forbidden"])
        if not item.is_file():
            raise AuraExactResidualBundleError(["aura_exact_residual_release_member_invalid"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        if _sha256(item) != _sha256(target):
            raise AuraExactResidualBundleError(["aura_exact_residual_release_copy_digest_mismatch"])
        files.append(_record(target, root=destination))
    return files


def _copy_wonderworld_marigold_runtime(source: Path, destination: Path) -> list[dict[str, Any]]:
    """Stage precisely the Apache helper modules Aura imports for Marigold.

    Aura imports these helpers at module import time.  Copying them from a clean,
    pinned source tree makes that dependency inspectable without inheriting an
    earlier provider packet or silently widening this packet to the full project.
    """

    records: list[dict[str, Any]] = []
    for staged_name, source_name in sorted(WONDERWORLD_MARIGOLD_RUNTIME_FILES.items()):
        relative = Path(staged_name)
        source_path = source / source_name
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not source_path.is_file()
            or source_path.is_symlink()
        ):
            raise AuraExactResidualBundleError(
                ["aura_exact_residual_bundle_wonderworld_runtime_source_invalid"]
            )
        target = destination / relative
        _link_or_copy(source_path, target)
        records.append(_record(target, root=destination))
    if not records:
        raise AuraExactResidualBundleError(
            ["aura_exact_residual_bundle_wonderworld_runtime_source_invalid"]
        )
    return records


def _git_value(root: Path, *args: str, code: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise AuraExactResidualBundleError([code]) from exc


def _verified_git_release(
    root: str | Path,
    *,
    repository: str,
    commit: str,
    tree: str,
    code: str,
) -> dict[str, Any]:
    path = Path(root).expanduser().resolve()
    if not path.is_dir() or path.is_symlink():
        raise AuraExactResidualBundleError([code])
    if (
        _git_value(path, "rev-parse", "HEAD", code=code) != commit
        or _git_value(path, "rev-parse", "HEAD^{tree}", code=code) != tree
        or _git_value(path, "status", "--porcelain", "--untracked-files=no", code=code)
    ):
        raise AuraExactResidualBundleError([code])
    return {
        "repository": repository,
        "commit": commit,
        "tree": tree,
        "tracked_files_clean": True,
    }


def _preflight(path: str | Path) -> tuple[Path, dict[str, Any]]:
    preflight_path = _file(path, code="aura_exact_residual_bundle_preflight_missing")
    value = _read(preflight_path, code="aura_exact_residual_bundle_preflight_unreadable")
    workflow = value.get("aura_workflow")
    backend = value.get("backend_admission")
    if (
        value.get("schema_version") != PREFLIGHT_SCHEMA
        or value.get("status") != "prepared_no_upload_no_execution"
        or value.get("preflight_digest") != canonical_digest(value, digest_field="preflight_digest")
        or not isinstance(workflow, Mapping)
        or workflow.get("released_entrypoints") != ["train.py", "remove.py", "inpaint.py"]
        or workflow.get("excluded_stock_stages")
        != ["utils/sam2_utils.py", "utils/LeftRefill/sdedit_utils.py"]
        or not isinstance(workflow.get("workflow_stages"), list)
        or not workflow["workflow_stages"]
        or workflow["workflow_stages"][0].get("train_dilate_mask_kernel_size") != 1
        or workflow["workflow_stages"][0].get("train_dilate_mask_iter") != 0
        or workflow.get("inpaint_dilate_mask_iter") != 0
        or workflow.get("inpaint_dilate_mask_kernel_size") != 1
        or workflow.get("agdd_dilate_iter") != 0
        or workflow.get("agdd_kernel_size") != 1
        or workflow.get("inpaint_init_finetune_iteration") != -1
        or not isinstance(backend, Mapping)
        or backend.get("backend_id") != "aurafusion360_exact_residual_multiview"
        or value.get("replacement_object_count") != len(value.get("lanes") or [])
        or not 1 <= int(value.get("replacement_object_count") or 0) <= 5
        or value.get("shared_retained_scene", {}).get("all_replacements_co_present") is not True
        or value.get("execution", {}).get("provider_mutations_performed") != 0
        or value.get("execution", {}).get("aura_inpainting_executed") is not False
    ):
        raise AuraExactResidualBundleError(["aura_exact_residual_bundle_preflight_invalid"])
    return preflight_path, value


def _image_and_mask(
    record: Any,
    *,
    dimensions: tuple[int, int],
    mask: bool,
    code: str,
) -> Path:
    path = _bound_file(record, code=code)
    try:
        with Image.open(path) as image:
            converted = image.convert("L" if mask else "RGB")
            if converted.size != dimensions:
                raise AuraExactResidualBundleError([code])
            if mask:
                values = set(converted.tobytes())
                if not values.issubset({0, 255}) or values != {0, 255}:
                    raise AuraExactResidualBundleError([code])
    except (OSError, ValueError) as exc:
        if isinstance(exc, AuraExactResidualBundleError):
            raise
        raise AuraExactResidualBundleError([code]) from exc
    return path


def _camera_row(row: Any) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(row, Mapping):
        raise AuraExactResidualBundleError(["aura_exact_residual_bundle_camera_invalid"])
    task_id = _safe_slug(row.get("task_id"), code="aura_exact_residual_bundle_camera_invalid")
    camera_id = _safe_slug(row.get("camera_id"), code="aura_exact_residual_bundle_camera_invalid")
    calibration = row.get("calibration")
    if not isinstance(calibration, Mapping) or calibration.get("id") != camera_id:
        raise AuraExactResidualBundleError(["aura_exact_residual_bundle_camera_invalid"])
    spec = calibration.get("spec")
    pose = spec.get("pose") if isinstance(spec, Mapping) else None
    intrinsics = spec.get("intrinsics") if isinstance(spec, Mapping) else None
    matrix = pose.get("T_world_camera_opencv") if isinstance(pose, Mapping) else None
    if (
        not isinstance(matrix, list)
        or len(matrix) != 4
        or any(not isinstance(line, list) or len(line) != 4 for line in matrix)
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for line in matrix
            for value in line
        )
        or [float(value) for value in matrix[3]] != [0.0, 0.0, 0.0, 1.0]
        or not isinstance(intrinsics, Mapping)
        or intrinsics.get("model") != "PINHOLE"
        or any(
            not isinstance(intrinsics.get(key), (int, float))
            or not math.isfinite(float(intrinsics[key]))
            for key in ("fx", "fy", "cx", "cy", "width", "height")
        )
        or float(intrinsics["fx"]) <= 0
        or float(intrinsics["fy"]) <= 0
        or int(intrinsics["width"]) <= 0
        or int(intrinsics["height"]) <= 0
        or float(intrinsics["cx"]) != int(intrinsics["width"]) / 2.0
        or float(intrinsics["cy"]) != int(intrinsics["height"]) / 2.0
    ):
        raise AuraExactResidualBundleError(["aura_exact_residual_bundle_camera_invalid"])
    return task_id, camera_id, {
        "T_world_camera_opencv": [[float(value) for value in line] for line in matrix],
        "intrinsics": {
            "model": "PINHOLE",
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
            "width": int(intrinsics["width"]),
            "height": int(intrinsics["height"]),
        },
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
    return -qvec if qvec[0] < 0 else qvec


def _write_colmap(scene_root: Path, cameras: Sequence[Mapping[str, Any]]) -> list[Path]:
    """Write one exact PINHOLE calibration record for every distinct intrinsic set."""

    sparse = scene_root / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    intrinsic_ids: dict[tuple[int, int, float, float, float, float], int] = {}
    camera_lines = ["# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    image_lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    for index, row in enumerate(cameras, start=1):
        intrinsic = row["calibration"]["intrinsics"]
        key = (
            int(intrinsic["width"]),
            int(intrinsic["height"]),
            float(intrinsic["fx"]),
            float(intrinsic["fy"]),
            float(intrinsic["cx"]),
            float(intrinsic["cy"]),
        )
        camera_number = intrinsic_ids.get(key)
        if camera_number is None:
            camera_number = len(intrinsic_ids) + 1
            intrinsic_ids[key] = camera_number
            camera_lines.append(
                f"{camera_number} PINHOLE {key[0]} {key[1]} "
                f"{key[2]:.17g} {key[3]:.17g} {key[4]:.17g} {key[5]:.17g}"
            )
        pose = np.asarray(row["calibration"]["T_world_camera_opencv"], dtype=np.float64)
        w2c = np.linalg.inv(pose)
        qvec = _rotmat_to_qvec(w2c[:3, :3])
        values = " ".join(f"{value:.17g}" for value in (*qvec, *w2c[:3, 3]))
        image_lines.extend([f"{index} {values} {camera_number} {row['staged_name']}.png", ""])
    camera_path = sparse / "cameras.txt"
    image_path = sparse / "images.txt"
    points_path = sparse / "points3D.txt"
    camera_path.write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    image_path.write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    points_path.write_text("# points3D.ply is an exact deterministic retained-splat conversion\n", encoding="utf-8")
    return [camera_path, image_path, points_path]


def _write_seed_points(standard_splat: Path, output: Path) -> Path:
    """Convert retained 3-DGS positions and diffuse color only; never make CAD."""

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


def _write_config(path: Path, values: Mapping[str, Any]) -> Path:
    text = "\n".join(f"{key} = {json.dumps(value)}" for key, value in values.items()) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_deterministic_zip(root: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            if path == output or not path.is_file() or path.is_symlink():
                continue
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_aura_exact_residual_bundle(
    *,
    preflight_path: str | Path,
    aura_source_directory: str | Path,
    lama_source_directory: str | Path,
    wonderworld_source_directory: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Materialize a fresh-source, private-derived Aura bundle and rehearse it."""

    preflight_file, preflight = _preflight(preflight_path)
    repo = Path(repo_root).expanduser().resolve()
    if not (repo / "scripts" / "run_public_scene_aura_exact_residual.sh").is_file() or not (
        repo / "scripts" / "public_scene_aura_exact_residual_runner.py"
    ).is_file():
        raise AuraExactResidualBundleError(["aura_exact_residual_bundle_runner_missing"])
    aura_identity = _verified_git_release(
        aura_source_directory,
        repository=AURA_REPOSITORY,
        commit=AURA_COMMIT,
        tree=AURA_TREE,
        code="aura_exact_residual_bundle_aura_source_identity_invalid",
    )
    lama_identity = _verified_git_release(
        lama_source_directory,
        repository=LAMA_REPOSITORY,
        commit=LAMA_COMMIT,
        tree=LAMA_TREE,
        code="aura_exact_residual_bundle_lama_source_identity_invalid",
    )
    wonderworld_identity = _verified_git_release(
        wonderworld_source_directory,
        repository=WONDERWORLD_SOURCE_REPOSITORY,
        commit=WONDERWORLD_SOURCE_COMMIT,
        tree=WONDERWORLD_SOURCE_TREE,
        code="aura_exact_residual_bundle_wonderworld_source_identity_invalid",
    )
    wonderworld_license = (
        Path(wonderworld_source_directory).expanduser().resolve()
        / "marigold_module"
        / "LICENSE.txt"
    )
    if (
        not wonderworld_license.is_file()
        or wonderworld_license.is_symlink()
        or _sha256(wonderworld_license) != WONDERWORLD_MARIGOLD_LICENSE_SHA256
    ):
        raise AuraExactResidualBundleError(
            ["aura_exact_residual_bundle_wonderworld_license_invalid"]
        )
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise AuraExactResidualBundleError(["aura_exact_residual_bundle_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)
    runtime = output / "provider_runtime"
    runtime.mkdir()

    backend_path = _bound_file(
        preflight.get("backend_admission"), code="aura_exact_residual_bundle_backend_invalid"
    )
    backend = _read(backend_path, code="aura_exact_residual_bundle_backend_invalid")
    source_archive = _bound_file(
        backend.get("source_archive"), code="aura_exact_residual_bundle_source_archive_invalid"
    )
    if (
        backend.get("source_repository") != aura_identity["repository"]
        or backend.get("source_revision") != aura_identity["commit"]
        or backend.get("source_tree") != aura_identity["tree"]
        or backend.get("source_archive_sha256") != _sha256(source_archive)
    ):
        raise AuraExactResidualBundleError(["aura_exact_residual_bundle_source_identity_mismatch"])
    big_lama = preflight.get("reference_completion", {}).get("backend_provenance", {})
    checkpoint = big_lama.get("checkpoint") if isinstance(big_lama, Mapping) else None
    big_lama_archive = _bound_file(
        checkpoint, code="aura_exact_residual_bundle_big_lama_checkpoint_invalid"
    )
    rights = big_lama.get("rights_authority") if isinstance(big_lama, Mapping) else None
    if (
        not isinstance(rights, Mapping)
        or rights.get("repository") != LAMA_REPOSITORY
        or rights.get("revision") != LAMA_COMMIT
        or rights.get("repository_tree") != LAMA_TREE
        or rights.get("license_id") != "Apache-2.0"
    ):
        raise AuraExactResidualBundleError(["aura_exact_residual_bundle_big_lama_rights_invalid"])

    shared_ply = _bound_file(
        preflight.get("shared_retained_scene"), code="aura_exact_residual_bundle_shared_ply_invalid"
    )
    expected_count = preflight["shared_retained_scene"].get("retained_gaussian_count")
    if read_standard_3dgs_ply(shared_ply).count != expected_count:
        raise AuraExactResidualBundleError(["aura_exact_residual_bundle_shared_ply_count_invalid"])

    staged_aura_files = _copy_release_tree(
        Path(aura_source_directory).expanduser().resolve(), runtime / "AuraFusion360_official"
    )
    staged_lama_files = _copy_release_tree(
        Path(lama_source_directory).expanduser().resolve(), runtime / "LaMa"
    )
    staged_wonderworld_files = _copy_wonderworld_marigold_runtime(
        Path(wonderworld_source_directory).expanduser().resolve(),
        runtime / "runtime_dependencies",
    )
    _link_or_copy(big_lama_archive, runtime / "big-lama.zip")
    _link_or_copy(shared_ply, runtime / "input" / "shared_retained_scene.ply")
    _link_or_copy(preflight_file, runtime / "input" / "preflight.json")
    _link_or_copy(backend_path, runtime / "input" / "backend_admission.json")
    _link_or_copy(source_archive, runtime / "input" / "aurafusion360_source_admission_anchor.zip")

    camera_rows = preflight.get("camera_inputs")
    if not isinstance(camera_rows, list) or not camera_rows:
        raise AuraExactResidualBundleError(["aura_exact_residual_bundle_camera_set_missing"])
    staged_cameras: list[dict[str, Any]] = []
    task_cameras: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    scene_root = runtime / "data" / "Other-360" / "shared_retained_scene"
    for directory in ("images", "object_masks", "unseen_masks", "reference"):
        (scene_root / directory).mkdir(parents=True, exist_ok=True)
    for row in camera_rows:
        task_id, camera_id, calibration = _camera_row(row)
        key = (task_id, camera_id)
        if key in seen:
            raise AuraExactResidualBundleError(["aura_exact_residual_bundle_camera_set_invalid"])
        seen.add(key)
        dimensions = (calibration["intrinsics"]["width"], calibration["intrinsics"]["height"])
        before = _image_and_mask(
            row.get("retained_scene_before"), dimensions=dimensions, mask=False,
            code="aura_exact_residual_bundle_retained_frame_invalid",
        )
        exact_mask = _image_and_mask(
            row.get("exact_residual_mask"), dimensions=dimensions, mask=True,
            code="aura_exact_residual_bundle_exact_mask_invalid",
        )
        staged_name = f"{task_id}--{camera_id}"
        image_destination = scene_root / "images" / f"{staged_name}.png"
        mask_destination = scene_root / "object_masks" / f"{staged_name}.png"
        _link_or_copy(before, image_destination)
        _link_or_copy(exact_mask, mask_destination)
        record = {
            "task_id": task_id,
            "camera_id": camera_id,
            "staged_name": staged_name,
            "calibration": calibration,
            "retained_scene_before": _record(image_destination, root=runtime),
            "exact_residual_mask": _record(mask_destination, root=runtime),
            "source_retained_scene_before": _record(before),
            "source_exact_residual_mask": _record(exact_mask),
        }
        staged_cameras.append(record)
        task_cameras.setdefault(task_id, []).append(record)
    if len(task_cameras) != len(preflight.get("lanes") or []) or any(not rows for rows in task_cameras.values()):
        raise AuraExactResidualBundleError(["aura_exact_residual_bundle_task_camera_join_invalid"])
    _write_colmap(scene_root, staged_cameras)
    _write_seed_points(
        runtime / "input" / "shared_retained_scene.ply", scene_root / "sparse" / "0" / "points3D.ply"
    )

    task_plans: list[dict[str, Any]] = []
    references = {
        str(row.get("task_id")): row
        for row in preflight.get("reference_completion", {}).get("references", [])
        if isinstance(row, Mapping)
    }
    for task_id in sorted(task_cameras):
        reference = references.get(task_id)
        if not isinstance(reference, Mapping):
            raise AuraExactResidualBundleError(["aura_exact_residual_bundle_reference_missing"])
        reference_id = _safe_slug(reference.get("camera_id"), code="aura_exact_residual_bundle_reference_missing")
        rows = task_cameras[task_id]
        by_camera = {row["camera_id"]: row for row in rows}
        if reference_id not in by_camera:
            raise AuraExactResidualBundleError(["aura_exact_residual_bundle_reference_missing"])
        workspace = runtime / "task_workspaces" / task_id
        for directory in ("unseen_masks", "reference_lama_input"):
            (workspace / directory).mkdir(parents=True, exist_ok=True)
        task_names = {row["staged_name"] for row in rows}
        zero_masks: dict[tuple[int, int], Path] = {}
        for row in staged_cameras:
            image_size = (
                row["calibration"]["intrinsics"]["width"],
                row["calibration"]["intrinsics"]["height"],
            )
            if row["staged_name"] in task_names:
                source_mask = runtime / row["exact_residual_mask"]["relative_path"]
                _link_or_copy(source_mask, workspace / "unseen_masks" / f"{row['staged_name']}.png")
            else:
                zero = zero_masks.get(image_size)
                if zero is None:
                    zero = workspace / f"zero_mask_{image_size[0]}x{image_size[1]}.png"
                    Image.new("L", image_size, 0).save(zero)
                    zero_masks[image_size] = zero
                _link_or_copy(zero, workspace / "unseen_masks" / f"{row['staged_name']}.png")
        reference_row = by_camera[reference_id]
        _link_or_copy(
            runtime / reference_row["retained_scene_before"]["relative_path"],
            workspace / "reference_lama_input" / f"{reference_row['staged_name']}.png",
        )
        _link_or_copy(
            runtime / reference_row["exact_residual_mask"]["relative_path"],
            workspace / "reference_lama_input" / f"{reference_row['staged_name']}_mask.png",
        )
        reference_index = [row["staged_name"] for row in sorted(staged_cameras, key=lambda item: item["staged_name"])].index(reference_row["staged_name"])
        task_plans.append(
            {
                "task_id": task_id,
                "reference_camera_id": reference_id,
                "reference_staged_name": reference_row["staged_name"],
                "reference_index_in_sorted_shared_camera_set": reference_index,
                "output_experiment": f"exact_residual--{task_id}",
                "review_camera_ids": sorted(row["camera_id"] for row in rows),
                "workspace": str(workspace.relative_to(runtime)),
                # Aura selects the released COLMAP loader only when the source
                # scene's parent directory is named ``Other-360``.  These paths
                # are evaluated with AuraFusion360_official as the working
                # directory, so keep both this and the shared source explicitly
                # outside the release tree.
                "task_scene": f"../data/Other-360/shared_retained_scene--{task_id}",
            }
        )
    train_config = _write_config(
        runtime / "configs" / "train.config",
        {
            "source_path": "../data/Other-360/shared_retained_scene",
            "model_path": "../work/model",
            "resolution": 1,
            "iterations": 30000,
            "dilate_mask_kernel_size": 1,
            "dilate_mask_iter": 0,
            "optimize_is_masked_iter": 20000,
            "is_masked_lr": 0.1,
            "is_masked_3d_lr": 0.0,
            "other_depth_lr": 0.0,
        },
    )
    remove_config = _write_config(
        runtime / "configs" / "remove.config",
        {
            "source_path": "../data/Other-360/shared_retained_scene",
            "model_path": "../work/model",
            "iteration": 30000,
            "resolution": 1,
            "skip_train": True,
            "skip_test": True,
            "skip_mesh": True,
            "render_path": False,
            "removal_thresh": 0.3,
            "outlier_factor": 1.0,
        },
    )
    inpaint_configs: list[dict[str, Any]] = []
    for plan in task_plans:
        config = _write_config(
            runtime / "configs" / f"inpaint_{plan['task_id']}.config",
            {
                "source_path": plan["task_scene"],
                "model_path": "../work/model",
                "iteration": 30000,
                "resolution": 1,
                # Retain native full-resolution training-camera renders for
                # review.  The input scene holds only exact cameras.
                "skip_train": False,
                "skip_test": True,
                "skip_mesh": True,
                "render_path": False,
                "dilate_mask_kernel_size": 1,
                "dilate_mask_iter": 0,
                "dilate_iter": 0,
                "kernel_size": 1,
                "finetune_iteration": -1,
                "reference_index": plan["reference_index_in_sorted_shared_camera_set"],
                "exp": plan["output_experiment"],
            },
        )
        inpaint_configs.append({"task_id": plan["task_id"], **_record(config, root=runtime)})
    for source_relative, destination in (
        ("scripts/run_public_scene_aura_exact_residual.sh", runtime / Path(ENTRYPOINT).name),
        ("scripts/public_scene_aura_exact_residual_runner.py", runtime / Path(AURA_RUNTIME_RUNNER).name),
    ):
        source = repo / source_relative
        shutil.copy2(source, destination)
        if source.suffix == ".sh":
            destination.chmod(destination.stat().st_mode | stat.S_IXUSR)
    request: dict[str, Any] = {
        "schema_version": RUNTIME_REQUEST_SCHEMA,
        "status": "ready_for_exact_mask_aura_execution",
        "preflight": {**_record(runtime / "input" / "preflight.json", root=runtime), "preflight_digest": preflight["preflight_digest"]},
        "backend_admission": _record(runtime / "input" / "backend_admission.json", root=runtime),
        "shared_retained_scene": {
            **_record(runtime / "input" / "shared_retained_scene.ply", root=runtime),
            "retained_gaussian_count": expected_count,
            "all_replacements_co_present": True,
        },
        "aura_release": aura_identity,
        "lama_release": lama_identity,
        "wonderworld_marigold_runtime": {
            **wonderworld_identity,
            "license": "Apache-2.0",
            "license_sha256": WONDERWORLD_MARIGOLD_LICENSE_SHA256,
            "files": staged_wonderworld_files,
        },
        # The provider is permitted to download this released checkpoint once
        # from this immutable revision.  The runner hashes every selected file,
        # writes the Hugging Face ref itself, then enables offline mode before
        # Aura begins.  No checkpoint bytes are part of the private-scene upload.
        "marigold_runtime_models": list(MARIGOLD_RUNTIME_MODELS),
        "big_lama_checkpoint": _record(runtime / "big-lama.zip", root=runtime),
        "camera_inputs": staged_cameras,
        "task_plans": task_plans,
        "commands": {
            "released_entrypoints": ["train.py", "remove.py", "inpaint.py"],
            "excluded_stock_entrypoints": ["utils/sam2_utils.py", "utils/LeftRefill/sdedit_utils.py"],
            "shared_train_config": _record(train_config, root=runtime),
            "shared_remove_config": _record(remove_config, root=runtime),
            "task_inpaint_configs": inpaint_configs,
            "inpaint_finetune_iteration": -1,
            "reference_completion": "Big-LaMa_then_exact_mask_composite_only",
            "task_initialization_checkpoint": (
                "one_shared_removal_point_cloud_digest_verified_before_and_after_each_task"
            ),
        },
        "private_derived_upload_only": True,
        "raw_dataset_bytes_included": False,
        "provider_training_authorized": False,
        "scene_fitting_is_provider_training": False,
        "automatic_paid_retry_allowed": False,
        "provider_zero_required_after_return": True,
        "learned_policy_outcomes_accessed": False,
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path = runtime / "aura_exact_residual_runtime_request.json"
    request_path.write_text(canonical_json(request) + "\n", encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "container_image": DEFAULT_IMAGE,
        "entrypoint": ENTRYPOINT,
        "runner": AURA_RUNTIME_RUNNER,
        "preflight_digest": preflight["preflight_digest"],
        "replacement_object_count": preflight["replacement_object_count"],
        "shared_camera_count": len(staged_cameras),
        "task_count": len(task_plans),
        "aura_release": aura_identity,
        "lama_release": lama_identity,
        "wonderworld_marigold_runtime": request["wonderworld_marigold_runtime"],
        "marigold_runtime_models": request["marigold_runtime_models"],
        "aura_release_files": staged_aura_files,
        "lama_release_files": staged_lama_files,
        "wonderworld_marigold_runtime_files": staged_wonderworld_files,
        "private_derived_upload_only": True,
        "raw_interiorgs_bytes_included": False,
        "stock_inpaint360gs_code_or_author_data_included": False,
        "automatic_paid_retry_allowed": False,
        "provider_zero_required_after_return": True,
        "claim_boundary": {
            "inpainting_result_qualified": False,
            "source_gaussian_removal_qualified": False,
            "native_simulator_import_qualified": False,
        },
    }
    manifest_path = runtime / "aura_exact_residual_bundle_manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    bundle_path = output / "aura_exact_residual_provider_bundle.zip"
    _write_deterministic_zip(output, bundle_path)
    if zipfile.ZipFile(bundle_path).testzip() is not None:
        raise AuraExactResidualBundleError(["aura_exact_residual_bundle_integrity_invalid"])
    rehearsal = rehearse_provider_bundle_entrypoint(
        bundle_path=bundle_path,
        entrypoint_relative_path=ENTRYPOINT,
        evidence_path=output / "aura_exact_residual_exact_bundle_rehearsal.json",
    )
    receipt = {
        **manifest,
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": _sha256(bundle_path),
        "runtime_request": _record(request_path),
        "exact_bundle_entrypoint_rehearsal": rehearsal,
    }
    receipt_path = output / "aura_exact_residual_bundle_receipt.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


__all__ = [
    "AURA_RUNTIME_RUNNER",
    "DEFAULT_IMAGE",
    "ENTRYPOINT",
    "SCHEMA_VERSION",
    "AuraExactResidualBundleError",
    "build_aura_exact_residual_bundle",
]
