"""Seal the exact GPU render inputs for one to five shared-scene replacements.

This is deliberately an input/bundle boundary, not a render result.  It lets a
provider receive only the two derived standard PLYs needed to render the
frozen task cameras: the byte-exact deleted shared source layer for residual
masks and the byte-exact shared retained scene for inpainting backgrounds.
The complete source conversion stays local provenance; raw InteriorGS bytes,
replacement generation, inpainting, and policy outcomes are all outside this
module.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import MAX_REPLACEMENT_OBJECTS, validate_task_freeze
from .gaussian_splat_decode import (
    read_standard_3dgs_ply,
    verify_standard_3dgs_ply_subset_exact,
)
from .provider_bundle_rehearsal import rehearse_provider_bundle_entrypoint


REQUEST_SCHEMA = "adp009d_retained_scene_gpu_render_request.v1"
BUNDLE_SCHEMA = "adp009d_retained_scene_gpu_render_bundle.v1"
PROBE_KIND = "adp-retained-scene-gpu-render"
ENTRYPOINT = "provider_runtime/run_adp_retained_scene_render_provider_runtime.sh"
DEFAULT_IMAGE = (
    "mcr.microsoft.com/playwright:v1.61.1-jammy@"
    "sha256:e4f20543d7da3faeddbce0176b447331ad652f0ca8c669dc7e7b205f2067677e"
)
_VENDOR_PACKAGES = (
    "@sparkjsdev/spark",
    "fflate",
    "playwright",
    "playwright-core",
    "three",
)
_ADMITTED_CANDIDATE_SET_SCHEMAS = frozenset(
    {
        "adp009b_direct_evidence_expansion_set.v1",
        "adp009b_ownership_coverage_cutout_set.v1",
        "adp009d_segment_contribution_cutout_set.v1",
    }
)


class RetainedSceneRenderPacketError(ValueError):
    """Stable fail-closed errors for retained-scene GPU render preparation."""

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
    if root is None:
        result["path"] = str(path)
    else:
        result["relative_path"] = path.relative_to(root).as_posix()
    return result


def _link_or_copy(source: Path, destination: Path) -> None:
    """Prefer a same-volume hardlink for immutable, receipt-verified inputs.

    This avoids duplicating hundreds of megabytes of derived PLY/vendor bytes
    while an archive is being built on a constrained workstation.  The final
    ZIP remains self-contained and immutable; cross-device sources fall back
    to a byte copy.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetainedSceneRenderPacketError([code]) from exc
    if not isinstance(value, dict):
        raise RetainedSceneRenderPacketError([code])
    return value


def _file(value: str | Path, *, code: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RetainedSceneRenderPacketError([code])
    return path


def _resolve_request_input(
    value: Any,
    *,
    repo: Path,
    scene_root: Path | None,
    code: str,
    directory: bool = False,
) -> Path:
    """Resolve a request input against the checkout or the staged scene root.

    A request that names absolute paths can only be rebuilt on the machine that
    wrote it, which is how the committed v1 manifest ended up pointing at two
    deleted ``/private/tmp`` directories. A relative input says *what* it needs
    and leaves *where* to the invocation, so the same manifest rebuilds the
    bundle on any checkout with the scene bytes staged anywhere. Absolute inputs
    stay supported: the earlier manifests are frozen evidence.
    """

    text = str(value or "").strip()
    if not text:
        raise RetainedSceneRenderPacketError([code])
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return _directory(candidate, code=code) if directory else _file(candidate, code=code)
    if ".." in candidate.parts:
        raise RetainedSceneRenderPacketError([code])
    for base in (repo, scene_root):
        if base is None:
            continue
        resolved = base / candidate
        if resolved.is_symlink():
            continue
        if resolved.is_dir() if directory else resolved.is_file():
            return _directory(resolved, code=code) if directory else _file(resolved, code=code)
    raise RetainedSceneRenderPacketError([code])


def _directory(value: str | Path, *, code: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir() or path.is_symlink():
        raise RetainedSceneRenderPacketError([code])
    return path


def _absolute_record(value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise RetainedSceneRenderPacketError([code])
    path = _file(str(value.get("path") or ""), code=code)
    if value.get("size_bytes") != path.stat().st_size or value.get("sha256") != _sha256(path):
        raise RetainedSceneRenderPacketError([code])
    return path


def _relative_record(root: Path, value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise RetainedSceneRenderPacketError([code])
    relative = str(value.get("relative_path") or "")
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise RetainedSceneRenderPacketError([code])
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file() or path.is_symlink():
        raise RetainedSceneRenderPacketError([code])
    if value.get("size_bytes") != path.stat().st_size or value.get("sha256") != _sha256(path):
        raise RetainedSceneRenderPacketError([code])
    return path


def _clone(value: Mapping[str, Any], *, code: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RetainedSceneRenderPacketError([code]) from exc
    if not isinstance(result, dict):
        raise RetainedSceneRenderPacketError([code])
    return result


def build_retained_scene_gpu_render_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and digest a pre-render request without opening scene bytes."""

    request = _clone(value, code="retained_scene_render_request_not_json")
    supplied_digest = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("retained_scene_render_request_schema_invalid")
    if (
        request.get("program_id") != "arm-decision-proof-v1"
        or request.get("adp_item") != "ADP-009D"
    ):
        errors.append("retained_scene_render_request_program_invalid")
    if request.get("frozen_before_render_execution") is not True:
        errors.append("retained_scene_render_request_not_frozen")
    if request.get("learned_policy_outcomes_accessed") is not False:
        errors.append("retained_scene_render_request_policy_outcome_leakage")
    if any(
        key in request for key in ("status", "render_qualified", "provider_mutations_performed")
    ):
        errors.append("retained_scene_render_request_caller_outcome_forbidden")
    for key in (
        "candidate_set_path",
        "execution_authority_path",
        "renderer_vendor_root",
    ):
        if not str(request.get(key) or "").strip():
            errors.append(f"retained_scene_render_request_{key}_missing")
    lanes = request.get("task_lanes")
    if not isinstance(lanes, list) or not 1 <= len(lanes) <= MAX_REPLACEMENT_OBJECTS:
        errors.append("retained_scene_render_request_task_lane_count_invalid")
        lanes = []
    ids: set[str] = set()
    for lane in lanes:
        if not isinstance(lane, Mapping):
            errors.append("retained_scene_render_request_task_lane_invalid")
            continue
        task_id = str(lane.get("task_id") or "").strip()
        if not task_id or task_id in ids:
            errors.append("retained_scene_render_request_task_id_invalid_or_duplicate")
        ids.add(task_id)
        if not str(lane.get("camera_contract_path") or "").strip():
            errors.append("retained_scene_render_request_camera_contract_missing")
    privacy = request.get("private_upload_policy")
    if not isinstance(privacy, Mapping) or (
        privacy.get("raw_dataset_bytes_upload") is not False
        or privacy.get("private_derived_upload") is not True
        or privacy.get("provider_training") is not False
        or privacy.get("publication") is not False
        or privacy.get("retention") != "bounded_to_goal_then_provider_zero"
    ):
        errors.append("retained_scene_render_request_private_upload_policy_invalid")
    if errors:
        raise RetainedSceneRenderPacketError(errors)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    if supplied_digest is not None and supplied_digest != request["request_digest"]:
        raise RetainedSceneRenderPacketError(["retained_scene_render_request_digest_mismatch"])
    return request


def _validate_authority(path: Path) -> dict[str, Any]:
    authority = _read(path, code="retained_scene_render_execution_authority_unreadable")
    paid = authority.get("paid_compute")
    valid = (
        authority.get("schema_version") == "third_scene_dual_task_execution_authority.v1"
        and authority.get("program_id") == "arm-decision-proof-v1"
        and authority.get("publisher_scene_id") == "840920"
        and authority.get("private_rights_admitted_scene_derived_uploads_authorized") is True
        and authority.get("raw_interiorgs_upload_authorized") is False
        and authority.get("training_authorized") is False
        and authority.get("public_dataset_bytes_publication_authorized") is False
        and authority.get("retention") == "bounded_to_goal_then_provider_zero"
        and isinstance(paid, Mapping)
        and paid.get("provider") == "vast"
        and paid.get("zero_retry") is True
        and paid.get("provider_zero_required_for_lane") is True
        and authority.get("authority_digest")
        == canonical_digest(authority, digest_field="authority_digest")
    )
    if not valid:
        raise RetainedSceneRenderPacketError(["retained_scene_render_execution_authority_invalid"])
    return authority


def _validate_candidate_set(
    path: Path,
) -> tuple[
    dict[str, Any],
    Path,
    Path,
    Path,
    dict[str, tuple[Path, dict[str, Any]]],
]:
    candidate = _read(path, code="retained_scene_render_candidate_set_unreadable")
    if (
        candidate.get("schema_version") not in _ADMITTED_CANDIDATE_SET_SCHEMAS
        or candidate.get("receipt_digest")
        != canonical_digest(candidate, digest_field="receipt_digest")
        or (candidate.get("claim_boundary") or {}).get("candidate_derived_layers_only") is not True
    ):
        raise RetainedSceneRenderPacketError(["retained_scene_render_candidate_set_invalid"])
    source = _absolute_record(
        candidate.get("source_standard_splat"),
        code="retained_scene_render_source_splat_invalid",
    )
    union = candidate.get("shared_scene_union")
    if not isinstance(union, Mapping):
        raise RetainedSceneRenderPacketError(["retained_scene_render_shared_union_missing"])
    outputs = union.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RetainedSceneRenderPacketError(["retained_scene_render_shared_outputs_missing"])
    retained = _relative_record(
        path.parent,
        outputs.get("retained_scene_gaussians"),
        code="retained_scene_render_shared_retained_splat_invalid",
    )
    deleted = _relative_record(
        path.parent,
        outputs.get("deleted_source_gaussians"),
        code="retained_scene_render_shared_deleted_source_layer_invalid",
    )
    deleted_indices_path = _relative_record(
        path.parent,
        outputs.get("deleted_source_indices"),
        code="retained_scene_render_shared_deleted_indices_invalid",
    )
    retained_indices_path = _relative_record(
        path.parent,
        outputs.get("retained_source_indices"),
        code="retained_scene_render_shared_retained_indices_invalid",
    )
    counts = union.get("counts")
    try:
        source_count = read_standard_3dgs_ply(source).count
        deleted_count = read_standard_3dgs_ply(deleted).count
        retained_count = read_standard_3dgs_ply(retained).count
    except (OSError, ValueError) as exc:
        raise RetainedSceneRenderPacketError(
            ["retained_scene_render_standard_ply_unreadable"]
        ) from exc
    try:
        deleted_indices = np.asarray(
            np.load(deleted_indices_path, allow_pickle=False), dtype=np.int64
        )
        retained_indices = np.asarray(
            np.load(retained_indices_path, allow_pickle=False), dtype=np.int64
        )
        byte_exact = verify_standard_3dgs_ply_subset_exact(source, retained, retained_indices).get(
            "retained_rows_byte_exact"
        )
        deleted_byte_exact = verify_standard_3dgs_ply_subset_exact(
            source, deleted, deleted_indices
        ).get("retained_rows_byte_exact")
    except (OSError, ValueError) as exc:
        raise RetainedSceneRenderPacketError(
            ["retained_scene_render_shared_indices_or_subset_unreadable"]
        ) from exc
    expected_indices = np.arange(source_count, dtype=np.int64)
    if (
        deleted_indices.ndim != 1
        or retained_indices.ndim != 1
        or not np.array_equal(np.unique(deleted_indices), deleted_indices)
        or not np.array_equal(np.unique(retained_indices), retained_indices)
        or np.intersect1d(deleted_indices, retained_indices, assume_unique=True).size
        or not np.array_equal(np.union1d(deleted_indices, retained_indices), expected_indices)
        or byte_exact is not True
        or deleted_byte_exact is not True
        or not isinstance(counts, Mapping)
        or counts.get("source") != source_count
        or counts.get("deleted_total") != int(deleted_indices.size)
        or counts.get("retained_total") != retained_count
        or deleted_count != int(deleted_indices.size)
        or retained_count <= 0
        or deleted_count <= 0
    ):
        raise RetainedSceneRenderPacketError(
            ["retained_scene_render_shared_retained_scene_not_byte_exact"]
        )
    tasks = candidate.get("task_candidates")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= MAX_REPLACEMENT_OBJECTS:
        raise RetainedSceneRenderPacketError(["retained_scene_render_task_count_invalid"])
    task_paths: dict[str, tuple[Path, dict[str, Any]]] = {}
    for row in tasks:
        if not isinstance(row, Mapping):
            raise RetainedSceneRenderPacketError(["retained_scene_render_task_row_invalid"])
        task_id = str(row.get("task_id") or "").strip()
        freeze_path = _absolute_record(
            row.get("task_freeze"), code="retained_scene_render_task_freeze_invalid"
        )
        freeze = validate_task_freeze(
            _read(freeze_path, code="retained_scene_render_task_freeze_invalid")
        )
        if (
            not task_id
            or task_id in task_paths
            or freeze.get("task_id") != task_id
            or freeze.get("task_freeze_digest") != row.get("task_freeze_digest")
        ):
            raise RetainedSceneRenderPacketError(["retained_scene_render_task_freeze_join_invalid"])
        task_paths[task_id] = (freeze_path, freeze)
    return candidate, source, deleted, retained, task_paths


def _camera_contract(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetainedSceneRenderPacketError(
            ["retained_scene_render_camera_contract_unreadable"]
        ) from exc
    if not isinstance(rows, list) or not rows:
        raise RetainedSceneRenderPacketError(["retained_scene_render_camera_contract_invalid"])
    seen: set[str] = set()
    dimensions: set[tuple[int, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise RetainedSceneRenderPacketError(["retained_scene_render_camera_contract_invalid"])
        camera_id = str(row.get("camera_id") or "")
        intrinsics = row.get("intrinsics")
        matrix = row.get("T_world_camera_provider_frame")
        if (
            not camera_id
            or camera_id in seen
            or not isinstance(intrinsics, Mapping)
            or not isinstance(matrix, list)
        ):
            raise RetainedSceneRenderPacketError(["retained_scene_render_camera_contract_invalid"])
        try:
            width, height = int(intrinsics["width"]), int(intrinsics["height"])
            _ = (
                float(intrinsics["fx"]),
                float(intrinsics["fy"]),
                float(intrinsics["cx"]),
                float(intrinsics["cy"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RetainedSceneRenderPacketError(
                ["retained_scene_render_camera_contract_invalid"]
            ) from exc
        if width <= 0 or height <= 0:
            raise RetainedSceneRenderPacketError(["retained_scene_render_camera_contract_invalid"])
        seen.add(camera_id)
        dimensions.add((width, height))
    if len(dimensions) != 1:
        raise RetainedSceneRenderPacketError(["retained_scene_render_camera_dimensions_mixed"])
    width, height = next(iter(dimensions))
    return [dict(row) for row in rows], width, height


def _git_identity(repo: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    if run("status", "--porcelain", "--untracked-files=all"):
        raise RetainedSceneRenderPacketError(["retained_scene_render_repository_not_clean"])
    commit, tree = run("rev-parse", "HEAD"), run("rev-parse", "HEAD^{tree}")
    if len(commit) != 40 or len(tree) != 40:
        raise RetainedSceneRenderPacketError(
            ["retained_scene_render_repository_identity_unavailable"]
        )
    return {"commit": commit, "tree": tree, "tracked_files_clean": True}


def _copy_tree(source: Path, destination: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise RetainedSceneRenderPacketError(["retained_scene_render_vendor_symlink_forbidden"])
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        _link_or_copy(path, target)
        files.append(_record(target, root=destination))
    if not files:
        raise RetainedSceneRenderPacketError(["retained_scene_render_vendor_package_empty"])
    return files


def _write_deterministic_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.resolve() == destination.resolve() or not path.is_file() or path.is_symlink():
                continue
            info = zipfile.ZipInfo(
                path.relative_to(source).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.external_attr = (stat.S_IMODE(path.stat().st_mode) | stat.S_IFREG) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as input_stream, archive.open(info, "w") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)


def build_retained_scene_gpu_render_bundle(
    *,
    request_path: str | Path,
    repo_root: str | Path,
    job_dir: str | Path,
    scene_input_root: str | Path | None = None,
) -> dict[str, Any]:
    """Copy verified derived inputs into an immutable, zero-network GPU bundle.

    ``scene_input_root`` is where the private scene bytes are staged on this
    machine. A request that names its inputs relative to that root, and to the
    checkout, rebuilds anywhere; one that names absolute paths rebuilds only
    where it was written.
    """

    request_file = _file(request_path, code="retained_scene_render_request_missing")
    request = build_retained_scene_gpu_render_request(
        _read(request_file, code="retained_scene_render_request_unreadable")
    )
    repo = Path(repo_root).expanduser().resolve()
    identity = _git_identity(repo)
    job = Path(job_dir).expanduser().resolve()
    if job.exists() and any(job.iterdir()):
        raise RetainedSceneRenderPacketError(["retained_scene_render_job_dir_not_empty"])
    scene_root = (
        _directory(scene_input_root, code="retained_scene_render_scene_input_root_invalid")
        if scene_input_root is not None
        else None
    )
    candidate_path = _resolve_request_input(
        request["candidate_set_path"],
        repo=repo,
        scene_root=scene_root,
        code="retained_scene_render_candidate_set_missing",
    )
    candidate, source_ply, deleted_ply, retained_ply, task_paths = _validate_candidate_set(
        candidate_path
    )
    authority_path = _resolve_request_input(
        request["execution_authority_path"],
        repo=repo,
        scene_root=scene_root,
        code="retained_scene_render_execution_authority_missing",
    )
    authority = _validate_authority(authority_path)
    requested = {str(row["task_id"]) for row in request["task_lanes"]}
    if requested != set(task_paths):
        raise RetainedSceneRenderPacketError(["retained_scene_render_candidate_task_set_mismatch"])
    vendor_root = _resolve_request_input(
        request["renderer_vendor_root"],
        repo=repo,
        scene_root=scene_root,
        code="retained_scene_render_vendor_root_invalid",
        directory=True,
    )
    job.mkdir(parents=True)
    runtime = job / "provider_runtime"
    input_root = runtime / "input"
    input_root.mkdir(parents=True)
    deleted_copy = input_root / "shared_deleted_source_layer.ply"
    retained_copy = input_root / "shared_retained_scene.ply"
    _link_or_copy(deleted_ply, deleted_copy)
    _link_or_copy(retained_ply, retained_copy)
    candidate_copy = input_root / "direct_evidence_successor_set.json"
    _link_or_copy(candidate_path, candidate_copy)
    task_bindings: dict[str, dict[str, Any]] = {}
    freezes = input_root / "task_freezes"
    freezes.mkdir()
    for task_id, (freeze_path, freeze) in task_paths.items():
        copied_freeze = freezes / f"{task_id}.json"
        _link_or_copy(freeze_path, copied_freeze)
        removal = freeze["removal_plan"]
        task_bindings[task_id] = {
            "task_freeze": _record(copied_freeze, root=runtime),
            "task_freeze_digest": freeze["task_freeze_digest"],
            "removal_id": removal["removal_id"],
            "mask_set_id": removal["mask_set_id"],
            "replacement_asset_id": removal["replacement_asset_id"],
        }
    lanes: list[dict[str, Any]] = []
    for lane in sorted(request["task_lanes"], key=lambda row: str(row["task_id"])):
        task_id = str(lane["task_id"])
        camera_path = _resolve_request_input(
            lane["camera_contract_path"],
            repo=repo,
            scene_root=scene_root,
            code="retained_scene_render_camera_contract_missing",
        )
        cameras, width, height = _camera_contract(camera_path)
        lane_root = input_root / "cameras" / task_id
        lane_root.mkdir(parents=True)
        copied = lane_root / "cameras.v1.json"
        _link_or_copy(camera_path, copied)
        lanes.append(
            {
                "task_id": task_id,
                **task_bindings[task_id],
                "camera_contract": _record(copied, root=runtime),
                "camera_count": len(cameras),
                "dimensions": {"width": width, "height": height},
                "render_variants": [
                    {"layer": "shared_deleted_source_layer", "background_rgb": "#000000"},
                    {"layer": "shared_deleted_source_layer", "background_rgb": "#ffffff"},
                    {"layer": "shared_retained_scene", "background_rgb": "#000000"},
                ],
            }
        )
    renderer = runtime / "renderer"
    for relative in (
        "render_splat.mjs",
        "harness.html",
        "package.json",
        "package-lock.json",
        "src/render_entry.mjs",
    ):
        source = repo / "tools/splat_render" / relative
        if not source.is_file() or source.is_symlink():
            raise RetainedSceneRenderPacketError(["retained_scene_render_renderer_source_missing"])
        target = renderer / relative
        _link_or_copy(source, target)
    vendor_files: dict[str, list[dict[str, Any]]] = {}
    for package in _VENDOR_PACKAGES:
        source = vendor_root / package
        target = renderer / "node_modules" / package
        if not source.is_dir() or source.is_symlink():
            raise RetainedSceneRenderPacketError(["retained_scene_render_vendor_package_missing"])
        vendor_files[package] = _copy_tree(source, target)
    for source_relative, destination_name in (
        (
            "scripts/run_adp_retained_scene_render_provider_runtime.sh",
            Path(ENTRYPOINT).name,
        ),
        (
            "scripts/adp_retained_scene_render_provider_runner.mjs",
            "adp_retained_scene_render_provider_runner.mjs",
        ),
    ):
        source = repo / source_relative
        if not source.is_file() or source.is_symlink():
            raise RetainedSceneRenderPacketError(["retained_scene_render_provider_runner_missing"])
        target = runtime / destination_name
        # The runtime shell is chmod-ed below; copying avoids a hard-link mode
        # mutation escaping into the sealed checkout.
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if source.suffix == ".sh":
            target.chmod(target.stat().st_mode | stat.S_IXUSR)
    authority_copy = runtime / "execution_authority.json"
    _link_or_copy(authority_path, authority_copy)
    # The request manifest is copied in for the same reason the authority is:
    # a receipt read on another host can only resolve what travelled with it.
    request_copy = runtime / "source_request_manifest.json"
    _link_or_copy(request_file, request_copy)
    renderer_identity = {
        "repository": identity,
        "harness_sha256": _sha256(renderer / "render_splat.mjs"),
        "render_entry_sha256": _sha256(renderer / "src/render_entry.mjs"),
        "package_manifest_sha256": _sha256(renderer / "package.json"),
        "package_lock_sha256": _sha256(renderer / "package-lock.json"),
        "graphics_backend": "egl",
        "vendor_packages": {name: len(rows) for name, rows in vendor_files.items()},
    }
    render_request = {
        "schema_version": "adp009d_retained_scene_gpu_renderer_runtime_request.v1",
        "candidate_set": _record(candidate_copy, root=runtime),
        "candidate_set_digest": candidate["receipt_digest"],
        "execution_authority": _record(authority_copy, root=runtime),
        "source_standard_splat_provenance": {
            **_record(source_ply),
            "gaussian_count": read_standard_3dgs_ply(source_ply).count,
            "included_in_provider_bundle": False,
        },
        "shared_deleted_source_layer": {
            **_record(deleted_copy, root=runtime),
            "gaussian_count": read_standard_3dgs_ply(deleted_ply).count,
            "source_layer_role": "shared_deleted_source_union",
        },
        "shared_retained_scene": _record(retained_copy, root=runtime),
        "shared_retained_gaussian_count": read_standard_3dgs_ply(retained_ply).count,
        "lanes": lanes,
        "renderer_identity": renderer_identity,
        "raw_dataset_bytes_upload_authorized": False,
        "private_scene_derived_upload_only": True,
        "provider_training_authorized": False,
        "publication_authorized": False,
        "request_digest": request["request_digest"],
    }
    render_request["runtime_request_digest"] = canonical_digest(
        render_request, digest_field="runtime_request_digest"
    )
    (runtime / "render_request.json").write_text(
        canonical_json(render_request) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "status": "ready",
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "probe_kind": PROBE_KIND,
        "container_image": DEFAULT_IMAGE,
        "blueprint_commit": identity["commit"],
        "blueprint_tree": identity["tree"],
        # Each of these records both where the bytes were read and where they
        # sit inside the job directory. The absolute path is provenance; the
        # relative one is what lets another host resolve the same bytes.
        "request": {
            **_record(request_file),
            "relative_path": request_copy.relative_to(job).as_posix(),
        },
        "candidate_set_digest": candidate["receipt_digest"],
        "execution_authority": {
            **_record(authority_path),
            "relative_path": authority_copy.relative_to(job).as_posix(),
            "authority_digest": authority["authority_digest"],
        },
        "source_standard_splat_provenance": render_request[
            "source_standard_splat_provenance"
        ],
        "shared_deleted_source_layer": {
            **_record(deleted_copy, root=runtime),
            "deleted_gaussian_count": read_standard_3dgs_ply(deleted_ply).count,
            "source_layer_role": "shared_deleted_source_union",
        },
        "shared_retained_scene": {
            **_record(retained_copy, root=runtime),
            "retained_gaussian_count": read_standard_3dgs_ply(retained_ply).count,
        },
        "task_lanes": lanes,
        "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
        "source_pair_per_task": True,
        "deleted_source_layer_pair_per_task": True,
        "retained_frame_per_task": True,
        "renderer_identity": renderer_identity,
        "provider_network_dependency_install_required": False,
        "raw_interiorgs_downloaded_bytes_included": False,
        "private_scene_derived_standard_splats_included": True,
        "automatic_paid_retry_allowed": False,
        "provider_zero_required_after_return": True,
        "hard_total_spend_cap_usd": authority["paid_compute"]["hard_total_spend_cap_usd"],
        "blockers": [],
    }
    (runtime / "adp_retained_scene_gpu_render_manifest.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    bundle_path = job / "adp_retained_scene_gpu_render_bundle.zip"
    partial_bundle_path = job / ".adp_retained_scene_gpu_render_bundle.partial.zip"
    try:
        _write_deterministic_zip(job, partial_bundle_path)
        with zipfile.ZipFile(partial_bundle_path) as archive:
            if archive.testzip() is not None:
                raise RetainedSceneRenderPacketError(
                    ["retained_scene_render_bundle_integrity_invalid"]
                )
        partial_bundle_path.replace(bundle_path)
    finally:
        if partial_bundle_path.exists():
            partial_bundle_path.unlink()
    rehearsal = rehearse_provider_bundle_entrypoint(
        bundle_path=bundle_path,
        entrypoint_relative_path=ENTRYPOINT,
        evidence_path=job / "adp_retained_scene_gpu_render_exact_bundle_rehearsal.json",
    )
    receipt = {
        **manifest,
        "bundle_path": str(bundle_path),
        # The receipt is written beside the archive, so this resolves wherever
        # the pair is staged. The absolute path above records only where it was
        # built.
        "bundle_relative_path": bundle_path.relative_to(job).as_posix(),
        "bundle_sha256": _sha256(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "exact_bundle_entrypoint_rehearsal": rehearsal,
    }
    (job / "adp_retained_scene_gpu_render_bundle_receipt.json").write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )
    return receipt


__all__ = [
    "BUNDLE_SCHEMA",
    "DEFAULT_IMAGE",
    "ENTRYPOINT",
    "PROBE_KIND",
    "REQUEST_SCHEMA",
    "RetainedSceneRenderPacketError",
    "build_retained_scene_gpu_render_bundle",
    "build_retained_scene_gpu_render_request",
]
