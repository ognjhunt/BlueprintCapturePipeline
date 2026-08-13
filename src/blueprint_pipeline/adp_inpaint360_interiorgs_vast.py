"""Build and execute one immutable Inpaint360GS InteriorGS packet on Vast."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

from .task_evaluation_artifact_manifest import seal_lane_terminal_artifacts
from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import PaidResourceAdmissionGrant
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers
from .vast_provider_adapter import run_vast_provider_adapter
from .vast_session_budget_contract import attempt_estimated_cost, attempt_runtime_seconds
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


PROBE_KIND = "adp-inpaint360-interiorgs"
PROVIDER_BUNDLE_KIND = "adp_inpaint360_interiorgs"
RESULT_SCHEMA_VERSION = "adp_inpaint360_interiorgs_vast_run.v1"
SOURCE_REPOSITORY = "https://github.com/dfki-av/Inpaint360GS"
SOURCE_COMMIT = "d54c893285c6cb27788e05cce607e7d3cca6388a"
SOURCE_TREE = "671626f4825cbf3d7c1ca37cc97a153d45e49b1c"
SOURCE_LICENSE_SHA256 = "sha256:41d805773f2aa0b36c2fb69491f64c3079fe3e0671c9848680645fc9e65d5a10"
LAMA_SOURCE_REPOSITORY = "https://github.com/advimman/lama"
LAMA_SOURCE_COMMIT = "786f5936b27fb3dacd2b1ad799e4de968ea697e7"
LAMA_SOURCE_TREE = "25f9902ca0c2ec4bf6c31c2b4427f0a4f05f2fd1"
LAMA_SOURCE_LICENSE_SHA256 = (
    "sha256:4ceeeac5a802e86c413c22b16cce8e9a22027b0250c97e6f8ac97c14cf0542c0"
)
LAMA_REQUIRED_RUNTIME_FILES = (
    "saicinpainting/training/data/__init__.py",
    "saicinpainting/training/data/aug.py",
    "saicinpainting/training/data/datasets.py",
    "saicinpainting/training/data/masks.py",
)
DEFAULT_IMAGE = (
    "docker.io/nvidia/cuda@sha256:60eda04ab6790aa76d73bf0df245b361eabc6d8f7b6f6cf9846c70f399b9a1eb"
)
PREREQUISITE_RECEIPT_DIGEST = (
    "sha256:ed309230e5dd216117789ef3abab947e3a0dabfe66afacd7b0de7732217b4902"
)
BIG_LAMA_SHA256 = "sha256:d7161bba4d68b438f9fa7f09dcb750a223804c300c68d214a5e0be16251fba8d"
BIG_LAMA_SIZE_BYTES = 381_428_720
VGG16_WEIGHTS_FILENAME = "vgg16-397923af.pth"
VGG16_WEIGHTS_SOURCE_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"
VGG16_WEIGHTS_SHA256 = "sha256:397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
VGG16_WEIGHTS_SIZE_BYTES = 553_433_881
MIN_RASTERIZER_COMPUTE_CAP = 890
PROVIDER_EXECUTION_TIMEOUT_SECONDS = 14_400
PROVIDER_HEARTBEAT_NO_PROGRESS_SECONDS = 1800
INPAINT360_GPU_SELECTION_POLICY = {
    "policy_id": "inpaint360_rtx_4090_observed_control",
    "allowed_gpu_keywords": ("RTX 4090",),
    "denied_gpu_keywords": (),
    "reason": "exact InteriorGS cameras passed only on the observed RTX 4090 control",
}
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/inpaint360-interiorgs"
_VAST_MUTATION_ENV = (
    "BLUEPRINT_ALLOW_VAST_API_CALLS",
    "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH",
)
_VAST_SINGLE_ATTEMPT_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("adp_inpaint360_json_object_required")
    return value


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_files(repo: Path) -> list[tuple[str, Path]]:
    return [
        (relative, repo / relative)
        for relative in _git(repo, "ls-files").splitlines()
        if relative and ((repo / relative).is_file() or (repo / relative).is_symlink())
    ]


def _file_manifest(rows: Sequence[tuple[str, Path]]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for relative, path in rows:
        if path.is_symlink():
            manifest.append({"path": relative, "type": "symlink", "target": os.readlink(path)})
        else:
            manifest.append(
                {
                    "path": relative,
                    "type": "file",
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return manifest


def _git_archive(repo: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", "-o", str(destination), "HEAD"],
        check=True,
    )


def _deterministic_zip(rows: Sequence[tuple[str, Path]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, source in sorted(rows):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = (0o755 if source.stat().st_mode & stat.S_IXUSR else 0o644) << 16
            archive.writestr(info, source.read_bytes())


def _write_executable(destination: Path, source: Path) -> None:
    shutil.copy2(source, destination)
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR)


def _validate_prerequisite(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    if (
        canonical_digest(receipt, digest_field="receipt_digest") != receipt.get("receipt_digest")
        or receipt.get("receipt_digest") != PREREQUISITE_RECEIPT_DIGEST
    ):
        raise ValueError("adp_inpaint360_prerequisite_receipt_digest_mismatch")
    method = (receipt.get("methods") or {}).get("inpaint360_author_smoke") or {}
    artifacts = method.get("artifacts") or []
    big_lama = next(
        (row for row in artifacts if row.get("artifact_id") == "big_lama_author_linked_archive"),
        None,
    )
    if (
        not isinstance(big_lama, Mapping)
        or big_lama.get("rights_established") is not True
        or big_lama.get("size_bytes") != BIG_LAMA_SIZE_BYTES
        or big_lama.get("sha256") != BIG_LAMA_SHA256
    ):
        raise ValueError("adp_inpaint360_big_lama_rights_or_identity_missing")
    return big_lama


def _adapter_target_binding(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the scene/task identity that owns one Inpaint360 packet."""

    scene = receipt.get("scene") or {}
    adapter = receipt.get("adapter") or {}
    scene_id = str(scene.get("publisher_scene_id") or "")
    target_instance_id = str(scene.get("target_instance_id") or "")
    task_value = scene.get("task_id")
    task_id = str(task_value or "") or None
    if not scene_id.isdigit() or not target_instance_id.isdigit():
        raise ValueError("adp_inpaint360_scene_target_identity_invalid")
    if task_id is not None and not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", task_id):
        raise ValueError("adp_inpaint360_task_id_invalid")
    expected_config_id = (
        f"{scene_id}__{target_instance_id}__{task_id}" if task_id else scene_id
    )
    method_config_id = str(adapter.get("method_config_id") or scene_id)
    if method_config_id != expected_config_id:
        raise ValueError("adp_inpaint360_method_config_identity_mismatch")
    method_instance_id = adapter.get("target_method_instance_id")
    if (
        isinstance(method_instance_id, bool)
        or not isinstance(method_instance_id, int)
        or not 1 <= method_instance_id <= 255
    ):
        raise ValueError("adp_inpaint360_method_target_identity_invalid")
    return {
        "scene_id": scene_id,
        "target_instance_id": target_instance_id,
        "task_id": task_id,
        "method_config_id": method_config_id,
        "target_method_instance_id": method_instance_id,
    }


def _validate_adapter(
    receipt: Mapping[str, Any], *, adapter_root: Path
) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    if canonical_digest(receipt, digest_field="receipt_digest") != receipt.get("receipt_digest"):
        raise ValueError("adp_inpaint360_adapter_receipt_digest_mismatch")
    source = receipt.get("source") or {}
    adapter = receipt.get("adapter") or {}
    if receipt.get("status") != "prepared_unexecuted":
        raise ValueError("adp_inpaint360_adapter_not_prepared")
    target = _adapter_target_binding(receipt)
    if source.get("commit") != SOURCE_COMMIT or source.get("tree") != SOURCE_TREE:
        raise ValueError("adp_inpaint360_adapter_source_identity_mismatch")
    if adapter.get("target_object_radius_derivation") != "max_distance_from_metric_obb_center":
        raise ValueError("adp_inpaint360_target_radius_not_evidence_derived")
    radius = adapter.get("target_object_radius_m")
    if not isinstance(radius, (int, float)) or not 0.0 < float(radius) < 1.0:
        raise ValueError("adp_inpaint360_target_radius_invalid")
    corners = adapter.get("target_obb_corners_m")
    if (
        not isinstance(corners, list)
        or len(corners) != 8
        or any(
            not isinstance(row, list)
            or len(row) != 3
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in row)
            for row in corners
        )
    ):
        raise ValueError("adp_inpaint360_target_obb_invalid")
    if adapter.get("target_removal_volume_contract") != (
        "gaussian_center_inside_exact_publisher_obb"
    ):
        raise ValueError("adp_inpaint360_target_removal_volume_unbound")
    rows: list[tuple[str, Path]] = []
    for record in adapter.get("staged_artifacts") or []:
        relative = str(record.get("relative_path") or "")
        path = (adapter_root / relative).resolve()
        if adapter_root != path and adapter_root not in path.parents:
            raise ValueError("adp_inpaint360_adapter_artifact_outside_root")
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise ValueError("adp_inpaint360_adapter_artifact_changed")
        rows.append((relative, path))
    required = {
        "config/distill.json",
        f"config/object_removal/blueprint/{target['method_config_id']}.json",
        f"config/object_inpaint/blueprint/{target['method_config_id']}.json",
        "vanilla_3dgs/cfg_args",
        "vanilla_3dgs/point_cloud/iteration_30000/point_cloud.ply",
    }
    if not required.issubset({relative for relative, _ in rows}):
        raise ValueError("adp_inpaint360_adapter_required_artifact_missing")
    return rows, target


def build_inpaint360_interiorgs_bundle(
    *,
    repo_root: str | Path,
    inpaint360_root: str | Path,
    lama_root: str | Path,
    adapter_root: str | Path,
    adapter_receipt_path: str | Path,
    prerequisite_receipt_path: str | Path,
    big_lama_path: str | Path,
    vgg16_weights_path: str | Path,
    job_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build one exact scene/method packet without allocating a provider."""

    repo = Path(repo_root).expanduser().resolve()
    source = Path(inpaint360_root).expanduser().resolve()
    lama_source = Path(lama_root).expanduser().resolve()
    packet = Path(adapter_root).expanduser().resolve()
    adapter_receipt_file = Path(adapter_receipt_path).expanduser().resolve()
    prerequisite_file = Path(prerequisite_receipt_path).expanduser().resolve()
    big_lama = Path(big_lama_path).expanduser().resolve()
    vgg16_weights = Path(vgg16_weights_path).expanduser().resolve()
    job = Path(job_dir).expanduser().resolve()
    if job.exists() and any(job.iterdir()):
        raise ValueError("adp_inpaint360_bundle_job_dir_not_empty")
    runtime = job / "provider_runtime"
    ensure_dir(runtime)
    try:
        blueprint_commit = _git(repo, "rev-parse", "HEAD")
        blueprint_tree = _git(repo, "rev-parse", "HEAD^{tree}")
        blueprint_dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
    except subprocess.CalledProcessError as exc:
        raise ValueError("adp_inpaint360_blueprint_repository_identity_missing") from exc
    if blueprint_dirty:
        raise ValueError("adp_inpaint360_blueprint_repository_tracked_state_dirty")
    if (
        _git(source, "rev-parse", "HEAD") != SOURCE_COMMIT
        or _git(source, "rev-parse", "HEAD^{tree}") != SOURCE_TREE
        or _git(source, "status", "--porcelain", "--untracked-files=no")
    ):
        raise ValueError("adp_inpaint360_source_identity_mismatch")
    source_license = source / "LICENSE.txt"
    if not source_license.is_file() or _sha256(source_license) != SOURCE_LICENSE_SHA256:
        raise ValueError("adp_inpaint360_source_license_identity_mismatch")
    if (
        _git(lama_source, "rev-parse", "HEAD") != LAMA_SOURCE_COMMIT
        or _git(lama_source, "rev-parse", "HEAD^{tree}") != LAMA_SOURCE_TREE
        or _git(lama_source, "status", "--porcelain", "--untracked-files=no")
    ):
        raise ValueError("adp_inpaint360_lama_dependency_identity_mismatch")
    lama_license = lama_source / "LICENSE"
    if not lama_license.is_file() or _sha256(lama_license) != LAMA_SOURCE_LICENSE_SHA256:
        raise ValueError("adp_inpaint360_lama_dependency_license_identity_mismatch")
    lama_rows = [(relative, lama_source / relative) for relative in LAMA_REQUIRED_RUNTIME_FILES]
    if any(not path.is_file() for _, path in lama_rows):
        raise ValueError("adp_inpaint360_lama_dependency_runtime_file_missing")
    if any((source / "LaMa" / relative).exists() for relative, _ in lama_rows):
        raise ValueError("adp_inpaint360_lama_dependency_would_overwrite_publisher_source")
    prerequisite = _read_json(prerequisite_file)
    big_lama_authority = _validate_prerequisite(prerequisite)
    if (
        not big_lama.is_file()
        or big_lama.stat().st_size != BIG_LAMA_SIZE_BYTES
        or _sha256(big_lama) != BIG_LAMA_SHA256
    ):
        raise ValueError("adp_inpaint360_big_lama_bytes_changed")
    if (
        not vgg16_weights.is_file()
        or vgg16_weights.stat().st_size != VGG16_WEIGHTS_SIZE_BYTES
        or _sha256(vgg16_weights) != VGG16_WEIGHTS_SHA256
    ):
        raise ValueError("adp_inpaint360_vgg16_weights_bytes_changed")
    adapter_receipt = _read_json(adapter_receipt_file)
    adapter_rows, target = _validate_adapter(adapter_receipt, adapter_root=packet)
    source_rows = _tracked_files(source)
    source_manifest = _file_manifest(source_rows)
    lama_manifest = _file_manifest(lama_rows)
    adapter_manifest = _file_manifest(adapter_rows)
    _git_archive(source, runtime / "inpaint360gs_source.tar")
    _deterministic_zip(lama_rows, runtime / "lama_training_data.zip")
    _deterministic_zip(adapter_rows, runtime / "interiorgs_adapter.zip")
    shutil.copy2(big_lama, runtime / "big-lama.zip")
    shutil.copy2(vgg16_weights, runtime / VGG16_WEIGHTS_FILENAME)
    scripts = repo / "scripts"
    _write_executable(
        runtime / "run_adp_inpaint360_interiorgs_provider_runtime.sh",
        scripts / "run_adp_inpaint360_interiorgs_provider_runtime.sh",
    )
    for name in (
        "adp_inpaint360_interiorgs_provider_runner.py",
        "materialize_inpaint360_virtual_masks.py",
        "probe_inpaint360_camera_rasterizer.py",
    ):
        shutil.copy2(scripts / name, runtime / name)
    spec = {
        "schema_version": "adp_inpaint360_interiorgs_spec.v1",
        "scene_id": target["scene_id"],
        "target_instance_id": target["target_instance_id"],
        "task_id": target["task_id"],
        "method_config_id": target["method_config_id"],
        "target_semantic_label": adapter_receipt["scene"]["target_semantic_label"],
        "target_method_instance_id": target["target_method_instance_id"],
        "target_object_radius_m": adapter_receipt["adapter"]["target_object_radius_m"],
        "target_obb_corners_m": adapter_receipt["adapter"]["target_obb_corners_m"],
        "target_removal_volume_contract": adapter_receipt["adapter"][
            "target_removal_volume_contract"
        ],
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "tree": SOURCE_TREE,
            "files": source_manifest,
        },
        "nested_dependencies": {
            "lama": {
                "repository": LAMA_SOURCE_REPOSITORY,
                "commit": LAMA_SOURCE_COMMIT,
                "tree": LAMA_SOURCE_TREE,
                "license": "Apache-2.0",
                "license_sha256": LAMA_SOURCE_LICENSE_SHA256,
                "files": lama_manifest,
                "materialization": "add_missing_publisher_runtime_modules_without_overwrite",
            }
        },
        "blueprint_repository": {
            "commit": blueprint_commit,
            "tree": blueprint_tree,
            "tracked_state": "clean",
        },
        "adapter": {
            "receipt_digest": adapter_receipt["receipt_digest"],
            "files": adapter_manifest,
        },
        "big_lama": {
            "size_bytes": BIG_LAMA_SIZE_BYTES,
            "sha256": BIG_LAMA_SHA256,
            "rights_authority_id": big_lama_authority["rights_authority_id"],
        },
        "runtime_dependencies": {
            "torchvision_vgg16_imagenet1k_v1": {
                "filename": VGG16_WEIGHTS_FILENAME,
                "source_url": VGG16_WEIGHTS_SOURCE_URL,
                "size_bytes": VGG16_WEIGHTS_SIZE_BYTES,
                "sha256": VGG16_WEIGHTS_SHA256,
                "materialization": "bundle_to_controlled_torch_hub_cache_before_method_execution",
            }
        },
        "runtime": {
            "container_image": DEFAULT_IMAGE,
            "main_python": "3.10",
            "main_torch": "2.0.0+cu118",
            "lama_python": "3.8",
            "lama_torch": "1.8.0+cu111",
            "publisher_documented_lama_python": "3.6.13",
            "publisher_documented_lama_cuda_toolkit": "10.2",
            "lama_environment_relation": "compatibility_environment_not_exact_publisher_conda_env",
            "source_modifications": [],
            "source_materialization_additions": list(LAMA_REQUIRED_RUNTIME_FILES),
            "virtual_view_count": 30,
            "expected_input_camera_count": 8,
            "input_resolution": "2048x1536",
            "method_resolution_argument": 2,
            "method_input_resolution": "1024x768",
            "mask_association_mode": (
                "pre_registered_single_target_resolution_divisor_2"
            ),
        },
        "claim_boundary": {
            "rendered_frames_have_no_hidden_background_truth": True,
            "publisher_splat_is_not_metric_surface_truth": True,
            "internal_synthetic_scene_edit_only": True,
            "replacement_or_physics_result": False,
        },
    }
    write_json(runtime / "execution_spec.json", spec)
    entrypoint = runtime / "run_adp_inpaint360_interiorgs_provider_runtime.sh"
    runner = runtime / "adp_inpaint360_interiorgs_provider_runner.py"
    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        entrypoint_text=entrypoint.read_text(encoding="utf-8"),
        runner_text=runner.read_text(encoding="utf-8"),
    )
    manifest = {
        "schema_version": "adp_inpaint360_interiorgs_provider_bundle.v1",
        "generated_at": generated_at or utc_now_iso(),
        "status": "ready" if not blockers else "blocked",
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "source_license": "Apache-2.0",
        "source_license_sha256": SOURCE_LICENSE_SHA256,
        "blueprint_repository_commit": blueprint_commit,
        "blueprint_repository_tree": blueprint_tree,
        "blueprint_repository_tracked_state": "clean",
        "source_archive_sha256": _sha256(runtime / "inpaint360gs_source.tar"),
        "source_manifest_digest": canonical_digest({"files": source_manifest}),
        "lama_source_commit": LAMA_SOURCE_COMMIT,
        "lama_source_tree": LAMA_SOURCE_TREE,
        "lama_source_license": "Apache-2.0",
        "lama_source_license_sha256": LAMA_SOURCE_LICENSE_SHA256,
        "lama_runtime_archive_sha256": _sha256(runtime / "lama_training_data.zip"),
        "lama_runtime_manifest_digest": canonical_digest({"files": lama_manifest}),
        "adapter_receipt_digest": adapter_receipt["receipt_digest"],
        "target_binding": target,
        "prerequisite_receipt_digest": prerequisite["receipt_digest"],
        "adapter_archive_sha256": _sha256(runtime / "interiorgs_adapter.zip"),
        "adapter_manifest_digest": canonical_digest({"files": adapter_manifest}),
        "big_lama_sha256": BIG_LAMA_SHA256,
        "vgg16_weights_source_url": VGG16_WEIGHTS_SOURCE_URL,
        "vgg16_weights_sha256": VGG16_WEIGHTS_SHA256,
        "vgg16_weights_size_bytes": VGG16_WEIGHTS_SIZE_BYTES,
        "container_image": DEFAULT_IMAGE,
        "runtime_environment_claim": (
            "exact_main_environment_method_source_unchanged_pinned_lama_dependency_materialized"
        ),
        "expected_output_filename": "adp_inpaint360_interiorgs_result.json",
        "retry_cap": 0,
        "local_bundle_ready_for_remote_staging": not blockers,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    write_json(runtime / "adp_inpaint360_interiorgs_provider_manifest.json", manifest)
    bundle = job / "adp_inpaint360_interiorgs_provider_runtime_bundle.zip"
    _deterministic_zip(
        [(path.relative_to(job).as_posix(), path) for path in runtime.rglob("*") if path.is_file()],
        bundle,
    )
    receipt = {
        **manifest,
        "bundle_path": str(bundle),
        "bundle_sha256": _sha256(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
    }
    write_json(job / "adp_inpaint360_interiorgs_bundle_receipt.json", receipt)
    return receipt


def _remaining_minutes(
    *, job: Path, hard_cap_usd: float, hard_ttl_seconds: int, max_hourly_rate_usd: float
) -> int:
    ledger = _read_json(job / "adp_inpaint360_vast_session_budget.json")
    attempts = [row for row in ledger.get("attempts") or [] if isinstance(row, Mapping)]
    prior_seconds = sum(attempt_runtime_seconds(row) for row in attempts)
    prior_cost = sum(attempt_estimated_cost(row) for row in attempts)
    runtime_minutes = math.floor(max(0.0, hard_ttl_seconds - prior_seconds) / 60.0)
    spend_minutes = math.floor(max(0.0, hard_cap_usd - prior_cost) * 60.0 / max_hourly_rate_usd)
    return max(0, min(runtime_minutes, spend_minutes))


def _bundle_target_key_prefix(bundle: Mapping[str, Any]) -> str:
    """Create a provider prefix unique to one sealed scene/task target."""

    target = bundle.get("target_binding")
    if not isinstance(target, Mapping):
        raise ValueError("adp_inpaint360_bundle_target_binding_invalid")
    scene_id = str(target.get("scene_id") or "")
    target_instance_id = str(target.get("target_instance_id") or "")
    config_id = str(target.get("method_config_id") or "")
    task_id = target.get("task_id")
    if (
        not scene_id.isdigit()
        or not target_instance_id.isdigit()
        or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", config_id)
        or task_id is not None
        and not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", str(task_id))
    ):
        raise ValueError("adp_inpaint360_bundle_target_binding_invalid")
    return f"{DEFAULT_KEY_PREFIX}/{scene_id}/{target_instance_id}/{config_id}"


def _extract(path: Path, destination: Path) -> dict[str, Any]:
    blockers: list[str] = []
    if not path.is_file():
        return {"status": "blocked", "blockers": ["inpaint360_provider_output_zip_missing"]}
    if destination.exists() and any(destination.iterdir()):
        return {"status": "blocked", "blockers": ["inpaint360_output_destination_not_empty"]}
    ensure_dir(destination)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    blockers.append("inpaint360_provider_output_zip_path_traversal")
            if not blockers:
                archive.extractall(destination)
    except (OSError, zipfile.BadZipFile):
        blockers.append("inpaint360_provider_output_zip_invalid")
    result_path = destination / "adp_inpaint360_interiorgs_result.json"
    execution = _read_json(result_path)
    if not execution:
        blockers.append("inpaint360_provider_result_missing")
    return {
        "status": "completed" if not blockers else "blocked",
        "result_path": str(result_path),
        "execution": execution,
        "blockers": sorted(set(blockers)),
    }


@contextmanager
def _authority_environment():
    names = (*_VAST_MUTATION_ENV, _VAST_SINGLE_ATTEMPT_ENV)
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in _VAST_MUTATION_ENV:
            os.environ[name] = "1"
        os.environ[_VAST_SINGLE_ATTEMPT_ENV] = "0"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_inpaint360_interiorgs_vast(
    *,
    job_dir: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float = 1.5,
    hard_cap_usd: float = 6.0,
    hard_ttl_seconds: int = 14_400,
    public_image: str = DEFAULT_IMAGE,
    machine_avoidlist_path: str | Path | None = None,
    allowed_active_instance_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Run one immutable InteriorGS edit with no automatic retry."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle = dict(prepared_bundle)
    bundle_path = Path(str(bundle.get("bundle_path") or "")).resolve()
    if public_image != DEFAULT_IMAGE:
        raise ValueError("adp_inpaint360_container_image_not_frozen")
    if (
        bundle.get("status") != "ready"
        or not bundle_path.is_file()
        or _sha256(bundle_path) != bundle.get("bundle_sha256")
    ):
        raise ValueError("adp_inpaint360_prepared_bundle_binding_invalid")
    key_prefix = _bundle_target_key_prefix(bundle)
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "dry_run_ready",
            "bundle": bundle,
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
        }
        write_json(job / "adp_inpaint360_interiorgs_vast_result.json", result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("adp_inpaint360_paid_resource_admission_grant_missing")
    remaining_minutes = _remaining_minutes(
        job=job,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        max_hourly_rate_usd=max_hourly_rate_usd,
    )
    if remaining_minutes < 120:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": ["adp_inpaint360_budget_below_minimum_live_window"],
        }
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=str(bundle_path),
        key_prefix=key_prefix,
        expiration_seconds=max(hard_ttl_seconds + 1800, 18_000),
    )
    if staging.get("status") != "completed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": staging.get("blockers") or ["inpaint360_object_store_staging_blocked"],
        }
    provider_run = job / "vast_provider_run"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    adapter: dict[str, Any] = {}
    try:
        with _authority_environment():
            adapter = run_vast_provider_adapter(
                job_dir=provider_run,
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=max_hourly_rate_usd,
                target_spend_usd=hard_cap_usd,
                hard_cap_usd=hard_cap_usd,
                max_live_minutes=remaining_minutes,
                session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=public_image,
                isaac_image=public_image,
                ngc_image_login_mode="never",
                provider_bundle=bundle_path,
                provider_bundle_url=(staging_dir / "provider_bundle_url.txt").read_text().strip(),
                provider_output_put_url=(staging_dir / "provider_output_put_url.txt")
                .read_text()
                .strip(),
                provider_output_get_url=(staging_dir / "provider_output_get_url.txt")
                .read_text()
                .strip(),
                provider_runtime_output_zip=output_zip,
                enable_isaac_smoke=False,
                enable_blueprint_bundle=True,
                provider_bundle_kind=PROVIDER_BUNDLE_KIND,
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=False,
                disk_gb=192,
                min_gpu_ram_mb=24_000,
                min_compute_cap=MIN_RASTERIZER_COMPUTE_CAP,
                poll_interval_seconds=15,
                startup_timeout_seconds=min(
                    PROVIDER_EXECUTION_TIMEOUT_SECONDS, remaining_minutes * 60
                ),
                heartbeat_no_progress_seconds=PROVIDER_HEARTBEAT_NO_PROGRESS_SECONDS,
                session_budget_ledger_path=job / "adp_inpaint360_vast_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=False,
                preferred_gpu_keywords=("RTX 4090",),
                prefer_isaac_rt=False,
                gpu_selection_policy=INPAINT360_GPU_SELECTION_POLICY,
                machine_avoidlist_path=machine_avoidlist_path,
                allowed_active_instance_ids=allowed_active_instance_ids,
                instance_label_prefix="blueprint-adp-inpaint360-interiorgs-",
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {
            "status": "blocked",
            "blockers": [f"adp_inpaint360_vast_adapter_failed:{type(exc).__name__}"],
            "raw_secret_values_recorded": False,
        }
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    extracted = _extract(output_zip, job / "immutable_execution")
    execution = dict(extracted.get("execution") or {})
    teardown = _read_json(provider_run / "vast_teardown_manifest.json")
    blockers = list(adapter.get("blockers") or []) + list(extracted.get("blockers") or [])
    if execution.get("status") != "completed":
        blockers.extend(execution.get("blockers") or ["inpaint360_edit_not_completed"])
    if teardown.get("continuing_spend_from_this_run") is not False:
        blockers.append("inpaint360_vast_provider_zero_not_proven")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("inpaint360_object_store_provider_zero_not_proven")
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "bundle_sha256": bundle["bundle_sha256"],
        "execution_result_path": extracted.get("result_path"),
        "adapter_result_path": str(provider_run / "vast_provider_adapter_result.json"),
        "teardown_manifest_path": str(provider_run / "vast_teardown_manifest.json"),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
    }
    # Seal the two terminal artifacts every production launch profile asks
    # this result for. Without them the run ends
    # `allocator_terminal_artifact_missing:` whatever happened on the provider.
    result = seal_lane_terminal_artifacts(
        result,
        attempt_root=job,
        lane="adp_inpaint360_interiorgs",
        binding={
            "bundle_sha256": bundle.get("bundle_sha256")
            if isinstance(bundle, Mapping)
            else None,
            "provider": "vast",
        },
    )
    write_json(job / "adp_inpaint360_interiorgs_vast_result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--inpaint360-root", required=True)
    parser.add_argument("--lama-root", required=True)
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--adapter-receipt", required=True)
    parser.add_argument("--prerequisite-receipt", required=True)
    parser.add_argument("--big-lama", required=True)
    parser.add_argument("--vgg16-weights", required=True)
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    receipt = build_inpaint360_interiorgs_bundle(
        repo_root=args.repo_root,
        inpaint360_root=args.inpaint360_root,
        lama_root=args.lama_root,
        adapter_root=args.adapter_root,
        adapter_receipt_path=args.adapter_receipt,
        prerequisite_receipt_path=args.prerequisite_receipt,
        big_lama_path=args.big_lama,
        vgg16_weights_path=args.vgg16_weights,
        job_dir=args.job_dir,
    )
    print(json.dumps({"status": receipt["status"], "bundle_sha256": receipt["bundle_sha256"]}))
    return 0 if receipt["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
