"""Build the distinct immutable AuraFusion360 InteriorGS Vast packet."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adp_aura_author_smoke_vast import (
    DEFAULT_IMAGE,
    SAM2_LICENSE_SHA256,
    SAM2_SOURCE_COMMIT,
    SAM2_SOURCE_REPOSITORY,
    SAM2_SOURCE_TREE,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    SOURCE_TREE,
    SUBMODULES,
    WONDERWORLD_MARIGOLD_LICENSE_SHA256,
    WONDERWORLD_MARIGOLD_RUNTIME_FILES,
    WONDERWORLD_SOURCE_COMMIT,
    WONDERWORLD_SOURCE_REPOSITORY,
    WONDERWORLD_SOURCE_TREE,
    _RUNTIME_MODELS,
    _SD2,
    _deterministic_zip_directory,
    _deterministic_zip_files,
    _git,
    _read_json,
    _sha256,
    _source_files,
    _source_manifest,
    _tracked_files,
    _validate_prerequisite,
    _write_executable,
)
from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import PaidResourceAdmissionGrant
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers
from .public_scene_aura_adapter import (
    BIG_LAMA_SHA256,
    BIG_LAMA_SIZE,
    LAMA_COMMIT,
    LAMA_TREE,
    SCHEMA_VERSION as ADAPTER_SCHEMA,
)
from .vast_provider_adapter import run_vast_provider_adapter
from .vast_session_budget_contract import attempt_estimated_cost, attempt_runtime_seconds
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)

PROBE_KIND = "adp-aurafusion360-interiorgs"
PROVIDER_BUNDLE_KIND = "adp_aura_interiorgs"
RESULT_SCHEMA_VERSION = "adp_aura_interiorgs_vast_run.v1"
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/aurafusion360-interiorgs"
MIN_RASTERIZER_COMPUTE_CAP = 890
PROVIDER_EXECUTION_TIMEOUT_SECONDS = 14_400
PROVIDER_HEARTBEAT_NO_PROGRESS_SECONDS = 1800
AURA_INTERIORGS_GPU_SELECTION_POLICY = {
    "policy_id": "aura_interiorgs_l40s_observed_control",
    "allowed_gpu_keywords": ("L40S",),
    "denied_gpu_keywords": (),
    "reason": "reuse the L40S class that completed the unchanged Aura author control",
}
_VAST_MUTATION_ENV = (
    "BLUEPRINT_ALLOW_VAST_API_CALLS",
    "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH",
)
_VAST_SINGLE_ATTEMPT_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"
AURA_RUNTIME_PREREQUISITE_RECEIPT_DIGEST = (
    "sha256:1b37189c60b55981bbb6f076109e476074aa570f53a1bbdaa66d01f8e052445a"
)
OPENCLIP_REPOSITORY = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
OPENCLIP_REVISION = "1c2b8495b28150b8a4922ee1c8edee224c284c0c"
OPENCLIP_PATH = "open_clip_pytorch_model.bin"
OPENCLIP_SIZE_BYTES = 3_944_692_325
OPENCLIP_SHA256 = (
    "sha256:9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4"
)
OPENCLIP_SNAPSHOT_DIGEST = (
    "sha256:9c94ad4897df15ae307d9c809d3d6a0ee7222350ca34a55da9f77a2b1af63110"
)

_SCENE_ID_PATTERN = re.compile(r"^[0-9]{6}$")
_TARGET_ID_PATTERN = re.compile(r"^ins[0-9]+$")
_CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _adapter_scene_binding(receipt: Mapping[str, Any]) -> dict[str, Any]:
    scene = receipt.get("scene")
    if not isinstance(scene, Mapping):
        raise ValueError("adp_aura_interiorgs_scene_binding_missing")
    scene_id = str(scene.get("publisher_scene_id") or "")
    target_instance_id = str(scene.get("target_instance_id") or "")
    scene_slug = str(scene.get("scene_slug") or "")
    reference_camera_id = str(scene.get("reference_camera_id") or "")
    if (
        _SCENE_ID_PATTERN.fullmatch(scene_id) is None
        or _TARGET_ID_PATTERN.fullmatch(target_instance_id) is None
        or scene_slug != f"{scene_id}_{target_instance_id}"
        or _CAMERA_ID_PATTERN.fullmatch(reference_camera_id) is None
    ):
        raise ValueError("adp_aura_interiorgs_scene_binding_invalid")
    camera_count = scene.get("camera_count")
    reference_index = scene.get("reference_camera_index")
    if (
        isinstance(camera_count, bool)
        or not isinstance(camera_count, int)
        or camera_count < 1
        or isinstance(reference_index, bool)
        or not isinstance(reference_index, int)
        or not 0 <= reference_index < camera_count
    ):
        raise ValueError("adp_aura_interiorgs_camera_binding_invalid")
    return {
        "publisher_scene_id": scene_id,
        "target_instance_id": target_instance_id,
        "target_semantic_label": str(scene.get("target_semantic_label") or ""),
        "scene_slug": scene_slug,
        "camera_count": camera_count,
        "reference_camera_id": reference_camera_id,
        "reference_camera_index": reference_index,
        "input_receipt_digest": str(scene.get("input_receipt_digest") or ""),
    }


def _validated_adapter(
    receipt: Mapping[str, Any], root: Path
) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    if (
        receipt.get("schema_version") != ADAPTER_SCHEMA
        or receipt.get("status") != "prepared_unexecuted"
        or canonical_digest(receipt, digest_field="receipt_digest") != receipt.get("receipt_digest")
    ):
        raise ValueError("adp_aura_interiorgs_adapter_receipt_invalid")
    source = receipt.get("source") or {}
    execution = receipt.get("execution") or {}
    scene = _adapter_scene_binding(receipt)
    if source.get("commit") != SOURCE_COMMIT or source.get("tree") != SOURCE_TREE:
        raise ValueError("adp_aura_interiorgs_source_identity_mismatch")
    if any(bool(value) for value in execution.values()):
        raise ValueError("adp_aura_interiorgs_caller_asserted_execution_forbidden")
    rows: list[tuple[str, Path]] = []
    for record in receipt.get("artifacts") or []:
        relative = str(record.get("relative_path") or "")
        path = (root / relative).resolve()
        if root != path and root not in path.parents:
            raise ValueError("adp_aura_interiorgs_adapter_artifact_outside_root")
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise ValueError("adp_aura_interiorgs_adapter_artifact_changed")
        rows.append((relative, path))
    scene_slug = scene["scene_slug"]
    reference_camera_id = scene["reference_camera_id"]
    required = {
        "aurafusion360_interiorgs_execution_spec.json",
        f"configs/Other-360/{scene_slug}/train.config",
        f"configs/Other-360/{scene_slug}/remove.config",
        f"configs/Other-360/{scene_slug}/inpaint.config",
        f"configs/Other-360/{scene_slug}/sdedit.config",
        f"reference_lama_input/{reference_camera_id}.png",
        f"reference_lama_input/{reference_camera_id}_mask.png",
        f"data/Other-360/{scene_slug}/sparse/0/points3D.ply",
    }
    if not required.issubset({relative for relative, _ in rows}):
        raise ValueError("adp_aura_interiorgs_adapter_required_artifact_missing")
    return rows, scene


def _validated_runtime_prerequisite(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if (
        receipt.get("receipt_digest") != AURA_RUNTIME_PREREQUISITE_RECEIPT_DIGEST
        or canonical_digest(receipt, digest_field="receipt_digest")
        != AURA_RUNTIME_PREREQUISITE_RECEIPT_DIGEST
    ):
        raise ValueError("adp_aura_interiorgs_runtime_prerequisite_digest_mismatch")
    method = (receipt.get("methods") or {}).get("aurafusion360_interiorgs_runtime")
    if not isinstance(method, Mapping) or method.get("checkpoint_rights_established") is not True:
        raise ValueError("adp_aura_interiorgs_runtime_prerequisite_rights_missing")
    snapshots = method.get("remote_snapshots") or []
    if len(snapshots) != 1 or not isinstance(snapshots[0], Mapping):
        raise ValueError("adp_aura_interiorgs_runtime_prerequisite_snapshot_invalid")
    snapshot = snapshots[0]
    publisher = snapshot.get("publisher")
    rights = snapshot.get("rights")
    if not isinstance(publisher, Mapping) or not isinstance(rights, Mapping):
        raise ValueError("adp_aura_interiorgs_openclip_publisher_or_rights_missing")
    identity = publisher.get("single_file_identity")
    if (
        snapshot.get("artifact_id") != "aurafusion360_openclip_vit_h_14"
        or snapshot.get("rights_established") is not True
        or rights.get("license_id") != "MIT"
        or publisher.get("repository") != OPENCLIP_REPOSITORY
        or publisher.get("revision") != OPENCLIP_REVISION
        or publisher.get("path_prefix") != OPENCLIP_PATH
        or publisher.get("snapshot_digest") != OPENCLIP_SNAPSHOT_DIGEST
        or publisher.get("gated") is not False
        or publisher.get("private") is not False
        or not isinstance(identity, Mapping)
        or identity.get("path") != OPENCLIP_PATH
        or identity.get("size_bytes") != OPENCLIP_SIZE_BYTES
        or identity.get("lfs_sha256") != OPENCLIP_SHA256
    ):
        raise ValueError("adp_aura_interiorgs_openclip_identity_invalid")
    return {
        "repository": OPENCLIP_REPOSITORY,
        "revision": OPENCLIP_REVISION,
        "snapshot_digest": OPENCLIP_SNAPSHOT_DIGEST,
        "materialized_files": [
            {
                "path": OPENCLIP_PATH,
                "size_bytes": OPENCLIP_SIZE_BYTES,
                "sha256": OPENCLIP_SHA256,
            }
        ],
        "materialized_total_size_bytes": OPENCLIP_SIZE_BYTES,
        "license": "MIT",
        "access": "public_ungated",
    }


def _validated_runtime_source_rows(
    root: Path,
    files: Mapping[str, str],
    *,
    error: str,
) -> list[tuple[str, Path]]:
    """Resolve every released runtime source before any large archive is written."""

    rows = [
        (archive_path, root / source_path)
        for archive_path, source_path in sorted(files.items())
    ]
    if any(not path.is_file() for _archive_path, path in rows):
        raise ValueError(error)
    return rows


def build_aura_interiorgs_bundle(
    *,
    repo_root: str | Path,
    aura_root: str | Path,
    sam2_root: str | Path,
    wonderworld_root: str | Path,
    lama_root: str | Path,
    prerequisite_receipt_path: str | Path,
    runtime_prerequisite_receipt_path: str | Path,
    adapter_root: str | Path,
    adapter_receipt_path: str | Path,
    big_lama_path: str | Path,
    job_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    aura = Path(aura_root).expanduser().resolve()
    sam2 = Path(sam2_root).expanduser().resolve()
    wonderworld = Path(wonderworld_root).expanduser().resolve()
    lama = Path(lama_root).expanduser().resolve()
    packet = Path(adapter_root).expanduser().resolve()
    adapter_file = Path(adapter_receipt_path).expanduser().resolve()
    prerequisite_file = Path(prerequisite_receipt_path).expanduser().resolve()
    runtime_prerequisite_file = Path(
        runtime_prerequisite_receipt_path
    ).expanduser().resolve()
    big_lama = Path(big_lama_path).expanduser().resolve()
    job = Path(job_dir).expanduser().resolve()
    if job.exists() and any(job.iterdir()):
        raise ValueError("adp_aura_interiorgs_job_dir_not_empty")
    runtime = job / "provider_runtime"
    ensure_dir(runtime)
    if (
        _git(repo, "status", "--porcelain", "--untracked-files=no")
        or _git(aura, "rev-parse", "HEAD") != SOURCE_COMMIT
        or _git(aura, "rev-parse", "HEAD^{tree}") != SOURCE_TREE
        or _git(aura, "status", "--porcelain")
    ):
        raise ValueError("adp_aura_interiorgs_source_or_blueprint_dirty")
    if {path: _git(aura / path, "rev-parse", "HEAD") for path in SUBMODULES} != SUBMODULES:
        raise ValueError("adp_aura_interiorgs_submodule_mismatch")
    if (
        _git(sam2, "rev-parse", "HEAD") != SAM2_SOURCE_COMMIT
        or _git(sam2, "rev-parse", "HEAD^{tree}") != SAM2_SOURCE_TREE
        or _git(sam2, "status", "--porcelain")
    ):
        raise ValueError("adp_aura_interiorgs_sam2_identity_mismatch")
    if not (sam2 / "LICENSE").is_file() or _sha256(sam2 / "LICENSE") != SAM2_LICENSE_SHA256:
        raise ValueError("adp_aura_interiorgs_sam2_license_mismatch")
    if (
        _git(wonderworld, "rev-parse", "HEAD") != WONDERWORLD_SOURCE_COMMIT
        or _git(wonderworld, "rev-parse", "HEAD^{tree}") != WONDERWORLD_SOURCE_TREE
        or _git(wonderworld, "status", "--porcelain")
    ):
        raise ValueError("adp_aura_interiorgs_wonderworld_identity_mismatch")
    if _sha256(wonderworld / "marigold_module/LICENSE.txt") != WONDERWORLD_MARIGOLD_LICENSE_SHA256:
        raise ValueError("adp_aura_interiorgs_wonderworld_license_mismatch")
    if (
        _git(lama, "rev-parse", "HEAD") != LAMA_COMMIT
        or _git(lama, "rev-parse", "HEAD^{tree}") != LAMA_TREE
        or _git(lama, "status", "--porcelain", "--untracked-files=no")
    ):
        raise ValueError("adp_aura_interiorgs_lama_source_identity_mismatch")
    if (
        not big_lama.is_file()
        or big_lama.stat().st_size != BIG_LAMA_SIZE
        or _sha256(big_lama) != BIG_LAMA_SHA256
    ):
        raise ValueError("adp_aura_interiorgs_lama_checkpoint_changed")

    prerequisite = _read_json(prerequisite_file)
    snapshots = _validate_prerequisite(prerequisite)
    runtime_prerequisite = _read_json(runtime_prerequisite_file)
    openclip_runtime_model = _validated_runtime_prerequisite(runtime_prerequisite)
    adapter = _read_json(adapter_file)
    adapter_rows, scene_binding = _validated_adapter(adapter, packet)
    aura_rows = _source_files(aura)
    sam2_rows = _tracked_files(sam2)
    wonderworld_rows = _validated_runtime_source_rows(
        wonderworld,
        WONDERWORLD_MARIGOLD_RUNTIME_FILES,
        error="adp_aura_interiorgs_wonderworld_runtime_source_missing",
    )
    lama_rows = _tracked_files(lama)
    _deterministic_zip_files(aura_rows, runtime / "aurafusion360_source.zip")
    _deterministic_zip_files(sam2_rows, runtime / "sam2_source.zip")
    _deterministic_zip_files(wonderworld_rows, runtime / "wonderworld_marigold_runtime.zip")
    _deterministic_zip_files(lama_rows, runtime / "lama_source.zip")
    _deterministic_zip_files(adapter_rows, runtime / "interiorgs_adapter.zip")
    shutil.copy2(big_lama, runtime / "big-lama.zip")
    source_manifest = _source_manifest(aura_rows)
    sam2_manifest = _source_manifest(sam2_rows)
    wonderworld_manifest = _source_manifest(wonderworld_rows)
    lama_manifest = _source_manifest(lama_rows)
    adapter_manifest = _source_manifest(adapter_rows)
    sd2 = snapshots["aurafusion360_sd2_inpainting_exact_checkpoint"]["publisher"][
        "single_file_identity"
    ]
    workflow_names = (
        "train",
        "render",
        "remove",
        "sam2_masks",
        "inpaint_init",
        "sdedit",
        "inpaint_finetune",
    )
    workflow = [
        {"stage": name, "command": command}
        for name, command in zip(
            workflow_names, adapter["commands"]["author_workflow"], strict=True
        )
    ]
    spec = {
        "schema_version": "adp_aura_interiorgs_spec.v1",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "source_files": source_manifest,
        "submodules": SUBMODULES,
        "scene": scene_binding,
        "sam2_source": {
            "repository": SAM2_SOURCE_REPOSITORY,
            "commit": SAM2_SOURCE_COMMIT,
            "tree": SAM2_SOURCE_TREE,
            "license_sha256": SAM2_LICENSE_SHA256,
            "source_files": sam2_manifest,
        },
        "wonderworld_marigold_runtime": {
            "repository": WONDERWORLD_SOURCE_REPOSITORY,
            "commit": WONDERWORLD_SOURCE_COMMIT,
            "tree": WONDERWORLD_SOURCE_TREE,
            "license": "Apache-2.0",
            "license_sha256": WONDERWORLD_MARIGOLD_LICENSE_SHA256,
            "archive": "wonderworld_marigold_runtime.zip",
            "archive_sha256": _sha256(runtime / "wonderworld_marigold_runtime.zip"),
            "source_files": wonderworld_manifest,
        },
        "lama": {
            "source_archive": "lama_source.zip",
            "source_files": lama_manifest,
            "checkpoint_archive": "big-lama.zip",
            "checkpoint_sha256": BIG_LAMA_SHA256,
        },
        "adapter": {
            "archive": "interiorgs_adapter.zip",
            "archive_sha256": _sha256(runtime / "interiorgs_adapter.zip"),
            "receipt_digest": adapter["receipt_digest"],
            "files": adapter_manifest,
        },
        "runtime_models": [*_RUNTIME_MODELS, openclip_runtime_model],
        "sd2_checkpoint": {
            **_SD2,
            "size_bytes": sd2["size_bytes"],
            "sha256": sd2["lfs_sha256"],
        },
        "workflow": workflow,
        "claim_boundary": {
            "hidden_background_truth_available": False,
            "publisher_splat_edited_in_place": False,
            "output_claim_ceiling": "visual_candidate_only",
        },
    }
    write_json(runtime / "execution_spec.json", spec)
    scripts = repo / "scripts"
    _write_executable(
        runtime / "run_adp_aura_interiorgs_provider_runtime.sh",
        scripts / "run_adp_aura_interiorgs_provider_runtime.sh",
    )
    for name in (
        "adp_aura_interiorgs_provider_runner.py",
        "adp_aura_author_smoke_provider_runner.py",
    ):
        shutil.copy2(scripts / name, runtime / name)
    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        entrypoint_text=(runtime / "run_adp_aura_interiorgs_provider_runtime.sh").read_text(),
        runner_text=(runtime / "adp_aura_interiorgs_provider_runner.py").read_text(),
    )
    manifest = {
        "schema_version": "adp_aura_interiorgs_provider_bundle.v1",
        "generated_at": generated_at or utc_now_iso(),
        "status": "ready" if not blockers else "blocked",
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "blueprint_commit": _git(repo, "rev-parse", "HEAD"),
        "blueprint_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "blueprint_repository_tracked_state": "clean",
        "adapter_receipt_digest": adapter["receipt_digest"],
        "prerequisite_receipt_digest": prerequisite["receipt_digest"],
        "runtime_prerequisite_receipt_digest": runtime_prerequisite[
            "receipt_digest"
        ],
        "container_image": DEFAULT_IMAGE,
        "expected_output_filename": "adp_aura_interiorgs_result.json",
        "retry_cap": 0,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    write_json(runtime / "adp_aura_interiorgs_provider_manifest.json", manifest)
    bundle = job / "adp_aura_interiorgs_provider_runtime_bundle.zip"
    _deterministic_zip_directory(runtime, bundle)
    receipt = {
        **manifest,
        "bundle_path": str(bundle),
        "bundle_sha256": _sha256(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
    }
    write_json(job / "adp_aura_interiorgs_bundle_receipt.json", receipt)
    return receipt


def _remaining_minutes(
    *, job: Path, hard_cap_usd: float, hard_ttl_seconds: int, max_hourly_rate_usd: float
) -> int:
    ledger = _read_json(job / "adp_aura_interiorgs_vast_session_budget.json")
    attempts = [row for row in ledger.get("attempts") or [] if isinstance(row, Mapping)]
    prior_seconds = sum(attempt_runtime_seconds(row) for row in attempts)
    prior_cost = sum(attempt_estimated_cost(row) for row in attempts)
    runtime_minutes = math.floor(max(0.0, hard_ttl_seconds - prior_seconds) / 60.0)
    spend_minutes = math.floor(
        max(0.0, hard_cap_usd - prior_cost) * 60.0 / max_hourly_rate_usd
    )
    return max(0, min(runtime_minutes, spend_minutes))


def _extract_provider_output(path: Path, destination: Path) -> dict[str, Any]:
    blockers: list[str] = []
    if not path.is_file():
        return {
            "status": "blocked",
            "execution": {},
            "result_path": str(destination / "adp_aura_interiorgs_result.json"),
            "blockers": ["aura_interiorgs_provider_output_zip_missing"],
        }
    ensure_dir(destination)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    blockers.append("aura_interiorgs_provider_output_zip_path_traversal")
            if not blockers:
                archive.extractall(destination)
    except (OSError, zipfile.BadZipFile):
        blockers.append("aura_interiorgs_provider_output_zip_invalid")
    result_path = destination / "adp_aura_interiorgs_result.json"
    execution = _read_json(result_path)
    if not execution:
        blockers.append("aura_interiorgs_provider_result_missing")
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


def run_aura_interiorgs_vast(
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
    """Run the frozen Aura InteriorGS challenger once, with no automatic retry."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle = dict(prepared_bundle)
    bundle_path = Path(str(bundle.get("bundle_path") or "")).resolve()
    if public_image != DEFAULT_IMAGE:
        raise ValueError("adp_aura_interiorgs_container_image_not_frozen")
    if (
        bundle.get("status") != "ready"
        or not bundle_path.is_file()
        or _sha256(bundle_path) != bundle.get("bundle_sha256")
    ):
        raise ValueError("adp_aura_interiorgs_prepared_bundle_binding_invalid")
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
        write_json(job / "adp_aura_interiorgs_vast_result.json", result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("adp_aura_interiorgs_paid_resource_admission_grant_missing")
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
            "blockers": ["adp_aura_interiorgs_budget_below_minimum_live_window"],
        }
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=str(bundle_path),
        key_prefix=DEFAULT_KEY_PREFIX,
        expiration_seconds=max(hard_ttl_seconds + 1800, 18_000),
    )
    if staging.get("status") != "completed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": staging.get("blockers") or ["aura_interiorgs_object_store_staging_blocked"],
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
                session_budget_ledger_path=job / "adp_aura_interiorgs_vast_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=False,
                preferred_gpu_keywords=("L40S",),
                prefer_isaac_rt=False,
                gpu_selection_policy=AURA_INTERIORGS_GPU_SELECTION_POLICY,
                machine_avoidlist_path=machine_avoidlist_path,
                allowed_active_instance_ids=allowed_active_instance_ids,
                vast_launch_lock_file=job.parent / "aura_interiorgs_paid_launch.lock",
                instance_label_prefix="blueprint-adp-aura-interiorgs-",
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {
            "status": "blocked",
            "blockers": [f"adp_aura_interiorgs_vast_adapter_failed:{type(exc).__name__}"],
            "raw_secret_values_recorded": False,
        }
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    extracted = _extract_provider_output(output_zip, job / "immutable_execution")
    execution = dict(extracted.get("execution") or {})
    teardown = _read_json(provider_run / "vast_teardown_manifest.json")
    blockers = list(adapter.get("blockers") or []) + list(extracted.get("blockers") or [])
    if execution.get("status") != "completed":
        blockers.extend(execution.get("blockers") or ["aura_interiorgs_edit_not_completed"])
    if teardown.get("continuing_spend_from_this_run") is not False:
        blockers.append("aura_interiorgs_vast_provider_zero_not_proven")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("aura_interiorgs_object_store_provider_zero_not_proven")
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
    write_json(job / "adp_aura_interiorgs_vast_result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo-root",
        "aura-root",
        "sam2-root",
        "wonderworld-root",
        "lama-root",
        "prerequisite-receipt",
        "runtime-prerequisite-receipt",
        "adapter-root",
        "adapter-receipt",
        "big-lama",
        "job-dir",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    receipt = build_aura_interiorgs_bundle(
        repo_root=args.repo_root,
        aura_root=args.aura_root,
        sam2_root=args.sam2_root,
        wonderworld_root=args.wonderworld_root,
        lama_root=args.lama_root,
        prerequisite_receipt_path=args.prerequisite_receipt,
        runtime_prerequisite_receipt_path=args.runtime_prerequisite_receipt,
        adapter_root=args.adapter_root,
        adapter_receipt_path=args.adapter_receipt,
        big_lama_path=args.big_lama,
        job_dir=args.job_dir,
    )
    print(json.dumps({"status": receipt["status"], "bundle_sha256": receipt["bundle_sha256"]}))
    return 0 if receipt["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
