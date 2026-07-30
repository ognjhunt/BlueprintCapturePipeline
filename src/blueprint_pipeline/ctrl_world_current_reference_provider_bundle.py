"""Deterministic provider bundle for one request-bound Ctrl-World generation."""

from __future__ import annotations

import shutil
import stat
import zipfile
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json
from .ctrl_world_current_reference_provider_runtime import (
    ARM_ID,
    EXPECTED_STATE_STAT_SHA256,
    EXPECTED_WORLD_MODEL_SHA256,
    MODEL_FREEZE,
    validate_staged_request,
)
from .ctrl_world_provider_bundle import (
    CTRL_WORLD_PUBLIC_IMAGE,
    CTRL_WORLD_TORCH_VERSION,
    MODEL_FREEZE as PROVIDER_MODEL_DOWNLOAD_FREEZE,
    PYTHON_DEPENDENCIES,
    REMOTE_ENTRYPOINT,
    _sha256_file,
    _source_commit,
    _source_status,
    _write_deterministic_zip,
)
from .policy_ranking_thesis import canonical_sha256


EXPERIMENT_ID = "policy_ranking_real_policy_closed_loop_confirmation_20260730"
BUNDLE_SCHEMA_VERSION = "ctrl_world_current_reference_provider_bundle.v1"
RECEIPT_SCHEMA_VERSION = "ctrl_world_current_reference_provider_bundle_receipt.v1"
DEFAULT_BUNDLE_FILENAME = "ctrl_world_current_reference_provider_runtime_bundle.zip"
SOURCE_FILES = (
    "LICENSE.txt",
    "requirements.txt",
    "config.py",
    "models/__init__.py",
    "models/ctrl_world.py",
    "models/pipeline_ctrl_world.py",
    "models/pipeline_stable_video_diffusion.py",
    "models/unet_spatio_temporal_condition.py",
    "models/utils.py",
    "dataset_meta_info/droid/stat.json",
)


def inspect_ctrl_world_current_reference_archive_inputs(
    archive: zipfile.ZipFile,
    *,
    manifest: Mapping[str, Any],
    rollout_manifest: Mapping[str, Any],
    names: set[str],
) -> tuple[dict[str, str], list[str]]:
    """Independently validate every request/source byte in a current-reference bundle."""

    blockers: list[str] = []
    prefix = "provider_runtime/"
    required = {
        prefix + "wam_provider_runtime_manifest.json",
        prefix + "wam_rollout_input_manifest.json",
        prefix + "wam_provider_runtime_runner.py",
        prefix + "ctrl_world_provider_runtime_support.py",
        prefix + "run_wam_provider_runtime.sh",
        prefix + "successor_retained_control.py",
        prefix + "ctrl_world_request/ctrl_world_current_reference_request.json",
    }
    missing = sorted(required - names)
    if missing:
        blockers.append("ctrl_world_current_reference_archive_entries_missing")
    if any(name.lower().endswith(".mp4") for name in names):
        blockers.append("ctrl_world_current_reference_archive_future_video_present")
    try:
        request_entry = prefix + "ctrl_world_request/ctrl_world_current_reference_request.json"
        request = json.loads(archive.read(request_entry).decode("utf-8"))
        if (
            request.get("schema_version")
            != "blueprint_ctrl_world_current_reference_staged_request.v1"
        ):
            blockers.append("ctrl_world_current_reference_archive_request_schema_invalid")
        request_digest_payload = dict(request)
        recorded_request_sha256 = request_digest_payload.pop("request_sha256", None)
        computed_request_sha256 = canonical_sha256(request_digest_payload)
        if recorded_request_sha256 != computed_request_sha256:
            blockers.append("ctrl_world_current_reference_archive_request_digest_invalid")
        request_root = prefix + "ctrl_world_request/"
        request_files: list[tuple[str, str]] = []
        for view_id in request.get("view_order") or []:
            for row in (request.get("selected_history_views") or {}).get(view_id, []):
                request_files.append(
                    (str(row.get("relative_path") or ""), str(row.get("sha256") or ""))
                )
            current = (request.get("current_views") or {}).get(view_id, {})
            request_files.append(
                (str(current.get("relative_path") or ""), str(current.get("sha256") or ""))
            )
        action = request.get("action_conditioning") or {}
        request_files.append(
            (str(action.get("relative_path") or ""), str(action.get("sha256") or ""))
        )
        for relative, expected_digest in request_files:
            entry = request_root + relative
            if not relative or entry not in names:
                blockers.append("ctrl_world_current_reference_archive_request_file_missing")
                continue
            observed_digest = hashlib.sha256(archive.read(entry)).hexdigest()
            if observed_digest != expected_digest:
                blockers.append("ctrl_world_current_reference_archive_request_file_hash_mismatch")

        source_manifest = manifest.get("source_manifest")
        if not isinstance(source_manifest, Mapping):
            blockers.append("ctrl_world_current_reference_archive_source_manifest_invalid")
            source_manifest = {}
        for row in source_manifest.get("files") or []:
            relative = str(row.get("relative_path") or "")
            entry = prefix + "ctrl_world_source/" + relative
            if not relative or entry not in names:
                blockers.append("ctrl_world_current_reference_archive_source_file_missing")
                continue
            observed = hashlib.sha256(archive.read(entry)).hexdigest()
            if observed != row.get("sha256"):
                blockers.append("ctrl_world_current_reference_archive_source_file_hash_mismatch")

        for value, reason in (
            (
                manifest.get("experiment_id"),
                "ctrl_world_current_reference_archive_experiment_invalid",
            ),
            (
                rollout_manifest.get("experiment_id"),
                "ctrl_world_current_reference_archive_rollout_experiment_invalid",
            ),
        ):
            if value != EXPERIMENT_ID:
                blockers.append(reason)
        if manifest.get("arm_id") != "blueprint_ctrl_world_current_reference":
            blockers.append("ctrl_world_current_reference_archive_arm_invalid")
        if rollout_manifest.get("arm_id") != "blueprint_ctrl_world_current_reference":
            blockers.append("ctrl_world_current_reference_archive_rollout_arm_invalid")
        if manifest.get("request_sha256") != computed_request_sha256:
            blockers.append("ctrl_world_current_reference_archive_manifest_request_mismatch")
        if rollout_manifest.get("request_sha256") != computed_request_sha256:
            blockers.append("ctrl_world_current_reference_archive_rollout_request_mismatch")
        if manifest.get("truth_boundary", {}).get("future_physical_rgb_forbidden") is not True:
            blockers.append("ctrl_world_current_reference_archive_future_rgb_boundary_invalid")
        if rollout_manifest.get("physical_future_rgb_provided_to_model") is not False:
            blockers.append("ctrl_world_current_reference_archive_rollout_future_rgb_invalid")
        if rollout_manifest.get("physical_outcome_labels_accessed") is not False:
            blockers.append("ctrl_world_current_reference_archive_rollout_label_boundary_invalid")

        embedded_hashes = {
            "runtime_manifest_file_sha256": hashlib.sha256(
                archive.read(prefix + "wam_provider_runtime_manifest.json")
            ).hexdigest(),
            "rollout_manifest_file_sha256": hashlib.sha256(
                archive.read(prefix + "wam_rollout_input_manifest.json")
            ).hexdigest(),
            "request_manifest_file_sha256": hashlib.sha256(archive.read(request_entry)).hexdigest(),
            "request_sha256": computed_request_sha256,
            "source_manifest_sha256": canonical_sha256(source_manifest),
            "direct_runner_sha256": hashlib.sha256(
                archive.read(prefix + "wam_provider_runtime_runner.py")
            ).hexdigest(),
            "support_runner_sha256": hashlib.sha256(
                archive.read(prefix + "ctrl_world_provider_runtime_support.py")
            ).hexdigest(),
            "entrypoint_sha256": hashlib.sha256(
                archive.read(prefix + "run_wam_provider_runtime.sh")
            ).hexdigest(),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        blockers.append("ctrl_world_current_reference_archive_unreadable")
        embedded_hashes = {}
    return embedded_hashes, sorted(set(blockers))


def build_ctrl_world_current_reference_provider_bundle(
    *,
    job_dir: str | Path,
    ctrl_world_source_dir: str | Path,
    staged_request_dir: str | Path,
    generated_at: str | None = None,
    bundle_filename: str = DEFAULT_BUNDLE_FILENAME,
) -> dict[str, Any]:
    """Freeze one direct WAM request without policy or future-physical inputs."""

    output = Path(job_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"ctrl_world_current_reference_bundle_job_exists:{output}")
    ensure_dir(output)
    source_root = Path(ctrl_world_source_dir).expanduser().resolve()
    request_root = Path(staged_request_dir).expanduser().resolve()
    request_manifest_path = request_root / "ctrl_world_current_reference_request.json"
    validated_request = validate_staged_request(request_manifest_path)
    bundle_root = output / "ctrl_world_current_reference_provider_bundle"
    runtime_dir = bundle_root / "provider_runtime"
    packaged_source = runtime_dir / "ctrl_world_source"
    ensure_dir(packaged_source)
    blockers: list[str] = []
    observed_commit = _source_commit(source_root)
    observed_status = _source_status(source_root)
    expected_revision = MODEL_FREEZE["ctrl_world_source"]["revision"]
    if observed_commit != expected_revision:
        blockers.append("ctrl_world_current_reference_source_revision_mismatch")
    if observed_status:
        blockers.append("ctrl_world_current_reference_source_worktree_not_clean")

    source_rows: list[dict[str, Any]] = []
    for relative in SOURCE_FILES:
        source = source_root / relative
        destination = packaged_source / relative
        if not source.is_file() or source.is_symlink():
            blockers.append(f"ctrl_world_current_reference_source_file_invalid:{relative}")
            continue
        ensure_dir(destination.parent)
        shutil.copyfile(source, destination)
        source_rows.append(
            {
                "relative_path": relative,
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256_file(destination),
            }
        )
    stat_source = source_root / "dataset_meta_info/droid/stat.json"
    if stat_source.is_file() and _sha256_file(stat_source) != EXPECTED_STATE_STAT_SHA256:
        blockers.append("ctrl_world_current_reference_state_stat_hash_mismatch")

    packaged_request = runtime_dir / "ctrl_world_request"
    shutil.copytree(request_root, packaged_request, symlinks=False)
    packaged_request_manifest = packaged_request / request_manifest_path.name
    try:
        packaged_validated = validate_staged_request(packaged_request_manifest)
    except (OSError, ValueError) as exc:
        blockers.append(
            f"ctrl_world_current_reference_packaged_request_invalid:{type(exc).__name__}"
        )
        packaged_validated = {}
    if packaged_validated.get("request_sha256") != validated_request["request_sha256"]:
        blockers.append("ctrl_world_current_reference_packaged_request_digest_mismatch")

    direct_runner_source = Path(__file__).with_name(
        "ctrl_world_current_reference_provider_runtime.py"
    )
    support_source = Path(__file__).with_name("ctrl_world_provider_runtime_runner.py")
    retained_source = Path(__file__).with_name("policy_ranking_successor_retained_remote.py")
    for source, filename in (
        (direct_runner_source, "wam_provider_runtime_runner.py"),
        (support_source, "ctrl_world_provider_runtime_support.py"),
        (retained_source, "successor_retained_control.py"),
    ):
        if not source.is_file():
            blockers.append(f"ctrl_world_current_reference_runtime_source_missing:{filename}")
            continue
        destination = runtime_dir / filename
        shutil.copyfile(source, destination)
        destination.chmod(destination.stat().st_mode | stat.S_IXUSR)
    entrypoint = runtime_dir / "run_wam_provider_runtime.sh"
    entrypoint.write_text(REMOTE_ENTRYPOINT, encoding="utf-8")
    entrypoint.chmod(entrypoint.stat().st_mode | stat.S_IXUSR)

    source_manifest = {
        "repository": MODEL_FREEZE["ctrl_world_source"]["repository"],
        "revision": expected_revision,
        "files": source_rows,
    }
    runtime_manifest: dict[str, Any] = {
        "schema_version": "wam_provider_runtime_manifest.v1",
        "runtime": "ctrl_world_current_reference_generated_only",
        "experiment_id": EXPERIMENT_ID,
        "arm_id": ARM_ID,
        "model_name": "Ctrl-World",
        "public_image": CTRL_WORLD_PUBLIC_IMAGE,
        "torch_version": CTRL_WORLD_TORCH_VERSION,
        "python_dependencies": list(PYTHON_DEPENDENCIES),
        "models": list(PROVIDER_MODEL_DOWNLOAD_FREEZE),
        "model_freeze": MODEL_FREEZE,
        "source_manifest": source_manifest,
        "source_files": source_rows,
        "world_model_checkpoint_sha256": EXPECTED_WORLD_MODEL_SHA256,
        "state_stat_sha256": EXPECTED_STATE_STAT_SHA256,
        "request_sha256": validated_request["request_sha256"],
        "seed": validated_request["seed"],
        "qualification_canary_request_count": 1,
        "scientific_matrix_request_count": 0,
        "total_initial_generation_request_count": 1,
        "truth_boundary": {
            "blueprint_current_reference_not_exact_paper_reproduction": True,
            "generated_only_outputs_required": True,
            "policy_identity_forbidden_from_wam_request": True,
            "future_physical_rgb_forbidden": True,
            "recorded_action_trace_forbidden": True,
            "outcome_labels_forbidden": True,
            "technical_generation_not_causal_or_ranking_credit": True,
        },
    }
    runtime_manifest["manifest_sha256"] = canonical_sha256(runtime_manifest)
    write_json(runtime_dir / "wam_provider_runtime_manifest.json", runtime_manifest)
    rollout_manifest: dict[str, Any] = {
        "schema_version": "ctrl_world_current_reference_rollout_input.v1",
        "experiment_id": EXPERIMENT_ID,
        "arm_id": ARM_ID,
        "request_manifest_path": (
            "provider_runtime/ctrl_world_request/ctrl_world_current_reference_request.json"
        ),
        "request_sha256": validated_request["request_sha256"],
        "seed": validated_request["seed"],
        "physical_outcome_labels_accessed": False,
        "physical_future_rgb_provided_to_model": False,
        "candidate_policy_loaded_by_wam_runtime": False,
        "recorded_action_trace_used": False,
        "closed_loop_transition_count": 1,
        "claim_ceiling": "label_free_single_policy_action_wam_generation_canary",
    }
    rollout_manifest["manifest_sha256"] = canonical_sha256(rollout_manifest)
    write_json(runtime_dir / "wam_rollout_input_manifest.json", rollout_manifest)

    bundle_path = output / bundle_filename
    zip_entries: list[str] = []
    if not blockers:
        zip_entries = _write_deterministic_zip(source_root=bundle_root, bundle_path=bundle_path)
        with zipfile.ZipFile(bundle_path) as archive:
            corrupt = archive.testzip()
        if corrupt is not None:
            blockers.append(f"ctrl_world_current_reference_bundle_zip_invalid:{corrupt}")
    embedded_hashes = {
        "runtime_manifest_file_sha256": _sha256_file(
            runtime_dir / "wam_provider_runtime_manifest.json"
        ),
        "rollout_manifest_file_sha256": _sha256_file(
            runtime_dir / "wam_rollout_input_manifest.json"
        ),
        "request_manifest_file_sha256": _sha256_file(packaged_request_manifest),
        "request_sha256": validated_request["request_sha256"],
        "source_manifest_sha256": canonical_sha256(source_manifest),
        "direct_runner_sha256": _sha256_file(runtime_dir / "wam_provider_runtime_runner.py"),
        "support_runner_sha256": _sha256_file(
            runtime_dir / "ctrl_world_provider_runtime_support.py"
        ),
        "entrypoint_sha256": _sha256_file(entrypoint),
    }
    bundle_sha256 = _sha256_file(bundle_path) if bundle_path.is_file() else None
    bundle_size = bundle_path.stat().st_size if bundle_path.is_file() else 0
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "arm_id": ARM_ID,
        "bundle_sha256": bundle_sha256,
        "bundle_size_bytes": bundle_size,
        **embedded_hashes,
    }
    receipt_path = output / "ctrl_world_current_reference_provider_bundle_receipt.json"
    write_json(receipt_path, receipt)
    result = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "experiment_id": EXPERIMENT_ID,
        "arm_id": ARM_ID,
        "provider_bundle_kind": "wam",
        "bundle_path": str(bundle_path),
        "bundle_present": bundle_path.is_file(),
        "bundle_sha256": bundle_sha256,
        "bundle_size_bytes": bundle_size,
        "zip_entry_count": len(zip_entries),
        "zip_entries": zip_entries,
        "source_revision": observed_commit,
        "source_worktree_clean": not observed_status,
        "request_sha256": validated_request["request_sha256"],
        "embedded_hashes": embedded_hashes,
        "receipt_path": str(receipt_path),
        "local_bundle_ready_for_remote_staging": not blockers,
        "blockers": blockers,
        "attribution": "Ctrl-World_not_OSCAR_not_Cosmos",
    }
    write_json(output / "ctrl_world_current_reference_provider_bundle_manifest.json", result)
    return result


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "DEFAULT_BUNDLE_FILENAME",
    "EXPERIMENT_ID",
    "RECEIPT_SCHEMA_VERSION",
    "SOURCE_FILES",
    "build_ctrl_world_current_reference_provider_bundle",
    "inspect_ctrl_world_current_reference_archive_inputs",
]
