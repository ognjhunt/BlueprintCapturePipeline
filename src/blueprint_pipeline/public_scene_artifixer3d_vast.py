"""Execute one sealed ArtiFixer/3D/3D+ candidate packet on Vast.

This adapter is deliberately narrow.  It accepts a rehearsed 1--5 object
packet containing only private-derived frames, exact masks, and released
source; consumes one file-backed authority; arms an independent watchdog
before allocation; and returns candidate appearance only after object-store
cleanup and API-confirmed provider zero.  It never promotes generated pixels
to observed hidden-background or physical evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import struct
import zipfile
from typing import Any

from .common import ensure_dir, redacted_failure_detail, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import (
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers
from .public_scene_artifixer3d_bundle import (
    CHECKPOINT_REUSE_SCHEMA_VERSION,
    DEFAULT_IMAGE,
    DUAL_TARGET_PIPELINE_MODE,
    DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
    RUNTIME_REQUEST_SCHEMA_VERSION,
    RUNTIME_RESULT_SCHEMA_VERSION,
    SCHEMA_VERSION as BUNDLE_SCHEMA_VERSION,
    USE_ATTESTATION_SCHEMA_VERSION,
)
from .task_evaluation_artifact_manifest import seal_lane_terminal_artifacts
from .vast_independent_watchdog_control import (
    EVIDENCE_NAME as WATCHDOG_EVIDENCE_NAME,
    arm_independent_vast_watchdog,
    close_independent_vast_watchdog,
)
from .vast_provider_adapter import run_vast_provider_adapter
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


PROBE_KIND = "adp-artifixer3d-exact-support"
PROVIDER_BUNDLE_KIND = "adp_artifixer3d"
RESULT_SCHEMA_VERSION = "public_scene_artifixer3d_vast_run.v1"
RAW_RESULT_SCHEMA_VERSION = "public_scene_artifixer3d_raw_result.v1"
NATIVE_APPEARANCE_EXPORT_SCHEMA = (
    "public_scene_artifixer3d_native_appearance_export.v1"
)
PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION = (
    "public_scene_artifixer3d_paid_attempt_authority.v1"
)
SUPPLEMENTAL_SPEND_SCHEMA_VERSION = (
    "artifixer3d_supplemental_prior_spend_reconciliation.v1"
)
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/artifixer3d-exact-support"
INSTANCE_LABEL_PREFIX = "blueprint-groot-oscar-canary-adp-artifixer3d-"
MIN_TTL_SECONDS = 7_200
MAX_TTL_SECONDS = 21_600
MAX_HARD_CAP_USD = 10.0
MIN_GPU_RAM_MB = 78_000
MIN_COMPUTE_CAP = 800
GPU_SELECTION_POLICY = {
    "policy_id": "artifixer3d_a100_80gb_author_control",
    "allowed_gpu_keywords": ("A100",),
    "denied_gpu_keywords": ("A100X",),
    "reason": (
        "the released 1.3B author workflow is documented for one 80 GB GPU; "
        "A100 uses the released cuDNN SDPA fallback without Hopper-only kernels"
    ),
}
AUTHORIZATION_CONSUMPTION_ROOT = Path.home() / ".blueprint-spend-authority" / "consumed"
_MUTATION_ENV = ("BLUEPRINT_ALLOW_VAST_API_CALLS", "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH")
_RETRY_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"
DUAL_TARGET_CANDIDATE_MEMBER = (
    "provider_runtime/input/public_scene_artifixer3d_dual_target_inputs.v1.json"
)
LEGACY_CANDIDATE_MEMBER = (
    "provider_runtime/input/public_scene_artifixer3d_candidate_inputs.v3.json"
)
DUAL_TARGET_PHASES = [
    "dual_target_input_validation",
    "artifixer3d_distillation",
    "artifixer3d_review_render",
    "native_appearance_export",
    "external_visual_and_multiview_review",
]
DUAL_TARGET_RENDER_ONLY_PHASES = [
    "reused_checkpoint_validation",
    "deterministic_distillation_input_replay",
    "artifixer3d_review_render",
    "native_appearance_export",
    "external_visual_and_multiview_review",
]


def inspect_artifixer3d_container_image(
    *, image_ref: str, output_path: str | Path
) -> dict[str, Any]:
    """Prove the exact digest is registry-resolvable before paid allocation."""

    output = Path(output_path).expanduser().resolve()
    ensure_dir(output.parent)
    blockers: list[str] = []
    decoded: dict[str, Any] = {}
    if "@sha256:" not in image_ref:
        blockers.append("artifixer3d_container_image_not_digest_pinned")
    try:
        completed = subprocess.run(
            ["docker", "manifest", "inspect", image_ref],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        completed = None
        blockers.append(
            f"artifixer3d_container_registry_probe_failed:{redacted_failure_detail(exc)}"
        )
    if completed is not None:
        if completed.returncode != 0:
            blockers.append("artifixer3d_container_image_not_registry_resolvable")
        else:
            try:
                value = json.loads(completed.stdout)
                decoded = dict(value) if isinstance(value, Mapping) else {}
            except json.JSONDecodeError:
                blockers.append("artifixer3d_container_registry_manifest_invalid")
            if not decoded:
                blockers.append("artifixer3d_container_registry_manifest_invalid")
    manifest = {
        "schema_version": "artifixer3d_container_registry_preflight.v1",
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "image_ref": image_ref,
        "digest_pinned": "@sha256:" in image_ref,
        "registry_manifest_available": bool(decoded),
        "media_type": decoded.get("mediaType"),
        "blockers": sorted(set(blockers)),
        "raw_registry_manifest_recorded": False,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "Registry metadata reachability only; this does not prove image pull, "
            "container startup, model execution, or scientific output."
        ),
    }
    write_json(output, manifest)
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read(path: Path, *, code: str = "artifixer3d_receipt_unreadable") -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def _bound(record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(code)
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ValueError(code)
    return path


def _provider_zero_inventory(path: Path) -> None:
    value = _read(path, code="artifixer3d_supplemental_provider_inventory_unreadable")
    if (
        value.get("provider") != "vast"
        or value.get("api_confirmed") is not True
        or value.get("live_resource_count") != 0
        or value.get("resources") != []
    ):
        raise ValueError("artifixer3d_supplemental_provider_inventory_invalid")


def _gaussian_excision_spend_entry(
    *, closeout_path: Path, inventory_path: Path
) -> dict[str, Any]:
    closeout = _read(
        closeout_path, code="artifixer3d_supplemental_excision_closeout_unreadable"
    )
    _provider_zero_inventory(inventory_path)
    inventory_record = closeout.get("provider_inventory")
    cost = closeout.get("combined_estimated_cost_usd")
    if (
        closeout.get("schema_version") != "adp_gaussian_excision_provider_closeout.v1"
        or closeout.get("status") != "lane_owned_provider_zero"
        or closeout.get("receipt_digest")
        != canonical_digest(closeout, digest_field="receipt_digest")
        or closeout.get("continuing_lane_owned_spend") is not False
        or closeout.get("global_provider_zero_claimed") is not True
        or closeout.get("external_live_instances") != []
        or not isinstance(inventory_record, Mapping)
        or inventory_record.get("size_bytes") != inventory_path.stat().st_size
        or inventory_record.get("sha256") != _sha256(inventory_path)
        or isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        raise ValueError("artifixer3d_supplemental_excision_closeout_invalid")
    return {
        "kind": "gaussian_excision_provider_closeout",
        "terminal_receipt": _record(closeout_path),
        "terminal_receipt_digest": closeout["receipt_digest"],
        "provider_zero_inventory": _record(inventory_path),
        "cost_usd": round(float(cost), 6),
    }


def _retained_render_spend_entry(
    *, result_path: Path, cleanup_path: Path, inventory_path: Path
) -> dict[str, Any]:
    result = _read(result_path, code="artifixer3d_supplemental_render_result_unreadable")
    cleanup = _read(cleanup_path, code="artifixer3d_supplemental_render_cleanup_unreadable")
    _provider_zero_inventory(inventory_path)
    watchdog = result.get("independent_watchdog")
    cost = result.get("estimated_cost_usd")
    if (
        result.get("schema_version") != "adp009d_retained_scene_gpu_render_vast_run.v1"
        or result.get("status") != "completed"
        or result.get("receipt_digest")
        != canonical_digest(result, digest_field="receipt_digest")
        or result.get("retry_cap") != 0
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or not isinstance(watchdog, Mapping)
        or watchdog.get("provider_absence_confirmed") is not True
        or cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        raise ValueError("artifixer3d_supplemental_render_result_invalid")
    return {
        "kind": "retained_scene_gpu_render_closeout",
        "terminal_receipt": _record(result_path),
        "terminal_receipt_digest": result["receipt_digest"],
        "object_store_cleanup": _record(cleanup_path),
        "provider_zero_inventory": _record(inventory_path),
        "cost_usd": round(float(cost), 6),
    }


def materialize_artifixer3d_supplemental_spend_reconciliation(
    *,
    gaussian_excision_closeouts: Sequence[Mapping[str, str | Path]],
    retained_scene_render_attempts: Sequence[Mapping[str, str | Path]],
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind paid closeouts executed after the predecessor ArtiFixer authority."""

    entries = [
        _gaussian_excision_spend_entry(
            closeout_path=Path(row["closeout_path"]).expanduser().resolve(),
            inventory_path=Path(row["provider_inventory_path"]).expanduser().resolve(),
        )
        for row in gaussian_excision_closeouts
    ]
    entries.extend(
        _retained_render_spend_entry(
            result_path=Path(row["result_path"]).expanduser().resolve(),
            cleanup_path=Path(row["cleanup_path"]).expanduser().resolve(),
            inventory_path=Path(row["provider_inventory_path"]).expanduser().resolve(),
        )
        for row in retained_scene_render_attempts
    )
    digests = [str(row["terminal_receipt_digest"]) for row in entries]
    if not entries or len(digests) != len(set(digests)):
        raise ValueError("artifixer3d_supplemental_spend_entries_invalid")
    value: dict[str, Any] = {
        "schema_version": SUPPLEMENTAL_SPEND_SCHEMA_VERSION,
        "status": "all_supplemental_spend_terminal_and_provider_zero",
        "entries": entries,
        "total_cost_usd": round(sum(float(row["cost_usd"]) for row in entries), 6),
        "continuing_spend": False,
        "provider_zero_confirmed_for_every_entry": True,
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError("artifixer3d_supplemental_spend_output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, value)
    return value


def _validate_supplemental_spend_reconciliation(
    path: Path,
) -> tuple[dict[str, Any], float]:
    value = _read(path, code="artifixer3d_supplemental_spend_unreadable")
    entries = value.get("entries")
    if (
        value.get("schema_version") != SUPPLEMENTAL_SPEND_SCHEMA_VERSION
        or value.get("status") != "all_supplemental_spend_terminal_and_provider_zero"
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or value.get("continuing_spend") is not False
        or value.get("provider_zero_confirmed_for_every_entry") is not True
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("artifixer3d_supplemental_spend_invalid")
    validated: list[dict[str, Any]] = []
    for row in entries:
        if not isinstance(row, Mapping):
            raise ValueError("artifixer3d_supplemental_spend_invalid")
        receipt_path = _bound(
            row.get("terminal_receipt"), code="artifixer3d_supplemental_terminal_unbound"
        )
        inventory_path = _bound(
            row.get("provider_zero_inventory"),
            code="artifixer3d_supplemental_inventory_unbound",
        )
        if row.get("kind") == "gaussian_excision_provider_closeout":
            expected = _gaussian_excision_spend_entry(
                closeout_path=receipt_path, inventory_path=inventory_path
            )
        elif row.get("kind") == "retained_scene_gpu_render_closeout":
            cleanup_path = _bound(
                row.get("object_store_cleanup"),
                code="artifixer3d_supplemental_cleanup_unbound",
            )
            expected = _retained_render_spend_entry(
                result_path=receipt_path,
                cleanup_path=cleanup_path,
                inventory_path=inventory_path,
            )
        else:
            raise ValueError("artifixer3d_supplemental_spend_kind_invalid")
        if dict(row) != expected:
            raise ValueError("artifixer3d_supplemental_spend_entry_mismatch")
        validated.append(expected)
    total = round(sum(float(row["cost_usd"]) for row in validated), 6)
    if total != value.get("total_cost_usd"):
        raise ValueError("artifixer3d_supplemental_spend_total_mismatch")
    return value, total


def _zip_json(archive: zipfile.ZipFile, name: str, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(archive.read(name).decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def _zip_text(archive: zipfile.ZipFile, name: str, *, code: str) -> str:
    try:
        return archive.read(name).decode("utf-8")
    except (KeyError, UnicodeDecodeError) as exc:
        raise ValueError(code) from exc


def _zip_bound_input_member(
    archive: zipfile.ZipFile, record: Any, *, code: str
) -> str:
    """Re-hash one input member against its immutable bundle record."""

    if not isinstance(record, Mapping):
        raise ValueError(code)
    relative = Path(str(record.get("relative_path") or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(code)
    name = f"provider_runtime/input/{relative.as_posix()}"
    try:
        info = archive.getinfo(name)
        digest = hashlib.sha256()
        with archive.open(info) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except KeyError as exc:
        raise ValueError(code) from exc
    if (
        info.is_dir()
        or info.file_size != record.get("size_bytes")
        or "sha256:" + digest.hexdigest() != record.get("sha256")
    ):
        raise ValueError(code)
    return name


def _validate_parent_execution_authority(
    attestation: Mapping[str, Any], *, publisher_scene_id: str
) -> tuple[Path, dict[str, Any]]:
    path = _bound(
        attestation.get("parent_execution_authority"),
        code="artifixer3d_parent_execution_authority_unbound",
    )
    authority = _read(path, code="artifixer3d_parent_execution_authority_unreadable")
    paid = authority.get("paid_compute")
    if (
        authority.get("schema_version") != "third_scene_dual_task_execution_authority.v1"
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
        or not publisher_scene_id
        or authority.get("publisher_scene_id") != publisher_scene_id
        or not isinstance(paid, Mapping)
        or paid.get("provider") != "vast"
        or paid.get("zero_retry") is not True
        or paid.get("provider_zero_required_for_lane") is not True
        or paid.get("hard_total_spend_cap_usd") != 12.0
    ):
        raise ValueError("artifixer3d_parent_execution_authority_invalid")
    return path, authority


def validate_artifixer3d_bundle(receipt_path: str | Path) -> dict[str, Any]:
    """Re-open every immutable bundle binding before paid admission."""

    path = Path(receipt_path).expanduser().resolve()
    receipt = _read(path, code="artifixer3d_bundle_receipt_unreadable")
    if (
        path.is_symlink()
        or receipt.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or receipt.get("status") != "sealed_rehearsal_passed_no_upload_no_execution"
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("provider_mutations_performed") != 0
        or receipt.get("private_derived_upload_performed") is not False
        or not 1 <= int(receipt.get("replacement_object_count") or 0) <= 5
        or receipt.get("local_rehearsal", {}).get("status") != "passed"
        or receipt.get("local_rehearsal", {}).get("provider_mutations_performed") != 0
    ):
        raise ValueError("artifixer3d_bundle_receipt_invalid")
    bundle_path = _bound(receipt.get("bundle"), code="artifixer3d_bundle_unbound")
    checkpoint_reuse: dict[str, Any] | None = None
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("artifixer3d_bundle_zip_integrity_failed")
            names = set(archive.namelist())
            manifest = _zip_json(
                archive,
                "provider_runtime/artifixer3d_bundle_manifest.json",
                code="artifixer3d_bundle_manifest_invalid",
            )
            request = _zip_json(
                archive,
                "provider_runtime/artifixer3d_runtime_request.json",
                code="artifixer3d_bundle_request_invalid",
            )
            pipeline_mode = str(request.get("pipeline_mode") or "")
            dual_target_family = pipeline_mode in {
                DUAL_TARGET_PIPELINE_MODE,
                DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
            }
            candidate_member = (
                DUAL_TARGET_CANDIDATE_MEMBER
                if dual_target_family
                else LEGACY_CANDIDATE_MEMBER
            )
            required = {
                "provider_runtime/run_public_scene_artifixer3d.sh",
                "provider_runtime/public_scene_artifixer3d_runner.py",
                "provider_runtime/artifixer3d_bundle_manifest.json",
                "provider_runtime/artifixer3d_runtime_request.json",
                candidate_member,
                "provider_runtime/artifixer3d_use_attestation.json",
            }
            if not required.issubset(names):
                raise ValueError("artifixer3d_bundle_required_entries_missing")
            candidate = _zip_json(
                archive,
                candidate_member,
                code="artifixer3d_bundle_candidate_invalid",
            )
            attestation = _zip_json(
                archive,
                "provider_runtime/artifixer3d_use_attestation.json",
                code="artifixer3d_bundle_attestation_invalid",
            )
            if pipeline_mode == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE:
                reuse = request.get("artifixer3d", {}).get("checkpoint_reuse")
                if (
                    not isinstance(reuse, Mapping)
                    or reuse.get("schema_version")
                    != CHECKPOINT_REUSE_SCHEMA_VERSION
                    or reuse.get("reuse_digest")
                    != canonical_digest(reuse, digest_field="reuse_digest")
                    or manifest.get("checkpoint_reuse") != reuse
                    or receipt.get("checkpoint_reuse_digest")
                    != reuse.get("reuse_digest")
                ):
                    raise ValueError("artifixer3d_checkpoint_reuse_binding_invalid")
                for field in (
                    "source_attempt_authority",
                    "source_attempt_result",
                    "source_provider_zero",
                    "source_runtime_result",
                ):
                    _zip_bound_input_member(
                        archive,
                        reuse.get(field),
                        code="artifixer3d_checkpoint_reuse_receipt_unbound",
                    )
                for row in reuse.get("checkpoints") or []:
                    _zip_bound_input_member(
                        archive,
                        row.get("checkpoint") if isinstance(row, Mapping) else None,
                        code="artifixer3d_checkpoint_reuse_checkpoint_unbound",
                    )
                checkpoint_reuse = dict(reuse)
            blockers = provider_runtime_contract_blockers(
                provider_bundle_kind=PROVIDER_BUNDLE_KIND,
                entrypoint_text=_zip_text(
                    archive,
                    "provider_runtime/run_public_scene_artifixer3d.sh",
                    code="artifixer3d_bundle_entrypoint_invalid",
                ),
                runner_text=_zip_text(
                    archive,
                    "provider_runtime/public_scene_artifixer3d_runner.py",
                    code="artifixer3d_bundle_runner_invalid",
                ),
            )
    except zipfile.BadZipFile as exc:
        raise ValueError("artifixer3d_bundle_zip_invalid") from exc
    if blockers:
        raise ValueError("artifixer3d_bundle_runtime_contract_invalid")
    identity = manifest.get("blueprint_source_identity")
    tasks = candidate.get("tasks")
    pipeline_mode = str(request.get("pipeline_mode") or "")
    render_only_mode = pipeline_mode == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
    dual_target_mode = pipeline_mode in {
        DUAL_TARGET_PIPELINE_MODE,
        DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
    }
    legacy_request_valid = (
        pipeline_mode in {"", "full_artifixer3d_plus"}
        and request.get("outside_exact_support_changed_pixels_permitted") == 0
        and request.get("repair_target")
        == "plausible_object_free_background_inside_exact_support_only"
        and request.get("direct_editor_backend")
        in {"artifixer", "qwen_image_edit_2511", "vibe_image_edit"}
    )
    dual_target_request_valid = (
        pipeline_mode == DUAL_TARGET_PIPELINE_MODE
        and candidate.get("schema_version")
        == "public_scene_artifixer3d_dual_target_inputs.v1"
        and candidate.get("pipeline_mode") == DUAL_TARGET_PIPELINE_MODE
        and manifest.get("pipeline_mode") == DUAL_TARGET_PIPELINE_MODE
        and receipt.get("pipeline_mode") == DUAL_TARGET_PIPELINE_MODE
        and request.get("direct_editor_backend") == "none"
        and request.get("semantic_editor_only") is False
        and request.get("phases") == DUAL_TARGET_PHASES
        and request.get("outside_exact_support_changed_pixels_permitted")
        in {None, "unconstrained_for_raw_representation_review"}
        and request.get("outside_support_invariance_gate")
        == "deferred_until_final_soft_composite"
        and request.get("repair_target")
        == "whole_frame_semantic_empty_scene_distillation_with_original_outside_support_anchors"
    )
    render_only_request_valid = (
        render_only_mode
        and candidate.get("schema_version")
        == "public_scene_artifixer3d_dual_target_inputs.v1"
        and candidate.get("pipeline_mode") == DUAL_TARGET_PIPELINE_MODE
        and manifest.get("pipeline_mode") == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
        and receipt.get("pipeline_mode") == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
        and request.get("direct_editor_backend") == "none"
        and request.get("semantic_editor_only") is False
        and request.get("phases") == DUAL_TARGET_RENDER_ONLY_PHASES
        and request.get("outside_exact_support_changed_pixels_permitted")
        == "unconstrained_for_raw_representation_review"
        and request.get("outside_support_invariance_gate")
        == "deferred_until_final_soft_composite"
        and request.get("repair_target")
        == "render_only_replay_of_zero_closed_dual_target_artifixer3d_checkpoint"
        and isinstance(checkpoint_reuse, Mapping)
        and checkpoint_reuse.get("source_pipeline_mode")
        == DUAL_TARGET_PIPELINE_MODE
        and checkpoint_reuse.get("source_candidate_input_receipt_digest")
        == candidate.get("receipt_digest")
        and checkpoint_reuse.get("training_reexecution_permitted") is False
        and checkpoint_reuse.get("direct_inference_permitted") is False
        and checkpoint_reuse.get("artifixer3d_plus_permitted") is False
        and checkpoint_reuse.get("provider_zero_confirmed_before_reuse") is True
        and request.get("artifixer3d", {}).get("training_permitted") is False
        and request.get("artifixer3d", {}).get("distillation_input_replay_only")
        is True
    )
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
        or request.get("schema_version") != RUNTIME_REQUEST_SCHEMA_VERSION
        or request.get("runtime_request_digest")
        != canonical_digest(request, digest_field="runtime_request_digest")
        or request.get("runtime_request_digest") != receipt.get("runtime_request_digest")
        or manifest.get("manifest_digest") != receipt.get("manifest_digest")
        or candidate.get("receipt_digest") != receipt.get("candidate_input_receipt_digest")
        or candidate.get("replacement_object_count") != receipt.get("replacement_object_count")
        or not isinstance(tasks, list)
        or len(tasks) != receipt.get("replacement_object_count")
        or request.get("task_ids") != receipt.get("task_ids")
        or request.get("source_object_restoration_permitted") is not False
        or not (
            legacy_request_valid
            or dual_target_request_valid
            or render_only_request_valid
        )
        or manifest.get("direct_editor_backend")
        != request.get("direct_editor_backend")
        or receipt.get("direct_editor_backend")
        != request.get("direct_editor_backend")
        or receipt.get("semantic_editor_only")
        != (request.get("semantic_editor_only") is True)
        or manifest.get("semantic_editor_only")
        != (request.get("semantic_editor_only") is True)
        or manifest.get("contains_raw_dataset_bytes") is not False
        or manifest.get("contains_model_weights") is not render_only_mode
        or manifest.get(
            "contains_reused_private_derived_3dgrut_checkpoint", False
        )
        is not render_only_mode
        or manifest.get("contains_released_direct_model_weights", False) is not False
        or manifest.get("container_image") != DEFAULT_IMAGE
        or receipt.get("container_image") != DEFAULT_IMAGE
        or not isinstance(identity, Mapping)
        or identity != receipt.get("blueprint_source_identity")
        or identity.get("tracked_files_clean") is not True
        or len(str(identity.get("commit") or "")) != 40
        or len(str(identity.get("tree") or "")) != 40
        or attestation.get("schema_version") != USE_ATTESTATION_SCHEMA_VERSION
        or attestation.get("attestation_digest")
        != canonical_digest(attestation, digest_field="attestation_digest")
        or attestation.get("attestation_digest") != receipt.get("use_attestation_digest")
        or attestation.get("private_derived_input_upload_authorized") is not True
        or attestation.get("raw_dataset_bytes_upload_authorized") is not False
        or attestation.get("provider_training_authorized") is not False
        or attestation.get("internal_noncommercial_research_and_development_only")
        is not True
        or attestation.get("simulator_or_generated_output_is_physical_evidence") is not False
    ):
        raise ValueError("artifixer3d_bundle_binding_invalid")
    task_ids: list[str] = []
    task_camera_counts: dict[str, int] = {}
    task_training_record_counts: dict[str, int] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("artifixer3d_bundle_task_invalid")
        task_id = str(task.get("task_id") or "")
        camera_count = (
            task.get("physical_camera_count")
            if dual_target_mode
            else task.get("camera_count")
        )
        training_record_count = (
            task.get("training_record_count") if dual_target_mode else camera_count
        )
        training_records = task.get("training_records")
        if (
            not task_id
            or task_id in task_camera_counts
            or isinstance(camera_count, bool)
            or not isinstance(camera_count, int)
            or camera_count < 2
            or isinstance(training_record_count, bool)
            or not isinstance(training_record_count, int)
            or training_record_count != (2 * camera_count if dual_target_mode else camera_count)
            or (
                dual_target_mode
                and training_records is not None
                and len(training_records) != training_record_count
            )
            or (
                dual_target_mode
                and len(task.get("frames") or []) != camera_count
            )
            or (
                not dual_target_mode
                and len(task.get("frames") or []) != camera_count
            )
        ):
            raise ValueError("artifixer3d_bundle_task_invalid")
        task_ids.append(task_id)
        task_camera_counts[task_id] = camera_count
        task_training_record_counts[task_id] = training_record_count
    if task_ids != receipt.get("task_ids"):
        raise ValueError("artifixer3d_bundle_task_order_invalid")
    reused_checkpoints: dict[str, dict[str, Any]] = {}
    if render_only_mode:
        assert isinstance(checkpoint_reuse, Mapping)
        checkpoint_rows = checkpoint_reuse.get("checkpoints")
        source_zip = checkpoint_reuse.get("source_provider_output_zip")
        steps = request.get("artifixer3d", {}).get("steps")
        if (
            not isinstance(checkpoint_rows, list)
            or len(checkpoint_rows) != len(task_ids)
            or not isinstance(source_zip, Mapping)
            or isinstance(source_zip.get("size_bytes"), bool)
            or not isinstance(source_zip.get("size_bytes"), int)
            or source_zip["size_bytes"] <= 0
            or not str(source_zip.get("sha256") or "").startswith("sha256:")
        ):
            raise ValueError("artifixer3d_checkpoint_reuse_binding_invalid")
        for task_id, row in zip(task_ids, checkpoint_rows):
            record = row.get("checkpoint") if isinstance(row, Mapping) else None
            if (
                not isinstance(row, Mapping)
                or row.get("task_id") != task_id
                or row.get("steps") != steps
                or not isinstance(row.get("source_provider_zip_member"), str)
                or not row["source_provider_zip_member"]
                or not isinstance(record, Mapping)
                or isinstance(record.get("size_bytes"), bool)
                or not isinstance(record.get("size_bytes"), int)
                or record["size_bytes"] <= 0
                or not str(record.get("sha256") or "").startswith("sha256:")
            ):
                raise ValueError("artifixer3d_checkpoint_reuse_binding_invalid")
            reused_checkpoints[task_id] = {
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
                "source_provider_zip_member": row["source_provider_zip_member"],
            }
    publisher_scene_id = str(candidate.get("publisher_scene_id") or "")
    if attestation.get("publisher_scene_id") != publisher_scene_id:
        raise ValueError("artifixer3d_bundle_scene_binding_invalid")
    parent_path, parent = _validate_parent_execution_authority(
        attestation, publisher_scene_id=publisher_scene_id
    )
    parent_allowlist = sorted(
        set(
            int(value)
            for value in parent.get("paid_compute", {}).get(
                "external_instance_allowlist", []
            )
        )
    )
    allowed_active_instance_ids = sorted(
        set(int(value) for value in manifest.get("allowed_active_instance_ids") or [])
    )
    return {
        "receipt_path": str(path),
        "receipt_sha256": _sha256(path),
        "receipt_digest": receipt["receipt_digest"],
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": _sha256(bundle_path),
        "manifest_digest": manifest["manifest_digest"],
        "runtime_request_digest": request["runtime_request_digest"],
        "candidate_input_receipt_digest": candidate["receipt_digest"],
        "use_attestation_digest": attestation["attestation_digest"],
        "parent_execution_authority_path": str(parent_path),
        "parent_execution_authority_digest": parent["authority_digest"],
        "aggregate_goal_spend_cap_usd": parent["paid_compute"]["hard_total_spend_cap_usd"],
        "blueprint_source_identity": dict(identity),
        "container_image": DEFAULT_IMAGE,
        "publisher_scene_id": publisher_scene_id,
        "allowed_active_instance_ids": allowed_active_instance_ids,
        "forbidden_external_instance_ids": sorted(
            set(parent_allowlist) - set(allowed_active_instance_ids)
        ),
        "replacement_object_count": receipt["replacement_object_count"],
        "task_ids": task_ids,
        "task_camera_counts": task_camera_counts,
        "task_training_record_counts": task_training_record_counts,
        "pipeline_mode": pipeline_mode or "legacy_exact_support_full_chain",
        "phases": list(request.get("phases") or []),
        "direct_editor_backend": request["direct_editor_backend"],
        "semantic_editor_only": request.get("semantic_editor_only") is True,
        "checkpoint_reuse_digest": (
            checkpoint_reuse.get("reuse_digest")
            if isinstance(checkpoint_reuse, Mapping)
            else None
        ),
        "reused_checkpoints": reused_checkpoints,
    }


def _validate_prior_authority_chain(path: Path, *, seen: set[Path] | None = None) -> dict[str, Any]:
    """Recursively re-open the predecessor authority's complete spend chain."""

    seen = set() if seen is None else seen
    resolved = path.expanduser().resolve()
    if resolved in seen or resolved.is_symlink() or not resolved.is_file():
        raise ValueError("artifixer3d_prior_authority_chain_invalid")
    seen.add(resolved)
    value = _read(resolved, code="artifixer3d_prior_authority_unreadable")
    if (
        value.get("schema_version")
        != "public_scene_aura_exact_residual_paid_attempt_authority.v1"
        or value.get("authorization_digest")
        != canonical_digest(value, digest_field="authorization_digest")
        or value.get("automatic_paid_retry_authorized") is not False
        or value.get("maximum_automatic_retries") != 0
        or value.get("maximum_paid_attempts") != 1
        or value.get("aggregate_goal_spend_cap_usd") != 12.0
        or isinstance(value.get("prior_goal_spend_usd"), bool)
        or not isinstance(value.get("prior_goal_spend_usd"), (int, float))
    ):
        raise ValueError("artifixer3d_prior_authority_invalid")
    for field in (
        "previous_terminal_execution_result",
        "previous_runtime_result",
        "previous_teardown",
        "previous_watchdog",
        "previous_object_store_cleanup",
        "prior_provider_runtime_campaign",
    ):
        _bound(value.get(field), code="artifixer3d_prior_authority_dependency_unbound")
    for record in value.get("additional_terminal_spend_receipts") or []:
        _bound(record, code="artifixer3d_prior_authority_spend_unbound")
    parent = value.get("prior_manual_corrected_attempt_authority")
    if parent is not None:
        _validate_prior_authority_chain(
            _bound(parent, code="artifixer3d_prior_authority_parent_unbound"),
            seen=seen,
        )
    return value


def _validate_prior_terminal_result(
    path: Path, *, prior_authority: Mapping[str, Any]
) -> tuple[dict[str, Any], float, float]:
    result = _read(path, code="artifixer3d_prior_terminal_result_unreadable")
    cost = result.get("estimated_cost_usd")
    if (
        result.get("schema_version") != "public_scene_aura_exact_residual_vast_run.v1"
        or result.get("status") != "completed"
        or result.get("retry_cap") != 0
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("all_staged_objects_absent") is not True
        or result.get("authorization_consumption", {}).get("authorization_digest")
        != prior_authority.get("authorization_digest")
        or result.get("authorization_consumption", {}).get("status") != "consumed"
        or isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        raise ValueError("artifixer3d_prior_terminal_result_invalid")
    teardown_path = Path(str(result.get("teardown_manifest_path") or "")).resolve()
    watchdog_path = Path(str(result.get("watchdog_receipt_path") or "")).resolve()
    cleanup_path = path.parent / "object_store_staging" / "wam_provider_object_store_cleanup.json"
    teardown = _read(teardown_path, code="artifixer3d_prior_teardown_unreadable")
    watchdog = _read(watchdog_path, code="artifixer3d_prior_watchdog_unreadable")
    cleanup = _read(cleanup_path, code="artifixer3d_prior_cleanup_unreadable")
    if (
        teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or watchdog.get("schema_version") != "groot_oscar_runpod_canary_watchdog.v1"
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or watchdog.get("final_global_inventory", {}).get("live_resource_count") != 0
        or watchdog.get("final_global_inventory", {}).get("api_confirmed") is not True
        or cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
    ):
        raise ValueError("artifixer3d_prior_terminal_closeout_invalid")
    return result, round(float(cost), 6)


def _validate_prior_artifixer_attempt(
    *,
    authority_path: Path,
    result_path: Path,
    cleanup_path: Path,
    provider_zero_path: Path,
) -> tuple[dict[str, Any], float]:
    """Re-open a predecessor ArtiFixer attempt, including its zero closeout."""

    authority = _read(
        authority_path, code="artifixer3d_predecessor_authority_unreadable"
    )
    result = _read(result_path, code="artifixer3d_predecessor_result_unreadable")
    cleanup = _read(cleanup_path, code="artifixer3d_predecessor_cleanup_unreadable")
    provider_zero = _read(
        provider_zero_path, code="artifixer3d_predecessor_provider_zero_unreadable"
    )
    inventory = provider_zero.get("inventory")
    cost = result.get("estimated_cost_usd")
    result_mutations = result.get("provider_mutations_performed")
    if result_mutations is None and provider_zero.get(
        "provider_mutations_performed_by_attempt"
    ) == 1:
        adapter_path = _bound(
            provider_zero.get("provider_adapter"),
            code="artifixer3d_predecessor_adapter_unbound",
        )
        adapter = _read(
            adapter_path, code="artifixer3d_predecessor_adapter_unreadable"
        )
        classification = adapter.get("provider_attempt_classification")
        if (
            adapter.get("schema_version") != "vast_provider_adapter_result.v1"
            or adapter.get("status") != "failed"
            or adapter.get("provider_create_attempted") is not True
            or adapter.get("api_call_performed") is not True
            or adapter.get("continuing_spend_from_this_run") is not False
            or not adapter.get("vast_instance_ids")
            or adapter.get("estimated_cost_usd") != cost
            or not isinstance(classification, Mapping)
            or classification.get("classification") != "pre_execution_provider_null"
            or classification.get("provider_bundle_started") is not False
            or classification.get("provider_entrypoint_started") is not False
            or classification.get("provider_output_returned") is not False
        ):
            raise ValueError("artifixer3d_predecessor_adapter_invalid")
        result_mutations = 1
    if cost is None and result.get("provider_mutations_performed") == 0:
        cost = 0.0
    if (
        authority.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION
        or authority.get("authorization_digest")
        != canonical_digest(authority, digest_field="authorization_digest")
        or authority.get("automatic_paid_retry_authorized") is not False
        or authority.get("maximum_paid_attempts") != 1
        or result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("status") not in {"blocked", "completed"}
        or result.get("retry_cap") != 0
        or result.get("authorization_consumption", {}).get("status") != "consumed"
        or result.get("authorization_consumption", {}).get("authorization_digest")
        != authority.get("authorization_digest")
        or result.get("all_staged_objects_absent") is not True
        or isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
        or cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or provider_zero.get("schema_version")
        != "artifixer3d_postblocked_provider_zero.v1"
        or provider_zero.get("attempt_authority_digest")
        != authority.get("authorization_digest")
        or provider_zero.get("provider_mutations_performed_by_attempt")
        != result_mutations
        or provider_zero.get("provider_zero_confirmed") is not True
        or not isinstance(inventory, Mapping)
        or inventory.get("api_confirmed") is not True
        or inventory.get("live_resource_count") != 0
    ):
        raise ValueError("artifixer3d_predecessor_attempt_invalid")
    attempt_cost = round(float(cost), 6)
    lineage_cost = attempt_cost
    predecessor = authority.get("prior_artifixer_attempt")
    if predecessor is not None:
        if not isinstance(predecessor, Mapping):
            raise ValueError("artifixer3d_predecessor_lineage_invalid")
        nested_authority_path = _bound(
            predecessor.get("authority"),
            code="artifixer3d_predecessor_lineage_authority_unbound",
        )
        nested_result_path = _bound(
            predecessor.get("terminal_result"),
            code="artifixer3d_predecessor_lineage_result_unbound",
        )
        nested_cleanup_path = _bound(
            predecessor.get("object_store_cleanup"),
            code="artifixer3d_predecessor_lineage_cleanup_unbound",
        )
        nested_zero_path = _bound(
            predecessor.get("provider_zero"),
            code="artifixer3d_predecessor_lineage_zero_unbound",
        )
        nested_authority, nested_attempt_cost, nested_lineage_cost = (
            _validate_prior_artifixer_attempt(
                authority_path=nested_authority_path,
                result_path=nested_result_path,
                cleanup_path=nested_cleanup_path,
                provider_zero_path=nested_zero_path,
            )
        )
        recorded_lineage_cost = predecessor.get(
            "lineage_cost_usd", predecessor.get("terminal_cost_usd")
        )
        if (
            predecessor.get("authority_digest")
            != nested_authority.get("authorization_digest")
            or predecessor.get("terminal_cost_usd") != nested_attempt_cost
            or recorded_lineage_cost != nested_lineage_cost
        ):
            raise ValueError("artifixer3d_predecessor_lineage_mismatch")
        lineage_cost = round(lineage_cost + nested_lineage_cost, 6)
    return authority, attempt_cost, lineage_cost


def materialize_artifixer3d_paid_attempt_authority(
    *,
    bundle_receipt_path: str | Path,
    prior_aura_authority_path: str | Path,
    prior_terminal_result_path: str | Path,
    authorization_reference: str,
    authorized_by: str,
    authorized_on: str,
    blueprint_commit: str,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    output_path: str | Path,
    prior_artifixer_authority_path: str | Path | None = None,
    prior_artifixer_result_path: str | Path | None = None,
    prior_artifixer_cleanup_path: str | Path | None = None,
    prior_artifixer_provider_zero_path: str | Path | None = None,
    supplemental_prior_spend_reconciliation_path: str | Path | None = None,
) -> dict[str, Any]:
    """Seal one zero-retry authority chained through all previous scene spend."""

    bundle = validate_artifixer3d_bundle(bundle_receipt_path)
    prior_authority_path = Path(prior_aura_authority_path).expanduser().resolve()
    prior_authority = _validate_prior_authority_chain(prior_authority_path)
    terminal_path = Path(prior_terminal_result_path).expanduser().resolve()
    _, latest_cost = _validate_prior_terminal_result(
        terminal_path, prior_authority=prior_authority
    )
    predecessor_paths = (
        prior_artifixer_authority_path,
        prior_artifixer_result_path,
        prior_artifixer_cleanup_path,
        prior_artifixer_provider_zero_path,
    )
    predecessor: dict[str, Any] | None = None
    predecessor_cost = 0.0
    if any(path is not None for path in predecessor_paths):
        if not all(path is not None for path in predecessor_paths):
            raise ValueError("artifixer3d_predecessor_attempt_incomplete")
        resolved_predecessor_paths = tuple(
            Path(str(path)).expanduser().resolve() for path in predecessor_paths
        )
        (
            predecessor_authority,
            predecessor_attempt_cost,
            predecessor_cost,
        ) = _validate_prior_artifixer_attempt(
            authority_path=resolved_predecessor_paths[0],
            result_path=resolved_predecessor_paths[1],
            cleanup_path=resolved_predecessor_paths[2],
            provider_zero_path=resolved_predecessor_paths[3],
        )
        predecessor = {
            "authority": _record(resolved_predecessor_paths[0]),
            "authority_digest": predecessor_authority["authorization_digest"],
            "terminal_result": _record(resolved_predecessor_paths[1]),
            "object_store_cleanup": _record(resolved_predecessor_paths[2]),
            "provider_zero": _record(resolved_predecessor_paths[3]),
            "terminal_cost_usd": predecessor_attempt_cost,
            "lineage_cost_usd": predecessor_cost,
        }
    supplemental: dict[str, Any] | None = None
    supplemental_cost = 0.0
    if supplemental_prior_spend_reconciliation_path is not None:
        supplemental_path = Path(supplemental_prior_spend_reconciliation_path).expanduser().resolve()
        supplemental_receipt, supplemental_cost = (
            _validate_supplemental_spend_reconciliation(supplemental_path)
        )
        supplemental = {
            **_record(supplemental_path),
            "receipt_digest": supplemental_receipt["receipt_digest"],
            "total_cost_usd": supplemental_cost,
        }
    prior_spend = round(
        float(prior_authority["prior_goal_spend_usd"])
        + latest_cost
        + predecessor_cost
        + supplemental_cost,
        6,
    )
    aggregate_cap = float(bundle["aggregate_goal_spend_cap_usd"])
    if (
        not authorization_reference.strip()
        or not authorized_by.strip()
        or not authorized_on.strip()
        or blueprint_commit != bundle["blueprint_source_identity"]["commit"]
        or not 0 < max_hourly_rate_usd <= hard_cap_usd <= MAX_HARD_CAP_USD
        or not MIN_TTL_SECONDS <= hard_ttl_seconds <= MAX_TTL_SECONDS
        or hard_ttl_seconds * max_hourly_rate_usd / 3600 > hard_cap_usd
        or prior_spend + hard_cap_usd > aggregate_cap
    ):
        raise ValueError("artifixer3d_paid_attempt_authority_configuration_invalid")
    authority: dict[str, Any] = {
        "schema_version": PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": authorization_reference.strip(),
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_on.strip(),
        "purpose": (
            "one_shot_artifixer3d_checkpoint_render_only_review_execution"
            if bundle.get("pipeline_mode") == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
            else "one_shot_artifixer3d_object_free_exact_support_candidate_execution"
        ),
        "provider": "vast",
        "paid_compute_authorized": True,
        "automatic_paid_retry_authorized": False,
        "maximum_automatic_retries": 0,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "zero_retry": True,
        "bundle_receipt": _record(Path(bundle["receipt_path"])),
        "bundle_receipt_digest": bundle["receipt_digest"],
        "bundle_sha256": bundle["bundle_sha256"],
        "manifest_digest": bundle["manifest_digest"],
        "runtime_request_digest": bundle["runtime_request_digest"],
        "checkpoint_reuse_digest": bundle.get("checkpoint_reuse_digest"),
        "candidate_input_receipt_digest": bundle["candidate_input_receipt_digest"],
        "use_attestation_digest": bundle["use_attestation_digest"],
        "parent_execution_authority_digest": bundle[
            "parent_execution_authority_digest"
        ],
        "blueprint_commit": blueprint_commit,
        "blueprint_tree": bundle["blueprint_source_identity"]["tree"],
        "container_image": bundle["container_image"],
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "aggregate_goal_spend_before_attempt_usd": prior_spend,
        "aggregate_goal_spend_cap_usd": aggregate_cap,
        "prior_aura_authority": _record(prior_authority_path),
        "prior_aura_authority_digest": prior_authority["authorization_digest"],
        "prior_terminal_result": _record(terminal_path),
        "prior_terminal_cost_usd": latest_cost,
        "prior_artifixer_attempt": predecessor,
        "supplemental_prior_spend_reconciliation": supplemental,
        "external_active_instance_allowlist": bundle["allowed_active_instance_ids"],
        "forbidden_external_instance_ids": bundle[
            "forbidden_external_instance_ids"
        ],
        "private_derived_upload_only": True,
        "raw_interiorgs_upload_authorized": False,
        "raw_dataset_bytes_upload_authorized": False,
        "provider_training_authorized": False,
        "publication_authorized": False,
        "commercial_use_authorized": False,
        "exact_mask_only_edits_required": True,
        "source_object_restoration_authorized": False,
        "generated_output_is_physical_evidence": False,
        "authorization_digest": "",
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError("artifixer3d_paid_attempt_authority_output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, authority)
    validate_artifixer3d_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=bundle["allowed_active_instance_ids"],
    )
    return authority


def validate_artifixer3d_paid_attempt_authority(
    authority: Mapping[str, Any],
    *,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    allowed_active_instance_ids: Sequence[int],
) -> dict[str, Any]:
    """Fail closed if any bundle, spend-chain, cap, or rights byte drifts."""

    value = dict(authority)
    if (
        value.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION
        or value.get("authorization_digest")
        != canonical_digest(value, digest_field="authorization_digest")
        or value.get("paid_compute_authorized") is not True
        or value.get("automatic_paid_retry_authorized") is not False
        or value.get("maximum_automatic_retries") != 0
        or value.get("maximum_paid_attempts") != 1
        or value.get("maximum_provider_allocations") != 1
        or value.get("zero_retry") is not True
        or value.get("purpose")
        != (
            "one_shot_artifixer3d_checkpoint_render_only_review_execution"
            if prepared_bundle.get("pipeline_mode")
            == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
            else "one_shot_artifixer3d_object_free_exact_support_candidate_execution"
        )
        or value.get("bundle_sha256") != prepared_bundle.get("bundle_sha256")
        or value.get("bundle_receipt_digest") != prepared_bundle.get("receipt_digest")
        or value.get("manifest_digest") != prepared_bundle.get("manifest_digest")
        or value.get("runtime_request_digest")
        != prepared_bundle.get("runtime_request_digest")
        or value.get("checkpoint_reuse_digest")
        != prepared_bundle.get("checkpoint_reuse_digest")
        or value.get("blueprint_commit")
        != prepared_bundle.get("blueprint_source_identity", {}).get("commit")
        or value.get("blueprint_tree")
        != prepared_bundle.get("blueprint_source_identity", {}).get("tree")
        or value.get("container_image") != prepared_bundle.get("container_image")
        or value.get("hard_attempt_spend_cap_usd") != hard_cap_usd
        or value.get("maximum_hourly_rate_usd") != max_hourly_rate_usd
        or value.get("maximum_single_resource_ttl_seconds") != hard_ttl_seconds
        or value.get("external_active_instance_allowlist")
        != sorted(set(int(item) for item in allowed_active_instance_ids))
        or set(value.get("external_active_instance_allowlist", []))
        & set(value.get("forbidden_external_instance_ids", []))
        or value.get("forbidden_external_instance_ids")
        != prepared_bundle.get("forbidden_external_instance_ids")
        or value.get("private_derived_upload_only") is not True
        or value.get("raw_interiorgs_upload_authorized") is not False
        or value.get("raw_dataset_bytes_upload_authorized") is not False
        or value.get("provider_training_authorized") is not False
        or value.get("commercial_use_authorized") is not False
        or value.get("exact_mask_only_edits_required") is not True
        or value.get("source_object_restoration_authorized") is not False
        or value.get("generated_output_is_physical_evidence") is not False
    ):
        raise ValueError("artifixer3d_paid_attempt_authority_invalid")
    bundle_receipt_path = _bound(
        value.get("bundle_receipt"), code="artifixer3d_authority_bundle_unbound"
    )
    if bundle_receipt_path != Path(str(prepared_bundle["receipt_path"])).resolve():
        raise ValueError("artifixer3d_authority_bundle_mismatch")
    prior_path = _bound(
        value.get("prior_aura_authority"),
        code="artifixer3d_authority_prior_chain_unbound",
    )
    prior = _validate_prior_authority_chain(prior_path)
    if prior.get("authorization_digest") != value.get("prior_aura_authority_digest"):
        raise ValueError("artifixer3d_authority_prior_chain_mismatch")
    terminal_path = _bound(
        value.get("prior_terminal_result"),
        code="artifixer3d_authority_terminal_unbound",
    )
    _, terminal_cost = _validate_prior_terminal_result(
        terminal_path, prior_authority=prior
    )
    predecessor_cost = 0.0
    predecessor = value.get("prior_artifixer_attempt")
    if predecessor is not None:
        if not isinstance(predecessor, Mapping):
            raise ValueError("artifixer3d_authority_predecessor_invalid")
        predecessor_authority_path = _bound(
            predecessor.get("authority"),
            code="artifixer3d_authority_predecessor_unbound",
        )
        predecessor_result_path = _bound(
            predecessor.get("terminal_result"),
            code="artifixer3d_authority_predecessor_result_unbound",
        )
        predecessor_cleanup_path = _bound(
            predecessor.get("object_store_cleanup"),
            code="artifixer3d_authority_predecessor_cleanup_unbound",
        )
        predecessor_zero_path = _bound(
            predecessor.get("provider_zero"),
            code="artifixer3d_authority_predecessor_zero_unbound",
        )
        (
            predecessor_authority,
            predecessor_attempt_cost,
            predecessor_cost,
        ) = _validate_prior_artifixer_attempt(
            authority_path=predecessor_authority_path,
            result_path=predecessor_result_path,
            cleanup_path=predecessor_cleanup_path,
            provider_zero_path=predecessor_zero_path,
        )
        if (
            predecessor.get("authority_digest")
            != predecessor_authority.get("authorization_digest")
            or predecessor.get("terminal_cost_usd") != predecessor_attempt_cost
            or predecessor.get("lineage_cost_usd") != predecessor_cost
        ):
            raise ValueError("artifixer3d_authority_predecessor_mismatch")
    supplemental_cost = 0.0
    supplemental = value.get("supplemental_prior_spend_reconciliation")
    if supplemental is not None:
        if not isinstance(supplemental, Mapping):
            raise ValueError("artifixer3d_authority_supplemental_spend_invalid")
        supplemental_path = _bound(
            supplemental,
            code="artifixer3d_authority_supplemental_spend_unbound",
        )
        supplemental_receipt, supplemental_cost = (
            _validate_supplemental_spend_reconciliation(supplemental_path)
        )
        if (
            supplemental.get("receipt_digest")
            != supplemental_receipt.get("receipt_digest")
            or supplemental.get("total_cost_usd") != supplemental_cost
        ):
            raise ValueError("artifixer3d_authority_supplemental_spend_mismatch")
    prior_spend = round(
        float(prior["prior_goal_spend_usd"])
        + terminal_cost
        + predecessor_cost
        + supplemental_cost,
        6,
    )
    if (
        terminal_cost != value.get("prior_terminal_cost_usd")
        or prior_spend != value.get("aggregate_goal_spend_before_attempt_usd")
        or value.get("aggregate_goal_spend_cap_usd")
        != prepared_bundle.get("aggregate_goal_spend_cap_usd")
        or prior_spend + hard_cap_usd > float(value["aggregate_goal_spend_cap_usd"])
        or not 0 < max_hourly_rate_usd <= hard_cap_usd <= MAX_HARD_CAP_USD
        or not MIN_TTL_SECONDS <= hard_ttl_seconds <= MAX_TTL_SECONDS
        or hard_ttl_seconds * max_hourly_rate_usd / 3600 > hard_cap_usd
    ):
        raise ValueError("artifixer3d_paid_attempt_authority_budget_invalid")
    return value


def consume_artifixer3d_paid_attempt_authority_once(
    authority: Mapping[str, Any], *, blueprint_commit: str
) -> dict[str, Any]:
    digest = str(authority.get("authorization_digest") or "")
    if digest != canonical_digest(dict(authority), digest_field="authorization_digest"):
        return {"status": "blocked", "blockers": ["artifixer3d_authority_identity_invalid"]}
    identity = digest.removeprefix("sha256:")
    if blueprint_commit != authority.get("blueprint_commit") or len(identity) != 64:
        return {"status": "blocked", "blockers": ["artifixer3d_authority_identity_invalid"]}
    try:
        AUTHORIZATION_CONSUMPTION_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_stat = AUTHORIZATION_CONSUMPTION_ROOT.stat()
        if (
            AUTHORIZATION_CONSUMPTION_ROOT.is_symlink()
            or root_stat.st_uid != os.getuid()
            or root_stat.st_mode & 0o077
        ):
            raise PermissionError
        destination = AUTHORIZATION_CONSUMPTION_ROOT / f"artifixer3d-{identity}.json"
        record = {
            "schema_version": "artifixer3d_paid_attempt_consumption.v1",
            "authorization_digest": digest,
            "bundle_sha256": authority.get("bundle_sha256"),
            "blueprint_commit": blueprint_commit,
            "maximum_provider_allocations": 1,
            "consumed_at": utc_now_iso(),
        }
        raw = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        temporary = AUTHORIZATION_CONSUMPTION_ROOT / f".{identity}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    except FileExistsError:
        return {"status": "blocked", "blockers": ["artifixer3d_paid_attempt_authority_consumed"]}
    except (OSError, PermissionError):
        return {"status": "blocked", "blockers": ["artifixer3d_authority_consumption_write_failed"]}
    return {
        "status": "consumed",
        "authorization_digest": digest,
        "consumption_record_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "record_location_disclosed": False,
    }


def materialize_artifixer3d_postblocked_provider_zero(
    *,
    attempt_authority_path: str | Path,
    result_path: str | Path,
    adapter_result_path: str | Path,
    cleanup_path: str | Path,
    watchdog_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal one terminal attempt's mutation count and API-confirmed zero state.

    The historical function and receipt-schema names are retained for compatibility,
    but a successfully completed attempt needs the same immutable provider-zero
    closeout before it can serve as a spend-chain predecessor.
    """

    authority_file = Path(attempt_authority_path).expanduser().resolve()
    result_file = Path(result_path).expanduser().resolve()
    adapter_file = Path(adapter_result_path).expanduser().resolve()
    cleanup_file = Path(cleanup_path).expanduser().resolve()
    watchdog_file = Path(watchdog_path).expanduser().resolve()
    authority = _read(authority_file)
    result = _read(result_file)
    adapter = _read(adapter_file)
    cleanup = _read(cleanup_file)
    watchdog = _read(watchdog_file)
    lane_inventory = watchdog.get("final_inventory")
    global_inventory = watchdog.get("final_global_inventory")
    if not isinstance(lane_inventory, Mapping):
        lane_inventory = {}
    if not isinstance(global_inventory, Mapping):
        global_inventory = {}
    recorded_teardown = watchdog.get("recorded_vast_instance_teardown")
    if not isinstance(recorded_teardown, Mapping):
        recorded_teardown = {}
    authority_digest = authority.get("authorization_digest")
    terminal_status = result.get("status")
    blockers: list[str] = []
    if (
        authority.get("schema_version") != PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION
        or authority_digest != canonical_digest(authority, digest_field="authorization_digest")
    ):
        blockers.append("artifixer3d_provider_zero_authority_invalid")
    if (
        result.get("schema_version") != RESULT_SCHEMA_VERSION
        or terminal_status not in {"blocked", "completed"}
        or result.get("authorization_consumption", {}).get("status") != "consumed"
        or result.get("authorization_consumption", {}).get("authorization_digest")
        != authority_digest
        or result.get("continuing_spend_from_this_run") is not False
    ):
        blockers.append("artifixer3d_provider_zero_result_invalid")
    provider_mutations = 1 if adapter.get("provider_create_attempted") is True else 0
    if (
        adapter.get("continuing_spend_from_this_run") is not False
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or watchdog.get("provider_absence_scope")
        != "recorded_instance_and_lane_prefix"
        or lane_inventory.get("api_confirmed") is not True
        or lane_inventory.get("live_resource_count") != 0
        or recorded_teardown.get("provider_absence_confirmed") is not True
    ):
        blockers.append("artifixer3d_provider_zero_closeout_invalid")
    if blockers:
        raise ValueError(";".join(sorted(set(blockers))))
    receipt: dict[str, Any] = {
        "schema_version": "artifixer3d_postblocked_provider_zero.v1",
        "generated_at": utc_now_iso(),
        "attempt_authority_digest": authority_digest,
        "attempt_terminal_status": terminal_status,
        "provider_mutations_performed_by_attempt": provider_mutations,
        "provider_zero_confirmed": True,
        "provider_zero_scope": "recorded_instance_and_lane_prefix",
        "inventory": dict(lane_inventory),
        "recorded_instance_teardown": dict(recorded_teardown),
        "provider_account_global_zero_confirmed": (
            global_inventory.get("api_confirmed") is True
            and global_inventory.get("live_resource_count") == 0
        ),
        "global_inventory": dict(global_inventory),
        "attempt_authority": _record(authority_file),
        "attempt_result": _record(result_file),
        "provider_adapter": _record(adapter_file),
        "object_store_cleanup": _record(cleanup_file),
        "watchdog_receipt": _record(watchdog_file),
        "continuing_spend_from_attempt": False,
        "all_staged_objects_absent": True,
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    write_json(output, receipt)
    return receipt


@contextmanager
def _authority_environment():
    previous = {name: os.environ.get(name) for name in (*_MUTATION_ENV, _RETRY_ENV)}
    for name in _MUTATION_ENV:
        os.environ[name] = "true"
    os.environ[_RETRY_ENV] = "0"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _extract_output(path: Path | None, destination: Path) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if path is None or not path.is_file() or path.is_symlink():
        return {}, ["artifixer3d_provider_output_missing"]
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(path) as archive:
            total = 0
            for info in archive.infolist():
                member = Path(info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if (
                    member.is_absolute()
                    or ".." in member.parts
                    or mode == 0o120000
                    or info.file_size > 2_000_000_000
                ):
                    raise ValueError("artifixer3d_provider_output_member_invalid")
                total += info.file_size
                if total > 4_000_000_000:
                    raise ValueError("artifixer3d_provider_output_too_large")
            archive.extractall(destination)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return {}, [
            f"artifixer3d_provider_output_extract_failed:{redacted_failure_detail(exc)}"
        ]
    result_path = destination / "public_scene_artifixer3d_runtime_result.json"
    if not result_path.is_file():
        blockers.append("artifixer3d_runtime_result_missing")
        return {}, blockers
    return _read(result_path, code="artifixer3d_runtime_result_unreadable"), blockers


def _local_runtime_path(root: Path, provider_path: Any, *, code: str) -> Path:
    value = str(provider_path or "").replace("\\", "/")
    marker = "/runtime_output/"
    if marker not in value:
        raise ValueError(code)
    relative = Path(value.split(marker, 1)[1])
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or path.is_symlink() or not path.is_file():
        raise ValueError(code)
    return path


def _validate_native_usdz_archive(
    path: Path, archive_contract: Mapping[str, Any]
) -> None:
    expected = archive_contract.get("members")
    if not isinstance(expected, list):
        raise ValueError("artifixer3d_runtime_native_appearance_invalid")
    observed: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle, zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if [info.filename for info in infos] != [
                "default.usda",
                "repaired_scene.nurec",
                "gauss.usda",
            ]:
                raise ValueError
            for info in infos:
                handle.seek(info.header_offset)
                header = handle.read(30)
                if len(header) != 30:
                    raise ValueError
                fields = struct.unpack("<IHHHHHIIIHH", header)
                data_offset = info.header_offset + 30 + fields[-2] + fields[-1]
                body = archive.read(info)
                if (
                    info.compress_type != zipfile.ZIP_STORED
                    or data_offset % 64
                    or len(body) != info.file_size
                    or (
                        info.filename.endswith(".nurec")
                        and (body[:3] != b"\x1f\x8b\x08" or body[4:8] != b"\0" * 4)
                    )
                ):
                    raise ValueError
                observed.append(
                    {
                        "filename": info.filename,
                        "size_bytes": info.file_size,
                        "data_offset_bytes": data_offset,
                        "sha256": "sha256:" + hashlib.sha256(body).hexdigest(),
                    }
                )
    except (OSError, ValueError, zipfile.BadZipFile, struct.error) as exc:
        raise ValueError("artifixer3d_runtime_native_appearance_invalid") from exc
    if observed != expected:
        raise ValueError("artifixer3d_runtime_native_appearance_invalid")


def _materialize_raw_result(
    *,
    execution: Mapping[str, Any],
    execution_root: Path,
    bundle: Mapping[str, Any],
    closeout: Mapping[str, Any],
) -> dict[str, Any]:
    render_only_mode = (
        bundle.get("pipeline_mode") == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
    )
    dual_target_mode = bundle.get("pipeline_mode") in {
        DUAL_TARGET_PIPELINE_MODE,
        DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
    }
    native_appearance_required = "native_appearance_export" in (
        bundle.get("phases") or []
    )
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in execution.get("tasks") or []:
        task_id = str(task.get("task_id") or "")
        if task_id not in bundle["task_ids"] or task_id in seen:
            raise ValueError("artifixer3d_runtime_task_invalid")
        seen.add(task_id)
        frames: list[dict[str, Any]] = []
        frame_field = (
            "artifixer3d_review_frames"
            if dual_target_mode
            else "final_candidate_frames"
        )
        frame_rows = task.get(frame_field) or []
        for row in frame_rows:
            path = _local_runtime_path(
                execution_root,
                row.get("path"),
                code="artifixer3d_runtime_frame_unbound",
            )
            if (
                path.stat().st_size != row.get("size_bytes")
                or _sha256(path) != row.get("sha256")
                or (
                    not dual_target_mode
                    and row.get("outside_support_changed_pixels") != 0
                )
            ):
                raise ValueError("artifixer3d_runtime_frame_invalid")
            frame = {
                "frame_index": row.get("frame_index"),
                "camera_id": row.get("camera_id"),
                **_record(path),
            }
            if dual_target_mode:
                frame.update(
                    {
                        "outside_support_invariance_status": (
                            "deferred_until_final_soft_composite"
                        ),
                        "outside_support_invariance_proven": False,
                    }
                )
            else:
                frame.update(
                    {
                        "repair_pixel_count": row.get("repair_pixel_count"),
                        "outside_support_changed_pixels": 0,
                    }
                )
            frames.append(frame)
        checkpoint_record = task.get("artifixer3d_checkpoint")
        native_appearance_record = task.get("native_appearance")
        native_appearance: dict[str, Any] | None = None
        if dual_target_mode and native_appearance_required:
            if not isinstance(native_appearance_record, Mapping):
                raise ValueError("artifixer3d_runtime_native_appearance_missing")
            coordinate = native_appearance_record.get("coordinate_contract")
            source_checkpoint = native_appearance_record.get("source_checkpoint")
            if (
                native_appearance_record.get("status")
                != "native_appearance_candidates_exported_pending_native_import_and_multiview_review"
                or not isinstance(coordinate, Mapping)
                or coordinate.get("source_gaussian_tensor_coordinates_preserved")
                is not True
                or coordinate.get("camera_derived_normalizing_transform_applied")
                is not False
                or coordinate.get("standard_gaussian_ply_transform_matrix")
                != [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
                or coordinate.get("isaac_nurec_usdz_wrapper_transform_matrix")
                != [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
                or coordinate.get("usdz_wrapper_transform_role")
                != "fixed_pinned_3dgrut_to_usd_axis_convention_only"
                or not isinstance(checkpoint_record, Mapping)
                or not isinstance(source_checkpoint, Mapping)
                or source_checkpoint.get("size_bytes")
                != checkpoint_record.get("size_bytes")
                or source_checkpoint.get("sha256") != checkpoint_record.get("sha256")
                or native_appearance_record.get("generated_output_is_capture_or_physical_evidence")
                is not False
                or native_appearance_record.get("native_import_qualified") is not False
            ):
                raise ValueError("artifixer3d_runtime_native_appearance_invalid")
            exports: dict[str, Any] = {}
            for field in ("standard_gaussian_ply", "isaac_nurec_usdz"):
                record = native_appearance_record.get(field)
                if not isinstance(record, Mapping):
                    raise ValueError("artifixer3d_runtime_native_appearance_invalid")
                export = _local_runtime_path(
                    execution_root,
                    record.get("path"),
                    code="artifixer3d_runtime_native_appearance_unbound",
                )
                if (
                    export.stat().st_size != record.get("size_bytes")
                    or _sha256(export) != record.get("sha256")
                ):
                    raise ValueError("artifixer3d_runtime_native_appearance_invalid")
                exports[field] = _record(export)
            archive_contract = native_appearance_record.get(
                "isaac_nurec_usdz_archive_contract"
            )
            if (
                not isinstance(archive_contract, Mapping)
                or archive_contract.get("compression") != "stored"
                or archive_contract.get("payload_alignment_bytes") != 64
                or archive_contract.get("all_payload_offsets_aligned") is not True
                or archive_contract.get("nurec_gzip_mtime_normalized_to_zero")
                is not True
                or not isinstance(archive_contract.get("members"), list)
                or len(archive_contract["members"]) != 3
            ):
                raise ValueError("artifixer3d_runtime_native_appearance_invalid")
            if (
                native_appearance_record.get("schema_version")
                != NATIVE_APPEARANCE_EXPORT_SCHEMA
                or native_appearance_record.get("export_digest")
                != canonical_digest(
                    native_appearance_record, digest_field="export_digest"
                )
            ):
                raise ValueError("artifixer3d_runtime_native_appearance_invalid")
            _validate_native_usdz_archive(
                Path(exports["isaac_nurec_usdz"]["path"]), archive_contract
            )
            native_appearance = {
                "schema_version": native_appearance_record["schema_version"],
                "status": native_appearance_record["status"],
                "source_checkpoint": {
                    "size_bytes": source_checkpoint["size_bytes"],
                    "sha256": source_checkpoint["sha256"],
                },
                "gaussian_count": native_appearance_record.get("gaussian_count"),
                "coordinate_contract": dict(coordinate),
                **exports,
                "isaac_nurec_usdz_archive_contract": dict(archive_contract),
                "usdz_tensor_precision": native_appearance_record.get(
                    "usdz_tensor_precision"
                ),
                "generated_output_is_capture_or_physical_evidence": False,
                "native_import_qualified": False,
                "source_export_digest": native_appearance_record["export_digest"],
            }
        checkpoint: Path | None = None
        reused_checkpoint_record: dict[str, Any] | None = None
        if render_only_mode:
            expected_reuse = bundle.get("reused_checkpoints", {}).get(task_id)
            if (
                not isinstance(checkpoint_record, Mapping)
                or not isinstance(expected_reuse, Mapping)
                or checkpoint_record.get("size_bytes")
                != expected_reuse.get("size_bytes")
                or checkpoint_record.get("sha256") != expected_reuse.get("sha256")
            ):
                raise ValueError("artifixer3d_runtime_checkpoint_reuse_mismatch")
            reused_checkpoint_record = {
                "size_bytes": expected_reuse["size_bytes"],
                "sha256": expected_reuse["sha256"],
                "checkpoint_reused": True,
                "checkpoint_reuse_digest": bundle.get("checkpoint_reuse_digest"),
                "source_provider_zip_member": expected_reuse.get(
                    "source_provider_zip_member"
                ),
            }
        elif checkpoint_record is not None:
            checkpoint = _local_runtime_path(
                execution_root,
                checkpoint_record.get("path"),
                code="artifixer3d_runtime_checkpoint_unbound",
            )
        if (
            (
                checkpoint is not None
                and (
                    checkpoint.stat().st_size != checkpoint_record.get("size_bytes")
                    or _sha256(checkpoint) != checkpoint_record.get("sha256")
                )
            )
            or (
                dual_target_mode
                and not render_only_mode
                and checkpoint is None
            )
            or (
                not dual_target_mode
                and (checkpoint is None)
                != (bundle.get("semantic_editor_only") is True)
            )
            or len(frames) != bundle["task_camera_counts"][task_id]
            or (
                dual_target_mode
                and (
                    task.get("pipeline_mode") != bundle.get("pipeline_mode")
                    or task.get("training_record_count")
                    != bundle["task_training_record_counts"][task_id]
                    or any(
                        isinstance(row.get("frame_index"), bool)
                        or not isinstance(row.get("frame_index"), int)
                        or not str(row.get("camera_id") or "")
                        for row in frames
                    )
                    or sorted(row.get("frame_index") for row in frames)
                    != list(range(bundle["task_camera_counts"][task_id]))
                    or len({row.get("camera_id") for row in frames})
                    != bundle["task_camera_counts"][task_id]
                )
            )
            or (
                render_only_mode
                and (
                    task.get("checkpoint_reused") is not True
                    or task.get("checkpoint_reuse_digest")
                    != bundle.get("checkpoint_reuse_digest")
                    or task.get("training_executed") is not False
                    or task.get("direct_artifixer_executed") is not False
                    or task.get("artifixer3d_plus_executed") is not False
                )
            )
            or (
                dual_target_mode
                and (
                    task.get("outside_support_invariance_status")
                    != "deferred_until_final_soft_composite"
                    or task.get("outside_support_changed_pixels_total") is not None
                )
            )
            or (
                not dual_target_mode
                and task.get("outside_support_changed_pixels_total") != 0
            )
        ):
            raise ValueError("artifixer3d_runtime_task_outputs_invalid")
        task_result = {
            "task_id": task_id,
            frame_field: frames,
            "artifixer3d_checkpoint": (
                reused_checkpoint_record
                if render_only_mode
                else (_record(checkpoint) if checkpoint is not None else None)
            ),
            "native_appearance": native_appearance,
            "semantic_object_free_review_passed": False,
            "multiview_consistency_review_passed": False,
        }
        if dual_target_mode:
            task_result.update(
                {
                    "pipeline_mode": bundle.get("pipeline_mode"),
                    "physical_camera_count": bundle["task_camera_counts"][task_id],
                    "training_record_count": bundle["task_training_record_counts"][task_id],
                    "outside_support_invariance_status": (
                        "deferred_until_final_soft_composite"
                    ),
                    "outside_support_invariance_proven": False,
                    "outside_support_changed_pixels_total": None,
                }
            )
        else:
            task_result["outside_support_changed_pixels_total"] = 0
        tasks.append(task_result)
    if seen != set(bundle["task_ids"]):
        raise ValueError("artifixer3d_runtime_task_coverage_invalid")
    raw: dict[str, Any] = {
        "schema_version": RAW_RESULT_SCHEMA_VERSION,
        "status": (
            "raw_artifixer3d_review_frames_ready_for_external_visual_and_multiview_review"
            if dual_target_mode
            else "candidate_frames_ready_for_external_visual_and_multiview_review"
        ),
        "pipeline_mode": bundle.get("pipeline_mode"),
        "bundle_sha256": bundle["bundle_sha256"],
        "manifest_digest": bundle["manifest_digest"],
        "runtime_request_digest": bundle["runtime_request_digest"],
        "replacement_object_count": bundle["replacement_object_count"],
        "tasks": tasks,
        "source_object_restoration_permitted": False,
        "outside_exact_support_changed_pixels_total": (
            None if dual_target_mode else 0
        ),
        "outside_support_invariance_status": (
            "deferred_until_final_soft_composite" if dual_target_mode else "proven"
        ),
        "outside_support_invariance_proven": not dual_target_mode,
        "appearance_repair_qualified": False,
        "simready_or_policy_gate_unlocked": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "provider_closeout": dict(closeout),
        "result_digest": "",
    }
    raw["result_digest"] = canonical_digest(raw, digest_field="result_digest")
    return raw


def _recovered_consumption(authority: Mapping[str, Any]) -> dict[str, Any]:
    """Re-open the one-time consumption record without consuming authority again."""

    digest = str(authority.get("authorization_digest") or "")
    identity = digest.removeprefix("sha256:")
    if len(identity) != 64:
        raise ValueError("artifixer3d_recovery_authority_identity_invalid")
    path = AUTHORIZATION_CONSUMPTION_ROOT / f"artifixer3d-{identity}.json"
    record = _read(path, code="artifixer3d_recovery_consumption_unreadable")
    if (
        record.get("schema_version")
        != "artifixer3d_paid_attempt_consumption.v1"
        or record.get("authorization_digest") != digest
        or record.get("bundle_sha256") != authority.get("bundle_sha256")
        or record.get("blueprint_commit") != authority.get("blueprint_commit")
        or record.get("maximum_provider_allocations") != 1
    ):
        raise ValueError("artifixer3d_recovery_consumption_invalid")
    return {
        "status": "consumed",
        "authorization_digest": digest,
        "consumption_record_sha256": _sha256(path),
        "record_location_disclosed": False,
    }


def _validate_recovered_provider_output(
    *, output_zip: Path, execution_root: Path, execution: Mapping[str, Any]
) -> None:
    """Prove every locally recovered scientific output came from the provider ZIP."""

    runtime_path = execution_root / "public_scene_artifixer3d_runtime_result.json"
    if not output_zip.is_file() or output_zip.is_symlink() or not runtime_path.is_file():
        raise ValueError("artifixer3d_recovery_provider_output_missing")
    records: list[Mapping[str, Any]] = []
    for task in execution.get("tasks") or []:
        if not isinstance(task, Mapping):
            raise ValueError("artifixer3d_recovery_runtime_task_invalid")
        for field in ("artifixer3d_review_frames", "final_candidate_frames"):
            for row in task.get(field) or []:
                if isinstance(row, Mapping):
                    records.append(row)
        checkpoint = task.get("artifixer3d_checkpoint")
        if isinstance(checkpoint, Mapping):
            records.append(checkpoint)
    try:
        with zipfile.ZipFile(output_zip) as archive:
            if archive.read("public_scene_artifixer3d_runtime_result.json") != (
                runtime_path.read_bytes()
            ):
                raise ValueError("artifixer3d_recovery_runtime_zip_mismatch")
            checked: set[str] = set()
            for record in records:
                provider_path = str(record.get("path") or "").replace("\\", "/")
                marker = "/runtime_output/"
                if marker not in provider_path:
                    raise ValueError("artifixer3d_recovery_output_path_invalid")
                member = provider_path.split(marker, 1)[1]
                if member in checked:
                    continue
                checked.add(member)
                info = archive.getinfo(member)
                digest = hashlib.sha256()
                with archive.open(info) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if (
                    info.is_dir()
                    or info.file_size != record.get("size_bytes")
                    or "sha256:" + digest.hexdigest() != record.get("sha256")
                ):
                    raise ValueError("artifixer3d_recovery_output_digest_mismatch")
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise ValueError("artifixer3d_recovery_provider_output_invalid") from exc


def recover_artifixer3d_local_closeout(
    *,
    job_dir: str | Path,
    bundle_receipt_path: str | Path,
    attempt_authority_path: str | Path,
) -> dict[str, Any]:
    """Recover deterministic local receipts after provider teardown and extraction.

    This never calls a provider, consumes authority, trains, or renders.  It exists
    for the narrow case where the immutable provider output and teardown evidence
    were retained but a local disk/write failure interrupted final receipt sealing.
    """

    job = Path(job_dir).expanduser().resolve()
    result_path = job / "public_scene_artifixer3d_vast_result.json"
    raw_path = job / "public_scene_artifixer3d_raw_result.json"
    if result_path.exists():
        raise ValueError("artifixer3d_recovery_result_exists")
    bundle = validate_artifixer3d_bundle(bundle_receipt_path)
    authority = _read(
        Path(attempt_authority_path).expanduser().resolve(),
        code="artifixer3d_recovery_authority_unreadable",
    )
    authority = validate_artifixer3d_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        max_hourly_rate_usd=float(authority.get("maximum_hourly_rate_usd") or 0),
        hard_cap_usd=float(authority.get("hard_attempt_spend_cap_usd") or 0),
        hard_ttl_seconds=int(authority.get("maximum_single_resource_ttl_seconds") or 0),
        allowed_active_instance_ids=bundle["allowed_active_instance_ids"],
    )
    consumption = _recovered_consumption(authority)
    provider_run = job / "vast_provider_run"
    staging_dir = job / "object_store_staging"
    execution_root = job / "immutable_execution"
    adapter_path = provider_run / "vast_provider_adapter_result.json"
    teardown_path = provider_run / "vast_teardown_manifest.json"
    final_path = provider_run / "vast_final_validation.json"
    watchdog_path = job / "independent_vast_watchdog" / WATCHDOG_EVIDENCE_NAME
    cleanup_path = staging_dir / "wam_provider_object_store_cleanup.json"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    adapter = _read(adapter_path, code="artifixer3d_recovery_adapter_unreadable")
    teardown = _read(teardown_path, code="artifixer3d_recovery_teardown_unreadable")
    cleanup = _read(cleanup_path, code="artifixer3d_recovery_cleanup_unreadable")
    watchdog = _read(watchdog_path, code="artifixer3d_recovery_watchdog_unreadable")
    execution = _read(
        execution_root / "public_scene_artifixer3d_runtime_result.json",
        code="artifixer3d_recovery_runtime_unreadable",
    )
    _validate_recovered_provider_output(
        output_zip=output_zip, execution_root=execution_root, execution=execution
    )
    lane_inventory = watchdog.get("final_inventory")
    if not isinstance(lane_inventory, Mapping):
        lane_inventory = {}
    if (
        adapter.get("status") != "completed"
        or adapter.get("continuing_spend_from_this_run") is not False
        or teardown.get("continuing_spend_from_this_run") is not False
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("provider_absence_confirmed") is not True
        or lane_inventory.get("api_confirmed") is not True
        or lane_inventory.get("live_resource_count") != 0
    ):
        raise ValueError("artifixer3d_recovery_provider_closeout_invalid")
    render_only_mode = (
        bundle.get("pipeline_mode") == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
    )
    dual_target_mode = bundle.get("pipeline_mode") in {
        DUAL_TARGET_PIPELINE_MODE,
        DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
    }
    if (
        execution.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION
        or execution.get("status")
        != (
            "raw_artifixer3d_candidate_completed_requires_visual_and_multiview_review"
            if dual_target_mode
            else "candidate_completed_requires_visual_and_multiview_review"
        )
        or execution.get("pipeline_mode") != bundle.get("pipeline_mode")
        or execution.get("model_loaded") is not True
        or execution.get("provider_zero_required_after_return") is not True
        or execution.get("source_object_restoration_permitted") is not False
        or execution.get("artifixer_direct_inference_executed") is not False
        or execution.get("semantic_editor_inference_executed") is not False
        or execution.get("artifixer3d_distillation_executed") is not (
            not render_only_mode
        )
        or execution.get("artifixer3d_plus_inference_executed") is not False
    ):
        raise ValueError("artifixer3d_recovery_runtime_not_completed")
    closeout = {
        "adapter_result": _record(adapter_path),
        "teardown_manifest": _record(teardown_path),
        "final_validation": _record(final_path),
        "watchdog_receipt": _record(watchdog_path),
        "object_store_cleanup": _record(cleanup_path),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "provider_mutations_performed": 1,
        "provider_zero_confirmed": True,
        "provider_zero_scope": "recorded_instance_and_lane_prefix",
        "all_staged_objects_absent": True,
    }
    raw = _materialize_raw_result(
        execution=execution,
        execution_root=execution_root,
        bundle=bundle,
        closeout=closeout,
    )
    if raw_path.exists():
        if _read(raw_path) != raw:
            raise ValueError("artifixer3d_recovery_raw_result_conflict")
    else:
        write_json(raw_path, raw)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed",
        "bundle_sha256": bundle["bundle_sha256"],
        "manifest_digest": bundle["manifest_digest"],
        "runtime_request_digest": bundle["runtime_request_digest"],
        "execution_result_path": str(
            execution_root / "public_scene_artifixer3d_runtime_result.json"
        ),
        "raw_result_path": str(raw_path),
        "adapter_result_path": str(adapter_path),
        "teardown_manifest_path": str(teardown_path),
        "final_validation_path": str(final_path),
        "watchdog_receipt_path": str(watchdog_path),
        "object_store_cleanup_path": str(cleanup_path),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "provider_mutations_performed": 1,
        "provider_closeout": closeout,
        "hard_cap_usd": authority["hard_attempt_spend_cap_usd"],
        "hard_ttl_seconds": authority["maximum_single_resource_ttl_seconds"],
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True,
        "authorization_consumption": consumption,
        "independent_watchdog": watchdog,
        "local_receipt_recovered_after_provider_teardown": True,
        "appearance_repair_qualified": False,
        "simready_or_policy_gate_unlocked": False,
        "blockers": [],
        "raw_secret_values_recorded": False,
    }
    result = seal_lane_terminal_artifacts(
        result,
        attempt_root=job,
        lane="public_scene_artifixer3d",
        extra_artifact_roots={"raw_result": raw_path},
    )
    write_json(result_path, result)
    return result


def run_artifixer3d_vast(
    *,
    job_dir: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    machine_avoidlist_path: str | Path | None = None,
    paid_attempt_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute exactly once through the canonical paid provider adapter."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle = dict(prepared_bundle)
    if (
        not 0 < max_hourly_rate_usd <= hard_cap_usd <= MAX_HARD_CAP_USD
        or not MIN_TTL_SECONDS <= hard_ttl_seconds <= MAX_TTL_SECONDS
        or hard_ttl_seconds * max_hourly_rate_usd / 3600 > hard_cap_usd
    ):
        raise ValueError("artifixer3d_budget_invalid")
    result_path = job / "public_scene_artifixer3d_vast_result.json"
    image_preflight = inspect_artifixer3d_container_image(
        image_ref=str(bundle.get("container_image") or ""),
        output_path=job / "artifixer3d_container_registry_preflight.json",
    )
    if image_preflight.get("status") != "completed":
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "prepared_bundle": bundle,
            "container_registry_preflight": image_preflight,
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "authority_consumed": False,
            "blockers": list(image_preflight.get("blockers") or []),
        }
        write_json(result_path, result)
        return result
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "dry_run_ready",
            "prepared_bundle": bundle,
            "container_registry_preflight": image_preflight,
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
        }
        write_json(result_path, result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("artifixer3d_paid_resource_admission_grant_missing")
    require_paid_resource_admission_grant(
        paid_resource_admission_grant,
        resource_class="vast_provider_adapter",
        require_allocation_binding=True,
    )
    if paid_attempt_authority is None:
        raise ValueError("artifixer3d_paid_attempt_authority_missing")
    authority = validate_artifixer3d_paid_attempt_authority(
        paid_attempt_authority,
        prepared_bundle=bundle,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=bundle["allowed_active_instance_ids"],
    )
    consumption = consume_artifixer3d_paid_attempt_authority_once(
        authority, blueprint_commit=str(authority["blueprint_commit"])
    )
    if consumption.get("status") != "consumed":
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "authorization_consumption": consumption,
            "blockers": list(consumption.get("blockers") or []),
        }
        write_json(result_path, result)
        return result
    bundle_path = Path(str(bundle["bundle_path"])).resolve()
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=bundle_path,
        key_prefix=DEFAULT_KEY_PREFIX,
        expiration_seconds=hard_ttl_seconds + 1800,
    )
    if staging.get("status") != "completed":
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "authorization_consumption": consumption,
            "blockers": staging.get("blockers")
            or ["artifixer3d_object_store_staging_blocked"],
        }
        write_json(result_path, result)
        return result
    allowed = tuple(int(value) for value in bundle["allowed_active_instance_ids"])
    handoff, handle = arm_independent_vast_watchdog(
        job_dir=job,
        max_live_minutes=hard_ttl_seconds // 60,
        generated_at=utc_now_iso(),
        allowed_active_instance_ids=allowed,
        pod_name_prefix=INSTANCE_LABEL_PREFIX,
    )
    if handle is None:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "authorization_consumption": consumption,
            "independent_watchdog": handoff,
            "blockers": ["artifixer3d_independent_watchdog_not_armed"],
        }
        write_json(result_path, result)
        return result
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
                max_live_minutes=hard_ttl_seconds // 60,
                session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=bundle["container_image"],
                isaac_image=bundle["container_image"],
                ngc_image_login_mode="never",
                provider_bundle=bundle_path,
                provider_bundle_url=(staging_dir / "provider_bundle_url.txt").read_text().strip(),
                provider_output_put_url=(staging_dir / "provider_output_put_url.txt").read_text().strip(),
                provider_output_get_url=(staging_dir / "provider_output_get_url.txt").read_text().strip(),
                provider_runtime_output_zip=output_zip,
                enable_isaac_smoke=False,
                enable_blueprint_bundle=True,
                provider_bundle_kind=PROVIDER_BUNDLE_KIND,
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=False,
                disk_gb=192,
                min_gpu_ram_mb=MIN_GPU_RAM_MB,
                min_compute_cap=MIN_COMPUTE_CAP,
                poll_interval_seconds=15,
                startup_timeout_seconds=hard_ttl_seconds,
                heartbeat_no_progress_seconds=3600,
                session_budget_ledger_path=job / "artifixer3d_vast_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=False,
                preferred_gpu_keywords=("A100",),
                prefer_isaac_rt=False,
                gpu_selection_policy=GPU_SELECTION_POLICY,
                machine_avoidlist_path=machine_avoidlist_path,
                allowed_active_instance_ids=allowed,
                vast_launch_lock_file=job.parent / "artifixer3d_paid_launch.lock",
                instance_label_prefix=INSTANCE_LABEL_PREFIX,
                started_instance_id_path=handle.started_instance_id_path,
                forward_hf_token=True,
                paid_resource_admission_grant=paid_resource_admission_grant,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {
            "status": "blocked",
            "blockers": [
                f"artifixer3d_adapter_failed:{redacted_failure_detail(exc)}"
            ],
            "raw_secret_values_recorded": False,
        }
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    teardown_path = provider_run / "vast_teardown_manifest.json"
    teardown = _read(teardown_path) if teardown_path.is_file() else {}
    instance_ids = [
        value
        for value in teardown.get("vast_instance_ids") or []
        if isinstance(value, int) and value > 0
    ]
    watchdog = close_independent_vast_watchdog(
        job_dir=job,
        handle=handle,
        instance_ids=instance_ids,
        provider_teardown_completed=teardown.get("continuing_spend_from_this_run") is False,
        provider_allocation_impossible=(
            not instance_ids and adapter.get("provider_create_attempted") is not True
        ),
    )
    execution_root = job / "immutable_execution"
    execution, blockers = _extract_output(output_zip, execution_root)
    adapter_path = provider_run / "vast_provider_adapter_result.json"
    final_path = provider_run / "vast_final_validation.json"
    watchdog_path = job / "independent_vast_watchdog" / WATCHDOG_EVIDENCE_NAME
    cleanup_path = staging_dir / "wam_provider_object_store_cleanup.json"
    closeout = {
        "adapter_result": _record(adapter_path) if adapter_path.is_file() else None,
        "teardown_manifest": _record(teardown_path) if teardown_path.is_file() else None,
        "final_validation": _record(final_path) if final_path.is_file() else None,
        "watchdog_receipt": _record(watchdog_path) if watchdog_path.is_file() else None,
        "object_store_cleanup": _record(cleanup_path) if cleanup_path.is_file() else None,
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "provider_mutations_performed": (
            1 if adapter.get("provider_create_attempted") is True else 0
        ),
        "provider_zero_confirmed": watchdog.get("status") == "provider_terminal",
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
    }
    if adapter.get("status") != "completed":
        blockers.append("artifixer3d_provider_adapter_not_completed")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("artifixer3d_object_store_zero_not_proven")
    if watchdog.get("status") != "provider_terminal":
        blockers.append("artifixer3d_watchdog_not_terminal")
    render_only_mode = (
        bundle.get("pipeline_mode") == DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
    )
    dual_target_mode = bundle.get("pipeline_mode") in {
        DUAL_TARGET_PIPELINE_MODE,
        DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
    }
    common_runtime_valid = (
        execution.get("schema_version") == RUNTIME_RESULT_SCHEMA_VERSION
        and execution.get("status")
        == (
            "raw_artifixer3d_candidate_completed_requires_visual_and_multiview_review"
            if dual_target_mode
            else "candidate_completed_requires_visual_and_multiview_review"
        )
        and execution.get("model_loaded") is True
        and execution.get("provider_zero_required_after_return") is True
        and execution.get("source_object_restoration_permitted") is False
    )
    dual_target_runtime_valid = (
        dual_target_mode
        and execution.get("pipeline_mode") == bundle.get("pipeline_mode")
        and execution.get("artifixer_direct_inference_executed") is False
        and execution.get("semantic_editor_inference_executed") is False
        and execution.get("artifixer3d_distillation_executed")
        is (not render_only_mode)
        and execution.get("artifixer3d_checkpoint_reused", False)
        is render_only_mode
        and execution.get("artifixer3d_plus_inference_executed") is False
        and execution.get("outside_exact_support_changed_pixels_permitted")
        == "unconstrained_for_raw_representation_review"
        and execution.get("outside_support_invariance_gate")
        == "deferred_until_final_soft_composite"
        and (
            not render_only_mode
            or execution.get("checkpoint_reuse_digest")
            == bundle.get("checkpoint_reuse_digest")
        )
    )
    legacy_runtime_valid = (
        not dual_target_mode
        and execution.get("artifixer_direct_inference_executed")
        == (bundle["direct_editor_backend"] == "artifixer")
        and execution.get("semantic_editor_inference_executed")
        == (bundle["direct_editor_backend"] != "artifixer")
        and execution.get("artifixer3d_distillation_executed")
        == (not bundle["semantic_editor_only"])
        and execution.get("artifixer3d_plus_inference_executed")
        == (not bundle["semantic_editor_only"])
        and execution.get("outside_exact_support_changed_pixels_permitted") == 0
    )
    if not common_runtime_valid or not (
        dual_target_runtime_valid or legacy_runtime_valid
    ):
        blockers.append("artifixer3d_runtime_not_completed")
    raw_path: Path | None = None
    if not blockers:
        try:
            raw = _materialize_raw_result(
                execution=execution,
                execution_root=execution_root,
                bundle=bundle,
                closeout=closeout,
            )
            raw_path = job / "public_scene_artifixer3d_raw_result.json"
            write_json(raw_path, raw)
        except (OSError, ValueError, KeyError) as exc:
            blockers.append(
                "artifixer3d_raw_result_materialization_failed:"
                f"{redacted_failure_detail(exc)}"
            )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "bundle_sha256": bundle["bundle_sha256"],
        "manifest_digest": bundle["manifest_digest"],
        "runtime_request_digest": bundle["runtime_request_digest"],
        "execution_result_path": str(
            execution_root / "public_scene_artifixer3d_runtime_result.json"
        ),
        "raw_result_path": str(raw_path) if raw_path else None,
        "adapter_result_path": str(adapter_path),
        "teardown_manifest_path": str(teardown_path),
        "final_validation_path": str(final_path),
        "watchdog_receipt_path": str(watchdog_path),
        "object_store_cleanup_path": str(cleanup_path),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "provider_mutations_performed": closeout["provider_mutations_performed"],
        "provider_closeout": closeout,
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "authorization_consumption": consumption,
        "independent_watchdog": watchdog,
        "appearance_repair_qualified": False,
        "simready_or_policy_gate_unlocked": False,
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
    }
    result = seal_lane_terminal_artifacts(
        result,
        attempt_root=job,
        lane="public_scene_artifixer3d",
    )
    write_json(result_path, result)
    return result


__all__ = [
    "MAX_HARD_CAP_USD",
    "MAX_TTL_SECONDS",
    "PAID_ATTEMPT_AUTHORITY_SCHEMA_VERSION",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "consume_artifixer3d_paid_attempt_authority_once",
    "inspect_artifixer3d_container_image",
    "materialize_artifixer3d_paid_attempt_authority",
    "materialize_artifixer3d_postblocked_provider_zero",
    "recover_artifixer3d_local_closeout",
    "run_artifixer3d_vast",
    "validate_artifixer3d_bundle",
    "validate_artifixer3d_paid_attempt_authority",
]
