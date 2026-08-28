"""Immutable authorities and state for scene-configuration warm diagnostics."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .common import utc_now_iso
from .decision_evidence_contracts import canonical_digest, canonical_json
from .retained_gpu_session_lifecycle import record_retained_gpu_state
from .task_evaluation_scene_configuration_bundle import (
    load_scene_configuration_provider_bundle_receipt,
)
from .task_evaluation_scene_configuration_diagnostic_checkpoint import (
    validate_scene_configuration_diagnostic_checkpoint,
)
from .task_evaluation_scene_configuration_artifixer_warm_checkpoint import (
    validate_artifixer_post_training_checkpoint,
)
from .task_evaluation_scene_configuration_diagnostic_mode import (
    CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE,
    FRESH_DIAGNOSTIC_BOOTSTRAP_MODE,
    validate_diagnostic_bootstrap_mode,
)
from .task_evaluation_scene_configuration_paid_authority import (
    validate_scene_configuration_paid_authority,
)
from .task_evaluation_scene_configuration_runtime_budget import (
    BOOTSTRAP_TRANSFER_AND_NO_SPEND_RESERVE_SECONDS,
    diagnostic_required_parent_ttl_seconds,
)
from .task_evaluation_scene_configuration_warm_overlay import (
    SceneConfigurationWarmDiagnosticError,
    _absolute_directory,
    _read,
    _record,
    _record_path,
    _validated_release_receipt,
    _write_exclusive,
    validate_scene_configuration_warm_source_overlay,
)
from .task_evaluation_scene_configuration_warm_allocation import (
    validate_warm_claim_boundary as _validate_claim_boundary,
)
from .task_evaluation_scene_configuration_vast import (
    _provider_transfer_byte_budget,
)


SESSION_AUTHORITY_SCHEMA_VERSION = "task_evaluation_scene_configuration_warm_session_authority.v1"
SESSION_SCHEMA_VERSION = "task_evaluation_scene_configuration_warm_session.v1"
ITERATION_AUTHORITY_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_warm_iteration_authority.v1"
)
ITERATION_RESULT_SCHEMA_VERSION = "task_evaluation_scene_configuration_warm_iteration_result.v1"
CLOSEOUT_SCHEMA_VERSION = "task_evaluation_scene_configuration_warm_session_closeout.v1"
SESSION_STATE_SCHEMA_VERSION = "task_evaluation_scene_configuration_warm_session_state.v1"
SESSION_JOURNAL_NAME = "task_evaluation_scene_configuration_warm_session.jsonl"
SESSION_STATE_NAME = "task_evaluation_scene_configuration_warm_session_state.v1.json"
SESSION_OWNER_LOCK_NAME = "task_evaluation_scene_configuration_warm_session.lock"
REMOTE_ROOT = "/workspace/task_evaluation_scene_configuration_warm"
BASE_RUNTIME_ROOT = (
    "/workspace/task_evaluation_scene_configuration_provider_bundle/provider_runtime"
)
BASE_OUTPUT_ROOT = "/workspace/task_evaluation_scene_configuration_provider_bundle/runtime_output"
MAX_WARM_ITERATIONS = 64
MIN_REMAINING_WATCHDOG_SECONDS = 180
WARM_CARRIED_PAID_MODEL_STAGES = (
    "artifixer_semantic_teacher",
    "artifixer_visual_review",
    "content_agents",
)
ARTIFIXER_POST_TRAINING_CONTINUATION_KIND = (
    "artifixer_post_training_visual_review_and_remaining_chain"
)
ARTIFIXER_POST_TRAINING_RERUN_PAID_MODEL_STAGES = (
    "artifixer_visual_review",
    "content_agents",
)
ARTIFIXER_WARM_SECRET_FILE_ENV_NAMES = (
    "OPENAI_ADMIN_API_KEY_FILE",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
    "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
    "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
    "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE",
)
ARTIFIXER_WARM_PUBLIC_ENV_NAMES = (
    "OPENAI_PROJECT_ID",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID",
    "OPENAI_CONTENT_AGENTS_API_KEY_ID",
    "BLUEPRINT_SCENE_CONFIGURATION_AUTHORITY_DIGEST",
)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ITERATION_ID = re.compile(r"i[0-9]{3}-[0-9a-f]{12}\Z")
_HOST = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?\Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _artifixer_warm_environment_authority(
    *, stage_caps: Mapping[str, Any], maximum_requests: int
) -> dict[str, Any]:
    public = {
        name: str(os.environ.get(name) or "").strip() for name in ARTIFIXER_WARM_PUBLIC_ENV_NAMES
    }
    file_paths = {
        name: Path(str(os.environ.get(name) or "")).expanduser()
        for name in ARTIFIXER_WARM_SECRET_FILE_ENV_NAMES
    }
    files_valid = True
    attestation_digests: dict[str, str] = {}
    for name, path in file_paths.items():
        try:
            valid = (
                path.is_absolute()
                and not path.is_symlink()
                and path.is_file()
                and path.stat().st_size > 0
                and path.stat().st_size <= 65_536
                and path.stat().st_mode & 0o077 == 0
            )
        except OSError:
            valid = False
        files_valid = files_valid and valid
        if valid and "COST_SCOPE_ATTESTATION" in name:
            attestation_digests[name] = _sha256_file(path)
    authorized = bool(
        files_valid
        and all(public.values())
        and _DIGEST.fullmatch(public["BLUEPRINT_SCENE_CONFIGURATION_AUTHORITY_DIGEST"]) is not None
        and maximum_requests > 0
        and all(
            float(stage_caps.get(stage) or 0) > 0
            for stage in ARTIFIXER_POST_TRAINING_RERUN_PAID_MODEL_STAGES
        )
        and len(attestation_digests) == 2
    )
    return {
        "authorized": authorized,
        "public_environment": public if authorized else {},
        "cost_scope_attestation_digests": (attestation_digests if authorized else {}),
        "required_secret_file_environment_names": list(ARTIFIXER_WARM_SECRET_FILE_ENV_NAMES),
    }


def materialize_scene_configuration_warm_session_authority(
    *,
    bundle_receipt_path: str | Path,
    paid_attempt_authority_path: str | Path,
    diagnostic_release_receipt_path: str | Path,
    checkpoint_root: str | Path | None,
    maximum_warm_iterations: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Derive one aggregate warm lease from an admitted one-allocation authority."""

    bundle_path = Path(bundle_receipt_path).expanduser().resolve()
    bundle = load_scene_configuration_provider_bundle_receipt(bundle_path, diagnostic_only=True)
    paid_path = Path(paid_attempt_authority_path).expanduser().resolve()
    paid = validate_scene_configuration_paid_authority(
        _read(paid_path, code="scene_configuration_warm_paid_authority_invalid"),
        bundle_receipt=bundle,
    )
    release_path = Path(diagnostic_release_receipt_path).expanduser().resolve()
    release = _validated_release_receipt(release_path, source_commit=str(bundle["source_commit"]))
    try:
        diagnostic_bootstrap_mode = validate_diagnostic_bootstrap_mode(
            bundle.get("diagnostic_bootstrap_mode")
        )
    except ValueError as exc:
        raise SceneConfigurationWarmDiagnosticError(str(exc)) from exc
    fresh_diagnostic_bootstrap = diagnostic_bootstrap_mode == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
    checkpoint: dict[str, Any] | None = None
    if fresh_diagnostic_bootstrap:
        if checkpoint_root is not None:
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_session_checkpoint_invalid"
            )
    else:
        if checkpoint_root is None:
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_session_checkpoint_invalid"
            )
        checkpoint_path = _absolute_directory(
            checkpoint_root,
            code="scene_configuration_warm_session_checkpoint_invalid",
        )
        checkpoint = validate_scene_configuration_diagnostic_checkpoint(
            checkpoint_root=checkpoint_path
        )
    _warm_download_ceiling, warm_output_ceiling = _provider_transfer_byte_budget(bundle)
    if (
        isinstance(maximum_warm_iterations, bool)
        or not 1 <= maximum_warm_iterations <= MAX_WARM_ITERATIONS
        or paid.get("maximum_paid_attempts") != 1
        or paid.get("maximum_provider_allocations") != 1
        or bundle.get("diagnostic_only") is not True
        or (
            checkpoint is not None
            and bundle.get("source_diagnostic_checkpoint_digest")
            != checkpoint.get("checkpoint_digest")
        )
        or (
            checkpoint is not None
            and bundle.get("diagnostic_scientific_binding_digest")
            != (checkpoint.get("scientific_bindings") or {}).get("binding_digest")
        )
        or (
            fresh_diagnostic_bootstrap
            and (
                bundle.get("source_diagnostic_checkpoint_digest") is not None
                or bundle.get("carried_completed_stage_count") != 0
                or _DIGEST.fullmatch(str(bundle.get("diagnostic_scientific_binding_digest") or ""))
                is None
            )
        )
        or release.get("source_commit") != bundle.get("source_commit")
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_session_authority_invalid"
        )
    authority: dict[str, Any] = {
        "schema_version": SESSION_AUTHORITY_SCHEMA_VERSION,
        "status": "authorized",
        "program_id": "arm-decision-proof-v1",
        "day_gate": "day-28",
        "purpose": "bounded_scene_configuration_diagnostic_warm_session",
        "provider": "vast",
        "provider_bundle_kind": "task_evaluation_scene_configuration",
        "source_commit": bundle["source_commit"],
        "remote_ref": release["remote_ref"],
        "diagnostic_release_receipt": _record(release_path),
        "bundle_receipt": _record(bundle_path),
        "bundle_sha256": bundle["bundle_sha256"],
        "run_id": bundle["run_id"],
        "toolchain_digest": bundle["toolchain_digest"],
        "construction_envelope_digest": bundle["portable_construction_envelope_digest"],
        "source_checkpoint_digest": (
            checkpoint["checkpoint_digest"] if checkpoint is not None else None
        ),
        "diagnostic_bootstrap_mode": diagnostic_bootstrap_mode,
        "bootstrap_carried_completed_stage_prefix_count": checkpoint["completed_stage_prefix_count"]
        if checkpoint is not None
        else 0,
        "bootstrap_carried_completed_stage_ids": [
            str(row["stage_id"]) for row in checkpoint["completed_stage_results"]
        ]
        if checkpoint is not None
        else [],
        "bootstrap_uses_one_shot_paid_authority": True,
        "warm_iterations_require_all_paid_model_stages_carried": True,
        "required_carried_paid_model_stages_for_retention": list(WARM_CARRIED_PAID_MODEL_STAGES),
        "scientific_binding_digest": (
            checkpoint["scientific_bindings"]["binding_digest"]
            if checkpoint is not None
            else bundle["diagnostic_scientific_binding_digest"]
        ),
        "diagnostic_stage_sequence_ids": list(bundle["diagnostic_stage_sequence_ids"]),
        "paid_attempt_authority": _record(paid_path),
        "paid_attempt_authority_digest": paid["authority_digest"],
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "maximum_warm_iterations": maximum_warm_iterations,
        "maximum_single_resource_ttl_seconds": paid["maximum_single_resource_ttl_seconds"],
        "maximum_hourly_rate_usd": paid["maximum_hourly_rate_usd"],
        "aggregate_provider_compute_spend_cap_usd": paid["provider_compute_spend_cap_usd"],
        "aggregate_hard_spend_cap_usd": paid["hard_attempt_spend_cap_usd"],
        "maximum_warm_output_archive_bytes": warm_output_ceiling,
        "diagnostic_only": True,
        "development_only": True,
        "qualification_eligible": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "arbitrary_command_permitted": False,
        "raw_secret_values_recorded": False,
        "authority_digest": "",
    }
    external = (paid.get("external_service_spend_caps") or {}).get("openai") or {}
    stage_caps = external.get("stage_max_cost_usd") or {}
    warm_environment = _artifixer_warm_environment_authority(
        stage_caps=stage_caps,
        maximum_requests=int(external.get("maximum_requests") or 0),
    )
    authority["artifixer_post_training_continuation"] = {
        "authorized": bool(
            fresh_diagnostic_bootstrap
            and float(stage_caps.get("artifixer_visual_review") or 0) > 0
            and float(stage_caps.get("content_agents") or 0) > 0
            and warm_environment["authorized"] is True
        ),
        "continuation_kind": ARTIFIXER_POST_TRAINING_CONTINUATION_KIND,
        "maximum_remote_continuations": 1,
        "maximum_provider_allocations": 0,
        "visual_review_provider_call_must_be_proven_absent": True,
        "rerun_paid_model_stages": list(ARTIFIXER_POST_TRAINING_RERUN_PAID_MODEL_STAGES),
        "stage_max_cost_usd": {
            stage: float(stage_caps.get(stage) or 0)
            for stage in ARTIFIXER_POST_TRAINING_RERUN_PAID_MODEL_STAGES
        },
        "maximum_openai_requests": int(external.get("maximum_requests") or 0),
        "public_environment": warm_environment["public_environment"],
        "cost_scope_attestation_digests": warm_environment["cost_scope_attestation_digests"],
        "required_secret_file_environment_names": warm_environment[
            "required_secret_file_environment_names"
        ],
        "credentials_via_pinned_ssh_stdin_private_files_only": True,
        "secret_values_in_authority_or_object_store": False,
    }
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
    destination = Path(output_path).expanduser()
    if not destination.is_absolute() or destination.is_symlink():
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_session_authority_output_invalid"
        )
    _write_exclusive(destination, authority)
    return authority


def validate_scene_configuration_warm_session_authority(
    authority_path: str | Path,
) -> dict[str, Any]:
    unresolved = Path(authority_path).expanduser()
    if not unresolved.is_absolute() or unresolved.is_symlink():
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_session_authority_invalid"
        )
    path = unresolved.resolve()
    authority = _read(path, code="scene_configuration_warm_session_authority_invalid")
    _validate_claim_boundary(authority, code="scene_configuration_warm_session_authority_invalid")
    artifixer_continuation = authority.get("artifixer_post_training_continuation")
    if (
        authority.get("schema_version") != SESSION_AUTHORITY_SCHEMA_VERSION
        or authority.get("status") != "authorized"
        or authority.get("program_id") != "arm-decision-proof-v1"
        or authority.get("day_gate") != "day-28"
        or authority.get("provider") != "vast"
        or authority.get("provider_bundle_kind") != "task_evaluation_scene_configuration"
        or _COMMIT.fullmatch(str(authority.get("source_commit") or "")) is None
        or _DIGEST.fullmatch(str(authority.get("bundle_sha256") or "")) is None
        or not str(authority.get("run_id") or "")
        or _DIGEST.fullmatch(str(authority.get("toolchain_digest") or "")) is None
        or _DIGEST.fullmatch(str(authority.get("construction_envelope_digest") or "")) is None
        or not isinstance(authority.get("diagnostic_stage_sequence_ids"), list)
        or len(authority.get("diagnostic_stage_sequence_ids") or []) != 6
        or (
            authority.get("diagnostic_bootstrap_mode") == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
            and authority.get("source_checkpoint_digest") is not None
        )
        or (
            authority.get("diagnostic_bootstrap_mode") != FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
            and (
                authority.get("diagnostic_bootstrap_mode")
                != CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE
                or _DIGEST.fullmatch(str(authority.get("source_checkpoint_digest") or "")) is None
            )
        )
        or _DIGEST.fullmatch(str(authority.get("scientific_binding_digest") or "")) is None
        or not 0 <= int(authority.get("bootstrap_carried_completed_stage_prefix_count") or 0) <= 6
        or authority.get("bootstrap_uses_one_shot_paid_authority") is not True
        or authority.get("warm_iterations_require_all_paid_model_stages_carried") is not True
        or authority.get("required_carried_paid_model_stages_for_retention")
        != list(WARM_CARRIED_PAID_MODEL_STAGES)
        or not isinstance(artifixer_continuation, Mapping)
        or artifixer_continuation.get("continuation_kind")
        != ARTIFIXER_POST_TRAINING_CONTINUATION_KIND
        or artifixer_continuation.get("maximum_remote_continuations") != 1
        or artifixer_continuation.get("maximum_provider_allocations") != 0
        or artifixer_continuation.get("visual_review_provider_call_must_be_proven_absent")
        is not True
        or artifixer_continuation.get("rerun_paid_model_stages")
        != list(ARTIFIXER_POST_TRAINING_RERUN_PAID_MODEL_STAGES)
        or artifixer_continuation.get("required_secret_file_environment_names")
        != list(ARTIFIXER_WARM_SECRET_FILE_ENV_NAMES)
        or not isinstance(artifixer_continuation.get("authorized"), bool)
        or not isinstance(artifixer_continuation.get("public_environment"), Mapping)
        or not isinstance(artifixer_continuation.get("cost_scope_attestation_digests"), Mapping)
        or (
            artifixer_continuation.get("authorized") is True
            and (
                set(artifixer_continuation["public_environment"])
                != set(ARTIFIXER_WARM_PUBLIC_ENV_NAMES)
                or not all(artifixer_continuation["public_environment"].values())
                or set(artifixer_continuation["cost_scope_attestation_digests"])
                != {
                    name
                    for name in ARTIFIXER_WARM_SECRET_FILE_ENV_NAMES
                    if "COST_SCOPE_ATTESTATION" in name
                }
                or any(
                    _DIGEST.fullmatch(str(value or "")) is None
                    for value in artifixer_continuation["cost_scope_attestation_digests"].values()
                )
            )
        )
        or (
            artifixer_continuation.get("authorized") is False
            and (
                artifixer_continuation.get("public_environment") != {}
                or artifixer_continuation.get("cost_scope_attestation_digests") != {}
            )
        )
        or artifixer_continuation.get("credentials_via_pinned_ssh_stdin_private_files_only")
        is not True
        or artifixer_continuation.get("secret_values_in_authority_or_object_store") is not False
        or authority.get("maximum_provider_allocations") != 1
        or authority.get("maximum_automatic_retries") != 0
        or not 1 <= int(authority.get("maximum_warm_iterations") or 0) <= MAX_WARM_ITERATIONS
        or float(authority.get("maximum_hourly_rate_usd") or 0) <= 0
        or float(authority.get("aggregate_provider_compute_spend_cap_usd") or 0) <= 0
        or int(authority.get("maximum_single_resource_ttl_seconds") or 0) <= 0
        or int(authority.get("maximum_warm_output_archive_bytes") or 0) <= 0
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_session_authority_invalid"
        )
    try:
        bundle_record = authority["bundle_receipt"]
        paid_record = authority["paid_attempt_authority"]
        release_record = authority["diagnostic_release_receipt"]
        if not all(
            isinstance(row, Mapping) for row in (bundle_record, paid_record, release_record)
        ):
            raise ValueError("record invalid")
        bundle = load_scene_configuration_provider_bundle_receipt(
            _record_path(
                bundle_record,
                code="scene_configuration_warm_session_authority_invalid",
            ),
            diagnostic_only=True,
        )
        paid = validate_scene_configuration_paid_authority(
            _read(
                _record_path(
                    paid_record,
                    code="scene_configuration_warm_session_authority_invalid",
                ),
                code="scene_configuration_warm_session_authority_invalid",
            ),
            bundle_receipt=bundle,
        )
        release = _validated_release_receipt(
            _record_path(
                release_record,
                code="scene_configuration_warm_session_authority_invalid",
            ),
            source_commit=str(authority["source_commit"]),
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_session_authority_invalid"
        ) from exc
    if (
        bundle.get("bundle_sha256") != authority.get("bundle_sha256")
        or bundle.get("run_id") != authority.get("run_id")
        or bundle.get("toolchain_digest") != authority.get("toolchain_digest")
        or bundle.get("portable_construction_envelope_digest")
        != authority.get("construction_envelope_digest")
        or bundle.get("diagnostic_bootstrap_mode") != authority.get("diagnostic_bootstrap_mode")
        or bundle.get("diagnostic_scientific_binding_digest")
        != authority.get("scientific_binding_digest")
        or bundle.get("diagnostic_stage_sequence_ids")
        != authority.get("diagnostic_stage_sequence_ids")
        or (
            authority.get("diagnostic_bootstrap_mode") == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
            and authority.get("source_checkpoint_digest") is not None
        )
        or paid.get("authority_digest") != authority.get("paid_attempt_authority_digest")
        or release.get("remote_ref") != authority.get("remote_ref")
        or release.get("source_commit") != authority.get("source_commit")
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_session_authority_binding_mismatch"
        )
    return authority


def materialize_scene_configuration_warm_session(
    *,
    session_authority_path: str | Path,
    adapter_result_path: str | Path,
    watchdog_handoff_path: str | Path,
    output_root: str | Path,
    advanced_checkpoint: Mapping[str, Any] | None,
    bootstrap_allocation_binding_digest: str,
    artifixer_post_training_checkpoint: Mapping[str, Any] | None = None,
    observed_now_epoch: float | None = None,
) -> dict[str, Any]:
    """Seal a retained adapter result as the sole owned warm session."""

    authority_path = Path(session_authority_path).expanduser().resolve()
    authority = validate_scene_configuration_warm_session_authority(authority_path)
    adapter_path = Path(adapter_result_path).expanduser().resolve()
    adapter = _read(adapter_path, code="scene_configuration_warm_adapter_result_invalid")
    watchdog_path = Path(watchdog_handoff_path).expanduser().resolve()
    watchdog = _read(watchdog_path, code="scene_configuration_warm_watchdog_handoff_invalid")
    decision = adapter.get("retention_decision")
    instance_ids = adapter.get("vast_instance_ids")
    now = time.time() if observed_now_epoch is None else float(observed_now_epoch)
    general_checkpoint = isinstance(advanced_checkpoint, Mapping)
    artifixer_checkpoint = isinstance(artifixer_post_training_checkpoint, Mapping)
    if general_checkpoint == artifixer_checkpoint:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_checkpoint_kind_invalid"
        )
    try:
        if general_checkpoint:
            assert isinstance(advanced_checkpoint, Mapping)
            carried_local = validate_scene_configuration_diagnostic_checkpoint(
                checkpoint_root=str(advanced_checkpoint.get("checkpoint_root") or "")
            )
            remote_relative = PurePosixPath(
                str(advanced_checkpoint.get("provider_output_relative_root") or "")
            )
            remote_checkpoint_root = PurePosixPath(BASE_OUTPUT_ROOT).joinpath(
                *remote_relative.parts
            )
            remote_artifixer_checkpoint_root: str | None = None
            continuation_kind = "stage_three_plus_no_openai"
        else:
            assert isinstance(artifixer_post_training_checkpoint, Mapping)
            carried_local = validate_artifixer_post_training_checkpoint(
                checkpoint_root=str(artifixer_post_training_checkpoint.get("checkpoint_root") or "")
            )
            remote_relative = PurePosixPath("input/artifixer_post_training_checkpoint")
            remote_checkpoint_root = PurePosixPath(BASE_RUNTIME_ROOT, "input/diagnostic_checkpoint")
            remote_artifixer_checkpoint_root = PurePosixPath(
                BASE_RUNTIME_ROOT, "input/artifixer_post_training_checkpoint"
            ).as_posix()
            continuation_kind = ARTIFIXER_POST_TRAINING_CONTINUATION_KIND
    except (OSError, RuntimeError, ValueError) as exc:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_advanced_checkpoint_invalid"
        ) from exc
    started_id_path = Path(str(watchdog.get("started_instance_id_path") or ""))
    try:
        started_id = started_id_path.read_text(encoding="utf-8").strip()
    except OSError:
        started_id = ""
    if (
        adapter.get("schema_version") != "vast_provider_adapter_result.v1"
        or adapter.get("retained_owned") is not True
        or adapter.get("continuing_spend_from_this_run") is not True
        or not isinstance(instance_ids, list)
        or len(instance_ids) != 1
        or isinstance(instance_ids[0], bool)
        or not isinstance(instance_ids[0], int)
        or instance_ids[0] <= 0
        or not isinstance(decision, Mapping)
        or decision.get("status") != "retained_owned"
        or decision.get("retention_mode")
        != "task_evaluation_scene_configuration_diagnostic_warm_worker"
        or decision.get("instance_ids") != instance_ids
        or adapter.get("provider_bundle_kind") != "task_evaluation_scene_configuration"
        or adapter.get("provider_bundle_sha256") != authority.get("bundle_sha256")
        or not isinstance(decision.get("warm_worker_evidence"), Mapping)
        or decision["warm_worker_evidence"].get("provider_bundle_kind")
        != "task_evaluation_scene_configuration"
        or decision["warm_worker_evidence"].get("scene_configuration_runtime_root_ready")
        is not True
        or watchdog.get("schema_version") != "vast_independent_watchdog_handoff.v1"
        or watchdog.get("status") != "armed"
        or watchdog.get("independent_process") is not True
        or watchdog.get("watchdog_armed_before_allocation") is not True
        or watchdog.get("watchdog_deadline_epoch") != decision.get("watchdog_deadline_epoch")
        or watchdog.get("watchdog_pid") != decision.get("watchdog_pid")
        or watchdog.get("watchdog_out_dir") != decision.get("watchdog_out_dir")
        or watchdog.get("pod_name_prefix") != decision.get("watchdog_pod_name_prefix")
        or watchdog.get("started_instance_id_path")
        != decision.get("watchdog_started_instance_id_path")
        or started_id != str(instance_ids[0])
        or (
            general_checkpoint
            and (
                remote_relative.is_absolute()
                or not remote_relative.parts
                or ".." in remote_relative.parts
                or int((advanced_checkpoint or {}).get("completed_stage_prefix_count") or 0) < 3
                or carried_local.get("checkpoint_digest")
                != (advanced_checkpoint or {}).get("checkpoint_digest")
                or (carried_local.get("scientific_bindings") or {}).get("binding_digest")
                != authority.get("scientific_binding_digest")
            )
        )
        or (
            artifixer_checkpoint
            and (
                authority.get("artifixer_post_training_continuation", {}).get("authorized")
                is not True
                or carried_local.get("checkpoint_digest")
                != (artifixer_post_training_checkpoint or {}).get("checkpoint_digest")
                or carried_local.get("scientific_binding_digest")
                != authority.get("scientific_binding_digest")
                or carried_local.get("visual_review_provider_call_started") is not False
            )
        )
        or _DIGEST.fullmatch(str(carried_local.get("checkpoint_digest") or "")) is None
        or _DIGEST.fullmatch(bootstrap_allocation_binding_digest) is None
        or float(watchdog.get("watchdog_deadline_epoch") or 0) - now
        < MIN_REMAINING_WATCHDOG_SECONDS
    ):
        raise SceneConfigurationWarmDiagnosticError("scene_configuration_warm_session_invalid")
    warm = dict(decision["warm_worker_evidence"])
    root = Path(output_root).expanduser()
    if not root.is_absolute() or root.is_symlink() or root.exists():
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_session_output_invalid"
        )
    root.mkdir(parents=True, mode=0o750)
    session: dict[str, Any] = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "status": "ready",
        "generated_at": utc_now_iso(),
        "provider": "vast",
        "provider_instance_id": instance_ids[0],
        "ssh_host": warm["ssh_host"],
        "ssh_port": warm["ssh_port"],
        "container_image": adapter.get("selected_container_image"),
        "source_commit": authority["source_commit"],
        "remote_ref": authority["remote_ref"],
        "bundle_sha256": authority["bundle_sha256"],
        "run_id": authority["run_id"],
        "toolchain_digest": authority["toolchain_digest"],
        "construction_envelope_digest": authority["construction_envelope_digest"],
        "bootstrap_source_checkpoint_digest": authority["source_checkpoint_digest"],
        "source_checkpoint_digest": (
            carried_local["checkpoint_digest"]
            if general_checkpoint
            else carried_local["source_diagnostic_checkpoint_digest"]
        ),
        "continuation_kind": continuation_kind,
        "artifixer_post_training_checkpoint_digest": (
            carried_local["checkpoint_digest"] if artifixer_checkpoint else None
        ),
        "artifixer_post_training_binding_digest": (
            carried_local["binding_digest"] if artifixer_checkpoint else None
        ),
        "carried_completed_stage_prefix_count": (
            carried_local["completed_stage_prefix_count"] if general_checkpoint else 0
        ),
        "carried_completed_stage_ids": (
            [
                str(row.get("stage_id") or "")
                for row in carried_local.get("completed_stage_results") or []
            ]
            if general_checkpoint
            else []
        ),
        "diagnostic_stage_sequence_ids": list(authority["diagnostic_stage_sequence_ids"]),
        "carried_paid_model_stages": (
            list(authority["required_carried_paid_model_stages_for_retention"])
            if general_checkpoint
            else ["artifixer_semantic_teacher"]
        ),
        "rerun_paid_model_stages": (
            [] if general_checkpoint else list(ARTIFIXER_POST_TRAINING_RERUN_PAID_MODEL_STAGES)
        ),
        "warm_openai_external_service_spend_permitted": artifixer_checkpoint,
        "artifixer_post_training_continuation_authority": (
            dict(authority["artifixer_post_training_continuation"])
            if artifixer_checkpoint
            else None
        ),
        "scientific_binding_digest": authority["scientific_binding_digest"],
        "session_authority_digest": authority["authority_digest"],
        "bootstrap_allocation_binding_digest": (bootstrap_allocation_binding_digest),
        "maximum_warm_iterations": authority["maximum_warm_iterations"],
        "maximum_hourly_rate_usd": authority["maximum_hourly_rate_usd"],
        "aggregate_provider_compute_spend_cap_usd": authority[
            "aggregate_provider_compute_spend_cap_usd"
        ],
        "maximum_warm_output_archive_bytes": authority["maximum_warm_output_archive_bytes"],
        "watchdog_pid": watchdog.get("watchdog_pid"),
        "watchdog_deadline_epoch": watchdog["watchdog_deadline_epoch"],
        "watchdog_out_dir": watchdog["watchdog_out_dir"],
        "watchdog_pod_name_prefix": watchdog["pod_name_prefix"],
        "adapter_result": _record(adapter_path),
        "watchdog_handoff": _record(watchdog_path),
        "base_runtime_root": BASE_RUNTIME_ROOT,
        "base_output_root": BASE_OUTPUT_ROOT,
        "current_remote_checkpoint_root": remote_checkpoint_root.as_posix(),
        "current_remote_artifixer_post_training_checkpoint_root": (
            remote_artifixer_checkpoint_root
        ),
        "remote_warm_root": REMOTE_ROOT,
        "continuing_spend": True,
        "diagnostic_only": True,
        "development_only": True,
        "qualification_eligible": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "arbitrary_command_permitted": False,
        "raw_secret_values_recorded": False,
        "session_digest": "",
    }
    session["session_digest"] = canonical_digest(session, digest_field="session_digest")
    session_path = root / f"{SESSION_SCHEMA_VERSION}.json"
    _write_exclusive(session_path, session)
    initial_state: dict[str, Any] = {
        "schema_version": SESSION_STATE_SCHEMA_VERSION,
        "status": "ready",
        "session_digest": session["session_digest"],
        "attempted_iteration_count": 0,
        "completed_iteration_count": 0,
        "last_iteration_authority_digest": None,
        "last_iteration_result_digest": None,
        "current_checkpoint_digest": session["source_checkpoint_digest"],
        "current_artifixer_post_training_checkpoint_digest": session[
            "artifixer_post_training_checkpoint_digest"
        ],
        "current_remote_checkpoint_root": session["current_remote_checkpoint_root"],
        "current_remote_artifixer_post_training_checkpoint_root": session[
            "current_remote_artifixer_post_training_checkpoint_root"
        ],
        "current_completed_stage_prefix_count": session["carried_completed_stage_prefix_count"],
        "current_completed_stage_ids": list(session["carried_completed_stage_ids"]),
        "current_carried_paid_model_stages": list(session["carried_paid_model_stages"]),
        "consumed_openai_cost_scope_attestation_digests": [],
        "scientific_binding_digest": session["scientific_binding_digest"],
        "continuing_spend": True,
        "state_digest": "",
    }
    initial_state["state_digest"] = canonical_digest(initial_state, digest_field="state_digest")
    _write_exclusive(root / SESSION_STATE_NAME, initial_state)
    record_retained_gpu_state(
        root,
        "allocated",
        evidence={
            "provider": "vast",
            "provider_instance_id": instance_ids[0],
            "session_digest": session["session_digest"],
        },
    )
    record_retained_gpu_state(
        root,
        "container_starting",
        evidence={"provider_instance_id": instance_ids[0]},
    )
    record_retained_gpu_state(
        root,
        "healthy",
        evidence={"provider_instance_id": instance_ids[0]},
    )
    record_retained_gpu_state(
        root,
        "retained_owned",
        evidence={"provider_instance_id": instance_ids[0]},
    )
    session["session_path"] = str(session_path)
    session["session_root"] = str(root)
    return session


def validate_scene_configuration_warm_session(
    *,
    session_root: str | Path,
    observed_now_epoch: float | None = None,
    require_iteration_window: bool = True,
    allowed_state_statuses: frozenset[str] = frozenset({"ready", "iteration_failed"}),
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _absolute_directory(session_root, code="scene_configuration_warm_session_root_invalid")
    session = _read(
        root / f"{SESSION_SCHEMA_VERSION}.json",
        code="scene_configuration_warm_session_invalid",
    )
    state = _read(
        root / SESSION_STATE_NAME,
        code="scene_configuration_warm_session_state_invalid",
    )
    _validate_claim_boundary(session, code="scene_configuration_warm_session_invalid")
    now = time.time() if observed_now_epoch is None else float(observed_now_epoch)
    artifixer_session = (
        session.get("continuation_kind") == ARTIFIXER_POST_TRAINING_CONTINUATION_KIND
    )
    artifixer_continuation = bool(
        artifixer_session and state.get("current_completed_stage_prefix_count") == 0
    )
    general_continuation = bool(
        session.get("continuation_kind") == "stage_three_plus_no_openai"
        or (
            session.get("continuation_kind") is None
            and isinstance(state.get("current_completed_stage_prefix_count"), int)
            and int(state["current_completed_stage_prefix_count"]) >= 3
        )
        or (
            artifixer_session
            and isinstance(state.get("current_completed_stage_prefix_count"), int)
            and int(state["current_completed_stage_prefix_count"]) >= 3
        )
    )
    if (
        session.get("schema_version") != SESSION_SCHEMA_VERSION
        or session.get("status") != "ready"
        or session.get("provider") != "vast"
        or not isinstance(session.get("provider_instance_id"), int)
        or _HOST.fullmatch(str(session.get("ssh_host") or "")) is None
        or isinstance(session.get("ssh_port"), bool)
        or not isinstance(session.get("ssh_port"), int)
        or not 1 <= int(session.get("ssh_port") or 0) <= 65_535
        or session.get("continuing_spend") is not True
        or _DIGEST.fullmatch(str(session.get("construction_envelope_digest") or "")) is None
        or session.get("session_digest") != canonical_digest(session, digest_field="session_digest")
        or state.get("schema_version") != SESSION_STATE_SCHEMA_VERSION
        or state.get("session_digest") != session.get("session_digest")
        or state.get("state_digest") != canonical_digest(state, digest_field="state_digest")
        or state.get("scientific_binding_digest") != session.get("scientific_binding_digest")
        or state.get("continuing_spend") is not True
        or state.get("status") not in allowed_state_statuses
        or not 0
        <= int(state.get("completed_iteration_count") or 0)
        <= int(state.get("attempted_iteration_count") or 0)
        <= int(session.get("maximum_warm_iterations") or 0)
        or not isinstance(state.get("consumed_openai_cost_scope_attestation_digests"), list)
        or not (general_continuation or artifixer_continuation)
        or (
            general_continuation
            and (
                not 3 <= int(state.get("current_completed_stage_prefix_count") or 0) <= 6
                or state.get("current_carried_paid_model_stages")
                != list(WARM_CARRIED_PAID_MODEL_STAGES)
                or (
                    not artifixer_session
                    and (
                        (session.get("rerun_paid_model_stages") or []) != []
                        or session.get("warm_openai_external_service_spend_permitted", False)
                        is not False
                    )
                )
            )
        )
        or (
            artifixer_continuation
            and (
                state.get("current_completed_stage_prefix_count") != 0
                or state.get("current_completed_stage_ids") != []
                or state.get("current_carried_paid_model_stages") != ["artifixer_semantic_teacher"]
                or session.get("rerun_paid_model_stages")
                != list(ARTIFIXER_POST_TRAINING_RERUN_PAID_MODEL_STAGES)
                or session.get("warm_openai_external_service_spend_permitted") is not True
                or not isinstance(
                    session.get("artifixer_post_training_continuation_authority"),
                    Mapping,
                )
                or session["artifixer_post_training_continuation_authority"].get("authorized")
                is not True
                or _DIGEST.fullmatch(
                    str(state.get("current_artifixer_post_training_checkpoint_digest") or "")
                )
                is None
                or not str(
                    state.get("current_remote_artifixer_post_training_checkpoint_root") or ""
                ).startswith(BASE_RUNTIME_ROOT + "/input/")
            )
        )
        or not isinstance(state.get("current_completed_stage_ids"), list)
        or len(state.get("current_completed_stage_ids") or [])
        != int(state.get("current_completed_stage_prefix_count") or 0)
        or (
            require_iteration_window
            and float(session.get("watchdog_deadline_epoch") or 0) - now
            < MIN_REMAINING_WATCHDOG_SECONDS
        )
    ):
        raise SceneConfigurationWarmDiagnosticError("scene_configuration_warm_session_invalid")
    return session, state


@contextmanager
def scene_configuration_warm_session_owner_lock(
    session_root: str | Path,
) -> Iterator[None]:
    root = _absolute_directory(session_root, code="scene_configuration_warm_session_root_invalid")
    path = root / SESSION_OWNER_LOCK_NAME
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_session_owner_lock_busy"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _state_path(root: Path) -> Path:
    return root / SESSION_STATE_NAME


def _write_state(root: Path, state: Mapping[str, Any]) -> None:
    destination = _state_path(root)
    temporary = root / f".{SESSION_STATE_NAME}.{os.getpid()}.tmp"
    payload = (canonical_json(state) + "\n").encode("utf-8")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o440,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short warm session state write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o440)
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)


def _append_session_event(root: Path, event: Mapping[str, Any]) -> str:
    journal = root / SESSION_JOURNAL_NAME
    previous = "sha256:" + "0" * 64
    if journal.is_file():
        try:
            last = json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])
            previous = str(last["event_digest"])
        except (OSError, IndexError, KeyError, json.JSONDecodeError) as exc:
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_session_journal_invalid"
            ) from exc
    row = {
        **dict(event),
        "previous_event_digest": previous,
        "event_digest": "",
    }
    row["event_digest"] = canonical_digest(row, digest_field="event_digest")
    descriptor = os.open(
        journal,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        payload = (canonical_json(row) + "\n").encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short warm session journal write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return str(row["event_digest"])


def _remote_checkpoint_root(value: object) -> str:
    text = str(value or "")
    path = PurePosixPath(text)
    base = PurePosixPath(BASE_RUNTIME_ROOT) / "input/diagnostic_checkpoint"
    iterations = PurePosixPath(REMOTE_ROOT) / "iterations"
    if path == base:
        return text
    try:
        relative = path.relative_to(iterations)
    except ValueError as exc:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_iteration_remote_checkpoint_invalid"
        ) from exc
    if (
        path.is_absolute() is not True
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_iteration_remote_checkpoint_invalid"
        )
    return text


def _remote_artifixer_checkpoint_root(value: object) -> str:
    text = str(value or "")
    path = PurePosixPath(text)
    base = PurePosixPath(BASE_RUNTIME_ROOT) / "input/artifixer_post_training_checkpoint"
    iterations = PurePosixPath(REMOTE_ROOT) / "iterations"
    if path == base:
        return text
    try:
        relative = path.relative_to(iterations)
    except ValueError as exc:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_artifixer_warm_remote_checkpoint_invalid"
        ) from exc
    if (
        path.is_absolute() is not True
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_artifixer_warm_remote_checkpoint_invalid"
        )
    return text


def materialize_scene_configuration_warm_iteration_authority(
    *,
    session_root: str | Path,
    overlay_receipt_path: str | Path,
    output_path: str | Path,
    observed_now_epoch: float | None = None,
) -> dict[str, Any]:
    """Seal the next single-use overlay/checkpoint attachment authority."""

    now = time.time() if observed_now_epoch is None else float(observed_now_epoch)
    with scene_configuration_warm_session_owner_lock(session_root):
        root = _absolute_directory(
            session_root, code="scene_configuration_warm_session_root_invalid"
        )
        session, state = validate_scene_configuration_warm_session(
            session_root=root, observed_now_epoch=now
        )
        attempted = int(state["attempted_iteration_count"])
        index = attempted + 1
        if index > int(session["maximum_warm_iterations"]):
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_iteration_limit_exhausted"
            )
        required_remaining_seconds = (
            diagnostic_required_parent_ttl_seconds(
                int(state["current_completed_stage_prefix_count"])
            )
            - BOOTSTRAP_TRANSFER_AND_NO_SPEND_RESERVE_SECONDS
        )
        if float(session["watchdog_deadline_epoch"]) - now < required_remaining_seconds:
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_iteration_runtime_budget_insufficient"
            )
        overlay_path = Path(overlay_receipt_path).expanduser()
        if not overlay_path.is_absolute() or overlay_path.is_symlink():
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_iteration_overlay_invalid"
            )
        overlay = validate_scene_configuration_warm_source_overlay(
            overlay_path,
            expected_checkpoint_digest=str(state["current_checkpoint_digest"]),
        )
        if overlay.get("scientific_binding_digest") != state.get("scientific_binding_digest"):
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_iteration_scientific_binding_mismatch"
            )
        iteration_id = f"i{index:03d}-{str(overlay['source_commit'])[:12]}"
        if _ITERATION_ID.fullmatch(iteration_id) is None:
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_iteration_id_invalid"
            )
        output = Path(output_path).expanduser()
        if not output.is_absolute() or output.is_symlink():
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_iteration_authority_output_invalid"
            )
        remote_checkpoint_root = _remote_checkpoint_root(
            state.get("current_remote_checkpoint_root")
            or f"{BASE_RUNTIME_ROOT}/input/diagnostic_checkpoint"
        )
        artifixer_continuation = bool(
            session.get("continuation_kind") == ARTIFIXER_POST_TRAINING_CONTINUATION_KIND
            and state.get("current_completed_stage_prefix_count") == 0
        )
        artifixer_authority = (
            dict(session.get("artifixer_post_training_continuation_authority") or {})
            if artifixer_continuation
            else {}
        )
        if artifixer_continuation and attempted >= int(
            artifixer_authority.get("maximum_remote_continuations") or 0
        ):
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_artifixer_warm_continuation_limit_exhausted"
            )
        remote_artifixer_checkpoint_root = (
            _remote_artifixer_checkpoint_root(
                state.get("current_remote_artifixer_post_training_checkpoint_root")
            )
            if artifixer_continuation
            else None
        )
        authority: dict[str, Any] = {
            "schema_version": ITERATION_AUTHORITY_SCHEMA_VERSION,
            "status": "authorized",
            "generated_at": utc_now_iso(),
            "iteration_id": iteration_id,
            "iteration_index": index,
            "session_digest": session["session_digest"],
            "previous_state_digest": state["state_digest"],
            "provider": "vast",
            "provider_instance_id": session["provider_instance_id"],
            "maximum_provider_allocations": 0,
            "maximum_instance_lifecycle_mutations": 0,
            "maximum_remote_workload_dispatches": 1,
            "maximum_automatic_retries": 0,
            "source_commit": overlay["source_commit"],
            "remote_ref": overlay["remote_ref"],
            "source_overlay_receipt": _record(overlay_path.resolve()),
            "source_overlay_receipt_digest": overlay["receipt_digest"],
            "source_overlay_archive_sha256": overlay["overlay_archive"]["sha256"],
            "source_overlay_manifest_digest": overlay["manifest_digest"],
            "source_checkpoint_digest": overlay["source_checkpoint_digest"],
            "scientific_binding_digest": overlay["scientific_binding_digest"],
            "remote_checkpoint_root": remote_checkpoint_root,
            "continuation_kind": (
                ARTIFIXER_POST_TRAINING_CONTINUATION_KIND
                if artifixer_continuation
                else "stage_three_plus_no_openai"
            ),
            "source_artifixer_post_training_checkpoint_digest": state.get(
                "current_artifixer_post_training_checkpoint_digest"
            ),
            "remote_artifixer_post_training_checkpoint_root": (remote_artifixer_checkpoint_root),
            "carried_completed_stage_prefix_count": state["current_completed_stage_prefix_count"],
            "carried_completed_stage_ids": list(state["current_completed_stage_ids"]),
            "carried_paid_model_stages": list(state["current_carried_paid_model_stages"]),
            "rerun_paid_model_stages": (
                list(ARTIFIXER_POST_TRAINING_RERUN_PAID_MODEL_STAGES)
                if artifixer_continuation
                else []
            ),
            "fresh_openai_cost_scope_attestation_digests": (
                dict(artifixer_authority.get("cost_scope_attestation_digests") or {})
                if artifixer_continuation
                else {}
            ),
            "warm_openai_external_service_spend_permitted": artifixer_continuation,
            "openai_public_environment": (
                dict(artifixer_authority.get("public_environment") or {})
                if artifixer_continuation
                else {}
            ),
            "openai_stage_max_cost_usd": (
                dict(artifixer_authority.get("stage_max_cost_usd") or {})
                if artifixer_continuation
                else {}
            ),
            "openai_maximum_requests": (
                int(artifixer_authority.get("maximum_openai_requests") or 0)
                if artifixer_continuation
                else 0
            ),
            "watchdog_deadline_epoch": session["watchdog_deadline_epoch"],
            "aggregate_provider_compute_spend_cap_usd": session[
                "aggregate_provider_compute_spend_cap_usd"
            ],
            "maximum_output_archive_bytes": session["maximum_warm_output_archive_bytes"],
            "required_remaining_runtime_seconds": required_remaining_seconds,
            "diagnostic_only": True,
            "development_only": True,
            "qualification_eligible": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "arbitrary_command_permitted": False,
            "raw_secret_values_recorded": False,
            "authority_digest": "",
        }
        authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
        _write_exclusive(output, authority)
        return authority


def validate_scene_configuration_warm_iteration_authority(
    *,
    session_root: str | Path,
    authority_path: str | Path,
    observed_now_epoch: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    now = time.time() if observed_now_epoch is None else float(observed_now_epoch)
    session, state = validate_scene_configuration_warm_session(
        session_root=session_root, observed_now_epoch=now
    )
    path = Path(authority_path).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_iteration_authority_invalid"
        )
    authority = _read(path.resolve(), code="scene_configuration_warm_iteration_authority_invalid")
    _validate_claim_boundary(authority, code="scene_configuration_warm_iteration_authority_invalid")
    overlay_record = authority.get("source_overlay_receipt")
    if not isinstance(overlay_record, Mapping):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_iteration_authority_invalid"
        )
    overlay_path = _record_path(
        overlay_record,
        code="scene_configuration_warm_iteration_overlay_invalid",
    )
    overlay = validate_scene_configuration_warm_source_overlay(
        overlay_path,
        expected_source_commit=str(authority.get("source_commit") or ""),
        expected_checkpoint_digest=str(state["current_checkpoint_digest"]),
    )
    expected_index = int(state["attempted_iteration_count"]) + 1
    artifixer_continuation = bool(
        session.get("continuation_kind") == ARTIFIXER_POST_TRAINING_CONTINUATION_KIND
        and state.get("current_completed_stage_prefix_count") == 0
    )
    artifixer_session_authority = dict(
        session.get("artifixer_post_training_continuation_authority") or {}
    )
    if (
        authority.get("schema_version") != ITERATION_AUTHORITY_SCHEMA_VERSION
        or authority.get("status") != "authorized"
        or authority.get("session_digest") != session.get("session_digest")
        or authority.get("previous_state_digest") != state.get("state_digest")
        or authority.get("provider") != "vast"
        or authority.get("provider_instance_id") != session.get("provider_instance_id")
        or authority.get("maximum_provider_allocations") != 0
        or authority.get("maximum_instance_lifecycle_mutations") != 0
        or authority.get("maximum_remote_workload_dispatches") != 1
        or authority.get("maximum_automatic_retries") != 0
        or authority.get("iteration_index") != expected_index
        or authority.get("iteration_index") > session.get("maximum_warm_iterations")
        or _ITERATION_ID.fullmatch(str(authority.get("iteration_id") or "")) is None
        or authority.get("source_overlay_receipt_digest") != overlay.get("receipt_digest")
        or authority.get("source_overlay_archive_sha256") != overlay["overlay_archive"]["sha256"]
        or authority.get("source_overlay_manifest_digest") != overlay.get("manifest_digest")
        or authority.get("source_checkpoint_digest") != state.get("current_checkpoint_digest")
        or authority.get("scientific_binding_digest") != session.get("scientific_binding_digest")
        or authority.get("carried_completed_stage_prefix_count")
        != state.get("current_completed_stage_prefix_count")
        or authority.get("carried_completed_stage_ids") != state.get("current_completed_stage_ids")
        or authority.get("carried_paid_model_stages")
        != state.get("current_carried_paid_model_stages")
        or authority.get("continuation_kind")
        != (
            ARTIFIXER_POST_TRAINING_CONTINUATION_KIND
            if artifixer_continuation
            else "stage_three_plus_no_openai"
        )
        or (
            not artifixer_continuation
            and (
                authority.get("rerun_paid_model_stages") != []
                or authority.get("fresh_openai_cost_scope_attestation_digests") != {}
                or authority.get("warm_openai_external_service_spend_permitted") is not False
                or authority.get("source_artifixer_post_training_checkpoint_digest") is not None
                or authority.get("remote_artifixer_post_training_checkpoint_root") is not None
                or authority.get("openai_public_environment") != {}
                or authority.get("openai_stage_max_cost_usd") != {}
                or authority.get("openai_maximum_requests") != 0
            )
        )
        or (
            artifixer_continuation
            and (
                authority.get("rerun_paid_model_stages")
                != list(ARTIFIXER_POST_TRAINING_RERUN_PAID_MODEL_STAGES)
                or authority.get("fresh_openai_cost_scope_attestation_digests")
                != artifixer_session_authority.get("cost_scope_attestation_digests")
                or authority.get("warm_openai_external_service_spend_permitted") is not True
                or authority.get("source_artifixer_post_training_checkpoint_digest")
                != state.get("current_artifixer_post_training_checkpoint_digest")
                or _remote_artifixer_checkpoint_root(
                    authority.get("remote_artifixer_post_training_checkpoint_root")
                )
                != authority.get("remote_artifixer_post_training_checkpoint_root")
                or authority.get("openai_public_environment")
                != artifixer_session_authority.get("public_environment")
                or authority.get("openai_stage_max_cost_usd")
                != artifixer_session_authority.get("stage_max_cost_usd")
                or authority.get("openai_maximum_requests")
                != artifixer_session_authority.get("maximum_openai_requests")
            )
        )
        or _remote_checkpoint_root(authority.get("remote_checkpoint_root"))
        != authority.get("remote_checkpoint_root")
        or authority.get("watchdog_deadline_epoch") != session.get("watchdog_deadline_epoch")
        or authority.get("maximum_output_archive_bytes")
        != session.get("maximum_warm_output_archive_bytes")
        or authority.get("required_remaining_runtime_seconds")
        != diagnostic_required_parent_ttl_seconds(
            int(state.get("current_completed_stage_prefix_count") or 0)
        )
        - BOOTSTRAP_TRANSFER_AND_NO_SPEND_RESERVE_SECONDS
        or float(authority.get("watchdog_deadline_epoch") or 0) - now
        < int(authority.get("required_remaining_runtime_seconds") or 0)
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_iteration_authority_invalid"
        )
    return session, state, authority
