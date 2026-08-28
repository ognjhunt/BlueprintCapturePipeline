"""Retain one Vast scene diagnostic worker for immutable source-only iterations.

The warm session is deliberately non-qualifying.  One ordinary scene diagnostic
bundle performs the expensive image, toolchain, checkpoint, and dependency
bootstrap.  Later iterations may replace only an inventory-bound source overlay
from the exact tip of a pushed ``codex/*`` branch and resume the exact scientific
checkpoint.  No interface accepts an arbitrary command.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shlex
import time
import urllib.error
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .common import ensure_dir, redacted_failure_detail, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest, canonical_json
from .retained_gpu_session_lifecycle import record_retained_gpu_state
from .gpu_render_providers import VastRenderProvider
from .native_task_arena_warm_vast import (
    _dispatch_warm_script_over_ssh,
    _run_pinned_ssh,
)
from .paid_resource_admission import (
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .task_evaluation_scene_configuration_bundle import (
    load_scene_configuration_provider_bundle_receipt,
)
from .task_evaluation_scene_configuration_diagnostic_checkpoint import (
    validate_scene_configuration_diagnostic_checkpoint,
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
    OVERLAY_RECEIPT_SCHEMA_VERSION,
    OVERLAY_SCHEMA_VERSION,
    SceneConfigurationWarmDiagnosticError,
    _absolute_directory,
    _read,
    _record,
    _record_path,
    _validated_release_receipt,
    _write_exclusive,
    build_scene_configuration_warm_source_overlay,
    validate_scene_configuration_warm_source_overlay,
)
from .task_evaluation_scene_configuration_warm_remote_protocol import (
    _remote_iteration_script,
)
from .task_evaluation_scene_configuration_warm_execution_contract import (
    warm_execution_binding_blockers as _warm_execution_binding_blockers,
)
from .task_evaluation_scene_configuration_warm_allocation import (
    scene_configuration_warm_closeout_allocation_binding,
    scene_configuration_warm_iteration_allocation_binding,
    validate_warm_claim_boundary as _validate_claim_boundary,
)
from .task_evaluation_scene_configuration_warm_transport import (
    SIGNED_URL_RETRIEVAL_RESERVE_SECONDS,
    _download_bounded_when_ready,
    _output_object_ready,
    validated_warm_staging_urls,
)
from .task_evaluation_scene_configuration_vast import (
    _extract_provider_output,
    _provider_transfer_byte_budget,
)
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)
from .watchdog_owner_teardown_contract import write_owner_teardown_cancel_request


SESSION_AUTHORITY_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_warm_session_authority.v1"
)
SESSION_SCHEMA_VERSION = "task_evaluation_scene_configuration_warm_session.v1"
ITERATION_AUTHORITY_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_warm_iteration_authority.v1"
)
ITERATION_RESULT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_warm_iteration_result.v1"
)
CLOSEOUT_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_warm_session_closeout.v1"
)
SESSION_STATE_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_warm_session_state.v1"
)
SESSION_JOURNAL_NAME = "task_evaluation_scene_configuration_warm_session.jsonl"
SESSION_STATE_NAME = "task_evaluation_scene_configuration_warm_session_state.v1.json"
SESSION_OWNER_LOCK_NAME = "task_evaluation_scene_configuration_warm_session.lock"
REMOTE_ROOT = "/workspace/task_evaluation_scene_configuration_warm"
BASE_RUNTIME_ROOT = (
    "/workspace/task_evaluation_scene_configuration_provider_bundle/provider_runtime"
)
BASE_OUTPUT_ROOT = (
    "/workspace/task_evaluation_scene_configuration_provider_bundle/runtime_output"
)
MAX_WARM_ITERATIONS = 64
MIN_REMAINING_WATCHDOG_SECONDS = 180
WARM_CARRIED_PAID_MODEL_STAGES = (
    "artifixer_semantic_teacher",
    "artifixer_visual_review",
    "content_agents",
)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ITERATION_ID = re.compile(r"i[0-9]{3}-[0-9a-f]{12}\Z")
_HOST = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?\Z")

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
    bundle = load_scene_configuration_provider_bundle_receipt(
        bundle_path, diagnostic_only=True
    )
    paid_path = Path(paid_attempt_authority_path).expanduser().resolve()
    paid = validate_scene_configuration_paid_authority(
        _read(paid_path, code="scene_configuration_warm_paid_authority_invalid"),
        bundle_receipt=bundle,
    )
    release_path = Path(diagnostic_release_receipt_path).expanduser().resolve()
    release = _validated_release_receipt(
        release_path, source_commit=str(bundle["source_commit"])
    )
    try:
        diagnostic_bootstrap_mode = validate_diagnostic_bootstrap_mode(
            bundle.get("diagnostic_bootstrap_mode")
        )
    except ValueError as exc:
        raise SceneConfigurationWarmDiagnosticError(str(exc)) from exc
    fresh_diagnostic_bootstrap = (
        diagnostic_bootstrap_mode == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
    )
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
    _warm_download_ceiling, warm_output_ceiling = _provider_transfer_byte_budget(
        bundle
    )
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
            != (checkpoint.get("scientific_bindings") or {}).get(
                "binding_digest"
            )
        )
        or (
            fresh_diagnostic_bootstrap
            and (
                bundle.get("source_diagnostic_checkpoint_digest") is not None
                or bundle.get("carried_completed_stage_count") != 0
                or _DIGEST.fullmatch(
                    str(bundle.get("diagnostic_scientific_binding_digest") or "")
                )
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
        "construction_envelope_digest": bundle[
            "portable_construction_envelope_digest"
        ],
        "source_checkpoint_digest": (
            checkpoint["checkpoint_digest"] if checkpoint is not None else None
        ),
        "diagnostic_bootstrap_mode": diagnostic_bootstrap_mode,
        "bootstrap_carried_completed_stage_prefix_count": checkpoint[
            "completed_stage_prefix_count"
        ] if checkpoint is not None else 0,
        "bootstrap_carried_completed_stage_ids": [
            str(row["stage_id"])
            for row in checkpoint["completed_stage_results"]
        ] if checkpoint is not None else [],
        "bootstrap_uses_one_shot_paid_authority": True,
        "warm_iterations_require_all_paid_model_stages_carried": True,
        "required_carried_paid_model_stages_for_retention": list(
            WARM_CARRIED_PAID_MODEL_STAGES
        ),
        "scientific_binding_digest": (
            checkpoint["scientific_bindings"]["binding_digest"]
            if checkpoint is not None
            else bundle["diagnostic_scientific_binding_digest"]
        ),
        "diagnostic_stage_sequence_ids": list(
            bundle["diagnostic_stage_sequence_ids"]
        ),
        "paid_attempt_authority": _record(paid_path),
        "paid_attempt_authority_digest": paid["authority_digest"],
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "maximum_warm_iterations": maximum_warm_iterations,
        "maximum_single_resource_ttl_seconds": paid[
            "maximum_single_resource_ttl_seconds"
        ],
        "maximum_hourly_rate_usd": paid["maximum_hourly_rate_usd"],
        "aggregate_provider_compute_spend_cap_usd": paid[
            "provider_compute_spend_cap_usd"
        ],
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
    authority["authority_digest"] = canonical_digest(
        authority, digest_field="authority_digest"
    )
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
    authority = _read(
        path, code="scene_configuration_warm_session_authority_invalid"
    )
    _validate_claim_boundary(
        authority, code="scene_configuration_warm_session_authority_invalid"
    )
    if (
        authority.get("schema_version") != SESSION_AUTHORITY_SCHEMA_VERSION
        or authority.get("status") != "authorized"
        or authority.get("program_id") != "arm-decision-proof-v1"
        or authority.get("day_gate") != "day-28"
        or authority.get("provider") != "vast"
        or authority.get("provider_bundle_kind")
        != "task_evaluation_scene_configuration"
        or _COMMIT.fullmatch(str(authority.get("source_commit") or "")) is None
        or _DIGEST.fullmatch(str(authority.get("bundle_sha256") or "")) is None
        or not str(authority.get("run_id") or "")
        or _DIGEST.fullmatch(str(authority.get("toolchain_digest") or "")) is None
        or _DIGEST.fullmatch(
            str(authority.get("construction_envelope_digest") or "")
        )
        is None
        or not isinstance(authority.get("diagnostic_stage_sequence_ids"), list)
        or len(authority.get("diagnostic_stage_sequence_ids") or []) != 6
        or (
            authority.get("diagnostic_bootstrap_mode")
            == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
            and authority.get("source_checkpoint_digest") is not None
        )
        or (
            authority.get("diagnostic_bootstrap_mode")
            != FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
            and (
                authority.get("diagnostic_bootstrap_mode")
                != CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE
                or _DIGEST.fullmatch(
                    str(authority.get("source_checkpoint_digest") or "")
                )
                is None
            )
        )
        or _DIGEST.fullmatch(
            str(authority.get("scientific_binding_digest") or "")
        )
        is None
        or not 0
        <= int(authority.get("bootstrap_carried_completed_stage_prefix_count") or 0)
        <= 6
        or authority.get("bootstrap_uses_one_shot_paid_authority") is not True
        or authority.get("warm_iterations_require_all_paid_model_stages_carried")
        is not True
        or authority.get("required_carried_paid_model_stages_for_retention")
        != list(WARM_CARRIED_PAID_MODEL_STAGES)
        or authority.get("maximum_provider_allocations") != 1
        or authority.get("maximum_automatic_retries") != 0
        or not 1
        <= int(authority.get("maximum_warm_iterations") or 0)
        <= MAX_WARM_ITERATIONS
        or float(authority.get("maximum_hourly_rate_usd") or 0) <= 0
        or float(authority.get("aggregate_provider_compute_spend_cap_usd") or 0)
        <= 0
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
            isinstance(row, Mapping)
            for row in (bundle_record, paid_record, release_record)
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
        or bundle.get("diagnostic_bootstrap_mode")
        != authority.get("diagnostic_bootstrap_mode")
        or bundle.get("diagnostic_scientific_binding_digest")
        != authority.get("scientific_binding_digest")
        or bundle.get("diagnostic_stage_sequence_ids")
        != authority.get("diagnostic_stage_sequence_ids")
        or (
            authority.get("diagnostic_bootstrap_mode")
            == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
            and authority.get("source_checkpoint_digest") is not None
        )
        or paid.get("authority_digest")
        != authority.get("paid_attempt_authority_digest")
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
    advanced_checkpoint: Mapping[str, Any],
    bootstrap_allocation_binding_digest: str,
    observed_now_epoch: float | None = None,
) -> dict[str, Any]:
    """Seal a retained adapter result as the sole owned warm session."""

    authority_path = Path(session_authority_path).expanduser().resolve()
    authority = validate_scene_configuration_warm_session_authority(authority_path)
    adapter_path = Path(adapter_result_path).expanduser().resolve()
    adapter = _read(
        adapter_path, code="scene_configuration_warm_adapter_result_invalid"
    )
    watchdog_path = Path(watchdog_handoff_path).expanduser().resolve()
    watchdog = _read(
        watchdog_path, code="scene_configuration_warm_watchdog_handoff_invalid"
    )
    decision = adapter.get("retention_decision")
    instance_ids = adapter.get("vast_instance_ids")
    now = time.time() if observed_now_epoch is None else float(observed_now_epoch)
    try:
        advanced_local = validate_scene_configuration_diagnostic_checkpoint(
            checkpoint_root=str(advanced_checkpoint.get("checkpoint_root") or "")
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_advanced_checkpoint_invalid"
        ) from exc
    remote_relative = PurePosixPath(
        str(advanced_checkpoint.get("provider_output_relative_root") or "")
    )
    remote_checkpoint_root = PurePosixPath(BASE_OUTPUT_ROOT).joinpath(
        *remote_relative.parts
    )
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
        or adapter.get("provider_bundle_kind")
        != "task_evaluation_scene_configuration"
        or adapter.get("provider_bundle_sha256") != authority.get("bundle_sha256")
        or not isinstance(decision.get("warm_worker_evidence"), Mapping)
        or decision["warm_worker_evidence"].get("provider_bundle_kind")
        != "task_evaluation_scene_configuration"
        or decision["warm_worker_evidence"].get(
            "scene_configuration_runtime_root_ready"
        )
        is not True
        or watchdog.get("schema_version")
        != "vast_independent_watchdog_handoff.v1"
        or watchdog.get("status") != "armed"
        or watchdog.get("independent_process") is not True
        or watchdog.get("watchdog_armed_before_allocation") is not True
        or watchdog.get("watchdog_deadline_epoch")
        != decision.get("watchdog_deadline_epoch")
        or watchdog.get("watchdog_pid") != decision.get("watchdog_pid")
        or watchdog.get("watchdog_out_dir") != decision.get("watchdog_out_dir")
        or watchdog.get("pod_name_prefix")
        != decision.get("watchdog_pod_name_prefix")
        or watchdog.get("started_instance_id_path")
        != decision.get("watchdog_started_instance_id_path")
        or started_id != str(instance_ids[0])
        or remote_relative.is_absolute()
        or not remote_relative.parts
        or ".." in remote_relative.parts
        or int(advanced_checkpoint.get("completed_stage_prefix_count") or 0) < 3
        or advanced_local.get("checkpoint_digest")
        != advanced_checkpoint.get("checkpoint_digest")
        or (advanced_local.get("scientific_bindings") or {}).get(
            "binding_digest"
        )
        != authority.get("scientific_binding_digest")
        or _DIGEST.fullmatch(
            str(advanced_checkpoint.get("checkpoint_digest") or "")
        )
        is None
        or _DIGEST.fullmatch(bootstrap_allocation_binding_digest) is None
        or float(watchdog.get("watchdog_deadline_epoch") or 0) - now
        < MIN_REMAINING_WATCHDOG_SECONDS
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_session_invalid"
        )
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
        "construction_envelope_digest": authority[
            "construction_envelope_digest"
        ],
        "bootstrap_source_checkpoint_digest": authority[
            "source_checkpoint_digest"
        ],
        "source_checkpoint_digest": advanced_checkpoint["checkpoint_digest"],
        "carried_completed_stage_prefix_count": advanced_checkpoint[
            "completed_stage_prefix_count"
        ],
        "carried_completed_stage_ids": [
            str(row.get("stage_id") or "")
            for row in advanced_local.get("completed_stage_results") or []
        ],
        "diagnostic_stage_sequence_ids": list(
            authority["diagnostic_stage_sequence_ids"]
        ),
        "carried_paid_model_stages": list(
            authority["required_carried_paid_model_stages_for_retention"]
        ),
        "rerun_paid_model_stages": [],
        "warm_openai_external_service_spend_permitted": False,
        "scientific_binding_digest": authority["scientific_binding_digest"],
        "session_authority_digest": authority["authority_digest"],
        "bootstrap_allocation_binding_digest": (
            bootstrap_allocation_binding_digest
        ),
        "maximum_warm_iterations": authority["maximum_warm_iterations"],
        "maximum_hourly_rate_usd": authority["maximum_hourly_rate_usd"],
        "aggregate_provider_compute_spend_cap_usd": authority[
            "aggregate_provider_compute_spend_cap_usd"
        ],
        "maximum_warm_output_archive_bytes": authority[
            "maximum_warm_output_archive_bytes"
        ],
        "watchdog_pid": watchdog.get("watchdog_pid"),
        "watchdog_deadline_epoch": watchdog["watchdog_deadline_epoch"],
        "watchdog_out_dir": watchdog["watchdog_out_dir"],
        "watchdog_pod_name_prefix": watchdog["pod_name_prefix"],
        "adapter_result": _record(adapter_path),
        "watchdog_handoff": _record(watchdog_path),
        "base_runtime_root": BASE_RUNTIME_ROOT,
        "base_output_root": BASE_OUTPUT_ROOT,
        "current_remote_checkpoint_root": remote_checkpoint_root.as_posix(),
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
    session["session_digest"] = canonical_digest(
        session, digest_field="session_digest"
    )
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
        "current_remote_checkpoint_root": session[
            "current_remote_checkpoint_root"
        ],
        "current_completed_stage_prefix_count": session[
            "carried_completed_stage_prefix_count"
        ],
        "current_completed_stage_ids": list(
            session["carried_completed_stage_ids"]
        ),
        "current_carried_paid_model_stages": list(
            session["carried_paid_model_stages"]
        ),
        "consumed_openai_cost_scope_attestation_digests": [],
        "scientific_binding_digest": session["scientific_binding_digest"],
        "continuing_spend": True,
        "state_digest": "",
    }
    initial_state["state_digest"] = canonical_digest(
        initial_state, digest_field="state_digest"
    )
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
    allowed_state_statuses: frozenset[str] = frozenset(
        {"ready", "iteration_failed"}
    ),
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _absolute_directory(
        session_root, code="scene_configuration_warm_session_root_invalid"
    )
    session = _read(
        root / f"{SESSION_SCHEMA_VERSION}.json",
        code="scene_configuration_warm_session_invalid",
    )
    state = _read(
        root / SESSION_STATE_NAME,
        code="scene_configuration_warm_session_state_invalid",
    )
    _validate_claim_boundary(
        session, code="scene_configuration_warm_session_invalid"
    )
    now = time.time() if observed_now_epoch is None else float(observed_now_epoch)
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
        or _DIGEST.fullmatch(
            str(session.get("construction_envelope_digest") or "")
        )
        is None
        or session.get("session_digest")
        != canonical_digest(session, digest_field="session_digest")
        or state.get("schema_version") != SESSION_STATE_SCHEMA_VERSION
        or state.get("session_digest") != session.get("session_digest")
        or state.get("state_digest")
        != canonical_digest(state, digest_field="state_digest")
        or state.get("scientific_binding_digest")
        != session.get("scientific_binding_digest")
        or state.get("continuing_spend") is not True
        or state.get("status") not in allowed_state_statuses
        or not 0
        <= int(state.get("completed_iteration_count") or 0)
        <= int(state.get("attempted_iteration_count") or 0)
        <= int(session.get("maximum_warm_iterations") or 0)
        or not isinstance(
            state.get("consumed_openai_cost_scope_attestation_digests"), list
        )
        or not 3 <= int(state.get("current_completed_stage_prefix_count") or 0) <= 6
        or not isinstance(state.get("current_completed_stage_ids"), list)
        or len(state.get("current_completed_stage_ids") or [])
        != int(state.get("current_completed_stage_prefix_count") or 0)
        or state.get("current_carried_paid_model_stages")
        != list(WARM_CARRIED_PAID_MODEL_STAGES)
        or (
            require_iteration_window
            and float(session.get("watchdog_deadline_epoch") or 0) - now
            < MIN_REMAINING_WATCHDOG_SECONDS
        )
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_session_invalid"
        )
    return session, state


@contextmanager
def scene_configuration_warm_session_owner_lock(
    session_root: str | Path,
) -> Iterator[None]:
    root = _absolute_directory(
        session_root, code="scene_configuration_warm_session_root_invalid"
    )
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
        if overlay.get("scientific_binding_digest") != state.get(
            "scientific_binding_digest"
        ):
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
            "carried_completed_stage_prefix_count": state[
                "current_completed_stage_prefix_count"
            ],
            "carried_completed_stage_ids": list(
                state["current_completed_stage_ids"]
            ),
            "carried_paid_model_stages": list(
                state["current_carried_paid_model_stages"]
            ),
            "rerun_paid_model_stages": [],
            "fresh_openai_cost_scope_attestation_digests": {},
            "warm_openai_external_service_spend_permitted": False,
            "watchdog_deadline_epoch": session["watchdog_deadline_epoch"],
            "aggregate_provider_compute_spend_cap_usd": session[
                "aggregate_provider_compute_spend_cap_usd"
            ],
            "maximum_output_archive_bytes": session[
                "maximum_warm_output_archive_bytes"
            ],
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
        authority["authority_digest"] = canonical_digest(
            authority, digest_field="authority_digest"
        )
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
    authority = _read(
        path.resolve(), code="scene_configuration_warm_iteration_authority_invalid"
    )
    _validate_claim_boundary(
        authority, code="scene_configuration_warm_iteration_authority_invalid"
    )
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
    if (
        authority.get("schema_version") != ITERATION_AUTHORITY_SCHEMA_VERSION
        or authority.get("status") != "authorized"
        or authority.get("session_digest") != session.get("session_digest")
        or authority.get("previous_state_digest") != state.get("state_digest")
        or authority.get("provider") != "vast"
        or authority.get("provider_instance_id")
        != session.get("provider_instance_id")
        or authority.get("maximum_provider_allocations") != 0
        or authority.get("maximum_instance_lifecycle_mutations") != 0
        or authority.get("maximum_remote_workload_dispatches") != 1
        or authority.get("maximum_automatic_retries") != 0
        or authority.get("iteration_index") != expected_index
        or authority.get("iteration_index")
        > session.get("maximum_warm_iterations")
        or _ITERATION_ID.fullmatch(str(authority.get("iteration_id") or ""))
        is None
        or authority.get("source_overlay_receipt_digest")
        != overlay.get("receipt_digest")
        or authority.get("source_overlay_archive_sha256")
        != overlay["overlay_archive"]["sha256"]
        or authority.get("source_overlay_manifest_digest")
        != overlay.get("manifest_digest")
        or authority.get("source_checkpoint_digest")
        != state.get("current_checkpoint_digest")
        or authority.get("scientific_binding_digest")
        != session.get("scientific_binding_digest")
        or authority.get("carried_completed_stage_prefix_count")
        != state.get("current_completed_stage_prefix_count")
        or authority.get("carried_completed_stage_ids")
        != state.get("current_completed_stage_ids")
        or authority.get("carried_paid_model_stages")
        != state.get("current_carried_paid_model_stages")
        or authority.get("rerun_paid_model_stages") != []
        or authority.get("fresh_openai_cost_scope_attestation_digests") != {}
        or authority.get("warm_openai_external_service_spend_permitted") is not False
        or _remote_checkpoint_root(authority.get("remote_checkpoint_root"))
        != authority.get("remote_checkpoint_root")
        or authority.get("watchdog_deadline_epoch")
        != session.get("watchdog_deadline_epoch")
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


def _consume_iteration_authority(
    *, root: Path, state: Mapping[str, Any], authority: Mapping[str, Any]
) -> dict[str, Any]:
    digest = str(authority["authority_digest"])
    consumed_root = root / "consumed_iterations"
    consumed_root.mkdir(mode=0o700, exist_ok=True)
    destination = consumed_root / f"{digest[7:]}.json"
    payload = {
        "schema_version": "task_evaluation_scene_configuration_warm_iteration_consumption.v1",
        "authority_digest": digest,
        "session_digest": authority["session_digest"],
        "iteration_id": authority["iteration_id"],
        "consumed_at": utc_now_iso(),
        "maximum_provider_allocations": 0,
        "consumption_digest": "",
    }
    payload["consumption_digest"] = canonical_digest(
        payload, digest_field="consumption_digest"
    )
    try:
        _write_exclusive(destination, payload)
    except FileExistsError as exc:
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_iteration_authority_replayed"
        ) from exc
    next_state = {
        **dict(state),
        "status": "iteration_running",
        "attempted_iteration_count": int(state["attempted_iteration_count"]) + 1,
        "active_iteration_id": authority["iteration_id"],
        "active_iteration_authority_digest": digest,
        "state_digest": "",
    }
    next_state["state_digest"] = canonical_digest(
        next_state, digest_field="state_digest"
    )
    _write_state(root, next_state)
    _append_session_event(
        root,
        {
            "schema_version": SESSION_STATE_SCHEMA_VERSION,
            "event": "iteration_authority_consumed",
            "iteration_id": authority["iteration_id"],
            "authority_digest": digest,
            "state_digest": next_state["state_digest"],
            "recorded_at": utc_now_iso(),
        },
    )
    return payload


def _wait_for_remote_output_or_exit(
    *,
    session: Mapping[str, Any],
    dispatch: Mapping[str, Any],
    output_get_url: str,
    attempt_key: str,
    deadline_monotonic: float,
) -> dict[str, Any]:
    pid = dispatch.get("remote_pid")
    enrollment = dispatch.get("host_key_enrollment")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or not isinstance(enrollment, Mapping)
    ):
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_dispatch_identity_invalid"
        )
    remote_log = f"/workspace/native_task_arena_warm_dispatches/{attempt_key}/run.log"
    started = time.monotonic()
    overlay_latency: float | None = None
    entrypoint_latency: float | None = None
    last_probe: dict[str, Any] = {}
    while time.monotonic() < deadline_monotonic:
        if _output_object_ready(output_get_url):
            return {
                "status": "output_ready",
                "overlay_applied_latency_seconds": overlay_latency,
                "entrypoint_started_latency_seconds": entrypoint_latency,
                "remote_probe": last_probe,
            }
        command = (
            f"if kill -0 {pid} 2>/dev/null; then state=running; else state=exited; fi; "
            "printf 'STATE:%s\\n' \"$state\"; "
            "grep -aoE 'BLUEPRINT_SCENE_WARM_(OVERLAY_APPLIED|ENTRYPOINT_STARTED|PROVIDER_OUTPUT_UPLOAD_OK|BLOCKED:[A-Za-z0-9:_.-]+)' -- "
            + shlex.quote(remote_log)
            + " | tail -n 40 || true"
        )
        last_probe = _run_pinned_ssh(
            session=session,
            known_hosts_file=str(enrollment.get("known_hosts_file") or ""),
            remote_argv=["sh", "-c", command],
            timeout_seconds=30,
        )
        stdout = str(last_probe.get("stdout") or "")
        elapsed = max(0.0, time.monotonic() - started)
        if overlay_latency is None and "BLUEPRINT_SCENE_WARM_OVERLAY_APPLIED" in stdout:
            overlay_latency = elapsed
        if entrypoint_latency is None and "BLUEPRINT_SCENE_WARM_ENTRYPOINT_STARTED" in stdout:
            entrypoint_latency = elapsed
        if "STATE:exited" in stdout:
            # One final object read closes the race between process exit and PUT
            # visibility before failing immediately instead of waiting to TTL.
            ready = _output_object_ready(output_get_url)
            return {
                "status": "output_ready" if ready else "remote_exited_without_output",
                "overlay_applied_latency_seconds": overlay_latency,
                "entrypoint_started_latency_seconds": entrypoint_latency,
                "remote_probe": last_probe,
            }
        time.sleep(2)
    return {
        "status": "timeout",
        "overlay_applied_latency_seconds": overlay_latency,
        "entrypoint_started_latency_seconds": entrypoint_latency,
        "remote_probe": last_probe,
    }


def _quiesce_remote_dispatch(
    *,
    session: Mapping[str, Any],
    dispatch: Mapping[str, Any],
    attempt_key: str,
) -> dict[str, Any]:
    """Boundedly stop the exact remote dispatch session before URL cleanup."""

    pid = dispatch.get("remote_pid")
    pgid = dispatch.get("remote_process_group_id")
    sid = dispatch.get("remote_session_id")
    enrollment = dispatch.get("host_key_enrollment")
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (pid, pgid, sid))
        or pid != pgid
        or pid != sid
        or not isinstance(enrollment, Mapping)
        or re.fullmatch(r"[0-9a-f]{16}", attempt_key) is None
    ):
        return {"status": "unproven", "remote_session_absent": False}
    expected_script = f"/workspace/native_task_arena_warm_dispatches/{attempt_key}/run.sh"
    command = f"""
set -eu
root={pid}
sid={sid}
attempt={shlex.quote(attempt_key)}
expected={shlex.quote(expected_script)}
session_members() {{
  for proc in /proc/[0-9]*; do
    [ -r "$proc/stat" ] || continue
    proc_pid=${{proc##*/}}
    stat=$(cat "$proc/stat" 2>/dev/null || true)
    [ -n "$stat" ] || continue
    rest=${{stat##*) }}
    set -- $rest
    [ "$#" -ge 4 ] || continue
    [ "$1" != Z ] || continue
    [ "$4" = "$sid" ] && printf '%s\\n' "$proc_pid"
  done
}}
members=$(session_members || true)
[ -n "$members" ] || {{ printf 'REMOTE_SESSION:absent\\n'; exit 0; }}
if kill -0 "$root" 2>/dev/null; then
  cmdline=$(tr '\\000' ' ' < "/proc/$root/cmdline" 2>/dev/null || true)
  case "$cmdline" in
    *"bash $expected"*) ;;
    *) printf 'REMOTE_SESSION:identity_mismatch\\n'; exit 74;;
  esac
fi
for member in $members; do
  kill -0 "$member" 2>/dev/null || continue
  tr '\\000' '\\n' < "/proc/$member/environ" 2>/dev/null \
    | grep -Fqx "BLUEPRINT_SCENE_WARM_DISPATCH_ATTEMPT=$attempt" \
    || {{ printf 'REMOTE_SESSION:identity_mismatch\\n'; exit 74; }}
done
for _pass in 1 2; do
  members=$(session_members || true)
  [ -z "$members" ] || kill -TERM $members 2>/dev/null || true
  for _wait in 1 2 3 4 5; do
    [ -n "$(session_members || true)" ] || break
    sleep 1
  done
  members=$(session_members || true)
  [ -z "$members" ] || kill -KILL $members 2>/dev/null || true
done
if [ -n "$(session_members || true)" ]; then
  printf 'REMOTE_SESSION:running\\n'
  exit 75
fi
printf 'REMOTE_SESSION:absent\\n'
"""
    observation = _run_pinned_ssh(
        session=session,
        known_hosts_file=str(enrollment.get("known_hosts_file") or ""),
        remote_argv=["/bin/bash", "-c", command],
        timeout_seconds=20,
    )
    absent = bool(
        observation.get("status") == "completed"
        and "REMOTE_SESSION:absent" in str(observation.get("stdout") or "")
    )
    return {
        "status": "quiesced" if absent else "unproven",
        "remote_pid": pid,
        "remote_process_group_id": pgid,
        "remote_session_id": sid,
        "remote_session_absent": absent,
        "observation": observation,
    }


def _mark_iteration_state(
    *,
    root: Path,
    state: Mapping[str, Any],
    authority: Mapping[str, Any],
    status: str,
    result_digest: str,
    advanced_checkpoint_digest: str | None = None,
    advanced_checkpoint_prefix_count: int | None = None,
    advanced_checkpoint_stage_ids: list[str] | None = None,
    advanced_remote_checkpoint_root: str | None = None,
) -> dict[str, Any]:
    completed = int(state["completed_iteration_count"])
    next_state = {
        **dict(state),
        "status": status,
        "active_iteration_id": None,
        "active_iteration_authority_digest": None,
        "last_iteration_authority_digest": authority["authority_digest"],
        "last_iteration_result_digest": result_digest,
        "completed_iteration_count": completed + (1 if status == "ready" else 0),
        "state_digest": "",
    }
    if advanced_checkpoint_digest is not None:
        if (
            advanced_checkpoint_prefix_count is None
            or advanced_checkpoint_stage_ids is None
            or len(advanced_checkpoint_stage_ids) != advanced_checkpoint_prefix_count
            or advanced_checkpoint_prefix_count < 3
            or advanced_remote_checkpoint_root is None
        ):
            raise SceneConfigurationWarmDiagnosticError(
                "scene_configuration_warm_advanced_checkpoint_state_invalid"
            )
        next_state["current_checkpoint_digest"] = advanced_checkpoint_digest
        next_state["current_remote_checkpoint_root"] = _remote_checkpoint_root(
            advanced_remote_checkpoint_root
        )
        next_state["current_completed_stage_prefix_count"] = (
            advanced_checkpoint_prefix_count
        )
        next_state["current_completed_stage_ids"] = list(
            advanced_checkpoint_stage_ids
        )
        next_state["current_carried_paid_model_stages"] = list(
            WARM_CARRIED_PAID_MODEL_STAGES
        )
    next_state["state_digest"] = canonical_digest(
        next_state, digest_field="state_digest"
    )
    _write_state(root, next_state)
    _append_session_event(
        root,
        {
            "schema_version": SESSION_STATE_SCHEMA_VERSION,
            "event": "iteration_completed" if status == "ready" else "iteration_failed",
            "iteration_id": authority["iteration_id"],
            "authority_digest": authority["authority_digest"],
            "result_digest": result_digest,
            "state_digest": next_state["state_digest"],
            "recorded_at": utc_now_iso(),
        },
    )
    return next_state


def run_scene_configuration_warm_iteration(
    *,
    session_root: str | Path,
    authority_path: str | Path,
    job_dir: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    provider: Any | None = None,
) -> dict[str, Any]:
    """Dispatch one fixed checkpoint resume to the retained exact instance."""

    job = Path(job_dir).expanduser()
    if not job.is_absolute() or job.is_symlink():
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_iteration_job_invalid"
        )
    ensure_dir(job)
    with scene_configuration_warm_session_owner_lock(session_root):
        session, state, authority = validate_scene_configuration_warm_iteration_authority(
            session_root=session_root, authority_path=authority_path
        )
        if not execute:
            result = {
                "schema_version": ITERATION_RESULT_SCHEMA_VERSION,
                "status": "dry_run_ready",
                "provider_allocations_performed": 0,
                "instance_lifecycle_mutations_performed": 0,
                "remote_workload_dispatches_performed": 0,
                "iteration_id": authority["iteration_id"],
                "diagnostic_only": True,
                "development_only": True,
                "qualification_eligible": False,
                "configured_revision_publication_permitted": False,
                "offering_publication_permitted": False,
                "terminal_e2e_completion_permitted": False,
                "arbitrary_command_permitted": False,
                "raw_secret_values_recorded": False,
                "blockers": [],
                "result_digest": "",
            }
            result["result_digest"] = canonical_digest(
                result, digest_field="result_digest"
            )
            write_json(job / f"{ITERATION_RESULT_SCHEMA_VERSION}.json", result)
            return result
        require_paid_resource_admission_grant(
            paid_resource_admission_grant,
            resource_class="vast_provider_adapter",
            allocation_binding_digest=canonical_digest(
                scene_configuration_warm_iteration_allocation_binding(
                    session=session, authority=authority
                )
            ),
            require_allocation_binding=True,
        )
        consumption = _consume_iteration_authority(
            root=_absolute_directory(
                session_root, code="scene_configuration_warm_session_root_invalid"
            ),
            state=state,
            authority=authority,
        )
        retained_provider = provider or VastRenderProvider()
        blockers: list[str] = []
        dispatch: dict[str, Any] = {}
        cleanup: dict[str, Any] = {}
        execution: dict[str, Any] = {}
        remote_dispatches = 0
        dispatch_start_latency_seconds: float | None = None
        stage_completion_seconds: float | None = None
        remote_progress: dict[str, Any] = {}
        remote_quiescence: dict[str, Any] = {}
        remote_writer_absent = False
        remote_setup_to_entrypoint_seconds: float | None = None
        try:
            observation = retained_provider.inspect(str(session["provider_instance_id"]))
            if (
                observation.get("status") != "observed"
                or observation.get("api_confirmed") is not True
                or observation.get("provider_absence_confirmed") is not False
                or observation.get("instance_id")
                != str(session["provider_instance_id"])
                or observation.get("ssh_host") != session["ssh_host"]
                or observation.get("ssh_port") != session["ssh_port"]
                or str(observation.get("actual_status") or observation.get("cur_state") or "").lower()
                not in {"running", "active"}
            ):
                raise SceneConfigurationWarmDiagnosticError(
                    "scene_configuration_warm_instance_identity_not_live"
                )
            overlay_record = authority["source_overlay_receipt"]
            overlay_receipt = validate_scene_configuration_warm_source_overlay(
                _record_path(
                    overlay_record,
                    code="scene_configuration_warm_iteration_overlay_invalid",
                ),
                expected_source_commit=authority["source_commit"],
                expected_checkpoint_digest=authority["source_checkpoint_digest"],
            )
            archive = _record_path(
                overlay_receipt["overlay_archive"],
                code="scene_configuration_warm_iteration_overlay_invalid",
            )
            staging_dir = job / "object_store_staging"
            staging = stage_wam_provider_bundle_object_store(
                job_dir=staging_dir,
                bundle_path=archive,
                key_prefix="blueprint/arm-decision-proof-v1/scene-configuration/warm",
                expiration_seconds=max(
                    600,
                    math.ceil(
                        float(session["watchdog_deadline_epoch"]) - time.time()
                    )
                    + SIGNED_URL_RETRIEVAL_RESERVE_SECONDS,
                ),
            )
            if staging.get("status") != "completed":
                blockers.extend(staging.get("blockers") or ["scene_configuration_warm_object_store_staging_blocked"])
            else:
                staging_urls = validated_warm_staging_urls(
                    staging_dir=staging_dir,
                    staging=staging,
                    overlay_archive=archive,
                    watchdog_deadline_epoch=float(session["watchdog_deadline_epoch"]),
                )
                remote_script = _remote_iteration_script(
                    authority=authority,
                    session=session,
                    overlay_url=staging_urls["overlay_url"],
                    output_put_url=staging_urls["output_put_url"],
                )
                attempt_key = str(authority["authority_digest"])[7:23]
                dispatch_started = time.monotonic()
                dispatch = _dispatch_warm_script_over_ssh(
                    job=job,
                    session=session,
                    remote_script=remote_script,
                    attempt_key=attempt_key,
                    require_dedicated_session=True,
                )
                dispatch_start_latency_seconds = max(
                    0.0, time.monotonic() - dispatch_started
                )
                if dispatch.get("status") != "completed":
                    blockers.extend(dispatch.get("blockers") or ["scene_configuration_warm_dispatch_blocked"])
                else:
                    remote_dispatches = 1
                    output_zip = job / "vast_provider_runtime_output.zip"
                    output_get_url = staging_urls["output_get_url"]
                    stage_started = time.monotonic()
                    remote_progress = _wait_for_remote_output_or_exit(
                        session=session,
                        dispatch=dispatch,
                        output_get_url=output_get_url,
                        attempt_key=attempt_key,
                        deadline_monotonic=(
                            time.monotonic()
                            + max(1, int(float(session["watchdog_deadline_epoch"]) - time.time() - 120))
                        ),
                    )
                    remote_quiescence = _quiesce_remote_dispatch(
                        session=session,
                        dispatch=dispatch,
                        attempt_key=attempt_key,
                    )
                    remote_writer_absent = (
                        remote_quiescence.get("remote_session_absent") is True
                    )
                    if not remote_writer_absent:
                        blockers.append(
                            "scene_configuration_warm_remote_writer_quiescence_unproven"
                        )
                    remote_stdout = str(
                        (remote_progress.get("remote_probe") or {}).get("stdout")
                        or ""
                    )
                    for marker in re.findall(
                        r"BLUEPRINT_SCENE_WARM_BLOCKED:([A-Za-z0-9:_.-]+)",
                        remote_stdout,
                    ):
                        blockers.append(f"scene_configuration_warm_remote_blocked:{marker}")
                    setup_match = re.search(
                        r"BLUEPRINT_SCENE_WARM_REMOTE_SETUP_STARTED_EPOCH_NS:([0-9]+)",
                        remote_stdout,
                    )
                    entrypoint_match = re.search(
                        r"BLUEPRINT_SCENE_WARM_ENTRYPOINT_STARTED_EPOCH_NS:([0-9]+)",
                        remote_stdout,
                    )
                    if setup_match and entrypoint_match:
                        elapsed_ns = int(entrypoint_match.group(1)) - int(
                            setup_match.group(1)
                        )
                        if elapsed_ns >= 0:
                            remote_setup_to_entrypoint_seconds = elapsed_ns / 1e9
                    output_ready = remote_progress.get("status") == "output_ready"
                    if output_ready and remote_writer_absent:
                        output_ready = _download_bounded_when_ready(
                        url=output_get_url,
                        destination=output_zip,
                        maximum_bytes=int(session["maximum_warm_output_archive_bytes"]),
                        deadline_monotonic=time.monotonic() + 60,
                        )
                    stage_completion_seconds = max(
                        0.0, time.monotonic() - stage_started
                    )
                    if not output_ready:
                        blockers.append("scene_configuration_warm_output_timeout")
                    else:
                        execution, extraction_blockers = _extract_provider_output(
                            output_zip,
                            job / "immutable_execution",
                            maximum_archive_bytes=int(
                                session["maximum_warm_output_archive_bytes"]
                            ),
                            diagnostic_only=True,
                        )
                        blockers.extend(extraction_blockers)
                        blockers.extend(
                            _warm_execution_binding_blockers(
                                execution=execution,
                                session=session,
                                authority=authority,
                            )
                        )
        except (OSError, RuntimeError, TypeError, ValueError, urllib.error.URLError) as exc:
            blockers.append(
                "scene_configuration_warm_iteration_failed:"
                + redacted_failure_detail(exc)
            )
        finally:
            if (job / "object_store_staging").is_dir() and remote_writer_absent:
                cleanup = cleanup_staged_wam_provider_objects(
                    job / "object_store_staging"
                )
                if cleanup.get("all_objects_absent") is not True:
                    blockers.append("scene_configuration_warm_object_store_cleanup_unproven")
        advanced = execution.pop("_validated_advanced_checkpoint", None)
        unsafe_checkpoint_blockers = [
            item
            for item in blockers
            if not str(item).startswith("provider_result_blocker:")
        ]
        if not unsafe_checkpoint_blockers and not isinstance(advanced, Mapping):
            blockers.append("scene_configuration_warm_advanced_checkpoint_missing")
            unsafe_checkpoint_blockers.append(
                "scene_configuration_warm_advanced_checkpoint_missing"
            )
        advanced_checkpoint_value: dict[str, Any] | None = None
        advanced_remote_checkpoint_root: str | None = None
        if not unsafe_checkpoint_blockers and isinstance(advanced, Mapping):
            try:
                advanced_checkpoint_value = (
                    validate_scene_configuration_diagnostic_checkpoint(
                        checkpoint_root=Path(str(advanced["checkpoint_root"]))
                    )
                )
                advanced_relative_root = PurePosixPath(
                    str(advanced["provider_output_relative_root"])
                )
                if (
                    advanced_relative_root.is_absolute()
                    or ".." in advanced_relative_root.parts
                ):
                    raise ValueError("unsafe advanced checkpoint root")
                advanced_remote_checkpoint_root = (
                    PurePosixPath(REMOTE_ROOT)
                    / "iterations"
                    / str(authority["iteration_id"])
                    / "output"
                    / advanced_relative_root
                ).as_posix()
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                blockers.append(
                    "scene_configuration_warm_advanced_checkpoint_state_invalid"
                )
                unsafe_checkpoint_blockers.append(
                    "scene_configuration_warm_advanced_checkpoint_state_invalid"
                )
        safe_checkpoint_advanced = bool(
            not unsafe_checkpoint_blockers
            and isinstance(advanced, Mapping)
            and advanced_checkpoint_value is not None
            and advanced_remote_checkpoint_root is not None
        )
        result: dict[str, Any] = {
            "schema_version": ITERATION_RESULT_SCHEMA_VERSION,
            "status": "completed_diagnostic_only" if not blockers else "blocked_diagnostic_only",
            "iteration_id": authority["iteration_id"],
            "iteration_index": authority["iteration_index"],
            "session_digest": session["session_digest"],
            "authority_digest": authority["authority_digest"],
            "authorization_consumption": consumption,
            "provider_instance_id": session["provider_instance_id"],
            "provider_allocations_performed": 0,
            "instance_lifecycle_mutations_performed": 0,
            "remote_workload_dispatches_performed": remote_dispatches,
            "fix_overlay_dispatch_start_latency_seconds": (
                round(dispatch_start_latency_seconds, 6)
                if dispatch_start_latency_seconds is not None
                else None
            ),
            "stage_completion_wait_seconds": (
                round(stage_completion_seconds, 6)
                if stage_completion_seconds is not None
                else None
            ),
            "remote_setup_to_provider_entrypoint_seconds": (
                round(remote_setup_to_entrypoint_seconds, 6)
                if remote_setup_to_entrypoint_seconds is not None
                else None
            ),
            "remote_setup_to_provider_entrypoint_measured": (
                remote_setup_to_entrypoint_seconds is not None
            ),
            "runtime_clone_strategy": "reflink_auto_with_full_copy_fallback",
            "persistent_isaac_process_reused": False,
            "remote_progress": remote_progress,
            "remote_quiescence": remote_quiescence,
            "base_bundle_source_commit": session["source_commit"],
            "diagnostic_source_overlay_commit": authority["source_commit"],
            "source_overlay_manifest_digest": authority["source_overlay_manifest_digest"],
            "source_checkpoint_digest": authority["source_checkpoint_digest"],
            "advanced_checkpoint_digest": (
                advanced.get("checkpoint_digest") if isinstance(advanced, Mapping) else None
            ),
            "checkpoint_advanced_despite_stage_blocker": bool(
                safe_checkpoint_advanced and blockers
            ),
            "scientific_binding_digest": authority["scientific_binding_digest"],
            "carried_completed_stage_prefix_count": authority["carried_completed_stage_prefix_count"],
            "carried_completed_stage_ids": authority["carried_completed_stage_ids"],
            "carried_paid_model_stages": authority["carried_paid_model_stages"],
            "rerun_paid_model_stages": [],
            "fresh_openai_cost_scope_attestation_digests": {},
            "warm_openai_external_service_spend_permitted": False,
            "continuing_spend_from_this_run": True,
            "watchdog_deadline_epoch": session["watchdog_deadline_epoch"],
            "dispatch": dispatch,
            "object_store_cleanup": cleanup,
            "diagnostic_only": True,
            "development_only": True,
            "qualification_eligible": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "arbitrary_command_permitted": False,
            "raw_secret_values_recorded": False,
            "blockers": sorted(set(str(item) for item in blockers if str(item))),
            "result_digest": "",
        }
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
        write_json(job / f"{ITERATION_RESULT_SCHEMA_VERSION}.json", result)
        _mark_iteration_state(
            root=_absolute_directory(session_root, code="scene_configuration_warm_session_root_invalid"),
            state=_read(_absolute_directory(session_root, code="scene_configuration_warm_session_root_invalid") / SESSION_STATE_NAME, code="scene_configuration_warm_session_state_invalid"),
            authority=authority,
            status=(
                "ready"
                if not blockers
                else (
                    "teardown_required"
                    if not remote_writer_absent
                    else "iteration_failed"
                )
            ),
            result_digest=result["result_digest"],
            advanced_checkpoint_digest=(
                str(advanced["checkpoint_digest"])
                if safe_checkpoint_advanced and isinstance(advanced, Mapping)
                else None
            ),
            advanced_checkpoint_prefix_count=(
                int(advanced_checkpoint_value["completed_stage_prefix_count"])
                if safe_checkpoint_advanced and advanced_checkpoint_value is not None
                else None
            ),
            advanced_checkpoint_stage_ids=(
                [
                    str(row["stage_id"])
                    for row in advanced_checkpoint_value["completed_stage_results"]
                ]
                if safe_checkpoint_advanced and advanced_checkpoint_value is not None
                else None
            ),
            advanced_remote_checkpoint_root=(
                advanced_remote_checkpoint_root if safe_checkpoint_advanced else None
            ),
        )
        return result


def close_scene_configuration_warm_session(
    *,
    session_root: str | Path,
    output_path: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    provider: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Destroy the exact retained instance and prove global provider-zero."""

    destination = Path(output_path).expanduser()
    if not destination.is_absolute() or destination.is_symlink():
        raise SceneConfigurationWarmDiagnosticError(
            "scene_configuration_warm_closeout_output_invalid"
        )
    with scene_configuration_warm_session_owner_lock(session_root):
        session, state = validate_scene_configuration_warm_session(
            session_root=session_root,
            require_iteration_window=False,
            allowed_state_statuses=frozenset(
                {
                    "ready",
                    "iteration_failed",
                    "iteration_running",
                    "teardown_required",
                }
            ),
        )
        if not execute:
            result = {
                "schema_version": CLOSEOUT_SCHEMA_VERSION,
                "status": "dry_run_ready",
                "session_digest": session["session_digest"],
                "provider_instance_id": session["provider_instance_id"],
                "instance_lifecycle_mutations_performed": 0,
                "provider_instance_absent": False,
                "global_provider_zero_proven": False,
                "continuing_spend_from_this_run": True,
                "diagnostic_only": True,
                "qualification_eligible": False,
                "raw_secret_values_recorded": False,
                "closeout_digest": "",
            }
            result["closeout_digest"] = canonical_digest(
                result, digest_field="closeout_digest"
            )
            _write_exclusive(destination, result)
            return result
        require_paid_resource_admission_grant(
            paid_resource_admission_grant,
            resource_class="vast_provider_adapter",
            allocation_binding_digest=canonical_digest(
                scene_configuration_warm_closeout_allocation_binding(
                    session=session
                )
            ),
            require_allocation_binding=True,
        )
        retained_provider = provider or VastRenderProvider()
        instance_id = str(session["provider_instance_id"])
        lifecycle_manifest_path = (
            Path(session_root) / "retained_gpu_session_manifest.json"
        )
        lifecycle_manifest = (
            _read(
                lifecycle_manifest_path,
                code="scene_configuration_warm_lifecycle_invalid",
            )
            if lifecycle_manifest_path.is_file()
            else {}
        )
        if lifecycle_manifest.get("state") != "teardown_requested":
            record_retained_gpu_state(
                session_root,
                "teardown_requested",
                evidence={
                    "provider": "vast",
                    "provider_instance_id": session["provider_instance_id"],
                    "session_digest": session["session_digest"],
                },
            )
        try:
            termination = retained_provider.terminate(instance_id)
        except Exception as exc:  # noqa: BLE001 - closeout must still inspect/seal
            termination = {
                "status": "terminate_failed",
                "error_type": type(exc).__name__,
                "raw_secret_values_recorded": False,
            }
        observations: list[dict[str, Any]] = []
        deadline = monotonic() + max(1.0, float(timeout_seconds))
        consecutive_absent = 0
        while monotonic() < deadline and consecutive_absent < 2:
            try:
                observed = retained_provider.inspect(instance_id)
            except Exception as exc:  # noqa: BLE001 - seal blocked closeout
                observed = {
                    "status": "observation_failed",
                    "api_confirmed": False,
                    "provider_absence_confirmed": False,
                    "blockers": [redacted_failure_detail(exc)],
                    "raw_secret_values_recorded": False,
                }
            observations.append(dict(observed))
            if (
                observed.get("status") == "absent"
                and observed.get("provider_absence_confirmed") is True
                and observed.get("api_confirmed") is True
            ):
                consecutive_absent += 1
            else:
                consecutive_absent = 0
            if consecutive_absent < 2:
                sleep(2.0)
        global_inventories: list[dict[str, Any]] = []
        for index in range(2):
            try:
                inventory = retained_provider.billable_inventory(name_prefix="")
            except Exception as exc:  # noqa: BLE001 - preserve blocked closeout
                inventory = {
                    "status": "blocked",
                    "api_confirmed": False,
                    "blockers": [redacted_failure_detail(exc)],
                    "raw_secret_values_recorded": False,
                }
            global_inventories.append(dict(inventory))
            if index == 0:
                sleep(2.0)
        exact_absent = consecutive_absent >= 2
        global_zero = bool(
            len(global_inventories) == 2
            and all(
                row.get("api_confirmed") is True
                and row.get("live_resource_count") == 0
                and row.get("resources") == []
                for row in global_inventories
            )
        )
        blockers: list[str] = []
        if not exact_absent:
            blockers.append("scene_configuration_warm_provider_absence_unproven")
        if not global_zero:
            blockers.append("scene_configuration_warm_global_provider_zero_unproven")
        watchdog_cancel: dict[str, Any] = {}
        if exact_absent and global_zero:
            watchdog_cancel = write_owner_teardown_cancel_request(
                root=Path(str(session["watchdog_out_dir"])),
                pod_name_prefix=str(session["watchdog_pod_name_prefix"]),
                provider_name="vast",
                instance_id=instance_id,
            )
            record_retained_gpu_state(
                session_root,
                "provider_absent",
                evidence={
                    "provider": "vast",
                    "provider_instance_id": session["provider_instance_id"],
                    "global_provider_zero_proven": True,
                },
            )
        result = {
            "schema_version": CLOSEOUT_SCHEMA_VERSION,
            "status": "completed" if not blockers else "blocked",
            "session_digest": session["session_digest"],
            "provider_instance_id": session["provider_instance_id"],
            "termination": termination,
            "provider_absence_observations": observations,
            "global_billable_inventory_observations": global_inventories,
            "watchdog_cancel_request": watchdog_cancel,
            "instance_lifecycle_mutations_performed": 1,
            "provider_instance_absent": exact_absent,
            "global_provider_zero_proven": global_zero,
            "continuing_spend_from_this_run": not (exact_absent and global_zero),
            "diagnostic_only": True,
            "development_only": True,
            "qualification_eligible": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
            "raw_secret_values_recorded": False,
            "blockers": blockers,
            "closeout_digest": "",
        }
        result["closeout_digest"] = canonical_digest(
            result, digest_field="closeout_digest"
        )
        _write_exclusive(destination, result)
        if not blockers:
            terminal_state = {
                **dict(state),
                "status": "closed",
                "continuing_spend": False,
                "active_iteration_id": None,
                "active_iteration_authority_digest": None,
                "closeout_digest": result["closeout_digest"],
                "state_digest": "",
            }
            terminal_state["state_digest"] = canonical_digest(
                terminal_state, digest_field="state_digest"
            )
            _write_state(
                _absolute_directory(
                    session_root,
                    code="scene_configuration_warm_session_root_invalid",
                ),
                terminal_state,
            )
        return result


__all__ = [
    "CLOSEOUT_SCHEMA_VERSION",
    "ITERATION_AUTHORITY_SCHEMA_VERSION",
    "ITERATION_RESULT_SCHEMA_VERSION",
    "OVERLAY_RECEIPT_SCHEMA_VERSION",
    "OVERLAY_SCHEMA_VERSION",
    "SESSION_AUTHORITY_SCHEMA_VERSION",
    "SESSION_SCHEMA_VERSION",
    "SceneConfigurationWarmDiagnosticError",
    "build_scene_configuration_warm_source_overlay",
    "close_scene_configuration_warm_session",
    "materialize_scene_configuration_warm_iteration_authority",
    "materialize_scene_configuration_warm_session",
    "materialize_scene_configuration_warm_session_authority",
    "run_scene_configuration_warm_iteration",
    "scene_configuration_warm_closeout_allocation_binding",
    "scene_configuration_warm_iteration_allocation_binding",
    "scene_configuration_warm_session_owner_lock",
    "validate_scene_configuration_warm_iteration_authority",
    "validate_scene_configuration_warm_session",
    "validate_scene_configuration_warm_session_authority",
    "validate_scene_configuration_warm_source_overlay",
]
