"""Required capture-build entrypoint for the Task Evaluation Supervisor."""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import math
import os
import json
from pathlib import Path
import re
import subprocess
from subprocess import run as _run_subprocess
from typing import Any, Mapping, Sequence

from ..agent_operator_runtime import LIVE_AGENTS_SDK_ENV, env_truthy
from ..decision_evidence_contracts import canonical_digest
from .agents_sdk import DEFAULT_SUPERVISOR_AGENT_MODEL
from .capabilities import SupervisorContext
from .capture_ingress import load_capture_build_ingress
from .capture_reconstruction_routing import build_capture_reconstruction_route
from .contracts import AutonomyMode
from .supervisor import TaskEvaluationSupervisor, default_authority_envelope
from .reconstruction_execution_readiness import (
    bound_tool_ids_for_control_plane_inspection,
    build_reconstruction_execution_readiness,
    validate_reconstruction_execution_readiness,
)
from .tools import ToolRegistry, non_spend_tool_bindings
from .phase2_artifacts import write_phase2_artifact


CAPTURE_SUPERVISOR_LIFECYCLE_SCHEMA_VERSION = "task_evaluation_capture_supervisor_lifecycle.v4"
CAPTURE_SUPERVISOR_ALLOW_LIVE_AGENTS_SDK_ENV = "BLUEPRINT_CAPTURE_SUPERVISOR_ALLOW_LIVE_AGENTS_SDK"
CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD_ENV = "BLUEPRINT_CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD"
CAPTURE_SUPERVISOR_AGENT_MODEL_ENV = "BLUEPRINT_CAPTURE_SUPERVISOR_AGENT_MODEL"
MAX_CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD = 100.0
RECONSTRUCTION_READINESS_POINTER_SCHEMA_VERSION = (
    "task_evaluation_reconstruction_readiness_pointer.v1"
)
RECONSTRUCTION_SUPERVISOR_CONTINUATION_SCHEMA_VERSION = (
    "task_evaluation_reconstruction_supervisor_continuation.v1"
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_RECONSTRUCTION_CONTEXT_FIELDS = {
    "reconstruction_dataset_compiler",
    "arkit_metric_scaffold_compiler",
    "arkit_depth_surface_compilation_request",
    "arkit_depth_surface_compiler",
    "arkit_reconstruction_dataset_request",
    "arkit_reconstruction_dataset_exporter",
    "native_360_normalizer",
    "equirectangular_virtual_rig_compiler",
    "pose_estimation_request",
    "pose_estimator",
    "pose_refinement_request",
    "pose_refiner",
    "reconstruction_training_request",
    "gaussian_reconstruction_trainer",
    "heldout_appearance_evaluation_request",
    "heldout_appearance_evaluator",
    "metric_geometry_source",
    "metric_geometry_compiler",
    "metric_geometry_manifest",
    "collision_candidate_compiler",
    "collider_candidate_manifest",
    "collider_qualification_request",
    "collision_candidate_qualifier",
    "nurec_packaging_request",
    "nurec_openusd_packager",
    "isaac_verification_request",
    "isaac_asset_verifier",
    "external_reconstruction_import_request",
    "external_reconstruction_importer",
    "reconstruction_failure_diagnosis_request",
    "reconstruction_terminal_report_request",
    "camera_rig_validation_request",
    "metric_scale_validation_request",
    "generated_repair_candidate_request",
    "fresh_scene_preparation_status",
    "fresh_scene_sam31_task_input_request",
    "fresh_scene_calibrated_mask_request",
    "fresh_scene_removal_freeze_request",
    "fresh_scene_segment_cutout_request",
    "fresh_scene_artifixer_candidate_request",
    "fresh_scene_semantic_teacher_edit_request",
}


def _capture_supervisor_source_commit_sha(
    environ: Mapping[str, str] | None = None,
) -> str:
    source = os.environ if environ is None else environ
    configured = str(source.get("BLUEPRINT_SOURCE_COMMIT") or "").strip().lower()
    if configured:
        if re.fullmatch(r"[0-9a-f]{40}", configured) is None:
            raise ValueError("capture_supervisor_source_commit_invalid")
        return configured
    repository_root = Path(__file__).resolve().parents[3]
    try:
        completed = _run_subprocess(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("capture_supervisor_source_commit_unavailable") from exc
    commit = completed.stdout.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("capture_supervisor_source_commit_invalid")
    return commit


def _readiness_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith("Z"):
        return text
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    raise ValueError("capture_supervisor_generated_at_not_utc")


def validate_reconstruction_readiness_pointer(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the replaceable pointer to an immutable readiness snapshot."""

    pointer = dict(value)
    required = {
        "schema_version",
        "run_id",
        "capture_build_digest",
        "source_commit_sha",
        "readiness_digest",
        "readiness_relative_path",
        "previous_readiness_digest",
        "control_plane_binding",
        "recorded_at",
        "proof_boundary",
        "pointer_digest",
    }
    previous = pointer.get("previous_readiness_digest")
    proof = pointer.get("proof_boundary")
    readiness_digest = str(pointer.get("readiness_digest") or "")
    if (
        set(pointer) != required
        or pointer.get("schema_version")
        != RECONSTRUCTION_READINESS_POINTER_SCHEMA_VERSION
        or not str(pointer.get("run_id") or "").strip()
        or _SHA256_RE.fullmatch(str(pointer.get("capture_build_digest") or "")) is None
        or re.fullmatch(r"[0-9a-f]{40}", str(pointer.get("source_commit_sha") or ""))
        is None
        or _SHA256_RE.fullmatch(readiness_digest) is None
        or pointer.get("readiness_relative_path")
        != (
            "reconstruction_execution_readiness_history/"
            f"{readiness_digest.removeprefix('sha256:')}.json"
        )
        or (previous is not None and _SHA256_RE.fullmatch(str(previous)) is None)
        or not isinstance(pointer.get("control_plane_binding"), Mapping)
        or not _readiness_timestamp(pointer.get("recorded_at"))
        or proof
        != {
            "pointer_is_execution_authority": False,
            "pointer_is_reconstruction_evidence": False,
            "prior_readiness_snapshots_preserved": True,
            "physical_task_success_established": False,
        }
        or pointer.get("pointer_digest")
        != canonical_digest(pointer, digest_field="pointer_digest")
    ):
        raise ValueError("reconstruction_readiness_pointer_invalid")
    return pointer


def capture_supervisor_execution_options_from_env(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load one strict service-side inference envelope for capture ingress."""

    source = os.environ if environ is None else environ
    raw_allow = str(source.get(CAPTURE_SUPERVISOR_ALLOW_LIVE_AGENTS_SDK_ENV, "")).strip().lower()
    if raw_allow in {"", "0", "false", "no", "off"}:
        allow_live = False
    elif raw_allow in {"1", "true", "yes", "on"}:
        allow_live = True
    else:
        raise ValueError("capture_supervisor_live_agents_sdk_env_invalid")

    raw_budget = str(source.get(CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD_ENV, "")).strip()
    try:
        budget = 0.0 if not raw_budget else float(raw_budget)
    except ValueError as exc:
        raise ValueError("capture_supervisor_inference_budget_invalid") from exc
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("capture_supervisor_inference_budget_invalid")
    if budget > MAX_CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD:
        raise ValueError("capture_supervisor_inference_budget_exceeds_service_ceiling")
    if allow_live and budget <= 0:
        raise ValueError("capture_supervisor_live_inference_requires_positive_budget")
    if not allow_live and budget != 0:
        raise ValueError("capture_supervisor_disabled_inference_budget_must_be_zero")

    model = str(source.get(CAPTURE_SUPERVISOR_AGENT_MODEL_ENV, "")).strip()
    if not model:
        model = DEFAULT_SUPERVISOR_AGENT_MODEL
    if len(model) > 256 or any(ord(character) < 32 for character in model):
        raise ValueError("capture_supervisor_agent_model_invalid")
    return {
        "agent_model": model,
        "allow_live_agents_sdk": allow_live,
        "agent_inference_budget_usd": budget,
    }


def capture_supervisor_execution_profile(
    *,
    agent_model: str = DEFAULT_SUPERVISOR_AGENT_MODEL,
    allow_live_agents_sdk: bool = False,
    agent_inference_budget_usd: float = 0.0,
    source_commit_sha: str | None = None,
) -> dict[str, Any]:
    """Bind the exact execution authority that determines lifecycle idempotency."""

    commit = source_commit_sha or _capture_supervisor_source_commit_sha()
    if re.fullmatch(r"[0-9a-f]{40}", str(commit or "")) is None:
        raise ValueError("capture_supervisor_source_commit_invalid")
    value = {
        "schema_version": "task_evaluation_capture_supervisor_execution_profile.v2",
        "agent_harness": "openai_agents_sdk",
        "agent_model": agent_model,
        "allow_live_agents_sdk": allow_live_agents_sdk,
        "live_operator_gate_enabled": (
            env_truthy(LIVE_AGENTS_SDK_ENV) if allow_live_agents_sdk else False
        ),
        "agent_inference_budget_usd": float(agent_inference_budget_usd),
        "source_commit_sha": commit,
        "autonomy_mode": AutonomyMode.EXECUTE_NON_SPEND.value,
    }
    value["execution_profile_digest"] = canonical_digest(
        value,
        digest_field="execution_profile_digest",
    )
    return value


def capture_supervisor_health_status(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a non-secret, fail-closed view of capture-supervisor readiness."""

    status = {
        "agent_harness": "openai_agents_sdk",
        "configuration_status": "invalid",
        "zero_spend_lifecycle_ready": False,
        "live_inference_configured": False,
        "live_operator_gate_enabled": False,
        "live_inference_ready": False,
        "execution_profile_digest": None,
        "proof_or_recovery_authority_granted": False,
    }
    try:
        options = capture_supervisor_execution_options_from_env(environ)
        profile = capture_supervisor_execution_profile(
            **options,
            source_commit_sha=_capture_supervisor_source_commit_sha(environ),
        )
    except ValueError:
        return status
    live_configured = options["allow_live_agents_sdk"] is True
    live_gate = profile["live_operator_gate_enabled"] is True
    return status | {
        "configuration_status": "valid",
        "zero_spend_lifecycle_ready": True,
        "live_inference_configured": live_configured,
        "live_operator_gate_enabled": live_gate,
        "live_inference_ready": live_configured and live_gate,
        "execution_profile_digest": profile["execution_profile_digest"],
    }


def run_capture_build_supervisor(
    *,
    capture_root: str | Path,
    agent_model: str = DEFAULT_SUPERVISOR_AGENT_MODEL,
    allow_live_agents_sdk: bool = False,
    agent_inference_budget_usd: float = 0.0,
    source_commit_sha: str | None = None,
) -> dict[str, Any]:
    """Enter every completed capture build into the required supervisor.

    A capture is enough to create the run. Missing customer, task, embodiment,
    success, testbed, or rights details remain explicit blockers for the agents
    to clarify; they are never synthesized into authoritative facts.
    """

    source = Path(capture_root).expanduser().resolve()
    capture_build = load_capture_build_ingress(source)
    root = source if source.is_dir() else source.parent
    digest_suffix = str(capture_build["capture_build_digest"]).removeprefix("sha256:")[:24]
    execution_profile = capture_supervisor_execution_profile(
        agent_model=agent_model,
        allow_live_agents_sdk=allow_live_agents_sdk,
        agent_inference_budget_usd=agent_inference_budget_usd,
        source_commit_sha=source_commit_sha,
    )
    profile_suffix = str(execution_profile["execution_profile_digest"]).removeprefix("sha256:")[:16]
    run_id = f"capture-supervisor-v4-{digest_suffix}-{profile_suffix}"
    output_dir = root / "pipeline" / "task_evaluation_supervisor" / "runs" / run_id
    execution = TaskEvaluationSupervisor(
        agent_model=agent_model,
        allow_live_agents_sdk=allow_live_agents_sdk,
        agent_inference_budget_usd=agent_inference_budget_usd,
        non_spend_action_ttl_seconds=14_400,
    ).run(
        SupervisorContext(
            run_id=run_id,
            customer_question=(
                "What task evaluations can this completed capture build support, and what "
                "customer decision, robot embodiment, task, success, rights, testbed, or "
                "evidence details must be clarified before Blueprint can decide?"
            ),
            capture_build=capture_build,
        ),
        output_dir=output_dir,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        resume=True,
    )
    report = execution.report.to_mapping()
    route = build_capture_reconstruction_route(capture_build)
    readiness = build_reconstruction_execution_readiness(
        capture_build_value=capture_build,
        route_value=route,
        tool_registry_manifest=ToolRegistry.default().manifest(),
        bound_tool_ids=[],
        source_commit_sha=str(execution_profile["source_commit_sha"]),
        recorded_at=_readiness_timestamp(report["generated_at"]),
    )
    readiness_path = write_phase2_artifact(
        output_dir,
        "reconstruction_execution_readiness.json",
        readiness,
    )
    registered_capabilities = list(execution.run.to_mapping().get("capabilities") or [])
    return {
        "schema_version": CAPTURE_SUPERVISOR_LIFECYCLE_SCHEMA_VERSION,
        "status": report["status"],
        "run_id": run_id,
        "capture_build_digest": capture_build["capture_build_digest"],
        "supervisor_run_digest": execution.run.digest,
        "terminal_report_digest": execution.report.digest,
        "output_dir": str(output_dir),
        "terminal_report_path": str(output_dir / "terminal_supervisor_report.json"),
        "customer_report_path": str(output_dir / "customer_decision_report.json"),
        "event_ledger_path": str(output_dir / "supervisor_events.jsonl"),
        "agent_harness": "openai_agents_sdk",
        "agent_model": agent_model,
        "execution_profile": execution_profile,
        "execution_profile_digest": execution_profile["execution_profile_digest"],
        "source_commit_sha": execution_profile["source_commit_sha"],
        "reconstruction_execution_readiness_status": readiness["status"],
        "reconstruction_execution_readiness_digest": readiness[
            "reconstruction_execution_readiness_digest"
        ],
        "reconstruction_execution_readiness_path": str(readiness_path),
        "autonomy_mode": AutonomyMode.EXECUTE_NON_SPEND.value,
        "capability_count": len(execution.capability_results),
        "triggered_capability_count": len(execution.capability_results),
        "registered_capability_count": len(registered_capabilities),
        "all_six_capabilities_present": len(registered_capabilities) == 6,
        "all_six_capabilities_registered": len(registered_capabilities) == 6,
        "manager_invocation_count": int(
            (report.get("inference_spend") or {}).get("manager_invocation_count") or 0
        ),
        "agent_inference_started": (
            int((report.get("inference_spend") or {}).get("live_invocation_count") or 0) > 0
            or int((report.get("inference_spend") or {}).get("reservation_count") or 0) > 0
        ),
        "actions_executed": bool(report.get("actions_executed")),
        "registered_tool_reads_executed": int(report.get("registered_tool_reads_executed") or 0),
        "registered_non_spend_actions_executed": int(
            report.get("registered_non_spend_actions_executed") or 0
        ),
        "proof_state_mutated_by_agent": False,
        "capture_build_alone_can_start_run": True,
    }


def _write_reconstruction_readiness_snapshot(
    *,
    output_dir: Path,
    lifecycle: Mapping[str, Any],
    initial: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    readiness_digest = str(readiness["reconstruction_execution_readiness_digest"])
    relative_snapshot = (
        "reconstruction_execution_readiness_history/"
        f"{readiness_digest.removeprefix('sha256:')}.json"
    )
    snapshot_path = (output_dir / relative_snapshot).resolve()
    latest_path = output_dir / "reconstruction_execution_readiness_latest.json"
    lock_path = output_dir / ".reconstruction_execution_readiness.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if snapshot_path.is_file():
            existing = validate_reconstruction_execution_readiness(
                json.loads(snapshot_path.read_text(encoding="utf-8"))
            )
            if existing != readiness:
                raise ValueError("reconstruction_readiness_snapshot_conflict")
        else:
            write_phase2_artifact(output_dir, relative_snapshot, readiness)

        previous_digest: str | None = str(
            initial["reconstruction_execution_readiness_digest"]
        )
        if latest_path.is_file():
            previous = validate_reconstruction_readiness_pointer(
                json.loads(latest_path.read_text(encoding="utf-8"))
            )
            if previous.get("readiness_digest") == readiness_digest:
                return dict(previous) | {
                    "status": readiness["status"],
                    "snapshot_path": str(snapshot_path),
                    "latest_pointer_path": str(latest_path),
                    "already_exists": True,
                }
            previous_digest = str(previous.get("readiness_digest") or "") or None
        pointer = {
            "schema_version": RECONSTRUCTION_READINESS_POINTER_SCHEMA_VERSION,
            "run_id": lifecycle["run_id"],
            "capture_build_digest": lifecycle["capture_build_digest"],
            "source_commit_sha": lifecycle["source_commit_sha"],
            "readiness_digest": readiness_digest,
            "readiness_relative_path": relative_snapshot,
            "previous_readiness_digest": previous_digest,
            "control_plane_binding": readiness["control_plane_binding"],
            "recorded_at": readiness["recorded_at"],
            "proof_boundary": {
                "pointer_is_execution_authority": False,
                "pointer_is_reconstruction_evidence": False,
                "prior_readiness_snapshots_preserved": True,
                "physical_task_success_established": False,
            },
        }
        pointer["pointer_digest"] = canonical_digest(
            pointer, digest_field="pointer_digest"
        )
        pointer = validate_reconstruction_readiness_pointer(pointer)
        write_phase2_artifact(
            output_dir,
            "reconstruction_execution_readiness_latest.json",
            pointer,
        )
        return pointer | {
            "status": readiness["status"],
            "snapshot_path": str(snapshot_path),
            "latest_pointer_path": str(latest_path),
            "already_exists": False,
        }


def refresh_capture_reconstruction_execution_readiness(
    *,
    capture_root: str | Path,
    control_plane_inspection: Mapping[str, Any],
    agent_model: str = DEFAULT_SUPERVISOR_AGENT_MODEL,
    allow_live_agents_sdk: bool = False,
    agent_inference_budget_usd: float = 0.0,
    source_commit_sha: str | None = None,
) -> dict[str, Any]:
    """Append an immutable readiness snapshot for one control-plane state.

    The durable supervisor run remains the owner of the readiness history. A
    replaceable latest pointer is updated only after the content-addressed
    snapshot exists; prior snapshots are never rewritten or deleted.
    """

    lifecycle = run_capture_build_supervisor(
        capture_root=capture_root,
        agent_model=agent_model,
        allow_live_agents_sdk=allow_live_agents_sdk,
        agent_inference_budget_usd=agent_inference_budget_usd,
        source_commit_sha=source_commit_sha,
    )
    output_dir = Path(lifecycle["output_dir"]).resolve()
    initial_path = Path(lifecycle["reconstruction_execution_readiness_path"]).resolve()
    initial = validate_reconstruction_execution_readiness(
        json.loads(initial_path.read_text(encoding="utf-8"))
    )
    capture_build = load_capture_build_ingress(capture_root)
    route = build_capture_reconstruction_route(capture_build)
    readiness = build_reconstruction_execution_readiness(
        capture_build_value=capture_build,
        route_value=route,
        tool_registry_manifest=ToolRegistry.default().manifest(),
        bound_tool_ids=bound_tool_ids_for_control_plane_inspection(
            control_plane_inspection
        ),
        source_commit_sha=str(lifecycle["source_commit_sha"]),
        recorded_at=str(initial["recorded_at"]),
        control_plane_inspection=control_plane_inspection,
    )
    return _write_reconstruction_readiness_snapshot(
        output_dir=output_dir,
        lifecycle=lifecycle,
        initial=initial,
        readiness=readiness,
    )


def validate_reconstruction_supervisor_continuation(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one immutable child run in a capture supervisor lifecycle."""

    receipt = dict(value)
    required = {
        "schema_version", "continuation_run_id", "parent_capture_supervisor_run_id",
        "capture_build_digest", "source_commit_sha",
        "reconstruction_execution_readiness_digest", "context_binding_digest",
        "bound_tool_ids", "supervisor_run_digest", "terminal_report_digest",
        "status", "output_relative_path", "agent_harness", "autonomy_mode",
        "actions_executed", "registered_non_spend_actions_executed",
        "proof_boundary", "recorded_at",
        "reconstruction_supervisor_continuation_digest",
    }
    proof = receipt.get("proof_boundary")
    if (
        set(receipt) != required
        or receipt.get("schema_version")
        != RECONSTRUCTION_SUPERVISOR_CONTINUATION_SCHEMA_VERSION
        or not str(receipt.get("continuation_run_id") or "").strip()
        or not str(receipt.get("parent_capture_supervisor_run_id") or "").strip()
        or _SHA256_RE.fullmatch(str(receipt.get("capture_build_digest") or "")) is None
        or re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("source_commit_sha") or "")) is None
        or any(
            _SHA256_RE.fullmatch(str(receipt.get(key) or "")) is None
            for key in (
                "reconstruction_execution_readiness_digest", "context_binding_digest",
                "supervisor_run_digest", "terminal_report_digest",
            )
        )
        or not isinstance(receipt.get("bound_tool_ids"), list)
        or receipt.get("bound_tool_ids")
        != sorted(set(str(item) for item in receipt.get("bound_tool_ids") or []))
        or receipt.get("agent_harness") != "openai_agents_sdk"
        or receipt.get("autonomy_mode") != AutonomyMode.EXECUTE_NON_SPEND.value
        or not isinstance(receipt.get("actions_executed"), bool)
        or isinstance(receipt.get("registered_non_spend_actions_executed"), bool)
        or not isinstance(receipt.get("registered_non_spend_actions_executed"), int)
        or int(receipt.get("registered_non_spend_actions_executed") or 0) < 0
        or not str(receipt.get("output_relative_path") or "").startswith(
            "reconstruction_continuations/"
        )
        or not _readiness_timestamp(receipt.get("recorded_at"))
        or proof
        != {
            "continuation_is_execution_authority": False,
            "agent_can_change_proof_state": False,
            "agent_can_change_splits_or_calibration": False,
            "agent_can_grant_rights_or_budget": False,
            "raw_capture_remains_authoritative": True,
            "prior_supervisor_run_preserved": True,
            "physical_task_success_established": False,
        }
        or receipt.get("reconstruction_supervisor_continuation_digest")
        != canonical_digest(
            receipt,
            digest_field="reconstruction_supervisor_continuation_digest",
        )
    ):
        raise ValueError("reconstruction_supervisor_continuation_invalid")
    return receipt


def _reconstruction_context_binding_manifest(
    *,
    capture_build_digest: str,
    source_commit_sha: str,
    requested_tool_ids: Sequence[str],
    context_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    unknown = sorted(set(context_bindings) - _RECONSTRUCTION_CONTEXT_FIELDS)
    if unknown:
        raise ValueError(
            "reconstruction_continuation_context_field_forbidden:" + ",".join(unknown)
        )
    mapping_digests: dict[str, str] = {}
    runtime_bindings: dict[str, dict[str, str]] = {}
    for name, value in sorted(context_bindings.items()):
        if isinstance(value, Mapping):
            mapping_digests[name] = canonical_digest(dict(value))
        elif callable(value):
            target = getattr(value, "func", value)
            module = str(getattr(target, "__module__", type(target).__module__) or "")
            qualified_name = str(
                getattr(target, "__qualname__", type(target).__qualname__) or ""
            )
            if not module or not qualified_name:
                raise ValueError(
                    f"reconstruction_continuation_runtime_identity_missing:{name}"
                )
            explicit_digest = str(
                getattr(value, "blueprint_runtime_digest", "")
                or getattr(target, "blueprint_runtime_digest", "")
            )
            if explicit_digest and _SHA256_RE.fullmatch(explicit_digest) is None:
                raise ValueError(
                    f"reconstruction_continuation_runtime_digest_invalid:{name}"
                )
            if explicit_digest:
                implementation_digest = explicit_digest
            else:
                try:
                    source = inspect.getsource(target).encode("utf-8")
                except (OSError, TypeError):
                    code = getattr(target, "__code__", None)
                    if code is None:
                        raise ValueError(
                            f"reconstruction_continuation_runtime_digest_missing:{name}"
                        )
                    source = code.co_code + repr(code.co_consts).encode("utf-8")
                implementation_digest = "sha256:" + hashlib.sha256(source).hexdigest()
            runtime_bindings[name] = {
                "runtime_identity": f"{module}:{qualified_name}",
                "implementation_digest": implementation_digest,
            }
        else:
            raise ValueError(f"reconstruction_continuation_context_binding_invalid:{name}")
    requested = sorted(set(str(item).strip() for item in requested_tool_ids if str(item).strip()))
    if not requested or len(requested) != len(requested_tool_ids):
        raise ValueError("reconstruction_continuation_tool_ids_invalid")
    manifest = {
        "schema_version": "task_evaluation_reconstruction_context_binding.v1",
        "capture_build_digest": capture_build_digest,
        "source_commit_sha": source_commit_sha,
        "requested_tool_ids": requested,
        "mapping_binding_digests": mapping_digests,
        "trusted_runtime_bindings": runtime_bindings,
        "runtime_handles_serialized_to_agent": False,
        "proof_effect": "none",
    }
    manifest["context_binding_digest"] = canonical_digest(
        manifest, digest_field="context_binding_digest"
    )
    return manifest


def run_capture_reconstruction_supervisor_continuation(
    *,
    capture_root: str | Path,
    control_plane_inspection: Mapping[str, Any],
    requested_tool_ids: Sequence[str],
    context_bindings: Mapping[str, Any],
    agent_model: str = DEFAULT_SUPERVISOR_AGENT_MODEL,
    allow_live_agents_sdk: bool = False,
    agent_inference_budget_usd: float = 0.0,
    source_commit_sha: str | None = None,
) -> dict[str, Any]:
    """Run one linear, content-addressed reconstruction continuation."""

    lifecycle = run_capture_build_supervisor(
        capture_root=capture_root,
        agent_model=agent_model,
        allow_live_agents_sdk=allow_live_agents_sdk,
        agent_inference_budget_usd=agent_inference_budget_usd,
        source_commit_sha=source_commit_sha,
    )
    capture_build = load_capture_build_ingress(capture_root)
    registry = ToolRegistry.default()
    binding_manifest = _reconstruction_context_binding_manifest(
        capture_build_digest=capture_build["capture_build_digest"],
        source_commit_sha=str(lifecycle["source_commit_sha"]),
        requested_tool_ids=requested_tool_ids,
        context_bindings=context_bindings,
    )
    digest_suffix = binding_manifest["context_binding_digest"][7:31]
    continuation_run_id = f"{lifecycle['run_id']}-reconstruction-{digest_suffix}"
    parent_output = Path(lifecycle["output_dir"]).resolve()
    output_dir = parent_output / "reconstruction_continuations" / continuation_run_id
    context = SupervisorContext(
        run_id=continuation_run_id,
        customer_question=(
            "Continue the parent capture reconstruction from its immutable readiness state. "
            "Inspect registered typed tools, execute only legal bounded actions, observe typed "
            "results, diagnose failures, and propose the next experiment or abstention."
        ),
        capture_build=capture_build,
        supervisor_output_dir=str(output_dir),
        **dict(context_bindings),
    )
    authority = default_authority_envelope(
        run_id=continuation_run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[
            capture_build["capture_build_digest"],
            binding_manifest["context_binding_digest"],
            *binding_manifest["mapping_binding_digests"].values(),
        ],
        agent_inference_budget_usd=agent_inference_budget_usd,
        allow_agent_inference=allow_live_agents_sdk,
        action_ttl_seconds=14_400,
    ).to_mapping()
    available = {
        binding.tool_id
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
    }
    requested = set(binding_manifest["requested_tool_ids"])
    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            "reconstruction_continuation_runtime_binding_missing:" + ",".join(missing)
        )

    initial_path = Path(lifecycle["reconstruction_execution_readiness_path"]).resolve()
    initial = validate_reconstruction_execution_readiness(
        json.loads(initial_path.read_text(encoding="utf-8"))
    )
    readiness = build_reconstruction_execution_readiness(
        capture_build_value=capture_build,
        route_value=build_capture_reconstruction_route(capture_build),
        tool_registry_manifest=registry.manifest(),
        bound_tool_ids=sorted(
            requested
            | set(bound_tool_ids_for_control_plane_inspection(control_plane_inspection))
        ),
        source_commit_sha=str(lifecycle["source_commit_sha"]),
        recorded_at=str(initial["recorded_at"]),
        control_plane_inspection=control_plane_inspection,
    )
    requested_stages = {
        row["stage_id"]: row
        for row in readiness["stages"]
        if row["stage_id"] in requested
    }
    # Fresh-scene preparation tools are support producers layered on the
    # capture route, not reconstruction-method stages selected by that route.
    # Their runtime availability was already established above from the
    # registered, digest-bound context. Do not require them to masquerade as a
    # reconstruction stage merely to pass readiness.
    route_independent_tools = {
        "inspect_fresh_scene_preparation",
        "materialize_sam31_task_inputs",
        "materialize_calibrated_object_masks",
        "materialize_fresh_scene_removal_freezes",
        "materialize_fresh_scene_segment_cutout",
        "materialize_fresh_scene_artifixer_candidate",
        "materialize_fresh_scene_semantic_teacher_edit_packet",
    }
    not_ready = sorted(
        tool_id
        for tool_id in requested
        if tool_id not in route_independent_tools
        and (tool_id not in requested_stages
        or requested_stages[tool_id]["readiness_status"]
        not in {"ready_for_bounded_tool_call", "recorded_support_completed"})
    )
    if not_ready:
        raise ValueError(
            "reconstruction_continuation_readiness_missing:" + ",".join(not_ready)
        )
    pointer = _write_reconstruction_readiness_snapshot(
        output_dir=parent_output,
        lifecycle=lifecycle,
        initial=initial,
        readiness=readiness,
    )
    execution = TaskEvaluationSupervisor(
        agent_model=agent_model,
        allow_live_agents_sdk=allow_live_agents_sdk,
        agent_inference_budget_usd=agent_inference_budget_usd,
    ).run(
        context,
        output_dir=output_dir,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        resume=True,
    )
    report = execution.report.to_mapping()
    receipt = {
        "schema_version": RECONSTRUCTION_SUPERVISOR_CONTINUATION_SCHEMA_VERSION,
        "continuation_run_id": continuation_run_id,
        "parent_capture_supervisor_run_id": lifecycle["run_id"],
        "capture_build_digest": capture_build["capture_build_digest"],
        "source_commit_sha": lifecycle["source_commit_sha"],
        "reconstruction_execution_readiness_digest": pointer["readiness_digest"],
        "context_binding_digest": binding_manifest["context_binding_digest"],
        "bound_tool_ids": binding_manifest["requested_tool_ids"],
        "supervisor_run_digest": execution.run.digest,
        "terminal_report_digest": execution.report.digest,
        "status": report["status"],
        "output_relative_path": output_dir.relative_to(parent_output).as_posix(),
        "agent_harness": "openai_agents_sdk",
        "autonomy_mode": AutonomyMode.EXECUTE_NON_SPEND.value,
        "actions_executed": bool(report.get("actions_executed")),
        "registered_non_spend_actions_executed": int(
            report.get("registered_non_spend_actions_executed") or 0
        ),
        "proof_boundary": {
            "continuation_is_execution_authority": False,
            "agent_can_change_proof_state": False,
            "agent_can_change_splits_or_calibration": False,
            "agent_can_grant_rights_or_budget": False,
            "raw_capture_remains_authoritative": True,
            "prior_supervisor_run_preserved": True,
            "physical_task_success_established": False,
        },
        "recorded_at": _readiness_timestamp(report["generated_at"]),
    }
    receipt["reconstruction_supervisor_continuation_digest"] = canonical_digest(
        receipt,
        digest_field="reconstruction_supervisor_continuation_digest",
    )
    receipt = validate_reconstruction_supervisor_continuation(receipt)
    write_phase2_artifact(
        output_dir,
        "reconstruction_supervisor_continuation.json",
        receipt,
    )
    return receipt


__all__ = [
    "CAPTURE_SUPERVISOR_AGENT_MODEL_ENV",
    "CAPTURE_SUPERVISOR_ALLOW_LIVE_AGENTS_SDK_ENV",
    "CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD_ENV",
    "CAPTURE_SUPERVISOR_LIFECYCLE_SCHEMA_VERSION",
    "MAX_CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD",
    "RECONSTRUCTION_READINESS_POINTER_SCHEMA_VERSION",
    "RECONSTRUCTION_SUPERVISOR_CONTINUATION_SCHEMA_VERSION",
    "capture_supervisor_execution_options_from_env",
    "capture_supervisor_health_status",
    "capture_supervisor_execution_profile",
    "run_capture_build_supervisor",
    "run_capture_reconstruction_supervisor_continuation",
    "refresh_capture_reconstruction_execution_readiness",
    "validate_reconstruction_readiness_pointer",
    "validate_reconstruction_supervisor_continuation",
]
