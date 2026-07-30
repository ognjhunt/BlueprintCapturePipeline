"""Required capture-build entrypoint for the Task Evaluation Supervisor."""

from __future__ import annotations

import fcntl
import math
import os
import json
from pathlib import Path
import re
import subprocess
from subprocess import run as _run_subprocess
from typing import Any, Mapping

from ..agent_operator_runtime import LIVE_AGENTS_SDK_ENV, env_truthy
from ..decision_evidence_contracts import canonical_digest
from .agents_sdk import DEFAULT_SUPERVISOR_AGENT_MODEL
from .capabilities import SupervisorContext
from .capture_ingress import load_capture_build_ingress
from .capture_reconstruction_routing import build_capture_reconstruction_route
from .contracts import AutonomyMode
from .supervisor import TaskEvaluationSupervisor
from .reconstruction_execution_readiness import (
    bound_tool_ids_for_control_plane_inspection,
    build_reconstruction_execution_readiness,
    validate_reconstruction_execution_readiness,
)
from .tools import ToolRegistry
from .phase2_artifacts import write_phase2_artifact


CAPTURE_SUPERVISOR_LIFECYCLE_SCHEMA_VERSION = "task_evaluation_capture_supervisor_lifecycle.v4"
CAPTURE_SUPERVISOR_ALLOW_LIVE_AGENTS_SDK_ENV = "BLUEPRINT_CAPTURE_SUPERVISOR_ALLOW_LIVE_AGENTS_SDK"
CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD_ENV = "BLUEPRINT_CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD"
CAPTURE_SUPERVISOR_AGENT_MODEL_ENV = "BLUEPRINT_CAPTURE_SUPERVISOR_AGENT_MODEL"
MAX_CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD = 100.0
RECONSTRUCTION_READINESS_POINTER_SCHEMA_VERSION = (
    "task_evaluation_reconstruction_readiness_pointer.v1"
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


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


__all__ = [
    "CAPTURE_SUPERVISOR_AGENT_MODEL_ENV",
    "CAPTURE_SUPERVISOR_ALLOW_LIVE_AGENTS_SDK_ENV",
    "CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD_ENV",
    "CAPTURE_SUPERVISOR_LIFECYCLE_SCHEMA_VERSION",
    "MAX_CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD",
    "RECONSTRUCTION_READINESS_POINTER_SCHEMA_VERSION",
    "capture_supervisor_execution_options_from_env",
    "capture_supervisor_health_status",
    "capture_supervisor_execution_profile",
    "run_capture_build_supervisor",
    "refresh_capture_reconstruction_execution_readiness",
    "validate_reconstruction_readiness_pointer",
]
