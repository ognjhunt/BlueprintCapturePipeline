"""Always-on live pipeline control-plane runner.

The control plane is intentionally thin. It audits local/live readiness, then
optionally consumes a WebApp-style robot-eval job request inbox through the
existing deterministic orchestrator. It does not promote proof claims and it
does not turn on simulator, vision-labeling, delivery, or live agent operators
unless the caller supplies the matching CLI and environment gates.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .agent_operator_runtime import LIVE_AGENTS_SDK_ENV, LIVE_CODEX_SDK_ENV
from .common import ensure_dir, utc_now_iso, write_json
from .live_pipeline_setup import (
    CONTROL_PLANE_NOT_PROOF,
    build_live_pipeline_setup_manifest,
)
from .robot_eval_job_orchestrator import (
    CPU_BACKENDS,
    CLAIM_BOUNDARY,
    AgentsSdkRobotEvalJobAdapter,
    FakeRobotEvalJobAgentAdapter,
    RobotEvalJobAgentAdapter,
    run_robot_eval_job_request_inbox,
)
from .safe_env import load_env_files


LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION = "blueprint_live_pipeline_control_plane_run.v1"

CAPTURE_ROOT_ENV = "BLUEPRINT_PIPELINE_CAPTURE_ROOT"
JOB_REQUEST_INBOX_ENV = "BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX"
PACKAGE_DIR_ENV = "BLUEPRINT_PIPELINE_PACKAGE_DIR"
ARENA_RESULTS_DIR_ENV = "BLUEPRINT_ARENA_RESULTS_DIR"
SIMULATOR_AUDIT_COMMAND_ENV = "BLUEPRINT_SIMULATOR_COMMAND"
VISION_LABELING_COMMAND_ENV = "BLUEPRINT_ROLLOUT_VISION_LABELING_COMMAND"
DELIVERY_COMMAND_ENV = "BLUEPRINT_PACKAGE_DELIVERY_UPLOAD_COMMAND"
CONTROL_PLANE_AGENT_MODE_ENV = "BLUEPRINT_CONTROL_PLANE_AGENT_MODE"
CONTROL_PLANE_ARENA_OPERATOR_MODE_ENV = "BLUEPRINT_CONTROL_PLANE_ARENA_OPERATOR_MODE"
CONTROL_PLANE_ALLOW_LIVE_AGENT_OPERATOR_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_LIVE_AGENT_OPERATOR"
CONTROL_PLANE_ALLOW_DIGITALOCEAN_READ_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_DIGITALOCEAN_READ"
CONTROL_PLANE_ALLOW_GPU_PROVISIONING_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_GPU_PROVISIONING"
CONTROL_PLANE_ALLOW_SIMULATOR_EXECUTION_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_SIMULATOR_EXECUTION"
CONTROL_PLANE_ALLOW_CPU_PREFLIGHT_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_CPU_PREFLIGHT"
CONTROL_PLANE_ALLOW_CPU_PREFLIGHT_RENDER_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_CPU_PREFLIGHT_RENDER"
CONTROL_PLANE_ALLOW_TRAINING_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_TRAINING"
CONTROL_PLANE_ALLOW_ROLLOUT_VISION_LABELING_ENV = (
    "BLUEPRINT_CONTROL_PLANE_ALLOW_ROLLOUT_VISION_LABELING"
)
CONTROL_PLANE_ALLOW_DELIVERY_UPLOAD_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_DELIVERY_UPLOAD"
CONTROL_PLANE_ALLOW_LIVE_AGENTS_SDK_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_LIVE_AGENTS_SDK"
CONTROL_PLANE_ALLOW_LIVE_CODEX_SDK_ENV = "BLUEPRINT_CONTROL_PLANE_ALLOW_LIVE_CODEX_SDK"
CONTROL_PLANE_SIMULATOR_ENV = "BLUEPRINT_CONTROL_PLANE_SIMULATOR"
CONTROL_PLANE_PROVISIONER_ENV = "BLUEPRINT_CONTROL_PLANE_PROVISIONER"
CONTROL_PLANE_TIMEOUT_SECONDS_ENV = "BLUEPRINT_CONTROL_PLANE_TIMEOUT_SECONDS"
ISAAC_LAB_ARENA_COMMAND_ENV = "BLUEPRINT_ISAAC_LAB_ARENA_COMMAND"
DIGITALOCEAN_DROPLET_NAME_ENV = "BLUEPRINT_DIGITALOCEAN_DROPLET_NAME"
DIGITALOCEAN_DROPLET_IP_ENV = "BLUEPRINT_DIGITALOCEAN_DROPLET_IP"

SECRET_ENV_NAMES = (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "WORLDLABS_API_KEY",
    "PIPELINE_SYNC_TOKEN",
    "DIGITALOCEAN_ACCESS_TOKEN",
)


def _string(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    return _truthy(os.getenv(name))


def _env_value(name: str, explicit: str | Path | None = None) -> str | None:
    value = _string(explicit)
    if value:
        return value
    env_value = _string(os.getenv(name))
    return env_value or None


def _env_int(name: str, default: int) -> int:
    value = _string(os.getenv(name))
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _restore_env(original_env: Mapping[str, str]) -> None:
    for key in list(os.environ):
        if key not in original_env:
            os.environ.pop(key, None)
    for key, value in original_env.items():
        os.environ[key] = value


def _unique_paths(paths: Sequence[Path]) -> List[Path]:
    unique: List[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _output_path(capture_root: Path | None, output_path: str | Path | None) -> Path:
    if output_path:
        return Path(output_path).resolve()
    if capture_root:
        return (
            capture_root
            / "pipeline"
            / "live_pipeline_control_plane"
            / "live_pipeline_control_plane_manifest.json"
        )
    return Path.cwd().resolve() / "live_pipeline_control_plane_manifest.json"


def _agent_adapter_from_mode(mode: str, *, allow_live_operator: bool) -> RobotEvalJobAgentAdapter | None:
    if mode == "fake":
        return FakeRobotEvalJobAgentAdapter()
    if mode == "agents-sdk":
        return AgentsSdkRobotEvalJobAdapter(allow_live_operator=allow_live_operator)
    return None


def _parse_simulator_commands(values: Sequence[str] | None) -> Dict[str, str]:
    commands: Dict[str, str] = {}
    for value in values or []:
        text = _string(value)
        if not text:
            continue
        framework, sep, command = text.partition("=")
        framework = framework.strip()
        command = command.strip()
        if not sep or not framework or not command:
            raise ValueError("simulator commands must be formatted as <framework>=<command>")
        commands[framework] = command
    env_command = _string(os.getenv(ISAAC_LAB_ARENA_COMMAND_ENV))
    if env_command and "isaac_lab_arena" not in commands:
        commands["isaac_lab_arena"] = env_command
    return commands


def _secret_values() -> List[str]:
    values: List[str] = []
    for name in SECRET_ENV_NAMES:
        value = _string(os.getenv(name))
        if len(value) >= 8 and value.lower() not in {"placeholder", "changeme", "example"}:
            values.append(value)
    return values


def _manifest_leaks_secret(manifest: Mapping[str, Any], secret_values: Sequence[str]) -> bool:
    if not secret_values:
        return False
    serialized = json.dumps(manifest, sort_keys=True)
    return any(value in serialized for value in secret_values)


def _inbox_status_not_configured(reason: str) -> Dict[str, Any]:
    return {
        "status": "not_configured",
        "processed": False,
        "processed_count": 0,
        "blockers": [reason],
        "manifest_path": None,
    }


def _overall_status(
    *,
    capture_root: Path | None,
    inbox: Mapping[str, Any],
    setup_manifest: Mapping[str, Any],
) -> str:
    if capture_root is None:
        return "blocked"
    if inbox.get("status") == "completed":
        return "processed_jobs"
    if inbox.get("status") == "empty":
        return "waiting_for_jobs"
    if setup_manifest.get("status") == "ready_for_live_external_execution":
        return "ready_for_live_external_execution"
    if setup_manifest.get("status") == "local_ready_live_external_blocked":
        return "local_ready_live_external_blocked"
    return "blocked"


def run_live_pipeline_control_plane(
    *,
    capture_root: str | Path | None = None,
    job_request_inbox: str | Path | None = None,
    package_dir: str | Path | None = None,
    arena_results_dir: str | Path | None = None,
    simulator_audit_command: str | None = None,
    vision_labeling_command: str | None = None,
    delivery_command: str | None = None,
    process_inbox: bool = True,
    load_local_env: bool = True,
    allow_digitalocean_read: bool | None = None,
    digitalocean_token_env: str = "DIGITALOCEAN_ACCESS_TOKEN",
    digitalocean_droplet_name: str | None = None,
    digitalocean_droplet_ip: str | None = None,
    agent_mode: str | None = None,
    allow_live_agent_operator: bool | None = None,
    provisioner: str | None = None,
    simulator: str | None = None,
    allow_gpu_provisioning: bool | None = None,
    allow_simulator_execution: bool | None = None,
    allowed_simulators: Sequence[str] = (),
    simulator_commands: Sequence[str] = (),
    allow_cpu_simulator_preflight: bool | None = None,
    cpu_preflight_backends: Sequence[str] = CPU_BACKENDS,
    cpu_preflight_smoke_steps: int = 10,
    allow_cpu_preflight_render: bool | None = None,
    allow_training: bool | None = None,
    training_command: str | None = None,
    timeout_seconds: int | None = None,
    budget_usd: float | None = None,
    arena_scenario_count: int = 500,
    arena_shard_size: int = 50,
    arena_num_envs: int = 16,
    arena_retry_budget: int = 2,
    allow_rollout_vision_labeling: bool | None = None,
    allow_delivery_upload: bool | None = None,
    arena_operator_mode: str | None = None,
    allow_live_agents_sdk: bool | None = None,
    allow_live_codex_sdk: bool | None = None,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    original_env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        initial_capture_text = _env_value(CAPTURE_ROOT_ENV, capture_root)
        initial_capture_path = Path(initial_capture_text).resolve() if initial_capture_text else None
        env_roots = _unique_paths(
            [repo_root, Path.cwd(), initial_capture_path]
            if initial_capture_path
            else [repo_root, Path.cwd()]
        )
        env_summary = (
            load_env_files(env_roots)
            if load_local_env
            else {
                "files": [],
                "loaded_keys": [],
                "skipped_existing_keys": [],
                "skipped_placeholder_keys": [],
            }
        )
        capture_text = _env_value(CAPTURE_ROOT_ENV, capture_root)
        capture_path = Path(capture_text).resolve() if capture_text else None
        if load_local_env and capture_path and capture_path.resolve() not in set(env_roots):
            capture_env_summary = load_env_files([capture_path])
            env_summary = {
                "files": env_summary["files"] + capture_env_summary["files"],
                "loaded_keys": sorted(
                    set(env_summary["loaded_keys"]) | set(capture_env_summary["loaded_keys"])
                ),
                "skipped_existing_keys": sorted(
                    set(env_summary["skipped_existing_keys"])
                    | set(capture_env_summary["skipped_existing_keys"])
                ),
                "skipped_placeholder_keys": sorted(
                    set(env_summary["skipped_placeholder_keys"])
                    | set(capture_env_summary["skipped_placeholder_keys"])
                ),
            }
        inbox_text = _env_value(JOB_REQUEST_INBOX_ENV, job_request_inbox)
        inbox_path = Path(inbox_text).resolve() if inbox_text else None
        package_text = _env_value(PACKAGE_DIR_ENV, package_dir)
        package_path = Path(package_text).resolve() if package_text else None
        arena_results_text = _env_value(ARENA_RESULTS_DIR_ENV, arena_results_dir)
        arena_results_path = Path(arena_results_text).resolve() if arena_results_text else None
        output = _output_path(capture_path, output_path)
        secret_values = _secret_values()

        resolved_agent_mode = _string(agent_mode or os.getenv(CONTROL_PLANE_AGENT_MODE_ENV)) or "none"
        resolved_arena_operator_mode = (
            _string(arena_operator_mode or os.getenv(CONTROL_PLANE_ARENA_OPERATOR_MODE_ENV))
            or "none"
        )
        resolved_provisioner = (
            _string(provisioner or os.getenv(CONTROL_PLANE_PROVISIONER_ENV)) or "fixture_local"
        )
        resolved_simulator = _string(simulator or os.getenv(CONTROL_PLANE_SIMULATOR_ENV)) or "fixture"
        resolved_timeout = (
            int(timeout_seconds)
            if timeout_seconds is not None
            else _env_int(CONTROL_PLANE_TIMEOUT_SECONDS_ENV, 120)
        )
        resolved_simulator_audit_command = (
            _string(simulator_audit_command or os.getenv(SIMULATOR_AUDIT_COMMAND_ENV)) or None
        )
        resolved_vision_command = (
            _string(vision_labeling_command or os.getenv(VISION_LABELING_COMMAND_ENV)) or None
        )
        resolved_delivery_command = _string(delivery_command or os.getenv(DELIVERY_COMMAND_ENV)) or None
        digitalocean_read_allowed = (
            bool(allow_digitalocean_read)
            if allow_digitalocean_read is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_DIGITALOCEAN_READ_ENV)
        )
        resolved_digitalocean_name = (
            _string(digitalocean_droplet_name or os.getenv(DIGITALOCEAN_DROPLET_NAME_ENV)) or None
        )
        resolved_digitalocean_ip = (
            _string(digitalocean_droplet_ip or os.getenv(DIGITALOCEAN_DROPLET_IP_ENV)) or None
        )
        live_agent_operator_allowed = (
            bool(allow_live_agent_operator)
            if allow_live_agent_operator is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_LIVE_AGENT_OPERATOR_ENV)
        )
        gpu_allowed = (
            bool(allow_gpu_provisioning)
            if allow_gpu_provisioning is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_GPU_PROVISIONING_ENV)
        )
        simulator_execution_allowed = (
            bool(allow_simulator_execution)
            if allow_simulator_execution is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_SIMULATOR_EXECUTION_ENV)
        )
        cpu_preflight_allowed = (
            bool(allow_cpu_simulator_preflight)
            if allow_cpu_simulator_preflight is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_CPU_PREFLIGHT_ENV)
        )
        cpu_preflight_render_allowed = (
            bool(allow_cpu_preflight_render)
            if allow_cpu_preflight_render is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_CPU_PREFLIGHT_RENDER_ENV)
        )
        training_allowed = (
            bool(allow_training)
            if allow_training is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_TRAINING_ENV)
        )
        vision_allowed = (
            bool(allow_rollout_vision_labeling)
            if allow_rollout_vision_labeling is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_ROLLOUT_VISION_LABELING_ENV)
        )
        delivery_allowed = (
            bool(allow_delivery_upload)
            if allow_delivery_upload is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_DELIVERY_UPLOAD_ENV)
        )
        live_agents_allowed = (
            bool(allow_live_agents_sdk)
            if allow_live_agents_sdk is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_LIVE_AGENTS_SDK_ENV)
            or _env_truthy(LIVE_AGENTS_SDK_ENV)
        )
        live_codex_allowed = (
            bool(allow_live_codex_sdk)
            if allow_live_codex_sdk is not None
            else _env_truthy(CONTROL_PLANE_ALLOW_LIVE_CODEX_SDK_ENV)
            or _env_truthy(LIVE_CODEX_SDK_ENV)
        )
        parsed_simulator_commands = _parse_simulator_commands(simulator_commands)

        setup_output = (
            capture_path / "pipeline" / "live_pipeline_setup" / "live_pipeline_setup_manifest.json"
            if capture_path
            else output.parent / "live_pipeline_setup_manifest.json"
        )
        try:
            setup_manifest = build_live_pipeline_setup_manifest(
                capture_root=capture_path,
                package_dir=package_path,
                arena_results_dir=arena_results_path,
                simulator_command=resolved_simulator_audit_command,
                vision_labeling_command=resolved_vision_command,
                delivery_command=resolved_delivery_command,
                load_local_env=False,
                allow_digitalocean_read=digitalocean_read_allowed,
                digitalocean_token_env=digitalocean_token_env,
                digitalocean_droplet_name=resolved_digitalocean_name,
                digitalocean_droplet_ip=resolved_digitalocean_ip,
                output_path=setup_output,
                timeout_seconds=min(resolved_timeout, 30),
            )
        except Exception as exc:  # pragma: no cover - exact exception varies by bad path
            setup_manifest = {
                "schema_version": "blueprint_live_pipeline_setup_blocked.v1",
                "generated_at": utc_now_iso(),
                "status": "blocked",
                "capture_root": str(capture_path) if capture_path else None,
                "blockers": [f"setup_audit_failed:{type(exc).__name__}"],
                "error_type": type(exc).__name__,
            }
            ensure_dir(setup_output.parent)
            write_json(setup_output, setup_manifest)

        inbox_run: Dict[str, Any]
        if not process_inbox:
            inbox_run = _inbox_status_not_configured("inbox_processing_disabled")
        elif capture_path is None:
            inbox_run = _inbox_status_not_configured("missing_capture_root")
        elif inbox_path is None:
            inbox_run = _inbox_status_not_configured("missing_job_request_inbox")
        else:
            ensure_dir(inbox_path)
            try:
                inbox_result = run_robot_eval_job_request_inbox(
                    capture_root=capture_path,
                    inbox_dir=inbox_path,
                    agent_adapter=_agent_adapter_from_mode(
                        resolved_agent_mode,
                        allow_live_operator=live_agent_operator_allowed,
                    ),
                    provisioner=resolved_provisioner,
                    simulator=resolved_simulator,
                    allow_gpu_provisioning=gpu_allowed,
                    allow_simulator_execution=simulator_execution_allowed,
                    allowed_simulators=allowed_simulators,
                    simulator_commands=parsed_simulator_commands,
                    allow_cpu_simulator_preflight=cpu_preflight_allowed,
                    cpu_preflight_backends=cpu_preflight_backends,
                    cpu_preflight_smoke_steps=cpu_preflight_smoke_steps,
                    allow_cpu_preflight_render=cpu_preflight_render_allowed,
                    allow_training=training_allowed,
                    training_command=training_command,
                    timeout_seconds=resolved_timeout,
                    budget_usd=budget_usd,
                    arena_results_dir=arena_results_path,
                    arena_scenario_count=arena_scenario_count,
                    arena_shard_size=arena_shard_size,
                    arena_num_envs=arena_num_envs,
                    arena_retry_budget=arena_retry_budget,
                    allow_rollout_vision_labeling=vision_allowed,
                    vision_labeling_command=resolved_vision_command,
                    allow_delivery_upload=delivery_allowed,
                    delivery_command=resolved_delivery_command,
                    arena_operator_mode=resolved_arena_operator_mode,
                    allow_live_agents_sdk=live_agents_allowed,
                    allow_live_codex_sdk=live_codex_allowed,
                )
                inbox_run = {
                    **inbox_result,
                    "processed": True,
                    "manifest_path": str(
                        capture_path
                        / "pipeline"
                        / "robot_eval_job_requests"
                        / "inbox_run_manifest.json"
                    ),
                    "blockers": [],
                }
            except Exception as exc:  # pragma: no cover - exact exception varies by bad path
                inbox_run = {
                    "status": "blocked",
                    "processed": False,
                    "processed_count": 0,
                    "blockers": [f"inbox_run_failed:{type(exc).__name__}"],
                    "error_type": type(exc).__name__,
                    "manifest_path": None,
                }

        blockers: List[str] = []
        if capture_path is None:
            blockers.append("missing_capture_root")
        for blocker in inbox_run.get("blockers") or []:
            blockers.append(f"inbox:{blocker}")

        manifest: Dict[str, Any] = {
            "schema_version": LIVE_PIPELINE_CONTROL_PLANE_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": _overall_status(
                capture_root=capture_path,
                inbox=inbox_run,
                setup_manifest=setup_manifest,
            ),
            "capture_root": str(capture_path) if capture_path else None,
            "job_request_inbox": str(inbox_path) if inbox_path else None,
            "output_path": str(output),
            "env_files": env_summary,
            "setup_manifest_path": str(setup_output),
            "setup_status": setup_manifest.get("status"),
            "setup_blockers": setup_manifest.get("blockers", []),
            "inbox_run": inbox_run,
            "operator_config": {
                "agent_mode": resolved_agent_mode,
                "arena_operator_mode": resolved_arena_operator_mode,
                "live_agent_operator_allowed_by_control_plane": live_agent_operator_allowed,
                "live_agents_sdk_allowed_by_control_plane": live_agents_allowed,
                "live_codex_sdk_allowed_by_control_plane": live_codex_allowed,
            },
            "digitalocean_config": {
                "read_allowed_by_control_plane": digitalocean_read_allowed,
                "droplet_name": resolved_digitalocean_name,
                "droplet_ip": resolved_digitalocean_ip,
                "token_env": digitalocean_token_env,
            },
            "execution_config": {
                "provisioner": resolved_provisioner,
                "simulator": resolved_simulator,
                "allowed_simulators": list(allowed_simulators),
                "simulator_commands_configured": sorted(parsed_simulator_commands),
                "allow_gpu_provisioning": gpu_allowed,
                "allow_simulator_execution": simulator_execution_allowed,
                "allow_cpu_simulator_preflight": cpu_preflight_allowed,
                "allow_cpu_preflight_render": cpu_preflight_render_allowed,
                "allow_training": training_allowed,
                "allow_rollout_vision_labeling": vision_allowed,
                "allow_delivery_upload": delivery_allowed,
                "arena_scenario_count": arena_scenario_count,
                "arena_shard_size": arena_shard_size,
                "arena_num_envs": arena_num_envs,
                "arena_retry_budget": arena_retry_budget,
                "timeout_seconds": resolved_timeout,
            },
            "control_plane_boundary": {
                **CONTROL_PLANE_NOT_PROOF,
                "public_claim_upgrade_allowed": False,
                "proof_boundary_authority": "deterministic_artifacts_and_owner_system_evidence",
            },
            "claim_boundary": dict(CLAIM_BOUNDARY),
            "blockers": blockers,
            "next_inputs_needed": [
                "Set BLUEPRINT_PIPELINE_CAPTURE_ROOT to a real capture root.",
                "Set BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX to the WebApp job request inbox path.",
                "Provide a real owner-system Isaac Lab-Arena command or result directory before "
                "claiming simulator execution.",
                "Provide a vision-labeling command and gate before model labels can be generated.",
                "Provide a delivery command and gate before package uploads or signed links can be "
                "created.",
            ],
        }
        manifest["secrets_leaked"] = _manifest_leaks_secret(manifest, secret_values)
        ensure_dir(output.parent)
        write_json(output, manifest)
        return manifest
    finally:
        _restore_env(original_env)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Blueprint live pipeline control plane once: audit readiness and optionally "
            "consume a robot-eval job request inbox."
        )
    )
    parser.add_argument("--capture-root")
    parser.add_argument("--job-request-inbox")
    parser.add_argument("--package-dir")
    parser.add_argument("--arena-results-dir")
    parser.add_argument("--simulator-audit-command")
    parser.add_argument("--vision-labeling-command")
    parser.add_argument("--delivery-command")
    parser.add_argument("--no-process-inbox", action="store_true")
    parser.add_argument("--no-load-env-files", action="store_true")
    parser.add_argument("--allow-digitalocean-read", action="store_true", default=None)
    parser.add_argument("--digitalocean-token-env", default="DIGITALOCEAN_ACCESS_TOKEN")
    parser.add_argument("--digitalocean-droplet-name")
    parser.add_argument("--digitalocean-droplet-ip")
    parser.add_argument("--agent-mode", choices=("none", "fake", "agents-sdk"), default=None)
    parser.add_argument("--allow-live-agent-operator", action="store_true", default=None)
    parser.add_argument("--provisioner", default=None)
    parser.add_argument("--simulator", default=None)
    parser.add_argument("--allow-gpu-provisioning", action="store_true", default=None)
    parser.add_argument("--allow-simulator-execution", action="store_true", default=None)
    parser.add_argument("--allow-simulator", action="append", default=[])
    parser.add_argument("--simulator-command", action="append", default=[])
    parser.add_argument("--allow-cpu-simulator-preflight", action="store_true", default=None)
    parser.add_argument("--cpu-preflight-backend", action="append", default=[])
    parser.add_argument("--cpu-preflight-smoke-steps", type=int, default=10)
    parser.add_argument("--allow-cpu-preflight-render", action="store_true", default=None)
    parser.add_argument("--allow-training", action="store_true", default=None)
    parser.add_argument("--training-command")
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument("--arena-scenario-count", type=int, default=500)
    parser.add_argument("--arena-shard-size", type=int, default=50)
    parser.add_argument("--arena-num-envs", type=int, default=16)
    parser.add_argument("--arena-retry-budget", type=int, default=2)
    parser.add_argument("--allow-rollout-vision-labeling", action="store_true", default=None)
    parser.add_argument("--allow-delivery-upload", action="store_true", default=None)
    parser.add_argument("--arena-operator-mode", choices=("none", "fake", "agents-sdk"), default=None)
    parser.add_argument("--allow-live-agents-sdk", action="store_true", default=None)
    parser.add_argument("--allow-live-codex-sdk", action="store_true", default=None)
    parser.add_argument("--output-path")
    args = parser.parse_args(argv)
    result = run_live_pipeline_control_plane(
        capture_root=args.capture_root,
        job_request_inbox=args.job_request_inbox,
        package_dir=args.package_dir,
        arena_results_dir=args.arena_results_dir,
        simulator_audit_command=args.simulator_audit_command,
        vision_labeling_command=args.vision_labeling_command,
        delivery_command=args.delivery_command,
        process_inbox=not args.no_process_inbox,
        load_local_env=not args.no_load_env_files,
        allow_digitalocean_read=args.allow_digitalocean_read,
        digitalocean_token_env=args.digitalocean_token_env,
        digitalocean_droplet_name=args.digitalocean_droplet_name,
        digitalocean_droplet_ip=args.digitalocean_droplet_ip,
        agent_mode=args.agent_mode,
        allow_live_agent_operator=args.allow_live_agent_operator,
        provisioner=args.provisioner,
        simulator=args.simulator,
        allow_gpu_provisioning=args.allow_gpu_provisioning,
        allow_simulator_execution=args.allow_simulator_execution,
        allowed_simulators=args.allow_simulator,
        simulator_commands=args.simulator_command,
        allow_cpu_simulator_preflight=args.allow_cpu_simulator_preflight,
        cpu_preflight_backends=args.cpu_preflight_backend or CPU_BACKENDS,
        cpu_preflight_smoke_steps=args.cpu_preflight_smoke_steps,
        allow_cpu_preflight_render=args.allow_cpu_preflight_render,
        allow_training=args.allow_training,
        training_command=args.training_command,
        timeout_seconds=args.timeout_seconds,
        budget_usd=args.budget_usd,
        arena_scenario_count=args.arena_scenario_count,
        arena_shard_size=args.arena_shard_size,
        arena_num_envs=args.arena_num_envs,
        arena_retry_budget=args.arena_retry_budget,
        allow_rollout_vision_labeling=args.allow_rollout_vision_labeling,
        allow_delivery_upload=args.allow_delivery_upload,
        arena_operator_mode=args.arena_operator_mode,
        allow_live_agents_sdk=args.allow_live_agents_sdk,
        allow_live_codex_sdk=args.allow_live_codex_sdk,
        output_path=args.output_path,
    )
    print(f"[live-pipeline-control-plane] manifest={result['output_path']}")
    print(f"[live-pipeline-control-plane] status={result['status']}")
    if result["blockers"]:
        print(f"[live-pipeline-control-plane] blockers={len(result['blockers'])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
