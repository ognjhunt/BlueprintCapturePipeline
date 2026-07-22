"""Provider-neutral worker contract helpers for robot policy runtimes.

This module defines the stable shape Blueprint expects from GPU-backed policy
workers. Provider adapters may start workers differently, but the policy/WAM
loop should talk to a worker once it is ready instead of launching a provider
instance for each action.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

from .common import ensure_dir, utc_now_iso, write_json


PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION = "provider_worker_contract.v1"

HEALTHZ_PATH = "/healthz"
READYZ_PATH = "/readyz"
INFER_PATH = "/infer"
SHUTDOWN_PATH = "/shutdown"
LEGACY_HEALTH_PATH = "/health"
LEGACY_POLICY_ACTION_PATH = "/policy/action"

ONE_SHOT_PROVIDER_COMMAND_MARKERS = (
    "unitree_groot_n17_sonic_vast_policy_command",
    "blueprint-run-vast-provider-adapter",
    "blueprint_pipeline.vast_provider_adapter",
    "vast_provider_adapter",
    "--allow-paid-vast-launch",
    "--allow-vast-instance-launch",
    "--mode vast-provider",
    " vast-provider",
)
ALLOW_PROVIDER_LAUNCH_PER_POLICY_INFERENCE_ENV = (
    "BLUEPRINT_ALLOW_PROVIDER_LAUNCH_PER_POLICY_INFERENCE"
)

PERSISTENT_BACKEND_CLIENT_MARKERS = (
    "provider_worker_policy_command_adapter",
    "blueprint-provider-worker-policy-command-adapter",
    "unitree_groot_n17_sonic_policy_server_command",
    "blueprint-unitree-groot-n17-sonic-policy-server-command",
    "unitree_unifolm_vla_server_bridge",
    "blueprint-unitree-unifolm-vla-server-bridge",
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _contains_any(text: str, markers: Sequence[str]) -> bool:
    lower = text.lower()
    return any(marker.lower() in lower for marker in markers)


def _env_truthy(name: str) -> bool:
    return _string(os.getenv(name)).lower() in {"1", "true", "yes", "y", "on"}


def classify_policy_worker_command(command: str | None) -> dict[str, Any]:
    """Classify whether a policy command is safe for repeated loop calls."""

    text = _string(command)
    if not text:
        return {
            "schema_version": "policy_worker_command_classification.v1",
            "status": "blocked",
            "invocation_kind": "missing",
            "command_configured": False,
            "command_value_redacted": None,
            "repeated_policy_loop_allowed": False,
            "provider_instance_launch_per_inference": None,
            "persistent_worker_required": True,
            "blockers": ["blocked_missing_policy_worker_command"],
            "warnings": [],
        }

    is_http_endpoint = text.startswith(("http://", "https://"))
    is_one_shot_provider_launcher = _contains_any(text, ONE_SHOT_PROVIDER_COMMAND_MARKERS)
    is_persistent_backend_client = _contains_any(text, PERSISTENT_BACKEND_CLIENT_MARKERS)
    blockers: list[str] = []
    warnings: list[str] = []

    if is_one_shot_provider_launcher:
        invocation_kind = "one_shot_provider_launcher"
        repeated_allowed = _env_truthy(ALLOW_PROVIDER_LAUNCH_PER_POLICY_INFERENCE_ENV)
        persistent_worker_required = True
        provider_instance_launch_per_inference = True
        if repeated_allowed:
            warnings.append(
                "provider_instance_launch_per_policy_inference_explicitly_allowed"
            )
            warnings.append(
                "persistent_policy_worker_still_recommended_for_long_repeated_loops"
            )
        else:
            blockers.append(
                "one_shot_provider_launcher_not_allowed_for_repeated_policy_loop"
            )
    elif is_http_endpoint:
        invocation_kind = "http_worker_endpoint"
        repeated_allowed = True
        persistent_worker_required = False
        provider_instance_launch_per_inference = False
    elif is_persistent_backend_client:
        invocation_kind = "persistent_backend_client_command"
        repeated_allowed = True
        persistent_worker_required = False
        provider_instance_launch_per_inference = False
    else:
        invocation_kind = "subprocess_command_adapter"
        repeated_allowed = True
        persistent_worker_required = False
        provider_instance_launch_per_inference = False
        warnings.append(
            "subprocess_adapter_should_be_wrapped_by_persistent_worker_for_gpu_backends"
        )

    return {
        "schema_version": "policy_worker_command_classification.v1",
        "status": "blocked" if blockers else "compatible",
        "invocation_kind": invocation_kind,
        "command_configured": True,
        "command_value_redacted": "<configured>",
        "repeated_policy_loop_allowed": repeated_allowed,
        "provider_instance_launch_per_inference": provider_instance_launch_per_inference,
        "persistent_worker_required": persistent_worker_required,
        "recommended_worker_contract": {
            "health": HEALTHZ_PATH,
            "ready": READYZ_PATH,
            "infer": INFER_PATH,
            "shutdown": SHUTDOWN_PATH,
            "legacy_policy_action": LEGACY_POLICY_ACTION_PATH,
        },
        "blockers": blockers,
        "warnings": warnings,
        "claim_boundary": {
            "classification_is_static_command_shape_only": True,
            "does_not_prove_worker_running": True,
            "does_not_run_provider": True,
            "raw_secret_values_recorded": False,
        },
    }


def build_provider_worker_contract(
    *,
    generated_at: str | None = None,
    provider: str | None = None,
    worker_role: str = "policy_action_worker",
    policy_command: str | None = None,
) -> dict[str, Any]:
    """Build the provider-neutral worker contract artifact."""

    generated = generated_at or utc_now_iso()
    command_classification = classify_policy_worker_command(policy_command)
    return {
        "schema_version": PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "ready_for_worker_implementation",
        "provider": _string(provider) or "provider_neutral",
        "worker_role": worker_role,
        "http_contract": {
            "canonical": {
                "health": {"method": "GET", "path": HEALTHZ_PATH},
                "ready": {"method": "GET", "path": READYZ_PATH},
                "infer": {"method": "POST", "path": INFER_PATH},
                "shutdown": {"method": "POST", "path": SHUTDOWN_PATH},
            },
            "legacy_compatibility": {
                "health": {"method": "GET", "path": LEGACY_HEALTH_PATH},
                "policy_action": {"method": "POST", "path": LEGACY_POLICY_ACTION_PATH},
            },
            "infer_request_json": {
                "observation": "Blueprint policy observation packet",
                "optional_context": "Adapter-specific metadata; must not contain raw secrets",
            },
            "infer_response_json": {
                "policy_id": "Worker-selected policy id",
                "action": "Blueprint action object",
                "endpoint_metadata": "Redacted worker/runtime metadata",
            },
        },
        "worker_lifecycle": {
            "provider_adapter_responsibilities": [
                "start_worker",
                "discover_endpoint",
                "wait_for_readyz",
                "send_infer_requests",
                "request_shutdown_after_eval_job",
                "record_teardown_and_cost_artifacts",
            ],
            "policy_loop_responsibilities": [
                "call_ready_worker",
                "reuse_worker_for_all_policy_steps",
                "never_launch_provider_inside_each_policy_action",
            ],
            "model_load_policy": {
                "load_model_once_per_worker": True,
                "download_or_compile_during_each_inference": False,
                "runtime_dependency_install_during_customer_job": False,
            },
        },
        "provider_portability": {
            "same_contract_for": [
                "runpod",
                "vast",
                "gcp",
                "aws_g6",
                "coreweave",
            ],
            "provider_specific_code_limited_to": [
                "allocation",
                "endpoint_discovery",
                "readiness_polling",
                "teardown",
                "cost_accounting",
            ],
        },
        "policy_command_classification": command_classification,
        "claim_boundary": {
            "contract_artifact_is_not_provider_execution_proof": True,
            "worker_readyz_required_before_customer_eval": True,
            "shutdown_response_is_not_cost_proof_without_provider_teardown_artifact": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
            "raw_secret_values_recorded": False,
        },
    }


def write_provider_worker_contract(
    *,
    output_dir: str | Path,
    generated_at: str | None = None,
    provider: str | None = None,
    worker_role: str = "policy_action_worker",
    policy_command: str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser()
    ensure_dir(output)
    contract = build_provider_worker_contract(
        generated_at=generated_at,
        provider=provider,
        worker_role=worker_role,
        policy_command=policy_command,
    )
    write_json(output / "provider_worker_contract.json", contract)
    return contract


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provider", default="provider_neutral")
    parser.add_argument("--worker-role", default="policy_action_worker")
    parser.add_argument("--policy-command")
    parser.add_argument("--policy-command-env")
    args = parser.parse_args(argv)
    command = args.policy_command
    if not command and args.policy_command_env:
        command = os.getenv(args.policy_command_env, "")
    contract = write_provider_worker_contract(
        output_dir=args.output_dir,
        provider=args.provider,
        worker_role=args.worker_role,
        policy_command=command,
    )
    print(contract["status"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
