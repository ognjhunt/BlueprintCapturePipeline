"""Discover completed model-runtime proof artifacts without exposing secrets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


OPENVLA_PROVIDER_SMOKE_JOB_ENV = "BLUEPRINT_OPENVLA_PROVIDER_SMOKE_JOB_DIR"
TRUSTED_OPENVLA_PROVIDER_OUTPUT_SCHEMAS = {
    "openvla_policy_provider_output.v1",
    "openvla_policy_command_adapter.v1",
}
UNITREE_UNIFOLM_PROVIDER_SMOKE_JOB_ENV = "BLUEPRINT_UNITREE_UNIFOLM_PROVIDER_SMOKE_JOB_DIR"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _openvla_payload_proves_policy_action(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("schema_version") in TRUSTED_OPENVLA_PROVIDER_OUTPUT_SCHEMAS
        and payload.get("status") == "completed"
        and payload.get("openvla_model_executed") is True
        and payload.get("openvla_predict_action_invoked") is True
        and payload.get("openvla_policy_action_command_ran") is True
        and isinstance(payload.get("action"), Mapping)
    )


def _openvla_provider_smoke_job_candidates(repo_root: Path) -> list[Path]:
    configured = os.getenv(OPENVLA_PROVIDER_SMOKE_JOB_ENV)
    candidates: list[Path] = [Path(configured).expanduser()] if configured else []
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def discover_openvla_provider_smoke_proof(*, repo_root: Path) -> dict[str, Any]:
    """Return completed OpenVLA provider-smoke proof, if one exists.

    This is action-model execution proof only. It is not endpoint closed-loop,
    dexterous manipulation, deployment, or physical-robot evidence.
    """

    blockers: list[str] = []
    for job_dir in _openvla_provider_smoke_job_candidates(repo_root):
        summary_path = job_dir / "openvla_policy_provider_smoke_summary.json"
        summary = _load_json(summary_path)
        if not summary:
            blockers.append(f"missing_or_invalid_summary:{job_dir}")
            continue
        output_path = job_dir / "openvla_provider_output" / "openvla_policy_provider_output.json"
        output_payload = _load_json(output_path)
        if not _openvla_payload_proves_policy_action(output_payload):
            blockers.extend(
                str(item)
                for item in output_payload.get("blockers", [])
                or summary.get("blockers", [])
                or ["openvla_provider_output_missing_trusted_runtime_proof"]
            )
            continue
        action = output_payload.get("action")
        return {
            "schema_version": "openvla_provider_smoke_proof.v1",
            "status": "completed",
            "provider_smoke_completed": True,
            "job_dir": str(job_dir),
            "summary_path": str(summary_path),
            "output_path": str(output_path) if output_path.is_file() else None,
            "openvla_model_executed": True,
            "openvla_model_loaded": bool(output_payload.get("openvla_model_loaded")),
            "openvla_predict_action_invoked": bool(
                output_payload.get("openvla_predict_action_invoked")
            ),
            "policy_action_model_command_ran": False,
            "openvla_policy_action_command_ran": False,
            "policy_action_model_provider_smoke_imported": True,
            "openvla_policy_action_command_imported": True,
            "action": dict(action) if isinstance(action, Mapping) else None,
            "model_execution_scope": (
                "provider_smoke_action_prediction_not_closed_loop_robot_control"
            ),
            "endpoint_closed_loop_policy_proven": False,
            "unitree_g1_dexterous_manipulation_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "blockers": [],
        }
    return {
        "schema_version": "openvla_provider_smoke_proof.v1",
        "status": "blocked",
        "provider_smoke_completed": False,
        "job_dir": None,
        "summary_path": None,
        "output_path": None,
        "openvla_model_executed": False,
        "openvla_model_loaded": False,
        "openvla_predict_action_invoked": False,
        "policy_action_model_command_ran": False,
        "openvla_policy_action_command_ran": False,
        "policy_action_model_provider_smoke_imported": False,
        "openvla_policy_action_command_imported": False,
        "action": None,
        "model_execution_scope": None,
        "endpoint_closed_loop_policy_proven": False,
        "unitree_g1_dexterous_manipulation_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "blockers": sorted(
            set(blockers or ["blocked_missing_completed_openvla_provider_smoke_job"])
        ),
    }


def discover_unitree_unifolm_provider_smoke_proof() -> dict[str, Any]:
    """Return explicitly configured Unitree UnifoLM provider-smoke proof."""

    configured = os.getenv(UNITREE_UNIFOLM_PROVIDER_SMOKE_JOB_ENV)
    if not configured:
        return {
            "schema_version": "unitree_unifolm_provider_smoke_proof.v1",
            "status": "skipped",
            "provider_smoke_completed": False,
            "job_dir": None,
            "summary_path": None,
            "output_path": None,
            "unitree_unifolm_model_executed": False,
            "unitree_unifolm_policy_action_command_ran": False,
            "policy_action_model_command_ran": False,
            "action": None,
            "model_execution_scope": None,
            "endpoint_closed_loop_policy_proven": False,
            "unitree_g1_dexterous_manipulation_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
            "blockers": ["skipped_unitree_unifolm_provider_smoke_not_configured"],
        }
    job_dir = Path(configured).expanduser().resolve()
    summary_path = job_dir / "unitree_unifolm_policy_provider_smoke_summary.json"
    summary = _load_json(summary_path)
    completed = bool(
        summary.get("status") == "completed"
        and summary.get("unitree_unifolm_model_executed") is True
        and summary.get("unitree_unifolm_policy_action_command_ran") is True
        and isinstance(summary.get("action"), Mapping)
    )
    output_path = job_dir / "unitree_unifolm_policy_provider_import.json"
    return {
        "schema_version": "unitree_unifolm_provider_smoke_proof.v1",
        "status": "completed" if completed else "blocked",
        "provider_smoke_completed": completed,
        "job_dir": str(job_dir),
        "summary_path": str(summary_path) if summary_path.is_file() else None,
        "output_path": str(output_path) if output_path.is_file() else None,
        "unitree_unifolm_model_executed": bool(summary.get("unitree_unifolm_model_executed")),
        "unitree_unifolm_policy_action_command_ran": bool(
            summary.get("unitree_unifolm_policy_action_command_ran")
        ),
        "policy_action_model_command_ran": bool(summary.get("policy_action_model_command_ran")),
        "action": dict(summary["action"]) if isinstance(summary.get("action"), Mapping) else None,
        "mode": summary.get("mode"),
        "model_execution_scope": (
            "unitree_unifolm_provider_smoke_action_prediction_not_closed_loop_robot_control"
            if completed
            else None
        ),
        "endpoint_closed_loop_policy_proven": False,
        "unitree_g1_dexterous_manipulation_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "blockers": list(summary.get("blockers", []) or ([] if completed else ["unitree_unifolm_provider_smoke_not_completed"])),
    }
