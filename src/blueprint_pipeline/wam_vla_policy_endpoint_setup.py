"""Write setup artifacts for local WAM/VLA policy endpoints."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .oscar_cosmos_wam_evaluator import (
    build_policy_model_endpoint_creation_plan,
    build_policy_model_endpoint_readiness_manifest,
)
from .policy_model_runtime_proofs import discover_openvla_provider_smoke_proof
from .provider_worker_contract import (
    HEALTHZ_PATH,
    INFER_PATH,
    LEGACY_HEALTH_PATH,
    LEGACY_POLICY_ACTION_PATH,
    PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION,
    READYZ_PATH,
    SHUTDOWN_PATH,
)
from .wam_vla_policy_endpoint_server import BUILTIN_REFERENCE_ADAPTER_COMMAND


SETUP_SCHEMA_VERSION = "wam_vla_policy_endpoint_setup.v1"
OSCAR_PROVIDER_COMMAND = (
    "python -m blueprint_pipeline.oscar_wam_provider_command_adapter "
    "--mode replay-existing-provider-output"
)
OSCAR_FRESH_PROVIDER_COMMAND = (
    "python -m blueprint_pipeline.oscar_wam_provider_command_adapter "
    "--mode vast-provider --allow-paid-vast-launch"
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _detect_oscar_checkpoint(repo_root: Path) -> Path | None:
    job_root = repo_root / "robot_eval_jobs"
    candidates = [
        job_root
        / "wam_model_runtime_bootstrap_oscar_20260621T025044Z"
        / "runtime_sources"
        / "oscar_wam"
        / "checkpoint",
    ]
    candidates.extend(
        sorted(job_root.glob("wam_model_runtime_bootstrap_oscar*/runtime_sources/oscar_wam/checkpoint"))
    )
    return _first_existing(candidates)


def _provider_runtime_result_proves_model_run(provider_job_dir: Path) -> bool:
    zip_path = provider_job_dir / "vast_provider_runtime_output.zip"
    if not zip_path.is_file():
        return False
    try:
        import zipfile

        with zipfile.ZipFile(zip_path) as archive:
            if "wam_runtime_result.json" not in archive.namelist():
                return False
            payload = json.loads(archive.read("wam_runtime_result.json").decode("utf-8"))
    except Exception:
        return False
    return bool(
        isinstance(payload, Mapping)
        and payload.get("status") == "completed"
        and payload.get("learned_wam_model_ran") is True
    )


def _detect_completed_oscar_provider_job(repo_root: Path) -> Path | None:
    job_root = repo_root / "robot_eval_jobs"
    preferred = [
        job_root
        / "oscar_wam_hands_pov_first_person_passthrough_fresh_vast_49f_20260622T065451Z"
        / "oscar_wam_provider_command_workspace"
        / "vast_provider_run",
        job_root
        / "oscar_wam_hands_pov_egocentric_mesh_conditioning_fresh_vast_49f_20260622T070946Z"
        / "oscar_wam_provider_command_workspace"
        / "vast_provider_run",
        job_root
        / "oscar_wam_hands_pov_texture_free_skeleton_fresh_vast_49f_20260622T072058Z"
        / "oscar_wam_provider_command_workspace"
        / "vast_provider_run",
    ]
    candidates = [
        path
        for path in preferred
        if _provider_runtime_result_proves_model_run(path)
    ]
    candidates.extend(
        path.parent
        for path in sorted(job_root.glob("**/vast_provider_runtime_output.zip"))
        if _provider_runtime_result_proves_model_run(path.parent)
    )
    return candidates[0] if candidates else None


def build_policy_model_runnable_env_artifact(
    *,
    repo_root: Path | None = None,
    generated_at: str | None = None,
) -> tuple[dict[str, Any], str]:
    root = (repo_root or _repo_root()).resolve()
    generated = generated_at or utc_now_iso()
    checkpoint = _detect_oscar_checkpoint(root)
    completed_provider_job = _detect_completed_oscar_provider_job(root)
    replay_ready = bool(checkpoint and completed_provider_job)
    fresh_provider_ready = bool(checkpoint)
    metadata = {
        "schema_version": "policy_model_runnable_env.v1",
        "generated_at": generated,
        "status": "ready" if replay_ready or fresh_provider_ready else "blocked",
        "oscar_replay_provider_ready": replay_ready,
        "oscar_fresh_provider_command_ready": fresh_provider_ready,
        "oscar_checkpoint_path": str(checkpoint) if checkpoint else None,
        "oscar_completed_provider_job_dir": (
            str(completed_provider_job) if completed_provider_job else None
        ),
        "commands": {
            "replay_completed_provider_output": OSCAR_PROVIDER_COMMAND,
            "fresh_vast_provider_launch": OSCAR_FRESH_PROVIDER_COMMAND,
        },
        "blockers": []
        if checkpoint
        else ["blocked_missing_oscar_wam_checkpoint_path"],
        "claim_boundary": {
            "endpoint_env_creation_is_not_model_execution_proof": True,
            "replay_completed_provider_output_is_not_fresh_model_run": True,
            "fresh_provider_launch_requires_paid_gpu_provider_success": True,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    if checkpoint and not completed_provider_job:
        metadata["blockers"].append("blocked_missing_completed_oscar_provider_job_for_replay")
    env_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Replay a completed OSCAR provider output through the WAM evaluator contract.",
        "# This proves import/binding of a prior learned WAM run, not a fresh model invocation.",
        "export BLUEPRINT_ALLOW_LOCAL_WAM_MODEL=true",
    ]
    if checkpoint:
        env_lines.append(f"export BLUEPRINT_OSCAR_WAM_CHECKPOINT={json.dumps(str(checkpoint))}")
    else:
        env_lines.append("# export BLUEPRINT_OSCAR_WAM_CHECKPOINT=/path/to/oscar/checkpoint")
    env_lines.append(f"export BLUEPRINT_OSCAR_WAM_COMMAND={json.dumps(OSCAR_PROVIDER_COMMAND)}")
    if completed_provider_job:
        env_lines.append(
            "export BLUEPRINT_OSCAR_WAM_PROVIDER_COMPLETED_JOB_DIR="
            f"{json.dumps(str(completed_provider_job))}"
        )
    else:
        env_lines.append(
            "# export BLUEPRINT_OSCAR_WAM_PROVIDER_COMPLETED_JOB_DIR=/path/to/completed/vast_provider_run"
        )
    env_lines.extend(
        [
            "",
            "# Fresh paid-provider path. Use only with an explicit spend cap and live inventory checks.",
            f"# export BLUEPRINT_OSCAR_WAM_COMMAND={json.dumps(OSCAR_FRESH_PROVIDER_COMMAND)}",
            "# export BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH=true",
            "# export BLUEPRINT_VAST_WAM_PUBLIC_IMAGE=docker.io/nijelhunt/blueprint-oscar-wam:20260622-cu128-shim",
            "",
        ]
    )
    return metadata, "\n".join(env_lines)


def build_wam_vla_policy_endpoint_setup(
    *,
    output_dir: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    root = output_dir or (
        _repo_root() / "policy_endpoint_setups" / f"wam_vla_policy_endpoint_setup_{_timestamp()}"
    )
    root = Path(root).resolve()
    ensure_dir(root)
    reference_adapter_script = (
        _repo_root() / "src" / "blueprint_pipeline" / "g1_endpoint_reference_adapter.py"
    )
    reference_adapter_subprocess_command = f"python {reference_adapter_script}"
    reference_adapter_command = BUILTIN_REFERENCE_ADAPTER_COMMAND
    openvla_provider_smoke_proof = discover_openvla_provider_smoke_proof(
        repo_root=_repo_root()
    )
    openvla_provider_smoke_completed = bool(
        openvla_provider_smoke_proof.get("provider_smoke_completed")
    )

    contract = {
        "schema_version": "wam_vla_policy_endpoint_contract.v1",
        "generated_at": generated,
        "status": "ready_for_local_endpoint_setup",
        "provider_worker_contract_schema_version": PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION,
        "http_contract": {
            "canonical": {
                "healthz": {"method": "GET", "path": HEALTHZ_PATH},
                "readyz": {"method": "GET", "path": READYZ_PATH},
                "infer": {"method": "POST", "path": INFER_PATH},
                "shutdown": {"method": "POST", "path": SHUTDOWN_PATH},
            },
            "legacy_compatibility": {
                "health": {"method": "GET", "path": LEGACY_HEALTH_PATH},
                "policy_action": {"method": "POST", "path": LEGACY_POLICY_ACTION_PATH},
            },
            "health": {"method": "GET", "path": LEGACY_HEALTH_PATH},
            "policy_action": {"method": "POST", "path": LEGACY_POLICY_ACTION_PATH},
            "request_json": {
                "observation": "Blueprint WAM/VLA observation packet",
                "optional_context": "adapter-specific context; must not contain raw secrets",
            },
            "response_json_any_of": [
                {"action": {"action_type": "waypoint", "waypoint": [0.5, 0.0, 0.79]}},
                {"action": {"action_type": "base_velocity", "linear_velocity_mps": 0.25}},
                {"action": {"action_type": "stop", "report": "done"}},
                {"action": {"action_type": "inspect_look", "yaw_rate_rad_s": 0.35}},
                {
                    "action": {
                        "action_type": "manipulation_contact",
                        "target_object_id": "blueprint_light_object",
                        "waypoint": [0.54, -0.65, 0.79],
                    }
                },
            ],
        },
        "supported_action_types": [
            "base_velocity",
            "heading_yaw",
            "waypoint",
            "stop",
            "inspect_look",
            "manipulation_contact",
        ],
        "auth": {
            "type": "bearer_token_from_file",
            "token_file_envs": [
                "WAM_POLICY_AUTH_TOKEN_FILE",
                "VLA_POLICY_AUTH_TOKEN_FILE",
                "TEAM_POLICY_AUTH_TOKEN_FILE",
            ],
            "raw_tokens_written_to_artifacts": False,
        },
        "evaluator_envs": {
            "team": {
                "TEAM_POLICY_WORKER_URL": "http://127.0.0.1:8765/infer",
                "TEAM_POLICY_WORKER_READY_URL": "http://127.0.0.1:8765/readyz",
                "TEAM_POLICY_ENDPOINT_URL": "http://127.0.0.1:8765/policy/action",
                "TEAM_POLICY_AUTH_TOKEN_FILE": "$HOME/.blueprint-secrets/team_policy_endpoint_token.txt",
            },
            "wam": {
                "WAM_POLICY_WORKER_URL": "http://127.0.0.1:8765/infer",
                "WAM_POLICY_WORKER_READY_URL": "http://127.0.0.1:8765/readyz",
                "WAM_POLICY_ENDPOINT_URL": "http://127.0.0.1:8765/policy/action",
                "WAM_POLICY_AUTH_TOKEN_FILE": "$HOME/.blueprint-secrets/wam_policy_endpoint_token.txt",
            },
            "vla": {
                "VLA_POLICY_WORKER_URL": "http://127.0.0.1:8765/infer",
                "VLA_POLICY_WORKER_READY_URL": "http://127.0.0.1:8765/readyz",
                "VLA_POLICY_ENDPOINT_URL": "http://127.0.0.1:8765/policy/action",
                "VLA_POLICY_AUTH_TOKEN_FILE": "$HOME/.blueprint-secrets/vla_policy_endpoint_token.txt",
            },
        },
        "claim_boundary": {
            "endpoint_running_is_not_policy_quality_proof": True,
            "model_response_must_control_mujoco_actions_successfully": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
        },
    }

    options = {
        "schema_version": "open_source_wam_vla_endpoint_options.v1",
        "generated_at": generated,
        "status": "advisory_setup_matrix",
        "recommended_first_path": "unitree_rl_gym_or_unitree_lerobot_for_g1_embodiment_actions_then_wrap_as_http_endpoint",
        "options": [
            {
                "id": "unitree_rl_gym",
                "source_url": "https://github.com/unitreerobotics/unitree_rl_gym",
                "best_fit": "G1 locomotion/control policy",
                "endpoint_role": "action_policy",
                "notes": "Embodiment-aligned for Unitree G1; best candidate for realistic navigation/contact once exported to a MuJoCo-runnable policy command.",
            },
            {
                "id": "unitree_lerobot",
                "source_url": "https://github.com/unitreerobotics/unitree_lerobot",
                "best_fit": "G1 manipulation data/training path",
                "endpoint_role": "vla_or_imitation_policy_adapter",
                "notes": "Use when the endpoint should consume Unitree G1 manipulation datasets or LeRobot-format policies.",
            },
            {
                "id": "unitree_unifolm_vla",
                "source_url": "https://github.com/unitreerobotics/unifolm-vla",
                "best_fit": "Unitree humanoid manipulation VLA",
                "endpoint_role": "vla_policy_adapter",
                "notes": "More G1/humanoid aligned than generic VLA, but still needs an observation preprocessor and action decoder for this evaluator schema.",
            },
            {
                "id": "openvla",
                "source_url": "https://github.com/openvla/openvla",
                "best_fit": "generic manipulation VLA comparison only",
                "endpoint_role": "comparison_candidate_not_default_g1_policy",
                "notes": "Do not use as the Unitree G1 policy path. It is useful only for non-Unitree or explicitly labeled comparison work.",
            },
            {
                "id": "cosmos_predict_2_5",
                "source_url": "https://github.com/nvidia-cosmos/cosmos-predict2.5",
                "best_fit": "world/action-conditioned video prediction and WAM review",
                "endpoint_role": "world_model_or_success_review_support",
                "notes": "Evaluator/support artifact only for Unitree G1. It is not a robot policy unless a Unitree-specific policy endpoint consumes it and emits G1 actions.",
            },
            {
                "id": "oscar_action_conditioned_world_model",
                "source_url": "https://arxiv.org/html/2606.04463v2",
                "best_fit": "action-conditioned world model evaluation",
                "endpoint_role": "world_model_or_success_review_support",
                "notes": "Treat as WAM evidence unless the implementation exposes a concrete action policy compatible with the Blueprint action schema.",
            },
        ],
        "minimum_real_endpoint_requirements": [
            "model_checkpoint_or_policy_weights_available_locally",
            "observation_preprocessor_maps_blueprint_packet_to_model_inputs",
            "action_decoder_maps_model_output_to_blueprint_action_schema",
            "auth_token_file_configured_without_printing_token",
            "http_endpoint_responds_to_post_policy_action",
            "mujoco_eval_reports_endpoint_policy_used_true_and_fixture_policy_used_false",
        ],
    }

    candidate_matrix = {
        "schema_version": "policy_model_candidate_matrix.v1",
        "generated_at": generated,
        "status": "adapter_boundary_defined",
        "candidates": [
            {
                "id": "command_policy",
                "runtime_role": "local_command_adapter",
                "default_local_command": reference_adapter_command,
                "fallback_subprocess_command": reference_adapter_subprocess_command,
                "console_script_command": "blueprint-g1-endpoint-reference-adapter",
                "recommended_local_command_reason": (
                    "the built-in wrapper path avoids Python subprocess startup on every closed-loop action"
                ),
                "current_repo_support": "implemented_reference_heuristic",
                "model_checkpoint_required": False,
                "real_model_claim_allowed": False,
            },
            {
                "id": "unitree_g1_policy",
                "runtime_role": "embodiment_aligned_locomotion_or_control_policy",
                "command_env": "BLUEPRINT_UNITREE_G1_POLICY_COMMAND",
                "checkpoint_env": "BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT",
                "current_repo_support": "adapter_contract_only",
                "model_checkpoint_required": True,
                "real_model_claim_allowed": "only_after_local_policy_runner_executes",
            },
            {
                "id": "openvla_policy",
                "runtime_role": "vla_or_imitation_policy_adapter",
                "command_env": "BLUEPRINT_OPENVLA_POLICY_COMMAND",
                "checkpoint_env": "BLUEPRINT_OPENVLA_POLICY_CHECKPOINT",
                "default_adapter_command": "blueprint-openvla-policy-command-adapter",
                "current_repo_support": "implemented_command_adapter_requires_runtime_checkpoint_and_visual_frame",
                "model_checkpoint_required": True,
                "provider_smoke_completed": openvla_provider_smoke_completed,
                "provider_smoke_job_dir": openvla_provider_smoke_proof.get("job_dir"),
                "openvla_model_executed": bool(
                    openvla_provider_smoke_proof.get("openvla_model_executed")
                ),
                "policy_action_model_command_ran": bool(
                    openvla_provider_smoke_proof.get("policy_action_model_command_ran")
                ),
                "openvla_policy_action_command_ran": bool(
                    openvla_provider_smoke_proof.get("openvla_policy_action_command_ran")
                ),
                "policy_action_model_provider_smoke_imported": bool(
                    openvla_provider_smoke_proof.get(
                        "policy_action_model_provider_smoke_imported"
                    )
                ),
                "openvla_policy_action_command_imported": bool(
                    openvla_provider_smoke_proof.get(
                        "openvla_policy_action_command_imported"
                    )
                ),
                "last_provider_action": openvla_provider_smoke_proof.get("action"),
                "real_model_claim_allowed": "provider_smoke_action_prediction_only"
                if openvla_provider_smoke_completed
                else "only_after_endpoint_response_proven",
                "endpoint_closed_loop_policy_proven": False,
                "unitree_g1_dexterous_manipulation_proven": False,
                "claim_boundary": {
                    "provider_smoke_is_model_execution_proof": openvla_provider_smoke_completed,
                    "provider_smoke_is_not_closed_loop_endpoint_control": True,
                    "provider_smoke_is_not_dexterous_manipulation_proof": True,
                },
            },
            {
                "id": "oscar_wam",
                "runtime_role": "action_conditioned_world_model_rollout_generator",
                "command_env": "BLUEPRINT_OSCAR_WAM_COMMAND",
                "checkpoint_env": "BLUEPRINT_OSCAR_WAM_CHECKPOINT",
                "current_repo_support": "oscar_cosmos_wam_evaluator_contract",
                "model_checkpoint_required": True,
                "real_model_claim_allowed": "only_after_generated_rollouts_are_written",
            },
            {
                "id": "cosmos_wam",
                "runtime_role": "world_video_rollout_or_review_substrate",
                "command_env": "BLUEPRINT_COSMOS_WAM_COMMAND",
                "checkpoint_env": "BLUEPRINT_COSMOS_WAM_CHECKPOINT",
                "current_repo_support": "oscar_cosmos_wam_evaluator_contract",
                "model_checkpoint_required": True,
                "real_model_claim_allowed": "only_after_generated_rollouts_are_written",
            },
        ],
        "stable_contracts": [
            "oscar_wam",
            "cosmos_wam",
            "openvla_policy",
            "unitree_g1_policy",
            "command_policy",
        ],
    }

    truth_boundary = {
        "schema_version": "policy_model_truth_boundary.v1",
        "generated_at": generated,
        "raw_capture_evidence_authoritative": True,
        "endpoint_running_is_not_model_quality_proof": True,
        "reference_command_policy_is_not_real_wam_vla": True,
        "real_wam_vla_proof_requires_model_endpoint_response_with_recorded_provenance": True,
        "openvla_provider_smoke_proof": openvla_provider_smoke_proof,
        "openvla_provider_smoke_model_executed": bool(
            openvla_provider_smoke_proof.get("openvla_model_executed")
        ),
        "policy_action_model_command_ran": bool(
            openvla_provider_smoke_proof.get("policy_action_model_command_ran")
        ),
        "openvla_policy_action_command_ran": bool(
            openvla_provider_smoke_proof.get("openvla_policy_action_command_ran")
        ),
        "policy_action_model_provider_smoke_imported": bool(
            openvla_provider_smoke_proof.get("policy_action_model_provider_smoke_imported")
        ),
        "openvla_policy_action_command_imported": bool(
            openvla_provider_smoke_proof.get("openvla_policy_action_command_imported")
        ),
        "openvla_provider_smoke_is_not_closed_loop_endpoint_or_dexterous_manipulation": True,
        "wam_rollouts_are_model_derived_support_artifacts": True,
        "mujoco_evidence_is_simulator_only": True,
        "isaac_proof": False,
        "splat_ply_spz_proof": False,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "official_unitree_controller_proven_by_reference_adapter": False,
    }

    adapter_manifest = {
        "schema_version": "policy_command_adapter_manifest.v1",
        "generated_at": generated,
        "status": "ready_for_local_reference_endpoint",
        "default_reference_adapter_command": reference_adapter_command,
        "fallback_reference_adapter_subprocess_command": reference_adapter_subprocess_command,
        "console_script_reference_adapter_command": "blueprint-g1-endpoint-reference-adapter",
        "openvla_policy_adapter_command": "blueprint-openvla-policy-command-adapter",
        "openvla_provider_smoke_proof": openvla_provider_smoke_proof,
        "recommended_local_command_reason": (
            "the built-in wrapper path avoids Python subprocess startup on every closed-loop action"
        ),
        "default_reference_adapter_invocation_mode": "in_process_builtin",
        "stdin_contract": {"observation": "Blueprint observation packet"},
        "stdout_contract": {"policy_id": "string", "action": "Blueprint supported action"},
        "supported_action_types": contract["supported_action_types"],
        "provider_worker_policy_adapter_command": (
            "blueprint-provider-worker-policy-command-adapter"
        ),
        "provider_worker_policy_adapter_contract": {
            "worker_url_envs": [
                "BLUEPRINT_PROVIDER_POLICY_WORKER_URL",
                "BLUEPRINT_POLICY_WORKER_URL",
                "TEAM_POLICY_WORKER_URL",
            ],
            "ready_url_envs": [
                "BLUEPRINT_PROVIDER_POLICY_WORKER_READY_URL",
                "BLUEPRINT_POLICY_WORKER_READY_URL",
                "TEAM_POLICY_WORKER_READY_URL",
            ],
            "requires_readyz_before_infer": True,
            "does_not_allocate_provider": True,
        },
        "adapter_families": [
            "command_policy",
            "provider_worker_policy",
            "unitree_g1_policy",
            "openvla_policy",
            "oscar_wam",
            "cosmos_wam",
        ],
        "raw_tokens_written_to_artifacts": False,
        "claim_boundary": truth_boundary,
    }
    endpoint_readiness = build_policy_model_endpoint_readiness_manifest(
        generated_at=generated,
    )
    endpoint_creation_plan = build_policy_model_endpoint_creation_plan(
        generated_at=generated,
        readiness_manifest=endpoint_readiness,
    )
    runnable_env_metadata, runnable_env_text = build_policy_model_runnable_env_artifact(
        repo_root=_repo_root(),
        generated_at=generated,
    )

    env_template = """#!/usr/bin/env bash
set -euo pipefail

# Create this file yourself and keep the raw token out of logs/artifacts:
#   blueprint-create-team-policy-endpoint-token

export TEAM_POLICY_WORKER_URL="http://127.0.0.1:8765/infer"
export TEAM_POLICY_WORKER_READY_URL="http://127.0.0.1:8765/readyz"
export TEAM_POLICY_ENDPOINT_URL="http://127.0.0.1:8765/policy/action"
export TEAM_POLICY_AUTH_TOKEN_FILE="$HOME/.blueprint-secrets/team_policy_endpoint_token.txt"

# The command receives JSON on stdin:
#   {"observation": { ... Blueprint observation packet ... }}
# It must write JSON on stdout:
#   {"action": {"action_type": "waypoint", "waypoint": [0.5, 0.0, 0.79]}}
export BLUEPRINT_WAM_VLA_POLICY_COMMAND="__REFERENCE_ADAPTER_COMMAND__"
export BLUEPRINT_WAM_VLA_POLICY_AUTH_TOKEN_FILE="$TEAM_POLICY_AUTH_TOKEN_FILE"

# To wrap OpenVLA after the real runtime/checkpoint are available:
# export BLUEPRINT_OPENVLA_POLICY_CHECKPOINT="/path/to/openvla-7b-or-finetuned-checkpoint"
# export BLUEPRINT_OPENVLA_POLICY_SOURCE_ROOT="/path/to/openvla/source"  # optional
# export BLUEPRINT_WAM_VLA_POLICY_COMMAND="blueprint-openvla-policy-command-adapter"

# To use an already-running provider worker for a Unitree policy candidate:
# export BLUEPRINT_UNITREE_G1_POLICY_COMMAND="blueprint-provider-worker-policy-command-adapter"
# export BLUEPRINT_UNITREE_G1_POLICY_CHECKPOINT="provider-worker/model-id"
# export BLUEPRINT_PROVIDER_POLICY_WORKER_URL="$TEAM_POLICY_WORKER_URL"
# export BLUEPRINT_PROVIDER_POLICY_WORKER_READY_URL="$TEAM_POLICY_WORKER_READY_URL"
""".replace("__REFERENCE_ADAPTER_COMMAND__", reference_adapter_command)

    runbook = """# WAM/VLA Policy Endpoint Setup

1. Pick the policy backend.

For Unitree G1 navigation/contact, start with a Unitree G1 locomotion/control stack and expose its policy as a command. For G1 hand or gripper manipulation, use a Unitree-specific LeRobot or UnifoLM adapter after it can emit one of Blueprint's supported action schemas. OpenVLA remains comparison-only for Unitree G1 unless a Unitree-specific decoder and checkpoint are configured and proven.

2. Implement the command adapter.

The command reads one JSON object from stdin with an `observation` field and writes one JSON object to stdout with an `action` field. It must not print secrets.

3. Start the local endpoint.

```bash
blueprint-create-team-policy-endpoint-token
source local_endpoint_env_template.sh
blueprint-serve-wam-vla-policy-endpoint --host 127.0.0.1 --port 8765
```

4. Run the MuJoCo endpoint evaluator.

```bash
python -m blueprint_pipeline.mujoco_g1_wam_vla_policy_endpoint_eval \\
  --job-root /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/robot_eval_jobs \\
  --g1-model-root /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/output/external_assets/mujoco_menagerie/unitree_g1
```

Success for endpoint plumbing means `endpoint_policy_used=true`, `fixture_policy_used=false`, endpoint attempts were invoked, and the returned actions normalized and controlled MuJoCo.

Why not just create OSCAR/Cosmos/OpenVLA endpoints automatically?

The HTTP wrapper is already generic: `blueprint-serve-wam-vla-policy-endpoint` can wrap any local command that reads Blueprint observation JSON and writes Blueprint action JSON. The default local reference adapter uses `builtin:g1_endpoint_reference_adapter` so repeated closed-loop MuJoCo action calls do not pay Python subprocess startup on every policy interval. A real model endpoint still needs the model runtime command, local checkpoint path, file-based credentials when required, and an adapter that maps model inputs/outputs to Blueprint contracts. Starting the wrapper around a missing command would only create a 503 endpoint, and starting it around the reference G1 adapter command proves endpoint plumbing rather than learned WAM/VLA behavior.

Unitree UnifoLM VLA bridge command:

```bash
export BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT=/path/to/UnifoLM-VLA-Base/checkpoints/pytorch_model.pt
# The provider-facing alias is accepted by the Blueprint adapter/image path too:
# export BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT="$BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT"
export BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT=/path/to/UnifoLM-VLM-Base
export BLUEPRINT_WAM_VLA_POLICY_COMMAND="blueprint-unitree-unifolm-vla-server-bridge --server-url http://127.0.0.1:8777/act"
blueprint-serve-wam-vla-policy-endpoint --host 127.0.0.1 --port 8765
```

That command assumes a real Unitree UnifoLM server is already running with the
configured checkpoints. The bridge only proves endpoint plumbing until the
server returns a real action chunk and the evaluator records the resulting G1
action.

OpenVLA comparison command:

```bash
export BLUEPRINT_OPENVLA_POLICY_CHECKPOINT=/path/to/openvla-7b-or-finetuned-checkpoint
export BLUEPRINT_WAM_VLA_POLICY_COMMAND=blueprint-openvla-policy-command-adapter
blueprint-serve-wam-vla-policy-endpoint --host 127.0.0.1 --port 8765
```

The adapter requires a simulated policy camera frame in the observation packet
and a runnable OpenVLA runtime. Generic OpenVLA output is decoded conservatively
into Blueprint endpoint actions; it is not Unitree dexterous manipulation proof
without a Unitree-specific action decoder and checkpoint.

See `policy_model_endpoint_readiness_manifest.json` for the exact missing env vars, checkpoints, gates, and provider/runtime blockers for OSCAR, Cosmos, OpenVLA, and Unitree G1 policy candidates.
"""

    paths = {
        "contract": root / "wam_vla_policy_endpoint_contract.json",
        "options": root / "open_source_wam_vla_endpoint_options.json",
        "policy_model_candidate_matrix": root / "policy_model_candidate_matrix.json",
        "policy_model_truth_boundary": root / "policy_model_truth_boundary.json",
        "policy_model_endpoint_readiness_manifest": root
        / "policy_model_endpoint_readiness_manifest.json",
        "policy_model_endpoint_creation_plan": root / "policy_model_endpoint_creation_plan.json",
        "policy_model_runnable_env_manifest": root / "policy_model_runnable_env_manifest.json",
        "policy_model_runnable_env": root / "policy_model_runnable_env.sh",
        "policy_command_adapter_manifest": root / "policy_command_adapter_manifest.json",
        "env_template": root / "local_endpoint_env_template.sh",
        "runbook": root / "README.md",
    }
    write_json(paths["contract"], contract)
    write_json(paths["options"], options)
    write_json(paths["policy_model_candidate_matrix"], candidate_matrix)
    write_json(paths["policy_model_truth_boundary"], truth_boundary)
    write_json(paths["policy_model_endpoint_readiness_manifest"], endpoint_readiness)
    write_json(paths["policy_model_endpoint_creation_plan"], endpoint_creation_plan)
    write_json(paths["policy_model_runnable_env_manifest"], runnable_env_metadata)
    _write_text(paths["policy_model_runnable_env"], runnable_env_text)
    write_json(paths["policy_command_adapter_manifest"], adapter_manifest)
    _write_text(paths["env_template"], env_template)
    _write_text(paths["runbook"], runbook)

    summary = {
        "schema_version": SETUP_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed",
        "output_dir": str(root),
        "artifacts": {key: str(path) for key, path in paths.items()},
        "next_command": "source local_endpoint_env_template.sh && blueprint-serve-wam-vla-policy-endpoint --host 127.0.0.1 --port 8765",
    }
    write_json(root / "wam_vla_policy_endpoint_setup_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    summary = build_wam_vla_policy_endpoint_setup(output_dir=args.output_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
