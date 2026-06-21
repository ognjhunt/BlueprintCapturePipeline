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


SETUP_SCHEMA_VERSION = "wam_vla_policy_endpoint_setup.v1"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


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

    contract = {
        "schema_version": "wam_vla_policy_endpoint_contract.v1",
        "generated_at": generated,
        "status": "ready_for_local_endpoint_setup",
        "http_contract": {
            "health": {"method": "GET", "path": "/health"},
            "policy_action": {"method": "POST", "path": "/policy/action"},
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
                "TEAM_POLICY_ENDPOINT_URL": "http://127.0.0.1:8765/policy/action",
                "TEAM_POLICY_AUTH_TOKEN_FILE": "$HOME/.blueprint-secrets/team_policy_endpoint_token.txt",
            },
            "wam": {
                "WAM_POLICY_ENDPOINT_URL": "http://127.0.0.1:8765/policy/action",
                "WAM_POLICY_AUTH_TOKEN_FILE": "$HOME/.blueprint-secrets/wam_policy_endpoint_token.txt",
            },
            "vla": {
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
                "best_fit": "general manipulation VLA baseline/fine-tuning",
                "endpoint_role": "vla_policy_adapter",
                "notes": "Useful for instruction-to-action manipulation after embodiment-specific fine-tuning; not a drop-in G1 locomotion controller.",
            },
            {
                "id": "cosmos_predict_2_5",
                "source_url": "https://github.com/nvidia-cosmos/cosmos-predict2.5",
                "best_fit": "world/action-conditioned video prediction and WAM review",
                "endpoint_role": "world_model_or_success_review_support",
                "notes": "Use for predicting/reviewing future video unless paired with an inverse-dynamics or policy head that emits actions.",
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
                "default_local_command": "blueprint-g1-endpoint-reference-adapter",
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
                "current_repo_support": "adapter_contract_only",
                "model_checkpoint_required": True,
                "real_model_claim_allowed": "only_after_endpoint_response_proven",
            },
            {
                "id": "oscar_wam",
                "runtime_role": "action_conditioned_world_model_evaluator",
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
        "default_reference_adapter_command": "blueprint-g1-endpoint-reference-adapter",
        "stdin_contract": {"observation": "Blueprint observation packet"},
        "stdout_contract": {"policy_id": "string", "action": "Blueprint supported action"},
        "supported_action_types": contract["supported_action_types"],
        "adapter_families": [
            "command_policy",
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

    env_template = """#!/usr/bin/env bash
set -euo pipefail

# Create this file yourself and keep the raw token out of logs/artifacts:
#   blueprint-create-team-policy-endpoint-token

export TEAM_POLICY_ENDPOINT_URL="http://127.0.0.1:8765/policy/action"
export TEAM_POLICY_AUTH_TOKEN_FILE="$HOME/.blueprint-secrets/team_policy_endpoint_token.txt"

# The command receives JSON on stdin:
#   {"observation": { ... Blueprint observation packet ... }}
# It must write JSON on stdout:
#   {"action": {"action_type": "waypoint", "waypoint": [0.5, 0.0, 0.79]}}
export BLUEPRINT_WAM_VLA_POLICY_COMMAND="blueprint-g1-endpoint-reference-adapter"
export BLUEPRINT_WAM_VLA_POLICY_AUTH_TOKEN_FILE="$TEAM_POLICY_AUTH_TOKEN_FILE"
"""

    runbook = """# WAM/VLA Policy Endpoint Setup

1. Pick the policy backend.

For Unitree G1 navigation/contact, start with a Unitree G1 locomotion/control stack and expose its policy as a command. For VLA manipulation, use a VLA/OpenVLA/UnifoLM/LeRobot adapter only after it can emit one of Blueprint's supported action schemas.

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

The HTTP wrapper is already generic: `blueprint-serve-wam-vla-policy-endpoint` can wrap any local command that reads Blueprint observation JSON and writes Blueprint action JSON. A real model endpoint still needs the model runtime command, local checkpoint path, file-based credentials when required, and an adapter that maps model inputs/outputs to Blueprint contracts. Starting the wrapper around a missing command would only create a 503 endpoint, and starting it around `blueprint-g1-endpoint-reference-adapter` proves endpoint plumbing rather than learned WAM/VLA behavior.

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
