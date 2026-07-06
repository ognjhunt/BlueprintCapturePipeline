"""Real policy family closed-loop eval harness.

Takes a LeRobot-format policy checkpoint through the full production chain —
``robot_eval_job_orchestrator`` -> real MuJoCo closed-loop rollout -> SC3
protocol artifact -> ``task_eval_run_report`` — and verifies the resulting
report is a *real-substrate* report (attempts from a command-adapter rollout
with the policy in the loop), not a fixture expansion.

The first registered family is the CPU-runnable scripted pick-place baseline,
which proves the plumbing without a GPU. Swapping in a learned/GPU LeRobot
policy is a config-only change: point ``checkpoint_dir`` at the learned
checkpoint and set ``adapter_command`` to its inference wrapper — the harness
code path is identical.

Registration boundary: evaluated families are registered in the
ranker-validation ladder only. Nothing here ever touches
``UNITREE_ACTION_COMMAND_CANDIDATES`` (the production candidate registry);
production registration stays blocked until a real GPU run exists.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import utc_now_iso, write_json
from .lerobot_policy_family import (
    create_scripted_baseline_checkpoint,
    load_lerobot_policy_checkpoint,
)
from .lerobot_torch_policy_adapter import (
    GROOT_LIBERO_CHECKPOINT_REPO_ID,
    GROOT_LIBERO_INTEGRATION_LABEL,
    LIBERO_VISUAL_FEATURE_KEYS,
    build_gpu_runtime_contract,
)
from .policy_ranking_ladder import build_known_ordering_policy_ladder

HARNESS_MANIFEST_SCHEMA_VERSION = "real_policy_family_eval_harness_manifest.v1"
FAMILY_REGISTRY_SCHEMA_VERSION = "real_policy_family_registry.v1"
FAMILY_CONFIG_SCHEMA_VERSION = "real_policy_family_config.v1"

REAL_SUBSTRATE = "classical_sim_mujoco"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return _mapping(payload)


def default_family_config(
    *,
    family_id: str,
    checkpoint_dir: str,
    adapter_command: str | None = None,
    requires_gpu: bool = False,
) -> dict[str, Any]:
    """Policy family as data: a GPU family differs only in these values."""
    return {
        "schema_version": FAMILY_CONFIG_SCHEMA_VERSION,
        "family_id": family_id,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_format": "lerobot_pretrained_dir",
        "adapter_command": adapter_command,
        "adapter_mode": "persistent" if adapter_command else "subprocess",
        "render_obs_frames": bool(adapter_command),
        "policy_execution_command": None,
        "requires_gpu": requires_gpu,
        "simulator": "mujoco",
        "control_hz": 20.0,
        "chunk_size": 25,
        "executed_horizon": 16,
        "max_rollout_seconds": 20.0,
    }


def default_groot_libero_family_config(
    *,
    checkpoint: str = GROOT_LIBERO_CHECKPOINT_REPO_ID,
    family_id: str = "nvidia_groot_n17_lerobot_libero_10_640",
    device: str = "cuda",
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Family config for NVIDIA's GR00T N1.7 LIBERO/Panda checkpoint.

    This is intentionally an integration-proof lane. It can prove that the
    Blueprint closed-loop harness invoked a real learned GR00T/LeRobot policy,
    but the first run is still a LIBERO/Panda action projection onto the
    Blueprint tabletop proxy, not a production manipulator-quality claim.
    """
    executable = _string(python_executable) or sys.executable
    adapter_command = " ".join(
        [
            shlex.quote(executable),
            "-m",
            "blueprint_pipeline.lerobot_torch_policy_adapter",
            "--checkpoint",
            shlex.quote(checkpoint),
            "--device",
            shlex.quote(device),
            "--chunk-size",
            "16",
            "--serve",
        ]
    )
    family = default_family_config(
        family_id=family_id,
        checkpoint_dir=checkpoint,
        adapter_command=adapter_command,
        requires_gpu=True,
    )
    family.update(
        {
            "adapter_mode": "persistent",
            "render_obs_frames": True,
            "chunk_size": 16,
            "executed_horizon": 16,
            "integration_proof_scope": GROOT_LIBERO_INTEGRATION_LABEL,
            "source_policy": {
                "repo_id": checkpoint,
                "policy_type": "groot",
                "trained_dataset": "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
                "embodiment_tag": "libero_sim",
                "action_decode_transform": "libero",
                "expected_visual_features": list(LIBERO_VISUAL_FEATURE_KEYS),
                "expected_state_dim": 8,
                "expected_action_dim": 7,
            },
            "gpu_runtime": build_gpu_runtime_contract(
                checkpoint=checkpoint,
                device=device,
                policy_type="groot",
            ),
            "meaningful_manipulator_scoring": {
                "status": "not_claimed_by_blueprint_tabletop_projection",
                "workflow_projection_status": "available_for_integration_plumbing",
                "required_for_quality_claim": (
                    "libero_panda_simulator_bridge_or_panda_task_evaluator"
                ),
                "blueprint_tabletop_proxy_is_panda_evaluator": False,
            },
            "claim_boundary": {
                "learned_policy_invocation_proof_target": True,
                "libero_panda_action_projection_to_blueprint_delta_ee": True,
                "panda_or_libero_task_success_proven": False,
                "meaningful_manipulator_scoring_proven": False,
                "blueprint_site_task_success_proven": False,
                "humanoid_readiness_proven": False,
                "physical_robot_readiness_proven": False,
                "buyer_facing_deployment_claim_allowed": False,
                "production_candidate_registration_allowed": False,
            },
        }
    )
    return family


def write_demo_pick_place_capture_root(root: str | Path) -> Path:
    """Bootstrap a minimal demo capture root so the plumbing can run locally.

    The cards define eval *scope* (task, scenario, zones); they are demo
    bootstrap content, labeled as such. Eval evidence still comes exclusively
    from the real rollout — nothing written here fabricates attempts, media,
    or outcomes.
    """
    capture_root = Path(root).expanduser().resolve()
    write_json(
        capture_root / "capture_descriptor.json",
        {
            "scene_id": "demo-scene-1",
            "capture_id": "demo-capture-1",
            "metadata": {"site_identity": {"site_id": "demo-site-1"}},
            "demo_bootstrap": True,
        },
    )
    write_json(
        capture_root / "raw" / "manifest.json",
        {
            "scene_id": "demo-scene-1",
            "capture_id": "demo-capture-1",
            "site_identity": {"site_id": "demo-site-1"},
            "video_uri": "walkthrough.mov",
            "width": 1280,
            "height": 720,
            "frame_count": 3,
            "demo_bootstrap": True,
        },
    )
    write_json(
        capture_root / "raw" / "capture_upload_complete.json",
        {
            "scene_id": "demo-scene-1",
            "capture_id": "demo-capture-1",
            "status": "complete",
        },
    )
    raw_video = capture_root / "raw" / "walkthrough.mov"
    if not raw_video.exists():
        raw_video.parent.mkdir(parents=True, exist_ok=True)
        raw_video.write_bytes(b"demo bootstrap capture placeholder\n")

    robot_eval_dir = capture_root / "pipeline" / "robot_eval_dataset"
    write_json(
        robot_eval_dir / "site_card.json",
        {
            "schema_version": "real_site_robot_eval_site_card.v0.1",
            "scene_id": "demo-scene-1",
            "capture_id": "demo-capture-1",
            "site_id": "demo-site-1",
            "site_type": "stockroom",
            "demo_bootstrap": True,
            "geometry": {
                "collider": {
                    "status": "review_input_present",
                    "collision_ready_claim_allowed": False,
                }
            },
            "provenance_rights_review_status": {
                "rights_privacy": {"blocked": False, "rights_status": "verified"}
            },
        },
    )
    write_json(
        robot_eval_dir / "task_cards.json",
        {
            "schema_version": "real_site_robot_eval_task_cards.v0.1",
            "task_card_count": 1,
            "cards": [
                {
                    "task_card_id": "task_card_place_return_in_bin",
                    "task_id": "place_return_in_bin",
                    "task_statement": "Place the return item in the labeled bin",
                    "task_category": "pick_place",
                    "required_metrics": [
                        "success_rate",
                        "cycle_time",
                        "intervention_rate",
                        "collision_risk",
                        "object_drop",
                        "wrong_object",
                        "timeout",
                        "placement_accuracy",
                    ],
                    "claim_boundary": "task_card_defines_eval_scope_not_robot_execution",
                }
            ],
        },
    )
    write_json(
        robot_eval_dir / "scenario_cards.json",
        {
            "schema_version": "real_site_robot_eval_scenario_cards.v0.1",
            "scenario_card_count": 1,
            "cards": [
                {
                    "scenario_card_id": "scenario_card_place_return_in_bin_tabletop",
                    "scenario_id": "scenario_place_return_in_bin_tabletop",
                    "task_id": "place_return_in_bin",
                    "robot_profile_id": "cartesian_ee_proxy_v1",
                    "target_object_ids": ["target_item"],
                    "target_objects": [
                        {
                            "object_id": "target_item",
                            "label": "return item",
                            "task_role": "target_object",
                            "center_xyz": [0.10, -0.12, 0.45],
                            "has_collision_hulls": True,
                            "has_support_surfaces": True,
                        }
                    ],
                    "start_zone": [0.0, 0.0, 0.66],
                    "goal_zone": [0.02, 0.16, 0.45],
                    "start_zone_id": "start_zone_tabletop_home",
                    "goal_zone_id": "goal_zone_tabletop_bin",
                    "spawn_candidates": [
                        {
                            "zone_id": "start_zone_tabletop_home",
                            "role": "robot_spawn",
                            "pose_xyz": [0.0, 0.0, 0.66],
                            "validation_status": "validated_finite_site_pose",
                            "validated": True,
                            "label_source": "demo_bootstrap_tabletop_layout",
                        }
                    ],
                    "target_candidates": [
                        {
                            "zone_id": "goal_zone_tabletop_bin",
                            "role": "task_goal",
                            "pose_xyz": [0.02, 0.16, 0.45],
                            "validation_status": "validated_finite_site_pose",
                            "validated": True,
                            "label_source": "demo_bootstrap_tabletop_layout",
                        }
                    ],
                    "semantic_spawn_target": {
                        "validated_spawn_target_pair": True,
                        "validated_spawn_candidate_count": 1,
                        "validated_target_candidate_count": 1,
                        "source": "demo_bootstrap_tabletop_layout",
                        "fallback_allowed_for_beta_release": False,
                    },
                    "normal_scenario": {
                        "statement": "Run the pick-place under the base tabletop layout.",
                        "ground_truth_status": "derived_from_capture_package",
                    },
                    "variation": {
                        "statement": "Run under object pose variation.",
                        "ground_truth_status": "derived_needs_review",
                    },
                    "edge_case": {
                        "statement": "Obstacle between item and bin.",
                        "ground_truth_status": "agent_inferred_needs_review",
                    },
                    "observed_vs_inferred_labels": {
                        "layout": "demo_bootstrap",
                        "variation": "derived",
                        "edge_case": "agent_inferred",
                    },
                    "required_missing_annotations": [
                        "needs_robot_pov",
                        "needs_actual_outcome",
                    ],
                    "claim_boundary": "scenario_card_is_review_scope_not_simulator_or_pilot_result",
                }
            ],
        },
    )
    write_json(
        robot_eval_dir / "eval_cards.json",
        {
            "schema_version": "real_site_robot_eval_eval_cards.v0.1",
            "eval_card_count": 1,
            "cards": [
                {
                    "eval_card_id": "eval_card_place_return_in_bin_tabletop",
                    "scenario_id": "scenario_place_return_in_bin_tabletop",
                    "task_id": "place_return_in_bin",
                    "prediction_source": "closed_loop_classical_sim",
                    "engine_used": "mujoco",
                    "validation": {"actual_status": "needs_actual_outcome"},
                    "blocked_upgrades": ["real_pilot_outcome_proven"],
                    "proof_boundary": "sim_rollout_no_actual_outcome_no_deployment_claim",
                }
            ],
        },
    )
    write_json(
        robot_eval_dir / "proof_boundaries.json",
        {
            "schema_version": "real_site_robot_eval_proof_boundaries.v0.1",
            "simulator_execution_proven": False,
            "physics_contact_validation_proven": False,
            "robot_policy_execution_proven": False,
            "non_ranking_operational_claim_proven": False,
            "real_pilot_outcome_proven": False,
            "generated_scenarios_are_real_world_proof": False,
        },
    )
    return capture_root


def build_real_policy_eval_job_request(
    *,
    capture_root: Path,
    family: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    family_id = _string(family.get("family_id"))
    return {
        "schema_version": "robot_eval_job_request.v1",
        "customer": {
            "id": "blueprint-real-policy-validation",
            "name": "Blueprint real policy family validation",
        },
        "site_package": {
            "capture_root": str(capture_root),
            "site_id": "demo-site-1",
            "package_uri": f"file://{capture_root}/pipeline",
        },
        "requested_tasks": [
            {
                "task_id": "place_return_in_bin",
                "scenario_ids": ["scenario_place_return_in_bin_tabletop"],
            }
        ],
        "robot_profile": {
            "robot_profile_id": "cartesian_ee_proxy_v1",
            "embodiment": "cartesian_ee_proxy",
            "sensors": ["proprioceptive_state"],
        },
        "policy_package": {
            "sim_controller_plugin": {
                "simulator_framework": "mujoco",
                "plugin_uri": "module://blueprint_pipeline.real_policy_closed_loop_rollout",
                "policy_family_id": family_id,
                "checkpoint_format": "lerobot_pretrained_dir",
                "checkpoint_sha256": checkpoint_sha256,
            }
        },
        "operation": "evaluate_only",
        "policy_candidates": [
            {
                "policy_id": family_id,
                "display_name": f"{family_id} (LeRobot-format checkpoint)",
                "candidate_role": "real_policy_family_under_validation",
                "source": "real_policy_family_eval_harness",
                "checkpoint_sha256": checkpoint_sha256,
                "reference_only": False,
                "candidate_behavior_distinctness_proven": False,
                "robot_team_policy_execution_proven": False,
            }
        ],
        "simulator_preference": "mujoco",
        "evaluation_substrate": REAL_SUBSTRATE,
        "cosmos_training_preference": {"mode": "export_only"},
        "budget": {"budget_usd": 0.0, "timeout_seconds": 600},
        "rights_privacy_scope": {
            "status": "cleared_for_robot_eval",
            "external_use_allowed": True,
            "privacy_scope": "synthetic_demo_bootstrap_scene_no_real_capture_content",
        },
        "owner_system": {
            "name": "blueprint-real-policy-validation",
            "request_id": f"real-policy-{family_id}",
        },
        "provenance": {
            "submitted_at": utc_now_iso(),
            "timestamp_alignment": "not_applicable_internal_validation",
        },
    }


def rollout_simulator_command(
    *,
    family: Mapping[str, Any],
    python_executable: str | None = None,
) -> str:
    executable = _string(python_executable) or sys.executable
    parts = [
        shlex.quote(executable),
        "-m",
        "blueprint_pipeline.real_policy_closed_loop_rollout",
        "--control-hz",
        f"{float(family.get('control_hz') or 20.0):g}",
        "--chunk-size",
        str(int(family.get("chunk_size") or 25)),
        "--executed-horizon",
        str(int(family.get("executed_horizon") or 16)),
        "--max-seconds",
        f"{float(family.get('max_rollout_seconds') or 20.0):g}",
    ]
    checkpoint_dir = _string(family.get("checkpoint_dir"))
    if checkpoint_dir:
        parts.extend(["--checkpoint", shlex.quote(checkpoint_dir)])
    adapter_command = _string(family.get("adapter_command"))
    if adapter_command:
        parts.extend(["--adapter-command", shlex.quote(adapter_command)])
        mode = _string(family.get("adapter_mode")) or "persistent"
        parts.extend(["--adapter-mode", mode])
    if family.get("render_obs_frames"):
        parts.append("--render-obs-frames")
    return " ".join(parts)


def family_adapter_command(
    *,
    family: Mapping[str, Any],
    python_executable: str | None = None,
) -> str:
    adapter_command = _string(family.get("adapter_command"))
    if adapter_command:
        return adapter_command
    executable = _string(python_executable) or sys.executable
    return " ".join(
        [
            shlex.quote(executable),
            "-m",
            "blueprint_pipeline.lerobot_policy_family",
            "--checkpoint",
            shlex.quote(_string(family.get("checkpoint_dir"))),
        ]
    )


def verify_real_substrate_task_eval_report(job_dir: str | Path) -> dict[str, Any]:
    """Fail-closed check that the report came from a real policy rollout."""
    resolved = Path(job_dir).resolve()
    blockers: list[str] = []

    trace = _read_json(resolved / "normalized_attempt_trace.json")
    attempts = [
        _mapping(row) for row in trace.get("attempts") or [] if isinstance(row, Mapping)
    ]
    if not attempts:
        blockers.append("normalized_attempt_trace_empty")
    runners = sorted({_string(row.get("runner")) for row in attempts})
    if runners and runners != ["command_adapter"]:
        blockers.append(f"attempt_runner_not_command_adapter:{','.join(runners)}")
    if _string(trace.get("backend")) != "mujoco":
        blockers.append("attempt_trace_backend_not_mujoco")
    if trace.get("simulator_execution_proven") is not True:
        blockers.append("simulator_execution_not_proven")

    simulator_result = _read_json(resolved / "simulator_service_result.json")
    if _string(simulator_result.get("framework")) == "fixture":
        blockers.append("simulator_framework_is_fixture")

    simulator_output: dict[str, Any] = {}
    for candidate in resolved.glob("*_simulator_output.json"):
        simulator_output = _read_json(candidate)
        break
    if not simulator_output:
        blockers.append("simulator_output_missing")
    else:
        if simulator_output.get("policy_in_the_loop") is not True:
            blockers.append("simulator_output_not_policy_in_the_loop")
        if _string(simulator_output.get("substrate")) != REAL_SUBSTRATE:
            blockers.append("simulator_output_substrate_not_classical_sim_mujoco")
        criteria_sources = {
            _string(_mapping(row.get("task_outcome")).get("success_criteria_source"))
            for row in _mapping(simulator_output).get("attempts") or []
            if isinstance(row, Mapping)
        }
        if criteria_sources and criteria_sources != {"measured_simulator_state"}:
            blockers.append("success_criteria_not_measured_from_simulator_state")

    report = _read_json(resolved / "task_eval_run_report.json")
    if not report:
        blockers.append("task_eval_run_report_missing")
    scorecard = _mapping(report.get("scorecard"))
    conditions = [
        _mapping(row)
        for row in scorecard.get("conditions") or []
        if isinstance(row, Mapping)
    ]
    trials = sum(int(row.get("trials") or 0) for row in conditions)
    if trials <= 0:
        blockers.append("task_eval_run_report_scorecard_has_no_trials")

    sc3 = _read_json(resolved / "sc3_eval_protocol.json")
    requery = _mapping(_mapping(sc3.get("data_requirements")).get("policy_requery_trace"))

    return {
        "real_rollout_report": not blockers,
        "blockers": blockers,
        "attempt_count": len(attempts),
        "attempt_runners": runners,
        "scorecard_trials": trials,
        "task_eval_run_report_status": _string(report.get("status")),
        "task_eval_run_report_evidence_level": _string(report.get("evidence_level")),
        "sc3_policy_requery_trace_status": _string(requery.get("status")),
        "task_success_summary": _mapping(trace.get("task_success_summary")),
    }


def register_family_in_validation_ladder(
    *,
    job_dir: Path,
    family: Mapping[str, Any],
    checkpoint_sha256: str,
    verification: Mapping[str, Any],
    generated_at: str,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Validation-ladder-only registration; production stays blocked."""
    family_id = _string(family.get("family_id"))
    adapter_command = family_adapter_command(
        family=family, python_executable=python_executable
    )
    ladder = build_known_ordering_policy_ladder(
        inner_policy_id=family_id,
        inner_command=adapter_command,
        generated_at=generated_at,
    )
    write_json(job_dir / "real_policy_family_ranking_ladder.json", ladder)
    registry = {
        "schema_version": FAMILY_REGISTRY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "family_id": family_id,
        "checkpoint_sha256": checkpoint_sha256,
        "registered_in": "validation_ladder_only",
        "ranking_ladder_path": "real_policy_family_ranking_ladder.json",
        "closed_loop_eval_verified": verification.get("real_rollout_report") is True,
        "integration_proof_scope": family.get("integration_proof_scope"),
        "source_policy": _mapping(family.get("source_policy")),
        "gpu_runtime": _mapping(family.get("gpu_runtime")),
        "meaningful_manipulator_scoring": _mapping(
            family.get("meaningful_manipulator_scoring")
        ),
        "production_candidate_registration": {
            "registered": False,
            "registry": "UNITREE_ACTION_COMMAND_CANDIDATES",
            "blocked_until": "real_gpu_provider_run_with_provider_reliability_manifest",
            "reason": (
                "CPU closed-loop validation proves plumbing and ranker "
                "discriminability inputs only; production candidacy requires a "
                "real GPU run of the actual learned policy."
            ),
        },
        "gpu_swap_contract": {
            "config_only": True,
            "changes_required": [
                "checkpoint_dir -> learned LeRobot checkpoint",
                "adapter_command -> learned-policy inference wrapper command",
                "requires_gpu -> true",
            ],
            "harness_code_change_required": False,
        },
        "claim_boundary": {
            "validation_ladder_registration_is_not_production_candidacy": True,
            "closed_loop_cpu_eval_is_not_gpu_or_physical_proof": True,
            **_mapping(family.get("claim_boundary")),
            "public_claim_upgrade_allowed": False,
        },
    }
    write_json(job_dir / "real_policy_family_registry.json", registry)
    return registry


def run_real_policy_family_eval(
    *,
    capture_root: str | Path,
    job_id: str,
    family_config: Mapping[str, Any],
    allow_simulator_execution: bool = False,
    bootstrap_demo_capture_root: bool = False,
    timeout_seconds: int = 600,
    python_executable: str | None = None,
) -> dict[str, Any]:
    import os

    from .robot_eval_job_orchestrator import build_robot_eval_job

    generated_at = utc_now_iso()
    family = _mapping(family_config)
    family_id = _string(family.get("family_id")) or "unnamed_policy_family"
    capture_path = Path(capture_root).expanduser().resolve()
    if bootstrap_demo_capture_root:
        write_demo_pick_place_capture_root(capture_path)

    loaded = load_lerobot_policy_checkpoint(_string(family.get("checkpoint_dir")))
    manifest_blockers: list[str] = []
    if loaded.blockers and not _string(family.get("adapter_command")):
        manifest_blockers.extend(loaded.blockers)
    if not allow_simulator_execution:
        manifest_blockers.append("simulator_execution_not_allowed_by_caller")

    if manifest_blockers:
        manifest = {
            "schema_version": HARNESS_MANIFEST_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "family_id": family_id,
            "blockers": sorted(set(manifest_blockers)),
        }
        return manifest

    os.environ["BLUEPRINT_ALLOW_SIMULATOR_EXECUTION"] = "true"
    simulator_command = rollout_simulator_command(
        family=family, python_executable=python_executable
    )
    request = build_real_policy_eval_job_request(
        capture_root=capture_path,
        family=family,
        checkpoint_sha256=loaded.checkpoint_sha256,
    )
    policy_execution_command = _string(family.get("policy_execution_command"))
    policy_execution_commands = (
        {"sim_controller_plugin": policy_execution_command}
        if policy_execution_command
        else None
    )
    if policy_execution_commands:
        os.environ["BLUEPRINT_ALLOW_POLICY_EXECUTION"] = "true"
    result = build_robot_eval_job(
        capture_root=capture_path,
        job_request=request,
        job_id=job_id,
        provisioner="fixture_local",
        simulator="mujoco",
        evaluation_substrate=REAL_SUBSTRATE,
        allow_simulator_execution=True,
        allowed_simulators=("mujoco",),
        simulator_commands={"mujoco": simulator_command},
        allow_policy_execution=bool(policy_execution_commands),
        policy_execution_commands=policy_execution_commands,
        timeout_seconds=timeout_seconds,
    )

    job_dir = capture_path / "pipeline" / "robot_eval_jobs" / job_id
    verification = verify_real_substrate_task_eval_report(job_dir)
    registry = register_family_in_validation_ladder(
        job_dir=job_dir,
        family=family,
        checkpoint_sha256=loaded.checkpoint_sha256,
        verification=verification,
        generated_at=generated_at,
        python_executable=python_executable,
    )

    status = (
        "real_closed_loop_eval_completed"
        if verification.get("real_rollout_report") is True
        else "blocked_report_not_real_substrate"
    )
    manifest = {
        "schema_version": HARNESS_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "family_id": family_id,
        "family_config": dict(family),
        "integration_proof_scope": family.get("integration_proof_scope"),
        "source_policy": _mapping(family.get("source_policy")),
        "gpu_runtime": _mapping(family.get("gpu_runtime")),
        "meaningful_manipulator_scoring": _mapping(
            family.get("meaningful_manipulator_scoring")
        ),
        "checkpoint_manifest": loaded.manifest(),
        "job_id": job_id,
        "job_dir": str(job_dir),
        "orchestrator_status": _string(_mapping(result).get("status")),
        "simulator_command": simulator_command,
        "verification": verification,
        "validation_ladder_registration": {
            "registered_in": registry.get("registered_in"),
            "production_registered": _mapping(
                registry.get("production_candidate_registration")
            ).get("registered"),
        },
        "artifact_paths": {
            "task_eval_run_report": str(job_dir / "task_eval_run_report.json"),
            "sc3_eval_protocol": str(job_dir / "sc3_eval_protocol.json"),
            "normalized_attempt_trace": str(job_dir / "normalized_attempt_trace.json"),
            "real_policy_family_registry": str(
                job_dir / "real_policy_family_registry.json"
            ),
            "real_policy_family_ranking_ladder": str(
                job_dir / "real_policy_family_ranking_ladder.json"
            ),
        },
        "claim_boundary": {
            "closed_loop_cpu_eval_is_not_gpu_or_physical_proof": True,
            "report_truth_comes_from_task_eval_run_report_ledger": True,
            **_mapping(family.get("claim_boundary")),
            "public_claim_upgrade_allowed": False,
        },
    }
    write_json(job_dir / "real_policy_family_eval_harness_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--family-config", default=None, help="Policy family config JSON path"
    )
    parser.add_argument(
        "--scripted-baseline-checkpoint",
        default=None,
        help=(
            "Create (if needed) and use the scripted CPU baseline checkpoint at "
            "this directory instead of --family-config"
        ),
    )
    parser.add_argument(
        "--groot-libero-checkpoint",
        nargs="?",
        const=GROOT_LIBERO_CHECKPOINT_REPO_ID,
        default=None,
        help=(
            "Use the NVIDIA GR00T N1.7 LIBERO/Panda LeRobot checkpoint as a "
            "GPU learned-policy integration proof."
        ),
    )
    parser.add_argument("--groot-libero-device", default="cuda")
    parser.add_argument("--allow-simulator-execution", action="store_true")
    parser.add_argument("--bootstrap-demo-capture-root", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args(argv)

    if args.family_config:
        family = _mapping(
            json.loads(Path(args.family_config).read_text(encoding="utf-8"))
        )
    elif args.scripted_baseline_checkpoint:
        checkpoint = create_scripted_baseline_checkpoint(
            args.scripted_baseline_checkpoint
        )
        family = default_family_config(
            family_id="blueprint_scripted_pick_place_v1",
            checkpoint_dir=str(checkpoint),
        )
        family["render_obs_frames"] = True
    elif args.groot_libero_checkpoint:
        family = default_groot_libero_family_config(
            checkpoint=args.groot_libero_checkpoint,
            device=args.groot_libero_device,
        )
    else:
        print(
            "[real-policy-eval] FAILED: provide --family-config, "
            "--scripted-baseline-checkpoint, or --groot-libero-checkpoint"
        )
        return 1

    manifest = run_real_policy_family_eval(
        capture_root=args.capture_root,
        job_id=args.job_id,
        family_config=family,
        allow_simulator_execution=args.allow_simulator_execution,
        bootstrap_demo_capture_root=args.bootstrap_demo_capture_root,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"[real-policy-eval] status={manifest.get('status')}")
    verification = _mapping(manifest.get("verification"))
    print(
        "[real-policy-eval] real_rollout_report="
        f"{verification.get('real_rollout_report')} "
        f"attempts={verification.get('attempt_count')} "
        f"trials={verification.get('scorecard_trials')}"
    )
    for blocker in verification.get("blockers") or []:
        print(f"[real-policy-eval] blocker={blocker}")
    report_path = _mapping(manifest.get("artifact_paths")).get("task_eval_run_report")
    if report_path:
        print(f"[real-policy-eval] task_eval_run_report={report_path}")
    return 0 if manifest.get("status") == "real_closed_loop_eval_completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
