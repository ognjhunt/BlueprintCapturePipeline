"""Compatibility adapter from the legacy G1/kitchen lane to Evaluation Run."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from .evaluation_run_contract import EVALUATION_RUN_SCHEMA_VERSION, EvaluationRunSpec


G1_KITCHEN_SCENE_ADAPTER = "openusd_scene_bundle"
G1_KITCHEN_ROBOT_ADAPTER = "isaac_unitree_g1"
G1_KITCHEN_TASK_PACK_ADAPTER = "manifest_task_scenario_pack"
G1_KITCHEN_RUNTIME_ADAPTER = "isaac_provider_runtime"
G1_KITCHEN_PROOF_ADAPTER = "declared_evidence_proof_contract"
G1_KITCHEN_EXECUTION_ADAPTER = "isaac_g1_kitchen_parity_compatibility"
G1_KITCHEN_CONTEXT_OPTION_KEYS = frozenset(
    {
        "allow_dirty_paid_launch",
        "articulated",
        "audit_boost_light_intensity",
        "audit_high_spp",
        "audit_warmup_frames",
        "capture_every",
        "cheap_collision",
        "cold",
        "cold_race_contenders",
        "collision_approximation",
        "max_seconds",
        "marker_timeout",
        "max_attempts",
        "container_disk_gb",
        "dynamic_episode_check_every",
        "dynamic_episode_termination",
        "dynamic_standing_contact_steps",
        "episode_max_steps",
        "fill_light_intensity",
        "focus_radius",
        "groot_policy_command",
        "groot_policy_command_timeout_seconds",
        "keep_objects",
        "kinematic_arm_pose",
        "manipulation_cam",
        "manipulation_look_at",
        "manipulation_reach",
        "manipulation_reach_arm",
        "manipulation_stand",
        "neutral_environment",
        "no_collision_probe",
        "placement_topdown_capture",
        "post_marker_progress_timeout",
        "physics_articulation_drive",
        "render_noise_audit",
        "render_subframes",
        "robot_review_material_mode",
        "robot_review_material_override",
        "serve_idle_timeout_s",
        "serve_max_jobs",
        "serve_ready_timeout",
        "settle_seconds",
        "startup_no_runtime_timeout",
        "supervised_startup",
        "vast_max_hourly_rate_usd",
        "verify_cam",
        "volume_gb",
        "warm_candidates",
        "warm_only",
        "worker_image_manifest_diagnostic",
    }
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._:-]+", "-", value).strip("-._:")
    return normalized[:160] or "evaluation-run"


def _sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _evidence_uri(value: str | None) -> str:
    """Retain resource identity without persisting signed URL credentials."""

    raw = _string(value)
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if parsed.scheme in {"http", "https"}:
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return raw


def _run_id(
    *,
    out_dir: str | Path,
    scenarios: Sequence[Mapping[str, Any]],
    policy_id: str,
) -> str:
    identity = json.dumps(
        {
            "out_dir": str(Path(out_dir).expanduser()),
            "scenario_ids": [
                _string(row.get("scenario_id") or row.get("id")) for row in scenarios
            ],
            "policy_id": policy_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return _safe_id(f"{Path(out_dir).name}-{suffix}")


def _task_pack(scenarios: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized_scenarios: list[dict[str, Any]] = []
    task_ids: list[str] = []
    for index, raw in enumerate(scenarios):
        scenario = dict(raw)
        scenario_id = _string(scenario.get("scenario_id") or scenario.get("id"))
        task_id = _string(scenario.get("task_id")) or "g1_kitchen_parity_task"
        normalized_scenarios.append(
            {
                **scenario,
                "scenario_id": scenario_id or f"scenario-{index + 1}",
                "task_id": task_id,
            }
        )
        if task_id not in task_ids:
            task_ids.append(task_id)
    return {
        "adapter_id": G1_KITCHEN_TASK_PACK_ADAPTER,
        "adapter_version": "1",
        "pack_id": "g1-kitchen-parity",
        "tasks": [{"task_id": task_id} for task_id in task_ids],
        "scenarios": normalized_scenarios,
        "source_contract": "legacy_g1_kitchen_scenarios",
    }


def _policy_adapter(policy_id: str) -> dict[str, Any]:
    learned = policy_id in {
        "groot_sonic",
        "groot",
        "groot_n17_sonic",
        "unitree_groot_n17_sonic_policy",
    }
    return {
        "adapter_id": (
            "unitree_groot_n17_sonic" if learned else "isaac_g1_deterministic_controller"
        ),
        "adapter_version": "1",
        "policy_id": policy_id,
        "observation_schema_ref": "blueprint://schemas/robot_eval_observation.v1",
        "action_schema_ref": "blueprint://schemas/robot_eval_action_trace.v1",
        "execution_kind": "persistent_worker_or_command" if learned else "in_process",
    }


def build_g1_kitchen_evaluation_run_spec(
    *,
    out_dir: str | Path,
    run_id: str | None = None,
    scenarios: Sequence[Mapping[str, Any]],
    kitchen_uri: str | None,
    kitchen_main_usd_relative: str,
    kitchen_asset_inventory: Mapping[str, Any] | None,
    g1_usd: str,
    policy_id: str,
    providers: Sequence[str],
    selected_image: str,
    allow_paid: bool,
    max_spend_usd: float | None,
    image_startup_canary: bool,
    serve: bool,
    requested_render_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate the legacy lane into the stable six-part run interface."""

    inventory = dict(kitchen_asset_inventory or {})
    archive_sha = _string(inventory.get("archive_sha256"))
    content_digest = (
        archive_sha if archive_sha.startswith("sha256:") else f"sha256:{archive_sha}"
    ) if archive_sha else None
    mode = "startup_canary" if image_startup_canary else "serve" if serve else "evaluate"
    scene_uri = _evidence_uri(kitchen_uri)
    identity_status = "verified" if content_digest else "legacy_unverified"
    if mode == "startup_canary" and not scene_uri:
        identity_status = "not_required_for_startup_canary"
    elif not scene_uri:
        # Historical dry-run callers prepared a bundle before choosing or
        # staging a kitchen archive. Preserve that behavior as an explicit,
        # non-proof compatibility reference instead of pretending an asset was
        # resolved.
        scene_uri = "legacy://g1-kitchen-scene-unmaterialized"
    return {
        "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
        "run_id": _safe_id(run_id) if run_id else _run_id(
            out_dir=out_dir, scenarios=scenarios, policy_id=policy_id
        ),
        "mode": mode,
        "scene_bundle": {
            "adapter_id": G1_KITCHEN_SCENE_ADAPTER,
            "adapter_version": "1",
            "bundle_id": "g1-kitchen-scene",
            "uri": scene_uri or None,
            "entrypoint": kitchen_main_usd_relative if mode != "startup_canary" else None,
            "format": "openusd",
            "content_digest": content_digest,
            "identity_status": identity_status,
            "inventory": {
                "file_count": inventory.get("file_count"),
                "total_bytes": inventory.get("total_bytes"),
            },
        },
        "robot_adapter": {
            "adapter_id": G1_KITCHEN_ROBOT_ADAPTER,
            "adapter_version": "1",
            "robot_profile_id": "unitree_g1",
            "asset_ref": g1_usd if mode != "startup_canary" else g1_usd or None,
            "simulator": "isaac_sim",
        },
        "task_scenario_pack": _task_pack(scenarios),
        "policy_adapter": _policy_adapter(policy_id),
        "runtime_provider_profile": {
            "adapter_id": G1_KITCHEN_RUNTIME_ADAPTER,
            "adapter_version": "1",
            "execution_adapter_id": G1_KITCHEN_EXECUTION_ADAPTER,
            "profile_id": "isaac-gpu-review",
            "providers": list(providers),
            "simulator": "isaac_sim",
            "worker_image_ref": selected_image or None,
            "paid_execution_requested": bool(allow_paid),
            "max_spend_usd": max_spend_usd,
            "render_settings": dict(requested_render_settings),
        },
        "proof_contract": {
            "adapter_id": G1_KITCHEN_PROOF_ADAPTER,
            "adapter_version": "1",
            "contract_id": "g1-kitchen-attempt-closure",
            "contract_schema_version": "g1_kitchen_attempt_closure.v1",
            "required_evidence": [
                "provider_startup",
                "worker_runtime",
                "policy_action_trace",
                "review_media",
                "task_state_change",
                "terminal_provider_teardown",
            ],
            "claim_ceiling": {
                "startup_only": bool(image_startup_canary),
                "simulator_task_success_requires_attempt_closure": True,
                "physical_robot_readiness": False,
                "deployment_readiness": False,
            },
            "prohibited_claims": [
                "physical_robot_readiness",
                "deployment_approval",
                "field_safety",
                "task_success_without_attempt_closure",
            ],
        },
        "metadata": {
            "compatibility_source": "isaac_g1_kitchen_parity_job.v1",
            "legacy_lane_retained_for_artifact_read_compatibility": True,
            "scene_specific_name_is_not_platform_contract": True,
            "source_identity": _sha256(
                f"{kitchen_main_usd_relative}\0{g1_usd}\0{policy_id}"
            ),
        },
    }


def g1_kitchen_job_kwargs_from_evaluation_run(
    spec: EvaluationRunSpec,
    *,
    output_dir: str | Path,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive legacy executor arguments from the authoritative run spec."""

    runtime = dict(spec.runtime_provider_profile)
    render = dict(runtime.get("render_settings") or {})
    task_pack = dict(spec.task_scenario_pack)
    options = dict(context.get("options") or {})
    unsupported = sorted(set(options) - G1_KITCHEN_CONTEXT_OPTION_KEYS)
    if unsupported:
        raise ValueError(f"unsupported_g1_kitchen_execution_options:{','.join(unsupported)}")
    providers = runtime.get("providers") or []
    if isinstance(providers, str):
        providers = [providers]
    scene_transport_uri = _string(context.get("scene_transport_uri"))
    scene_asset_dir = _string(context.get("scene_asset_dir"))
    allow_paid = bool(context.get("allow_paid"))
    if allow_paid and runtime.get("paid_execution_requested") is not True:
        raise ValueError("paid_execution_not_declared_by_runtime_provider_profile")
    kwargs: dict[str, Any] = {
        "evaluation_run_id": spec.run_id,
        "scenarios": list(task_pack.get("scenarios") or []),
        "out_dir": Path(output_dir),
        "kitchen_asset_dir": scene_asset_dir or None,
        "kitchen_url": scene_transport_uri or None,
        "g1_usd": spec.robot_adapter.get("asset_ref"),
        "policy_id": spec.policy_adapter.get("policy_id"),
        "provider": ",".join(_string(value) for value in providers if _string(value)),
        "allow_paid": allow_paid,
        "image": runtime.get("worker_image_ref"),
        "max_spend_usd": runtime.get("max_spend_usd"),
        "image_startup_canary": spec.mode == "startup_canary",
        "serve": spec.mode == "serve",
    }
    render_keys = {
        "steps": "steps",
        "width": "width",
        "height": "height",
        "fps": "fps",
        "warmup_frames": "warmup",
        "per_scenario_seconds": "per_scenario_seconds",
    }
    for source, target in render_keys.items():
        if render.get(source) is not None:
            kwargs[target] = render[source]
    kwargs.update(options)
    return kwargs


class G1KitchenEvaluationRunExecutor:
    """Execute the historical kitchen implementation from the generic seam."""

    adapter_id = G1_KITCHEN_EXECUTION_ADAPTER

    def execute(
        self,
        *,
        spec: EvaluationRunSpec,
        output_dir: Path,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from .isaac_g1_kitchen_parity_job import run_isaac_g1_kitchen_parity_job

        try:
            kwargs = g1_kitchen_job_kwargs_from_evaluation_run(
                spec,
                output_dir=output_dir,
                context=context,
            )
        except ValueError as exc:
            return {
                "schema_version": "g1_kitchen_evaluation_run_execution.v1",
                "status": "blocked",
                "blockers": [str(exc)],
            }
        result = run_isaac_g1_kitchen_parity_job(**kwargs)
        return {
            **dict(result),
            "evaluation_run_execution_adapter_schema_version": (
                "g1_kitchen_evaluation_run_execution.v1"
            ),
            "evaluation_run_id": spec.run_id,
            "source_spec_is_execution_authority": True,
        }


def build_g1_kitchen_cli_evaluation_run(
    args: Any,
    *,
    scenarios: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate the legacy CLI namespace into spec plus ephemeral context."""

    render_settings = {
        "steps": int(args.steps),
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
        "warmup_frames": int(args.warmup),
        "per_scenario_seconds": int(args.per_scenario_seconds),
        "expected_frame_count_per_scenario": int(args.steps),
    }
    providers = [value.strip() for value in str(args.provider).split(",") if value.strip()]
    spec = build_g1_kitchen_evaluation_run_spec(
        out_dir=args.out_dir,
        scenarios=scenarios,
        kitchen_uri=args.kitchen_url,
        kitchen_main_usd_relative="Collected_KitchenRoom/KitchenRoom.usd",
        kitchen_asset_inventory=None,
        g1_usd=args.g1_usd,
        policy_id=args.policy,
        providers=providers,
        selected_image=args.image or "",
        allow_paid=bool(args.allow_paid),
        max_spend_usd=args.max_spend_usd,
        image_startup_canary=bool(args.image_startup_canary),
        serve=bool(args.serve),
        requested_render_settings=render_settings,
    )
    direct_options = {
        "allow_dirty_paid_launch": args.allow_dirty_paid_launch,
        "cold": args.cold,
        "max_seconds": args.max_seconds,
        "marker_timeout": args.marker_timeout,
        "max_attempts": args.max_attempts,
        "post_marker_progress_timeout": args.post_marker_progress_timeout,
        "startup_no_runtime_timeout": args.startup_no_runtime_timeout,
        "cold_race_contenders": args.cold_race_contenders,
        "container_disk_gb": args.container_disk_gb,
        "volume_gb": args.volume_gb,
        "warm_candidates": tuple(args.warm_candidate or ()),
        "warm_only": args.warm_only,
        "serve_idle_timeout_s": args.serve_idle_timeout,
        "serve_max_jobs": args.serve_max_jobs,
        "serve_ready_timeout": args.serve_ready_timeout,
        "supervised_startup": args.supervised_startup,
        "vast_max_hourly_rate_usd": args.vast_max_hourly_rate,
        "articulated": args.articulated,
        "physics_articulation_drive": args.physics_articulation_drive,
        "dynamic_standing_contact_steps": args.dynamic_standing_contact_steps,
        "cheap_collision": args.cheap_collision,
        "settle_seconds": args.settle_seconds,
        "focus_radius": args.focus_radius,
        "keep_objects": args.keep_objects,
        "manipulation_cam": args.manipulation_cam,
        "manipulation_look_at": args.manipulation_look_at,
        "render_subframes": args.render_subframes,
        "manipulation_reach": args.manipulation_reach,
        "manipulation_reach_arm": args.manipulation_reach_arm,
        "dynamic_episode_termination": not args.no_dynamic_episode_termination,
        "episode_max_steps": args.episode_max_steps,
        "dynamic_episode_check_every": args.dynamic_episode_check_every,
        "capture_every": args.capture_every,
        "fill_light_intensity": args.fill_light_intensity,
        "neutral_environment": args.neutral_environment,
        "robot_review_material_override": args.robot_review_material_override,
        "robot_review_material_mode": args.robot_review_material_mode,
        "kinematic_arm_pose": args.kinematic_arm_pose,
        "collision_approximation": args.collision_approximation,
        "verify_cam": args.verify_cam,
        "manipulation_stand": args.manipulation_stand,
        "placement_topdown_capture": not args.no_placement_topdown_capture,
        "render_noise_audit": args.render_noise_audit,
        "audit_high_spp": args.audit_high_spp,
        "audit_warmup_frames": args.audit_warmup_frames,
        "audit_boost_light_intensity": args.audit_boost_light_intensity,
        "groot_policy_command": args.groot_policy_command,
        "groot_policy_command_timeout_seconds": args.groot_policy_command_timeout_seconds,
        "worker_image_manifest_diagnostic": args.worker_image_manifest_diagnostic,
    }
    context = {
        "allow_paid": bool(args.allow_paid),
        "scene_asset_dir": args.kitchen_asset_dir,
        "scene_transport_uri": args.kitchen_url,
        "options": direct_options,
    }
    return spec, context
