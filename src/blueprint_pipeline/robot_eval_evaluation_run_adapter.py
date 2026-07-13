"""Adapt the generic robot-eval request into ``evaluation_run.v1``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import read_json, write_json
from .evaluation_run_contract import EVALUATION_RUN_SCHEMA_VERSION, EvaluationRunSpec


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _first(*values: Any) -> str:
    for value in values:
        text = _string(value)
        if text:
            return text
    return ""


def _requested_tasks(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = _rows(request.get("requested_tasks") or request.get("requestedTasks"))
    normalized: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        task_id = _first(task.get("task_id"), task.get("taskId"), task.get("id"))
        normalized.append({**task, "task_id": task_id or f"task-{index + 1}"})
    return normalized


def _matrix_scenarios(
    scenario_eval_matrix: Mapping[str, Any],
    *,
    fallback_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = _rows(
        scenario_eval_matrix.get("runs")
        or scenario_eval_matrix.get("scenario_eval_runs")
        or scenario_eval_matrix.get("scenarios")
    )
    normalized: list[dict[str, Any]] = []
    fallback_task_id = _string(fallback_tasks[0].get("task_id")) if fallback_tasks else ""
    for index, row in enumerate(candidates):
        scenario_id = _first(
            row.get("scenario_eval_run_id"),
            row.get("scenario_id"),
            row.get("scenarioId"),
            row.get("id"),
        )
        task_id = _first(row.get("task_id"), row.get("taskId"), fallback_task_id)
        normalized.append(
            {
                **row,
                "scenario_id": scenario_id or f"scenario-{index + 1}",
                "source_scenario_id": _first(
                    row.get("scenario_id"), row.get("scenarioId")
                )
                or None,
                "task_id": task_id or "unspecified-task",
            }
        )
    if normalized:
        return normalized
    for task in fallback_tasks:
        task_id = _string(task.get("task_id"))
        raw_scenarios = task.get("scenario_ids") or task.get("scenarioIds") or []
        if isinstance(raw_scenarios, str):
            raw_scenarios = [raw_scenarios]
        if isinstance(raw_scenarios, list):
            for scenario_id in raw_scenarios:
                if _string(scenario_id):
                    normalized.append(
                        {"scenario_id": _string(scenario_id), "task_id": task_id}
                    )
    return normalized


def _policy_binding(
    request: Mapping[str, Any], policy_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    policy_package = _mapping(
        request.get("policy_package") or request.get("policyPackage")
    )
    selected_modalities = policy_manifest.get("selected_modalities") or []
    if not isinstance(selected_modalities, list):
        selected_modalities = []
    policy_id = _first(
        request.get("policy_id"),
        request.get("policyId"),
        policy_package.get("policy_id"),
        policy_package.get("policyId"),
        _mapping(request.get("default_test_policy")).get("policy_id"),
        "request-policy",
    )
    return {
        "adapter_id": "robot_eval_policy_package",
        "adapter_version": "1",
        "policy_id": policy_id,
        "observation_schema_ref": "blueprint://schemas/robot_eval_observation.v1",
        "action_schema_ref": "blueprint://schemas/robot_eval_action_trace.v1",
        "selected_modalities": selected_modalities,
        "policy_package": policy_package,
        "source_manifest_schema_version": policy_manifest.get("schema_version"),
    }


def build_robot_eval_evaluation_run_spec(
    *,
    job_id: str,
    request: Mapping[str, Any],
    capture_root: str | Path,
    scene_preflight: Mapping[str, Any],
    scenario_eval_matrix: Mapping[str, Any],
    policy_manifest: Mapping[str, Any],
    provisioner: str,
    simulator: str,
    budget_usd: float | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Build the canonical six-part interface from a robot-eval request."""

    site_package = _mapping(request.get("site_package") or request.get("sitePackage"))
    robot_profile = _mapping(request.get("robot_profile") or request.get("robotProfile"))
    tasks = _requested_tasks(request)
    scenarios = _matrix_scenarios(scenario_eval_matrix, fallback_tasks=tasks)
    scene_uri = _first(
        site_package.get("package_uri"),
        site_package.get("packageUri"),
        site_package.get("capture_root"),
        request.get("capture_root"),
        str(Path(capture_root)),
    )
    scene_entrypoint = _first(
        site_package.get("scene_entrypoint"),
        site_package.get("sceneEntrypoint"),
        scene_preflight.get("selected_scene_path"),
        scene_preflight.get("scene_path"),
        "capture_root",
    )
    scene_digest = _first(
        site_package.get("content_digest"),
        site_package.get("contentDigest"),
        scene_preflight.get("content_digest"),
    )
    if scene_digest and not scene_digest.startswith("sha256:"):
        scene_digest = f"sha256:{scene_digest}"
    robot_profile_id = _first(
        robot_profile.get("robot_profile_id"),
        robot_profile.get("robotProfileId"),
        robot_profile.get("id"),
        "request-robot-profile",
    )
    robot_asset_ref = _first(
        robot_profile.get("asset_ref"),
        robot_profile.get("assetRef"),
        robot_profile.get("usd_ref"),
        robot_profile.get("urdf_ref"),
        f"profile://{robot_profile_id}",
    )
    return {
        "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
        "run_id": job_id,
        "mode": "evaluate",
        "scene_bundle": {
            "adapter_id": "capture_site_scene_bundle",
            "adapter_version": "1",
            "bundle_id": _first(
                site_package.get("site_id"),
                site_package.get("siteId"),
                request.get("scene_id"),
                job_id,
            ),
            "uri": scene_uri,
            "entrypoint": scene_entrypoint,
            "content_digest": scene_digest or None,
            "identity_status": "verified" if scene_digest else "legacy_unverified",
            "source_preflight_schema_version": scene_preflight.get("schema_version"),
        },
        "robot_adapter": {
            "adapter_id": "robot_profile_adapter",
            "adapter_version": "1",
            "robot_profile_id": robot_profile_id,
            "asset_ref": robot_asset_ref,
            "embodiment": robot_profile.get("embodiment"),
            "sensors": robot_profile.get("sensors") or [],
        },
        "task_scenario_pack": {
            "adapter_id": "robot_eval_matrix_task_scenario_pack",
            "adapter_version": "1",
            "pack_id": f"{job_id}-task-scenarios",
            "tasks": tasks,
            "scenarios": scenarios,
            "source_matrix_schema_version": scenario_eval_matrix.get("schema_version"),
        },
        "policy_adapter": _policy_binding(request, policy_manifest),
        "runtime_provider_profile": {
            "adapter_id": "robot_eval_runtime_provider",
            "adapter_version": "1",
            "execution_adapter_id": "robot_eval_job_orchestrator",
            "profile_id": f"{provisioner}-{simulator}",
            "providers": [provisioner],
            "simulator": simulator,
            "max_spend_usd": budget_usd,
            "timeout_seconds": int(timeout_seconds),
        },
        "proof_contract": _mapping(request.get("evaluation_run_proof_contract")) or {
            "adapter_id": "robot_eval_proof_contract",
            "adapter_version": "1",
            "contract_id": "robot-eval-job-proof-boundary",
            "contract_schema_version": "robot_eval_job_proof_boundary.v1",
            "required_evidence": [
                "scene_identity",
                "scenario_eval_matrix",
                "policy_execution_trace",
                "simulator_result",
                "artifact_freshness",
                "provider_teardown_when_paid",
            ],
            "claim_ceiling": {
                "simulator_execution_requires_runtime_evidence": True,
                "policy_execution_requires_action_trace": True,
                "task_success_requires_declared_task_contract": True,
                "physical_robot_readiness": False,
                "deployment_approval": False,
            },
            "prohibited_claims": [
                "physical_robot_readiness",
                "deployment_approval",
                "safety_validation_without_owner_evidence",
                "task_success_without_task_contract",
            ],
            "rights_privacy_scope": _mapping(
                request.get("rights_privacy_scope")
                or request.get("rightsPrivacyScope")
            ),
        },
        "metadata": {
            "source_contract": request.get("schema_version"),
            "source_job_id": job_id,
            "operation": request.get("operation") or "evaluate_only",
            "evaluation_substrate": request.get("evaluation_substrate")
            or request.get("evaluationSubstrate"),
            "capture_first": True,
            "scene_specific_names_are_adapter_configuration": True,
        },
    }


def robot_eval_job_request_from_evaluation_run(
    spec: EvaluationRunSpec,
    *,
    capture_root: str | Path,
    spec_digest: str | None = None,
) -> dict[str, Any]:
    """Translate the authoritative run contract into the legacy job request."""

    scene = dict(spec.scene_bundle)
    robot = dict(spec.robot_adapter)
    task_pack = dict(spec.task_scenario_pack)
    policy = dict(spec.policy_adapter)
    runtime = dict(spec.runtime_provider_profile)
    proof = dict(spec.proof_contract)
    tasks = _rows(task_pack.get("tasks"))
    scenarios = _rows(task_pack.get("scenarios"))
    requested_tasks: list[dict[str, Any]] = []
    for task in tasks:
        task_id = _first(task.get("task_id"), task.get("id"))
        scenario_ids = [
            _first(row.get("source_scenario_id"), row.get("scenario_id"), row.get("id"))
            for row in scenarios
            if _first(row.get("task_id"), row.get("taskId")) == task_id
        ]
        requested_tasks.append(
            {
                **task,
                "task_id": task_id,
                "scenario_ids": list(dict.fromkeys(value for value in scenario_ids if value)),
            }
        )
    policy_package = _mapping(policy.get("policy_package"))
    if not policy_package:
        policy_package = {
            "sim_controller_plugin": {
                "simulator_framework": _string(runtime.get("simulator")),
                "plugin_uri": f"adapter://{_string(policy.get('adapter_id'))}",
            }
        }
    return {
        "schema_version": "robot_eval_job_request.v1",
        "job_id": spec.run_id,
        "capture_root": str(Path(capture_root).expanduser()),
        "site_package": {
            "capture_root": str(Path(capture_root).expanduser()),
            "site_id": scene.get("bundle_id"),
            "package_uri": scene.get("uri"),
            "scene_entrypoint": scene.get("entrypoint"),
            "content_digest": scene.get("content_digest"),
        },
        "requested_tasks": requested_tasks,
        "robot_profile": {
            "robot_profile_id": robot.get("robot_profile_id"),
            "asset_ref": robot.get("asset_ref"),
            "embodiment": robot.get("embodiment"),
            "sensors": robot.get("sensors") or [],
        },
        "policy_id": policy.get("policy_id"),
        "policy_package": policy_package,
        "operation": spec.metadata.get("operation") or "evaluate_only",
        "simulator_preference": runtime.get("simulator"),
        "evaluation_substrate": spec.metadata.get("evaluation_substrate"),
        "budget": {
            "budget_usd": runtime.get("max_spend_usd"),
            "timeout_seconds": runtime.get("timeout_seconds"),
        },
        "rights_privacy_scope": _mapping(proof.get("rights_privacy_scope")),
        "evaluation_run_proof_contract": proof,
        "provenance": {
            "evaluation_run_id": spec.run_id,
            "evaluation_run_schema_version": EVALUATION_RUN_SCHEMA_VERSION,
            "evaluation_run_spec_digest": spec_digest,
            "source_spec_is_execution_authority": True,
        },
    }


class RobotEvalEvaluationRunExecutor:
    """Execute a generic run through the existing robot-eval implementation."""

    adapter_id = "robot_eval_job_orchestrator"

    def execute(
        self,
        *,
        spec: EvaluationRunSpec,
        output_dir: Path,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from .robot_eval_job_orchestrator import build_robot_eval_job

        capture_root = _string(context.get("capture_root"))
        if not capture_root:
            return {
                "schema_version": "robot_eval_evaluation_run_execution.v1",
                "status": "blocked",
                "blockers": ["robot_eval_evaluation_run_capture_root_missing"],
            }
        providers = spec.runtime_provider_profile.get("providers") or []
        if isinstance(providers, str):
            providers = [providers]
        providers = [_string(value) for value in providers if _string(value)]
        if len(providers) != 1:
            return {
                "schema_version": "robot_eval_evaluation_run_execution.v1",
                "status": "blocked",
                "blockers": ["robot_eval_evaluation_run_requires_one_provisioner"],
            }
        simulator = _string(spec.runtime_provider_profile.get("simulator"))
        gates = _mapping(context.get("gates"))
        request = robot_eval_job_request_from_evaluation_run(
            spec,
            capture_root=capture_root,
            spec_digest=_string(
                _mapping(context.get("_evaluation_run_binding")).get("spec_digest")
            )
            or None,
        )
        result = build_robot_eval_job(
            capture_root=capture_root,
            job_request=request,
            job_id=spec.run_id,
            agent_adapter=gates.get("agent_adapter"),
            provisioner=providers[0],
            simulator=simulator,
            evaluation_substrate=_string(spec.metadata.get("evaluation_substrate")) or None,
            allow_wam_provider=bool(gates.get("allow_wam_provider")),
            wam_provider_commands=_mapping(gates.get("wam_provider_commands")),
            wam_artifact_output_uri=_string(gates.get("wam_artifact_output_uri")) or None,
            wam_provider_max_retries=int(gates.get("wam_provider_max_retries") or 0),
            wam_provider_timeout_seconds=(
                int(gates["wam_provider_timeout_seconds"])
                if gates.get("wam_provider_timeout_seconds") is not None
                else None
            ),
            allow_gpu_provisioning=bool(gates.get("allow_gpu_provisioning")),
            allow_simulator_execution=bool(gates.get("allow_simulator_execution")),
            allowed_simulators=list(gates.get("allowed_simulators") or []),
            simulator_commands=_mapping(gates.get("simulator_commands")),
            allow_cpu_simulator_preflight=bool(
                gates.get("allow_cpu_simulator_preflight")
            ),
            cpu_preflight_backends=list(gates.get("cpu_preflight_backends") or ())
            or ["mujoco", "pybullet"],
            cpu_preflight_smoke_steps=int(gates.get("cpu_preflight_smoke_steps") or 10),
            allow_cpu_preflight_render=bool(gates.get("allow_cpu_preflight_render")),
            allow_training=bool(gates.get("allow_training")),
            training_command=_string(gates.get("training_command")) or None,
            allow_policy_execution=bool(gates.get("allow_policy_execution")),
            policy_execution_commands=_mapping(gates.get("policy_execution_commands")),
            timeout_seconds=int(
                spec.runtime_provider_profile.get("timeout_seconds") or 120
            ),
            budget_usd=(
                float(spec.runtime_provider_profile["max_spend_usd"])
                if spec.runtime_provider_profile.get("max_spend_usd") is not None
                else None
            ),
            arena_results_dir=_string(gates.get("arena_results_dir")) or None,
            arena_scenario_count=int(gates.get("arena_scenario_count") or 500),
            arena_shard_size=int(gates.get("arena_shard_size") or 50),
            arena_num_envs=int(gates.get("arena_num_envs") or 16),
            arena_retry_budget=int(gates.get("arena_retry_budget") or 2),
            allow_rollout_vision_labeling=bool(
                gates.get("allow_rollout_vision_labeling")
            ),
            vision_labeling_command=_string(gates.get("vision_labeling_command")) or None,
            allow_delivery_upload=bool(gates.get("allow_delivery_upload")),
            delivery_command=_string(gates.get("delivery_command")) or None,
            arena_operator_mode=_string(gates.get("arena_operator_mode")) or "none",
            allow_live_agents_sdk=bool(gates.get("allow_live_agents_sdk")),
            allow_live_codex_sdk=bool(gates.get("allow_live_codex_sdk")),
        )
        return {
            **dict(result),
            "evaluation_run_execution_adapter_schema_version": (
                "robot_eval_evaluation_run_execution.v1"
            ),
            "evaluation_run_id": spec.run_id,
            "source_spec_is_execution_authority": True,
        }


def execute_legacy_robot_eval_request_as_evaluation_run(
    *,
    capture_root: str | Path,
    job_request: Mapping[str, Any] | str | Path,
    job_id: str,
    output_dir: str | Path | None = None,
    agent_adapter: Any = None,
    provisioner: str = "fixture_local",
    simulator: str = "fixture",
    evaluation_substrate: str | None = None,
    allow_wam_provider: bool = False,
    wam_provider_command: str | None = None,
    wam_provider_commands: Mapping[str, str] | None = None,
    wam_artifact_output_uri: str | None = None,
    wam_provider_max_retries: int = 0,
    wam_provider_timeout_seconds: int | None = None,
    allow_gpu_provisioning: bool = False,
    allow_simulator_execution: bool = False,
    allowed_simulators: Sequence[str] = (),
    simulator_commands: Mapping[str, str] | None = None,
    allow_cpu_simulator_preflight: bool = False,
    cpu_preflight_backends: Sequence[str] = ("mujoco", "pybullet"),
    cpu_preflight_smoke_steps: int = 10,
    allow_cpu_preflight_render: bool = False,
    allow_training: bool = False,
    training_command: str | None = None,
    allow_policy_execution: bool = False,
    policy_execution_commands: Mapping[str, str] | None = None,
    timeout_seconds: int = 120,
    budget_usd: float | None = None,
    arena_results_dir: str | Path | None = None,
    arena_scenario_count: int = 500,
    arena_shard_size: int = 50,
    arena_num_envs: int = 16,
    arena_retry_budget: int = 2,
    allow_rollout_vision_labeling: bool = False,
    vision_labeling_command: str | None = None,
    allow_delivery_upload: bool = False,
    delivery_command: str | None = None,
    arena_operator_mode: str = "none",
    allow_live_agents_sdk: bool = False,
    allow_live_codex_sdk: bool = False,
) -> Mapping[str, Any]:
    """Route a legacy request through the canonical six-part execution authority."""

    from .evaluation_run import compile_evaluation_run
    from .robot_eval_job_orchestrator import build_robot_eval_job

    request = (
        dict(job_request)
        if isinstance(job_request, Mapping)
        else read_json(Path(job_request).expanduser())
    )
    if evaluation_substrate:
        request = {**request, "evaluation_substrate": evaluation_substrate}
    request_budget = _mapping(request.get("budget"))
    authoritative_budget = (
        float(request_budget["budget_usd"])
        if request_budget.get("budget_usd") is not None
        else budget_usd
    )
    authoritative_timeout = int(
        request_budget.get("timeout_seconds") or timeout_seconds
    )
    request = {
        **request,
        "provenance": {
            **_mapping(request.get("provenance")),
            "evaluation_run_id": job_id,
            "evaluation_run_schema_version": EVALUATION_RUN_SCHEMA_VERSION,
            "source_spec_is_execution_authority": True,
        },
    }
    commands = dict(wam_provider_commands or {})
    if wam_provider_command and evaluation_substrate:
        commands.setdefault(evaluation_substrate, wam_provider_command)
    # Let the legacy builder resolve dataset-card task expansion and write its
    # structured blocked/completed artifacts before compiling the canonical
    # Evaluation Run mirror. Compiling from an empty matrix here used to block
    # valid task-only requests before that enrichment could happen.
    result = build_robot_eval_job(
        capture_root=capture_root,
        job_request=request,
        job_id=job_id,
        agent_adapter=agent_adapter,
        provisioner=provisioner,
        simulator=simulator,
        evaluation_substrate=evaluation_substrate,
        allow_wam_provider=allow_wam_provider,
        wam_provider_commands=commands,
        wam_artifact_output_uri=wam_artifact_output_uri,
        wam_provider_max_retries=wam_provider_max_retries,
        wam_provider_timeout_seconds=wam_provider_timeout_seconds,
        allow_gpu_provisioning=allow_gpu_provisioning,
        allow_simulator_execution=allow_simulator_execution,
        allowed_simulators=allowed_simulators,
        simulator_commands=simulator_commands,
        allow_cpu_simulator_preflight=allow_cpu_simulator_preflight,
        cpu_preflight_backends=cpu_preflight_backends,
        cpu_preflight_smoke_steps=cpu_preflight_smoke_steps,
        allow_cpu_preflight_render=allow_cpu_preflight_render,
        allow_training=allow_training,
        training_command=training_command,
        allow_policy_execution=allow_policy_execution,
        policy_execution_commands=policy_execution_commands,
        timeout_seconds=authoritative_timeout,
        budget_usd=authoritative_budget,
        arena_results_dir=arena_results_dir,
        arena_scenario_count=arena_scenario_count,
        arena_shard_size=arena_shard_size,
        arena_num_envs=arena_num_envs,
        arena_retry_budget=arena_retry_budget,
        allow_rollout_vision_labeling=allow_rollout_vision_labeling,
        vision_labeling_command=vision_labeling_command,
        allow_delivery_upload=allow_delivery_upload,
        delivery_command=delivery_command,
        arena_operator_mode=arena_operator_mode,
        allow_live_agents_sdk=allow_live_agents_sdk,
        allow_live_codex_sdk=allow_live_codex_sdk,
    )
    job_dir = Path(_string(result.get("job_dir"))) if result.get("job_dir") else None
    execution_root = Path(
        output_dir
        or Path(capture_root).expanduser() / "pipeline" / "evaluation_runs" / job_id
    )
    enriched_request = request
    scene_preflight: Mapping[str, Any] = {}
    scenario_matrix: Mapping[str, Any] = {}
    policy_manifest: Mapping[str, Any] = {}
    if job_dir:
        for name, target in (
            ("job_request.json", "request"),
            ("scene_asset_preflight.json", "scene"),
            ("scenario_eval_matrix.json", "matrix"),
            ("policy_package_manifest.json", "policy"),
        ):
            path = job_dir / name
            if not path.is_file():
                continue
            value = read_json(path)
            if target == "request":
                enriched_request = value
            elif target == "scene":
                scene_preflight = value
            elif target == "matrix":
                scenario_matrix = value
            else:
                policy_manifest = value
    spec = build_robot_eval_evaluation_run_spec(
        job_id=job_id,
        request=enriched_request,
        capture_root=capture_root,
        scene_preflight=scene_preflight,
        scenario_eval_matrix=scenario_matrix,
        policy_manifest=policy_manifest,
        provisioner=provisioner,
        simulator=simulator,
        budget_usd=authoritative_budget,
        timeout_seconds=authoritative_timeout,
    )
    compile_evaluation_run(spec, output_dir=execution_root)
    write_json(
        execution_root / "evaluation_run_execution.json",
        {
            "schema_version": "evaluation_run_execution.v1",
            "run_id": job_id,
            "status": result.get("status") or "blocked",
            "adapter_result": dict(result),
        },
    )
    return dict(result)


def execute_robot_eval_cli_evaluation_run(
    args: Any,
    *,
    agent_adapter: Any,
    simulator_commands: Mapping[str, str],
    policy_execution_commands: Mapping[str, str],
    wam_provider_commands: Mapping[str, str],
) -> Any:
    """Run the generic CLI flags through the authoritative Evaluation Run engine."""

    from .evaluation_run_execution import execute_evaluation_run

    raw_spec = read_json(Path(args.evaluation_run_spec))
    spec = EvaluationRunSpec.from_mapping(raw_spec)
    output_dir = (
        Path(args.evaluation_run_output_dir).expanduser()
        if args.evaluation_run_output_dir
        else Path(args.capture_root).expanduser()
        / "pipeline"
        / "evaluation_runs"
        / spec.run_id
    )
    gates = {
        "agent_adapter": agent_adapter,
        "allow_wam_provider": args.allow_wam_provider,
        "wam_provider_commands": dict(wam_provider_commands),
        "wam_artifact_output_uri": args.wam_artifact_output_uri,
        "wam_provider_max_retries": args.wam_provider_max_retries,
        "wam_provider_timeout_seconds": args.wam_provider_timeout_seconds,
        "allow_gpu_provisioning": args.allow_gpu_provisioning,
        "allow_simulator_execution": args.allow_simulator_execution,
        "allowed_simulators": list(args.allow_simulator),
        "simulator_commands": dict(simulator_commands),
        "allow_cpu_simulator_preflight": args.allow_cpu_simulator_preflight,
        "cpu_preflight_backends": list(args.cpu_preflight_backend),
        "cpu_preflight_smoke_steps": args.cpu_preflight_smoke_steps,
        "allow_cpu_preflight_render": args.allow_cpu_preflight_render,
        "allow_training": args.allow_training,
        "training_command": args.training_command,
        "allow_policy_execution": args.allow_policy_execution,
        "policy_execution_commands": dict(policy_execution_commands),
        "arena_results_dir": args.arena_results_dir,
        "arena_scenario_count": args.arena_scenario_count,
        "arena_shard_size": args.arena_shard_size,
        "arena_num_envs": args.arena_num_envs,
        "arena_retry_budget": args.arena_retry_budget,
        "allow_rollout_vision_labeling": args.allow_rollout_vision_labeling,
        "vision_labeling_command": args.vision_labeling_command,
        "allow_delivery_upload": args.allow_delivery_upload,
        "delivery_command": args.delivery_command,
        "arena_operator_mode": args.arena_operator_mode,
        "allow_live_agents_sdk": args.allow_live_agents_sdk,
        "allow_live_codex_sdk": args.allow_live_codex_sdk,
    }
    return execute_evaluation_run(
        raw_spec,
        output_dir=output_dir,
        allow_execution=True,
        context={"capture_root": args.capture_root, "gates": gates},
    )
