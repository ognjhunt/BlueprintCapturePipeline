"""Adapt the generic robot-eval request into ``evaluation_run.v1``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .evaluation_run import EVALUATION_RUN_SCHEMA_VERSION


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
        "proof_contract": {
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
        },
        "metadata": {
            "source_contract": request.get("schema_version"),
            "source_job_id": job_id,
            "capture_first": True,
            "scene_specific_names_are_adapter_configuration": True,
        },
    }
