"""ADP-050/day 28: inject controlled policy execution into the job orchestrator.

Only trusted simulator startup supplies these factories. Request payloads
cannot select Python modules, shell commands, scene paths or sandbox profiles.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from .company_policy_container_contract_v2 import validate_company_policy_container_contract_v2
from .controlled_policy_session import ControlledPolicyClient, customer_hosted_client, run_controlled_policy_episode
from .policy_model_onnx import validate_model_task_binding


class ControlledPolicyExecutor:
    def __init__(self, *, task_contract: Callable[..., Mapping[str, Any]],
                 environment_factory: Callable[..., Any], sandbox_factory: Callable[..., Any],
                 allowed_origins: tuple[str, ...] = (), model_image_builder: Callable[..., str] | None = None,
                 credential_resolver: Callable[..., str | None] | None = None,
                 outcome_reader: Callable[..., Mapping[str, Any]] | None = None,
                 max_queries: int = 128, deadline_seconds: float = 120):
        self.task_contract = task_contract
        self.environment_factory = environment_factory
        # This runner uses company-policy v2 qualification, denial probes and
        # cleanup, then calls qualified_session with the Unix-proxy transport.
        self.sandbox_factory = sandbox_factory
        self.allowed_origins = allowed_origins
        self.model_image_builder = model_image_builder
        self.credential_resolver = credential_resolver
        self.outcome_reader = outcome_reader
        self.max_queries = max_queries
        self.deadline_seconds = deadline_seconds

    def __call__(self, *, modality: str, payload: Mapping[str, Any],
                 job_request: Mapping[str, Any], job_dir: Path,
                 observations: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        if modality not in {"policy_api_endpoint", "docker_container", "sim_controller_plugin"}:
            raise ValueError("controlled_policy_modality_invalid")
        tasks = job_request.get("requested_tasks", [])
        if not isinstance(tasks, list) or len(tasks) != 1 or not observations:
            raise ValueError("controlled_policy_frozen_task_required")
        task = tasks[0]
        if not isinstance(task, Mapping) or len(task.get("scenario_ids", [])) != 1:
            raise ValueError("controlled_policy_single_scenario_required")
        contract = dict(self.task_contract(job_request=job_request))
        contract.pop("contract_digest", None)
        if modality != "policy_api_endpoint":
            image = payload.get("image_ref")
            artifact = payload.get("model_artifact")
            if artifact is not None:
                if not isinstance(artifact, Mapping) or self.model_image_builder is None:
                    raise ValueError("controlled_policy_model_builder_required")
                validate_model_task_binding(artifact, contract)
                image = self.model_image_builder(artifact=artifact, contract=contract, job_request=job_request)
                contract["container"] = {**contract["container"],
                    "serve_command": ["python", "-m", "blueprint_pipeline.policy_model_server"],
                    "port": 8600, "run_as_uid": 65532, "run_as_gid": 65532, "gpu_required": False}
            if not isinstance(image, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}", image):
                raise ValueError("controlled_policy_immutable_image_required")
            contract["container"] = {**contract["container"], "image": image}
        contract = validate_company_policy_container_contract_v2(contract)
        output = job_dir / "controlled_policy_execution"
        output.mkdir(mode=0o700, exist_ok=True)
        attempts = []
        for index, observation in enumerate(observations):
            if (observation.get("task_id") != task.get("task_id")
                    or observation.get("scenario_id") != task["scenario_ids"][0]):
                raise ValueError("controlled_policy_episode_binding_mismatch")
            path = output / f"{modality}-{index:05d}.jsonl"
            # Exclusive retention prevents a retry from silently overwriting a
            # prior execution; orchestrator attempts have their own job roots.
            with path.open("x") as stream:
                path.chmod(0o600)
                def retain(row: Mapping[str, Any]) -> None:
                    stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                    stream.flush()
                def run_session(transport: Any) -> Mapping[str, Any]:
                    policy = transport if modality == "policy_api_endpoint" else ControlledPolicyClient(
                        contract=contract, transport=transport, transport_name="qualified_sandbox_unix_proxy")
                    environment = self.environment_factory(job_request=job_request, observation=observation)
                    receipt = run_controlled_policy_episode(environment=environment, policy=policy,
                        max_queries=self.max_queries, deadline_seconds=self.deadline_seconds, retain=retain)
                    if self.outcome_reader is not None:
                        from .controlled_policy_outcome import validate_controlled_outcome
                        outcome = validate_controlled_outcome(self.outcome_reader(
                            environment=environment, job_request=job_request, observation=observation,
                            episode_receipt=receipt), executed_motor_steps=receipt["executed_motor_steps"])
                        retain({"kind": "independent_outcome", "receipt": outcome})
                        receipt = {**receipt, "task_success": outcome["task_success"],
                                   "outcome_evidence_required": False, "independent_outcome": outcome}
                    retain({"kind": "episode_terminal", "receipt": receipt})
                    return receipt

                if modality == "policy_api_endpoint":
                    token = self.credential_resolver(job_request=job_request) if self.credential_resolver else None
                    policy = customer_hosted_client(endpoint=str(payload.get("endpoint_url", "")),
                        allowed_origins=self.allowed_origins, contract=contract, bearer_token=token)
                    receipt = run_session(policy)
                else:
                    sandbox = self.sandbox_factory(contract=contract, job_request=job_request,
                                                   qualified_session=run_session)
                    retain({"kind": "sandbox_terminal", "receipt": sandbox.get("terminal_receipt")})
                    if (sandbox.get("status") != "controlled_session_completed"
                            or sandbox.get("terminal_receipt", {}).get("cleanup_complete") is not True):
                        raise ValueError("controlled_policy_sandbox_session_or_cleanup_failed")
                    receipt = sandbox["controlled_session"]
                attempts.append({"attempt_id": f"{modality}_{index:05d}", **{
                    key: observation.get(key) for key in ("observation_id", "task_id", "scenario_id", "scenario_eval_run_id")},
                    "status": ("completed" if receipt.get("outcome_evidence_required") is False
                               else "executed_outcome_pending") if receipt["executed_motor_steps"] else "blocked",
                    "success": receipt.get("task_success"),
                    "evidence_scope": ("native_simulator_independent_outcome" if receipt.get("outcome_evidence_required") is False
                                       else "controlled_policy_execution_without_outcome"),
                    "metrics": receipt, "artifact_paths": {"controlled_policy_episode": str(path)}})
        completed = bool(attempts) and all(row["status"] == "completed" for row in attempts)
        return {"status": "completed" if completed else "executed_outcome_pending", "attempts": attempts,
            "independent_outcome_proven": completed,
            "execution_performed": any(row["metrics"]["executed_motor_steps"] > 0 for row in attempts),
            "blockers": [] if completed else ["independent_task_outcome_evidence_required"]}
