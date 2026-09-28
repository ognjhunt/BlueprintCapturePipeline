"""Provider entry point for a frozen controlled-policy task; no policy imports."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

from blueprint_pipeline.controlled_policy_configuration import validate_native_configuration, canonical_request_digest
RESULT_FILENAME = "controlled_native_policy_result.v1.json"
from blueprint_pipeline.controlled_native_isaac import build_controlled_native_environment, read_controlled_native_outcome
from blueprint_pipeline.controlled_policy_dispatch import ControlledPolicyExecutor
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def main() -> int:
    root = Path(__file__).resolve().parent
    output = Path(os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR", root.parent / "runtime_output"))
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": "controlled_native_policy_result.v1", "status": "blocked",
        "evidence_scope": "development_only", "qualification_eligible": False,
        "physical_success_proven": False, "public_claim_upgrade_allowed": False,
        "candidate_policy_queried": False, "blockers": []}
    environments = []
    started = time.monotonic()
    try:
        config = validate_native_configuration(json.loads((root / "runtime_inputs/controlled_native_configuration.json").read_text()))
        request = json.loads((root / "runtime_inputs/controlled_job_request.json").read_text())
        observations = json.loads((root / "runtime_inputs/controlled_observations.json").read_text())
        def create_environment(**context):
            env = build_controlled_native_environment(runtime_root=root, configuration=config,
                evidence_root=output, **context)
            environments.append(env)
            return env
        def sandbox_required(**_context):
            raise ValueError("controlled_native_qualified_sandbox_not_configured")
        def resolve_credential(*, job_request):
            path = root / "runtime_inputs/policy_credential.json"
            if not path.exists():
                return None
            credential = json.loads(path.read_text())
            if (credential.get("job_id") != job_request["job_id"]
                    or credential.get("endpoint_url") != job_request["policy_package"]["policy_api_endpoint"]["endpoint_url"]):
                raise ValueError("controlled_native_policy_credential_binding_mismatch")
            return credential["bearer_token"]
        executor = ControlledPolicyExecutor(task_contract=lambda **_: config["contract"],
            environment_factory=create_environment, sandbox_factory=sandbox_required,
            allowed_origins=tuple(config["allowed_origins"]), outcome_reader=read_controlled_native_outcome,
            max_queries=config["max_queries"], deadline_seconds=config["deadline_seconds"],
            credential_resolver=resolve_credential)
        package = request["policy_package"]
        selected = [name for name in ("policy_api_endpoint", "docker_container", "sim_controller_plugin") if package.get(name)]
        if len(selected) != 1 or len(observations) != 1:
            raise ValueError("controlled_native_one_policy_episode_required")
        execution = executor(modality=selected[0], payload=package[selected[0]], job_request=request,
            job_dir=output, observations=observations)
        result.update(execution)
        result["candidate_policy_queried"] = execution["execution_performed"]
        result["job_id"] = request["job_id"]
        result["canonical_request_digest"] = canonical_request_digest(request)
        result["native_runtime_receipts"] = [env.native_runtime_receipt for env in environments]
        result["native_final_joint_positions_rad"] = [list(env.native.read_arm_joint_positions()) for env in environments]
    except Exception as exc:
        result["blockers"] = [str(exc)[:500]]
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
        # Isaac close can end the interpreter: seal before closing it.
        (output / RESULT_FILENAME).write_text(json.dumps(result, sort_keys=True, allow_nan=False))
        for env in environments:
            env.outcome_recorder.close()
            env.native_built.env.close()
            env.native_simulation_app.close()
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
