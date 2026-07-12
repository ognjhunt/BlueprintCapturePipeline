# Evaluation Run Interface

`evaluation_run.v1` is the provider-neutral front door for robot evaluation.
It describes one run using exactly six replaceable parts:

1. `scene_bundle`
2. `robot_adapter`
3. `task_scenario_pack`
4. `policy_adapter`
5. `runtime_provider_profile`
6. `proof_contract`

The interface is compiled by
`python -m blueprint_pipeline.evaluation_run --spec ... --output-dir ...` into
an `evaluation_run_plan.v1` artifact before a scene is executed. Compilation
is side-effect free: it does not stage an asset, launch a provider, invoke a
policy, or upgrade a proof claim.

Execution goes through the separately gated authority:

```bash
blueprint-run-evaluation-run \
  --spec evaluation_run_spec.json \
  --output-dir output/evaluation-run
```

That command compiles only. `--allow-execution` is required before it resolves
and invokes the runtime profile's `execution_adapter_id`. Local materialization
paths, ephemeral transport credentials, and runtime gates belong in a local
`--context` JSON file; context values are passed to the adapter but never
persisted. Every `evaluation_run_execution.v1` artifact binds the adapter result
to the compiled spec digest.

```json
{
  "schema_version": "evaluation_run.v1",
  "run_id": "warehouse-policy-eval-001",
  "mode": "evaluate",
  "scene_bundle": {
    "adapter_id": "openusd_scene_bundle",
    "adapter_version": "1",
    "bundle_id": "warehouse-a",
    "uri": "gs://blueprint-scenes/warehouse-a.zip",
    "entrypoint": "Warehouse.usd",
    "content_digest": "sha256:..."
  },
  "robot_adapter": {
    "adapter_id": "isaac_robot_asset",
    "adapter_version": "1",
    "robot_profile_id": "mobile-manipulator-a",
    "asset_ref": "Robots/MobileManipulator/robot.usd"
  },
  "task_scenario_pack": {
    "adapter_id": "manifest_task_scenario_pack",
    "adapter_version": "1",
    "pack_id": "warehouse-pick-pack",
    "tasks": [{"task_id": "pick-tote"}],
    "scenarios": [{"scenario_id": "pick-near-shelf", "task_id": "pick-tote"}]
  },
  "policy_adapter": {
    "adapter_id": "http_policy_worker",
    "adapter_version": "1",
    "policy_id": "customer-policy-17",
    "observation_schema_ref": "blueprint://schemas/robot_eval_observation.v1",
    "action_schema_ref": "blueprint://schemas/robot_eval_action_trace.v1"
  },
  "runtime_provider_profile": {
    "adapter_id": "isaac_provider_runtime",
    "adapter_version": "1",
    "execution_adapter_id": "robot_eval_job_orchestrator",
    "profile_id": "isaac-a40",
    "providers": ["runpod"],
    "simulator": "isaac_sim",
    "max_spend_usd": 2.0
  },
  "proof_contract": {
    "adapter_id": "declared_evidence_proof_contract",
    "adapter_version": "1",
    "contract_id": "warehouse-task-eval",
    "required_evidence": ["action_trace", "task_state_change", "teardown"],
    "claim_ceiling": {"physical_robot_readiness": false},
    "prohibited_claims": ["physical_robot_readiness"]
  }
}
```

## Adapter rule

Scene- or task-specific names belong in adapter packages, not the Evaluation
Run interface. Existing `g1_kitchen_*` schemas remain readable historical
evidence contracts. `g1_kitchen_evaluation_run_adapter.py` translates that
lane into `evaluation_run.v1`; new generic orchestration must not import
kitchen modules.

Every binding resolves through `EvaluationRunAdapterRegistry`. Unknown adapter
IDs fail closed. A new robot, policy, scene format, runtime, or proof adapter is
added by registering an `EvaluationRunAdapterDescriptor`; the compiler itself
does not change. The built-in registry contains multiple implementations at
each seam so these are real variation points rather than speculative wrappers.

Runtime dispatch resolves independently through
`EvaluationRunExecutionRegistry`. The built-in implementations are the generic
robot-eval orchestrator and the backward-compatible G1/kitchen executor. New
execution implementations register at that seam without changing the compiler.

## Public execution paths

`blueprint-run-evaluation-run` is the canonical execution command. The
installed `blueprint-run-robot-eval-job`, robot-eval request inbox, provider
input preparation, worker, `blueprint-run-e2e`, and real-policy-family harness
remain supported compatibility inputs. Each translates its legacy request into
the six-part spec and invokes the same execution authority before the low-level
robot-eval implementation runs. The low-level `build_robot_eval_job` function
is an implementation port used by `RobotEvalEvaluationRunExecutor`; it is not a
separate public contract.

`blueprint-isaac-g1-kitchen-parity` follows the same rule: its CLI arguments are
translated into an Evaluation Run, and the registered kitchen compatibility
executor invokes the historical job implementation. Kitchen names remain only
in that adapter and its evidence artifacts.

## Proof rule

An `evaluation_run_plan.v1` proves only that the six bindings are structurally
compatible and ready for an explicitly gated runtime adapter. It does not
prove provider startup, policy execution, simulator task success, semantic
success, safety, deployment readiness, or physical robot readiness.
