# Simulator-Agnostic G1 Execution Contract

Status: implementation contract for MuJoCo and Isaac Unitree G1 simulator evidence.

This contract lets robot-eval jobs compare simulator command outputs without
treating backend evidence as interchangeable. MuJoCo proof does not count as
Isaac proof. Isaac proof does not count as MuJoCo proof. Neither backend proves
safety validation, deployment approval, physical robot readiness, WAM
consistency, or generated-world rank fidelity.

## Required Output Fields

Every simulator command must emit a JSON payload at `BLUEPRINT_SIMULATOR_OUTPUT`
or the command-specific `--simulator-output` path with these fields:

- `simulator_backend`
- `simulator_version`
- `simulator_execution_proven`
- `scenario_eval_run_count`
- `attempt_count`
- `attempt_count_matches_matrix_count`
- `scenario_eval_run_coverage_complete`
- `required_scenario_eval_run_ids`
- `covered_scenario_eval_run_ids`
- `missing_scenario_eval_run_ids`
- `attempts`
- `normalized_attempt_trace`
- `failure_labels`
- `policy_evaluation_summary`
- `realistic_video_manifest` or backend-equivalent camera/media manifest
- `collision_contact_report` or backend-equivalent contact summary
- `artifact_manifest`
- `batch_trace_package`
- `batch_closure_manifest`
- `artifact_paths`
- `proof_boundary`

The command must run every row in `scenario_eval_matrix.json`. If the runtime is
unavailable, it may write blocked attempts for every row, but it must set
`simulator_execution_proven=false` and keep `scenario_eval_run_coverage_complete`
separate from proof of real simulator execution.

## Isaac Command

Local command:

```bash
python -m blueprint_pipeline.isaac_g1_site_3dgs_realistic_eval \
  --capture-root /path/to/capture \
  --scenario-eval-matrix /path/to/scenario_eval_matrix.json \
  --simulator-output /path/to/isaac_g1_simulator_output.json
```

Equivalent console script:

```bash
blueprint-run-isaac-g1-simulator-command \
  --capture-root /path/to/capture \
  --scenario-eval-matrix /path/to/scenario_eval_matrix.json \
  --simulator-output /path/to/isaac_g1_simulator_output.json
```

Optional runtime-result ingestion:

```bash
python -m blueprint_pipeline.isaac_g1_site_3dgs_realistic_eval \
  --capture-root /path/to/capture \
  --scenario-eval-matrix /path/to/scenario_eval_matrix.json \
  --runtime-result /path/to/isaac_runtime_result.json \
  --simulator-output /path/to/isaac_g1_simulator_output.json
```

An Isaac runtime result can prove Isaac execution only when it contains
Isaac-specific runtime facts such as `isaac_runtime_executed=true`,
`isaac_sim_execution_proven=true`, Unitree G1 asset proof, attempts keyed by
`scenario_eval_run_id`, and media/contact artifacts that exist on disk after
teardown or upload finalization.

## Environment Contract

The local/provider path detects these variables when present:

- `BLUEPRINT_MUJOCO_G1_MODEL_ROOT`
- `BLUEPRINT_ISAAC_PYTHON` or `ISAAC_SIM_ROOT`
- `BLUEPRINT_ISAAC_UNITREE_G1_USD`
- `BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF`
- `BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF_FILE`
- `BLUEPRINT_ISAAC_PROVIDER_BUNDLE_URI`
- `BLUEPRINT_ARTIFACT_OUTPUT_URI`
- `BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL`
- `NGC_API_KEY_FILE`
- `RUNPOD_API_KEY_FILE`

`BLUEPRINT_ARTIFACT_OUTPUT_URI` or
`BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL` is required before paid
provider work can be treated as finalizable. File-based secrets are recorded as
present or missing; raw secret values must not be written to artifacts.

RunPod Isaac launches are fail-closed by default unless a prebuilt, versioned
Isaac eval worker image is configured through `BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF`,
`BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF_FILE`, or the generic
`BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF`. The direct
`nvcr.io/nvidia/isaac-sim:6.0.0` base image path is intentionally not
provider-ready because live evidence showed it can spend bounded RunPod time
without reaching the Blueprint fetch/finalizer wrapper or uploading an output
zip. Missing image configuration blocks before spend with
`prebuilt_isaac_eval_worker_image_ref_missing`. Only set
`BLUEPRINT_ALLOW_DIRECT_ISAAC_BASE_IMAGE_RUNPOD=true` for a deliberate debug
run with a wider manual observation window.

## Provider Closeout Requirements

Paid provider execution must report:

- runtime phase log
- spend boundary and billable time estimate/actual
- output zip or uploaded artifact state
- teardown/shutdown proof
- continuing-spend status
- exact blocker when no runtime/provider is usable:
  `isaac_runtime_or_authorized_gpu_unavailable`

Do not accept `local_validation_report.status=passed_with_runtime_blockers` as
an Isaac parity pass. That status means artifacts were internally consistent
despite blocked runtime proof. A real Isaac pass requires
`simulator_execution_proven=true`, exact matrix coverage, runtime attempts,
media evidence, contact/collision summaries, trace package closure, artifact
manifest, and teardown/upload evidence.

## Robot-Eval Orchestrator

For `simulator=isaac_sim`, the orchestrator supplies this default command when
no explicit simulator command is provided:

```bash
python -m blueprint_pipeline.isaac_g1_site_3dgs_realistic_eval
```

The worker passes `BLUEPRINT_CAPTURE_ROOT`, `BLUEPRINT_SCENARIO_EVAL_MATRIX`,
`BLUEPRINT_SIMULATOR_OUTPUT`, and `BLUEPRINT_SIMULATOR_FRAMEWORK`. If the
command writes an output payload with `simulator_execution_proven=false`, the job
must remain blocked even if the process exits cleanly.

## Simulator Beta Readiness

The simulator beta readiness manifest keeps the legacy MuJoCo gate but also
evaluates `site_capture_isaac_g1_run`. The combined
`site_capture_simulator_g1_run` gate can be satisfied by either backend only
when that backend has real evidence. Isaac evidence must include exact matrix
coverage, a completed media manifest, required artifacts, trace package closure,
and an artifact manifest. Placeholder scene assets and blocked runtime outputs
cannot satisfy the Isaac gate.

Run:

```bash
python -m blueprint_pipeline.simulator_beta_readiness \
  --capture-root /path/to/capture \
  --isaac-output /path/to/isaac_g1_simulator_output.json
```

## Policy Autoresearch

MuJoCo policy autoresearch still uses:

```bash
python -m blueprint_pipeline.policy_autoresearch_mujoco_evaluator
```

Isaac/owner-GPU policy autoresearch uses:

```bash
python -m blueprint_pipeline.policy_autoresearch_owner_gpu_evaluator
```

`isaac_sim` is listed in `proven_simulator_engines` only when the evaluator
output includes Isaac-specific proof such as `isaac_sim_execution_proven=true`.
Generic owner-GPU proof is not enough to count an Isaac simulator engine.
