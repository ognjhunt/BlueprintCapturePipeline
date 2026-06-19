# Policy Autoresearch Owner-GPU Evaluator

`blueprint-run-policy-autoresearch-owner-gpu-evaluator` adapts an owner
simulator command, such as an Isaac Sim or Isaac Lab runner, into the
`blueprint-run-policy-autoresearch` evaluator contract.

The adapter is proof-bounded:

- it runs the owner simulator command through `blueprint-run-owner-gpu-proof`;
- it validates the owner GPU proof manifest with the existing Pipeline verifier;
- it still requires a per-scenario policy attempt trace before task success can
  count toward promotion;
- it never upgrades physical robot readiness, real-world outcome proof, safety
  validation, or public claims.

Use it from policy autoresearch with an engine-specific command:

```bash
BLUEPRINT_POLICY_AUTORESEARCH_OWNER_COMMAND="/path/to/owner-isaac-policy-runner" \
BLUEPRINT_POLICY_AUTORESEARCH_OWNER_SYSTEM_ID="runpod-isaac-l40s" \
BLUEPRINT_POLICY_AUTORESEARCH_OWNER_SIMULATOR_VERSION="Isaac Sim 6.0.0" \
BLUEPRINT_POLICY_AUTORESEARCH_OWNER_GPU_MODEL="L40S 48GB" \
BLUEPRINT_POLICY_AUTORESEARCH_OPERATOR_ID="operator-id" \
BLUEPRINT_POLICY_AUTORESEARCH_OPERATOR_ATTESTATION="I ran this command on the owner GPU VM." \
blueprint-run-policy-autoresearch \
  --capture-root /path/to/<capture-root> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --policy-recipe /path/to/seed_policy_recipe.json \
  --simulator-engine mujoco \
  --simulator-engine isaac_sim \
  --evaluator-command-by-engine "mujoco=python -m blueprint_pipeline.policy_autoresearch_mujoco_evaluator" \
  --evaluator-command-by-engine "isaac_sim=python -m blueprint_pipeline.policy_autoresearch_owner_gpu_evaluator" \
  --parallel-branch-limit 2 \
  --max-candidate-evaluations 4
```

The owner command receives `BLUEPRINT_POLICY_AUTORESEARCH_RECIPE`,
`BLUEPRINT_POLICY_AUTORESEARCH_MATRIX`, and
`BLUEPRINT_POLICY_AUTORESEARCH_OWNER_ATTEMPT_TRACE`. It must write the attempt
trace as JSON or JSONL with `scenario_eval_run_id`, `task_success` or `success`,
failure modes, and safety/contact metrics. Accepted owner simulator proof without
that attempt trace is preserved as simulator execution evidence, but cannot
promote a policy candidate.
