# Policy Autoresearch MuJoCo Evaluator

`blueprint-run-policy-autoresearch-mujoco-evaluator` is the real sim execution
bridge for the sim-only policy autoresearch lane. The source
`scenario_eval_matrix.json` remains the frozen verifier. The evaluator derives a
candidate split matrix that changes only policy-generated route waypoints and
policy ids, then executes `blueprint_pipeline.mujoco_g1_simulator_command`.

Example:

```bash
BLUEPRINT_MUJOCO_G1_MODEL_ROOT=/path/to/mujoco_menagerie/unitree_g1 \
BLUEPRINT_POLICY_AUTORESEARCH_MUJOCO_STEPS=16 \
blueprint-run-policy-autoresearch \
  --capture-root /path/to/<capture-root> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --policy-recipe /path/to/seed_policy_recipe.json \
  --evaluator-command "python -m blueprint_pipeline.policy_autoresearch_mujoco_evaluator" \
  --simulator-engine mujoco \
  --max-iterations 1 \
  --agent-count 1
```

Proof boundary:

- Proves candidate route/control execution in local MuJoCo when the emitted
  attempts include `metrics.simulator_execution_performed=true`.
- Does not prove balanced humanoid locomotion, robot-team policy execution,
  physical robot readiness, safety validation, real-world outcome, or public
  claim upgrade.
- Promotion still requires heldout task success and zero normalized
  safety/contact events against the frozen verifier.

Concrete smoke evidence currently lives under:

`output/policy_autoresearch_smoke/mujoco_real_policy_job/policy_autoresearch_actual_mujoco/`

That run freezes three scenario runs with safe start/target pairs but a bad seed
route through a colliding center point. The seed route fails heldout clearance;
the promoted candidate replaces only route waypoints and reaches heldout task
success with clean MuJoCo contact/safety metrics.
