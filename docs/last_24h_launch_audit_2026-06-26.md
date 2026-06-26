# Last 24h Launch/Test Audit - 2026-06-26

This handoff audits the current `BlueprintCapturePipeline` state for launching or testing the sim-only policy-comparison product path: answering, with high accuracy inside the configured evaluator, which robot policy is better for a customer's site. It is intentionally scoped to sim/evaluator evidence. Physical robot evidence, safety validation, real-world anchors, and deployment approval are optional future claim-upgrade lanes, not blockers for this sim-only product launch.

## Current Verdict

Do not present this as physical-robot-ready, safety-validated, or deployment-approved.

For the sim-only policy-comparison product, the current evidence shows strong progress but still blocks launch at these layers:

- Sim-only beta release is still blocked.
- Production forwarding/deployment parity is still blocked.
- Sim-only policy-comparison closure is still blocked.
- Evaluator-bounded policy-comparison quality must stay explicit and evidence-backed.
- Remote/cloud provider execution is not proven for this launch path.
- WAM/generated media remains support/review evidence only.

## Evidence Snapshot

Repository:

- `cwd`: `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline`
- `HEAD`: `aa37e89ef1798260c3c6bf9c35bf83b72a545022`
- `origin/main`: `aa37e89ef1798260c3c6bf9c35bf83b72a545022`
- current worktree after the stabilization pass: 96 dirty status entries, including broad changes across docs, scripts, live pipeline, G1 controlled-run evidence, WAM/perception harnesses, OSCAR visual augmentation, live-pipeline forwarding setup, and tests
- `git diff --check`: passed on 2026-06-26 after the stabilization pass
- `python -m ruff check src/blueprint_pipeline scripts tests`: passed on 2026-06-26 after lint cleanup
- disk headroom after the stabilization pass: about 12 GiB free on `/System/Volumes/Data`

Primary capture root:

```text
/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/output/first-gpu-walkthrough2-storage/local-blueprint/scenes/first-gpu-walkthrough-2/captures/downloads-walkthrough2-20260611
```

Primary sim-only local-gate job:

```text
robot-eval-sim-only-beta-local-gate-capture-intent-first-gpu-humanoid-navigation-smoke-61099fcc72
```

Current evidence artifacts inspected:

- `output/beta_launch_readiness_deep_audit_current.json`
- `pipeline/live_pipeline_control_plane/sim_only_beta_local_gate/sim_only_beta_local_gate_report.json`
- `pipeline/live_pipeline_control_plane/sim_only_beta_release_gate_report.json`
- `pipeline/live_pipeline_control_plane/sim_only_beta_production_deployment_proof.json`
- `pipeline/robot_eval_jobs/<job_id>/job_run_manifest.json`
- `pipeline/robot_eval_jobs/<job_id>/scenario_eval_matrix.json`
- `pipeline/robot_eval_jobs/<job_id>/robot_team_grade_eval_closure_manifest.json`
- `pipeline/robot_eval_jobs/<job_id>/live_eval_closure_manifest.json`
- `pipeline/robot_eval_jobs/<job_id>/failure_labels.json`
- `pipeline/robot_eval_jobs/<job_id>/simulator_command_batch_closure_manifest.json`
- `pipeline/robot_eval_jobs/<job_id>/sim_vs_real_calibration_report.json`
- `pipeline/robot_eval_jobs/<job_id>/robot_camera_profile_launch_readiness.json`
- `pipeline/robot_eval_jobs/<job_id>/owner_robot_camera_calibration_request.json`
- `pipeline/simulation_automation/mujoco_g1_simulator_command/mujoco_digital_twin_fidelity_qa.json`
- `pipeline/g1_controlled_proof_setup/g1_controlled_proof_setup_manifest.json`
- `pipeline/g1_controlled_proof_setup/assembled_live_inputs/g1_controlled_run_evidence_assembly_manifest.json`

Current verification after the stabilization pass:

- `python -m pytest tests/test_live_pipeline_forwarding_secret_setup.py tests/test_sim_only_beta_local_gate.py tests/test_sim_only_beta_release_gate.py tests/test_sim_only_beta_deployment_parity_proof.py tests/test_robot_eval_job_orchestrator.py tests/test_robot_initial_observation.py tests/test_g1_controlled_proof_setup.py tests/test_g1_controlled_run_evidence.py tests/test_g1_field_run_capture.py tests/test_wam_derived_observation_harness.py tests/test_wam_fixture_evaluator.py tests/test_wam_perception_harness_gpu_image.py -q --basetemp=/private/tmp/blueprint-pytest-basetemp-current-launch-audit`: `262 passed in 310.94s`
- `python -m pytest tests/test_live_pipeline_control_plane.py tests/test_live_pipeline_control_plane_coverage_edges.py tests/test_live_pipeline_input_intake.py tests/test_live_pipeline_proof_audit.py tests/test_live_robot_eval_closure_coverage_edges.py tests/test_unitree_groot_n17_sonic_vast_persistent_session.py tests/test_post_training_data_package.py tests/test_oscar_visual_augmentation_packet.py tests/test_oscar_visual_augmentation_generation_runner.py -q --basetemp=/private/tmp/blueprint-pytest-basetemp-current-launch-audit-extra`: `115 passed in 12.82s`
- `python -m pytest tests/test_arena_result_ingest.py tests/test_vast_bundle_staging.py -q --basetemp=/private/tmp/blueprint-pytest-basetemp-current-launch-audit-linttouch`: `40 passed in 6.84s`
- Full `python -m pytest` suite was not rerun in this stabilization pass.
- `output/beta_launch_readiness_deep_audit_current.json` remains stale. It was generated on `2026-06-25T15:09:17.496910Z`, records git `1b63d7f2795c7b8730c75b734fe2d1c9b247cf47`, 30 dirty entries, and 1.67 GiB free disk. No durable generator for this JSON was found by `rg`, so this markdown handoff carries the current evidence instead of inventing a new JSON proof artifact.

## What Improved In The Last 24h

Sim-only local execution is much closer than the stale audit JSON suggests.

- The older audit JSON still says semantic spawn target coverage is incomplete and deterministic fallback targets were used.
- Current artifacts say `scenario_eval_matrix.status=completed`, `semantic_spawn_target_coverage_complete=true`, `deterministic_fallback_spawn_target_run_count=0`, and 11 scenario eval runs are present.
- Current `job_run_manifest.status=simulator_command_completed`.
- Current `job_run_manifest.simulator_execution_proven=true`.
- Current local gate proof boundary says local WebApp route forwarding, Pipeline intake staging, control-plane processing, local MuJoCo simulator execution, and simulator execution are proven for the sim-only path.

Production/deployment proof is narrower and clearer now.

- Current deployment parity proof has `git_parity_proven=true` and `webapp_health_ready=true`.
- It is still blocked by `pipeline_intake_token_missing` and `pipeline_intake_health_not_ready`.
- Current release gate is blocked by production forwarding preflight not being ready, missing forwarding token, unreachable/unattempted forwarding probe, and pipeline intake health not ready.

Camera profile launch wording is safer.

- Current robot camera profile gate is `status=ready` for `launch_scope=sim_only`.
- It explicitly says owner calibration is not required for sim-only launch, but is required for physical robot launch.
- The owner calibration request lists optional physical robot calibration inputs for Unitree G1 head and chest RGB-D cameras.

G1 field evidence shape is better, but still not proof.

- `g1_controlled_proof_setup_manifest.json` is `setup_ready_external_operator_inputs_required`.
- The generated templates and field kit are not proof.
- `g1_controlled_run_evidence_assembly_manifest.json` is `not_requested_for_sim_only` with `physical_evidence_required=false`.
- The controlled field anchor packet is `not_requested_for_sim_only`, not accepted real-world evidence.

WAM/perception harness changes are moving in the right direction.

- Missing validation rows now become optional diagnostics for sim-only harness runs instead of blocking generated-provider support paths.
- Candidate selection now avoids claiming a winner when visual review, fixture-only labels, low confidence, or inconclusive scorecards remain.
- These changes improve claim boundaries, but do not prove generated-world rank fidelity or real-world performance.

## What Is Still Blocking

### P0 - Blocks sim-only policy-comparison launch

The product scope for this lane is sim-only policy comparison. Required launch evidence should focus on evaluator-bounded policy ranking quality, scenario coverage, failure/closure artifacts, and customer-facing delivery/forwarding where needed. Physical robot evidence is not a P0 requirement for this launch.

Current `sim_only_beta_release_gate_report.json` has `status=blocked` and `ready_for_beta_release=false`.

Local sim-only gate:

- `sim_only_beta_local_gate_report.status=blocked`
- blocker: `sim_only_beta_core_not_complete`
- simulator execution itself is proven for the local sample path

Current closure manifest state:

- `sim_only_beta_core_complete=false`
- `sim_only_beta_blocked_requirement_ids`: `failure_diagnosis`, `closure_audit`
- `failure_diagnosis` blocker: `failure_labels_not_accepted_or_reviewable`
- `closure_audit` blocker: `task_metric_closure_incomplete`

This means the immediate sim-only beta work is not another MuJoCo execution proof. It is closing the failure-diagnosis and closure-audit contract mismatch.

Additional sim-only launch requirements to keep explicit:

- policy A/B/C comparison must be evaluator-bounded and backed by symmetric scenario coverage
- the scorecard must not claim a winner when evidence is fixture-only, visually blocked, low-confidence, or inconclusive
- generated/WAM support media can support review but cannot by itself prove external rank fidelity
- production forwarding/intake must be proven only if this beta is customer-facing through the WebApp/Pipeline handoff

### Not P0 For Sim-Only Launch - Optional Claim Upgrades

These are useful future validation lanes, but they are not required before launching/testing the sim-only policy-comparison product:

- physical Unitree G1 controlled-run evidence
- accepted real robot POV/action/timestamp evidence
- owner-provided physical robot camera calibration
- safety validation, contact/physics acceptance, field abort/e-stop evidence
- accepted real-world anchors for sim-vs-real calibration, SRCC, MMRV, or Pearson

Keep these as claim-boundary safeguards: missing physical/deployment evidence means "do not claim physical readiness or real-world calibration," not "do not launch the sim-only policy-ranking product."

### P1 - Blocks broader robot-team-grade or claim-upgrade evaluation

Current robot-team-grade blockers:

- `failure_diagnosis`
- `digital_twin_fidelity_qa`
- `remote_cloud_execution_path`
- `closure_audit`

Digital twin fidelity blockers:

- `digital_twin_object_semantics_missing`
- `hidden_obstacle_or_proxy_truncation_review_required`
- `visible_objects_without_physics_coverage`
- `visual_collision_alignment_not_validated`

Remote/cloud execution:

- local execution does not require remote cloud proof
- the broader robot-team-grade path still needs pinned remote/provider inputs, cost controls, timeout handling, artifact output, and clean shutdown proof

### P1 - Blocks production beta release

Production release gate blockers include:

- production forwarding preflight not ready with probe
- forwarding token not configured
- forwarding probe not attempted or reachable
- forwarding probe audit not staged for control plane
- production deployment status not ready
- production deployment not proven
- Pipeline intake health not ready

Current `sim_only_beta_production_deployment_proof.json` narrows this to:

- `pipeline_intake_token_missing`
- `pipeline_intake_health_not_ready`

The new helper `blueprint-setup-live-pipeline-forwarding` is the right local setup path, but a local env file is not enough. The same token must exist in WebApp forwarding config and Pipeline intake service config.

### P2 - Blocks public/customer confidence, not local plumbing

- The stale `docs/CHANGELOG.md` caveat about only one uncommitted item was corrected during the stabilization pass.
- The duplicated generated-world rank-fidelity wording in `README.md`, `docs/CHANGELOG.md`, and `src/blueprint_pipeline/isaac_g1_site_3dgs_realistic_eval.py` was corrected during the stabilization pass.
- The stale audit JSON references an older HEAD and older blocker set. Regenerate it before treating it as current evidence.

## Recommended Closure Order

1. Stabilize the dirty worktree and run targeted tests over the changed surfaces.
2. Fix sim-only beta core closure: failure labels and closure audit.
3. Regenerate local gate and release gate artifacts after the fix.
4. Configure and prove production forwarding and Pipeline intake health with a non-secret env contract.
5. Fix digital-twin fidelity blockers.
6. Add remote/cloud provider execution proof only after local and production gates are clean.
7. Keep physical G1, safety, and real-world-anchor work as optional claim-upgrade loops, not sim-only launch blockers.
8. Clean stale docs/changelog language and push only after the user explicitly requests publishing.

## Copy/Paste Goal Loops

Each loop below is deliberately scoped so the next session can work for a long time without confusing sim-only policy comparison, WAM support, production delivery, and optional physical/deployment claim upgrades.

### Loop 1 - Worktree Stabilization And Current Evidence Refresh

```text
/goal In /Users/nijelhunt_1/workspace/BlueprintCapturePipeline, deeply stabilize the current dirty worktree and refresh the launch-readiness evidence without changing scope or overclaiming.

Read first:
- /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/PLATFORM_CONTEXT.md
- /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/WORLD_MODEL_STRATEGY_CONTEXT.md
- /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/README.md
- /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/pyproject.toml
- /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/last_24h_launch_audit_2026-06-26.md

Current facts to verify, not assume:
- HEAD and origin/main are aa37e89ef1798260c3c6bf9c35bf83b72a545022 at the time this prompt was written.
- The current tree has about 96 dirty status entries across docs, scripts, src, tests, plus untracked live-pipeline forwarding setup and OSCAR visual augmentation files.
- output/beta_launch_readiness_deep_audit_current.json is stale against the current checkout and older blockers.

Do:
- Inspect `git status --short --branch`, `git diff --stat`, and all dirty files.
- Separate existing user/session changes from any changes you make. Do not revert user changes.
- Fix stale or internally inconsistent docs only when the inconsistency is proven current, especially the `docs/CHANGELOG.md` caveat about only one uncommitted file.
- Regenerate or replace the stale launch-readiness audit JSON only if there is an existing command/script path for it. If no durable command exists, document the current manifest-derived evidence in the markdown handoff and do not invent an ad hoc proof format.
- Run `git diff --check`.
- Run targeted tests for every changed code surface first, then run the broad suite only if disk/time allows.

Preferred commands:
- `git status --short --branch`
- `git diff --stat`
- `git diff --check`
- `python -m ruff check src/blueprint_pipeline scripts tests`
- `python -m pytest tests/test_live_pipeline_forwarding_secret_setup.py tests/test_sim_only_beta_local_gate.py tests/test_sim_only_beta_release_gate.py tests/test_sim_only_beta_deployment_parity_proof.py tests/test_robot_eval_job_orchestrator.py tests/test_robot_initial_observation.py tests/test_g1_controlled_proof_setup.py tests/test_g1_controlled_run_evidence.py tests/test_g1_field_run_capture.py tests/test_wam_derived_observation_harness.py tests/test_wam_fixture_evaluator.py tests/test_wam_perception_harness_gpu_image.py -q --basetemp=/private/tmp/blueprint-pytest-basetemp-current-launch-audit`
- If targeted tests pass and disk is healthy: `PYTHONDONTWRITEBYTECODE=1 python -m pytest --cache-clear --basetemp=/private/tmp/blueprint-pytest-basetemp-full-launch-audit`

Claim boundaries:
- Do not claim deployment approval, safety validation, physical robot readiness, or generated-world rank fidelity from local sim or generated artifacts.
- Do not push or publish unless the user explicitly asks.

Stop only when:
- The current dirty tree is understood file-by-file.
- Stale docs/audit artifacts are either corrected or called out with exact paths.
- Current tests run or any blocker is named with exact command output.
- docs/last_24h_launch_audit_2026-06-26.md is updated with the latest evidence.
```

### Loop 2 - Close The Sim-Only Beta Core Blockers

```text
/goal In /Users/nijelhunt_1/workspace/BlueprintCapturePipeline, close the current sim-only beta core blockers for the first-gpu walkthrough capture root. Do not expand this into physical robot readiness or production deployment.

Use this capture root:
/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/output/first-gpu-walkthrough2-storage/local-blueprint/scenes/first-gpu-walkthrough-2/captures/downloads-walkthrough2-20260611

Use this job id:
robot-eval-sim-only-beta-local-gate-capture-intent-first-gpu-humanoid-navigation-smoke-61099fcc72

Read first:
- docs/last_24h_launch_audit_2026-06-26.md
- scripts/run_sim_only_beta_local_gate.py
- scripts/run_sim_only_beta_release_gate.py
- src/blueprint_pipeline/robot_eval_job_orchestrator.py
- src/blueprint_pipeline/live_robot_eval_closure.py
- tests/test_sim_only_beta_local_gate.py
- tests/test_sim_only_beta_release_gate.py
- tests/test_robot_eval_job_orchestrator.py
- tests/test_live_robot_eval_closure_coverage_edges.py

Current blockers to verify:
- `pipeline/live_pipeline_control_plane/sim_only_beta_local_gate/sim_only_beta_local_gate_report.json` has `status=blocked`.
- Its only blocker should be `sim_only_beta_core_not_complete`.
- `pipeline/robot_eval_jobs/<job_id>/robot_team_grade_eval_closure_manifest.json` has `sim_only_beta_blocked_requirement_ids` equal to `failure_diagnosis` and `closure_audit`.
- `failure_diagnosis` blocks on `failure_labels_not_accepted_or_reviewable`.
- `closure_audit` blocks on `task_metric_closure_incomplete`.
- Simulator execution is already proven for this local sim-only path. Do not rerun MuJoCo until you prove it is necessary.

Do:
- Trace why deterministic sim failure labels with `review_status=available_for_human_audit_not_required_for_sim_only_metric` still fail the sim-only `failure_diagnosis` gate.
- Decide whether the bug is in failure-label artifact shape, closure interpretation, or local-gate requirements.
- Preserve the claim boundary: deterministic sim labels can satisfy sim-only metric closure only if explicitly marked reviewable for sim-only, but they must not become rank-fidelity, real-world, or physical success labels.
- Fix the smallest correct contract/code/test surface.
- Regenerate the affected job manifest and gate reports using the repo's existing commands or scripts. Do not hand-edit output JSON as proof.
- Re-run the local gate until it either passes or blocks on a new concrete reason.
- Re-run the release gate after local gate changes so production blockers are clearly separated from local blockers.

Expected verification:
- `python -m pytest tests/test_sim_only_beta_local_gate.py tests/test_sim_only_beta_release_gate.py tests/test_robot_eval_job_orchestrator.py tests/test_live_robot_eval_closure_coverage_edges.py -q --basetemp=/private/tmp/blueprint-pytest-basetemp-sim-only-beta-core`
- `git diff --check`

Acceptance:
- `sim_only_beta_local_gate_report.json` no longer blocks on `sim_only_beta_core_not_complete`, or the remaining blocker is newly identified and more precise.
- `robot_team_grade_eval_closure_manifest.json` has `sim_only_beta_core_complete=true` only if the actual sim-only required requirements pass.
- Production release may still be blocked. That is acceptable and should be reported separately.

Stop only when:
- The local sim-only blocker is fixed or replaced by a concrete, newly discovered blocker with exact artifact paths and commands.
```

### Loop 3 - Production Forwarding, Token, Intake Health, And Deployment Parity

```text
/goal In /Users/nijelhunt_1/workspace/BlueprintCapturePipeline, close the production WebApp-to-Pipeline forwarding and deployment parity blockers for the sim-only beta gate without printing secrets or making physical-readiness claims.

Read first:
- docs/last_24h_launch_audit_2026-06-26.md
- docs/LIVE_PIPELINE_SETUP.md
- README.md live pipeline and sim-only beta sections
- src/blueprint_pipeline/live_pipeline_forwarding_secret_setup.py
- src/blueprint_pipeline/live_pipeline_intake_service.py
- scripts/run_sim_only_beta_deployment_parity_proof.py
- scripts/run_sim_only_beta_release_gate.py
- tests/test_live_pipeline_forwarding_secret_setup.py
- tests/test_sim_only_beta_deployment_parity_proof.py
- tests/test_sim_only_beta_release_gate.py

Current blockers to verify:
- `sim_only_beta_release_gate_report.json` has production forwarding preflight blockers, missing forwarding token, probe not reachable/not attempted, and pipeline intake health not ready.
- `sim_only_beta_production_deployment_proof.json` currently blocks on `pipeline_intake_token_missing` and `pipeline_intake_health_not_ready`.

Do:
- Use `blueprint-setup-live-pipeline-forwarding` or `python -m blueprint_pipeline.live_pipeline_forwarding_secret_setup` to create or reuse a local ignored env file at `$HOME/.blueprint-secrets/live_pipeline_forwarding.env`.
- Confirm the helper writes the raw token only to the env file and never to stdout or a manifest.
- Start or verify the Pipeline intake service using the same token.
- From the WebApp repo, run the forwarding preflight with the same env file and an intake probe, if WebApp is available.
- Produce a route-forwarding proof for the exact same capture root, not an older mismatched server-side path.
- Run `scripts/run_sim_only_beta_deployment_parity_proof.py` with the forwarding env file, WebApp URL, Pipeline intake URL, and route proof.
- Re-run `scripts/run_sim_only_beta_release_gate.py`.

Commands to adapt:
- `python -m blueprint_pipeline.live_pipeline_forwarding_secret_setup --env-file "$HOME/.blueprint-secrets/live_pipeline_forwarding.env" --forward-url "https://paperclip.tryblueprint.io/api/live-pipeline/job-requests" --capture-root "$CAPTURE_ROOT" --site-slug "$WEBAPP_SITE_SLUG"`
- `set -a; source "$HOME/.blueprint-secrets/live_pipeline_forwarding.env"; set +a`
- `blueprint-live-pipeline-intake-service --host 127.0.0.1 --port 8765`
- From `/Users/nijelhunt_1/workspace/Blueprint-WebApp`: `npm run pipeline:forwarding:preflight -- --require-forwarding --probe-intake-audit --forwarding-env-file "$HOME/.blueprint-secrets/live_pipeline_forwarding.env"`
- `python scripts/run_sim_only_beta_deployment_parity_proof.py --capture-root "$CAPTURE_ROOT" --route-forwarding-proof "$ROUTE_PROOF" --webapp-url https://www.tryblueprint.io --pipeline-intake-url https://paperclip.tryblueprint.io/api/live-pipeline/job-requests --forwarding-env-file "$HOME/.blueprint-secrets/live_pipeline_forwarding.env"`

Claim boundaries:
- A local env file is not production authentication proof by itself.
- WebApp health does not prove Pipeline intake health.
- Production forwarding does not prove simulator correctness, physical robot readiness, safety, or real-world success.
- Do not print the token or commit secrets.

Expected tests:
- `python -m pytest tests/test_live_pipeline_forwarding_secret_setup.py tests/test_sim_only_beta_deployment_parity_proof.py tests/test_sim_only_beta_release_gate.py -q --basetemp=/private/tmp/blueprint-pytest-basetemp-forwarding`
- `git diff --check`

Stop only when:
- The production deployment parity proof has current, exact blockers or passes.
- Release gate output separates production blockers from local sim-only blockers.
- No secret has been written to stdout, markdown, git-tracked files, or JSON manifests.
```

### Loop 4 - Digital Twin Fidelity, Object Semantics, And Collision Coverage

```text
/goal In /Users/nijelhunt_1/workspace/BlueprintCapturePipeline, close the robot-team-grade digital-twin fidelity blockers for the first-gpu walkthrough MuJoCo G1 sim path while preserving the boundary that this remains simulator/support evidence.

Use this capture root:
/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/output/first-gpu-walkthrough2-storage/local-blueprint/scenes/first-gpu-walkthrough-2/captures/downloads-walkthrough2-20260611

Read first:
- docs/last_24h_launch_audit_2026-06-26.md
- src/blueprint_pipeline/mujoco_g1_simulator_command.py
- src/blueprint_pipeline/robot_eval_job_orchestrator.py
- tests/test_mujoco_g1_wam_vla_policy_endpoint_eval.py
- tests/test_robot_eval_job_orchestrator.py
- output/.../pipeline/simulation_automation/mujoco_g1_simulator_command/mujoco_digital_twin_fidelity_qa.json
- output/.../pipeline/robot_eval_jobs/<job_id>/robot_team_grade_eval_closure_manifest.json

Current blockers to verify:
- `digital_twin_object_semantics_missing`
- `hidden_obstacle_or_proxy_truncation_review_required`
- `visible_objects_without_physics_coverage`
- `visual_collision_alignment_not_validated`

Do:
- Inspect the digital-twin QA generator and the current QA artifact.
- Identify which visible objects lack simulator physics coverage, which semantics are missing, and whether collision alignment can be validated from existing scene/trace artifacts or requires a rerun.
- Fix the contract or generation path so the QA artifact names concrete missing objects/coverage gaps, not only generic blockers.
- If possible, add object-level evidence refs into the QA artifact so a reviewer can see why each blocker remains or passes.
- Rerun only the necessary MuJoCo/simulation automation commands after code fixes.
- Regenerate the robot-team-grade closure manifest and confirm the blocker list changes.

Do not:
- Treat visual collision alignment as physical collision validation.
- Treat simulator object semantics as raw capture truth.
- Claim generated-world rank fidelity from digital-twin QA.

Expected tests:
- `python -m pytest tests/test_mujoco_g1_wam_vla_policy_endpoint_eval.py tests/test_robot_eval_job_orchestrator.py tests/test_sim_only_beta_local_gate.py -q --basetemp=/private/tmp/blueprint-pytest-basetemp-digital-twin-qa`
- `git diff --check`

Stop only when:
- The digital-twin QA artifact either passes with concrete evidence or blocks with object-level actionable evidence.
- The robot-team-grade closure manifest reflects the updated state.
```

### Loop 5 - Remote/Cloud Provider Execution Closure With Spend Controls

```text
/goal In /Users/nijelhunt_1/workspace/BlueprintCapturePipeline, prepare and prove the remote/cloud provider execution path for robot-team-grade closure only after local sim-only evidence is clean. Keep provider spend bounded and separate from launch approval.

Read first:
- docs/last_24h_launch_audit_2026-06-26.md
- docs/GPU_VM_RUNBOOK.md
- docs/WAM_POLICY_EVALUATION_SERVICE.md
- src/blueprint_pipeline/robot_eval_job_orchestrator.py
- src/blueprint_pipeline/robot_eval_execution.py
- src/blueprint_pipeline/runpod_provider_adapter.py
- src/blueprint_pipeline/runpod_wam_async_runner.py
- tests/test_robot_eval_job_orchestrator.py
- tests/test_runpod_provider_adapter.py
- tests/test_runpod_wam_async_runner.py

Current blocker to verify:
- `remote_cloud_execution_path` blocks on `remote_cloud_execution_not_proven`.
- `remote_cloud_execution_closure_manifest.json` is `not_required_for_local_execution` and `remote_cloud_execution_proven=false`.

Preconditions:
- Local sim-only beta core blockers should be closed first.
- Production forwarding/token/intake should be known current.
- Paid provider run must be explicitly authorized by the user before spend.

Do:
- Build or verify a provider/worker package with pinned input manifest, exact job id, capture root, output URI, timeout, max spend, and teardown behavior.
- Ensure artifact output destination is provider-writable and fetchable.
- Run dry-run/package validation first.
- If the user authorizes paid provider execution, run one bounded provider attempt.
- Track phase, pod/job id, start time, max wait, watchdog boundary, output zip/object size, artifact manifest, and teardown status.
- Download and validate provider artifacts.
- Write or refresh `remote_cloud_execution_closure_manifest.json`.
- Do not collapse policy-only success, WAM provider success, valid output artifact, and clean shutdown into one boolean.

Claim boundaries:
- Remote/cloud execution does not prove physical robot readiness.
- Provider execution does not prove generated-world rank fidelity.
- Valid artifact upload does not prove rollout visual quality.
- Continuing spend must be reported explicitly.

Expected tests before live spend:
- `python -m pytest tests/test_robot_eval_job_orchestrator.py tests/test_runpod_provider_adapter.py tests/test_runpod_wam_async_runner.py tests/test_robot_eval_execution_coverage_edges.py -q --basetemp=/private/tmp/blueprint-pytest-basetemp-remote-cloud-closure`
- `git diff --check`

Stop only when:
- The provider path is proven by current artifacts or blocked by one named external condition.
- Spend state and teardown state are explicit.
```

### Loop 6 - Optional Physical Unitree G1 Controlled-Run Evidence Kit

```text
/goal In /Users/nijelhunt_1/workspace/BlueprintCapturePipeline, turn the current G1 controlled-run templates into a real evidence collection path for a physical Unitree G1 controlled test, but do not claim physical readiness until owner evidence is actually supplied and accepted.

Read first:
- docs/last_24h_launch_audit_2026-06-26.md
- src/blueprint_pipeline/g1_controlled_proof_setup.py
- src/blueprint_pipeline/g1_field_run_capture.py
- src/blueprint_pipeline/g1_controlled_run_evidence.py
- tests/test_g1_controlled_proof_setup.py
- tests/test_g1_field_run_capture.py
- tests/test_g1_controlled_run_evidence.py
- output/.../pipeline/g1_controlled_proof_setup/g1_controlled_proof_setup_manifest.json
- output/.../pipeline/g1_controlled_proof_setup/field_run_capture_kit/*
- output/.../pipeline/g1_controlled_proof_setup/owner and physical evidence templates

Current facts:
- `g1_controlled_proof_setup_manifest.json` is `setup_ready_external_operator_inputs_required`.
- Templates are not proof.
- Current assembled live inputs are `not_requested_for_sim_only`.
- Owner robot camera calibration is optional for sim-only and required for physical robot launch.

Do:
- Validate the generated field-run capture kit and expected evidence files.
- Ensure the field kit requires exact `scenario_eval_run_id`, `policy_id`, `task_id`, and `scenario_variation_instance_id`.
- Make sure allowed task set, operator site checklist, abort criteria, robot calibration refs, camera calibration refs, real robot POV, action logs, timestamp alignment, hardware validation, contact/collision logs, policy metrics, and robot-team review are all required before any physical claim.
- Generate a clear owner/operator checklist and evidence-drop structure.
- Add or strengthen tests so placeholder evidence, mismatched keys, missing owner evidence, or unsigned attestations fail closed.
- If real field evidence is present, assemble it through the CLI and inspect the resulting accepted/blocked manifest.

Do not:
- Use generated simulator POV as real robot POV.
- Use field-run templates as evidence.
- Accept loose task/scenario names instead of exact join keys.
- Claim safety validation unless reviewed safety evidence exists.

Expected tests:
- `python -m pytest tests/test_g1_controlled_proof_setup.py tests/test_g1_field_run_capture.py tests/test_g1_controlled_run_evidence.py -q --basetemp=/private/tmp/blueprint-pytest-basetemp-g1-field-evidence`
- `git diff --check`

Stop only when:
- The physical evidence kit fails closed on every missing/mismatched input.
- The next human/operator step is concrete and path-specific.
```

### Loop 7 - Owner Robot Camera Calibration For Physical G1 Launch

```text
/goal In /Users/nijelhunt_1/workspace/BlueprintCapturePipeline, close the owner robot camera calibration gap for physical Unitree G1 testing while preserving sim-only defaults as support-only.

Use this current request artifact:
output/first-gpu-walkthrough2-storage/local-blueprint/scenes/first-gpu-walkthrough-2/captures/downloads-walkthrough2-20260611/pipeline/robot_eval_jobs/robot-eval-sim-only-beta-local-gate-capture-intent-first-gpu-humanoid-navigation-smoke-61099fcc72/owner_robot_camera_calibration_request.json

Read first:
- docs/last_24h_launch_audit_2026-06-26.md
- src/blueprint_pipeline/robot_initial_observation.py
- tests/test_robot_initial_observation.py
- output/.../robot_camera_profile_launch_readiness.json
- output/.../owner_robot_camera_calibration_request.json

Current facts:
- `robot_camera_profile_launch_readiness.status=ready` only for `launch_scope=sim_only`.
- `physical_robot_launch_ready=false` for Unitree G1 profiles.
- Missing physical inputs include owner intrinsics and FOV for head and chest RGB-D cameras.
- Owner extrinsics are still required for physical robot launch even when default/derived values let sim-only proceed.

Do:
- Make the calibration request artifact impossible to misread as calibration proof.
- Validate owner-provided camera profile input shape for Unitree G1 head and chest RGB-D cameras.
- Require pixel-unit intrinsics, FOV degrees, and robot-base-to-camera extrinsics with clear frame names.
- Add fixture tests for accepted calibration, partial calibration, default-only sim-only readiness, and physical-launch blocking.
- Regenerate robot camera profile launch readiness and owner calibration request artifacts after code changes.

Claim boundaries:
- Sim-only camera launch ready does not prove physical robot sensor truth.
- Owner calibration request does not prove owner calibration.
- Defaults can support sim-only smoke, not physical launch.

Expected tests:
- `python -m pytest tests/test_robot_initial_observation.py -q --basetemp=/private/tmp/blueprint-pytest-basetemp-camera-calibration`
- `git diff --check`

Stop only when:
- Physical launch calibration blockers are explicit, machine-readable, and tested.
```

### Loop 8 - Real-World Anchors And Sim-Vs-Real Calibration

```text
/goal In /Users/nijelhunt_1/workspace/BlueprintCapturePipeline, implement and verify the optional real-world anchor ingestion/calibration path for the current policy-eval job. Do not treat this as required for the sim-only policy-comparison launch, and do not compute or claim SRCC/MMRV/Pearson until exact accepted anchors exist.

Read first:
- docs/last_24h_launch_audit_2026-06-26.md
- docs/WAM_POLICY_EVALUATION_SERVICE.md
- src/blueprint_pipeline/g1_controlled_run_evidence.py
- src/blueprint_pipeline/robot_eval_execution.py
- src/blueprint_pipeline/robot_eval_job_orchestrator.py
- tests/test_g1_controlled_run_evidence.py
- tests/test_robot_eval_job_orchestrator.py

Current artifact:
- `sim_vs_real_calibration_report.json` is `status=not_measured`.
- Blockers: `insufficient_anchor_count`, `insufficient_policy_group_count`, `unmatched_prediction_rows`.
- All current prediction rows are unmatched because no accepted anchors exist for the exact `scenario_eval_run_id`, `policy_id`, `task_id`, and `scenario_variation_instance_id` join keys.

Do:
- Confirm current prediction rows and required exact join keys.
- Accept only anchors with owner evidence, physical evidence where requested, signed/reviewed decisions, and exact join keys.
- Reject loose/inferred joins.
- Add tests for no anchors, insufficient anchors, one-policy-only anchors, mismatched keys, unsigned anchors, and accepted multi-policy anchors.
- Regenerate `sim_vs_real_calibration_report.json` only through the existing pipeline path.

Claim boundaries:
- Real-world anchors are calibration evidence only after accepted review.
- Single-policy anchors do not prove policy ranking correlation.
- MMRV/Pearson/Spearman require enough paired real-world anchors and must remain `not_measured` otherwise.

Expected tests:
- `python -m pytest tests/test_g1_controlled_run_evidence.py tests/test_robot_eval_job_orchestrator.py -q --basetemp=/private/tmp/blueprint-pytest-basetemp-real-world-anchors`
- `git diff --check`

Stop only when:
- The calibration report either measures from accepted anchors or blocks with exact missing anchor counts/keys.
```

### Loop 9 - WAM/Perception Harness Review-Quality And Candidate Selection

```text
/goal In /Users/nijelhunt_1/workspace/BlueprintCapturePipeline, audit and harden the WAM/perception harness and candidate-selection path so generated WAM artifacts can support review without becoming unsupported winner, physical-readiness, or rank-fidelity claims.

Read first:
- docs/last_24h_launch_audit_2026-06-26.md
- docs/WAM_POLICY_EVALUATION_SERVICE.md
- src/blueprint_pipeline/wam_derived_observation_harness.py
- src/blueprint_pipeline/wam_fixture_evaluator.py
- src/blueprint_pipeline/wam_real_provider_validation_probe.py
- src/blueprint_pipeline/wam_perception_harness_gpu_image.py
- tests/test_wam_derived_observation_harness.py
- tests/test_wam_fixture_evaluator.py
- tests/test_wam_perception_harness_gpu_image.py

Current behavior to verify:
- Missing labeled validation rows should be `not_requested` or diagnostics for sim-only generated-provider support paths, not blockers.
- False-success reduction must remain `not_measured` without accepted labels.
- Candidate selection must not claim a single winner when visual review blockers, fixture-only labels, OOD blockers, low confidence, or inconclusive scorecards exist.
- Generated videos/review queues must not claim forward/inverse episode consistency unless an external scorer result exists.

Do:
- Inspect current WAM fixture outputs under `candidate-selection-failure-diagnosis-handoff-20260626-fixture`.
- Verify policy scorecard, candidate selection report, visual review blocker summary, customer handoff report, WAM claim boundary, and validation reports all agree.
- Add or fix tests for visual-review-required candidate shortlist, fixture-only label blocking, missing validation labels as diagnostics, timeout stdout/stderr capture, and false-success metrics boundaries.
- Regenerate only fixture/support artifacts needed to prove the path.

Claim boundaries:
- WAM-generated observations are support artifacts downstream of capture.
- Reviewability is not task success.
- Candidate shortlist is not a winner claim.
- Generated-world rank fidelity requires scoped accepted evidence and cannot be inferred from fixture rollouts.

Expected tests:
- `python -m pytest tests/test_wam_derived_observation_harness.py tests/test_wam_fixture_evaluator.py tests/test_wam_perception_harness_gpu_image.py -q --basetemp=/private/tmp/blueprint-pytest-basetemp-wam-harness`
- `git diff --check`

Stop only when:
- WAM artifacts distinguish support, diagnostics, reviewable output, candidate shortlist, and blocked winner claims with tests.
```

### Loop 10 - Package Delivery, Review Acceptance, And Customer Handoff

```text
/goal In /Users/nijelhunt_1/workspace/BlueprintCapturePipeline, close the package delivery, review acceptance, rights/privacy, and customer-handoff gaps for the sim-only beta and robot-team-grade paths without treating delivery as deployment approval.

Read first:
- docs/last_24h_launch_audit_2026-06-26.md
- src/blueprint_pipeline/live_robot_eval_closure.py
- src/blueprint_pipeline/post_training_data_package.py
- src/blueprint_pipeline/robot_eval_job_orchestrator.py
- tests/test_live_robot_eval_closure_coverage_edges.py
- tests/test_robot_eval_job_orchestrator.py
- tests/test_post_training_data_package.py

Current closure blockers to verify:
- `review_acceptance_evidence_missing`
- `signed_delivery_evidence_missing`
- `signed_delivery_access_not_proven`
- `failure_labels_not_accepted_or_reviewable`
- `task_metric_closure_incomplete`
- WebApp upstream truth still missing or ungrounded for some live closure paths.

Do:
- Determine which blockers are required for sim-only beta core and which are only robot-team-grade or evaluation-readiness.
- Add exact proof-boundary fields to any package or delivery artifact that could be confused for deployment approval.
- Ensure post-training data package export manifests, proof boundaries, review acceptance, signed delivery, and rights/privacy records are all referenced by closure.
- Add tests for missing signed URL, missing reviewer, ungrounded WebApp IDs, and sim-only optional vs required evidence.
- Regenerate closure artifacts through the real CLI/path.

Claim boundaries:
- Signed delivery proves access/delivery only.
- Review acceptance proves reviewer acceptance of the specified artifact, not safety validation or robot readiness.
- Rights/privacy clearance must be exact to the use.

Expected tests:
- `python -m pytest tests/test_live_robot_eval_closure_coverage_edges.py tests/test_robot_eval_job_orchestrator.py tests/test_post_training_data_package.py -q --basetemp=/private/tmp/blueprint-pytest-basetemp-package-delivery`
- `git diff --check`

Stop only when:
- Closure artifacts show exactly which evidence is required for sim-only beta, robot-team-grade evaluation, and real-world readiness.
```

## Non-Negotiable Claim Boundaries For Every Loop

- Local route forwarding is not production forwarding.
- Production WebApp health is not Pipeline intake health.
- MuJoCo execution is not physical robot execution.
- Generated WAM video is not raw capture evidence.
- Review-quality media is not task success.
- Sim-only task metrics are not generated-world rank fidelity.
- Candidate shortlist is not a winner claim.
- Physical field-run templates are not physical field-run evidence.
- Owner calibration request is not owner calibration proof.
- Real-world calibration needs accepted anchors with exact join keys.
- Sim-only policy-comparison launch claims require evaluator-bounded policy-ranking evidence and honest delivery/proof boundaries. Physical robot, safety, and real-world calibration evidence are only required for those optional claim upgrades.

## Short Next Step Recommendation

The highest leverage next session is Loop 2. The current local sim-only path already proves route/intake/control-plane processing, semantic target coverage, and MuJoCo execution for 11 runs. Closing `failure_diagnosis` and `closure_audit` should convert the local sim-only beta gate from "blocked despite simulator execution" into a cleaner pass or a more precise blocker. After that, Loop 3 can isolate production forwarding and intake health without mixing in robot-team-grade proof.
