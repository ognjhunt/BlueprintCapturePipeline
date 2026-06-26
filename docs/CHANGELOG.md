# BlueprintCapturePipeline Changelog

## 2026-06-25

### User-Facing

- Hardened WAM provider rollout review for generated-video support artifacts.
  RunPod/Vast WAM paths now carry review queues, synthetic seed metadata, and
  provider artifact handling through the pipeline, but generated videos remain
  review/support evidence only, not raw capture truth, live-robot proof,
  deployment proof, safety validation, or generated-world rank-fidelity proof.
- Connected scene WAM episode packets to capture-derived robot POV synthesis.
  For each task and robot profile, the packet can now write source QA,
  coverage/quality reports, contact sheets, and recapture guidance when no
  depth-splat candidate passes. Passing synthesized/splatted frames can seed the
  WAM initial-observation lane, but remain explicitly labeled as support
  artifacts, not raw capture truth, owner-run POV evidence, safety
  validation, or generated-world rank-fidelity result.
- Added a WAM-derived observation/perception harness lane for policy/WAM loops.
  The new harness can package WAM-derived observations, perception checks,
  adapter reports, step traces, and optional external perception backend
  requests/results, while keeping those artifacts downstream of capture
  provenance and separate from deployment-readiness claims.
- Clarified WAM/substrate evaluation as evaluator-bounded policy comparison:
  policy ranking scorecards can compare policy A/B/C inside the configured
  evaluator, while MMRV/Pearson/Spearman require real-world anchors and do not
  create deployment-readiness or physical-readiness claims.

### Employee-Facing

- Added and documented new runtime modules and tests for generated-video review,
  synthetic WAM seeding, persistent short visual sanity checks, capture-derived
  initial policy observations, WAM auxiliary observations, WAM-derived
  observation harnesses, WAM perception harness GPU image packaging, real
  provider validation probes, and sim-provider E2E support. Key paths include
  `src/blueprint_pipeline/wam_generated_video_review.py`,
  `src/blueprint_pipeline/synthetic_2d_wam_seed.py`,
  `src/blueprint_pipeline/persistent_wam_short_visual_sanity.py`,
  `src/blueprint_pipeline/robot_initial_observation.py`,
  `src/blueprint_pipeline/wam_auxiliary_observation.py`,
  `src/blueprint_pipeline/wam_derived_observation_harness.py`,
  `src/blueprint_pipeline/wam_perception_harness_gpu_image.py`,
  `src/blueprint_pipeline/wam_real_provider_validation_probe.py`, and
  `src/blueprint_pipeline/wam_sim_provider_e2e.py`.
- Added CLI entrypoints in `pyproject.toml` for short WAM visual sanity,
  WAM real-provider validation, WAM sim-provider E2E, WAM perception harness GPU
  image builds, rollout vision labeling, post-training data package builds, and
  several live-pipeline / arena package audit and delivery commands.
- Expanded release/local gate scripts and live-robot closure paths so rollout
  review, failure diagnosis, image remediation, visual labels, simulator
  command artifacts, and webapp status projections stay explicit instead of
  being collapsed into a single readiness claim.

### Future-Agent-Facing

- Contract changes: WAM jobs can now emit
  `robot_policy_wam_closed_loop/wam_derived_observation_harness/*`,
  `vision_success_labels.json`, `wam_vision_success_review_queue.json`,
  `wam_episode_consistency_request.json`, and short visual sanity manifests.
  Treat these as support/review artifacts unless a separate accepted scorer,
  provider runtime output, or real-world anchor upgrades the claim.
- Runtime behavior changes: capture-derived initial observations and WAM-derived
  observations may seed policy loops, but raw capture/provenance, rights, and
  privacy metadata still outrank downstream generated frames, labels, and
  review queues.
- Launch/readiness gate changes: short learned-WAM visual sanity is now a
  first-class precondition before longer review-quality learned-WAM rollout
  claims. Provider probes and sim-provider E2E outputs remain opt-in runtime
  evidence, not deployment approval.
- Proof boundary: the repo-wide claim language was aligned to rank-fidelity
  scope. Evaluator-bounded policy comparisons can be recorded, but
  MMRV/Pearson/Spearman, public readiness, simulator
  validity, and generated-world rank fidelity require separate accepted
  evidence.
- Local caveat: this June 25 entry predates the June 26 stabilization pass. The
  current checkout has broad uncommitted docs, scripts, source, tests, and
  untracked support files; use `docs/last_24h_launch_audit_2026-06-26.md` for
  current worktree evidence.

## 2026-06-24

### User-Facing

- Committed provider-worker session contracts for robot-eval jobs through
  `src/blueprint_pipeline/provider_worker_contract.py`,
  `src/blueprint_pipeline/provider_worker_endpoint_manifest.py`,
  `src/blueprint_pipeline/provider_worker_policy_command_adapter.py`, and
  `src/blueprint_pipeline/provider_worker_session_runner.py`. Repeated policy
  calls can now target one ready provider worker with `/readyz`, `/infer`, and
  optional `/shutdown` semantics instead of treating every inference as a fresh
  provider launch.
- Committed a Unitree GR00T N1.7 + SONIC Vast lane with
  `src/blueprint_pipeline/unitree_groot_n17_sonic_vast_image_canary.py`,
  `src/blueprint_pipeline/unitree_groot_n17_sonic_vast_persistent_session.py`,
  `src/blueprint_pipeline/unitree_groot_n17_sonic_vast_policy_command.py`, and
  `deploy/docker/robot_eval_worker/unitree_groot_sonic_vast/Dockerfile`. These
  are provider/runtime scaffolds for Task Evaluation Run support, not physical
  generated-world rank fidelity, off-scope validation, generated-world rank-fidelity result, or public claim proof.
- Extended the MuJoCo Unitree policy/WAM loop with a local OSCAR-style support
  backend for no-live-provider runs. The generated next-observation frames,
  short MP4 segments, and Unitree re-query attempts are loop/debug evidence only
  and explicitly do not prove a learned OSCAR/Cosmos checkpoint or physical
  robot sensor loop ran.

### Employee-Facing

- Added CLI entrypoints in `pyproject.toml` for provider-worker contracts,
  endpoint manifests, policy-command adapters, provider-worker sessions, and
  Unitree GR00T/SONIC Vast image canary, persistent session, and policy-command
  flows.
- Updated RunPod/Vast startup planning so provider endpoint discovery can be
  recorded as `provider_worker_endpoint_manifest.json`, with cost/teardown proof
  kept separate from endpoint discovery and readiness checks.
- Hardened Vast/OSCAR provider-bundle support for Unitree GR00T/SONIC runtime
  packaging, including provider-kind routing, HF token-file handling, runtime
  output/import checks, and tests across the provider adapters and bundle
  builders.
- Uncommitted local June 24 edits in
  `src/blueprint_pipeline/oscar_wam_command_adapter.py`,
  `tests/test_oscar_wam_command_adapter.py`,
  `src/blueprint_pipeline/runpod_provider_adapter.py`, and
  `tests/test_runpod_provider_adapter.py` add OSCAR subprocess timeout blocking
  and a configurable RunPod REST API base. Related provider-session edits
  continued after midnight on June 25 and are not summarized here as June 24
  material.

### Future-Agent-Facing

- Contract changes: provider-worker manifests now distinguish endpoint
  discovery from allocation, runtime readiness, teardown, cost control,
  simulator execution, safety, deployment, and rank-fidelity proof.
- Runtime behavior changes: repeated WAM/policy loops should use the
  provider-worker adapter/session path when a ready worker URL is available;
  one-shot provider launchers remain inappropriate for repeated inference loops.
- Launch/readiness gate changes: Vast/RunPod provider adapters still require
  explicit live API gates, artifact-output/finalizer destinations, and
  provider-native runtime evidence before any simulator or provider proof is
  upgraded.
- Proof boundary: Unitree GR00T/SONIC Vast canary, bundle, persistent-session,
  and policy-command artifacts are startup/runtime support artifacts unless they
  are paired with accepted provider execution outputs and downstream eval
  evidence. They do not supersede raw capture/provenance evidence and do not
  prove generated-world rank fidelity, off-scope validation, or
  real-world manipulation success.

## 2026-06-23

### User-Facing

- Committed MuJoCo-backed initial policy observation rendering for scene/WAM
  episode packets in `src/blueprint_pipeline/scene_wam_policy_episode_packet.py`.
  USD scenes can now fall back through a generated visual MJCF with texture
  export, bbox proxies for oversized meshes, and blank/uniform frame rejection
  before an observation is treated as useful review evidence. This is visual
  scene/render support only; it does not validate physics contact, safety, or
  generated-world rank fidelity.
- Committed eval-ready task grounding for WAM policy loops through
  `src/blueprint_pipeline/eval_ready_task_grounding.py` and the
  `blueprint-build-eval-ready-task-grounding` CLI. The new artifacts identify
  task targets, camera calibration quality, FK/projected skeleton support, and
  lightweight handle-state proxies for learned rollout requests while keeping
  those outputs downstream of raw capture/provenance truth.
- Extended OSCAR/Cosmos WAM evaluation to consume eval-ready grounding,
  projected skeleton traces, and optional policy-ranking outcome ledgers, then write
  `wam_prediction_outcome_correlation_ledger.json`. The correlation ledger is an
  audit/support artifact; generated rollouts, VLM labels, calibration gates, and
  handle proxies still do not prove physical contact, torque, task success,
  generated-world rank-fidelity result, or generated-world rank fidelity.

### Employee-Facing

- Added README artifact-contract coverage for
  `eval_ready_task_grounding.json`, `camera_calibration_quality_gate.json`,
  `robot_fk_projection_manifest.json`,
  `robot_fk_projected_skeleton_trace.jsonl`, `handle_proxy_state_check.json`,
  and `wam_prediction_outcome_correlation_ledger.json`.
- Updated object-index support to derive task-aware detector prompts from
  customer task text, giving downstream grounding a more direct target selection
  path without hardwiring the pipeline to one scene or model backend.
- Added focused tests for the MuJoCo render fallback, USD texture/MJCF export,
  blank-frame rejection, eval-ready task grounding, object-index prompt
  derivation, and WAM evaluator grounding/correlation behavior.
- Uncommitted local June 23 work extends the MuJoCo Unitree policy/WAM loop with
  a default local OSCAR-style support generator for no-live-provider runs,
  including action-conditioned next-observation frames, short MP4 segments,
  projected-skeleton/proprioception conditioning, and fresh Unitree policy
  re-query attempts. Several related test/runtime files were touched after
  midnight on June 24, so treat this as local, unmerged state rather than a
  committed June 23 release.

### Future-Agent-Facing

- Contract changes: `pyproject.toml` now exposes
  `blueprint-build-eval-ready-task-grounding`; WAM evaluation may copy grounding
  support artifacts into job directories and include them in substrate,
  rollout-input, scorecard, claim-boundary, and handoff manifests.
- Runtime behavior changes: scene/WAM packet rendering now verifies image
  content before accepting a frame and can render MJCF scenes directly or convert
  USD visual meshes into MuJoCo-renderable support geometry.
- Proof boundary: eval-ready grounding, projected skeleton traces, generated
  visual MJCFs, and prediction/outcome correlation records are support layers.
  They do not supersede raw capture/provenance evidence and do not prove live
  provider execution, public readiness, off-scope validation, or real-world
  manipulation success without separate accepted proof.
- Uncommitted caveat: local Unitree GR00T/SONIC Vast/provider packaging work
  touches `src/blueprint_pipeline/vast_provider_adapter.py`,
  `src/blueprint_pipeline/oscar_wam_provider_bundle.py`,
  `src/blueprint_pipeline/unitree_groot_n17_sonic_provider_smoke.py`, related
  tests, docs, and `deploy/docker/robot_eval_worker/unitree_groot_sonic_vast/`.
  It adds HF token-file handling, optional Docker image-login controls,
  provider-bundle preflight/runtime output checks, and a CUDA 12.4 runtime image,
  but remains uncommitted and should not be described as live provider proof.

## 2026-06-22

### User-Facing

- Committed Unitree-native G1 policy lanes through
  `docs/UNITREE_G1_POLICY_ENDPOINT_LANE.md`, the Unitree LeRobot/UnifoLM/GR00T
  adapters, and the MuJoCo endpoint evaluation path. The buyer-facing meaning is
  narrower than "robot ready": Unitree action-command plumbing, endpoint smoke
  results, and simulator-only MuJoCo artifacts can support Task Evaluation Run
  review, but they do not prove generated-world rank fidelity, generated-world rank-fidelity result,
  off-scope validation, or task success without separate accepted evidence.
- Committed a clearer WAM proof boundary: generated WAM rollouts and generated
  video success labels are support evidence, while forward/inverse episode
  consistency now requires a separate scorer output before it can be summarized
  in `wam_consistency_checks.json`.
- Recorded June 22 local proof artifacts for Unitree UnifoLM provider import,
  endpoint replay, and WAM requery attempts in generated `robot_eval_jobs/`
  directories. Those artifacts show action output and action chunks flowing
  through endpoint/replay paths, while fresh per-observation Unitree
  hand/manipulation policy execution remains blocked unless a live
  Unitree-specific command, server, or provider call runs for the current
  observation.

### Employee-Facing

- Added CLI entrypoints in `pyproject.toml` for OpenVLA comparison adapters,
  OSCAR/WAM provider commands and images, Unitree UnifoLM GPU/server/provider
  smoke paths, Unitree LeRobot runtime, GR00T N1.7 + SONIC preflight/runtime
  commands, and WAM episode-consistency labeling.
- Extended provider/runtime scaffolding across RunPod, Vast, OSCAR/Cosmos WAM,
  Unitree UnifoLM, and endpoint setup code while preserving file/env-secret
  boundaries and fail-closed runtime gates.
- Added focused tests for the new adapter, provider, runtime, image, endpoint,
  startup, and consistency-scorer contracts under `tests/`, matching the June 22
  code expansion rather than claiming live provider or real-robot proof.
- Uncommitted local June 22 work in
  `src/blueprint_pipeline/scene_wam_policy_episode_packet.py` and
  `tests/test_scene_wam_policy_episode_packet.py` adds MJCF/MuJoCo scene-target
  inspection, USD-to-MuJoCo visual MJCF fallback rendering, texture export,
  blank-frame rejection, and renderer content checks for initial policy
  observations. Treat it as local changelog-worthy state until committed.

### Future-Agent-Facing

- Added `docs/UNITREE_G1_POLICY_ENDPOINT_LANE.md` and updated README/WAM/manipulation
  docs to make Unitree-native policy endpoints the preferred G1 hand/manipulation
  path. OpenVLA remains a generic VLA candidate, while OSCAR/Cosmos/Unitree WMA
  remain evaluator/world-model support unless a real Unitree policy endpoint
  consumes observations and emits normalized G1 actions.
- Added a `unitree_unifolm` Vast provider-bundle kind and a self-contained
  Unitree UnifoLM provider bundle so remote policy smoke runs look for
  `run_unitree_unifolm_provider_runtime.sh` and
  `unitree_unifolm_policy_provider_output.json` instead of WAM runtime files.
- Added a fresh Unitree UnifoLM server/endpoint proof boundary: the current G1
  policy path uses the Unitree-native `/act` endpoint bridge for robot
  action-command execution, not OpenVLA or WAM as the G1 controller. The proof
  can mark endpoint action-command plumbing true while keeping dexterous task
  success and WAM re-observation blocked until those loops actually run.
- Added `blueprint-build-unitree-unifolm-gpu-image` to create a reusable CUDA
  12.4 Unitree UnifoLM VLA image context with torch 2.5.1/cu124, flash-attn,
  the Unitree source install, server launcher, and image healthcheck.
- Extended the OSCAR/Cosmos WAM evaluator's model candidate contract with
  `unitree_unifolm_vla_policy` and `unitree_unifolm_wma_policy`, including
  explicit command/checkpoint envs and checkpoint-source hints. Public checkpoint
  existence is still not endpoint execution proof.
- Added a separate WAM episode-consistency scorer contract through
  `docs/WAM_EPISODE_CONSISTENCY_SCORER.md`, keeping forward/inverse consistency
  labels outside WAM/provider execution and outside evaluator-owned scoring.
- The OSCAR/Cosmos WAM evaluator now treats `wam_episode_consistency_request.json`
  as scorer input, `wam_episode_consistency.command.json` as external scorer
  output, and `wam_consistency_checks.json` as normalized proof-bound support
  evidence. Generated rollout existence and generated-video success labels still
  do not prove forward/inverse consistency.

## 2026-06-21

### User-Facing

- Committed provider endpoint evaluation lanes through
  `src/blueprint_pipeline/mujoco_g1_wam_vla_policy_endpoint_eval.py`,
  `src/blueprint_pipeline/wam_vla_policy_endpoint_setup.py`,
  `src/blueprint_pipeline/wam_vla_policy_endpoint_server.py`, and
  `src/blueprint_pipeline/g1_endpoint_reference_adapter.py`, keeping the lane
  simulator/provider-bound rather than deployment proof.
- Committed OSCAR/Cosmos WAM support through
  `src/blueprint_pipeline/oscar_cosmos_wam_evaluator.py`,
  `src/blueprint_pipeline/oscar_wam_command_adapter.py`, and
  `src/blueprint_pipeline/oscar_wam_provider_bundle.py`, with generated WAM
  rollouts treated as downstream support artifacts.

### Employee-Facing

- Committed RunPod/Vast WAM provider planning and runner plumbing through
  `src/blueprint_pipeline/runpod_wam_async_runner.py`,
  `src/blueprint_pipeline/vast_provider_adapter.py`,
  `src/blueprint_pipeline/vast_wam_async_runner.py`,
  `src/blueprint_pipeline/vast_wam_authorized_runner.py`,
  `src/blueprint_pipeline/vast_bundle_staging.py`, and
  `src/blueprint_pipeline/wam_provider_object_store.py`.
- Hardened the OSCAR WAM provider bundle with dependency probing fixes and a
  transformer-engine shim in commits `9736381` and `813b12b`, covered by
  `tests/test_oscar_wam_provider_bundle.py`.
- Uncommitted local June 21 work added private hardware/IP controls for Policy
  Improvement Runs in `src/blueprint_pipeline/policy_improvement_run.py` and
  `docs/POLICY_IMPROVEMENT_RUN.md`, including `private_hardware_integration_plan.json`
  and sealed eval capsule language.

### Future-Agent-Facing

- Contract changes: committed entrypoints in `pyproject.toml` cover provider
  adapters, WAM/VLA endpoint setup/server/token helpers, OSCAR/Cosmos WAM
  commands, MuJoCo endpoint evaluation, and G1/3DGS support lanes.
- Uncommitted local June 21 work separated generated-video success labeling from
  episode consistency via `docs/WAM_EPISODE_CONSISTENCY_SCORER.md`,
  `src/blueprint_pipeline/wam_generated_video_success_label_gemini.py`, and
  `src/blueprint_pipeline/wam_episode_consistency_label_gemini.py`.
- Proof boundary: June 21 work does not by itself prove live provider runtime
  success, public generated-world rank fidelity, off-scope validation, or
  customer-specific sim-to-real correlation. Generated videos,
  VLM labels, endpoint probes, and owner-hosted connector outputs remain support
  evidence unless paired with accepted runtime and real-world validation proof.

## 2026-06-20

### User-Facing

- Uncommitted local work added a simulator-only MuJoCo G1 WAM/VLA policy-endpoint lane via
  `src/blueprint_pipeline/mujoco_g1_wam_vla_policy_endpoint_eval.py`, with setup and local
  HTTP wrapper support in `src/blueprint_pipeline/wam_vla_policy_endpoint_setup.py` and
  `src/blueprint_pipeline/wam_vla_policy_endpoint_server.py`.
- Uncommitted local work added G1/3DGS support lanes for local MuJoCo preview and
  fail-closed Isaac/3DGS realistic evaluation through
  `src/blueprint_pipeline/g1_site_3dgs_mujoco_preview.py` and
  `src/blueprint_pipeline/isaac_g1_site_3dgs_realistic_eval.py`.
- June 20 generated local artifacts under `robot_eval_jobs/g1_site_3dgs_mujoco_preview_20260620T135100Z/`
  and `policy_endpoint_setups/` record preview media, endpoint setup, and readiness outputs as
  support evidence only, not real-robot or deployment proof.

### Employee-Facing

- Uncommitted local work added Vast.ai provider planning/startup support and WAM provider bundle
  paths through `src/blueprint_pipeline/vast_provider_adapter.py`,
  `src/blueprint_pipeline/vast_wam_authorized_runner.py`,
  `src/blueprint_pipeline/vast_wam_async_runner.py`,
  `src/blueprint_pipeline/vast_bundle_staging.py`, and
  `src/blueprint_pipeline/oscar_wam_provider_bundle.py`.
- `pyproject.toml` now has local CLI entrypoints for the Vast provider adapter, OSCAR/WAM
  adapters, WAM/VLA endpoint setup/server/token helpers, MuJoCo endpoint eval, G1 endpoint
  reference adapter, G1/3DGS MuJoCo preview, and Isaac realistic eval.
- Model-access handling now prefers file-based Hugging Face and NGC secrets through
  `src/blueprint_pipeline/model_access_env.py`, and operational logging redacts sensitive fields
  through `src/blueprint_pipeline/logging_utils.py`.
- `.gitignore` now excludes local generated runtime outputs such as `pipeline/`,
  `robot_eval_jobs/`, `policy_endpoint_setups/`, `frame_*.png`, and `MUJOCO_LOG.TXT`.

### Future-Agent-Facing

- Contract changes: new uncommitted artifact families include WAM/VLA endpoint setup contracts,
  policy endpoint readiness manifests, team policy endpoint token manifests, Vast provider
  adapter/runtime phase artifacts, OSCAR/WAM provider bundles, MuJoCo G1 WAM/VLA scenario
  matrices, WAM/VLA action/output traces, and G1/3DGS preview/evaluation manifests.
- Runtime behavior changes: provider and endpoint lanes are gated, file-secret based, and
  fail closed when explicit local model commands, checkpoints, auth files, or provider gates are
  missing.
- Proof boundary: the MuJoCo lane is simulator-only; the 3DGS/MuJoCo preview is review/support
  media; the Isaac lane can write blocked attempts when runtime prerequisites are missing; and
  generated WAM/OSCAR outputs are downstream support artifacts. None of these prove physical
  generated-world rank fidelity, off-scope approval, public readiness, provider runtime success, or customer
  sim-to-real correlation without paired runtime and real-world validation evidence.
- Validation caveat: the June 20 work is still uncommitted in this checkout. Treat it as local
  changelog-worthy state, not a merged release.

## 2026-06-19

### User-Facing

- Added a first-class WAM/substrate policy-evaluation lane for Task Evaluation Runs and
  Policy Improvement Runs via `docs/WAM_POLICY_EVALUATION_SERVICE.md`,
  `src/blueprint_pipeline/wam_eval_substrate.py`, and
  `src/blueprint_pipeline/wam_fixture_evaluator.py`.
- Added a local deterministic WAM fixture evaluator and policy-autoresearch bridge through
  `src/blueprint_pipeline/policy_autoresearch_wam_fixture_evaluator.py`, keeping generated
  WAM rollouts as model-derived support evidence rather than raw capture, real-robot, or
  deployment-readiness proof.
- Added the `blueprint-run-major-capability-scenarios` CLI and
  `src/blueprint_pipeline/major_capability_scenario_suite.py` to evaluate five major
  product capabilities against concrete artifact criteria: capture-to-robot-eval packaging,
  Task Evaluation Run execution, Post-Training Data Package export, WAM/substrate policy
  evaluation, and hosted runtime/support-artifact review.

### Employee-Facing

- `docs/POLICY_IMPROVEMENT_RUN.md` now names WAM/substrate evaluation as first-class while
  preserving classical simulation as fallback, cross-check, or stricter physics support.
- Robot-eval job orchestration and worker paths now propagate WAM provider settings,
  artifact-output URI, retry, and timeout controls through
  `src/blueprint_pipeline/robot_eval_job_orchestrator.py`,
  `src/blueprint_pipeline/robot_eval_worker.py`, and
  `src/blueprint_pipeline/wam_provider_runtime.py`.
- Live or owner-provided WAM adapters fail closed unless the explicit local gate,
  environment gate, provider command, and env-only auth are present. Secrets must remain
  out of artifacts.

### Future-Agent-Facing

- Contract changes: new WAM artifacts include `evaluation_substrate_registry.json`,
  `wam_provider_runtime_package.json`, `wam_provider_execution_manifest.json`,
  `wam_rollout_results.json`, `wam_eval_claim_boundary.json`, SRCC/real-world validation
  follow-up artifacts, and customer handoff reports when WAM evaluation is requested.
- CLI/script changes: `pyproject.toml` adds `blueprint-run-wam-fixture-evaluator`,
  `blueprint-run-wam-eval-job`, `blueprint-run-policy-autoresearch-wam-fixture-evaluator`,
  and `blueprint-run-major-capability-scenarios`.
- Proof boundary: WAM heldout success, generated rollout labels, ranking scorecards, and
  major-capability scenario reports are support artifacts only. They do not prove physical
  generated-world rank fidelity, public readiness, off-scope approval, or customer-specific sim-to-real
  correlation without paired real-world validation evidence.
- Launch/readiness caveat: no June 19 commit by itself proves live provider runtime success,
  public deployment parity, or real-robot generated-world rank-fidelity result.
- Uncommitted local carryover: `src/blueprint_pipeline/object_geometry_stage.py`,
  `tests/test_qualification_alpha.py`, and `tests/test_robot_eval_job_orchestrator.py` had
  late-June-19 local edits around PNG helper behavior and test coverage/restructuring.
  Related local changes continued after midnight on June 20, so they are not treated here
  as finalized June 19 contract or runtime changes.
