# BlueprintCapturePipeline Changelog

## 2026-06-22

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
  simulator/provider-bound rather than physical-robot proof.
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
  success, public deployment readiness, safety validation, physical robot
  readiness, or customer-specific sim-to-real correlation. Generated videos,
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
  robot readiness, safety approval, public readiness, provider runtime success, or customer
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
  robot readiness, public readiness, safety approval, or customer-specific sim-to-real
  correlation without paired real-world validation evidence.
- Launch/readiness caveat: no June 19 commit by itself proves live provider runtime success,
  public deployment parity, or real-robot deployment approval.
- Uncommitted local carryover: `src/blueprint_pipeline/object_geometry_stage.py`,
  `tests/test_qualification_alpha.py`, and `tests/test_robot_eval_job_orchestrator.py` had
  late-June-19 local edits around PNG helper behavior and test coverage/restructuring.
  Related local changes continued after midnight on June 20, so they are not treated here
  as finalized June 19 contract or runtime changes.
