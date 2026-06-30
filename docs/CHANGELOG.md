# BlueprintCapturePipeline Changelog

## 2026-06-29

### User-Facing

- Closed the no-GPU dry-render evidence gap for the Isaac/G1 kitchen-parity
  lane. Local dry-render previews now carry explicit
  `X-Blueprint-Render-Source=dry_render_preview` PNG metadata and JSON
  provenance stating that they are NOT rendered Isaac frames.
- Added a fail-fast CPU environment contract for the canonical interpreter:
  `PIL`, `pxr`/`usd-core`, `mujoco`, `trimesh`, and `boto3` must be present so
  dry-render, USD placement, and MuJoCo-parity tests run instead of skipping
  green.
- Added a dirty-worktree paid-launch guard for the Isaac/G1 provider job. Paid
  GPU launch requests now record git evidence and block from a dirty or
  unverifiable tree unless an explicit override preserves that provenance risk
  in the manifest.
- Added WAM backend strategy and runtime-quality gates so OSCAR/Cosmos-style
  WAM candidates stay behind a replaceable adapter boundary and generated-video
  labels, backend readiness, and episode-consistency requests remain separate
  from deployment or physical-robot proof.
- Completed the tracked MuJoCo/Isaac parity backlog for the no-GPU portion of
  the lane: Isaac now has per-frame camera-contract, depth, segmentation,
  learned-policy requery, completion-gating, success-evaluator, gravity-step,
  and effort/contact-material wiring; MuJoCo gained depth, segmentation,
  photoreal observation handoff, texture/material, lighting, and collision-proxy
  improvements. Isaac items marked `gpu-pending` still require real GPU/RTX
  confirmation before stronger provider-runtime claims.
- Defaulted short and closed-loop WAM planning toward Vast and added budget,
  heartbeat, allowlist, runtime-env, snapshot-retry, future-frame, and evidence
  hardening for the Unitree GR00T N1.7 SONIC / OSCAR WAM lanes. These changes
  improve paid-run launch discipline; they do not themselves prove useful WAM
  visual quality or live provider completion.

### Employee-Facing

- Added `blueprint-check-cpu-env` and `src/blueprint_pipeline/cpu_env_doctor.py`
  for no-GPU dependency diagnosis, plus a meta-test that fails rather than
  skips if the canonical CPU stack is missing.
- Hardened `scene_placement` edge cases: suffix-only USD labels are dropped,
  multi-target task strings expose a deterministic target-group diagnostic,
  openable targets can receive conservative extra standoff, degenerate
  perception cameras fail closed, room-spanning perception boxes are skipped,
  and validation can flag a flipped forward-axis convention.
- Hardened Gemini-backed support gates with reconciled model cascades,
  balanced JSON extraction for reasoning-brace preambles, bounded transient
  retry, boolean-confidence rejection, diagnostic logging, and best-effort
  uploaded-file deletion after Gemini video inference.
- Extended the Isaac/G1 provider bundle with a required-file namelist and
  `bundle_manifest.json` so future runner/module extraction cannot silently drop
  worker dependencies.
- Made the full no-GPU test suite pass on a bare `python3`-only interpreter:
  the live-pipeline control-plane and Unitree-GR00T policy-server-preflight
  readiness tests now reference `sys.executable` for their command fixtures
  instead of assuming a bare `python` binary. Production command-runnability
  validators were left strict (they still report `blocked` when a named binary
  is genuinely absent); only the test fixtures changed.
- Added `opencv-python-headless` (`cv2`) to the canonical no-GPU stack (the
  `dev` extra, the `dev` dependency-group, and the CPU env contract). Without it
  ~32 oscar/cosmos/WAM/video tests silently skipped; they now run and pass.
- Closed a `uv sync` footgun with a PEP 735 default `[dependency-groups].dev`
  group: a bare `uv sync` now installs the full no-GPU stack
  (`pxr`/`mujoco`/`trimesh`/`cv2`/`boto3`) instead of UNINSTALLING 31 packages
  (including `usd-core`/`mujoco`/`trimesh`) and silently re-breaking the
  dry-render / placement / POV / video gates. `docs/DEV_SETUP.md` and the
  Makefile document `uv sync` as the canonical command.
- Added the durable parity roadmap and closeout notes in
  `docs/MUJOCO_VS_ISAAC_LANE_GAP_ANALYSIS.md` and
  `docs/MUJOCO_ISAAC_PARITY_BACKLOG.md`; those docs explicitly keep MuJoCo
  physics evidence, Isaac render evidence, WAM generated observations, and
  provider-runtime evidence non-interchangeable.
- Added shared paid-launch provenance and provider-runtime convergence helpers
  in `src/blueprint_pipeline/launch_provenance.py` and
  `src/blueprint_pipeline/isaac_worker_runtime_preflight.py`, and extended
  `src/blueprint_pipeline/provider_race.py`,
  `src/blueprint_pipeline/gpu_render_providers.py`, and
  `src/blueprint_pipeline/unitree_groot_n17_sonic_vast_persistent_session.py`
  around heartbeat stalls, teardown semantics, render budget caps, and runtime
  preflight markers.
- Extended MuJoCo/Isaac runtime surfaces in
  `scripts/run_isaac_g1_kitchen_parity_eval.py`,
  `src/blueprint_pipeline/mujoco_g1_simulator_command.py`,
  `src/blueprint_pipeline/mujoco_g1_wam_vla_policy_endpoint_eval.py`,
  `src/blueprint_pipeline/oscar_isaac_closed_loop_eval.py`,
  `src/blueprint_pipeline/oscar_wam_provider_bundle.py`,
  `src/blueprint_pipeline/wam_backend_strategy.py`, and related provider/WAM
  adapters, with focused coverage added across the corresponding test files.

### Future-Agent-Facing

- Render-seed proof boundary for the 2026-06-29 render-visibility work:
  CPU/hermetic only. No live GPU frame was produced in this session on
  2026-06-29, so the G1 refrigerator/faucet render changes remain local
  logic, dry-render, and unit-test evidence, not live Isaac frame proof,
  deployment approval, physical-robot readiness, manipulation success, or
  safety validation.
- Evidence boundary: base checkout was `8715581de51851b898451ed528ed4d0dab3d1cc1`
  on `main`; audit-start dirty files were `docs/CHANGELOG.md`,
  `scripts/run_isaac_g1_kitchen_parity_eval.py`,
  `tests/test_isaac_g1_kitchen_parity_runner.py`,
  `tests/test_local_render_preview.py`, and untracked
  `docs/cpu-work-audit-2026-06-29.md`.
- Local CPU proof on 2026-06-29: `.venv/bin/python -m pytest -q -o
  addopts=''` completed with `2556 passed, 30 skipped, 10 warnings` in
  695.66s. The focused no-spend G1/placement/provider evidence command over
  scene placement, perception, provider race, render lock, warm server, spend
  guard, local render preview, and the Isaac/G1 runner completed with
  `367 passed`; its matching `--collect-only` pass collected all 367 tests with
  no collection errors.
- Local CPU proof update on 2026-06-29 (continued): after the test-interpreter
  portability fix and the `cv2` dependency addition, the full
  `.venv/bin/python -m pytest tests/` run completed with 0 failures
  (`2567 passed, 32 skipped`); those 32 skips were all `cv2`-gated and now run
  after installing `opencv-python-headless`. The render-visibility/G1 work
  remains CPU/hermetic-only — still no live GPU frame produced in this session.
- Later same-day focused proofs recorded in commit subjects include green
  no-GPU test runs for the WAM backend gates, Vast WAM selection/env forwarding,
  OSCAR input-contract diagnostics, MuJoCo RGBD/segmentation/material/lighting
  and collision-proxy paths, Isaac depth/segmentation/gravity/effort-drive
  paths, and provider-runtime convergence. Treat those as focused unit or
  hermetic proofs unless a future run supplies real provider artifacts.
- Launch/readiness caveat: the June 29 parity backlog marks several Isaac
  tasks `done (gpu-pending)`. Do not cite them as accepted live RTX frames,
  provider closure, physical manipulation success, safety validation, or
  deployment approval until the matching GPU run artifacts, upload/finalizer
  evidence, cost/teardown proof, and review-quality outputs exist.
- Uncommitted local state at changelog finalization: none found by
  `git status --short`.

## 2026-06-28

### User-Facing

- Added dynamic, task-aware scene placement for Isaac/G1 kitchen-parity review
  media. Tasks can now resolve a target object from USD scene bounds or injected
  perception views, compute a stand pose from open-floor probes, and fail closed
  when placement validation is weak instead of relying on hardcoded kitchen
  coordinates.
- Improved G1 manipulation POV review quality for faucet/fridge-style tasks
  with corrected reach poses, arm/hand visibility checks, lighting/framing
  updates, low-lens mount corrections, and stricter manipulation seed POV
  validation. These frames remain simulator/render support artifacts, not raw
  capture truth, physical manipulation success, safety validation, deployment
  approval, or live robot readiness.
- Hardened GPU/provider spend and warm-run behavior with spend guards, render
  locks, provider race handling, longer image-pull/startup tolerance, object
  store warm-inbox presigning, and a persistent warm render server whose control
  loop is implemented and hermetically tested. Live multi-request reuse after
  one real Isaac scene load still needs on-GPU proof.

### Employee-Facing

- Added the `src/blueprint_pipeline/scene_placement/` package with USD and
  perception-backed spatial indexes, perception-view fusion, task target
  resolution, obstacle/degenerate-box handling, geometric placement validation,
  and a self-validating `place_and_validate_robot_for_task` orchestration path.
  See `src/blueprint_pipeline/scene_placement/README.md`.
- Updated `scripts/run_isaac_g1_kitchen_parity_eval.py` and
  `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py` so provider bundles
  ship the placement package, re-add the bundle path before worker imports,
  support warm `--serve` execution, route manipulation camera/arm reach through
  resolved targets, and keep startup/image-pull watchdog behavior explicit.
- Added provider and concurrency helpers in
  `scripts/gpu_spend_guard.py`, `src/blueprint_pipeline/provider_race.py`,
  `src/blueprint_pipeline/render_lock.py`, and
  `src/blueprint_pipeline/warm_render_server.py`; extended
  `src/blueprint_pipeline/wam_provider_object_store.py` with warm-inbox
  presign support.
- Expanded render/placement QC in `src/blueprint_pipeline/render_visual_qc.py`
  and related tests so placement, robot-POV, manipulation-POV, provider, warm
  server, spend guard, perception adapter/fusion, render lock, and scene
  placement behavior are covered by focused unit tests.

### Future-Agent-Facing

- Contract changes: `scene_placement` is dependency-light and swappable; GPU
  work stays behind injected render/SAM3/DA3/perception hooks, while capture,
  package, evaluation, and provenance contracts above it remain stable.
- Runtime behavior changes: Isaac/G1 jobs can now reuse a warm scene load and
  accept task requests through a signed warm inbox. Treat warm-provider success
  as provider/runtime scaffolding unless matching result, upload, teardown,
  cost-control, and review artifacts are present.
- CLI/script changes: `scripts/gpu_spend_guard.py` is a new spend-safety helper,
  and `scripts/run_isaac_g1_kitchen_parity_eval.py` now owns more of the
  dynamic placement, manipulation POV, local render-preview, and warm-run
  harness behavior.
- Proof-boundary changes: placement validation, visual QC, robot POV frames,
  manipulation POV frames, and Isaac/G1 review media are downstream support
  evidence only. They can flag whether review media is useful, but they do not
  override raw capture/provenance evidence or prove live robot execution,
  physical contact, task success, safety, deployment, or generated-world rank
  fidelity.
- Launch/readiness gate changes: image-pull/startup tolerance increased and
  warm provider reuse reduces repeated cold-start pressure, but live provider
  closure still requires accepted runtime artifacts, upload/finalizer evidence,
  cost/teardown proof, and review-quality outputs.
- Provenance note: this June 28 entry is based on committed history for the
  previous completed calendar day. Do not read it as a claim that the June 29
  audit checkout was clean; the June 29 entry records the dirty-tree evidence
  boundary separately.

## 2026-06-27

### User-Facing

- Added a provider-agnostic GPU launch and Isaac/G1 kitchen-parity evaluation
  lane for captured-scene review, including splat/NuRec support, G1 policy
  scaffolding, particle-field USD helpers, and a kitchen-parity runner. These
  outputs are simulator/render/runtime support artifacts; they do not prove
  raw capture truth, live robot readiness, physical manipulation success,
  safety validation, or deployment approval.
- Expanded WAM compute and robot-POV support with provider-agnostic WAM compute
  planning, object-index splat analysis, OSCAR provider command adapter updates,
  generated-video review improvements, and WAM real-provider validation probe
  wiring. The lane strengthens Task Evaluation Run and review-package
  infrastructure while keeping generated media and provider outputs downstream
  of capture/provenance evidence.
- Added a per-step OSCAR/SAM3 closed-loop evaluation path and GPU pod startup
  builder for policy/WAM/perception experiments. The closed loop can prepare and
  test provider-side inference paths, but it remains evaluator/runtime evidence,
  not an accepted forward/inverse consistency score or live deployment proof.
- Hardened Isaac manipulation review media with manipulation-camera modes,
  lighting/framing fixes, rest-pose skeleton conditioning, crash-safe USD arm
  reach, convex-hull collision geometry, a centered third-person verify camera,
  and a manipulation-stand mode that places the robot at the task start pose
  without claiming navigation.
- Added a VLM-backed visual sanity QC helper for rendered frames and WAM outputs
  so blank, irrelevant, or weak review media can be flagged before it is treated
  as useful support evidence.

### Employee-Facing

- Added or updated runtime modules and scripts for provider-agnostic GPU/WAM
  execution and Isaac/G1 parity flows. Key paths include
  `src/blueprint_pipeline/gpu_render_providers.py`,
  `src/blueprint_pipeline/wam_compute_providers.py`,
  `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`,
  `src/blueprint_pipeline/isaac_g1_policy.py`,
  `src/blueprint_pipeline/isaac_particlefield_render_job.py`,
  `src/blueprint_pipeline/isaac_nurec_export.py`,
  `src/blueprint_pipeline/particlefield_usd.py`,
  `src/blueprint_pipeline/splat_backends.py`,
  `scripts/run_isaac_g1_kitchen_parity_eval.py`,
  `scripts/run_isaac_splat_nurec_render.py`, and
  `scripts/object_index_splat_analyzer_runner.py`.
- Added OSCAR closed-loop and provider startup surfaces through
  `src/blueprint_pipeline/oscar_isaac_closed_loop_eval.py` and
  `src/blueprint_pipeline/oscar_isaac_closed_loop_gpu_launch.py`, with focused
  tests for injectable WAM backends, real OSCAR-2B pod-side inference,
  next-frame extraction, CLI wiring, and startup-package generation.
- Hardened provider reliability in `src/blueprint_pipeline/runpod_wam_async_runner.py`,
  `src/blueprint_pipeline/vast_provider_adapter.py`,
  `src/blueprint_pipeline/vast_wam_async_runner.py`, and
  `src/blueprint_pipeline/oscar_wam_provider_bundle.py` by addressing stale
  object-store output, heartbeat/poll behavior, dud detection, dependency
  handling, and teardown behavior.
- Added `src/blueprint_pipeline/render_visual_qc.py` with focused coverage in
  `tests/test_render_visual_qc.py`; expanded tests across the Isaac/G1,
  provider, WAM compute, object-index, generated-video review, and closed-loop
  paths.
- Added June 27 design/goal docs under
  `docs/superpowers/specs/2026-06-27-isaac-g1-kitchen-parity-design.md`,
  `docs/superpowers/specs/2026-06-27-provider-agnostic-wam-compute-design.md`,
  and `docs/goals/2026-06-27-provider-agnostic-wam-compute-loop.md`.

### Future-Agent-Facing

- Contract changes: provider-agnostic compute/render abstractions were added for
  GPU render and WAM compute paths. Keep model/provider backends replaceable and
  preserve capture, package, evaluation, and provenance contracts above those
  adapters.
- Runtime behavior changes: RunPod/Vast WAM polling and object-store staging now
  account for stale outputs, heartbeats, dud provider behavior, dependency
  setup, and teardown more explicitly. Do not treat poll completion or artifact
  presence as provider-runtime proof without matching runtime/provenance output.
- CLI/script changes: the Isaac/G1 kitchen-parity and splat/NuRec render
  scripts are now first-class support surfaces for review media generation;
  `pyproject.toml` also picked up related GPU render entrypoints.
- Proof-boundary changes: Isaac review frames, G1 skeleton videos,
  manipulation-stand renders, visual QC labels, and generated WAM outputs are
  downstream review/support artifacts. They may help decide whether to continue
  a run, but they do not override raw capture/provenance truth or establish
  physical robot readiness, navigation success, safety validation, deployment
  approval, or forward/inverse episode consistency.
- Launch/readiness gate changes: cold Isaac image pulls now tolerate a longer
  marker timeout and additional attempts, which improves startup robustness but
  does not remove the need for accepted provider execution, artifact upload,
  cost/teardown, and review-quality evidence before stronger readiness claims.
- Uncommitted local state: none found in the current checkout for June 27; this
  entry is based on committed history for the previous completed calendar day.

## 2026-06-26

### User-Facing

- Hardened the sim-only policy-comparison launch path with clearer local-gate,
  release-gate, deployment-parity, and live-pipeline intake evidence. The June
  26 audit records that simulator execution is proven for the local sample path,
  while beta release, production forwarding, and sim-only closure still remain
  blocked by failure-diagnosis / closure-audit and intake-health evidence gaps.
- Expanded Task Evaluation Run and Post-Training Data Package support through
  policy/package contract work, provider closure auditing, RL post-training
  handoff artifacts, OSCAR visual augmentation packets, and failure/scorecard
  guardrails. These are package and evaluator support artifacts; they do not
  upgrade generated media, simulator outputs, or WAM labels into raw capture
  truth, physical robot readiness, safety validation, or deployment approval.
- Added Isaac/RunPod startup proof and Gaussian-splat rendering support so
  captured 3DGS scenes can be decoded, analyzed, and rendered as reference
  review media. Reference Spark renders show the captured splat can display,
  but they are not Isaac RTX/NuRec proof, physics proof, navigation proof, or
  public readiness proof.

### Employee-Facing

- Added or updated runtime/docs/tests for live-pipeline forwarding setup,
  sim-only beta local/release/deployment gates, G1 controlled-run evidence
  assembly, robot-eval orchestration, policy endpoint boundaries,
  provider-closure audits, RL post-training handoff, OSCAR visual augmentation,
  WAM/perception harnesses, and post-training package generation. Key paths
  include `docs/last_24h_launch_audit_2026-06-26.md`,
  `docs/OSCAR_VISUAL_AUGMENTATION_PACKET.md`,
  `src/blueprint_pipeline/live_pipeline_forwarding_secret_setup.py`,
  `src/blueprint_pipeline/provider_closure_audit.py`,
  `src/blueprint_pipeline/rl_post_training_handoff.py`,
  `src/blueprint_pipeline/oscar_visual_augmentation_packet.py`, and
  `src/blueprint_pipeline/oscar_visual_augmentation_generation_runner.py`.
- Added simulator-agnostic G1/Isaac contracts and splat tooling through
  `docs/simulator-agnostic-g1-execution-contract.md`,
  `docs/superpowers/specs/2026-06-26-isaac-splat-render-parity-design.md`,
  `src/blueprint_pipeline/gaussian_splat_decode.py`,
  `src/blueprint_pipeline/splat_scene_analysis.py`,
  `src/blueprint_pipeline/splat_scene_render.py`, and
  `tools/splat_render/`, with focused tests for decode, scene analysis, render
  wiring, RunPod adapter behavior, and live execution proof handling.
- Added CLI/script surfaces for Isaac worker image builds and support flows,
  including `scripts/build_push_isaac_worker_image.sh` plus `pyproject.toml`
  entrypoints for provider closure, live-pipeline forwarding setup, OSCAR
  augmentation, rollout labeling, post-training packages, and splat rendering.

### Future-Agent-Facing

- Contract changes: sim-only launch evidence now distinguishes local simulator
  execution, beta release closure, production forwarding, Pipeline intake
  health, robot-team-grade blockers, and optional physical/deployment claim
  upgrades. Use `docs/last_24h_launch_audit_2026-06-26.md` for the current
  blocker hierarchy instead of stale generated audit JSON.
- Runtime behavior changes: reference splat rendering can attach display
  evidence to Isaac/G1 evaluation artifacts, but the proof boundary must keep
  `rendered_by: reference_spark_renderer` separate from Isaac RTX/NuRec,
  physics, navigation, provider runtime, and readiness proof.
- Launch/readiness gate changes: production beta is still blocked on Pipeline
  intake token/health and forwarding proof; broader robot-team-grade paths still
  require remote/cloud execution, digital-twin fidelity, failure-diagnosis, and
  closure-audit evidence.
- Uncommitted local June 26 carryover: `.gitignore`,
  `src/blueprint_pipeline/isaac_g1_site_3dgs_realistic_eval.py`,
  `src/blueprint_pipeline/splat_scene_analysis.py`,
  `src/blueprint_pipeline/splat_scene_render.py`,
  `scripts/run_isaac_splat_nurec_render.py`,
  `src/blueprint_pipeline/isaac_nurec_export.py`,
  `src/blueprint_pipeline/particlefield_usd.py`,
  `src/blueprint_pipeline/splat_backends.py`, and related tests had June 26
  mtimes in the current dirty worktree. Treat them as local follow-on state
  until committed; adjacent kitchen-parity/provider work continued after
  midnight on June 27 and is intentionally excluded from this June 26 entry.

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
