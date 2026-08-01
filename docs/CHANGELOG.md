# BlueprintCapturePipeline Changelog

## 2026-08-01

### User-Facing

- Added the authorized source-video semantic runner needed before 3D lifting.
  A configured SAM 3.1 Object Multiplex runtime can now turn exact retained
  frames into persistent, hash-bound 2D object-mask tracks for the existing
  semantic pipeline. This does not make the model's labels observed facts or
  establish metric location, collision, physics, or task success.
- Added the first executable source-frame-to-3DGS semantic bridge. Persistent
  object masks from retained frames can now be projected through exact calibrated
  cameras onto a standard 3DGS using deterministic front-to-back Gaussian
  contribution weights, then passed directly to the existing semantic lifting
  contract. The result remains candidate semantic support, not metric object,
  collision, physics, task-success, or physical evidence.

### Employee-Facing

- Published PR #282 after all hosted checks passed, including 6,345 fast-lane
  and 8,020 full-lane tests. Protected main and isolated staging now use commit
  `7063d719` with exact tested tree `0354737d`; production remains independently
  pinned to `3bb376e7`.
- Added a bounded adapter for Meta's official SAM 3.1 multiplex predictor API.
  It fails closed unless checkpoint, code revision, runtime, license terms/use,
  privacy, trade-controls, customer-data use, and execution authorization are exact; verifies
  every retained-frame JPEG derivative; runs offline; validates untrusted
  object IDs, scores, and mask shapes; and emits both the existing provider
  result and its ready-to-import request. The gated checkpoint was not
  downloaded or executed locally.
- Published PR #281 after all 16 hosted checks passed, including 6,336 fast-lane
  and 8,011 full-lane tests. Protected main and isolated staging now use commit
  `b3a6ce1c` with the tested tree; production remains independently pinned to
  `3bb376e7`. The Pub/Sub listener uses a dedicated least-privilege identity and
  corrected recurring systemd cadence.
- Added a bounded NumPy standard-3DGS contribution renderer and file stage. It
  rejects stale source-track, frame, camera, splat, and Gaussian-mapping digests;
  unrectified cameras; nonstandard PLY inputs; and projected-work overflows. It
  is a small-scene/conformance path while accelerated chunked transport remains.

### Future-Agent-Facing

- The official `facebook/sam3.1` checkpoint repository has no Transformers
  integration. Preserve the pinned Meta `build_sam3_multiplex_video_predictor`
  interface and do not silently fall back to `facebook/sam3` Transformers or
  the legacy placeholder detector.
- Preserve the source-video/3DGS authority split: source frames establish mask
  and track evidence, calibrated cameras lift it into the splat, and separate
  independently qualified collision geometry is required before physics use.
- Do not reinstall the upstream Splat Analyzer locally merely to claim support.
  Its current rough boxes remain candidate-only, and the machine does not have
  safe disk headroom for an unbounded Torch/model installation.

## 2026-07-29

### User-Facing

- Closed the powered DROID native-Cosmos causal confirmation as
  `thesis_not_supported` for the frozen tested stack. The complete 17-session,
  51-window matrix returned 612/612 valid videos but passed zero windows and
  zero sessions; correct-action scene effects were consistently weaker than
  ordinary seed variation, so Blueprint abstained without producing a policy
  ranking (`docs/experiments/policy_ranking_roboarena_powered_droid_confirmation_20260729/final_report_v1.md`).
- Added the claim-level Decision/Evidence Router above stable leaf evaluators.
  Task Evaluation Run requests now bind maintained testbeds, qualifications,
  budgets, rights, evidence plans, normalized results, decisions or explicit
  abstentions, and append-only physical-outcome joins. Legacy Policy Improvement
  and post-training exports remain opt-in compatibility machinery
  (`docs/architecture/decision-evidence-router.md`,
  `src/blueprint_pipeline/decision_evidence_router.py`).

### Employee-Facing

- Hardened the powered GPU lane with exact source/runtime identities, immutable
  object transport, spend/TTL admission, stale-callback rejection, teardown,
  and provider-zero evidence. Two Vast allocations consumed 4,615.574 live
  seconds and an estimated USD 2.226112; this proves bounded execution and
  cleanup, not invoice settlement or scientific validity
  (`src/blueprint_pipeline/policy_ranking_powered_droid_analysis.py`,
  `docs/experiments/policy_ranking_roboarena_powered_droid_confirmation_20260729/provider_zero_and_object_closure_v1.json`).
- Added isolated new-site diagnostic adapters plus an allocator-only one-arm GPU
  canary, and made OpenPI monitoring retry transient output failures. These are
  diagnostic/runtime seams and do not qualify a WAM, admit a ranking, or prove
  captured-site transfer (`src/blueprint_pipeline/new_site_diagnostic_smoke.py`,
  `src/blueprint_pipeline/new_site_diagnostic_canary_gpu.py`,
  `src/blueprint_pipeline/openpi_policy_ranking_runpod.py`).

### Future-Agent-Facing

- The America/Chicago window contains sixteen first-parent `main` commits,
  `3d6c4044` through `21e49c3d` (PRs #237 and #239--#245). At review, the
  checkout is clean on a later feature branch; `main == origin/main` at
  `4aa4e056`, so later July 30 history and checkout-only July 29 branches are
  excluded. No attributable uncommitted July 29 work is recorded.
- Do not rerun unchanged native Cosmos or promote the router's hermetic vertical
  slice into provider, deployment, safety, or physical evidence. A new WAM arm
  needs prospectively frozen causal controls before evaluator spend; physical
  claim upgrades require separately accepted, exactly joined outcomes.
- Raw capture, timestamps, poses, provenance, rights, privacy, and accepted
  physical outcomes remain authoritative. Generated videos, simulator results,
  router plans, qualification records, provider receipts, readiness summaries,
  and this changelog are downstream support artifacts.

## 2026-07-28

### User-Facing

- Closed the Cosmos3-Nano successor follow-up `inconclusive`. The frozen
  one-session screen produced ten valid, nonduplicate videos and all eight
  active rows differed from same-seed zero-action output, but only one of eight
  rejected the strongest temporal placebo and zero of four conditions passed
  both-seed robustness. The powered causal, evaluator, benchmark, and
  captured-site arms were therefore not admitted
  (`docs/experiments/policy_ranking_cosmos3_followup_20260728/final_verdict.json`).
- Closed the RoboArena full-stack calibration `inconclusive`. The complete
  63-session, seven-policy, 441-episode frozen GPT-5 mini reproduction failed
  its registered ranking and selective-use gates (Spearman 0.357143, Kendall
  0.238095, pairwise accuracy 0.619048, selective coverage 0.050182). Gemini
  3.6 Flash ranked better only as a post-unseal diagnostic and never abstained;
  no disjoint closed-loop Phase B or captured-site Phase C ran
  (`docs/experiments/policy_ranking_roboarena_full_stack_calibration_20260728/final_verdict_v1.json`).
- Aligned the product doctrine around capture-first Task Evaluation Runs,
  bounded Policy Improvement Runs, and Post-Training Data Packages. Added a
  documented doctrine-precedence order, clarified site-operator lifecycle
  roles and model replaceability, and tightened SC3-Eval and OSCAR evidence
  claims (`PLATFORM_CONTEXT.md`, `WORLD_MODEL_STRATEGY_CONTEXT.md`,
  `docs/DOCTRINE_PRECEDENCE.md`).

### Employee-Facing

- Added fail-closed evaluator and paid-GPU admission paths for OpenAI, Gemini,
  and Cosmos Reasoner diagnostics, including schema/runtime binding, signed
  object transport, resumable media staging, idempotent submission, and a
  pre-download Vast CUDA compatibility probe. Reasoner attempts V4--V6 yielded
  no valid scientific ranking; V6 failed before model load with CUDA error 803
  (`src/blueprint_pipeline/policy_ranking_evaluator_diagnostic.py`,
  `src/blueprint_pipeline/vast_cuda_runtime_probe.py`).
- Added a WAM rollout-reliability gate for action-motion, timing, degeneracy,
  and rot6d validity, then made timing reliability aggregate explicitly by
  session while hard failures remain immediate. Thresholds remain experiment-
  specific and require calibration before scientific use
  (`src/blueprint_pipeline/wam_rollout_reliability.py`).
- Renamed the capture-to-package spine to
  `src/blueprint_pipeline/site_package_orchestrator.py`; the deprecated
  `blueprint_pipeline.qualification` import and existing artifact contracts
  remain compatible. Known conservative RoboArena provider spend was
  USD 6.909436375 excluding unavailable storage/transfer invoice evidence;
  low request cost did not establish faster or cheaper physical evaluation.

### Future-Agent-Facing

- The America/Chicago window contains seventeen first-parent `main` commits,
  `a00856dd` through `029a705b` (PRs #214--#225 and #227--#231). At review,
  the worktree is clean and no attributable uncommitted July 28 work is
  recorded. Local `HEAD == main == 029a705b`; `origin/main` is six later commits
  ahead, so they are excluded from this day-bounded entry.
- Do not rerun exposed-snapshot judges or another short Reasoner diagnostic.
  The next scientifically valid experiment needs a genuinely new disjoint
  labeled RoboArena/DROID snapshot, runnable frozen policies, prospectively
  powered independent sessions, and the frozen reliability/evaluator gates.
- Raw capture, provenance, rights, and privacy evidence remain authoritative.
  Generated videos, benchmark reports, provider receipts, review galleries,
  readiness summaries, and this changelog are downstream support artifacts;
  none proves deployment, public readiness, or physical-robot success.
## 2026-07-27

### User-Facing

- Closed policy-ranking thesis Experiment 2 as `thesis_not_supported` for the
  frozen OSCAR representation/evaluator and released DROID-compatible policy
  cohort. The powered 49-session causal gate measured 0.039976 mean excess and
  a 0.387755 clustered-bootstrap lower validity pass rate against the required
  0.8; the incomplete 43/686 GPT-5 matrix remained unscored. Captured-site
  transfer is separately `not_supported` because retained InteriorGS execution
  used a different OpenPI/MuJoCo stack and supplied neither frozen evaluator
  transfer nor site-specific physical labels
  (`docs/experiments/policy_ranking_thesis_experiment_2_20260727/final_verdict.json`).
- Added a replaceable Cosmos3-Nano forward-dynamics successor lane, but its
  exact-main campaign closed `inconclusive`: Blackwell CUDA/BF16 admission and
  model load passed, while the only direct clip returned 640x528 for a frozen
  640x540 request and independently failed the static-video check. No causal
  matrix, evaluator call, benchmark ranking, or captured-site arm was admitted
  (`docs/experiments/policy_ranking_successor_experiment_20260727/final_verdict.json`).

### Employee-Facing

- Hardened allocator-only Vast execution with portable reviewed bundles,
  pre-create session-budget checks, detached watchdog evidence, bounded cold
  startup, fail-closed authorization replacement, retained-session lifecycle,
  and personal-path sanitization. Three successor allocations consumed
  791.277 seconds and an estimated USD 0.226920; the final authenticated
  inventory proved provider zero and zero continuing hourly burn
  (`src/blueprint_pipeline/policy_ranking_successor_gpu_admission.py`,
  `src/blueprint_pipeline/vast_provider_adapter.py`,
  `src/blueprint_pipeline/retained_gpu_session_lifecycle.py`).

### Future-Agent-Facing

- The America/Chicago window contains fourteen first-parent `main` commits,
  `115e9cae` through `39965efc` (PRs #198, #199, #201--#209, and #211--#213).
  At review, the worktree is clean and no attributable uncommitted July 27 work
  is recorded. Local `HEAD == main == 39965efc`; `origin/main` is seven commits
  ahead, all dated July 28, so July 27 history remains bounded to the range above.
- Do not spend evaluator budget until a newly preregistered Cosmos arm passes
  recorded, zero, shuffled/reversed, and policy-swapped causal controls. The
  successor's decodable clip and successful model load prove runtime execution,
  not action conditioning, ranking fidelity, simulator success, deployment,
  public readiness, or physical-robot performance.
- Raw capture, provenance, rights, and privacy evidence remain authoritative.
  Frozen experiment artifacts, generated clips, provider receipts, qualification
  summaries, and this changelog are downstream support artifacts.
## 2026-07-26

### User-Facing

- Completed the frozen policy-ranking thesis experiment with an `inconclusive`
  verdict. Calibration failed the selective-coverage and action-following gates,
  while rate limits left the required held-out matrix unmeasured; the frozen
  protocol therefore forbids either a support or definitive-falsification claim
  (`docs/experiments/policy_ranking_thesis_20260726/final_verdict.md`).
- Executed 24 contract-valid OpenPI/MuJoCo GPU episodes across NVIDIA Warehouse
  and InteriorGS-derived scene lanes. Both lanes abstained from a total ranking;
  this proves pipeline ingestion and learned-policy execution, not useful policy
  ordering, site-specific physical success, or transfer of the OSCAR/GPT
  evaluator stack (`src/blueprint_pipeline/openpi_policy_ranking_runpod.py`,
  `src/blueprint_pipeline/captured_site_policy_ranking.py`).

### Employee-Facing

- Split orchestrator authority, release source, runtime image, and mutable
  overlay identities in qualification and paid-resource admission. Spend
  watchdog ownership now survives authorized identity migration, and a
  WAM-primary evaluation authority cannot silently fall back to legacy Isaac
  scoring (`src/blueprint_pipeline/single_g1_kitchen_qualification_contract.py`,
  `src/blueprint_pipeline/paid_resource_allocator.py`,
  `scripts/gpu_spend_guard.py`,
  `src/blueprint_pipeline/wam_isaac_evaluation_hierarchy.py`).
- Added an allocator-governed OpenPI GPU build/run lane with checkpoint,
  action-space, scene, image-result, lease, budget, and teardown controls. Final
  Vast absence was API-confirmed and the reservation settled; teardown proof is
  separate from ranking or task-success proof
  (`src/blueprint_pipeline/openpi_policy_ranking_gpu_admission.py`,
  `src/blueprint_pipeline/paid_lane_guard.py`,
  `src/blueprint_pipeline/gpu_render_providers.py`).

### Future-Agent-Facing

- The America/Chicago window contains ten first-parent `main` commits,
  `dd38227a` through `1a7376e7` (including PRs #192 and #197). At review,
  `HEAD == main == origin/main`, divergence is `0 0`, the worktree is clean,
  and no attributable uncommitted July 26 source work is recorded.
- Preserve the frozen retry limit and held-out abstention. A provider reset or
  new evaluator arm requires a separately preregistered experiment; do not
  retrofit the missing held-out result or equate deterministic MuJoCo scoring
  with the unexecuted OSCAR/GPT cross-lane transfer.
- Raw capture, provenance, rights, and privacy evidence remain authoritative.
  Benchmark reports, generated media, simulator episodes, ranking summaries,
  provider receipts, and this changelog are downstream support artifacts.

## 2026-07-25

### User-Facing

- Vast qualification now rejects GPU architectures that the pinned TensorRT
  10.4 engine cannot build for and, for Isaac workloads only, NVIDIA driver
  branches above the runtime's evidence-backed compatibility ceiling. Both
  constraints expose explicit diagnostics and operator overrides; admission of
  a compatible offer still does not prove renderer startup, episode completion,
  or task success (`src/blueprint_pipeline/vast_compute_capability.py`,
  `src/blueprint_pipeline/isaac_driver_support.py`,
  `src/blueprint_pipeline/vast_provider_adapter.py`).
- The closed-loop evaluator now consumes the sealed producer's nested initial
  observation, resolves the bundle's canonical evidence path, and fails closed
  before executing a step when a configured learned-policy endpoint cannot
  supply an initial action. The action's manipulation target, rather than its
  camera-framing point, is now authoritative for directional progress
  (`src/blueprint_pipeline/initial_policy_observation_contract.py`,
  `src/blueprint_pipeline/oscar_isaac_closed_loop_eval.py`).

### Employee-Facing

- Runtime termination now replays the live simulator timeline before measuring
  completion and requires both task-joint and FK-approach streams to stall.
  Run 7 reached first door contact before a supported stance abort, while its FK
  approach remains command-intent evidence because each chunk replays from the
  frozen initial state (`src/blueprint_pipeline/isaac_runtime_task_backend.py`,
  `docs/runbooks/qualification-debug-funnel.md`).
- GPU-residency attribution can fall back from unavailable host-PID translation
  to namespace-local, per-GPU device handles. It refuses multi-GPU inference and
  generic NVIDIA control nodes, and records the attribution mode in the sealed
  sample (`src/blueprint_pipeline/gpu_residency_attribution.py`).
- The qualification debug funnel now records that a Tier-2 refresh needs a
  healthy box allocated while checkout, release, and protected main still
  match; merging first can force a thin-image rebuild before another allocation
  (`docs/runbooks/qualification-debug-funnel.md`).

### Future-Agent-Facing

- The America/Chicago window contains seven first-parent `main` commits,
  `24404ba6` through `9be733e8` (PRs #181--#183 and #186--#189). At review,
  `HEAD == main == origin/main`, the worktree is clean, and no attributable
  uncommitted July 25 source work is recorded.
- Preserve sealed bundle argv/evidence compatibility, action-target semantics,
  scoped hardware gates, and disclosed residency attribution. Live attempts
  066--069 diagnose and constrain failure modes; they do not prove a completed
  simulator task, policy ranking, deployment, public readiness, or physical
  robot performance.
- Raw capture, provenance, rights, and privacy evidence remain authoritative.
  Runtime manifests, diagnostics, generated review media, qualification
  ledgers, and this changelog are downstream support artifacts.

## 2026-07-24 (third entry)

### User-Facing

- Generated media no longer inherits its source capture's redaction status.
  Capture-side redaction protects captured pixels; it cannot cover pixels a
  world model invents, which can re-synthesise a face, a badge, or a whiteboard
  that redaction removed. Generated artifacts now start unverified, must be
  conditioned on redaction-verified assets, require their own redaction pass
  over the generated pixels before customer-visible release, and carry takedown
  keys so a later consent revocation reaches the derivative. At the hosted
  serving boundary a chunk that carries a contract is held to it unconditionally
  and withheld behind a labelled placeholder if it is not cleared for
  customer-visible release. Enforcement of the *no-contract* case is staged: no
  producer attaches contracts to runtime chunks yet, so uncontracted media is
  served with an explicit `X-Blueprint-Generated-Media-Privacy: unverified`
  label rather than silently, and `BLUEPRINT_ENFORCE_GENERATED_MEDIA_PRIVACY`
  withholds it outright. Until a producer exists this defect is contained and
  visible, not closed (`src/blueprint_pipeline/generated_media_privacy.py`,
  `src/blueprint_pipeline/native_runtime_backend.py`,
  `docs/CUSTOMER_OUTPUT_CONTRACTS.md`).
- Every Post-Training Data Package now ships a frozen held-out cut carved from
  the same capture, plus a check that fails closed when the training payload
  contains held-out clips. Nothing previously stopped a buyer's evaluation set
  from overlapping the clips they were sold
  (`src/blueprint_pipeline/post_training_holdout_split.py`).
- Site package, proof pack, and rights review manifests now carry a
  `delivery_integrity` block with per-member digests and a root digest over the
  member set. A URI without a digest is a blocker: it records where bytes were,
  not which bytes they were
  (`src/blueprint_pipeline/signed_delivery_bundle.py`,
  `src/blueprint_pipeline/proof_contracts.py`).
- Added anchor return kits so a physical trial's outcomes can actually join to
  the prediction they answer. Join keys are pre-populated per prediction, and a
  returned file is validated before ingest rather than after the robot time is
  spent (`src/blueprint_pipeline/anchor_return_kit.py`).
- Added per-site difficulty profiles so cross-site policy numbers are
  interpretable. Difficulty is reported beside a success rate as a covariate,
  never divided into it (`src/blueprint_pipeline/site_difficulty_profile.py`).

### Employee-Facing

- The OEM handoff summary now reports its own completeness against the required
  inputs its skill declares, instead of degrading to a one-line prose string
  when evidence is missing. The hosted runtime's placeholder card is explicitly
  labelled so it cannot be presented as a rendered site observation
  (`src/blueprint_pipeline/agent_runtime/orchestrator.py`,
  `src/blueprint_pipeline/native_runtime_backend.py`).

## 2026-07-24 (second entry)

### User-Facing

- GPU selection is now an explicit per-workload policy rather than one global
  rule. The Isaac RT-core exclusion (A100/H100, extended to H200/B200/GB200) was
  being applied to every Vast offer selection, barring generation and training
  campaigns from the hardware they need. `generation`, `training` and `open`
  policies carry no denylist and travel with their own rate/VRAM envelopes;
  Blackwell RTX PRO 6000 96GB, H200, B200 and RTX 5090 are recognised. An
  unknown policy name fails closed to the Isaac policy
  (`src/blueprint_pipeline/vast_provider_adapter.py`,
  `docs/EXECUTION_COST_AND_ARCHITECTURE_GATES.md`).
- Separated adopting a published world-model architecture from admitting an
  upstream release. Building on already-pinned permissively licensed components
  is no longer blocked by a third party's unreleased code; the resulting model
  is Blueprint-authored, may not use the upstream name or metrics, and must pass
  ordinary evaluator qualification. Authorisation is to build, not to claim
  (`src/blueprint_pipeline/world_model_architecture_adoption.py`).

### Employee-Facing

- Added a resident OSCAR worker so a closed-loop rollout loads the checkpoint
  once instead of once per step. A dead or desynchronised worker fails the step
  closed rather than falling back to per-step spawning, restarts must be
  explicitly budgeted and are counted, and cold-start versus warm-step latency
  is reported separately. This is a throughput change, not evidence of
  generation quality or task success
  (`src/blueprint_pipeline/oscar_resident_worker.py`,
  `src/blueprint_pipeline/oscar_resident_worker_main.py`).
- Added judge spend governance mirroring the GPU envelope: target spend, hard
  cap, request and frame ceilings, TTL, a ledger, and a cohort hard stop. Prices
  are operator-supplied and an unpriceable request is denied rather than waved
  through; failed requests are still settled. The graded-progress lane treats an
  absent policy as a refusal (`src/blueprint_pipeline/judge_spend_governor.py`).
- Retired the platform-wide 7-dimensional action invariant in favour of a
  registered action-space registry covering the SC3 7-D delta end-effector
  layout, the 78-D Unitree G1 whole-body command, and a 43-D arm/hand layout.
  The default remains SC3, so existing callers and blocker strings are
  unchanged; unregistered spaces fail closed
  (`src/blueprint_pipeline/action_space_registry.py`,
  `src/blueprint_pipeline/action_normalization.py`,
  `src/blueprint_pipeline/oscar_cosmos_wam_command_adapter.py`).
- Registered a second embodiment as a zero-GPU conformance fixture, differing on
  base, arm count, action interface and camera rig, and made unknown profile
  lookups raise a typed `UnknownRobotProfileError`
  (`src/blueprint_pipeline/scene_placement/robot_profile.py`).
- Added a gate-reachability audit that probes real validators and source rather
  than asserting. It records that `validate_external_study` never returns
  `validated` — making `sc3_eval_protocol`'s `public_rank_fidelity_claim_eligible`,
  `claim_ready` and `eligible_preregistered_external_rank_fidelity` unreachable
  by construction — that two claim fields are emitted as literal `False`, and
  that the two OOD axis vocabularies diverge. Blocker lists can now be split
  into what waiting could clear and what it never will
  (`src/blueprint_pipeline/gate_reachability_audit.py`).

## 2026-07-24

### User-Facing

- Added a public real-world benchmark anchor path so the evaluation harness can
  be validated against independently published robot outcomes before any
  customer anchor exists. This produces the previously unproduced
  `roboarena_snapshot_sha256` digest and the previously unproduced
  `external_reference_results.v1` artifact. Results carry the distinct
  `harness_validation_public_anchor` scope and are structurally barred from
  upgrading site-specific rank fidelity, other embodiments, or deployment
  readiness (`src/blueprint_pipeline/public_benchmark_anchor.py`,
  `docs/EVALUATOR_ATTRIBUTION_AND_PUBLIC_ANCHOR.md`).
- Demoted Pearson to supporting evidence in the external rank-fidelity report
  and promoted pairwise-ordering accuracy to the headline, alongside a
  resolving-power (minimum-detectable-difference) curve. Correlation degrees of
  freedom come from the policy cohort, not the rollout count
  (`src/blueprint_pipeline/benchmark_protocol.py`).
- Shipped the producer for `roboworld_progress_score.v1` graded task-progress
  scores, whose consumers (rubric validation, segment aggregation, aggregation
  ablation, judge calibration) already existed with nothing to feed them.
  Scores remain generated-media review evidence; a score of 5 does not prove
  physical task success (`src/blueprint_pipeline/roboworld_progress_judge.py`).

### Employee-Facing

- Added `rank_fidelity_statistics`: Fisher-z correlation intervals, Wilson
  proportion intervals, two-proportion minimum-detectable-difference, exact
  one-sided Fisher tests, and bootstrap-reliability judgement. These make the
  small-sample behaviour of every published evaluator number computable rather
  than implicit (`src/blueprint_pipeline/rank_fidelity_statistics.py`).
- Repaired the policy-ranking-ladder acceptance statistic. The ladder accepted
  `recovered` on a strict ordering of per-rung Bernoulli means at three
  replicate seeds, where the exact one-sided p-value for an adjacent-rung
  difference is 0.5. Adjacent-pair separation is now tested exactly and computed
  before the pass/fail decision, unresolvable separation blocks acceptance with
  the new `inconclusive_underpowered_separation` status, and the replicate seed
  count is a builder parameter defaulting to a value derived from the separation
  it must resolve (`src/blueprint_pipeline/policy_ranking_ladder.py`).
- Stopped the external rank-fidelity bootstrap from silently discarding
  undefined replicates, which narrowed rather than widened the published
  interval. Attempted/defined replicate counts, the undefined fraction, and an
  explicit reliability verdict are now reported per metric
  (`src/blueprint_pipeline/benchmark_protocol.py`).
- Added a world-model-free control ranker (action-chunk jerk, gripper toggle
  rate, timeout rate, first-frame prior, plus null controls) that attributes an
  evaluator's rank agreement against the best baseline using a paired bootstrap
  over policies. A winning baseline is not an evaluator; the arm exists to price
  the evaluator's marginal contribution
  (`src/blueprint_pipeline/control_ranker.py`).
- Raised VLM judge frame budgets from 5-6 to 16 frames and added a rubric-aware
  sampling contract that fails closed below 2.0 samples/second. Six frames
  across a 25-second rollout is 0.24 fps, which cannot localise where a rollout
  diverged (`src/blueprint_pipeline/wam_generated_video_success_label_gemini.py`,
  `src/blueprint_pipeline/wam_episode_consistency_label_openai.py`).

## 2026-07-23

### User-Facing

- Published the model-neutral RoboWorld-inspired progress evaluator,
  blinded judge-calibration study, and hierarchical benchmark-uncertainty
  contracts. Results remain digest-bound and claim-ineligible for public rank
  fidelity without separately accepted frozen external anchors
  (`src/blueprint_pipeline/roboworld_evaluator.py`,
  `src/blueprint_pipeline/benchmark_uncertainty.py`,
  `docs/ROBOWORLD_EVALUATOR_INTEGRATION.md`).
- Corrected qualification and GEAR-SONIC controller binding so the executable
  runtime uses the requested controller and each horizon reply must be tied to
  a fresh, exact action/frame convention. Controller startup, fresh replies,
  and generated review media still do not establish task success
  (`src/blueprint_pipeline/single_g1_kitchen_qualification_contract.py`,
  `src/blueprint_pipeline/gear_sonic_official_zmq_executor.py`).
- Added a scoped Agentic Real2Sim paper analysis that maps useful planning and
  repair ideas onto Blueprint's capture-first architecture without adopting
  the paper's system as a production dependency or inheriting its reported
  results (`docs/AGENTIC_REAL2SIM_PAPER_ANALYSIS_2026-07-23.md`).

### Employee-Facing

- Hardened OSCAR release provenance and preflight: the image now verifies
  sealed source/assets without relying on a runtime Git checkout, upgrades the
  legacy shim before sealing, and propagates verified runtime identity into
  the qualification path (`src/blueprint_pipeline/oscar_runtime_source_provenance.py`,
  `deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Release.Dockerfile`).
- Hardened RunPod WAM teardown for malformed or incomplete provider responses,
  and corrected controller-FK readiness hand ordering. These are closure and
  startup-contract improvements, not evidence that a provider allocation,
  simulator episode, or teardown occurred
  (`src/blueprint_pipeline/runpod_wam_teardown.py`,
  `src/blueprint_pipeline/groot_oscar_worker_startup_script.py`).
- A clean checkout-only branch added late-day controller freshness/head-POV
  review quality, policy-authoritative OSCAR transitions and action
  conditioning, GPU PID attribution, and a bounded-finetune probe skip
  (`src/blueprint_pipeline/oscar_isaac_closed_loop_eval.py`,
  `src/blueprint_pipeline/isaac_task_review_renderer.py`,
  `src/blueprint_pipeline/g1_microwave_groot_finetune_component.py`).

### Future-Agent-Facing

- The America/Chicago window contains nine first-parent `origin/main` commits,
  `1a2135a2` through `0247bee5` (PRs #169--#177), plus six cleanly committed
  checkout-only changes, `b0b14622` through `e241a47e`. At review time
  `main == origin/main`, while detached `HEAD` was 17 commits ahead; no
  attributable uncommitted July 23 work is recorded.
- RoboWorld paper metrics, Agentic Real2Sim paper results, evaluator fixtures,
  unit tests, startup/preflight success, controller freshness, and review
  renders are not Blueprint ranking-fidelity, live-provider, deployment,
  public-readiness, simulator-semantic-success, or physical-robot proof.
- Keep policy actions authoritative over OSCAR state transitions, preserve
  exact source/runtime/controller bindings, and keep WAM rollout execution,
  generated-video labels, external forward/inverse consistency scoring, and
  policy ranking as separate evidence layers.

## 2026-07-22

### User-Facing

- Reworked the core product spine around typed, provider-neutral capture,
  packaging, evaluation-run, optional-support, and hosted-runtime contracts;
  removed verified dead or superseded pivots, made qualification outputs
  opt-in, and added aggregate run summaries plus a documented production
  profile (`src/blueprint_pipeline/core/`,
  `src/blueprint_pipeline/run_summary_aggregation.py`,
  `docs/PRODUCTION_PIPELINE_PROFILE.md`).
- Preserved actionable Vast qualification startup diagnostics across process
  exit races and diagnostic failures, including component state, log tails,
  GPU/process observations, and continuing-spend blockers
  (`src/blueprint_pipeline/single_g1_kitchen_qualification_observability.py`).
- Added a model-neutral RoboWorld-inspired 0--5
  task-progress evaluator with explicit world-model failure stages,
  criterion-scoped camera authority,
  judge confidence/abstention, evidence and model/prompt/calibration digests,
  and preservation through the existing WAM success-label normalization path
  (`src/blueprint_pipeline/roboworld_evaluator.py`,
  `docs/ROBOWORLD_EVALUATOR_INTEGRATION.md`).
- Added comparison-only segment aggregation for
  terminal, mean, minimum, maximum, regression-aware, and stable-maintenance
  scores. Maximum remains experimental and cannot become the default without
  a measured ablation.

### Employee-Facing

- Sealed the reviewed OSCAR source commit and post-patch runtime-tree digest
  into the thin image, re-verifies that tree before allocation/execution, and
  rejects missing seals, tree drift, unsafe links, or unsealed bytecode caches
  (`src/blueprint_pipeline/oscar_runtime_source_provenance.py`,
  `deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Foundation.Dockerfile`).
- Hardened OSCAR image package transport with HTTPS Ubuntu sources, bounded
  retries/timeouts, and disabled HTTP pipelining. Detached `cpu-build`
  supervisors now survive local `SIGINT` while retaining intentional `SIGTERM`
  and independent resource watchdog controls
  (`deploy/docker/robot_eval_worker/groot_oscar_closed_loop/apt_transport_hardening.conf`,
  `src/blueprint_pipeline/paid_resource_allocator.py`).
- Added an executable blinded GPT/Gemini/human
  calibration study with confusion, confidence calibration, false-success,
  policy-rank, and task/view/contact/artifact bias reports, plus hierarchical
  policy/site/task/initial-condition uncertainty, trial-count convergence, and
  leave-one-out sensitivity (`src/blueprint_pipeline/benchmark_uncertainty.py`).
- Added frozen schemas and tracked
  evaluator/admission artifacts. The current RoboWorld admission status is
  `awaiting_upstream_release`; paper-only Step Forcing reimplementation and
  backend integration remain deferred until licensed code, weights, and
  reproducible runtime artifacts exist.

### Future-Agent-Facing

- The original America/Chicago window contained six commits,
  `d873dd80` through `35ea6a3f` (including the July 21 changelog merge).
  The RoboWorld evaluator, uncertainty tooling, schemas, tests, and this
  expanded July 22 entry were prepared locally afterward and are included in
  the subsequent source publish. They must not be reported as deployed.
- RoboWorld's reported `0.989` Pearson, `0.970` progress-rubric Spearman,
  `0.922` binary-score Spearman, and `0.862` wrist-as-success Spearman values
  are external paper context only. They are not Blueprint measurements.
- The evaluator profile is backend-neutral. Do not hardwire it to a Step
  Forcing implementation, and do not treat generated-media progress, judge
  calibration, or uncertainty reports as physical success or public rank
  fidelity without independently accepted frozen real anchors.
- July 22 establishes contracts, fail-closed gates, diagnostics, and tested
  build/runtime behavior only. It does not establish a successful OSCAR image
  build, provider allocation, live simulator episode, ranking fidelity,
  deployment, public readiness, or physical-robot performance.

## 2026-07-21

### User-Facing

- Added model-neutral evaluator runtime-evidence and qualification workflows,
  plus reproducible benchmark bindings for exact environment, adapter, and
  evaluator digests. These contracts fail closed on stale or mismatched
  evidence and do not turn evaluator startup or generated outputs into ranking
  success (`src/blueprint_pipeline/evaluator_runtime_evidence.py`,
  `src/blueprint_pipeline/evaluator_qualification_workflow.py`,
  `src/blueprint_pipeline/benchmark_protocol.py`).
- Added a provider-neutral NVIDIA simulation integration framework for
  SimReady validation, Omniverse library preflight, gsplat conformance, Cosmos
  3 edge experiments, and external worker handoffs. The checked-in completion
  matrix and receipt describe implemented/tested contracts; they are not proof
  of deployed NVIDIA services, a successful live-provider run, or physical
  robot performance (`docs/NVIDIA_SIGGRAPH_2026_IMPLEMENTATION_RUNBOOK.md`,
  `docs/evidence/nvidia_siggraph_2026_completion_matrix.json`).

### Employee-Facing

- Hardened paid-resource admission and closure: GPU canaries require fresh
  protected-main source binding, Vast qualification collection is bound to the
  exact launched release and task contract, model-volume cleanup records
  terminal evidence, and watchdog cancellation requires exact-attempt plus
  global provider-zero proof (`src/blueprint_pipeline/paid_resource_allocator.py`,
  `src/blueprint_pipeline/single_g1_kitchen_qualification_contract.py`,
  `scripts/gpu_spend_guard.py`).
- Added Python 3.10-safe TOML loading for sealed runtime paths and narrowed the
  RunPod adapter/image health-check imports, preserving compatibility without
  weakening source or runtime identity checks
  (`src/blueprint_pipeline/toml_compat.py`,
  `src/blueprint_pipeline/runpod_provider_adapter.py`).

### Future-Agent-Facing

- The America/Chicago window contains ten first-parent `origin/main` commits,
  `2ec960d9` through `95b10eb1` (PRs #142–#153, including the July 20 changelog
  merge); the checkout is clean with `HEAD == main == origin/main`, so no
  attributable uncommitted July 21 source work is recorded.
- Keep evaluator evidence, evaluator qualification, benchmark reproducibility,
  simulation preflight, external worker results, provider allocation, teardown,
  semantic/ranking success, deployment, and physical-robot execution as
  separate proof layers. July 21's contracts and test receipts do not establish
  public readiness or a successful live episode.

## 2026-07-20

### User-Facing

- Added retry-safe, per-lane resume markers keyed to normalized raw capture
  inputs, so completed lanes can be reused without weakening changed-input or
  missing-output checks. Raw `scenes/` and `targets/` evidence now transitions
  to Archive storage instead of deletion (`src/blueprint_pipeline/lane_resume.py`,
  `deploy/storage/primary-capture-bucket-lifecycle.json`).
- Added a backward-compatible `frames_index.v2` reader for packed frame archives
  and stopped materialization from overwriting an existing rich frame index.
  Added feature-flagged Cloud CDN URL signing, default off with direct-GCS
  fallback; CDN provisioning and deployment remain unproven owner actions
  (`src/blueprint_pipeline/frames_layout.py`,
  `src/blueprint_pipeline/arena_package_delivery_local.py`,
  `docs/BUYER_DELIVERY_CDN_DESIGN_2026-07-20.md`).

### Employee-Facing

- Generalized the sim-only policy-evaluation stack and separated evaluator
  evidence profiles from model backends. Rankings now fail closed unless fresh,
  structured, profile-specific evaluator evidence is identity- and
  digest-bound; WAM execution, generated review media, and ranking evidence
  remain distinct claims (`src/blueprint_pipeline/policy_evaluation_contracts.py`,
  `src/blueprint_pipeline/evaluator_evidence_profiles.py`,
  `src/blueprint_pipeline/decision_grade_ranking.py`).
- Hardened real-site evaluation admission by binding independently verified
  site evidence, task grounding, site-task contracts, and complete
  out-of-distribution axes. Scaniverse imports and local site records do not by
  themselves prove live-site evaluation readiness
  (`src/blueprint_pipeline/site_reference_database.py`,
  `docs/SCANIVERSE_ASSET_IMPORT.md`).
- Preserved teardown obligations for ambiguous fine-tune launches while
  distinguishing pre-create refusals from potentially billable allocation
  outcomes. Provider absence remains required before ambiguous attempts close
  (`src/blueprint_pipeline/g1_microwave_finetune_provider_job.py`,
  `src/blueprint_pipeline/gpu_render_providers.py`).
- Removed Terraform indexes for the unwritten `captures` collection, retained
  scale-to-zero GPU defaults with measured model-load telemetry and an
  in-process runtime cache, and documented modeled warm-pool economics. Applying
  Terraform, enabling a warm pool, and provisioning the CDN were not performed
  by these repo changes (`deploy/terraform/main.tf`,
  `src/blueprint_pipeline/privacy_service_runtime.py`,
  `docs/GPU_WARM_POOL_ECONOMICS_2026-07-20.md`).

### Future-Agent-Facing

- The America/Chicago window contains six cleanly committed changes on `main`,
  `ee403467` through `c8a0f890` (PRs #135, #130, #137, #138, #140, and #141);
  no attributable uncommitted July 20 source work is recorded.
- Keep raw capture/provenance truth authoritative. Lane completion, packed-frame
  readability, startup, model caching, review media, evaluator evidence,
  ranking admission, real-site admission, provider allocation, teardown,
  deployment, and physical-robot execution are separate evidence layers. The
  July 20 contracts and tests do not establish public readiness, a successful
  live-provider episode, deployed infrastructure, or physical-robot success.

## 2026-07-17

### User-Facing

- Hardened the governed RunPod model-cache path so full post-upload verification
  re-downloads objects with the volume endpoint's supported streaming `GetObject`
  API. Verification still hashes the downloaded cache and fails closed on
  corruption; this is transport-integrity evidence, not model-runtime or episode
  success (`src/blueprint_pipeline/groot_oscar_runpod_s3_model_cache.py`).

### Employee-Facing

- Made runtime-carrier library-path manifests accept shell-quotable paths while
  rejecting delimiter, expansion, control-character, traversal, and
  out-of-root inputs. The generated loader environment is now shell-quoted before
  reuse, preserving the fail-closed carrier contract without rejecting valid
  runtime layouts (`src/blueprint_pipeline/groot_oscar_runpod_carrier_volume.py`,
  `src/blueprint_pipeline/groot_oscar_model_cache_s3_remote_executor.py`).
- The committed America/Chicago window contains two `origin/main` commits,
  `201146ea` (PR #128) and `01545646` (PR #129). The reviewed feature checkout
  is clean and diverges from `origin/main` (`81` checkout-only commits and `39`
  main-only commits), so no July 17 uncommitted source change is recorded.

### Future-Agent-Facing

- Keep loader-path serialization, cache upload, cache re-download, digest
  verification, provider allocation, startup readiness, semantic task success,
  artifact review, and teardown as separate evidence layers. Ignored July 17
  operational files under `output/single_g1_kitchen_episode_20260716/` are
  support artifacts, not source changes or standalone proof of a completed
  episode, deployment, public readiness, or physical-robot execution.

## 2026-07-15

### User-Facing

- Added a governed RunPod FlashBoot kitchen-campaign path and a persistent,
  pre-baked DigitalOcean GPU-host path to reduce repeated cold-start work while
  retaining exact image/cache/source bindings, bounded leases, watchdogs, and
  teardown controls (`src/blueprint_pipeline/production_gpu_runpod_autoscaler.py`,
  `src/blueprint_pipeline/groot_oscar_digitalocean_prebaked_host.py`). These are
  provider/runtime paths, not proof of semantic task success or deployment.

### Employee-Facing

- Merged the canonical paid-resource and GPU reliability closure (PR #80), then
  added a strict RunPod policy smoke probe, hardened S3 multipart transfer and
  recovery, and corrected the offline Cosmos cache layout. Model-cache reuse is
  now bound to verified transport, inventory, lifecycle policy, an armed
  watchdog deadline, and byte-exact handoff evidence
  (`src/blueprint_pipeline/paid_resource_allocator.py`,
  `src/blueprint_pipeline/groot_oscar_runpod_s3_model_cache.py`,
  `src/blueprint_pipeline/groot_oscar_model_cache.py`).
- Reconciled the authorized GPU campaign budget across cold-start retries and
  terminal canaries without turning cumulative authority into one unbounded
  job. Persistent DigitalOcean bake storage and RunPod warm/cache retention now
  remain explicit, bounded resource lanes rather than implicit leftovers
  (`src/blueprint_pipeline/production_gpu_campaign_budget.py`,
  `src/blueprint_pipeline/paid_provider_lane_lease.py`,
  `docs/runbooks/groot-oscar-thin-release.md`).
- The committed America/Chicago window contains 16 first-parent `origin/main`
  commits from `9cba1c2e` through `e4ebfc23` (PRs #80–#95 and the FlashBoot
  campaign merge). At review time the local checkout was still at July 14 head
  `8de9115d`; the only tracked worktree change was the prior changelog update,
  so no July 15 uncommitted source change is recorded.

### Future-Agent-Facing

- Preserve the canonical allocator boundary for every CPU build, model-volume,
  and GPU-canary mutation. Cache transfer, cache retention, a prebaked host,
  FlashBoot startup, provider readiness, review-media validity, semantic task
  success, buyer claims, and teardown are separate evidence layers.
- July 15 local output directories contain operational attempt artifacts, but
  they are downstream support evidence rather than source truth. Do not infer a
  successful episode, public readiness, deployment, or physical-robot result
  without validating the exact attempt closure and final provider inventory.

## 2026-07-14

### User-Facing

- Added the production GPU startup path that binds customer jobs only to an
  already-ready, exact-release worker. When no worker is ready, the request
  remains queued and writes asynchronous scale demand; it never allocates a VM,
  installs host software, logs in to a registry, or pulls the 47 GB worker image
  inline (`src/blueprint_pipeline/production_gpu_worker_pool.py`,
  `docs/runbooks/production-gpu-startup-and-warm-pool.md`).
- Added one durable campaign lifecycle after warm binding: smoke seed 1000 must
  pass before seeds 1001–1003 can run; episodes stop dynamically, review media
  is at least 640x480, artifacts resume by verified offset/hash, and customer
  status never equates startup or artifact arrival with task success
  (`src/blueprint_pipeline/production_gpu_campaign_control_plane.py`).
- Split the GR00T+OSCAR worker into a cached robot foundation, an external
  checksum-bound model volume, and a thin Blueprint release so normal releases
  do not rebuild or repull the full robot/model stack. Paid CPU builds,
  model-volume creation, and GPU canaries now enter only through the shared
  fail-closed allocator (`docs/runbooks/groot-oscar-thin-release.md`,
  `src/blueprint_pipeline/paid_resource_allocator.py`).

### Employee-Facing

- Added the next-release GR00T+OSCAR reliability closure after the July 13 G4
  campaign: the official image path now emits BuildKit SBOM/provenance
  attestations, scans only the immutable registry digest, admits disk for both
  build and scan scratch, records layer/startup evidence, and runs the finished
  digest as the OCI runtime user. The worker closure also pins and verifies the
  WBC/GEAR runtime assets, resolves the Cosmos processor from its offline
  snapshot, and composes the pinned G1 USD when a scenario omits `/World/G1`.
- Added a provider-neutral, resumable GPU campaign state machine with immutable
  configuration, OS-level single ownership, budget and duplicate-allocation
  gates, strict smoke-to-episode admission, paid-lifetime-capped stage
  deadlines, explicit same-allocation canary handoff schemas, and
  finally-equivalent teardown. Added a pinned GCP G4 host-image template,
  startup self-test, regional mirror equivalence/planning contracts, release
  SLO instrumentation, and focused regression/contract tests.
- Added a GCP Packer contract for an immutable GPU host image containing the
  driver payload, Docker, NVIDIA Container Toolkit, and exact digest-pinned
  worker cache. GCP/AWS VM startup now verifies the baked image marker and local
  Docker cache instead of pulling at boot (`deploy/packer/gcp_g4_gpu_worker_host.pkr.hcl`,
  `src/blueprint_pipeline/cloud_vm_render_providers.py`).
- Added restart-safe ready-worker leases, stale-worker quarantine, exact release
  fingerprints, atomic autoscaler claims, a private hardened systemd service,
  and a fail-closed promotion gate. The gate cannot report
  `customer_launch_ready` without exact-release live p95 bind/replenishment,
  rollback, inventory, and teardown evidence.
- Classified the 47.1 GB exact RunPod release as `active_worker_only`, added a
  same-session worker registration agent that joins all nine readiness checks,
  and enforced one Secure L40S attempt with A40 fallback only after an explicit
  no-allocation capacity rejection. The fallback uses an honest GPU serving
  class while retaining the actual GPU model as evidence
  (`production_gpu_image_contract.py`, `production_gpu_worker_agent.py`,
  `production_gpu_runpod_autoscaler.py`).
- Added a deterministic release-candidate closure, repeated-campaign
  promotion/quarantine evaluator, scheduled contract qualification, opt-in
  private warm-bind p95 canary, hardened campaign service, and the ten-control
  operating model (`production_gpu_release_candidate.py`,
  `production_gpu_reliability_qualification.py`,
  `.github/workflows/production-gpu-reliability.yml`,
  `docs/PRODUCTION_GPU_RELIABILITY_OPERATING_MODEL.md`).
- Added an atomic dual-cap campaign budget ledger and independent hard-TTL
  termination watchdog so a crashed warm-worker owner cannot silently reuse
  spend/time or leave a paid allocation running
  (`production_gpu_campaign_budget.py`, `production_gpu_warm_watchdog.py`).
- Added live launch qualification, three-cycle asynchronous replenishment
  measurement, and candidate quarantine/alternate-worker rollback drills; none
  can promote from local or historical evidence
  (`production_gpu_launch_qualification.py`,
  `production_gpu_replenishment_probe.py`, `production_gpu_rollback_drill.py`).
- Added guarded remote foundation/release builds, RunPod canary and network
  model-volume allocators, exact-source build packets, pre-allocation
  prerequisite checks, watchdog handoff verification, global inventory/budget
  accounting, sanitized failure evidence, and explicit GPU/datacenter capacity
  admission (`src/blueprint_pipeline/groot_oscar_digitalocean_builder.py`,
  `src/blueprint_pipeline/groot_oscar_runpod_model_volume.py`,
  `scripts/verify_paid_resource_allocator.py`). Legacy build scripts are hard
  disabled as launchers; the canonical allocator commands are the supported
  paid-resource entrypoints.
- Isolated incompatible Isaac/GR00T/OSCAR Python and CUDA dependencies inside
  the pinned foundation, bound cache sizing and runtime imports to governed
  metadata, and required admitted model-cache identity before worker bootstrap
  (`deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Foundation.Dockerfile`,
  `src/blueprint_pipeline/isaac_g1_worker_bootstrap.py`).
- Merged the production reliability control plane as PR #78, including the
  first-class GCP/AWS adapters that were uncommitted in the July 13 snapshot.
  The July 14 window then continued through commit `8de9115d`; the worktree was
  clean at this review, so no uncommitted July 14 changes are recorded.

### Future-Agent-Facing

- These changes are next-release source hardening. They do not alter the July
  13 immutable release candidate or upgrade its blocked kitchen campaign into
  simulator-step, learned-action, semantic-success, buyer-claim, or physical
  robot evidence. A new image build, host image, mirror copy, and live campaign
  remain separately gated provider operations.
- Treat the immutable 20–25 minute cold-start campaign as release-engineering
  evidence, not the production-serving architecture. Packer build and local
  pool tests do not prove a live GPU host image, warm capacity, or latency SLO;
  preserve `local_contract_ready_live_proof_required` until the exact promoted
  tuple passes the live evidence gate.
- Keep foundation-image construction, model-volume materialization, thin-release
  publication, provider allocation, readiness, review-media validity, artifact
  retrieval, semantic task success, and teardown as separate claims. The new
  admission and watchdog controls reduce launch/spend risk but do not prove a
  published exact image, populated external cache, live customer-ready worker,
  successful episode, public readiness, deployment, or physical-robot result.
- Use only `python -m blueprint_pipeline.paid_resource_allocator cpu-build`,
  `model-volume`, or `gpu-canary` for new paid resources. Provider-specific
  modules remain adapters, and every launch must retain exact source/image/cache
  bindings plus provider absence evidence after teardown.

## 2026-07-13

### User-Facing

- Published the provider-neutral Evaluation Run and shared consent-normalization
  work recorded in the July 12 entry. The July 13 merge made those contracts
  available on `main`; it did not add live provider, semantic-task-success,
  public-readiness, or deployment proof (`src/blueprint_pipeline/evaluation_run_contract.py`,
  `src/blueprint_pipeline/consent_normalization.py`).

### Employee-Facing

- Hardened the sealed GR00T+OSCAR worker startup path by sealing and pinning the
  nested Cosmos backbone for offline use, making required runtime trees writable,
  detecting child-process death during readiness waits, and moving persistent
  Isaac articulation control to the supported `SingleArticulation` API
  (`deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile`,
  `src/blueprint_pipeline/isaac_runtime_task_backend.py`).
- Made review-renderer canaries fail closed on a bounded timeout and on blocked
  verdicts, while preserving the distinction between startup/review-media checks
  and task success. Legacy robot-eval requests now execute through their existing
  builder before compiling the canonical Evaluation Run mirror, preserving
  dataset-card task enrichment (`src/blueprint_pipeline/isaac_review_renderer_canary.py`,
  `src/blueprint_pipeline/robot_eval_evaluation_run_adapter.py`).
- **Uncommitted local work (July 13):** added first-class GCP Compute Engine and
  AWS EC2 render-provider adapters, operator setup guidance, and spend-guard /
  termination inventory integration; also generalized RunPod capacity checks for
  the requested secure or community pool (`src/blueprint_pipeline/cloud_vm_render_providers.py`,
  `docs/GCP_AWS_GPU_PROVIDER_SETUP.md`, `scripts/gpu_spend_guard.py`). This is a
  dirty-tree snapshot, not merged provider-runtime or paid-run proof.

### Future-Agent-Facing

- The committed July 13 America/Chicago window runs from `4a2cae3b` through
  `a4b01fd0`. Treat image construction, health checks, and canary controls as
  release/runtime safeguards only; they do not prove a rebuilt image was
  published or that a live episode completed successfully.
- Preserve the current uncommitted cloud-provider files when continuing work.
  Before upgrading their status, run focused adapter/spend tests and bind any
  provider claim to authenticated inventory, exact resource identity, teardown,
  and absence verification.
## 2026-07-12

### User-Facing

- Introduced provider-neutral Evaluation Runs as one fail-closed composition of
  scene bundle, robot adapter, task/scenario pack, policy adapter,
  runtime/provider profile, and proof contract (`evaluation_run.v1`). The
  compiler is side-effect free; execution requires `--allow-execution`, and
  runtime completion remains separate from required-evidence satisfaction and
  public claim upgrades (`src/blueprint_pipeline/evaluation_run_contract.py`,
  `src/blueprint_pipeline/evaluation_run_execution.py`,
  `docs/architecture/evaluation-run-interface.md`).
- Published the shared consent normalizer across materialization,
  qualification, takedown, Post-Training Data Packages, readiness, and proof
  contracts. Malformed, contradictory, or revoked consent fails closed and can
  only downgrade grants (`src/blueprint_pipeline/consent_normalization.py`,
  `tests/test_consent_rights_cross_surface_invariants.py`).

### Employee-Facing

- Generalized the historical G1 kitchen lane behind a pack registry and stable
  adapter/execution seams while preserving its legacy schemas and entrypoints.
  `g1_warehouse` is a configuration-only second pack; no GPU run or task
  success is claimed. The final compatibility fix preserves dataset-card task
  enrichment before compiling the canonical mirror and points
  `blueprint-compile-evaluation-run` at the correct contract CLI
  (`src/blueprint_pipeline/evaluation_run.py`,
  `src/blueprint_pipeline/robot_eval_evaluation_run_adapter.py`,
  `docs/specs/evaluation-run-pack-architecture-2026-07-12.md`).
- Merged PR #68's FABLE-001..007 integrity controls and subsequent sealed-image
  lifecycle fixes, including runtime-user interpreter access, supplementary
  Isaac group resolution, and a pre-push OCI runtime smoke gate. These are
  fail-closed release/runtime controls, not live episode or semantic-success
  proof (`deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile`,
  `scripts/build_push_groot_oscar_closed_loop_image.sh`).
- **Local runtime evidence (not committed):** July 12 America/Chicago artifacts
  record an initial exact-image canary failure caused by runtime-user
  permission defects, followed by a DigitalOcean fast startup and
  review-renderer canary pass for exact image digest `sha256:fd722b07...f975`,
  with API-confirmed teardown and zero final live resources
  (`output/groot_oscar_live_episode_20260712/canary_root_cause_imgfix.json`,
  `output/groot_oscar_live_episode_20260712/worker_image_runtime_evidence_fd722b07_final.json`).
  The passing artifact binds a source snapshot plus dirty-patch digest and
  explicitly leaves task success, semantic success, and physical-robot
  readiness unproven.

### Future-Agent-Facing

- The committed July 12 window runs from merge `dbdbe95a` through
  `dbf787c3`. The worktree was clean at the July 13 review; the local runtime
  artifacts above are ignored support evidence, not committed source or a
  substitute for provider episode closure.
- Keep `evaluation_run_plan.v1` compilation, provider/runtime completion,
  required-evidence satisfaction, semantic task success, and public claim
  upgrades as separate proof layers. Preserve the legacy robot-eval and G1
  kitchen adapters until callers have migrated to the canonical six-part
  interface.

## 2026-07-11

### User-Facing

- Closed the seven code-level blockers from
  `docs/archive/specs/claude-fable-5-hardest-blockers-audit-2026-07-11.md`
  (FABLE-001..FABLE-007): the attempt closure now reconstructs every worker
  proof row from hash-verified, Ed25519-attested leaf artifacts instead of
  trusting worker `passed` booleans and never repairs a missing worker
  identity; the review-media horizon is bound to the immutable attempt/task
  request with contiguous per-camera indices, per-step action-SHA bindings,
  and global duplicate-frame rejection; G1 proprioception uses an explicit
  canonical DOF map instead of substring grouping; the GEAR-SONIC
  controller-to-MuJoCo mapping is named, digest-pinned, and permutation
  rejecting with real loopback ZMQ tests; a mandatory pre-allocation identity
  gate compares attempt/launch/registry/worker-evidence image digests and
  source identities before any paid capacity call; relative task success is
  evaluated against a signed episode baseline (current minus episode initial)
  end to end; and the Bandit findings are fixed at source through a
  centralized fail-closed outbound-HTTP boundary plus gated, revision-pinned
  model retrieval. All controls are fail-closed contracts; none of this is
  proof of a live provider episode, task success, rank fidelity, or physical
  robot readiness.
- Hardened the sealed Isaac 6 / G1 worker image path: the build now installs
  the pinned TensorRT closure, uses the OSCAR virtual-environment interpreter
  for checkpoint prefetch, bakes digest-pinned Isaac assets, inspects runnable
  child manifests when a registry tag resolves to an OCI image index, and
  bounds descendant processes during canary shutdown. These changes improve
  image reproducibility and failure containment; they do not prove that the
  rebuilt image was published or that a live provider episode succeeded.

### Employee-Facing

- New modules: `g1_kitchen_proof_row_validation.py`,
  `g1_kitchen_pre_allocation_identity.py`, `g1_proprioception_map.py`,
  `task_episode_baseline.py`, `gear_sonic_joint_order_contract.py`,
  `gear_sonic_container_smoke.py`, `safe_outbound_http.py`. The sealed
  GR00T+OSCAR healthcheck now emits `configured_g1_asset_binding_valid` and
  `configured_g1_usd_exists` as distinct claims through one shared
  runtime-metadata schema validated by the real evidence assembler. The
  hermetic pyarrow fail-closed test no longer skips. `pyzmq` joined the dev
  extras so the real ZMQ transport tests cannot skip green in the Full Test
  Lane. Fixed a cross-test `MUJOCO_GL` environment leak in
  `tests/test_g1_site_3dgs_mujoco_preview.py`.
- Attempt-local Ed25519 trust generation now publishes the raw public keys and
  role mappings required by host verification, bound to the complete immutable
  attempt identity. Private keys remain mode-0600 inside the allocation;
  closure auto-discovers the collected public-pin manifest and rejects a
  manifest replayed from another attempt.
- Live-run status and the exact remaining external blockers (sealed-image
  rebuild, spend cap, scorer/semantic-review services, attestation key pins,
  and protected-PR merge approval) are tracked in
  `docs/archive/specs/fable-remediation-and-live-readiness-2026-07-11.md`.
- The FABLE-001..007 patch series is published as PR #68. `main` branch
  protection is enabled and was negative-tested against direct and force
  pushes; hosted proof must come from every required check passing on the
  latest PR head, never from a superseded run.
- Release workflows now retain and verify command-attestation artifacts, and
  the quality-gap ledger was rebound to committed source bytes after the
  release/image dependency changes. Worker-image evidence inspection now
  follows OCI index children to the runnable manifest instead of treating an
  index descriptor as the final image configuration.
- **Uncommitted local work (July 11):** a shared fail-closed consent normalizer
  is wired across materialization, qualification, takedown, PTDP, readiness,
  and proof-contract surfaces, with cross-surface hostile-input invariants in
  `tests/test_consent_rights_cross_surface_invariants.py`. Malformed,
  contradictory, or revoked consent can only downgrade grants. The associated
  ledger digests are worktree bindings and must be rebound to committed bytes
  before they can support release evidence.

### Future-Agent-Facing

- Earlier entries in this file that describe work as "uncommitted local
  changes" (July 4-8 snapshots) are historical: that work was merged in
  `df030e45`. Read those entries as dated snapshots, never as the current
  source state.
- New serve/closure surfaces must emit worker rows with their own
  `identity_binding` plus `leaf_artifacts` refs (path, sha256, size, schema,
  Ed25519 attestation role); the host compares and validates - it never
  injects identity or accepts a bare status boolean.
- `docs/archive/specs/fable-live-run-handoff-2026-07-11.md` is an uncommitted operator
  handoff, not provider proof. It records the sealed-image digest and the
  remaining paid-run setup, while explicitly preserving blocked external
  scorer, semantic-review, and attestation-pin rows.

## 2026-07-10

### User-Facing

- Hardened release and launch-quality contracts across buyer claim ceilings,
  paid-spend admission, release-evidence retention, supply-chain/source
  governance, runtime secret handling, and critical-capability CI
  (`src/blueprint_pipeline/buyer_package_readout.py`,
  `docs/release_evidence_requirements.json`,
  `docs/source_governance_policy.json`, `.github/workflows/ci.yml`,
  `.github/workflows/critical-capability-lanes.yml`). These are fail-closed
  repository controls and evidence requirements, not proof of a live
  deployment, public launch, paid provider completion, or robot task success.
- Added feedback-driven G1 stance search and strict graded episode
  trace-consistency scoring to the Isaac parity lane
  (`src/blueprint_pipeline/stance_configuration_agent.py`,
  `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`,
  `scripts/run_isaac_g1_kitchen_parity_eval.py`). A configured or locally
  graded episode remains separate from live simulator articulation, calibrated
  external consistency scoring, and physical-robot validation.

### Employee-Facing

- Merged Isaac startup-reliability controls: atomic canary-to-worker
  supervision, durable machine quarantine, separate fast/review canaries,
  provider phase tracing, spend reconciliation, content-addressed kitchen-asset
  admission, worker-image health checks, and more honest capacity handling
  (`src/blueprint_pipeline/isaac_startup_supervisor.py`,
  `src/blueprint_pipeline/machine_quarantine_registry.py`,
  `src/blueprint_pipeline/provider_phase_trace.py`,
  `src/blueprint_pipeline/startup_spend_reconciliation.py`,
  `src/blueprint_pipeline/kitchen_asset_startup_gate.py`). The merged controls
  improve startup diagnosis and teardown accounting; they do not establish a
  newly published exact-digest worker image or a successful live task episode.
- **Uncommitted local changes:** The July 10 G1 kitchen audit/remediation wave
  added attempt-bound closure and lineage contracts, self-identifying worker
  image evidence, visual-mesh geometry provenance, official GEAR-SONIC
  controller/FK adapters, a persistent Isaac task executor, strict external
  action-consistency scorer contracts, and ordered full-episode review-media
  admission (`docs/archive/point-in-time/G1_KITCHEN_RUN_DEEP_AUDIT_2026-07-10.md`,
  `docs/archive/point-in-time/G1_KITCHEN_RUN_DEEP_AUDIT_REMEDIATION_2026-07-10.md`,
  `src/blueprint_pipeline/g1_kitchen_attempt_closure.py`,
  `src/blueprint_pipeline/gear_sonic_official_zmq_executor.py`,
  `src/blueprint_pipeline/isaac_persistent_task_executor_service.py`,
  `src/blueprint_pipeline/wam_action_consistency_contract.py`,
  `src/blueprint_pipeline/isaac_review_media.py`). This is an mtime-bounded
  dirty-tree snapshot and is not merged history.

### Future-Agent-Facing

- The committed July 10 America/Chicago window runs from `95a11a0f` through
  merge `d1220f78`; the main feature merges are PR #66 for stance/strict trace
  consistency and PR #67 for Isaac startup reliability. Preserve unrelated
  dirty-tree work when continuing from this snapshot.
- The audit's honest lane status remains
  `local_contracts_advanced_live_end_to_end_task_success_not_proven`. The
  strongest cited live result is an RTX A6000 startup/pixel canary; fresh
  exact-digest canaries, live visual-mesh reach, official controller execution,
  persistent articulation success, calibrated external scoring, and accepted
  full-episode review media remain unproven. Physical validation is a separate
  future claim-upgrade lane, not a sim-only closure blocker.
- Keep raw capture/provenance and rights/privacy evidence authoritative.
  Startup markers, image manifests, generated media, semantic-review outputs,
  local consistency labels, readiness ledgers, and prepared bundles are
  downstream support artifacts and cannot substitute for their corresponding
  live runtime or semantic-success proof layers.

## 2026-07-09

### User-Facing

- Published fail-closed beta controls for data residency/transfer, retention,
  secret-artifact disclosure, output retention, operator incident response, and
  capture-root site coverage (`docs/BETA_DATA_RESIDENCY_TRANSFER_POLICY_2026-07-09.md`,
  `docs/BETA_DATA_RETENTION_POLICY_2026-07-09.md`,
  `docs/SECRET_ARTIFACT_DISCLOSURE_POLICY.md`,
  `docs/runbooks/output-artifact-retention.md`,
  `docs/runbooks/beta-ops-incident-response.md`,
  `scripts/validate_capture_root_by_site_coverage.py`). These are policy and
  validation contracts, not proof that production retention, residency,
  incident response, or site coverage has been exercised live.
- Published stricter buyer/readiness boundaries across buyer package readouts,
  Arena result ingestion, external/paid launch gates, and launch-readiness
  packets (`src/blueprint_pipeline/buyer_package_readout.py`,
  `src/blueprint_pipeline/arena_result_ingest.py`,
  `scripts/run_external_alpha_launch_gate.py`,
  `scripts/run_paid_marketplace_launch_gate.py`,
  `scripts/build_launch_readiness_packet.py`). Prepared packages and passing
  local checks remain separate from live provider execution, semantic task
  success, public readiness, and deployment approval.

### Employee-Facing

- Published provider/runtime hardening for model-access secrets, object-store
  handoff, Vast/RunPod staging, worker-image startup, and Unitree/WAM readiness
  (`src/blueprint_pipeline/secret_artifact_policy.py`,
  `src/blueprint_pipeline/model_access_env.py`,
  `src/blueprint_pipeline/wam_provider_object_store.py`,
  `src/blueprint_pipeline/vast_bundle_staging.py`,
  `src/blueprint_pipeline/runpod_wam_async_runner.py`,
  `src/blueprint_pipeline/wam_model_runtime_bootstrap.py`). These changes make
  execution fail closed and improve artifact handling; they do not establish a
  successful paid run, useful generated output, or physical-robot result.
- Added CI Python-interpreter contract checks, expanded full/sim-only lanes,
  beta capacity/backup validation, and deploy/Terraform updates
  (`docs/CI_PYTHON_INTERPRETER_MATRIX.md`,
  `scripts/validate_python_interpreter_matrix.py`,
  `.github/workflows/full-test-lane.yml`,
  `.github/workflows/sim-only-local-gate.yml`,
  `scripts/validate_beta_capacity_storage.py`,
  `scripts/validate_capture_truth_backup_policy.py`,
  `deploy/terraform/main.tf`). Repository wiring and validation artifacts are
  not evidence that current CI, infrastructure, backups, or deployment are
  healthy in production.
- **Uncommitted local changes:** Late-July-9 work added release-evidence,
  supply-chain/security, paid-spend admission, transactional output, warm-render
  brokering, public scientific-claim linting, and PTDP scalability surfaces
  (`scripts/build_release_evidence_bundle.py`,
  `scripts/build_supply_chain_evidence.py`,
  `src/blueprint_pipeline/spend_admission_lock.py`,
  `src/blueprint_pipeline/output_run_transaction.py`,
  `src/blueprint_pipeline/warm_render_broker.py`,
  `scripts/lint_public_scientific_claims.py`,
  `scripts/benchmark_ptdp_scalability.py`). This is a file-mtime-bounded local
  snapshot and remains unmerged; related post-midnight July 10 edits are not
  included in this dated entry.

### Future-Agent-Facing

- The committed evidence window contains `cfe742adc` and `7a462e070` on July 9
  America/Chicago. The first commit publishes the July 8 remediation snapshot;
  the second adds the July 9 beta-hardening contracts and focused regressions.
- Keep raw capture/provenance and rights/privacy evidence authoritative. Policy
  documents, readiness packets, CI matrices, generated summaries, provider
  manifests, and release-evidence graphs are downstream support artifacts and
  must not be promoted into simulator fidelity, live provider/runtime success,
  public launch readiness, deployment approval, or robot task-success proof.

## 2026-07-08

### User-Facing

- **Uncommitted local changes:** Added buyer-claim-ceiling safeguards that pin
  task-evaluation and sim-only beta release language to the highest truthful
  success claim and block buyer-facing copy that asserts live simulator or live
  policy execution without matching gates
  (`src/blueprint_pipeline/buyer_claim_ceiling.py`,
  `src/blueprint_pipeline/task_eval_run_report.py`,
  `src/blueprint_pipeline/live_robot_eval_closure.py`,
  `scripts/run_sim_only_beta_release_gate.py`). This is a copy/proof-boundary
  gate, not live runtime or task-success proof.
- **Uncommitted local changes:** Added industrial-site launch controls for
  warehouse/manufacturing/fulfillment/factory/brownfield captures, including
  legal/EHS authorization evidence, worker-PII/proprietary-data posture, PPE
  and escort acknowledgements, restricted-zone controls, and stricter
  industrial privacy redaction requirements
  (`src/blueprint_pipeline/alpha_readiness.py`,
  `src/blueprint_pipeline/proof_contracts.py`,
  `docs/operator_launch_evidence.template.json`,
  `docs/PAID_MARKETPLACE_BETA_LAUNCH_GATE.md`).
- **Uncommitted local changes:** Added a 100-user beta capacity/storage model,
  primary bucket lifecycle policy, dry-run soak harness, and capture-truth
  backup/DR runbook/scripts
  (`docs/BETA_CAPACITY_COST_STORAGE_MODEL_2026-07-08.md`,
  `docs/beta_capacity_cost_storage_model_2026-07-08.json`,
  `deploy/storage/primary-capture-bucket-lifecycle.json`,
  `scripts/validate_beta_capacity_storage.py`,
  `scripts/run_beta_intake_soak_test.py`,
  `docs/CAPTURE_TRUTH_BACKUP_DR_RUNBOOK_2026-07-08.md`,
  `scripts/apply_capture_truth_backup_policy.sh`). These artifacts document
  and validate intended controls; they do not prove the real production bucket,
  Firestore backup, restore drill, or live soak run was completed.

### Employee-Facing

- **Uncommitted local changes:** Extended Arena/PTDP delivery from local-copy
  only toward GCS-backed package artifacts and WebApp entitlement patch inputs
  while keeping signed access and entitlement verification separate
  (`src/blueprint_pipeline/arena_package_delivery_local.py`,
  `src/blueprint_pipeline/arena_result_ingest.py`,
  `src/blueprint_pipeline/webapp_sync.py`).
- **Uncommitted local changes:** Hardened provider reliability and spend-control
  surfaces: provider-reliability manifests now name ordered phases and first
  failing phase, Lambda termination verifies terminal state through the
  provider API, provider-race launch uses a fixed adapter registry and only
  selects a winner after a fresh job-bound terminal artifact plus verified
  teardown, render pods carry hard/idle ownership leases, and
  `scripts/gpu_spend_guard.py` fails closed on provider-inventory uncertainty
  while persisting atomic spend-ledger and fleet-budget reports
  (`docs/PROVIDER_RELIABILITY_MANIFEST.md`,
  `src/blueprint_pipeline/lambda_provider_adapter.py`,
  `src/blueprint_pipeline/robot_eval_provider_race_launcher.py`,
  `src/blueprint_pipeline/isaac_particlefield_render_job.py`,
  `scripts/gpu_spend_guard.py`).
- **Uncommitted local changes:** Added industrial task-grounding support and a
  minimal warehouse fixture for containment/zone-arrival proxy checks, plus
  capture-batch registry quarantine/dead-letter behavior so one malformed
  capture does not abort the whole registry update
  (`src/blueprint_pipeline/eval_ready_task_grounding.py`,
  `tests/fixtures/warehouse_task_min/`,
  `src/blueprint_pipeline/capture_batch_registry.py`).

### Future-Agent-Facing

- Treat this entry as a snapshot of uncommitted local work attributable to
  2026-07-08 by file modification time, not as merged `main` history. The only
  commit dated 2026-07-08 in the review window was the prior changelog commit
  (`2e7abbf03`) for July 7.
- Keep proof layers separate: buyer copy ceilings, industrial/EHS evidence
  templates, storage lifecycle docs, backup scripts, GCS delivery URIs, provider
  reliability ledgers, and spend-guard snapshots are support contracts. They do
  not by themselves prove public beta readiness, live provider execution,
  generated-world rank fidelity, physical robot readiness, backup readiness, or
  deployment approval.

## 2026-07-07

### User-Facing

- Closed several no-spend launch-readiness blockers with generated
  readiness-packet support, sim-only beta local-gate fixtures/CI, live
  Pipeline intake and Pub/Sub handoff deployment assets, production runtime
  environment guards, and buyer/package truth gates
  (`scripts/build_launch_readiness_packet.py`,
  `scripts/run_sim_only_beta_local_gate.py`,
  `src/blueprint_pipeline/live_pipeline_intake_service.py`,
  `src/blueprint_pipeline/pubsub_handoff_listener.py`,
  `src/blueprint_pipeline/production_runtime_env_guard.py`). These changes
  improve launch evidence collection and fail-closed gating; they do not by
  themselves prove public launch readiness, paid provider completion,
  physical-robot readiness, or deployment approval.
- Hardened launch/readiness proof boundaries. Readiness packets now record
  canonical repo heads, reject unclean or blocked evidence repos, require fresh
  CI and sim-only evidence, consume operator/legal evidence explicitly, require
  probed non-local WebApp forwarding proof, and reject stale proof artifacts
  (`scripts/build_launch_readiness_packet.py`,
  `scripts/collect_github_actions_evidence.py`,
  `src/blueprint_pipeline/source_metadata.py`). Local forwarding and stale
  generated artifacts remain blockers rather than launch proof.
- Added buyer artifact access gating for paid marketplace beta checks and
  WebApp sync: buyer readout status and signed access-contract checks now feed
  readiness rather than being assumed from package existence
  (`scripts/run_paid_marketplace_launch_gate.py`,
  `src/blueprint_pipeline/webapp_sync.py`,
  `docs/PAID_MARKETPLACE_BETA_LAUNCH_GATE.md`).
- Added and hardened a DigitalOcean GR00T/OSCAR closed-loop launcher with
  region/size capacity checks, direct launch planning, sealed image contracts,
  and hardware pre-spend preflight
  (`src/blueprint_pipeline/groot_oscar_digitalocean_closed_loop_job.py`,
  `src/blueprint_pipeline/gpu_render_providers.py`,
  `docs/runbooks/groot-oscar-closed-loop-sealed-image.md`). Prepared/local
  launch plans are not live DigitalOcean runtime, artifact quality, or task
  success proof.
- Converted the July 6 T4 quality post-mortem into fail-closed contracts:
  per-lane GPU hardware floors, native OSCAR generation-resolution checks, and
  generated-clip coherence checks now block under-provisioned or degraded WAM
  runs before stronger claims can be made
  (`src/blueprint_pipeline/lane_hardware_requirements.py`,
  `src/blueprint_pipeline/oscar_isaac_closed_loop_eval.py`,
  `src/blueprint_pipeline/provider_reliability_manifest.py`).

### Employee-Facing

- Added full-lane and sim-only GitHub Actions workflows plus launch-gate
  helpers for external alpha, paid marketplace, and sim-only beta local checks
  (`.github/workflows/full-test-lane.yml`,
  `.github/workflows/sim-only-local-gate.yml`,
  `scripts/run_external_alpha_launch_gate.py`,
  `scripts/run_paid_marketplace_launch_gate.py`).
- Added live-control-plane/intake deployment assets and validation contracts:
  systemd units/timers, install/postcheck scripts, Terraform handoff wiring,
  intake-service freshness checks, manifest alerts, and local agent-provider
  support (`deploy/systemd/*`, `deploy/scripts/deploy.sh`,
  `deploy/terraform/main.tf`,
  `src/blueprint_pipeline/live_pipeline_control_plane.py`,
  `src/blueprint_pipeline/live_pipeline_manifest_alert.py`,
  `src/blueprint_pipeline/agent_runtime/providers/local.py`).
- Added `robot_eval_job_request.v1` schema verification and tighter
  multi-root robot-eval inbox processing coverage
  (`src/blueprint_pipeline/robot_eval_job_request_contract.py`,
  `scripts/verify_robot_eval_job_request_contract.py`,
  `tests/test_robot_eval_job_request_contract.py`,
  `tests/test_robot_eval_job_orchestrator.py`).
- Hardened default task and environment fallbacks: eval-ready grounding derives
  targets from site objects instead of falling back to legacy sink tasks,
  manipulation targets fail closed when defaults are unsupported, live setup can
  honor explicit environment fallbacks, and object-index default-environment
  fallback is surfaced (`src/blueprint_pipeline/eval_ready_task_grounding.py`,
  `src/blueprint_pipeline/manipulation_task_stack.py`,
  `src/blueprint_pipeline/live_pipeline_setup.py`,
  `src/blueprint_pipeline/object_index_stage.py`).
- Added focused regression coverage across launch packets, paid/external gates,
  WebApp sync, live pipeline setup/intake/control plane, DigitalOcean
  closed-loop jobs, hardware requirements, sim-only local gates, deploy systemd
  contracts, and manipulation/default-target edges.

### Future-Agent-Facing

- Treat July 7 as launch-readiness automation and proof-boundary hardening, not
  as final launch approval. Stronger claims still require current CI artifacts,
  probed production forwarding, operator/legal evidence, buyer access proof,
  live provider/runtime artifacts, cost/teardown closure, and task-specific
  evaluation proof as applicable.
- Keep prepared DigitalOcean plans, WAM coherence checks, and sealed-image
  manifests separate from live provider execution. They can block bad paid runs
  before spend, but they do not prove useful generated video, simulator task
  success, physical robot readiness, or deployment readiness without accepted
  run artifacts.
- Evidence boundary: this entry covers committed history dated 2026-07-07 from
  `1c6175882` through `e0bb4ae5b`. `git status --short` was clean during this
  changelog run, so no uncommitted July 7 local changes were included.

## 2026-07-06

### User-Facing

- Added real policy-family evaluation support for LeRobot-format checkpoints,
  ACT torch inference, and GR00T/SONIC endpoint integration
  (`src/blueprint_pipeline/lerobot_policy_family.py`,
  `src/blueprint_pipeline/lerobot_torch_policy_adapter.py`,
  `src/blueprint_pipeline/real_policy_family_eval_harness.py`,
  `src/blueprint_pipeline/real_policy_closed_loop_rollout.py`). The lane
  records simulator framework, evaluation substrate, SC3-derived measurements,
  visual-media coverage, and validation-ladder registration without promoting
  outputs to production-candidate, live-robot, deployment, or field-success
  proof.
- Hardened buyer and delivery truth boundaries: live consent is re-read at
  emit/copy time for webapp status projections, task-eval reports, robot-eval
  datasets, arena package delivery, qualification, and alpha readiness
  (`src/blueprint_pipeline/task_eval_run_report.py`,
  `src/blueprint_pipeline/robot_eval_dataset.py`,
  `src/blueprint_pipeline/arena_package_delivery_local.py`,
  `src/blueprint_pipeline/qualification.py`,
  `src/blueprint_pipeline/alpha_readiness.py`). Task-eval scorecards now
  withhold success rates when review-task success is insufficient instead of
  converting media validity into a task-success claim.
- Added a local Scaniverse support-asset import lane
  (`src/blueprint_pipeline/scaniverse_asset_import.py`,
  `docs/SCANIVERSE_ASSET_IMPORT.md`) and CLI entrypoint
  `blueprint-import-scaniverse-assets`. It stages and checksums Scaniverse
  USDZ/PLY/SPZ/mesh/USD exports under a capture root with an explicit proof
  boundary: these assets are optional downstream support artifacts, not raw
  Blueprint capture truth or simulator/task-success proof.
- Added `docs/100_BETA_TESTER_LAUNCH_BLOCKER_AUDIT_2026-07-06.md`, which says
  the service should not launch to 100 external beta testers yet. The audit
  records local pytest success but blocks external beta on paid marketplace
  gate failure, missing WebApp->Pipeline forwarding configuration/probe,
  missing current sim-only gate artifacts, incomplete real-policy closeout,
  blocked WAM real-provider validation, no paid provider canary proof, no
  live-robot/device/money/payout/legal/KYC readiness proof, and broad ruff
  failures.

### Employee-Facing

- Added sealed GR00T x OSCAR worker-image tooling:
  `deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile`,
  image healthcheck/requirements files, `blueprint-groot-oscar-closed-loop-image`,
  `scripts/build_push_groot_oscar_closed_loop_image.sh`,
  `scripts/snapshot_groot_oscar_eval_pod.sh`, and
  `docs/runbooks/groot-oscar-closed-loop-sealed-image.md`. A follow-up fix
  corrected the mutually-exclusive CLI default for
  `--print-sealed-contract`.
- Extended `oscar_isaac_closed_loop_eval.py` with GR00T/SONIC policy-server
  wiring, skeleton-conditioning video support, generated-video success labels,
  and separate episode-consistency artifacts. WAM rollout execution, visual
  success labels, and forward/inverse consistency remain separate evidence
  layers; WAM execution alone does not claim consistency.
- Strengthened runtime/CI contracts: provider launchers and
  `runpod_wam_async_runner.py` gained harder failure handling, CI video-codec
  checks now fail rather than silently skipping, and the command-safety matrix
  includes the Scaniverse import CLI.
- Added buyer/PTDP and policy-eval regression coverage across
  `tests/test_lerobot_policy_family.py`,
  `tests/test_lerobot_torch_policy_adapter.py`,
  `tests/test_real_policy_family_eval_harness.py`,
  `tests/test_post_training_data_package.py`,
  `tests/test_scaniverse_asset_import.py`,
  `tests/test_consent_gate_prod_wiring.py`,
  `tests/test_orchestrator_consent_toctou.py`,
  `tests/test_task_eval_run_report.py`, and
  `tests/test_video_codec.py`.

### Future-Agent-Facing

- Keep the July 6 real-policy stack validation-ladder-only unless a later run
  supplies current live simulator execution, live policy execution, full trace
  package, task metrics, WebApp lineage, delivery access, and closure audit
  evidence. Current artifacts do not prove live robot readiness, deployment
  approval, public beta readiness, or semantic field success.
- Treat Scaniverse imports as provider-derived support assets behind a
  replaceable boundary. Do not let imported Scaniverse assets override raw
  Blueprint capture/provenance authority, and do not claim simulator load,
  collision/contact/scale validation, policy execution, or task success without
  separate owner-system proof.
- Evidence boundary: this entry covers four commits dated 2026-07-06
  (`201e2531a`, `30488efb3`, `ff82bb5db`, and `64edcc8e6`) and no
  uncommitted local changes; the working tree was clean during this changelog
  run.

## 2026-07-05

### User-Facing

- PTDP real-data-fraction floor + synthesized-state honesty gate: the
  lerobot_v3 / gr00t_lerobot training exports now compute
  `real_state_fraction` / `real_action_fraction` (measured vs
  zero-fill-synthesized `observation.state` and fallback-synthesized action
  rows), per episode and package-wide. Each export manifest carries a
  `state_action_provenance` block (fractions, configurable floor via
  `BLUEPRINT_PTDP_MEASURED_STATE_FRACTION_FLOOR`, default 0.5, per-episode
  provenance counts so a buyer can filter episodes); below the floor the
  export downgrades to `written_degraded` with
  `insufficient_measured_state_fraction` and the buyer readout's
  robot-POV-evidence section blocks
  (`insufficient_measured_state_fraction:<format>`; a claimed lerobot export
  with no provenance report fails closed as
  `measured_state_fraction_unknown:<format>`). Frame rows gain an
  `action_synthesized_fallback` column alongside
  `state_synthesized_zero_fill`, and the package manifest surfaces the
  fractions + floor verdict in `export_policy` and `claim_boundary`
  (`measured_state_fraction_floor_passed`). A fully-measured package passes
  with fractions = 1.0.

## 2026-07-04

### User-Facing

- Added the buyer package readout
  (`src/blueprint_pipeline/buyer_package_readout.py`, schema
  `buyer_package_readout.v1`): every Post-Training Data Package export now
  writes `buyer_package_readout.json` + `buyer_package_summary.md`, a
  fail-closed summary across nine buyer-critical sections (cards,
  rights/privacy/provenance, robot POV evidence, failure evidence, task
  success criteria, calibration, media provenance, export integrity,
  replay/review instructions). Missing sections block the readout even when
  the pipeline export itself is ready; the claim boundary echoes the
  success-claim ledger and can never invent a higher claim. Exports also ship
  `replay_review_instructions.md` (verify → review → replay protocol), and
  `docs/BUYER_PACKAGE_TRUST_GUIDE_2026-07-04.md` documents the deliverable for
  robot-team buyers.
- Overclaim fixes across sellable surfaces: `post_training_data_package`
  export_policy RL flags (`rl_sparse_reward_signal_included`, concurrent A/B,
  bottleneck, speed curriculum, action-chunk QA, safety ledger) are now derived
  from actual handoff content instead of hardcoded `True`;
  `policy_improvement_run` downgrades
  `improvement_candidate_ready_for_customer_review` to
  `blocked_improvement_claim_unsupported` when the heldout delta is missing or
  non-positive or the concurrent-A/B claim is not allowed;
  `evaluation_prep_stage` proven flags now require strict booleans (proof
  boundary authoritative both directions, truthy strings never count);
  WebApp sync projections label every task success rate with its evaluation
  substrate, list evidence manifests behind each proven flag, carry
  fail-closed rights/privacy status, mark `evaluation_readiness` advisory
  only, and expose optional `product_handoff` (SKU/entitlement/review URL)
  wiring without gating evidence. Robot POV evidence requirements now include
  a camera metadata contract (intrinsics, extrinsics, calibration status;
  uncalibrated footage supports review-grade labels only).

- Added the provider reliability manifest
  (`src/blueprint_pipeline/provider_reliability_manifest.py`,
  `docs/PROVIDER_RELIABILITY_MANIFEST.md`): one fail-closed
  `provider_reliability_manifest.v1` JSON per paid GPU run recording the exact
  failed phase and blocker across pre-spend preflight, provider launch,
  container startup, runtime execution, artifact collection, artifact quality,
  task evaluation, and teardown — with pre-spend preflight (capacity, pinned
  image, marker/timeout runtime contract, credentials, spend gate), a
  post-marker stall policy, teardown proof that requires a provider-reported
  terminal state (RunPod STOPPED is not terminal), and stale-artifact-rejecting
  collection contracts. Infrastructure phases never imply artifact quality or
  task success.
- `isaac_particlefield_render_job` paid runs now write
  `provider_reliability_manifest.json` on every attempt (including
  fail-before-spend capacity/credential blocks), enable the post-marker
  no-progress watchdog by default
  (`BLUEPRINT_POST_MARKER_NO_PROGRESS_TIMEOUT_SECONDS`, 900s), and record
  keep-alive teardowns as open billing risk instead of silence.
- Local MP4 repair in the Isaac G1 kitchen parity job now checks the run's
  expected frame count: a repair over a partial provider upload is labeled
  `repaired_truncated` with blocker `mp4_repair_truncated_frames:*` instead of
  `repaired`, so locally assembled review videos can no longer mask truncated
  provider renders.
- `scripts/gpu_spend_guard.py` gained `--json-report <path>`: a persisted
  `gpu_spend_guard.v1` snapshot of live allocations, burn rate, protected ids,
  reap candidates, and reap results, giving ops durable teardown evidence
  instead of stdout-only reports.

- Added layered, fail-closed success-claim contracts
  (`src/blueprint_pipeline/success_claim_contracts.py`) separating media
  validity, review task success, the task success contract, simulator/runtime
  execution, policy/action execution, contact/state-change proof, and
  physical/deployment readiness into independent fields with their own
  blockers. A composed ledger reports the highest truthful claim; a higher
  claim can never be asserted while a lower layer is unproven.
- Closed audited false-positive success paths: provider runtime success no
  longer reads as task success, media validity no longer makes a
  generated-video success label authoritative, status strings and stringly
  typed verdicts no longer coerce to task success, visible arm presence no
  longer satisfies reach-required tasks, and stale artifacts no longer count
  as current-run truth without freshness evidence.
- Closed the July 3 Pipeline beta-remediation items for capture handoff wiring
  and rights/privacy fail-closed behavior in committed history
  (`docs/beta-launch-audit-2026-07-03/REMEDIATION-STATUS.md`), including
  Pub/Sub storage-trigger handoff validation and PIPE-01/02/03/04/05/06
  remediation markers. This records blocker closure in source/tests; it is not
  external beta readiness, buyer delivery proof, or live provider proof.
- Uncommitted July 4 work adds further paid-lane, consent-revocation,
  LeRobot-export, WAM-score, provider-race, buyer-readout, PTDP, WebApp sync,
  and run-e2e hardening. Treat those files as local work until committed; they
  improve fail-closed package/runtime contracts but do not prove live paid runs,
  deployment readiness, physical robot readiness, or task success.

### Employee-Facing

- Isaac/G1 kitchen parity runner and job now attach a per-scenario
  `success_claim_ledger` plus a result-level `success_claim_summary`; the
  Stage A kinematic lane fails the policy-execution layer closed
  (`action_source_not_policy:kinematic_preview_controller`) and a scenario
  that declares `success_state_change` metadata withholds simulator-level
  task claims until a measured state change exists.
- `oscar_cosmos_wam_evaluator._normalize_wam_success_labels` requires strict
  boolean reviewer verdicts (`wam_success_label_verdict_not_strict_boolean`
  blocker otherwise) and computes `authoritative_task_success_label` from
  media validity AND verdict, never media validity alone.
- `wam_fixture_evaluator` re-derives `review_grade_success_label` from its
  gates instead of passing the upstream field through, and rejects
  non-boolean `task_success` label values.
- `runpod_wam_async_runner` splits `provider_runtime_operational` from
  `runtime_task_success` (strict boolean from the runtime result only) in the
  poll manifest.
- `robot_eval_execution` and `isaac_g1_site_3dgs_realistic_eval` fail closed
  with `task_success_not_reported_failing_closed` when an episode completes
  without an explicit boolean verdict.
- `proof_contracts.build_site_package_manifest` blocks on
  `launchable_export_not_ready` / `site_world_runtime_not_launchable`;
  `evaluation_prep_stage` proven-flags treat `proof_boundary.json` as
  authoritative over the run manifest; `live_robot_eval_closure` requires
  evidence refs behind `robot_policy_execution_proven`
  (`policy_execution_proof_flag_without_evidence_refs`).
- Regression tests in `tests/test_success_claim_contracts.py` (98 tests)
  parametrize over the real faucet/stovetop/microwave/sink task artifacts
  under `output/kitchen_task_scaling_preflight_*` when present and skip
  hermetically when absent. Requirements are derived from task contract
  metadata (affordance ids, declared `success_state_change`), never task-id
  string matching.
- `scripts/pytest_fast.sh` now blocks the old false-green path by requiring the
  full no-GPU validation dependencies before running the fast lane, and
  `scripts/pytest_full.sh` provides a full `python -m pytest tests/` wrapper.
  Hermetic kitchen task fixtures under
  `tests/fixtures/kitchen_task_min/` keep claim tests meaningful even when
  local generated `output/` artifacts are absent.
- Committed remediation touched capture handoff infrastructure
  (`deploy/terraform/main.tf`, `functions/storage_trigger.py`,
  `scripts/validate_pubsub_handoff_infra.py`,
  `src/blueprint_pipeline/pubsub_handoff_listener.py`) and Pipeline
  rights/privacy gates (`alpha_readiness.py`, `evaluation_prep_stage.py`,
  `proof_contracts.py`, `qualification.py`) with focused tests.
- Uncommitted July 4 modules include
  `src/blueprint_pipeline/paid_lane_guard.py`,
  `src/blueprint_pipeline/consent_takedown.py`,
  `src/blueprint_pipeline/lerobot_export_validation.py`,
  `src/blueprint_pipeline/wam_score_claim_gate.py`, and
  `src/blueprint_pipeline/robot_eval_provider_race_launcher.py`, plus tests.
  `pyproject.toml` also has an uncommitted
  `blueprint-run-robot-eval-provider-race` CLI entrypoint.

### Future-Agent-Facing

- When adding a new success-claiming surface, emit the layer fields from
  `success_claim_contracts` (or the runner's bundle-safe mirror) instead of a
  bare `success`/`ready` boolean. `physical_deployment_ready` can only come
  from real-robot evidence plus a named approval — no combination of WAM,
  generated-video, review, or simulator evidence upgrades it.
- Tasks that change object state must declare
  `success_state_change: {object, property}` in their task metadata; the
  ledger then withholds simulator/policy task claims until a measured
  before/after change of that property exists.
- Evidence boundary: this entry covers six commits dated 2026-07-04
  (`c47eeea3d`, `376a58139`, `31082785b`, `4f4b0201e`, `f93d97c09`, and
  `19d996359`) plus explicitly labeled uncommitted local changes whose file
  mtimes were also on 2026-07-04. Keep the uncommitted paid-lane/takedown/WAM
  score/provider-race work separate from shipped proof until it is committed
  and validated.

## 2026-07-03

### User-Facing

- Added a cross-repo beta-launch blocker audit under
  `docs/beta-launch-audit-2026-07-03/`, covering capture app wiring,
  capture-to-pipeline handoff, WebApp money/security issues, and Pipeline
  rights/privacy gates. The audit says the external beta path is not ready; it
  is a blocker map, not readiness proof.
- Added shared `VISION.md` strategy framing for the robot-eval wedge, with
  OSCAR and SC3-Eval cited as the scientific backbone for generated-world
  policy-ranking correlation. The document keeps rank fidelity and calibrated
  prediction separate from guaranteed field outcomes, deployment proof, or live
  robot execution.
- Committed Isaac/G1 kitchen-parity, GPU render-provider, GR00T/SONIC provider
  smoke/persistent-session, and WAM generated-video review hardening from the
  in-progress local tree. These improve review/runtime support paths, but they
  remain downstream simulator/provider artifacts unless matching run, artifact,
  cost, teardown, and closure evidence exists.
- Uncommitted July 3 work began closing beta audit findings for capture handoff
  wiring, rights/privacy fail-closed behavior, and WorldLabs preview gating.
  Those changes are not committed yet and should not be treated as shipped.

### Employee-Facing

- Added `docs/beta-launch-audit-2026-07-03/INDEX.md` plus repo-specific specs
  for BlueprintCapturePipeline, BlueprintCapture, Blueprint-WebApp, and
  cross-repo blockers. Stable IDs include `PIPE-01` through `PIPE-06` and
  `XR-01` through `XR-05`.
- Added and then refined `VISION.md` as a shared cross-repo doctrine document,
  including the SC3-Eval `0.929` headline correlation attribution, OSCAR
  RoboArena correlation caveats, and explicit swappable-model proof boundaries.
- Updated committed runtime/review code in
  `scripts/run_isaac_g1_kitchen_parity_eval.py`,
  `src/blueprint_pipeline/gpu_render_providers.py`,
  `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`,
  `src/blueprint_pipeline/isaac_particlefield_render_job.py`,
  `src/blueprint_pipeline/unitree_groot_n17_sonic_provider_smoke.py`,
  `src/blueprint_pipeline/unitree_groot_n17_sonic_vast_persistent_session.py`,
  `src/blueprint_pipeline/wam_generated_video_review.py`, and
  `src/blueprint_pipeline/wam_generated_video_success_label_gemini.py`, with
  focused tests across those surfaces.
- Uncommitted July 3 edits add a dedicated capture-bridge Pub/Sub handoff topic,
  raw-upload-complete handoff publishing, synthesized `pipeline_handoff.json`
  from iOS raw sidecars, rights/privacy launch blockers in evaluation prep and
  proof contracts, delivery-run privacy gating in qualification, and dynamic
  visible-reach episode termination for Isaac/G1 review clips.

### Future-Agent-Facing

- Keep the July 3 beta audit as a decision and blocker artifact. It does not
  certify beta readiness, public readiness, production forwarding, paid-provider
  closure, physical-robot readiness, or buyer delivery.
- Keep `VISION.md` subordinate to `PLATFORM_CONTEXT.md` and
  `WORLD_MODEL_STRATEGY_CONTEXT.md`: OSCAR/SC3-Eval support the evaluation
  strategy, but generated-world correlation is not deployment approval,
  universal grading proof, or guaranteed real-world task success.
- Evidence boundary: this entry covers three commits dated 2026-07-03
  (`b96e85bca`, `cd26ca2c3`, and `ba7968bc4`) plus explicitly labeled
  uncommitted local changes whose file mtimes were also on 2026-07-03.

## 2026-07-02

### User-Facing

- Added 3DGS/InteriorGS scene placement support for labels-free PLY sidecar
  bootstrapping, local depth/composite helpers, robot-only probe passes, and G1
  ParticleField visual compositing. These are source-observation and render
  support paths; they do not prove physical robot execution, contact fidelity,
  task success, or deployment readiness.
- Added SC3 protocol and provider-agnostic robot-eval adapter contracts,
  including closure planning, WAM/scorer separation, and the start of a
  ranker-validation policy ladder. SC3 consistency, WAM execution, and
  generated-video labels remain separate evidence layers.
- Added launch/beta readiness audit specs for geometry truth, clip curation,
  semantic deduplication, action normalization, Cosmos3 WAM adapter work, SC3
  scoring, calibration, temporal alignment, immutable raw captures, enrichment,
  launch gates, CPU safety, and city-launch refresh.
- Hardened capture truth and PTDP quality gates so fabricated geometry
  fallbacks, malformed action data, curation gaps, dedup drift, and absent vs.
  invalid SC3 action payloads are handled more explicitly.
- Added scene-eval auto-generation from a single PLY/USD scene and made corrupt
  or malformed scene files fail closed rather than raising through the caller.
- Expanded paid-provider launch discipline: RunPod offer retry/error capture,
  datacenter RTX pool pinning, DigitalOcean GPU Droplets provider support,
  Lambda runtime handoff hardening, built-in Vast launcher automation, and
  Pub/Sub handoff deployment infrastructure.
- Guarded paid GR00T/SONIC WAM runs behind runtime plus sealed-image proof,
  added sealed WAM image packaging, strict generated-video task-success judging,
  remote-build packet generation, and a no-spend provider-readiness audit.

### Employee-Facing

- Added or extended CLI/script surfaces in `pyproject.toml` and scripts for
  SC3 protocol handling, scene eval autogen, Pub/Sub handoff listening,
  provider launcher automation, sealed GR00T/SONIC WAM image build/push,
  remote-build packets, and provider-readiness audits.
- Added new core modules including `sc3_eval_protocol.py`,
  `scene_eval_autogen.py`, `action_normalization.py`,
  `clip_curation_stage.py`, `semantic_dedup_stage.py`,
  `cosmos3_wam_command_adapter.py`, `policy_ranking_ladder.py`,
  `pubsub_handoff_listener.py`,
  `unitree_groot_sonic_wam_image_remote_build_packet.py`, and
  `unitree_groot_sonic_provider_readiness.py`.
- Updated geometry, retrieval, native runtime, WAM backend/substrate,
  PTDP/export, robot-eval orchestration, provider launch, Isaac/G1 parity, and
  webapp-sync paths with focused tests around the new contracts and blockers.
- Updated docs in `README.md`, `docs/SC3_EVAL_PROTOCOL.md`,
  `docs/WAM_EPISODE_CONSISTENCY_SCORER.md`,
  `docs/FIRST_GPU_E2E_RUNBOOK.md`, `docs/architecture/*`, and
  `docs/specs/launch-audit-2026-07-02/*`.
- Uncommitted local work was present in the working tree around the July 2/3
  boundary, touching Isaac/G1 render quality, provider bootstrap quoting,
  provider smoke/persistent-session checks, WAM generated-video review, and
  related tests. Because several file mtimes are after midnight on July 3, this
  entry labels that work as uncommitted and does not treat it as completed
  July 2 proof.

### Future-Agent-Facing

- Preserve the proof hierarchy: raw capture/provenance evidence remains
  authoritative. The July 2 SC3, Cosmos3, WAM, generated-video, render,
  readiness, and provider artifacts are downstream support/evaluation layers
  unless a separate artifact proves a stronger claim.
- The launch/beta audit specs identify blockers and implementation direction;
  they are not themselves public readiness, deployment approval, safety
  validation, physical-robot readiness, or successful task execution.
- Provider changes improve launch paths, cost controls, image sealing, and
  handoff infrastructure. They do not prove live paid-provider completion
  without matching runtime, artifact upload, spend, teardown, and closure
  evidence.
- Evidence boundary: this entry covers committed history with July 2 committer
  dates from `483bde16` through `006616a3`. The current working tree also has
  uncommitted local changes spanning late July 2 and early July 3; keep those
  separate in any later closeout or push summary.

## 2026-07-01

### User-Facing

- Rebuilt and pinned the reusable OSCAR WAM image to the official
  `oscar-public` source plus the Blueprint TransformerEngine RoPE/Torch-SDPA
  compatibility shim. WAM provider defaults now point at the pinned official
  OSCAR image contract instead of falling back to a generic PyTorch carrier
  image.
- Fixed the immediate visual-collapse path for the G1/fridge OSCAR run. A fresh
  two-step GR00T/SONIC -> OSCAR WAM -> generated-observation -> GR00T/SONIC
  loop completed with visual-quality gate pass, preserved edge structure, and an
  external episode-consistency scorer result.
- Upgraded the G1/fridge action-conditioning bridge from hand-drawn/projected
  screen axes to sidecar kinematic-chain FK over Isaac seed arm-link landmarks
  where those sidecars exist. This is still not full G1 URDF FK, official
  WholeBodyControl execution, physical robot proof, contact validation, or task
  success proof.
- Added kitchen task scaling preflight and G1 render-noise audit support for
  Isaac/G1 fridge review media. The audit now separates texture asset
  resolution, render sample budget, denoiser behavior, material response,
  lighting, and camera/pose issues before WAM seed frames are treated as useful
  support artifacts.
- Promoted provider startup paths with stronger RunPod/Lambda/live-proof
  handling, warm render server behavior, GPU startup manifests, and no-spend or
  dry-run modes where applicable. Provider launch, endpoint readiness,
  simulator execution, artifact upload, cost/teardown closure, safety, and
  rank-fidelity proof remain separate claims.
- Fixed the headless MuJoCo `--skip-render-frames` Linux path so GL-less runners
  default `MUJOCO_GL=disable` for non-rendering simulator commands while keeping
  EGL for actual render-frame runs.
- Raised path-traced manipulation/verify review defaults after the first G1
  render-noise audit diagnosed 64-spp sample starvation and clean 384-spp
  variants. The stock Isaac G1 asset still had no texture asset references, so
  textured outputs must remain labeled `textured_unverified` unless a future
  asset resolves real texture refs.

### Employee-Facing

- Added explicit Isaac scene sidecar routing, WAM edge-structure collapse
  detection, OpenAI episode-consistency scoring, SC3-style distinct-view guards,
  rank-fidelity calibration requirements, and sidecar FK metadata propagation
  across the persistent Unitree GR00T N1.7 SONIC / OSCAR WAM path.
- Added local and OpenAI WAM episode-consistency scorer entrypoints, rank
  fidelity anchor requests/calibration reports, and the accepted-anchor
  computation path in
  `src/blueprint_pipeline/unitree_groot_n17_sonic_vast_persistent_session.py`.
- Added contract tests for official-image defaults, OSCAR provider bundle
  diagnostics, sidecar FK skeleton traces, external episode consistency, visual
  quality blockers, multiview unavailability, and calibration guardrails.
- Added `src/blueprint_pipeline/kitchen_task_scaling_preflight.py`,
  `src/blueprint_pipeline/g1_render_noise_audit.py`, and
  `scripts/run_g1_render_noise_audit.py`, with runner/job coverage for the
  kitchen scaling and render-noise variant matrix.
- Hardened provider startup and kitchen parity flow across
  `src/blueprint_pipeline/robot_eval_gpu_startup_pipeline.py`,
  `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`,
  `src/blueprint_pipeline/lambda_provider_adapter.py`,
  `src/blueprint_pipeline/runpod_provider_adapter.py`,
  `src/blueprint_pipeline/runpod_live_execution_proof.py`,
  `src/blueprint_pipeline/gpu_render_providers.py`, and
  `src/blueprint_pipeline/warm_render_server.py`.
- Wired `lambda_cloud` into managed-provider priority, live-provider gate
  metadata, provider credential contracts, and focused Lambda adapter coverage.
- Added official SAM3 depth harness provider support in
  `src/blueprint_pipeline/wam_real_provider_validation_probe.py` and hardened
  related test fixtures for SAM3, host `ffmpeg`, placement yaw validation, and
  dual-stream skeleton visibility counters.

### Future-Agent-Facing

- Treat the July 1 G1/fridge WAM proof as evaluator-bounded visual-review
  evidence only. It does not prove generated-world rank fidelity or real-world
  rank correlation until an accepted prediction-vs-actual calibration anchor set
  exists.
- If reusing RunPod pods, verify image compatibility first. Older hot pods may
  have been launched from the PyTorch carrier image; current WAM defaults expect
  the pinned official OSCAR image.
- The G1 render-noise audit is a simulator/render-quality diagnostic only. Its
  proxy, simplified-diffuse, and `textured_unverified` labels can gate WAM seed
  media choices, but they do not prove physical robot readiness, task success,
  contact correctness, policy quality, verified G1 material fidelity, or WAM
  rank fidelity.
- The Linux MuJoCo fix is scoped to GL selection for packaged simulator
  commands. It fixes GL-less `--skip-render-frames` execution and does not add
  new render-frame, physics-fidelity, provider-runtime, or deployment proof.
- Evidence boundary: this entry covers committed history dated 2026-07-01 from
  `681dd698` through `38771dc3`. Current working tree inspection found no
  uncommitted local changes to attribute to July 1.

## 2026-06-30

### User-Facing

- Hardened the learned-WAM/OSCAR review lane around real future-frame
  materialization, visual-success labeling, and materialization blockers. WAM
  rollouts now fail more explicitly when they fall back to frame zero, degraded
  future frames, or incomplete OSCAR input materialization instead of presenting
  those artifacts as useful generated-video success.
- Added clearer WAM input-review, projected-skeleton, SONIC action-bridge, and
  episode-consistency contracts for Unitree GR00T N1.7 SONIC / OSCAR loops.
  These contracts can support evaluator-bounded policy comparison and external
  consistency scoring, but they do not prove task success, physical-robot
  readiness, safety validation, deployment approval, or raw capture truth.
- Improved paid RunPod WAM lifecycle handling with stronger polling, completed
  persistent-session finalization, dynamic stopped-pod reuse, an explicit stop
  command, and hot-pod retention after successful runs. This improves spend and
  reuse discipline; it is still provider-runtime scaffolding unless the matching
  output, upload, cost, teardown, and visual-quality artifacts exist.
- Added a Lambda provider adapter stub as a second managed-provider lane behind
  the same provider boundary. It is a launch/readiness integration surface, not
  proof that Lambda-hosted runtime execution has occurred.

### Employee-Facing

- Added official OSCAR release and runtime compatibility hardening across
  `src/blueprint_pipeline/oscar_official_release.py`,
  `src/blueprint_pipeline/oscar_wam_provider_bundle.py`,
  `src/blueprint_pipeline/oscar_wam_command_adapter.py`,
  `src/blueprint_pipeline/oscar_wam_gpu_image.py`, and
  `src/blueprint_pipeline/wam_compute_providers.py`, including the official
  source/checkpoint pin contract, TransformerEngine RoPE compatibility, visual
  metrics/review contracts, and DeepInfra/Cosmos API-first WAM adapter work.
- Extended WAM/policy loop contracts in
  `src/blueprint_pipeline/unitree_groot_n17_sonic_vast_persistent_session.py`,
  `src/blueprint_pipeline/oscar_isaac_closed_loop_eval.py`, and
  `src/blueprint_pipeline/oscar_cosmos_wam_evaluator.py` so skeleton traces,
  bridgeable SONIC action chunks, action-conditioning risk summaries, and
  forward/inverse consistency requests stay separate from WAM execution itself.
- Hardened CPU-only and capture-core surfaces: `live_pipeline_control_plane`
  dropped dead next-input flags and gained import-isolation/static guards;
  `materialization.py` and `scene_semantics.py` were refactored without public
  shape changes; privacy runner HTTP/fail-closed edges, object-index detection,
  task-target grounding, agent-review threshold constants, and scene-placement
  lint/clearance regressions gained focused coverage.
- Updated docs and command surfaces in `README.md`,
  `docs/WAM_POLICY_EVALUATION_SERVICE.md`,
  `docs/OSCAR_VISUAL_AUGMENTATION_PACKET.md`,
  `docs/PRIVACY_RUNNER_SERVICES.md`,
  `docs/architecture/command-safety-matrix.md`, and `pyproject.toml`.

### Future-Agent-Facing

- Contract changes: learned OSCAR WAM claims now depend on the official
  `oscar-public` source/checkpoint/image contract and visual-smoke/import
  evidence. The repo-local OSCAR-style generator remains support/test plumbing
  and must not be cited as a learned OSCAR checkpoint run.
- Runtime behavior changes: RunPod WAM runners can poll/finalize more carefully,
  reuse stopped pods, keep successful pods hot, and stop pods explicitly. Treat
  those as lifecycle controls, not provider-output proof without the run
  artifacts and spend/teardown evidence.
- Launch/readiness caveat: WAM materialization, skeleton conditioning, visual
  labels, and official OSCAR compatibility checks are downstream support
  artifacts. They do not override raw capture/provenance evidence and do not
  prove live robot execution, physical contact, manipulation success, safety,
  deployment, or generated-world rank fidelity.
- Evidence boundary: this entry covers committed history dated 2026-06-30 from
  `208c2a7bdd1f16d832d790733027c407d80ac67d` through
  `d7bf8c5ee344046fe49fda2a2ab7691d832ad45c`. Current working tree inspection
  found no uncommitted local changes to attribute to June 30.
- Recorded commit-body verification for the final June 30 OSCAR/WAM checkpoint:
  `python -m ruff check changed source/test files` and `python -m pytest` over
  the RunPod WAM, OSCAR bundle/image/command, Unitree SONIC sim2sim/persistent
  session, closed-loop eval/GPU launch, WAM compute, generated-video review, and
  runtime-bootstrap focused tests.

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
- Pinned learned OSCAR WAM execution to the official `oscar-public` source
  commit, `zywu2115/OSCAR-2B` HF revision, and checked provider image digest.
  The repo-local OSCAR-style generator remains deterministic fallback/test
  support and still cannot claim a learned OSCAR checkpoint, deployment proof,
  safety validation, physical readiness, or generated-world rank fidelity.

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
  `docs/archive/point-in-time/cpu-work-audit-2026-06-29.md`.
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
- Added an API-first DeepInfra Cosmos3-Nano WAM compute adapter behind
  `WamComputeProvider`. It emits redacted request/execution/cost/checksum
  artifacts, downloads generated MP4 output, packages
  `deepinfra_provider_runtime_output.zip`, and preserves the same generated
  support-media proof ceiling as RunPod/Vast.
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
  `docs/archive/superpowers/specs/2026-06-27-isaac-g1-kitchen-parity-design.md`,
  `docs/archive/superpowers/specs/2026-06-27-provider-agnostic-wam-compute-design.md`,
  and `docs/archive/goals/2026-06-27-provider-agnostic-wam-compute-loop.md`.

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
  include `docs/archive/point-in-time/last_24h_launch_audit_2026-06-26.md`,
  `docs/OSCAR_VISUAL_AUGMENTATION_PACKET.md`,
  `src/blueprint_pipeline/live_pipeline_forwarding_secret_setup.py`,
  `src/blueprint_pipeline/provider_closure_audit.py`,
  `src/blueprint_pipeline/rl_post_training_handoff.py`,
  `src/blueprint_pipeline/oscar_visual_augmentation_packet.py`, and
  `src/blueprint_pipeline/oscar_visual_augmentation_generation_runner.py`.
- Added simulator-agnostic G1/Isaac contracts and splat tooling through
  `docs/simulator-agnostic-g1-execution-contract.md`,
  `docs/archive/superpowers/specs/2026-06-26-isaac-splat-render-parity-design.md`,
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
  upgrades. Use `docs/archive/point-in-time/last_24h_launch_audit_2026-06-26.md` for the then-current
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
  untracked support files; use `docs/archive/point-in-time/last_24h_launch_audit_2026-06-26.md` for
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
