# Agentic Real2Sim — Paper Analysis and Blueprint Fit

Date: 2026-07-23

Paper: *Agentic Real2Sim: Physics-based World Modeling with Vision-Language
Agents* — https://agentic-real2sim.github.io/ ·
PDF https://agentic-real2sim.github.io/static/pdfs/agentic-real2sim.pdf

Status: internal advisory assessment. This memo does not install any component,
run any model, or upgrade any Blueprint proof boundary. It is a
"should we adopt / does it fill a real gap" review in the same spirit as
`docs/NVIDIA_SIGGRAPH_2026_STACK_IMPACT_2026-07-21.md`. No claim here is buyer-
facing or a rank-fidelity result.

## TL;DR

The paper is **adjacent to Blueprint, and mostly behind it**. Almost every stage
it presents as a contribution already exists in this repo — usually with far
stricter honesty discipline — and its single most important finding
("the bottleneck is upstream perception, not the orchestration agent") is a
conclusion Blueprint already reached independently in
`docs/MUJOCO_VS_ISAAC_LANE_GAP_ANALYSIS.md`.

There is **no stack to adopt**: no released code/checkpoints are offered, the
physics target is single-backend rigid-body MuJoCo replay, and the success
metric is deliberately lenient. But there are **three real takeaways**: one
genuine capability seam worth a bounded experiment (physical-prior inference),
one strategic option to note but not pursue now (demo→sim ingestion of existing
robot-demonstration datasets), and useful external corroboration for three
architectural bets Blueprint has already made.

## Decision Summary

1. **Do not treat this as an integration.** It is a research demonstration and a
   useful external data point, not a component or a benchmark Blueprint should
   inherit. Handle it the way `NVIDIA_SIGGRAPH_2026_STACK_IMPACT` handles ArtiFixer
   / MotionBricks: watch-list, not critical path.
2. **The one bounded experiment worth scoping** is a *physical-prior proposal*
   pass (object identity / material class / mass / friction hints) feeding the
   existing agent-proposal seam. This is Blueprint's weakest current analog
   (mass is largely placeholder), and the paper's stage-2 is directly about it.
   It must sit behind the existing "agents propose, cannot set proof booleans"
   boundary.
3. **Do not adopt the paper's grasp-position sweep or its success metric.**
   Blueprint already treats the grasp-sweep move as a fidelity lie and already
   rejects any-judge / best-of-N aggregation. Importing either would regress the
   evidence discipline that is the product's moat.
4. **Bank the corroboration.** The paper independently supports (a) the
   deterministic-tools + narrow-schema-agent architecture, (b) perception as the
   real bottleneck, and (c) that a cheap open VLM is sufficient for the
   orchestration seat — which is evidence for keeping `capture_enrichment_llm`
   and the agent-review surfaces on inexpensive models and investing the delta
   in capture and perception fidelity.

## What the paper actually is (and isn't)

**Is:** a four-agent VLM pipeline that converts one recorded robot-manipulation
episode (sampled from DROID) into a runnable MuJoCo "episodic twin." A VLM makes
narrow, schema-constrained judgment calls (which object matters, is this mask
good enough, which object defines the ground, retry?) while deterministic
specialist tools do all the heavy lifting: SAM 3 (segmentation), SAM 3D (mesh),
FoundationStereo (scale), FoundationPose (6-DoF object pose), MuJoCo (physics).
It reports 48/100 successes on DROID with an open 31B model at a **model** bill
of $2.62, and finds backend choice reduces largely to cost — the headroom is in
perception/sim, not the agent.

**Isn't:** a released stack, a physics-fidelity result, or a rigorous evaluator.
Three caveats govern how to read it:

- **Lenient success metric.** Up to five candidate reconstructions per episode
  (ranked by peak displacement) crossed with three VLM judges scoring keyframes
  out of 10, success if *any one* judge ≥ 8. That is an OR over candidates and
  judges. A consensus / single-candidate criterion lands materially lower.
- **Cost figure is model-only.** $2.62 covers VLM tokens; it excludes the GPU
  time for SAM 3D / FoundationStereo / FoundationPose / MuJoCo that dominates
  real per-episode cost.
- **The grasp sweep is a self-inflicted fidelity risk.** Stage 4 nudges object
  positions until a grasp succeeds, then reports success. For pure visual replay
  that is fine; for downstream policy learning it bakes a geometry lie into the
  twin that the replay metric cannot catch (the metric is what selected the
  shift).

## The core architectural difference

The paper reconstructs a scene that **already contains a robot acting**, from a
dataset episode. Blueprint starts from a **robot-free facility walkthrough**
(iPhone / 3DGS capture) and must *synthesize* the robot and its point of view
into the reconstructed site. That inversion changes what "6-DoF pose tracking"
even means: Blueprint has no recorded manipulation trajectory to track and no
robot pixels, so its analog is **robot forward-kinematics projected through a
calibrated camera to condition a world model that generates the robot POV**
(`eval_ready_task_grounding.py` FK projection → `oscar_cosmos_wam_evaluator.py`
egocentric WAM cameras). The paper never solves — and never has to solve —
Blueprint's actual hard problem: getting a physically usable, robot-ready scene
out of a walkthrough where no robot was present.

Consequence: the paper's unit of conversion (one dataset episode) and
Blueprint's unit (a facility, expanded into a task/scenario/eval family) do not
compete. They are complementary.

## Capability & overlap matrix

| Paper stage / idea | Blueprint analog (file / symbol) | Overlap or gap | Recommendation |
| --- | --- | --- | --- |
| Object selection (VLM picks which objects matter) | `object_index_stage.py` scene-typed prompt banks + optional LLM expansion; `episode_spec.py:build_task_anchor_proposals` (`proposal_authority = review_input_not_execution_or_proof`) | Full overlap; Blueprint keeps it review-required | Keep. No action. |
| Segmentation (SAM 3) | `object_index_stage.py` optional SAM3 / GroundingDINO / YOLO-World **subprocess** runners | Full overlap; Blueprint runs detectors as pluggable tools, off by default | Keep. |
| Mesh recovery (SAM 3D) | `object_geometry_stage.py` (trimesh) + World Labs / Marble asset materialization; **Palatial PhysReady** `twin_candidate_manifest.json` (task-critical object twins) | Overlap, but Blueprint downloads/authors assets rather than single-image mesh gen | Keep; Palatial lane already covers "task-critical object twin." |
| Scale (FoundationStereo) | `object_index_stage.py` conservative calibrated-evidence allowlist (`calibrated_depth_ray`, `multiview_triangulation`, validated provider reconstruction) | Overlap; Blueprint is deliberately conservative, falls back to proxy extents | Keep. |
| 6-DoF object pose (FoundationPose) | No object-pose tracking (no robot/trajectory in capture). Analog = robot FK projection through calibrated camera (`eval_ready_task_grounding.py`) | Genuine architectural difference, by design | N/A — different problem. |
| **Physical priors: identity / material / mass** | `capture_enrichment_llm.py` `articulation_prior_writer` skill; Palatial `twin_candidate_manifest.json` scale/articulation hints; but mass/material largely placeholder (`simready_assets.py` URDF `mass=1.0`) | **Weakest Blueprint analog — real seam** | **Bounded experiment (below).** |
| Scene prep: camera calibration, ground plane, robot base pose | `camera_geometry_validation.py`; `eval_ready_task_grounding.py` calibration quality gate; `splat_scene_analysis.py` floor/up-axis + occupancy-grid `suggest_robot_start` | Full overlap; Blueprint places a robot that was never in the scene | Keep. |
| Grasp optimization: sweep object shifts until grasp succeeds | `cpu_simulator_preflight.py:_build_spawn_pose_validation` (validity **filter**, not a success sweep); `g1_microwave_grasp_arc_seed.py` (prescribed, not contact-driven, door angle) | Overlap in ingredients; **opposite posture** — Blueprint refuses to credit success from a manufactured pose | **Do not adopt the sweep.** |
| Load into a physics backend | `simulation_automation.py` `SIMULATOR_FRAMEWORKS` = isaac_sim / isaac_lab_arena / mujoco / pybullet / newton behind `simulator_engine_plugin_registry` | Blueprint is multi-backend and swappable; paper commits to MuJoCo | Keep; Blueprint is ahead. |
| Deterministic tools vs narrow agent decisions | `simulation_automation.py` `CLAIM_BOUNDARY.agents_may_mutate_proof_booleans = False`; `capture_enrichment_llm.py` 6 schema-constrained skills; `eval_ready_task_grounding.py` explicit split of `vlm_or_human_review_checks` vs `deterministic_or_lightweight_checks` | Same philosophy, pushed further into fail-closed proof gating | Keep; corroborated by paper. |
| Replay-success VLM judge panel | `wam_vision_success_judge.py` / `rollout_vision_label_openai.py` + `wam_vision_success_review_queue.json` | Overlap in concept, **opposite aggregation** (see below) | Keep Blueprint's; do not import any-judge rule. |

## Evaluation rigor: the false-success comparison (the crux)

This is where the two systems most diverge, and where importing anything from the
paper would be a regression.

**Paper:** optimizes a lenient success rate. best-of-5 candidates × 3 judges,
success if `max(judge) ≥ 8/10`. It openly acknowledges the grasp-sweep can
manufacture false success that the replay metric cannot detect.

**Blueprint:** designed to *refuse to manufacture a success number* without
accepted real-world anchors. Concretely:

- **Judge aggregation is AND, not OR.** `oscar_cosmos_wam_evaluator.py`
  `_normalize_wam_success_labels` requires all three criteria
  (`end_effector_reaches_target`, `target_state_change_visible`,
  `robot_caused_target_motion`), each with evidence refs; a low-confidence or
  non-boolean verdict **abstains** to `wam_vision_success_review_queue.json` for
  human adjudication. There is no "best judge wins."
- **False success has a dedicated numeric kernel.**
  `wam_action_consistency_contract.py` (`strict_action_consistency_blockers`)
  requires an *inverse-dynamics-recovered action vector* that numerically matches
  the commanded action (per-dimension error recomputed and cross-checked against
  a threshold; `commanded_action_sha256` pin; replay-reuse blockers). Boolean-only
  "looks right" labels can never satisfy it. `WAM_EPISODE_CONSISTENCY_SCORER.md`
  states the rule: no `forward_inverse_consistency_proven=true` unless an external
  scorer ran with both visual **and** action-trace evidence.
- **The gain is measured, not assumed.**
  `wam_derived_observation_harness.py` withholds low-confidence / OOD /
  inconsistent steps from success scoring and emits
  `wam_false_success_reduction_metrics.json` — the fraction of cases where the
  naive video judge said "success" but the real outcome was failure that gating
  caught — and reports `not_measured` unless capture-backed validation labels
  exist.
- **A 7-layer fail-closed claim ladder bounds every "success."**
  `success_claim_contracts.py` `CLAIM_LADDER`:
  `no_claim → media_valid → review_task_success → simulator_task_success →
  policy_task_success → contact_state_change_proven → physical_deployment_ready`.
  A higher layer can never be asserted while a lower one is unproven, and
  `physical_readiness` is never derivable from the others. `buyer_claim_ceiling.py`
  lints buyer copy against the ladder.
- **Ranking blocks proxies.** `decision_grade_ranking.py` (Bradley-Terry) refuses
  to rank unless judge calibration is independently accepted and no
  fixture/proxy/fallback output is used — the structural opposite of accepting a
  lenient best candidate.
- **Calibration to reality exists on both the judge and the rankings.** Brier
  ≤ 0.1 with inter-rater ≥ 0.6 on the labeler; Pearson / Spearman / MMRV / MAE
  sim-vs-real with preregistered CI-lower-bound thresholds (7 checkpoints, 3
  criteria, 2 splits, 20 trials/cell) in `robot_eval_calibration.py`.

Blueprint has also already **encoded and self-audited** the rigorous cousin of
this paper: `sc3_eval_protocol.py` pins the SC3-Eval paper (arXiv:2606.18610v3)
as its north star, and `docs/PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md`
enumerates the exact false-success / leniency traps (SC3-01…SC3-12) and marks
several already remediated in code.

**Honest caveat:** much of the above is *contract and gating* rather than a fully
executed live evaluation program with real-world anchors — the SC3 gap audit and
the calibration module's small-N diagnostics say so directly. That is Blueprint's
real frontier, and it is a *higher* bar than the paper, not a lower one.

One idea worth a light cross-check (not an adoption): the paper's rubric
decomposition — wrong target object / wrong final location / wrong action /
wrong gripper position — is clean and manipulation-specific. Blueprint's three
AND-criteria plus root-cause failure categories
(`failure_diagnosis_contract.py`) already cover the concept, but comparing the
two axis sets could sharpen the criterion wording. The aggregation rule is *not*
worth taking.

## Scenario diversity comparison

The paper does essentially **no** diversity generation — move-object / change-mass
/ add-glare / run-variations appear only as motivating future work. Blueprint
implements each as a concrete, measured pipeline stage:

- A **16-axis, site-type-scoped variation vocabulary**
  (`robot_eval_dataset.py` `KNOWN_SCENARIO_VARIATION_DEFINITIONS`) covering
  lighting, object rotation, cart shift, blocked path, occlusion, distractor /
  wrong-object, human crossing, forklift / AGV traffic, conveyor motion, machine
  guarding, thermal surface, narrow approach.
- A **deterministic instantiator** (`scenario_variation_instantiator.py`) that
  turns each axis into concrete physical mutations and emits **per-engine mutation
  payloads** for all five simulators (`usd_stage.*`, `arena_cfg.*`, `mjcf_patch.*`,
  `pybullet_api.*`, `newton_scene_graph.*`).
- The direct **"add glare / change appearance"** analog
  (`oscar_visual_augmentation_packet.py`): visual distribution-shift variants that
  hold source motion and camera geometry constant, behind a swappable
  OSCAR/Cosmos/future backend registry with an explicit claim boundary.
- **Robot-POV synthesis from a robot-free capture**
  (`scene_wam_policy_episode_packet.py` capture-derived robot POV via
  depth/splat re-projection).
- A **stance search axis** (`adaptive_task_stance_configurator.py`) that varies
  the robot's base pose under bounded search with gates an agent cannot waive.
- A **failure taxonomy + harvested breakage library**, and an **eval matrix**
  (`robot_eval_execution.py:build_scenario_eval_matrix`) that joins every
  variation instance to a scored `scenario_eval_run_id` with exact coverage
  accounting.

All of it stays gated at `simulator_execution_proven: False` until owner
evidence exists. This is a large, real subsystem the paper only gestures at — it
also happens to be the strategic point (`VISION.md` cites the data-scaling-law
result that generalization scales with environment diversity, which is the whole
per-site-capture moat).

## Fit with Blueprint's product strategy

The paper strengthens the existing strategy; it does not challenge it.
`WORLD_MODEL_STRATEGY_CONTEXT.md` says the durable moat is capture supply,
rights/privacy/provenance-safe pipelines, and Task Evaluation Runs / Post-Training
Data Packages grounded in real sites — with model backends kept replaceable. The
paper is a supplier-ecosystem signal that the *conversion* of demonstrations to
sim is becoming cheap and automatable, which raises the value of proprietary
real-site capture and rigorous evaluation, not lowers it. It should be treated as
external evidence behind Blueprint's existing replaceable-adapter and
evidence-ledger boundaries, exactly like the NVIDIA components.

## What should NOT be adopted or rebuilt

- The **grasp-position sweep** (stage 4). Blueprint deliberately treats
  "shift geometry until the task passes" as a fidelity lie; `spawn_pose_validation`
  filters candidates, and the only in-repo perturbation *degrades* actions for
  robustness testing. Do not add success-seeking geometry perturbation.
- The **any-judge / best-of-N success aggregation.** Blueprint's AND-criteria +
  abstention + claim ladder is the moat; importing OR-aggregation would regress it.
- The **model-bill-only cost framing** as a headline. Blueprint already accounts
  for GPU/provider cost through the paid-resource allocator and cost-control
  ledgers; do not quote a model-token figure as total cost.
- Do not present any paper metric (the paper's 48% success rate, or SC3-Eval's
  0.929) as a Blueprint result; these are **not Blueprint measurements**, and
  `PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md` already forbids restating them.

## Recommended bounded experiment: physical-prior proposals

This is the one place the paper maps to a genuine gap. Blueprint's mass / material
/ friction is largely placeholder (`simready_assets.py` `mass=1.0`); only
articulation is inferred, via the `articulation_prior_writer` skill.

Proposed scope (design-only until explicitly authorized):

- A `capture_enrichment_llm.py` skill (or extension of the Palatial
  `twin_candidate_manifest.json` path) that emits **object identity, material
  class, mass range, and friction range as proposals** with confidence and
  provenance, for task-critical objects only.
- Outputs are **proposals**, written under the existing review-required boundary:
  they cannot set proof booleans, cannot upgrade any claim, and must carry a
  `claim_boundary` mirroring the Palatial and episode-spec proposal contracts.
- Deterministic validation where evidence exists: reject mass/scale proposals that
  violate measured bounding-box extents or calibrated scale; keep the value a
  labeled proposal, never a fact.
- Acceptance gate: a proposal only becomes an authored physical property through
  the same human/owner review path Blueprint already uses for accepted variations,
  and never through the agent itself.

Value: better contact-rich object parameters for the MuJoCo lane's
manipulation targets without weakening capture truth. Cost: bounded — it reuses
existing seams and adds no new backend dependency.

## Strategic option (note, do not pursue now): demo→sim ingestion

The paper's actual capability — turning an *existing* recorded robot-demonstration
episode into a runnable sim episode — is something Blueprint does not do, because
its unit is robot-free capture. A "demo→sim ingestion adapter" could convert
buyer-supplied or public robot-demonstration datasets (DROID, Open X-Embodiment —
both already tracked in `VISION.md` and `wam_backend_strategy.py`) into
`episode_spec.v1` + scenario variations, i.e. a Post-Training Data Package built
from a buyer's own demos rather than a captured site.

This is adjacent to, not on, the capture-first flywheel. Record it as a possible
future package variant for buyers who arrive with demonstration data; do not build
it speculatively. It would still have to pass the same claim ladder and rights /
provenance contracts as any other package.

## Stop rules

Abandon any experiment derived from this paper if:

- a physical-prior proposal can flip a proof boolean or upgrade a claim without
  human/owner acceptance;
- success-seeking geometry perturbation is introduced anywhere in the sim lanes;
- any paper metric is restated as a Blueprint measurement;
- the experiment requires raw/unredacted capture to leave the accepted privacy
  and rights boundary; or
- it does not catch a useful failure class earlier or more cheaply than the
  current pipeline.

## Bottom line

Blueprint has independently built a stricter, multi-backend, facility-scale
version of what this paper prototypes at single-episode scale — and its
evaluation stack is engineered around exactly the false-success failure mode the
paper flags but does not defend against. The paper's lasting value to Blueprint
is threefold: it **corroborates** the deterministic-tools-plus-narrow-agent
architecture and the perception-is-the-bottleneck roadmap; it supplies **evidence**
that a cheap VLM suffices for the orchestration seat; and it points at the one
underbuilt seam — **physical-prior inference** — worth a bounded, proof-boundaried
experiment. Everything else is watch-list.

## Source map (files referenced)

- Pipeline / perception: `src/blueprint_pipeline/run_e2e.py`,
  `object_index_stage.py`, `object_geometry_stage.py`, `splat_scene_analysis.py`,
  `camera_geometry_validation.py`, `eval_ready_task_grounding.py`,
  `episode_spec.py`, `cpu_simulator_preflight.py`, `simulation_automation.py`,
  `g1_microwave_grasp_arc_seed.py`, `capture_enrichment_llm.py`,
  `simready_assets.py`; `docs/SIMULATION_AUTOMATION_LANE.md`,
  `docs/PALATIAL_PHYSREADY_LANE.md`, `docs/MUJOCO_VS_ISAAC_LANE_GAP_ANALYSIS.md`.
- Evaluation / claims: `oscar_cosmos_wam_evaluator.py`,
  `wam_vision_success_judge.py`, `rollout_vision_label_openai.py`,
  `wam_action_consistency_contract.py`, `wam_derived_observation_harness.py`,
  `closed_loop_consistency_scoring.py`, `success_claim_contracts.py`,
  `claim_contract_keys.py`, `buyer_claim_ceiling.py`, `decision_grade_ranking.py`,
  `evaluation_run_contract.py`, `robot_eval_calibration.py`,
  `failure_diagnosis_contract.py`, `sc3_eval_protocol.py`, `benchmark_protocol.py`;
  `docs/WAM_EPISODE_CONSISTENCY_SCORER.md`,
  `docs/PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md`,
  `tests/test_success_claim_contracts.py`.
- Diversity: `robot_eval_dataset.py`, `scenario_variation_instantiator.py`,
  `oscar_visual_augmentation_packet.py`, `scene_wam_policy_episode_packet.py`,
  `adaptive_task_stance_configurator.py`, `robot_eval_execution.py`.
- Strategy: `PLATFORM_CONTEXT.md`, `WORLD_MODEL_STRATEGY_CONTEXT.md`, `VISION.md`,
  `wam_backend_strategy.py`, `docs/NVIDIA_SIGGRAPH_2026_STACK_IMPACT_2026-07-21.md`.
