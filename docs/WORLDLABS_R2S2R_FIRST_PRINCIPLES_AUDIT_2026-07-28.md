# World Labs R2S2R: First-Principles Audit of the 2026-07-28 Strategy Analysis

Status: independent audit, 2026-07-28. This document audits a prior AI-agent
analysis of the World Labs "Real-to-Sim-to-Real" announcement (2026-07-28) and
its follow-on proposal that Blueprint become a maximal control plane / wrapper
on a hypothetical World Labs R2S2R service. Every external claim below was
re-verified today against primary sources; every repo claim was re-verified
against the code on `main` (through PR #224).

Truth labels used here follow house convention: **verified** (checked against a
primary source or repo code today), **unverified** (asserted by the source
analysis but not confirmable from available text), **inference** (reasoned
judgment, labeled as such).

---

## 1. Sources independently checked today

External:

- World Labs R2S2R announcement — https://www.worldlabs.ai/blog/real-to-sim-to-real
- World Labs SceniX acquisition post (2026-07-21) — https://www.worldlabs.ai/blog/scenix
- SceniX-lineage evaluation paper — arXiv:2511.04665 (Real-to-Sim Robot Policy
  Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions)
- Agentic Real2Sim paper — arXiv:2607.19190 (already reviewed in-repo:
  `docs/AGENTIC_REAL2SIM_PAPER_ANALYSIS_2026-07-23.md`)
- World Labs public API reference (worlds:generate) and Terms of Service —
  docs.worldlabs.ai
- Competitive scan of the real-to-sim evaluation category (Lightwheel, Hillbot,
  NVIDIA Isaac Lab-Arena, Real-is-Sim, RoboDojo)

Internal: full-tree audit of `BlueprintCapturePipeline` (contracts, WAM/GPU
lanes, simulator lanes, anchors, ranking, admission harness, prior strategy
analyses), plus doctrine docs (`PLATFORM_CONTEXT.md`,
`WORLD_MODEL_STRATEGY_CONTEXT.md`, `VISION.md`, `AGENTS.md`).

---

## 2. Verdict at a glance

The source analysis is **largely accurate on the facts and directionally right
on strategy**: this is one of the most Blueprint-relevant announcements yet; it
validates the Task Evaluation Run concept; the right integration shape is a
replaceable provider behind the existing six-part evaluation contract; and the
four never-delegate items (evidence authority, exam definition, result
interpretation, buyer relationship) are exactly right.

The audit found **five material problems** with it as a plan:

1. It frames the bet as "wrapper on World Labs" when the evidence supports
   "control plane over an emerging *provider class*" (World Labs is the most
   visible entrant, not the only one).
2. Its "independent verification" story is under-built: if all execution is
   delegated, verification collapses into recomputing statistics over
   provider-authored rollouts. Independence requires non-provider evidence —
   physical anchors above all — and the repo's own registry records that
   Blueprint currently collects **zero** real-world task outcomes.
3. Its legal read is too soft. Under World Labs' standard terms, the
   provider-qualification harness the analysis itself demands is arguably
   prohibited, and customer captures/policy weights cannot be sent at all.
   A negotiated agreement is a precondition, not a nicety.
4. It under-credits the repo: much of its "what we need to build" list already
   exists (admission harness, rank-fidelity metrics, provider-neutral anchor
   schema, policy gateway modes, digest-bound canonical package). The real
   build list is much shorter.
5. Its retirement table, read as a to-do list, would bet roughly 40–50% of a
   ~475k-LOC codebase on a product that today has **no API, no pricing, no
   export guarantees, no named customers, and no announced availability**. As a
   *trigger-gated end state* it is defensible; as a near-term plan it is not.

The single biggest strategic implication the source analysis under-weights: the
scarce input for R2S2R-grade evaluation is **robot-in-the-loop task calibration
capture**, which Blueprint's supply network does not currently collect. That —
not the adapter — is the long-lead pivot, and it pays off under every future
(World Labs, a competitor, or in-house lanes).

---

## 3. Fact-check of the source analysis (external claims)

| Claim in source analysis | Status |
|---|---|
| Two announcements: SceniX acquisition 07-21, R2S2R results 07-28; tech demonstration, not GA product | **Verified** |
| Closed loop: capture task → aligned sim → variations → train/evaluate → transfer → refine | **Verified** |
| Matched open-loop sim/real comparisons for box packing, handover, cable sliding/plugging, elastic cable insertion, power-cord routing | **Verified**; the article also shows Flexiv test-tube transfer and xArm marker/pencil singulation from clutter |
| Policies trained with "zero real-world data" transferred to ALOHA, RB-Y1, YAM, Flexiv, xArm | **Verified** (verbatim: "trained entirely in simulation, with zero real-world training data") |
| "Several showcased policies operated one hour without intervention" | **Approximately verified** — the article states **five tasks** ran autonomously for one hour |
| Cube-handover figure: 2,000 sim trials (1,000 ID / 1,000 OOD) and 100 real trials (50/50) per checkpoint | **Verified** |
| The figure compares **GR00T N1.6 and π0.5** checkpoints | **Unverified** — the article text and captions name no policy architectures at all (only generic "policies", VLAs, WAMs). Possibly present in an embedded chart legend; treat as unconfirmed and do not repeat as fact |
| No published Pearson/Spearman/Kendall/MMRV/CI for the new engine | **Verified** — no numbers anywhere in the post |
| SceniX-lineage paper: Pearson 0.944 / 0.901 / 0.915 on toy packing, rope routing, T-block pushing; small real sets | **Verified exactly** (arXiv:2511.04665, Table I; real sets 20 / 27 / 16 configurations; MMRV 0.076 / 0.174 / 0.108; policies ACT, Diffusion Policy, π0, SmolVLA) |
| Public World Labs API = Marble world generation only, no R2S2R/robotics endpoints | **Verified for the load-bearing part**: no robotics, physics, policy-training, or evaluation endpoints exist. Generate + operations endpoints confirmed directly; media/export surfaces (SPZ splats, collider mesh URL, panoramas) match in-repo API notes at `docs/MARBLE_SIM_ASSET_HANDOFF.md:94-110` |
| ToS restricts competitive benchmarking; broad content rights with paid opt-out | **Verified and stronger than stated** — see §6.4 below |
| No SDK/API/pricing/customer-access announced for R2S2R; no named customers ("customer-inspired" tasks) | **Verified**; the site has no enterprise/robotics contact program at all. SceniX's founder (Yunzhu Li) publicly thanked SceniX's *customers*, so commercial motion exists privately |

Net: the source analysis's research is trustworthy, with one attribution
(GR00T N1.6 / π0.5) that must be downgraded to unconfirmed.

---

## 4. What the source analysis gets right (endorsed)

1. **High relevance and concept validation.** R2S2R is a task-specific,
   calibrated, closed-loop evaluation engine — the same product category as the
   Task Evaluation Run. Their "simulation preserves relative ranking, tracks
   improvements and plateaus across checkpoints" framing is Blueprint's rank-
   fidelity thesis, stated by a heavily funded competitor. This also confirms
   the moat doctrine already in `WORLD_MODEL_STRATEGY_CONTEXT.md:106-117`: if
   world models become easier to buy, capture supply and product workflow
   become more valuable, not less. That predicted event is now happening.
2. **The control-plane direction is already doctrine — with a sharper edge
   than the source analysis gives it.** `VISION.md` rung 2 defines Blueprint as
   the "measurement standard — the thing the industry transacts *against*,"
   with the strategic logic stated as Aggregation Theory: "own the 'which one
   is best' decision and the suppliers being rated become interchangeable
   beneath you… It is worth more than any single model we could own"
   (`VISION.md:140-142`). The maximal-wrapper proposal is a special case of
   this — but doctrine's version makes World Labs one interchangeable supplier
   beneath the decision layer, not the layer itself. VISION also carries the
   governing caution (the Nielsen precedent, `VISION.md:137-138`): "a
   measurement monopoly cracks when the substrate shifts and clients fund
   challengers. A trust layer must continuously re-validate against the
   frontier or die." R2S2R is exactly such a substrate shift; re-validating
   against it is mandatory, merging into it is not.
3. **Integration shape.** Mapping a future R2S2R service onto the six-part
   contract is correct and cheap: the six components exist verbatim in
   `src/blueprint_pipeline/evaluation_run_contract.py:26-33` and
   `docs/architecture/evaluation-run-interface.md`. An R2S2R provider is a new
   `runtime_provider_profile` + `scene_bundle` source behind the same boundary.
4. **The Marble handoff gap is real and R2S2R targets it.**
   `docs/MARBLE_SIM_ASSET_HANDOFF.md:130-141` refuses exactly the claims
   (simulator load, contact validity, policy execution, rank fidelity) that
   R2S2R exists to supply. If a real service exports those artifacts, that
   bridge has served its purpose.
5. **The four never-delegate items.** Raw evidence authority, definition of the
   exam, interpretation of results, buyer relationship/package. Fully aligned
   with `PLATFORM_CONTEXT.md` truth hierarchy and the moat doctrine.
6. **Training/evaluation branch separation** in the proposed provider state
   machine (trained policies frozen before hidden-split evaluation). This is a
   genuinely good design detail; it protects the hidden-split integrity that
   makes Blueprint's grading trustworthy.
7. **No speculative integration code before access is confirmed.** Matches the
   house pattern set by `docs/NVIDIA_SIGGRAPH_2026_STACK_IMPACT_2026-07-21.md`
   ("bounded experiments, not a wholesale stack change") and
   `docs/AGENTIC_REAL2SIM_PAPER_ANALYSIS_2026-07-23.md` ("watch-list, not
   integration").
8. **Capture mismatch identified.** The optional `task_calibration_bundle`
   claim-upgrade lane is the right mechanism (this audit raises its priority —
   see §6.5).
9. **The diligence-gate question list and the retirement gate list** are both
   good and worth adopting nearly verbatim — as *gates*, not as a schedule.

---

## 5. Repo reality vs the "what we need to build" list

The source analysis writes as if most of its proposed control-plane machinery
is net-new. Most of it exists. What follows is the mapping (all verified in
code today):

| Proposed build | Already in repo | Genuinely missing |
|---|---|---|
| Canonical R2S2R input package | `canonical_site_package.py` (722 LOC, digest-bound via `source_checksums`, includes `semantic_task_context` with task statement/success criteria/target objects, rights/privacy/provenance, `adapter_mappings` for `world_labs_marble`) | Robot/policy identity fields (deliberately live in `evaluation_run_contract` bindings instead); a composed, immutable "task package" that joins site package + eval-run spec + calibration bundle |
| R2S2R execution adapter | Real World Labs client exists and is production-wired: `provider_preview.py` calls `api.worldlabs.ai` `/marble/v1/worlds:generate` with API-key auth and ~20-min polling; checksum-bound materialization (`worldlabs_asset_materialization.py`, `blueprint-materialize-worldlabs-assets`); spend/admission gating patterns in `docs/PAID_SPEND_ADMISSION_LOCK.md` | Everything R2S2R-specific (there is no R2S2R API to call). Zero references to SceniX/R2S2R exist in-tree — nothing has been prematurely built, which is correct |
| Task & scenario compiler | `scenario_variation_instantiator.py` already emits **declarative engine-neutral mutation payloads** with claim boundaries (`requires_owner_engine_adapter: True`, `simulator_execution_proven: False`, five engine targets incl. `isaac_lab_arena`) | A compiler from buyer language to frozen scenario contracts remains partially manual; provider-proposed-scenario acceptance flow |
| Secure policy gateway | Four modes registered in `evaluation_run_contract.py:179-193` (`in_process`, `persistent_worker`/command, `multi_modality`, `http`); GR00T / π0-openpi / OpenVLA / UnifoLM adapters; `action_space_registry.py`; checkpoint digest pinning in `decision_grade_ranking.py` | Provider-side policy delivery (upload/container handoff to an external R2S2R service) and its retention/no-training terms |
| Provider-output evidence envelope | `evaluator_runtime_evidence.py` + `docs/EVALUATOR_RUNTIME_EVIDENCE_CONTRACT.md`; the admission rule "a declared completed result is never trusted as a substitute" (`docs/EVALUATOR_QUALIFICATION_WORKFLOW.md`) | An R2S2R-shaped rollout/privileged-state/contact-trace schema instance |
| Provider qualification harness ("golden corpus") | `evaluator_qualification_workflow.py` (≥7 policy checkpoints, ≥4 sites, ≥20 matched trials/cell, frozen splits, teardown + billing reconciliation) and — the exact precedent — `docs/ROBOWORLD_EVALUATOR_INTEGRATION.md` with a frozen digest-pinned admission profile for an external evaluator backend | A `worldlabs_r2s2r` admission profile instance (a document + JSON profile, not new machinery) |
| Rank-fidelity metrics (Pearson/Spearman/Kendall tau-b/MMRV/bootstrap CIs) | All implemented: `benchmark_uncertainty.py:30-34` (10,000-replicate hierarchical bootstrap), `decision_grade_ranking.py`, `robot_eval_calibration.py`, `docs/BLUEPRINT_BENCHMARK_PROTOCOL.md`; plus world-model-free control-arm rankers (`control_ranker.py`) the analysis never mentions | Nothing — except the *data* to compute them on (see §6.3) |
| Physical-anchor ingestion | Provider-neutral `accepted_real_world_anchor.v1` schema declared across 8 modules with fixed join keys `(scenario_eval_run_id, policy_id, task_id, scenario_variation_instance_id)`; G1 capture kits (`g1_field_run_capture.py`, `g1_controlled_run_evidence.py`, `anchor_return_kit.py`) | **Operations, not schema**: `docs/external_anchor_candidate_registry_2026-07-20.json` records `blueprint_collects_real_world_task_outcomes: false`, `correlation_not_measured`, every candidate `blocked_candidate`. Kits are G1-specific, not robot-neutral. Closing this is already the repo's standing goal (`docs/goals/2026-07-02-sc3-eval-robot-policy-agnostic-service-plan.md` slices 2–3: human/owner-accepted anchors only, preregistered minimum-N, fail-closed `correlation_not_measured`) |
| Simpler buyer product | `buyer_package_readout.py`, `buyer_claim_ceiling.py`, WebApp sync — already provider-opaque by design | Nothing structural |

Conclusion (**inference**): the true incremental build for a real R2S2R
provider is roughly — one adapter package, one admission profile, one input
compiler, one result-envelope schema instance, deletion/consent propagation,
and a calibration-capture spec. That is weeks of contract work, not a company
re-architecture. The six-part contract already did the hard part.

---

## 6. Where this audit disagrees or reframes

### 6.1 Provider class, not provider

The source analysis's own conclusion ("integrate as a replaceable,
independently validated provider, never as the source of truth") is right, but
its maximal-wrapper answer then designs everything around World Labs
specifically. The category is forming, not settled: Lightwheel (~$145M Series B;
SimReady assets, sim-first evaluation tooling), Hillbot (ManiSkill lineage),
NVIDIA Isaac Lab-Arena (open-source scalable policy evaluation), Real-is-Sim
(Embodied Gaussians dynamic twins), and the academic real2sim-eval line all
target overlapping capability. Doctrine already forbids single-provider
overfit (`WORLD_MODEL_STRATEGY_CONTEXT.md:36-44,87-96`), and VISION invariant 2
makes it rung-independent: "Model backends stay swappable. No rung couples the
company to one checkpoint, provider, or world model" (`VISION.md:242-256`).

**Reframe:** define an `r2s2r_calibrated_sim` *capability class* on the
evaluation-run interface (a `runtime_provider_profile` + `scene_bundle` source
that must export rollouts, privileged state, and calibration evidence), with
World Labs as first candidate — exactly how the WAM lane treats Cosmos 3 vs
OSCAR. The control-plane architecture is then identical, but the company story
and the negotiation posture are materially better.

### 6.2 Independent verification requires independent evidence generation

The maximal plan retires every non-World-Labs execution lane and keeps
"independent ranking and calibration." But recomputing campaign statistics over
provider-authored rollouts verifies arithmetic, not the simulator. If World
Labs builds the twin, generates the variations, runs the rollouts, and grades
success, then Blueprint's only *independent* checks are: (a) matched physical
anchors, (b) cross-provider replication, (c) a retained in-house replay/
cross-check lane, and (d) world-model-free control arms (already built,
`control_ranker.py`). The plan keeps only (a), and only as a thin ingestion
contract.

**Reframe:** "retire local simulation from production" should be "demote local
simulation to a verification/reference lane." Concretely: the MuJoCo
policy-in-the-loop lane is the natural keeper (per
`docs/MUJOCO_VS_ISAAC_LANE_GAP_ANALYSIS.md`, MuJoCo owns the learned-policy
closed loop; Isaac owns pixels/placement and can shrink first). Spot-replaying
a sample of provider rollouts and running control-arm rankers against provider
rankings is cheap insurance against exactly the failure mode the analysis
warns about: the same provider defining, administering, and grading the exam.

### 6.3 The binding constraint in *both* futures is physical anchors

Ground truth in-repo today: rank fidelity against reality is **unmeasured**
(`decision_grade_ranking.py` emits `pearson/spearman/mmrv: None` pending
anchors; the anchor registry says collection is not happening). That is the
single biggest gap in the current stack's sellable claims — and it is *also*
the admission instrument for any R2S2R provider, *and* the negotiating asset
World Labs cannot easily replicate across third-party sites it does not
control.

The source analysis lists physical-anchor ingestion seventh of eight builds.
It should be first or second, and the work is operational (get real trials
flowing through the existing `accepted_real_world_anchor.v1` schema; generalize
the G1-specific kits), not schema design. This is not even a new priority: it
is the repo's standing goal file
(`docs/goals/2026-07-02-sc3-eval-robot-policy-agnostic-service-plan.md`), whose
slices 2–3 (owner-accepted anchors, preregistered minimum-N, computed-only
correlation claims) have been planned but unexecuted since early July. World
Labs validated their sim with 100 physical trials per checkpoint comparison;
Blueprint currently has zero.

One statistical footnote that cuts both ways
(`docs/EVALUATOR_ATTRIBUTION_AND_PUBLIC_ANCHOR.md`): correlation degrees of
freedom come from **policy count**, not rollout count — at n=7 policies a
measured r=0.95 certifies only ≥0.69 at standard confidence, and certifying
≥0.90 needs ~33 policies. Applied outward: World Labs' cube-handover figure,
built on a handful of checkpoints in one task, cannot statistically certify
high rank fidelity either — which both justifies Blueprint's stricter
admission bar and defines what a real pilot must contain (many policies and
checkpoints, not many rollouts of two).

### 6.4 The legal wall is load-bearing, not a checklist item

Verified from the current ToS:

- §2.8(d) prohibits "competitive analysis or benchmarking, develop[ing]
  competing AI models, or develop[ing] systems that replicate the Services'
  core functionality." Read literally, the provider-qualification harness the
  analysis itself requires (comparing World Labs against other providers and
  against physical anchors, then publishing rankings to buyers) is prohibited
  under standard terms — and even *retaining* in-house R2S2R-like lanes could
  be attacked under the "replicate core functionality" clause while Blueprint
  is a customer.
- §3.6 grants World Labs a license to use customer content for training —
  **irrevocable** for free accounts; revocable with **prospective-only** opt-out
  for paid accounts. Incompatible with Blueprint's rights/privacy doctrine for
  third-party site captures and with buyer policy weights under NDA.
- §11.4: no post-term retention obligations; deletion at World Labs'
  discretion. Incompatible with consent-revocation propagation promises.

**Consequence:** no customer capture, no buyer policy artifact, and no
provider-comparison result may touch the service under standard terms. A
negotiated enterprise agreement (benchmarking carve-out, no-training clause,
deletion SLA, export rights, resale/sublicensing) is a *precondition to the
first real pilot*, not a step at the end. The source analysis says "contract
review is essential" but sequences it as diligence; it is a gate.

### 6.5 The capture mismatch is the actual pivot

R2S2R's strongest results start from robot-side capture: demonstrations,
matched open-loop interactions, robot/camera calibration, and per-task physical
validation. Blueprint's supply network collects human walkthrough capture only —
verified hard in the capture contract: `BlueprintCapture`'s
`docs/CAPTURE_RAW_CONTRACT_V3.md` defines a rich five-modality bundle
(walkthrough video, poses, intrinsics, depth+confidence, motion/IMU, meshes,
rights/provenance) with **zero robot channels** — no joint states, no commanded
actions, no teleoperation, no demonstration paths anywhere in the app or
contract. The doctrine data-priority list
(`WORLD_MODEL_STRATEGY_CONTEXT.md:149-165`) likewise contains no robot-side
fields. The source analysis notices this ("capture mismatch") but files it as
one build item. Helpfully, the capture contract's own stated goal — "Keep raw
capture independent of any single world-model provider. Preserve enough truth
to swap downstream backends without re-capturing sites" — is exactly the
design principle a `task_calibration_bundle.v1` extension should inherit.

**Reframe (inference):** this is the strategic headline. If task-level
calibration capture is the scarce input, then either (a) Blueprint's capturer
network learns to deliver it (a kit + protocol + app problem — BlueprintCapture
roadmap), or (b) providers like World Labs capture tasks themselves at customer
facilities and Blueprint's walkthrough capture is relegated to context/scenery.
Owning calibration-grade task capture at third-party sites is the version of
"capture-first" that survives this announcement, and it pays off under every
backend future. It is also the piece with the longest lead time (hardware
kits, capturer training, site relationships), which is why it should start
before any provider access exists.

### 6.6 WAM and GPU-fleet retirement should ride their own evidence timetable

The repo already concluded — before this announcement — that WAM-based ranking
is not currently reliable and is sequenced second
(`docs/policy_and_wam_benchmark_research_2026-07-26.md`: OSCAR validated
open-loop only, r=0.750 on RoboArena; chained rollouts collapse; "policies
first"). R2S2R corroborates that internal conclusion; it does not create it,
and an unreleased third-party demo is not the trigger for deleting ~111k LOC of
WAM lane plus ~127k LOC of GPU/provider infrastructure (together roughly 40-50%
of `src/`). Some of that fleet also serves policy *hosting* (sealed customer
checkpoints), which persists in every future — the analysis concedes this in
one row ("unless needed for sealed customer policy hosting") and the concession
swallows more of the table than it admits.

**Reframe:** hold retirements to the analysis's own gate list (adopted below),
and let the WAM lane's fate be decided by the already-planned Cosmos-3-vs-
frozen-cells experiments, not by a competitor's blog post.

### 6.7 Keep the product taxonomy stable

The proposed five-subsystem end state drops Policy Improvement Runs as a named
product; doctrine lists TER / PIR / PTDP / hosted access as the primary
sellable outputs (`PLATFORM_CONTEXT.md:63-70`). If a provider owns training,
PIR becomes an orchestration of the provider's training branch under
Blueprint's before/after evidence gates — the product does not disappear, and
the contracts (`docs/POLICY_IMPROVEMENT_RUN.md`) should remain the stable
frame. Post-Training Data Packages likewise depend on curated clips, labels,
and generated variations; retiring all generation/reconstruction infrastructure
before provider exports demonstrably fill PTDP needs would break a primary
product to serve a support layer.

### 6.8 Economics and the competitor posture deserve harder numbers

The wrapper strategy's margin is a function of undisclosed provider pricing,
and the provider has visible intent to serve robot teams directly
("customer-inspired" tasks; SceniX had customers; explicit "serve customers
with different robots, sensors, policy stacks" language). **Inference:** for
tasks at facilities a robot team already controls, World-Labs-direct will
likely beat Blueprint-as-reseller. Blueprint's defensible ground is where the
provider is weak: third-party sites it does not control, rights-cleared
multi-site portfolios, cross-provider and cross-policy-vendor neutrality, and
buyer-side evaluation governance. Pricing, wholesale terms, and a
no-direct-conflict understanding belong in the first partner conversation, and
the diligence gate should add: per-task reconstruction cost/time, marginal
rollout cost, and whether World Labs will sell evaluation results directly to
Blueprint's buyer segments.

---

## 7. What to do now (this audit's recommendation)

Ordered; items 1–2 are valuable under every future and have the longest lead
times. Nothing here writes speculative provider-integration code, consistent
with house precedent.

1. **Unblock physical-anchor collection** (operational). Get real matched
   trials flowing through `accepted_real_world_anchor.v1`; generalize the
   G1-only capture kits toward robot-neutral `physical_trial`-style intake.
   This simultaneously (a) closes the current stack's unmeasured-rank-fidelity
   gap, (b) builds the instrument that qualifies any R2S2R provider, and
   (c) creates the negotiation asset. This is the standing goal file's
   slices 2–3, re-ranked to the top by this audit.
2. **Spec `task_calibration_bundle.v1`** (cross-repo with BlueprintCapture):
   optional claim-upgrade capture lane — synchronized robot camera video,
   joint/EE states, commanded actions, intrinsics/extrinsics, robot model and
   base pose, object start/end states, matched interactions, success criteria,
   rights/provenance per recording. Design the capturer kit and protocol; pilot
   at one friendly site.
3. **Open the World Labs partner/diligence conversation** with the source
   analysis's question list plus: enterprise terms (benchmarking carve-out,
   no-training, deletion SLA, export and resale rights), per-task cost/time,
   rollout-artifact export fidelity, and direct-sales intent toward robot
   teams. No integration code until access and terms exist.
4. **Author the frozen admission profile** `worldlabs_r2s2r` following the
   RoboWorld pattern (`docs/ROBOWORLD_EVALUATOR_INTEGRATION.md`): digest-pinned
   capability profile, required export artifacts, admission metrics
   (existing `benchmark_uncertainty.py` set), and preregistered thresholds,
   with status `awaiting_upstream_release` until there is something to pin —
   the exact posture `docs/EXECUTION_COST_AND_ARCHITECTURE_GATES.md` §6
   already codifies (admitting an upstream backend requires released,
   pinnable code/service; adopting the recipe is a separate decision that
   confers no evaluator standing). Add World Labs R2S2R to the same
   watch-list mechanism used for ArtiFixer / MotionBricks / Cosmos-Dreams.
   Documents and JSON profiles only.
5. **Name the capability class** `r2s2r_calibrated_sim` in
   `docs/architecture/evaluation-run-interface.md` as a provider-profile
   flavor, so World Labs, Lightwheel, Isaac Lab-Arena, or an in-house lane can
   compete for the same slot.
6. **Retire nothing now.** Adopt the source analysis's retirement gate list
   (end-to-end non-sensitive task; exportable raw rollouts + privileged state;
   digest-bound versions; deterministic reruns; ≥2 policy families ranked;
   agreement with physical anchors at a preregistered threshold; distinguishable
   provider-vs-policy failures; working deletion; acceptable cost/limits;
   contract terms permitting customer use, independent benchmarking, and
   resale; provider outage cannot corrupt authoritative evidence) into
   `docs/EXECUTION_COST_AND_ARCHITECTURE_GATES.md`, and sequence any future
   retirements: Marble-preview bridge and Isaac pixel-parity work shrink first;
   MuJoCo policy-loop lane is retained as the verification/reference lane until
   cross-provider replication exists; WAM lane follows its own already-planned
   evidence timetable; GPU fleet shrinks to sealed-policy hosting.

**If the maximal-wrapper future arrives** (provider passes all gates, terms
signed, pricing works): the source analysis's end-state architecture is
approximately right with three amendments — the adapter is a provider-class
slot, physical-anchor operations are a first-class product input rather than an
ingestion contract, and one in-house reference execution lane survives as part
of `evaluation_verification`. Blueprint's five subsystems then read:
capture_and_rights (now including calibration capture), canonical task
packages, r2s2r provider adapters (plural), evaluation verification (anchors +
reference lane + control arms + stats), and buyer package delivery — selling
Task Evaluation Runs, Policy Improvement Runs, and Post-Training Data Packages
exactly as doctrine already defines them.

---

## 8. Doctrine compatibility

No doctrine change is required to act on this audit. The announcement is the
event `WORLD_MODEL_STRATEGY_CONTEXT.md` was written for ("If world models
become easier to buy, proprietary real-site capture and product workflow should
become more valuable, not less"). The maximal-wrapper proposal, taken literally
today, would violate the single-provider prohibition and the capture-truth
hierarchy; taken as a gated end state with this audit's amendments, it fits.
The one doctrine-adjacent addition worth making when action item 2 lands:
extend the data-priority list to include robot-side calibration capture fields,
since "collect data now as if future training and evaluation depend on it" now
demonstrably includes robot-in-the-loop task data.

Housekeeping noticed during the audit: `BlueprintCapture`'s shared doctrine
blocks are a stale older copy (missing Policy Improvement Runs, the wedge
overlay, and the Cosmos 3 paragraph; `VISION.md:343-345` records the mirror as
pending). Any capture-side calibration work should land after that mirror
sync, per the `docs/DOCTRINE_PRECEDENCE.md` most-recently-edited rule.
