# Arm Decision Proof v1 Implementation Backlog

This is the sole active backlog. Existing issues and documents outside this file
are compatibility, historical evidence, or paused ideas unless an item below
links to them as the smallest way to remove an observed blocker.

## Operating Rules

- Work critical-path order; do not optimize parallel lanes.
- Use existing code, fixtures, runtimes, and provider-neutral contracts before
  adding infrastructure.
- Never hand-author evidence to satisfy a qualification gate.
- Every artifact binds source digests, versions, condition IDs, and claim ceiling.
- Agents propose and assemble; deterministic code authorizes or abstains.
- Humans own task truth, resets, rights, safety, physical execution, and holdout
  release.
- No paid compute, provider upload, robot motion, or external publication without
  its existing explicit authorization.
- A failed experiment is retained. There is no automatic retry or verdict edit.

## Critical Path

```text
ADP-001 focus lock
  -> ADP-002 public reference admission
  -> ADP-003 through ADP-007 complete harness seams
  -> ADP-008 one-command retrospective harness qualification
  -> ADP-009 public-scene inpainting and SimReady replacement qualification
  -> ADP-010 partner admission
  -> ADP-020 protocol freeze
  -> ADP-021 fresh partner capture
  -> ADP-022 robot/site/task registration
  -> ADP-030 task-truth and condition compilation
  -> ADP-040 candidate MSER qualification
  -> ADP-050 policy/runtime integration
  -> ADP-060 sealed simulation decision
  -> ADP-070 physical holdout
  -> ADP-080 outcome join and verdict
  -> ADP-090 case study and reuse decision
```

Partner discovery and protocol conversations may run in parallel as a small
human lane, but engineering does not wait for partner capture. ADP-008 is
observed complete. Until ADP-009 passes, nearly all implementation effort stays
on exact public-scene admission, metric-frame preservation, 3DGS
removal/inpainting, SimReady USD replacement, hybrid-scene qualification,
variation, abstention, and full simulator-side rehearsal. Fresh capture feature
work stays at zero; use only the existing Raw V3.2 capture path unless a measured
ADP-009 or partner blocker identifies the smallest missing measurement.

ADP-002 through ADP-009 must converge into ADP-040 through ADP-080 rather than
becoming a separate fixture product.

## Foundation: Focus And Harness Readiness

### ADP-001 — Enforce the sole-focus contract

Deliverables:

- machine-valid north-star contract and schema;
- canonical active-document index;
- shared doctrine, agent instructions, and onboarding map aligned;
- deterministic test that fails if the focus contract, active docs, or required
  evidence boundaries drift.

Acceptance: one command validates the contract and canonical focus references;
legacy readers remain intact.

### ADP-002 — Admit one public reference substrate

Audit and pin SIMPLER first, following
[`PUBLIC_REFERENCE_SUBSTRATE.md`](PUBLIC_REFERENCE_SUBSTRATE.md). Bind exact
repository/submodule commits, assets, licenses, two policies/checkpoints, one
rigid task, public physical-reference outcomes, control/evaluator semantics, and
compute requirements.

Use local fixtures only to isolate Blueprint-owned seams:

- `decision_evidence_rigid_object_v1` for routing and outcome learning;
- `new_site_loading_bay_v1` for compiler shape;
- `kitchen_task_min` only for existing USD/runtime plumbing;
- at most one existing SimReady/OpenUSD candidate only if it exercises a missing
  runtime seam that SIMPLER and the committed fixtures cannot.

Deliverables:

- exact public source digests, submodule pins, assets, and licenses;
- feature-to-fixture coverage table;
- explicit fixture claim ceilings;
- selection of one SIMPLER task/runtime scene, not a bakeoff;
- two genuine public policy/checkpoint identities and the external physical
  performance source.

Acceptance: every selected asset has a reason tied to a later partner seam; the
public corpus is labeled `retrospective_external_reference`; no fixture appears
as qualified real-site evidence.

### ADP-003 — Generalize the matrix from exactly five candidates to two-or-more

Preserve existing five-policy compatibility. Add a generic path driven by the
candidate list in the Evaluation Run request.

Deliverables:

- two-candidate planning and execution fixtures;
- stable candidate/condition IDs;
- compatibility translation for existing five-policy packets;
- tests rejecting fabricated padding candidates.

Acceptance: the rigid-object vertical slice plans two genuine candidates without
requiring three fake identities.

### ADP-004 — Complete the condition-to-episode receipt seam

Wire:

```text
condition cell -> environment reset -> policy query -> simulator steps
-> independent environment metric -> lossless policy-input frames
-> derived human-review video -> validity status -> digest-bound episode receipt
```

Acceptance: replay produces the same normalized receipt or an explicit
non-reproducibility failure; candidate self-reported success cannot grade itself;
every completed episode includes decodable lossless policy-input frames, a
terminal observation, a frame manifest that reproduces the observation-trace
digest, and a digest-bound review video. A missing or changed media artifact
invalidates the new execution version rather than silently degrading review.

### ADP-005 — Enforce calibration/holdout separation and decision sealing

Deliverables:

- immutable partitions;
- holdout custodian/release receipt;
- sealed decision digest before release;
- rejection of late policy, metric, threshold, condition, or reset changes;
- append-only amendment path that creates a new experiment version.

Acceptance: a test proves that seeing or joining a holdout outcome before sealing
cannot produce a qualified verdict.

### ADP-006 — Power and decision-rule compiler

Inputs include baseline, minimum decision-relevant difference, alpha, power,
paired/independent design, invalid-trial handling, multiplicity, and stop rule.

Acceptance: arbitrary trial counts are rejected for qualification; the output
states the detectable effect and exact select/eliminate/inconclusive/abstain rule.

### ADP-007 — Evidence matrix and case-study renderer

Each policy-by-condition cell links source evidence, replica configuration,
simulation trace/video, metric, matched physical trial, validity, failure class,
versions, digests, and qualification status.

Acceptance: fixture data renders with a visible `development_only` ceiling and
missing physical outcomes stay missing.

### ADP-008 — One-command public-reference harness qualification

Run exactly one pinned task and two genuine public candidates through:

```text
external source manifest -> normalized EvaluationRunSpec
-> condition matrix -> closed-loop receipts -> sealed development decision
-> programmatic release of external physical-reference outcomes
-> exact join -> bounded verdict -> evidence matrix and replay
```

Acceptance:

- the command is hermetic apart from explicitly pinned external inputs;
- outcome labels are inaccessible to the execution/decision process until after
  sealing;
- the result is labeled retrospective and `development_only`;
- a contradiction, missing condition, or invalid trial remains visible;
- another engineer can reproduce it from the immutable manifest;
- no Blueprint capture/reconstruction feature was added unless a recorded
  blocker proved it necessary.

Harness clock: public source pinned by day 7, two-candidate receipts by day 14,
sealed/joined result by day 21, complete rerun and evidence matrix by day 28.

## Public Scene Qualification: Day 7 Through Day 28

Follow [`PUBLIC_EVIDENCE_LADDER.md`](PUBLIC_EVIDENCE_LADDER.md). This is an
additive qualification phase; it cannot rewrite ADP-008 evidence.

### ADP-009A — Admit exact datasets, code, and assets

Record exact, rights-reviewed scenes and revisions for:

- one matched InteriorGS PLY/metadata, SAGE-3D USDZ, and SAGE collision scene,
  with exact rights admission; an authored positive control is separate and can
  never substitute for this pair;
- one exact targeted ScanNet++ real measured scene after gated access/terms
  acceptance; an access/rights blocker leaves ADP-009C incomplete;
- one exact Inpaint360GS code/weights/dependency graph and unchanged author-data
  reproducibility smoke;
- one exact InFusion revision as the primary Blueprint interface-adapter
  candidate and one exact AuraFusion360 revision as the 360-quality challenger;
- 3DGIC, GPGS, and GOR-IS only as separately preregistered conditional research
  ablations, subject to official-code, dependency, and rights admission;
- one exact SimReady USD replacement object;
- NVIDIA USD Content Agents v0.5.2 as a candidate authoring backend after
  deterministic geometry exists; SimReadyGen remains a comparison only if
  separately admitted.

Deliverables:

- `public_scene_suite_manifest.v1` schema and fail-closed component validator;
- one component manifest and receipt per coherent executable case, plus a suite
  index that binds the required component roles and is the only artifact allowed
  to claim the ADP-009A matrix is complete;
- source units, handedness, up axis, normalization history, partitions, artifact
  roles, allowed use, exact revisions, hashes, sizes, and claim ceilings;
- released-code smoke receipts; paper-only methods remain inadmissible.

Observed 2026-08-06: the matched InteriorGS/SAGE-3D `840313` roles, the bounded
NVIDIA USD Content Agents authoring comparison, AuraFusion360's unchanged
author-data smoke, and the Blueprint-controlled native Isaac/PhysX positive
control are admitted. The exact project-owner-approved match-v2 SimReady can is
also admitted after its static profile and four native Isaac probes passed. The
ten-role index remains blocked at six admitted roles
because the Inpaint360GS author smoke, InFusion primary adapter,
controlled-background truth, and ScanNet++ transfer are still missing. The Aura author
smoke is not an InteriorGS result, and the Content Agents execution is not an
inpainting-method smoke.

Acceptance: missing/expired rights, mismatched scene IDs, changed digests,
unknown frames, calibration/test-trajectory overlap, DA3-as-scale-authority, paper-only
code, or claim elevation fails closed with the smallest blocker.

Day-7 gate: exact mandatory inputs and one physics-authored positive control are
admitted. An InteriorGS or matching SAGE rights blocker stops the required gate.
An authored scene may still run as a labeled positive control, but it cannot
substitute for the exact public pair. A runnable-code gap likewise stops the
phase rather than authorizing an unlabeled substitute.

### ADP-009B — Synthetic removal, inpainting, and replacement

On one exact rights-admitted InteriorGS/SAGE-3D scene:

1. bind appearance, semantics, collision, and their shared metric frame;
2. select one rigid source object and its exact collision body;
3. conserve every Gaussian in background/object/uncertain partitions;
4. remove the object from both appearance and collision;
5. freeze collision-aware calibration and test trajectories; render
   object-present RGB from the publisher splat and bind exact cameras as
   `render_derived_synthetic_method_inputs`; keep external metric depth separate
   as a validation oracle unless an admitted method explicitly accepts it and
   the substitution is preregistered; never describe SAGE collision depth as a
   measurement-authoritative surface; withhold all clean-background truth until
   the completion digest is sealed;
6. run the pinned Inpaint360GS revision only on its unchanged author data as a
   reproducibility control; run the exact InFusion method-native
   incomplete-splat-depth/c2w/intrinsics to supplemental-Ply to
   original-Ply-composition workflow through a verified format/license/frame
   adapter; run AuraFusion360 through its separate
   representation/checkpoint adapter as the 360-quality challenger;
7. insert one digest-bound SimReady USD with distinct visual/collision meshes,
   dimensions, pose, mass, center of mass, inertia, friction, and restitution;
8. load and probe the composed scene in Isaac Sim;
9. compare a known-good manually authored object, deterministic parametric
   CAD plus pinned NVIDIA USD Content Agents, and SimReadyGen only if admitted;
   keep CAD/mesh generation independent from VLM authoring.

Acceptance: no source-object visual or collision ghost remains; the replacement
exists exactly once; transforms round-trip; frozen test-trajectory rendering,
visual/collider alignment, support, penetration, contact, drop/settle, slide/tip, and gripper
probes pass the preregistered task-derived tolerances. Generated background is
`visual_candidate_only`.

InFusion's hidden depth and every other generated completion remain plausible,
not factual. Its explicit original/supplemental PLY composition earns only the
claims actually demonstrated by the format and frame adapter. AuraFusion360
starts from its own trained/checkpoint representation and cannot claim to edit
the publisher PLY in place without a separately tested adapter.

Day-14 gate: the exact synthetic replacement case is independently replayable.

### ADP-009C — Real capture transfer, metrology, and factual-background benchmark

After exact ScanNet++ access and rights are admitted, train or import one exact scene without
discarding the official laser-aligned metric camera model. Preserve its DSLR
images, iPhone RGB-D where useful, laser geometry, native units, and transforms.
Measure from laser mesh or observed depth, never Gaussian extent. Run the same
frozen selection/removal/inpainting/replacement checks without silently tuning
the method or thresholds to the real scene.

Separately run a known-background firebreak: withhold true RGB/depth for an
observed region, seal the completion result, then release and score factual
recovery. Ordinary hidden ScanNet++ regions can earn only visual-plausibility
evidence when no object-free observation exists.

Acceptance: the receipt reports frozen test-trajectory
RGB/depth/surface/boundary metrics, view-subset uncertainty, edit locality, and
every unsupported region. DA3 and SAM
remain proposal/cross-check aids.

### ADP-009D — Deterministic variation, abstention, and full rehearsal

Use the admitted InteriorGS/SAGE control and the rights-admitted ScanNet++
transfer plus deterministic mutations. Test scale/unit errors, handedness/up
swaps, pose drift, missing/noisy depth, view coverage, mask errors, object
size/pose, collider offsets, mass/inertia, friction/restitution, appearance, and
lighting. Use one-factor diagnosis, pairwise coverage, and task-model
interactions with fixed seeds.

Acceptance:

- each case preserves the bounded output inside tolerance or emits a typed
  abstention naming the smallest missing measurement;
- `x0.1`, `x10`, centimeter/meter, scene-ID mismatch, missing clean background,
  and tampered-digest faults never pass silently;
- one command reproduces the admitted scene matrix, editing and replacement
  receipts, hybrid Isaac qualification, complete simulator-side two-candidate
  Task Evaluation Run, lossless policy-input media, evidence matrix, claim
  ceilings, and replay;
- variation results are labeled construction robustness, not multi-site or
  general policy-ranking evidence.

Day-21 gate: the exact real measured transfer, known-background scoring, and the
variation matrix are complete. A ScanNet++ access/rights blocker stops this gate;
it is not a passing abstention. Day-28 gate: the full public-data rehearsal is
reproducible.

## Partner Proof: Day 7 From Partner-Phase Start

### ADP-010 — Select one design partner

Use [`PARTNER_SELECTION_PACKET.md`](PARTNER_SELECTION_PACKET.md). Do not code for
a hypothetical robot before admission. This human lane may proceed while the
public-reference harness is being completed, but it does not redirect harness
engineering.

Acceptance: score at least `20/24`, no zero in a required row, and obtain durable
authority for capture, protocol, holdout, outcome use, and bounded case study.

### ADP-011 — Freeze the partner integration surface

Record the robot, gripper, cameras, policy runtime, observation/action schemas,
control rate, resets, logging, simulator already in use, and exact versions.

Acceptance: choose the thinnest adapter route. Engine/provider selection follows
the partner stack and measured gaps; it is not a platform preference.

## Partner Proof: Day 14 Protocol Freeze, Day 21 Capture

### ADP-020 — Preregister the experiment

Freeze the decision, candidates, task distribution, conditions, reset, outcomes,
minimum meaningful difference, power, partitions, randomization/interleaving,
blinding, exclusions, invalid region, stop rule, and amendment policy.

Acceptance: partner task owner, physical holdout custodian, and Blueprint approve
the same digest.

### ADP-021 — Capture the workcell

First accept and qualify the partner's existing CAD, calibration, simulator
scene, and task assets behind the same source contract. After ADP-009 passes and
the partner protocol is frozen, collect one fresh guided iPhone Pro Raw Contract
V3.2 capture session of the previously unseen workcell. Include a registered
clean-background segment, followed by object-present close views and the known
metric control required by the preregistered replacement task. Collect any
additional walk or probe only when the observation, scale, collision,
registration, background, or dynamics gate identifies a specific missing
measurement.

Acceptance: an immutable rights/provenance-bound bundle retains decoded-time
alignment, intrinsics, poses, RGB, depth/mesh where available, gravity/up, and
the known metric control. The clean-background and object-present observations
share one measured frame; missing or unobserved regions remain explicit rather
than generatively promoted.

### ADP-022 — Register robot, cameras, work surface, tray, fixture, and objects

Acceptance: independent residual and held-out checks satisfy preregistered
thresholds; otherwise request the smallest new measurement or abstain.

## Partner Proof: Day 28

### ADP-030 — Compile task truth and condition distribution

Owner-approved task/reset/outcome facts remain distinct from captured
observations. Frequencies and invalid conditions cannot be inferred from pixels.

Acceptance: every condition has provenance, reset parameters, success evaluator,
and calibration/holdout partition.

### ADP-040 — Construct and qualify the candidate MSER

Use appearance only where the policy observation requires it. Use independently
qualified simplified collision geometry for the robot, work surface, tray,
fixture, and rigid objects. Do not model the rest of the facility.

Acceptance: observation, geometry, collision, scale, placement, reset, and
runtime gates each report their own ceiling; no overall `simready` label hides a
failed component.

### ADP-050 — Integrate the two frozen candidates

Acceptance: both candidates run through the same normalized observation/action
contract, cannot access evaluator or hidden state, and emit complete execution
receipts. At least one real learned policy runs closed loop by day 28.

## Partner Proof: Day 35

### ADP-060 — Execute the preregistered simulation matrix

Acceptance: all admitted condition/candidate cells complete or retain typed
failure; no failed cell disappears from aggregation.

### ADP-061 — Seal the prospective decision

Output select, eliminate, equivalent/inconclusive, or abstain plus uncertainty,
coverage, one predicted failure boundary, invalid region, and next cheapest
measurement.

Acceptance: decision, evidence, code, environment, policies, protocol, and
artifacts share an immutable digest graph before holdout release.

## Partner Proof: Day 42

### ADP-070 — Run randomized/interleaved held-out physical trials

Humans retain robot-motion and safety authority. Record condition ID, reset
receipt, policy identity, timestamps, observations/actions, outcome, intervention,
invalidity, and raw media/log references.

Acceptance: planned trial count or preregistered stop condition is reached with
all invalid/missing trials accounted for.

### ADP-080 — Join outcomes and publish the bounded verdict

Acceptance:

- exact condition/candidate join;
- decision agreement or contradiction stated without spin;
- failure-boundary localization assessed;
- uncertainty and claim ceiling reported;
- historical simulated decision remains immutable;
- method calibration updates only through a new testbed version.

### ADP-081 — Measure partner value

Record physical trials/hours redirected or avoided, replica construction hours,
manual authoring, compute/spend, expected reuse, and the partner's actual next
testing allocation.

Acceptance: conclude `reuse_supported`, `capture_service_pivot`, `economics_not_supported`,
or `inconclusive` under a frozen rubric.

## Showcase And Next Decision

### ADP-090 — Produce the SiteBench case study

Deliver a two-minute split-screen narrative and inspectable evidence matrix.
State the exact bounded claim and every non-claim.

Acceptance: no digital-twin, deployment, safety, universal ranking, or humanoid
claim exceeds the evidence.

### ADP-091 — Decide whether the program continues

Only after ADP-080 and ADP-081 decide among:

- repeat the same method on a second fixed-arm site-task;
- offer bounded capture/testbed preparation;
- stop the thesis;
- investigate the smallest measured fidelity blocker.

Humanoid work is not an automatic next phase.
