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
  -> ADP-010 partner admission
  -> ADP-020 protocol freeze
  -> ADP-030 partner capture and registration
  -> ADP-040 candidate MSER qualification
  -> ADP-050 policy/runtime integration
  -> ADP-060 sealed simulation decision
  -> ADP-070 physical holdout
  -> ADP-080 outcome join and verdict
  -> ADP-090 case study and reuse decision
```

Partner discovery and protocol conversations may run in parallel as a small
human lane, but engineering does not wait for partner capture. Until ADP-008
passes, nearly all implementation effort stays on the public-reference harness.
New capture/reconstruction feature work stays at zero unless a measured blocker
identifies the smallest missing measurement.

ADP-002 through ADP-008 must converge into ADP-040 through ADP-080 rather than
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
-> metric -> validity status -> digest-bound episode receipt
```

Acceptance: replay produces the same normalized receipt or an explicit
non-reproducibility failure; candidate self-reported success cannot grade itself.

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

## Partner Proof: Day 14

### ADP-020 — Preregister the experiment

Freeze the decision, candidates, task distribution, conditions, reset, outcomes,
minimum meaningful difference, power, partitions, randomization/interleaving,
blinding, exclusions, invalid region, stop rule, and amendment policy.

Acceptance: partner task owner, physical holdout custodian, and Blueprint approve
the same digest.

### ADP-021 — Capture the workcell

First accept and qualify the partner's existing capture, CAD, calibration,
simulator scene, and task assets behind the same source contract. Collect a new
guided iPhone Pro Raw Contract V3.2 walk or any additional probe only when the
task's observation, scale, collision, registration, or dynamics gate identifies
a specific missing measurement that existing inputs cannot supply.

Acceptance: immutable bundle with decoded-time alignment, intrinsics, poses,
depth/mesh where available, rights/provenance, and task-specific capture context.

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
