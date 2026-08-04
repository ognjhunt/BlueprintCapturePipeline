# Arm Decision Proof v1

Status: **sole active Blueprint program**
Approved: 2026-08-03
Machine-readable contract: [`north_star_contract.json`](north_star_contract.json)

## North Star

From one qualified, rights-cleared representation of one previously unseen
fixed-arm workcell—imported from an existing scene/capture when possible and
newly captured only for measured gaps—prospectively decide which of two frozen
policy or configuration candidates
deserves the next scarce physical-test budget, or explicitly abstain, then
verify that decision and at least one predicted failure boundary with randomized
held-out physical trials.

The company metric is
`prospectively_physically_validated_new_site_task_decisions`. Its current value
is `0`; the next target is `1`.

The customer-facing product remains one **Task Evaluation Run**. A **candidate
Minimum Sufficient Evaluation Replica** (candidate MSER) is the construction
method. A maintained **Site-Task Testbed** is the reusable substrate. The
**Physical Outcome Join** adjudicates the proof. **SiteBench** is only an
optional name for the public case study.

The replica is called a *candidate* until held-out physical outcomes establish
that it was sufficient for the exact decision. Visual coherence, simulator
execution, or a completed report cannot establish sufficiency.

## Execution Strategy: Public Harness First

The north star still requires one eventual partner-specific physical proof, but
the immediate engineering program is **harness-first**.

Until the harness is complete:

- dominant engineering effort goes to accepting an already-built environment,
  normalizing the two candidates, executing the condition matrix, producing
  receipts, sealing the result, joining external physical-reference outcomes,
  and rendering the evidence matrix;
- new capture or reconstruction feature development is zero unless a measured
  harness or later partner blocker identifies the smallest missing measurement;
- partner discovery, protocol design, rights, and physical-access planning run
  as a small parallel human lane so the harness is not built around a fictional
  interface.

[`PUBLIC_REFERENCE_SUBSTRATE.md`](PUBLIC_REFERENCE_SUBSTRATE.md) selects SIMPLER
as the first candidate to pin and audit. Its public environments, policies, and
real performance tables can exercise the complete harness retrospectively. They
cannot produce the prospective north-star claim because those physical outcomes
are already public.

## Exact v1 Envelope

- one design partner;
- one previously unseen workcell;
- one fixed robot arm;
- one bounded rigid-object pick-and-place task owned by the partner;
- fixed cameras and a parallel-jaw gripper;
- two real, runnable, frozen policy checkpoints or configurations;
- a small set of owner-approved, site-grounded conditions;
- one prospective selection, elimination, or explicit abstention;
- one predicted failure boundary;
- one randomized or interleaved physical holdout using the same condition IDs.

The task must have a machine-verifiable success definition and a repeatable,
owner-approved reset. The preferred first shape is tray-to-bin or
tray-to-fixture placement with generous tolerance. The partner, not Blueprint,
supplies the real task and the candidates it would otherwise spend hardware time
comparing.

## Why Existing Captures And SimReady Scenes Are Used Now

They should be used. Waiting for the final capture would serialize work that can
be completed safely today.

Existing captures, fixtures, OpenUSD scenes, and SimReady candidates may exercise:

- request and testbed compilation;
- robot, policy, runtime, and proof adapter seams;
- scenario condition IDs and deterministic matrix generation;
- reset/evaluator orchestration;
- closed-loop execution plumbing;
- complete episode receipts and replay;
- decision aggregation and abstention;
- calibration/holdout partition enforcement;
- exact Physical Outcome Join mechanics;
- the evidence matrix and case-study renderer.

Use the smallest fixed corpus:

1. `tests/fixtures/decision_evidence_rigid_object_v1/vertical_slice.json` for
   claim routing, partial decisions, abstention, and physical-outcome versioning;
2. `tests/fixtures/new_site_loading_bay_v1` for capture-to-testbed compiler shape;
3. `tests/fixtures/kitchen_task_min` only where an existing USD/runtime artifact
   is required to exercise execution and receipt plumbing.

These are **development substrates**, not qualification evidence. They cannot
qualify:

- the new partner capture or capture application;
- the partner's task distribution, reset truth, or business decision;
- robot/camera/workcell registration;
- task-specific collision geometry, friction, compliance, latency, or dynamics;
- the partner policy's observation-domain match;
- sim-to-real decision fidelity;
- partner value, reuse, or avoided physical work.

This creates two phases of one program, not two product lanes:

```text
public and existing development substrates
  -> qualify the complete harness retrospectively and fail-closed
  -> replace fixture inputs with the partner capture and owner truth
  -> seal the prospective simulated decision
  -> adjudicate it with the physical holdout
```

No fixture datum may be copied into a production evidence field merely to make a
gate pass. Missing partner evidence must remain missing and produce the smallest
specific blocker.

## Evidence Chain

The proof is complete only when the following chain is digest-bound and
replayable:

```text
partner-owned decision and task truth
-> immutable raw capture
-> qualified frames, objects, and coordinate frames
-> candidate MSER with observation and collision claim ceilings
-> frozen condition distribution and reset protocol
-> two frozen candidate receipts
-> simulation episode ledger
-> sealed prospective decision and predicted failure boundary
-> randomized/interleaved held-out physical trials
-> exact condition-ID Physical Outcome Join
-> bounded verdict, uncertainty, invalid region, and partner-value result
```

Raw capture and authoritative physical outcomes outrank every derived artifact.
Generated appearance, candidate geometry, SimReady assets, simulation, learned
evaluators, and provider outputs retain their individual claim ceilings.

## Experimental Protocol

Before any holdout result is available, freeze:

- the decision the partner needs to make;
- the minimum practically meaningful performance difference;
- candidate identities, code, weights, prompts, and runtime configuration;
- observation/action interfaces and control rate;
- task distribution and condition frequencies;
- reset, success, partial-success, failure, timeout, and intervention rules;
- calibration conditions and disjoint physical holdout conditions;
- randomization/interleaving and evaluator-blinding procedure;
- exclusions and known-invalid region;
- statistical test, uncertainty method, power, and stop rule;
- exact decision rule for select, eliminate, equivalent/inconclusive, or abstain.

Trial count comes from the preregistered minimum decision-relevant difference.
The existing independent two-rate approximation shows why an arbitrary count is
unsafe: near a 50% baseline, 25 trials per candidate resolves only about a
40-point difference at 80% power, while 50 per candidate resolves about a
28-point difference. A paired design may improve power, but its dependence and
analysis must be frozen rather than assumed.

Two candidates can prove one bounded selection or elimination decision. They do
not establish general rank correlation or universal policy ordering.

## Acceptance

The technical showcase passes only when all of the following are observed:

1. A real learned partner policy executes closed loop in the candidate replica.
2. The simulation result is sealed before physical holdout access.
3. The physical holdout supports the same practical selection or elimination
   decision under the preregistered rule.
4. At least one site-relevant failure boundary is correctly localized at useful
   resolution.
5. Every condition, reset, execution, metric, and outcome is traceable and
   replayable.
6. The partner confirms the result changed or reduced its next physical-test
   allocation.
7. The result states uncertainty, unsupported conditions, and the exact claim
   ceiling.

An honest abstention demonstrates integrity but does not complete the showcase
target. A wrong prospective decision fails the experiment. It must be retained
as evidence and diagnosed, never relabeled as success.

## Stop Conditions

Stop or pivot rather than adding infrastructure when:

- the public reference cannot be pinned by harness day 7;
- two public candidates cannot emit complete receipts by harness day 14;
- the development result cannot be sealed and joined to programmatically
  withheld public outcomes by harness day 21;
- the one-command harness and evidence matrix are incomplete by harness day 28;
- after the partner-proof clock starts, no suitable partner, task, candidates,
  holdout authority, or rights are secured by partner day 7;
- the partner task/protocol/inputs cannot be frozen by partner day 14;
- a real partner policy cannot run closed loop by partner day 28;
- the prospective decision cannot be sealed by partner day 35;
- the physical holdout cannot be exactly joined by partner day 42;
- direct physical testing is consistently cheaper than constructing and reusing
  the replica;
- manual one-off scene authoring dominates;
- the partner would not change its testing plan;
- the physical result contradicts the simulated decision;
- repeated inconclusive results do not reveal a measurable missing input.

If the technical method works but reuse economics do not, investigate a bounded
capture/testbed-preparation service. Do not respond by building a broader
simulator platform.

## Sole-Focus Rule

Every proposed task must answer:

> What exact day-7, day-14, day-28, day-35, or day-42 blocker does this remove,
> and what observed artifact will prove that it removed it?

If it cannot answer, it is not active work.

The following are frozen unless a recorded, observed Arm Decision Proof blocker
requires the smallest possible use of them:

- humanoid/G1 and locomotion work;
- deformable, cable, cloth, granular, insertion, and force-task expansion;
- five-policy and general rank-correlation campaigns;
- world-model or evaluator expansion;
- reconstruction/provider bakeoffs and marketplaces;
- universal robot, simulator, or provider support;
- dynamic 3DGS and general scene research;
- post-training and policy-improvement products;
- multi-site generalization;
- unrelated WebApp, growth, city-launch, and public-site polish.

Historical artifacts, schemas, readers, and stable compatibility paths remain
readable. They are not active roadmap authority and do not authorize new work.

## Program Files

- [`north_star_contract.json`](north_star_contract.json) — machine-validated focus lock
- [`PARTNER_SELECTION_PACKET.md`](PARTNER_SELECTION_PACKET.md) — partner qualification and intake
- [`IMPLEMENTATION_BACKLOG.md`](IMPLEMENTATION_BACKLOG.md) — sole active backlog
- [`PUBLIC_REFERENCE_SUBSTRATE.md`](PUBLIC_REFERENCE_SUBSTRATE.md) — public harness substrate decision
- [`MASTER_GOAL_PROMPT.md`](MASTER_GOAL_PROMPT.md) — paste-ready autonomous execution goal
