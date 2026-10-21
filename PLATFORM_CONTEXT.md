# Platform Context

> Current owner direction (2026-10-05, recorded in AGENTS.md; owner record `gs://blueprint-8c1ca.appspot.com/operations/recovery/2026-10-05/owner-decisions/owner-decision-demand-discovery-20261005.json`): **demand discovery is open.** Prospecting, screening, outreach, and partner discovery span every robot form (fixed arm, humanoid, wheeled, bimanual, and others) and task family, deformables included. Arm Decision Proof v1 stays as the first evaluation-proof track; its fixed-arm, rigid-object envelope is the current default build target, not a ban on other forms. Building evaluation support for a new form or task family needs a recorded owner focus decision based on demand evidence; it does not have to be earned by this proof first. Read "Sole Active Program" and the robot-form and task-family items of "Frozen Work" below as limits on build-out, not on demand discovery; the other frozen items are unchanged. Spend stays inside approved allowances; sending and automatic first contact stay off. The canonical shared doctrine below incorporates the current owner direction through the three-repo sync.

> Current owner direction (2026-08-13, recorded in AGENTS.md): **artifixer3D+ with gpt-image-2** is the appearance path. AuraFusion360 and Inpaint360GS are retired execution lanes; older method-selection passages below and in the frozen north-star v3 contract are historical context, not launch requirements. Use the sealed public scene for the development-only two-candidate Franka rehearsal, then the existing fresh Raw V3.2 capture path. The historical 7/10 index remains unchanged. See [current lane reachability](docs/arm_decision_proof_v1/LIVE_LANE_REACHABILITY.md).


<!-- SHARED_PLATFORM_CONTEXT_START -->
## Shared Platform Doctrine

### Active Programs

Blueprint has two active programs. Both bind every repository, and neither
loosens the engineering invariants in this document.

1. **Free Beta: Job-First Automation** is the customer-facing program, updated
   by the owner on 2026-10-21. The historical program identifier
   `free-beta-evaluated-pilot` remains readable for existing records.
2. **Arm Decision Proof v1** (`arm-decision-proof-v1`) is the evidence program.
   It tests whether a simulated decision agrees with held-out physical trials.

No other research, product, growth, infrastructure, provider, embodiment, or
market-expansion lane is active unless an observed blocker in one of these two
programs shows that it is the smallest necessary dependency.

Shared seams are built once and serve both programs: capture, scene
preparation, policy execution, scoring, and result publication. When the two
compete for the same engineering time, work that a site or suitable partner
is blocked on comes first, and the founder resolves anything else. Arm
Decision Proof keeps sole authority over physical-validation claims. No beta
result may be described as physically validated without Physical Outcome Join
evidence.

### Free Beta: Job-First Automation

"Find the right way to automate your work." A site records or uploads its job
video; a written description is not required. The later owner correction on
2026-10-21 requires actual retained video before assessment. A draft or inquiry
may wait for video, but a description, checkbox or allocated upload ID cannot
substitute for a successful storage receipt. Existing video analysis drafts the
description and structured job facts from the same output; the customer corrects
the reading and supplies material facts the footage cannot establish.
Preserve customer-authored or confirmed text and source-bound revisions. Keep
observed, measured, customer-declared and inferred values distinct; estimated
dimensions do not become authoritative hardware exclusions. Retain metric units,
coordinate frames and calibration across geometry, objects and runtime inputs.
Visual appearance alone does not establish contact, dynamics or sensor fidelity.

The six service stages reuse the existing capture-to-simulation foundation:
retain the original video, permissions and job identity; understand and confirm
the job; prepare the capture-backed site package and measurable task testbed;
shortlist complete executable configurations; run matched evaluations with
independent scoring; and prepare an evidence-bounded recommendation and real
delivery proposals. Reuse valid source-bound artifacts and applicable permitted
evidence instead of repeating unchanged work. Missing measurements, unavailable
execution or unsuitable options return their precise next action or abstention;
they cannot be presented as completed simulation, execution or delivery proof.
The evidence bridge's read-only adapters are not policy evaluation.

This replaces the superseded optional-video interpretation, not the original
walkthrough or approved example animations. Keep the experience bare bones:
video, review the reading, understandable progress/results and comparable
delivery options. Use the existing private workspace and admin tools. After
recommendation, the customer chooses an exact real proposal revision; the
customer and deployment partner agree their bounded physical pilot directly.
Selection is not contract award or authority to start physical work.

- The beta is free. There are no assessment or success fees, paid provider
  evaluations, balances, top-ups, checkout, escrow, percentage cut, or future
  pricing promises. Partner installation and operating prices concern their
  physical delivery, not a Blueprint charge.
- Capture current process cost separately from desired future spending. An
  optional planning budget is neither a supplier price nor resource-spending
  authority. Firm proposals follow the agreed scope and acceptance criteria.
  Prefer two or three comparable proposals when real suppliers exist; one
  credible option is sufficient. Revisions retain their brief and scope.
- Deployment partners explicitly commit equipment, integration, installation,
  preparation, assistance, timing and support. Technology providers may supply
  policies or integrations without offering deployment. Reviewed applications
  do not establish runtime admission or verified delivery coverage.
- Blueprint can execute accessible licensed policies through existing approved
  self-hosted, provider-API or hosted-endpoint adapters. A technology team can
  supply an approved endpoint or integration when access is missing; operating
  every evaluation themselves is not mandatory. Preserve independent scoring,
  controlled observations, exact runtime identities, retained policy inputs,
  rights, authorized spending, watchdog and teardown controls.
- A scientific result keeps its question, candidate/site/task/configuration
  scope, assumptions, uncertainty, provenance and claim ceiling. It does not
  choose a commercial proposal. The operator's recommendation also considers
  installed and operating cost, assistance, disruption, timing and support.
- A separately permitted customer-reported pilot outcome remains reported
  evidence. Only authoritative physical records with the required exact
  identities and rights can enter the existing Physical Outcome Join.

Use the existing owner workspace and operator workflow. Keep six commercial
stages: `job_submitted`, `brief_agreed`, `options_assessed`, `proposals_ready`,
`pilot_agreed`, and `results_recorded`, each with a next action and responsible
actor. Do not add review stages, dashboards, certifications or approval loops.
Existing signed evidence publication remains separate from commercial state.
Budget-only quote revisions do not invalidate unchanged observations; task or
configuration changes require reassessing technical applicability. Stale
evidence stays historical. Neither assessment nor publication may award a
contract, invent an agreement, send outreach, or start a physical operation.

The historical metric `evidence_backed_recommended_pilots_booked` belongs to
the earlier journey; it is not proof that a selected proposal was agreed or
delivered. The useful beta outcome is a real job with an agreed scope, a
supportable customer choice, a separately agreed pilot, and recorded outcomes.

Accept jobs across sites and task families into truthful capability routing.
The first runnable route is the exact admitted fixed-arm/DROID-compatible
rigid-object profile, not every arm or task. An articulated open/close fidelity
template does not establish an executable adapter, reset or scorer. Atlas and
other unavailable embodiments or policies remain unavailable without blocking
supported configurations. New task families need real assets, adapters,
resets, scoring and evidence; registry labels and fixture names cannot supply
them. Keep task requirements, site data, embodiment/tool/sensors, policy
interfaces and delivery availability separate behind existing contracts.

### Durable Job Workflow

Site, job, task, capture and tenant identities are data, never demo constants.
Derive stage completion from retained receipts and artifacts. Reuse existing
stores, queues, intent/outbox records, idempotency keys, claims and leases; do
not create a parallel workflow platform or state database. At-least-once events,
worker restarts and signed publication retries must not duplicate expensive
work, proposals or sends. Persist intent before effects, reconcile ambiguous
provider timeouts before another purchase, and resume from the last valid
artifact with bounded retry/backoff and the existing retry/error surface.

Bind video, technical brief, testbed, configuration and result revisions.
Changed technical sources cannot inherit a stale recommendation; retain late
old results as history. Quote or budget edits do not rerun unchanged science.
Cancellation, withdrawal and permission revocation stop applicable future work
while preserving existing minimum audit obligations. Owner links and staff
authority remain distinct, and tenant evidence remains isolated. Retain bounded
summaries/references in the existing database and large immutable artifacts in
authorized company object storage, not chat sessions or provider file IDs.

Executable technology access, deployment capability and a partner's scoped
proposal are separate facts. Record exact robot/tool/sensor/controller,
observation/action units and normalization, policy/runtime versions and access
terms. Research rankings do not prove compatibility, stock or delivery support.
Only actual partner availability, cost and support commitments make a proposal
theirs. Unknown cost, assistance and cycle time stay unmeasured or unknown;
simulation time is not physical end-to-end cycle time. Existing rights, spend,
founder holds, watchdog, teardown and no-send controls remain intact.

### Arm Decision Proof v1

Its north star is:

> After qualifying every reusable service seam on a digest-bound,
> rights-cleared public evidence ladder—including one metric 3DGS/collision
> object removal, released-code background completion, and exact SimReady USD
> replacement—take one fresh capture of one previously unseen fixed-arm
> workcell, prospectively decide which of exactly two frozen candidates deserves
> the next scarce physical-test budget or abstain, and adjudicate that decision
> and one predicted failure boundary with randomized held-out physical trials.

The north-star metric is
`prospectively_physically_validated_new_site_task_decisions`: current `0`, next
target `1`.

ADP-008's SIMPLER decision-harness replay is observed complete and immutable.
The recorded earlier engineering phase was **ADP-009 public scene qualification**:
exact dataset/code/rights admission; an unchanged Inpaint360GS author-data
reproducibility smoke; an InFusion primary adapter and AuraFusion360 quality
challenger on one exact rights-admitted InteriorGS/SAGE-3D pair using a frozen
render-derived camera bundle; exact SimReady USD replacement;
NVIDIA USD Content Agents as a bounded authoring candidate against a manual USD
control; targeted ScanNet++ real measured transfer after access; hybrid Isaac
tests; and a complete simulator-side rehearsal. Its outputs remain
`development_only`. This historical method sequence is superseded for execution
by the current lane-reachability record described below.

After ADP-009 and partner protocol freeze, collect one fresh Raw V3.2 workcell
capture with registered clean-background and object-present observations. That
is the final new construction input, followed by replica qualification,
two-candidate execution, sealing, held-out physical trials, and the Physical
Outcome Join. New capture feature development remains zero unless a measured
gate proves the smallest missing measurement. Partner discovery, protocol,
rights, and physical access remain a small parallel human lane.

### Product And Artifact Boundary

The customer-facing service helps a business find a suitable, deliverable way
to automate its job. The **Task Evaluation Run** is its bounded technical
assessment, not a mandatory purchase or a substitute for retained video.
An inquiry awaiting footage needs no invented capture, testbed or run and
cannot advance through assessment before its real video is retained.

- A maintained **Site-Task Testbed** is the versioned reusable substrate.
- A **candidate Minimum Sufficient Evaluation Replica** is the construction
  method for this program. It is not proven sufficient before physical
  adjudication.
- A **Physical Outcome Join** binds the prospective result to authoritative
  held-out physical outcomes.
- **SiteBench** may name the bounded public case study. It is not a second
  product, general benchmark platform, or digital-twin offer.

A run returns a bounded selection, elimination, equivalence/inconclusive result,
partial decision, explicit abstention, or the smallest missing measurement. It
does not guarantee ranking, deployment, physical success, or safety.

### Arm Decision Proof v1 Scope

The first proof is deliberately limited to:

- one admitted design partner;
- one previously unseen workcell;
- one fixed robot arm;
- one partner-owned bounded rigid-object pick-and-place task;
- fixed cameras, rigid opaque objects, and a parallel-jaw gripper;
- two genuine runnable frozen policy checkpoints or configurations;
- a small owner-approved condition distribution and repeatable reset;
- one prospective decision and one predicted failure boundary;
- one randomized or interleaved held-out physical trial set using the same
  condition IDs.

The task, candidates, minimum meaningful difference, protocol, partitions,
decision rule, code, assets, and evidence are frozen before holdout release.
Two candidates may establish one bounded decision; they cannot establish general
rank correlation.

### System Responsibilities

- `BlueprintCapture` records immutable, provenance-linked real-site evidence.
- `BlueprintCapturePipeline` compiles the testbed, routes claims to qualified
  evidence, qualifies methods, executes authorized leaf runs using supported
  independent or team-supplied access, seals the prospective decision,
  performs the Physical Outcome Join, and owns normalized scientific results,
  uncertainty, abstention and claim ceilings.
- `Blueprint-WebApp` handles site and robot-team intake, consent, invitation,
  authorization, commercial workflow state, briefs, delivery proposals,
  customer selection, direct pilot agreement records, reported outcomes and
  evidence display. It does not recompute verdicts, raise claim ceilings or
  turn commercial screening into scientific evidence.
- Agents may inspect, propose, implement, validate, and assemble artifacts.
  Deterministic contracts authorize or abstain. Humans retain authority over
  task truth, resets, rights/privacy, physical robot motion and safety, holdout
  release, publication, and claim promotion.

### Existing Capture And Scene Policy

The public-scene method list below records the earlier ADP construction plan.
The owner-selected appearance path is artifixer3D+ with `gpt-image-2` as recorded
on 2026-08-13; Inpaint360GS and AuraFusion360 are retired execution lanes. Fresh
Raw V3.2 capture replaces ScanNet++ as the required real metric transfer.
Follow the current ADP lane-reachability record for execution. Preserve the
historical `7/10` suite and frozen proof definitions.

Existing captures, exact public datasets, fixtures, released editing code,
OpenUSD scenes, SimReady candidates, and simulator artifacts must exercise every
seam that does not require new physical truth. This is one phase of Arm Decision
Proof, not a separate fixture or editing product.

SIMPLER remains the pinned decision/outcome reference. Exact InteriorGS plus its
matching SAGE-3D scene is the first synthetic metric-frame appearance/static-
collision editing control after explicit rights; an authored scene is a
separate physics positive control and cannot substitute for that dataset gate.
Because the published pair lacks the original training cameras and a local
measurement-authoritative surface, Blueprint may fly frozen virtual cameras,
render object-present RGB and method-native splat depth/disparity, and emit
digest-bound intrinsics and poses without rerunning unscaled SfM. External
metric depth/geometry and clean-background truth stay out of method inputs.
Calibration and test trajectories are synthetic self-consistency evidence,
never independent capture truth. ScanNet++ is the second and only public real
measured transfer after access. Inpaint360GS is the unchanged author-data
reproducibility control; InFusion is the primary world-frame supplemental-Ply
adapter; AuraFusion360 is the multiview quality challenger; GOR-IS is a
conditional glossy/shadow ablation. NVIDIA USD Content Agents may convert and physicalize
already-authoritative geometry, but cannot
create measurement authority. Paper-only methods cannot enter the critical
path.

They may validate contract shape, metric frames, source-object partition and
removal, visual background completion, exact SimReady USD insertion, static and
dynamic collision/contact checks, deterministic variation, testbed compilation,
two-candidate matrix planning, condition IDs, reset/evaluator plumbing,
policy/runtime adapters, closed-loop execution, episode receipts, replay,
decision sealing, holdout separation, Physical Outcome Join mechanics,
aggregation, abstention, and report rendering.

They remain `development_only` and cannot qualify the fresh partner capture,
owner task truth, robot/camera/workcell registration, real task-specific
dynamics, observation-domain match, factual recovery of an unobserved surface,
sim-to-real decision fidelity, or partner value. No fixture/generated field may
be copied into qualified evidence to make a gate pass.

Beta site captures are required real inputs to assessment. Qualified simulation
results can contribute to a recommendation within their stated
task/configuration/site limits. Reusing applicable evidence does not establish
a newly completed reconstruction or evaluation. Those results stay
separate from Arm Decision Proof evidence. They cannot qualify a partner capture, promote a
physical-validation claim, or be copied into Arm Decision Proof qualified
evidence.

### Truth Hierarchy

1. Raw capture, timestamps, poses, device metadata, rights/privacy records, and
   provenance are authoritative for captured truth.
2. Owner-approved task distribution, reset, outcome, and invalid-condition
   records are authoritative for task truth within their stated scope.
3. Accepted physical trial logs and exact join identifiers are authoritative for
   physical outcomes.
4. Site/Task/Scenario/Eval Cards, replicas, geometry, generated appearance,
   SimReady assets, simulations, learned evaluators, provider outputs, reports,
   and hosted views are derived artifacts with independent claim ceilings.

Derived artifacts never rewrite higher-authority truth. Realistic appearance,
successful loading, completed execution, or provider status does not establish
collision validity, physical fidelity, task success, deployment, or safety.

### Evidence And Result Contract

Every scientific decision exposes the applicable fields, with absent physical
evidence left absent rather than invented:

- the requested decision and minimum meaningful difference;
- exact site, task, robot, candidate, condition, reset, and runtime identities;
- selected evidence methods and why they qualified;
- measurements, coverage, uncertainty, invalid trials, disagreements, and known
  unsupported conditions;
- the sealed pre-holdout result;
- exact physical outcome joins;
- the final agreement, contradiction, or abstention;
- the correctly or incorrectly localized failure boundary;
- the next smallest measurement;
- exact artifact versions and digests;
- the partner-value verdict and reuse estimate.

An honest abstention is valid behavior but does not complete the showcase target.
A wrong prospective decision is a failed experiment and remains immutable
evidence.

### Arm Decision Proof Partner And Economic Gate

The first partner must bring a real testing decision, two runnable candidates,
a bounded task, holdout authority, a named operator/task owner, documented
rights, and a credible reuse or physical-scarcity reason.

The technical result alone is insufficient. The program must measure whether the
partner changed or reduced its next physical-test allocation, how much manual
authoring and compute the replica required, and whether direct physical testing
would have been cheaper. If reuse economics are not supported, Blueprint stops
or considers a bounded capture/testbed-preparation service rather than expanding
the platform.

### Compatibility

Historical schemas, evidence, URLs, records, entitlements, readers, paid-resource
safety gates, and provider adapters remain readable and secure. They are not
active product surfaces or roadmap authority. Legacy behavior is preserved by
explicit translation or deprecation rather than silent reinterpretation.

The existence of compatibility code does not authorize new work on its former
lane.

### Frozen Work

Unless one of the two active programs records a direct blocker, do not start
or expand:

- humanoid/G1 or locomotion work;
- deformable, cable, cloth, granular, insertion, or force-task work;
- five-policy campaigns or general rank-correlation claims;
- world-model/evaluator expansion;
- reconstruction/provider bakeoffs or provider marketplaces;
- universal robot, simulator, or provider support;
- dynamic-scene and general 3DGS research;
- post-training or policy-improvement products;
- cross-site generalization claims;
- city-launch or growth campaigns, and WebApp work unrelated to the beta
  journeys.

Humanoids are not automatically the next phase. A later embodiment must be
earned by demand evidence and a recorded owner focus decision.

### Practical Rule For Agents

Before accepting any work, answer:

1. Which free beta journey step, or which public-harness or partner-proof Arm
   Decision Proof gate, does it unblock?
2. What observed artifact will prove the blocker was removed?
3. Why is existing infrastructure insufficient?
4. What is the smallest reversible change?

If any answer is missing, the work is out of focus.

Prefer thin adapters over new infrastructure. Preserve raw truth and stable
contracts. Never hand-author fake evidence. Do not run paid providers, upload
data, move a robot, release holdouts, or publish externally without the existing
explicit authority. Use the smallest risk-based verification that protects the
changed claim.
<!-- SHARED_PLATFORM_CONTEXT_END -->

## Pipeline Embodiment Rule

The admitted design partner's fixed arm and working interface determine the Arm
Decision Proof embodiment. Franka/DROID assets may serve as development-only
defaults before admission, but they cannot substitute for the partner binding.
Humanoid/G1 defaults are inactive compatibility settings and do not authorize
humanoid work.
