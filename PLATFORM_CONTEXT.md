# Platform Context

<!-- SHARED_PLATFORM_CONTEXT_START -->
## Shared Platform Doctrine

### System Framing

- `BlueprintCapture` is the capture client and supply-side evidence collection
  tool. It collects immutable, provenance-linked real-site evidence.
- `BlueprintCapturePipeline` composes capture bundles, cards, evaluators, and
  reset artifacts into maintained Site-Task Testbeds, then produces claim-level
  Task Evaluation Run plans, evidence, decisions, or explicit abstentions. It
  owns the versioned Site-Task Testbed manifest, Decision/Evidence Request,
  Evidence Plan, Evidence Method Profile, normalized Evidence Result, Decision
  Envelope, and Physical Outcome Join, plus method qualification, routing,
  aggregation, and scientific verdicts.
- `Blueprint-WebApp` is the buyer, licensing, ops, and hosted-access surface for
  Task Evaluation Runs. It owns authenticated intake, request validation,
  idempotency, entitlement and authorization, durable queue/outbox state, status
  projection, artifact access, redacted presentation, and operator workflows. It
  collects constraints and displays Pipeline-owned results; it does not select
  providers or recompute scientific verdicts.
- `BlueprintValidation` remains optional downstream infrastructure for
  benchmarking, runtime checks, robot evaluation support, and specialized
  validation after the primary package or run is scoped.

This platform is capture-first and real-site decision-evaluation first. Raw
capture, timestamps, poses, device metadata, rights/privacy records, and
provenance remain authoritative. World models, site-world routes, simulation
outputs, generated media, editing assets, legacy data-package exports, and
model-derived artifacts are evidence or support artifacts inside a run. They are
not additional public offers, and they do not silently upgrade the claim.

### One Customer-Facing Product

Blueprint sells one product: **Task Evaluation Run**.

The maintained Site-Task Testbed is the reusable substrate behind runs. Robot
teams and site operators are personas using the same service, decision request,
workflow, result model, pricing concept, and call to action. Post-training is
only a permitted use of qualifying evidence inside a run; it is not a SKU,
add-on, navigation item, checkout flow, or delivery promise.

The normal customer describes the site-task, decision, candidates when
applicable, claims, thresholds, false-safe consequence, acceptable risk, budget,
deadline, available evidence, rights/privacy restrictions, and physical-testing
constraints. The WebApp does not ask ordinary users to choose a simulator, world
model, or provider.

### Product Center of Gravity

The center of gravity is:

- broad real-world capture coverage
- strong capture quality and provenance
- Task Evaluation Runs for robot teams
- maintained, immutable-version Site-Task Testbeds reused across successive runs
- claim-level routing to the cheapest currently qualified evidence method or
  combination, with explicit escalation and abstention
- rights-cleared evidence-use determinations inside a run; post-training use is
  allowed only after rights, provenance, action-alignment, quality, and leakage
  gates
- hosted access for request-scoped review
- rights, privacy, and commercialization controls
- buyer-facing product surfaces that make real sites easy to browse, buy, run,
  and manage

The center of gravity is not:

- generic marketplace browsing as the main story
- qualification/readiness as the main thing Blueprint sells
- world models are not the primary public product, and Blueprint is not a generic
  world-model marketplace
- one-off model demos disconnected from real capture
- a single permanent world-model backend

### Market Structure

The core business engine is two-sided:

- **Capturers** supply real-site evidence packages.
- **Robot teams** buy Task Evaluation Runs.

`Site operators` remain important, but they are an optional third lane at
capture time, covering:

- access control
- rights / consent / privacy boundaries
- commercialization and revenue sharing

The platform must support lawful capture and packaging even when a site has not
already gone through a pre-negotiated intake flow. Site-operator involvement is
a supported workflow branch, not a universal prerequisite for all capture.

Optional at capture time is not the same as unimportant at adoption time. The
long-horizon direction (see `VISION.md`, rung 2) is that site operators become
the demand-side channel that routes deployment decisions through Blueprint
evaluation — requiring a Task Evaluation Run before a robot reaches their floor.
That is a strategic adoption bet about where the standard gets enforced, not a
current capture prerequisite; the two statements describe different lifecycle
stages and do not conflict.

### Decision And Evidence Router

Pipeline routes every decision-relevant claim to the least expensive currently
qualified combination of fixture data, geometry, real observations, traditional
simulation, world models, provider tools, and physical evidence. It escalates
only when stronger evidence is required.

A valid run outcome may be:

- bounded positive;
- bounded negative;
- elimination of an incompatible candidate;
- partial decision;
- explicit abstention;
- blocked or failed;
- a request for the next evidence needed.

A run does not guarantee ranking, shortlist, winner, deployment, pilot
readiness, physical success, or safety approval. Unknown future states fail
closed. An abstained result never implies a winner from raw scores.

### Truth Hierarchy

- raw capture, timestamps, poses, device metadata, and provenance are
  authoritative
- rights / privacy / consent metadata are authoritative
- Site Cards, Task Cards, Scenario Cards, Eval Cards, package manifests,
  generated/model-derived support assets, and hosted-session artifacts are
  downstream artifacts with explicit proof boundaries
- Task Evaluation Run is the one primary sellable downstream product
- maintained Site-Task Testbeds are reusable substrates, not a second product
- post-training is a permitted use of qualifying run evidence, never proof that
  training occurred or a policy improved
- qualification / readiness / review outputs are optional trust layers that can
  guide buying, commercialization, and deployment decisions
- downstream outputs must not rewrite capture truth or provenance truth

### Result Contract

Buyer-facing results expose the requested decision, per-claim outcomes, selected
methods and selection reasons, measurements, validation envelope, unsupported
conditions, coverage, uncertainty, disagreements and correlated-evidence
warnings, claim ceiling, next cheapest experiment, physical-evidence
requirements, cost/time when available, exact artifact versions and digests, and
permitted evidence uses.

An evidence export does not prove training happened or a policy improved.
Physical outcome ingestion requires authoritative evidence and exact join
identifiers; a user note alone cannot recalibrate a method.

### Product Stack

1. supply and truth layer: real-site capture, rights, privacy, and provenance
2. reusable substrate: maintained Site-Task Testbeds
3. single buyer product: Task Evaluation Runs
4. evidence methods: geometry, captured observations, traditional simulation,
   learned/world-model evaluation, provider tools, physical evidence, and
   bounded owner inputs
5. access and support layer: hosted review, licensing, entitlements, operator
   workflows, evidence export/use, legacy compatibility, generated/model-derived
   data, editing, and augmentation

### Commercial Wedge Overlay

The current PMF wedge is the Task Evaluation Run: a decision request bound to an
exact maintained testbed. The router decomposes the decision into claims,
selects only qualified methods, and returns a partial or complete decision or an
explicit abstention with the next cheapest experiment. A run may emit
rights-cleared evidence for later evaluation or post-training use, but the
evidence export is not another product and does not imply training or
improvement.

Wedge claims stay inside the proof boundary. The current comparative
policy-ranking scientific verdict is `thesis_not_supported`; physical success,
deployment readiness, and safety claims require separately accepted physical
evidence. Generated frames are support, never real-world proof.

### Commercial And Compatibility Rules

- One run is scoped and quoted according to decision, evidence, candidates,
  scenarios, compute, deadline, rights, and physical requirements.
- The server owns authoritative pricing. The client cannot supply it.
- No new subscription, standalone evidence package, improvement add-on, or
  vendor submission fee.
- Historical data, URLs, transactions, and entitlements remain readable through
  explicit compatibility paths.
- Legacy paid or customer-visible intent is never silently reinterpreted.
- Live provider, physical robot, deployment, payment, rights, calibration, and
  customer claims require proof from the system that owns them.

### Default Lifecycle

1. A capture is sourced proactively or through a buyer / site / ops request.
2. `BlueprintCapture` records and uploads a truthful evidence bundle.
3. `BlueprintCapturePipeline` composes a versioned Site-Task Testbed and routes a
   Task Evaluation Run at claim level through qualified evidence adapters.
4. `Blueprint-WebApp` exposes the request, plan status, decision envelope,
   abstentions, and proof-bound supporting artifacts.
5. Optional world-model, simulation, deeper evaluation, validation, or managed
   support follows only when commercially useful and proof-bounded.

### Practical Rule For Agents

When changing any Blueprint repo, optimize for:

1. stronger real-site capture supply and capture truth
2. better Task Evaluation Runs, maintained testbeds, routing, abstention, and
   learning from authoritative physical outcomes
3. stable rights / privacy / provenance contracts and stable versioned testbeds
4. secure decision intake, durable state, and buyer and ops surfaces that make
   those outputs easy to sell and use
5. optional trust, readiness, world-model, simulation, generated-data, and
   validation layers that support the product without becoming the product story

Do not assume that every capture must begin with formal site qualification.
Do not treat qualification/readiness as the universal center of the company.
Do not overstate world-model quality beyond what capture, privacy, and runtime
artifacts support.
Do not move scientific routing or scoring into WebApp.
<!-- SHARED_PLATFORM_CONTEXT_END -->
