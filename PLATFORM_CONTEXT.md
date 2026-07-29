# Platform Context

<!-- SHARED_PLATFORM_CONTEXT_START -->
## Shared Platform Doctrine

### System Framing

- `BlueprintCapture` is the capture client and supply-side evidence collection tool.
- `BlueprintCapturePipeline` composes capture bundles, cards, evaluators, and reset
  artifacts into maintained Site-Task Testbeds, then produces claim-level Task
  Evaluation Run plans, evidence, decisions, or explicit abstentions.
- `Blueprint-WebApp` is the buyer, licensing, ops, and hosted-access surface for
  Task Evaluation Runs. It collects constraints and displays Pipeline-owned
  results; it does not select providers or recompute scientific verdicts.
- `BlueprintValidation` remains optional downstream infrastructure for benchmarking, runtime checks, robot evaluation support, and specialized validation after the primary package or run is scoped.

This platform is capture-first and real-site decision-evaluation first. World
models, site-world routes, simulation outputs, generated media, editing assets,
legacy data-package exports, and model-derived artifacts are evidence or support
artifacts inside a run. They are not additional public offers.

### Product Center of Gravity

The center of gravity is:

- broad real-world capture coverage
- strong capture quality and provenance
- Task Evaluation Runs for robot teams
- maintained, immutable-version Site-Task Testbeds reused across successive runs
- claim-level routing to the cheapest currently qualified evidence method or
  combination, with explicit escalation and abstention
- rights-cleared evidence-use determinations inside a run; post-training use is
  allowed only after rights, provenance, action-alignment, quality, and leakage gates
- hosted access for request-scoped review
- rights, privacy, and commercialization controls
- buyer-facing product surfaces that make real sites easy to browse, buy, run, and manage

The center of gravity is not:

- generic marketplace browsing as the main story
- qualification/readiness as the main thing Blueprint sells
- not world models as the primary public product or a generic world-model marketplace
- one-off model demos disconnected from real capture
- a single permanent world-model backend

### Market Structure

The core business engine is two-sided:

- **Capturers** supply real-site evidence packages.
- **Robot teams** buy Task Evaluation Runs.

`Site operators` remain important, but they are an optional third lane at capture time, covering:

- access control
- rights / consent / privacy boundaries
- commercialization and revenue sharing

The platform must support lawful capture and packaging even when a site has not already gone through a pre-negotiated intake flow. Site-operator involvement is a supported workflow branch, not a universal prerequisite for all capture.

Optional at capture time is not the same as unimportant at adoption time. The long-horizon direction (see `VISION.md`, rung 2) is that site operators become the demand-side channel that routes deployment decisions through Blueprint evaluation — requiring a Task Evaluation Run before a robot reaches their floor. That is a strategic adoption bet about where the standard gets enforced, not a current capture prerequisite; the two statements describe different lifecycle stages and do not conflict.

### Truth Hierarchy

- raw capture, timestamps, poses, device metadata, and provenance are authoritative
- rights / privacy / consent metadata are authoritative
- Site Cards, Task Cards, Scenario Cards, Eval Cards, package manifests, generated/model-derived support assets, and hosted-session artifacts are downstream artifacts with explicit proof boundaries
- Task Evaluation Run is the one primary sellable downstream product
- maintained Site-Task Testbeds are reusable substrates, not a second product
- post-training is a permitted use of qualifying run evidence, never proof that training occurred or a policy improved
- qualification / readiness / review outputs are optional trust layers that can guide buying, commercialization, and deployment decisions
- downstream outputs must not rewrite capture truth or provenance truth

### Product Stack

1. supply substrate: truthful capture and real-site coverage
2. reusable substrate: maintained Site-Task Testbeds
3. buyer product: Task Evaluation Runs
4. evidence methods: geometry, captured observations, traditional simulation,
   learned/world-model evaluation, provider tools, physical evidence, and bounded owner inputs
5. support layer: hosted review, evidence export/use, legacy compatibility,
   generated/model-derived data, editing, and augmentation

### Commercial Wedge Overlay

The current PMF wedge is the Task Evaluation Run: a decision request bound to an
exact maintained testbed. The router decomposes the decision into claims,
selects only qualified methods, and returns a partial or complete decision or an
explicit abstention with the next cheapest experiment. A run may emit
rights-cleared evidence for later evaluation or post-training use, but the
evidence export is not another product and does not imply training or improvement.

Wedge claims stay inside the proof boundary. The current comparative
policy-ranking scientific verdict is `thesis_not_supported`; physical success,
deployment readiness, and safety claims require separately accepted physical
evidence. Generated frames are support, never real-world proof.

### Default Lifecycle

1. A capture is sourced proactively or through a buyer / site / ops request.
2. `BlueprintCapture` records and uploads a truthful evidence bundle.
3. `BlueprintCapturePipeline` composes a versioned Site-Task Testbed and routes a
   Task Evaluation Run at claim level through qualified evidence adapters.
4. `Blueprint-WebApp` exposes the request, plan status, decision envelope,
   abstentions, and proof-bound supporting artifacts.
5. Optional world-model, simulation, deeper evaluation, validation, or managed support follows only when commercially useful and proof-bounded.

### Practical Rule For Agents

When changing any Blueprint repo, optimize for:

1. stronger real-site capture supply
2. better Task Evaluation Runs, maintained testbeds, routing, abstention, and learning
3. stable rights / privacy / provenance contracts
4. buyer and ops surfaces that make those outputs easy to sell and use
5. optional trust, readiness, world-model, simulation, generated-data, and validation layers that support the product without becoming the product story

Do not assume that every capture must begin with formal site qualification.
Do not treat qualification/readiness as the universal center of the company.
Do not overstate world-model quality beyond what capture, privacy, and runtime artifacts support.
<!-- SHARED_PLATFORM_CONTEXT_END -->
