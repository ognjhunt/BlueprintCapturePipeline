# Platform Context

<!-- SHARED_PLATFORM_CONTEXT_START -->
## Shared Platform Doctrine

### System Framing

- `BlueprintCapture` is the capture client and supply-side evidence collection tool.
- `BlueprintCapturePipeline` turns capture bundles into Site Cards, Task Cards, Scenario Cards, Eval Cards, Post-Training Data Package artifacts, generated/model-derived support assets, hosted-session artifacts, and proof boundaries.
- `Blueprint-WebApp` is the buyer, licensing, ops, and hosted-access surface for Task Evaluation Runs and Post-Training Data Packages.
- `BlueprintValidation` remains optional downstream infrastructure for benchmarking, runtime checks, robot evaluation support, and specialized validation after the primary package or run is scoped.

This platform is capture-first and real-site robot-evaluation/data-package first.
World models, site-world routes, simulation outputs, generated media, editing assets, and model-derived artifacts are allowed as internal compatibility names or support artifacts inside data packages. They are not the primary public offer.

### Product Center of Gravity

The center of gravity is:

- broad real-world capture coverage
- strong capture quality and provenance
- Task Evaluation Runs for robot teams
- Post-Training Data Packages with curated robot POV clips, labels, generated/model-derived variations, failure cases, task metadata, QA notes, and export manifests
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
- **Robot teams** buy Task Evaluation Runs and Post-Training Data Packages.

`Site operators` remain important, but they are an optional third lane for:

- access control
- rights / consent / privacy boundaries
- commercialization and revenue sharing

The platform must support lawful capture and packaging even when a site has not already gone through a pre-negotiated intake flow. Site-operator involvement is a supported workflow branch, not a universal prerequisite for all capture.

### Truth Hierarchy

- raw capture, timestamps, poses, device metadata, and provenance are authoritative
- rights / privacy / consent metadata are authoritative
- Site Cards, Task Cards, Scenario Cards, Eval Cards, package manifests, generated/model-derived support assets, and hosted-session artifacts are downstream artifacts with explicit proof boundaries
- Task Evaluation Runs and Post-Training Data Packages are the primary sellable downstream products
- qualification / readiness / review outputs are optional trust layers that can guide buying, commercialization, and deployment decisions
- downstream outputs must not rewrite capture truth or provenance truth

### Product Stack

1. primary product: capture supply and real-site coverage
2. buyer product: Task Evaluation Runs
3. buyer product: Post-Training Data Packages
4. support layer: hosted review, generated/model-derived data, simulation, editing, augmentation, and world-model compatibility artifacts
5. downstream support: validation, deeper benchmarking, managed tuning, licensing, and deployment support

### Default Lifecycle

1. A capture is sourced proactively or through a buyer / site / ops request.
2. `BlueprintCapture` records and uploads a truthful evidence bundle.
3. `BlueprintCapturePipeline` materializes site/task/scenario/eval artifacts, post-training data artifacts, hosted artifacts, generated/model-derived support assets, and optional trust outputs.
4. `Blueprint-WebApp` exposes Task Evaluation Runs, Post-Training Data Packages, and those proof-bound artifacts through buyer, ops, licensing, and hosted-session surfaces.
5. Optional world-model, simulation, deeper evaluation, validation, or managed support follows only when commercially useful and proof-bounded.

### Practical Rule For Agents

When changing any Blueprint repo, optimize for:

1. stronger real-site capture supply
2. better Task Evaluation Runs and Post-Training Data Packages
3. stable rights / privacy / provenance contracts
4. buyer and ops surfaces that make those outputs easy to sell and use
5. optional trust, readiness, world-model, simulation, generated-data, and validation layers that support the product without becoming the product story

Do not assume that every capture must begin with formal site qualification.
Do not treat qualification/readiness as the universal center of the company.
Do not overstate world-model quality beyond what capture, privacy, and runtime artifacts support.
<!-- SHARED_PLATFORM_CONTEXT_END -->
