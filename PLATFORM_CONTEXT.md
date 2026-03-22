# Platform Context

<!-- SHARED_PLATFORM_CONTEXT_START -->
## Shared Platform Doctrine

### System Framing

- `BlueprintCapture` is the capture client and supply-side evidence collection tool.
- `BlueprintCapturePipeline` turns capture bundles into site-specific world-model packages, hosted-session artifacts, and optional trust / review outputs.
- `Blueprint-WebApp` is the buyer, licensing, ops, and hosted-access surface around those packages.
- `BlueprintValidation` remains optional downstream infrastructure for deeper benchmarking, robot evaluation, and specialized runtime checks.

This platform is capture-first and world-model-product-first.

### Product Center of Gravity

The center of gravity is:

- broad real-world capture coverage
- strong capture quality and provenance
- site-specific world models and hosted access for robot teams
- rights, privacy, and commercialization controls
- buyer-facing product surfaces that make real sites easy to browse, buy, run, and manage

The center of gravity is not:

- generic marketplace browsing as the main story
- qualification/readiness as the main thing Blueprint sells
- one-off model demos disconnected from real capture
- a single permanent world-model backend

### Market Structure

The core business engine is two-sided:

- **Capturers** supply real-site evidence packages.
- **Robot teams** buy site-specific world models, hosted access, and related outputs.

`Site operators` remain important, but they are an optional third lane for:

- access control
- rights / consent / privacy boundaries
- commercialization and revenue sharing

The platform must support lawful capture and packaging even when a site has not already gone through a pre-negotiated intake flow. Site-operator involvement is a supported workflow branch, not a universal prerequisite for all capture.

### Truth Hierarchy

- raw capture, timestamps, poses, device metadata, and provenance are authoritative
- rights / privacy / consent metadata are authoritative
- site-specific world-model packages and hosted-session artifacts are the primary sellable downstream products
- qualification / readiness / review outputs are optional trust layers that can guide buying, commercialization, and deployment decisions
- downstream outputs must not rewrite capture truth or provenance truth

### Product Stack

1. primary product: capture supply and real-site coverage
2. second product: site-specific world models and hosted access
3. third product: optional trust / review / readiness outputs
4. fourth product: deeper evaluation, managed tuning, licensing, and deployment support

### Default Lifecycle

1. A capture is sourced proactively or through a buyer / site / ops request.
2. `BlueprintCapture` records and uploads a truthful evidence bundle.
3. `BlueprintCapturePipeline` materializes site-specific packages, hosted artifacts, and optional trust outputs.
4. `Blueprint-WebApp` exposes those outputs through buyer, ops, licensing, and hosted-session surfaces.
5. Optional review, deeper evaluation, or managed support follows only when commercially useful.

### Practical Rule For Agents

When changing any Blueprint repo, optimize for:

1. stronger real-site capture supply
2. better site-specific world-model outputs and hosted access
3. stable rights / privacy / provenance contracts
4. buyer and ops surfaces that make those outputs easy to sell and use
5. optional trust / readiness layers that support the product without becoming the product story

Do not assume that every capture must begin with formal site qualification.
Do not treat qualification/readiness as the universal center of the company.
Do not overstate world-model quality beyond what capture, privacy, and runtime artifacts support.
<!-- SHARED_PLATFORM_CONTEXT_END -->
