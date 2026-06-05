# World Model Strategy Context

<!-- SHARED_WORLD_MODEL_STRATEGY_START -->
## Strategic Doctrine

Blueprint should assume world models will improve rapidly and that multiple viable model providers, checkpoints, papers, and hosted services will exist over time.

Blueprint should not build the company around owning one permanent model.

Blueprint's durable moat should be:

1. capture supply and real-site coverage
2. rights-safe, provenance-safe, privacy-safe data pipelines
3. Task Evaluation Runs and Post-Training Data Packages grounded in real captured sites
4. buyer, licensing, and ops product surfaces around those runs and packages
5. a compounding capture -> package -> buyer usage -> more capture flywheel

The model backend matters, but it should remain a replaceable engine behind stable capture, packaging, and product contracts.

## Core Belief

Blueprint is not qualification-first and not model-first.

Blueprint is capture-first and real-site robot-evaluation/data-package first.

That means:

- real capture supply is the entry point
- Task Evaluation Runs and Post-Training Data Packages are the primary sellable outputs
- world models, site-world internals, generated media, simulation outputs, editing/augmentation assets, and model-derived artifacts are support substrates inside packages, not the primary public offer
- qualification / readiness can exist as optional trust layers, especially for high-stakes buyers, commercialization decisions, or deployment review
- those trust layers should support the product, not define the company

## Practical Strategic Conclusion

Do not overfit the platform to any one of:

- a single model paper
- a single checkpoint family
- a single provider
- a single inference trick
- a single hardware profile

Instead, build the stack so that better model backends can be dropped in later with minimal changes above the model-adapter layer.

## Current Product Truth

Today, the strongest near-term value comes from:

1. capturing real indoor spaces at scale
2. turning those captures into site/task/scenario/eval artifacts for robot evaluation
3. preserving strong rights, privacy, and provenance metadata around those assets
4. giving robot teams clear buyer surfaces for Task Evaluation Runs before expensive pilots
5. packaging curated robot POV clips, labels, generated/model-derived variations, failure cases, and export manifests into Post-Training Data Packages
6. using qualification / readiness outputs only when they materially improve trust, pricing, commercialization, or deployment decisions

Native SWM-like interaction remains an important direction, but it is not the only thing customers need in order for the product to be valuable now.

## How To Think About The Runtime

The runtime should be treated as a bridge architecture:

- immediate interaction should come from truthful, site-grounded rendering and hosted-session paths
- more generative continuation can sit behind that as optional refinement
- the browser/runtime contract should not assume one model family

This keeps the product useful now while preserving room for stronger native world-model behavior later.

## What Must Stay Stable Across Model Swaps

These should be treated as long-lived platform contracts:

- raw capture bundle structure
- timestamps, poses, intrinsics, depth, and device metadata
- consent, rights, privacy, and provenance metadata
- site-specific package manifests
- hosted-session and runtime session contracts
- buyer attachment, licensing, and sync contracts
- truth labeling in UI and APIs
- Task Evaluation Run and Post-Training Data Package contracts

Qualification / readiness outputs should stay compatible where they exist, but they should be treated as optional support contracts rather than the only source of product value.

## What Must Remain Swappable

These should be deliberately replaceable:

- world-model checkpoints
- world-model providers
- inference services
- retrieval-conditioned generation strategies
- refinement models
- training/export adapters

No repo should assume one specific model or provider is permanent.

## Platform Moat

Blueprint's moat should come from assets that get stronger when models commoditize:

- better real-site capture coverage
- better capture quality and provenance
- better rights / privacy / commercialization handling
- better Task Evaluation Runs and Post-Training Data Packages
- better buyer UX and operational surfaces
- better feedback loops from real buyer usage on real sites

If world models become easier to buy, proprietary real-site capture and product workflow should become more valuable, not less.

## Product Implication

The company should be able to say:

- we do not depend on owning the single best world model
- we are the best system for turning real sites into robot evaluation runs and post-training data packages
- we can use world models, hosted experiences, trust, review, and readiness layers when they help, without making unsupported qualification or world models the center of the company

## Build Priorities Right Now

For the current stage, prioritize:

1. capture quality and coverage
2. packaging captures into strong site/task/scenario/eval artifacts
3. Task Evaluation Runs, Post-Training Data Packages, hosted access, and buyer usability
4. rights / privacy / provenance rigor
5. stable product contracts that survive backend swaps
6. generated/model-derived data, world-model, simulation, and readiness support for the cases that need them

Do not spend disproportionate time pushing qualification/readiness or world-model access into the lead product story when the main commercial value comes from capture supply, Task Evaluation Runs, and Post-Training Data Packages.

## Data Priority

Collect and preserve data now as if future world-model training and evaluation will depend on it.

That means preserving:

- walkthrough video
- motion / trajectory logs
- camera poses
- intrinsics
- depth when available
- timestamps and temporal alignment data
- device / modality metadata
- site / scenario / deployment context
- privacy / consent / rights metadata
- retrieval / reference relationships when derived

Future model quality and package quality will depend heavily on data quality and structure.

## Repo-Level Guidance

Each repo should optimize for the same posture:

- `BlueprintCapture`: capture the richest, cleanest, most reusable real-site evidence possible
- `BlueprintCapturePipeline`: turn that evidence into Task Evaluation Run artifacts, Post-Training Data Package artifacts, hosted-session artifacts, generated/model-derived support assets, and optional trust layers without coupling the platform to one backend
- `Blueprint-WebApp`: sell, deliver, and operate those runs and packages through clear buyer and ops surfaces

## Non-Goal

Do not assume the platform is "done" only when a perfect SWM runtime exists.

The correct goal is:

- build everything around capture, packaging, and buyer workflow so stronger world-model backends can be adopted later without a company-wide rebuild
- keep world-model language as internal compatibility or generated/data-package support unless a public surface clearly labels it as advisory support

## Decision Rule For Future Sessions

When choosing between:

- investing in model-specific hacks
- investing in reusable capture / packaging / product infrastructure

default toward reusable infrastructure unless a model-specific change materially improves near-term user-visible value without increasing long-term coupling.
<!-- SHARED_WORLD_MODEL_STRATEGY_END -->
