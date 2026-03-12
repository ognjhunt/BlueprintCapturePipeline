# Platform Context

<!-- SHARED_PLATFORM_CONTEXT_START -->
## Shared Platform Doctrine

### System Framing

- `BlueprintCapture` captures raw evidence packages.
- `BlueprintCapturePipeline` converts evidence plus intake into canonical scene memory, evaluation-prep bundles, and runtime-ready site-world records.
- `Blueprint-WebApp` is the operating and commercial system around site-world records, runtime status, and derived workflow packages.
- `BlueprintValidation` consumes site-world packages for robot evaluation, adaptation, and tuning work.

This platform is site-world-first.

### Truth Hierarchy

- capture-backed scene memory, evaluation-prep manifests, and site-world runtime eligibility are authoritative
- legacy qualification/readiness artifacts are compatibility overlays and must not override site-world grounding
- preview simulations, advanced-geometry bundles, and trained policies are derived downstream assets; they do not rewrite capture-backed site-world truth

### Product Stack

1. primary product: site-specific world model and runtime-ready package
2. secondary product: scene memory / preview simulation / robot eval package
3. third product: world-model-based adaptation, managed tuning, training data, licensing
4. fourth product: legacy reporting and handoff overlays for systems that still expect them

### Downstream Training Rule

- world-model RL and world-model-based post-training are first-class downstream paths for site adaptation, checkpoint ranking, synthetic rollout generation, and bounded robot-team evaluation
- those paths sit behind the capture-backed site world and do not by themselves replace stricter validation for contact-critical, safety-critical, or contractual deployment claims
- Isaac-backed, physics-backed, or otherwise stricter validation remains the higher-trust lane when reproducibility, contact fidelity, or formal signoff matters

### Data Rule

- passive site capture and walkthrough evidence are valuable context for scene memory, preview simulation, and downstream conditioning
- strong robot adaptation gains usually require action-conditioned robot interaction data such as play, teleop logs, or task rollouts; site video alone is usually not enough for reliable policy training from scratch
- derived assets may inform routing and downstream work, but they must not mutate capture-backed site-world truth
<!-- SHARED_PLATFORM_CONTEXT_END -->

This repo is the site-world assembly engine.

The default output should be a capture-backed site-world package, not a qualification packet.

## What This Repo Owns

`BlueprintCapturePipeline` is the site-world assembly engine.

Its main job is to turn raw capture evidence plus structured intake into:

- scene-memory bundles
- evaluation-prep manifests
- runtime-eligibility and launch records
- backend adapter manifests
- optional advanced-geometry follow-ons

Legacy qualification, readiness, and handoff artifacts still exist for compatibility, but they are not the center of gravity.

This repo is where the site moves from “we captured something” to “we have a launchable site-world package.”

## Upstream And Downstream Boundaries

### Upstream

This repo expects a capture bundle from `BlueprintCapture`, not just a loose video.

The raw capture contract is:

```text
raw/
  manifest.json
  intake_packet.json
  capture_context.json
  capture_upload_complete.json
  walkthrough.mov
  optional arkit/...
```

### Downstream

This repo produces records that should be routed into `Blueprint-WebApp`, including:

- scene-memory manifests
- evaluation-prep bundles
- site-world runtime status
- backend adapter manifests
- compatibility summaries when older consumers still require them

## Product Context

The correct product stack is:

1. primary product: site-specific world model and runtime-ready package
2. secondary product: scene memory / preview simulation / evaluation package
3. third product: world-model-based adaptation / managed tuning / training data / licensing
4. fourth product: compatibility overlays for reporting and routing systems

This repo sits in the middle and owns the transition from evidence to site-world package.

This repo should treat the default output as:

- a reusable scene-memory bundle
- a runtime-ready site-world spec
- adapter manifests for downstream runtimes
- compatibility overlays only when a consumer still requires them

This repo should not assume that every capture should become:

- a CRM record
- a qualification packet
- a tuning engagement

Those are consumers or follow-on lanes, not the default result.

## Role In The Business

This repo is the middle of the system.

It exists to answer:

- can we ground a usable site-specific world model from this capture?
- is the evidence sufficient for scene memory and runtime launchability?
- what holes, blockers, or hidden zones still limit the site world?
- which downstream runtimes or evaluation paths can consume it?
- what follow-on capture or validation is still required?

## System Lifecycle

The intended end-to-end lifecycle is:

1. `Blueprint-WebApp` creates `site_submission_id` and intake.
2. `BlueprintCapture` attaches that context to the evidence package.
3. This repo produces:
   - scene-memory artifacts
   - evaluation-prep manifests
   - site-world runtime records
   - backend adapter manifests
   - later geometry artifacts when justified
4. `Blueprint-WebApp` ingests those outputs and updates:
   - site-world status
   - runtime status
   - artifact and handoff references
5. Downstream evaluation and adaptation consume the site-world package.

Biggest missing system boundary:

- this repo already emits the right artifacts
- the webapp already has the right state model
- the production bridge between the two is still missing

Agents should treat that bridge as a top-tier integration concern.

## Phase 2 Decision Boundary

Phase 2 should stay deterministic for the site-world grounding core:

- object indexing
- scene-memory packaging
- adapter manifest generation
- runtime eligibility checks
- dashboard action buckets and rollout counts

LLM synthesis is allowed only for narrative or recommendation artifacts:

- memos
- reviewer summaries
- operator-facing summaries
- recapture recommendations

That means agents should not let a model override site-world grounding, runtime eligibility, or the
scene dashboard contract. The scene-local `pipeline/dashboard_summary.json` is the stable
frontend-facing contract for Phase 2 scene rollups.

## Practical Rule For Agents In This Repo

When making changes here, optimize for:

1. grounded site-world outputs
2. fail-closed behavior
3. explicit human-review boundaries
4. clean handoff records for the webapp and later technical teams

Do not let CRM, legacy reporting, or marketplace assumptions replace the site-specific world model as the center of gravity.
Do not let generative simulation replace grounded capture evidence; runtime claims are advisory unless separately validated.
