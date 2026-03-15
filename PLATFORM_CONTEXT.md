# Platform Context

<!-- SHARED_PLATFORM_CONTEXT_START -->
## Shared Platform Doctrine

### System Framing

- `BlueprintCapture` is the contributor evidence-capture tool inside Blueprint's three-sided marketplace.
- `BlueprintCapturePipeline` is the authoritative qualification, provenance, and provider-routing service.
- `Blueprint-WebApp` is the three-sided marketplace and operating system connecting capturers, robot teams, and site operators around qualification records and downstream work.
- `BlueprintValidation` is optional downstream infrastructure for provider benchmarking, runtime-backed demos, and deeper robot evaluation after qualification.

This platform is qualification-first.

### Three-Sided Marketplace

- **Capturers** supply evidence packages from real sites.
- **Robot teams** are the primary demand-side buyers of trusted qualification outcomes and downstream technical work.
- **Site operators** control access, rights, and commercialization boundaries for their facilities.

### Truth Hierarchy

- qualification records, readiness decisions, trust signals, and provenance links are authoritative
- capture-backed scene memory and evaluation-prep packages are preferred downstream technical substrates once qualification justifies them
- preview simulations, provider outputs, advanced-geometry bundles, and trained policies are derived downstream assets; they do not rewrite qualification truth

### Product Stack

1. primary product: qualification record / readiness decision / buyer-safe evidence bundle
2. secondary product: qualified opportunity exchange and provider-backed preview lane
3. third product: scene memory / evaluation-prep / runtime-backed robot evaluation
4. fourth product: world-model-based adaptation, managed tuning, training data, licensing

### Downstream Training Rule

- world-model RL and world-model-based post-training are first-class downstream paths for site adaptation, checkpoint ranking, synthetic rollout generation, and bounded robot-team evaluation
- those paths sit behind qualification and do not by themselves replace stricter validation for contact-critical, safety-critical, or contractual deployment claims
- Isaac-backed, physics-backed, or otherwise stricter validation remains the higher-trust lane when reproducibility, contact fidelity, or formal signoff matters

### Data Rule

- passive site capture and walkthrough evidence are valuable context for qualification, scene memory, preview simulation, and downstream conditioning
- strong robot adaptation gains usually require action-conditioned robot interaction data such as play, teleop logs, or task rollouts; site video alone is usually not enough for reliable policy training from scratch
- derived assets may inform routing and downstream work, but they must not mutate qualification truth
<!-- SHARED_PLATFORM_CONTEXT_END -->

This repo is the qualification and provider-routing engine.

## What This Repo Owns

`BlueprintCapturePipeline` is the authoritative middle of the alpha service.

Its main job is to turn raw capture evidence plus structured intake into:

- deterministic QA aggregation
- qualification artifacts and readiness decisions
- buyer-trust and rights/compliance summaries
- provider preview routing and normalized provider manifests
- recapture requirements and follow-up guidance
- scene-memory bundles
- evaluation-prep manifests
- runtime-facing or geometry follow-ons only when requested

Scene/world outputs still exist, but they are not the center of gravity for alpha.

This repo is where the site moves from “we captured something” to “we have a trustworthy qualification bundle and optional downstream routes.”

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

- qualification summaries
- buyer trust and rights/compliance summaries
- provider preview manifests and provenance
- scene-memory manifests
- evaluation-prep bundles
- runtime and adapter manifests when deeper work is explicitly requested

## Product Context

The correct product stack is:

1. primary product: qualification packet and trusted routing object
2. secondary product: provider-backed preview and qualified opportunity handoff
3. third product: scene memory / evaluation-prep / runtime-backed evaluation
4. fourth product: world-model-based adaptation / managed tuning / training data / licensing

This repo sits in the middle and owns the transition from evidence to qualification-first outputs.

This repo should treat the default output as:

- a normalized qualification bundle
- buyer-safe quality, rights, and provenance summaries
- provider preview state that can fail without blocking qualification
- a reusable scene-memory bundle
- evaluation/runtime manifests only when a downstream lane needs them

This repo should not assume that every capture should become:

- a world model
- a live runtime
- a tuning engagement

Those are downstream lanes, not the default result.

## Role In The Business

This repo exists to answer:

- is the evidence sufficient to make a qualification decision?
- what holes, blockers, rights issues, or hidden zones still limit trust?
- what recapture is still required?
- which downstream provider, scene-memory, runtime, or evaluation paths are justified?
- what provenance must be preserved for buyer review?

## System Lifecycle

The intended end-to-end lifecycle is:

1. `Blueprint-WebApp` creates `site_submission_id` and intake.
2. `BlueprintCapture` attaches that context to the evidence package.
3. This repo produces:
   - qualification artifacts
   - trust, rights, and recapture summaries
   - normalized provider preview state
   - scene-memory artifacts
   - evaluation-prep manifests
   - runtime or geometry artifacts only when justified
4. `Blueprint-WebApp` ingests those outputs and updates:
   - qualification state
   - buyer review surfaces
   - artifact and handoff references
5. `BlueprintValidation` or another downstream system is used only for benchmarking, demos, or deeper robot evaluation.

## Advisory Model Boundary

Deterministic code should stay authoritative for:

- qualification state
- trust-score assembly
- provenance
- rights/compliance status
- fail-closed preview/provider behavior

API and LLM synthesis are allowed only for advisory artifacts:

- semantic evidence review
- task/blocker summaries
- recapture recommendations
- buyer-safe quality narration

That means agents should not let a model override deterministic QA, final qualification state, rights status, or provenance.

## Practical Rule For Agents In This Repo

When making changes here, optimize for:

1. grounded qualification outputs
2. fail-closed behavior
3. explicit human-review boundaries
4. clean handoff records for the webapp and later technical teams
5. optional downstream world-model/runtime lanes that never block qualification

Do not let world-model or runtime assumptions replace qualification as the center of gravity.
Do not let provider previews or generative simulation replace grounded capture evidence; they remain advisory unless separately validated.
