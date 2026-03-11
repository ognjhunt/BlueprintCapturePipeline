# Platform Context

<!-- SHARED_PLATFORM_CONTEXT_START -->
## Shared Platform Doctrine

### System Framing

- `BlueprintCapture` captures raw evidence packages.
- `BlueprintCapturePipeline` converts evidence plus intake into qualification artifacts, readiness decisions, and handoffs.
- `Blueprint-WebApp` is the operating and commercial system around qualification records and derived downstream lanes.
- `BlueprintValidation` performs post-qualification scene derivation, robot evaluation, adaptation, and tuning work.

This platform is qualification-first.

### Truth Hierarchy

- qualification records, readiness decisions, and supporting evidence links are authoritative
- capture-backed scene memory is the preferred downstream substrate when deeper technical work is justified
- preview simulations, world-model outputs, and world-model-trained policies are derived downstream assets; they do not rewrite qualification truth

### Product Stack

1. primary product: site qualification / readiness pack
2. secondary product: qualified opportunity exchange for robot teams
3. third product: scene memory / preview simulation / robot eval package
4. fourth product: world-model-based adaptation, managed tuning, training data, licensing

### Downstream Training Rule

- world-model RL and world-model-based post-training are first-class downstream paths for site adaptation, checkpoint ranking, synthetic rollout generation, and bounded robot-team evaluation
- those paths sit behind qualification and do not by themselves replace stricter validation for contact-critical, safety-critical, or contractual deployment claims
- Isaac-backed, physics-backed, or otherwise stricter validation remains the higher-trust lane when reproducibility, contact fidelity, or formal signoff matters

### Data Rule

- passive site capture and walkthrough evidence are valuable context for scene memory, preview simulation, and downstream conditioning
- strong robot adaptation gains usually require action-conditioned robot interaction data such as play, teleop logs, or task rollouts; site video alone is usually not enough for reliable policy training from scratch
- derived assets may inform routing and downstream work, but they must not mutate qualification state or source-of-truth readiness records
<!-- SHARED_PLATFORM_CONTEXT_END -->

This repo is the qualification engine.

The default output should be a qualification record plus routing object, not a scene package by default.

## What This Repo Owns

`BlueprintCapturePipeline` is the qualification engine.

Its main job is to turn raw capture evidence plus structured intake into:

- qualification artifacts
- readiness decisions
- blocker and evidence-gap outputs
- opportunity handoffs
- optional advanced-geometry follow-ons

This repo is where the site moves from “we captured something” to “we have a structured decision object.”

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

- qualification state
- readiness report
- opportunity handoff
- human actions required
- later geometry/evaluation artifacts when justified

## Product Context

The correct product stack is:

1. primary product: site qualification / readiness pack
2. secondary product: qualified opportunity exchange for robot teams
3. third product: scene memory / preview simulation / evaluation package
4. fourth product: world-model-based adaptation / managed tuning / training data / licensing

This repo sits in the middle and owns the transition from evidence to decision.

This repo should treat the default output as:

- a reusable qualification record
- a routing object
- a human-review-aware handoff
- a scene-memory bundle only as a derived downstream substrate

This repo should not assume that every capture should become:

- a marketplace scene
- a sim package
- a tuning engagement

Those are later-stage follow-on lanes, not the default result.

## Role In The Business

This repo is the middle of the system.

It exists to answer:

- is the site/task scoped well enough?
- is the evidence good enough?
- what blockers are visible?
- is the site ready, risky, or not ready yet?
- should this move into deeper evaluation?

## System Lifecycle

The intended end-to-end lifecycle is:

1. `Blueprint-WebApp` creates `site_submission_id` and intake.
2. `BlueprintCapture` attaches that context to the evidence package.
3. This repo produces:
   - qualification artifacts
   - readiness decision
   - opportunity handoff
   - human actions required
   - later geometry/evaluation artifacts when justified
4. `Blueprint-WebApp` ingests those outputs and updates:
   - `qualification_state`
   - `opportunity_state`
   - report / handoff references
5. Only qualified records move into exchange and later paid lanes.

Biggest missing system boundary:

- this repo already emits the right artifacts
- the webapp already has the right state model
- the production bridge between the two is still missing

Agents should treat that bridge as a top-tier integration concern.

## Phase 2 Decision Boundary

Phase 2 should stay deterministic for the decision core:

- scoring
- gates
- blocker classification
- capability-envelope checks
- readiness state
- dashboard action buckets and rollout counts

LLM synthesis is allowed only for narrative or recommendation artifacts:

- memos
- reviewer summaries
- OEM-facing summaries
- recapture recommendations

That means agents should not let a model override blocker truth, readiness state, or the
scene dashboard contract. The scene-local `pipeline/dashboard_summary.json` is the stable
frontend-facing contract for Phase 2 scene rollups.

## Practical Rule For Agents In This Repo

When making changes here, optimize for:

1. grounded qualification outputs
2. fail-closed behavior
3. explicit human-review boundaries
4. clean handoff records for the webapp and later technical teams

Do not let geometry, scenes, validation, or marketplace assumptions replace qualification as the center of gravity.
Do not let generative simulation replace grounded qualification evidence; world-model outputs are advisory unless separately validated.
