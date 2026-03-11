# Platform Context

This repo is one part of a three-repo system.

## System Framing

- `BlueprintCapture` creates the evidence package.
- `BlueprintCapturePipeline` creates the qualification record and handoff.
- `Blueprint-WebApp` is the operating system around those records:
  - intake
  - routing
  - admin review
  - qualified opportunity exchange
  - later evaluation / tuning packaging
  - monetization

This platform is qualification-first.

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
3. third product: deeper evaluation / geometry / simulation package
4. fourth product: training data / managed tuning / licensing

This repo sits in the middle and owns the transition from evidence to decision.

This repo should treat the default output as:

- a reusable qualification record
- a routing object
- a human-review-aware handoff

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

## Practical Rule For Agents In This Repo

When making changes here, optimize for:

1. grounded qualification outputs
2. fail-closed behavior
3. explicit human-review boundaries
4. clean handoff records for the webapp and later technical teams

Do not let geometry, scenes, validation, or marketplace assumptions replace qualification as the center of gravity.
