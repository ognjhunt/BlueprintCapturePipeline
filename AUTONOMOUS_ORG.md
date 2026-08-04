# Blueprint Autonomous Organization Guide

> **Source of truth:** [Blueprint Hub on Notion](https://www.notion.so/16d80154161d80db869bcfba4fe70be3)
> The live company package and current org definition are maintained from `Blueprint-WebApp`.
> This repo-side file is a synced mirror for people working in `BlueprintCapturePipeline`.

## Current Authority

Treat these as the canonical operating files, all in the `Blueprint-WebApp`
sibling checkout (its location is environment-dependent — see the
sibling-checkout convention in [`AGENTS.md`](AGENTS.md)):

1. `AUTONOMOUS_ORG.md`
2. `ops/paperclip/blueprint-company/.paperclip.yaml`
3. `ops/paperclip/BLUEPRINT_AUTOMATION.md`

If this file drifts from those, update this file immediately rather than inventing a repo-local org shape. If the `Blueprint-WebApp` checkout is not present in your environment, treat cross-checks against these files as blocked instead of guessing.

## Platform Posture

Blueprint remains:

- capture-first
- solely focused on Arm Decision Proof v1: one prospective two-candidate
  fixed-arm Task Evaluation Run on a new partner workcell, physically adjudicated
- world-model backends remain replaceable support infrastructure
- exact-site package and hosted-access focused
- rights-safe, privacy-safe, provenance-safe
- built so Paperclip owns execution state while agents operate on top of software and product systems

Do not reframe the company as qualification-first, model-checkpoint-first, or generic marketplace-first.

All listed roles are capabilities, not parallel mandates. Until Arm Decision
Proof v1 is adjudicated, every active routine or issue must directly unblock its
day-7, day-14, day-28, day-35, or day-42 gate. Growth, city launch, humanoid,
world-model, marketplace, post-training, and unrelated product routines are
paused even if their role definitions remain for compatibility.

## Current Org Reality

Blueprint now runs as a Paperclip-centered operating system with Hermes used selectively for continuity-heavy roles.

Current high-level split:

- `Paperclip` is the execution record for issues, routines, assignments, and work state.
- `Notion` is the workspace, knowledge, review, and operator-visibility surface.
- `Hermes` is the persistent runtime for the chief of staff plus selected ops, growth, research, and commercial roles.
- `Claude` remains the default executive/review lane and the default lane for sensitive ops/review roles.
- `Codex` remains the default implementation lane for repo specialists.

## Department Snapshot

### Executive

- `blueprint-ceo` — Claude
- `blueprint-chief-of-staff` — Hermes
- `blueprint-cto` — Claude
- `investor-relations-agent` — Hermes
- `notion-manager-agent` — Hermes
- `revenue-ops-pricing-agent` — Hermes

### Engineering

- `webapp-codex`, `webapp-claude`
- `pipeline-codex`, `pipeline-claude`
- `capture-codex`, `capture-claude`
- `beta-launch-commander`
- `docs-agent`

### Ops

- `ops-lead`
- `intake-agent`
- `capture-qa-agent`
- `field-ops-agent`
- `finance-support-agent`
- `buyer-solutions-agent`
- `solutions-engineering-agent`
- `rights-provenance-agent`
- `security-procurement-agent`
- `capturer-success-agent`
- `site-catalog-agent`
- `buyer-success-agent`

### Growth

- `growth-lead`
- `conversion-agent`
- `analytics-agent`
- `community-updates-agent`
- `market-intel-agent`
- `supply-intel-agent`
- `capturer-growth-agent`
- `city-launch-agent`
- `demand-intel-agent`
- `robot-team-growth-agent`
- `site-operator-partnership-agent`
- `city-demand-agent`
- `outbound-sales-agent`

## What Touches BlueprintCapturePipeline

These are the primary agents with direct responsibility for this repo:

- `pipeline-codex` — implementation specialist for `BlueprintCapturePipeline`
- `pipeline-claude` — review and planning specialist for `BlueprintCapturePipeline`
- `capture-qa-agent` — quality, completeness, privacy, and recapture review
- `rights-provenance-agent` — consent, rights, privacy-processing, provenance, and commercialization fail-closed gate

These agents regularly read this repo or create work against it:

- `blueprint-cto`
- `blueprint-chief-of-staff`
- `ops-lead`
- `beta-launch-commander`
- `docs-agent`
- `solutions-engineering-agent`

## Current Operational Shape

The live Paperclip package currently runs from the shared trusted host and includes:

- a continuous chief-of-staff managerial loop
- active repo autonomy loops for Capture, Pipeline, and WebApp
- active executive reporting routines
- active weekly or daily routines across growth and commercial lanes
- explicit paused routines for lanes that are intentionally not running continuously yet

For the exact current task and routine inventory, read (in the
`Blueprint-WebApp` sibling checkout):

- `ops/paperclip/blueprint-company/.paperclip.yaml`

## Rules For Pipeline Repo Work

When autonomous-org work touches `BlueprintCapturePipeline`, keep these constraints explicit:

- optimize only for the Arm Decision Proof critical path, using existing
  captures/scenes to complete downstream seams before partner capture arrives
- require every issue to name its ADP backlog item, gate, observed blocker, and
  completion artifact; reject general platform improvement work
- keep geometry, simulator, world-model/provider, and physical-evidence choices
  swappable behind stable request, plan, result, and leaf-run contracts
- preserve rights, privacy, and provenance metadata as first-class truth
- route blockers, delegation, and validation through Paperclip issues, not prose alone
- do not let buyer, launch, or growth pressure overstate package quality beyond what artifacts and contracts support

## Maintenance Rule

This mirror should change whenever one of these changes (all in the
`Blueprint-WebApp` sibling checkout):

- the role registry in `AUTONOMOUS_ORG.md`
- the live company package in `ops/paperclip/blueprint-company/.paperclip.yaml`
- the Paperclip/Hermes runtime split in `ops/paperclip/BLUEPRINT_AUTOMATION.md`

Do not keep an older repo-local org chart alive once the shared control plane has moved on.
