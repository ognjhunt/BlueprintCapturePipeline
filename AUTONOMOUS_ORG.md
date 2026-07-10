# Blueprint Autonomous Organization Guide

> **Source of truth:** [Blueprint Hub on Notion](https://www.notion.so/16d80154161d80db869bcfba4fe70be3)
> The live company package and current org definition are maintained from `Blueprint-WebApp`.
> This repo-side file is a synced mirror for people working in `BlueprintCapturePipeline`.

## Current Authority

Treat these as the canonical operating files:

1. `$HOME/workspace/Blueprint-WebApp/AUTONOMOUS_ORG.md`
2. `$HOME/workspace/Blueprint-WebApp/ops/paperclip/blueprint-company/.paperclip.yaml`
3. `$HOME/workspace/Blueprint-WebApp/ops/paperclip/BLUEPRINT_AUTOMATION.md`

If this file drifts from those, update this file immediately rather than inventing a repo-local org shape.

## Platform Posture

Blueprint remains:

- capture-first
- world-model-product-first
- exact-site package and hosted-access focused
- rights-safe, privacy-safe, provenance-safe
- built so Paperclip owns execution state while agents operate on top of software and product systems

Do not reframe the company as qualification-first, model-checkpoint-first, or generic marketplace-first.

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

For the exact current task and routine inventory, read:

- `$HOME/workspace/Blueprint-WebApp/ops/paperclip/blueprint-company/.paperclip.yaml`

## Rules For Pipeline Repo Work

When autonomous-org work touches `BlueprintCapturePipeline`, keep these constraints explicit:

- optimize for stronger site-specific package quality and hosted-session truth
- keep backend/model choices swappable behind stable package and runtime contracts
- preserve rights, privacy, and provenance metadata as first-class truth
- route blockers, delegation, and validation through Paperclip issues, not prose alone
- do not let buyer, launch, or growth pressure overstate package quality beyond what artifacts and contracts support

## Maintenance Rule

This mirror should change whenever one of these changes:

- the role registry in `$HOME/workspace/Blueprint-WebApp/AUTONOMOUS_ORG.md`
- the live company package in `$HOME/workspace/Blueprint-WebApp/ops/paperclip/blueprint-company/.paperclip.yaml`
- the Paperclip/Hermes runtime split in `$HOME/workspace/Blueprint-WebApp/ops/paperclip/BLUEPRINT_AUTOMATION.md`

Do not keep an older repo-local org chart alive once the shared control plane has moved on.
