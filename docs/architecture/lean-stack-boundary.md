# Lean Stack Boundary

Status: proposed, version 0 (2026-07-31)

Companion to [`decision-evidence-router.md`](decision-evidence-router.md)
(accepted, 2026-07-29) and
[`site-task-adaptation-layer.md`](site-task-adaptation-layer.md) (proposed).
The router says *what* Blueprint routes across. This says *how much of that we
are allowed to own*.

## Decision

Blueprint owns the Task Pack and the results. Everything underneath is rented.

**Owned** — the artifacts that make a site reusable and a decision defensible:

- raw capture truth, provenance, rights, consent, privacy
- Site-Task Testbed compilation (the Task Pack) and its immutable versioning
- the claim router, Evidence Method Profiles, and qualification records
- normalized Evidence Results, the Decision Envelope, abstention, next-cheapest-
  experiment
- the Physical Outcome Join and the calibration ledger built from it

**Rented** — replaceable behind adapters, never a company asset:

- reconstruction and real2sim conversion
- simulators and physics engines
- world models and learned evaluators
- compute, storage, and orchestration providers
- policy endpoints and checkpoints

This is not new strategy. It is the router ADR's non-goals applied to the
repository itself.

## Why this needs to be written down

Measured 2026-07-31 on `src/blueprint_pipeline/` (624 modules):

| Surface | Distinct modules |
| --- | --- |
| World-model backends (OSCAR, Cosmos, Ctrl-World, GR00T, WAM) | 129 |
| Compute providers (RunPod, Vast, DigitalOcean, Lambda, Cloud Run, GPU pools) | 67 |
| Traditional simulation (Isaac, MuJoCo) | 34 |
| Policy-ranking experiment machinery | 50 |

Against that:

- **three** authorized executable evidence adapters — analytic reachability,
  captured visibility, swept-AABB collision
- the frozen policy-ranking verdict is `thesis_not_supported`; the specialized
  successor is `inconclusive`
- **zero** rights-cleared real captures have been through the pipeline

The 129 world-model modules and 50 ranking modules exist to serve claims that no
qualification record currently supports. That is not a criticism of the
experiments — they produced honest negative results, which is their job. It is
an observation that experiment residue is now the largest thing in the
repository, and nothing distinguishes it from infrastructure.

## The admission rule

The router already refuses to treat runnability as qualification:

> Provider identity, visual realism, parameter count, and runnable defaults are
> not qualification.

Applied to the stack: **a backend or provider integration whose method has never
held a qualification record is a parked experiment, not infrastructure.** It
does not get maintenance, dependency upgrades, CI time, or a place in the
onboarding map.

Adding a new backend or provider requires naming, in advance:

1. the claim it is expected to qualify for, and the claim type's authority tier;
2. why an already-integrated method cannot serve that claim more cheaply;
3. what result would retire it.

Point 3 is the one currently missing everywhere. An integration with no
retirement condition never leaves.

## Disposition

Three states, applied per integration:

- **Load-bearing** — holds a qualification record, or is required by one.
  Maintained normally.
- **Parked** — an experiment reached a terminal verdict and the integration is
  retained as reproduction evidence. Frozen: excluded from dependency upgrades,
  excluded from the fast lane, marked in the onboarding map, not extended.
  Preserving the negative result matters; carrying it as live code does not.
- **Delete** — neither load-bearing nor evidence for a published result.

Parking is preferred over deletion wherever an integration underwrites a
published experiment, because those results are the honest record and the repo's
existing discipline is to preserve failures rather than erase them.

## Consequence for the moat

If the durable moat is the operating network — site access, capture density,
robot-team relationships, and partnership terms — then engineering exists to
make each additional site cheaper than the last. A stack whose weight sits in
backends serving unqualified claims does the opposite: it spends the capacity
that would otherwise go to site throughput.

The reusable Task Pack is what converts field work into an asset. It is the one
place where owning more, not less, is correct.

## Non-goals

This decision does not delete anything by itself, does not retire any published
experimental result, does not change the router or any contract, and does not
authorize dropping a provider with live paid-resource obligations.

## Open questions

- Which single compute lane and which single simulation backend are the
  supported defaults, with the rest parked.
- Whether "parked" needs a machine-checkable marker so CI enforces the
  exclusions rather than relying on convention.
- Who decides disposition per integration, and on what cadence.
