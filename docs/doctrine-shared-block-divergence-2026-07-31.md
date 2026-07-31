# Shared Doctrine Block Divergence — 2026-07-31

Point-in-time snapshot. Records a blocker found while preparing a doctrine
update; it is not standing authority and should be marked superseded once the
reconciliation lands.

## Finding

`PLATFORM_CONTEXT.md` and `WORLD_MODEL_STRATEGY_CONTEXT.md` are required by
[`DOCTRINE_PRECEDENCE.md`](DOCTRINE_PRECEDENCE.md) to be byte-identical across
`BlueprintCapture`, `BlueprintCapturePipeline`, and `Blueprint-WebApp` inside
their shared markers. **They are not.** The divergence is substantive, not
whitespace.

| Shared block | Pipeline | WebApp | State |
| --- | --- | --- | --- |
| `PLATFORM_CONTEXT` | 119 lines | 63 lines | diverged |
| `WORLD_MODEL_STRATEGY_CONTEXT` | 205 lines | 72 lines | diverged |
| `VISION` (shared block) | 282 lines | 270 lines | diverged |

`BlueprintCapture` was not available in the session that found this and has not
been compared. Treat the three-way state as unknown, not as two-way.

## Why the precedence rule does not resolve it

`DOCTRINE_PRECEDENCE.md` says a disagreeing shared block "is resolved by the
most recently edited copy, then mirrored."

Both copies were edited on 2026-07-29:

- Pipeline `98484a5` "Add claim-level Decision/Evidence Router (#243)" — 13:31:54 −05:00
- WebApp `92e4eac` "Unify Blueprint around Task Evaluation Runs (#426)" — 14:11:57 −05:00

By that rule the WebApp copy wins. But the WebApp copy is also the *shorter*
one, and it is not a superset. Mechanically mirroring it would delete doctrine
that exists only in the Pipeline copy, including:

- **Product Center of Gravity** — the explicit is/is-not list
- **Market Structure** — capturers supply, robot teams buy, site operators as an
  optional third lane, and the rung-2 note on where the standard gets enforced
- **Commercial Wedge Overlay** and **Default Lifecycle**
- **Platform Moat**, **Data Priority**, **Repo-Level Guidance**, and the
  **Decision Rule For Future Sessions** in the world-model block

Conversely the WebApp copy carries material the Pipeline copy lacks, including
the **Result contract** and a tighter **Decision and evidence router** section.

The rule was written for small drift between near-identical files. It does not
survive two substantively different documents, and applying it mechanically
would silently destroy doctrine. `DOCTRINE_PRECEDENCE.md` is explicit that the
wrong move here is to "silently pick a side and move on."

## What this blocks

Any edit to the shared blocks — including the pending
network-as-moat and route-across-real2sim updates — must wait. Editing a
diverged block deepens the divergence and makes reconciliation harder, because
each repo then carries a third variant.

The repo-local ADRs
([`architecture/lean-stack-boundary.md`](architecture/lean-stack-boundary.md),
[`architecture/site-task-adaptation-layer.md`](architecture/site-task-adaptation-layer.md))
are unaffected and can proceed independently.

## Decision needed

Three options, owner's call:

1. **WebApp copy is canonical.** Accept it as an intentional tightening, mirror
   it to the other repos, and accept the loss of Pipeline-only sections.
2. **Merge.** Produce a union document — WebApp's tightened router/result
   language plus the Pipeline-only strategy sections — and mirror that. Larger
   than either current copy.
3. **Split the contract.** Concede that a single byte-identical block across
   three repos with different jobs was the wrong shape; define a genuinely
   shared minimal core plus explicit per-repo appendices, and amend
   `DOCTRINE_PRECEDENCE.md` accordingly.

Option 2 preserves the most and is the safest default. Option 3 is the honest
one if the blocks keep diverging every time a repo ships, which is what
happened here — two PRs on the same day, each editing its own copy.

Whichever is chosen, `BlueprintCapture` must be compared before the mirror, and
the `VISION.md` footer already records a mirror as pending from the 2026-07-29
edit.
