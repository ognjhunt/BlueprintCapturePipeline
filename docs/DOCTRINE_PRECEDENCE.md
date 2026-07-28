# Doctrine Precedence

This repo's documentation moves at different speeds: shared doctrine blocks
update slowly and mirror across three repos, `README.md` and `docs/` move
weekly, and code moves daily. That speed difference produces apparent
contradictions. This document is the resolution order for agents (Claude,
Codex, or other) and human engineers when documents disagree. It governs how
to *read* conflicts; it is not a license to ignore doctrine.

## For "what exists / what is true right now"

Highest authority first:

1. Code, JSON Schemas, and tests — contracts as actually implemented.
2. Artifact truth per
   [`architecture/source-of-truth-map.md`](architecture/source-of-truth-map.md)
   (raw capture truth outranks every derived artifact; this hierarchy is
   absolute and is not softened by anything below).
3. `README.md` and the current lane docs under `docs/`.
4. The shared doctrine blocks in
   [`../PLATFORM_CONTEXT.md`](../PLATFORM_CONTEXT.md) and
   [`../WORLD_MODEL_STRATEGY_CONTEXT.md`](../WORLD_MODEL_STRATEGY_CONTEXT.md).
5. [`../VISION.md`](../VISION.md) — long-horizon direction and bets. By its own
   rule it never overrides the two documents above, and rungs 3–5 are
   direction, not current capability.

If a lower layer claims a capability the higher layer does not support, the
claim is wrong at the higher layer's level: for example, a doc statement that a
lane is live loses to code that gates the lane off.

## For "what to optimize for / what should be built"

Highest authority first:

1. The Practical Rule For Agents and product doctrine in
   [`../PLATFORM_CONTEXT.md`](../PLATFORM_CONTEXT.md).
2. The strategy and build priorities in
   [`../WORLD_MODEL_STRATEGY_CONTEXT.md`](../WORLD_MODEL_STRATEGY_CONTEXT.md).
3. [`../VISION.md`](../VISION.md) as direction for sequencing bets.
4. Repo-level guides ([`../AGENTS.md`](../AGENTS.md), architecture docs) as
   implementation discipline.

Code being shaped a certain way is evidence about what exists, not an argument
about what should exist. If current code contradicts doctrine on direction,
raise it — do not quietly re-derive strategy from the commit log.

## Dated versus living documents

- A doc with a date in its filename or a "point-in-time" marker is a snapshot.
  Snapshots are historical evidence: label them stale or superseded when the
  world moves, and do not rewrite their contents to match the present.
- Undated docs are living. When a living doc is wrong, fix it in the same
  session you notice, or file the conflict explicitly. Never silently pick a
  side and move on — the next reader inherits the ambiguity.
- Files under `output/` are point-in-time evidence snapshots, never standing
  authority (see the source-of-truth map).

## Shared doctrine blocks

`PLATFORM_CONTEXT.md`, `WORLD_MODEL_STRATEGY_CONTEXT.md`, and the shared block
of `VISION.md` are byte-identical across `BlueprintCapture`,
`BlueprintCapturePipeline`, and `Blueprint-WebApp`. Edit the shared block in
one repo, then mirror to the other repos promptly; until mirrored, note the
pending mirror in the editing repo (see the `VISION.md` footer for the current
mirror state). A shared block that disagrees between repos is resolved by the
most recently edited copy, then mirrored.

## Cross-repo paths

Sibling-repo references follow the sibling-checkout convention in
[`../AGENTS.md`](../AGENTS.md): `$HOME/workspace/<repo>` means "the local
checkout of `<repo>`, wherever it lives in your environment", and a missing
sibling checkout blocks the dependent step rather than being guessed around.
