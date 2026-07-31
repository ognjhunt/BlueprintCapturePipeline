# Shared Doctrine — Single Source

This directory is the **one editable place** for doctrine shared across
`BlueprintCapture`, `BlueprintCapturePipeline`, and `Blueprint-WebApp`.

Edit here. Never edit a shared block in a repo's `PLATFORM_CONTEXT.md`,
`VISION.md`, or `WORLD_MODEL_STRATEGY_CONTEXT.md` directly — those regions are
generated, and CI will reject a hand edit.

## Files

| Fragment | Generated into |
| --- | --- |
| `platform-context.md` | the `SHARED_PLATFORM_CONTEXT` block of every repo's `PLATFORM_CONTEXT.md` |
| `vision.md` | the `SHARED_VISION` block of every repo's `VISION.md` |
| `world-model-strategy.md` | the `SHARED_WORLD_MODEL_STRATEGY` block of every repo's `WORLD_MODEL_STRATEGY_CONTEXT.md` |

Fragments are plain Markdown with no wrappers and no marker comments. Each
repo keeps its own header and footer around the generated region, which is why
the shared content is spliced rather than provided as a whole file — and why a
git submodule does not solve this by itself.

## Changing shared doctrine

```bash
$EDITOR doctrine/platform-context.md

# splice into every repo this checkout can see, and re-lock
python3 scripts/sync_shared_doctrine.py --write

# then commit in each repo the sync touched
```

Sibling repos are found by the sibling-checkout convention in
[`../AGENTS.md`](../AGENTS.md). A repo that is not checked out is reported and
skipped — never guessed around — so a sync run with a missing sibling leaves
that repo stale and its CI will fail until it is synced. That is deliberate: a
stale repo should be loud, not silent.

## Two halves, deliberately separate

**Propagation** — `scripts/sync_shared_doctrine.py` writes fragments into every
repo and updates `contracts/shared-doctrine.lock.json`. Runs from this repo,
needs sibling checkouts.

**Enforcement** — `scripts/verify_shared_doctrine.py` (and its TypeScript twin
`scripts/doctrine/verify-shared-doctrine.ts` in `Blueprint-WebApp`) compares
committed content against the lock. Runs in every repo's CI, needs no sibling
checkout, no network, and no provider.

The split matters. The previous mechanism was convention plus a sibling
comparison that reported `proposed_dependency_unmerged` and passed whenever the
sibling was absent — which is always true in CI. That is how all three blocks
diverged on 2026-07-29 without any gate firing. Enforcement must not depend on
anything CI does not have.

## Lock status

`contracts/shared-doctrine.lock.json` carries either:

- **`unreconciled`** — the repos do not yet agree. Each is pinned to its own
  recorded baseline, so no new variant can appear, but they are not identical.
- **`locked`** — one canonical digest per block; every repo must match it.

`--write` moves the lock to `locked`. Until the fragments here are populated and
synced, the lock stays `unreconciled` and the reconciliation options are
recorded in
[`../docs/doctrine-shared-block-divergence-2026-07-31.md`](../docs/doctrine-shared-block-divergence-2026-07-31.md).

## Moving the canonical location later

`CANONICAL_REPO` and `DOCTRINE_DIRECTORY` in `scripts/sync_shared_doctrine.py`
are the only two places that name where doctrine lives. Promoting this
directory to a dedicated `blueprint-doctrine` repo is a change to those
constants plus a checkout path — not a rewrite.
