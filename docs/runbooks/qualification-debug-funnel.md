# Qualification debug funnel

Default operating procedure for the single-G1-kitchen qualification lane (and any
future GPU episode lane). Encoded 2026-07-25 after attempts 065–067 each paid a
full merge → thin-rebuild → cold-boot cycle (~90 minutes, ~$0.50) to discover a
single defect at the very end of the toll. Every defect gets found at the
highest tier that can find it; discovery at Tier 3 prices is the anti-pattern.

## The three tiers

| Tier | Cost | What runs | Finds |
| --- | --- | --- | --- |
| 1 — hermetic | free, seconds | bare `pytest`: contract tests, argv welds, producer-shape round-trips | anything expressible against committed fixtures |
| 2 — live-box refresh | ~$0.13, ~10 min | `refresh-bootstrap` overlay push to a warm GPU, re-run on the same box | overlay-plane runtime defects |
| 3 — merge + sealed run | ~$0.50, ~90 min | CI, thin rebuild, fresh allocate on merged main | image/build-plane defects; final confirmation |

Rules:

- Before any paid tier, exhaust Tier 1. Attempt 067's defect (the sealed
  bundle argv omits `--start-frame-evidence`) was findable from the bundle zip
  already on local disk; it was paid for at Tier 3. The argv fixture test
  (`tests/test_sealed_bundle_argv_contract.py`) now pins that class.
- Tier 2 is the default debugging vehicle for overlay-plane code. Tier 3 is
  **confirmation on the same merged commit a Tier-2 loop already verified** —
  never the discovery vehicle.
- A refresh-verified run is debug evidence only: `overlay_revision > 1` in the
  session manifest says so. Evidence-grade results still require the sealed
  path from merged main through the full allocate gate.
- Skip Tier 2 only for a high-confidence single fix in a contract-tested class,
  where merge-first costs one boot instead of two.

## 10-second triage: which tier can find it?

Read the failing traceback path.

- `/workspace/runtime_overlay/…` → overlay plane → **Tier 2**.
- `/opt/…`, the GEAR-SONIC binary, TensorRT, CUDA, embedded checkpoints, venv
  deps, the entrypoint model verifier → image plane → **Tier 3** (front-load
  with build-time probes; no legitimate shortcut exists — hand-patching a
  container breaks provenance and the no-hand-fixes rule).
- Not on the GPU at all (builder, allocator, staging) → build plane →
  fix + hermetic test, no GPU needed.

## The Tier-2 loop, exactly as first driven (attempt 067, instance 45807515)

1. Fix code locally; run the relevant hermetic suites.
2. `refresh-bootstrap` (no URL) → generates the payload from the working tree,
   records `pending_refresh` (path, sha256, from/to revision) in the session
   manifest.
3. Stage the payload with `wam_provider_object_store` → signed URL files.
4. Stop all components first — the refresh gate requires it:
   `stop-component` for each of `episode groot controller isaac bridge`, else
   `qualification_refresh_requires_all_components_stopped` (rc 70).
5. `refresh-bootstrap --provider-bootstrap-url-file … --execute` →
   `bootstrap_refreshed_continuing_spend`, `overlay_revision` increments.
6. `collect` the prior attempt (the session refuses a re-run before collection:
   `qualification_collect_required_before_episode_rerun`), then `run`.

## Tier-2 precondition: allocate the box BEFORE merging the fix

Tier 2 refreshes a **live** box; it cannot create one. Allocation requires all
of `checkout == release.source_commit`, `checkout == origin/main`,
`checkout == remote main`, and a clean tree
(`_source_checkout_blockers`, `paid_resource_allocator.py`). The moment `main`
advances past the commit the release image was built from, **no new box can be
allocated at all** until a thin rebuild republishes the release at the new
HEAD. Tearing down the last box and then merging therefore forfeits Tier 2 and
forces a ~77-minute rebuild — the exact toll the funnel exists to avoid.

Order of operations when a fix is headed for a live check:

1. Allocate (or keep) the box while `HEAD == origin/main == release commit`.
2. Fix, merge, whatever — the box is already bound and stays refreshable.
3. `refresh-bootstrap` the overlay onto it.

Corollary: do not tear down a healthy box just because its episode failed. A
box whose failure was diagnosed as overlay-plane is the cheapest asset in the
loop; the teardown reflex is what converts a $0.13 fix into a $0.70 one. Tear
down only after the fix is verified, or when the defect is image-plane and a
rebuild is unavoidable anyway. (First paid: attempt 069, 2026-07-25 — box torn
down, fix merged, then Tier 2 found unreachable.)

## Known Tier-2 trap: overlay import closure (split-brain)

`blueprint_pipeline` resolves as a namespace package across TWO roots on the
pod: the refresh overlay (`/workspace/runtime_overlay/package/`) and the sealed
image (`/opt/blueprint/release-src/`). A shipped member that imports a sibling
NOT in `RUNTIME_PACKAGE_OVERLAY_MODULES`
(`single_g1_kitchen_episode_runpod.py`) silently resolves that sibling from the
older image tree. First hit live: overlay rev 2 shipped the new eval but not
`initial_policy_observation_contract.py` → `ImportError: cannot import name
'resolve_start_frame_evidence_path' … (/opt/blueprint/release-src)`.

Until the refresh builder validates import closure against the bound release
commit (open follow-up): when a change adds or modifies any module imported by
an overlay member, add it to `RUNTIME_PACKAGE_OVERLAY_MODULES` in the same
change.

## Why the image plane gets no fast loop — and what defends it instead

Image-layer iteration has a hard floor (merge → thin rebuild → boot). Its
defenses are front-loaded and already exist: build-time import matrices,
`pip check`, the embedded-carrier compatibility audit and `--build-time`
healthcheck on the $0.17/hr CPU builder; the free architecture and live
prerequisite gates before any allocation; startup gates that fail in the first
minutes of a boot; and class-killer admission gates (the TensorRT compute-cap
ceiling is the template — an un-refreshable defect class made unrentable).
Empirically the plane has earned this: the foundation is unchanged since
`@ab8fbccb`, and every failure since #171 has been builder, selector, or
overlay — not image content.

## Reading episode verdicts: failure is the product

A stance abort, stall, or criterion miss with a truthful measurement chain is a
product-grade negative, not a defect. Run 7 (first door contact) is the
template: dead-upright gravity through the approach, ~30 degrees at contact,
~35 and growing while pulling, abort. The gate measured a real balance failure.
Fix gates only when the MEASUREMENT lies (the directional gate aimed at the
camera-framing point; the joint-only stall guillotining approach phases) —
never to make verdicts friendlier.

## Host-capability defects: the third plane

Attempts 066–067 taught overlay-vs-image. Attempt 068 added a plane neither
tier covers: the **host**. Its residency proof reported all four roles absent
while they held ~45 GiB on the one visible GPU. Cause: `nvidia-smi` reports
root-namespace PIDs, and this Vast host's container runtime never exposed the
outer `NSpid` chain, so the host→local translation was not merely unparsed but
*absent in principle*. Run 7 passed the identical gate on a host whose runtime
did leak the chain.

Diagnosing this class:

- The tell is a measurement that is *uniformly* empty rather than wrong —
  `role_observed_sample_counts` all zero across 212 samples, while
  `peak_gpu_memory_used_mib` proves the work was happening.
- Corroborate against the last passing run's report before touching code. Run 7
  (243 observations) vs 068 (0) localized this to the host in one comparison.
- `process_name: "[Not Found]"` and a one-element `ancestor_chain` are the
  host saying "I cannot attribute this PID." Never let a fallback silently
  reinterpret that as "this PID is someone else's."

The durable fix is never "retry on another host" — that is a coin flip that
re-bills the same discovery. Prefer, in order: (1) an alternative measurement
that does not depend on the missing capability, (2) a distinct blocker naming
the capability, (3) an admission gate making the host class unrentable. The
device-handle fallback in `gpu_residency_attribution.py` is (1): it asks our
own processes whether they hold `/dev/nvidia*` instead of asking the GPU which
PIDs it holds, so it is namespace-local and works on both runtimes. It refuses
to conclude anything when more than one GPU is visible, because an open handle
cannot then identify which GPU — bounded soundness beats a friendlier verdict.

## Branch hygiene: squash-merge orphans its own follow-ups

A squash merge rewrites history, so any branch cut from the pre-merge head goes
`mergeState: DIRTY` and cannot be rebased cleanly. Cut follow-up branches from
the post-merge `main`, and when a PR does strand, rebuild it as fresh files on
current `main` rather than fighting the conflict — that is what turned #183's
orphan into the clean #186.

## Known measurement limitation: frozen-seed conditioning FK

The official-executor FK replays every chunk from the canonical initial state
(run 7: identical wrist start across all six steps, even after live contact).
Per-step approach gains are therefore command-intent, not live progress;
termination evidence carries `approach_measurement_source` saying so, and a
limitation pin in the closed-loop tests flips red when live seeding lands.
The live protocol-v4 joint state needed for seeding is already available in
every completion result (`post_action_policy_state`).
