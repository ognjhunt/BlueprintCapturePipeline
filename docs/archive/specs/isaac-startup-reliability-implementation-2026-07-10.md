# Isaac startup reliability implementation status

> Archived implementation snapshot. Verify current release runbooks before use.

Date: 2026-07-10

Implements the code-only work packages from
`docs/specs/isaac-startup-reliability-cloud-handoff-2026-07-10.md`. This lane
is simulator support only: nothing here proves kitchen task completion, policy
quality, physical Unitree G1 readiness, or real-world safety.

## Implemented in this change

### P0-1: Atomic canary-to-worker supervisor

`blueprint_pipeline.isaac_startup_supervisor` — a provider-neutral startup
transaction, not more flags on `isaac_g1_kitchen_parity_job`:

- immutable run ID + launch nonce input contract; digest-pinned image and
  manifest checksum are validated before any provider call;
- inventory + cumulative-spend admission before every allocation; at most one
  billable resource; wall-clock and dollar caps;
- allocate -> machine identity -> durable quarantine check -> attempt-bound
  marker -> strict canary;
- terminal modes `promote_to_warm_job` (ownership-transfer receipt, pending
  teardown handed to the full job, never a second cold pull) and
  `terminate_after_canary` (API-confirmed `not_found` teardown proof plus a
  zero-inventory snapshot);
- one shared finalizer for failure, process exception, and SIGTERM.

Artifacts per run: `startup_supervisor_manifest.json`, `startup_attempts.json`
(reservations + elapsed upper bounds), `provider_phase_trace.json`,
`provider_machine_quarantine.json`, and either
`final_zero_spend_inventory.json` or `ownership_transfer_receipt.json`.

Hermetic fake-provider tests cover: no-capacity, no-runtime, stale marker, bad
driver, empty frame, success-and-delete, success-and-promote, exception during
promotion, cap exhaustion, one-live-resource, no-second-cold-launch, and
API-confirmed teardown on every non-promoted terminal path.

### P0-2: Persistent cross-run machine quarantine

`blueprint_pipeline.machine_quarantine_registry` — durable, non-secret entries
keyed by provider/machine ID/image digest/Isaac version/failure class with
first/last timestamps, evidence checksums, attempt counts, and TTL expiry.
Placement/policy/kitchen/task-validation failures are refused. Entries record
`provider_exclusion_supported: false` — quarantine never pretends to guarantee
a different host. The supervisor terminates a still-pre-runtime reallocation of
a quarantined machine immediately and quarantines driver incompatibilities
learned by the canary before teardown.

### P0-4: Split fast-startup proof from review-renderer proof

- `isaac_worker_runtime_preflight` now carries an explicit
  `canary_contract: fast_startup_canary` block stating the 64x64 check does not
  validate DLSS or review quality and may never set
  `isaac_review_renderer_operational`.
- `blueprint_pipeline.isaac_review_renderer_canary` renders a deterministic lit
  scene at 480x640/640x480 with a visible G1 + floor/target marker, saves the
  real PNG (contact sheet only for multiple frames), and fails on missing G1,
  blank/flat/non-finite/clipped frames, wrong dimensions, missing marker,
  RTX driver-verifier errors, or a checksum copied from a prior attempt. Every
  image is bound to the launch nonce and image digest.

### P1-1: Honest RunPod capacity confidence

`capacity_preflight` now reports `catalog_reported_stock`,
`single_gpu_count_known`, `reservation_proven=false`, per-type and overall
`capacity_confidence` (`advisory|unknown|unavailable`), and names the create
response as the only authoritative capacity source. A create failure that
allocates nothing returns `capacity_outcome=true, allocation_created=false,
spend_occurred=false`. Default review-GPU priority is price-aware and
capability-gated: A40, RTX A6000, L40, L40S, RTX 6000 Ada; H100/H200 remain
excluded from the RTX review lane. No machine's observed driver is encoded as
a guarantee for its GPU model.

### P1-2: Automatic phase and heartbeat artifacts

`blueprint_pipeline.provider_phase_trace` — atomically rewritten trace with the
full phase vocabulary (pre-spend inventory through final inventory), rows
carrying run/attempt IDs, launch nonce, allocation ID, UTC timestamp, elapsed
seconds, and monotonic sequence; sub-interval heartbeats; stale/duplicate/
out-of-order callback rejection; signed-URL query strings and raw provider
responses are refused at write time.

### P1-3: Startup telemetry and spend reconciliation

`blueprint_pipeline.startup_spend_reconciliation` — explicit separation of
reserved worst-case, elapsed-rate upper bound, provider-reported actual (only
from an authoritative billing API), standing stopped-disk cost, and
`not_configured` reconciliation; an estimate is never labeled actual. The
`CumulativeSpendLedger` includes failed and successful attempts and enforces
the total cap before each allocation (used by the supervisor).

### P1-4: Content-addressed kitchen asset startup gate

`blueprint_pipeline.kitchen_asset_startup_gate` — checksum inventory contract
for the materialized `Collected_KitchenRoom` tree, pre-extraction archive
safety (path traversal/links/device members), digest-matched reuse, free-disk
and progress records, and hard failure before simulator or policy startup on
any incompleteness. Proves provider-side asset presence only.

### P1-5: Adaptive task stance configurator

`blueprint_pipeline.adaptive_task_stance_configurator` — bounded deterministic
search around the resolved affordance with injected Isaac measurement backend;
seven deterministic gates plus a fresh-render-evidence gate (schematic/dry
previews cannot pass); an agent hook may propose candidates or diagnose
failures but every waiver attempt is refused and recorded; budget/watchdog
stops are `blocked`, never success; local acceptance is explicitly not provider
acceptance.

### P0-3 (prepared, not published)

`scripts/build_push_isaac_worker_image.sh` now emits the registry manifest
diagnostic v2 via `blueprint_pipeline.isaac_worker_image_manifest`, resolves
and records the immutable digest after push (a mutable tag is refused as final
evidence), and states that build completion requires both the fast startup
canary and the review renderer canary on the new digest before any claim.
Publishing requires Docker + registry push credentials that are not available
in this environment; no new digest is claimed here.

## Explicitly not claimed

- No live RunPod/Vast/DigitalOcean allocation ran here; no new image digest
  was built or pushed.
- P2 remains unproven: kitchen scene load, accepted microwave stance in Isaac,
  fresh GR00T N1.7 + `UNITREE_G1_SONIC` actions, controller/FK routes,
  articulation completion, episode quality, and semantic success all still
  require the live assets/endpoints listed in the handoff.
