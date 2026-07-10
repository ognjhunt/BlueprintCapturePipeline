# Isaac worker startup reliability cloud handoff

Date: 2026-07-10

Scope: make the digest-pinned Isaac worker image, provider startup, runtime
canaries, artifact transport, and teardown reliable. This is a simulator support
lane. It does not prove kitchen task completion, policy quality, physical Unitree
G1 readiness, or real-world safety.

## Current evidence snapshot

The current carrier is:

`docker.io/nijelhunt/blueprint-isaac-eval-worker@sha256:435f6ffa1ddb6cfbf72681e30f212d92ab7826420ea026f613e4a4f4c4679acd`

Registry inspection records 24 layers, 10,612,276,453 compressed bytes, a
2,418,853,396-byte largest layer, five layers over 1 GB, a suitable split-layer
layout, and a conservative 1,446-second pre-runtime timeout. The image is split
well enough to avoid one giant monolithic layer, but it is still a large cold
pull.

Live outcomes:

- Attempt 020, RunPod L40S at $0.99/hour: image and Isaac started, but RTX
  rendering failed. The worker log identified NVIDIA driver 570.124.06 as part
  of Isaac Sim 6's rejected Linux R570 interval. This was a host driver problem,
  not an L40S VRAM or RT-core problem.
- Attempt 021, RunPod A40 at $0.44/hour: read-only inventory reported `Medium`
  stock, but create returned HTTP 500, `This machine does not have the resources
  to deploy your pod`. No allocation and no spend occurred.
- Attempt 022, RunPod RTX A6000 at $0.49/hour: PASS. The image marker arrived in
  319.3 seconds. NVIDIA RTX A6000, 49,140 MiB, driver 570.211.01, Isaac Sim 6,
  `RayTracedLighting`, and a 64x64 RGBA frame with 16,384 values all passed.
  The stopped allocation was subsequently deleted and provider inspection
  returned `not_found`. The final inventory contains zero RunPod, Vast, or
  DigitalOcean resources and $0/hour live burn.

Primary evidence:

- `output/kitchen_random_task_e2e_20260710T131557Z/isaac_worker_image_manifest_diagnostic_fresh_v2.json`
- `output/kitchen_random_task_e2e_20260710T131557Z/attempt_020_image_startup_canary/render_output/isaac_worker_runtime_preflight.json`
- `output/kitchen_random_task_e2e_20260710T131557Z/attempt_021_image_startup_canary_a40/isaac_g1_kitchen_parity_job_manifest.json`
- `output/kitchen_random_task_e2e_20260710T131557Z/attempt_022_image_startup_canary_a6000/render_output/isaac_worker_runtime_preflight.json`
- `output/kitchen_random_task_e2e_20260710T131557Z/attempt_022_image_startup_canary_a6000/isaac_g1_kitchen_parity_job_manifest.json`
- `output/kitchen_random_task_e2e_20260710T131557Z/attempt_022_image_startup_canary_a6000/pending_teardown_reap_report.json`
- `output/kitchen_random_task_e2e_20260710T131557Z/attempt_022_image_startup_canary_a6000/final_zero_spend_inventory.json`

The current affected test lane passes 277 tests with:

```bash
python -m pytest -q -m '' \
  tests/test_isaac_worker_runtime_preflight.py \
  tests/test_gpu_render_providers.py \
  tests/test_gpu_startup_reliability.py \
  tests/test_isaac_g1_kitchen_parity_job.py \
  tests/test_isaac_particlefield_render_job.py \
  tests/test_groot_oscar_closed_loop_image.py \
  tests/test_groot_oscar_digitalocean_closed_loop_job.py \
  tests/test_kitchen_task_scaling_preflight.py \
  tests/test_kitchen_task_scaling_task_file.py \
  tests/test_kitchen_random_task_selection.py \
  tests/test_isaac_worker_image_healthcheck.py \
  tests/test_isaac_worker_image_manifest.py
```

## Already repaired in the current worktree

Do not replace these with one-off flags or weaken them:

- The review renderer requires RTX-capable GPUs. H100/H200 may be compute-policy
  workers but are rejected for the Isaac RTX review lane.
- The RunPod default list is L40S, RTX 6000 Ada, RTX A6000, L40, and A40. One
  billable resource is the default; same-provider cold racing is off by default.
- The exact registry manifest is inspected without pulling. The resulting image
  size controls the pre-runtime and marker timeouts.
- The pre-spend guard includes startup plus render/watchdog time for every
  sequential attempt. A cap firing blocks before provider allocation.
- The canary uses the same digest-pinned carrier and a signed Blueprint bundle.
  It verifies `nvidia-smi`, structured driver identity, Isaac import,
  `RayTracedLighting`, and real RGB pixels.
- Isaac Sim 6 Linux R570 versions in `[570.0.0, 570.158.1)` fail before RTX
  initialization. Drivers outside that narrow known-bad range must still pass
  the rendered-frame gate.
- The current source caps empty-scene Replicator asset waiting at ten seconds,
  permits up to three renderer steps, and exits on the first non-empty frame.
  This speed optimization is locally tested; attempt 022 used the immediately
  preceding three-step implementation and remains a valid live pass.
- Runtime JSON is written before `SimulationApp.close()`, preventing Isaac fast
  shutdown from erasing the result.
- A blocked startup canary is terminated rather than kept as a warm worker.
- RunPod HTTP 500 resource errors are now classified as provider capacity
  failures, not as a booted flaky pod.
- Provider snapshots before teardown include machine ID, GPU rate, image, and
  runtime/public-IP state without raw API responses or secrets.
- The base image is digest-pinned in the Dockerfile and build script. A static
  build-time healthcheck verifies package import, image family, Isaac Python,
  and the Unitree G1 USD contract. GPU and RTX checks remain runtime-only.
- Negative manipulation coordinates are serialized as `--arg=<negative-value>`
  so the worker parser does not treat them as options.
- Every artifact and marker is bound to a fresh launch-session nonce.

These changes are currently uncommitted in a heavily dirty worktree. A clean
remote clone will not contain them unless the complete worktree snapshot or a
patch plus all untracked files is supplied.

## Remaining work packages

### P0-1: Atomic canary-to-worker supervisor

Problem: the current job can run a canary and can warm-restart a stopped pod, but
there is no single transaction that validates the worker and then either hands
the same allocation to the full job or deletes it. A successful canary is
stopped for potential reuse and leaves an intentionally open teardown record.
The operator currently has to perform the warm handoff or orphan reap manually.

Implement a provider-neutral startup supervisor, preferably a new module rather
than more flags in `isaac_g1_kitchen_parity_job.py`.

Required input contract:

- immutable run ID and launch nonce;
- provider and ordered GPU type list;
- digest-pinned image and matching manifest diagnostic checksum;
- cumulative attempt count, wall-clock cap, and total dollar cap;
- terminal mode: `promote_to_warm_job` or `terminate_after_canary`;
- optional full-job request to execute after a passing canary.

Required behavior:

1. Run inventory and spend admission before each allocation.
2. Keep at most one billable GPU resource at a time.
3. Allocate, record provider machine identity, and wait for an attempt-bound
   early marker.
4. Run the strict driver/Isaac/RTX canary.
5. On failure, upload all logs/results, terminate, verify provider `not_found`,
   close pending teardown, update the cumulative spend ledger, then retry only
   if the overall cap permits.
6. On pass with `promote_to_warm_job`, transfer ownership of the same allocation
   and pending-teardown record to the full job. Do not cold-pull a second pod.
7. On pass with `terminate_after_canary`, delete the allocation and verify no
   residual disk/volume billing.
8. A process exception or SIGTERM must execute the same finalizer.

Required artifacts:

- `startup_supervisor_manifest.json`;
- `startup_attempts.json` with cumulative spend reservation and actual elapsed
  upper-bound estimates;
- `provider_phase_trace.json`;
- `provider_machine_quarantine.json` references;
- final pending-teardown proof and zero-inventory snapshot, or an explicit
  ownership-transfer receipt when promoted.

Acceptance tests:

- Hermetic fake-provider tests for no-capacity, no-runtime, stale marker, bad
  driver, empty frame, success-and-delete, success-and-promote, exception during
  promotion, and cap exhaustion.
- Assert one live resource maximum and no second cold launch after canary pass.
- Assert every terminal non-promoted path has API-confirmed teardown proof.
- Existing provider, parity-job, spend-guard, and 277-test lanes stay green.

### P0-2: Persistent cross-run machine quarantine

Problem: pre-runtime machine quarantine currently exists only inside one
`launch_with_marker_retry` call. A later command can select the same dead or
driver-incompatible RunPod machine again. Provider inventory usually cannot
reveal the driver until the container executes.

Add a durable, non-secret registry keyed by:

- provider;
- provider machine ID;
- image digest;
- Isaac version;
- failure class;
- observed GPU and driver when available.

Each row must include first/last observed timestamps, evidence paths/checksums,
attempt count, expiry/TTL, and whether the failure happened before runtime or in
the runtime canary. Never quarantine solely because placement, policy, kitchen,
or task validation failed.

Launcher behavior:

- Inspect immediately after allocation.
- If a still-valid quarantine matches the machine/image/Isaac identity and the
  container has not started useful work, terminate it immediately and try a
  different allocation within the cumulative spend cap.
- If the provider cannot exclude a machine during create, explicitly record
  that limitation; do not pretend quarantine guarantees a different host.
- Driver incompatibility learned by the canary must create a quarantine entry
  before teardown.

Acceptance tests must cover TTL expiry, image-digest change, same machine with a
different failure class, corrupted registry, concurrent writers, and no secret
or raw provider payload persistence.

### P0-3: Build and publish a self-contained worker overlay

Problem: the currently proven digest is an Isaac carrier. It does not expose the
new `BLUEPRINT_WORKER_IMAGE_FAMILY` or Blueprint package overlay from the image
itself; the signed runtime bundle supplies the code. This split contract works,
but the Dockerfile healthcheck and baked package improvements are not present in
the immutable digest used by attempt 022.

Build from `deploy/docker/robot_eval_worker/isaac/Dockerfile` using the pinned
Isaac Sim 6 base digest. Do not bake kitchen assets, provider secrets, or policy
tokens into the image. Keep the content-addressed kitchen and policy packages
separate.

Required build properties:

- linux/amd64 manifest;
- versioned tag plus resolved immutable digest;
- small Blueprint overlay layers with apt and pip caches removed;
- `python3 -m blueprint_pipeline.isaac_worker_image_healthcheck --build-time`
  passes during build;
- `/isaac-sim/python.sh` can import `blueprint_pipeline` at runtime;
- image-family and simulator-family environment variables are present;
- Unitree G1 USD exists at the configured path;
- registry manifest diagnostic v2 is generated after push;
- no mutable tag is accepted as final evidence.

After publishing, run both the fast canary and the review-renderer canary from
P0-4 on the new digest. Do not claim success from build completion or registry
inspection alone.

### P0-4: Split fast startup proof from review-renderer proof

Problem: the 64x64 canary proves RTX pixels, but it is below review resolution
and cannot establish that the final Isaac review lane is visually usable. It
must not be used as kitchen placement or review-media evidence. Conversely, a
full kitchen attempt is too expensive to diagnose basic image/driver failure.

Keep two explicit contracts:

1. `fast_startup_canary`: current CUDA, structured driver, Isaac import,
   RayTracedLighting, and first-non-empty-frame check. It may remain 64x64 and
   must state that it does not validate DLSS or review quality.
2. `review_renderer_canary`: render a deterministic, lit USD scene at 480x640
   (or 640x480 with the recorded orientation), save the actual PNG, and include
   a visible Unitree G1 plus floor/target marker. It does not need the kitchen.

The review canary must fail on missing G1, blank/black/white/flat frames,
non-finite pixels, severe clipping, wrong dimensions, missing target marker,
RTX driver-verifier errors, or a frame checksum copied from a prior attempt.
Store a contact sheet only if multiple frames are generated. Bind every image
to the launch nonce and image digest.

The fast canary may gate whether the review canary runs. Only the review canary
may establish `isaac_review_renderer_operational=true`; neither may establish
kitchen placement, policy execution, or task success.

### P1-1: Honest RunPod capacity confidence

Problem: RunPod's read-only catalog returned `Medium` A40 stock with an empty
`availableGpuCounts`, yet create immediately failed for lack of resources. The
probe is advisory and not a reservation.

Change capacity output to distinguish:

- `catalog_reported_stock`;
- `single_gpu_count_known`;
- `reservation_proven=false`;
- `capacity_confidence=advisory|unknown|unavailable`;
- authoritative create outcome.

An empty count plus a textual stock label must not be called immediately
available without the advisory qualifier. A create failure that allocates no
pod is a capacity outcome, not a startup failure and not spend.

Default review-GPU priority should be price-aware but capability-gated:

- A40 first when truly allocatable;
- RTX A6000 next (live-proven on driver 570.211.01 at $0.49/hour);
- L40/L40S/RTX 6000 Ada after that;
- H100/H200 excluded from the RTX review lane but still available to explicitly
  compute-only policy/model workers.

Do not encode one machine's driver as a guarantee for an entire GPU model.

### P1-2: Automatic phase and heartbeat artifacts

Problem: the current run has manual 30-45 second spend-guard snapshots, but the
launcher does not automatically persist every provider phase. This makes a
long pull look opaque and makes postmortems depend on operator polling.

Persist an append-safe or atomically rewritten phase trace at least every 60
seconds and on every state change:

- pre-spend inventory;
- capacity probe;
- allocation requested/created;
- machine identity observed;
- image pull / no runtime;
- public IP/runtime present;
- early marker;
- bundle download;
- CUDA/driver check;
- Isaac start;
- RTX frame;
- result upload;
- stop/promote/delete;
- teardown verification;
- final inventory.

Every row must carry run ID, attempt ID, launch nonce, provider allocation ID,
UTC timestamp, elapsed seconds, and monotonic sequence number. Reject stale,
duplicate, or out-of-order callbacks. Do not store signed URL query strings or
raw API responses.

### P1-3: Startup telemetry and spend reconciliation

Problem: spend admission is bounded and teardown is proven, but the final
inventory reports billing reconciliation as `not_configured`. We can compute a
conservative upper bound from provider rate and allocation age, not an exact
invoice amount.

Add per-phase durations and an explicit distinction among:

- reserved worst-case spend;
- elapsed-rate upper-bound spend;
- provider-reported actual spend, when an authoritative API exists;
- standing stopped-disk/volume cost;
- unknown billing reconciliation.

Do not label an estimate `actual`. The goal-level cumulative ledger must include
failed and successful attempts and enforce the user's total cap before each new
allocation.

### P1-4: Content-addressed kitchen asset startup gate

Problem: the startup canary deliberately ships no kitchen assets. A carrier
pass therefore does not prove the required 1.24 GB kitchen tree is present or
extractable on the worker. Re-uploading it for every diagnostic wastes time and
creates more failure modes.

Add a separate asset-readiness stage using the materialized
`Collected_KitchenRoom` tree or a content-addressed archive:

- expected main USD: `Collected_KitchenRoom/KitchenRoom.usd`;
- current reference inventory: 185 files, approximately 1.24 GB materialized;
- current archive: approximately 699.6 MB;
- verify every file checksum, total count/bytes, main USD presence, and no path
  traversal before extraction;
- record download/extraction progress and free disk;
- allow reuse only when the exact archive digest matches;
- fail before simulator or policy startup if the bundle is incomplete.

This proves provider-side asset presence only. It does not prove Isaac can load
the scene, correct robot placement, or task success.

### P1-5: Adaptive task stance configurator

Problem: scene-grounded spawn logic exists, but the sink candidate exhausted
140 stance candidates without satisfying the provider reach/placement gate.
The selected task was invalidated from fresh scene evidence and the reproducible
random selection moved to `microwave_door`. Its accepted local reference stance
is not provider acceptance.

Implement a bounded scene-configuration loop around deterministic tools. An
OpenAI Agents SDK agent may propose the next candidate or diagnose a failure,
but it must never waive, alter, or fabricate a gate.

Inputs:

- exact kitchen scene digest;
- task registry entry and completion contract;
- resolved target `/root/Microwave017`;
- resolved affordance `/root/Microwave017/Microwave017_Door`;
- G1 robot profile and collision geometry;
- current local reference pose `[-1.229635, 1.471274, 0.84]`, yaw `3.141593`;
- camera and reach/navigation limits.

Loop:

1. Generate candidates around the resolved affordance, not unrelated fixture
   coordinates.
2. Spawn and settle the G1 in Isaac.
3. Measure floor support/uprightness, collisions/clearance, target-facing yaw,
   navigation/reach envelope, affordance visibility, and robot/target framing.
4. Persist exact metrics, rendered views, and rejection reasons for each
   candidate.
5. Update the search region based on measured failures.
6. Terminate only on all deterministic gates PASS or a bounded search budget.

Acceptance requires a fresh robot POV and third-person PNG showing the correct
G1, microwave, door affordance, orientation, and floor support. A schematic dry
preview cannot pass. A watchdog/search-budget stop is blocked, never success.

### P2: Full task/policy path remains unproven

This startup handoff must not be mistaken for the original episode objective.
No live attempt after the pivot has proven:

- full kitchen scene load;
- accepted microwave stance in Isaac;
- fresh GR00T N1.7 + `UNITREE_G1_SONIC` manipulation actions;
- the real controller/FK/skeleton-conditioning route;
- action-aware forward/inverse consistency;
- carried-forward proprioception/state;
- microwave door articulation completion;
- full robot POV/third-person episode quality;
- semantic full-episode success.

Those require a reachable policy server or a genuinely prebaked policy image,
the exact SONIC checkpoint/runtime, and the full kitchen asset package. OSCAR or
another WAM remains an action-conditioned evaluator/support model, never the G1
policy.

## Cloud inputs and permissions

### Code-only work: no local machine required

P0-1, P0-2, P1-1, P1-2, and most of P1-3 can be implemented and tested entirely
in a cloud container with fake provider responses. Supply:

- a complete snapshot of this dirty worktree, including untracked files;
- the attempt 020, 021, and 022 artifacts listed above;
- Python 3.12 with the repo's dev dependencies.

A clean remote clone is insufficient because the current fixes are not
committed or pushed. If a full snapshot cannot be supplied, provide:

1. `git diff --binary` for tracked changes;
2. every untracked file listed by `git status --short`;
3. the fresh run artifact directory.

### Image build/publish work

P0-3 needs:

- a Docker daemon with BuildKit/buildx and linux/amd64 support;
- network access to pull the pinned `nvcr.io/nvidia/isaac-sim:6.0.0` digest;
- any NGC authentication/license acceptance required by that pull;
- write credentials for `docker.io/nijelhunt/blueprint-isaac-eval-worker` or a
  replacement registry namespace;
- permission to push a new immutable tag.

Do not paste credentials into a prompt or commit them. Mount them through the
cloud secret manager. If Claude cannot receive registry push authority, it can
prepare and test the Dockerfile/build command but cannot complete or prove the
new digest.

### Live RunPod/DO canaries

Mount secrets as files with mode 0600. The existing code looks for:

- `~/.blueprint-secrets/runpod_api_key`;
- `~/.blueprint-secrets/isaac_eval_worker_image_ref`;
- for DigitalOcean probes/launches,
  `~/.blueprint-secrets/digitalocean_api_token` or `DIGITALOCEAN_TOKEN_FILE`;
- one S3-compatible object-store set, preferably the existing DigitalOcean
  Spaces files:
  - `~/.blueprint-secrets/digitalocean_spaces_access_key_id`;
  - `~/.blueprint-secrets/digitalocean_spaces_secret_access_key`;
  - `~/.blueprint-secrets/digitalocean_spaces_endpoint_url`;
  - `~/.blueprint-secrets/digitalocean_spaces_bucket`;
  - `~/.blueprint-secrets/digitalocean_spaces_region`.

Equivalent RunPod S3, R2, or AWS files supported by
`wam_provider_object_store.py` are acceptable. Never copy signed URLs from an
old attempt; mint fresh URLs bound to the new attempt.

Live execution also needs explicit authority for paid allocation, a dollar cap,
and a one-resource-at-a-time rule. Without those, Claude should stop after
hermetic tests and prepared manifests.

### Kitchen/placement work

P1-4 and P1-5 need one of:

- the complete materialized `Collected_KitchenRoom` directory;
- the original source archive plus materialization inputs; or
- a fresh signed, content-addressed archive URL plus the expected checksum
  inventory.

The kitchen tree is not safely reconstructible from task names or old
coordinates. Do not allow Claude to fabricate a substitute scene.

### Full GR00T/SONIC episode work

The later P2 lane additionally needs the configured policy runtime:

- `BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL`, or a prebaked worker
  image explicitly confirmed by the existing readiness contract;
- optional token file through
  `BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_TOKEN_FILE`;
- if weights are fetched, the HF token file through
  `BLUEPRINT_UNITREE_GROOT_N17_SONIC_HF_TOKEN_FILE`;
- the official GR00T/WholeBodyControl roots and SONIC deploy assets/checkpoint
  required by `unitree_groot_n17_sonic_policy_runtime.py`;
- any OSCAR/WAM endpoint credentials separately from policy credentials.

If those assets or endpoints are unavailable, Claude can improve contracts and
tests but cannot honestly produce fresh learned-policy actions.

## Non-negotiable acceptance boundary

- Never reduce resolution or quality gates to fit a GPU.
- Never accept H100/H200 as an RTX review GPU merely because CUDA works.
- Never treat a marker, process exit, frame count, cap, timeout, or completed
  video as task success.
- Never accept fixture, replay, zero, synthetic, stale, or fabricated actions in
  a live policy lane.
- Never let an agent or operator waive deterministic placement, collision,
  reach, camera, driver, render, freshness, spend, or teardown gates.
- Keep carrier startup, kitchen asset presence, scene load, placement, policy,
  simulator state success, semantic video review, consistency scoring, and
  real-world truth as separate claims.
- Every paid terminal path must end with API-confirmed deletion or an explicit,
  owned warm-handoff receipt. A stopped RunPod pod is not zero residual billing.
