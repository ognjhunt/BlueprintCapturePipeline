# Control-plane disk: never block on space again

Status: design, 2026-09-26. Owner: control-plane storage.

Extends [CONTROL_PLANE_CAPACITY_PLAN.md](CONTROL_PLANE_CAPACITY_PLAN.md) (2026-09-05) and
[CONTROL_PLANE_STORAGE.md](CONTROL_PLANE_STORAGE.md). Those documents landed pins, a reclaim
timer, streaming evidence offload to Backblaze B2, a work volume and a capacity controller. This
document records why the host still ran out of space on 2026-09-26, what is missing, and a phased
plan. Quick wins come first, then the volume split and ephemeral workers.

Backlog: platform capacity that every paid lane depends on. On 2026-09-26 it blocked the
website dishwasher scene (`site-capture-e70b9764…`, development_only; ADP-009 downstream
seams). It also held deploys and other lanes. The implementing session confirms the exact
backlog item in its PR.

## 1. What happened on 2026-09-26 (measured)

| Observation | Evidence |
| --- | --- |
| Root disk 165 GB, **94.5 % used**; work volume 105 GB, **88.8 % used**; forecast "floor within 0.39 days" on both | `blueprint-control-plane-capacity.service` journal, 13:56 and 14:07 UTC |
| A deploy was refused on disk: `need_bytes=2147483648 available_bytes=323833856 floor_bytes=8589934592` | door request `20260926T135524Z-deploy-ca29d053` |
| A scene that was accepted and paid for could not start its build: `scene_whole_chain_capacity_insufficient`, with 10.84 GB free against 8 GiB floor + 5 × 2 GiB flat role footprints | scene intent `scene-ec693ebc…`, 17:22 UTC |
| The reclaim timer ran hourly with offload enabled and found **nothing**: `candidate_count: 0`, `offloaded_bytes: 0`, `removed_bytes: 0`, and derived caches `retained_hot_or_active` | storage-gc journal |
| Release retirement at deploy retired **0 of 95** release trees: `protected_commit_count: 513` | deploy receipt `iteration_414f16fb68c2_door.json` |
| Finished website scene working copies are never reclaimed. `pubsub-handoffs` is classed `work` (queues). An expired scene's message is retried forever and re-staged | `control_plane_storage_roots.py`; the listener retries `consent_expired` every pass |
| "What uses the space?" could not be answered from the door. `task-evaluation-inputs/prepared-references` and `compiled-episodes` count as ~470 GB on a 165 GB disk because of hardlinks | bounded door listing |
| Out-of-band deploys (scratch source checkouts through transient units) bypassed the door. One left GPU admission refusing every sponsored GPU step until a door deploy of a newer commit | `iteration_294564ee_drawer.json`, `gpu_canary_deployed_release_receipt_unverified` |
| Alerts went nowhere actionable: "floor within 0.39 days" repeated all day, and an agent noticed first | capacity journal |

### Root causes

1. **One fixed disk shares everything.** OS, release trees, control-plane state, every lane's
   scene working copies, caches and evidence compete for one root disk (plus one work volume
   capped at 100 GB by the provider). A full scratch area stops deploys, the listener and
   teardown.
2. **The host is treated as the system of record.** Retention is "keep until proven safe". The
   release reference roots (`DEFAULT_RELEASE_RETIREMENT_REFERENCE_ROOTS`) pin 513 commits,
   most of them through stale bindings. Evidence offload only reaches `evidence_cold` run
   directories past their hot window, which on this host was none.
3. **Reservations are guesses.** `ROLE_FOOTPRINT_BYTES` is a flat 2 GiB per role, and
   `whole_chain_admission` sums five of them (10.74 GB) regardless of actual use.
4. **Alerts do not cause action.** `BLUEPRINT_CAPACITY_AUTORESIZE_ACK` and
   `BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL` either weren't set or didn't reach a human. The
   work volume is at the provider's 100 GB limit.
5. **Out-of-band work.** SSH deploys and ad-hoc scripts create data and state (release trees,
   provenance files, stopped timers) that no retention or admission logic knows about.

## 2. Principles

1. **Object storage is the source of truth; host disk is a disposable cache.** A stage is done
   only when its outputs are sealed, content-addressed and uploaded to B2 (bulk) or Firebase
   Storage (website capture), with readback. Local copies are then evictable at any time, and
   consumers re-fetch by digest. Evidence is never only on the host.
2. **Every run owns a workspace with a lifecycle.** Its quota is reserved at admission and sized
   from measured history. The run's state machine deletes the workspace at a terminal state,
   after verifying the upload. A GC should not have to prove safety after the fact.
3. **Storage classes live on separate volumes.** System (OS and releases), control-plane state
   (small, durable) and scratch (large, growable). A full scratch volume queues new runs; it never
   blocks deploys, the listener, teardown or provider-zero.
4. **Every pin expires.** Pins and bindings carry an owner, a reason, a TTL and the run they serve.
   They lapse when that run ends.
5. **Capacity is an SLO with automatic action.** Measure, forecast, grow inside a budget, and
   page a human at three days of headroom.
6. **The control host coordinates; workers compute.** Heavy CPU and GPU work runs on ephemeral
   workers with object-store I/O.
7. **Only the door mutates the host.** No SSH writes or out-of-band deploys, except a logged
   break-glass path.

## 3. Phased plan

Each item lands as its own PR with hermetic tests, following `AGENTS.md`: fail closed, never
delete evidence (offload instead), and keep provenance intact.

### Phase 0: owner actions (hours; unblocks today)

- Resize the root disk, or free space with the survey and deletions in §5.
- Request a DigitalOcean volume limit above 100 GB (see CONTROL_PLANE_CAPACITY_PLAN.md).
  Then raise `BLUEPRINT_CAPACITY_VOLUME_MAX_GIB` and set
  `BLUEPRINT_CAPACITY_AUTORESIZE_ACK=grow-control-plane-volume`.
- Point `BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL` at something that pages a person.

### Phase 1: quick wins (days; no architecture change)

**1a. Measured reservations.** Files: `control_plane_disk_budget.py`,
`control_plane_capacity_controller.py`, `task_evaluation_scene_progression.py`,
`task_evaluation_scene_capacity_recovery.py`.

- When a `DiskReservation` is released, record the role's observed peak bytes: the workspace
  byte delta, counting unique inodes, for the directory it reserved against. Store the record in
  the reservation ledger history.
- Add `measured_footprint_bytes(role)`: p95 of the last N (≥ 10) samples × 1.25, clamped to
  `[64 MiB, ROLE_FOOTPRINT_BYTES[role]]`. It falls back to the constant when history is short,
  so a new role stays conservative.
- `whole_chain_admission` and each stage's `reserve_control_plane_disk` use the measured value.
  The receipt names the basis (`measured_p95` or `declared_default`) and the sample count.
- `control_plane_deploy` measures the real staged release size, not 2 GiB.
- Tests: short history keeps the constant; a measured p95 admits a chain that the constant
  refuses; a sample above the clamp cannot raise the reservation above the declared ceiling;
  the receipt basis is recorded.

**1b. Expiring pins and release bindings.** Files: `control_plane_storage_pins.py` (pins already
expire after 30 days), `task_evaluation_release_retention.py`,
`control_plane_release_retirement.py`, `scripts/deploy_control_plane_commit.py`.

- Give every file under `task-evaluation-release-retention-bindings` and `standing-authorizations`
  `owner`, `reason`, `expires_at_epoch` and `run_ref`. `run_ref` is the queue envelope, scene
  intent or launch it serves.
- Retirement ignores a binding when it has expired or its `run_ref` is terminal (settled,
  revoked, stranded). Legacy bindings without these fields get a one-time migration that
  stamps `expires_at_epoch = now + 14d` and reports them. They are never silently dropped.
- The deploy receipt lists protected commits grouped by reason, and alerts when more than 20
  commits are protected by bindings.
- `in_use_by_live_process` protection (#2315) stays.
- Tests: an expired binding no longer protects a commit; a live `run_ref` still does; a binding
  for a terminal run lapses; migration stamps and reports.

**1c. Scene workspace retirement.** Files: `control_plane_storage_roots.py`,
`pubsub_handoff_listener.py`, a new `website_scene_workspace_retention.py`, the storage-gc wiring,
and the door.

- Give `pubsub-handoffs/**/scenes/<scene>` a storage class `scene_workspace` (today it's
  `work`).
- A workspace is retirable when **all** of these hold:
  - its handoff job ledger is terminal (completed and acknowledged, or its authority expired or
    was revoked and its message was acknowledged or dead-lettered);
  - every pipeline output named in its manifests has a `gs://` URI whose object exists with a
    matching digest (readback);
  - it has been idle ≥ 48 h;
  - no live pin, process or queue message references it.
- Retirement writes `<scene>.retired.v1.json`. That's a receipt with the manifest digests and
  object URIs, so a later consumer re-stages from cloud storage.
- The listener treats `consent_expired`, `source_revoked` and similar authority endings as
  **terminal**. It acknowledges the message with a terminal receipt instead of retrying forever.
  Today an expired scene re-stages and re-fails every few minutes, and its working copy can never
  be retired.
- Door action `retire-scene-workspace <scene_id>`: plan, then apply, with a receipt. It's the
  supported way to clean up by hand.
- Tests: a workspace isn't retired when any output's readback fails, while it's pinned, or while
  it's non-terminal; an expired authority is acknowledged terminally; the retire receipt replays.

**1d. Accurate usage reporting.** Files: `control_plane_capacity_controller.py` and the door
status.

- The capacity report adds per-root **unique-inode** bytes (hardlinks counted once) and the top
  ten consumers by storage class and by owner (lane, scene, run).
- Door `status` shows those numbers, so "what's using space?" is one call.
- Tests: hardlinked files count once; unclassified roots are reported.

### Phase 2: volume split (a week)

- Move every bulk root to the scratch volume with `deploy/host/mount_work_volume.sh`: `cache`,
  `evidence_cold`, `scene_workspace`, `pubsub-handoffs`, `task-evaluation-inputs`, and native
  run work.
  - Keep the root disk for the OS, releases and `evidence_hot` / `ledger` state.
  - Put control-plane state on its own small, durable volume.
- Admission floors are per volume. `control_plane_deploy`, listener staging of small files,
  teardown and provider-zero reserve against the system volume, so they aren't blocked by a full
  scratch volume.
- The scratch volume grows through the existing resize path (`plan_volume_resize`). If the
  provider's 100 GB limit persists, use a pool of volumes per storage class instead.
- Releases: replace per-commit full checkouts with a single git object store plus worktrees, or
  release images.
  - Keep the last N (3) releases plus live-process and live-binding releases.
  - Content-address runtime trees by recipe digest (CONTROL_PLANE_CAPACITY_PLAN.md already
    names this).
- Tests: a filled scratch volume refuses a new chain but admits a deploy and a teardown; the
  resize plan stays within its budget; release retention with the shared object store.

### Phase 3: ephemeral workers (weeks; launch/beta)

- Run scene preparation, episode compilation, and the SAM, registration and CPU pre-stage work
  on ephemeral workers.
  - Workers pull from the existing queues and read and write only content-addressed objects in
    B2 or Firebase Storage.
  - A worker's disk disappears with it.
  - Concurrency is governed by paid-launch slots (3 today) and worker counts, not by host disk.
- Stream, don't download: lanes seal from the object-store manifest and fetch only what
  interpretation needs (the 4.2 GB `vast_provider_runtime_output.zip` pattern in 16 adapters).
- Queue with an ETA instead of refusing: intake accepts `queued_for_capacity` with the capacity
  forecast.
- The control host keeps state, queues, the door and the WebApp bridge, at megabytes per run.
- Before beta, load-test N concurrent scenes (N = expected beta concurrency × 2). Record the
  per-stage p95 disk, CPU and wall time, and size worker counts and per-tenant quotas from it.

### Phase 4: process (continuous)

- Every host mutation goes through the door: deploy, unit start/stop, workspace retirement, and
  reclaim. `deploy_control_plane_commit.py` refuses sources outside the canonical checkout (see
  the queued task on out-of-band deploys and GPU admission).
- Break-glass SSH writes a signed note to `cleanup-receipts/` that the next deploy reports.
- Capacity runbook: page at three days of headroom, and grow or reclaim by the procedure.
  Nobody hand-deletes evidence.
- Timers that another lane pauses (listener, scene progression) are paused through a door
  "hold" with an owner and an expiry, not a bare `systemctl stop`.

## 4. Definition of done

- A full website scene chain is admitted with measured reservations, and its workspace is
  retired automatically after upload verification.
- The deploy receipt shows fewer than 20 binding-protected commits, and releases older than the
  last three retire.
- The capacity report attributes more than 90 % of used bytes to a storage class and owner.
- A filled scratch volume cannot block a deploy, the listener or teardown (tested).
- A human is paged at three days of headroom.
- Load test: N concurrent scenes complete with host disk flat.

## 5. Appendix: 2026-09-26 manual relief (owner, SSH)

Read-only survey:

```bash
sudo df -h / /mnt/blueprint-work
sudo du -xsh --one-file-system /var/lib/blueprint/* /opt/blueprint/* 2>/dev/null | sort -h | tail -15
sudo journalctl --disk-usage
command -v docker >/dev/null && sudo docker system df
```

Safe to delete: finished website scene working copies. Their runs completed, their handoffs
were acknowledged, and their inputs and outputs are in Firebase Storage.

```bash
cd /var/lib/blueprint/pubsub-handoffs/blueprint-8c1ca.appspot.com/scenes
sudo rm -rf site-capture-ae539f2c-f6aa-4cbd-9f99-3ed72017791e \
            site-capture-f8f79b26-d813-4976-bb0a-43f282a0fc14 \
            site-capture-7d655e25-34f2-4961-ae9c-6da11621ce8f
```

Do not delete `site-capture-e41a08cd…` (expired). Its message is still retried and would
re-stage it; Phase 1c fixes this.
