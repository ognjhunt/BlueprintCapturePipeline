# Control-plane storage: budget, content stores, and reclaim

Status: operating contract for the production control plane (the single
Task Evaluation host). Measured 2026-09-02 on a 154 GB root disk at 98 %.

## Why the disk kept filling

Three structural causes, each now closed by code rather than by cleanup:

1. **Wrong content-addressing granularity.** The runtime-source wrapper embedded
   the invariant 4.29 GB runtime packet together with a per-release identity
   manifest and was stored by whole-file digest. Two same-size wrappers on the
   host differed only in `task_evaluation_adapter_bundle_manifest.v1.json`, so
   every deploy minted a new 4.29 GB blob (11 blobs, 47 GB).
2. **Immutable per-commit trees with no lifecycle.** Each deploy published a
   release worktree and two runtime trees per SHA and nothing retired them.
3. **No admission control.** Nothing on the host reserved space before a
   deploy, preparation, compile, or activation began; failure arrived as
   `ENOSPC` mid-write.

## Admission gate (`blueprint_pipeline.control_plane_disk_budget`)

Every write-heavy control-plane role reserves its footprint in a shared ledger
(`/var/lib/blueprint/pipeline-control-plane/disk-reservations`, `root:blueprint`
`2770`) before it mutates anything. Admission is
`free - floor - live reservations >= need`, where the floor is
`max(8 GiB, 5 % of the disk)` (`BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES`) and a
live reservation is a ledger entry on the same device, inside its TTL, whose
pid is alive.
A refusal is the typed blocker
`control_plane_disk_budget_exceeded:<role>:need_bytes=..:available_bytes=..:free_bytes=..:floor_bytes=..:reserved_bytes=..`
and never a host path.

| Role | Declared ceiling | Its reservation holds | Where it refuses |
|---|---|---|---|
| `control_plane_deploy` | 2 GiB | the release's git-tree estimate: blob bytes × 1.25, plus 4 KiB per file, plus 256 MiB (the measured footprint if the tree cannot be listed) | `deploy_control_plane_commit.py` before provenance or staging |
| `launch_preparation` | 2 GiB | exact bytes of references the content store lacks, plus 256 MiB; the runtime-source layer separately on a miss | preparation worker, before any fetch |
| `episode_compilation` | 2 GiB | exact bytes of runtime members the member store lacks, plus 256 MiB | compile worker, before the output directory exists |
| `launch_activation` | 2 GiB | the measured footprint, never less than the reference bytes the request declares plus 256 MiB | activation worker |
| `policy_canary_dispatch` | 2 GiB | the measured footprint | canary dispatcher queue boundary |

`launch_dispatch` has no reservation call site yet (PR 6 adds one), so
whole-chain admission counts it at its declared ceiling.

The intake version endpoint reports `disk_headroom` with `refused_roles` and each
role's `footprints`; the launch-preparation, launch-activation, and
task-evaluation-launch intakes refuse a submission (HTTP 503, typed blocker)
while its role is refused.

Declared ceilings can be tuned with
`BLUEPRINT_CONTROL_PLANE_DISK_FOOTPRINT_<ROLE>_BYTES`.

### Measured footprints (`blueprint_pipeline.control_plane_disk_footprints`)

A declared footprint is a ceiling, not what admission keeps reserving. Each
reservation names the per-job directory it writes (its `workspace`: the
preparation, compilation, activation or canary run directory under the shared
parent it reserves against) and, when released, records how much that directory
grew. Growth counts each inode once, in allocated blocks
(`control_plane_disk_usage.tree_usage`), so hardlinked names are not double
counted. Names hardlinked from a content store still count as this job's bytes;
that over-counts cache hits, which errs conservative, and the ceiling bounds
it. The cpu prestage and semantic pretraining jobs sample just before they remove
their scratch trees, and the deploy observes the release checkout and runtime
trees it created (a redeploy that created nothing records no sample).

Samples are appended to `<ledger>/history/<role>.jsonl` (`root:blueprint`
`2770`, installed and verified by the deploy), one line per release with the
workload, outcome, reserved bytes, duration, the workspace's baseline and
whether it was fresh, and compacted to the newest 200 lines under the ledger
lock. Every sample names how its job ended, and only `completed` samples shape
admission:

| Outcome | Meaning |
|---|---|
| `completed` | the job finished from a fresh workspace and every byte was read |
| `failed` | the job raised (its reservation context exited on the exception, or a worker caught it and released with this outcome: preparation, compilation, activation) |
| `blocked` | the job returned a blocked result before finishing: a preparation paused on its children or on capacity, a canary run with a `blocked…` status, a scene factory short of publication, or a replay that did not succeed (only a child replay whose stage completed, or a parent replay queued for production, counts) |
| `resumed` | the pass started from a workspace an earlier pass had filled, so its growth is not the job's footprint |
| `incomplete` | part of the workspace could not be read, so the measurement under-counts |

A workspace is fresh when it is bound if it did not exist or held less than
1 MiB. Callers that know better say so (`fresh=` on the reservation or on
`bind_workspace`): the cpu prestage asserts freshness right after it clears its
work dir, and these passes reserve as not fresh because an earlier pass already
wrote there: a canary whose session authority or allocator start is on disk (it
resumes a run awaiting billing, provider zero or interpretation), a public
bootstrap whose progress file exists, a launch preparation whose directory
exists, and a scene preparation attempt whose materialized output exists.

Once a role has at least 10 completed samples among its newest 50, its footprint
is the nearest-rank p95 of each workload's samples, taken at the largest
workload (a rare large workload is never averaged away by a frequent small
one), × 1.25, clamped as `min(declared ceiling, max(64 MiB, p95 × 1.25))`, so
the declared ceiling always wins. Until then, or whenever the history cannot be
read, it is the declared ceiling, so a new role stays conservative; a sample
above the ceiling can never raise a reservation. Activation samples are
labelled by lane. Reservations made without explicit bytes hold this
footprint (or `minimum_bytes`, what the job declares, when that is larger), and
every reservation receipt names its `footprint_basis` (`measured_p95`,
`declared_default`, `declared_minimum` or `caller_exact`), its
`footprint_sample_count` and its `workload`.

Intake headroom, the capacity controller (`measure_mount`),
`whole_chain_admission` and the chain preflight use the ledger's own floor,
live-reservation rule and footprints, so a projected refusal is the refusal the
workers will make. A reservation ledger that cannot be read is never read as
empty: the controller reports the mount as unreadable and the preflight reports
a `disk_reservations_unreadable` blocker. `whole_chain_admission` sums the chain
roles' footprints into `required_workspace_bytes` and reports
`required_workspace_basis`: `measured_p95` when every chain role is measured,
`declared_default` when none is, `mixed` otherwise, with the per-role
`footprints` beside it. Until PR 6 gives `launch_dispatch` a call site, its
footprint stays declared, so `mixed` is the steady state once the other chain
roles have history. The capacity controller reports these footprints, like its
forecast, only in `latest.json`; its never-pruned `history.jsonl` keeps each
tick's measurement alone.

Pid liveness is the primary liveness signal and the TTL only a backstop for a
recycled pid. A job holds its reservation for at most its systemd unit's
`TimeoutStartSec`, so each role's TTL outlives that timeout (a test pins this
against `deploy/systemd`): `cpu_prestage` and `semantic_pretraining` 12 h,
`stage_replay`, `policy_canary_dispatch` and `launch_dispatch` 6 h,
`control_plane_deploy` and `evidence_offload` 4 h, every other role 2 h.

The ledger directory is group-writable and root uses it too, so nothing in it
is followed through a symlink: the ledger, its lock and the history directory
are opened with `O_NOFOLLOW`, history writes work relative to the history
directory's descriptor, and a mode is repaired only through a descriptor, only
by the file's owner.

## Runtime-source wrappers with external layers

`build_task_evaluation_runtime_source_bundle` accepts an external layer store.
Members at or above `--external-layer-min-bytes` (default 64 MiB) are written
once to `<store>/sha256/<digest>/<name>` and listed in the wrapper manifest as
`{"external_layer": {"transport": "content_addressed_external_layer.v1", "uri": ...}}`
instead of being archived. The wrapper stays a few kilobytes and keeps its
per-release identity bindings; the layer digest is stable across releases, so
the preparation content store holds one copy however many wrappers name it.

Build and publish (the URI prefix must be the artifact bucket and the
`native-runtime-source-layer` kind, because the publisher derives object keys
from content digests and the wrapper embeds the resulting URI verbatim):

```bash
python -m blueprint_pipeline.task_evaluation_native_arena_preparation_adapter build-runtime-source \
  --source-root <dir with native_task_runtime_sources.zip and its receipt> \
  --output <wrapper.zip> \
  --expected-production-commit <sha> --runtime-id native-arena --runtime-version isaac-2026-1 \
  --external-layer-store-root <local layer store> \
  --external-layer-uri-prefix s3://<artifact-bucket>/blueprint/arm-decision-proof-v1/configured-scenes/artifacts/native-runtime-source-layer \
  --receipt-out <build-receipt.json>
```

```bash
python -m blueprint_pipeline.task_evaluation_native_arena_preparation_adapter publish-runtime-source-layers \
  --receipt <build-receipt.json>
```

Then publish the wrapper itself exactly as before (`publish_configured_scene_artifact`,
kind `native-runtime-source`) and reference it from the Website request.

Consumers:

- **Preparation** materializes the wrapper, reads its manifest, and fetches every
  declared layer into `prepared-references/content-addressed/sha256/` once. The
  rows are recorded under
  `execution_adapter.runtime_source_bundle.external_layers.<n>` and flow to the
  compile envelope like every other verified reference. Preparation engages only
  for wrappers that declare external layers; such a wrapper that fails
  validation blocks with `launch_preparation_runtime_source_bundle_invalid:<reason>`
  before any fetch. Wrappers without declared layers keep their existing
  contract: the compile step validates them.
- **Compile** resolves each layer by digest through the adapter member store
  (`compiled-episodes/content-addressed/adapter-members/sha256/`): copied once on
  the first miss, hardlinked into every later compiled episode, verified against
  the manifest digest on every read. A missing or tampered layer is a typed
  refusal and the partial output is removed.
- **v1 wrappers** (embedded payload) keep working unchanged.

## Shared member and runtime stores

- Compiled episodes hardlink every verified adapter member from the member store
  instead of extracting a private copy.
- Splat-render runtime trees hardlink immutable prerequisite files (Node,
  Chromium, `node_modules`) from the prerequisite root; a per-commit tree costs
  directories and renderer sources only.

## Storage classes

`blueprint_pipeline.control_plane_storage_roots` is the single table of
production roots and their retention law: `evidence_hot` (never evicted or
offloaded: spend guard, deploy receipts, standing authorizations),
`evidence_cold` (sealed run directories; offloadable behind a pointer), `cache`
(reproducible derived inputs; evictable when unpinned), `work` (queues),
`release` (per-commit trees), `ledger`, `container`, `staging`. A governance test
requires every root a production unit names to be classified, and the reclaim
tools refuse a configured root whose class is not the one they may touch.

## Volume layout

`deploy/host/mount_work_volume.sh` keeps bulk bytes off the root disk:

| Where | What |
|---|---|
| Root disk | The OS, the `/opt/blueprint` releases, and small durable state: `evidence_hot` (spend guard, deploy receipts, standing authorizations, the manifest), `ledger` (disk reservations, pins, locks) and the queues. |
| Scratch volume, `/mnt/blueprint-work`, growable | Every `cache`, `evidence_cold` and `scratch` root. Also the handoff spool `pubsub-handoffs` (every scene's raw capture and workspace), native run work (`native-g1-team-campaign-work`), the whole `task-evaluation-inputs` tree, and `/workspace`. |

Each moved root is bound back at its original path (`/mnt/blueprint-work/<rel>`
at `/var/lib/blueprint/<rel>`, `/mnt/blueprint-work/workspace` at `/workspace`)
and recorded in `/etc/fstab`, so unit sandboxes and recorded paths do not change.
`tests/test_mount_work_volume_script.py` ties the script's root list to
`control_plane_storage_roots.STORAGE_ROOTS`. The test fails when a bulk root is
left on the root disk, when a queue, ledger or hot evidence root would move, or
when a unit that writes under a moved root would keep running during the move.

**Why `task-evaluation-inputs` is one bind.** `link(2)` returns `EXDEV` across
mount points even on one filesystem. The episode compiler hardlinks
prepared-reference files into compiled episodes and falls back to a copy
(`task_evaluation_native_arena_episode_compiler.py`). The September layout bound
`prepared-references` and `compiled-episodes` separately, so every one of those
links silently became a full copy. The runtime builder links
`system-runtime-prerequisites` into `system-runtimes` the same way
(`scripts/build_task_evaluation_splat_render_runtime.py`). One bind for the
whole tree keeps all of these links, whichever stores they join.

The tree carries three small `evidence_hot` entries onto the volume:
`sam31-profile-registry`, `task-evaluation-terminal-results` and
`g1-team-campaign-registry.json`. This is deliberate. The volume is durable
block storage, and splitting the tree would break the hardlinks that keep it
small. `--plan` lists them under `evidence_hot on volume:`.

**Per-volume floors.** Admission measures the filesystem that holds each role's
target root, so after the move bulk roles reserve against the volume and state
writers against the root disk. Per-volume admission gives each volume its own
floor, plus a reserved band that keeps deploys and the listener's own state
writable when the volume is full. See
[Admission gate](#admission-gate-blueprint_pipelinecontrol_plane_disk_budget)
(design: [phase 2](CONTROL_PLANE_DISK_REDESIGN_2026-09-26.md#phase-2-volume-split-a-week)).
The capacity controller must measure both disks. Set
`BLUEPRINT_CAPACITY_MOUNTS=/:/var/lib/blueprint:/mnt/blueprint-work` in
`/etc/blueprint/pipeline-control-plane.env`, which overrides the unit's default.

### Consolidating the September binds

The production host still has one bind per store below `task-evaluation-inputs`
from the September migration, and the spool, capture reconstruction, the result
artifact cache and the other new roots still live on the root disk. One run moves
the new roots and consolidates the old binds:

1. Quiesce paid work first. The move stops the launch and canary dispatchers,
   the launch reconciler, the existing-canary continuation and its watchdog, and
   the spend guard. A paid GPU run left in flight would go unwatched until they
   start again. Before applying, `python3 scripts/operator_door.py status` must
   show no `paid_launch_locks` holders (every paid-launch lock free) and no
   deploy unit in flight. The move refuses while a transient
   `blueprint-operator-door-*` unit runs.
2. Check room. The volume serves every bound root, so the copy must not fill it.
   `df -h /mnt/blueprint-work` must leave headroom after the plan's
   `total to move`. Grow the volume first if it does not (see the 100 GB limit
   below). Apply refuses on its own when free space is below what the roots
   still need plus 5 %, and states both numbers.
3. Plan, and read it:

   ```bash
   sudo deploy/host/mount_work_volume.sh --device /dev/disk/by-id/<volume> --plan
   ```

   - `consolidate /var/lib/blueprint/task-evaluation-inputs … (bound children: …)`
     names the old per-store binds.
   - `move` lines name the new roots, and `bound` lines the six September roots
     that stay as they are.
   - `evidence_hot on volume:` lines name the hot entries the tree carries.
   - A `blocked` line is a refusal in advance: apply moves nothing until it is
     fixed.
   - `bound … not in /etc/fstab` marks a hand-made bind that would not survive a
     reboot. `/workspace` was bound by hand on 2026-09-20, so check it. Record
     such a bind in `/etc/fstab` by hand.
   - `missing` roots do not exist yet and are not created. Rerun the script
     once they appear.
4. Apply:

   ```bash
   sudo deploy/host/mount_work_volume.sh --device /dev/disk/by-id/<volume> --apply --ack move-work-roots-to-volume
   ```

   The run does the following, in order:
   - Stops every unit in the script's `WORKER_UNITS`. The units in
     `UNITS_LEFT_RUNNING` stay up, each for the reason written beside it; intake
     is one of them.
   - Copies the tree around the old binds, and compares the copy with its root
     in both directions.
   - Before each swap, checks again that no worker unit and no door request is
     running.
   - Prepares the new mount point and the rewritten `/etc/fstab`, after backing
     up the old one to `/etc/fstab.blueprint-<epoch>.bak`. A full root disk
     therefore refuses before anything changes.
   - Unmounts the old binds, deepest first, and binds them back if one will not
     go.
   - Swaps the root for its new mount point with two renames, binds the tree
     whole, and renames the new `/etc/fstab` into place.
   - Compares each original with its volume copy again, and only then removes
     it.
   - Reloads systemd and starts again the units that were running.
5. Check. `findmnt -R /var/lib/blueprint/task-evaluation-inputs` shows the
   tree's bind and no mount below it. `--plan` reports `bound` for every root
   that exists. After the next compile,
   `find /var/lib/blueprint/task-evaluation-inputs/compiled-episodes -type f -links +1 | head`
   lists hardlinked members.

When a run stops:

- `refusing: …` before `copying` means nothing moved. Fix the named cause and
  rerun. This covers a volume without room, a device blkid cannot name, a
  running door request, and worker units that would not stop.
- `refusing: worker units are running` or `refusing: operator door requests are
  running` after `copying` means the root about to swap was left as it was.
  Find what started the unit, let the request finish, and rerun.
- `refusing to swap: copy differs from source` means nothing was swapped. Rerun,
  and rsync resumes.
- `refusing to swap: the volume copy holds entries its root lacks` means nothing
  was swapped. The copy never deletes, and the bind would expose everything on
  the volume. Something there (an earlier run's copy of what was since removed,
  or a store copy that is no longer bound) would go live. Check the listed paths
  under `/mnt/blueprint-work`, move them aside, and rerun. This advice does not
  apply after an interrupted consolidation (next item). There the listed
  entries are the old binds' live bytes, only unmounted: restore the binds
  instead.
- After an abnormal stop, the plan may show
  `blocked  <root> (/etc/fstab mounts <path> inside the root)`, and apply refuses
  the same way. An abnormal stop is a run that was killed, lost its terminal, or
  saw the host go down in the middle of a consolidation. It unmounted some old
  binds before it could record the tree in `/etc/fstab`, which still lists them.
  Nothing was deleted: the old binds' bytes are on the volume, and the tree's own
  bytes are still in place.
  - Put the old layout back first. Reboot, or mount each path the line names
    from `/etc/fstab` (`sudo mount <path>`, or `sudo mount -a`), until
    `findmnt -R /var/lib/blueprint/task-evaluation-inputs` shows the old binds
    again. Then rerun.
  - Do not delete those `/etc/fstab` lines to get past the block.
  - If the plan instead says that an earlier move kept
    `<root>.migrated-to-volume` or left `<root>.new-mount-point`, and the root
    is not mounted, the stop came in the middle of the swap itself. Move the
    original back to `<root>` in place of the empty mount point, restore the
    old binds the same way, and rerun.
- `refusing: could not unmount …` means the old binds are back in place.
  `fuser -vm <path>` shows what holds the path. Rerun once it is free.
- `refusing to remove <root>.migrated-to-volume` means the root is already bound
  to the volume. The kept original holds the listed files, which the volume copy
  lacks or holds differently: bytes an old bind hid, or a write after the
  verification. Compare them with
  `sudo rsync -a -n --itemize-changes <root>.migrated-to-volume/ /mnt/blueprint-work/<rel>/`,
  and copy what belongs on the volume. Remove the kept copy only when nothing it
  holds is still needed. `--plan` shows a `kept` line until then.
- `refusing: <root> is no longer mounted from the volume` means the root's bind
  went away before its original was removed. The original is kept and the
  worker units stay stopped. Bind the volume copy at the root again (or move the
  kept original back), then start the units.
- `leaving the worker units stopped: <root> is between its old and new mounts`
  means a swap failed halfway. Finish it by hand: bind the volume copy at the
  root and move the rewritten fstab the message names into place. Or undo it:
  move the `.migrated-to-volume` original back, bind the old children again, and
  restore `/etc/fstab` from the backup. Then start the units the message names.

### Owner decisions

- A separate small durable state volume is optional. The root disk plus
  snapshots is enough for `evidence_hot`, `ledger` and the queues.
- DigitalOcean refuses volumes above 100 GB on this account. While that limit
  holds, attach a second volume and bind a subset of the roots from it (a pool of
  volumes per class) instead of growing one volume. The script binds every root
  under one `--mount` today, so this needs a root-subset option first.

## Pins

Producers pin the derived directories they create under
`/var/lib/blueprint/pipeline-control-plane/storage-pins/<kind>/<owner>.json`:
the preparation worker pins its preparation directory, the compile worker pins
its compiled episode (depending on the preparation), and the activation worker
pins its launch set (depending on both). The policy-canary dispatcher releases
the activation pin when it writes the terminal `dispatch_receipt.json`, and the
release cascades to dependencies no other live pin still needs. Pins expire
after 30 days so a release that never arrives cannot protect bytes forever.

Automatic scene-configuration launches also release their activation pin after
writing a sealed terminal `launch_receipt.json`. Release requires matching
staging and launch identities, byte-verified retained copies under the run's
`immutable_inputs`, no other pending or processing launch using the activation,
and completed teardown of every recorded provider instance. A launch refused
before paid admission may release only when its receipt explicitly records no
provider mutation and no provider evidence contradicts it. Missing or uncertain
evidence retains the pin. Reading an existing terminal receipt retries this
metadata-only release; an already released pin requires no repeated hashing.
The dispatcher never deletes files or changes the launch outcome during release.
The normal reclaim timer still enforces queue references and its idle window.
The launch reconciler retries release after creating or validating a retained
post-teardown provider-zero receipt, so launches that never allocated a GPU can
release their pins once that later closure evidence exists. Cleanup failures
are reported separately and never invalidate proven resource closure.

## Reclaim timer

`blueprint-control-plane-storage-gc.timer` runs
`python -m blueprint_pipeline.control_plane_storage_gc run --apply --ack reclaim-control-plane-storage`
every six hours as the `blueprint` service account and writes
`/var/lib/blueprint/pipeline-control-plane/storage-gc/latest.json`. One tick:

1. **Derived directories** under the configured `cache` roots are retired when
   no live pin names them, no pending or processing queue message mentions
   them, and they have been idle for seven days.
2. **Content-store blobs** whose link count is one (nothing hardlinks them any
   more), whose bytes still match their digest, and which are older than a day
   are removed. Retiring directories first is what frees blobs.
3. **Evidence offload** lists sealed run directories (terminal receipt present,
   idle past the 14-day hot window) under the `evidence_cold` roots. It applies
   only when `BLUEPRINT_CONTROL_PLANE_EVIDENCE_OFFLOAD=1` is set in
   `/etc/blueprint/pipeline-control-plane.env`: the directory is packed, published
   to the artifact store under kind `control-plane-evidence` with full readback,
   replaced by `<name>.offloaded.v1.json` (URI, digest, size, per-member digests),
   and only then removed. Bytes are migrated, never deleted; the spend guard and
   every other `evidence_hot` root are outside the tool's reach.

Restore an offloaded run with
`python -c 'from blueprint_pipeline.control_plane_evidence_offload import restore_offloaded_evidence as r; r(pointer_path=..., destination=...)'`;
every member digest is verified before the directory is exposed.

The manual single-root form
`python -m blueprint_pipeline.control_plane_storage_gc --content-store-root <root>/sha256 [--apply --ack reap-unreferenced-content]`
remains for operators.

## Release retirement at deploy

Deploy is the only event that creates per-commit release worktrees and runtime
trees, so deploy retires them. After the new release is proven live, a commit's
trees (release, `splat-render`, `scene-configuration` and their publication
receipts) stay only while the commit is:

- the active release or the commit being deployed;
- among the newest three releases;
- in use by a live process (its cwd, executable or an argv path lies in the
  commit's release or runtime tree), checked when planning and again just before
  each commit's paths are removed;
- younger than a day; or
- held by a typed protection row: a lease (`live_queue`, `standing_authorization`,
  `retention_binding`) or current configuration (`configured_runtime`).

Protection no longer comes from searching every JSON file for 40-hex tokens.
That search protected git tree ids, commits embedded in profile ids and the
consumption records of expired authorizations; on 2026-09-26 it protected 513
commits and retired none of 95 trees. Every lease now has an owner, a reason,
the run it serves and an expiry, and lapses when that run ends. Leases for
immutable retention bindings live in sidecars under
`/var/lib/blueprint/pipeline-control-plane/release-leases/bindings` (a root-only
`ledger` root). The runbook
[`runbooks/task-evaluation-release-retention.md`](runbooks/task-evaluation-release-retention.md)
covers the lease rules, migration, and how an owner renews or ends a lease.

Every release-reference publisher (queue writers, the profile publisher, the
standing-authorization materializer, release activation and the SAM prefix
binding writer) locks the control-plane root shared. Retirement holds it
exclusively, waiting at most 300 seconds for it, only while it collects
protection, plans and renames each candidate into `<its root>/.retiring` (on a
filesystem too full for that, it deletes the candidate directly). It deletes
and measures the moved trees only after the deploy has released the lock, its
paid-launch gate and its disk reservation, prunes the source clone's worktree
registrations so a retired commit can be redeployed, and sweeps `.retiring`
leftovers of an interrupted run before the deploy reserves disk.
`retired_bytes` counts only bytes actually freed (each inode once, and only
when its last link was deleted); hardlinked bytes are reported as
`shared_bytes`. Any protection
blocker (an unreadable or unsettled queue, a missing protection source, an
unreadable configuration file, a live reference to a missing profile, a
malformed standing authorization, an invalid or changed binding) retires
nothing.

The deploy receipt records `release_retirement`: `applied`, `skipped` with
blockers, or `blocked`, together with `protected_by_kind` (tree counts per
kind), `protected_tree_count`, `lease_protected_tree_count`, `lapsed_count`,
`migrated_binding_count` and `alerts`. The same summary is written to
`release-retention/latest-deploy-retirement.json` (0644). More than 20 trees
held only by leases raises `release_retirement_lease_protected_trees:<n>`; a
blocked or skipped retirement raises `release_retirement_blocked:<blocker>`. A
retirement problem never fails a deploy whose surfaces already moved.


## Streaming offload and whole-chain admission (2026-09-08)

The artifact store is the durable copy of sealed evidence. Local run directories are a working cache once their terminal seal, age, pins, queue references, and active readers permit eviction. The September 8 host had a 5.53 GB archive eligible for offload but insufficient room to build that archive locally; the existing reaper therefore could not free the space it was meant to recover.

Default evidence offload now makes a deterministic hashing pass over a tar stream, uploads a second identical stream in bounded multipart chunks, and verifies the complete remote object before writing the small pointer and removing local files. It reserves space for pointer metadata, not a second local copy of the evidence. Stream changes and interrupted uploads abort incomplete multipart uploads; remote readback failures or changed local evidence retain the source. Existing explicit file-publisher integrations keep their compatibility path. The receipt identifies the transfer mode and local archive bytes.

Restore supplies the complete artifact identity required by the actual downloader and checks every restored member. The regression uses the real download implementation, so a fixture cannot hide a missing reference field. The GC service can preserve pointer ownership and inspect active process references; ptrace and process-vm syscalls remain denied.

New scene-preparation installations require a whole-chain capacity check before creating an attempt. A workspace that fits only the next stage waits for capacity before work starts. The existing stage reservations remain authoritative and account for competing work; this initial check is an admission forecast, not an additional reservation or a guarantee against untracked external disk writers. Already-started attempts can continue. Cloud-backed reclamation can recover space without needing archive-sized local scratch, allowing the automatic scene timer to retry admission.
