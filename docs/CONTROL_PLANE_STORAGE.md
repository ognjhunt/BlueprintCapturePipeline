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
`free - floor - live reservations >= need`, where the bulk floor is
`max(8 GiB, 5 % of the disk)` (`BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES`) and a
live reservation is a ledger entry on the same device, inside its TTL, whose
pid is alive.
`control_plane_deploy` may use the protected band below that bulk floor down to
`max(1 GiB, 1 % of the disk)`
(`BLUEPRINT_CONTROL_PLANE_DISK_CRITICAL_FLOOR_BYTES`). Other bulk writes stop at
the bulk floor, leaving room for deploys, the listener's small state writes,
teardown and provider-zero evidence.
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
| `handoff_staging` | 4 GiB | sizes of the capture blobs being downloaded plus 64 MiB | listener before downloading; refusal remains retryable and unacknowledged |
| `launch_dispatch` | 2 GiB | unique immutable input file sizes plus 64 MiB | dispatcher before copying and before any allocator call |

The intake version endpoint reports `disk_headroom` with `refused_roles` and each
role's `footprints` and `targets` (device, floor, reservations and available
bytes). `BLUEPRINT_CONTROL_PLANE_DISK_ROLE_TARGETS` maps each bulk role to its
absolute write root; unspecified roles use the default target. A malformed map
fails closed for the whole chain. The launch-preparation, launch-activation, and
task-evaluation-launch intakes refuse a submission (HTTP 503, typed blocker)
while its role is refused.
The staging reservation is renewed while a blob download is in progress, so a
long download does not release its bytes merely because the original lease
period elapsed.

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
a `disk_reservations_unreadable` blocker. `whole_chain_admission` groups chain
roles by target device and admits only when each device has room for its roles
together. It reports each device's required and available bytes and whether it
passed. It also sums the chain roles' footprints into `required_workspace_bytes` and reports
`required_workspace_basis`: `measured_p95` when every chain role is measured,
`declared_default` when none is, `mixed` otherwise, with the per-role
`footprints` beside it. The capacity controller reports these footprints, like its
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
`release` (per-commit trees), `ledger`, `container`, `staging`, `scratch`, and
`scene_workspace` (one website scene's working copy under
`pubsub-handoffs/<bucket>/scenes/<scene>`, retired as described below). A root may
contain `*` segments that match exactly one path component; the most specific
match wins (more segments, then more literal characters), so
`pubsub-handoffs/*/scenes/*.retired.v1.json` receipts are `evidence_hot` while the
workspaces beside them are `scene_workspace`. A governance test requires every root
a production unit names to be classified, and the reclaim tools refuse a
configured root whose class is not the one they may touch.

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

The inputs tree carries three small `evidence_hot` entries onto the volume:
`sam31-profile-registry`, `task-evaluation-terminal-results` and
`g1-team-campaign-registry.json`. The handoff spool also carries
`pubsub-handoffs/*/scenes/*.retired.v1.json` retirement receipts. This is deliberate. The volume is durable
block storage, and splitting the tree would break the hardlinks that keep it
small. `--plan` lists them under `evidence_hot on volume:`.

**Per-volume floors.** Admission measures the filesystem that holds each role's
target root, so after the move bulk roles reserve against the volume. Per-volume
admission gives each volume its own floor. "Full" here means below the bulk
floor, with physical free space still reserved: the listener's job ledger lives
inside its scratch-volume capture tree and can use that remaining space after a
new capture download is refused. Deploys use the separate critical floor on
their target device. At zero physical free bytes, no local state write can be
guaranteed. See
[Admission gate](#admission-gate-blueprint_pipelinecontrol_plane_disk_budget)
(design: [phase 2](CONTROL_PLANE_DISK_REDESIGN_2026-09-26.md#phase-2-volume-split-a-week)).
The capacity unit measures `/`, `/var/lib/blueprint` and `/mnt/blueprint-work`.
A configured mount that is not present is reported as absent; an existing
unreadable mount remains critical. The intake, scene progression and capacity
units share the same role-target map.

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

## Usage attribution

The capacity controller (`blueprint-control-plane-capacity.service`, every ten
minutes as `root`) also surveys disk usage with
`control_plane_disk_usage.survey_usage`, so one door call answers "what uses the
space?". It surveys at most hourly (`BLUEPRINT_CAPACITY_SURVEY_INTERVAL_SECONDS`,
default 3600, judged by the last attempted survey even when it failed or the
process was killed); `--survey` forces one.

**What it counts.** The survey walks the controller's mounts
(`BLUEPRINT_CAPACITY_MOUNTS`) plus `/` and `/mnt/blueprint-work` when that
physical work volume is mounted. Before it is mounted, the survey skips it.
Each root is walked within its own filesystem like
`du -x`: a directory on another device or listed as a mount point in
`/proc/self/mountinfo` is skipped, a listed mount nested inside another is walked
once, and symlinks are never followed. Every inode counts once, in allocated
bytes. On 2026-09-26 a per-name listing counted about 470 GB on the 165 GB disk,
because the content stores and the trees built from them share bytes through
hardlinks. Bytes on the work volume are attributed at the paths the pipeline uses
(`/mnt/blueprint-work/workspace` → `/workspace`, any other
`/mnt/blueprint-work/<rel>` → `/var/lib/blueprint/<rel>`). The walk stops after
3,000,000 entries or 240 s with `status: "truncated"`. A 20,000-entry memory
bound on buffered directory entries, pending directories, owner rows and shared
inodes also truncates the survey before the capacity unit's 512 MiB limit is
at risk. Unreadable entries are counted. Paths the unit's sandbox hides
(`ProtectHome=`, `PrivateTmp=`), its own `ReadWritePaths=`/`ReadOnlyPaths=` bind
mounts skipped by the mountinfo rule, and deleted files still held open by a
process cannot be attributed. These gaps lower `attributed_fraction`.

**Class and root.** A path takes the storage class and root of its
`control_plane_storage_roots` row; a row with `*` segments reports the concrete
directory it matched. A path under a `container`, or under `/var/lib/blueprint`,
`/opt/blueprint` or `/workspace`, that no row claims is `unclassified`, rooted at
the child it lies in. Everything else is `host`, rooted at its first two
components (`/var/log`, `/usr/lib`).

**Owner.** The first matching rule wins:

| Path | Owner |
|---|---|
| `…/pubsub-handoffs/<bucket>/scenes/<scene>/…` | `scene:<scene>` |
| `…/system-runtimes/<component>/<sha>/…`, `…/task-evaluation-control-plane-releases/<sha>/…` | `release:<sha[:12]>` |
| `…/content-addressed/…` | `store:<root basename>` |
| `…/task-evaluation-launch-runs/<id>/…`, `…/task-evaluation-policy-canaries/<id>/…` | `run:<id>` |
| `…/task-evaluation-scene-intents/<id>/…` | `scene-intent:<id>` |
| any other classified path | `<root basename>/<first child>`, or `<root basename>` for a file directly under the root |
| `unclassified` or `host` | the root |

`<id>`, `<scene>` and `<sha>` are directories, and a `<sha>` starts with a 40-hex
commit; a pointer or marker file beside them keeps the generic owner. An inode with several
names belongs to the smallest of its names under a `content-addressed` directory,
else to its smallest name, whatever order the walk met them in.

**Files** in `/var/lib/blueprint/pipeline-control-plane/capacity` (`0755`, so the
door, which runs as `blueprint`, can reach the public files):

| File | Mode | Holds |
|---|---|---|
| `latest.json`, `history.jsonl` | `0600` | the full report, with project spend and provider funding, and its history |
| `usage-latest.json` | `0644` | the last survey (`control_plane_disk_usage_survey.v1`), with every unclassified root |
| `usage-attempt.json` | `0644` | the last survey attempt, including a failed or interrupted attempt's retry clock |
| `summary.json` | `0644` | `control_plane_capacity_summary.v1`, written every tick: level, alerts, mounts, the usage projection and the resize status. It is projected by named keys, so it carries no spend, funding or URLs, and it stays under 128 KiB. |

Credential-shaped filesystem names are redacted from the public survey and
summary before publication. Their byte totals and storage classes remain in the
report. Existing surveys are sanitized when the controller reads them.

`latest.json` and `summary.json` carry the same `usage` projection: the survey's
age and status, its filesystem rows, bytes per class, the top ten roots and
owners, and the 20 largest unclassified roots. The controller warns with
`usage_unclassified_root` for each unclassified root over 1 GiB, and with
`usage_attribution_low` when a filesystem's `attributed_fraction` (surveyed bytes
over used bytes, capped at 1) is under 0.9. Either warning raises an `ok` report
to `warning`. A survey exception keeps the last result and names the error
(`usage_survey_failed:<type>`) without stopping the capacity tick. Failed and
interrupted attempts do not retry on every ten-minute tick. A new non-usage
warning still pages when a usage warning has already raised the report to
`warning`. A warning on another mount pages even when its code is already
present, and a failed webhook post is retried on the next tick.

**Reading it.** `python3 scripts/operator_door.py usage` prints `capacity.usage`
from door `status` as tables ([`OPERATOR_DOOR.md`](OPERATOR_DOOR.md)):

- The mount table's `attributed` column is the fraction of the filesystem's used
  bytes that the survey found. A low value means bytes it could not see: a
  truncated walk, unreadable or sandbox-hidden paths, or deleted files still held
  open by a process.
- The owner table says what to retire. `scene:` workspaces, `run:` evidence
  (offloadable by the reclaim timer), `release:` trees (retired by deploy) and
  `store:` blobs (reaped once nothing hardlinks them) each have their own
  retention rule.
- An unclassified root is a tree the storage table does not know. Classify it in
  `control_plane_storage_roots` before any tool may reclaim it.

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
hourly (`OnUnitInactiveSec=1h`: an hour after the previous tick finished) as
`root`, confined by its unit to the roots it may write, and writes
`/var/lib/blueprint/pipeline-control-plane/storage-gc/latest.json`. One tick runs
eight phases in order:

1. **Stranded queue rows**: pending rows bound to a release other than the
   running one move to `stranded/` beside a receipt, so they stop counting as
   live queue references. Nothing is deleted.
2. **Terminal cache pins** whose run is proven closed by archived-run evidence
   are released. Only the pin ledger changes.
3. **Derived directories** under the configured `cache` roots are retired when
   no live pin names them, no pending or processing queue message mentions
   them, and they have been idle for an hour
   (`BLUEPRINT_CONTROL_PLANE_GC_DERIVED_MINIMUM_AGE_SECONDS=3600`).
4. **Content-store blobs** whose link count is one (nothing hardlinks them any
   more), whose bytes still match their digest, and which are older than a day
   are removed. Retiring directories first is what frees blobs.
5. **Evidence offload** lists run directories under the `evidence_cold` roots
   that are sealed (terminal receipt present) and idle past the two-day hot
   window (`BLUEPRINT_CONTROL_PLANE_EVIDENCE_HOT_WINDOW_SECONDS=172800`), or that
   have no receipt and have not changed for three days (abandoned by a superseded
   or torn-down worker). It applies only when
   `BLUEPRINT_CONTROL_PLANE_EVIDENCE_OFFLOAD=1` is set in
   `/etc/blueprint/pipeline-control-plane.env`: the directory is packed, published
   to the artifact store under kind `control-plane-evidence` with full readback,
   replaced by `<name>.offloaded.v1.json` (URI, digest, size, per-member digests),
   and only then removed. Bytes are migrated, never deleted; the spend guard and
   every other `evidence_hot` root are outside the tool's reach.
6. **Scratch directories** idle for three days
   (`BLUEPRINT_CONTROL_PLANE_GC_SCRATCH_MINIMUM_AGE_SECONDS=259200`) are reaped by
   age alone: nothing references them.
7. **Workspace bundles**: the reproducible `bundle/` copy inside a
   semantic-pretraining workspace that has been idle and unpinned for six hours
   is removed behind a sealed marker.
8. **Scene workspaces** are retired only after terminal, acknowledgement,
   reference, and remote-copy checks pass. This phase plans until
   `BLUEPRINT_CONTROL_PLANE_SCENE_WORKSPACE_RETIREMENT=1` enables it; its
   detailed contract is below.

Restore an offloaded run with
`python -c 'from blueprint_pipeline.control_plane_evidence_offload import restore_offloaded_evidence as r; r(pointer_path=..., destination=...)'`;
every member digest is verified before the directory is exposed.

The manual single-root form
`python -m blueprint_pipeline.control_plane_storage_gc --content-store-root <root>/sha256 [--apply --ack reap-unreferenced-content]`
remains for operators.

## Scene workspace retirement

The Pub/Sub listener stages each website capture under
`/var/lib/blueprint/pubsub-handoffs/<bucket>/scenes/<scene_id>/captures/<capture_id>`:
raw capture downloaded from Firebase Storage plus the pipeline's outputs. Nothing
reclaimed these working copies, and the website `pipeline/**` outputs exist nowhere
else (their `gs://` names are local aliases; nothing uploads them), so they cannot
simply be deleted. `blueprint_pipeline.website_scene_workspace_retention` retires a
scene only when everything in it can come back and nothing can still need it.

**Plan** (read-only; cheap checks first; any failure keeps the scene and says why):

1. the scene path and its parents are real directories, nothing inside is a link or
   special file (`workspace_path_unsafe`, `unsafe_entry:<path>`);
2. every capture is a website capture, by the listener's own test
   (`is_website_capture_manifest` on its `raw/manifest.json`); a device or mixed scene,
   or a capture whose manifest cannot be read, is kept (`not_a_website_scene`). Every
   capture is terminal with its own proof and holds no live lease: a `completed`
   ledger with a committed output, or `terminal_authority_ended` with a terminal
   receipt whose payload digest is the ledger's
   (`capture_not_terminal:<c>`, `capture_lease_held:<c>`, `capture_ledger_unreadable:<c>`,
   `capture_lock_missing:<c>`, `scene_has_no_captures`);
3. its last terminal message was acknowledged: an ack receipt with the matching
   disposition written after the terminal state, or a ledger idle past Pub/Sub's
   7-day message retention, which can then no longer redeliver it
   (`acknowledgement_unproven:<c>`);
4. nothing in the tree changed for 48 hours (`recently_active`);
5. no live storage pin names it, lies inside it or contains it (`pinned`);
6. no pending or processing queue message names the scene, across the reclaim
   timer's queues plus `sam31-preparation-executions`,
   `task-evaluation-scene-configuration-activation-intents` and
   `capture-reconstruction-queue` (`queue_referenced`);
7. no live process holds it (`in_use`; an unreadable process table counts as in use);
8. no scene intent that can still run resolves a website source registered inside
   it (`open_scene_intent:<intent>`). An intent is finished once its progression
   completed, it was revoked, or seven days have passed since its (possibly extended)
   execution window elapsed: an owner may still extend an expired window, and
   progression would then resolve the source again. A revoked or expired intent is
   also held by any attempt row progression still treats as live (`attempts/*.json`
   with no validated cancellation or settlement:
   `open_scene_attempt:<intent>/<attempt>`), but only for seven days after it finished
   (the revocation time recorded in `revoked.json`, else that file's time; for an
   expired intent that period ends with its grace period). Every hold expires:
   materialization copies the workspace inputs into the attempt's own staging, and no
   factory pass runs for days. A completed intent's rows never hold it, since only
   retired predecessors are ever settled and progression completes only after the
   attempt's terminal result. A registration no intent has claimed protects the
   scene for 72 hours
   (`unclaimed_source_registration`); a registration or intent that cannot be read
   protects every scene (`reference_index_unreadable`); a workspace whose website
   handoff names a registration outside the indexed binding root is kept too
   (`source_registration_unindexed:<c>`);
9. every file is recoverable: it matches its Firebase Storage object
   `gs://<bucket>/scenes/<scene>/<path>` by size and MD5 (CRC32C when the object has
   no MD5), or it is marked for the archive. Raw capture bytes are never archived:
   a raw file that does not verify keeps the scene
   (`raw_not_verified_in_cloud:<path>`). On the timer each file's digests are cached
   by (path, size, mtime, ctime, device, inode) in
   `pubsub-handoffs/.scene-workspace-inventory/<bucket>/<scene>.json` (root, `0600`), so an
   hourly plan re-reads only what changed. A tick normally hashes at most 20 GiB
   or five minutes of uncached bytes. One file larger than 20 GiB may use an
   exceptional window sized for 8 MiB/s, capped at two hours below the GC unit's
   three-hour timeout. An unfinished ordinary file waits for the next tick
   (`inventory_deferred`); an exceptional file that exceeds its window stays local
   with `oversized_hash_timeout`. Retirement never trusts the cache: it re-reads every file
   it is about to delete.

**Retire** re-proves checks 1-8 and the planned file snapshot, and re-reads
every file the cloud copy replaces. It sizes the receipt before publishing;
one above its readers' 16 MiB limit is refused (`receipt_too_large`). It streams
unverified files as `workspace.tar` to the private artifact store with full
readback. This upload holds no listener ledger lock. Before mutation it takes
the per-capture locks without waiting (`candidate_busy`) and the storage-pin
lock, then rechecks readers, file identity and Firebase Storage metadata. It
writes a digest-bound receipt including the cloud generations, archive hashes,
capture records and a rename token, then renames the workspace to
`.retiring-<scene>-<token>`. After releasing the locks, it verifies the renamed
tree's capture IDs and every file against the receipt, and revalidates the
receipt digest, before removal. An incomplete removal reports zero reclaimed
bytes and its hidden tree remains available for the next enabled sweep. A new
capture appearing at the rename boundary is returned to the live path and its
receipt is preserved under a recovery name. An enabled applying tick finishes
a crash-left copy only if its receipt token and bytes match. A receipt beside a
live workspace finishes the rename if the bytes match; a changed workspace moves
the old receipt aside. Failed post-upload attempts record the published archive
reference in the GC report for orphan review.

**Restore** replays the receipt: `python -m blueprint_pipeline.website_scene_workspace_retention
restore --receipt <receipt> --destination <dir>` downloads each verified object's
recorded GCS generation and
re-checks it, materializes the archive, re-checks its digest and every member's
SHA-256, requires exactly the retired file set, and only then moves the tree into
place (owned like the destination's parent). An in-place restore moves the
historical receipt aside. The door exposes this as `restore-scene-workspace
<scene_id> --bucket <bucket> --wait`.

**Redeliveries.** Before claiming a capture whose workspace is absent, the
listener reads the receipt: a message the receipt proves terminal (the acknowledged
payload, or the payload that ended the capture's authority) is acknowledged as
`skipped_retired_terminal` without staging. A different payload is a new request
and stages again from Firebase Storage. A redelivery that races a retirement never
recreates the workspace: after taking the ledger lock the listener checks that the
lock file it holds is still the capture's and that the capture still exists, and a
claim for a capture that existed when the message arrived never creates it again.
Either way it asks the receipt again, and a payload the receipt does not cover is
left for its next delivery (`capture_retired_retryable`). A retired completed capture
covers every payload, as its ledger would. A present invalid or unreadable receipt
raises an alert signal and leaves the message retryable
(`retirement_lookup_failed_retryable`) rather than restaging.

**Where it runs.** The reclaim timer plans every scene workspace in
`BLUEPRINT_CONTROL_PLANE_GC_SCENE_WORKSPACE_ROOTS` (intents from
`BLUEPRINT_CONTROL_PLANE_GC_SCENE_INTENT_ROOT`; registrations from
`BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT`, else `<intent root parent>/website-source-bindings`)
and reports them under `scene_workspaces` (candidates, retired count and bytes,
`retained_counts` by reason). It attempts at most 20 retirements per tick (attempts,
not successes, since each can publish a large archive), and only with its own explicit
opt-in, `BLUEPRINT_CONTROL_PLANE_SCENE_WORKSPACE_RETIREMENT=1` in the operator
environment file: unset, each tick only plans. It never follows
`BLUEPRINT_CONTROL_PLANE_EVIDENCE_OFFLOAD`. Any other value disables it and puts
`scene_workspace_retirement_setting_invalid` in the report's `alerts` without aborting
the tick. Neither the unit nor the example environment enables it. The operator door offers the same operation by
hand (`retire-scene-workspace`, see `docs/OPERATOR_DOOR.md`), plan first and
`--apply` second. Every phase of the tick is isolated: a phase that raises is recorded
as `{"status": "error", "error": "<type>"}` under its key, later phases still run, and
the tick exits non-zero after writing the report.

**Privacy.** Raw capture bytes stay only in Firebase Storage. Derived local-only
files from ordinary completed captures go to the existing private artifact store
(B2). Authority-ended captures stay local (`authority_ended_capture_kept_local`)
until the owner approves a revocation and deletion lifecycle for those derivatives.

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
