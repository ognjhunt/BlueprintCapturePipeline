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
| `launch_activation` | 2 GiB | the measured footprint | activation worker |
| `policy_canary_dispatch` | 2 GiB | the measured footprint | canary dispatcher queue boundary |

`launch_dispatch` has no reservation call site; whole-chain admission counts it
at its declared ceiling until it has measured history.

The intake version endpoint reports `disk_headroom` with `refused_roles` and each
role's `footprints`; the launch-preparation, launch-activation, and
task-evaluation-launch intakes refuse a submission (HTTP 503, typed blocker)
while its role is refused.

Declared ceilings can be tuned with
`BLUEPRINT_CONTROL_PLANE_DISK_FOOTPRINT_<ROLE>_BYTES`.

### Measured footprints

A declared footprint is a ceiling, not what admission keeps reserving. Each
reservation names the per-job directory it writes (its `workspace`: the
preparation, compilation, activation or canary run directory under the shared
parent it reserves against) and, when released, records how much that directory
grew. Growth counts each inode once, in allocated blocks
(`control_plane_disk_usage.tree_usage`), so hardlinked names are not double
counted. Names hardlinked from a content store still count as this job's bytes;
that over-counts cache hits, which errs conservative, and the clamp below bounds
it. The cpu prestage and semantic pretraining jobs sample just before they remove
their scratch trees, and the deploy observes the release checkout and runtime
trees it created.

Samples are appended to `<ledger>/history/<role>.jsonl` (`root:blueprint`
`2770`, installed and verified by the deploy), one line per release with the
workload, outcome, reserved bytes and duration, and compacted to the newest 200
lines under the ledger lock. Every sample names how its job ended, and only
`completed` samples shape admission. A job that raised is recorded as `failed`,
whether it left a reservation's context on the exception or a worker caught the
exception and released with that outcome (preparation, compilation,
activation). A job that returned a blocked result before finishing its work is
recorded as `blocked`: a preparation paused on its children or on capacity, a
canary run that returned a `blocked…` status, a scene factory that stopped short
of publication, or a replay that was refused.

Once a role has at least 10 completed samples among its newest 50, its footprint
is the nearest-rank p95 of those samples × 1.25, clamped to
`[64 MiB, declared ceiling]`. Until then, or whenever the history cannot be
read, it is the declared ceiling, so a new role stays conservative; a sample
above the ceiling can never raise a reservation. Reservations made without
explicit bytes hold this footprint, and every reservation receipt names its
`footprint_basis` (`measured_p95`, `declared_default` or `caller_exact`), its
`footprint_sample_count` and its `workload`.

Intake headroom, the capacity controller (`measure_mount`),
`whole_chain_admission` and the chain preflight use the ledger's own floor,
live-reservation rule and footprints, so a projected refusal is the refusal the
workers will make. `whole_chain_admission` sums the chain roles' footprints into
`required_workspace_bytes` and reports `required_workspace_basis`:
`measured_p95` when every chain role is measured, `declared_default` when none
is, `mixed` otherwise, with the per-role `footprints` beside it.

Pid liveness is the primary liveness signal and the TTL only a backstop for a
recycled pid, so long roles get TTLs that outlive them: `cpu_prestage` and
`semantic_pretraining` 12 h, `stage_replay` 6 h, `control_plane_deploy` 4 h,
every other role 2 h.

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
seven phases in order:

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

Restore an offloaded run with
`python -c 'from blueprint_pipeline.control_plane_evidence_offload import restore_offloaded_evidence as r; r(pointer_path=..., destination=...)'`;
every member digest is verified before the directory is exposed.

The manual single-root form
`python -m blueprint_pipeline.control_plane_storage_gc --content-store-root <root>/sha256 [--apply --ack reap-unreferenced-content]`
remains for operators.

## Release retirement at deploy

Deploy is the only event that creates per-commit release worktrees and runtime
trees, so deploy retires them: after the new release is proven live, commits
that are not the active release, not the commit being deployed, not named by any
launch profile, standing authorization, or pending/processing queue envelope,
not among the newest three releases, and older than a day are removed together
with their runtime publication receipts. The deploy receipt records
`release_retirement` (`applied`, `skipped` with blockers, or `blocked`); a
retirement problem never fails a deploy whose surfaces already moved.


## Streaming offload and whole-chain admission (2026-09-08)

The artifact store is the durable copy of sealed evidence. Local run directories are a working cache once their terminal seal, age, pins, queue references, and active readers permit eviction. The September 8 host had a 5.53 GB archive eligible for offload but insufficient room to build that archive locally; the existing reaper therefore could not free the space it was meant to recover.

Default evidence offload now makes a deterministic hashing pass over a tar stream, uploads a second identical stream in bounded multipart chunks, and verifies the complete remote object before writing the small pointer and removing local files. It reserves space for pointer metadata, not a second local copy of the evidence. Stream changes and interrupted uploads abort incomplete multipart uploads; remote readback failures or changed local evidence retain the source. Existing explicit file-publisher integrations keep their compatibility path. The receipt identifies the transfer mode and local archive bytes.

Restore supplies the complete artifact identity required by the actual downloader and checks every restored member. The regression uses the real download implementation, so a fixture cannot hide a missing reference field. The GC service can preserve pointer ownership and inspect active process references; ptrace and process-vm syscalls remain denied.

New scene-preparation installations require a whole-chain capacity check before creating an attempt. A workspace that fits only the next stage waits for capacity before work starts. The existing stage reservations remain authoritative and account for competing work; this initial check is an admission forecast, not an additional reservation or a guarantee against untracked external disk writers. Already-started attempts can continue. Cloud-backed reclamation can recover space without needing archive-sized local scratch, allowing the automatic scene timer to retry admission.
