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
`max(8 GiB, 5 % of the disk)` (`BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES`).
A refusal is the typed blocker
`control_plane_disk_budget_exceeded:<role>:need_bytes=..:available_bytes=..:free_bytes=..:floor_bytes=..:reserved_bytes=..`
and never a host path.

| Role | Reserves | Where it refuses |
|---|---|---|
| `control_plane_deploy` | 2 GiB | `deploy_control_plane_commit.py` before provenance or staging |
| `launch_preparation` | exact bytes of references the content store lacks, plus 512 MiB; the runtime-source layer separately on a miss | preparation worker, before any fetch |
| `episode_compilation` | exact bytes of runtime members the member store lacks, plus 2 GiB | compile worker, before the output directory exists |
| `launch_activation` | 2 GiB | activation worker |
| `policy_canary_dispatch` | 2 GiB | canary dispatcher queue boundary |

The intake version endpoint reports `disk_headroom` with `refused_roles`; the
launch-preparation, launch-activation, and task-evaluation-launch intakes refuse
a submission (HTTP 503, typed blocker) while its role is refused.

Per-role defaults can be tuned with
`BLUEPRINT_CONTROL_PLANE_DISK_FOOTPRINT_<ROLE>_BYTES`.

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

## Pins

Producers pin the derived directories they create under
`/var/lib/blueprint/pipeline-control-plane/storage-pins/<kind>/<owner>.json`:
the preparation worker pins its preparation directory, the compile worker pins
its compiled episode (depending on the preparation), and the activation worker
pins its launch set (depending on both). The policy-canary dispatcher releases
the activation pin when it writes the terminal `dispatch_receipt.json`, and the
release cascades to dependencies no other live pin still needs. Pins expire
after 30 days so a release that never arrives cannot protect bytes forever.

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
trees, so deploy retires them: after the new release is proven live, commits
that are not the active release, not the commit being deployed, not named by any
launch profile, standing authorization, or pending/processing queue envelope,
not among the newest three releases, and older than a day are removed together
with their runtime publication receipts. The deploy receipt records
`release_retirement` (`applied`, `skipped` with blockers, or `blocked`); a
retirement problem never fails a deploy whose surfaces already moved.
