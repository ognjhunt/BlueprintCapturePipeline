# PR 7 — Volume layout: every bulk root on the scratch volume, hardlinks intact (design doc phase 2, layout)

> Read `00-index.md` first.

**Goal:**
- The root disk holds only the OS, releases and small durable state (`evidence_hot`, `ledger`, queues).
- Every bulk root moves to the growable scratch volume: `cache`, `evidence_cold`, `scratch`,
  `scene_workspace`, the handoff spool, `task-evaluation-inputs` and native run work.
- `task-evaluation-inputs` is bound as **one tree**, so hardlinks between its stores keep working.

**Branch / base / worktree:** `claude/disk-2-volume-layout` from `origin/main`,
`/Users/nijelhunt_1/workspace/BlueprintCapturePipeline-disk-2b-20260926`.

## Facts (verified)

- `deploy/host/mount_work_volume.sh` moves 11 roots, each as its **own** bind mount, and records
  them in `/etc/fstab`. It plans by default and needs `--apply --ack move-work-roots-to-volume`.
- `--root-prefix` runs it hermetically (no mounts). `tests/test_mount_work_volume_script.py` runs it
  that way.
- `link(2)` returns `EXDEV` across mount points even on one filesystem, and the episode compiler
  hardlinks prepared-reference files into compiled episodes with a copy fallback
  (`task_evaluation_native_arena_episode_compiler.py:198-200`, `830-833`). With `prepared-references`
  and `compiled-episodes` as separate bind mounts, every such link silently becomes a full copy.
  The same applies to `system-runtime-prerequisites` → `system-runtimes`.
- The spool `/var/lib/blueprint/pubsub-handoffs` holds every scene's raw capture and was never on
  the volume. `/workspace` (CPU prestage work) was bind-mounted by hand onto the volume on
  2026-09-20.
- The production host already has the 11 per-root binds from the September migration, so the new
  layout has to **consolidate** existing child binds into one parent bind.

## Design

**Target roots.** Relative to `--state-root`, except the one absolute root:

```bash
ROOTS=(
  task-evaluation-inputs                                # one tree: its stores hardlink into each other
  pubsub-handoffs                                       # scene workspaces and the handoff spool
  production-gpu-artifacts
  pipeline-control-plane/task-evaluation-launch-runs
  pipeline-control-plane/task-evaluation-policy-canaries
  pipeline-control-plane/capture-reconstruction-runs
  pipeline-control-plane/capture-reconstruction-derived
  pipeline-control-plane/episode-interpretation-backfills
  pipeline-control-plane/policy-canary-preprovider-audits
  pipeline-control-plane/scene-configuration-diagnostics
  pipeline-control-plane/result-artifact-cache
  pipeline-control-plane/profile-install-staging
  pipeline-control-plane/policy-canary-presubmission
  pipeline-control-plane/native-g1-team-campaign-work
  pipeline-control-plane/engineering
  pipeline-control-plane/render-probes
  pipeline-control-plane/diagnostic-checkouts
  pipeline-control-plane/release-builds
)
ABSOLUTE_ROOTS=(/workspace)                              # bound to ${MOUNT}/workspace
```

`task-evaluation-inputs` carries three small `evidence_hot` entries onto the volume:
`sam31-profile-registry`, `task-evaluation-terminal-results` and `g1-team-campaign-registry.json`.
This is deliberate. The volume is durable block storage, and splitting the tree would break the
hardlinks that keep it small. The plan output lists these under `evidence_hot on volume:` so the
owner sees them.

**Consolidation.** A root `R` is either:
- *bound*: `R` itself is a mount point, so skip it;
- *partially bound*: descendants of `R` are mount points whose source is `${MOUNT}/R/...`. These are
  the old per-root binds.

Detect mount points with `findmnt -rn -o TARGET` when not in `--root-prefix` mode, or from
`--bound-roots-file FILE` (one absolute path per line) in hermetic tests. For a partially bound
`R`, `apply` does, in order:

1. rsync `R/` → `${MOUNT}/R/` with the existing flag probing, plus one anchored `--exclude` for
   each bound child (`/child/rel/`), so the old binds' content, already on the volume, is not
   copied onto itself.
2. Verify with the same excludes: rsync dry-run itemize, or `diff -rq` in minimal mode.
3. For each bound child, deepest first:
   - `umount "$child"`;
   - delete its exact ` $child ` line from `/etc/fstab` with a temp file plus `mv`, after a
     `cp /etc/fstab /etc/fstab.blueprint-<epoch>.bak` backup.

   In hermetic mode, remove the child from the bound-roots file instead.
4. Continue as for an unbound root: `mv R R.migrated-to-volume`, `mkdir R`, restore owner and
   mode, append the fstab line, `mount --bind`, `rm -rf R.migrated-to-volume`.

Plan output adds lines like `consolidate <R> (bound children: a b)` and `evidence_hot on volume: …`.
Every existing refusal (ack, root, block device, copy drift) is unchanged.

**Governance.** A test reads `ROOTS`/`ABSOLUTE_ROOTS` out of the script text and checks them
against `control_plane_storage_roots.STORAGE_ROOTS`:
- every root of class `cache`, `evidence_cold`, `scratch` or `scene_workspace` under
  `/var/lib/blueprint` lies under some listed root;
- no listed root contains a `work`, `ledger` or `evidence_hot` root, except inside
  `task-evaluation-inputs` (the documented exception) and the handoff spool's own `work` root
  (`pubsub-handoffs`).

## Tasks (each: failing test → implement → pass → commit)

- [ ] **7.1 Governance test**, in `tests/test_mount_work_volume_script.py`:
  `test_every_bulk_storage_class_root_is_on_the_volume_and_queues_never_move`. It fails today:
  `pubsub-handoffs`, `capture-reconstruction-*`, `result-artifact-cache` and others are missing.
  Then update `ROOTS` and `ABSOLUTE_ROOTS`, and `WORKER_UNITS` (add the listener
  `blueprint-pubsub-handoff-listener.timer`/`.service`, the scene progression timer/service, the
  capture-reconstruction units and the agent stage-replay timer). Update
  `test_plan_lists_every_bulk_root_and_changes_nothing` for the tree-level root. Commit "Put every bulk root on the scratch volume".
- [ ] **7.2 Consolidation.** Tests:
  - `test_apply_consolidates_previously_bound_children_into_one_tree_bind`:
    - build a state where `task-evaluation-inputs/prepared-references` is listed in
      `--bound-roots-file` and its payload lives at
      `${MOUNT}/task-evaluation-inputs/prepared-references/payload.bin`;
    - the local mount point is an empty directory;
    - `task-evaluation-inputs/launch-activations` has local data.
    - After apply: both payloads are under `${MOUNT}/task-evaluation-inputs/…`, the local tree is an
      empty directory, and the child is gone from the bound-roots file.
  - `test_plan_reports_consolidation_and_evidence_hot_on_volume`
  - `test_hardlinks_across_input_stores_survive_the_move` (skip unless the local rsync supports `--hard-links`)

  Commit "Consolidate per-root volume binds into one tree bind so hardlinks keep working".
- [ ] **7.3 Docs.**
  - `docs/CONTROL_PLANE_STORAGE.md`: new section "Volume layout": what lives where; per-volume
    floors (link PR 6); why `task-evaluation-inputs` is one bind (EXDEV); the consolidation
    procedure with exact commands:

```bash
sudo deploy/host/mount_work_volume.sh --device /dev/disk/by-id/<volume> --plan
sudo deploy/host/mount_work_volume.sh --device /dev/disk/by-id/<volume> --apply --ack move-work-roots-to-volume
```

    plus the capacity env (`BLUEPRINT_CAPACITY_MOUNTS=/:/var/lib/blueprint:/mnt/blueprint-work`)
    and the owner decisions:
    - a separate small durable state volume is optional (root disk plus snapshots suffices);
    - when DigitalOcean keeps the 100 GB volume limit, attach a second volume and add a second
      `--mount` for a subset of roots (a pool of volumes per class) instead of growing one volume.
  - `docs/CONTROL_PLANE_CAPACITY_PLAN.md` host procedure: point step 3 at the new section.
  - Commit "Document the volume layout and its consolidation procedure".

## PR verification

`tests/test_mount_work_volume_script.py`, `tests/test_control_plane_storage_roots.py`.
