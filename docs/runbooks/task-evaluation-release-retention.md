# Task Evaluation release retention

This is the ADP-009D day-28 disk-safety policy for the website-driven scene
configuration control plane. Each deploy publishes three immutable trees for
one protected-main commit:

- `/opt/blueprint/task-evaluation-control-plane-releases/<sha>`
- `/var/lib/blueprint/task-evaluation-inputs/system-runtimes/splat-render/<sha>`
- `/var/lib/blueprint/task-evaluation-inputs/system-runtimes/scene-configuration/<sha>`

The three trees have one retention identity. A commit is retained everywhere
when any protected binding exists and is eligible everywhere only when none
exists. A deployment must never remove just the browser or toolchain behind a
still-retained checkout.

## Retention policy

A tree is never eligible until its youngest managed artifact is at least 24
hours old. Age is only a grace period; it does not override a binding. The
following commits are always retained:

- the exact target of the active-release symlink;
- the explicitly named current deploy candidate;
- an explicit operator `--keep-commit` pin (two-step tool) or one of the newest
  three releases (deploy);
- any commit a live process runs from (deploy);
- any commit, or profile resolving to a commit, in a live Task Evaluation
  queue document;
- any published profile whose standing authorization is valid, unexpired, and
  still has launch and spend capacity;
- any commit named by a required-evidence binding whose lease has not lapsed;
- any runtime path named by the public-scene machinery or the
  scene-preparation bootstrap configuration.

An expired, launch-exhausted, or spend-exhausted standing authorization does not
pin a runtime forever. A malformed authorization does not count as exhausted:
it blocks the entire retention operation. The same fail-closed rule applies to
an unreadable or malformed queue document, profile, public catalog, dry-run
plan, or required-evidence binding; a symlink at a managed boundary; an unknown
child under a managed root; or a managed target whose inode, mtime, or byte
count changes between review and apply.

Required evidence that still needs the executable tree must have an immutable
JSON binding in the configured evidence-binding root (its lease is described
under Release leases below):

```json
{
  "schema_version": "task_evaluation_release_retention_binding.v1",
  "status": "required",
  "source_commit": "0123456789abcdef0123456789abcdef01234567",
  "reason": "terminal qualification replay remains open"
}
```

Receipts and terminal evidence that already bind a protected Git commit do not
implicitly require a host-resident checkout. Use the explicit binding only
when replay or open qualification genuinely depends on those local bytes.

## Release leases

Deploy-time retirement reads protection from typed sources, never by searching
files for 40-hex tokens. That search caught git tree ids, commits embedded in profile ids and the
consumption records of expired authorizations; on 2026-09-26 it held 513
"protected commits" over 95 release trees and deploy retired none of them.
`blueprint_pipeline.control_plane_release_leases` now produces one row per
protected commit, each naming an `owner`, a `reason`, the `run_ref` it serves
and an `expires_at_epoch`:

| Kind | Source | Protects | Lapses |
|---|---|---|---|
| `live_queue` | `*.json` in a live state of the launch, preparation (including `awaiting_source_preparation` and `awaiting_capacity`), SAM execution (`pending`, `processing`, `waiting_external`), episode compilation, activation, policy canary, scene construction and terminal resource release queues | `expected_production_commit`, `source_commit`, `expected_source_commit` (top level or under `request`), `release.commit`, `launch_profile_id` resolved through its profile, and paths naming `…/task-evaluation-control-plane-releases/<sha>` or `…/system-runtimes/<component>/<sha>` | when the envelope leaves the live states; at the latest 30 days after its mtime (with a warning) |
| `standing_authorization` | `standing-authorizations/<profile_id>.json` | the commit its profile runs, while the authorization can still admit a launch | when it expires or runs out of launches or spend |
| `retention_binding` | `task-evaluation-release-retention-bindings/*.json` plus its lease | `source_commit` and `retained_release.source_commit` (never `retained_release.tree`, a git tree id) | when its run is terminal, when its lease expires with the run unknown, and at the lease's maximum lifetime |
| `configured_runtime` | `/etc/blueprint/task-evaluation-public-scene-machinery.json`, the scene-preparation bootstrap, and the machinery file the bootstrap names in `public_scene_machinery_path` | runtime and release paths in those files | when the configuration stops naming them (re-read every deploy) |

Wake-pending markers, consumption records, standing-authorization step logs,
launch profiles on their own, and bare commits in configuration protect nothing.

A binding's run is the scene intent whose directory under
`task-evaluation-scene-intents` is the first component of its `evidence.path`
(factory output is `<factory_output_root>/<intent_id>/<attempt_id>/…`). The run
is terminal when the intent's progression is `completed`, when `revoked.json`
exists, or when the intent's effective execution window (including approved
extensions) has passed. It is live while the intent directory exists otherwise,
and unknown when it cannot be read or the binding names no intent.

### Binding sidecars and migration

Binding bytes never change: `publish_adoption_release_binding` compares the
whole document on republish, and a different byte would make it refuse its own
binding. A binding's lease therefore lives in a sidecar,
`/var/lib/blueprint/pipeline-control-plane/release-leases/bindings/<binding>.lease.v1.json`
(root:root, directory 0750, file 0640):

```json
{
  "schema_version": "control_plane_release_lease.v1",
  "binding": "sam31-prefix-<digest>.json",
  "binding_sha256": "sha256:<digest of the binding bytes>",
  "commits": ["<sha>"],
  "owner": "legacy-migration",
  "reason": "<the binding's reason>",
  "run_ref": {"kind": "scene_intent", "intent_id": "scene-<id>"},
  "created_at_epoch": 1790000000.0,
  "expires_at_epoch": 1791209600.0,
  "max_expires_at_epoch": 1792592000.0,
  "migrated": true,
  "lease_digest": "sha256:<canonical digest of this document without lease_digest>"
}
```

The first deploy after this change migrates every binding without a sidecar:
it creates the sidecar exclusively with a 14-day TTL and a 30-day maximum
lifetime, and the deploy receipt counts it in `migrated_binding_count`. A
binding whose run is already terminal lapses on that same deploy. While a run is
live, deploy renews its sidecar by atomic replace once less than half the TTL
remains, to `min(now + 14 days, max_expires_at_epoch)`; past the maximum
lifetime the lease lapses with the warning
`release_protection_lease_past_max_lifetime:<binding>`. A new binding writer may
instead carry `owner`, `expires_at_epoch`, `run_ref` and optionally
`max_expires_at_epoch` inline; inline leases are never renewed.

Blocking codes, each of which retires nothing until fixed:
`release_protection_queue_unreadable:<queue>/<state>/<name>`,
`release_protection_profile_missing:<profile_id>`,
`release_protection_standing_authorization_invalid:<profile_id>`,
`release_protection_binding_invalid:<binding>`,
`release_protection_binding_changed:<binding>` (binding bytes no longer match
the sidecar's `binding_sha256`), `release_protection_lease_invalid:<binding>`,
`release_protection_lease_write_failed:<binding>`,
`release_protection_config_unreadable:<name>`, and
`release_protection_control_plane_root_missing`. A retention plan misplaced
into the binding directory is only the warning `misplaced_retention_plan:<name>`
at deploy; the two-step tool still refuses it until it is reconciled (below).

### Renewing or ending a lease

A lease owner never edits host files by hand. To keep a release past what
automatic renewal allows, or to end a lease early, land a reviewed commit whose
tool writes the sidecar: keep `binding`, `binding_sha256` and `commits`, set the
new `expires_at_epoch` (end a lease by setting it to the present) or a new
`max_expires_at_epoch`, and recompute `lease_digest` with
`decision_evidence_contracts.canonical_digest(lease, digest_field="lease_digest")`.
A sidecar that fails those checks blocks all retirement with
`release_protection_lease_invalid:<binding>`. The ordinary way to end a lease is
to let its run end: the next deploy lapses it as `run_terminal`. The binding
itself is evidence and is never edited or removed to end a lease.

## Deploy-time retirement

Every deploy retires superseded trees after the new release is proven live
(`control_plane_release_retirement`). It holds every release-reference
publisher lock exclusively (the control-plane root, which queue writers, the
profile publisher, the standing-authorization materializer and release
activation lock shared) while it collects protection with migration, plans,
and applies. It keeps the newest three releases, the active and current
commits, anything a live process runs from (release or runtime tree, re-checked
immediately before each commit is removed; a busy commit is skipped as
`in_use_at_apply`), anything younger than a day, and every lease or configured
runtime above.

The receipt's `release_retirement` records `status` (`applied`, `skipped` with
`blockers`, or `blocked`), `retired_commits`, `retired_bytes`, `skipped`,
`protected_by_kind` (tree counts per kind: `active_release`, `current_deploy`,
`keep_last`, `in_use_by_live_process`, `younger_than_minimum_age`, and the four
protection kinds), `protected_tree_count`, `lease_protected_tree_count` (trees
held only by leases), `lapsed_count`, `migrated_binding_count`,
`renewed_lease_count`, `warning_count`, `alerts` and `lock_roots`. The same
summary, with `generated_at_epoch` and `source_commit`, replaces
`/var/lib/blueprint/pipeline-control-plane/release-retention/latest-deploy-retirement.json`
(0644) on every deploy. Its alerts are:

- `release_retirement_lease_protected_trees:<n>` when more than 20 trees are held
  only by leases, which means leases are not lapsing and someone must look;
- `release_retirement_blocked:<first blocker>` when retirement skipped or could
  not run.

A `blocked` retirement carries `release_reference_lock_root_unavailable` (or
another `release_reference_lock_*` code), `deploy_release_lease_root_unsafe`,
or `deploy_release_retirement_failed:<exception type>`. A retirement problem
never fails a deploy whose surfaces already moved. For an out-of-band look,
`python -m blueprint_pipeline.control_plane_release_retirement` takes the same
sources (`--control-plane-root`, `--profile-dir`,
`--standing-authorization-dir`, `--binding-root`, `--lease-root`,
`--config-file`, `--intent-root`, `--launch-run-root`); pass `--no-migrate` for
a dry run that writes no lease.

## Two-step operation

The dry run is mandatory and performs no deletion:

```bash
python -m blueprint_pipeline.task_evaluation_release_retention \
  --current-deploy-commit "$SHA" \
  --receipt-out /var/lib/blueprint/pipeline-control-plane/release-retention/dry-run.json
```

Review `eligible_commits`, all three artifacts for each commit,
`protected_commits`, and `predicted_removed_bytes`. Apply only those exact
reviewed bytes:

```bash
python -m blueprint_pipeline.task_evaluation_release_retention \
  --apply \
  --dry-run-plan /var/lib/blueprint/pipeline-control-plane/release-retention/dry-run.json \
  --ack reap-task-evaluation-release-artifacts \
  --receipt-out /var/lib/blueprint/pipeline-control-plane/release-retention/applied.json
```

Apply re-reads every liveness document and re-stats every target before the
first deletion. Any difference from the reviewed plan refuses the operation.
The success receipt records predicted and actually removed bytes. Deploy
retires automatically (above); the two-step process remains the operator's
reviewed path for anything more targeted, such as `--keep-commit` pins.

The two-step tool evaluates each binding's lease exactly as deploy does, from
`--lease-root`, `--intent-root` and `--launch-run-root` (the host's roots by
default), but never migrates or renews one: a binding without a sidecar is
evaluated as the lease its migration would write. Its reading of live queue
documents is unchanged and stricter than deploy's: any commit token in a
pending or processing document still protects. A binding whose lease lapsed
does not protect and is listed under `lapsed_evidence_bindings`; the sidecar
bytes are digest-bound reference documents of the plan, so a lease that changes
between review and apply refuses the apply.

The tool neither contacts Vast nor reads credentials. It must not be used as a
substitute for provider teardown, provider-zero, or evidence-retention policy.

The dry-run plan belongs under
`/var/lib/blueprint/pipeline-control-plane/release-retention/`; the CLI refuses
to write it into `task-evaluation-release-retention-bindings/`. If an older
operator invocation already placed a plan in the binding namespace, reconcile
that one exact file before another scan:

```bash
python -m blueprint_pipeline.task_evaluation_release_retention \
  --reconcile-misplaced-plan \
    /var/lib/blueprint/pipeline-control-plane/task-evaluation-release-retention-bindings/plan-<timestamp>.json \
  --receipt-out \
    /var/lib/blueprint/pipeline-control-plane/release-retention/reconciliation-<timestamp>.json
```

Reconciliation accepts only canonical bytes with a valid retention-plan
digest, creates the same filename in the retention-plan root without
overwriting, removes the misplaced source, and records both byte digests. It
refuses evidence bindings, unknown JSON or bytes, symlinks, nested paths, and
pre-existing destinations or receipts.
