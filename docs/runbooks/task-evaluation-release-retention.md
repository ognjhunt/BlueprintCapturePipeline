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
  three releases (release-retirement CLI);
- any commit a live process runs from (retirement CLI);
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

Explicit retirement reads protection from typed sources, never by searching
files for 40-hex tokens. That search caught git tree ids, commits embedded in profile ids and the
consumption records of expired authorizations; on 2026-09-26 it held 513
"protected commits" over 95 release trees and deploy retired none of them.
`blueprint_pipeline.control_plane_release_leases` now produces one row per
protected commit, each naming an `owner`, a `reason`, the `run_ref` it serves
and an `expires_at_epoch`:

| Kind | Source | Protects | Lapses |
|---|---|---|---|
| `live_queue` | `*.json` in a live state of the launch, preparation (including `awaiting_source_preparation` and `awaiting_capacity`), SAM execution (`pending`, `processing`, `waiting_external`), episode compilation, activation, policy canary, scene construction and terminal resource release queues | `expected_production_commit`, `source_commit`, `expected_source_commit` (top level or under `request`), `release.commit`, `launch_profile_id` resolved through its profile, and paths naming `…/task-evaluation-control-plane-releases/<sha>` or `…/system-runtimes/<component>/<sha>` | when the envelope leaves the live states; at the latest 30 days after its mtime (with a warning) |
| `standing_authorization` | `standing-authorizations/<profile_id>.json` | the commit its profile runs (its `source_commit`, else its allocator's `--expected-source-commit`, else the first 40-hex segment of the profile id), while the authorization can still admit a launch | when it expires or runs out of launches or spend |
| `retention_binding` | `task-evaluation-release-retention-bindings/*.json` plus its lease | `source_commit` and `retained_release.source_commit` (never `retained_release.tree`, a git tree id), and the same for every ancestor binding it builds on | when its run is terminal and no protected descendant builds on it, when its lease expires with the run unknown, and at the lease's maximum lifetime |
| `configured_runtime` | `/etc/blueprint/task-evaluation-public-scene-machinery.json`, the scene-preparation bootstrap, and the machinery file the bootstrap names in `public_scene_machinery_path` | runtime and release paths in those files | when the configuration stops naming them (re-read every deploy) |

Wake-pending markers, consumption records, standing-authorization step logs,
launch profiles on their own, and bare commits in configuration protect nothing.

The live queues are read as one snapshot. Workers move envelopes between
states during the scan, so after each pass every live state is listed again;
if an envelope vanished between listing and reading, or one is present that the
pass never read, the pass is repeated, and a queue still moving after three
passes blocks with `release_protection_queue_unstable`.

A binding's run is the scene intent whose directory under
`task-evaluation-scene-intents` is the first component of its `evidence.path`
(factory output is `<factory_output_root>/<intent_id>/<attempt_id>/…`).
Revocation and expiry only close *new* execution, so:

- a `completed` progression is terminal;
- a revoked or expired intent with a paid attempt still in flight (a row under
  `attempts/` with no validated cancellation or terminal settlement, the rule
  progression uses for execution ownership) is live;
- a revoked intent with nothing in flight is terminal;
- an expired intent with nothing in flight is unknown, because its owner may
  still extend the window after expiry: the lease runs out its TTL first;
- an intent that is neither revoked nor expired is live, and one that cannot be
  read is unknown.

Chained SAM prefix adoptions share releases. An adoption whose source profile
adopted an earlier completed prefix (`completed_prefix_adoption`) reopens that
earlier adoption's release on replay, which is why
`publish_adoption_release_binding` republishes the ancestor's binding.
Protection is therefore transitive: the collector follows each binding's
evidence (its adoption record) through the source profile to the earlier
adoption and its binding, `sam31-prefix-<digest>.json`, and keeps every
ancestor of a protected binding, with the reason `ancestor_of:<binding>` and the
descendant's run and expiry. An ancestor whose binding is missing still has its
commits kept, with the warning `release_protection_binding_ancestor_missing`.
When a binding's chain cannot be read, its run is never treated as terminal; it
keeps its lease until the lease expires.

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
it writes the sidecar to a temporary file, fsyncs it and links it into place
(so a full disk leaves no partial lease behind) with a 14-day TTL and a 30-day
maximum lifetime, and the deploy receipt counts it in `migrated_binding_count`. A
binding whose run is already terminal lapses on that same deploy. While a run is
live, deploy renews its sidecar by atomic replace once less than half the TTL
remains, to `min(now + 14 days, max_expires_at_epoch)`; past the maximum
lifetime the lease lapses with the warning
`release_protection_lease_past_max_lifetime:<binding>`. A new binding writer may
instead carry `owner`, `expires_at_epoch`, `run_ref` and optionally
`max_expires_at_epoch` inline; inline leases are never renewed.

Blocking codes, each of which retires nothing until fixed:
`release_protection_queue_unreadable:<queue>[/<state>[/<name>]]`,
`release_protection_queue_unstable`,
`release_protection_profile_missing:<profile_id>`,
`release_protection_standing_authorization_invalid:<profile_id>`,
`release_protection_standing_authorization_commit_unknown:<profile_id>` (the
authorization can still launch, but no profile or profile id names its
release), `release_protection_live_profile_unreadable:<profile_id>` (a live
launch or authorization names a profile whose file cannot be read),
`release_protection_ancestry_unreadable:<binding>` (a protected binding's
adoption chain cannot be read to its root, so ancestors past the break cannot
be kept), `release_protection_source_missing:<directory>` (the
standing-authorization directory or binding root is missing while the
control-plane root exists), `release_protection_binding_invalid:<binding>`,
`release_protection_binding_changed:<binding>` (binding bytes no longer match
the sidecar's `binding_sha256`), `release_protection_lease_invalid:<binding>`,
`release_protection_lease_write_failed:<binding>`,
`release_protection_config_unreadable:<name>`, and
`release_protection_control_plane_root_missing`. A permission error, a symlink
or a FIFO where a document or directory belongs is a blocker, never "empty";
documents are opened without blocking. An identity in a code that is not a
plain identifier appears as `invalid-<digest>`, never raw.

Warnings, which do not block: `misplaced_retention_plan:<name>` (a retention
plan misplaced into the binding directory; the two-step tool still refuses it
until it is reconciled, below), `release_protection_queue_root_missing:<queue>`,
`release_protection_profile_commit_unpinned:<profile_id>` (a readable profile
that pins no release runs from the active one),
`release_protection_binding_ancestor_missing:<binding>`, and the lapse and
renewal warnings above. The two-step tool refuses a missing binding root
(`release_retention_evidence_binding_root_missing`) and any directory it cannot
examine (`…_unreadable`) the same way.

### Renewing or ending a lease

A lease owner never edits host files by hand. To keep a release past what
automatic renewal allows, or to end a lease early, land a reviewed commit whose
tool writes the sidecar: keep `binding`, `binding_sha256` and `commits`, set the
new `expires_at_epoch` (end a lease by setting it to the present) or a new
`max_expires_at_epoch`, and recompute `lease_digest` with
`decision_evidence_contracts.canonical_digest(lease, digest_field="lease_digest")`.
A sidecar that fails those checks blocks all retirement with
`release_protection_lease_invalid:<binding>`. The ordinary way to end a lease is
to let its run end: the next explicit retention evaluation reports it as `run_terminal`. The binding
itself is evidence and is never edited or removed to end a lease.

## Deployment preserves trees; retirement is separate

Ordinary deployment preserves all existing release/runtime trees and publication
receipts, including generated/untracked files and interrupted `.retiring` trees.
It does not sweep before disk admission, plan or apply retirement after activation,
or delete trees during closeout. Its receipt records `release_retirement.status`
`not_requested`, reason `requires_separate_action`, and zero retired bytes; it
leaves the last actual retirement summary unchanged. Disk admission still fails
closed when space is insufficient. Release identity, paid-launch gates, automation
state and owned holds retain their existing deployment safeguards.

Use the separate reviewed two-step operation below for retirement. Its checks
and explicit apply acknowledgment are unchanged. Review deletion authority and
actual recovery availability separately from code reproducibility; a Git commit
alone does not reconstruct generated or untracked state.

The existing `control_plane_release_retirement` module remains a separate tool.
With `--no-migrate` and without `--apply`, it takes no lock, writes no lease and
only plans/measures candidate sizes. Otherwise it holds the publisher lock roots
exclusively while collecting protection and planning; `--apply` additionally
requires its explicit acknowledgment, rechecks live processes, stages candidates
and deletes after releasing the locks. Its existing ENOSPC/EDQUOT fallback can
delete a candidate directly, and apply also deletes `.retiring` leftovers.
Deployment never invokes this tool or supplies its acknowledgment.

Retirement keeps the active/current commits, the newest three releases, trees
younger than a day, live-process references, and typed leases/configured runtimes.
Unproven protection blocks the operation. Previously staged `.retiring` trees
remain in place until an independently authorized retirement operation; another
deployment is not a cleanup trigger.

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
The success receipt records predicted and actually removed bytes. Deployment
preserves existing trees; this two-step process is the operator's reviewed
retirement path, including targeted `--keep-commit` pins.

The two-step tool evaluates each binding's lease through the typed retention
rules, from
`--lease-root`, `--intent-root` and `--launch-run-root` (the host's roots by
default), but never migrates or renews one: a binding without a sidecar is
evaluated as the lease its migration would write. Any commit token in a
pending or processing queue document still protects. A binding whose lease lapsed
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
