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
- an explicit operator `--keep-commit` pin;
- any commit, or profile resolving to a commit, in a pending or processing
  Task Evaluation queue document;
- any published profile whose standing authorization is valid, unexpired, and
  still has launch and spend capacity;
- any commit named by a required-evidence binding.

An expired, launch-exhausted, or spend-exhausted standing authorization does not
pin a runtime forever. A malformed authorization does not count as exhausted:
it blocks the entire retention operation. The same fail-closed rule applies to
an unreadable or malformed queue document, profile, public catalog, dry-run
plan, or required-evidence binding; a symlink at a managed boundary; an unknown
child under a managed root; or a managed target whose inode, mtime, or byte
count changes between review and apply.

Required evidence that still needs the executable tree must have an immutable
JSON binding in the configured evidence-binding root:

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
The success receipt records predicted and actually removed bytes. This release
does not install an automatic deploy hook or timer; an operator invokes the
two-step process after a deploy while rollout behavior is being observed.

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
