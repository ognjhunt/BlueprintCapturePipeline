# Provider billing audit retention

This is the ADP-009D day-28 disk-safety procedure for the official billing
evidence used by paid-resource admission. The billing reconciler runs every ten
minutes. Provider APIs commonly return identical cumulative response bytes, but
historically each refresh wrote another full copy under
`gpu_spend_guard/billing-audit/<timestamp>/`.

The reconciler now keeps the evidence contract unchanged:

- every `blueprint.provider_billing_source_receipt.v1` remains immutable;
- every receipt retains its original absolute `retained_path`;
- each receipt-local response remains a regular `0600` file with the bound
  size and SHA-256 digest;
- equal responses are hard links to one object under
  `billing-audit/objects/sha256/<prefix>/<digest>`.

The audit-root directory inode is the shared lock. The reconciler and migration
both take an exclusive lock without creating a lock file. Object publication is
same-filesystem and no-overwrite; an existing object is reused only after its
owner, group, mode, size, and digest are revalidated.

## Historical migration

Run the historical scan as root because a small number of legacy response files
are root-owned and not readable by the service account. Dry-run is mandatory
and does not create the object directory or alter production artifacts:

```bash
python -m blueprint_pipeline.provider_billing_audit_retention \
  --audit-root /var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/billing-audit \
  --receipt-out /var/lib/blueprint/pipeline-control-plane/billing-audit-retention/dry-run.json
```

Review the receipt count, response groups, every inode snapshot,
`metadata_excluded_response_paths`, `directory_repairs`, and
`predicted_relinked_bytes`. Also review
`unreconciled_incomplete_transactions`: a historical timestamp directory that
has no source receipt is inventory-bound, retained exactly, and excluded from
directory repair, object publication, relink, and deletion. Its secure regular
`response-N-provider.json` files are hashed and snapshotted; an unknown child,
symlink, insecure metadata, or cross-filesystem file blocks the scan globally.
Secure legacy response files whose owner/group/mode
differs from the current audit-owner `0600` contract remain byte-, path-, and
inode-preserved rather than being silently coerced into a hard-link group.
Apply only the exact reviewed plan:

```bash
python -m blueprint_pipeline.provider_billing_audit_retention \
  --apply \
  --dry-run-plan /var/lib/blueprint/pipeline-control-plane/billing-audit-retention/dry-run.json \
  --ack deduplicate-provider-billing-audit \
  --receipt-out /var/lib/blueprint/pipeline-control-plane/billing-audit-retention/applied.json
```

Apply re-locks the root, validates the plan digest, rescans every receipt and
response, and re-stats every directory, receipt, object, and response before the
first mutation. Reviewed legacy timestamp directories are conditionally repaired
to the audit owner and `0700`, with a readback, before any response is relinked.
A changed inode, byte count, mode, owner, digest, receipt, symlink, unknown
directory child, cross-filesystem response, or malformed binding blocks the
entire operation. Receipt bytes are never rewritten. Each eligible duplicate
path is atomically replaced by a hard link carrying the exact same bytes.

Do not apply the migration until the content-addressed reconciler is deployed;
an older writer does not participate in the audit-root lock. This process does
not contact a provider, change provider resources, replace provider-zero, or
authorize paid execution.
