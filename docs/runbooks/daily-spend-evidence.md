# Daily spend evidence

This owner-requested finance visibility slice reads already received evidence.
It does not add provider calls, pages, grouping, credentials, services, timers,
admission authority or reservation changes. The active AWS removal owner owns
the small `provider_billing_reconciler.py` hook and Pipeline deployment.
The actual ADP backlog/day-gate metadata was not supplied; this is not a program
closeout or physical proof artifact.

The existing reconciliation refresh writes the unchanged billing-v1 export.
`MeteredTransport` fsyncs an immutable private attempted record before each
dispatch, then records received bytes or a separate error. A crash leaves an
unresolved attempt. Run/attempt/retry/page identities are distinct; no replay
command exists. Events contain fixed provider/service/endpoint names and hashes,
never headers, tokens, bodies or exception messages. Cost remains null until
separately reconciled. Forbidden AWS endpoints emit a denied local event and
cannot reach the wrapped transport. The B2 S3-compatible path is untouched.

The same existing refresh then calls the offline observer, independently of
admission. It writes `daily_spend_snapshot.json` next to the existing billing
export, readable through the existing operator file route. The snapshot includes
local request-day counts, received/errors/unresolved/denied outcomes and warnings.
These cover only the instrumented Pipeline billing transport, not all research,
communications or company workflows. Request day is not usage day.

The offline entrypoint works on retained files without credentials or network:

```bash
PYTHONPATH=src python -m blueprint_pipeline.daily_spend_snapshot \
  --billing-export /path/to/provider_billing_export.json \
  --source-receipt /path/to/billing-audit/run/provider_billing_source_receipt.json
```

It verifies receipt/export and response hash/size bindings. Retained response
basenames are resolved beside the receipt, supporting authorized copied exports.
Historical AWS sources are excluded. No GCP data source is invented. By default
offline files are unverified; custom transports are development-only. Neither
creates publishable actual-spend rows.

- Vast rows retain official intervals/day, pseudonymous resource/project and
  GPU/disk/download/upload components. Sparse/malformed components, invalid
  amounts/currencies and conflicting rows are partial. No-ID rows are retained
  with correction uncertainty, excluded from daily slices and Notion upserts;
  supersession is unresolved without a stable provider identity. Disk-only scene
  costs are never replaced or repriced by this observer.
- RunPod responses keep year grain and explicit query cohort separately from
  unknown official intervals. They cannot establish daily costs.
- DigitalOcean invoices are references. Previews remain estimates with the
  balance's original observation time, including stale values; finalized/preview
  overlap across pages is flagged. Cash/prepaid/reservations are not synthesized.
- Cumulative changes are posting changes, never proven usage-day costs. Signed
  credits and revisions are preserved; exact duplicate source revisions collapse.
- UTC intervals are only allocated when wholly inside one Chicago day, using
  timezone-aware, end-exclusive boundaries including 23/25-hour DST days.
  Unallocatable intervals remain visible with a gap. Observed slices are partial;
  all-provider/day totals remain null. Source age is separate from pointer time.

`notion_projection` is a sanitized local plan for the existing authorized daily
spend page/ledger, whose current schema was read through the connected Notion
route. It performs no new connector request and has no schedule or dependency on
the parent digest. The existing publisher must look up exact Source key, upsert
that row, clear corrected unknown amounts/dates, and read back the revision and
snapshot digest. Every row remains reference-only until scope/coverage is
reconciled; monthly references, reserves, estimates and cash never enter daily
expense totals. No fixture plan may be published as live.

Validation covers unchanged dispatch count, zero AWS dispatch, persistence before
dispatch, denied/error/crash/unknown outcomes, retry/page identity, offline
hash-bound parsing, duplicate/credit/correction/unknown currency, partial pages,
invoice overlap, source freshness vs pointer freshness, DST, secret redaction,
and development-only publication refusal. The normal impacted-test/CI gates and
independent Sol review remain release requirements.

Runtime boundary: this delegated environment's existing operator `whoami` read
returned a proxy tunnel 403. Thus no live billing files, this candidate's runtime
snapshot or Notion actual-spend upsert/readback have been obtained here. Minimal
remaining input is an already authorized owner read of the retained export,
matching source receipt/responses and the resulting snapshot; no new grants or
provider calls are needed. Deployment must coordinate with the AWS removal owner.
