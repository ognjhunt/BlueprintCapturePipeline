# Paid Spend Admission Lock

Status: production fail-closed contract. The code and hermetic tests do not
claim a current provider invoice, delivered page, or completed live teardown.

The beta cohort hard stop is exactly `$5,000`. In production,
`require_pre_spend_preflight` refuses every paid provider launch unless the
systemd spend guard has written a fresh
`blueprint.paid_spend_admission_lock.v1` artifact with:

- a current, complete USD billing reconciliation for RunPod, Vast, and
  DigitalOcean;
- the conservative allocation-ledger total;
- `effective_spend_usd` equal to the larger of billing and allocation totals;
- `effective_spend_usd < 5000`, or a valid short-lived override;
- no inventory, burn-rate, daily-budget, billing, or override blocker.

Equality is a crossing: `$5,000.00` stops new paid work and emits a critical
page event. A valid audited override may reopen admission for its bounded
interval, but never suppresses that page. The lock preserves active owned work
as `draining`, reaps only eligible orphans through the existing
provider-confirmed teardown path, and does not mark teardown complete until no
live allocation remains and every reap candidate has a successful termination
result.

## Billing export input

The external billing synchronizer must atomically replace the file configured
by `BLUEPRINT_GPU_BILLING_EXPORT` with this shape:

```json
{
  "schema_version": "blueprint.provider_billing_export.v1",
  "generated_at": "2026-07-09T18:00:00+00:00",
  "currency": "USD",
  "scope": "blueprint_beta_100_user_cohort",
  "provider_totals_usd": {
    "runpod": 0.0,
    "vast": 0.0,
    "digitalocean": 0.0
  }
}
```

The guard rejects a file older than 24 hours, a future timestamp, another
currency/scope/schema, negative/non-numeric totals, or any omitted provider.
The resulting artifact records the billing input digest and basename, not an
absolute runner path. Supplying this file is external billing evidence; the
repository does not manufacture it.

## Audited override

An override is normally absent. If incident recovery requires one, set
`BLUEPRINT_PAID_SPEND_OVERRIDE_PATH` to a non-symlink regular file that is not
group/world writable. It must use
`blueprint.paid_spend_override.v1`, name distinct requester and approver
identities, include a durable HTTPS ticket and substantive reason, bind the
`$5,000` threshold and `paid_spend_hard_stop` scope, and expire within four
hours. Expired, malformed, overlong, same-person, or permission-unsafe
overrides lock admission and page; they never fall back to an unaudited open
state.
The maximum override validity interval is four hours.

An override applies only to the cohort-total crossing. It cannot waive missing
billing, unknown inventory, daily/burn ceilings, or other fleet blockers.

## Page and teardown evidence

`blueprint-gpu-spend-guard.service` writes both the spend snapshot and admission
lock every two minutes. Its `ExecStopPost` always evaluates the lock with the
operator webhook alert path. A required webhook that is absent or fails keeps
the unit failed. The admission lock's `page_event.delivery_status` remains
`external_pending`; only the separate alert audit can say a webhook was sent.

Likewise, `controlled_drain.status=draining` is not provider absence proof.
Provider termination is closed only by the existing API-confirmed teardown
proof, and current billing/page/teardown execution remain external until those
artifacts arrive for the live environment.
An API-confirmed teardown proof is required before closure.
