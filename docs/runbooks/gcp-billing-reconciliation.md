# GCP billing reconciliation

This repairs the observed GCP accounting blocker for ADP-009D's Day-35 runtime
comparison and the beta service's shared paid-resource admission. The existing
reconciler had no GCP collector, and project inventory left every stopped VM's
retained attachments unpriced. The change adds a bounded report reader and
read-only attachment pricing to those existing paths.

## Official report input

For a project without a historical billing export, an operator may download the
official Google Cloud Billing report, grouped by project and filtered to exactly
the configured GCP inventory projects. Use the existing cumulative cohort start,
not the monthly budget period. Download posted net costs, not the forecast. Keep
the original CSV bytes and the visible report context together in a private,
regular JSON file using `blueprint.gcp_official_billing_report.v1`:

- `source_kind`: `operator_downloaded_official_report`;
- `charge_type`: `posted_net_cost`, `currency`: `USD`;
- `billing_account_id`, `source_url`, and `report_context_text` from the report;
- `captured_at`: actual UTC download/observation timestamp, never refresh time;
- `period_start_date` and `period_end_date`: the report's calendar dates;
- `csv_utf8`: original CSV text, with `csv_sha256` binding its UTF-8 bytes.

Set `BLUEPRINT_GCP_BILLING_REPORT_RECEIPT` to this retained file and independently
set `BLUEPRINT_GCP_BILLING_ACCOUNT_ID` to the intended billing account. The CLI
also accepts `--gcp-report-receipt` and `--gcp-billing-account-id`. Project scope
comes from the existing `BLUEPRINT_GCP_INVENTORY_PROJECTS`/`BLUEPRINT_GCP_PROJECT`
configuration. Do not put private reports or live configuration in source control.

The reader requires every configured project exactly once, USD columns, a
matching CSV digest, and consistent unrounded project/subtotal/tax/total rows.
It supports calendar-date cohorts beginning at midnight; it cannot establish
sub-day charge attribution. Google's monthly chart can extend the report end
date through the current month's last day. This does not extend freshness:
reports older than 24 hours or dated in the future fail. The export's timestamp
remains at the oldest included source observation, so a timer cannot renew an
old report. Missing or invalid GCP evidence leaves GCP uncovered and live GCP
inventory continues to block paid admission. This is an operator report bridge,
not an automatic historical billing API or a replacement for monthly budget
notifications.

The complete envelope is retained through the existing content-addressed billing
audit. The reconciler never writes a hand-entered provider total or edits a guard
result. The separate monthly guard continues to enforce its configured limit.

## Retained VM rate

Stopped VMs remain in project inventory. Supported attachments are priced with
authenticated GETs to the current public Cloud Billing price API:

- Zonal `pd-balanced`: SKU `6AE1-525F-8B80`, only the regions in that SKU's
  catalog mapping. Live disk identity, size, type, status, and VM association
  must match the attachment.
- Associated external IPv4: standard SKU `C054-7F72-A02E`, or Spot SKU
  `4AF8-7C1F-39C4`. Complete paginated address inventory must prove association.

The hourly number is a **list-price bound**, not a provider charge receipt. The
disk's monthly list price uses Google's published 730-hour normalization; IP
pricing uses the highest tier rather than assuming account-wide free usage or
credits remain available. Component identities and returned prices accompany
the rate. The instance is read again after pricing; an identity, state, disk,
network, or scheduling change leaves the rate unknown. Unsupported or incomplete
inputs likewise remain unknown. Pricing never changes cleanup ownership.

This rate covers the VM's retained attachments. It does not inventory all Google
Cloud services or prove that a disk has no unique files. Preservation and a
separate content audit remain required before deletion.

Sources: [public price API](https://docs.cloud.google.com/billing/docs/how-to/get-pricing-information-api),
[disk pricing](https://cloud.google.com/compute/disks-image-pricing), and
[external IP pricing](https://cloud.google.com/vpc/network-pricing#ipaddress).

## Focused verification

`tests/test_gcp_billing_report.py` covers scope, exact posted totals, stale/future
sources, malformed reports, private-file handling, and export age retention.
`tests/test_gcp_retained_vm_cost.py` covers disk plus IP accounting, tier bounds,
unknown prices, missing inventory, VM races, and unchanged cleanup ownership.
The existing provider billing and cloud VM adapter suites protect compatibility.
