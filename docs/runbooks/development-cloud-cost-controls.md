# Development cloud cost controls

Owner direction, October 23, 2026: keep developing while lowering retained
infrastructure cost; `blueprint-8c1ca` has a USD 100/calendar-month ceiling.
This does not replace or increase a capture's provider allowance. This removes
a beta capture-to-scene development blocker and protects ADP-040/day-28.

## Before enabling Blaze

- Keep billing unchanged until the owner enables it. Billing disabled is not
  evidence that a VM was deleted or its attachments retired.
- Bind a project-scoped USD 100 monthly budget to project number 744608654760
  on its selected billing account; read it back. Preserve existing budgets.
  Cost reports already show more than USD 100 this October, so the existing
  month has no new GCP development allowance. Do not reset its accounting.
- Verify `blueprint-policy-sandbox-20260929` in `us-central1-a` is stopped,
  retain its data and inspect its active consumers before retirement. Its
  last start found in the audit was September 29. Do not restart it simply to
  restore Firebase storage access.
- Configure `BLUEPRINT_GCP_INVENTORY_PROJECTS=blueprint-8c1ca` on the existing
  spend-checking execution context. Run the existing guard read-only, with
  `--require-provider gcp`, and retain its report. It now queries all zones,
  includes CPU VMs and stopped VMs, and does not require GPU launch readiness.
  Failure to read inventory is a blocker, not a fleet-zero receipt. Expanded
  inventory cannot authorize automatic deletion of unowned VMs.
- Read Artifact Registry's inventory and every active run, release, rollback
  and proof image digest. Keep every referenced or unresolved digest. Preview
  deletion only for independently proven unused versions. A tagless image can
  still be pinned by digest; deleting all untagged images is unsafe. Verify
  retained digests before applying cleanup. Preserve stopped/report-only
  retention programs; do not substitute a broad deletion rule.

Google refuses Compute/Artifact Registry reads while project billing is
disabled. If those checks cannot run, report remaining activation work; do not
represent source rules or a local test as live cloud enforcement.

## Existing code enforcement

`GCPRenderProvider` requires `BLUEPRINT_GCP_MAX_RUN_SECONDS`, from 30 seconds
through Google's documented 120-day maximum. Choose it from the admitted
job's runtime/budget. Native `scheduling.maxRunDuration` and
`instanceTerminationAction=DELETE` protect a temporary VM if its controller
dies. Its temporary boot disk already uses `autoDelete=true`. Launch checks
the native deadline again before API access: removing it from a prepared
request cannot bypass the check. This ceiling is not permission to allocate.

The existing spend guard uses separate GCP inventory readiness, explicit
project scopes, all-zone paginated inventory, unknown-cost rejection and
report-only handling outside the configured managed launcher scope. Existing
paid admission, rights and teardown controls still apply.

## Develop using retained assets

Replay retained inputs with the production stage handler without network,
GPU or model calls before resubmission. Adopt completed stages for identical
inputs. Keep raw files, cleaned images, returned RGBD and exported assets;
changing planning code does not require recreating them. Use local CPU work
for small replays and governed temporary cloud workers for work that does not
fit locally. Do not move large artifacts back onto a full local disk.

Registry storage and stopped-VM disks remain paid assets while billing is on.
A lower VM runtime does not solve image growth. Reuse dependency layers across
code-only builds and retain reviewed required versions, with deletion proof.

Google's alerts-only budgets do not stop spending. Native spend caps currently
cover selected services (including Gemini API, Cloud Run and Cloud Run
functions), have reporting delay, and do not cover Compute Engine or Artifact
Registry. A whole-project USD 100 ceiling also needs bounded allocations,
VM lifetimes and storage controls. Do not promise an instantaneous Google-wide
dollar cutoff.

References:
- https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime
- https://docs.cloud.google.com/artifact-registry/docs/repositories/cleanup-policy
- https://docs.cloud.google.com/billing/docs/how-to/budgets-spend-caps

## Monthly admission and automatic stop

The existing `blueprint-gpu-spend-guard.timer` runs the optional controller in
`gcp_monthly_budget_guard.py`. Set `BLUEPRINT_GCP_BUDGET_GUARD_CONFIG` to a
root-managed policy file outside the checkout. The file binds one project ID
and number, billing account, monthly USD amount, exact budget ID, Pub/Sub topic
and subscription, and separate state/report paths. It uses the existing ADC
credential integration; do not create or copy service-account keys.

The controller rechecks the live budget scope and amount, consumes its budget
notifications, inventories CPU/GPU VMs across the project's zones, and retains
the highest observed cost for that month. Duplicate, late or lower cost
notifications cannot reopen an exhausted month. A retained cost-report digest
can seed an over-budget refusal; a seed below the limit is rejected. A new
month waits for a real current-month observation. Missing/stale observations,
inaccessible APIs, corrupt state and failed cutoff readbacks remain refusals.

The selected reconstruction job also sets
`BLUEPRINT_GCP_MONTHLY_BUDGET_REPORT` to that policy's report. The shared paid
admission checks it both before issuing a grant and immediately before provider
mutation, including previously issued grants. Reports must match the policy,
project, month and amount, and be no more than five minutes old. This project
allowance does not replace unrelated providers' fleet allowances.

The policy defaults `billing_shutdown_authorized` to false. At or above the
limit, it refuses spending and reports that activation is required. If the
owner authorizes whole-project shutoff, setting that field true enables only
unlinking that exact project from its configured billing account, followed by
independent billing-status readback. It never re-enables billing. Keep the
controller on the existing external host so a disabled Google project cannot
erase its blocked-month state or silently restart it.

Activation has a material consequence: Google's documented shutoff stops all
services in the project, including Firebase/Storage, and warns that resources
may be irretrievably deleted. Review that consequence with the owner before
activating it. Budget notifications and cost reporting are delayed; even this
automatic shutoff cannot guarantee an invoice of exactly USD 100. Do not
describe alerts, a local fixture or a configured callback as verified native
cutoff. Retain the actual admission refusal and Google billing readback.

Required read access is budget viewing on the selected billing account,
subscriber access on the specific budget subscription, project billing-info
read access and Compute inventory access. Billing-unlink authority, when
activated, must be limited to the selected project and must not include
permission to re-enable billing or raise the budget.

References:
- https://docs.cloud.google.com/billing/docs/how-to/budgets-programmatic-notifications
- https://docs.cloud.google.com/billing/docs/how-to/disable-billing-with-notifications
