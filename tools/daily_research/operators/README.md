# Scoped research operator commands

Code and instructions are maintained in Blueprint's GitHub repository. Firestore
is the durable research ledger; provider IDs are provenance. Operator commands
must use an exact reviewed Git commit and the exact installed standalone package.
They have no ChatGPT Library dependency and require no Pipeline application,
GPU installation, new key, service, OAuth connection or security setting.

## Disabled October 2 migration

`research-oct2-control.py plan` performs Firestore GETs and writes a new private
local plan (0600, exclusive create, fsync). It makes no provider calls and does
not take a lease. `apply-disabled` is a Firestore-writing operator action. It
uses the existing fenced lease, compares the full current control and retained
Oct1 row/raw with the plan, then configures and verifies exact readback. A stale
plan refuses; generate a new file rather than editing or bypassing its digests.

This helper is pinned to Pipeline `35f5c9ad43f84aa053aa7616a63a9aa4f6e32a61`,
archive SHA256 `1aa932767fe9ec73c06ece6b5ba1e573027a636a3249363d62df7bf6415a3651`
(409600 bytes, 41 source files). It verifies the archive, manifest, all installed
source hashes and actual imported module. WebApp
`b0cbd5e4c84120cf79ca704554433b8723a08b01` vendors this same receipt; operator
must verify the currently deployed installation rather than infer it from Git.

Run from `/opt/render/project/src` after materializing the unchanged reviewed
helper into a private directory using the existing authenticated operator route:

```sh
PYTHONPATH=/opt/render/project/src/dist/daily-research/release \
  /opt/render/project/src/dist/daily-research/venv/bin/python \
  /tmp/blueprint-oct2-reviewed/research-oct2-control.py plan \
  --package /opt/render/project/src/dist/daily-research/release \
  --archive /opt/render/project/src/vendor/daily-research/blueprint-research.tar \
  --output /tmp/blueprint-oct2-reviewed/live-disabled-plan.json
```

After scoped migration authorization, apply the same fresh plan:

```sh
PYTHONPATH=/opt/render/project/src/dist/daily-research/release \
  /opt/render/project/src/dist/daily-research/venv/bin/python \
  /tmp/blueprint-oct2-reviewed/research-oct2-control.py apply-disabled \
  --package /opt/render/project/src/dist/daily-research/release \
  --archive /opt/render/project/src/vendor/daily-research/blueprint-research.tar \
  --input /tmp/blueprint-oct2-reviewed/live-disabled-plan.json
```

Expected readback: root and config enabled are false, source is the exact pinned
Pipeline source, contract v3, search `perplexity-fast-v1`, discovery
`adaptive-sites-v1`, total runtime 1800 seconds, QA reserve 600 (research 1200),
soft total target $5 and recurring budget reference exactly
`Sentinel_c30352f247c88191bbd695cc2bd99de1`. That receipt approves model, search
and hosted environment total as a soft target; it is not a hard API billing cap.

Preserve first_date, existing workflow settings/authorities, approval/scheduler
references, unrelated control fields, history, failed intent, raw bytes and
cleanup state. Existing lease bookkeeping is preserved through configure and
released normally. No provider session, run, input, publication, outreach,
deployment or deletion occurs. Root false makes the preserved workflow dormant.
The retained Oct1 date remains occupied, so an early restart cannot recreate it;
October 2 07:00 America/Chicago is 12:00 UTC. Enabled cutover is a separate action
after test, publication and cleanup evidence; do not simply flip a Boolean now.

## Private artifacts and portable handoff

Stable normal identity: `blueprint-researcher:YYYY-MM-DD`, with canonical document
`blueprintDailyResearch/sites-first/runs/YYYY-MM-DD`. Complete requests, raw
artifacts and tool evidence already survive restart in Firestore immutable blobs
and chunks. The installed `render export` command verifies bindings and exports
ordinary JSON status/artifact/evidence/review/QA/tool files. Export is read-only
for Firestore and writes a new private local directory; it is not publication.

Existing company object storage candidate is
`gs://blueprint-8c1ca.appspot.com`. The native runtime owner independently
verified a US Standard bucket, existing identity create/get/list, and no public
IAM bindings. Exact destination approval for private backup writes is still
required before any transfer. Do not provision credentials, grant access, make
objects public, mint Firebase download tokens, or delete existing copies.

Use a Blueprint-owned prefix such as
`blueprint-workflows/daily-research/sites-first/YYYY-MM-DD/exports/SHA256/`.
Publish a standard JSON manifest containing schema version, Blueprint workflow
and run IDs, file names/media types/byte counts/SHA256, source commit, object
generations and provider/session provenance. Save ordinary JSON/JSONL files
optionally gzip compressed, create-only (`ifGenerationMatch: 0`), with existing
private bucket/object access and authenticated exact-object hash readback.
Same-name retries read and compare, never overwrite. The manifest references
Blueprint IDs; no Library ID or provider session ID is a canonical identity or
required consumer lookup. Retain local/Library copies as secondary copies only.

## Fresh Perplexity canary

The private one-time adapter reuses the reviewed Store, leases, Runner, Consumer
and publisher, mapping only to
`blueprintDailyResearch/sites-first/canaries/perplexity-fast-20261001`.
It must leave normal control, the failed Oct1 run and Oct2 slot unchanged.
Stable test identity is `blueprint-research-canary:perplexity-fast-20261001`;
provider session, turn and environment IDs are provenance. Date-scoped Store
internals retain the reviewed `blueprint-researcher:2026-10-01` key. The test
input explicitly labels the brief as a test and prioritizes publicly verified
named decision contacts without expanding the strict output schema.

Required admission: exact existing $25 one-time approval reference, freshly
checked package/agent/template/tool/instruction/skills bindings, complete CRM,
checked knowledge/refresh policy, reconciled original cleanup receipt, canonical
publication authorities, normal root/config disabled. Test uses the production
$5 soft target and a separate one-time $25 soft total testing allowance; the
legacy `ceiling_usd` admission field records that allowance, not a hard provider
billing cap. It never replaces recurring authority or creates a timer.

Research deadline is 1200 seconds and total research+QA 1800 seconds, with 600
seconds reserved for QA. Existing independent Linux watchdog convention is
`timeout --signal=TERM --kill-after=60s 1860s`. Process stop/cancellation does not
guarantee provider teardown or a total-dollar cap. Model usage is observed for
baseline measurement; an arbitrary $8 stop and cancellation for absent usage
are removed. Selected tools retain existing resource ceilings, and actual
billing still needs reconciliation.
No fixed prospect quota; use defined scope, coverage and diminishing returns.

One durable create claim only, persisted full immutable intent first; an unknown
POST reconciles by GET and is never resent. One bounded QA turn in that session;
publication only after terminal evidence-backed agent QA and dedup. Recovery
must verify exact terminal turns, downloaded immutable artifacts, canonical
Sheets/Notion receipts and no duplicate publication. No Dot review/publication
dependency, outreach or permanent deletion. Cleanup needs its own exact
action-time approval; $25 test approval does not authorize deletion.

Use the exact reviewed GitHub commit containing `research-perplexity-canary.py`,
its sibling `.mjs`, and the unchanged `research-oct2-control.py`. Materialize all
three ordinary source files into `/tmp/blueprint-oct2-reviewed` through the
existing authenticated operator route and compare their reviewed SHA256 before
execution. Do not install the Pipeline application or deploy to a Pipeline host.
The adapter verifies the exact installed package receipt before every command.
Keep the production worker settings and normal research control disabled.

Create a private approval JSON file from the **existing** one-time $25 receipt;
use its exact evidence reference, never invent one or reuse recurring/deletion
authority. Required fields (the example reference must be replaced by the
actual already-approved receipt) are:

```json
{
  "schema_version": "blueprint.perplexity-canary-admission.v1",
  "test_id": "perplexity-fast-20261001",
  "authority_reference": "PENDING-actual-existing-one-time-25-receipt",
  "ceiling_usd": 25,
  "scope": "one-time-fresh-research-agent-qa-canonical-publication-no-outreach"
}
```

The fixed admission expires at `2026-10-02T10:00:00Z`; starting a new test after
that time refuses. Existing uncertain intent remains reconcilable and never
allows another create. The code does not change a template, account connection,
credentials, network policy, or production scheduler.

From `/opt/render/project/src`, run the complete read-only admission first:

```sh
PYTHONPATH=/opt/render/project/src/dist/daily-research/release \
  /opt/render/project/src/dist/daily-research/venv/bin/python \
  /tmp/blueprint-oct2-reviewed/research-perplexity-canary.py inspect \
  --package /opt/render/project/src/dist/daily-research/release \
  --archive /opt/render/project/src/vendor/daily-research/blueprint-research.tar \
  --approval /tmp/blueprint-oct2-reviewed/canary-approval.json \
  --output /tmp/blueprint-oct2-reviewed/canary-plan.json
```

`inspect` reads the complete canonical CRM, checks its digest and identity keys,
validates current knowledge/refresh policy and saved-agent/template/instruction/
four inline skill-file bindings using the pinned package's real preflight.
It verifies original cleanup and migrated controls. It makes no Firestore
writes or paid provider calls; only the new private plan file is written.
Read access alone does not prove publication writes or final report quality.
The authenticated runtime owner must also retain current exact canonical
Sheets Editor and Notion parent/integration permission evidence before execution.

Stage the same unedited plan (Firestore writes to the private canary namespace,
no provider inference or changes to normal control):

```sh
PYTHONPATH=/opt/render/project/src/dist/daily-research/release \
  /opt/render/project/src/dist/daily-research/venv/bin/python \
  /tmp/blueprint-oct2-reviewed/research-perplexity-canary.py stage \
  --package /opt/render/project/src/dist/daily-research/release \
  --archive /opt/render/project/src/vendor/daily-research/blueprint-research.tar \
  --plan /tmp/blueprint-oct2-reviewed/canary-plan.json
```

Under the existing one-time test authorization, execute once with the exact
independent watchdog. The direct parent must be `timeout` with these arguments;
an unbounded invocation refuses before provider creation:

```sh
PYTHONPATH=/opt/render/project/src/dist/daily-research/release \
  timeout --signal=TERM --kill-after=60s 1860s \
  /opt/render/project/src/dist/daily-research/venv/bin/python \
  /tmp/blueprint-oct2-reviewed/research-perplexity-canary.py execute \
  --package /opt/render/project/src/dist/daily-research/release \
  --archive /opt/render/project/src/vendor/daily-research/blueprint-research.tar
```

For lost replies, interruption or restart, use the same bounded command with
`reconcile` in place of `execute`. It never creates a new session; it may finish
the already-authorized research tool responses, one QA turn, canonical
publication and authenticated readback. This recovery command is not read-only.
Use `status` with the same package/archive arguments for read-only sanitized
state, exact root/QA turn IDs and terminal states, artifact SHA256, cleanup
guard and destination readback receipts. Do not treat an idle session or process
exit as successful research.

For a new private standard-file export use `export` with
`--output /tmp/blueprint-oct2-reviewed/canary-export`. It is read-only for
Firestore and writes an exclusive private directory. It reuses the package's
artifact/evidence/QA/tool binding verification. Require complete terminal root
and QA artifacts, acknowledged canonical Sheets and Notion readbacks, and
reported missing-file review before calling the test successful.

Archive that export with a standard hash manifest to the authorized existing
company storage destination, read back the exact generation and hashes, then
obtain separate action-time approval for the exact test session's permanent
deletion. Record cleanup only after authenticated session and environment GETs
prove absence: use `record-cleanup` with the same package/archive arguments and
`--receipt /tmp/blueprint-oct2-reviewed/canary-cleanup-receipt.json`. The receipt
must name the exact test `session_id`, `environment_id`, and separate actual
`action_time_approval_reference`; retain stable Blueprint ID and private backup
manifest/URI/generation/hash evidence alongside it. This command reuses
`Runner.record_cleanup`, makes authenticated absence GETs, verifies retained
artifact/QA/evidence bindings, then writes only the private test cleanup record.
It never deletes a provider resource. The normal `render record-cleanup` CLI
addresses normal history and must not be used for this isolated test.

Before daily cutover, verify this private canary's cleanup guard
as well as normal dated history; normal production Store does not query the
private test namespace. Verify Oct2 remains unopened, the old Dot trigger stays
disabled, and only the standalone 07:00 America/Chicago trigger is enabled.
No canonical publication through Dot or ChatGPT Library is required.

## Baseline measurement and pending usage

The public [Agents API usage contract](https://developers.openai.com/api/docs/guides/agents-api/observability)
is best-effort: turn/session usage can be null, recorded counts may change, and
counts are not a final bill. Missing counts are neither zero nor evidence of
overspend. The measurement adapter records `usage_state=pending` and
`estimate_usd=null`; if some turns have counts, `reported_estimate_usd` describes
only that reported subset. A failed telemetry GET is explicitly `unavailable`.
Reported counts are labeled `reported_best_effort` and retain exclusions for
unreported usage, tools and hosted compute. Root and QA work may proceed under
the already-authorized soft-total scope while counts are pending; exact current
status, immutable phase deadlines, independent watchdog, one-create/one-QA
claims and resource ceilings continue to apply. No prospect-count stopping
rule or billing cap is introduced. Terminal reconciliation refreshes accounting
by GET without repeating research, QA or publication.

The first accepted test was cancelled by the historical missing-usage guard.
Its create has been consumed. This code correction preserves cancellation and
never clears an intent, resets its start/deadline, resubmits a create, or opens
another test identity. The runtime owner must reconcile that exact session's
terminal status, available usage and tools/hosted charges, retain portable
evidence, and complete separately approved cleanup. New paid testing waits for
the owner's total testing allowance and recovery authority; unknown charges
must remain explicit in that reconciliation. This fix is preparation, not a
new paid-session or permanent-deletion authorization. Production remains
disabled until real coverage/QA/publication, cleanup and cutover are verified.
