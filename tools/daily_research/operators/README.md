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

## Fresh Perplexity canary under preparation

The private one-time adapter is being tested and independently reviewed. It is
not yet an approved execution command. It reuses the reviewed Store, leases,
Runner, Consumer and publisher, mapping only to
`blueprintDailyResearch/sites-first/canaries/perplexity-fast-20261001`.
It must leave normal control, the failed Oct1 run and Oct2 slot unchanged.

Required admission: exact existing $25 one-time approval reference, freshly
checked package/agent/template/tool/instruction/skills bindings, complete CRM,
checked knowledge/refresh policy, reconciled original cleanup receipt, canonical
publication authorities, normal root/config disabled. Test uses the production
$5 soft target and a separate one-time $25 authorization ceiling; the latter
never replaces recurring authority or creates a timer.

Research deadline is 1200 seconds and total research+QA 1800 seconds, with 600
seconds reserved for QA. Existing independent Linux watchdog convention is
`timeout --signal=TERM --kill-after=60s 1860s`. Process stop/cancellation does not
guarantee provider teardown or a total-dollar cap. Installed conservative model
estimate stop threshold is $8, unknown usage stops paid work, selected tools have
their existing resource ceilings, and actual billing still needs reconciliation.
No fixed prospect quota; use defined scope, coverage and diminishing returns.

One durable create claim only, persisted full immutable intent first; an unknown
POST reconciles by GET and is never resent. One bounded QA turn in that session;
publication only after terminal evidence-backed agent QA and dedup. Recovery
must verify exact terminal turns, downloaded immutable artifacts, canonical
Sheets/Notion receipts and no duplicate publication. No Dot review/publication
dependency, outreach or permanent deletion. Cleanup needs its own exact
action-time approval; $25 test approval does not authorize deletion.
