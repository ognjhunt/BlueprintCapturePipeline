# Exporting a job packet (offline)

The job packet is what a deployment company gets to evaluate one job without a
visit (owner direction 2026-11-12). `scripts/export_job_packet.py` writes it
offline from files you pulled; it sends nothing. Sharing is the service: a job
whose site accepted recording statement `2026-11-12.v4` or later goes to the
deployment companies Blueprint has approved, and a job under an older statement
is not shown to any company until its site accepts the current one. A site's
name, street address and contact details are never shared without asking the
site first. The site asks for an introduction with one press, at most three per job; Blueprint then gives each side the other's name and contact details by email. Nothing else shares them. This tool is an offline export; it is not in the
automatic path, where a ready job is shown to every approved company. A packet is not held
for the simulation: it goes when the video and the first look exist, and says
plainly what is not finished.

## 1. Pull the job's files

Pull into one root laid out as the host's site roots (`/var/lib/blueprint/pubsub-handoffs`,
`BLUEPRINT_WEBSITE_WITHDRAWAL_SITE_ROOTS`): `<root>[/<bucket>]/scenes/site-<request>/`.

```bash
D=python3\ scripts/operator_door.py; H=/var/lib/blueprint/pubsub-handoffs; P=/var/lib/blueprint/pipeline-control-plane
$D ls $H                          # the buckets: every directory beside scenes/ ("" is the root itself)
# For EACH bucket b (and the root): the withdrawal state. A refused pull means none on the host.
$D pull $H/[b/]scenes/site-<request>/website_withdrawal ./pulled/[b/]scenes/site-<request>/website_withdrawal
$D pull $H/<b>/scenes/site-<request>/captures/<capture id> ./pulled/<b>/scenes/site-<request>/captures/<capture id>
# (the capture brings raw/manifest.json, pipeline_job_ledger.json and pipeline/consent_takedown when present)
$D pull $P/beta-policy-screening/publications/<campaign>.json ./publication.json      # when a trial exists
```

Then write `./pulled/pull_record.json`: `{"schema_version": "job_packet_pull_record.v1",
"buckets": [every bucket from ls, "" for the root], "absent_on_host": [each
`[b/]scenes/site-<request>/website_withdrawal` and `<capture path>/pipeline/consent_takedown`
the door said is missing]}`. A place neither pulled nor listed absent is refused.
The record is your own statement of what you pulled: marking a place "absent"
without having looked is how a withdrawn job could get a packet. It is kept,
whole, in the operator record. Restore a cold run first; full level also needs
`$P/team-scene-packets/<packet>` and the launch run's `configured_scene_publication`.

## 2. Run it

```bash
python3 scripts/export_job_packet.py --pull-root ./pulled --capture-dir <the capture> \
  --operator <you> --output-root ./out [--screening-publication ./publication.json]
```

`config/beta_policy_candidates.v1.json` must not be group-writable (`chmod 644`), or every summary
with a trial is refused (`job_packet_policy_catalog_unreadable`). `--level summary` is the default. It takes no
free text from any input: names come from the Pipeline's catalogs, verdicts and
statuses are fixed words, and rows that do not match are left out and counted.
Its folder (`packet-<random>`) and manifest carry no site, request or capture id.

`--level full` adds the sentence, the video, the simulation, the success rule,
the clips and an optional `--first-look-file`. It is refused until the site's
recording statement covers sharing (`SHARING_PERMISSION_STATEMENT_VERSIONS`,
empty today; adding `2026-11-12.v4` is in build, and until it ships only
`--level summary` can be made for a site's job), or, for Blueprint's own capture, with `--blueprint-owned-capture
--capture-id <id> --approved-by <owner> --approved-on <date> --approval-evidence
"<the owner's words>"` and the id in `config/blueprint_owned_captures.v1.json`.
Every input must belong to this job; the full manifest keeps the job's ids. Withdrawn,
unconsented, authority-ended and owner-closed jobs are refused at both levels.

## 3. Before it leaves

Read `JOB_SHEET.md` whole. Beside the folder, `packet-<random>.operator-record.json` holds the job's ids, takedown keys, the
permission basis, input digests, the pull record and who ran it. Keep it; never send it.
Record recipient, date and `packet_id` in the website's register (`recipients` stays
empty). On a withdrawal, find packets by the record's takedown keys; ask each recipient to delete.
