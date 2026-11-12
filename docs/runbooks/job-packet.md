# Making a job packet by hand (V1)

The job packet is what a deployment company gets to evaluate one job without a
visit (owner direction 2026-11-12). `scripts/export_job_packet.py` writes it
offline from files you pulled; it sends nothing. Every approach to an outside
company needs the owner's go-ahead first.

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
the door said is missing]}`. A place neither pulled nor listed absent is refused,
naming it. A run in cold storage must be restored first. Full level also needs
`$P/team-scene-packets/<packet>` and the launch run's `configured_scene_publication`.

## 2. Run it

```bash
python3 scripts/export_job_packet.py --pull-root ./pulled --capture-dir <the capture> \
  --operator <you> --output-root ./out [--screening-publication ./publication.json]
```

`--level summary` (the default) is all that may be shared today. It takes no
free text from any input: names come from the Pipeline's catalogs, verdicts and
statuses are fixed words, and rows that do not match are left out and counted.
Its folder (`packet-<random>`) and manifest carry no site, request or capture id.

`--level full` adds the sentence, the video, the simulation, the success rule,
the clips and an optional `--first-look-file`. It is refused until the site's
recording statement covers sharing (`SHARING_PERMISSION_STATEMENT_VERSIONS`,
empty), or, for Blueprint's own capture, with `--blueprint-owned-capture
--capture-id <id> --approved-by <owner> --approved-on <date> --approval-evidence
"<the owner's words>"` and the id in `config/blueprint_owned_captures.v1.json`.
Every input must belong to this job. The full manifest keeps the job's ids,
because the company is being given the job. Withdrawn, unconsented,
authority-ended and owner-closed jobs are refused at both levels.

## 3. Before it leaves

Read `JOB_SHEET.md` whole. Beside the folder is
`packet-<random>.operator-record.json`: the job's ids, takedown keys, the
permission basis, input digests and who ran it. Keep it; never send it.
Record recipient, date and `packet_id` in the register on the website side;
`recipients` in the manifest stays empty. On a withdrawal, find the packets by
the record's takedown keys and ask each recipient to delete them.
