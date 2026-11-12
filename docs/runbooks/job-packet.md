# Making a job packet by hand (V1)

The job packet is what a deployment company gets to evaluate one job without a
visit (owner direction 2026-11-12). `scripts/export_job_packet.py` writes it
offline from files you pulled; it sends nothing. Every approach to an outside
company needs the owner's go-ahead first.

## 1. Pull the job's files

Pull into one root laid out as the host's site roots
(`/var/lib/blueprint/pubsub-handoffs`, the units'
`BLUEPRINT_WEBSITE_WITHDRAWAL_SITE_ROOTS`): `<root>[/<bucket>]/scenes/site-<request>/`.
Pull every bucket's `scenes/site-<request>/website_withdrawal` (a refused pull:
none exists). The tool checks withdrawal across the whole root as the
Pipeline does and refuses when the layout is not that.

```bash
D=python3\ scripts/operator_door.py
S=<site root>[/<bucket>]/scenes/site-<request>
$D pull $S/captures/<capture id> ./pulled[/<bucket>]/scenes/site-<request>/captures/<capture id>
# only when they exist (full level):
$D pull /var/lib/blueprint/pipeline-control-plane/team-scene-packets/<packet> ./team-scene
$D pull /var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs/<launch>/configured_scene_publication ./configured
$D pull /var/lib/blueprint/pipeline-control-plane/beta-policy-screening/publications/<campaign>.json ./publication.json
```

The capture directory must hold `pipeline/website_task_context.json` and
`pipeline_job_ledger.json`. A run in cold storage must be restored first.

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
