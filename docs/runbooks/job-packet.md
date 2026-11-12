# Making a job packet by hand (V1)

The job packet is what a deployment company gets to evaluate one job without a
visit (owner direction 2026-11-12). `scripts/export_job_packet.py` writes it
offline from files you pulled; it sends nothing. Every approach to an outside
company needs the owner's go-ahead first.

## 1. Pull the job's files

Keep the host layout for the site directory: the tool looks for the withdrawal
tombstone at `scenes/site-<request>/website_withdrawal/tombstone.json` and
refuses when it cannot (`job_packet_withdrawal_state_unknown`). Site roots are
`/var/lib/blueprint/pubsub-handoffs[/<bucket>]/scenes/` (the units'
`BLUEPRINT_WEBSITE_WITHDRAWAL_SITE_ROOTS`).

```bash
D=python3\ scripts/operator_door.py
S=<site root>/scenes/site-<request>
$D pull $S/website_withdrawal ./job/scenes/site-<request>/website_withdrawal   # a refused pull: none exists
$D pull $S/captures/<capture id> ./job/scenes/site-<request>/captures/<capture id>
# only when they exist:
$D pull /var/lib/blueprint/pipeline-control-plane/team-scene-packets/<packet> ./team-scene
$D pull /var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs/<launch>/configured_scene_publication ./configured
$D pull /var/lib/blueprint/pipeline-control-plane/beta-policy-screening/publications/<campaign>.json ./publication.json
```

The capture directory must hold `pipeline/website_task_context.json` and
`pipeline_job_ledger.json`. A run that went to cold storage must be restored
first. The review clips are under the capture's
`pipeline/beta_screening_review_videos/<attempt>/`.

## 2. Run it

```bash
python3 scripts/export_job_packet.py --capture-dir ./job/scenes/site-<request>/captures/<capture id> \
  --output-root ./out [--team-scene-dir ./team-scene] [--configured-scene-dir ./configured] \
  [--screening-publication ./publication.json] [--review-videos-dir <clips dir>] \
  [--first-look-file first-look.md --first-look-checked-by <you> --redact-term "<site name>"]
```

`--level summary` (the default) is all that may be shared today: the kind of
job, the first look and the early trial. `--level full` (sentence, video,
simulation, success rule, clips) is refused until the site's recording
statement covers sharing (`SHARING_PERMISSION_STATEMENT_VERSIONS`, empty), or,
for Blueprint's own capture, with `--blueprint-owned-capture --capture-id <id>
--approved-by <owner> --approval-evidence "<the owner's words>"` and the id in
`config/blueprint_owned_captures.v1.json`. Withdrawn, authority-ended and
owner-closed jobs are refused at both levels.

## 3. Before it leaves

Read `JOB_SHEET.md` whole. Record the recipient, date and `packet_id` from
`manifest.json` in the register on the website side; `recipients` in the
manifest stays empty. On a withdrawal, match `takedown_keys` and ask each
recipient to delete the packet.
