# Exporting a job packet (offline)

The job packet is what an approved deployment company reads to judge one job
without a visit (owner direction 2026-11-12). A job is shared in full: the
site's sentence and constraints, the video without sound, the first look on
paper, the 3D reconstruction, the simulation and the success rule, and the
early trial. Sharing is the service: a job whose site accepted recording
statement `2026-11-12.v4` or later goes to the deployment companies Blueprint
has approved, and a job under an older statement is not shown to any company
until its site accepts the current one. A site's name, street address and
contact details reach a company only through an introduction the site asked
for on its job page (one press, at most three per job).

`scripts/export_job_packet.py` writes the packet offline from files you
pulled. It sends nothing, and it is not in the automatic path, where a listed
job is shown to every approved company on the website. Every approach to an
outside company needs the owner's go-ahead first.

## 1. Pull the job's files, one by one

The door stops a folder pull at 512 MiB without an error, so a whole-capture
pull loses files. Pull each file below on its own. When the door says a file
is missing, put its path in `absent_on_host` (step 2). The tool refuses, naming
the file, when a file it reads is neither pulled nor recorded absent.

Pull into one root laid out as the host's site roots:
`<root>[/<bucket>]/scenes/site-<request>/captures/<capture>/`.

```bash
D=python3\ scripts/operator_door.py; H=/var/lib/blueprint/pubsub-handoffs; P=/var/lib/blueprint/pipeline-control-plane
$D ls $H        # the buckets: each folder beside scenes/ whose name does not start with a dot
```

A hidden folder (a name that starts with a dot) is never a bucket. Do not
pull it or list it. Folders such as `_pipeline_failures` are buckets.

For the root and each bucket `b`, the withdrawal folder:
`[b/]scenes/site-<request>/website_withdrawal`.

From the capture `<b>/scenes/site-<request>/captures/<capture>/`, each of:

- `pipeline/website_task_context.json`
- `pipeline_job_ledger.json`
- `pipeline_job_terminal_receipt.json`
- `raw/rights_consent.json`, `rights_consent.json`, `raw/manifest.json`, `capture_descriptor.json`
- `pipeline/consent_takedown` (a folder; check the door's reply says `truncated: false`)
- `pipeline/website_scene_preparation/handoff.json`, `preparation.json`, `base_scene.json`
- `pipeline/website_scene_preparation/revisions` (a folder, when it exists)
- the robot fit table the handoff names (`robot_fit_table.path`, normally
  `pipeline/website_scene_preparation/native/robot_fit_table.json`)
- the video `raw/manifest.json` names in `video_uri` (normally `raw/walkthrough.mp4`)
- the two reconstruction files `base_scene.json` names in `splat_path` and `collision_mesh_path`

When they exist, also pull, each to its own folder: the team scene packet
(`$P/team-scene-packets/<packet>`), the launch run's
`configured_scene_publication`, the trial's
`$P/beta-policy-screening/publications/<campaign>.json` and its
`pipeline/beta_screening_review_videos/<attempt>/`. Check `truncated: false`
on every folder pull. Restore a cold run first.

## 2. Write the pull record

Write `./pulled/pull_record.json`:

```json
{"schema_version": "job_packet_pull_record.v1",
 "buckets": ["", "<each bucket from ls>"],
 "absent_on_host": ["<each place above the door said is missing, relative to ./pulled>"]}
```

The record is your own statement of what you pulled. Marking a place absent
without having looked is how a withdrawn job could get a packet. It is kept,
whole, in the operator record.

## 3. Run it

```bash
python3 scripts/export_job_packet.py --pull-root ./pulled --capture-dir <the capture> \
  --operator <you> --output-root ./out [--screening-publication ./publication.json] \
  [--review-videos-dir <clips>] [--team-scene-dir <packet>] [--configured-scene-dir <publication>]
```

The machine needs `ffmpeg` and `ffprobe`: the video goes in without its sound
or its recorded details such as place and time (the video stream is copied as
it is), and the tool refuses rather than copy a video that keeps either. `config/beta_policy_candidates.v1.json` must not be group-writable
(`chmod 644`).

A packet is made only when one of these holds:

- the site accepted the current statement: `capture_rights.statement_version`
  in the task context is `2026-11-12.v4`. Under any older version the tool
  refuses (`job_packet_sharing_statement_not_accepted`) until the site accepts
  the current one.
- the capture is Blueprint's own: `--blueprint-owned-capture --capture-id <id>
  --approved-by <owner> --approved-on <date> --approval-evidence "<the owner's
  words>"`, and the id is in `config/blueprint_owned_captures.v1.json`
  (read by `sharing_statement.owned_captures`, the same reader the company
  video uses; a list it cannot read refuses with
  `job_packet_owned_capture_list_invalid`). Every row must name its video by
  `video_sha256_prefix` (12 to 64 hex digits); a row without one makes the
  list unreadable.

The first look lists robots that fit, then those that fit if conditions hold,
then those that do not fit, then unknown. Each condition is written in the
Pipeline's own fixed sentence with the table's numbers: what the robot can do
beside what the job seems to need. A row whose condition the tool does not
know is left out and counted, never printed as it stands.

Withdrawn, unconsented, authority-ended and owner-closed jobs are refused.
Every refusal names the file or place it is about. A packet is never held for
the simulation, the success rule or the early trial: the status line says what
is not in it yet.

The simulation is in the packet only as one composed stage,
`files/simulation/scene.usda`. Open that file: it places the room, its hidden
collision geometry and the job's object where Blueprint's simulation puts them,
in metres, with the up axis the scene plan states. Each part's file is
referenced on a child prim (`/World/<part>/Asset`), so its own transform is
kept under the plan's pose. The plan itself, a record, is never copied. When a
plan row cannot be placed (a missing pose, a file that was not copied), there is
no `scene.usda`, and the sheet says the simulation is not finished. The success
rule (`files/task/success_rule.json`) passes through the same allow-list as the
company download: numbers and fixed mode words only, so no label the site typed
is in it.

## 4. Before it leaves

Read the whole sheet, `JOB_SHEET.md`, before the packet leaves. The tool
removes email addresses, phone numbers and web links from the site's own words,
and writes every simulation file itself instead of copying it: in
`files/simulation/` and `files/object/` each USD layer is written by `Sdf`,
each usdz is rebuilt from its members, and each PNG or JPEG is rewritten without
its text, EXIF, ICC and comment data (`outbound_file_writer`; pixels unchanged).
Any other kind of file there refuses the packet. Then the shared scan
(`outbound_file_scan`; its rules are a table in the module's docstring) reads
the whole packet as the proof. Kinds are decided by content, not by the file's
ending. Parsed text (a USD layer's text form, image metadata, text files, a
PLY's header, a GLB's JSON) is read for ids, keys, paths, links, email and IP
addresses and the site's identity; the raw bytes of binary files (the video,
clips, the PLY and GLB bodies) for the job's long ids and keys, long-prefix
tokens, links and long host paths. The one limit: numeric payload (pixels,
coding tables, palette and transparency tables, numeric arrays) is not searched
as text. Textures lose their colour
profile and their EXIF, so a photo that relied on an EXIF rotation shows
unrotated. The site's identity is its name, organisation, typed
address, contact and submitter names, email and phone. It is matched as whole
words, ignoring case and accents; a value of two or more words is also matched
with separators removed (`Zebra Plant Foods` as `ZebraPlant_Foods`). Single
words under five letters, common scene words and address parts that are only a
unit, a number or a country are not matched. The site's description of its job
is not looked for: the site agreed to share it, and the sheet quotes it on
purpose. So a name or a street address the site typed into its description is
still there: take it out by hand.
The packet holds no site name, address, contact, token, link or storage path
from the records; an introduction is made only when the site asks for one on
its job page.
Beside the folder, `packet-<number>.operator-record.json` holds the job's ids,
takedown keys, the permission basis, input digests, the pull record and who ran
it. Keep it; never send it. Record recipient, date and packet number in the
website's register (`recipients` stays empty). On a withdrawal, find packets by
the record's takedown keys; ask each recipient to delete.
