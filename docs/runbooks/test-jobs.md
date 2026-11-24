# Internal test jobs (real host, real money)

A test job is a whole real job, started from Blueprint's own test footage instead of a site's upload. It runs
through every real stage and every paid admission a site's job does. It is never shown to a deployment company
and nothing about it is sent to anyone. For a $0 check of the code paths, use `test-job-matrix.md` instead.

The existing monthly limit is the brake; no cap, limit or price is raised for test jobs.

## Footage

Test footage lives in the website's storage bucket, one folder per clip:

```
internal-test-footage/<footage_id>/walkthrough.mp4   (or .mov, exactly one of the two)
internal-test-footage/<footage_id>/source.json
```

`footage_id` is lower-case letters, digits and hyphens (at most 64). A clip is added only on the owner's word for
it, by the owner or by a session the owner told to add it. Any video may be used for internal tests (owner,
2026-11-15). Keep `source.json` beside it anyway so the next reader knows where it came from:

```json
{"source_page": "https://...", "licence_text": "the licence line exactly as the page states it",
 "sha256": "<hex of the video file>", "added_by": "<name>", "added_at": "YYYY-MM-DD"}
```

## Run one job

```bash
python3 scripts/operator_door.py test-job --idempotency-key 20261201:library-aisle --label library-aisle \
  --footage-id pexels-854533 --description "Move the blue book from the cart to the top shelf." --wait
```

The door answers with the job id, its capture id and the label. The same key always gives the same job; the same
key with another label, footage or description is refused.

## Run a batch

Write a matrix file:

```json
{"jobs": [{"label": "library-aisle", "footage_id": "pexels-854533", "description": "Move the blue book ..."},
          {"label": "kitchen-push", "footage_id": "pexels-3773483", "description": "Push the tray ..."}]}
```

```bash
. ~/.blueprint-cloud/env.sh   # the Pipeline's Python: the report reads intents with its digest library
"$BLUEPRINT_PIPELINE_PYTHON" scripts/test_jobs.py submit --matrix jobs.json -k 2 --state run.json
"$BLUEPRINT_PIPELINE_PYTHON" scripts/test_jobs.py follow --state run.json --report report.json
"$BLUEPRINT_PIPELINE_PYTHON" scripts/test_jobs.py report --state run.json
```

Under another Python it says so in one line and exits 2 before doing anything.

`submit` starts the rows in order, never more than two at a time. Before each one it reads `door status` and
stops when the month's counted spend (spend plus held reservations) plus one job's expected spend ($20) would
reach the stop line; the door checks the same against the host's own spend report and refuses
`test_job_month_stop_line`, or `test_job_month_closed:<reason>` when paid admission is closed. The stop line is one
of the account's existing alert lines, `STOP_LINE` in `src/blueprint_pipeline/internal_test_job_door.py`: `"100"`
is the monthly limit itself. `submit` also stops at any refusal or missing answer rather than going on. It waits through the hour
of the daily deploy window (15:00 UTC). `run.json` remembers what was started, so running `submit` again skips
rows already started. The key is the run id and the label with spaces made hyphens, so the labels "a b" and
"a-b" in one run share a key; give them different labels.

## Two at a time

The host has 8 GB of memory and no swap. The door refuses a third test job while two are in flight
(`test_job_two_in_flight`). A job stops counting when the website refused it (nothing was made), when its zip
has been delivered, when its preparation ended for good (`pipeline_job_ledger.json` `terminal_*` other than
`terminal_success`), when its scene finished or ended for good (the scene intent's `progression.json` reads
`completed`, or its `outcome.json` is terminal), when its preparation ended on a hold that waits outside the
pipeline (the ledger reads `completed` and the handoff is one the preparation status reads as `awaiting_inputs`,
`website_preparation_status.held_preparation`: a stop only a new video clears, a target correction, a place or
owner question, no robot fitting; nobody answers a test job), when storage retention retired its workspace (the
capture folder is gone and the scene's `<bucket>/scenes/<scene>.retired.v1.json` receipt verifies as retention
writes it and records the capture's ledger as ended, with its proof:
`website_scene_workspace_retention.retired_capture_ended`; a file by that name that does not verify frees no
slot, and a capture a new delivery stages again counts from its live folder), or 7 days after it was asked for. A preparation
waiting on Blueprint or the provider (the 3D reading, an Atlas job, an armed retry) keeps its slot. An ask whose
answer was lost keeps its slot until it is asked again with the same key.

The door reads this afresh at each ask. A held capture that is later reopened counts again from the moment its
ledger leaves `completed`, but the door only refuses new asks: it cannot stop a reopen. So the worst case is two
running plus every held test job reopened at the same moment. Some holds reopen with no release at all: a
robot-fit hold when the team registry or the fit rule changes (`website_robot_fit_hold`), a navigation hold when
its intake receipt is bound (`website_held_handoff_replay`), and a job whose brief is confirmed later. A release
re-evaluates every hold that names a missing `pipeline:*` piece. The host has 8 GB of memory and no swap, so do not leave many held test jobs lying around:
retire them (an owner-approved retirement) or let them age out, and check `report` for `held:` lines before
starting a batch.

## What it costs

Up to about $20 a job. The one complete run so far (2026-11-16, a rigid job) settled $6.04 at World Labs and
about $2 of object authoring. Every website job's scene, a door or drawer job's too, is built on the host with no
rented GPU (stage 5 records that no simulator ran, `website_native_background.select_stage_five`). The account
limit counts test jobs like any other job. It is $400 for November 2026 and $300 again from December.

## Reading the report

One line per job: label, the job kind the Atlas review returned, the stage reached (a preparation's stage is
read from its own `pipeline/website_scene_preparation/handoff.json` when there is one, so the 3D reading's wait
shows as `preparation:awaiting_reconstruction`, not the ledger's `failed_retryable`), settled spend ("spend not
shown" when `door status`, which lists only its top jobs, does not name the job), the first failure code
(including a package the website refused, `held:<code>` for a preparation that ended on a hold, naming what the
site would have been asked, and "stalled" for a job with no package and no ending after 7 days) and the start of
the zip digest. The JSON adds the footage id, job and capture ids, times (asked
for, job ledger last changed, package built), spend by provider (settled and held, from `door status`, matched by
job or capture id), the handoff's status and blocker codes (`preparation`), the package's file names
(`package_files`) and its `permission` (`internal_test`, with the label). The package manifest does not record
the README's note codes, so the report does not show them.

A test job is found again by its label in: the website's request (`internal_test.label`), the capture's task
context (`pipeline/website_task_context.json`), the clip and package manifests (`permission.label`), the
permission check (`pipeline/company_clip/statement_check.json`) and the door's ledger
(`/var/lib/blueprint/pipeline-control-plane/internal-test-jobs/`). Spend rows carry the job id, not the label.
Nothing here deletes anything; retiring a test job's files is an owner-approved retirement like any other.

## When the monthly limit refuses

`submit` and the door refuse and say why; nothing is queued. A job already running keeps its own reservations and its next
paid step is refused at admission with `account_monthly_spend_limit_reached`, exactly as a site's job would be.
Do not raise any limit for a test. Wait for the next month, or ask the owner.
