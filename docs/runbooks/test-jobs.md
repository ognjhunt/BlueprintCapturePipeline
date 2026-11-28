# Internal test jobs (real host, real money)

A test job is a whole real job, started from Blueprint's own test footage instead of a site's upload. It runs
through every real stage and every paid admission a site's job does. It is never shown to a deployment company
and nothing about it is sent to anyone. For a $0 check of the code paths, use `test-job-matrix.md` instead.
To find where a running test job stops next against your branch, replay it offline: `job-replay.md`.

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

**The simulation.** Since 2026-11-28 a site's job builds its simulation only once an approved deployment company
asks for it (`website_simulation_request`). A test job asks for its own by default: the website records an
`internal_test` request when it makes the job and pushes it to the Pipeline, so the full path runs as before.
To test the wait itself, add `--simulation on_request`: the job prepares its sound-free clip and nothing else,
its preparation completes waiting (status `awaiting_inputs`, reason `awaiting_simulation_request`) and costs $0
on the host, and its slot is free while it waits. Ask for its simulation as a company would: the internal test
company's "Ask for a simulation" on the website's job page, which sends the same signed request a company's press
does. The job then takes a slot again until it ends. A waiting test job's workspace is kept while its grant is live
(at most seven days from upload), then retired like any finished test job. The option is sent only when given, so a
door installed before it still takes every other test job; the option itself needs the installed door upgraded
(`install.sh --upgrade`, `docs/OPERATOR_DOOR.md`), and the website must know the field.

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

`docs/test-footage/clip-mix-v1.json` is a matrix too: the job types the $0 matrix cannot model (rotation, 4K, a
long clip, low light, people in shot, a large room) and the ones no real job has run, in the order to run them,
each with its source page and licence line. `submit --matrix docs/test-footage/clip-mix-v1.json` skips a row marked
`"footage_uploaded": false` or with no `footage_id`, and a row the website finds no clip for
(`internal_test_footage_not_found`: nothing was made) is named and skipped; any other refusal still stops the run.
The file says how to add a row's clip.

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

## Run it again now, after a fix

A job that failed a step waits out its own retry schedule, even once the fix it needed is live: the Atlas preview's
refusal schedule (15 min, 1 h, 3 h, 6 h, 12 h, 24 h, `job_retry_policy.REFUSAL_RETRY_DELAYS_SECONDS`) and the
preparation retry's (`website_preparation_retry`). A release reopens a job parked at `needs_input`; it does not move a
job that is processing with a retry scheduled. To bring that retry forward:

```bash
python3 scripts/operator_door.py test-job-retry-now itest-<40 hex> --reason "the fix for X is live" --wait
"$BLUEPRINT_PIPELINE_PYTHON" scripts/test_jobs.py retry-now --state run.json --label <label> --reason "..."
```

It changes only when the job's next attempt is due. Under the lock its own tick or listener takes, it sets the Atlas
preview job's `next_attempt_at_epoch`, and the preparation retry record's `next_attempt_at` (the ledger's next
revision, as the listener commits its own retry's time), to now. Nothing else changes: the refusal, retry and reopen
counts stay as they are and still count the retry that runs, no wall, grant or limit moves, and every paid call still
goes through its own admission. The next tick or listener pass runs the step as it would have when its time came.

- Only internal test jobs: the id is the website's `itest-` id, the door's test-job ledger
  (`/var/lib/blueprint/pipeline-control-plane/internal-test-jobs/`) has its submitted row, and the door's result of
  the `test-job` request that row names says it made this job. Anything else is refused.
- A withdrawn capture is refused. A record whose last stop reads as privacy, rights or money (`hard_stop_kind`, as the
  preparation retry reads a refusal; the job's spending cap; a refusal by the paid-resource chokepoint,
  `PaidResourceAdmissionBlocked`) is left on its own schedule, and the answer names it (`stop_is_a_wall`).
- A record whose step is running is left as it is and the answer says so (`step_running`): the Atlas lock is held, the
  preparation is `processing`, or the listener began its retry less than 15 minutes ago and has not claimed it yet. A
  step mark left by a killed Atlas step is `step_interrupted`: the next tick schedules that retry itself.
- Once per job per release. A retry-now that changed a record is kept on the job's ledger row (`retry_now`: the
  release, who asked, why); a second one on the same release is refused `test_job_retry_now_used_on_this_release`.
  One that changed nothing does not count. The door's receipt keeps every ask with who asked and why
  (`results/<id>.test-job-retry-now.json`).

The answer names each record's result: `brought_forward`, `already_due`, `not_waiting_on_a_retry` (a parked or
finished Atlas job, which a release reopens; a preparation with no timed retry), `waits_for_a_release` (a
preparation retry gated on a release or a deterministic key, not a time), `step_running`, `stop_is_a_wall`, `absent`.
For the preparation it also gives `listener_due_at`: a grant past its expiry first leaves the silence sweep its
window. The door's outcome is `brought_forward` when either record moved, `unchanged` when neither did, and
`refused` when neither did and a record's stop is a wall.

The installed door's request list is fenced to owner-approved kinds; this kind works there only once the owner has
approved adding it to that list (`DISPATCHER_HOLD_ONLY` in `deploy/operator-door/operator_door/requests.py`). Until
then the door refuses it as `kind_unknown`.

## What it costs

Up to about $20 a job once its simulation is asked for; $0 on the host while it waits (`--simulation on_request`). The one complete run so far (2026-11-16, a rigid job) settled $6.04 at World Labs and
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
the zip digest. A job whose workspace storage retention retired reads `workspace_retired`, with the scene's
`<scene>.retired.v1.json` named in the JSON (`retired_receipt`): the door does not serve that receipt, so the report
names it without verifying it and says nothing of how the job ended. The JSON adds the footage id, job and capture ids, times (asked
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
