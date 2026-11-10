# Retries and deadlines: how a job keeps going and how it ends

An audit on 2026-11-10 found three reasons a website job stops being
autonomous. This note records the rule that answers each, where the rule
lives, and what is still not covered. It describes code, not a plan.

| Root cause | Rule | Status |
|---|---|---|
| The code stops by default | One retry policy. Only privacy, rights and money stop a job. | Built for the controllers in section 2. |
| A deploy is the retry mechanism, and it destroys work | Every retry is timed. A deploy waits for paid work. | Timed retries: built for the controllers in section 2. Waiting for paid work: section 3 (needs a door upgrade). Attempts are still bound to a commit: section 5. |
| No job has a deadline | The job's window is its deadline, and the job ends there by itself. | Built in the WebApp (section 4). Pipeline gaps: section 5. |

## 1. One retry policy

`src/blueprint_pipeline/job_retry_policy.py`. Pure: no I/O, no clock.

| A step fails with | What the job does | What ends it |
|---|---|---|
| A refusal that protects privacy or rights | Ends or holds at once, as before. | The refusal. |
| A refusal that protects money | Waits and asks the same gate again. | The job's deadline, or the owner raising the limit. |
| An outage (429, 5xx, transport, storage) | Retries with capped backoff. No attempt count. | The job's deadline. |
| Anything nobody has classified | Six timed retries: 15 min, 1 h, 3 h, 6 h, 12 h, 24 h. | The schedule: after about 46 hours the job ends by itself with the refusal's own code. |

Why a retry is safe: a retry never bypasses a gate, it asks it again. Consent,
withdrawal and every paid admission are checked by the step itself on each
attempt, so the gates stay fail-closed while the job keeps going. A paid call
with a retained receipt is reused, not bought again.

Why about 46 hours: the schedule spans two daily deploy windows (`AGENTS.md`,
"Beta Deploy Policy"). A fix that ships in the next window, or the one after,
is picked up by a timed retry. No deploy and no operator command is needed to
trigger a retry; a deploy only changes what the next retry finds.

`hard_stop_kind` reads the first two rows from a refusal code's words. It errs
toward naming a hard stop. A controller uses it only to stop or wait sooner,
never to skip a check.

## 2. Where the policy is used

| Controller | Before | Now |
|---|---|---|
| Atlas reconstruction (`website_atlas_preview.py`; owned captures share it) | Any refusal off a short list was a permanent `failed`; an operator retry was the only way back. Eight transient failures (about three hours) ended the job. A step killed for memory re-ran every minute. | The explicit lists are unchanged. An outage retries until the job's 7-day deadline. A money refusal is `needs_input` with its code and is asked again every 30 minutes, like the other refusals an owner can lift. Any other refusal, a defect, or a step that never returned gets the six timed retries (twelve in a job's life), then `failed` with its own code. A privacy or rights refusal still fails at once. `retry_failed_job` reopens a job that has ended. |
| Beta screening (`beta_policy_screening_coordinator.py`) | A candidate refused before any allocation was planned again only after a deploy. One candidate's harness failure held every other candidate until a deploy. | An unspent refusal is planned again at the same commit on the six timed retries, per site and candidate, counted on the ledger. Privacy, rights and money refusals, and answers about the candidate, are never retried by the clock. A harness failure holds only candidates on the same backend. |
| Team scene packet (`team_scene_packet_worker.py`) | A refusal not on the transient list went back to pending once per deployed release. | It also goes back on the six timed retries. The count is on the status and nothing resets it. |
| Rigid probe (`rigid_robot_free_probe_runner.py`) | A staging refusal was set aside once per deployed release. | It is also set aside on the six timed retries, through the same supersession markers, which are the count. |
| WebApp scene intake (`taskEvaluationSceneIntake.ts`) | Forwarding gave up for good after 20 one-minute tries. | It retries until the intent's expiry: five tries a minute apart, then doubling to 30 minutes. |

What stays as it was, on purpose:

- A harness failure that spent money is planned again only after a deploy
  (`HARNESS_RETRY_LIMIT`). A second paid run at the same commit would usually
  fail the same way and use one of the candidate's seven retries.
- A funded rigid-probe attempt that failed in our runtime keeps its
  once-per-release rule and `MAX_RUNTIME_REDRIVES`.
- Every cap, limit, grant and reservation. A timed retry changes when money
  can be spent, never how much.

## 3. A deploy waits for paid work

Each scene attempt is bound to the commit of the release that created it, and
later stages refuse work bound to another release. So a deploy retires the
in-flight attempt.

- The operator door holds a deploy while a paid stage is live on the running
  release: a scene-configuration activation (as before), a submitted
  scene-configuration launch, a held paid-launch slot, or a screening candidate
  `launching` or `running`
  (`deploy/operator-door/operator_door/activation_hold.py`). One cap of 285
  minutes bounds the hold. `--activation-hold-override <reason>` deploys at
  once for a fix to money, privacy, rights or the stuck stage. It reads only,
  and anything it cannot read means "deploy now".
- The installed door gets this only after the owner's door upgrade
  (`docs/OPERATOR_DOOR.md`). A normal deploy does not update the door.
- `scripts/operator_door.py deploy` does not send a deploy whose only
  difference from the active release is under `docs/launch-readiness/`. Such a
  deploy changes no code a job runs and would only retire attempts.
  `--even-if-no-runtime-change` sends it anyway.

## 4. Every job has a deadline

The deadline is the job's window: the grant's expiry with the owner's one
extension, or seven days after creation for a website job that never received
a grant (WebApp `websiteJobDeadline`).

- The WebApp's beta ops monitor pass ends each website job whose deadline has
  passed while its page still shows it open (`websiteJobDeadlineSweep.ts`). It
  writes one ending record and, in the same transaction, the intent for one
  email. It runs in the web process, so it does not depend on the Pipeline
  host being alive.
- The job page shows a job past its deadline as ended without waiting for the
  sweep.
- A result or a hold that arrives later still wins on the page. A withdrawn job
  is never ended or emailed by the sweep.
- On the Pipeline, the same window already ends an Atlas job
  (`atlas_preview_expired`), a scene intent (`authority_expired`) and a silent
  preparation whose grant the WebApp refuses as expired (WP-03).

## 5. Not covered

1. **Attempts are bound to a commit.** A deploy that changes runtime code
   still retires in-flight attempts once nothing paid is live or the hold's cap
   passes; the CPU preparation is rebuilt. The complete fix is that the old
   release drains its own attempts. The pieces exist: each release is its own
   tree, kept at least 24 hours and longer under a lease; every queue document
   names its commit; workers act only on documents of their own commit. What
   is missing: a pass per draining release for each worker, and new-release
   workers that skip a foreign document instead of blocking it. This needs the
   host to validate and was not built.
2. **Preparation after Pub/Sub's five deliveries.** Only
   `--resume-preparation` moves a `failed_retryable` preparation. The handoff
   bytes are retained only for the two explicit waits, and "an unknown provider
   outcome never becomes fresh retry capacity" (`website_reconstruction_resume`)
   is a money rule. An automatic re-drive needs the owner's rule for an
   unanswered paid call, as SAM got on 2026-11-07.
3. **A screening run that failed on a download.** It is still a harness
   failure and waits for a deploy. The launcher gives no typed code that tells
   a failed download from a harness defect.
4. **Three preparation states with no ending on the Pipeline at expiry**
   (readiness plan G50). The WebApp now ends the job for the site; the
   Pipeline ledger stays as it was.
5. **Refusals cached by a revision constant** (Atlas levelling). A timed retry
   of the same inputs and code gives the same answer.
6. **WebApp:** an email with an unknown send outcome is alerted, not resent
   (the provider call has no idempotency key). Result schemas still reject
   unknown fields. A sponsorship policy edit still revokes accepted intakes.
