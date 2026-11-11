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
| Beta screening (`beta_policy_screening_coordinator.py`) | A candidate refused before any allocation was planned again only after a deploy. One candidate's harness failure held every other candidate until a deploy. | #3550 on `main` added the self re-plans: a transient failure on the rented machine retries at the same commit, a policy-load failure holds only its candidate and other harness failures hold same-backend siblings, and a row out of GPU offers or with an expired quote is planned again by itself, three times each. This work adds the default beside them: any other refusal before allocation that no rule owns is planned again on the six timed retries, per site and candidate, only for its own reason. Privacy, rights and money refusals, and answers about the candidate, are never retried by the clock. |
| Team scene packet (`team_scene_packet_worker.py`) | A refusal not on the transient list went back to pending once per deployed release. | It also goes back on the six timed retries. The count is on the status and nothing resets it. |
| Rigid probe (`rigid_robot_free_probe_runner.py`) | A staging refusal was set aside once per deployed release. | It is also set aside on the six timed retries, through the same supersession markers, which are the count. |
| Scene configuration (`task_evaluation_scene_launch_recovery.py`, from `_recover_configuration_capacity`) | A launch refused before renting anything (no GPU offer, an object-storage or bundle staging error, a watchdog that would not arm, a Vast inventory query error) or one that ended with no allocator result waited at `scene_configuration_failed` for a deploy. A machine lost mid-run ended the job (`max_retries: 0`). | When the launch's own records prove no machine and no paid API call, a $0 successor follows on the six timed retries, then the job ends by itself (`capacity_recovery_cap_exhausted`). The same failure without that proof keeps its named blocker. The wait's blocker names the hours left, so it changes hourly and the stall alarm does not page it. A lost machine spends the intent's signed `max_retries` (1 for a Website grant issued since 2026-11-10), inside the same dollar cap and `max_paid_attempts`. Credit, disk and authentication refusals keep their measured gates. |
| WebApp scene intake (`taskEvaluationSceneIntake.ts`) | Forwarding gave up for good after 20 one-minute tries. | It backs off from 1 to 15 minutes and retries until the job's window ends (WebApp #1257). |

What stays as it was, on purpose:

- A harness defect that spent money is planned again only after a deploy
  (`HARNESS_RETRY_LIMIT`). A second paid run at the same commit would usually
  fail the same way and use one of the candidate's seven retries. A transient
  failure on the rented machine is different and retries at the same commit
  (#3550).
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
extension. The WebApp half landed as WebApp #1257.

- The WebApp worker's sweep (`websiteJobWindowSweep.ts`) ends each website job
  whose window has passed with no outcome on record. One transaction writes the
  stored ending the job pages show and queues one email, which is never sent
  twice. It runs in the WebApp, so it does not depend on the Pipeline host
  being alive.
- It never ends a job that already has an outcome, a withdrawal or an owner
  closure on record, or whose window an extension moved into the future.
- A window that ended before 2026-11-11T00:00Z gets the stored ending and no
  email.
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
2. **Preparation after Pub/Sub's five deliveries: now built**
   (`website_preparation_retry.py`). Every claimed website attempt keeps its
   message bytes beside its ledger, and the listener's tick re-runs a
   `failed_retryable` preparation on this policy: an outage or a money refusal
   until the grant's expiry, anything unclassified on the six timed retries,
   then left for the grant-expiry ending and the WebApp's window sweep. A
   privacy or rights refusal is never re-run. A dead process's attempt is
   recorded `failed_retryable` once its lease has expired and its process is
   gone, and runs again within 10 minutes. A delivery refused before its claim
   keeps its bytes until a run claims it. An unanswered paid call is sent
   again only under a new reservation: SAM resends once (the owner's
   2026-11-07 rule). Gemini (owner approval 2026-11-10, every call site through
   `retained_gemini_call`): a dispatch with no answer may have been billed, so
   it is counted as spent (its reservation stays held at its maximum, which the
   WebApp ledger debits; there is no Google settlement route) and the call is
   sent once more under one new reservation, which the WebApp admits only
   inside the job's ceiling and the account limit. A refused reservation, or an
   unanswered call with no reservation on record, is a money wait, never a
   send. A Gemini error in the clean-plate analysis is raised
   (`GeminiProviderError`) and retried by this policy, never retained as a
   completed "blocked" plan; a "blocked" plan an older release retained is
   still read and reused as it stands.
3. **Three preparation states with no ending on the Pipeline at expiry**
   (readiness plan G50). The WebApp now ends the job for the site; the
   Pipeline ledger stays as it was.
4. **Refusals cached by a revision constant** (Atlas levelling). A timed retry
   of the same inputs and code gives the same answer.
5. **A website job that never received a grant has no deadline.** The WebApp
   sweep reads the grant's window, and such a job has none, so it is not
   ended. On the Pipeline, `end_expired_preparation` needs a grant too, so
   nothing ends such a preparation; the host retry therefore gives every
   failure of a grant-less job, an outage included, only the six timed retries
   (`has_deadline` in `website_preparation_retry.py`) and then leaves it. Its
   ledger is not marked ended.

## 6. Reconstruction refusals: continue on an estimate, or wait on a key (2026-11-10)

A reconstruction quality refusal decided only by retained inputs and code repeats on every timed
retry: the same video and release give the same answer. Two changes answer it.

- **Continue on a labelled estimate.** Levelling with nothing observed to level on takes the
  provider's +Y, unverified, with both observed tilts recorded (`scene_estimate:level`); a splat with
  view-dependent colour keeps its base colour (`scene_estimate:splat_view_dependence_dropped`). A
  mesh that fits no declared frame, or two, takes the frame of smallest median residual and keeps
  only faces near observed geometry (`scene_estimate:collider_registration`; no splat-aligned
  collider exists at $0 today). A scene review's citation of an unusable frame is dropped
  (`scene_estimate:review_citations_dropped`); a surrounding view past the frame budget is left out
  and reported uncaptured (`scene_estimate:surrounding_views_dropped`). Each estimate is named on the
  base scene (`atlas_scene_estimates`), copied into the screening plan, and published in v3
  `scene_feasibility.flags`; the website names each in its own sentence. Counts and intervals are
  shown as measured: no estimate adjusts them.
- **Wait on a key, not a timer.** What still refuses on retained inputs alone raises
  `DeterministicRefusal` (`deterministic_refusal.py`): its key binds the code, the step's revision
  constant and the input digests. The Atlas controller and the preparation retry keep the record and
  do not run the step again while the running release computes the same key, so nothing is bought or
  redone; a release that changes the step's revision runs it once. The job's deadline still ends it.
- **End with a site ask.** A video that never connects or sharply shows the task area
  (`atlas_task_and_scene_coverage_gap`, `atlas_insufficient_overlapping_distinct_views`,
  `atlas_coverage_anchor_quality_insufficient`), and a scale no reference supports
  (`atlas_scale_review_unresolved`), already end at once; the website now tells the site what to film.

This replaces item 4 of section 5 for the refusals named here.

## 7. Preparation stops only a new video can clear (2026-11-11)

`website_front_normal_undetermined` and five `website_registration_*` fit codes (`ambiguous`,
`poor_fit`, `conflicts_provider_anchor`, `anchor_deviation`, `ground_plane_inconsistent`) are decided
by the video alone. The handoff keeps the stop (`website_recapture_required`), the stage completes as
a hold, so Pub/Sub acknowledges it and the preparation retry never arms, and preparation status v7
names the code as the outcome; the website asks the site to film again. A new video is a new
capture (the owner's "update video" issues a replacement capture, which becomes the active one) and
runs from the start. `website_registration_anchor_frame_missing`, `..._anchor_frame_invalid` and
`..._scale_conflicts_declared` are not video faults and keep their earlier handling.
