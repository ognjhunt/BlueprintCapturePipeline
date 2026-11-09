# Goal brief: invited-beta readiness (one planner, many executors)

This is the brief for one long Claude Code session, started with `/goal`. The
main session is the **planner**: it decides, plans and reviews. Its subagents
are the **executors**: they read, build, test and open PRs. The goal command
that starts the session names the model for each role.

The facts below were read on 2026-11-09. Other sessions change these repos and
the host every few hours, so re-check a fact before you act on it.

## 1. The outcome

Blueprint can take jobs from invited outside sites without an engineer driving
each job by hand, and without a mistake in privacy, rights or money.

"Ready" means the exit criteria in section 4 have evidence. At the end of the
session, each criterion has one of three states:

- **PASS**, with a link to the evidence: a merged PR, a CI run, a drill
  record or a tracker line.
- **OWNER**, with the one decision or action only the owner can take, and your
  recommendation.
- **OPEN**, with the next work package and the reason it is not done.

## 2. Roles

- **The planner (the main session).** You own the exit criteria, the gap
  register, the plan, the work-package briefs, the review of each result and
  the final report. Planner tokens cost about twice as much as executor tokens,
  so spend them on judgment, not on reading. Do not read source files longer
  than about 150 lines, run broad searches or write code yourself. Read this
  brief, both `AGENTS.md` files, the tracker sections named in section 3, and
  the briefs your executors return.
- **Executors (subagents).** They do the recon, the code, the tests, the
  replays and the PRs. Set the executor model that the goal command names on
  every Agent call and every workflow agent, even when the session already
  forces it.
- **Verifiers (subagents).** A verifier is a fresh executor that did not write
  the change. Give it the diff, the package's acceptance test and the files
  involved, not the author's conclusion. Use one for every change that touches
  money, privacy, rights, deploys or the job state machine.

Do not start a subagent for work that takes three tool calls or fewer.

## 3. What you already know (do not rediscover it)

### The product and its rules

- A site sends an email and a video of one recurring job. The Pipeline builds a
  simulation of the site, Blueprint runs candidate robot policies in it, and the
  site gets a bounded estimate or "held: we can't simulate this kind of job
  yet". The beta is free. Blueprint-WebApp is the customer side; this repo is
  the back end.
- **The only hard stops are privacy, rights and money.** Everything else is
  flagged and keeps going. Do not add a gate, approval step, review stage or
  readiness check unless it protects one of the three. Delete checks that
  protect none of them.
- **Do not make things up.** No invented results, captures, approvals or
  readiness. Unknown stays unknown. Simulation results never certify physical
  performance or safety. Beta results never enter Arm Decision Proof v1
  evidence (`docs/arm_decision_proof_v1/`).
- **Money.** The account-wide limit is USD 200 per UTC month (owner,
  2026-11-07). Every ceiling is in the `AGENTS.md` "Spend limits" table. Paid
  compute starts only through `paid_resource_allocator`. A change to any money
  value is the owner's decision.
- **Host access** goes through the operator door
  (`python3 scripts/operator_door.py`, `docs/OPERATOR_DOOR.md`), never SSH.
  Cloud setup: WebApp `docs/runbooks/cloud-scene-runs.md`.
- **Deploys** follow `AGENTS.md` "Beta Deploy Policy" (D-08): batched, in one
  window a day from 15:00 UTC, never during a paid stage, and recorded in the
  tracker. Every Pipeline deploy retires and rebuilds in-flight website
  attempts.
- **Replay before merge.** A fix for a stopped production job is replayed on
  its branch at $0 (`task_evaluation_stage_replay`, `atlas_preview_replay`)
  before it merges.
- **Checks.** Pipeline: `python -m blueprint_pipeline.impacted_test_selection`
  and `ruff check <changed files>`. WebApp: `npm run check`,
  `npx vitest run <paths>` and the affected `e2e/` specs.

### Where things stand on 2026-11-09

- The source of truth is
  [`2026-11-beta-launch-tracker.md`](2026-11-beta-launch-tracker.md). Read its
  section tables A to H (near the end) and its newest dated entry only. The
  rows were last updated on 11-03 and some are stale: C-01 (#3196) and B-04
  (#3315, WebApp #1204) have merged since.
- Two other cloud sessions are live: the **G1 session** and the
  **library-aisle session**. They share the `cloud-scene-runs` door token and
  deploy. Do not drive their jobs: capture-447 and the library-aisle
  navigation job (`capture-5b067498…`). Tell them before each deploy, as the
  tracker describes.
- The owner gave a standing permission to deploy the latest `main` once its CI
  is green, until 2026-11-11 14:26Z. Every other deploy rule still holds.
- This package has about 2,400 modules (1.28 M lines) and about 2,700 test
  files. Never ask an executor to "read the codebase". Give it a question, a
  starting path and an output limit.

### The pattern to break

Since 11-06, most work has gone into one live job. Of the newest 50 commits in
this repo, 37 are about levelling, floors, splats, G1 or Atlas for the library
aisle, and levelling has reached v10. Each fix was found by deploying and
waiting for the next stop. On 11-07 the tracker already said "More would be
fitting one scene". Meanwhile most items in sections A to H have not moved
since 11-03.

One job proves one path. A beta needs every job to end truthfully, on time and
inside its money, whatever kind of job it is.

### Leads from a read-only sweep on 2026-11-09

These are starting points for recon, not findings. Verify each one. "T:n" is a
line of the tracker.

**Only one job has ended, and it was driven by hand.**

- capture-447 (rigid pick-and-place) is the only website job with a terminal
  customer result: `estimate` on 11-06, 0 successes in 39 scored episodes
  (T:425). It took about 8 deploys, several re-plans and owner money calls.
- The library-aisle job (navigation) is at step 14 with no valid episode
  (WebApp `docs/library-aisle-navigation-e2e.md`). It was requeued with
  `door selected-handoff --atlas-retry` and resumed with
  `--resume-preparation` many times.
- The drawer job ended `blocked` with no scored episode. The dishwasher job's
  steps 10 to 14 are "unproven".
- Only rigid pick-and-place has reached a result. The library-aisle doc lists
  the task-kind gates in both repos. In WebApp, the robot-fit receipt still
  accepts only `rigid_pick_place` and `articulated_open_close`
  (`server/utils/websitePreparationStatus.ts`, a strict `z.enum`).
- WebApp tells a waiting site "Simulation evaluation in progress…" with no
  time bound (`server/utils/betaPolicyScreeningContract.ts`).

**The host and the deploy model limit concurrency.**

- The host has 4 CPUs and 8.3 GB of RAM with no swap. A 4K clip caused
  out-of-memory kills (T:127, T:336). The work volume was below its 15.8 GB
  floor on 11-09.
- Every deploy retires in-flight attempts. C-06 (release succession and a
  door-enforced paid-stage hold) is not done.
- Paid admission needs a spend report under 5 minutes old, so a resume right
  after a deploy is refused (T:141, T:156).
- Provider credit is not reserved across concurrent launches (C-07).

**Money.**

- B-07: the internal ledger read $165.55 while reconciled billing read
  DigitalOcean $325.38 and Vast $455.88 (T:1002). The gap is unexplained.
- Stage 2 of metered spend is planned, not built
  (`docs/architecture/metered-spend.md`). The `AGENTS.md` table marks values
  held in Render or on the host as "unverified".
- `max_paid_attempts` may never bind website scene reservations (B-03, found by
  reading code only).

**Privacy, rights and abuse.**

- G-02: the consent-withdrawal drill has never been run. WebApp confirms a
  deletion only after the Pipeline acknowledges it
  (`server/utils/websiteCaptureWithdrawal.ts`).
- The retention policy says it is not proof that deletions run
  (`docs/BETA_DATA_RETENTION_POLICY_2026-07-09.md`). Retiring a scene leaves G1
  overlay rows and packets in place (T:687).
- G-05: credential rotation is open, and both repos are public. G-01: counsel
  status is due before 2026-11-15.
- WebApp has no invitee allowlist in code (F-04). The intake cap is 5 jobs a
  day in total but 20 per IP address
  (`server/utils/betaAbuseCaps.ts`, `INTAKE_CAP_DEFAULTS`), so one network can
  use the whole day's quota; the code comment says the opposite. App Check is
  off by default.

**Change safety.**

- E-06 is not done. Of the scheduled nightly Full Test Lane runs from 11-03 to
  11-09, 2 of 7 passed (11-03 and 11-07). The 11-09 run failed, mostly on
  import-isolation tests.
- `docs/CI_REQUIRED_CHECKS.md` says a deploy needs a Full Test Lane promotion
  run on the exact commit. The door deploys on main CI's attestation alone
  (`docs/OPERATOR_DOOR.md`, "Request a deploy right after merging"). Decide
  which gate is real, and make the documents and the door agree.
- Main CI has failed on external downloads (GitHub release assets, the Rekor
  transparency log), and sessions cannot re-run jobs (HTTP 403).
- WebApp CI never runs three intake suites (`playwright.intake.config.ts`,
  `playwright.intake-backend.config.ts`, `playwright.intake-durability.config.ts`).
  A change limited to `server/utils/**` or `client/src/lib/**` gets release
  scope `code` and skips the browser suites. No e2e spec covers a finished
  estimate on the job page.

**Operations.**

- No on-call backup is named (D-24). Only the owner's computer has SSH to the
  host, and there is no host rebuild runbook.
- The DR runbook requires a backup and restore drill before beta, and none is
  recorded. As of the 10-18 WebApp audit, Firestore had no scheduled backups,
  and 16 declared indexes were never deployed (F-02, H-03).
- The D-04 preflight reports 186 blockers and is "not a health signal".
- The GitHub uptime backstop is scheduled every 10 minutes but ran about every
  3 to 4 hours on 11-08 and 11-09. The Render uptime cron waits on a manual
  deploy (D-02).

## 4. Proposed exit criteria

Turn these into measurable thresholds in the plan. Take the numbers from
evidence: the cost model ([`2026-11-cost-model.md`](2026-11-cost-model.md)),
the invited cohort (3 to 5 sites, per the intake cap's comment), and the host
size. A threshold that is a money value goes to the owner as a recommendation;
do not apply it yourself.

- **E1. Every job ends.** Each admitted job reaches a terminal customer state
  (estimate, reference-only, held with a reason, withdrawn or ended) with no
  operator command, inside a wall-clock bound and a cost bound. While it runs,
  the job page says truthfully what it is waiting on and when the site will
  hear next.
- **E2. "Held" is first class and cheap.** A job kind the system cannot
  simulate holds before its first paid call wherever that can be known, with
  one plain sentence for the site. The time and the money spent before a hold
  are measured.
- **E3. Fixes generalize.** An offline replay set of retained jobs of different
  kinds (pick-and-place and navigation; 1080p and 4K; short and long clips)
  runs at $0 in CI or nightly. A fix made for one job passes the whole set
  before it merges. If no such set exists, building it is one of the first
  packages.
- **E4. Money is correct.** No paid call can bypass admission, and a test
  proves it. The internal ledger reconciles with provider billing inside a
  stated tolerance. A $0 drill proves the USD 200 account limit and its
  alerts. Concurrent launches cannot over-commit provider credit.
- **E5. Privacy and rights hold under a drill.** A consent withdrawal on a test
  job stops the work and removes or retires every copy the policy names:
  WebApp, the host, derived scenes, provider uploads where deletion is
  possible, and pending emails. The retention schedule runs. Every clip has a
  rights record. Exposed credentials are rotated (owner action).
- **E6. Changes are safe.** `main` CI is green and not flaky for 7 consecutive
  nightly runs. External downloads that broke CI are cached or pinned. One
  documented deploy gate matches what the door enforces. A deploy does not
  destroy paid work in flight, or drains it first. A rollback has been
  rehearsed.
- **E7. Capacity fits the cohort.** The planned number of concurrent jobs runs
  on the current host without out-of-memory kills or disk refusals. Inputs are
  normalized at intake (resolution, frame count, duration), so one 4K clip
  cannot exhaust an 8 GB host.
- **E8. Someone notices and acts.** A stuck job is detected and either ends
  itself or pages. A primary and a backup are on call. The runbooks have been
  run once in a game day. Health checks are quiet when the system is healthy,
  and the uptime probes run on schedule.
- **E9. The site's journey works.** The invitee allowlist, the intake caps and
  App Check are on. CI runs the e2e specs for intake, the job page, every
  outcome and withdrawal. Every outcome sends the right email. All copy is
  true.

## 5. How to plan efficiently

1. **Read** this brief, both `AGENTS.md` files and the tracker sections named
   in section 3. Nothing else yet.
2. **Recon in one parallel pass.** Use a workflow (the goal command opts in) or
   parallel Agent calls: at most 8 read-only executors, one per group of
   criteria. Give each one the question, the criteria it covers, the leads
   from section 3 to verify, two or three starting paths, and this output
   schema, in at most 500 words:

   `gap | criterion | evidence (file:line, PR, tracker ID or command output) | verified or inferred | severity (blocks beta, degrades, polish) | size (S, M, L) | owner-only? | smallest fix`

   Tell each one not to propose new gates and not to invent facts.
3. **Write the plan yourself** to
   `docs/launch-readiness/2026-11-beta-readiness-plan.md` in this repo. It
   holds:
   - the exit criteria with thresholds;
   - the gap register, with the recon rows merged and de-duplicated;
   - the owner list: each decision, your recommendation and what it unblocks;
   - the work packages in waves, as a dependency graph, ordered by beta risk
     retired per day of work, not by ease.

   Open it as a PR, and add one line to the tracker that points to it. The plan
   file is your memory. Update it after every wave, so a compacted or resumed
   session can continue from it.
4. **Do not wait for plan approval.** Start the first wave as soon as the plan
   PR is open. The owner reads the plan in parallel.

## 6. Work-package format

One package is one PR in one repo, or a linked pair in this repo and WebApp.
Each brief to an executor holds:

- the goal in one sentence, and the exit criterion it serves;
- the files to start from;
- the acceptance test: the exact command, and the output that proves it;
- the replay it needs (for a fix to a production stop), and the expected first
  stop;
- what it must not touch: the live jobs, the money values, the shared doctrine
  blocks;
- the expected spend: $0 unless the owner approved a figure.

The executor branches from the latest `main`, builds the change with its tests,
runs the repo checks, opens a ready-for-review PR, and reports in at most 300
words: what changed, the test output, the PR link, and anything it could not
do. Run independent packages in parallel, at most 4 at a time, each in its own
worktree.

## 7. Execution rules

- One driver per live job. Never resume, requeue or deploy for a job another
  session drives.
- No per-scene tuning. A fix names its general cause and passes the replay set
  (E3), or two retained jobs until that set exists.
- Every new check protects privacy, rights or money. Otherwise flag and
  continue.
- Paid work runs only through the allocator, inside an approved figure.
- Before any deploy, read `door status` and `door requests`, follow D-08, tell
  the other sessions, and record the deploy in the tracker.
- When two facts disagree, trust the newest primary source (host status, a
  provider bill, a CI run) over a document, and correct the document.

## 8. What is approved, and what needs the owner

These steps need no further approval in this session:

- reading anything; running tests, linters and $0 replays; creating branches;
  opening PRs; editing the tracker and the plan;
- retrying a command that is safe to repeat;
- merging your own PR when its CI is green and a verifier approved it;
- deploying the latest green `main` inside D-08. While the owner's standing
  permission lasts (until 2026-11-11 14:26Z), such a deploy needs no further
  ask. The permission keeps every other deploy rule (the daily batching and
  its out-of-window exceptions, a quiet door, no deploy during a paid stage,
  telling the other sessions, a tracker line, and never
  `install-native-g1-policies`), and it widens no money grant;
- door operations that only read.

These need the owner first. Put each one in the owner list with your
recommendation, then carry on with the work that does not depend on it:

- a change to any money value, or a paid run without an approved figure;
- email to anyone outside Blueprint, or contact with outside companies;
- deleting retained data;
- credential rotation, legal and counsel questions, repo visibility, the
  invitee list and the on-call names.

## 9. Finishing work

Do not stop while you still owe work. Carry on with the work that does not
depend on an owner answer. Ordinary obstacles are not blockers: a flaky CI job,
a timeout, a busy door or an empty result. Diagnose them, then retry, fix the
request or take another route. A deliberate blocker is left alone and
reported: a hard stop, a permission that was withheld, or another session's
job. This does not remove the owner approvals in section 8.

## 10. Reporting

After each wave, give the owner a short update: the criteria that moved, the
PRs merged and the owner decisions waiting. Write the final report for a reader
who saw none of the work. Lead with the readiness verdict, then a table of E1
to E9 with status and evidence, then the owner list, then what is left. Use
plain words and full sentences, and do not use names you made up during the
work.
