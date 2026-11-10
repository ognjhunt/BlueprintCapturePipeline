# Invited-beta readiness plan (2026-11-09)

This plan turns the exit criteria of the 2026-11-09 goal brief
(`2026-11-09-beta-readiness-goal-brief.md`, on branch
`claude/zealous-carson-f49k6s` until it merges) into thresholds, lists the gaps
found by a read-only recon of both repos on 2026-11-09, and orders the work.

It is the planning session's memory. Section 6 is updated after every wave, so a
resumed session continues from it. The live record of deploys and jobs stays in
[`2026-11-beta-launch-tracker.md`](2026-11-beta-launch-tracker.md).

Rules that bound every package: the only hard stops are privacy, rights and
money; no new gate unless it protects one of them; no paid run outside the
allocator or an approved figure; no per-scene tuning; nobody in this plan drives
capture-447 or the library-aisle job (`capture-5b067498…`).

## 1. Two facts that outrank the rest

Both were read from `door status` on 2026-11-09 (21:26Z and 21:32Z).

1. **The account limit will bind before the cohort runs.** November spend is
   $136.43 of the $200 limit: OpenAI $42.11, Vast $34.52, DigitalOcean $24.14,
   Google Cloud $21.87, World Labs $13.79. DigitalOcean rent alone is about
   $2.68 a day, so about $56 of the $63.57 left is already spoken for by
   2026-11-30. At the limit every new paid admission is refused, so an invited
   job would hold at once. This is owner decision O-01.
2. **The work volume is below its floor and reclaim frees nothing.**
   `/mnt/blueprint-work` had 3.55 GB available against a 15.8 GB floor, with
   `reclaimable_bytes=0`. Launches refuse at $0 until space returns. 406 orphan
   scratch directories hold 96.8 GB but cannot be retired without a reference
   census, and deleting them needs the owner (O-02).

Until both are settled, no invited job can finish, whatever else this plan
fixes.

## 2. Exit criteria with thresholds

"Today" is the recon reading on 2026-11-09. A threshold marked *(owner)* is a
money value or an owner fact; the plan recommends it and does not apply it.

| ID | Criterion | Threshold (how it is measured) | Today |
|---|---|---|---|
| E1 | Every job ends | (a) Every non-terminal job state has a deadline that writes a terminal customer state with no operator command; a test lists the states and fails when one has no deadline. (b) Wall-clock: a job ends inside its 7-day job window (`website_job_window.py`; 14 days with the one allowed extension). Target, not yet measured without an operator: 72 h from video to a terminal state. (c) Cost: no job's spend across all ledgers passes its job ceiling ($40 default; cost model p50 $27, p90 $44). (d) Every non-terminal sentence on the job page names what the job waits on and an "ends by" date; 0 sentences without one. (e) Three consecutive admitted jobs end with 0 door commands. | A stalled preparation never ends by itself. Atlas retry and preparation resume are door-only. Money waits have no deadline of their own. No sentence carries a date. 1 job has ended, after about 8 deploys. |
| E2 | "Held" is first class and cheap | (a) An unsupported job kind holds after at most the Atlas planner review and before any World Labs purchase: spend before the hold is at most $0.50 (3 reviews measured $0.353), and the clean plate (SAM and Gemini, $0.69 measured) does not start for a held job. (b) The site sees one sentence with no code, on the job page and in the email; a test maps every early-hold reason to it. (c) Seconds from video to hold and dollars settled before the hold are recorded on 100% of holds. Target of 60 minutes is provisional until 3 holds are measured. | The early hold exists and precedes every World Labs stage. The site is told "could not finish processing". Neither time nor spend is recorded. The clean plate runs beside the review. |
| E3 | Fixes generalize | (a) A replay set of at least 4 job records covers pick-and-place and navigation, 1080p and 4K, a short and a long clip. (b) It runs at $0, with no host and no network, in the nightly lane and on pull requests that touch preparation, levelling or screening, in at most 15 minutes. (c) A fix for a stopped production job passes 100% of the set before it merges. | Four $0 replay tools exist; each reads inputs pulled from the host. No committed set, no CI step. Levelling v8 to v10 tests cite one scene. |
| E4 | Money is correct | (a) The paid-call verifier covers API-priced providers (OpenAI, Gemini, World Labs, SAM) as well as compute, in both repos, with 0 unclassified call sites; it runs in CI. (b) A month-scoped reconciliation per provider reports settled meter spend against the provider's billing delta. Tolerance *(owner)*: the larger of $5 or 10%. (c) A $0 drill on the real admission code shows a warning at $100, pages at $160 and $200, and refusal with `account_monthly_spend_limit_reached` at $200. (d) With 3 launch slots, the sum of admitted Vast caps never passes the balance less the $1 reserve; a test runs the 3 launches together. (e) A test proves `max_paid_attempts` refuses a scene-configuration reservation past the cap. | The verifier sees compute and storage only. WebApp admin routes buy World Labs worlds with no admission. Ledger $181.76 against $824.15 of billing, on different scopes, compared nowhere. Unit tests only for the limit. Credit is checked per launch. |
| E5 | Privacy and rights hold under a drill | (a) An offline drill across both repos passes in CI: after a withdrawal, 0 emails are pending, no new paid admission is granted for that site, and the receipt lists every copy the policy names as removed or as remaining with a reason. (b) One host drill on a synthetic job *(owner approves the deletion step)*. (c) A withdrawal produces its deletion plan with no operator within 1 hour of acknowledgement; completion inside the policy's 30 days, target 24 h. (d) The 14-day temporary class is enforced by a daily timer, with at least one receipt. (e) 100% of jobs carry a rights record: the site's attestation, or the source and license of an operator clip. (f) The four exposed keys are revoked, with provider and time recorded *(owner)*. | Deletion is a manual operator command. No drill has run. Provider copies (World Labs) are never deleted. No retention timer. Stock clips have no rights record. Keys not rotated. |
| E6 | Changes are safe | (a) 7 consecutive green scheduled Full Test Lane runs. (b) Import-closure tests run on pull requests. (c) The ffmpeg asset and the Playwright browser are cached in CI. (d) One deploy gate, the same in `CI_REQUIRED_CHECKS.md`, `OPERATOR_DOOR.md` and the door. (e) The door refuses a deploy while a website or Atlas stage is paid, except the D-08 exceptions. (f) One rollback rehearsal recorded in the tracker. | 2 of 7 green (11-03, 11-07); the streak is 0. Import-isolation tests fail only at night. No cache. Two documented gates. The hold covers Vast launches and scene activation only. No rehearsal. |
| E7 | Capacity fits the cohort | (a) Cohort: 3 to 5 sites, at most 5 new jobs a day, 2 jobs in preparation at once. (b) 0 host-wide out-of-memory kills in 7 days; the memory caps of the heavy units sum to at most 6.5 GB on the 8.3 GB host. (c) Every clip is normalized before the first preparation stage to a long edge of at most 1920 px and at most 30 frames a second, with the original kept; a synthetic 4K clip proves it. (d) The work volume stays above its 15.8 GB floor with 0 disk refusals in 7 days. | 6 host-wide kills in 24 h; two units each capped at 8 GB. No normalization; fixes so far are per stage. 3.55 GB free against the floor. |
| E8 | Someone notices and acts | (a) A preparation whose lease expired and whose process is dead becomes `failed_retryable` and releases its reservation within 30 minutes, with no operator. (b) 0 paging alerts while the system is healthy. (c) A primary and a backup are named and a test page reaches both *(owner)*. (d) An uptime probe runs at least every 15 minutes *(owner deploys the Render cron)*. (e) One game day: a stall, a full disk and a Firestore restore into a scratch database. | Stalls page and end nothing. 22 alerts raised, 6 paging, mostly noise. Names are "TBD". The GitHub probe ran 3 times on 11-09. No game day or restore drill. |
| E9 | The site's journey works | (a) An invitee allowlist is enforced on every route that starts a job once the list is set *(owner supplies it)*. (b) The per-address daily cap is below the global cap *(owner; recommended 2 against 5)*, and the cap covers every route that starts a job. (c) App Check enforces *(owner sets the site key)*. (d) CI runs the three intake suites, an e2e case for each outcome (estimate, reference-only, held, infeasible, no match) and withdrawal; a change to a job-page or intake file gets the full release scope. (e) 0 sentences known to be false. | No allowlist. Per-address 20 against global 5. App Check off. Three suites never run in CI. No reference-only case. Three sentences promise something nothing delivers. |

## 3. Gap register

Merged and de-duplicated from the recon. Basis: V = the executor saw the file,
run or command output; I = inferred. Severity: B = blocks the beta, D =
degrades it, P = polish. "WP" is the package in section 5; "O" is the owner
item in section 4.

| # | Gap | Crit. | Evidence | Basis | Sev. | Size | Goes to |
|---|---|---|---|---|---|---|---|
| G01 | A stalled preparation records "overdue" and never ends | E1 | `website_preparation_terminal.py:1-8`; `pubsub_handoff_listener.py:1585` | I | B | M | WP-03 |
| G02 | Atlas retry and preparation resume run only from the door | E1 | `selected_handoff_recovery.py:18`; tracker lines 94, 108, 126, 371 | V | B | L | WP-10 (dead process), WP-20; the rest stays open |
| G03 | A money wait is non-terminal with no deadline of its own | E1 | `task_evaluation_scene_outcome.py:72-74` | V | D | S | O-12 |
| G04 | Job-page sentences carry no date | E1, E9 | WebApp `betaPolicyScreeningContract.ts:327,341` | V | D | S | WP-14 |
| G05 | An early hold reaches the site as "could not finish processing" | E2 | WebApp `taskPreparationOutcome.ts:87-115`; `website_early_task_hold.py:45-47` | V | B | S | WP-06 |
| G06 | Time and spend before a hold are not recorded | E2 | `website_early_task_hold.py:150-155` | V | B | S | WP-12 |
| G07 | The clean plate is not stopped by an early hold | E2 | `website_early_task_hold.py:25-27` | I | D | M | WP-12 |
| G08 | Coverage and subject holds come after the scene purchase | E2 | `website_early_task_hold.py:19-29` | V | D | M | open (after WP-12) |
| G09 | The robot-fit receipt's task kind is a closed list of two | E2 | WebApp `websitePreparationStatus.ts:106,212` | V | P | S | WP-06 |
| G10 | No committed replay set; every replay reads host inputs | E3 | `docs/OPERATOR_DOOR.md:774-831`; `tests/fixtures` | V | B | L | WP-02 |
| G11 | Data derived from real clips cannot go in a public repo without a decision | E3 | both repos are public | I | B | S | O-03 |
| G12 | No CI step replays anything | E3 | `.github/workflows` | V | B | M | WP-02 |
| G13 | Levelling tests cite one scene | E3 | `test_atlas_world_levelling.py:285,334,399` | V | D | M | after O-03 |
| G14 | The paid-call verifier misses API-priced providers | E4 | `scripts/verify_paid_resource_allocator.py` | V | B | M | WP-05 |
| G15 | WebApp buys World Labs worlds with no admission | E4 | WebApp `admin-site-worlds.ts:259,334`; `worldReconstruction.ts:242` | V | B | M | WP-16 |
| G16 | Ledger and billing are on different scopes, with no tolerance | E4 | `spend_admission_lock.py:469-475` | V | D | M | WP-13, O-13 |
| G17 | No $0 drill of the $200 limit | E4 | `tests/test_account_monthly_spend.py` only | I | D | S | WP-13, O-14 |
| G18 | Vast credit is not reserved across launch slots | E4 | `provider_credit_admission.py` docstring; slot locks held live | V | D | M | WP-09 |
| G19 | `max_paid_attempts` may not bind configuration reservations | E4 | `task_evaluation_scene_configuration_allocator.py` | I | D | S | WP-09 |
| G20 | Atlas preflight is refused after a deploy on a stale report, with no retry | E1, E4 | `selected_atlas_preview_recovery.py:375` | V | D | S | WP-20 |
| G21 | Withdrawal deletion is a manual operator command | E5 | `website_capture_withdrawal_deletion.py:1-66` | V | B | M | WP-08 |
| G22 | The withdrawal drill has never run | E5 | tracker G-02 | V | B | M | WP-08, O-14 |
| G23 | Provider copies are never deleted | E5 | `website_capture_withdrawal_deletion.py:106-115` | V | D | M | open (needs World Labs contact: owner) |
| G24 | Some Pipeline roots are reported, not scanned | E5 | same file, lines 101-105 | V | D | M | WP-08 |
| G25 | Nothing enforces the retention schedule | E5 | WebApp `Privacy.tsx:83,89`; no timer | I | B | L | WP-21, O-17 |
| G26 | Exposed credentials are not rotated | E5 | WebApp `docs/launch-readiness/blockers/2026-07-20-provider-credential-rotation.md` | V | B | S | O-06 |
| G27 | Operator clips have no rights record | E5 | tracker lines 16-21 | I | D | S | O-07 |
| G28 | Nightly lane 2 of 7 green; import-isolation tests repeat | E6 | runs 42677465046, 42493732732, 42247595232, 42064845740, 41866136730 | V | B | M | WP-01 |
| G29 | One-off failures that depend on the runner | E6 | same run logs | V | D | M | WP-15 |
| G30 | Two documented deploy gates | E6 | `scripts/operator_door.py:85,871`; `CI_REQUIRED_CHECKS.md:118` | V | B | S | WP-15 |
| G31 | A deploy retires every in-flight attempt | E6 | `beta-operations.md:457-470` | V | B | L | WP-17 (hold); succession stays open |
| G32 | No rollback rehearsal | E6 | `beta-operations.md` section 5 | V | B | S | O-11 |
| G33 | CI downloads fresh each run | E6 | `full-test-lane.yml:149-164`; `ci.yml:186-201` | V | D | S | WP-15 |
| G34 | Work volume below its floor; reclaim frees 0 | E7 | door status 21:32Z | V | B | M | O-02, WP-22 |
| G35 | No normalization at intake | E7 | WebApp `capture-uploads.ts:82`; `website_assembly_coverage.py:288` | V | B | M | WP-07 |
| G36 | Unit memory caps at or above the host's memory | E7 | unit files; 6 host-wide kills in 24 h | V | D | S | WP-18 |
| G37 | A stall pages and ends nothing; a dead process is not covered | E8 | `beta-operations.md` section 1; tracker lines 127-130 | V | D | M | WP-10 |
| G38 | No backup on call; the page route is unverified | E8 | `beta-ops-incident-response.md:17-21` | V | B | S | O-05 |
| G39 | The uptime probe runs every 3.5 to 10 hours | E8 | WebApp Actions `uptime.yml` runs on 11-09 | V | B | S | O-10 |
| G40 | Alerts are loud when healthy; no game day or restore drill | E8 | door status: 22 alerts, 6 paging | V | D | M | WP-19, O-11 |
| G41 | No invitee allowlist | E9 | WebApp `access-control.ts:131-136` | V | B | S | WP-04, O-04 |
| G42 | Per-address cap above the global cap | E9 | WebApp `betaAbuseCaps.ts:47-49` | V | D | S | O-08 |
| G43 | The cap skips signed-in job starts | E9 | WebApp `inbound-request.ts:1721`; `robot-eval-job-requests.ts:950` | V | D | M | WP-16 |
| G44 | App Check is off | E9 | WebApp `appCheck.ts:28-31` | V | B | S | O-09 |
| G45 | Three intake suites never run in CI; scope `code` skips browser suites | E9 | WebApp `playwright.config.ts:30`; `ci.yml:271-294`; `scripts/release/checks.mjs:14-15` | V | B | M | WP-11 |
| G46 | No reference-only case on the job page | E9 | WebApp `e2e/` | V | D | S | WP-11 |
| G47 | Copy that promises what nothing delivers | E9 | WebApp `betaPolicyScreeningContract.ts:327`; `taskLifecycleNotifications.ts:176`; `Privacy.tsx:99` | I | P | S | WP-14; `Privacy.tsx` with O-07 |

Leads from the brief that the recon found wrong or stale:

- A finished estimate on the job page is covered by
  `e2e/site-private-link-reliability.spec.ts`, which runs in CI.
- Navigation is no longer "held here": intake and screening for it merged
  (Pipeline #3453, #3481, #3486; WebApp #1233, #1243).
- Main CI was green on all 14 pushes of 11-09; no Rekor or download failure
  appeared from 11-04 to 11-09.
- Retiring a scene leaves G1 rows on disk, but a withdrawal deletes them.
- Tracker rows B-07, C-07, D-01, D-04, E-01 and E-06 carry 11-02 or 11-03
  figures that no longer hold.

## 4. Owner list

Ordered by what each unblocks. None of these is applied by this plan.

| ID | Decision or action | Recommendation | Unblocks |
|---|---|---|---|
| O-01 | The November account limit: $136.43 of $200 was spent by 11-09. | Decide before inviting anyone. Either invite from 2026-12-01, or set the limit for the invited period at fixed costs (about $110 a month: host rent and Google Cloud) plus $60 for each invited job (the cost model's figure with screening). For 5 jobs that is about $410. Also decide whether the library-aisle job keeps spending from this month's limit (O-15). | E1, E4: any invited job finishing in November |
| O-02 | The work volume: 3.55 GB against a 15.8 GB floor. | Approve retiring orphan scratch (406 directories, 96.8 GB) once WP-22's census shows nothing refers to them. If that takes longer than two days, grow the volume by 200 GiB (about $20 a month; declined on 11-02). | E7; every launch |
| O-03 | Where data derived from real clips may live for the replay set. | Synthetic stand-ins in the public repo now (WP-02). Redacted JSON records of retained jobs in a private bucket read by the nightly lane. Never video, frames or meshes. | E3 (a) with real jobs; G13 |
| O-04 | The invitee list (D-13). | Send the emails or domains; set them in Render as the allowlist that WP-04 reads. | E9 (a) |
| O-05 | On-call primary and backup (D-24), and one test page to both. | Name two people before the first invitation. | E8 (c) |
| O-06 | Revoke the four exposed keys (Lindy, Perplexity, Firecrawl, Render) and record provider and time (G-05). | Do it this week; the repos are public and the keys stay in history. | E5 (f) |
| O-07 | Counsel status before 2026-11-15 (G-01). | Add three questions to the packet: the license of the two stock clips (Pexels terms section 8), the footage retention windows, and the "booking records" line in the Privacy page. | E5 (e); G47 |
| O-08 | Per-address intake cap: 20 a day against a global 5. | Set it to 2. | E9 (b) |
| O-09 | App Check. | Set `VITE_FIREBASE_APPCHECK_SITE_KEY`, run in monitor mode for a day, then enforce. | E9 (c) |
| O-10 | Deploy the `blueprint-uptime-probe` Render cron (D-02). | Deploy it and confirm one run in its log. | E8 (d) |
| O-11 | A window for the rollback rehearsal and the game day. Both retire in-flight attempts. | Hold both in one deploy window, after the library-aisle job has ended or paused. | E6 (f), E8 (e) |
| O-12 | A deadline for a money wait. | A job waiting on a spend approval for 48 hours ends as held for budget, and says so. | E1 (a) |
| O-13 | The reconciliation tolerance. | The larger of $5 or 10% per provider per month; outside it, a warning, never a refusal. | E4 (b) |
| O-14 | Two host drills: the $200 limit (a dated override below the month's spend, then reverted) and the withdrawal deletion on a synthetic job. | Approve both; each costs $0. | E4 (c), E5 (b) |
| O-15 | The library-aisle job has taken most of the work since 11-06 (levelling v10) and has no valid episode. | Give it one more run. If that run has no valid episode, end it as held with a true sentence and move its lessons into the replay set. | E3; the month's money (O-01) |
| O-16 | Host size (D-11): 4 CPUs and 8.3 GB with no swap. | Keep the host; merge WP-18's memory caps first and re-read the kills after 7 days. Resize only if kills continue. | E7 (b) |
| O-17 | Turning on automatic deletion for the 14-day temporary class. | Run WP-21 in report-only mode for a week, then approve. | E5 (d) |

One decision the brief delegated, taken here: **the real deploy gate for the
beta is main CI's supply-chain attestation**, which is what the door enforces
and what the owner's standing permission describes. WP-15 rewrites
`CI_REQUIRED_CHECKS.md` to say so, and keeps the Full Test Lane promotion run
for production deploys.

## 5. Work packages

One package is one pull request. Order is by beta risk retired per day of
work. "Check" names the fresh verifier a package needs because it touches
money, privacy, rights, deploys or job state.

Dependencies: WP-10 after WP-03 (same terminal records). WP-14 after WP-03 and
WP-06 (the expiry date and the same outcome files). WP-15 after WP-01 (same CI
files). WP-16 after WP-04 (same intake routes). WP-17 after WP-15 (the gate
document). WP-12 is independent of WP-06 (different repo, same reasons).

| ID | Wave | Repo | Goal | Serves | Check | Size |
|---|---|---|---|---|---|---|
| WP-01 | 1 | Pipeline | Cut the control-plane import of the hot lane, and run the import-closure tests on pull requests | E6 a, b | no | M |
| WP-02 | 1 | Pipeline | A committed replay set of synthetic job records for two job kinds, a runner test, and a nightly step | E3 | no | L |
| WP-03 | 1 | Pipeline | A preparation whose grant has expired ends with a terminal outcome by itself | E1 a | job state | M |
| WP-04 | 1 | WebApp | An invitee allowlist on every route that starts a job, enforced once the list is set | E9 a | money | S |
| WP-05 | 2 | Pipeline | The paid-call verifier covers OpenAI, Gemini, World Labs and SAM call sites | E4 a | money | M |
| WP-06 | 2 | WebApp | An early hold shows one held sentence on the page and in the email; task kind becomes an open display string | E2 b | no | S |
| WP-07 | 2 | Pipeline | Normalize every clip before the first preparation stage; keep the original | E7 c | no | M |
| WP-08 | 2 | Pipeline | An offline withdrawal drill in CI, an automatic deletion plan after acknowledgement, the uncovered roots scanned, a runbook section | E5 a, c | privacy | M |
| WP-09 | 3 | Pipeline | Vast credit admission subtracts open holds; a test for `max_paid_attempts` on configuration reservations | E4 d, e | money | M |
| WP-10 | 3 | Pipeline | A preparation with an expired lease and a dead process becomes `failed_retryable` and releases its reservation | E8 a, E1 | job state, money | M |
| WP-11 | 3 | WebApp | CI runs the three intake suites; job-page and intake files get the full scope; a reference-only e2e case | E9 d | no | M |
| WP-12 | 3 | Pipeline | Record seconds and dollars before a hold; the clean plate stops at $0 for a held job | E2 a, c | money | M |
| WP-13 | 4 | Pipeline | A $0 drill of the $200 limit on the real admission code; a month-scoped reconciliation report | E4 b, c | money | M |
| WP-14 | 4 | WebApp | Non-terminal sentences carry "ends by" from the grant; remove two promises nothing delivers | E1 d, E9 e | no | S |
| WP-15 | 4 | Pipeline | Runner-dependent tests made deterministic; CI caches ffmpeg and the browser; one deploy gate in the documents | E6 c, d | deploys | M |
| WP-16 | 4 | WebApp | The intake cap on signed-in job starts; the World Labs admin routes refuse or pass an admission | E4 a, E9 b | money | M |
| WP-17 | 5 | Pipeline | The door refuses a deploy while a website or Atlas stage is paid, except the D-08 exceptions | E6 e | deploys, money | M |
| WP-18 | 5 | Pipeline | Memory caps on the heavy units that sum to at most 6.5 GB | E7 b | deploys | S |
| WP-19 | 5 | Pipeline | Storage-class and unclassified-root alerts warn and no longer page | E8 b | no | M |
| WP-20 | 5 | Pipeline | A bounded retry of an Atlas preflight refused only for a stale spend report | E1, E4 | money | S |
| WP-21 | 5 | Pipeline | A daily retention report for the 14-day temporary class; deletion stays off until O-17 | E5 d | privacy | L |
| WP-22 | 5 | Pipeline | A reference census that says which orphan scratch nothing refers to; no deletion | E7 d | no | M |

Not packaged, and why:

- Automatic Atlas retry and preparation resume beyond the dead-process and
  stale-report cases (G02): the general cause is not yet named; it needs the
  replay set (WP-02) to find it without deploying.
- Release succession that keeps unchanged preparations across a deploy (G31):
  large; WP-17's hold removes the paid loss first.
- Coverage and subject holds before the scene purchase (G08): after WP-12
  measures what a hold costs.
- Provider-side deletion at World Labs (G23): needs contact with an outside
  company, which is the owner's.

## 6. Progress

Updated after every wave. Last update: 2026-11-10 04:00Z. Eight packages merged: WP-01, 04, 05, 06, 07, 08, 11, 15.
Status: merged, PR (open, with its CI result), or deferred with a reason. "CI"
counts the checks that ran; checks the workflow skipped are not counted.

| ID | Status | Pull request | CI | Note |
|---|---|---|---|---|
| WP-01 | merged | [#3517](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3517) | green, 3 passed | The leak came from one import added by #3488. The import-closure tests now run on every pull request (about 25 s). |
| WP-02 | PR open, CI re-running | [#3515](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3515) | was red on one full-suite shard (the 180 s collection limit, G58); updated from `main` after #3523 fixed that limit, CI re-running | 5 synthetic records, 2 job kinds, 7 s. Covers the downstream replay only. Merge when the re-run is green. |
| WP-03 | PR open, verifier approved, merge held | [#3516](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3516) | green, 4 passed | Merge held on purpose: see O-18. The first version was refused by its verifier (it ended extended and renewed jobs); the second asks the WebApp and ends only on its refusal. |
| WP-04 | merged | [WebApp #1245](https://github.com/ognjhunt/Blueprint-WebApp/pull/1245) | green, 9 passed | The first version was refused by its verifier (3 bypasses). Enforced only once `BLUEPRINT_INVITEE_ALLOWLIST` is set. |
| WP-05 | merged | [#3518](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3518) | green, 16 passed | Second verifier approved. 109 call sites: 36 admitted, 24 not paid, 49 can spend with no admission. None of the 49 is reachable from an automatic website job without an operator flag or environment opt-in; DreamLabs is reachable from screening only if an enabled Dream candidate and its credential are configured (WP-23). |
| WP-06 | merged | [WebApp #1246](https://github.com/ognjhunt/Blueprint-WebApp/pull/1246) | green, 9 passed | Two verifier passes. The one red `test` was a real race in the job workspace form (a late reset could erase the first keystroke); fixed in the same pull request. |
| WP-07 | merged | [#3520](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3520) | green, 3 passed | Verifier approved. New clips are normalized to at most 1920 px and 30 fps; jobs in flight keep their original. Operator note: O-19. |
| WP-08 | merged, split | [#3519](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3519) | green, 3 passed | Fourth verifier pass approved the split version: the offline drill, the receipt's policy copy classes, unresolved roots, the runbook section. The automatic deletion plan failed three reviews and was removed; it continues as WP-24. |
| WP-09 | deferred | | | Not started: this session reached its turn limit after wave 2. Nothing blocks it. Add to it: a planned screening campaign can still reserve for a withdrawn site (G52). |
| WP-10 | deferred | | | Not started (turn limit). Depends on WP-03, which is not merged. |
| WP-11 | merged, in part | [WebApp #1247](https://github.com/ognjhunt/Blueprint-WebApp/pull/1247) | green, 9 passed | Full release scope for 28 job-page and intake files, and a reference-only case. The three intake suites are NOT in CI: all three fail on `main` (G59, O-20). |
| WP-12 | deferred | | | Not started (turn limit). Nothing blocks it. |
| WP-13 | deferred | | | Not started (turn limit). The host drill also needs O-14. |
| WP-14 | deferred | | | Not started (turn limit). Depends on WP-03 and WP-06. Add to it: the `authority_ended` sentence (G51). |
| WP-15 | merged | [#3523](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3523) | green, 28 check runs, none failed | Verifier approved. Suite collection from about 185 s to about 45 s; one deploy gate per deploy kind in the documents; a `/proc` race fixed; ffmpeg and browser caches. Left: the `registered_feature` nightly failure needs a production budget change (G60). |
| WP-16 | deferred | | | Not started (turn limit). Add to it: `/v1/creator/captures` is outside the allowlist (G53). |
| WP-17 | deferred | | | Not started (turn limit). Depends on WP-15. |
| WP-18 | deferred | | | Not started (turn limit). Its merge needs the library-aisle session's agreement, because a lower memory cap can kill a stage that survives today. |
| WP-19 | deferred | | | Not started (turn limit). Nothing blocks it. |
| WP-20 | deferred | | | Not started (turn limit). Nothing blocks it. |
| WP-21 | deferred | | | Not started (turn limit). Deletion stays off until O-17 either way. |
| WP-22 | deferred | | | Not started (turn limit). The most urgent of the deferred packages: O-02 waits on it. |
| WP-23 | new, deferred | | | Put the 49 unadmitted API-priced call sites (WP-05's list) behind the existing account-limit check, or refuse them when no reservation exists. Serves E4 (a). |
| WP-24 | new, deferred | | | Produce the Pipeline's deletion plan after a withdrawal with no operator, run OUTSIDE the intake service. Work so far and three reviews' findings are on branch `claude/inspiring-turing-xbkcyh-wp-24`. Serves E5 (c). |

Gaps found while doing the work, added to the register:

| # | Gap | Crit. | Evidence | Goes to |
|---|---|---|---|---|
| G48 | 49 call sites can spend on OpenAI, Gemini, World Labs, Anthropic or another model host with no admission. Of the first 30, none is reachable from an automatic website job; the other 19 are unchecked | E4 | `scripts/paid_api_call_sites.json` in #3518 | WP-23 |
| G49 | Anthropic calls, the paid `codex` command and one Agents SDK client were invisible to the verifier | E4 | `capture_enrichment_llm.py:195,314-340`; `task_object_agent_session.py:118` | WP-05 rework |
| G50 | Three states still have no path to an end at expiry: the assessment wait, a completed job held for inputs, a job with no grant file | E1 | WP-03 executor's report | open, after WP-03 |
| G51 | The site is told "Processing is on hold while our team checks its authorization" for a job that has ended | E1, E9 | WebApp `taskStatusProjection.ts:599` | WP-14 |
| G52 | A planned screening campaign can reserve its budget for a withdrawn site; the launch refuses before any spend | E5, E4 | `beta_policy_screening_coordinator.py:305,909` | WP-09 |
| G53 | `/v1/creator/captures` is outside the invitee allowlist; whether it can cause Pipeline spend is unknown | E9 | WebApp `creator.ts:490` | WP-16 |
| G54 | The WebApp cannot show the deletion plan (its receipt schema is strict), and no door request runs `--apply` | E5 | WebApp `websiteCaptureWithdrawal.ts`; `scripts/operator_door.py` | open; O-14 |
| G55 | The replay set covers one of four replay tools, and no real job | E3 | #3515 | after O-03 |
| G56 | Normalization leftovers: a killed encode leaves a scratch folder, a failing encode runs twice, the one-hour limit may be short for long 4K 60 fps clips | E7 | #3520 verifier | polish |
| G57 | None of the nine uncovered storage roots has a per-site binding, so a withdrawal cannot find a site's files in them | E5 | #3519 | open |
| G58 | Collecting the whole test suite takes about 185 s on `main` against a 180 s limit, so one full-suite shard fails on any pull request that runs it, and the nightly lane is at risk | E6 | `tests/test_cpu_env_contract.py::test_canonical_cpu_env_collects_without_errors`; #3515 shard 11 | WP-15 |
| G59 | The three intake browser suites fail on `main`: 7 cases assert headings the pages no longer show, and 15 backend cases hit the 5-a-day intake cap, which the local launcher cannot raise | E9 | WebApp `playwright.intake*.config.ts`; `scripts/qa/reliability-local-app.mjs` | open; O-20 |
| G60 | The nightly `registered_feature` test fails when the runner is busy: one 20,000-entry budget covers every process's file-descriptor listing | E6 | #3523 report; `reference_entries_limit` | open: needs a production budget change |

Added to the owner list:

| ID | Decision or action | Recommendation | Unblocks |
|---|---|---|---|
| O-18 | Merge WP-03 ([#3516](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3516))? After it deploys, a job that is silent and whose grant the WebApp says has expired is ended within about two hours, permanently. Today such a job waits, which leaves time for the operator renewal script. | Merge it once the library-aisle job has ended, or its driver confirms the job's grant is current. Renew a grant before it expires, not after. | E1 (a) |
| O-19 | Operator note, no decision: since #3520 an owner row in `BLUEPRINT_WEBSITE_ATLAS_AUTHORIZATIONS_JSON` for a new job must name the normalized clip's digest (`pipeline/source_video/normalization.json`). A row naming the upload's digest does not match, and the job falls back to the automatic grant. | Read the digest from the job's normalization record. | Owner-row grants for new jobs |
| O-04 (note) | When you set the invitee list, use exact emails, not domains: the public form trusts a typed email. Tenant users, robot teams and people invited through account invitations need listing too. | | E9 (a) |
| O-20 | Allow a test-only intake cap in the local emulator launcher (`scripts/qa/reliability-local-app.mjs`). | Yes: no real job or spend is involved there. | Running the intake backend suite in CI (E9 d) |

Criteria on 2026-11-10: none passes yet. E2 (b) is done on `main` (WebApp #1246). E5 (a) is done on `main` (#3519). E6 (c) and (d) are done on `main` (#3523). E4 (a) is done except that 49 call sites remain unadmitted (WP-23). E6 (b) is done on `main`
(#3517) and the 7-night count for E6 (a) starts with the 2026-11-10 run. E7 (c)
is done on `main` (#3520) and takes effect at the next deploy. E9 (a) is ready
on `main` (WebApp #1245) and waits on the list (O-04).

## 7. How a resumed session continues

1. Read this file, then the newest dated entry of the tracker.
2. Re-read `door status` before trusting section 1's two figures.
3. Take the next wave from section 6: at most 4 packages at once, each in its
   own worktree and on its own branch from the latest `main`.
4. Give a fresh verifier the diff and the acceptance test of every package
   marked in the "Check" column, before it merges.
5. Update section 6 and the owner list, and add one line to the tracker.
