# Beta launch tracker (November 2026)

Goal: make Blueprint ready for its invited beta, then its product launch. Every
item keeps the ID from the 2026-11-02 goal brief. Times are UTC unless marked.

Status: **done** (verified, with evidence) · **in progress** · **owner** (waits
on an owner decision or action) · **next deploy** (merged, ships with the next
batched Pipeline deploy) · **not started**.

Last updated: 2026-11-02 17:10Z. Pipeline live: `ed6d0befb` (deployed 15:18Z by the cloud session). Next deploy (window 15:00 UTC, D-08) carries #3260, #3261, #3266, #3267 and #3274 once merged. WebApp live: `282ba6452`.

## Operating notes (2026-11-02)

- A second operator (the cloud session) uses the shared `cloud-scene-runs` door token. It deployed `b631b23fa` at 14:05:57Z and `ed6d0befb` at 15:18Z. This session makes no Pipeline deploys while it drives capture-447; merged fixes ship with the next deploy. At 15:32Z no paid launch lock was held.
- Five `footage_before_account` emails, the first ever, went to the owner at 14:05–14:08Z about old guest-uploaded test jobs. Each job gets one. The email promises "We email you when that changes", but nothing will change for those jobs (wording follow-up under F-03).

## Owner decisions (2026-11-02)

The owner delegated every decision to the recommendations in the brief, except
where an action strictly needs the owner. That covers D-03, D-04, D-05, D-07,
D-08, D-12, D-13, D-15, D-16, D-17, D-19, D-20, D-21, D-22 and D-23.

about 15:45Z, the owner, on the Privacy/Terms correction and the doctrine line: "change it so it doesn't say this anymore" (G-04).

About 16:35Z, the owner: the corrected Privacy/Terms take effect today (G-04), and no volume growth for now (D-10).

Money and data-deletion items still need an explicit, specific approval:

| Ask | Amount or scope | Status |
|---|---|---|
| D-02: raise capture-447's job ceiling | $70 → $110 (Render `BLUEPRINT_WEBSITE_ATLAS_AUTHORIZATIONS_JSON` row) | owner |
| D-10: grow the work volume | +200 GiB (about +$20/month). **Declined for now** (owner, about 16:35Z: "NO - dont grow rn"). Reclaim instead (E-01). At the floor, launches refuse at $0 and resume by themselves (C-02) | owner |
| D-10: evict the SAM-3.1 preparation cache | 22 directories, 10.1 GB, idle 47–51 days, rebuildable. `#2380` requires a separate owner decision | owner |
| D-09: Vast auto-recharge | owner account setting; suggested: recharge to $50 below $25 | owner |
| D-01 / D-06 / B-04 defaults | amounts come from the B-03 cost model | not started |
| D-11: host resize | deferred: remove CPU waste first (E-02) | — |

Facts and actions only the owner has: counsel status before 2026-11-15 (G-01),
the on-call backup (D-24), the invitee allowlist (D-13), credential rotation
(G-05), and repo visibility (both repos are public; making them private moves CI
onto paid Actions minutes).

## A. Prove one website job end to end

| ID | Status | Evidence / next step |
|---|---|---|
| A-01 | owner | **Driven by the cloud session (branch prefix `bold-darwin-iketye`)**. It deployed `b631b23fa` at 14:05:57Z and `ed6d0befb` at 15:18Z (the active release at 15:32Z), and merged [#3259](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3259), which sizes the CPU prestage by what it writes. This session stood down on capture-447 at 14:45Z to avoid two drivers. History: the export-only relaunch was refused before allocation at $0 (13:39:42Z, `scene_configuration_provider_output_disk_budget_exceeded`: needs 35.19 GB on the work volume, had 8.06 GB). The scene intent was then revoked at 14:08:33Z. Root cause (cloud session, WebApp #1182): the brief-reading lane, idle since 10-24, drained its backlog from 14:03Z, and one reading folded an inferred answer into the unconfirmed brief. That moved the task-context digest the scene grant is bound to (`website_scene_sponsorship_changed` → revoke). |
| A-02 | not started | Runs in the same GPU session after stage 3. The 10-31 destination gates are fixed on main (#3136; `ef276779a` for the stage-5 bind tolerance; `ae0e4ae2b`). |
| A-03 | owner | WebApp #1168 merged 14:04Z (deploy pending CI). capture-447 still needs the D-02 ceiling row ($110) before its screening plan can fit. |
| A-04 | not started | Host has never created the team-scene-packet status or rigid-probe roots. |
| A-05 | not started | Pre-flight (read-only):<br>• The 10-26 canary's 8.4 GB output failed its single signed PUT (probably the 5 GB single-upload limit). The automatic pinned-SSH fallback (`vast_provider_output_recovery`, 900 s) covers the screening lane, which writes about 4.2 GB per candidate. Multipart upload remains a SHOULD.<br>• Fixed: candidate order. [#3260](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3260) (D-20) funds π0.5-DROID, Cosmos3-Edge-DROID and GR00T-N1.7-DROID first when there is no intake ranking.<br>• Fixed: the job ceiling's native over-count. WebApp [#1184](https://github.com/ognjhunt/Blueprint-WebApp/pull/1184) bounds it by the budget the Pipeline enforces; it had counted about $361 against a $70 cap.<br>• Still open: ingest needs a confirmed `siteTaskBriefs/<job>`. Root has 4.58 GB of headroom for screening attempt folders (13:19Z). |
| A-06 | not started | After A-05 and C-01…C-07. |

## B. Velocity, cost model, money correctness

| ID | Status | Evidence / next step |
|---|---|---|
| B-01 | in progress | Phase 1 merged: [#3256](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3256), `python -m blueprint_pipeline.task_evaluation_downstream_replay` replays completion → readiness → team packet → probe enqueue → screening plan offline, with paid calls, network and WebApp stubbed (18 tests). Phase 2 (stages 3–6) needs a Linux mount namespace (absolute `/workspace` paths); the cloud session's #3253 covers the stage-3/4 ARTISAN export replay. |
| B-02 | not started | |
| B-03 | done | [`2026-11-cost-model.md`](2026-11-cost-model.md): capture-447 measured spend $25.08 vs about $83 held across caps (scene holds 2.8× actual). Modelled job: p50 about $27 (plus unread upstream), p90 about $44, $59 with screening. Recommended defaults are listed in section 7; they need owner money approval (D-01/D-06). Found by code reading, not runtime-tested: `max_paid_attempts` never binds website scene-configuration reservations. |
| B-04 | owner | Accounting policy change; needs owner approval once implemented. |
| B-05 | done | [#3265](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3265) merged: a "Spend limits" table in `AGENTS.md` lists 18 money ceilings with the file that sets each. Every code value was checked against `main`. Values that live only in Render or on the host are marked unverified. The B-03 defaults are not applied; they wait for the owner (D-01/D-06). |
| B-06 | not started | |
| B-07 | not started | `door status` 13:19Z: internal ledger $165.55; billing reconciled DigitalOcean $325.38 + Vast $455.88. The spend guard's RunPod inventory call succeeds (no errors in its journal over the last hour), so the audit's "1,065 failing RunPod calls a day" does not reproduce there; its source is unknown. |
| B-08 | in progress | Building alerts for OpenAI, through the Organization Costs API client `openai_cost_authority.py` already uses (OpenAI exposes no balance); Gemini, through the GCP monthly guard, paging before its $100 ceiling; and World Labs, through recorded spend, since no balance endpoint was found. Each follows the C-07 pattern: missing credentials only warn, and nothing blocks a launch. Recharge settings are owner account facts. |

## C. Unattended operation

| ID | Status | Evidence / next step |
|---|---|---|
| C-01 | not started | #3196 merges after capture-447 settles. The configured-controls timer, path and service are all inactive (13:30Z). |
| C-02 | done (live) | [#3251](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3251) is in the active release `ed6d0befb`: a $0 pre-allocation disk refusal of a website scene resumes like credit and auth refusals, bounded by the 6-recovery ceiling. The new defaults from B-03 still need owner money approval (D-01/D-06). |
| C-03 | owner | WebApp #1180 merged: the sponsorship TTL may be up to 7 days, the Pipeline's own window. **Owner action:** set `ttl_seconds: 604800` in Render `BLUEPRINT_WEBSITE_SCENE_SPONSORSHIP_JSON`. |
| C-04 | next deploy | Done in code. WebApp [#1190](https://github.com/ognjhunt/Blueprint-WebApp/pull/1190) is live in `8264ffd40` (deployed 16:27Z, verified at the exact SHA). Pipeline [#3266](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3266) merged 16:32Z as `d971aeb40` and ships with the next Pipeline deploy. Checked before merge: the record carries exactly the 9 fields the WebApp's strict schema expects, and `outcome_digest` uses the RFC 8785 digest that the live `status_digest` already uses. The first pass emails no outside person: all 12 scene intakes are on the owner's account. |
| C-05 | not started | |
| C-06 | in progress | Deploy policy written ([#3272](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3272), D-08): one batched deploy of `main` a day in a 15:00 UTC window; held while an external job is in a paid stage (held paid-launch lock, activation hold, or screening launching or running); outside the window only money, privacy, rights or stuck-paid-stage fixes. The door already holds deploys during a live scene activation. Not started: a door-enforced paid-stage hold, release succession reusing unchanged preparations, and automatic keep-last-3 release retirement. |
| C-07 | done (live) | [#3257](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3257) is in the active release `ed6d0befb`: a Vast balance under $11 (two launches' $5 compute plus the $1 reserve) pages hourly, with the balance in the alert. Still open: a credit reservation across concurrent launches, needed once dispatcher concurrency is above 1. Auto-recharge is the owner's Vast setting (D-09). |

## D. Observability and operations

| ID | Status | Evidence / next step |
|---|---|---|
| D-01 | done (live) | #3226 is in the active release `ed6d0befb`: stall paging on the capacity alert route (D-12 approved by delegation). Follow-up: page daily on cap exhaustion. |
| D-02 | in progress | WebApp [#1176](https://github.com/ognjhunt/Blueprint-WebApp/pull/1176) merged: the external uptime probe also checks the door `/healthz`. Pending: the WebApp deploy, then a manual deploy of the `blueprint-uptime-probe` Render cron (`autoDeploy: false`). Capacity-tick freshness is a follow-up. |
| D-03 | not started | |
| D-04 | next deploy | [#3249](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3249) is live. The next check then failed on every run: `preflight_summary_private_parent_unsafe` (journal 15:31Z), because the private report directory was created long ago by another account. The door shows it as `preflight_summary_symlink_or_open_refused`, since no summary is written. [#3267](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3267) merged 15:35Z: the unit claims the directory (root, 0711) before each run. It takes effect at the next Pipeline deploy. |
| D-05 | in progress | Seven preparation-status deliveries have been pending since 10-28, about 21:30Z, with 188–193 failures each. The cause is local: `PreparationStatusUnavailable` comes from `read_preparation_status` before any WebApp call. Six never reached the WebApp; capture-36ad8b1d made 113 WebApp attempts first. All seven are old test captures, including guest captures. They retry every 30 min forever, so `overloaded` stays true. Next: find the failing check (it needs a run on the host), park permanently unavailable deliveries with a typed reason, and page when the oldest live delivery is over 1 h. |
| D-06 | in progress | `blueprint-provider-billing-reconciler.timer` has been inactive since 2026-09-30 14:14Z, though enabled and in the deploy list. The reconciler still runs, because `blueprint-gpu-spend-guard.service` `Wants=` it: 31 runs an hour (see E-02). `launch-supervisor.timer` (optional OpenAI supervision) and `capture-reconstruction-dispatcher.{timer,path}` are inactive. To decide per unit: re-armed, retired or documented manual. `reset-failed` on the transient failed units is not done; another operator uses the host. |
| D-07 | owner | Runbooks done: [#3265](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3265) adds `docs/runbooks/beta-operations.md` (stuck job, Vast credit, full work volume, door or host down, deploy rollback). Still needs the on-call backup's name (D-24). |
| D-08 | not started | |
| D-09 | not started | |

## E. Capacity, infrastructure, security

| ID | Status | Evidence / next step |
|---|---|---|
| E-01 | in progress | Owner, about 16:35Z: no volume growth for now (D-10 declined). [#3274](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3274) (auto-merge): an hourly timer runs the activation worker as `blueprint`, never root. It retires a claim only when the claim is older than 255 + 15 min, no worker holds its lock, its envelope is valid and it has no result; a door hold on the queue still stops every run. After the next deploy, the first timer run retires the 10-31 claim and the next GC tick reclaims about 14.8 GB. The 11 capture-447 activation pins: 6 `run_hot` release after the 2-day hot window, 2 `run_not_sealed` only at pin expiry (about 11-30), 2 were `pin_young`, and 1 `activation_result_not_stale` waits until about 11-09. The GC's stranded-rows phase never runs on the host because it lacks its running commit; enabling it would add root writes into `blueprint`-owned queues, so it moves off root first (follow-up). Orphan scratch (26 GB) needs the owner's OK. |
| E-02 | in progress | [#3278](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3278) (in review). The runtime-env guard reuses a ready verdict only when nothing it reads has changed, for at most 15 min; a verdict that repaired the catalog, or any refusal, is never reused, and the ledger and lock-slot steps that write still run on every start. Idle Pub/Sub listener ticks no longer import the dispatcher. The billing refresh exits at once while its last export is under 10 min old (the spend guard accepts 24 h). The preflight runs every 30 min, inside the door's 1 h freshness limit. Expected saving: about 1,650–1,950 CPU-s/h, about a third of the 15:00–16:00Z sample. A resize to 8 vCPU (D-11) stays with the owner. |
| E-03 | not started | |
| E-04 | owner | Credential rotation (G-05). |
| E-05 | not started | |
| E-06 | in progress | Both nightly Full Test Lane failures since 2026-11-01 are fixed on main: the control-plane import leak through `paid_provider_job_scope` ([#3261](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3261)), and the door's G1 contract refusing the new `stance_reach` assumption ([#3263](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3263), issue #2236; takes effect at the next `upgrade-door`). Next: 7 green nightlies, and a documented beta deploy gate. |
| E-07 | not started | |

## F. Website: result delivery, truthfulness, journey

| ID | Status | Evidence / next step |
|---|---|---|
| F-01 | done | WebApp [#1179](https://github.com/ognjhunt/Blueprint-WebApp/pull/1179) is live: one deduplicated email per terminal screening outcome (estimate, reference_only, held, infeasible, no_match). It re-checks the page at dispatch, is cancelled on consent withdrawal, and carries the simulation disclaimer. No backfill (0 production campaigns). |
| F-02 | done | The job-evidence index is READY. Created `creatorCaptures(creator_id, created_at desc)` (READY 14:10Z) for the website's capturer page. 16 other declared indexes serve ops, growth and payout routes and were never deployed (H-03). |
| F-03 | done | WebApp [#1177](https://github.com/ognjhunt/Blueprint-WebApp/pull/1177) merged and deployed (`ba27cd6e4`, 14:33Z): one sentence per outcome on both the headline and the body; reference-only runs are not called estimates; no raw codes, budgets or operator text reach sites; exhausted allowances now read as ended (the prefix bug). Follow-up: the `footage_before_account` email promises "We email you when that changes". |
| F-04 | in progress | WebApp #1181 merged: the default intake cap is 5 a day. **Owner checks:** Render must not pin `BLUEPRINT_INTAKE_DAILY_LIMIT=50`. The allowlist needs the invitee list. App Check `monitor` needs `VITE_FIREBASE_APPCHECK_SITE_KEY`. |
| F-05 | not started | |
| F-06 | in progress | Sites are recorded as attesting statement `2026-09-18.v1` ("…a scene robot teams can evaluate against"), which the start form does not show. Building: a v2 sentence taken from the Terms in force, shown verbatim where it is attested. The Pipeline (`capture_original_owner_observer.py`, which accepts only v1) is changed to accept v1 and v2 and deploys first; the WebApp then sends v2. |
| F-07 | done (mitigated) | WebApp #1068 (10-18) accepts footage only for a saved job, so a new guest upload cannot strand mid-run. Old guest jobs get one `footage_before_account` email; the first five were sent 14:05–14:08Z, all to the owner's test jobs. |
| F-08 | in progress | WebApp [#1185](https://github.com/ognjhunt/Blueprint-WebApp/pull/1185) documents the 10 website-beta settings in `render.optional.env.example`. Readiness checks are a follow-up. |
| F-09 … F-11 | not started | |

## G. Privacy, rights, legal

| ID | Status | Evidence / next step |
|---|---|---|
| G-01 | owner | Counsel status before 2026-11-15 is still open. Add to the counsel packet: Terms §12 and the Privacy Policy's "Changes" section promise an email to account holders before a material change. On 2026-11-02 the owner made the corrected documents effective at once and no one was emailed (G-04). |
| G-02 | in progress | WebApp [#1178](https://github.com/ognjhunt/Blueprint-WebApp/pull/1178) is live. Pipeline [#3262](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3262): all five review findings are fixed (`9d32897c6`). One stop signal per allocator, and teardown ignores repeats. Transport blips are retryable; a definite revoked, changed or expired answer still blocks. Reconcile never voids a finished result. Retries publish `beta_screening_waiting_for_capacity`, so the site reads "Waiting for room in our simulator". An independent re-review is in progress; then a withdrawal drill. |
| G-03 | superseded | D-16 (owner, 14:15Z): no blurring or person screening. The residual-person check in #1980 is moot; its stage-reuse part can be revisited separately. |
| G-04 | done | Owner, about 15:45Z and 16:35Z: correct the text, effective today. WebApp [#1183](https://github.com/ognjhunt/Blueprint-WebApp/pull/1183) (the corrected wording) and [#1192](https://github.com/ognjhunt/Blueprint-WebApp/pull/1192) (versions `2026-11-02.1`, effective at once) are live in `282ba6452`, verified in the site bundle. People are not removed from footage, faces are blurred only in the frames sent to World Labs, and Meta is listed. An acceptance of the earlier November 15 pairs no longer counts; the identical `2026-11-15.4`/`.5` text does. The previous versions stay frozen at `/terms/previous` and `/privacy/previous`. Doctrine line replaced ([#3269](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3269), WebApp [#1188](https://github.com/ognjhunt/Blueprint-WebApp/pull/1188)). |
| G-05 | owner | Rotation is the owner's action. Both repos are public (verified 13:30Z). |

## H. Cleanup and consistency

| ID | Status | Evidence / next step |
|---|---|---|
| H-01 | done | Pipeline [#3268](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3268) archives four obsolete launch gates (one moved; three keep their paths with a banner because tests read them). WebApp [#1187](https://github.com/ognjhunt/Blueprint-WebApp/pull/1187) brings the beta guides and README to the 10-28 model. Found false but needing code or an owner call: robot-team copy says 200 episodes per evaluation (screening plans at most 10 per candidate); `main` still has the delivery-proposal screen and booking route (#1143, owner); `docs/beta/ROBOT_TEAM_SUPPLY_AGREEMENTS.md` describes the old evaluate-and-deliver rule. |
| H-02 | done | 126 PRs and 3 issues closed (D-23). Also: WebApp #1142 was carried as #1178; #1103 was closed as superseded (main no longer caps Atlas requests, per the owner's 10-30 decision). #3051 is a 59-file tangle that overlaps #3042, so it is left open; its core, truthful worker status, already shows in `door status`. |
| H-03, H-04 | not started | |
