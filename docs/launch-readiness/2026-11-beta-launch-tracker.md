# Beta launch tracker (November 2026)

Goal: make Blueprint ready for its invited beta, then its product launch. Every
item keeps the ID from the 2026-11-02 goal brief. Times are UTC unless marked.

Status: **done** (verified, with evidence) · **in progress** · **owner** (waits
on an owner decision or action) · **next deploy** (merged, ships with the next
batched Pipeline deploy) · **not started**.

Last updated: 2026-11-02 16:00Z. WebApp: #1183 and #1187 merged 15:50Z, deploys follow CI.

## Operating notes (2026-11-02)

- A second operator (the cloud session) uses the shared `cloud-scene-runs` door token. It deployed `b631b23fa` at 14:05:57Z and `ed6d0befb` at 15:18Z. This session makes no Pipeline deploys while it drives capture-447; merged fixes ship with the next deploy. At 15:32Z no paid launch lock was held.
- Five `footage_before_account` emails, the first ever, went to the owner at 14:05–14:08Z about old guest-uploaded test jobs. Each job gets one. The email promises "We email you when that changes", but nothing will change for those jobs (wording follow-up under F-03).

## Owner decisions (2026-11-02)

The owner delegated every decision to the recommendations in the brief, except
where an action strictly needs the owner. That covers D-03, D-04, D-05, D-07,
D-08, D-12, D-13, D-15, D-16, D-17, D-19, D-20, D-21, D-22 and D-23.

about 15:45Z, the owner, on the Privacy/Terms correction and the doctrine line: "change it so it doesn't say this anymore" (G-04).

Money and data-deletion items still need an explicit, specific approval:

| Ask | Amount or scope | Status |
|---|---|---|
| D-02: raise capture-447's job ceiling | $70 → $110 (Render `BLUEPRINT_WEBSITE_ATLAS_AUTHORIZATIONS_JSON` row) | owner |
| D-10: grow the work volume | +200 GiB (about +$20/month); the owner resizes it in DigitalOcean, then runs `resize2fs` on the host. **Urgent: 2.89 days to the floor at 15:32Z** | owner |
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
| B-08 | not started | |

## C. Unattended operation

| ID | Status | Evidence / next step |
|---|---|---|
| C-01 | not started | #3196 merges after capture-447 settles. The configured-controls timer, path and service are all inactive (13:30Z). |
| C-02 | in progress | [#3251](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3251) (auto-merge): a $0 pre-allocation disk refusal of a website scene (all have `max_retries: 0`) now resumes like credit/auth refusals instead of ending at `retry_cap_exhausted`; bounded by the 6-recovery ceiling. Defaults from B-03 still need owner money approval. capture-447 intent: `max_paid_attempts` 16, `max_retries` 0, base $20 (raised to $70 by grants). |
| C-03 | owner | WebApp #1180 merged: the sponsorship TTL may be up to 7 days, the Pipeline's own window. **Owner action:** set `ttl_seconds: 604800` in Render `BLUEPRINT_WEBSITE_SCENE_SPONSORSHIP_JSON`. |
| C-04 | in progress | Pipeline [#3266](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3266), revised (`59223530d`): each website scene intent gets one outcome record. Blueprint faults stay fixable: `setup_repair` is non-terminal. Terminal means only withdrawn, held, one of four exhausted non-extendable caps, or the authority expired or was revoked. There is no 24-hour rule. Consequence to note: a window extension issued after expiry no longer revives a job. The WebApp half, which accepts the optional `outcome`, maps it to one state and sends one email per terminal outcome, is being built and must deploy before #3266 merges. |
| C-05 | not started | |
| C-06 | not started | |
| C-07 | next deploy | [#3257](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3257) merged: a Vast balance under $11 (two launches' $5 compute plus the $1 reserve) pages hourly, with the balance in the alert. Credit reservation across concurrent launches is still open (needed once dispatcher concurrency is above 1). Auto-recharge is the owner's Vast setting (D-09). |

## D. Observability and operations

| ID | Status | Evidence / next step |
|---|---|---|
| D-01 | next deploy | #3226 merged 14:0xZ (stall paging on the capacity alert route; D-12 approved by delegation). Follow-up: page daily on cap exhaustion. |
| D-02 | in progress | WebApp [#1176](https://github.com/ognjhunt/Blueprint-WebApp/pull/1176) merged: the external uptime probe also checks the door `/healthz`. Pending: the WebApp deploy, then a manual deploy of the `blueprint-uptime-probe` Render cron (`autoDeploy: false`). Capacity-tick freshness is a follow-up. |
| D-03 | not started | |
| D-04 | next deploy | [#3249](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3249) is live. The next check then failed on every run: `preflight_summary_private_parent_unsafe` (journal 15:31Z), because the private report directory was created long ago by another account. The door shows it as `preflight_summary_symlink_or_open_refused`, since no summary is written. [#3267](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3267) merged 15:35Z: the unit claims the directory (root, 0711) before each run. It takes effect at the next Pipeline deploy. |
| D-05 | not started | `door status` 13:19Z: `due_backlog` 1, oldest 4.5 days, not overloaded. |
| D-06 | not started | `door status` 13:19Z: 206 failed units. |
| D-07 | owner | Runbooks done: [#3265](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3265) adds `docs/runbooks/beta-operations.md` (stuck job, Vast credit, full work volume, door or host down, deploy rollback). Still needs the on-call backup's name (D-24). |
| D-08 | not started | |
| D-09 | not started | |

## E. Capacity, infrastructure, security

| ID | Status | Evidence / next step |
|---|---|---|
| E-01 | in progress | 15:32Z: `/mnt/blueprint-work` 90.6% used (29.7 GB free), **2.89 days to floor**; `/` 92.2% (12.9 GB free). Capacity `critical`; pages: `floor_within_three_days`, `reclaim_ineffective`, `orphan_scratch_large` (625 folders, 26.2 GB; the survey never grants their cleanup, so they need an owner retention decision). Refused roles: `embodiment_preparation`, `g1_checkpoint_cache`. Root cause of the 14.8 GB `queue_referenced` folder: the 10-31 capture-447 activation worker hit the unit's then-1 h `TimeoutStartSec` and was killed at 18:38:43Z (`8d8bd7b86` has since raised it to 255 min). Its claim stays in `processing/` forever, because the worker only claims `pending/` and the GC never touches `processing/`. Fix in progress: retire a claim automatically when no live worker holds its lock and it is older than the unit timeout. Also open: 11 activation pins (6.6 GB) and the owner's D-10 volume growth. |
| E-02 | not started | Load 5.2 on 4 vCPU. The preflight timer is `OnUnitActiveSec=15min`, and a run takes about 13 min. |
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
| F-06 | not started | |
| F-07 | done (mitigated) | WebApp #1068 (10-18) accepts footage only for a saved job, so a new guest upload cannot strand mid-run. Old guest jobs get one `footage_before_account` email; the first five were sent 14:05–14:08Z, all to the owner's test jobs. |
| F-08 | in progress | WebApp [#1185](https://github.com/ognjhunt/Blueprint-WebApp/pull/1185) documents the 10 website-beta settings in `render.optional.env.example`. Readiness checks are a follow-up. |
| F-09 … F-11 | not started | |

## G. Privacy, rights, legal

| ID | Status | Evidence / next step |
|---|---|---|
| G-01 | owner | Counsel status before 2026-11-15. |
| G-02 | in progress | WebApp [#1178](https://github.com/ognjhunt/Blueprint-WebApp/pull/1178) is live. Pipeline [#3262](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3262) was reviewed: no rights blocker. Fixing before merge: a stop sends two SIGTERMs, and the second can lose the run's result and strand the campaign; a transport blip becomes a terminal block at launch preflight and G1 create; reconcile can void a completed result. Then a withdrawal drill. |
| G-03 | superseded | D-16 (owner, 14:15Z): no blurring or person screening. The residual-person check in #1980 is moot; its stage-reuse part can be revisited separately. |
| G-04 | done | Owner, about 15:45Z: "change it so it doesn't say this anymore". [#1183](https://github.com/ognjhunt/Blueprint-WebApp/pull/1183) merged 15:50Z (deploy follows CI). The corrected Terms 2026-11-15.4 and Privacy 2026-11-15.5 say people are not removed from footage, faces are blurred only in frames sent to World Labs, and Meta is listed (`meta_sam31.py` uploads to `api.meta.ai`). Under the owner's 10-15 notice rule the corrected text is published now and takes effect 2026-11-15. Until then the in-force 2026-10-13.1 policy (`/privacy/previous`) still says "We remove people from the frames used as input to reconstruction". **Owner question:** make the correction effective now? The doctrine line "people in captures are screened" is replaced in the shared block and both `AGENTS.md` files ([#3269](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3269), WebApp [#1188](https://github.com/ognjhunt/Blueprint-WebApp/pull/1188), auto-merge). Runbook §3a was corrected in WebApp [#1187](https://github.com/ognjhunt/Blueprint-WebApp/pull/1187). |
| G-05 | owner | Rotation is the owner's action. Both repos are public (verified 13:30Z). |

## H. Cleanup and consistency

| ID | Status | Evidence / next step |
|---|---|---|
| H-01 | done | Pipeline [#3268](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3268) archives four obsolete launch gates (one moved; three keep their paths with a banner because tests read them). WebApp [#1187](https://github.com/ognjhunt/Blueprint-WebApp/pull/1187) brings the beta guides and README to the 10-28 model. Found false but needing code or an owner call: robot-team copy says 200 episodes per evaluation (screening plans at most 10 per candidate); `main` still has the delivery-proposal screen and booking route (#1143, owner); `docs/beta/ROBOT_TEAM_SUPPLY_AGREEMENTS.md` describes the old evaluate-and-deliver rule. |
| H-02 | done | 126 PRs and 3 issues closed (D-23). Also: WebApp #1142 was carried as #1178; #1103 was closed as superseded (main no longer caps Atlas requests, per the owner's 10-30 decision). #3051 is a 59-file tangle that overlaps #3042, so it is left open; its core, truthful worker status, already shows in `door status`. |
| H-03, H-04 | not started | |
