# Beta launch tracker (November 2026)

Goal: make Blueprint ready for its invited beta, then its product launch. Every
item keeps the ID from the 2026-11-02 goal brief. Times are UTC unless marked.

Status: **done** (verified, with evidence) · **in progress** · **owner** (waits
on an owner decision or action) · **next deploy** (merged, ships with the next
batched Pipeline deploy) · **not started**.

Last updated: 2026-11-02 14:45Z.

## Operating notes (2026-11-02)

- A second operator uses the shared `cloud-scene-runs` door token. It started `configured-controls-progression` at 13:35:32Z and deployed `b631b23fa` at 14:05:57Z (`deploy_activation_hold` none). Two drivers on one job risk collisions; the owner has been asked. This session is holding its own Pipeline deploys until that is resolved.
- Five `footage_before_account` emails, the first ever, went to the owner at 14:05–14:08Z about old guest-uploaded test jobs. Each job gets one. The email promises "We email you when that changes", but nothing will change for those jobs (wording follow-up under F-03).

## Owner decisions (2026-11-02)

The owner delegated every decision to the recommendations in the brief, except
where an action strictly needs the owner. That covers D-03, D-04, D-05, D-07,
D-08, D-12, D-13, D-15, D-16, D-17, D-19, D-20, D-21, D-22 and D-23.

Money and data-deletion items still need an explicit, specific approval:

| Ask | Amount or scope | Status |
|---|---|---|
| D-02: raise capture-447's job ceiling | $70 → $110 (Render `BLUEPRINT_WEBSITE_ATLAS_AUTHORIZATIONS_JSON` row) | owner |
| D-10: grow the work volume | +200 GiB (about +$20/month); the owner resizes it in DigitalOcean, then runs `resize2fs` on the host | owner |
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
| A-01 | owner | **Driven by the cloud session (branch prefix `bold-darwin-iketye`)**, which deployed `b631b23fa` at 14:05:57Z and opened Pipeline #3253 and WebApp #1182. This session stood down on capture-447 at 14:45Z to avoid two drivers. History: the export-only relaunch was refused before allocation at $0 (13:39:42Z, `scene_configuration_provider_output_disk_budget_exceeded`: needs 35.19 GB on the work volume, had 8.06 GB). The scene intent was then revoked at 14:08:33Z. Root cause (cloud session, WebApp #1182): the brief-reading lane, idle since 10-24, drained its backlog from 14:03Z, and one reading folded an inferred answer into the unconfirmed brief, which moved the task-context digest the scene grant is bound to (`website_scene_sponsorship_changed` → revoke). |
| A-02 | not started | Runs in the same GPU session after stage 3. The 10-31 destination gates are fixed on main (#3136; `ef276779a` for the stage-5 bind tolerance; `ae0e4ae2b`). |
| A-03 | owner | WebApp #1168 merged 14:04Z (deploy pending CI). capture-447 still needs the D-02 ceiling row ($110) before its screening plan can fit. |
| A-04 | not started | Host has never created the team-scene-packet status or rigid-probe roots. |
| A-05 | not started | Pre-flight (read-only):
- The 10-26 canary's 8.4 GB output failed its single signed PUT (HTTP 400, probably the 5 GB single-upload limit). There is still no multipart upload, and screening uses the same path (about 4.2 GB per candidate).
- With no ranking file the plan picks Cosmos3-Edge, GR00T and MolmoAct2-YAM ($12 of $15). π0.5, the one policy proven in production, is fifth.
- WebApp #1168 counts the sum of every attempt's maximum, so a job with many retries leaves $0 for screening (fix: B-04).
- Ingest needs `siteTaskBriefs/<job>`.
- Root volume has 4.58 GB above its floor for screening attempt folders. |
| A-06 | not started | After A-05 and C-01…C-07. |

## B. Velocity, cost model, money correctness

| ID | Status | Evidence / next step |
|---|---|---|
| B-01 | in progress | Phase 1 merged: [#3256](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3256), `python -m blueprint_pipeline.task_evaluation_downstream_replay` replays completion → readiness → team packet → probe enqueue → screening plan offline, with paid calls, network and WebApp stubbed (18 tests). Phase 2 (stages 3–6) needs a Linux mount namespace (absolute `/workspace` paths); the cloud session's #3253 covers the stage-3/4 ARTISAN export replay. |
| B-02 | not started | |
| B-03 | done | [`2026-11-cost-model.md`](2026-11-cost-model.md): capture-447 measured spend $25.08 vs about $83 held across caps (scene holds 2.8× actual). Modelled job: p50 about $27 (plus unread upstream), p90 about $44, $59 with screening. Recommended defaults are listed in section 7; they need owner money approval (D-01/D-06). Found by code reading, not runtime-tested: `max_paid_attempts` never binds website scene-configuration reservations. |
| B-04 | owner | Accounting policy change; needs owner approval once implemented. |
| B-05 | not started | |
| B-06 | not started | |
| B-07 | not started | `door status` 13:19Z: internal ledger $165.55; billing reconciled DigitalOcean $325.38 + Vast $455.88. |
| B-08 | not started | |

## C. Unattended operation

| ID | Status | Evidence / next step |
|---|---|---|
| C-01 | not started | #3196 merges after capture-447 settles. The configured-controls timer, path and service are all inactive (13:30Z). |
| C-02 | in progress | [#3251](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3251) (auto-merge): a $0 pre-allocation disk refusal of a website scene (all have `max_retries: 0`) now resumes like credit/auth refusals instead of ending at `retry_cap_exhausted`; bounded by the 6-recovery ceiling. Defaults from B-03 still need owner money approval. capture-447 intent: `max_paid_attempts` 16, `max_retries` 0, base $20 (raised to $70 by grants). |
| C-03 | owner | WebApp #1180 merged: the sponsorship TTL may be up to 7 days, the Pipeline's own window. **Owner action:** set `ttl_seconds: 604800` in Render `BLUEPRINT_WEBSITE_SCENE_SPONSORSHIP_JSON`. |
| C-04 | in progress | Mapped. Only `completed` stops the progression driver; everything else retries every minute forever. The WebApp's `ENDED_BLOCKERS` lists unprefixed codes while the Pipeline writes `scene_progression_*`/`scene_intake_*`, so exhausted caps show "waiting" (fix folded into the F-03 PR). `failed_terminal` is never emitted. Plan: a write-once Pipeline `outcome.json` (held/setup_repair/awaiting_spend_approval/awaiting_site/withdrawn) surfaced as an optional `outcome` on the intent status. The WebApp ships `.optional()` first (its schema is strict) and maps it to one state plus one email. |
| C-05 | not started | |
| C-06 | not started | |
| C-07 | next deploy | [#3257](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3257) merged: a Vast balance under $11 (two launches' $5 compute plus the $1 reserve) pages hourly, with the balance in the alert. Credit reservation across concurrent launches is still open (needed once dispatcher concurrency is above 1). Auto-recharge is the owner's Vast setting (D-09). |

## D. Observability and operations

| ID | Status | Evidence / next step |
|---|---|---|
| D-01 | next deploy | #3226 merged 14:0xZ (stall paging on the capacity alert route; D-12 approved by delegation). Follow-up: page daily on cap exhaustion. |
| D-02 | in progress | WebApp [#1176](https://github.com/ognjhunt/Blueprint-WebApp/pull/1176) merged: the external uptime probe also checks the door `/healthz`. Pending: the WebApp deploy, then a manual deploy of the `blueprint-uptime-probe` Render cron (`autoDeploy: false`). Capacity-tick freshness is a follow-up. |
| D-03 | not started | |
| D-04 | next deploy | Root cause: `run_chain` stores `EnvironmentFiles` as tuples, so `json.loads(raw) != report` on every run (journal 13:22:46Z: `preflight_report_bytes_mismatch`) and `door status` shows `preflight: refused`. Fix: [#3249](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3249). Triage of the 186–193 blockers starts once the summary is published. |
| D-05 | not started | `door status` 13:19Z: `due_backlog` 1, oldest 4.5 days, not overloaded. |
| D-06 | not started | `door status` 13:19Z: 206 failed units. |
| D-07 | owner | Needs the on-call backup's name (D-24). |
| D-08 | not started | |
| D-09 | not started | |

## E. Capacity, infrastructure, security

| ID | Status | Evidence / next step |
|---|---|---|
| E-01 | in progress | 13:19Z: `/mnt/blueprint-work` 90.4% used, 3.11 days to floor at 4.7 GB/day; `/` 91.9%. Capacity level `critical`, `reclaim_ineffective` paging. The usage survey attributes only 0.1% of the work volume (truncated). Storage GC retains: `result_registry` 17.1 GB (29 runs), one `queue_referenced` directory 14.8 GB, 11 activation pins 6.6 GB. All 13 launch-activation directories belong to capture-447 retries. The 14.8 GB one is the 10-31 activation (`6e9f0cac`), pinned by a stranded `processing/` row whose standing authorization expired 2026-11-01 17:38Z. Fixes: reconcile stranded activation rows and release superseded attempts' pins (code, next deploy window), plus the owner asks above. |
| E-02 | not started | Load 5.2 on 4 vCPU. The preflight timer is `OnUnitActiveSec=15min`, and a run takes about 13 min. |
| E-03 | not started | |
| E-04 | owner | Credential rotation (G-05). |
| E-05 | not started | |
| E-06 | not started | |
| E-07 | not started | |

## F. Website: result delivery, truthfulness, journey

| ID | Status | Evidence / next step |
|---|---|---|
| F-01 | in progress | WebApp #1179 (one email per terminal screening outcome) is in review. |
| F-02 | done | `taskEvaluationPolicyRuns(job_evidence_binding.job_id, created_at_iso desc)` is READY in `blueprint-8c1ca` (gcloud, 14:00Z). Website scene-intake queries are equality-only. 17 declared indexes in `firestore.indexes.json` were never deployed. Created the one a live website page needs, `creatorCaptures(creator_id, created_at desc)` for `CapturerAccount` → `/v1/creator/captures`; READY 14:10Z. The rest serve ops, growth and payout routes and are left for H-03. |
| F-03 | done | WebApp [#1177](https://github.com/ognjhunt/Blueprint-WebApp/pull/1177) merged and deployed (`ba27cd6e4`, 14:33Z): one sentence per outcome on both the headline and the body; reference-only runs are not called estimates; no raw codes, budgets or operator text reach sites; exhausted allowances now read as ended (the prefix bug). Follow-up: the `footage_before_account` email promises "We email you when that changes". |
| F-04 | in progress | WebApp #1181 merged: the default intake cap is 5 a day. **Owner checks:** Render must not pin `BLUEPRINT_INTAKE_DAILY_LIMIT=50`. The allowlist needs the invitee list. App Check `monitor` needs `VITE_FIREBASE_APPCHECK_SITE_KEY`. |
| F-05 … F-11 | not started | |

## G. Privacy, rights, legal

| ID | Status | Evidence / next step |
|---|---|---|
| G-01 | owner | Counsel status before 2026-11-15. |
| G-02 | in progress | WebApp [#1178](https://github.com/ognjhunt/Blueprint-WebApp/pull/1178) (auto-merge) carries #1142's two reviewed rights commits onto main (82 tests, `npm run check` pass); #1142 is closed. Next: split the rights chain out of Pipeline #3042 (screening-runtime revocation: `b57cde47c`, `dd1c2ffee`, `b994839ab`, `81f57c38c`, `c371cec63`, `6f62b422c`, validators `fde308f66`) and run a withdrawal drill. |
| G-03 | superseded | D-16 (owner, 14:15Z): no blurring or person screening. The residual-person check in #1980 is moot; its stage-reuse part can be revisited separately. |
| G-04 | owner | Owner decision D-16: no blurring or screening; rely on the site's consent. The published Terms and Privacy Policy overclaim ("we remove people from the frames used as input to reconstruction") and the Privacy Policy omits Meta. Draft correction: WebApp [#1183](https://github.com/ognjhunt/Blueprint-WebApp/pull/1183) (Terms 2026-11-15.4 / Privacy 2026-11-15.5; needs the owner's or counsel's OK on wording; a material-change notice is likely). The runbook §3a correction is still to do. |
| G-05 | owner | Rotation is the owner's action. Both repos are public (verified 13:30Z). |

## H. Cleanup and consistency

| ID | Status | Evidence / next step |
|---|---|---|
| H-01 | not started | |
| H-02 | done | 126 PRs and 3 issues closed (owner-authorized D-23), each with one comment; 10 already merged and skipped; #2245 and #443 marked stale. Closing disarmed auto-merge on WebApp #858 and Pipeline #1348, #1462 and #116. All of it is reversible. |
| H-03, H-04 | not started | |
