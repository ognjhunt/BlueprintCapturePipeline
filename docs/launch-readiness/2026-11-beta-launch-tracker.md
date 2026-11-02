# Beta launch tracker (November 2026)

Goal: make Blueprint ready for its invited beta, then its product launch. Every
item keeps the ID from the 2026-11-02 goal brief. Times are UTC unless marked.

Status: **done** (verified, with evidence) · **in progress** · **owner** (waits
on an owner decision or action) · **next deploy** (merged, ships with the next
batched Pipeline deploy) · **not started**.

Last updated: 2026-11-02 14:15Z.

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
| A-01 | owner | capture-447 (intent `scene-819be616…`) attempt `source-690b1233…` on release `4c2640ba9` (#3248, export-only ARTISAN reuse). Activation succeeded 13:34:58Z; the launch was dispatched 13:37:52Z and **refused before allocation at $0** (13:39:42Z, `scene_configuration_provider_output_disk_budget_exceeded`). It needs 35.19 GB available on the work volume (CPU-prestage peak = 2× the 16.3 GB bundle) and has 8.06 GB (free 23.9 GB − floor 15.8 GB). Intent now `blocked/configuration_capacity/preallocation_capacity_not_recovered` (13:45:12Z); it relaunches once capacity returns. Unblock: owner volume growth (E-01). Execution window open to 2026-11-07 10:30Z. Owned by this session from 13:30Z. |
| A-02 | not started | Runs in the same GPU session after stage 3. The 10-31 destination gates are fixed on main (#3136; `ef276779a` for the stage-5 bind tolerance; `ae0e4ae2b`). |
| A-03 | owner | WebApp #1168 merged 14:04Z (deploy pending CI). capture-447 still needs the D-02 ceiling row ($110) before its screening plan can fit. |
| A-04 | not started | Host has never created the team-scene-packet status or rigid-probe roots. |
| A-05 | not started | `door status`: `beta_screening.campaign_count` = 0. |
| A-06 | not started | After A-05 and C-01…C-07. |

## B. Velocity, cost model, money correctness

| ID | Status | Evidence / next step |
|---|---|---|
| B-01 | in progress | Mapping stages 3–6 and the post-completion chain for an offline replay command. |
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
| C-03 | not started | capture-447 needed a window extension to 2026-11-07 10:30Z. |
| C-04 | in progress | Mapped. Only `completed` stops the progression driver; everything else retries every minute forever. The WebApp's `ENDED_BLOCKERS` lists unprefixed codes while the Pipeline writes `scene_progression_*`/`scene_intake_*`, so exhausted caps show "waiting" (fix folded into the F-03 PR). `failed_terminal` is never emitted. Plan: a write-once Pipeline `outcome.json` (held/setup_repair/awaiting_spend_approval/awaiting_site/withdrawn) surfaced as an optional `outcome` on the intent status. The WebApp ships `.optional()` first (its schema is strict) and maps it to one state plus one email. |
| C-05 | not started | |
| C-06 | not started | |
| C-07 | not started | |

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
| F-01 | in progress | Verified on WebApp `8ead858b5`: screening ingest sends no email. A campaign that reaches evaluated/held between scans also gets its queued `scene_tests_started` cancelled at dispatch, so the site gets no email at all. Plan: one deduplicated `screening_outcome` outbox email per terminal outcome, re-checked at dispatch, with a reconciler. |
| F-02 | done | `taskEvaluationPolicyRuns(job_evidence_binding.job_id, created_at_iso desc)` is READY in `blueprint-8c1ca` (gcloud, 14:00Z). Website scene-intake queries are equality-only. 17 declared indexes in `firestore.indexes.json` were never deployed. Created the one a live website page needs, `creatorCaptures(creator_id, created_at desc)` for `CapturerAccount` → `/v1/creator/captures`; READY 14:10Z. The rest serve ops, growth and payout routes and are left for H-03. |
| F-03 | in progress | `no_match` headline/body mismatch still present. #1174 added a second mismatch (a preparation hold's body vs the generic held headline). Scripted-only runs are still called an estimate. Raw codes and "for development only" are still shown, and operator next-action text still reaches sites. |
| F-04 | not started | |
| F-05 … F-11 | not started | |

## G. Privacy, rights, legal

| ID | Status | Evidence / next step |
|---|---|---|
| G-01 | owner | Counsel status before 2026-11-15. |
| G-02 | not started | Approved (D-15): rebase #1142, split #3042. |
| G-03 | not started | Approved (D-15): rebase #1980. |
| G-04 | not started | Decision D-16: consent plus real face-blurring on every frame that leaves to a provider. |
| G-05 | owner | Rotation is the owner's action. Both repos are public (verified 13:30Z). |

## H. Cleanup and consistency

| ID | Status | Evidence / next step |
|---|---|---|
| H-01 | not started | |
| H-02 | not started | Bulk close authorized (D-23). |
| H-03, H-04 | not started | |
