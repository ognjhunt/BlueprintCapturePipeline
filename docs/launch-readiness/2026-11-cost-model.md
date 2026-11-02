# B-03: Measured cost model for one website job

As of 2026-11-02 13:35Z. Built from retained production receipts. Read-only: 134 door-ro calls, no spend, nothing changed on the host. Code reference: `wt-pipe` at `4c2640ba` (origin/main), the same commit as capture-447's live attempt.

Labels:
- **[M]** measured (source path given)
- **[E]** estimated (basis given)
- **[C]** from code (file given)
- **[U]** unknown (no receipt was read)

The pulled files were kept in the analysing session's scratch space; every source path is listed in section 9.

This model replaces the cost figures in `docs/BETA_CAPACITY_COST_STORAGE_MODEL_2026-07-08.md` ($3.56 per capture, 1,200 GPU-s per capture, Cloud Run). See section 7.

---

## 1. Headline

| | capture-447 to date (job still live, attempt 18 pending) |
|---|---|
| Real spend, measured | **$25.08** = ARTISAN/OpenAI $17.02 + Vast billed $1.65 + Atlas completed operations $5.50 + Atlas-side OpenAI $0.91. Atlas pose/view purchases and the upstream ledger are not included **[U]**. |
| Held against caps | Scene budget **$58.24** of $70 ($52.23 kept by 17 settled rows + $6.01 live). Atlas allowance ≈$19.5 of $40 (from the brief; not re-read). Upstream ledger ≤ $5. |
| Hold-to-actual ratio, scene stage | $52.23 kept for $18.67 measured = **2.8×** |
| GPU | 5 Vast instances, 6,216 GPU-s (1.73 h) **[M]**. Billed $1.648, or $0.954 per GPU-hour all-in. |
| Attempts | 27 preparation attempts ($0 each) → 18 scene-configuration reservation rows → 10 dispatched launches → 5 created a GPU instance → 0 succeeded |

Only capture-447 reached a GPU in the last 30 days. I found 14 other website jobs on the host in the window:
- None has an October launch run.
- Only cd43cdc3 has an October scene intent, and that intent has no paid rows.
- Their Atlas, OpenAI and Gemini spend lives in the WebApp ledger, which I did not read (section 8).

---

## 2. capture-447 by stage and provider: actual vs reserved

Job identifiers: capture `447ae4f4-c253-44ae-a897-c999245e0ef7`, intent `scene-819be616…88cc`, team namespace `website-566f5be8f9c868d8c49215ca`. Job root:
`/var/lib/blueprint/pubsub-handoffs/blueprint-8c1ca.appspot.com/scenes/site-capture-447ae4f4-…/captures/walkthrough-capture-447ae4f4-…/pipeline/` (shown below as `P/`).

| Stage | Provider / item | Reserved or held | Actual | Source |
|---|---|---|---|---|
| Atlas splat | World Labs `posedImages2Splats` ×3 | $15.00 (3 × $5) [M] | **$5.00** [E]: 1 completed × $5 rate card. The other 2 failed with code 13 and failed runs are free, so $0. | `P/atlas/automation/background_completed_splat-…-refined/native/{controller_admission.json,atlas_pose/submission.json}` and `…/native/retry/…` (both failed); `…-refined-minimal/native/…` (completed) |
| Atlas mesh | World Labs `splats2Mesh` ×1 | $0.50 [M] | **$0.50** [E]: completed, rate card | `P/atlas/automation/background_completed_mesh-…/controller_admission.json`, `result.json` present |
| Atlas pose and views | Pose inputs $0.01 each; RGBD view $0.05 + $0.01 per extra view; RGB view $0.04 + $0.008 per extra view [M rate card] | [U] | [U]. About $3.1 [E] if the brief's ≈$19.5 allowance usage is right: $19.5 − $16.41 of measured rows. | Rate card: `P/atlas/configuration.json`. Purchases not enumerated. |
| Atlas planner | OpenAI gpt-6.1-sol ×3 | held at max until settled | **$0.353** [M], settled | `P/atlas/scene_planner/{ccfe544c…,02690767…,c327227b…}.settlement.json` |
| Atlas automation reviews | OpenAI gpt-6.1-sol ×3 | $12.00 (3 × $4 `maximum_review_cost_usd`) [M] | **$0.561** [M], token-priced usage: 0.472 + 0.058 + 0.031. Settlement not read. | `P/atlas/automation/result.json` (`…planner.admission` / `…usage`) |
| Atlas allowance total | | cap **$40 / 40 requests** [M], owner-approved | | `P/atlas/configuration.json` |
| Upstream ledger | SAM, Gemini, other OpenAI, rigid probe | cap **$5** [M] (`upstream_max_spend_usd`, amendable to 3×) | [U] | `P/website_scene_sponsorship.json` |
| Scene preparation | control plane, 27 attempts | $0 [M] | $0 [M] | `…/scene-intents/scene-819be…/preparation-attempts/source-7d107400….json` (`maximum_spend_usd: 0`, `provider: control_plane`) |
| Scene configuration GPU | Vast, 5 instances | compute share of each row: $5 or $6 [M] | **$1.648 billed** [M]. Duration × rate $0.916; adapter estimate $3.448. | `gpu_spend_guard/daily_spend_snapshot.json` (rows keyed by `sha256(json("instance-<id>"))`); `…/vast_provider_run/vast_budget_ledger.json` |
| Scene configuration stage 3 | ARTISAN on OpenAI (gpt-6-astra, gpt-5.6-sol compactions, 1 design-board image) | cumulative cap $15, raised to **$25** on 11-02 [M] | **$17.019 metered** [M], 55 calls | `…/stages/stage-3/producer/artisan_runtime/blueprint_usage.json` (22c9f59f, 7f4e4a4a); `…/continuation/source/adoption.v1.json` |
| Scene configuration teacher and visual review | OpenAI | $0. Website jobs set `requires_artifixer=False`, so the caps are 0 [C] | $0 [C] | `website_native_submission.py:145-151` |
| Scene-configuration holds | — | **$52.23 kept + $6.01 live** of $70 [E: code applied to receipts, replay-validated (section 4)] | — | settlements in `…/cancelled-unstarted-controls/*.json`; `task_evaluation_terminal_scene_attempt_settlement.py:235-341` |
| Rigid robot-free probe | Vast via the upstream ledger | cap $1.00 at ≤ $0.80/h [C] | not run, $0 | `rigid_robot_free_probe_lane.py:71-72,275` |
| Beta screening | per candidate | $15 per site [C]; ≤ $4 per candidate (brief) | not run, $0 [M]: `campaign_count 0` | door `status` → `beta_screening`; `beta_policy_screening_coordinator.py:41` |

Notes:
- World Labs has no per-operation bill. Atlas "actual" means the rate-card price of a completed operation.
- The WebApp has no `world_labs_atlas` settlement variant (memory note dated 10-30, consistent with `wt-web/server/utils/websiteSceneSponsorship.ts:470-520`). So the 2 failed splats keep $10 held in the allowance even though they cost $0.

---

## 3. capture-447 scene-configuration attempts (all 18 reservation rows)

Paths:
- Rows: `…/scene-intents/scene-819be…/attempts/` and `cancelled-unstarted-controls/`.
- Launches: `/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs/website-566f5be8…-<id>-…-activation-auto-launch/`.

Column notes:
- **dur×rate** is $ from `vast_budget_ledger.json`.
- **billed** is $ from the daily snapshot, split as gpu / download / disk / upload.
- **kept** is the hold left after settlement.

| # | Reserved (UTC) | Launch | Row hold | Outcome (blocker from `launch_receipt.json` or stage-3 log) | Instance, GPU-s, $/h | dur×rate | billed | ARTISAN | Kept (basis) | Hold time | Fault |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10-31 15:38 | 9a384717 | $20 | never queued; superseded by a newer release | — | 0 | 0 | 0 | $0 (never_queued) [M] | 1:43 | Blueprint release churn |
| 2 | 10-31 17:36 | 6e9f0cac | $20 | never queued | — | 0 | 0 | 0 | $0 [M] | 0:47 | churn |
| 3 | 10-31 18:30 | 947d5dcd | $20 | never queued | — | 0 | 0 | 0 | $0 [M] | 1:14 | churn |
| 4 | 10-31 19:49 | ccbeb1fc | $20 | never queued | — | 0 | 0 | 0 | $0 [M] | 1:04 | churn |
| 5 | 10-31 20:57 | 7d27dc64 | $20 | contained before run | — | 0 | 0 | 0 | $0 [M] | 4:55 | churn |
| 6 | 11-01 01:58 | 237194fe | $20 | stopped before allocator | — | 0 | 0 | 0 | $0 [M] | 3:52 | churn |
| 7 | 11-01 05:54 | d63579c5 | $20 | `provider_runtime_bundle_json_parse_failed`; no instance | — | 0 | 0 | 0 | $0 [E] | 6:47 | Blueprint (bundle) |
| 8 | 11-01 12:45 | 95d87185 | $20 | `asset_sandbox_unavailable` in stage 3, before any model call | 58665957, 1073 s, $0.507 | 0.151 | **0.412** (.137/.257/.016/.002) | 0 (no meter file) | **$5** [E] (compute allowance kept; authoring never entered) | 2:12 | Blueprint (sandbox) |
| 9 | 11-01 16:17 | a00c7494 | $20 | never queued | — | 0 | 0 | 0 | $0 [M] | 1:54 | churn |
| 10 | 11-01 19:15 | f35b93e3 | $20 | `provider_output_disk_budget_exceeded`; no instance (no ledger; no unmatched billed row on 11-01) | — | 0 | 0 | 0 | $0 [E] | 2:36 | Blueprint (host disk) |
| 11 | 11-01 21:55 | 77509c4c | $20 | `artisan_sandboxed_interpreter_unavailable` (libpython missing in image) | 58725584, 707 s, $0.654 | 0.129 | **0.162** (.139/.015/.008/0) | 0 | **$5** [E] | 1:45 | Blueprint (image) |
| 12 | 11-01 23:45 | 22c9f59f | $20 | `artisan_framework_terminal_failure` after 44 calls | 58739608, 2074 s, $0.507 | 0.292 | **0.598** (.278/.258/.024/.038) | **13.761** (44 calls; cap $15) | **$20** full [M] (`terminal_launch_unreconciled`) | 2:02 | ARTISAN run inside Blueprint sandbox; root cause not in receipt |
| 13 | 11-02 02:14 | fe0a6a75 | $17 | never queued | — | 0 | 0 | 0 | $0 [M] | 0:39 | churn |
| 14 | 11-02 02:57 | 813c7cb5 | $21 | never queued | — | 0 | 0 | 0 | $0 [M] | 0:52 | churn |
| 15 | 11-02 03:55 | b8dfa68e | $30 | `provider_credit_insufficient`; no instance | — | 0 | 0 | 0 | $0 [E] | 1:28 | Blueprint ops (Vast balance) |
| 16 | 11-02 07:52 | 97a472e7 | $30 | `artisan_continuation_restored_inventory_mismatch`, before any model call | 58792846, 1002 s, $0.490 | 0.136 | **0.208** (.106/.087/.013/.002) | 0 | **$5** [E] | 2:07 | Blueprint (code) |
| 17 | 11-02 11:18 | 7f4e4a4a | $17.23 | `artisan_not_done_and_registered`; continuation of #12, which carried $13.76 / 44 calls | 58815449, 1360 s, $0.548 | 0.207 | **0.268** (.229/.026/.012/.001) | **3.258** (11 calls; increment $11.23) | **$17.23** full [M] | 1:39 | Blueprint-run ARTISAN |
| 18 | 11-02 13:14 | 4c2640ba (530202b7) | $6.01 | live, awaiting execution ($0.01 ARTISAN export-only + $6 compute) | — | — | — | — | live | — | — |
| **Σ** | | | | | 6,216 GPU-s | **0.916** | **1.648** | **17.019** (55 calls) | **$52.23 + $6.01 live** | | |

Row-hold structure [C]: hold = remaining cumulative ARTISAN cap + compute cap ($6; it shrinks when the intent ceiling is short).
- $20 = $15 + $5 (ceiling then $20)
- $30 = $25 + $5
- $17.23 = $11.23 + $6
- $6.01 = $0.01 + $6

Source: `task_evaluation_scene_configuration_submission_records.py:195-234`, `website_native_submission.py:130-151`, `…_runtime_budget.py:47-48`.

Per-attempt measured distributions (n = 5 GPU attempts):

| Measure | Values | p50 | p90 (nearest-rank) |
|---|---|---|---|
| Vast billed | 0.162, 0.208, 0.268, 0.412, 0.598 | $0.268 | $0.598 |
| Adapter estimate (incl. projected transfer) | 0.202, 0.302, 0.452, 1.234, 1.259 | $0.452 | $1.259 |
| GPU-s | 707, 1002, 1073, 1360, 2074 | 1,073 s | 2,074 s |

- Billed GPU component $0.889 ≈ duration × rate $0.916. Bandwidth is 39% of the bill ($0.643 download).
- The adapter's projected transfer ($2.53) is 3.7× the billed bandwidth ($0.686).

---

## 4. Holds vs ceiling, and what Blueprint-fault retries consumed

Owner grants [M] (`…/execution-budget-extensions/*.json`, `host_operator` channel; `max_paid_attempts` stayed 16):

| When | Ceiling |
|---|---|
| intake | $20 |
| 11-01 16:14 | $25 |
| 11-01 23:12 | $30 |
| 11-02 02:07 | $47 |
| 11-02 02:09 | $60 |
| 11-02 12:41 | $70 |

Replay check: I replayed all 18 reservations and 17 settlements in time order.
- With the code's adjusted holds (`budget_retained_hold`), every admission fits the ceiling in force, several exactly: $25 at 16:17, 19:15 and 21:55; $30 at 23:45; $60 at 03:55 and 07:52.
- With raw full holds, it would reach $183.24.
- So the kept holds above ($5 / $5 / $5 / $20 / $17.23, $0 elsewhere) are what the budget code uses: **$52.23**. This matches the brief's ≈$52.

**Answer: how much of the ceiling a Blueprint-fault retry consumed on capture-447**

- **Per pre-authoring Blueprint-fault GPU retry** (#8 sandbox, #11 interpreter, #16 continuation restore):
  - Kept $5 for good: 7.1% of the final $70, 25% of the original $20.
  - Actual billed: $0.16–$0.41 (12–31× over-held).
  - While in flight each held the full row of $20–$30 for about 2 h: up to 100% of the ceiling then in force ($20 of $20 for #8; $30 of $60 for #16).
  - The three together kept **$15 (21% of $70) for $0.78 billed**.
- **ARTISAN-entered failures** (#12, #17; Blueprint-run): kept $37.23 (53% of $70) for $17.88 actual. That spend is not wasted: #17 adopted #12's work, and attempt 18 is export-only.
- **No-instance Blueprint faults** (#7 bundle parse, #10 host disk, #15 Vast credit) and **9 release-churn rows**: kept $0, but each blocked $17–$30 of ceiling for 39 min to 6 h 47 min.
- **Total:** $52.23 kept (75% of $70) for $18.67 measured scene spend. Every one of the 17 retired rows was a Blueprint-side failure; none was a Vast or OpenAI failure. The only provider-side failures were the 2 World Labs splat failures (Atlas ledger, $10 held, $0 actual).

**Attempt-cap finding** [C, not runtime-tested]:
- `reserve_scene_attempt` counts only "ordinary" source rows against `max_paid_attempts` (`task_evaluation_scene_intake.py:60,318-322`). `scene-configuration-*` rows are dependent rows.
- A terminal-unreconciled settlement has `counts_as_attempt: False` (`…settlement.py:208`). Website source attempts live in `preparation-attempts/` at $0.
- So for website jobs **`max_paid_attempts` never binds**: 18 rows ran under a cap of 16 with `max_retries 0`. Only the dollar ceiling stopped retries.

---

## 5. Website jobs found on the host in the last ~30 days

Scenes are listed under `/var/lib/blueprint/pubsub-handoffs/blueprint-8c1ca.appspot.com/scenes`.

| Job (capture) | Created | Furthest stage | Atlas allowance [M: `…/pipeline/atlas/configuration.json`] | GPU | Measured spend |
|---|---|---|---|---|---|
| 447ae4f4 | 10-30 | scene config, attempt 18 pending | $40 / 40 req, review cap $4, refined splat priced | 5 instances | ≥ $25.08 (section 2) |
| 4b60431e (job 3; same video and task as 447) | 10-29 | Atlas refined splat; ran out of allowance | $25 / 32 req, review cap $4 | none | Planner $0.472 settled [M: `…/atlas/scene_planner/*.settlement.json`]. Refined splat ×2 held $10, actual $0: failed code 13, then rejected [M: `…-refined/native/{,retry/}…`]. Allowance remaining $12.59 per memory note 10-30 [not re-read], so usage ≈ $12.41 [E]. |
| f88f1bd0 | 10-29 | Atlas, planner abstained, then supervisor skipped (memory) | $25 / 32, review cap $1 | none | [U] |
| 98a79ce3 | 10-29 | target clarification (memory) | $25 / 32, review cap $1 | none | [U] |
| c33427c3 | 10-28 | target clarification (memory) | $25 / 32, review cap $1 | none | [U] |
| 05d7f712 | 10-27 | target track ambiguous; 41 attempts (memory) | $25 / 32, review cap $1 | none | [U] |
| 6940ee82 | 10-23 | target clarification, then consent expired (memory) | $25 / 16, review cap $1 | none | [U] |
| cd43cdc3 | 10-11 | scene intent `scene-2bd41…` (10-12): dishwasher open/close, $20 / 16, 3 preparation attempts, **no paid rows** [M] | not read (pre-Atlas era) | none | scene $0 [M]; preparation [U] |
| 70454ab0, 8656e744, 36ad8b1d, 89d24b73 | 10-12 | not read | — | none found | [U] |
| f8f79b26, 29abb4cb, e70b9764 | retired 10-17 / 10-05 / 10-04 (activity dates not read) | retired workspace records only | — | none found | [U] |

Out of window, for attempt-count shape only: Sept website namespaces have offloaded launch runs. Dispatched launches per job, counted from `task-evaluation-launch-runs` names [M]:
- 52ff463b: 10
- f857014e: 10
- c577df51: 5
- 337b48b9: 4
- b7d45d1b: 2
- a9ec304c: 1
- 9dda3f7d: 1

Their Vast ledgers are offloaded (only `launch_receipt.json` stays inline, `control_plane_retained_receipt.py:15`), so their cost is [U].

Vast context [M, daily snapshot]:
- Window 10-03 → 11-02: $32.45 billed across 69 charge rows.
- capture-447 accounts for $1.648 (5 rows). The other $30.80 is unattributed non-website work. It includes 10-30 ×4 ($0.847) and 10-31 ×2 ($0.406), which are not capture-447 launches.

---

## 6. p50 / p90 cost per job, including retries

An empirical per-job distribution is not possible: 1 of 15 jobs in the window reached a GPU, and the preparation spend for the other 14 is in the WebApp ledger I did not read. Instead, below is a component model built from the measured per-attempt values. Every input is labelled.

| Component | p50 | p90 | Basis |
|---|---|---|---|
| Atlas completed operations | $5.50 | $5.50 | [M] 447: splat $5 + mesh $0.5. Failed operations are $0 but each holds $5 in the allowance. |
| Atlas pose and views | $3.10 | $3.10 | [E] residual from the brief's ≈$19.5 |
| Atlas-side OpenAI (planner + reviews) | $0.91 | $0.91 | [M] 447, 6 calls |
| ARTISAN per object (cumulative) | $17.02 | $25.00 | [M] n = 1 / [C] cumulative cap |
| GPU attempts × billed per attempt | 3 × $0.268 = $0.80 | 6 × $0.598 = $3.59 | [M] per-attempt p50/p90. Counts are [E]: 447 needed 5 + 1. Across 8 website intents (7 from Sept + 447), dispatched launches were p50 4.5, p90 10; × the 0.5 instance-per-dispatch ratio measured on 447. |
| Upstream (SAM / Gemini / OpenAI) | U | $5.00 | [C] cap; actual [U] |
| Rigid probe | U | $1.00 | [C] cap; never run |
| **Subtotal, no screening** | **$27.3 + U** | **$44.1** | |
| Beta screening, if run | +$0 to $15 | +$15.00 | [C] site limit; never run |
| **Total** | **≈ $27–42** | **≈ $59** | |

Arithmetic:
- p50 = 5.50 + 3.10 + 0.91 + 17.02 + 0.80 = 27.33
- p90 = 5.50 + 3.10 + 0.91 + 25.00 + 3.59 + 5 + 1 = 44.10, then + 15 = 59.10

A job-level retry on the same video (447 after job 3) adds job 3's spend: ≈ $2.4 actual [E] ($0.47 planner [M] + ≈ $1.94 other [E]), but $12.41 of Atlas allowance held.

Holds needed per end-to-end job under today's accounting: Atlas ≈ $19.5 + scene $58.24 + upstream $5 ≈ **$83**, plus $15 for screening.

---

## 7. Recommended defaults (arithmetic shown)

1. **Scene `max_total_spend_usd` = $65 now; $35 after the settlement fix.**
   - Now: $25 ARTISAN cumulative cap + (5 kept compute holds + 1 live) × $6 = $61, + $4 for overlap and rounding = **$65**. The overlap is ARTISAN allocation kept but not spent; on #12 it was $1.24.
   - capture-447 needed $58.24 only because its compute holds shrank to $5 to fit the lower ceilings. The compute cap shrinks to fit what remains (`submission_records.py:224-233`), and the hourly cap shrinks with it (compute × 3600 / 27,000 s): $5 → $0.667/h, $2 → $0.267/h. Below about $4 of compute, no RTX-4090-class offer ($0.49–0.65/h measured) qualifies.
   - After holds settle to measured spend (item 3): $25 + $6 live compute + 6 × $0.598 p90 billed ≈ $34.6, rounded to **$35**.
2. **`max_paid_attempts` = 6 instance-creating scene-configuration attempts, and make the cap count them.**
   - (65 − 25 − 4) / 6 = 6 compute holds → 5 kept + 1 live.
   - Today the cap does not count `scene-configuration-*` rows (section 4).
3. **Settle kept holds at measured spend.** This is the biggest lever.
   - Vast: the billed per-instance charge is already on the host (`daily_spend_snapshot.json`), and the WebApp already has a `vastSettlementRequest` shape.
   - ARTISAN: the `blueprint_usage.json` meter.
   - On 447 this turns $52.23 kept into $18.67.
   - Also add the missing `world_labs_atlas` settlement: $0 for failed operations. On 447 and job 3 that is $10 held each for $0 actual.
4. **Per-job Blueprint-paid ceiling** = Atlas $25 + upstream $5 + scene $65 = **$95**, or **$110** with screening.
   - After item 3: $25 + $5 + $35 = **$65** (or $80 with screening).
   - This still covers the modelled p90 actual: $44, or $59 with screening.
5. **Atlas allowance: keep the $25 default.**
   - 447's ≈ $19.5 includes $10 of failed-splat holds. Without them it is ≈ $9.5, which leaves room for 2 more failed splats.
   - Job 3 hit its request count, not dollars. Request counts are no longer a ceiling (owner decision 10-30, `websiteSceneSponsorship.ts:943`).
6. **Max preparation attempts = 12 per intent** (2 × `max_paid_attempts`). Do not re-reserve a row when a release only supersedes an activation that was never dispatched.
   - 447: 27 preparation attempts → 18 rows → 9 never dispatched, each holding $17–21 for 39–115 min.
   - Preparation itself costs $0, so this cap protects ceiling and time, not dollars.
7. **Update the old model** (`docs/BETA_CAPACITY_COST_STORAGE_MODEL_2026-07-08.md`): $3.56 per capture → p50 ≈ $27, p90 ≈ $44 (+$15 screening); 1,200 GPU-s per capture → 1,073 GPU-s p50 per attempt, 6,216 GPU-s for 447. Fixed host cost is separate: DigitalOcean October invoice $84.19 [M], about $5.6 per job over 15 jobs [E, upper bound because the host also serves non-website lanes].

---

## 8. Gaps (unknown stays unknown)

1. **WebApp ledger not read.** Firestore `inboundRequests/<request_id>.website_preparation_reservations` holds every Atlas, upstream and probe reservation and settlement per job. This is the single read that would close the Atlas and upstream gaps for all 15 jobs. It was outside the host-only access rules, so I did not use the Firestore tool.
2. **capture-447 Atlas pose and view purchases not enumerated.** Not read: `P/atlas/pose/task-coverage-*`, `automation/{scene_completion-v14,completion_*,fallback_pose-v3,background}`. The ≈ $3.1 residual is an estimate.
3. **Atlas "actual" is the rate card, not an invoice.** World Labs exposes no per-operation bill. The claim that failed runs are free comes from World Labs policy, as recorded in the memory note and `atlas_spend_settlement.py`.
4. **ARTISAN is metered, not invoiced.** It is tokens × price table plus one $2.43 image-budget estimate. The official OpenAI per-run settlement (`openai_official_cost_run_settlement.v1`) has no production caller.
5. **Vast rows may still change.** Billed rows are flagged partial: day-grain interval and no stable charge id. Charges for 11-02 may still post; re-read the snapshot after 24 h.
6. **Kept holds for #7–#17 are derived, not printed by the host.** I applied `budget_retained_hold` to the receipts and validated by replay (section 4). I did not read the per-launch result or teardown for #7, #10, #15.
7. **Rigid probe and screening never ran**, so only their caps are known.
8. **Outcomes for 6940ee82, 05d7f712, c33427c3, 98a79ce3 and f88f1bd0, and job 3's $12.59, come from memory notes** and were not re-verified. The 10-11/10-12 jobs and the retired jobs were not read.
9. **Billing context.** DigitalOcean $325.38 + Vast $455.88 = $781.26 billed vs the guard ledger's $165.55 [M: door `status`]. The guard ledger is a polling estimate of GPU allocation: it excludes the host, Vast bandwidth and disk, and short instances. On 447 the adapter's runtime estimate ($0.916) is 56% of billed ($1.648).

---

## 9. Host paths read (134 door-ro calls, read-only)

Pipeline control plane, shown below as `CP` = `/var/lib/blueprint/pipeline-control-plane/`:
- `CP/task-evaluation-scene-intents/` (ls)
- `…/scene-819be6165e6a9d56f535cc977ae3559df595d4ffeebae42a04db7e4f81b588cc/`:
  - ls of the directory and of `attempts/`, `cancelled-unstarted-controls/`, `execution-budget-extensions/`, `preparation-attempts/`, `preparations/`, `execution-window-extensions/`
  - `intent.json`
  - `attempts/scene-configuration-{530202b7ac87eeb75e41b426,824c33a7bbe66bceccb764f7}.json`
  - `preparations/530202b7…227375ee9.activation.json`
  - all 17 `cancelled-unstarted-controls/scene-configuration-*.json`
  - all 5 `execution-budget-extensions/*.json`
  - `preparation-attempts/source-7d107400ae81cd47e1b605f9.json`
- `…/scene-2bd41095…7e89/`: ls, `intent.json`, ls `preparation-attempts/`
- `…/scene-bd07d5f7…380b/`: ls
- `CP/gpu_spend_guard/`: ls, `daily_spend_snapshot.json`, `spend_ledger.json`
- `CP/task-evaluation-launch-runs/` (ls `--match website-*`)
- For each of `…-{d63579c5,95d87185,f35b93e3,77509c4c,22c9f59f,b8dfa68e,97a472e7,7f4e4a4a}…-activation-auto-launch/`: `launch_receipt.json` and `allocator/scene-configuration-job/vast_provider_run/vast_budget_ledger.json` (f35b93e3: not found)
- 7f4e4a4a only:
  - ls of the run dir, `allocator/`, `scene-configuration-job/`, `vast_provider_run/`, `immutable_execution/`, `stages/`, `stages/stage-3/`
  - `task_evaluation_scene_configuration_vast_result.v1.json`
  - `stages/artisan_same_run_resume_binding.json`
- `…/immutable_execution/stages/stage-3/producer/artisan_runtime/blueprint_usage.json` for 7f4e4a4a and 22c9f59f (95d87185, 77509c4c, 97a472e7: not found)
- `…/continuation/source/adoption.v1.json` for 7f4e4a4a (others not found)
- `…/stage-3/producer/stage_producer.log` for 22c9f59f, 7f4e4a4a, 95d87185, 97a472e7
- ls of stage-3 `producer/` for 95d87185 and 97a472e7
- ls of `scene-configuration-job/` for f35b93e3

Website jobs:
- `/var/lib/blueprint/pubsub-handoffs/` (ls) and `…/blueprint-8c1ca.appspot.com/scenes/` (ls)
- capture-447 job dir `…/site-capture-447ae4f4-…/captures/walkthrough-capture-447ae4f4-…/`:
  - ls of the capture dir, `pipeline/`, `pipeline/atlas/` and its `automation/`, `pose/`, `scene_planner/`
  - `pipeline/provider_run_manifest.json`, `pipeline/website_scene_sponsorship.json`
  - `pipeline/atlas/{configuration.json,preview_job.json,automation/result.json}`
  - `scene_planner/{ccfe544c…,02690767…,c327227b…}.settlement.json`
  - refined splat `native/`: ls, `controller_admission.json`, `atlas_pose/submission.json`, and the same two files under `native/retry/`
  - refined-minimal `native/`: ls, `controller_admission.json`, `atlas_pose/submission.json`
  - mesh dir: ls, `controller_admission.json`
  - ls of the `-refined-whole` dir
  - `automation/scene_completion-v14/` (ls, `completion.json`)
  - `pose/task-coverage-bf4b0f3646cee779/` (ls)
  - `pose/atlas_pose_inputs.json`
- `pipeline/atlas/configuration.json` for 4b60431e, f88f1bd0, 98a79ce3, c33427c3, 05d7f712, 6940ee82
- 4b60431e (job 3): ls `atlas/automation/` and `atlas/scene_planner/`; 3 × `scene_planner/*.settlement.json`; refined splat `native/{controller_admission.json,atlas_pose/submission.json}` and the same two under `native/retry/`

Also used (pulled earlier by the parent): `scratchpad/status-1.json` (door `status`, 13:19Z) and `scratchpad/prog-447.json` (`progression.json`).
