# Metered spend: one account-wide monthly limit

Owner decision, 2026-11-07: **"Meter + monthly cap"**, set the same day to
**USD 200 a month**.

Future jobs get no budget or confirmation of their own. Every paid call is
metered against its job and shown. Alerts go at 50 %, 80 % and 100 % of one
account-wide monthly limit ($100, $160, $200). Spend stops only at that limit.
The provider runaway guards stay.

| Stage | Scope | Status |
|---|---|---|
| 1 | Add the meter, the account limit, its report and its alerts. Remove nothing. | Built (this change). |
| 2 | Remove the per-job reservations, grants and confirmations listed under [Stage 2](#stage-2-inventory-and-plan). | Planned here, not built. |

Until stage 2, both systems apply at once. A paid admission must pass every
existing per-job cap, grant and reservation, the provider guards, and now the
account limit.

## The model

```
 paid admission ──► paid_resource_admission (the one chokepoint)
                      │  refuses: GCP $100 guard, account limit, RunPod retired, ...
                      │  journals every grant ──► paid_spend_meter (per job)
                      ▼
                 provider call

 every minute: GPU spend guard ──► account_monthly_spend_report.refresh_report
     reads: billing export + receipts (Vast, DigitalOcean, GCP)
            GCP monthly budget report
            capacity tick's OpenAI and World Labs readings
            meter: journal + screening ledger + Atlas attempts
     writes: account-monthly-spend/report.json  ◄── read by the chokepoint

 every 10 min: capacity tick ──► provider_spend_alerts
     writes its OpenAI / World Labs readings for the guard
     reads the account report ──► 50 / 80 / 100 % alerts
```

## What is metered, and where

`src/blueprint_pipeline/paid_spend_meter.py` (the light half, imported on
every paid admission) and `paid_spend_meter_sources.py` (the readers).

One meter row is one paid admission and, once known, its settlement. Each row
is keyed by a job. For a website job the key is the WebApp request id: the
Pipeline names the job's scene and screening site `site-<request id>`.

| Source | What it records | Held | Settled | Job key |
|---|---|---|---|---|
| `paid_resource_admission` journal (new) | Every grant `require_paid_resource_admission` issues, in `<meter root>/events/<YYYY-MM>/uid-<uid>.jsonl`. | `admission["meter"]["held_usd"]` when the admission names one. No lane names one yet, so in stage 1 these rows count calls (`unpriced_admissions`), not dollars. | `record_settlement(...)`, the stage-2 API. No caller yet. | `admission["meter"]["job_id"]`, else the admission's `job_id` / `website_request_id` / `request_id`, else `BLUEPRINT_PAID_SPEND_JOB_ID`, else `unattributed` |
| Beta screening ledger (existing, read in place) | Each attempt in `screening_ledger.v1.json`. | Its cap while `reserved`, or while `committed` without a verified settlement. | `settled_cost` (`beta_policy_screening_settlement`), dated `settled_at`. | campaign `site_id` |
| Atlas attempts (existing, read in place) | Each attempt `provider_spend_alerts.observe_world_labs_spend` counts this month. | Its reviewed price while the outcome is unknown. | Its reviewed price once completed. A failed or refused attempt counts $0 (`atlas_spend_settlement`). | the `scenes/<scene>` path component |

Nothing is copied. The screening ledger and the Atlas records already hold
admission and settlement per job, so the meter reads them.

How the view counts:

- A settled row counts in the UTC month it settled.
- A held row counts in every month its attempt could still be billed in
  (`hold_counts_in_month`). The attempt's own record decides. A screening
  attempt that is committed and keeps its whole cap for good
  (`settlement_held`) ended no later than its `held_at`; a journal admission
  may record `billable_until_epoch`. A hold whose attempt ended before the
  month began is not counted: the provider billed it, if at all, in an earlier
  month. An attempt live across 00:00 UTC on the 1st counts in both months.
- A hold with no recorded end is open and counts every month: a reservation,
  a committed attempt not decided yet, or a record whose times do not agree.
- A hold that no longer counts is listed under `old_unsettled_holds`, with the
  sum in `old_unsettled_held_usd`, so someone settles it. `door status` shows
  both under `spend_guard.account_monthly_spend`.
- The totals are the sums of the job rows (tested).

The meter root is `BLUEPRINT_PAID_SPEND_METER_ROOT`. On the host the default is
`/var/lib/blueprint/pipeline-control-plane/paid-spend-meter`, used only where
`/var/lib/blueprint/pipeline-control-plane` exists. Writing the journal is best
effort: a unit whose sandbox cannot write there records nothing, and its
admission proceeds. The limit reads reconciled billing, not the journal.

To see the meter:

- `python -m blueprint_pipeline.paid_spend_meter_sources [--job <request id>]`
  prints every job, Atlas included: it reads the capacity tick's retained World
  Labs reading, as the account report does;
- `door status` shows the ten largest jobs under
  `spend_guard.account_monthly_spend.jobs`.

## The account's month

`src/blueprint_pipeline/account_monthly_spend_report.py`, called by
`scripts/gpu_spend_guard.py` (`_refresh_account_monthly_spend`) on every run,
about once a minute.

```
counted = Σ_provider max(reconciled month-to-date, meter settled this month)
        + meter holds outside the provider readings
```

| Provider | Reconciled month-to-date |
|---|---|
| Vast, DigitalOcean | The provider billing export (`provider_billing_reconciler`) is cumulative since the cohort start (2026-01-01). The month's spend is that total less the same provider's total in the last retained reconciliation receipt (`billing-audit/<time>/provider_billing_source_receipt.json`) written before the month began. With no such receipt, the first receipt of the month stands in, flagged `billing_month_baseline_after_month_start`. |
| Google Cloud | The larger of two readings: the monthly budget guard's report, counted only when its billing month (Los Angeles) equals the UTC month, and the export's month delta. |
| OpenAI | The capacity tick's organization Costs API reading, persisted to `account-monthly-spend/provider_observation.json`. |
| World Labs | The capacity tick's recorded Atlas purchases. These already include held attempts, so Atlas holds are not added a second time. |

**Holds.** The meter's open holds from every source except Atlas: screening
caps, and journal holds once lanes price them. Holds are counted on top of
billing. An in-flight attempt is therefore counted twice: once in its partial
bill, once in its whole hold. That only makes the limit trip earlier. A hold
whose attempt ended in an earlier month is not counted (see "How the view
counts"); before 2026-11-10 it reduced every later month for good.

**Missing readings.** A monitoring failure never lowers the count within a
month. Each run reads the previous report:

- A provider never counts less than it did there. If its reading failed, or
  came back lower, the earlier figure is kept, marked `stale` (failed) or
  `partial` (lower), with `held_from_previous_report` and the `observed_at`
  of the reading it keeps.
- A source of holds that cannot be read (the screening ledger, the journal)
  keeps the holds it last had.

**A failed billing call.** The billing reconciler keeps the last good Vast or
DigitalOcean total for 24 hours after a call fails
([`PAID_SPEND_ADMISSION_LOCK.md`](../PAID_SPEND_ADMISSION_LOCK.md)). The report
counts that total from the time it was really read and marks the provider
`carried_forward`, with the failure, never `observed`.

A held or carried figure, or a reading with nothing earlier to keep, sets
`complete: false`, names the provider under `gaps`, and raises the
`account_spend_unverifiable` warning. It does not stop work by itself:
everything except money keeps going, and the provider guards still bound each
provider. A new month starts from its own readings. One consequence: a World
Labs attempt held at full price that later fails keeps counting until the
month ends.

**Claim boundary.** Provider readings lag their bills by hours. The limit
stops new paid admissions, not charges already running. It is a monthly stop,
not an invoice-exact ceiling.

## The limit

**Configuration.** `config/account_monthly_spend_limit.v1.json`, committed and
reviewed:

```json
{
  "schema_version": "blueprint.account_monthly_spend_limit.v1",
  "monthly_limit_usd": "200",
  "monthly_limit_overrides": {},
  "report_path": "/var/lib/blueprint/pipeline-control-plane/account-monthly-spend/report.json",
  "decision": {"by": "owner", "on": "2026-11-07", "note": "..."}
}
```

- **Where it is read.** The deploy promotes the source tree to
  `/opt/blueprint/task-evaluation-control-plane`. The code reads
  `/opt/blueprint/task-evaluation-control-plane/config/account_monthly_spend_limit.v1.json`
  (`INSTALLED_CONFIG_PATH`). `BLUEPRINT_ACCOUNT_MONTHLY_SPEND_LIMIT_CONFIG`
  can name another file.
- **How it is read.** As `gcp_monthly_budget_admission.load_policy` reads the
  GCP policy: a regular file, never a symlink, at most 64 KiB, strictly
  validated. `monthly_limit_usd` must be a string of dollars, with at most two
  decimals and greater than 0. Unknown keys are refused, and `report_path` must
  be absolute. Its sha256 is bound into every report.
- **No file on the control plane refuses.** The limit must exist in
  production (`limit_required`). Production is either of two signs:
  `BLUEPRINT_LAUNCH_PROOF_MODE=production` in the process environment, or the
  host's installed environment file `/etc/blueprint/pipeline-control-plane.env`
  (only the host install writes it). The state directory is not a sign,
  because any process that runs with the default roots creates it. There,
  with no file at the path, every new paid admission is refused with
  `account_monthly_spend_limit_not_configured`, the report says
  `cap_status: missing`, and the capacity tick pages. Stops are still admitted.
- **No file anywhere else means report only.** On a machine with neither sign
  (tests, development, a replay off the host), nothing is refused. The report
  and the alerts still run, and say `cap_status: not_set`.
- **A file that does not validate refuses** every new paid admission with
  `account_monthly_spend_limit_config_invalid` and pages. Money fails closed.

**To change the amount**, edit `monthly_limit_usd` in a reviewed commit and
deploy. A dated exception for one month goes in `monthly_limit_overrides`
(`{"2026-12": "300"}`); it never changes the recurring amount. Host files are
not edited by hand. Changing the amount is a money decision for the owner.

**Enforcement.** `paid_resource_admission._account_monthly_limit_blockers`
runs in `require_paid_resource_budget` (the pre-reservation check) and in
`require_paid_resource_admission` (where every grant is issued). It refuses
when:

| Refusal | When |
|---|---|
| `account_monthly_spend_limit_reached` | The report's `counted_usd` is at or over the month's limit. |
| `account_monthly_spend_report_stale` | The report is older than 15 min (`REPORT_MAX_AGE_SECONDS`) or is for another month. |
| `account_monthly_spend_report_missing_or_invalid` | The report is unreadable, not bound to this config's sha256 and limit, or not `cap_status: set`. |
| `account_monthly_spend_limit_config_invalid` | The config exists but does not validate. |
| `account_monthly_spend_limit_not_configured` | No config file, on the production control plane. |
| `account_monthly_spend_limit_unavailable` | The check itself raised. |

**Stops are never blocked.** `require_paid_resource_admission` takes
`spend_effect`, which defaults to `new_spend`, so an unmarked caller is
checked. A stop must be declared twice: the caller passes `spend_effect="stop"`
and the admission it built and wrote carries `"spend_effect": "stop"`. Only
then is the account limit skipped. A stop declared once is refused
(`paid_resource_admission_spend_effect_mismatch`), and so is an unknown value
(`paid_resource_admission_spend_effect_invalid`). A stop still passes every
other admission check.

The stops that use the chokepoint today:

| Stop | Caller |
|---|---|
| Stop a qualification-session component | `qualification_control_admission.py` (`action="stop"`); `run`, `restart`, `refresh` and `install-checkpoint` stay `new_spend`. |
| Close out a warm scene-configuration session (destroys the retained instance) | `task_evaluation_scene_configuration_allocator.py` (`warm_action="closeout"`); `iterate` stays `new_spend`. |

Terminal resource release and reaping do not pass through the chokepoint.

The limit is never checked when an issued grant is re-validated
(`require_paid_resource_admission_grant`). Work already admitted, teardown and
reconciliation are never stopped by it.

The GCP monthly check (`_monthly_budget_blockers`) is unchanged. It runs on
every `require_paid_resource_admission`, stops included, wherever it is active:
`BLUEPRINT_GCP_MONTHLY_BUDGET_REPORT` set (no committed unit sets it), or
`provider_reconstruction_api` with `BLUEPRINT_GCP_BUDGET_GUARD_CONFIG`.

**Files on the host.** Directory
`/var/lib/blueprint/pipeline-control-plane/account-monthly-spend/`, owned by
`blueprint` with mode 0755:

- `report.json`: written by the spend guard, mode 0644;
- `provider_observation.json`: written by the capacity tick, mode 0644.

The capacity unit creates the directory (`ExecStartPre=-+`) and may write to it
(`ReadWritePaths=-`). The guard also creates it on its first run.

To read the decision on the host:

```bash
python -m blueprint_pipeline.account_monthly_spend          # summary + admission decision
python -m blueprint_pipeline.account_monthly_spend --full   # the whole report
```

`door status` shows the same summary under `spend_guard.account_monthly_spend`,
and under `capacity.provider_spend.providers.account`.

## Alerts

`provider_spend_alerts` (each capacity tick) reads the account report:

| Code | Severity | When |
|---|---|---|
| `account_spend_month_50pct` | warn (delivered once) | counted ≥ $100 |
| `account_spend_month_80pct` | page (hourly) | counted ≥ $160 |
| `account_spend_month_limit_reached` | page (hourly) | counted ≥ $200: new paid admissions are refused |
| `account_spend_limit_not_set` | warn | No config file, off the control plane. The row carries the month's spend. |
| `account_spend_limit_missing` | page (hourly) | No config file on the production control plane: new paid admissions are refused. |
| `account_spend_limit_config_invalid` | page | The config exists but does not validate. |
| `account_spend_report_stale` | page | A limit is set and the report is stale or unreadable: admission is refused. |
| `account_spend_unverifiable` | warn | A reading is missing, or (with no limit) the report is unreadable (`reason`). |

The existing OpenAI, World Labs and GCP pages are unchanged.

## Known edges in stage 1

- **First deploy.** Until the guard writes its first report (the guard timer
  starts 15 s after a deploy and runs about once a minute), every paid
  admission is refused with `account_monthly_spend_report_missing_or_invalid`.
  A beta-screening candidate refused in that window is blocked by name and
  releases its reservation. Nothing is spent. Deploy outside a paid stage, as
  D-08 already requires. Later deploys keep the last report, which stays valid
  for 15 min.
- **Month rollover.** For up to a minute after 00:00 UTC on the 1st, the
  report names the previous month and admission refuses with
  `account_monthly_spend_report_stale`. The GCP guard behaves the same way
  (its month is Los Angeles).
- **Unattributed calls.** Admissions that carry no job id are journaled as
  `unattributed`. The website preparation receipts (Atlas, upstream) carry the
  request id only if the WebApp echoes it. Stage 2 passes `meter` blocks.
- **Lanes refuse in their own way.** A lane meets the new refusal codes as it
  meets any other admission refusal. Only the Atlas automation treats a
  refusal as a wait, and only for the GCP report. Stage 2 makes
  `account_monthly_spend_limit_reached` a wait everywhere (hold the job, do not
  block a candidate).
- **Posting time.** A Vast charge posted after midnight on the 1st for usage
  on the 31st counts in the new month.

## Stage 2: inventory and plan

Every existing per-job (or per-site, per-intent, per-capture) budget, grant,
reservation and confirmation found on 2026-11-07, with what stage 2 does to it.

- *Remove*: the account limit replaces it.
- *Keep (runaway)*: it bounds a single provider run, call or attempt count,
  not a job's total.
- *Replace*: its record becomes meter admissions and settlements.

Pipeline paths are under `src/blueprint_pipeline/` unless shown. WebApp paths
are in `Blueprint-WebApp`.

### Pipeline: the website job's ledgers

| # | Item | File:line | Stage 2 | Risk |
|---|---|---|---|---|
| P1 | Screening site allowance, $30 per site | `beta_policy_screening_coordinator.py:62` (`SITE_LIMIT`); `beta_policy_screening_plan.py:58` (`SITE_BUDGET_USD`); WebApp mirror W8 | Remove | One site can use more of the month. The account limit and the per-candidate cap (P3) bound it. The WebApp publication contract mirrors the figure, so change both repos together. |
| P2 | Screening reservations and commitments | `beta_policy_screening_coordinator.py:177` (`_budget`), `:701` (`_new_attempt` reserves the cap) | Replace: journal admission at allocation (`held_usd` = cap), settlement at the settled cost | Screening publications carry `site_budget` and `spent_upper_bound_usd`, and the WebApp (`screeningPublicationTransition`, `betaPolicyScreeningContract.ts:375`) refuses decreases. Version the publication. |
| P3 | Per-candidate cap, $4 default | `beta_policy_screening_capabilities.py:24` (`DEFAULT_CANDIDATE_CAP_USD`) | Keep (runaway) | Low. |
| P4 | Screening settlement evidence | `beta_policy_screening_settlement.py:157` (`attempt_settlement`), `:264` (`settled_cost`) | Keep the evidence; call `record_settlement` with its figure | Low. This proof is what makes a settlement trustworthy. |
| P5 | Screening plans sized to the job ceiling | `beta_policy_screening_job_ceiling.py:93` (`plan_within_job_ceiling`), `:54` (`local_screening_holds`), `:164` (`site_screening_remaining`) | Remove with W1 | Low. It also removes a hold (`website_job_ceiling_unavailable`) that stops planning when the WebApp is unreachable. |
| P6 | Website preparation reservation (upstream ledger and Atlas allowance) | `website_task_context.py:138` (`reserve_website_preparation_spend`; `require_paid_resource_budget` at `:168`; WebApp `preparation-spend`) | Replace: issue the paid-lane admission locally with a `meter` block (job, held = `maximum_cost_usd`); settle into the journal | **High.** The same WebApp transaction checks the sponsorship's consent, owner and provider terms, which are rights hard stops. Split rights from money first; keep the rights checks. |
| P7 | Atlas settlement to the WebApp ledger | `atlas_spend_settlement.py:72` (`settle_closed_atlas_attempt`), `:137` | Replace the target with `record_settlement`; keep the outcome rules (`:45`) | Low. |
| P8 | Scene spend cap per intent, and owner budget grants | `task_evaluation_scene_intake.py:337` (`spend_cap_exhausted`); `task_evaluation_scene_execution_budget.py:32` (`extend_scene_execution_budget`); `task_evaluation_scene_execution_budget_evidence.py:100` (`effective_execution_budget`); `task_evaluation_scene_execution_budget_door.py:24` (`door extend-scene-execution-budget`) | Remove the dollar cap and the grant path (a per-job confirmation) | Medium. The WebApp's job ceiling (`nativeIntakeMicros`) and the scene outcome (`spend_cap_reached`, `task_evaluation_scene_outcome.py:96`) read it. The stall alarm maps it to `money_approval` (`control_plane_stall_alarm.py:117`). |
| P9 | Scene paid-attempt cap per intent (1–32, grants to 100) | `task_evaluation_scene_intake.py:333` (`attempt_cap_exhausted`); `task_evaluation_scene_execution_budget_evidence.py:22` (`MAX_EXTENDED_PAID_ATTEMPTS`) | Keep (runaway retry limit); drop only the owner-grant path | Low. Retry limits stay. |
| P10 | Scene reservation exposure: holds at full cap until proven | `task_evaluation_scene_reservation_spend_evidence.py:19`; `task_evaluation_scene_spend.py:49`; `task_evaluation_terminal_scene_attempt_settlement.py:266` (`budget_retained_hold`); `task_evaluation_retained_controls_evidence.py:491`; `task_evaluation_unstarted_controls_reservations.py:20`; `task_evaluation_unentered_authoring_budget.py:180` | Replace with meter rows. The terminal-settlement evidence becomes `record_settlement`. Retire the project-exposure publication once the meter covers it. | Medium. Activation gates on the publication's freshness (`project_spend_stale`). The account report must cover that before the gate goes. |
| P11 | Per-attempt runtime caps: $6 compute, $0.80/h, $12 attempt, $6 external, Astra $5–$25 | `task_evaluation_scene_configuration_runtime_budget.py:47`, `:48`, `:52`, `:53`, `:62`, `:67`; `task_evaluation_scene_configuration_submission_records.py:195` (`spend_block`) | Keep (runaway). Stop shrinking them to "scene budget left". | Low. |
| P12 | ARTISAN authoring cap, $12 per object; its meter | `task_object_authoring_backend.py:47` (`ARTISAN_AUTHORING_CAP_USD`); `task_object_artisan_metering.py:24` | Keep (runaway); emit journal settlements per job | Low. |
| P13 | Rigid robot-free probe: $1 cap, $0.80/h, 1 h TTL, reserved on the upstream ledger | `rigid_robot_free_probe_lane.py:77` (`HARD_CAP_USD`), `:542`; `rigid_robot_free_probe_settlement.py:79` | Keep the cap (runaway); replace the upstream-ledger reservation with the meter | Low. |
| P14 | Pre-reservation budget check | `paid_resource_admission.py:150` (`require_paid_resource_budget`) | Keep. It then checks only the account limit, GCP and the class. | Low. |

### Pipeline: account-wide guards (all kept, stage 1 and 2)

| Guard | File:line | Why it stays |
|---|---|---|
| GCP project, $100 a month | `gcp_monthly_budget_admission.py:75`; `paid_resource_admission.py:118` | Provider runaway guard; owner said keep. |
| Cohort hard stop, $5,000, and its two-person override | `spend_admission_lock.py:25` (`HARD_STOP_USD`), `:243` (`validate_spend_override`) | An account-level backstop above the monthly limit. The override is an account confirmation, not a per-job one. |
| GPU fleet: 10 live, $10/h, $100/day, $5,000 total | `scripts/gpu_spend_guard.py:509`; `deploy/systemd/blueprint-gpu-spend-guard.service` | Runaway guard. |
| Vast credit admission | `provider_credit_admission.py:125` | Funding guard. |
| Provider spend pages | `provider_spend_alerts.py:92` | Alerts, not ceilings. |

### Pipeline: other lanes' campaign budgets (not on the website job path)

`production_gpu_campaign_budget.py:25`, `judge_spend_governor.py:45`,
`isaac_g1_spend_guard.py:9`, `native_task_arena_spend_ceiling.py:9`,
`policy_ranking_cosmos_reasoner_gpu_admission.py:61`. These belong to
development, ADP or research lanes. Keep them as runaway guards; stage 2 does
not touch them.

### WebApp

| # | Item | File:line | Stage 2 | Risk |
|---|---|---|---|---|
| W1 | Website job ceiling, $40 default | `server/utils/websiteJobCeiling.ts:22`, `:41`, `:192`; callers `server/utils/websiteSceneSponsorship.ts:1043`, `server/routes/internal-capture-worlds.ts:346` | Remove the refusal. Keep `websiteJobSpend` (`:169`) as the job page's read-only metered view. | Medium. Both callers are files another agent is editing now; sequence after that work lands. |
| W2 | Atlas automatic grant, at most $100 and 32 requests | `server/utils/websiteAtlasAuthorization.ts:91`, `:92`, `:100` (`BLUEPRINT_WEBSITE_ATLAS_GRANT_POLICY_JSON`) | Remove the dollar figure; keep the request count (runaway) | Low. |
| W3 | Atlas owner authorization rows, at most $1,000 and 1,000 requests | `server/utils/websiteAtlasAuthorization.ts:29`, `:30`, `:68` (`BLUEPRINT_WEBSITE_ATLAS_AUTHORIZATIONS_JSON`) | Remove the money fields (per-job confirmation) | **High.** The same rows carry rights: provider terms, private processing, training and retention (`:24`–`:28`). Keep those. |
| W4 | Atlas review cost, at most $5 per review | `server/utils/websiteAtlasAuthorization.ts:230` | Keep (runaway, per call) | Low. |
| W5 | Website scene sponsorship caps: upstream, native, total, 1–32 paid attempts, TTL | `server/utils/websiteSceneSponsorship.ts:51`–`:63` | Remove the dollar caps; keep `max_paid_attempts` (runaway) and the job window | **High.** The same grant carries owner and provider terms (rights). Edit only the money fields. |
| W6 | Upstream-ledger owner amendment, up to 3×; request-limit amendment | `server/utils/websiteSceneSponsorship.ts:677` (`preparationSpendLimit`), `:729`, `:766` | Remove (per-job confirmations) | Low once W5 is done. |
| W7 | Preparation reservations and settlement | `server/utils/websiteSceneSponsorship.ts:927` (`reserveWebsitePreparationSpend`), `:608` (`settleWebsitePreparationSpend`), `:829`; route `server/routes/internal-capture-worlds.ts:246`, `:274`, `:307` | Replace with the Pipeline meter (P6), or keep as a display mirror | **High.** A mirrored contract; the consent checks live in the same transaction. |
| W8 | Screening site limit mirror, $30 | `server/utils/betaPolicyScreeningContract.ts:22` (`SCREENING_SITE_LIMIT_USD`) | Remove with P1 | Mirrored contract; change both repos together. |
| W9 | Open intake, 5 jobs a day | `server/utils/betaAbuseCaps.ts:44` | Keep: an abuse cap, not a job budget | Low. |

### Stage 2 order

1. Make `account_monthly_spend_limit_reached` a hold-and-wait in every lane:
   the screening coordinator, the scene progression and Atlas, like the
   existing GCP stale wait (`paid_resource_allocator.py:799`). Then reaching
   the limit pauses jobs instead of blocking candidates.
2. Pass a `meter` block (job id, held amount) at every chokepoint call. Call
   `record_settlement` from each lane's existing settlement evidence (P4, P7,
   P10, P12, P13). Check that per-job totals match the old ledgers for a week.
3. Separate rights from money in the WebApp grants (W3, W5, W7). Keep every
   consent, terms and retention check.
4. Remove the per-job dollar caps and confirmations in mirrored pairs: P1+W8,
   P5+W1, P8+W6, P6+W7, then W2 and W3's money fields.
5. Retire the scene project-exposure publication (P10) once the account report
   carries its holds, and the screening site budget (P2) once publications are
   versioned.
6. Keep every runaway guard listed as *Keep* above. Revisit the $5,000 cohort
   stop with the owner.

## Scene ceiling as an expected budget (2026-11-10)

Owner decision 2026-11-10. A website scene's spend ceiling is its expected
budget. The account monthly limit is the one hard money stop.

- **Standing extension.** The WebApp may set `automatic_budget_extension: true`
  on a newly issued grant. A reservation that does not fit such a job's
  ceiling then raises the ceiling by exactly the shortfall, when all of these
  hold:
  - the account's fresh report has the headroom, after counting every
    standing extension issued since that report;
  - the job has paid attempts left;
  - the new ceiling stays within the $1,000 schema bound.

  Records go in `<intent>/standing-budget-extensions/`
  (`scene_budget_standing_extension`). That store is separate from the owner's
  `execution-budget-extensions/` and keeps its own sealed chain under the same
  intake lock. An older release ignores it. Otherwise the job waits, and its
  next timed re-ask asks again. Every paid launch is still checked at
  `require_paid_resource_admission`. A flagged job sizes each attempt at its
  full standard allowance instead of shrinking compute into what is left.
- **Settling at the bill.** A settled GPU run keeps its final Vast charge
  instead of its whole GPU allowance, capped at the reservation
  (`scene_attempt_measured_compute`). A charge is final when the teardown is
  complete, the spend snapshot is at least 48 h newer than the teardown, and
  the instance's rows cover every day of the run.
- **Early warning.** `scene_budget_doubled` pages once per job per whole
  multiple of the grant's original figure (`scene_budget_alert`, in the stall
  tick). It gates nothing.
