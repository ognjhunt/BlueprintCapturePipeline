# Beta operations runbooks

Short runbooks for the invited beta (tracker items D-07 and B-05 in
[`../launch-readiness/2026-11-beta-launch-tracker.md`](../launch-readiness/2026-11-beta-launch-tracker.md)).
Each section names the alarm, what it means and what to do. Every fact cites a
file. "Unknown" means that the repo does not record it.

In this file, `door` means `python3 scripts/operator_door.py`
([`../OPERATOR_DOOR.md`](../OPERATOR_DOOR.md)). Host paths are on the
control-plane host.

Before you act:

- Run `door status`. Read `active_release`, `deploys`, `door_requests`,
  `deploy_activation_hold`, `holds`, `failed_units` and `capacity`.
- Ask the owner before you spend money, delete retained data or contact an
  outside company ([`../../AGENTS.md`](../../AGENTS.md)).
- Owners and escalation:
  [`beta-ops-incident-response.md`](beta-ops-incident-response.md#owners-and-escalation).

| Section | Use it when |
|---|---|
| [1. A job stops moving](#1-a-job-stops-moving) | A stall page arrives, or a job shows the same state for hours. |
| [1a. A unit failed](#1a-a-unit-failed) | A `unit_failed:<unit>:…` post arrives, or `failed_units` names a unit. |
| [1b. The batch memory slice](#1b-the-batch-memory-slice) | A batch unit was killed for memory, or runs slowly near 4.5G. |
| [2. Vast credit runs out](#2-vast-credit-runs-out) | `provider_credit_insufficient` pages, or a launch is refused for credit. |
| [2a. Provider funding](#2a-provider-funding) | An OpenAI, World Labs or Google Cloud spend page arrives (`*_spend_*_high`). |
| [2b. Account monthly limit](#2b-account-monthly-limit) | An `account_spend_*` alert arrives, or a paid admission is refused with `account_monthly_spend_*`. |
| [3. The work volume is full](#3-the-work-volume-is-full) | A launch is refused for disk, or a capacity page arrives. |
| [4. The operator door or the host is down](#4-the-operator-door-or-the-host-is-down) | The uptime probe pages, or `door` calls fail. |
| [5. Roll back a bad deploy](#5-roll-back-a-bad-deploy) | A deploy broke production. |
| [6. Deploy policy (D-08)](#6-deploy-policy-d-08) | You are about to deploy the Pipeline. |
| [7. Triggers that are off on purpose (D-06)](#7-triggers-that-are-off-on-purpose-d-06) | `door units` shows an enabled timer or path watcher as `inactive`. |
| [8. Consent withdrawal](#8-consent-withdrawal) | A site owner withdraws consent, or you run the withdrawal drill (G-02). |

## 1. A job stops moving

The capacity controller runs every 10 minutes
(`deploy/systemd/blueprint-control-plane-capacity.timer`). Each tick also runs
three read-only stall checks (PR #3226,
`src/blueprint_pipeline/control_plane_stall_alarm.py`). A stall pages on the
capacity alert route, `BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL`. A new stall posts
at once. A stall that continues posts again each hour.

| Code | Meaning | Default limit | Do this |
|---|---|---|---|
| `scene_progression_stalled` | The `status`, `phase` and `blockers` of a scene intent did not change. | 90 min while the row names blockers. 8 h while it is handed off (`awaiting_execution` or `running`, no blockers). | Read the progression and the journal (commands below). |
| `launch_lease_stuck` | A request in `task-evaluation-launches/processing/` has no live dispatcher. | 30 min after the launch reconciler's own limit: 15 min before the allocator starts, otherwise the run's `hard_ttl_seconds`. | Make sure that `blueprint-task-evaluation-launch-reconciler.timer` is active. Read its journal. |
| `trigger_inactive_unheld` | A `.timer` or `.path` that the deploy tool always arms is inactive, with no door hold. | 60 min | Start it: `door unit start <unit>`. If it is off on purpose, put an owned hold on it with `door hold`. |
| `stall_check_incomplete` (warn) | A check could not read what it needs. | none | Read the `stall` section: `door cat /var/lib/blueprint/pipeline-control-plane/capacity/summary.json`. |

These never page: completed rows, `needs_input` rows, revoked or expired
execution windows, and intended holds. Intended holds are pauses, privacy,
consent, rights and money approvals, for example
`scene_intake_spend_cap_exhausted` and `scene_intake_attempt_cap_exhausted`.

The limits are minutes in the environment file:
`BLUEPRINT_STALL_PROGRESSION_MINUTES`, `BLUEPRINT_STALL_HANDOFF_MINUTES`,
`BLUEPRINT_STALL_LAUNCH_GRACE_MINUTES` and `BLUEPRINT_STALL_TRIGGER_MINUTES`.
`BLUEPRINT_STALL_ALARM=0` turns the alarm off. More detail is in
[`control-plane-capacity.md`](control-plane-capacity.md#stalled-jobs).

Read the job:

```bash
I=/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents/<intent>
door status
door cat "$I/progression.json"
door ls "$I/progression-events" --sort mtime
door journal blueprint-task-evaluation-scene-progression.service -n 200 --since -1h
door journal blueprint-task-evaluation-launch-reconciler.service -n 200 --since -2h
```

What frequent states mean:

| `progression.json` shows | Meaning | Do this |
|---|---|---|
| `running` / `previous_release`, blocker `previous_release_attempt_not_terminal` | A deploy moved the release. The old attempt must end first. Then scene progression starts a successor on the new release (`_release_successor` in `src/blueprint_pipeline/task_evaluation_scene_progression.py`). | Wait. See section 5. |
| `blocked` / `configuration_capacity` | A $0 refusal before allocation waits for credit or disk. | Section 2 or 3. |
| blocker `scene_intake_spend_cap_exhausted` or `scene_intake_attempt_cap_exhausted` | The scene's own budget is used up. This is a money hold, so it does not page. | Owner decision. Preview, then apply, `door extend-scene-execution-budget` ([`../OPERATOR_DOOR.md`](../OPERATOR_DOOR.md)). |
| blocker `scene_intake_authority_expired` | The execution window expired. | Owner decision. Plan, then apply, `door extend-scene-execution-window`. |

## 1a. A unit failed

Thirteen units start `blueprint-unit-failure@<unit>.service` when they fail
(`OnFailure=`): the spend guard, the intake, the capacity controller and the
units that move a website job (the handoff listener, Atlas preview and provider
poll, scene progression, launch preparation, activation and dispatcher, team
scene packet, the rigid probe runner, screening). The list is
`UNIT_FAILURE_NOTIFIED_UNITS` in `tests/test_deploy_systemd_contract.py`.

The notifier (`src/blueprint_pipeline/unit_failure_notifier.py`) does two things:

- It rewrites one record,
  `/var/lib/blueprint/pipeline-control-plane/unit-failures/<unit>.json`:
  `result`, `exit_status`, `failure_count`, first and last failure, last post.
  `door status` and the capacity summary do not show this record yet; read it
  with `door cat`. `door status` still lists the unit under `failed_units`.
- It posts on the capacity alert route, `BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL`,
  with the capacity controller's own posting code. The text reads
  `systemd unit_failed:<unit>:<result>`. A unit that fails on every run posts
  once, then once an hour.

| `result` | Meaning | Do this |
|---|---|---|
| `timeout` | The run passed its `TimeoutStartSec` and systemd stopped it. A timer starts the next run. | Read the journal for the step it hung in. |
| `oom-kill` | The kernel killed it for memory, in its own cgroup or in the batch slice (1b). | `door status`, `memory.oom_reports`. |
| `exit-code` | The worker exited non-zero. For the spend guard, status 2 means that paid admission is closed or a reap failed. | Read the journal. For the guard read `gpu_spend_guard/latest.json`. |
| `signal`, `core-dump` | The worker was killed or crashed. | Read the journal. |
| `null` | systemd did not pass the result (older than 251). | Read the journal. |

```bash
door cat /var/lib/blueprint/pipeline-control-plane/unit-failures/<unit>.json
door journal <unit> -n 200 --since -1h
```

Two units on the job's path have no notifier on purpose, because a non-zero
exit is an ordinary result there: the launch reconciler (exit 2 while a
teardown is still being proven) and the billing reconciler (exit 2 when a
provider read is blocked; the spend guard then reports it).

## 1b. The batch memory slice

The 23 units that decode video, fuse or load scenes, compile episodes or stage
GPU bundles run in `system-blueprintbatch.slice`
(`deploy/systemd/system-blueprintbatch.slice`; the list is `BATCH_SLICE_UNITS`
in `tests/test_deploy_systemd_contract.py`). The slice bounds them together:
`MemoryHigh=4.5G`, `MemoryMax=5G` (binary units; 5G is 5.37 GB). The host has
8.33 GB and 6.19 GB of it free with nothing heavy running (2026-11-10).

- Above 4.5G the kernel reclaims the slice's page cache and slows its
  allocations. Above 5G it kills inside the slice, by each unit's
  `OOMScoreAdjust`. The killed unit fails with `oom-kill` and its timer or path
  unit starts it again.
- The intake, the operator door, the spend guard, the billing reconciler, the
  capacity controller and the other money-safety and housekeeping units are
  outside the slice.
- `door status`, `memory.system_units`, lists the slice as one row with the
  batch units' total and peak. It no longer lists the batch units one by one;
  each unit's journal still records its peak when a run ends.
- A unit joins the slice the next time it starts with an empty cgroup. A run in
  flight at the deploy, and a unit whose GPU watchdog is still alive, keep the
  cgroup they have until they end.

A stage that needs more than 5G by itself now fails here every time. That is a
capacity decision for the owner (a larger host), not a setting to raise: 5G
plus the always-on units is what this host has.

## 2. Vast credit runs out

What breaks: Vast stops every instance when the account balance runs out
(`PAGE_ALERT_CODES` in `src/blueprint_pipeline/control_plane_capacity_controller.py`).
The credit guard is on by default in code; only `BLUEPRINT_VAST_CREDIT_GUARD_ENABLED=false`
(or `0`) turns it off. Every unit that can allocate Vast also sets `=true` explicitly
(`PAID_ALLOCATION_UNITS` in
`src/blueprint_pipeline/task_evaluation_production_chain_preflight.py`; the chain
preflight refuses one without it). That includes the rigid robot-free probe runner
and beta policy screening. A Vast launch then needs a fresh balance that covers
its cap plus a reserve before it allocates (`record_vast_credit_admission` in
`src/blueprint_pipeline/vast_provider_adapter.py`). If not, the launch stops
before allocation, at $0, with `provider_credit_insufficient`. Each launch is
checked alone: up to 3 launches can hold slots at once, and their caps are not
added up against the balance.

The alarm (PR #3257, `src/blueprint_pipeline/provider_credit_admission.py`):

- `provider_credit_insufficient` pages when the balance is below
  `PAGE_CREDIT_USD` ($10, two launches' compute allowance) plus the $1 reserve.
  That is, below $11. It repeats each hour while it lasts.
- The alert row in `door status` (`capacity.alerts`) carries `credit_usd` and
  `required_usd`.
- `BLUEPRINT_VAST_CREDIT_WARNING_USD` changes the $10 line.
  `BLUEPRINT_VAST_CREDIT_RESERVE_USD` changes the $1 reserve.
- `provider_credit_credentials_missing`, `provider_credit_unverifiable` and
  `provider_credit_transport_failed` are warnings only. They raise the
  capacity level to `warning` at most. Only `provider_credit_insufficient`
  makes it `critical`. A release without PR #3279 shows any of them as
  `critical`.
- Disk pages such as `reclaim_ineffective` follow the disks' own level, not
  a level that a credit or spend alert raised.
- PR #3257 merged as `882d3bd59` on 2026-11-02. A release without it warns
  once at $5 + $1 and does not repeat. Check that the active release contains
  `882d3bd59`.

Recover:

1. Ask the owner to add credit in the Vast account. This is a money action.
   The owner controls automatic recharge in the Vast account (tracker D-09).
   The code only reads the balance. It never adds credit.
2. A website scene that was refused at $0 recovers by itself. Scene progression
   reads the refusal as `preallocation_credit` (`CREDIT_KIND` in
   `src/blueprint_pipeline/task_evaluation_scene_capacity_recovery.py`).
3. Each tick reads the balance again (`capacity_admission`). While the balance
   is short, the scene shows `blocked` / `configuration_capacity` with
   `provider_credit_insufficient`.
4. When the balance covers the attempt's compute cap plus the reserve, scene
   progression makes a $0 preparation successor. The scene shows `preparing` /
   `capacity_recovery_reserved`. This works with `max_retries: 0`, because
   nothing ran.
5. The limit is 6 capacity recoveries for each intent
   (`MAX_PREALLOCATION_CAPACITY_RECOVERIES` in
   `src/blueprint_pipeline/task_evaluation_scene_recovery.py`). After that, the
   blocker is `preallocation_capacity_recovery_cap_exhausted`.

This recovery covers only a refusal before allocation. A run that started and
then lost its machine is a different case. On a website scene
(`max_retries: 0`) it stops at `scene_configuration_failed` or
`retry_cap_exhausted` (`_recover_configuration_capacity` in
`src/blueprint_pipeline/task_evaluation_scene_progression.py`).

## 2a. Provider funding

Only Vast gives this host a balance to read (section 2). For OpenAI, World
Labs and Google Cloud, each capacity tick (every 10 min) reads what Blueprint
spent and pages when the spend crosses a line (beta tracker B-08,
`src/blueprint_pipeline/provider_spend_alerts.py`). The alarm only reads. It
never adds credit, and no launch waits on it. A page goes to the capacity alert
route, `BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL`, and repeats each hour while it
lasts. As for Vast, a page makes the capacity level `critical`, and a warning
raises it to `warning` at most. Neither raises a disk page.

| Provider | How it is funded | What pages | Default line | Override in `/etc/blueprint/pipeline-control-plane.env` |
|---|---|---|---|---|
| Vast | Prepaid credit. Automatic recharge: owner to confirm (tracker D-09). | `provider_credit_insufficient`: the balance (section 2). | Below $11 | `BLUEPRINT_VAST_CREDIT_WARNING_USD`, `BLUEPRINT_VAST_CREDIT_RESERVE_USD` |
| OpenAI | Owner to confirm: prepaid credit (with or without automatic recharge) or a monthly invoice. | `openai_spend_24h_high`: organization spend in one UTC day, today so far or yesterday. `openai_spend_month_high`: the UTC month to date. | $75 a day. $560 a month. | `BLUEPRINT_OPENAI_SPEND_24H_PAGE_USD`, `BLUEPRINT_OPENAI_SPEND_MONTH_PAGE_USD` |
| World Labs (Atlas) | Prepaid credit: tasks draw from the account balance, 1,000 credits per US dollar in the beta. Automatic recharge and the account's monthly spend limit: owner to confirm. | `world_labs_spend_24h_high`: Blueprint's recorded Atlas purchases in the last 24 h. `world_labs_spend_month_high`: the UTC month to date. | $60 a day. $450 a month. | `BLUEPRINT_WORLD_LABS_SPEND_24H_PAGE_USD`, `BLUEPRINT_WORLD_LABS_SPEND_MONTH_PAGE_USD` |
| Google Cloud project `blueprint-8c1ca` (Gemini) | A Cloud Billing account that Google charges after use. Payment setup: owner to confirm. | `gcp_spend_month_high`: the budget guard's cost for the month. | 80% of the month's limit: $80 of $100. | `BLUEPRINT_GCP_SPEND_PAGE_FRACTION`, more than 0 and at most 1 |

**RunPod is retired** (owner decision, 2026-11-05). Nothing can start RunPod
spend: the canonical allocator, the shared paid admission
(`RUNPOD_PROVIDER_RETIRED` in `src/blueprint_pipeline/paid_resource_admission.py`)
and the RunPod render launcher refuse with `runpod_provider_retired` before any
RunPod resource is created. The billing reconciler no longer requests RunPod billing (it had
failed on every call: 85 of 85 on 2026-11-05). While a RunPod key exists, the
GPU spend guard still reads the RunPod inventory as a required provider: an
orphan pod is reaped, and it keeps provider-zero and paid admission closed until
it is gone. The paid-spend admission lock lists RunPod as retired, not as
uncovered billing. Owner step: close the RunPod account, then delete the RunPod
key file (`runpod_api_key` in `/etc/blueprint/provider-secrets`, and the file
`RUNPOD_API_KEY_FILE` names where an env file sets it). With no key the guard
records RunPod as retired and not required, and Vast, DigitalOcean and GCP stay
required exactly as before.

The lines come from the B-03 cost model
([`../launch-readiness/2026-11-cost-model.md`](../launch-readiness/2026-11-cost-model.md)):

- One website job at its caps costs $30.91 of OpenAI: ARTISAN's $25
  cumulative cap, $0.91 of Atlas planner and review calls (measured on
  capture-447), and the $5 upstream ledger. It costs $25 of World Labs: the
  default Atlas allowance. capture-447 measured $17.93 of OpenAI. Its World
  Labs cost was about $8.60 at rate-card prices, part of it estimated.
- The plan is 2 jobs a day and 15 a month. B-03 found 15 website jobs in the
  30 days to 2026-11-02. The WebApp admits at most 5 new jobs a day.
- Each line is jobs × cost per job × 1.2, rounded up to $5. The 20% is for
  spend outside website jobs. Raise the lines when the job rate goes up.

These are alert lines, not ceilings. A bad override (not a positive number,
or a fraction above 1) keeps the default and warns
`provider_spend_config_invalid`. `BLUEPRINT_PROVIDER_SPEND_ALERTS=0` turns the
alarm off.

**OpenAI.** OpenAI has no balance API. The alarm reads the organization Costs
API (`src/blueprint_pipeline/openai_organization_costs.py`) with the Admin key
in `OPENAI_ADMIN_API_KEY_FILE`. The capacity unit sets it to
`/etc/blueprint/provider-secrets/openai_admin_api_key`, the file that the
launch dispatcher uses. Every project in the organization counts.

- Cost data lags. OpenAI reports costs by UTC day, and a day's figure keeps
  growing for hours after the spend. The alarm sees a cost only when OpenAI
  reports it, so a page can come hours after the spend. The alarm also checks
  yesterday, because late costs land in yesterday's figure after midnight. A
  page therefore lasts until the day that crossed the line is older than
  yesterday: about 48 h at most.
- The balance and the recharge settings are on the OpenAI platform's billing
  page (owner login).
- Whether the WebApp's OpenAI key is in the same organization: owner to
  confirm. If it is not, the WebApp's OpenAI spend is not counted.

**World Labs.** The Atlas API has no balance, usage or billing endpoint (its
API reference, read 2026-11-02). The alarm counts Blueprint's own record of
each Atlas attempt under
`/var/lib/blueprint/pubsub-handoffs/*/scenes/*/captures/*/pipeline/atlas`
(`BLUEPRINT_WEBSITE_ATLAS_PREVIEW_STORAGE_ROOT`):

- A completed operation counts at its reviewed price. An operation that may
  still be charged counts in full. A failed operation or a refused send counts
  $0, the same rule as `src/blueprint_pipeline/atlas_spend_settlement.py`.
- A purchase is dated by its last send.
- World Labs use outside website jobs is not counted.
- Check the balance itself on the World Labs dashboard: Settings → Billing
  (`https://atlas-beta.worldlabs.ai/settings/billing?tab=developer`). Daily
  spend is at `https://atlas-beta.worldlabs.ai/usage?tab=developer`. With no
  credit left, Atlas refuses tasks with HTTP 402 `CREDITS_UNAVAILABLE`. At the
  account's monthly limit it refuses them with `SPEND_LIMIT_EXCEEDED` (World
  Labs billing documentation, read 2026-11-02).

**Google Cloud (Gemini).** The WebApp and the Pipeline call the Gemini API
with `GEMINI_API_KEY` or `GOOGLE_GENAI_API_KEY`. The Google Cloud project that
owns the key pays. That this project is `blueprint-8c1ca`: owner to confirm.
The Gemini lanes already handle the error "prepayment credits are depleted"
(`src/blueprint_pipeline/wam_generated_video_success_label_gemini.py`).
Whether the key uses Gemini API prepayment: owner to confirm. This alarm
cannot see a prepaid Gemini balance.

- The alarm reads the monthly budget guard's report. That is the
  `report_path` in the policy that `BLUEPRINT_GCP_BUDGET_GUARD_CONFIG` names,
  `/var/lib/blueprint/pipeline-control-plane/gcp-monthly-budget/latest.json`
  on the host. `blueprint-gpu-spend-guard.timer` writes it every few minutes
  ([`development-cloud-cost-controls.md`](development-cloud-cost-controls.md#monthly-admission-and-automatic-stop)).
- The line is a share of the month's effective limit, so a dated owner
  exception moves it too. The alarm never changes the limit. At 100%, the
  guard itself refuses paid work.
- Google's cost reports are hours late, so the 80% line leaves room.

These codes warn and never page:

| Code | Meaning |
|---|---|
| `openai_spend_credentials_missing` | The Admin key file is not set, is missing, or is not owner-only (`reason`). |
| `openai_spend_transport_failed` | The request failed. `http_status` 401 or 403 means OpenAI refused the key. |
| `openai_spend_unverifiable` | OpenAI's answer was malformed (`reason`). |
| `world_labs_spend_unverifiable` | The job root is missing, an attempt record is unreadable, or the walk stopped early. A walk that stopped early is a lower bound, and it still pages on what it counted. |
| `gcp_spend_unverifiable` | No guard policy, a report that is unreadable, stale or for another month, or no cost reported yet this month (`reason`). |
| `provider_spend_config_invalid` | An override is bad (`reason` names it). The default line applies. |
| `provider_spend_check_failed` | The check stopped with an error. The tick went on. |

When a spend page arrives:

1. Read the alert in `door status` (`capacity.alerts`: `spend_usd`,
   `threshold_usd`, `window`). For the current figures of all three
   providers, read `provider_spend` in
   `door cat /var/lib/blueprint/pipeline-control-plane/capacity/summary.json`.
2. Ask the owner to check the provider's balance on its own dashboard, and to
   add credit if it is low. This is a money action. Recharge settings are the
   owner's account settings.
3. If the spend is not expected, find what spent it: OpenAI usage by project,
   World Labs usage by API key or project, or the Google Cloud billing report.
   For World Labs, also look at the Atlas folders of recent jobs.
4. A page for an expected busy day stops by itself when that day leaves the
   window. To move a line, the owner edits the environment file over SSH
   (section 4) and records a break-glass note ([`break-glass.md`](break-glass.md)).

## 2b. Account monthly limit

The owner set one account-wide limit on 2026-11-07: **USD 200 per UTC
month**, across every provider. The design and the stage-2 plan are in
[`../architecture/metered-spend.md`](../architecture/metered-spend.md).

- **Where the amount lives.** `monthly_limit_usd` in
  `config/account_monthly_spend_limit.v1.json`. It is committed and the
  deploy installs it.
- **What counts.** Reconciled provider month-to-date plus held reservations.
  The spend guard rewrites the report about once a minute, to
  `/var/lib/blueprint/pipeline-control-plane/account-monthly-spend/report.json`.
- **Where it is enforced.** At `paid_resource_admission`. Admitted work,
  teardown and reconciliation are never stopped by it.

| Alert or refusal | Means | Do |
|---|---|---|
| `account_spend_month_50pct` (warn) | The month has counted $100. | Nothing. Read the jobs (below) if it is early in the month. |
| `account_spend_month_80pct` (page) | $160. | Tell the owner. Find the largest jobs and any unexpected spend. |
| `account_spend_month_limit_reached` (page); refusal `account_monthly_spend_limit_reached` | $200 counted: new paid admissions are refused until the month ends or the owner raises the limit. | Tell the owner. Only the owner can raise the limit (a money decision). A one-month raise goes in `monthly_limit_overrides` (`{"2026-11": "300"}`), in a reviewed commit, then a deploy. |
| `account_spend_report_stale` (page); refusals `account_monthly_spend_report_stale` / `_missing_or_invalid` | The spend guard has not rewritten the report for 15 min, so every paid admission is refused. | Check `blueprint-gpu-spend-guard.service` (`door status`, `failed_units`). For up to a minute after a deploy or after 00:00 UTC on the 1st, this is expected. |
| `account_spend_limit_config_invalid` (page) | The installed config does not validate, so every paid admission is refused. | Fix the file in a reviewed commit and deploy. |
| `account_spend_unverifiable` (warn) | A provider reading is missing or stale (`reason`). The month counts what it can see. | See section 2a for that provider. |
| `account_spend_limit_not_set` (warn) | No config is installed: the month is reported, nothing is refused. | Expected only off the host or before the deploy. |

To read the month:

1. `door status`: `spend_guard.account_monthly_spend` (limit, counted, held,
   providers, the ten largest jobs) and
   `capacity.provider_spend.providers.account`.
2. On the host: `python -m blueprint_pipeline.account_monthly_spend` (the
   admission decision) and `python -m blueprint_pipeline.paid_spend_meter_sources`
   (every job's metered spend).

Holds count in full on top of billing. A busy day can therefore read high
until attempts settle. Provider bills lag by hours.

## 3. The work volume is full

What breaks: a scene-configuration launch stops before allocation, at $0, with
`scene_configuration_provider_output_disk_budget_exceeded`. The disk
reservation ledger could not hold the output's upload ceiling plus 512 MiB
(`MEASURED_BLOCKER` in
`src/blueprint_pipeline/task_evaluation_scene_capacity_recovery.py`). Ceiling
mode uses `scene_configuration_provider_output_disk_capacity_insufficient`.

Beta screening and the rigid robot-free probe stage new attempts on the work
volume: `/mnt/blueprint-work/beta-policy-screening-work` and
`/mnt/blueprint-work/rigid-robot-free-probe-work` (each unit's `*_WORK_ROOT`
setting). Their `policy_canary_output` disk hold therefore draws on this
volume, not on `/`. Their queues and the screening budget ledger stay on `/`.
An attempt staged before that setting keeps its queue paths. A screening
candidate refused with `policy_canary_output_disk_admission_refused` is out
of room on the disk its attempt directory is on.

The capacity controller pages for disk with these codes: `floor_within_three_days`,
`floor_within_hours`, `admission_refused`, `critical_admission_refused`,
`reclaim_ineffective`, `reclaim_state_unknown` and `volume_growth_blocked`.
`reclaim_state_unknown` pages a critical mount once when the GC summary is
missing, more than two hours old, not `applied`, or reports phase errors or
skipped roots; it names the reason and the summary's age and clears when a
fresh clean summary arrives. It never fires with `reclaim_ineffective` for the
same mount. `reclaim_ineffective` is the one page for a critical mount the GC
cannot relieve: it names the mount, its
free, floor and available bytes and the top retained reasons, and its webhook
`cites` the usage warnings behind it. `storage_class_over_budget`,
`orphan_scratch_large` and `usage_unclassified_root` warn (WP-19, 2026-11-10).
See [`control-plane-capacity.md`](control-plane-capacity.md).

Automatic recovery (PR #3251):

- The launch result must prove that nothing started (`nothing_started`): no
  instance, no allocation, no API pretraining, and
  `provider_mutations_performed` 0.
- Then scene progression treats the refusal like a credit refusal. The scene
  shows `blocked` / `configuration_capacity` with
  `preallocation_capacity_not_recovered`.
- Each tick measures free space for the CPU chain, the output and the next
  bundle (`capacity_admission`). When the space is there, it makes a $0
  preparation successor.
- The same limit of 6 recoveries for each intent applies.
- A refusal after paid API pretraining does not retry. Its blockers gain
  `preallocation_capacity_recovery_withheld:<reason>`.

Make space:

1. Run `door status`, then `door usage`. Find the full mount and what holds it.
2. Run storage GC: `door unit start blueprint-control-plane-storage-gc.service`.
   Read its plan and receipt.
3. If GC frees too little, the volume must grow. Growth costs money, so it is
   an owner decision (tracker D-10: +200 GiB is about +$20 a month).

Automatic growth is opt-in for each host
(`deploy/systemd/blueprint-control-plane-capacity.service`). When the volume's
mount is `critical`, the controller grows the volume one step, up to the
maximum. It calls the DigitalOcean API, then runs `resize2fs`
(`plan_volume_resize` and `resize_volume` in
`src/blueprint_pipeline/control_plane_capacity_controller.py`). These lines go
in `/etc/blueprint/pipeline-control-plane.env`:

```bash
BLUEPRINT_CAPACITY_VOLUME_ID=<DigitalOcean volume id>
BLUEPRINT_CAPACITY_VOLUME_MOUNT=<mount of the work volume>
BLUEPRINT_CAPACITY_VOLUME_DEVICE=<block device under /dev>
BLUEPRINT_CAPACITY_VOLUME_MAX_GIB=<owner-approved maximum>
BLUEPRINT_CAPACITY_AUTORESIZE_ACK=grow-control-plane-volume
# Optional. The default step is 50 GiB.
BLUEPRINT_CAPACITY_VOLUME_STEP_GIB=50
```

- The controller also reads the DigitalOcean token from
  `DIGITALOCEAN_API_TOKEN_FILE`, which the unit sets.
- Whether the host has these lines today is unknown. `door status` shows
  `capacity.volume_resize` only when they are set.
- `volume_at_maximum` needs a provider limit increase first.
- Only SSH from the owner's Mac can edit the file (section 4). Record a
  break-glass note for the edit ([`break-glass.md`](break-glass.md)).

## 4. The operator door or the host is down

The alarm is the WebApp's external uptime probe (WebApp PR #1176,
`scripts/ops/uptime-probe.mjs` in the WebApp checkout):

- It checks
  `https://paperclip.tryblueprint.io/api/live-pipeline/operator/v1/healthz`.
  The check needs no token.
- It runs every 10 min as the Render cron job `blueprint-uptime-probe`. The
  WebApp's `.github/workflows/uptime.yml` is the backstop.
- A failure posts to the ops Slack webhook as
  `control-plane=HTTP_<status> (Pipeline control-plane host or operator door down)`.
- At 2026-11-02 14:45Z, two steps were still open (tracker D-02): the WebApp
  deploy, and a manual deploy of the cron job, which has `autoDeploy: false`.

First check, from any machine:

```bash
curl -fsS https://paperclip.tryblueprint.io/api/live-pipeline/operator/v1/healthz   # expect {"ok": true}
```

If the check fails, a cloud session cannot help: the door is its only path to
the host. The door runs apart from the pipeline release, from
`/opt/blueprint/operator-door` ([`../OPERATOR_DOOR.md`](../OPERATOR_DOOR.md)).

Only the owner, with SSH from the owner's Mac, can do these (WebApp
`docs/runbooks/cloud-scene-runs.md`, "What still needs the Mac"):

- Look at the door units and Caddy:
  `systemctl status blueprint-operator-door.service blueprint-operator-door-runner.path caddy`.
- Probe the door on the host:
  `curl -fsS http://127.0.0.1:8767/api/live-pipeline/operator/v1/healthz`.
- Repair the door: run `install.sh --upgrade` as root, from a commit on `main`
  ([`../OPERATOR_DOOR.md#install-tokens-upgrade-removal`](../OPERATOR_DOOR.md#install-tokens-upgrade-removal)).
  If the health check fails, the installer restores the previous door.
- Edit `/etc/blueprint` config or env files, and rotate provider keys.
- Debug a process (py-spy, gdb), or SSH into a GPU machine.
- Record a break-glass note for each change ([`break-glass.md`](break-glass.md)).

If SSH also fails, the host itself is down. Host recovery is the owner's
action. This repo has no runbook to restart or rebuild the host (unknown).

## 5. Roll back a bad deploy

Deploy an earlier commit through the door:

```bash
python3 scripts/operator_door.py deploy <previous sha on main> --wait
```

Before you deploy:

1. Find the last good commit. Use `door status` (`deploys.recent_receipts`, the
   five most recent), or
   `door ls /var/lib/blueprint/pipeline-control-plane/deploy-receipts --sort mtime`.
   The commit must be on `origin/main`.
2. List what the rollback removes: `git log --oneline <previous>..<active>`.
   Every commit in that range leaves production. Fixes from other sessions in
   that range also leave.
3. Do not redeploy an older commit before you read that list. If it removes
   work of another session, ask the owner first.
4. A revert merged on `main` is a newer commit. It removes only what it names.
5. Read `door status` (`deploy_activation_hold`). While a scene-configuration
   activation is live, the door holds the deploy. The hold ends when the
   activation ends, or after at most 285 min (`CAP_SECONDS` in
   `deploy/operator-door/operator_door/activation_hold.py`).
6. `--activation-hold-override <reason>` deploys at once. Use it only for
   money, privacy, rights or a fix to that activation.
7. Request the rollback when no other deploy runs. A rollback that waits behind
   another deploy ends `parked_rollback_refused`
   ([`../OPERATOR_DOOR.md`](../OPERATOR_DOOR.md#what-a-door-deploy-does)).

After the deploy:

- Every deploy, a rollback too, retires each in-flight website attempt and
  rebuilds it on the new release. That is about 15 min of preparation, then
  about an hour of activation (`activation_hold.py`). Later stages refuse work
  that is bound to an older release.
- Until the old attempt ends, the scene shows `running` / `previous_release`
  with `previous_release_attempt_not_terminal`.
- Read the receipt `deploy-receipts/iteration_<sha12>_door.json` and
  `door status` (`active_release`).

## 6. Deploy policy (D-08)

Why: every deploy retires each in-flight website attempt and rebuilds it on the
new release (section 5), and it pauses paid launches for about 7 minutes. So
deploys are batched once a day and held while an external job is in a paid
stage. The rule is in `AGENTS.md` ("Beta Deploy Policy").

The window opens at 15:00 UTC (09:00 CST). Deploy what `main` holds then.

Before you deploy:

1. Read `door status`. Do not deploy while any of these hold:
   - a `paid_launch_locks` entry with `held: true`;
   - `deploy_activation_hold` is not null (the door holds the deploy itself
     while a scene-configuration activation is live, section 5);
   - `beta_screening.candidate_counts.launching` or `.running` is above 0.
   Wait for them to clear, then deploy.
2. Read `door requests` and `deploys.recent_receipts`. If another operator has a
   deploy pending, processing, or finished in the last hour, do not stack a
   second one: use theirs, or wait for it.
3. List what ships: `git log --oneline <active release>..origin/main`. A fix for
   a stopped production job needs its replay first (`AGENTS.md`, "Commands").
4. Request it:

   ```bash
   python3 scripts/operator_door.py deploy <sha on origin/main> --wait
   ```

   Without `--release-provenance` this is an iteration deploy. Its gate is main
   CI's supply-chain attestation for the commit, not a Full Test Lane run: a
   commit main CI has not attested does not deploy, and nothing pauses before
   the attestation is listed
   ([`../CI_REQUIRED_CHECKS.md`](../CI_REQUIRED_CHECKS.md), "Deploy gates").

5. Record the deploy in
   [`../launch-readiness/2026-11-beta-launch-tracker.md`](../launch-readiness/2026-11-beta-launch-tracker.md):
   the commit, the time, and what it carried.

Outside the window:

- Deploy only a fix for money, privacy, rights, or a paid stage that is stuck.
  Say which in the request.
- `--activation-hold-override <reason>` is for the same cases only.
- A rollback follows section 5.

WebApp merges are separate. A WebApp merge to `main` deploys to Render through
its CI-gated workflow, so treat every WebApp merge as a deploy, but it does not
retire Pipeline attempts.

## 7. Triggers that are off on purpose (D-06)

`door units` shows these timers and path watchers as enabled but `inactive`.
They are off on purpose. A deploy restores each unit's earlier state unless the
unit is in `DEFAULT_ALWAYS_ARM_TIMER_UNITS`
(`scripts/deploy_control_plane_commit.py`), so they stay off. Checked
2026-11-03 at 01:58Z.

| Unit | What it does | Why it is off | To turn it on |
|---|---|---|---|
| `blueprint-provider-billing-reconciler.timer` | Refreshes provider billing evidence every 10 minutes. | Not needed. `blueprint-gpu-spend-guard.service` `Wants=` the reconciler, so the reconciler runs with every guard start. Since #3278 (E-02), it refreshes at most every 10 minutes, and only while a live provider needs it. | Nothing to do. |
| `blueprint-task-evaluation-launch-supervisor.timer` | Supervises launches with the OpenAI Agents SDK. This is optional. | It spends OpenAI money on every launch. | The owner decides the money first. |
| `blueprint-task-evaluation-terminal-resource-release.path` | Watches the terminal resource release queue. It releases one terminal Vast record through the canonical allocator. | The queue has been empty since 2026-08-12 (one completed record). When `BLUEPRINT_ALLOW_TASK_EVALUATION_TERMINAL_RESOURCE_RELEASE_TRIGGER` is set, the intake starts the release unit directly (`live_pipeline_intake_runtime_controls.py`). Teardown runs in the paid lane. The launch reconciler files the provider-zero receipt. | Add it to `DEFAULT_ALWAYS_ARM_TIMER_UNITS` if anything writes to its `pending/` folder again. |
| `blueprint-capture-reconstruction-dispatcher.timer` and `.path` | Advanced the legacy capture reconstruction queue. | Retired: the website beta reconstructs through Atlas. The unit files, the dispatcher module and the listener's enqueue are gone from the repo (P-RT-01); the queue (`capture-reconstruction-queue/{pending,processing,completed}`) was empty. | The host still has the installed unit files until the host cleanup (H-03) disables and removes them. |

Not done: `reset-failed` on the transient failed units (O-24). It needs a door
unit action, and another operator uses the host, so do it in a deploy window.

## 8. Consent withdrawal

A site owner withdraws in the WebApp. The WebApp's Privacy page promises an
acknowledgement within 10 business days and completion within 30 days
(Blueprint-WebApp `client/src/pages/Privacy.tsx`).

### What happens by itself

1. The WebApp's withdrawal worker runs every 60 seconds. It sends the Pipeline a
   signed, non-destructive withdrawal command
   (Blueprint-WebApp `server/utils/websiteCaptureWithdrawal.ts`), and sends the
   same command again every hour while the withdrawal is pending.
2. The intake writes a tombstone at `<site>/website_withdrawal/tombstone.json`
   and answers with an acknowledgement (`website_capture_withdrawal.py`,
   `acknowledge_withdrawal`). From then on the site's capture reads as revoked
   (`consent_takedown.read_consent_state`).
   - The paid screening launch refuses the capture with
     `team_scene_capture_withdrawn`, through
     `beta_policy_screening_worker.require_current_source`, which resolves the
     capture with `team_scene_capture_root.resolve_capture_root`.
   - The Atlas preview queue and the scene preview publisher also refuse a
     withdrawn capture.
3. The WebApp's deletion worker deletes the WebApp's own copies for the site
   (Blueprint-WebApp `server/utils/websiteCaptureDeletion.ts`). These are the
   raw and upload prefixes of the site's captures, the job item photos and the
   published scene preview reference. It waits for the Pipeline's
   acknowledgement, deletes every object generation, writes a receipt, and stops
   if the Pipeline reports a legal hold.

Nothing in the Pipeline is deleted by itself. A person runs the Pipeline's
deletion plan and its `--apply`
(`python -m blueprint_pipeline.website_capture_withdrawal_deletion`):

- Only the owner can run them today, because only the owner has SSH access to
  the host. Operators reach the host only through the door
  ([`../OPERATOR_DOOR.md`](../OPERATOR_DOOR.md)), and no door request kind runs
  the deletion.
- Every `--apply` needs the owner's approval for that site.
- The plan reads every run's records, so run it on a quiet host.

Known gaps:

- The screening ledger's own reservation
  (`beta_policy_screening_coordinator._reserve`) and `enqueue_screening_campaign`
  do not read the tombstone. A campaign that was already planned can still
  reserve its cap. The launch then refuses before any spend, as step 2 says.
- The WebApp's acknowledgement schema is `.strict()` (`websiteWithdrawalReceipt`).
  So the acknowledgement cannot carry anything about a Pipeline deletion until
  the WebApp accepts a new optional field.

### Make and read a plan

On the host, the owner runs:

```bash
python -m blueprint_pipeline.website_capture_withdrawal_deletion \
    --capture-root <site>/captures/walkthrough-<request id> --object-store --bucket <source bucket>
```

The plan only reads and deletes nothing. The `<site>` directory is
`scenes/site-<request id>/` under the intake's capture root. The printed
`plan` has these fields:

- `unresolved`: each copy the plan could not resolve, with a reason. This
  includes a run root it cannot list (`website_withdrawal_deletion_root_unreadable`).
  `--apply` refuses while any row remains.
- `runs`, `objects`, `scene_files`, `screening`, `native_g1`, `team_scene`:
  what an apply would delete.
- `uncovered_pipeline_roots`: Pipeline roots the deletion does not scan. In the
  receipt, each one is listed under `remaining`, with what the root holds.
- `legal_hold`: if this is `true`, `--apply` refuses.
- `digest`: an apply needs exactly this digest.

The owner applies with the same command plus `--expected-plan-digest <digest>
--apply --authorize-complete-deletion`. The sealed receipt is at
`<site>/website_withdrawal/deletion_receipts/<digest>.json`. Read it with
`door cat`. It contains:

- `deleted`: what the apply removed.
- `remaining`: what stays and why. This covers provider copies, WebApp and
  delivered copies, the capture source bucket when no bucket was named, the
  roots the deletion does not cover and the retained audit journal.
- `policy_copy_classes`: one row for each data class in
  [`../beta_data_retention_policy_2026-07-09.json`](../beta_data_retention_policy_2026-07-09.json),
  with what was removed and what stays, and why.

### Drill

The drill runs offline, costs $0 and runs in CI
(`tests/test_website_capture_withdrawal_drill.py`):

```bash
python -m blueprint_pipeline.website_capture_withdrawal_drill
```

It works in a new temporary directory, with one synthetic site, one other site,
an in-memory B2 store and an in-memory source bucket. It:

1. acknowledges the shared WebApp vector
   (`tests/fixtures/website-withdrawal-vector.json`);
2. checks that a dry-run plan deletes nothing;
3. checks that a paid launch for the site is refused;
4. plans with the object store and the bucket, then applies with the plan's
   digest and the authorization;
5. prints one row for each copy class: the count before, the count after, and
   why anything remains.

It fails if a covered copy remains, if a policy data class has no row, if
anything of the other site changed, or if it wrote, removed or renamed anything
outside its temporary directory.

A drill on a fresh synthetic job on the host:

1. Ask the owner first. A new job can start paid preparation, and an apply
   deletes retained data.
2. Submit a synthetic job from an operator test account, with a synthetic
   walkthrough video. Note its request id.
3. Withdraw that job in the WebApp. Within a few minutes the WebApp record shows
   `pipelineAcknowledged: true`.
4. Find the site with `door ls <capture root>/scenes --match site-<request id>`
   and read `door cat <site>/website_withdrawal/tombstone.json`.
5. The owner makes the plan over SSH, as in "Make and read a plan". Record its
   `unresolved`, `runs`, `objects` and `uncovered_pipeline_roots` in tracker row
   G-02.
6. Apply only if the owner approves it for this synthetic site. Then read the
   receipt with `door cat`, and record its `remaining` and
   `policy_copy_classes` in G-02.
