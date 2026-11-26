# BlueprintCapturePipeline Agent Guide

This guide applies to everyone working here, human or agent. `CLAUDE.md` is a
pointer to it.

## Mission

A site films one recurring job once and uploads the video through the WebApp.
Blueprint turns it into a **job packet** that any robot deployment company can
evaluate without visiting, sends it to deployment companies Blueprint has
approved, and tells the site who will take the job and on what evidence. This
repo is the packet factory: it reads the video, builds the simulation of the
site with the job's object in it, and carries the job's success rule. WebApp
owns the site side: the job page, the first look on paper, the site's
permissions and the responses of deployment companies.

Owner direction, 2026-11-12 (it replaces the direction of 2026-11-11): the
simulation is how a job becomes testable without a visit. It is a means, not
the product. The site's answer is the responses of deployment companies; see
the shared block of `PLATFORM_CONTEXT.md`.

Owner direction, 2026-11-17: "why cant we just accept every job then determine
what is needed from there?" and "we need to think from first principles in
regards to our pipeline - only keep stuff that is needed/no extra 'rules' that
we have for no reason - and utilize ai/agents where possible when logic or
decisions are needed". Privacy, rights and money stay plain code that no model
decision can move: the site's consent and its withdrawal, the owner closing a
run, the provider terms and retention, faces blurred in the frames sent to
World Labs, and the spend limits and admissions below. In build, Pipeline
first: a website job's room step ships the room as filmed, flagged, for
anything else, and a stuck job retries by itself.

Blueprint no longer runs robot software in the simulation. The owner dropped
the early trial of openly available robot software on 2026-11-14: its timer is
retired, no release arms it, and the job packet has no trial section. A
scripted run only checks this simulation; it is never a result for any robot.
The trial's stored records stay only where withdrawal deletion and money
accounting still read them (`screening_retained_ledger`,
`website_capture_withdrawal_screening`). Some trial code is still imported by
live modules and is removed as those imports are cut; do not build on it.

Shared doctrine is generated from `doctrine/` (see `doctrine/README.md`).

## The Only Hard Stops

Fail closed only for:

- **Privacy:** the site's consent is respected, including its withdrawal.
  Faces are blurred in the frames sent to World Labs only for a job started
  under a Privacy Policy before 2026-11-17.1; footage of a later job is not
  blurred, and people in captures are not otherwise screened or blurred.
- **Rights:** capture permissions, revocation, and dataset or model licenses.
- **Money:** one cap per job (USD 40 for everything a job spends, owner
  decision 2026-11-19), the account-wide monthly limit (USD 300, owner decisions
  2026-11-07 and 2026-11-12; USD 400 for November 2026 only, owner decision
  2026-11-20), the Google Cloud
  monthly ceiling (USD 100), and the paid-resource allocator below. Never
  start paid compute or provider calls outside it. All ceilings: "Spend limits"
  below.

Everything else keeps going. If an input is missing, ambiguous or unverified,
flag it, make the best supported guess, and continue the same job. Fixing a
failure means fixing its cause; do not add a gate, approval step, review stage
or readiness check unless it protects privacy, rights or money. Delete checks
that protect none of the three.

After a failed step a job retries by itself: `job_retry_policy` decides when,
and the job's window is its deadline. A new stage controller uses that module
instead of its own retry rule; never make a deploy or an operator command the
only way to retry
([`docs/architecture/job-retry-and-deadlines.md`](docs/architecture/job-retry-and-deadlines.md)).

Contacting a company that Blueprint has not approved is outreach and needs the owner's word first. Ask the owner first before sending email by
hand or deleting retained data.

## Spend limits

The money ceilings found in code and config on 2026-11-02. Code values were
read in the files named; paths without a repo name are in this repo.
"Unverified" means that the value lives in Render or on the host, not in a
repo. The measured cost model is
[`docs/launch-readiness/2026-11-cost-model.md`](docs/launch-readiness/2026-11-cost-model.md)
(B-03). Its recommended new defaults (section 7) are not approved: they wait
for the owner (tracker D-01/D-06). Changing any value below is a money
decision for the owner.

| Limit | Value | Set in | Status |
|---|---|---|---|
| Account, all providers, per UTC month | $300 (raised from $200 on 2026-11-12); $400 for the month 2026-11 only (owner decision 2026-11-20), $300 again from 2026-12. Alerts at 50% (warn), 80% and 100% (page): $150, $240 and $300 ($200, $320 and $400 in 2026-11). At the limit, new paid admissions are refused with `account_monthly_spend_limit_reached`. Counts reconciled provider month-to-date plus held reservations, plus the paid admissions made since that count was written. A call that names its maximum cost is admitted only when the count plus that cost fits under the limit, as the job's $40 cap admits a call. | `monthly_limit_usd` in `config/account_monthly_spend_limit.v1.json`, and its one dated entry `monthly_limit_overrides` `"2026-11": "400"`, installed by the deploy. Read by `src/blueprint_pipeline/account_monthly_spend.py` at `paid_resource_admission`. If the file is missing on the control plane, new paid admissions are refused with `account_monthly_spend_limit_not_configured` and the capacity tick pages. Design and stage-2 plan: [`docs/architecture/metered-spend.md`](docs/architecture/metered-spend.md). | committed config (owner decisions 2026-11-07, "Meter + monthly cap", and 2026-11-20, $400 for November); stage 1, additive: every limit below still applies |
| Website job, everything it spends (Gemini, Meta SAM, World Labs, OpenAI with ARTISAN, Vast; every retry) | $40. Before each paid call, what the job spent plus what its calls in flight hold plus this call's quote must fit; a reservation is settled or released when its call ends. At the cap no further paid call starts: the job ships what it has, flagged `spending_cap_reached`, and waits for no one. Nothing raises it: no owner grants, no automatic extensions. | `per_job_cap_usd` in `config/website_job_spend_cap.v1.json`, mirrored byte for byte in WebApp `contracts/pipeline/website-job-spend-cap.v1.json`. The WebApp enforces it on the job's ledger (`server/utils/websiteJobCeiling.ts`, at every preparation reservation and scene intake); the Pipeline holds each scene intent to it (`task_evaluation_scene_intake.scene_spend_budget`). Design: [`docs/architecture/metered-spend.md`](docs/architecture/metered-spend.md#one-cap-at-40-2026-11-19). | committed config (owner decision 2026-11-19, "One cap at $40") |
| Atlas review, per review | at most $5 (schema) | Render `BLUEPRINT_WEBSITE_ATLAS_RUNTIME_POLICY_JSON` (`maximum_review_cost_usd`); same file | unverified (B-03 read $4 and $1) |
| Website scene sponsorship | its scene allowance (below) and 1–32 paid attempts; TTL up to 7 days (schema). Its upstream and total dollar figures, and the Atlas grants' dollar figures, are still carried for the grants' digests but no longer read as money: the job's $40 is | Render `BLUEPRINT_WEBSITE_SCENE_SPONSORSHIP_JSON`; schema in WebApp `server/utils/websiteSceneSponsorship.ts` | unverified |
| Scene budget, per website scene intent | sponsorship `native_max_spend_usd`, never above the job's $40, counted in full on the job's ledger while the intent can still start an attempt. Nothing extends it. An intent whose holds leave no room ends `spending_cap_reached`. | WebApp sponsorship; `scene_spend_budget` in `src/blueprint_pipeline/task_evaluation_scene_intake.py` | unverified (B-03: capture-447 $20 at intake) |
| ARTISAN authoring, per object, cumulative across resumes | $12 (gpt-6.1-sol; owner direction 2026-11-02, down from gpt-6-astra's $25). The build plan's first object authors inside the scene intake's allowance; each further object it generates reserves this cap as its own row on the job's WebApp ledger, in the plan's order, inside the job's $40, and settles at what it cost. When the WebApp refuses one, that object and every lower-ranked generated one stay as filmed, named, under the machine reason `spending_cap_reached`; the README says "the job reached its limit on paid work". The refusal code, `website_scene_preparation_budget_exhausted`, is kept as received: the WebApp sends it for the $40 cap and for the job's paid-request count alike, so no record says which. Library and parametric objects reserve nothing (owner decision 2026-11-23) | `ARTISAN_AUTHORING_CAP_USD` in `src/blueprint_pipeline/task_object_authoring_backend.py`, the code default and what the deploy installs as host `BLUEPRINT_ARTISAN_AUTHORING_MAX_COST_USD`; extra objects: `website_build_budget.admit_extra_objects`, through `reserve_website_preparation_spend` and WebApp `checkWebsiteJobCeiling`. Design: [`docs/architecture/metered-spend.md`](docs/architecture/metered-spend.md#extra-objects-inside-the-40-2026-11-23) | code; host value set by the next deploy (B-03: capture-447 ran at $25). Extra objects: code, called by no live route yet: since several objects per job stage 5 a job builds its free further objects (library or parametric, nothing reserved), and one that would need ARTISAN stays in the scan as filmed until stage 6 |
| Astra CAD/Blender authoring, per stage (when that backend runs) | $5 default, at most $25 | `MIN_ASTRA_AUTHORING_SPEND_USD`, `MAX_ASTRA_AUTHORING_SPEND_USD` in `src/blueprint_pipeline/task_evaluation_scene_configuration_runtime_budget.py` | code |
| Compute, per scene-configuration GPU attempt | at most $6 and $0.80/h; both shrink to fit the scene budget left | `MAX_PROVIDER_COMPUTE_SPEND_USD`, `MAX_HOURLY_RATE_USD` in `task_evaluation_scene_configuration_runtime_budget.py`; `spend_block` in `task_evaluation_scene_configuration_submission_records.py` | code |
| Rigid robot-free probe | $1.00, at most $0.80/h, 1 h TTL | `HARD_CAP_USD` in `src/blueprint_pipeline/rigid_robot_free_probe_lane.py`; reserved on the upstream ledger, then settled to the run's budget-ledger estimate once its teardown is proven (`rigid_robot_free_probe_settlement.py`) | code |
| Google Cloud project `blueprint-8c1ca` | $100 per calendar month | `limit_usd` in the root policy file named by `BLUEPRINT_GCP_BUDGET_GUARD_CONFIG`; `src/blueprint_pipeline/gcp_monthly_budget_guard.py`; [`docs/runbooks/development-cloud-cost-controls.md`](docs/runbooks/development-cloud-cost-controls.md) | owner direction 2026-10-23; host policy file unverified |
| GPU fleet guard | 10 live instances, $10/h, $100/day, $5,000 total | `BLUEPRINT_GPU_FLEET_MAX_*` in `deploy/systemd/blueprint-gpu-spend-guard.service`; `/etc/blueprint/pipeline-control-plane.env` can override | unit default; host override unverified. Totals are the guard ledger's estimate, not the bill (B-03 §8) |
| RunPod (retired 2026-11-05) | no new RunPod spend: the allocator, the shared paid admission and the RunPod render launcher refuse with `runpod_provider_retired` before any RunPod resource is created, and RunPod billing is no longer collected | `RUNPOD_PROVIDER_RETIRED` in `src/blueprint_pipeline/paid_resource_admission.py`. While a RunPod key exists, `scripts/gpu_spend_guard.py` still reads the RunPod inventory (required) and reaps an orphan pod; with no key RunPod is recorded as retired and not required (`RETIRED_PROVIDERS` in `src/blueprint_pipeline/spend_admission_lock.py`) | code. Owner step: close the RunPod account, then delete the RunPod key file (`runpod_api_key` in the provider secrets directory, and the file `RUNPOD_API_KEY_FILE` names where an env file sets it) |
| Vast credit page | pages below $11 ($10 `PAGE_CREDIT_USD` + $1 reserve) | `src/blueprint_pipeline/provider_credit_admission.py`; overrides `BLUEPRINT_VAST_CREDIT_WARNING_USD`, `BLUEPRINT_VAST_CREDIT_RESERVE_USD` | code default; host override unverified |
| Vast launch credit check | balance must cover the launch's cap + $1 reserve (each launch alone; nothing is reserved across the 3 launch slots) | `record_vast_credit_admission` in `src/blueprint_pipeline/vast_provider_adapter.py`. On by default in code (`credit_guard_setting` in `src/blueprint_pipeline/provider_credit_admission.py`; only `BLUEPRINT_VAST_CREDIT_GUARD_ENABLED=false` or `0` turns it off), and every unit that can allocate Vast still sets `=true` explicitly (`PAID_ALLOCATION_UNITS` in `src/blueprint_pipeline/task_evaluation_production_chain_preflight.py`; the chain preflight refuses one that does not) | code default and unit; host env-file override unverified |
| OpenAI spend page, one UTC day | pages at $75 of organization spend in one UTC day (today so far, or yesterday) | `OPENAI_24H_PAGE_USD` in `src/blueprint_pipeline/provider_spend_alerts.py`; override `BLUEPRINT_OPENAI_SPEND_24H_PAGE_USD` | alert line, not a ceiling; code default; host override unverified |
| World Labs spend page, 24 h | pages at $60 of Blueprint's recorded Atlas purchases in the last 24 h | `WORLD_LABS_24H_PAGE_USD`, same file; override `BLUEPRINT_WORLD_LABS_SPEND_24H_PAGE_USD` | alert line, not a ceiling; code default; host override unverified |
| Google Cloud spend page | pages at 80% of the month's effective limit ($80 of $100) | `GCP_PAGE_FRACTION`, same file; override `BLUEPRINT_GCP_SPEND_PAGE_FRACTION` | alert line, not a ceiling; the limit is unchanged; code default; host override unverified |
| WebApp open intake | 5 new jobs a day (20 per IP) | WebApp `server/utils/betaAbuseCaps.ts`; Render `BLUEPRINT_INTAKE_DAILY_LIMIT` overrides | code default; Render value unverified (tracker F-04) |

Runbooks for credit, provider spend, disk and stalls: [`docs/runbooks/beta-operations.md`](docs/runbooks/beta-operations.md).

## Two Truths

- Don't make things up: no invented results, captures, approvals or readiness.
  Unknown stays unknown.
- Simulation results are bounded estimates; they never certify physical
  performance or safety.

## Sibling Checkouts

`BlueprintCapture`, `Blueprint-WebApp` and this repo are sibling checkouts;
where they live depends on the environment. If a sibling you need is missing,
say so. The `BlueprintCapture` phone apps are not in use: a person who works at
the site films the job on their own phone and uploads it through the WebApp.
That repository is kept for its `extractFrames` upload hand-off; see "How
Capture Works" in `PLATFORM_CONTEXT.md`.

## Cloud Sessions

Cloud sessions reach the control-plane host through the operator door
(`python3 scripts/operator_door.py`, see `docs/OPERATOR_DOOR.md`), never SSH.
Setup lives in the WebApp checkout's `docs/runbooks/cloud-scene-runs.md`.

## Beta Deploy Policy

Decision D-08 (2026-11-02, delegated by the owner): every Pipeline deploy
retires and rebuilds each in-flight website attempt, so deploys are batched.

- Deploy `main` once a day, in the window that opens at 15:00 UTC.
- The door holds a deploy itself while a website job has work in flight on the
  running release (it parks, never touching the job): a paid stage until it
  ends or for 285 min at most, work that is not paid yet for 60 min at most.
  Request the deploy; if it parks, `door status` `deploy_activation_hold`
  (`stages`, `bound`) says why. This needs the installed door upgraded
  (`install.sh --upgrade` as root, `docs/OPERATOR_DOOR.md`); until then, also
  skip the window while `door status` shows a held `paid_launch_locks` entry.
- Outside the window, deploy only a fix for money, privacy, rights or a stuck
  paid stage, and name which in the request.
- Before you deploy, read `door status` (`deploys.recent_receipts`) and
  `door requests` for another operator's deploy, and record yours in the beta
  tracker.

Procedure: [`docs/runbooks/beta-operations.md`](docs/runbooks/beta-operations.md)
section 6.

## Commands

Paid resource allocation (the only supported launchers):

```bash
python -m blueprint_pipeline.paid_resource_allocator gpu-canary <arguments>
python -m blueprint_pipeline.paid_resource_allocator remote-cpu-job <arguments>
```

Install and test:

```bash
python -m pip install -e .[dev]
python -m blueprint_pipeline.impacted_test_selection  # changed tests, capped at 120s
ruff check <changed Python files>
scripts/pytest_fast.sh                                # broader, slower
scripts/dev/nonroot_pytest.sh <test paths>          # as CI's non-root user, on a root machine
```

Before merging or deploying a fix for a stopped production job, replay the job's
next step on the fix branch (it pays for nothing) and include its first stop in
the PR. Deploy to find out only what a replay cannot show, such as provider
behavior:

```bash
python -m blueprint_pipeline.task_evaluation_stage_replay --child <id> --isolate
python -m blueprint_pipeline.atlas_preview_replay --job <pulled preview_job.json>
```

Entrypoints:

```bash
python main.py
python -m blueprint_pipeline.capture_orchestrator
python -m blueprint_pipeline.runtime_service_app
```
