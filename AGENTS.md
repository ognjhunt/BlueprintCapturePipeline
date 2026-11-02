# BlueprintCapturePipeline Agent Guide

This guide applies to everyone working here, human or agent. `CLAUDE.md` is a
pointer to it.

## Mission

A site sends a video of one recurring job. This repo turns it into a
simulation of the site, runs the candidate robot policies and embodiments in it,
and returns a bounded estimate of what works, or "held — we can't simulate this
kind of job yet". Blueprint runs the candidates itself, then works alongside a
deployment company on the real site. WebApp owns the customer side.

Shared doctrine is generated from `doctrine/` (see `doctrine/README.md`).

## The Only Hard Stops

Fail closed only for:

- **Privacy:** people in captures are screened, and consent is respected.
- **Rights:** capture permissions, revocation, and dataset or model licenses.
- **Money:** the per-site budget (USD 15 for screening candidates), the Google
  Cloud monthly ceiling (USD 100), and the paid-resource allocator below. Never
  start paid compute or provider calls outside it. All ceilings: "Spend limits"
  below.

Everything else keeps going. If an input is missing, ambiguous or unverified,
flag it, make the best supported guess, and continue the same job. Fixing a
failure means fixing its cause; do not add a gate, approval step, review stage
or readiness check unless it protects privacy, rights or money. Delete checks
that protect none of the three.

Ask the owner first before contacting outside companies, sending email, or
deleting retained data.

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
| Beta screening, per site | $15 | `SITE_LIMIT` in `src/blueprint_pipeline/beta_policy_screening_coordinator.py`; `SITE_BUDGET_USD` in `beta_policy_screening_plan.py`; WebApp `SCREENING_SITE_LIMIT_USD` in `server/utils/betaPolicyScreeningContract.ts` | code |
| Beta screening, per candidate | $4.00 default | `DEFAULT_CANDIDATE_CAP_USD` in `src/blueprint_pipeline/beta_policy_screening_capabilities.py`; a catalog profile can set `screening_candidate_cap_usd` | code default |
| Website job, all ledgers (Atlas, upstream, scene, screening) | $40 default | WebApp `WEBSITE_JOB_CEILING_DEFAULT_USD` in `server/utils/websiteJobCeiling.ts`. An Atlas authorization row, a stored grant or the grant policy sets the figure first. | code default; per-job figures unverified |
| Atlas automatic grant, per capture | at most $100 and 32 requests (schema) | Render `BLUEPRINT_WEBSITE_ATLAS_GRANT_POLICY_JSON`; schema in WebApp `server/utils/websiteAtlasAuthorization.ts` | unverified (per-job allowances in B-03 §5) |
| Atlas owner row, per capture | at most $1,000 and 1,000 requests (schema) | Render `BLUEPRINT_WEBSITE_ATLAS_AUTHORIZATIONS_JSON`; same file | unverified (tracker D-02: capture-447 row $70, raise to $110 asked) |
| Atlas review, per review | at most $5 (schema) | Render `BLUEPRINT_WEBSITE_ATLAS_RUNTIME_POLICY_JSON` (`maximum_review_cost_usd`); same file | unverified (B-03 read $4 and $1) |
| Website scene sponsorship | each cap above $0 and at most $1,000; upstream + scene ≤ total; 1–32 paid attempts; TTL up to 7 days (schema) | Render `BLUEPRINT_WEBSITE_SCENE_SPONSORSHIP_JSON`; schema in WebApp `server/utils/websiteSceneSponsorship.ts` | unverified |
| Upstream ledger (SAM, Gemini, other OpenAI, rigid probe) | sponsorship `upstream_max_spend_usd`; an owner amendment can raise it to at most 3× | WebApp `server/utils/websiteSceneSponsorship.ts` (`preparationSpendLimit`) | unverified (B-03 read $5 on capture-447) |
| Scene budget, per website scene intent | sponsorship `native_max_spend_usd`; owner grants raise it, at most $1,000 | WebApp sponsorship; `python3 scripts/operator_door.py extend-scene-execution-budget` | unverified (B-03: capture-447 $20 at intake, $70 after grants) |
| ARTISAN authoring, per object, cumulative across resumes | $15 default | host `BLUEPRINT_ARTISAN_AUTHORING_MAX_COST_USD`; default in `src/blueprint_pipeline/task_object_authoring_backend.py` | code default; host value unverified (B-03: capture-447 ran at $25) |
| Astra CAD/Blender authoring, per stage (when that backend runs) | $5 default, at most $25 | `MIN_ASTRA_AUTHORING_SPEND_USD`, `MAX_ASTRA_AUTHORING_SPEND_USD` in `src/blueprint_pipeline/task_evaluation_scene_configuration_runtime_budget.py` | code |
| Compute, per scene-configuration GPU attempt | at most $6 and $0.80/h; both shrink to fit the scene budget left | `MAX_PROVIDER_COMPUTE_SPEND_USD`, `MAX_HOURLY_RATE_USD` in `task_evaluation_scene_configuration_runtime_budget.py`; `spend_block` in `task_evaluation_scene_configuration_submission_records.py` | code |
| Rigid robot-free probe | $1.00, at most $0.80/h, 1 h TTL | `HARD_CAP_USD` in `src/blueprint_pipeline/rigid_robot_free_probe_lane.py`; reserved on the upstream ledger | code |
| Google Cloud project `blueprint-8c1ca` | $100 per calendar month | `limit_usd` in the root policy file named by `BLUEPRINT_GCP_BUDGET_GUARD_CONFIG`; `src/blueprint_pipeline/gcp_monthly_budget_guard.py`; [`docs/runbooks/development-cloud-cost-controls.md`](docs/runbooks/development-cloud-cost-controls.md) | owner direction 2026-10-23; host policy file unverified |
| GPU fleet guard | 10 live instances, $10/h, $100/day, $5,000 total | `BLUEPRINT_GPU_FLEET_MAX_*` in `deploy/systemd/blueprint-gpu-spend-guard.service`; `/etc/blueprint/pipeline-control-plane.env` can override | unit default; host override unverified. Totals are the guard ledger's estimate, not the bill (B-03 §8) |
| Vast credit page | pages below $11 ($10 `PAGE_CREDIT_USD` + $1 reserve) | `src/blueprint_pipeline/provider_credit_admission.py`; overrides `BLUEPRINT_VAST_CREDIT_WARNING_USD`, `BLUEPRINT_VAST_CREDIT_RESERVE_USD` | code default; host override unverified |
| Vast launch credit check | balance must cover the launch's cap + $1 reserve | `record_vast_credit_admission` in `src/blueprint_pipeline/vast_provider_adapter.py`, with `BLUEPRINT_VAST_CREDIT_GUARD_ENABLED=true` in `deploy/systemd/blueprint-task-evaluation-launch-dispatcher.service` | code and unit |
| WebApp open intake | 5 new jobs a day (20 per IP) | WebApp `server/utils/betaAbuseCaps.ts`; Render `BLUEPRINT_INTAKE_DAILY_LIMIT` overrides | code default; Render value unverified (tracker F-04) |

Runbooks for credit, disk and stalls: [`docs/runbooks/beta-operations.md`](docs/runbooks/beta-operations.md).

## Two Truths

- Don't make things up: no invented results, captures, approvals or readiness.
  Unknown stays unknown.
- Simulation results are bounded estimates; they never certify physical
  performance or safety.

## Arm Decision Proof v1

`arm-decision-proof-v1` is a separate evidence program with its own protocol in
`docs/arm_decision_proof_v1/`. That protocol applies only to ADP work. Beta
results never enter ADP evidence, and only ADP may claim physical validation.

## Sibling Checkouts

`BlueprintCapture`, `Blueprint-WebApp` and this repo are sibling checkouts;
where they live depends on the environment. If a sibling you need is missing,
say so.

## Cloud Sessions

Cloud sessions reach the control-plane host through the operator door
(`python3 scripts/operator_door.py`, see `docs/OPERATOR_DOOR.md`), never SSH.
Setup lives in the WebApp checkout's `docs/runbooks/cloud-scene-runs.md`.

## Commands

Paid resource allocation (the only supported launchers):

```bash
python -m blueprint_pipeline.paid_resource_allocator cpu-build <arguments>
python -m blueprint_pipeline.paid_resource_allocator model-volume <arguments>
python -m blueprint_pipeline.paid_resource_allocator gpu-canary <arguments>
python -m blueprint_pipeline.paid_resource_allocator provider-reconstruction <arguments>
python -m blueprint_pipeline.paid_resource_allocator remote-cpu-job <arguments>
```

Install and test:

```bash
python -m pip install -e .[dev]
python -m blueprint_pipeline.impacted_test_selection  # changed tests, capped at 120s
ruff check <changed Python files>
scripts/pytest_fast.sh                                # broader, slower
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
