# System map

What connects to what in the free beta, end to end, as of 2026-11-05. This page
describes the current state only. Read it to find the unit, module, store and
service at each step, then follow the links for detail. Arm Decision Proof v1
(`docs/arm_decision_proof_v1/`) has its own protocol and is not described here,
except where beta code shares its paths.

Evidence for every claim here, and what was not checked, is in
[`audits/2026-11-simplification-audit.md`](audits/2026-11-simplification-audit.md).

Notation:
- `pcp/` = `/var/lib/blueprint/pipeline-control-plane/`, `tei/` = `/var/lib/blueprint/task-evaluation-inputs/`, `ph/` = `/var/lib/blueprint/pubsub-handoffs/<bucket>/scenes/<scene>/captures/<capture>/`.
- Units are `deploy/systemd/<name>`. Modules are `src/blueprint_pipeline/<name>.py`.
- Gates: **[PRIVACY]**, **[RIGHTS]**, **[MONEY]**. These are the only hard stops (`AGENTS.md`). Everything else flags and continues.

## How a unit starts its module

Most units do not run their module directly. Read the `--worker`, `--continuous-module` or `--action-module` argument, not the first `-m`:

- `python -m blueprint_pipeline.task_evaluation_scene_retirement_supervisor --worker blueprint_pipeline.<module> -- <args>`. The supervisor accepts only workers in its `_WORKERS` allowlist (`task_evaluation_scene_retirement_supervisor.py:1124`), loads them from the bound release source and runs them (`main`, lines 1168-1188). Fourteen units use it. Retiring a worker means updating that allowlist too.
- `/usr/bin/python3 -I -S /usr/lib/blueprint/scene-retirement-runtime/continuous_bootstrap.py --continuous-module blueprint_pipeline.<module>`. This is installed from `scripts/scene_retirement_continuous_bootstrap.py` by `scripts/install_scene_retirement_runtime.py`. Used by the intake, agent execution and storage GC.

The full list of what every unit, drop-in, door script and deploy step starts is in [`audits/2026-11-inventory.json`](audits/2026-11-inventory.json) (rows with `kind: "unit"`).

The deploy tool `scripts/deploy_control_plane_commit.py` installs the units in `DEFAULT_DEPLOYED_SYSTEMD_UNITS` (line 428). It always arms the triggers in `DEFAULT_ALWAYS_ARM_*` (lines 540-650). Every other trigger keeps its earlier state. Triggers that are off on purpose: [`runbooks/beta-operations.md` §7](runbooks/beta-operations.md#7-triggers-that-are-off-on-purpose-d-06).

## The website job, step by step

```
 site (browser)                        WebApp (Render)                          control-plane host (DigitalOcean)                 rented GPU (Vast)
 ─────────────                         ───────────────                          ─────────────────────────────────                 ─────────────────
 1 email + task ──────────────▶ /api/inbound-request → inboundRequests
 2 brief  ◀──────────────────▶ /api/site-task-brief → siteTaskBriefs
 3 video ─────────────────────▶ /capture-upload/:token → Firebase Storage
                                 raw/capture_upload_complete.json ──▶ Cloud Function storage-trigger ──▶ Pub/Sub ──▶ 4 pubsub-handoff-listener
                                                                                                                    │ [RIGHTS][PRIVACY]
                                                                                       5 website-atlas-preview ◀────┤ (World Labs, faces blurred) [MONEY][PRIVACY]
                                 scene sponsorship + intent ──────────── signed POST ──▶ 6 intake :8765 → scene intent [MONEY][RIGHTS]
                                                                                       7 scene-progression → preparation queues
                                                                                       8 launch-activation → launch-dispatcher ──▶ paid_resource_allocator ──▶ scene configuration (ARTISAN) [MONEY]
                                                                                       9 launch-reconciler: provider-zero receipt ◀───────────────────────────── teardown
                                                                                      10 team-scene-packet (CPU)
                                                                                      11 rigid-robot-free-probe-runner ──▶ allocator ($1 cap) ──────────▶ robot-free probe [MONEY]
                                                                                      12 beta-policy-screening ──▶ allocator ──────────────────────────▶ policy canary worker (Isaac Sim) [MONEY]
                                 /api/internal/pipeline/beta-policy-screening-runs ◀── 13 publication
 14 result page + email ◀──── taskEvaluationPolicyRuns → outcome email (Resend)
```

| # | Step | Unit or route | Module | Reads | Writes (store) | External service | Gate |
|---|---|---|---|---|---|---|---|
| 1 | Start a job | `POST /api/inbound-request` (App Check), WebApp `server/routes/inbound-request.ts` | — | email, one-sentence task | Firestore `inboundRequests` | — | intake caps `server/utils/betaAbuseCaps.ts` |
| 2 | Brief | `/api/site-task-brief` (`server/routes/site-task-brief.ts`) | WebApp | inbound request, video analysis (Gemini) | `siteTaskBriefs`, `siteTaskItemInventories` | Gemini | — |
| 3 | Upload | page `/capture-upload/:token` (`client/src/pages/SelfCaptureUpload.tsx`), `/api/self-capture/uploads`, `/api/capture-uploads` | WebApp `server/utils/captureParts.ts`, `websiteCaptureDelivery.ts` | video bytes | Firebase Storage `blueprint-8c1ca.appspot.com/scenes/<scene>/captures/<capture>/raw/…`, then `raw/capture_upload_complete.json` | Firebase Storage | recording-rights attestation `2026-11-02.v2` (`client/src/lib/recordingRightsStatement.ts`, `server/utils/websiteCaptureOwnerObservation.ts`; the Pipeline accepts only the versions in `capture_original_owner_observer.py`) **[RIGHTS]**; footage only for a saved job (F-07); no people screen at upload (D-16) |
| 4 | Hand-off to the host | Cloud Function `storage-trigger` (`functions/storage_trigger.py:437` `on_storage_finalize`, deployed by `deploy/scripts/deploy.sh:775`) → topic `capture_bridge_handoff` → subscription `blueprint-pipeline-handoff-listener` (`deploy/terraform/main.tf:988,1069`) → `blueprint-pubsub-handoff-listener.timer` (every minute). The host owns the retry: each tick re-runs a failed preparation from its kept message bytes on the one retry policy (`job_retry_policy`), reclaims an attempt whose process died, and keeps a delivery refused before its claim (`website_preparation_retry`); Pub/Sub's five deliveries are no longer the bound | `pubsub_handoff_listener` → `run_e2e` → `site_package_orchestrator.run_qualification_pipeline` → `website_capture_preparation` | Pub/Sub message, raw capture | `ph/` (capture copy, `pipeline/*`, `capture_descriptor.json`), `/var/lib/blueprint/pubsub-handoffs/.website_preparation_*` | GCP Pub/Sub, GCS; OpenAI (candidate ranking `website_candidate_ranking`), Meta hosted SAM 3.1 (`meta_sam31`) | revocation and authority endings stop the job (`authority_ending`) **[RIGHTS]**; clean-plate privacy gate (`site_package_orchestrator._privacy_postprocess_gate`, applied in `website_capture_preparation.py:109`) **[PRIVACY]**; upstream ledger (SAM, Gemini, OpenAI) under the website sponsorship **[MONEY]** |
| 5 | Atlas reconstruction | `blueprint-website-atlas-preview.timer`, `blueprint-website-atlas-provider-poll.{timer,path}` | `website_atlas_preview` → `website_atlas_pipeline` → `website_worldlabs`; route `BLUEPRINT_WEBSITE_ATLAS_ROUTE` (full or `geometry_only`) | `ph/pipeline/atlas/preview_job.json` | Atlas bundle under `ph/pipeline/atlas/`; served by the intake at `GET /api/live-pipeline/website-atlas-previews/<capture>/bundle` | World Labs | faces blurred in every frame sent (`atlas_privacy_screen.py`, refuses if the model is missing) **[PRIVACY]**; Atlas grant and per-capture cap from the WebApp **[MONEY]** |
| 6 | Scene intent | WebApp `server/utils/websiteSceneSponsorship.ts` builds the intent (`server/utils/taskEvaluationSceneIntake.ts`, Firestore `taskEvaluationSceneIntakes`) → signed `POST /api/live-pipeline/task-evaluation-scene-intents` (`taskEvaluationSceneIntake.ts:543`) → Caddy (`deploy/caddy/Caddyfile`) → `blueprint-pipeline-intake.service` on 127.0.0.1:8765 | `live_pipeline_intake_service` + `task_evaluation_scene_intake_http` | the signed intent | `pcp/task-evaluation-scene-intents/<intent>/intent.json` | — | scene sponsorship and execution window (`website_scene_sponsorship.json`; door `extend-scene-execution-budget` / `-window`) **[MONEY][RIGHTS]** |
| 7 | Scene preparation | `blueprint-task-evaluation-scene-progression.timer` (every minute) | `task_evaluation_scene_progression` (config `/etc/blueprint/task-evaluation-scene-progression.json`) → `task_evaluation_scene_preparation_service`; child queues `task_evaluation_sam31_preparation_queue`, `task_evaluation_launch_preparation_queue` | intent, `ph/` | `pcp/task-evaluation-scene-intents/<intent>/progression.json`, `pcp/task-evaluation-owned-scene-preparations`, `tei/completed-scene-preparation/<intent>/<attempt>/` | B2 (artifact store) | scene budget and attempt caps (`scene_intake_spend_cap_exhausted`; for a Website scene `max_paid_attempts` counts its scene-configuration rows reserved since 2026-11-10) **[MONEY]**; whole-chain disk admission (refuses at $0); a configuration launch whose own records prove $0 gets a $0 successor by itself, with backoff, at most 6 (`task_evaluation_scene_launch_recovery`; [`runbooks/beta-operations.md` §2](runbooks/beta-operations.md#2-vast-credit-runs-out)) |
| 8 | Paid scene configuration | `…-launch-activation.{path,timer}` → `pcp/task-evaluation-launches/pending` → `…-launch-dispatcher.path` | `task_evaluation_scene_configuration_activation_automation`, `task_evaluation_launch_activation_worker`, `task_evaluation_launch_dispatcher` → `paid_resource_allocator gpu-canary` → `vast_provider_adapter`; on the GPU: `scripts/task_evaluation_scene_configuration_provider_runner.py` → `task_evaluation_scene_configuration_artisan_driver` → `task_object_artisan_process` | activation intent, prepared references | `pcp/task-evaluation-launch-activations`, `pcp/task-evaluation-launch-runs/<launch>/launch_receipt.json`; configured scene in B2 under `…/arm-decision-proof-v1/configured-scenes/website-…` (the prefix name is historical; this is beta data) | Vast; OpenAI (ARTISAN authoring) | the allocator is the only paid launcher (`AGENTS.md` Commands); Vast credit admission (`provider_credit_admission.record_vast_credit_admission`, called from `vast_provider_adapter.py:8227`); per-attempt caps (`task_evaluation_scene_configuration_runtime_budget.py`); ARTISAN cap $12 per object **[MONEY]**; a machine lost mid-run is retried only within the intent's signed `max_retries` (1 for a Website grant issued since 2026-11-10, else 0) and the same caps |
| 9 | Teardown proof | `blueprint-task-evaluation-launch-reconciler.timer` | `task_evaluation_launch_reconciler` (reads `pcp/gpu_spend_guard/latest.json`) | launch state, spend-guard inventory | provider-zero receipt, `pcp/task-evaluation-launch-reconciliation/latest.json`; launch progress to the WebApp (`task_evaluation_launch_webapp_sync`) | Vast (read) | provider-zero before a lease closes **[MONEY]** |
| 10 | Completion and team half | inside scene progression (`task_evaluation_scene_preparation_completion`), then `blueprint-team-scene-packet.path` | `team_scene_packet_worker` → `team_scene_packet`; offer via `agent_execution_offer.publish_agent_execution_offer` | completed preparation | `pcp/team-scene-packets/{pending,…}`, `tei/team-scene-halves`, `tei/team-scene-profiles`; robot fit table (`website_robot_fit_*`) | WebApp (offer republish) | a site's unconfirmed standard success thresholds are assumed and marked so (`website_owner_success_assumption`); the site can still confirm (door `confirm-owner-success-contract` or the job page), which is recorded in the status (`confirmed_later`) and recompiles nothing; withdrawal deletes the site's halves, derived rows and packet records (`website_capture_withdrawal_team_scene`) **[PRIVACY]** |
| 11 | Robot-free probe | `blueprint-rigid-robot-free-probe-runner.{path,timer}` | `rigid_robot_free_probe_runner` → allocator; `rigid_robot_free_probe_settlement` | `pcp/rigid-robot-free-probes/pending` | `pcp/rigid-robot-free-probes/{jobs,done}`, `/mnt/blueprint-work/rigid-robot-free-probe-work` | Vast | $1.00 cap, $0.80/h, 1 h TTL, on the upstream ledger, settled after teardown **[MONEY]** |
| 12 | Candidate screening | `blueprint-beta-policy-screening.timer` | `beta_policy_screening_worker` → `_coordinator` (ledger, `SITE_LIMIT`), `_plan` (catalog `config/beta_policy_candidates.v1.json`), `_launch` (allocator `gpu-canary --probe-kind …`), `_settlement`, `_holds`; bundle `native_task_arena_policy_canary_bundle`; on the GPU: `native_task_arena_policy_canary_worker` → `adp009d_isaac_episode_adapter` (live, despite the name) in NGC Isaac Sim 6.0.1 | `pcp/team-scene-packets`, catalog, `docs/experiments/policy_ranking_thesis_20260726/*.json` (OpenPI inventories, read at runtime) | `pcp/beta-policy-screening/{screening_ledger.v1.json,plans,selections,attempts,publications}`, `/mnt/blueprint-work/beta-policy-screening-work`; review videos | Vast, Hugging Face (public checkpoints), NGC | site allowance (`SITE_LIMIT`), per-candidate cap, job ceiling, settlement to measured cost **[MONEY]**; withdrawal deletes screening work (`website_capture_withdrawal_screening`) **[PRIVACY]** |
| 13 | Publication | same worker (`publish_screening`, `deliver_publications`) | `beta_policy_screening_worker` + `webapp_sync` (signed Pipeline sync token) | publications queue | WebApp `POST /api/internal/pipeline/beta-policy-screening-runs` and `…-holds` (`server/routes/internal-beta-policy-screening.ts`) → `taskEvaluationPolicyRuns`, `captureTaskEvaluationRuns` | WebApp | contract parity: `SCREENING_SITE_LIMIT_USD` (`server/utils/betaPolicyScreeningContract.ts`) must equal `SITE_LIMIT` **[MONEY]** |
| 14 | Result and email | job page `/capture-upload/:token` (`JobWorkspace` → `ManagedScreeningResults` → `PolicyScreeningCandidates`); owner workspace `/app/tasks/:taskId`; videos `GET /api/screening-practice-run-videos/…` (from the intake's `GET /api/live-pipeline/beta-screening-review-videos/<id>`) | WebApp `server/utils/screeningOutcomeNotifications.ts` → `captureOutbox` | `taskEvaluationPolicyRuns` | outbox `deliveryReceipts` | Resend | the email is cancelled on consent withdrawal (F-01) **[PRIVACY]**; practice-run cache drops lapsed jobs (#1220) |

The bounded estimate or "held" is decided in step 12 (`beta_policy_screening_result`, `beta_policy_screening_holds`). A held job publishes a hold in step 13, not a run.

## Side flows

| Flow | Unit or route | Module | Store | Notes |
|---|---|---|---|---|
| Withdrawal and deletion | WebApp withdrawal and deletion workers (`server/worker.ts`), then an operator-run Pipeline deletion | `website_capture_withdrawal_deletion`, `website_capture_withdrawal_screening`, `website_capture_withdrawal_native_g1` (G1 site registration rows and `native-g1/` packets), `website_capture_withdrawal_team_scene` (the team scene halves and every record pointing at them: derived-profile rows, packet queue requests and statuses, rigid probe entries) and `website_capture_withdrawal_rigid_probe` (a settled probe job's bundles, inputs and run outputs, its money records kept; the deletion waits, unresolved, while any job's money may still settle); `owner_approved_deletion` (`blueprint-owner-approved-deletion{-plan,}.service`, started by hand) | WebApp `websiteCaptureDeletionReceipts`; `pcp/storage-gc/owner-approved-deletions` | **[PRIVACY][RIGHTS]**. The drill on a test job is still open (tracker G-02) |
| GPU spend guard | `blueprint-gpu-spend-guard.timer` | `scripts/gpu_spend_guard.py`, `gcp_monthly_budget_guard`, `account_monthly_spend_report` (the account monthly report, read at `paid_resource_admission`); `Wants=` `blueprint-provider-billing-reconciler.service` (`provider_billing_reconciler`; its own timer is off on purpose) | `pcp/gpu_spend_guard/{latest.json,billing-audit,…}`, `pcp/gcp-monthly-budget/`, `pcp/account-monthly-spend/report.json` | **[MONEY]**. Inventories Vast, RunPod, DigitalOcean and GCP; reaps orphans; fleet caps in the unit |
| Scene spend refresh | `blueprint-scene-project-spend-refresh.service` (no timer; pulled by capacity and the replay GC) | `task_evaluation_scene_spend.refresh_configured_scene_project_spend` | `pcp/scene-project-spend/` | **[MONEY]** |
| Capacity, stalls and paging | `blueprint-control-plane-capacity.timer` (10 min) | `control_plane_capacity_controller`, `control_plane_stall_alarm`, `provider_spend_alerts`, `provider_credit_admission` | `pcp/capacity/summary.json` (read by `door status`) | pages on `BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL`; runbook [`beta-operations.md` §1-3](runbooks/beta-operations.md) |
| Unit failures | `blueprint-unit-failure@.service`, started by `OnFailure=` in 13 units (the spend guard, the intake, the capacity controller and the units of steps 4 to 12, except the launch reconciler) | `unit_failure_notifier` | `pcp/unit-failures/<unit>.json` (read with `door cat`) | posts on `BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL` with the capacity controller's posting code, once and then hourly; runbook [`beta-operations.md` §1a](runbooks/beta-operations.md#1a-a-unit-failed) |
| Batch memory | `system-blueprintbatch.slice` (`Slice=` in 23 batch units) | — | — | bounds the batch units' memory together (`MemoryHigh=5G`, `MemoryMax=5.5G`); the always-on and money-safety units are outside it; runbook [`beta-operations.md` §1b](runbooks/beta-operations.md#1b-the-batch-memory-slice) |
| Preflight | `blueprint-control-plane-preflight.timer` | `task_evaluation_production_chain_preflight` | `pcp/preflight/latest.json` | the blocker count is not a health signal yet (tracker D-04) |
| Storage | `blueprint-control-plane-storage-gc.timer`, `blueprint-completed-replay-cache-gc.timer`, `blueprint-control-plane-release-lifecycle.timer` | `control_plane_storage_gc`, `completed_replay_cache_retention`, `control_plane_release_lifecycle` | `pcp/storage-gc/`, `pcp/release-retention/`; B2 offload | [`CONTROL_PLANE_STORAGE.md`](CONTROL_PLANE_STORAGE.md) |
| Deploys and host access | operator door: `blueprint-operator-door.service` (127.0.0.1:8767, Caddy `/api/live-pipeline/operator/*`), `-runner.path`, `-hold-sweep.timer` | `deploy/operator-door/operator_door`; client `scripts/operator_door.py`; deploy `scripts/deploy_control_plane_commit.py` | `/var/lib/blueprint-operator-door/requests/`, `pcp/deploy-receipts/` | D-08: one batched deploy a day from 15:00 UTC, skipped while a paid stage is live ([`AGENTS.md`](../AGENTS.md) "Beta Deploy Policy"); [`OPERATOR_DOOR.md`](OPERATOR_DOOR.md) |
| Uptime | Render cron `blueprint-uptime-probe` (`scripts/ops/uptime-probe.mjs`, every 10 min) and WebApp `.github/workflows/uptime.yml` | — | Slack | probes WebApp `/health`, `/version.json`, worker health, door `/healthz` |
| Provider output resume | door `provider-output-resume` | `provider_output_promotion` | canary dispatch root, B2 | promotes a staged provider archive before anything deletes it |
| Embodiment preparation | door `public-embodiment-preparation` → `blueprint-embodiment-preparation.{path,timer}` | `deployment_embodiment_preparation_controller` | `pcp/embodiment-preparation/`, `/var/lib/blueprint/robot-models/batches/` | one sealed, funded grant per batch **[MONEY]** |
| Agent runs (paused) | `blueprint-agent-run-dispatcher.timer` (founder stop since 2026-10-02, explicit release only) → `pcp/robot-eval-job-requests` (its consumer, the legacy `blueprint-pipeline-control-plane` loop, was retired on 2026-11-06, P-RT-06; resuming agent runs needs a new consumer) and `blueprint-team-evaluation-runner.path` | `agent_run_executor`; `team_candidate_quick10_launch` | `pcp/robot-eval-job-requests`, `pcp/team-evaluations` | do not release the hold without the founder |
| ADP agent execution | `blueprint-agent-execution.service` (continuous), `blueprint-agent-stage-replay.{path,timer}` | `agent_execution.production`, `agent_execution.stage_recovery` | `pcp/agent-execution/`, `tei/stage-replays/` | Arm Decision Proof assessment pilot; WebApp `/api/internal/pipeline` agent-execution routes |

Lanes that are installed but switched off on purpose (configured controls, launch supervisor, terminal resource release, the Cloud Run team-scene lane, native G1 campaigns, SAM-3.1 execution) are listed with their reasons in the audit's "Dormant on purpose" table.

## WebApp side

- Render (`render.yaml`, all `autoDeploy: false`; deployed only by WebApp `.github/workflows/deploy.yml` at an exact SHA):
  - web `Blueprint-WebApp` (`server/index.ts`): routes, customer outbox pump, beta ops monitor;
  - worker `blueprint-webapp-worker` (`server/worker.ts`, launch-forward-only): launch forwarder, company-policy outbox, withdrawal and deletion workers;
  - cron `blueprint-uptime-probe`.
- Signed traffic both ways:
  - WebApp → Pipeline: `signedWebsitePipelineRequest` to `/api/live-pipeline/*` on the intake;
  - Pipeline → WebApp: `/api/internal/pipeline/*`, with the URLs taken from host env files, so grep finds no caller for many of them.
- Contracts that must change in both repos together are listed in the audit (section "Mirrored contracts"). Most important: the site limit, screening holds, owner-observation versions, the scene-intent `outcome.json`, the 7-day job window and the canonical-JSON digests.

## Money ledgers

Every ceiling, with the file that sets it, is in the "Spend limits" table of [`AGENTS.md`](../AGENTS.md). A website job draws on four ledgers:

- Atlas: World Labs (step 5);
- upstream: SAM, Gemini, OpenAI and the rigid probe (steps 4 and 11);
- scene: scene configuration and ARTISAN (step 8);
- screening (step 12).

The WebApp's job ceiling (`server/utils/websiteJobCeiling.ts`) bounds all four.

One account-wide monthly limit (USD 200, `config/account_monthly_spend_limit.v1.json`) bounds every job and every provider together. It is enforced at `paid_resource_admission`. `paid_spend_meter` meters each paid admission against its job. Stage 1 adds them beside the ledgers above; the plan to remove the per-job ledgers is [`architecture/metered-spend.md`](architecture/metered-spend.md).

## Where to look next

| Question | Doc |
|---|---|
| A job stopped; Vast credit; full disk; rollback | [`runbooks/beta-operations.md`](runbooks/beta-operations.md) |
| What is happening right now | `python3 scripts/operator_door.py status`; [`launch-readiness/2026-11-beta-launch-tracker.md`](launch-readiness/2026-11-beta-launch-tracker.md) |
| Storage classes and retention | [`CONTROL_PLANE_STORAGE.md`](CONTROL_PLANE_STORAGE.md) |
| Screening catalog and candidates | [`runbooks/beta-policy-screening.md`](runbooks/beta-policy-screening.md), [`runbooks/beta-policy-catalog.md`](runbooks/beta-policy-catalog.md) |
| Running a website scene from a cloud session | WebApp `docs/runbooks/cloud-scene-runs.md` |
| What is dead, dormant or duplicated | [`audits/2026-11-simplification-audit.md`](audits/2026-11-simplification-audit.md) |
