# Full Beta Launch Readiness Audit — 2026-07-07

Scope: everything after capture/upload — autonomous ingest → validate → package → eval → hosted/buyer outputs → delivery proof, for **any** captured site. Capture itself out of scope. Audited at Pipeline `7a7db94` (= `origin/main`), WebApp `5e27bae` (+1 local preflight artifact). `BlueprintCapture` repo was **not available** in this environment; everything iOS-side is unverifiable here and marked as such.

Old audits (2026-07-03 cross-repo audit, 2026-07-04 remediation status, 2026-07-06 100-tester audit) were treated as hypotheses and re-verified against current code, current commands, and current artifacts.

> **Post-audit addendum (added before merge).** After this audit's snapshot was taken at `7a7db94`, `main` advanced by 22 commits (`df4d64d`..`5e6fa82`, 2026-07-07 evening) that respond to this ledger — including a launch readiness evidence packet with operator-evidence/CI/probed-forwarding/clean-repo requirements (B-04/B-06/B-10/B-26), DO closed-loop hardware preflight enforcement (B-13), site-object-derived default eval grounding (B-09), buyer-readout-gated WebApp readiness (B-17), signed buyer access checks (B-07), multi-root inbox coverage (B-02), full-test-lane fixes (B-19), and object-index fallback flagging (B-24). **None of those commits have been re-verified by this audit**; every finding and status below describes the `7a7db94` snapshot. Re-run the evidence commands against current `main` before treating any ledger item as closed.

---

## Verdict: **not ready** for a full "any captured site flows autonomously to buyer-ready outputs" beta.

**Reason.** The contract layer is genuinely strong — fail-closed claim ladders, rights/privacy gates (PIPE-01..04 verified fixed in code since 07-03), upstream-ID enforcement, consent-takedown wiring, spend-guarded providers — but the *operational spine is not connected*. The deployed GCS→Pub/Sub capture handoff has **no deployed consumer** (the listener is a one-shot CLI; real uploads sit as an undrained subscription backlog and expire after the 7-day retention window), the GCP ingest stack and the DigitalOcean robot-eval control plane are disjoint stacks coupled only by a shared filesystem and manual staging, WebApp→Pipeline forwarding is unconfigured with the probe never attempted, the cross-repo paid gate is still red on the same Stripe-parity test as 07-06 (and the gate script itself crashes fail-open when a sibling repo is missing), no current gate artifacts exist anywhere (`output/` is empty in a fresh clone and no capture root exists to regenerate them), buyer authenticated delivery has no implemented signed-URL path, the default manipulation task grounding is hardwired to kitchen-sink/tote assumptions, and the whole fail-closed production posture hinges on `BLUEPRINT_LAUNCH_PROOF_MODE=production`, which nothing in `deploy/` sets. A narrower, operator-attended, sim-only, single-site-class beta is close (see "Conditionally ready scope" at the end), but the launch claim as defined — any site, autonomous, buyer-safe, delivery-proven — is not supported today.

---

## Top 10 launch blockers

1. **No deployed consumer for the capture handoff subscription** — real capture uploads publish to `capture_bridge_handoff` and then sit as an undrained backlog on the pull subscription until the 7-day retention window expires (the 5-attempt dead-letter policy only applies once a consumer is pulling and failing to ack); `blueprint-pubsub-handoff-listener` is a one-shot CLI a human must run per batch. Autonomy is broken at step one.
2. **Two disjoint post-capture stacks** — GCP serverless ingest (Stack A) and the DO intake/control-plane (Stack B) are only coupled by a shared filesystem; Stack B is single-capture (`BLUEPRINT_PIPELINE_CAPTURE_ROOT` is one path). No multi-site autonomous fleet exists.
3. **Cross-repo paid gate still red + gate script fails open** — `stripe-native-parity.test.ts:237` still expects 200 and gets 403 (stale test vs. the WEB-06 CSRF hardening); and `run_paid_marketplace_launch_gate.py` crashes with an unhandled traceback (writing **no** gate artifact) when `BlueprintCapture` is absent, plus its closeout readout says "Automated repository contracts passed" unconditionally, even on `automation_failed`.
4. **Production WebApp→Pipeline forwarding unconfigured and never probed** — `forwarding_preflight.json` = `not_configured`, no URL/token, `probe.attempted=false`; worse, unconfigured now reports `blockers=0` (false calm).
5. **No current gate/proof artifacts exist at all** — `output/` is absent in a clean clone; sim-only local/release/deployment-proof, buyer readout, live-setup manifests are all machine-local and unregenerable here (the sim-only gate requires a `--capture-root` that doesn't exist in the repo). Nothing currently proves the chain end-to-end.
6. **Buyer authenticated delivery is not implemented** — WebApp has no buyer-facing signed-URL artifact delivery endpoint (`getSignedUrl` exists only on an admin route), `storage.rules` has no purchaser read grant, and the Pipeline's `signed_delivery_access` gate is attestation-based (never an executed authenticated fetch; entitlement check enforced only if the field is present).
7. **`BLUEPRINT_LAUNCH_PROOF_MODE=production` is set nowhere** — the raw-video bypass demotion, sync-required, placeholder-forbidden, synthetic-geometry-forbidden postures all key off this flag; `deploy/` never exports it (Cloud Run does set `PRIVACY_PIPELINE_ENABLED=true` and `PIPELINE_SYNC_REQUIRED=true`; the DO systemd units set neither and hardcode `/Users/nijelhunt_1/...` paths).
8. **Any-site manipulation grounding is hardwired** — `eval_ready_task_grounding.py` defaults to "turn on the sink right handle" with sink/handle/right token scoring; `manipulation_task_stack.py` defaults to a synthesized tote at a fixed pose and never reads the site's object index. The generic geometry-grounded lane (`scene_eval_autogen.py`) exists but the sink/tote lineage is what the WAM episode packet consumes.
9. **Live/provider proof is entirely open** — every provider lane is prepared-only or no-spend-readiness; no paid startup canary, task-episode, SAM3/depth/pose provider, or live simulator-in-production proof exists; the new DO GR00T×OSCAR closed-loop launcher (the one truly executable paid path) does not enforce its own 40 GB lane hardware floor pre-spend (a 20 GB droplet fallback can be provisioned).
10. **Operator/legal/payments evidence stack untouched** — live Stripe payment/payout settlement, Connect readiness, KYC/background-check decisions, finance owner, payout-exception monitoring, legal consent-posture sign-off, DPA, real-device claim recordings: all still open manual checks; additionally nothing pages an operator (monitoring alert has `notification_channels = []`, control plane exits 0 when blocked).

---

## Blocker ledger

Severity: P0 blocks truthful full beta; P1 blocks high-quality autonomy / major buyer risk; P2 hardening with explicit limitation; P3 cleanup.

---

**B-01 · P0 · Intake autonomy — handoff subscription has no deployed consumer**
- Blocker: Deployed config (`SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF=true`) routes every real capture upload to pull subscription `blueprint-pipeline-handoff-listener`; nothing drains it. `pubsub_handoff_listener.main()` pulls one batch and exits.
- Evidence: `functions/storage_trigger.py:460-478`; `deploy/terraform/main.tf:467-485,1090-1093` (7-day `message_retention_duration`; the `max_delivery_attempts=5` dead-letter policy applies only to pulled-but-unacked deliveries, not to an idle subscription); `src/blueprint_pipeline/pubsub_handoff_listener.py:658,777-828`; `deploy/systemd/` contains only intake + control-plane units — no listener unit; no Cloud Run/Scheduler runner in `deploy/`.
- Claim blocked: "a completed capture flows autonomously into the Pipeline."
- Why it matters: with zero humans in the loop, uploads accumulate as a silent, unmonitored backlog and expire after 7 days — they do not even reach the DLQ, so DLQ-based monitoring would show nothing; the beta's first step requires a manual command per batch.
- Proves current state: `grep -rn "pubsub_handoff_listener" deploy/` → only a terraform output string.
- Fix: deploy the listener as a durable consumer (systemd service on the droplet, or a Cloud Run push subscription / scheduler loop); add a drain-lag metric + alert.
- Closes when: a deployed unit/service definition exists and a live handoff message is consumed with `job ledger` proof; verification = publish a test handoff, observe automated staging without a human command.
- Requires: deploy + GCP secrets. No spend beyond infra.

**B-02 · P0 · Intake autonomy — GCP ingest and DO control plane are disjoint; control plane is single-capture**
- Blocker: Stack B's `capture-handoffs` endpoint requires `pipeline/robot_eval_dataset/task_cards.json` to already exist on the same filesystem (Stack A output); capture root is a single env value.
- Evidence: `live_pipeline_intake_service.py:192-234,315-357,835-838`; `live_pipeline_control_plane.py:89`; `deploy/systemd/pipeline-control-plane.env.example`.
- Claim blocked: "any captured site" + "autonomous" simultaneously — the always-on plane serves exactly one capture root at a time.
- Why it matters: N concurrent beta sites need per-site roots and automated wiring between package-build and task-eval staging; today that is one env var and a manual copy.
- Proves current state: `python -m blueprint_pipeline.live_pipeline_setup --no-load-env-files` → `status=blocked`, includes `webapp_upstream_truth:capture_root_not_provided`.
- Fix: make capture-root resolution per-request (the `capture_root_by_site_json` map on the WebApp side already anticipates this), and drive Stack B staging from Stack A completion events instead of shared-disk coincidence.
- Closes when: two different capture roots flow through intake→control-plane in one deployment without editing env; verification = inbox manifests for two sites in one control-plane pass.
- Requires: code + deploy. No spend.

**B-03 · P0 · Launch gates — paid marketplace gate red; gate script crashes fail-open; closeout text misleading**
- Blocker: (a) `webapp_request_sync_contracts` blocking check fails: `server/tests/stripe-native-parity.test.ts:237` expects 200 on a native DELETE with no `Authorization`, but WEB-06 CSRF hardening (`server/middleware/csrf.ts:50-56`, commit `6165f55` 07-04) now requires a Bearer token → 403. Test is stale (last touched 06-06); endpoint is correctly hardened. (b) `run_paid_marketplace_launch_gate.py` raises `FileNotFoundError` and writes **no JSON/markdown** when the Capture sibling repo is missing (`run_command` → `subprocess.run(cwd=...)`, no existence check; same in `run_external_alpha_launch_gate.py:236`). (c) `closeout_summary()` hardcodes "Automated repository contracts passed..." regardless of `overall_status` (`scripts/run_paid_marketplace_launch_gate.py:477-480`), reproducing the 07-06 P0 "misleading markdown" verbatim.
- Evidence observed: ran the gate → traceback, `GATE_EXIT=1`, no artifact; ran the vitest subset directly → `1 failed | 35 passed`; pipeline subset → `55 passed`.
- Claim blocked: "paid beta automation is green" and "the gate is a trustworthy machine-readable closeout."
- Why it matters: the launch gate is the thing operators trust; today it can (a) fail red on a stale test, (b) die without a verdict, (c) print a passing readout over a failing status.
- Proves current state: `python scripts/run_paid_marketplace_launch_gate.py` (crashes); `npx vitest run server/tests/stripe-native-parity.test.ts` (red).
- Fix: add the Bearer token to the DELETE parity test; guard `run_command` on `spec.cwd.is_dir()` → `skipped_result(..., "repo missing")`; make `closeout_summary` status-conditional.
- Closes when: gate exits 0 with all three repos present, and exits non-zero *with a written artifact + truthful readout* when a repo is missing or a check fails.
- Requires: nothing external.

**B-04 · P0 · Forwarding — production WebApp→Pipeline forwarding unconfigured; unconfigured reads as zero blockers**
- Blocker: `ROBOT_EVAL_JOB_REQUEST_FORWARD_URL/_TOKEN` unset; probe never attempted; and with nothing configured the preflight reports `status=not_configured`, `forwarding_required=false`, `blockers: []` — a production-critical unconfigured path that presents as calm.
- Evidence: ran `npm run pipeline:forwarding:preflight` → `/home/user/Blueprint-WebApp/output/pipeline/robot_eval_job_requests/forwarding_preflight.json` (`status=not_configured`, `probe.attempted=false`, `blockers=0`).
- Claim blocked: "WebApp can hand a buyer robot-eval request to Pipeline production intake."
- Why it matters: the buyer-request path (as opposed to the capture path) has zero live wiring; and a dashboard reading `blockers=0` would wrongly conclude health.
- Proves current state: command above.
- Fix: run `python -m blueprint_pipeline.live_pipeline_forwarding_secret_setup` to mint the shared token, set both envs (`ROBOT_EVAL_JOB_REQUEST_FORWARD_*` on WebApp, `BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN` on Pipeline), set `ROBOT_EVAL_JOB_REQUEST_FORWARD_REQUIRED=true` in prod, run the preflight with probe; also make the preflight treat `not_configured` as a named blocker when the deployment intends production forwarding.
- Closes when: `forwarding_preflight.json` shows `status=ready`, `probe.status=ok`, and Pipeline `intake_audit` shows the staged request.
- Requires: deploy + secrets. No spend.

**B-05 · P0 · Live pipeline setup blocked (18 blockers)**
- Blocker: simulator execution, rollout vision labeling, delivery upload, and live operator lanes are all unconfigured; ffmpeg absent for clip/keyframe lanes in this environment.
- Evidence: ran `python -m blueprint_pipeline.live_pipeline_setup --no-load-env-files --output-path output/launch_audit_live_pipeline_setup_20260707.json` → `status=blocked`, 18 blockers (07-06 had 15) incl. `missing_env_BLUEPRINT_ALLOW_SIMULATOR_EXECUTION`, `missing_delivery_command`, `missing_vision_labeling_command`, `local_deterministic_lane:missing_ffmpeg_for_clip_keyframe_paths`.
- Claim blocked: "the always-on pipeline is launch-configured."
- Why it matters: even with intake fixed, the control plane cannot execute sim, label rollouts, or upload deliveries.
- Proves current state: command above; artifact in `output/`.
- Fix: per `docs/LIVE_PIPELINE_SETUP.md`, set the `BLUEPRINT_ALLOW_*` gates + commands on the deployment host and re-run the setup audit; add ffmpeg to the deploy image.
- Closes when: `live_pipeline_setup` reports `status=ready` (or the narrower documented lane statuses) on the production host.
- Requires: deploy + secrets; simulator/labeling lanes may imply provider/LLM spend when actually exercised.

**B-06 · P0 · Evidence — no current gate/proof artifacts exist and none can be regenerated from the repo**
- Blocker: `output/` does not exist in a clean clone; `paid_marketplace_launch_gate.json/md`, `sim_only_beta_local/release/production` proofs, `buyer_package_readout.json`, `robot_eval_dataset/*`, `post_training_data_package*`, closure manifests — none are committed, and the sim-only local gate requires a `--capture-root` (real capture) that doesn't exist in the repo. Its own tests exercise only `_validate_sim_only_outputs` with hand-written JSON.
- Evidence: `ls output` → missing; `python scripts/run_sim_only_beta_local_gate.py` → `error: the following arguments are required: --capture-root`; `tests/test_sim_only_beta_local_gate.py:123-269` (synthetic fixtures only).
- Claim blocked: any status that inherits 07-03/07-04/07-06 artifact results as "current proof."
- Why it matters: all launch evidence is machine-local on one laptop; the repo cannot demonstrate its own chain; this is exactly the stale-proof failure mode the audit rules warn about.
- Proves current state: commands above.
- Fix: commit a minimal synthetic-but-honest capture-root fixture that can drive intake→control-plane→MuJoCo locally (clearly stamped as fixture, `fallback_allowed_for_beta_release=false` as done for demo bootstrap), and make CI regenerate + archive the sim-only local gate artifact; keep real-site proofs in a versioned evidence store, not a laptop `output/` tree.
- Closes when: a fresh clone can run `run_sim_only_beta_local_gate.py --capture-root <committed-fixture>` green, and a canonical launch-readiness packet links every real-site artifact by checksum.
- Requires: nothing external for the fixture lane; real-site proof needs a real capture.

**B-07 · P0 · Buyer delivery — authenticated buyer artifact delivery is not implemented end-to-end**
- Blocker: WebApp `resolveAccessUrl` returns static catalog/detail links only when `access_state === "provisioned"`; the only `getSignedUrl` is on an admin route; `storage.rules` grants read to owner/admin only (no purchaser/entitlement grant); Pipeline's `_signed_delivery_access_gate` passes on declared signed URLs + attestation without an executed fetch, and enforces `entitlement_verified` only if present.
- Evidence: `Blueprint-WebApp/server/routes/marketplace-entitlements.ts:17-99`; `server/routes/admin-growth.ts:165`; `storage.rules:47-101`; `src/blueprint_pipeline/live_robot_eval_closure.py:2646-2690`; `arena_package_delivery_local.py:122-134` (honestly `signed_delivery_access_proven=false`).
- Claim blocked: "buyers get durable authenticated access to purchased artifacts."
- Why it matters: package generation ≠ package delivery; today a paying buyer has no implemented path to download a Post-Training Data Package.
- Proves current state: grep evidence above; paid-gate manual check `buyer_artifact_access` open.
- Fix: implement a buyer-facing signed-URL endpoint gated by entitlement (mirror the hosted-session gating), make the Pipeline gate optionally execute an authenticated fetch against `PIPELINE_BUYER_ACCESS_CHECK_URL`, and make `entitlement_verified` required rather than if-present.
- Closes when: an authenticated buyer session downloads a real package via a signed URL, logged, with the closure gate consuming that proof.
- Requires: deploy; a live buyer session (human) for final proof.

**B-08 · P0 · Config posture — `BLUEPRINT_LAUNCH_PROOF_MODE=production` set nowhere; DO stack missing privacy/sync env**
- Blocker: the production fail-closed posture (raw-bypass demotion disabled, sync required, placeholder-forbidden, synthetic-geometry-forbidden, privacy forced on) is keyed off an env var that no deploy artifact sets. Cloud Run deploy does set `PRIVACY_PIPELINE_ENABLED=true` + `PIPELINE_SYNC_REQUIRED=true`; the DO systemd units set neither and hardcode `BLUEPRINT_PIPELINE_REPO=/Users/nijelhunt_1/workspace/...`.
- Evidence: `grep -rn BLUEPRINT_LAUNCH_PROOF_MODE deploy/ configs/ ops/ scripts/` → no hits; `deploy/scripts/deploy.sh:373` (Cloud Run flags); `deploy/systemd/blueprint-pipeline-*.service:9` (laptop paths); `launch_proof_policy.py:20-75`; `canonical_site_package.py:148-181` (bypass demotes blockers→warnings outside production).
- Claim blocked: "the pipeline fails closed in production."
- Why it matters: every escape hatch the code carefully fences is fenced by a flag that the deployment never sets.
- Proves current state: grep above.
- Fix: export `BLUEPRINT_LAUNCH_PROOF_MODE=production` (and privacy/sync env) in every production runtime unit; parametrize the systemd repo path; add a boot-time assertion in the intake/control-plane services that refuses to start in an ambiguous mode.
- Closes when: deployed env dumps (redacted) show the flag set on both stacks; `live_pipeline_proof_audit` records production mode.
- Requires: deploy.

**B-09 · P0 · Any-site — default manipulation/articulation grounding is kitchen/sink/tote hardwired**
- Blocker: `eval_ready_task_grounding.py` defaults `DEFAULT_TASK_TEXT="turn on the sink right handle"` with +0.45/+0.25/+0.2 scoring for handle/sink/right tokens; `manipulation_task_stack.py` defaults to `simready_tote_001` at hardcoded pose `(2.0, 4.0, 0.16)` and never reads the site's object index; `object_geometry_stage.py` carries sink-specific pose synthesis and kitchen-named MJCF scene composition.
- Evidence: `eval_ready_task_grounding.py:17-24,270-278,348-357,979-984`; `manipulation_task_stack.py:27-28,176-212,699-772`; `object_geometry_stage.py:466-617,2315-2428`; consumed by `scene_wam_policy_episode_packet.py:3371-3416`.
- Claim blocked: "the pipeline works for any captured site" for the manipulation/WAM lane.
- Why it matters: a warehouse/office/retail capture flowing through defaults gets a sink task with no sink, or a synthesized tote unrelated to the site — plausible-looking artifacts that are semantically wrong for the site (a buyer-truth failure, not just a quality one).
- Proves current state: run `build_eval_ready_task_grounding` on any capture with no sink → target `None`/weak score; code refs above.
- Fix: route default task selection through the generic `scene_eval_autogen.py` lineage (geometry-grounded core tasks + object-grounded tasks), require an explicit task contract for the sink/tote lanes, and fail closed with recapture/task-spec guidance when no groundable target exists.
- Closes when: a non-kitchen fixture produces site-grounded task cards with no sink/tote defaults, verified by a new test asserting the default path never emits the sink task for sinkless object indexes.
- Requires: nothing external.

**B-10 · P0 · Ops/payments/legal — live and operator evidence entirely open**
- Blocker: live Stripe buyer payment, capturer payout settlement, Connect live readiness, payout-exception monitor, KYC decision, background-check decision, finance owner, legal consent-posture sign-off, DPA, real-device claim recordings (iPhone/glasses/Android), buyer access session.
- Evidence: `docs/PAID_MARKETPLACE_BETA_LAUNCH_GATE.md` required evidence ids; `operator-actions/XR-05-live-evidence-runbook.md`, `CAP-10-consent-posture-signoff.md` (prepared, unsigned); no evidence artifacts in either repo.
- Claim blocked: any *paid* beta claim; consent-posture legal cover for external users.
- Why it matters: these cannot be produced by code; they gate truthful paid-launch messaging.
- Proves current state: gate doc; absence of `pipeline/operator_launch_evidence.json` with verified ids.
- Fix: execute `operator-actions/XR-05-live-evidence-runbook.md`; record each evidence id in `operator_launch_evidence.json` (schema `operator_launch_evidence.v1`).
- Closes when: `launch_gate_summary.json` flips past `automated_contracts_passed_manual_ops_required`.
- Requires: spend (live Stripe), human approval (legal, finance owner), real devices.

---

**B-11 · P1 · Delivery — WebApp sync silently no-ops when unconfigured**
- Blocker: with `PIPELINE_SYNC_WEBAPP_URL/TOKEN` unset and sync not required, sync returns `status="skipped", reason="sync_not_configured"` and the run succeeds.
- Evidence: `webapp_sync.py:780-802`; only `BLUEPRINT_LAUNCH_PROOF_MODE=production` or `PIPELINE_SYNC_REQUIRED=true` force fail-closed (Cloud Run sets the latter; DO stack does not).
- Claim blocked: "package generation implies buyer-visible delivery."
- Fix: fold into B-08 (set the flags everywhere); additionally surface `sync_skipped` as a control-plane blocker rather than success.
- Closes when: an unconfigured-sync run reports blocked in the control-plane manifest. Requires: deploy.

**B-12 · P1 · Intake truth — stale-artifact and duplicate-identity risks at intake**
- Blocker: (a) `_select_dataset_task` hands off whatever `task_cards.json`/`scenario_cards.json` are on disk with no run-id/mtime binding — a stale dataset from a prior capture at the same root is forwarded as fresh; (b) capture-handoff `job_id` is a digest of the handoff payload only, and staging refuses overwrite → a true re-capture with identical ids is rejected as duplicate (old artifacts win); (c) control-plane inbox globs all `*.json` every 5-minute tick with no processed marker.
- Evidence: `live_pipeline_intake_service.py:161-234,260-413,307-310`; `live_pipeline_input_intake.py:923-924`; `robot_eval_job_orchestrator.py:11049-11085`. (Contrast: the proof side *does* have a freshness contract — `success_claim_contracts.py:141-172`.)
- Claim blocked: "stale inbox rows/artifacts cannot masquerade as fresh proof" at the intake boundary.
- Fix: bind dataset selection to the capture's `capture_upload_complete` timestamp/run id; include a content digest (or upload timestamp) in the handoff job identity; add processed markers to the inbox.
- Closes when: a test proves a stale dataset at the same root is rejected/flagged, and a re-upload produces a distinct job. Requires: nothing external.

**B-13 · P1 · Providers — DO closed-loop launcher does not enforce the lane hardware floor pre-spend**
- Blocker: the newest truly-executable paid path (`groot_oscar_digitalocean_closed_loop_job.py`) records `lane_hardware_requirements` (40 GB VRAM floor for `kitchen_g1_groot_sonic_eval`) as metadata but never routes the selected droplet size through `build_lane_hardware_contract`/`require_pre_spend_preflight`; the DO size list falls back to `gpu-4000adax1-20gb` (20 GB) when 48 GB sizes are region-unavailable — the exact under-provisioning class the T4 post-mortem contracts (commit `1c61758`) were written to prevent. Coherence gates fail the run closed only *after* spend.
- Evidence: `groot_oscar_digitalocean_closed_loop_job.py:368,372-410`; `groot_oscar_closed_loop_image.py:216-219`; `gpu_render_providers.py:385,418,502-507,973-1004`.
- Claim blocked: "paid runs cannot be provisioned below their measured hardware floor."
- Fix: enforce the hardware contract against the actually-selected size pre-launch, as the classic lanes do.
- Closes when: a unit test proves a 20 GB fallback raises `PreSpendPreflightBlocked` for a 40 GB-floor lane. Requires: nothing external.

**B-14 · P1 · Ops — no operator notification anywhere in the loop**
- Blocker: the only alert policy has `notification_channels = []` and watches only Cloud Run Job failures; the DO control plane deliberately exits 0 when blocked (so systemd never flags it); no Slack/Pager/email wiring exists.
- Evidence: `deploy/terraform/main.tf:1246-1272`; `docs/LIVE_PIPELINE_SETUP.md:59-61`.
- Claim blocked: "fails closed with actionable blockers" — blockers are written to manifests nobody is paged about.
- Fix: wire notification channels; add an alert on handoff-subscription drain lag and on control-plane manifest `status=blocked`.
- Closes when: a forced failure produces a page/message. Requires: deploy.

**B-15 · P1 · Buyer truth — measured-state honesty floor is env-overridable to zero**
- Blocker: `BLUEPRINT_PTDP_MEASURED_STATE_FRACTION_FLOOR` overrides the 0.5 floor, clamped to [0,1]; at 0, a fully zero-filled synthesized-state export passes both the provenance gate and the readout's robot-POV check.
- Evidence: `post_training_data_package.py:66-77,2670`; `buyer_package_readout.py:312-341`.
- Claim blocked: "buyer readouts fail closed when measured data is insufficient."
- Fix: forbid lowering below default in production mode (`production_forces_*` pattern), or require an explicit reviewed waiver artifact.
- Closes when: test proves floor=0 is rejected under production mode. Requires: nothing external.

**B-16 · P1 · Buyer truth — presence-only readout sections**
- Blocker: `task_success_criteria` passes on task+eval card *presence* (never verifies measured `clearance`/`path_deviation`/outcomes exist); `failure_evidence` passes on the failure-labels artifact being included even when empty (`failure_label_count` may be 0/None).
- Evidence: `buyer_package_readout.py:365-378`.
- Claim blocked: "artifact validity, not just presence" for two of the nine buyer-critical sections.
- Fix: gate `task_success_criteria` on the batch metrics artifact with the standard-required metrics present; gate `failure_evidence` on `failure_label_count >= 1` or an explicit `zero_failures_reviewed` attestation.
- Closes when: unit tests prove empty metrics/labels block the readout. Requires: nothing external.

**B-17 · P1 · Buyer truth — two-tier readiness surface (manifest vs readout)**
- Blocker: `manifest.status == "export_ready_review_required"` can coexist with `buyer_readout.status == "blocked_incomplete_package"` because round-trip loadability and measured-state floors live only in the readout; any consumer gating on manifest status overreads readiness.
- Evidence: `post_training_data_package.py:4136-4142,4241-4249,4608,4718`; `buyer_package_readout.py:406-437`.
- Fix: fold the readout blockers back into manifest status, or document + enforce (in WebApp sync) that only `buyer_readout.status` gates buyer-facing state.
- Closes when: sync consumes the readout status; contract test added. Requires: nothing external.

**B-18 · P1 · Cross-repo — robot-eval job-request wire schema has no single source of truth**
- Blocker: `robot_eval_job_request.v1` / inbox contract strings are hand-duplicated in 3 Pipeline modules + a ~1000-line hand-rolled TS validator in WebApp; `blueprint-contracts@9c076bd` covers only the site-world/runtime surface; WebApp has no contracts dependency at all.
- Evidence: `first_gpu_sample_video_stage.py:25-26`; `first_gpu_run_packet.py:1505-1506`; `live_pipeline_control_plane.py:86-87`; `Blueprint-WebApp/server/utils/robotEvalJobRequests.ts:5-6,629-1017`.
- Claim blocked: durable cross-repo contract stability for the most active wire.
- Fix: move the job-request schema into BlueprintContracts (JSON Schema consumable by both languages) and pin both repos.
- Closes when: both sides validate against the same generated schema in CI. Requires: nothing external.

**B-19 · P1 · Tests — full (slow/gpu) lane unproven; fast lane green**
- Blocker: fast lane green (`2396 passed, 5 skipped, 1419 deselected` of 3820 collected) but the 1419 slow/gpu tests were not run in this audit and there is no CI evidence they run anywhere; 5 skips are missing-dep (torch, lerobot, ffmpeg, google-cloud-pubsub) — i.e., the LeRobot round-trip and privacy-ffmpeg tests silently skip on thin environments.
- Evidence: `python -m pytest -q` output; `-rs` skip reasons; `pyproject.toml` addopts.
- Claim blocked: "the test suite proves the launch surface" — the heavy lanes (subprocess/Isaac/render/entrypoints) are exactly the launch-critical ones.
- Fix: a scheduled full-lane CI job (CPU-safe subset at minimum) + archived results; make lerobot/ffmpeg deps explicit for the packaging lane.
- Closes when: a recorded `pytest -m ''` run (or documented CPU-safe partition) is green on a defined interpreter matrix. Requires: CI infra; GPU tests need GPU spend.

**B-20 · P1 · Cross-repo — Capture-side producer contract unverifiable; iOS chain fixes unconfirmed**
- Blocker: XR-02/03/04 are fixed on the Pipeline consumer side (commit `c47eeea`, 07-04), but the iOS producer's exact `raw/capture_upload_complete.json` shape and the CAP-01..04 fixes could not be verified (repo absent here); 07-04 remediation notes those fixes were then-uncommitted working-tree changes.
- Evidence: `functions/storage_trigger.py:28-52`; `docs/beta-launch-audit-2026-07-03/REMEDIATION-STATUS.md` ("No code was committed or pushed").
- Claim blocked: capture→pipeline handoff proven end-to-end.
- Fix: confirm the Capture repo commits landed; run one real-device upload against a staging bucket and observe the handoff message parse.
- Closes when: a real upload produces a staged capture root autonomously. Requires: real device, deploy.

**B-21 · P1 · Autonomy — core flow depends on live LLM providers with no committed no-LLM lane**
- Blocker: `run_e2e` hard-requires `--provider {claude,openai}`; capture review/agent review stages need live LLM keys; the bare CLI default also skips `evaluation_prep` (and therefore WebApp sync) — only the pubsub-listener path sets `run_evaluation_prep=True`.
- Evidence: `python -m blueprint_pipeline.run_e2e --help` (required arg); `run_e2e.py:437-456,697-704`; `pubsub_handoff_listener.py:500,552-586`.
- Claim blocked: autonomous flow without per-run manual flags; cost/keys are a standing dependency (fine, but must be provisioned + spend-guarded in the deployed env).
- Fix: document/provision LLM keys in both stacks; align bare-CLI defaults with the autonomous path or clearly mark the CLI as a dev tool.
- Closes when: deployed env provisions keys and the listener path runs e2e including sync. Requires: secrets + LLM spend.

---

**B-22 · P2 · Lint — repo-wide Ruff red and growing**
- Evidence: `python -m ruff check src/blueprint_pipeline scripts tests` → 34 errors (31 E402 from `pytest.importorskip` pattern, 2 F841, 1 F401); 07-06 had 31.
- Fix: per-file-ignores for E402 in slow tests (or move importorskip patterns); fix F841/F401. Closes when ruff exits 0. Requires: nothing.

**B-23 · P2 · Privacy defense-in-depth — bypass demotion + fallback allow-set outside production**
- Evidence: `canonical_site_package.py:148-181` (raw-bypass demotes privacy/rights blockers to warnings outside production); `evaluation_prep_stage.py:1838-1845` (`_PRIVACY_CLEARED_STATUSES` fallback includes `face_anonymized_fallback` — looser than the rights-review verdict).
- Why: only safe because of B-08's flag; both should be tightened independently of env.
- Fix: align the fallback set with `proof_contracts` verdicts; require explicit waiver artifacts for bypass demotion. Requires: nothing.

**B-24 · P2 · Any-site — narrow template libraries and silent source default**
- Evidence: `episode_spec.py:218-239` (7 site types); `object_index_stage.py:420-438` (4 environments, else default bank — degrades gracefully); `capture_bridge.py:252-263` (unknown capture source silently labeled "iphone"; conservative downstream but fabricated label).
- Fix: emit `site_type_unrecognized` note; flag inferred source. Requires: nothing.

**B-25 · P2 · Deterministic recapture guidance lacks spatial specificity**
- Evidence: `qualification.py:1860-1938,4306-4325` — deterministic follow-ups name failed checks; spatial "re-shoot X" guidance exists only via the optional LLM writer; no deterministic coverage-gap recapture computation.
- Why: any-site autonomy needs actionable recapture instead of silent degradation for sparse/partial captures.
- Fix: deterministic coverage/zone-gap analysis feeding `recapture_requirements.json`. Requires: nothing.

**B-26 · P2 · Evidence hygiene — laptop-pathed docs/units, no canonical launch packet**
- Evidence: `deploy/systemd/*.service:9` and `CLAUDE.md`/`README.md`/docs referencing `/Users/nijelhunt_1/...`; launch evidence split across markdown/JSON/laptop `output/`/memory (07-06 #97 still true).
- Fix: parametrize paths; build the single launch-readiness packet linking every artifact/command/owner. Requires: nothing.

**B-27 · P3 · Misc quality**
- `pytest` module-reexecution `RuntimeWarning`s for entrypoint tests; `src/blueprint_capture_pipeline.egg-info/` committed noise in grep surface; forwarding-preflight artifact churn tracked in WebApp git status; Python interpreter matrix undocumented (this audit: 3.11.15; 07-06 used 3.12/3.13).

---

## False-ready risks

1. **Gate crash = no verdict**: the paid gate dying without writing artifacts means "no news" can be read as "not red" (B-03).
2. **`forwarding_preflight blockers=0`** while production forwarding is entirely unconfigured (B-04).
3. **`manifest.status=export_ready_review_required`** read as buyer-ready when the readout is blocked (B-17).
4. **Presence-only readout sections** (`task_success_criteria`, `failure_evidence`) look substantive (B-16).
5. **Closeout markdown says "contracts passed"** even under `automation_failed` (B-03c).
6. **Fast-lane pytest green** (2396) mistaken for full-suite green (1419 deselected; skips hide missing lerobot/ffmpeg) (B-19).
7. **`sync_not_configured` = skipped = success** on the DO stack (B-11).
8. **Stale dataset at a reused capture root forwarded as fresh** by intake (B-12).
9. **No-spend "ready_for_paid_canary" readiness** reads like provider readiness; it is credentials/image readiness only (correctly labeled in artifacts, easily misquoted).
10. **07-03/07-04 remediation docs** describe working-tree fixes; only per-repo git history proves what actually landed (Pipeline `376a581`/`c47eeea`/`ff82bb5`, WebApp `6165f55` — landed; Capture — unverifiable).

## Any-site gaps

- Sink-handle default task + sink/handle/right token scoring (`eval_ready_task_grounding.py`) — B-09.
- Tote-at-fixed-pose default manipulation task ignoring the site's object index (`manipulation_task_stack.py`) — B-09.
- Kitchen/sink-specialized spawn-pose and MJCF scene composition (`object_geometry_stage.py`) — B-09.
- Success-claim CI truth anchored to the single kitchen fixture (`tests/fixtures/kitchen_task_min/`).
- 7-site-type task-hint library; 4-environment prompt-bank inference (degrade to generic; B-24).
- Unknown capture source silently labeled iPhone (B-24).
- No dedicated handling for multi-room/long/low-light/people/partial captures beyond generic QA gates; recapture guidance lacks spatial specificity (B-25).
- Healthy counterweights: `scene_eval_autogen.py` (geometry-grounded any-scene tasks), `capture_bridge.py` modality downgrade fail-closed, open-vocab object indexing with no per-site config, demo bootstrap strictly opt-in and stamped.

## Autonomy gaps (manual steps still needed after capture/upload)

1. Manually run `blueprint-pubsub-handoff-listener` per batch (B-01).
2. Manually stage/couple Stack A output into Stack B (shared-disk assumption; single capture root) (B-02).
3. Manually configure forwarding secrets/URLs (one-time, but currently absent) (B-04).
4. Manually set every `BLUEPRINT_ALLOW_*` execution gate + commands (B-05) — appropriate for spend gates, but simulator-on-CPU and labeling lanes need at least a configured default.
5. Manually run delivery upload (no configured delivery command; local delivery is a gated CLI) (B-05/B-07).
6. Manually watch manifests — no paging (B-14).
7. Operator approvals that are *legitimately* manual and documented: paid provider canaries, live payments/payouts, legal sign-off, KYC decisions (B-10) — these are the acceptable "documented operator approvals" class; the rest above are not.

## Buyer-truth gaps

- Signed delivery/entitlement proof is attestation-based (B-07); `entitlement_verified` only if-present.
- Measured-state floor env-overridable to 0 (B-15).
- Presence-only `task_success_criteria` / `failure_evidence` (B-16).
- Manifest-vs-readout two-tier readiness (B-17).
- `success_rate` withholding below `review_task_success` landed (`ff82bb5`) — healthy; keep it.
- Healthy: claim ladder cannot be minted downstream; generated media segregation defaults worst-case; consent takedown TOCTOU re-reads; WAM score-claim gate blocks visual-only upgrades.

## Live-proof gaps (locally proven, not live-proven)

- Provider execution: RunPod/Vast/Lambda adapters prepared-only; no paid startup canary, no task-episode, no teardown-in-anger evidence; DO closed-loop executable but hardware floor unenforced pre-spend (B-13).
- SAM3/depth/pose providers unconfigured (fail closed correctly — but the WAM real-provider validation lane therefore has zero live proof).
- Live simulator execution in the deployed control plane: gated off, never configured (B-05).
- Production deployment parity/health: no current proof artifacts (B-06); deployed-commit provenance unknown.
- Stripe/live money movement: nothing (B-10).
- MuJoCo local execution is genuinely proven by code (real library execution) — sim-only *local* claims are supportable; production sim-only claims are not yet.

## Cross-repo gaps

- Capture repo absent → producer-side contract, CAP-01..04, real-device flows unverifiable from here (B-20).
- Job-request schema quadruplicated with no shared contract source (B-18).
- Forwarding token pair (`ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN` ↔ `BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN`) is a deploy-time coupling with no parity check in either repo's CI (B-04).
- Stale WebApp test vs hardened middleware keeps the shared gate red (B-03).
- WebApp buyer delivery endpoint missing (B-07).
- Fixed and verified since 07-03 (credit where due): WEB-02 auth + ownership, PIPE-02 rights-verdict gating in `pipelineStateMachine.ts`, placeholder fail-closed on both sides, XR-02/03/04 consumer-side, PIPE-01/03/04/06 in Pipeline.

## Recommended fix order (smallest path to a truthful beta)

1. **Fix the red gate + gate robustness** (B-03): Bearer token in the parity test; cwd-guard + truthful closeout in both gate scripts. *(hours)*
2. **Deploy the handoff consumer** (B-01) and **wire Stack A→Stack B staging** (B-02 minimal: per-request capture roots). This is the autonomy spine.
3. **Set the production posture everywhere** (B-08/B-11): `BLUEPRINT_LAUNCH_PROOF_MODE=production`, privacy/sync env on the DO stack, parametrized systemd paths.
4. **Configure + probe forwarding** (B-04) with the secret-setup CLI; flip `not_configured` to a blocker.
5. **Close intake stale-proof holes** (B-12) — freshness binding + content-digest identity + processed markers.
6. **Un-hardwire default task grounding** (B-09) — route defaults through `scene_eval_autogen`, fail closed to recapture/task-spec guidance.
7. **Implement buyer signed-URL delivery + executed access check** (B-07).
8. **Notification channels + drain-lag/blocked-manifest alerts** (B-14).
9. **Readout tightening** (B-15/16/17) and **DO hardware-floor pre-spend enforcement** (B-13).
10. **Commit a synthetic capture-root fixture + CI regeneration of the sim-only local gate** (B-06), then produce one real-site end-to-end proof run and archive the packet.
11. **Run the operator/live evidence runbook** (B-10) — payments, payouts, KYC, legal, devices — before any *paid* claim.
12. Contracts consolidation (B-18), Ruff (B-22), full-lane CI (B-19), docs/paths (B-26).

Items 1–6 are code/deploy work with no spend; 7–9 small code + deploy; 10 needs one real capture; 11 is the only human/legal/spend block.

## Conditionally ready narrower scope (what could launch sooner, honestly)

An **operator-attended, sim-only, unpaid, single-site-class pilot** is close: kitchen-family captures, operator manually drains the handoff listener and stages the control plane, MuJoCo-only local simulator execution, packages delivered manually with the local delivery CLI, all buyer copy restricted to the claim ladder's actual rungs. To run even that truthfully, launch copy must **remove**: "autonomous", "any site", "live provider execution", "policy ranking" (unless two-candidate symmetric runs exist), "authenticated buyer delivery", and every payment/payout claim. Blockers B-03, B-08 (flag on the host actually used), B-12, and B-16 should still be closed first because they affect truthfulness even at pilot scale.

---

## Evidence appendix (commands run)

Environment: fresh clones at `/home/user/BlueprintCapturePipeline` (branch `claude/blueprint-pipeline-beta-audit-928ka6` = `origin/main` @ `7a7db94`) and `/home/user/Blueprint-WebApp` (@ `5e27bae`). Python 3.11.15. `BlueprintCapture` absent. No secrets/env files loaded.

| Command | Result |
|---|---|
| `git status --short --branch` (Pipeline) | clean, `claude/blueprint-pipeline-beta-audit-928ka6`; HEAD `7a7db94` = `origin/main` |
| `git diff --check` | clean |
| `git status` (WebApp) | HEAD `5e27bae`; 1 modified file after preflight run (its own output artifact) |
| `pip install -e '.[dev]'` | exit 0 (env repair needed: distro `cryptography` missing `_cffi_backend`; fixed via `pip install cffi cryptography`) |
| `python -m pytest -q` | **2396 passed, 5 skipped, 1419 deselected**, 29s. Skips: torch ×2, lerobot ×1, ffmpeg ×1, google-cloud-pubsub ×1 |
| `python -m pytest -m '' --collect-only -q` | 3820 tests collected |
| `python -m ruff check src/blueprint_pipeline scripts tests` | **exit 1 — 34 errors** (31 E402, 2 F841, 1 F401) |
| `python scripts/run_paid_marketplace_launch_gate.py` | **CRASH**: unhandled `FileNotFoundError` (`/home/user/BlueprintCapture/cloud/extract-frames`); exit 1; **no gate JSON/markdown written** |
| `python scripts/run_external_alpha_launch_gate.py` | **CRASH**: same missing-repo failure at `_ensure_extract_frames_dependencies` |
| `npx vitest run` (paid-gate WebApp suite ×7 files) | **1 failed / 35 passed** — `stripe-native-parity.test.ts:237` expects 200, got 403 |
| `pytest tests/test_webapp_sync.py tests/test_qualification_alpha.py tests/test_alpha_readiness.py tests/test_run_e2e.py` | 55 passed (pipeline gate subset green) |
| `npm run pipeline:forwarding:preflight` (WebApp) | `status=not_configured`, `blockers=0`, `probe.attempted=false` → `output/pipeline/robot_eval_job_requests/forwarding_preflight.json` |
| `python -m blueprint_pipeline.live_pipeline_setup --no-load-env-files` | **`status=blocked`, 18 blockers** → `output/launch_audit_live_pipeline_setup_20260707.json` |
| `python scripts/run_sim_only_beta_local_gate.py` | exit 2 — requires `--capture-root`; no capture root exists in repo |
| `ls output/` (pre-audit) | directory absent — zero gate artifacts in a clean clone |
| `grep -rn pubsub_handoff_listener deploy/` | only a terraform output string; no deployed runner |
| `grep -rn BLUEPRINT_LAUNCH_PROOF_MODE deploy/ configs/ ops/ scripts/` | no hits |
| `python -m blueprint_pipeline.run_e2e --help` | `--provider {claude,openai}` is required |

Code-inspection evidence (file:line) is embedded per blocker above; key verification sweeps covered `src/blueprint_pipeline/` (intake, control plane, orchestrators, providers, WAM, packaging, readout, claims, privacy, consent), `scripts/`, `deploy/`, `tests/`, and WebApp `server/` (routes, middleware, pipelineStateMachine, storage.rules, functions).

## Not-run appendix (blocked on spend/deploy/secrets/human approval)

| Command / action | Why not run | Would prove |
|---|---|---|
| `python -m blueprint_pipeline.run_e2e --capture-root <real> --provider openai` | requires live LLM API key (spend) + a real capture root | full post-capture lane incl. agent review + sync |
| `python scripts/run_sim_only_beta_local_gate.py --capture-root <real>` | no capture root available; producing one requires a real capture or authorized fixture creation | intake→control-plane→MuJoCo local closure |
| `scripts/pytest_full.sh` (`pytest -m ''`) | slow/gpu lanes need GPU + hours; several would spend or hit providers | full-suite green claim |
| Paid provider startup canaries (`--allow-paid` DO/RunPod/Vast/Lambda paths) | provider spend; explicitly gated | live provider startup/teardown/artifact proof |
| WAM real-provider probe with SAM3/depth/pose configured | model weights + GPU + spend | real perception-provider validation |
| Forwarding probe against production intake | needs `ROBOT_EVAL_JOB_REQUEST_FORWARD_URL/TOKEN` secrets + deployed endpoint | production forwarding reachability |
| `firebase deploy` / terraform apply / systemd installs | production mutation | deployment parity/health proofs |
| Live Stripe payment/payout, Connect readiness, KYC/background decisions, legal sign-off, real-device recordings | money movement + human/legal approval | the paid-gate operator evidence ids |
| Android gradle / iOS xcodebuild gate lanes | toolchains absent (and Capture repo absent) | device-side contract suites |
