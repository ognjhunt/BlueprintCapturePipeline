# Simplification audit, November 2026

Read-only audit of both repositories and the production host, done 2026-11-05.

- Map of what connects to what: [`../SYSTEM_MAP.md`](../SYSTEM_MAP.md).
- Every item, with its evidence and removal plan: [`2026-11-inventory.json`](2026-11-inventory.json) (1,161 rows plus a `plans` section).
- Nothing was deleted, deployed or changed on the host. Host evidence comes from `scripts/operator_door.py` read commands, 2026-11-05 20:50–21:20 UTC. A screening of capture-447 was running throughout.

Confidence levels:
- **proven**: static, runtime, data and intent evidence were all checked.
- **likely**: one gap, which the inventory row names.
- **owner-call**: a product or money decision.

"Grep found nothing" never counts as proof. Units, `--worker` targets, door scripts, GPU bundle file lists and docs read as data were all treated as callers.

## Executive summary (for the owner)

Blueprint's running system is much smaller than its codebase.
- **Website job chain:** about 25 systemd units on the main path, plus the guards around them, and one GPU worker carry a website job from upload to result (map: [`SYSTEM_MAP.md`](../SYSTEM_MAP.md)).
- **Code size and reach:** the Pipeline has 1.35 million lines of Python. 94% of it is reachable from something production starts. Only 6% (86,000 lines in 228 modules) is reachable from nothing.
- **Where the extra weight is:** not mainly in dead code. It is in duplication (132 copies of "hash this JSON", 350 hand-written file writes), in lanes that are off but still installed, in 2,700 environment switches, and in docs and catalog rows nobody can use.

Before anything is removed, these checks protect the system:
- The guards (privacy, rights, money, deploy gates and the door) stay as they are.
- Arm Decision Proof v1 code and evidence stay.
- Records already stored stay readable.

Top 10 simplifications, safest and highest payoff first:

| # | Change | Effect | Risk | Plan |
|---|---|---|---|---|
| 1 | One system map (this PR). Then point the docs index at it (P-DT-05; the index is an owner decision) | A new person or agent can find the unit, module and store for each step in minutes | none (docs only) | P1-01, P-DT-05 |
| 2 | Delete about 16,600 lines of unreachable research and one-off lanes, with their tests (splat and semantic research, one-off task-evaluation lanes, compatibility shims) | about −30k lines including tests; a smaller nightly test lane | low: no unit, image, bundle or record uses them; each PR has a narrow test plan | P-A06, P-A08, P-A10 |
| 3 | Archive the policy-ranking research (14,000 lines) after recording the commit its evidence manifest verifies at | −21k lines; the research stays reproducible | low | P-A02 |
| 4 | Retire the capture reconstruction dispatcher: 3 units and the dispatcher module. The owner already decided this (D-06, H-03) | −3 units; one fewer legacy queue to explain | low: its queue has been empty since 2026-08-24 | P-RT-01 |
| 5 | Stop the idle Cloud Run compile poll from costing a quarter of a CPU, or rebuild its image so it can work | frees about 0.25 of a core on the 4-core host (load 4.46), relevant to D-11. The team-scene lane shipped today (#3331) **probably cannot engage** until the image is rebuilt (pillow 12.3.0 on the image vs 13.0.0 on the host) | low; the rebuild is a small Cloud Run spend (owner) | P-RT-02, R1 |
| 6 | One meaning for the gripper width key, and one function that computes pad centres | prevents a repeat of today's inverted-gripper incident. #3404/#3405 fixed the workers, but `finger_separation_m` still means two different distances | low; keeps a read alias for stored receipts | P-B01 |
| 7 | WebApp: delete about 50 unreachable client files and 8 unreferenced server files; drop 2 unused packages and move 4 build-only ones | about −7.5k lines, −6 runtime dependencies | low; UI only, no route the Pipeline calls | P-W01, P-W02, P-W03a, P-W03b |
| 8 | Install the host's Python packages from `uv.lock` (today only 8 pins are applied); pin Node on Render | removes a whole class of skew: msgpack below its floor, scipy unpinned, pillow apart from the worker image, Node unpinned while CI tests the end-of-life Node 20 | medium; needs a deploy window | R1, R4 |
| 9 | Split the screening catalog: website jobs see only candidates that can run | removes about 82 rows that can never run on a website job (58 partner placeholders, 24 scripted baselines). They make up most of the 2,588 "Did not run" rows across 20 campaigns and most of the 1 MB ledger | owner product decision; Pipeline and WebApp change together | P-DT-12 |
| 10 | Archive about 31 MB of Pipeline docs (one 20 MB HTML file) and about 38 MB of WebApp design binaries | faster clones; less noise for agents | low; none of them is read at runtime | P-DT-01, P-DT-03, P-DT-13 |

Also worth doing soon, outside the top 10:
- **Legacy control-plane loop (P-RT-06):** it burns 27 CPU-seconds every 5 minutes and sends a "control plane blocked" webhook that means nothing.
- **Replay-cache GC (P-RT-05):** it writes two receipt files every 5 minutes; there are 22,756 already.

Problems found on the way (not simplifications; each needs a fix or an owner look):
- RunPod billing reconciliation fails on every call: 85 of 85 on 11-05, 138 of 138 on 11-04. This is the unexplained source in tracker B-07. The guard stays; its call needs fixing.
- Nothing reconciles B2 storage, Gemini or hosted Meta SAM spend. The spend guard does not inventory Lambda Labs, which has an adapter; whether a Lambda key exists is unknown.
- The WebApp's 24-lane ops scheduler runs in no deployed process. That includes the `agent_run_settlement` sweep, which releases holds left by robot teams that stopped calling. Is that intended?
- The intake answered 503 to 246 of 267 `website-preparation-status` calls between 12:55 and 20:58 UTC. Not investigated.
- Embodiment preparation has re-checked one 10-30 grant (`public-robot-imports-20261030-01`) every 5 minutes for 6 days. It spends nothing, but the grant should be settled or left to expire.
- The nightly full test lane failed 4 of its last 5 runs (tracker E-06).
- Four workflows have never run, two of them the signed-release path, and every deploy is recorded as "iteration".

## Findings by area

Each table shows the main items only. The inventory has every row with its evidence (file:line, unit and journal time, host path and mtime) and its plan.

### Host runtime (units and lanes)

There are 99 unit files: 49 services, 30 timers and 20 paths. The deploy installs 74 of them (`DEFAULT_DEPLOYED_SYSTEMD_UNITS`, `scripts/deploy_control_plane_commit.py:428`). On the host, `door units` lists 317 units. 206 of those are failed experiment one-offs from 2026-09-26 to 10-27 (tracker O-24: `reset-failed`, an operator action in a deploy window). 11 are door hold-expiry timers.

| Item | Class | Confidence | Evidence (UTC) | Plan |
|---|---|---|---|---|
| `blueprint-capture-reconstruction-dispatcher.{service,timer,path}` | REMOVE-CANDIDATE | likely (units proven idle) | Timer and path inactive; no journal in 14 days. `capture-reconstruction-queue/{pending,completed,blocked}`, `-runs` and `-derived` are empty; the lock is dated 08-24. D-06 says retire. Gap: the host keeps one dev-fixture policy in `/etc/blueprint/capture-reconstruction-policies`, so the listener may still call the module and abstain | P-RT-01 |
| `blueprint-pipeline-control-plane.{service,timer}` (`live_pipeline_control_plane`) | OUTDATED | owner-call | Every 5 min: 27 CPU-s, a 6.7 MB manifest rewrite, `status=blocked`. Its 13 blockers are feature switches. Inbox last fed 09-29 (only by the paused agent-run dispatcher). Webhook last sent 20:38Z | P-RT-06 |
| `blueprint-task-evaluation-episode-compilation-remote.*` | CONSOLIDATE | likely | 20.6–22.6 CPU-s every 60–90 s with zero rows. Last remote job 10-01 | P-RT-02 |
| `blueprint-completed-replay-cache-gc.*` | CONSOLIDATE | likely | Removes 0 bytes every 5 min but writes 2 receipts (22,756+). Its `Wants=` also starts the 35 CPU-s spend refresh | P-RT-05 |
| `blueprint-scene-object-discovery.{service,path}` | REMOVE-CANDIDATE | owner-call | Every queue and output dir is empty since the 08-26 deploy created them. Only producer: a panel on the WebApp's `/ops/task-evaluation-launches`, which Arm Decision Proof cites (keep the page) | P-RT-03 |
| `blueprint-production-gpu-{campaign-control-plane,worker-agent,worker-pool}`, `blueprint-pipeline-intake-staging` | REMOVE-CANDIDATE | owner-call | Not installed on the host; not in the deploy list. A weekly workflow and a runbook still name them | P-RT-07+P-A09 |
| `blueprint-scene-project-spend-refresh.service` | KEEP-GUARD (only its entry point needs renaming) | proven | Its `python -c` entry makes preflight report `entry_module_unrecognised` on every run | P-RT-09 |
| Off on purpose: agent-run dispatcher (founder stop since 10-02), team evaluation runner, launch supervisor (OpenAI money), configured-controls progression (C-01), terminal resource release, Cloud Run team-scene lane (#3331), SAM-3.1 execution (idle since 09-16), native G1 campaigns | DORMANT-INTENDED / KEEP-ADP | proven | `beta-operations.md` §7 (D-06); tracker | none |
| Existing-policy canary continuation and watchdog | DORMANT-INTENDED | owner-call | Installed, never started in 14 days. The watchdog is a teardown guard. Keep while a manual hand-off is possible | P-RT-08 |
| Stuck: `blueprint-embodiment-preparation` | KEEP-LIVE | proven | Re-checks the 10-30 grant every 5 min; blockers `independent_vast_watchdog_exact_resource_name_invalid`, `spend_admission_lock_stale` | operator |

Armed but with no real work for 30+ days: launch preparation (09-27), SAM-3.1 (09-16), host episode compilation (09-27), remote episode compilation (10-01) and native G1 dispatcher and settlement (09-26, ADP). They are cheap and dormant on purpose, apart from the remote poll (P-RT-02).

### Providers, images, Terraform, CI

**Providers:**
- **Live:** Vast; DigitalOcean (the host itself); GCP (Cloud Run remote CPU, Pub/Sub, Storage, Artifact Registry, budget guard); B2; Hugging Face; OpenAI; Gemini; World Labs; Meta hosted SAM 3.1; NGC; Firebase; Render; Resend; Slack; Paperclip; Notion.
- **In the spend guard only:** RunPod and GCP compute. They are required inputs to the provider-zero check (KEEP-GUARD).
- **Fallback:** Redis (waived on Render).
- **Dormant:** Stripe, Gmail, Parallel/Exa, Varjo Teleport.
- **No caller:** Lambda Labs, SkyPilot.
- **Retired:** AWS.

In the current screening ledger (20 campaigns since 11-04), only `robolab.v1` has completed candidates: pi05 ×3 and groot_n17 ×1. The G1, task-family and hosted-endpoint backends show no run there. Older ledgers were not read.

| Item | Class | Confidence | Evidence | Plan |
|---|---|---|---|---|
| `deploy/docker/robot_eval_worker/unitree_groot_sonic_vast/Dockerfile` | REMOVE-CANDIDATE | likely | Referenced only by `.github/dependabot.yml:50` and CHANGELOG history | P-INFRA-01 |
| Cloud Run remote-CPU worker image `gcr.io/blueprint-8c1ca/blueprint-pipeline@sha256:822b04aa…` | OUTDATED | likely (inference) | Probed 10-01 with pillow 12.3.0 and msgpack 1.2.1. The host has had pillow 13.0.0 since the 11-05 17:25Z deploy. `remote_cpu_environment.py:43-48` digests pillow, so the planner refuses with `environment_mismatch`. The team-scene stage was never probed | R1 |
| Terraform resources with no consumer under the configured routing: the `swap_dispatch_worker` function, `google_cloud_run_v2_job.pipeline`, 2 Cloud Tasks queues, Secret Manager data sources, `video_to_world`, `large_video_ingest` | OUTDATED | owner-call | `deploy.sh:786` sets `SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF=true`, so `functions/storage_trigger.py:446-498` publishes only to the hand-off topic. Deployed state was not readable | P3 |
| Privacy Cloud Run services (SAM3, VIP, DeepPrivacy2) | KEEP-GUARD | owner-call | They serve the non-website capture lane (`privacy_processing`). Retire only with that lane | P3 |
| Images with no live consumer: reconstruction worker, robot-eval isaac/mujoco (they pin a different Isaac Sim 6.0.1 digest), GR00T OSCAR set (its workflow has never run), `policy_ranking_openpi`, `video_to_world`, two Packer templates for one G4 host | OUTDATED | owner-call | `infra` rows in the inventory | P3 |
| Workflows that never run: `release-evidence-retention`, `release-signature-verification`, `critical-capability-lanes`, `groot-oscar-thin-release` | OUTDATED | owner-call | GitHub run history | P3 |
| Two fileless workflow registrations (`groot-oscar-registry-recovery-9309f28a.yml`, `policy-dependency-replay.yml`) | ARCHIVE | likely | Registered in GitHub; no file on main; can still run from a branch | P-INFRA-02 |
| `production-gpu-reliability.yml` | OUTDATED | owner-call | Runs on matching PR paths and weekly. It is **not** a required check (only the `ci.yml` "Impacted test and sentinel gate" is, `docs/CI_REQUIRED_CHECKS.md:9`) | P-RT-07+P-A09 |

### Python code and dependencies

Reachability: 2,592 modules, 1,348,789 lines.
- **Production-reachable (2,364 modules, 1,262,820 lines):** reached from units (including `--worker`), door scripts, deploy tools, Dockerfiles, the Cloud Function, workflows, `scripts/` and the GPU bundle's file-name lists.
- **Unreachable (228 modules, 85,969 lines):**
  - 68 are operator CLIs named in docs;
  - 41 are named only in docs or tools;
  - 119 are reached by nothing.

| Cluster | Modules / lines | Class | Confidence | Plan |
|---|---|---|---|---|
| Splat, semantic and reconstruction research (`semantic_*_stage`, `splat_backends`, `gaussian_*`, `native_360_*`, `isaac_nurec_export`, …) | 24 / 7,970 removable | REMOVE-CANDIDATE | likely | P-A06 |
| One-off task-evaluation lanes (external scene, provider NuRec, ARKitScenes, new-site, diagnostic scene configuration, MoveIt adapter, …) | 22 / 8,258 | REMOVE-CANDIDATE | likely | P-A08 |
| Compat shims and retired entrypoints (`aws_independent_watchdog_control` is a refusing stub, not a guard; `output_run_transaction`, `pipeline_settings`, `legacy_*`) | 7 / 361 | REMOVE-CANDIDATE | likely; needs the full suite | P-A10 |
| Policy-ranking research | 35 / 13,944 | ARCHIVE | likely | P-A02 |
| Articulated door task | 17 / 5,103 | REMOVE-CANDIDATE | owner-call (ADP-adjacent) | P-A05 |
| Measurement lane (only user of `newton`, `pyelastica`, `sapien`, `coal`) | 31 / 12,389 | DORMANT-INTENDED | owner-call | P-A01 |
| `production_gpu_*` / `production_autonomy_*` | 6 / 1,460 | OUTDATED | owner-call | P-RT-07+P-A09 |
| Public-scene ADP-009A lane, ADP modules | 51 / 21,797 | KEEP-ADP | proven | none |
| Retention and reclamation tools | 11 / 4,245 | KEEP-RECORDS | owner-call | P-A07 |
| Operator diagnostics (`vast_safe_status`, `atlas_*`, …) | 19 / 6,421 | RENAME/DOCUMENT | likely | P-A11 |

Duplicate implementations. Each needs one owner. Guards are consolidated only with proof of identical decisions.

| Concept | Copies | Proposed owner | Note | Plan |
|---|---|---|---|---|
| Gripper width (Robotiq 2F-85) | `finger_separation_m` means pad-centre distance in the construction probe (`native_task_arena_construction_worker.py` ~780-815) and finger-body-origin distance in `adp009d_newton_gripper_drive.py:274`. Pad centres are computed twice (`native_franka_pose_servo.py:2220`, `native_task_arena_construction_worker.py:360`) | `native_franka_pose_servo.current_gripper_pad_readback` | #3404/#3405 (merged 11-05) already fixed the policy workers; this remainder is still open | P-B01 |
| Canonical digest | 132 local helpers, about 20 byte variants | `decision_evidence_contracts.canonical_digest` | Only 15 byte-identical copies may delegate. The rest seal stored records (KEEP-RECORDS). WebApp has 2 orderings on purpose plus 9 private helpers | P-B03 |
| Atomic write | 350 call sites in 225 files; 64 files skip fsync | `core/common.py` writer | Durability differs today | P-B04 |
| File locks | 133 files | `core/file_lock.py` (mechanism only) | Paid-launch, withdrawal and nonce lock paths must not change (KEEP-GUARD) | P-B05 |
| Retry and backoff | 12 hand-rolled loops; the shared module has 2 users | `transport_retry_policy` | Reads only. A paid mutation must never gain a retry | P-B06 |
| Quaternions | 93 helpers; 2 angle helpers don't normalise (`native_task_rigid_gate_evaluation.py:30`, `native_task_construction_plan.py:186`) | `rigid_frame_transforms` | xyzw everywhere except the MuJoCo and USD edges | P-B02 |
| OpenGL↔OpenCV camera flip | 8 independent definitions | `camera_geometry_validation` | | P-B07 |
| Spend-cap arithmetic (rate × TTL ≤ cap) | 8 lanes, rounding differs | `paid_resource_admission` | KEEP-GUARD; consolidate only with per-lane tests | P-B08 |

Line caps (`docs/source_governance_policy.json`): 102 modules are at 90% or more of their cap, and 45 are at exactly 100%. The ones that matter, with proposed split points (P-C):
- `native_task_arena_policy_canary_worker.py` at 2295/2297. A split must also be added to the GPU bundle's `TEAM_RUNTIME_MODULE_NAMES`, or the rented machine fails to import it.
- `adp009d_isaac_episode_adapter.py` at 1986/2000.
- `vast_provider_adapter.py` at 10420/10420. The credit admission stays where it is.
- `paid_resource_allocator.py` at 7456/7456. Split per subcommand behind the single chokepoint.

Dependencies:
- Imported in production but not declared in core: `pydantic`, `starlette`, `httpx`, `requests`.
- Three HTTP clients are in use (stdlib in 116 files, `requests` in 3, `httpx` in 2).
- websockets is handled three ways on GPU clients. See the version table.
- `langgraph*` has no import in the repo. It is presumably needed by the external CAD package (owner-call).

### WebApp

| Item | Class | Confidence | Evidence | Plan |
|---|---|---|---|---|
| 5 pages with no route (`CaptureAppPlaceholder`, `CaptureLaunchAccess`, `pages/Settings.jsx`, `pages/app/Overview`, `OperatorOverview`) plus about 47 unreachable components and libs | REMOVE-CANDIDATE | likely | knip `--production`, each re-checked by grep; `lib/contactRequestPrefill.ts` is live (agent CLI) | P-W01 |
| 8 unreferenced server files (`server/utils/{ai-prompts,business-logic,…}.ts`) | REMOVE-CANDIDATE | likely | no importer | P-W02 |
| `@heroicons/react`, `@google/generative-ai` | REMOVE-CANDIDATE | proven | zero imports (Gemini is called over REST) | P-W03a |
| `bcryptjs`, 3 Radix packages, remotion ×3, the Replit Vite plugin; undeclared runtime imports (`body-parser`, `@google-cloud/storage`, `gaxios`, `google-auth-library`) | REMOVE / CONSOLIDATE / RENAME | likely | package.json versus imports | P-W03b |
| 9 legacy endpoints (`/api/generate-image`, `/api/submit-to-sheets`, `/api/process-waitlist`, `/api/googlePlaces`, `/api/upload-to-b2`, `/api/storage/uploads`, `/api/help/reschedule`, …) | REMOVE-CANDIDATE | likely | No caller in any repo. Gap: the Render request logs were not readable | P-W04 |
| About 70 endpoints of dormant lanes, plus their admin pages (growth/city-launch 36, creator 9, admin agent 9, robot-team admin 6, Stripe payouts 4, voice 4, checkout 1) | DORMANT-INTENDED | owner-call | no caller in code; routed and auth-protected | P3 |
| Ops automation scheduler (24 lanes, including the `agent_run_settlement` money sweep) | DORMANT-INTENDED; the sweep is KEEP-GUARD | owner-call | Worker runs launch-forward-only (`render.yaml:55-56`); web needs `BLUEPRINT_RUN_OPS_AUTOMATION_IN_WEB=1` | P3 |
| "Book the call" (Calendly) on the brief page for `needs_conversation` | OUTDATED | owner-call | `client/src/components/site/TaskBriefReview.tsx:367-370` conflicts with F-09 (introduction by reply) | P3 |
| 8 of 14 Playwright configs and 6 e2e specs that no CI job runs | CONSOLIDATE | likely | `ci.yml:270-347` | P-W05, P-DT-10 |
| 30 declared Firestore indexes; 16 never deployed (F-02) | OUTDATED | owner-call | The growth, payout and shard indexes serve dormant routes. The live index list was not readable | P-W07 |
| Write-only collections that are evidence (`websiteCaptureDeletionReceipts`, Stripe dead letters and events, `unallocatedCohortCosts`) | KEEP-RECORDS | proven | | none |
| Rules-only collections (the iOS app's surface, kept in parity with the iOS repo by CI) | KEEP-GUARD | likely | `ci.yml:80` | none |

### Docs, tests, scripts and the catalog

**Docs read as data:** 373 files.
- 227 of 272 schemas are read at runtime, and the whole `docs/schemas` folder ships to remote CPU workers.
- 50 other docs are read at runtime and 12 by CI validators.

**Archive candidates:**
- Pipeline: about 31 MB.
  - `docs/verification`: 20 MB, one HTML file nothing reads.
  - Experiment trees nothing reads: 3.7 MB.
  - Dated audits and superpowers plans: 1.2 MB.
  - Experiments read only by research tests: 7.9 MB, an owner-call.
- WebApp: about 38 MB of `docs/design` and `docs/visuals` binaries (only 3 files there are read), plus 3.6 MB of dated history.

**Docs that are wrong:**
- `AGENTS.md` (lines 24 and 51) still states the per-site screening limit as $15. The code and the WebApp have been at $30 since #3395 (11-05). This is a money figure in a binding guide, so the owner confirms it (P-DT-14).
- `docs/README.md` names ten ADP files as the only active documents and no beta document.
- `beta-ops-incident-response.md` sends rollback to the legacy Cloud Run job; the real path is `door deploy`, `beta-operations.md` §5.
- The July SC3 gap ledger still says it is authoritative, with 107 gaps.
- `PRODUCTION_PIPELINE_PROFILE.md` and `LIVE_PIPELINE_SETUP.md` describe the July product.

Every unit, module and script name in the runbooks resolves.

**Tests:**
- 2,746 files, about 1.06M lines, about 34.5k cases.
- 21 test files (3,184 lines) test only removal-candidate modules. They go in the same PR as their module.
  - A first pass listed 36 files. The cross-check found that 15 of those test modules that stay: 4 KEEP-ADP, 2 KEEP-RECORDS, 1 dormant, 3 operator tools, 4 archive and 1 owner-call.
- About 40 `external_runtime`/`external_data` cases, including every Isaac case and both `gpu`-marked tests, run in no CI lane (P-DT-09, owner-call: a GPU lane costs money).

**Scripts** (300 in `scripts/`):

| Status | Count |
|---|---:|
| Wired to a unit, CI, the Makefile or `src` | 142 |
| Called only by other scripts | 53 |
| Called only by their own test | 48 |
| No caller at all | 26 |
| Named only in docs | 20 |

Most of the uncalled ones are break-glass or ADP tools; document them, don't remove them. Six (MuSHRoom/Postshot bakeoff and similar) are likely removals (P-DT-11).

**Catalog** (owner decision P-DT-12): 136 candidates, 39 enabled on the host.

| Family | Rows | Can run today | Ran since 11-04 |
|---|---:|---|---|
| OpenPI DROID | 17 | 4 enabled | pi05 ×11 attempts, pi0-100k-polaris ×1 |
| GR00T N1.7 DROID | 1 | yes | ×5 |
| Cosmos3 Edge/Nano | 2 | yes | Edge ×15 |
| MolmoAct2 DROID | 1 | yes | ×11 |
| OpenWAM Franka/Piper | 2 | yes, with a fitting job | no |
| MolmoAct2 YAM | 1 | yes, with a fitting job | no |
| HumanoidArena G1 | 4 | needs a humanoid job | no |
| Scripted baselines | 24 | barred on website jobs (#3358) | 13 on 11-04 |
| Team route templates | 58 | needs a partner artifact | never |
| Vendor private controllers | 14 | needs vendor access and a licence | never |
| Dream Labs hosted | 7 | needs an account and terms | never |
| Physical Intelligence API | 2 | needs API access | never |
| Licence-blocked DROID | 2 | needs licence review | never |
| TRLC π0.5 | 1 | needs tokenizer access | never |

## Looks dead but is live (do not delete)

- `adp009d_isaac_episode_adapter.py` is the live screening adapter (#3404 changed it today). The same goes for `adp009d_policy_episode` (imported lazily by the canary worker), `adp009d_policy_server_worker` and `adp009d_policy_provisioning`, all of which ship in the GPU bundle.
- `task_evaluation_scene_retirement_supervisor` is the entry wrapper for 14 units, despite the name. `/usr/lib/blueprint/scene-retirement-runtime/continuous_bootstrap.py` starts the intake, agent execution and storage GC.
- The GPU bundle ships modules named only as strings (`CANARY_SHIPPED_PACKAGE_FILES`, `TEAM_RUNTIME_MODULE_NAMES`, `native_task_arena_policy_canary_bundle.py:61-83`). Remote CPU jobs get the whole `src` tree and `docs/schemas` through `git archive`.
- Docs read at runtime:
  - `docs/experiments/policy_ranking_thesis_20260726/*.json`: the OpenPI inventories every screening plan reads.
  - `docs/arm_decision_proof_v1/manifests/scene839873_policy_canary_*`, `adp009d_scene_840920_policy_readiness.v1.json`, `policy_canary_episode_interpreter_profile.v1.json` and `image_editor_backends.v1.json`: beta screening and website image completion use them, despite the ADP names.
  - `docs/website-controls-pause-2026-09-21.md`: the authorization reference written into records.
  - `docs/runbooks/beta-operations.md` and `control-plane-capacity.md`: their paths and anchors are emitted in pages.
- CI spend-guard verifier inputs: `docs/FIRST_GPU_E2E_RUNBOOK.md` (despite its "Archived" banner), `UNITREE_G1_POLICY_ENDPOINT_LANE.md`, `runbooks/groot-oscar-thin-release.md`, `architecture/command-safety-matrix.md` and `architecture/paid-resource-mutation-surfaces.json`.
- Configured scenes in B2 live under `…/arm-decision-proof-v1/configured-scenes/website-…`, but they are beta data. The remote-CPU B2 prefix sits under the same name. Do not tie either to ADP retirement.
- The `storage-trigger` Cloud Function and the Pub/Sub "capture bridge" hand-off look legacy, but they carry every website upload (last hand-off 11-04 16:56Z).
- DigitalOcean looks like a dead builder provider, but it bills the host. The RunPod and GCP-compute inventories in the spend guard are required provider-zero inputs.
- Some units look off but run:
  - The billing reconciler timer is off, but the reconciler runs with every spend-guard start.
  - The scene spend refresh has no timer and runs every 3–5 min.
  - Agent execution writes nothing to its journal but refreshes `worker_health.json` every minute.
- Top-level `pcp/<lane>` directory mtimes are touched by every deploy and preflight, so they say nothing about activity. Read the subfolders.
- WebApp items that look unused but are live:
  - `/api/internal/pipeline/*` endpoints with no literal caller: the Pipeline reads their URLs from host env files. They include the rights/privacy revocation and settlement endpoints.
  - Eight env names built at runtime (`server/agents/provider-config.ts:271`).
  - `ops/paperclip/plugins/blueprint-automation/src/*`, which is compiled into the server bundle.
  - `knowledge/` and `vendor/daily-research`.
  - The iOS-only endpoints (`/api/client/runtime-config`, `/api/qr/*`).
  - The ADP-cited pages `/ops/task-evaluation-launches` and `/admin/jobs`.
- Write-only Firestore collections that are evidence (deletion receipts and similar).

## Version skew

Columns are the declared floor (pyproject), `uv.lock`, the host venv, the remote-CPU worker image, the GPU runtime and CI.

The host venv is never synced to `uv.lock`. Each deploy installs only 6 pins (`deploy/systemd/production-control-plane-requirements.txt`) and 2 security pins. Host values come from the 11-05 17:25Z deploy receipt and the 10-01 parity probe.

| Dependency | pyproject | uv.lock | Host venv | Worker image (Cloud Run) | GPU runtime | CI |
|---|---|---|---|---|---|---|
| pillow | ≥13.0.0 | 13.0.0 | 13.0.0 (security pin) | **12.3.0** (below the floor) | — | 13.0.0 |
| msgpack | ≥1.2.3 | 1.2.3 | **1.2.1** | **1.2.1** | 1.1.2 (cosmos thin client) | 1.2.3 |
| scipy | ≥1.10.0 | 1.18.0 (py3.12) | **unpinned, unmeasured** | not measured | — | 1.18.0 |
| websockets | ≥12.0 | 16.0 | unpinned | — | **12.x** (Isaac Sim 6.0.1; #3399 shim); OpenWAM needs 16; cosmos vendors 16.0 | 16.0 |
| numpy | ≥1.24 | 2.3.1 | 2.3.1 | 2.3.1 | Isaac Sim bundled (unverified) | 2.3.1 |
| torch | ≥2.2.0 | 2.10.0 | — | not installed | Isaac Sim bundled; reconstruction image 2.4.1+cu124 | 2.10.0 CPU |
| opencv-python-headless | ≥4.10 | 4.11.0.86 | unpinned | from lock | 4.10.0.84 (RoboLab bundle) | 4.11.0.86 |
| Isaac Sim | — | — | — | — | 6.0.1@b1c542b2 (live); `robot_eval_worker/isaac` 6.0.1@783444c7; groot_oscar 6.0.0 | never pulled |
| jsonschema | ≥4.23 | 4.26.0 | 4.26.0 | 4.26.0 | — | 4.26.0 |
| openai (Python) | ≥1.0 | 2.45.0 | unpinned | from lock | — | 2.45.0; `tools/daily_research` 3.22.1 |
| google-cloud-storage / pubsub | ≥2.10 / ≥2.21 | 3.12.1 / 2.39.0 | unpinned | from lock at build (~10-01) | — | same as lock |
| boto3 | ≥1.34 | 1.43.45 | unpinned | — | — | 1.43.45 |
| Python | ≥3.11,<3.13 | — | 3.12.13 | 3.12.13 | Isaac Sim bundled | 3.12 (and 3.11). A `tomli` shim cites Python 3.10 images |
| Node (WebApp) | — | — | — | — | — | Render unpinned (no `engines`, `.nvmrc` or `NODE_VERSION`); CI Node 20 (end of life 2026-04-30); functions nodejs22 |

Shims that imply skew:
- `openpi_droid_policy_runtime.py:525`: the `ping_timeout` check for websockets 12.
- `toml_compat.py`: tomllib or tomli.
- `remote_cpu_environment.py:120`: NumPy 1 or 2.
- `atlas_shared_view_alignment.py:53`: OpenCV SIFT.
- `native_task_arena_preconstruction.py:148`: the Isaac Lab API.
- `atlas_privacy_screen.py:37`: `FaceDetectorYN`. This is the face-blur guard, so it stays fail-closed.

WebApp majors behind:

| Package | Current major | Latest major |
|---|---:|---:|
| React | 18 | 19 |
| Vite | 5 | 8 |
| Express | 4 | 5 |
| firebase | 11 | 13 |
| firebase-admin | 13 | 14 |
| vitest | 3 | 5 |
| TypeScript | 5.6 | 7 |
| Stripe | 17 | 23 |

zod is current.

## Environment variables and flags

Scale of the scan:
- Pipeline: 1,936 env names are read. 155 are set in a repo unit and 72 by a deploy tool or door script.
- WebApp: 796 names, 278 of them read but in no env example.

"Production sets it" below means a repo deploy file sets it. Host and Render env files were not read, so "no" is an upper bound. The inventory has 815 flag rows: every guard, dormant or removal-candidate flag, and every flag production sets. No values are shown.

| Name | Default (code) | Production sets it | Switches | Class / plan |
|---|---|---|---|---|
| `BLUEPRINT_VAST_CREDIT_GUARD_ENABLED` | false | yes, all 11 allocator units | the Vast credit check before launch | KEEP-GUARD. Flip the default to true so a new unit fails safe (P-D01, owner) |
| `BLUEPRINT_LAUNCH_PROOF_MODE` | — | yes, 5 units (`production`) | forces safety flags on | KEEP-GUARD |
| `BLUEPRINT_RIGID_ROBOT_FREE_PROBE_ENABLED` | off | yes, the probe runner unit | the $1 rigid probe | KEEP-GUARD |
| `BLUEPRINT_TEAM_SCENE_PACKET_EXECUTION` | unset: auto (Cloud Run only when the remote config names the stage, else host) | no | the Cloud Run team-scene lane (#3331) | DORMANT-INTENDED |
| `BLUEPRINT_EPISODE_COMPILATION_EXECUTION` | unset: auto (same rule) | no; the host config names the stage, so auto picks Cloud Run | remote episode compilation | DORMANT-INTENDED; P-RT-02 |
| `BLUEPRINT_WEBSITE_ATLAS_ROUTING` | off | host env (unread) | which uploads get an Atlas preview | KEEP-LIVE |
| `BLUEPRINT_WEBSITE_ATLAS_ROUTE` | full | owner sets it for a test only | geometry-only Atlas route | KEEP-LIVE |
| `BLUEPRINT_ARTISAN_AUTHORING_MAX_COST_USD` | 12 | yes, by the deploy tool | the ARTISAN per-object cap | KEEP-GUARD |
| `BLUEPRINT_CAPTURE_RECONSTRUCTION_POLICY_ROOT`, `…_QUEUE_ROOT` | unset | env example sets them | listener enqueue to the legacy reconstruction queue | goes with P-RT-01 |
| `BLUEPRINT_STALL_ALARM` and `BLUEPRINT_STALL_*_MINUTES` | on; 90/480/30/60 | no | stall paging | KEEP-GUARD |
| 67 `BLUEPRINT_*ALLOW_*` opt-ins | off | none in the repo | 44 open a paid or external call (KEEP-GUARD); 23 protect none of privacy, rights or money (for example `…_LEGACY_MARBLE_EVAL_PREP`, `…_SYNTHETIC_GEOMETRY`) | P-D02, owner. Check the host env names first |
| `BLUEPRINT_UNITREE_ALLOW_REAL_ROBOT_COMMANDS` | off | no | physical-safety interlock | KEEP-GUARD |
| 21 flags read only by unreachable modules (cloud-tasks transport, RoboArena/thesis OpenAI switches, Chrono/Drake paths, …) | — | no | — | go with P-A02/P-A06/P-A08/P-A11 |
| WebApp `BLUEPRINT_TASK_EVALUATION_LAUNCH_FORWARD_ONLY_WORKER` | unset (off) | `render.yaml` sets it on the worker | worker topology; why the ops scheduler is off | KEEP-LIVE |
| WebApp `BLUEPRINT_RUN_OPS_AUTOMATION_IN_WEB` | off | no | runs the 24-lane ops scheduler (including the settlement sweep) in web | owner-call |
| WebApp `BLUEPRINT_READINESS_REDIS_WAIVED` | — | `render.yaml` = 1 | Redis waived (founder decision 10-14) | KEEP-LIVE |
| WebApp `BLUEPRINT_INTAKE_DAILY_LIMIT` | 5 | Render (unverified) | open-intake cap | KEEP-GUARD (tracker F-04) |
| WebApp `VITE_BLUEPRINT_STORAGE_PROVIDER`, `BACKBLAZE_B2_KEY_NAME` | — | documented | read by no code | RENAME/DOCUMENT (P-W06) |

## Mirrored contracts (Pipeline ↔ WebApp)

Pairs that must change in both repos together. The rows marked **no parity test** have nothing that fails when one side changes alone:

| Concept | WebApp | Pipeline | Parity test |
|---|---|---|---|
| Screening site limit ($30) | `server/utils/betaPolicyScreeningContract.ts:22` `SCREENING_SITE_LIMIT_USD` | `beta_policy_screening_coordinator.py:62` `SITE_LIMIT` | **no** |
| Screening holds | `server/utils/betaPolicyScreeningHold.ts:33` | `beta_policy_screening_holds.py:54` | **no** |
| Owner-observation versions | `client/src/lib/recordingRightsStatement.ts:16` | `capture_original_owner_observer.py:44` | **no** |
| Scene-intent `outcome.json` | `server/utils/sceneIntentOutcome.ts:29,38` | `task_evaluation_scene_outcome.py:51,165` | **no** |
| Website job window (7 days) | `server/utils/websiteJobWindow.ts` | `website_job_window.py` | Python side only |
| Canonical JSON digest | `server/utils/crossRuntimeCanonical.ts` | `rfc8785` / `cross_runtime_canonical_digest` | yes (golden vectors) |
| Doctrine, robot-eval request, iOS rules | — | — | yes (CI) |

The full list (17 named pairs and 270 shared schema literals) is in the inventory.

## Owner decisions needed

One line each: the choice, the trade-off, and the recommendation.

1. **Catalog split (P-DT-12).** Hide never-runnable rows (58 placeholders, 24 scripted baselines) from website campaigns. Website results and the ledger get much shorter, at the cost of a Pipeline and WebApp change. *Recommend: yes; filter at plan time and keep ids readable.*
2. **Legacy control-plane loop (P-RT-06).** Retire it: it costs CPU and sends a false "blocked" alarm. The trade-off is that the paused agent-run lane would need it again if that lane resumes. *Recommend: an owned door hold now, then retire after a quiet week.*
3. **GCP warm pool and staging intake (P-RT-07+P-A09).** These are not installed. Retiring them drops 4 units, 6 modules and one weekly workflow; it needs the full suite. *Recommend: retire.*
4. **Scene object discovery (P-RT-03).** It has never been used. Retiring it drops 2 units and an admin panel. *Recommend: retire, keeping the `/ops/task-evaluation-launches` page.*
5. **Remote-CPU image (R1).** Rebuild and probe for a small Cloud Run spend, so #3331 can work; otherwise switch the lane off. *Recommend: rebuild.*
6. **RunPod.** Fix the billing call, or retire the account and drop it from the guard and the reconciler together. *Recommend: retire if unused (inventory 0 rows).*
7. **Vast credit guard default (P-D01).** Flip `false` to `true`. It costs nothing and makes the guard fail safe. *Recommend: yes.*
8. **Measurement lane (P-A01).** Retire it, with its monthly workflow and 4 heavy dependencies, or keep researching. *Recommend: retire unless it is planned.*
9. **Articulated door lane (P-A05).** 5,100 lines, ADP-adjacent. *Recommend: ask the ADP owner; retire if the second-scene work is done.*
10. **WebApp dormant lanes.** Growth and city-launch, the capturer network, Stripe payouts and checkout, and voice: about 70 endpoints and their pages. *Recommend: retire the growth and voice lanes. Keep capturer and Stripe until the iOS app's needs are checked.*
11. **WebApp ops scheduler.** Confirm it is meant to be off, including the `agent_run_settlement` sweep. *Recommend: turn on just the settlement sweep if agent-team spend restarts.*
12. **Legacy capture lane infrastructure.** Swap dispatch, the `pipeline` Cloud Run job, Cloud Tasks, video-to-world and, with the lane, the privacy services. *Recommend: first read what is actually deployed (owner gcloud), then retire with the non-website capture lane.*
13. **Outdated images and the signed-release workflows.** *Recommend: retire the images with no consumer. Decide whether the signed-release path is wanted; today every deploy is "iteration".*
14. **Docs index (P-DT-05).** `docs/README.md` limits "active execution documents" to ADP files (an owner decision of 10-14). *Recommend: list the beta's operating docs and `SYSTEM_MAP.md` there too.*
15. **Binaries in git.** The 20 MB HTML and 38 MB of WebApp design files could move to storage. *Recommend: archive in-tree first; moving them out of git is optional.*
16. **Never-run GPU test suites (P-DT-09).** Run them in a paid GPU lane, run them manually, or delete them. *Recommend: manual and documented; keep the rigid-probe suite.*
17. **Non-guard `ALLOW_*` switches (P-D02).** Delete the 23 that protect none of privacy, rights or money, once the host env shows they are unset. *Recommend: yes.*
18. **"Book the call" leftover.** Remove the Calendly button from the brief page. *Recommend: yes (F-09).*

## Phased plan

Every plan below is PR-sized. Its full detail is in the inventory's `plans` section: files, units, dependencies, tests, replacement, impacted-test selection, records checked, deploy order, rollback and diff size.

Deploy rule for every Pipeline plan:
- Ship in the D-08 window (15:00 UTC, 9:00 am CST), never while a screening is launching or running.
- WebApp-only plans deploy through `deploy.yml`.
- Pipeline before WebApp whenever a contract changes.

### Phase 1: docs-only, and items the owner has already decided

| Plan | Change | Diff | Tests |
|---|---|---|---|
| P1-01 | This PR: `SYSTEM_MAP.md`, this report, the inventory | +3 files | 10 doc sentinels (passed, see the PR) |
| P-DT-06 | `beta-ops-incident-response.md` rollback section → link to `beta-operations.md` §5 | ~40 lines | doc sentinels plus `validate_beta_capacity_storage` |
| P-DT-04 | Fold `CONTROL_PLANE_CAPACITY_PLAN.md` and `CONTROL_PLANE_DISK_REDESIGN_2026-09-26.md` into `CONTROL_PLANE_STORAGE.md`; archive them | ~40 lines + 3 moves | doc sentinels |
| P-DT-01, P-DT-03 | Archive `docs/verification`, `docs/superpowers`, dated audits and plans, and `audit/policy-comparison-20260907` (no reader) | ~135 moves | doc sentinels |
| P-W06 | WebApp `DEPLOYMENT.md`: storage-provider paragraph, the ops scheduler topology, Node policy; the `AGENTS.md` Stack line | ~30 lines | `npm run check` |
| P-INFRA-01 | Delete the orphaned `unitree_groot_sonic_vast` Dockerfile and its dependabot line | −36 lines | 17 selected, no full suite |
| P-W03a | Remove `@heroicons/react` and `@google/generative-ai` (proven unused) | package files | `npm ci && npm run build && npm run check` |
| P-DT-02 | Archive three experiment directories nothing reads (keep `policy_ranking_thesis_20260726` and `argus_shadow_20261001`, which are read at runtime) | ~325 moves | doc sentinels |
| P-DT-14 | `AGENTS.md`: correct the stated site screening limit from $15 to $30 (owner confirms the wording) | 2 lines | `--plan-only` on `AGENTS.md` |
| P-INFRA-02 | Disable the two fileless workflow registrations (GitHub UI, reversible) | none | none |
| P-RT-01 (step a) | H-03 host step: disable and remove the 3 capture-reconstruction units (operator, in a D-08 window) | host only | none |

### Phase 2: code removals and consolidations (likely; review each)

| Plan | Change | Diff | Test plan |
|---|---|---|---|
| P-A06 | splat and semantic research | about −17k | narrow, 12 |
| P-A08 | one-off task-evaluation lanes | about −13k | narrow, 11 |
| P-A02 | archive policy-ranking research (record the manifest commit first) | about −21k | narrow, 12 |
| P-A10 | compat shims | about −0.4k | **full suite** |
| P-RT-01 (step b) | the dispatcher module, its listener branch and env names (after the host step and an env-name check); the wider `capture_reconstruction_*` family stays dormant | about −2k | 10 selected / 61 wide |
| P-W01, P-W02 | WebApp dead client and server files | about −7.5k | check, vitest, build, CI e2e |
| P-W03b, P-W05, P-DT-10 | WebApp dependency hygiene; Playwright configs and the e2e specs no CI job runs | small | same |
| P-W04 | 9 legacy endpoints (after 30 days of Render logs show no hits) | about −1.2k | server tests |
| P-DT-11 | 6 likely-dead scripts (after Postshot is confirmed retired); runbook steps for 11 break-glass tools | about −0.9k | per-file selection |
| P-RT-02, P-RT-05, P-RT-09 | idle fast path for the remote compile poll; no receipt on a no-op GC; a `-m` entry for the spend refresh. Guard coverage stays equal | small each | 14 / 20 / 20 |
| P-B01 | gripper key rename and one pad-centre function | about +60/−40 | adapter, environment and lifecycle tests |
| P-B02, P-B03, P-B04, P-B05, P-B06, P-B07 | quaternion normalisation, 15 identical digest copies, one atomic writer, one lock mechanism (lock paths unchanged), read-only retries, one camera flip | one PR each | per-file selection |
| P-DT-08, P-FLAGS-A | tests and flags whose only user is a removed module go in that module's PR, never alone | — | — |
| P-E, P-F, P-W08 | retire skew shims after R1; declare the packages production imports; check the shared fixture copies | small | per-file selection |
| P-C | split the four modules at their cap (keep the GPU bundle lists in step) | moves only | per-file selection plus a bundle import-closure test |
| R1, R4 | sync the host venv to `uv.lock` for digested packages plus scipy; rebuild and probe the remote-CPU image (owner spend); pin Node | config | deploy-tool tests; WebApp CI |
| R2, R3 | fix the RunPod billing call; add B2 to billing evidence | small | reconciler tests |

### Phase 3: after owner decisions

Owner decisions 1–18 above. Each one, once approved, becomes a plan of the same shape: P-DT-12, P-RT-06, P-RT-07+P-A09, P-RT-03, P-A01, P-A05, P-A07, P-A11, P-W07, P-DT-07, P-DT-09, P-DT-13, P-D01, P-D02 and the Terraform and image retirements.

## Re-check of every removal candidate against the rules

Every REMOVE-CANDIDATE in the inventory (46 rows after this check) was checked once more against three questions:
- Is it a guard? None is. The AWS "watchdog" is a refusing stub for a retired provider. The spend-guard verifier inputs and the privacy face-blur shim are marked KEEP-GUARD.
- Does Arm Decision Proof v1 cite it? Every ADP-cited module, doc and page is KEEP-ADP.
- Does a stored record depend on it?
  - Retention tools and digest helpers are KEEP-RECORDS.
  - The gripper rename keeps a read alias.
  - The catalog split keeps candidate ids readable.

Four items changed in this check:
- The capture-reconstruction dispatcher went from proven to likely, because of the host policy folder.
- The 36 "tests of dead code" became 21.
- `public_scene_simready_validation_stage` (no importer anywhere) moved to KEEP-ADP. It sits in the ADP-009A lane, so the ADP owner decides.
- `GIT_CONFIG_COUNT` moved to KEEP-LIVE: 32 unit files set it for git.

## Gaps (not checked, or refused)

- **Door refusals and errors:** `/etc/blueprint/provider-secrets` (`path_hidden`; not worked around); a `500 internal_error` listing `pcp/preflight/` and `pcp/storage-gc/owner-approved-deletions`.
- **Not readable:** host env files (by rule); host package list (no `pip list`); Render dashboard, env and request logs; GCP console and Terraform state; the live Firestore index list; Firestore data (the Firebase connector failed to connect); Isaac Sim image contents; the BlueprintCapture iOS repo.
- **Journal window:** journals keep 14 days, so "idle 30+ days" comes from record mtimes.
- **Not inspected:** `door show` does not list drop-ins, so the hotfix-overlay drop-in is unverified.
- **Static only:** "no caller" for WebApp endpoints and unreachable Python modules is static evidence. Per-module journals and the stored records that name a producer module were not checked.
- **History and PRs:** both clones are shallow, so there are no per-file last-change dates. Two open PRs (#3056, #2808) list more than 100 files and were read only in part.
- **Assumed:** the KEEP-ADP status of agent execution and stage replay rests on the `adp-assessment-pilot` directory name.
