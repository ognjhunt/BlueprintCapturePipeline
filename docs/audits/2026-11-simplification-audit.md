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

## Status of every plan (reconciled 2026-11-10)

This section is the ledger for the 60 plans in the inventory's `plans` section. The rest of this document is the 2026-11-05 audit as written and is not updated; where the two disagree, this section is current.

- Checked against Pipeline `main` at `4f2c74c84` and WebApp `main` at `772800816`, and against the host through `scripts/operator_door.py units`, `show` and `journal` on 2026-11-10 03:10 UTC.
- "#3407" is Pipeline PR #3407 (merged 2026-11-06). "#1224" is WebApp PR #1224 (merged 2026-11-06). Both were approved by the owner on 2026-11-05.
- **Status** is one of: `done`, `partly done`, `still open`, `no longer valid`. Rows that begin "Merged 2026-11-10" give the status after the cleanup PRs listed below the tables, and keep the evidence read before the merge.
- **Needs owner** repeats the inventory's `owner_decision_needed`, or says which host or GitHub step only the owner can take.
- The last column names the PR that advances each plan. The cleanup PRs of 2026-11-10, the reasons for plans without a PR, what was found past the inventory, and the decision list follow the three tables.

Three things the reconciliation found that the audit could not know:

1. **The legacy control-plane loop still runs on the host.** #3407 removed `blueprint-pipeline-control-plane.{service,timer}` from the repo and from the deploy list, but the host timer is still `enabled` and `active`: it last fired 2026-11-10 03:08:30 UTC and used 12.5 CPU-seconds (`door show`, `door journal`). Removing a unit file from the repo does not stop it on the host. Until an operator disables the timer, `live_pipeline_control_plane.py` must stay in the repo, or the unit fails every 5 minutes.
2. **The three capture-reconstruction dispatcher units are still loaded on the host** (`inactive`, `dead`). Their module is gone from the repo, so starting one would fail. The host step (P-RT-01 step a) is still to do.
3. **`AGENTS.md` still states $15 in the spend table** (line 53); the code and the WebApp are at $30. Line 24 no longer names a figure. This is P-DT-14 and stays with the owner.

### Phase 1

| Plan | What | Status | Evidence (2026-11-10) | Needs owner | PR |
|---|---|---|---|---|---|
| P1-01 | System map, audit, inventory | done | The three files are on `main` (#3407) | no | #3407 |
| P-DT-06 | Incident runbook rollback → `beta-operations.md` §5 | done | Merged 2026-11-10 (PR in the last column). Before the merge: `docs/runbooks/beta-ops-incident-response.md:54-66` still gives the WebApp rollback command and "restore the last known good image tag" for a Cloud Run job; buyer, checkout and payout wording remains (lines 23-32) | no | #3530 (merged; completes it) |
| P-DT-04 | One control-plane storage doc | done | Merged 2026-11-10 (PR in the last column). Before the merge: `docs/superpowers/plans/2026-09-26-control-plane-disk/` is archived (#3407). `docs/CONTROL_PLANE_CAPACITY_PLAN.md` and `docs/CONTROL_PLANE_DISK_REDESIGN_2026-09-26.md` are still top-level | no | #3529 (merged; completes it) |
| P-DT-01 | Archive `docs/verification` | done | `docs/verification` is gone; the files are under `docs/archive/verification` (#3407) | moving the 20 MB HTML out of git is still an owner call | #3407 |
| P-DT-03 | Archive dated history docs | partly done | `docs/superpowers` and 14 dated docs are archived (#3407). `audit/policy-comparison-20260907/` and 38 dated top-level docs remain; several of those are read by tests or at runtime and stay | no | none; reason below |
| P-W06 | WebApp deployment doc and Stack line | done | Merged 2026-11-10 (PR in the last column). Before the merge: #1224 rewrote 124 lines of `DEPLOYMENT.md` (scheduler topology). The storage-provider paragraph still documents `VITE_BLUEPRINT_STORAGE_PROVIDER`, which no file under `client/src` or `server` reads; no Node policy is written down | no | WebApp #1248 (merged; completes it) |
| P-INFRA-01 | Delete the orphaned `unitree_groot_sonic_vast` Dockerfile | done | Merged 2026-11-10 (PR in the last column). Before the merge: `deploy/docker/robot_eval_worker/unitree_groot_sonic_vast/Dockerfile` exists; its only references are `.github/dependabot.yml`, `docs/CHANGELOG.md` and this audit | no | #3526 (merged; completes it) |
| P-W03a | Remove two unused WebApp packages | done | Neither `@heroicons/react` nor `@google/generative-ai` is in `package.json` or `vite.config.ts` (#1224) | no | #1224 |
| P-INFRA-02 | Disable two fileless workflow registrations | still open | GitHub setting; not changed by any PR | yes (GitHub settings) | |
| P-RT-01 | Retire the capture reconstruction dispatcher | partly done | Step b is done: the module, the three unit files and the listener branch are gone (#3407). Step a is not: the host still has the three units loaded (`door units`: `inactive`, `dead`) | host step: disable and remove the three units | #3407 |
| P-DT-02 | Archive three unreferenced experiment directories | partly done | `argus_video_explanation_default_20261001` and `measurement_isaac_vast_20260802_attempt_1` are under `docs/archive/experiments` (#3407). `policy_ranking_thesis_experiment_2_20260727` waits for P-A02 | no | none; reason below |
| P-DT-14 | `AGENTS.md` states the old $15 site limit | partly done | Line 24 no longer names a figure. The spend table (line 53) still says $15; `SITE_LIMIT` is `Decimal("30")`, `SITE_BUDGET_USD` is `Decimal("30.00")`, WebApp `SCREENING_SITE_LIMIT_USD` is 30 | yes (money figure in an owner-owned section) | |

### Phase 2

| Plan | What | Status | Evidence (2026-11-10) | Needs owner | PR |
|---|---|---|---|---|---|
| P-A06 | Splat, semantic and reconstruction research | partly done | 20 of the 25 listed modules are gone (#3407). Still present: `frames_layout` (imported by `materialization.py`, live), `gaussian_object_partition` (imported by `simready_asset_lane.py`), `artifixer_heldout_evaluation` (imported by `nvidia_siggraph_completion.py`, a P-A11 operator tool), `particlefield_upstream_parity` and `capture_reconstruction_publication` (imported only by their own tests) | no | none; reason below |
| P-A08 | One-off task-evaluation lanes | partly done | 18 of the 24 listed modules are gone (#3407). Still present: `external_scene_task_evaluation`, `external_scene_robot_placement_evidence`, `provider_nurec_task_evaluation`, `provider_robot_placement_evidence` (import only each other and their tests), `task_evaluation_sam31_service_provisioning` (imported only by its test), and `task_evaluation_downstream_replay` (kept on purpose, tracker B-01) | no | none; reason below |
| P-A02 | Archive policy-ranking research | partly done | Merged 2026-11-10 (PR in the last column). Before the merge: All 36 listed modules exist | no | #3533 (merged; 22 of the 36 modules) |
| P-A10 | Compat shims and retired entrypoints | done | All 7 modules are gone (#3407) | no | #3407 |
| P-W01 | Unreachable WebApp client files | done | The 5 pages and the listed components and libs are gone; the 3 Radix packages are gone (#1224) | no | #1224 |
| P-W02 | Unreferenced WebApp server files | done | All 8 files are gone (#1224) | no | #1224 |
| P-W03b | WebApp dependency hygiene | partly done | Merged 2026-11-10 (PR in the last column). Before the merge: `bcryptjs`, `node-fetch`, `remotion`, `@remotion/bundler`, `@remotion/renderer` and the Replit plugin are all still in `dependencies` | no | WebApp #1251 (merged; part) |
| P-W04 | 9 legacy WebApp endpoints | still open | 8 of the 9 handlers exist; `admin-creative.ts` went with the growth lane (#1224). The plan's condition (30 days of Render request logs) cannot be met from the repo | Render logs are not readable from a session | none; decision list |
| P-W05 | Playwright configs no CI job runs | partly done | Merged 2026-11-10 (PR in the last column). Before the merge: 14 `playwright*.config.ts` files; `ci.yml` runs 3 by name (`invited-beta`, `reliability-intake`, `privacy`) plus the default | no for the 5 deletions; the 3 intake configs are an owner call | WebApp #1250 (merged; 4 configs) |
| P-DT-11 | Scripts with no caller | done | Merged 2026-11-10 (PR in the last column). Before the merge: All 6 scripts exist | no | #3527 (merged; completes it) |
| P-RT-02 | Idle fast path for the remote compile poll | still open | No fast path in `task_evaluation_episode_compilation_collector.py` | no | none; reason below |
| P-RT-05 | No receipts for a no-op replay-cache GC tick | partly done | Merged 2026-11-10 (PR in the last column). Before the merge: `completed_replay_cache_retention.py` `main()` still writes `-plan.json` and `-result.json` on every tick; the unit still has `Wants=blueprint-scene-project-spend-refresh.service` | the `Wants=` line is a money-freshness question | #3528 (merged; the receipts) |
| P-RT-09 | A `-m` entry for the scene spend refresh | still open | The unit still starts `python -c "from … import refresh_configured_scene_project_spend; …"` | no | none; reason below |
| P-B01 | One meaning for gripper separation | still open | `finger_separation_m` is in 10 source files; neither `pad_center_separation_m` nor `finger_body_origin_separation_m` exists | no | none; reason below |
| P-B02 | Normalising quaternion helpers | still open | Not changed since the audit | no | none; reason below |
| P-B03 | Delegate identical digest copies | still open | Not changed since the audit | no | none; reason below |
| P-B04 | One atomic JSON writer | still open | 175 source files call `os.replace`; 67 of them never call `fsync` | no | none; reason below |
| P-B05 | One file-lock mechanism | still open | 124 source files call `flock` or `lockf` | no | none; reason below |
| P-B06 | Read retries through `transport_retry_policy` | still open | The shared module still has 2 importers (`vast_inventory_read_retry`, `provider_transport`) | no | none; reason below |
| P-B07 | One OpenGL/OpenCV camera flip | still open | 8 source files define `diag(1, -1, -1)` | no | none; reason below |
| P-C | Split modules at their line cap | still open | Line counts are unchanged: 2295, 1986, 10420, 7456 | no | none; reason below |
| P-E | Retire version-skew shims | still open | Waits for R1 (host and worker image aligned). `toml_compat.py` has one importer, `runpod_provider_adapter.py` | R1 first | none; reason below |
| P-F | Python dependency declarations | still open | `requests` is declared only in the `cloud` extra and `httpx` only in the `dev` extra; `pydantic` and `starlette` are not named in `pyproject.toml` | no for the declarations; the `langgraph*` and `groot-libero-cpu` extras are an owner call | none; reason below |
| P-FLAGS-A | Flags read only by removed modules | partly done | 3 of the 15 flag names are gone with their readers; 12 still have a reader and go with that reader's plan | no | |
| P-DT-08 | Tests go only with their module | partly done | A rule, not a change. #3407 followed it; each later PR does the same | no | |
| P-DT-10 | WebApp e2e specs that never run in CI | still open | The 6 specs exist | yes | |
| P-W08 | Shared fixture copies | still open | `screen-admission-snapshot.json` still differs between the repos (`cmp`: first difference at byte 21) | no | none; reason below |
| P-A11 | Document operator diagnostics as tools | done | Merged 2026-11-10 (PR in the last column). Before the merge: Of 4 sampled tools, only `website_atlas_operator` is named in a runbook | no | #3535 (merged; completes it) |
| R1 | Host venv from `uv.lock`; rebuild the remote-CPU image | still open | `production-control-plane-requirements.txt` still pins a handful of packages | yes (deploy window, Cloud Run spend) | |
| R2 | Fix RunPod billing reconciliation | no longer valid | RunPod was retired on 2026-11-05 (`RUNPOD_PROVIDER_RETIRED = True`); the reconciler no longer makes the request. What is left is the owner's account closure (`AGENTS.md`, RunPod row) | account closure | #3407 |
| R3 | Billing evidence for B2, Gemini, hosted SAM | still open | `PROVIDERS` in `provider_billing_reconciler.py` is unchanged | yes | |
| R4 | Pin Node | done | Merged 2026-11-10 (PR in the last column). Before the merge: No `.nvmrc`, no `engines`, no `NODE_VERSION` in `render.yaml`; `ci.yml` uses Node 20 in 6 jobs | no | WebApp #1248 (merged; completes it) |

### Phase 3 (owner decisions)

| Plan | What | Status | Evidence (2026-11-10) | Needs owner | PR |
|---|---|---|---|---|---|
| P-DT-12 | Website campaigns plan only runnable candidates | done | #3407 changed `beta_policy_screening_plan.py`; #1224 added the WebApp rendering tests | approved 2026-11-05 | #3407, #1224 |
| P-RT-06 | Retire the legacy control-plane loop | partly done | The unit files and the deploy entry are gone (#3407). The host timer is still enabled and firing (see above). `live_pipeline_control_plane.py`, `live_pipeline_proof_audit.py` and `live_pipeline_manifest_alert.py` remain | host step: disable the timer, then the module can go | #3407 |
| P-RT-07+P-A09 | Retire the GCP warm pool and staging intake | done | Merged 2026-11-10 (PR in the last column). Before the merge: The 4 units, the env examples, the install script, the workflow and 5 of the 6 modules are gone (#3407); none of the units is on the host. `production_gpu_runpod_autoscaler.py` and its test remain | approved 2026-11-05 | #3407; #3531 (merged; completes the repo side) |
| P-RT-03 | Retire scene object discovery | still open | Both unit files exist; on the host the path unit is `active` | yes | |
| P-A01 | Retire the measurement lane | partly done | The 31 modules and the workflow are gone (#3407). `scripts/measurement_research_monitor.py`, `measurement_research_monitor.py` and `measurement_research_monitoring.py` remain | approved 2026-11-05 | #3407; none; reason below |
| P-A05 | Retire the articulated door lane | done | 15 of the 18 modules are gone (#3407). The other 3 stay: `articulated_simready_replacement` is ADP-cited and imports `articulated_dynamics_profiles` and `articulated_dynamics_realism` | approved 2026-11-05 | #3407 |
| P-A07 | Retention and reclamation tools | still open | All 11 modules exist | yes (they delete retained data) | |
| P-W07 | Firestore index manifest | still open | Not changed | yes | |
| P-DT-07 | Archive the July SC3 launch-evidence program | still open | The SC3 ledger and its tests exist | yes | |
| P-DT-09 | Never-run external suites | still open | Not changed | yes (a GPU lane costs money) | |
| P-DT-13 | Archive WebApp design binaries | partly done | `docs/design` is under `docs/archive/design` (#1224), apart from one JSON file. `docs/visuals` (9.1 MB) is still in place | yes | #1224 |
| P-D01 | Vast credit guard on by default | done | `credit_guard_setting()` reads `os.getenv(ENABLED_ENV, "true")` (#3407) | approved 2026-11-05 | #3407 |
| P-D02 | Non-guard `ALLOW_*` switches | still open | 54 of the 55 flag names in the plan still have a reader | yes | |
| P-DT-05 | Docs index lists the beta's operating docs | still open | `docs/README.md` still limits "active execution documents" to the ten ADP files | yes | |
| P-RT-08 | Existing-policy canary continuation and watchdog | still open | The 3 unit files exist; the continuation units are not on the host | yes | |
| P-B08 | Spend-cap arithmetic | still open | Not changed (money guard) | yes | |

### Cleanup PRs opened on 2026-11-10

The PR column above says which plan each PR advances. The owner asked for all thirteen to be merged on 2026-11-10, and they were; this ledger PR merged last. Pipeline merges reach the host with the next daily deploy; the three WebApp merges deploy to Render by themselves.

| PR | What it does |
|---|---|
| #3525 | This ledger |
| #3526 | Deletes the orphaned `unitree_groot_sonic_vast` Dockerfile and its dependabot entry (P-INFRA-01) |
| #3527 | Deletes six uncalled scripts and the three modules only they used, with their tests (P-DT-11) |
| #3528 | Replay-cache GC writes no receipt for a tick with nothing to reclaim (P-RT-05) |
| #3529 | Archives the two superseded control-plane capacity plans (P-DT-04) |
| #3530 | Incident runbook sends host rollbacks to the operator door (P-DT-06) |
| #3531 | Removes five leftover GCP warm-pool modules, their four console scripts and tests (P-RT-07+P-A09) |
| #3532 | Deletes two stray screenshots from the repo root (past the inventory) |
| #3533 | Archives 22 policy-ranking research modules and 20 test files (P-A02, part 1) |
| #3535 | Adds `docs/OPERATOR_TOOLS.md` for the 19 operator tools (P-A11) |
| WebApp #1248 | Pins Node to 22.x in one place; corrects the storage-provider paragraph (R4, P-W06) |
| WebApp #1250 | Deletes four Playwright configs that nothing runs (P-W05) |
| WebApp #1251 | Drops `bcryptjs` and its unused helper; declares `gaxios` and `google-auth-library` (P-W03b) |

All ten Pipeline branches merge into one tree without conflict. Measured with the same commands on main and on that merged tree: source files 2,534 → 2,504, source lines 1,334,234 → 1,323,054, test files 2,734 → 2,706, test lines 1,064,152 → 1,059,558, scripts 299 → 293, console scripts 175 → 171, Dockerfiles 22 → 21.

### Open plans with no PR, and why

| Plan | Why there is no PR |
|---|---|
| P-DT-03 | `audit/policy-comparison-20260907/` cannot move under `docs/`: `scripts/verify_source_governance.py` then fails on personal absolute paths in `evidence/identity.json:8` and `evidence/worktrees.txt:1`. Of the 38 dated top-level docs, most are read by a script, a test or runtime code; the 2026-09-21 authorization notes and the two G1 rights reviews may be cited by stored records and stay |
| P-DT-02 | `docs/experiments/policy_ranking_thesis_experiment_2_20260727/` is cited by path in four sealed evidence files in other experiment directories |
| P-A06 | The five remaining modules are live (`frames_layout`, `gaussian_object_partition`), imported by an operator tool (`artifixer_heldout_evaluation`), tied to an Arm Decision Proof scene audit (`particlefield_upstream_parity`), or the writer of a record shape that `capture_reconstruction_downstream` reads (`capture_reconstruction_publication`) |
| P-A08 | `external_scene_task_evaluation` is file 16 in `docs/arm_decision_proof_v1/ADP_009D_FRANKA_PUBLIC_SCENE_SIM_GOAL_PROMPT.md`, and the other three modules of that island exist only for it. `task_evaluation_sam31_service_provisioning` is the documented provisioning step in `docs/operations/sam31_preparation_services.md` |
| P-A01 | `measurement_research_monitor` and `measurement_research_monitoring` are imported by `task_evaluation_supervisor/agents_sdk.py` and `capabilities.py` |
| P-RT-02 | The collector also cancels orphaned Cloud Run executions (a teardown path). An idle exit must know every place a live execution can be recorded. The cheaper fix depends on decision R1: if the remote lane is switched off, the unit is simply disabled |
| P-RT-09 | With a `-m` entry the chain preflight stops skipping this unit and runs its sandbox probe on it for the first time; a probe failure is a `blocker` finding. That can only be tried on the host, as root |
| P-B01 | Not attempted. The rename touches the construction worker, the pose servo, the readback and the live screening adapter, and needs the calibration test the plan describes written first |
| P-B02, P-B04, P-B05, P-B06 | Not attempted. Each changes behaviour if done wrong (normalisation, durability, lock paths, retries) and needs its own proof per lane |
| P-B03 | A scan for single-statement helpers found 25 local sha256-of-JSON helpers and none byte-identical to `canonical_digest` (none passes exactly `canonical_json`'s arguments to `json.dumps`, and 22 of them return no `sha256:` prefix). The audit's list of 15 was not committed, so it has to be rebuilt before anything is delegated |
| P-B07 | Of the 8 definitions, 2 are in Arm Decision Proof modules, 1 (`robot_arm_chains.py:310`) is not a camera flip, 1 is deleted by #3527, and the other 4 are one-line constants of different shape. No longer worth a PR |
| P-C | Not attempted. Three of the four modules are guards or ship in the GPU bundle; a move-only split still needs the bundle lists changed in step |
| P-E | Waits for R1. `tomli` must stay in `uv.lock` in any case: `scripts/install_scene_retirement_runtime.py:1511-1537` loads a verified `tomli` wheel from the lock |
| P-F | Not attempted. Declaring `pydantic` and `starlette` changes `uv.lock`, both generated `requirements*.txt` files and the supply-chain evidence; do it with the next dependency refresh |
| P-W08 | Not attempted |

### Found past the inventory

1. **Console scripts keep switched-off lanes installed.** 61 of the 175 console scripts in `pyproject.toml` are the only thing outside tests that references their target module. If those 61 entries went, 67 more modules (51,076 lines) would be reachable from nothing. #3531 removes the 4 that belong to the already-retired GCP warm pool; the other 57 need a lane-by-lane owner decision. The 61: `blueprint-adapt-polycam-developer-raw`, `blueprint-agent-review`, `blueprint-audit-first-gpu-candidates`, `blueprint-audit-first-gpu-cross-repo-readiness`, `blueprint-audit-first-gpu-e2e-readiness`, `blueprint-audit-first-gpu-sample-video`, `blueprint-audit-provider-closure`, `blueprint-audit-unitree-groot-sonic-provider-readiness`, `blueprint-autogen-scene-eval-tasks`, `blueprint-backfill-site-reference-db`, `blueprint-bootstrap-wam-model-runtime`, `blueprint-build-benchmark-uncertainty-report`, `blueprint-build-capture-batch-registry`, `blueprint-build-exact-workcell-variation-matrix`, `blueprint-build-first-gpu-run-packet`, `blueprint-build-lightwheel-kitchen-isaac-scenarios`, `blueprint-build-manipulation-task-stack`, `blueprint-build-mujoco-scene-scenario-packet`, `blueprint-build-production-gpu-launch-qualification`, `blueprint-build-production-handoff-readiness`, `blueprint-build-public-scene-simready-control`, `blueprint-build-unitree-unifolm-gpu-image`, `blueprint-build-wam-perception-harness-gpu-image`, `blueprint-check-cpu-env`, `blueprint-compile-articulated-public-scene`, `blueprint-compile-site-task-testbed`, `blueprint-create-team-policy-endpoint-token`, `blueprint-deliver-arena-package-local`, `blueprint-g1-endpoint-reference-adapter`, `blueprint-import-scaniverse-assets`, `blueprint-launch-unitree-unifolm-runpod-server`, `blueprint-materialize-public-scene-suite`, `blueprint-openvla-policy-command-adapter`, `blueprint-prepare-unitree-groot-sonic-wam-remote-build`, `blueprint-qualify-capture-reconstruction`, `blueprint-review-g1-field-run-evidence`, `blueprint-route-task-evaluation`, `blueprint-run-gpu-provider-launcher`, `blueprint-run-kitchen-task-scaling-preflight`, `blueprint-run-lucky-g1-reference-adapter`, `blueprint-run-major-capability-scenarios`, `blueprint-run-manipulation-physics-sim`, `blueprint-run-openvla-policy-provider-smoke`, `blueprint-run-oscar-visual-augmentation-generation`, `blueprint-run-policy-autoresearch-local-evaluator`, `blueprint-run-policy-autoresearch-wam-fixture-evaluator`, `blueprint-run-production-gpu-replenishment-probe`, `blueprint-run-production-gpu-rollback-drill`, `blueprint-run-production-gpu-worker-pool`, `blueprint-run-provider-worker-session`, `blueprint-run-robot-eval-provider-race`, `blueprint-run-unitree-groot-n17-sonic-policy-server-preflight`, `blueprint-run-unitree-unifolm-policy-provider-smoke`, `blueprint-run-wam-sim-provider-e2e`, `blueprint-serve-wam-vla-policy-endpoint`, `blueprint-setup-wam-vla-policy-endpoint`, `blueprint-smoke-arena-package-local`, `blueprint-stage-first-gpu-sample-video`, `blueprint-task-candidate-control`, `blueprint-unitree-lerobot-policy-command-adapter`, `blueprint-write-owner-gpu-default-smoke-artifacts`.
2. **34 unreachable modules the inventory does not list** (13,649 lines): `policy_ranking_successor_cosmos_provider_runtime`, `policy_improvement_run`, `public_scene_inpaint360_adapter`, `articulated_excision_join`, `gaussian_source_removal_qualification`, `public_reference_admission`, `public_scene_simready_replacement`, `gaussian_suppression_volume`, `paired_target_native_render_request`, `cosmos_edge_closed_loop_provider_bundle`, `task_evaluation_task_parameter_proposal`, `paired_target_native_preflight`, `fresh_scene_paired_target_preparation`, `policy_ranking_successor_cosmos_bundle`, `task_evaluation_planar_push_readiness_candidate`, `public_scene_artifixer3d_final_composite`, `task_evaluation_preparation_claim_recovery`, `semantic_teacher_pending_spend`, `public_scene_infusion_compose`, `public_scene_hybrid_replacement_seal`, `public_scene_task_selection`, `openai_unattributable_spend`, `task_evaluation_passive_destination_simready`, `task_object_artisan_replay`, `capture_reconstruction_checkpoints`, `capture_reconstruction_downstream`, `capture_reconstruction_postshot_allocator`, `company_policy_conformance`, `openpi_checkpoint_inventory`, `skypilot_promotion_gate`, `native_g1_packet_choice_handoff`, `optional_dependencies`, `stage_outcome`, `lane_resume`. Several look like Arm Decision Proof code (`public_scene_*`, `paired_target_*`) and two are spend records (`semantic_teacher_pending_spend`, `openai_unattributable_spend`); none was touched.
3. **RunPod code outlives RunPod.** 26 modules with `runpod` in the name hold 25,045 lines. RunPod is retired for new spend; the spend guard still needs its inventory read while a key exists.
4. **Three more compatibility shims**, imported only by tests that pin them: `optional_dependencies.py`, `stage_outcome.py`, `lane_resume.py` (16 lines).
5. **ruff `target-version = "py310"`** while the package requires Python 3.11 or later. Changing it to `py311` adds no finding with the locked ruff 0.15.21. Not sent as its own PR because any `pyproject.toml` change runs the full suite; it can ride with the next one.
6. **Three tests fail when the suite runs as root** (`tests/test_task_evaluation_result_artifact_store.py`: `test_root_eviction_lease_needs_no_fowner`, `test_existing_lock_owned_by_registry_owner_is_not_chmodded[0600-left-by-a-failed-tick]`, `test_non_root_reader_takes_the_flock_on_a_lock_someone_else_owns`). Seen on unchanged main in a cloud session with uid 0.
7. **WebApp:** `ops/paperclip/reports/` has 949 tracked files (8.25 MB) although `.gitignore:64` ignores the directory; the audit's row "city-launch utils read it at runtime" is out of date, since #1224 deleted those utils. Its one remaining reader is an optional input root of `scripts/paperclip/agent-improvement-observer.ts`. The 27 files under `scripts/` dated 20261006 and 20261007 form one incident-evidence set that pins its own file hashes. `@replit/vite-plugin-shadcn-theme-json` cannot move to devDependencies as P-W03b proposed: production boot imports it through `server/vite.ts`.

How the reachability figures were measured: a static import graph over `src/blueprint_pipeline`, with string module names counted as edges, and with every module named in `deploy/`, `scripts/`, `functions/`, `.github/`, `config/`, `configs/`, the root `Dockerfile`, `main.py`, `Makefile`, `docs/runbooks/`, `docs/OPERATOR_DOOR.md` and `AGENTS.md` as a root; once with `pyproject.toml` as a root too (153 unreachable modules, 63,456 lines) and once without (220 modules, 114,532 lines). It is a lead, not proof: each module still needs the caller checks this audit lists.

### Decision list (2026-11-10)

Ordered by payoff. Items 14 to 16 repeat audit decisions that have not moved.

| # | Decision | Evidence | Recommendation | If approved |
|---|---|---|---|---|
| 1 | Host step: disable `blueprint-pipeline-control-plane.timer`; remove it, its service and the three `blueprint-capture-reconstruction-dispatcher` units | The timer is `enabled` and fired at 2026-11-10 03:08:30 UTC, 12.5 CPU-seconds a run; the repo retired it in #3407. The dispatcher units are loaded and their module is gone | Do it in the next deploy window | Then delete `live_pipeline_control_plane.py` and what only it uses, with tests |
| 2 | Which lanes behind the 57 remaining single-use console scripts are retired | Finding 1 above | Name the retired lanes (first-GPU July lane, OpenVLA, Unitree UnifoLM, the WAM VLA endpoint, policy autoresearch, kitchen task scaling and the rest of the list) | One PR per lane: console scripts, modules, tests, docs |
| 3 | RunPod: close the account and delete the key | `AGENTS.md` spend table, RunPod row; finding 3 | Close it | Remove the RunPod launchers and, once no key exists, the guard's RunPod read |
| 4 | The other 14 policy-ranking modules | #3533 lists each with its reason | Remove the four OpenAI judge modules with their entries in `scripts/paid_api_call_sites.json`; decide the six contract-doc modules with those docs | One PR that also edits the paid-call-site list and the verifier's expected rows |
| 5 | Arm Decision Proof leftovers | `external_scene_task_evaluation` and its three helpers, `particlefield_upstream_parity`, the 34 modules in finding 2 | Ask the ADP owner | Delete what ADP no longer cites |
| 6 | The 23 non-guard `ALLOW_*` switches (P-D02) | 54 of the 55 flag names in the plan still have a reader | Delete them once the host env shows them unset | One PR per reader |
| 7 | Scene object discovery (P-RT-03) | Both unit files exist; on the host the path unit is `active`; the audit found no record was ever written | Retire | Units, worker, intake routes, the WebApp panel |
| 8 | WebApp: 8 legacy endpoints (P-W04) | No caller in either repo; Render request logs not readable | Read 30 days of Render logs, then remove | Routes, their rate-limiter lines, tests |
| 9 | WebApp: `ops/paperclip/reports/` | Finding 7 | Delete from the tree (git history keeps it), or decide it is a record to keep | One deletion PR; fix the links in `knowledge/` |
| 10 | WebApp: the dated incident scripts | Finding 7 | Keep as evidence, or archive the set as one unit | Move scripts, tests and docs together |
| 11 | `AGENTS.md` line 53 states 15 dollars; the limit in code is 30 (P-DT-14) | Code constants in both repos | Owner edits the figure | Nothing else |
| 12 | `audit/policy-comparison-20260907/` | Reason under P-DT-03 | Delete it (60 files, history is in git) | One deletion PR |
| 13 | `Wants=blueprint-scene-project-spend-refresh.service` on the replay-cache GC unit (P-RT-05) | Starts the spend refresh every 5 minutes; the capacity unit also pulls it | Drop the line if a 10-minute refresh is fresh enough | One-line unit change, deployed in the window |
| 14 | R1: host venv from `uv.lock`, rebuild or switch off the remote-CPU image | Unchanged since the audit | Rebuild | Unblocks P-RT-02 and P-E |
| 15 | Docs index (P-DT-05) | `docs/README.md` still lists only ADP files as active | Add the beta's operating docs, `SYSTEM_MAP.md` and `OPERATOR_TOOLS.md` | About 30 lines |
| 16 | Unchanged audit owner calls | P-A07, P-W07, P-DT-07, P-DT-09, P-DT-10, P-DT-13, P-RT-08, P-B08, R3, P-INFRA-02 | As the audit recommends | As each plan says |

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
