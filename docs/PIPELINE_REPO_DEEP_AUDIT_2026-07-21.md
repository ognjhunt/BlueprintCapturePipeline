# Pipeline Repo Deep Audit — 2026-07-21

Status: point-in-time audit (actionable backlog). Supersedes nothing; complements
`PIPELINE_CURRENT_PROCESS_AUDIT_2026-06-06.md` (now stale) and the July launch ledgers.

Goal: make this repo navigable and trustworthy for AI agents and human engineers at a
Series-A bar — remove what successive pivots left behind, consolidate what drifted apart,
and name the gaps in the product core. Every removal below states **why** and the evidence.

## Post-audit revalidation and remediation status

The two-commit PR #149 candidate was revalidated against code, tests, command strings,
current runbooks, and the live SC3 governed ledger before remediation. That pass corrected
additional false positives: several alleged small orphans and `native_runtime_backend` are
live operator/runtime surfaces; the paid-path giants have substantial hermetic tests but no
GPU-marked live-provider proof; the July launch/beta audit trees, the SC3 goal, and the KYC
decision must remain at their canonical paths while current contracts reference them.

The remediation branch has removed the safe ~12.6k-LOC dead-code set, migrated the live
model-volume watchdog helpers, gated legacy lanes, deduplicated evaluation prep, made agent
review opt-in, neutralized the Task Evaluation adapter vocabulary and visual-augmentation
registry, removed model-specific exporter execution from evaluation prep, and moved the
retired Cosmos-Predict2.5 `run_e2e` path behind an explicitly admitted lazy compatibility
adapter. It also wires the model-neutral evaluator workflow into Task Evaluation jobs as an
optional committed support artifact, adds typed sellable-artifact contracts and strict
entrypoint settings, writes per-run timing/spend summaries, documents a production profile,
and adds one CPU end-to-end regression from capture through cards, WebApp projection, and a
signed/verified Post-Training Data Package archive.

Verification for remediation commit `5bfc2091`: the complete default pytest lane
passed with **4,656 passed, 1 skipped, and 1,595 deselected**. The previously reported
failures were reproduced as host disk-exhaustion effects, then their affected storage,
materialization, package, product-spine, local-bundle, and governed-ledger modules were
rerun successfully before the clean full-suite result.

After that verification, `origin/main` advanced through #150/#151. The remediation
branch was rebased onto `894583dd`; the NVIDIA/SIGGRAPH integration makes
`splat_backends` and its `isaac_nurec_export` adapter live, so both modules and their
direct tests were restored. This is a post-audit reachability change, not evidence that
the original graph scan was fabricated. The safe removal estimate and P0 ledger below
have been reduced accordingly. Post-rebase validation is recorded separately from the
pre-rebase full-suite result.

Post-rebase validation on the reconciled tree includes: 64 NVIDIA/SIGGRAPH integration
and restored splat/NuRec tests, 6 additional SimReady asset tests, 84 selected core
overlap tests (artifact/evaluation contracts, `run_e2e`, simulation automation, and the
CPU product spine), all 198 slow robot-eval orchestrator tests, and all 12 governed SC3
ledger tests. All passed. These are hermetic results; they do not claim live GPU/provider
execution, ranking fidelity, deployment, or paid-resource proof.

The first P2 seam slice now puts the hosted native runtime behind a typed strategy
catalog. `site_splat` is the provider-neutral default; the legacy Cosmos-Predict2.5
adapter requires explicit `cosmos_wam` selection, conflicting old/new settings fail
closed, and the runtime no longer discovers a model checkout from hard-coded workspace
paths. The stable runtime-service contract is unchanged. Selection/readiness/prewarm,
security, script-pin, and all slow native-service tests pass; extracting the remaining
Cosmos-specific generation helpers from the 3k-LOC store remains a later incremental
split, not a safe deletion.

Live Vast revalidation was authorized with a hard `$15` cap. A read-only, account-wide
inventory query returned HTTP 200 with zero non-terminal instances, so observed spend
remained `$0`. No instance was launched: the allocator correctly requires a clean image
and release-evidence binding for the exact `origin/main` commit, while the retained
qualification image is bound to `fca4712e` plus a non-empty dirty-patch digest and the
newest clean registry image is bound to audit commit `2edb48b`, not current main
`1974536b`. The normal qualification watchdog contract is bounded to at most `$5.50`,
but budget authorization does not override source identity. A clean, immutable
`1974536b` image plus its generated thin-release evidence is the concrete prerequisite
for live GPU startup/teardown proof; startup would still not prove policy quality,
semantic task success, or ranking fidelity.

## Method

Six parallel audit passes over the tree at commit `2edb48b` (#146):

1. AST import-graph reachability over all 461 modules in `src/blueprint_pipeline`,
   seeded from 157 pyproject console scripts plus `main.py`, `scripts/`, `Makefile`,
   `Dockerfile`, `docker-compose.yml`, `.github/workflows`, `functions/`, `ops/`,
   `deploy/`, `infra/`, `tools/` (including `python -m` string references).
2. World-model backend seam audit (WAM/OSCAR/Cosmos/OpenVLA/UniFOLM/splat/Isaac/MuJoCo).
3. GPU/provider infrastructure audit (RunPod/Vast/Lambda/DigitalOcean generations).
4. Campaign one-off module audit (`g1_*`, `groot_oscar_*`, proof/evidence/closure families).
5. Core product path trace (`run_e2e`, storage trigger, cards, packages, eval runs).
6. Periphery audit (docs, scripts, deps, CI, root files, top-level dirs).

Conflicting findings between passes were re-verified by hand (grep) before landing here.

**Evidence caveats that shaped every verdict:**

- Commit `9309f28` (2026-07-15) is a squashed 1,328-file baseline import. Git dates
  before it are meaningless; the ~50 commits since (#82–#147) are the only activity
  signal. ~102 modules touched since 7/15 mark the actively developed surface.
- Import-graph reachability alone produces **false positives**. The post-review pass
  caught additional command-string, runbook, cross-repo-contract, and test-contract
  surfaces; all are explicitly excluded below.

## Headline numbers

| Metric | Value |
|---|---|
| Modules in `src/blueprint_pipeline` | 461 (~440k LOC), flat directory |
| Console-script entry points | 157 (a third of the package is CLI tails) |
| Unreachable from any entry point | 34 modules (~19.5k LOC) |
| Verified-dead removal shortlist | ~12.6k LOC src + associated tests |
| Deprecate-then-remove (documented operator surface) | ~5.5k LOC src |
| Campaign-specific code (`g1_*`/`groot_oscar_*`/`oscar_*`/`wam_*`/`unitree_*`/`isaac_*`/`mujoco_*`/`kitchen_*`) | ~195k LOC / 174 files (~46% of package) |
| Mapping escape hatches in src | 2,463 `Dict[str, Any]` + 3,656 `dict[str, Any]` occurrences |
| Direct env reads in src | 1,144 `os.getenv` + 554 `os.environ.get` occurrences |
| docs/ files | 147 (~55 are dated point-in-time records to archive) |
| scripts/ files | 94 (12 orphaned or stale) |

The single biggest structural fact: the product core (capture → qualification →
cards → packages → eval runs) is roughly **15 modules** out of 461. Everything else is
provider plumbing, campaign machinery, readiness/review layers, or dead pivots.

---

## P0 — Remove: verified dead code (~12.9k LOC src, plus tests)

Each removal row was verified: no importers outside the listed cluster, no console-script
that anything references, no CI workflow, no `python -m` string reference, no
worker-bundle inclusion. Section 5 records a later main-branch reclassification and is
explicitly excluded from removal.

**Command-surface rule (added after review):** "unreachable by import graph" is not
sufficient when a module is an installed console script or a command documented in a
*living* runbook — that is operator surface, not dead code. Rows below that carry a
console script state so explicitly; the removal PR must delete the pyproject entry,
update `docs/source_governance_policy.json`, and fix any runbook rows in the same
change. Modules whose commands living docs still instruct operators to run are listed
separately under "Deprecate-then-remove", not here.

### 1. `robot_eval_team_closure` cluster — 2,343 LOC, zero tests

`robot_eval_team_closure.py` (1,420), `robot_eval_webapp_projection.py` (635),
`robot_eval_closure_common.py` (288).
**Why:** `robot_eval_closure_common` is imported *only* by the other two, which have
zero importers, zero tests, no console script, no CI/doc references. This is a
completed closure campaign's reporting code. (The similarly named
`live_robot_eval_closure` and `robot_eval_closure_decisions` are live — imported by
`robot_eval_job_orchestrator` — and stay.)
The adjacent Blueprint-WebApp and BlueprintCapture checkouts were also searched for
the three module names before removal; neither contains an invocation.

### 2. `gpu_campaign_state_machine` + `gpu_campaign_provider_adapters` — 1,221 LOC

**Why:** zero src consumers, zero script/CI/doc references, test-only. Functionally
superseded by `production_gpu_campaign_control_plane` + `production_gpu_campaign_budget`
+ `paid_resource_allocator` — the current campaign control generation.

### 3. `real_policy`/`lerobot` policy-family cluster — 3,264 LOC

`real_policy_family_eval_harness.py` (852), `real_policy_closed_loop_rollout.py`,
`lerobot_policy_family.py`, `lerobot_torch_policy_adapter.py`.
**Why:** closed dead cluster — the members import only each other; the root
(`real_policy_family_eval_harness`) is referenced only by its own test. The live
LeRobot code path (`lerobot_episode_export`, `lerobot_export_validation`, used by
`post_training_data_package`) is unrelated and stays.

### 4. Splat/InteriorGS dead cluster — 2,329 LOC

`interiorgs_task_preflight.py` (970), `splat_scene_bootstrap.py`, `splat_occupancy.py`,
`splat_depth.py` (163), `splat_robot_composite.py` (115).
**Why:** closed cluster reachable from nothing; test-only. The *live* splat surface
(`gaussian_splat_decode`, `splat_scene_analysis`, `splat_scene_render`,
`synthesis/depth_splat`, `tools/splat_render`) is separate and stays.

### 5. Reclassified after #150: `splat_backends` + `isaac_nurec_export` — keep

At the audited `2edb48b` snapshot, the registry was test-only and the export adapter was
reachable only through it. Main commit `449a66bc` (#150) extended the registry for the
NVIDIA/SIGGRAPH integration and made the 3dgrut/NuRec export adapter part of that live
swappable-backend surface. The remediation rebase therefore retained both modules and
restored their direct tests. Re-evaluate reachability only if the NVIDIA integration is
later removed; they are not deletion candidates now.

### 6. `realistic_readiness_rehearsal` — 1,873 LOC

**Why:** zero importers (verified by hand — one audit pass initially called it an
"active eval lane" because it *imports* live modules, but nothing imports or invokes
*it*). Readiness-rehearsal one-off; readiness is doctrinally secondary anyway.

### 7. Small verified orphans — ~1,200 LOC

| Module | LOC | Why dead |
|---|---|---|
| `synthetic_2d_wam_seed.py` | 467 | test-only; abandoned 2D-seed WAM experiment |
| `reference_image_utils.py` | 534 | test-only, zero importers |
| `oscar_isaac_closed_loop_gpu_launch.py` | 170 | zero importers (superseded by the carrier/allocator launch path) |

Four modules in the original small-orphan list were false positives and are now in
the do-not-remove ledger: the GEAR-SONIC container smoke, the SC3 action-control
contract, the GR00T+OSCAR cached-footprint audit, and the strict scorer service client.

### 8. `groot_oscar_runpod_model_volume` disabled stub — ~250 of 466 LOC

**Why:** the module docstring says it is the "disabled legacy GPU preparation seam";
`run_model_volume` unconditionally fails with
`legacy_gpu_model_volume_preparation_disabled_use_storage_only_allocator`. The current
implementation is `groot_oscar_runpod_storage_volume` (#145). **But** its deadline
watchdog + compat-admission helpers are imported by `runpod_preflight`,
`runpod_storage_volume`, and `paid_provider_lane_lease` — move those helpers into
`groot_oscar_runpod_storage_volume` (or a watchdog module), then delete the stub.

### 9. Non-src removals

- **`agent_skills/` (8 files)** — its own README says "legacy drafting notes";
  canonical skill sources live in `skillpacks/` (loaded by `agent_runtime/skill_sync.py`).
  Only reference is one dated June audit doc. Delete.
- **8 orphaned scripts** (zero references anywhere; all thin wrappers over live
  modules): `audit_production_gpu_startup.py`, `build_production_handoff_readiness.py`,
  `prepare_runpod_serverless_kitchen_campaign.py`, `render_official_g1_policy_trace.py`,
  `rotate_gpu_provider_keys.py`, `single_g1_kitchen_runtime_patch.py`,
  `prepare_groot_oscar_model_cache.sh`, `setup_cuda.sh`. Delete or wire into a runbook —
  an entry point nothing references is dead weight either way. (If any are genuinely
  used operator commands, the fix is one line in the relevant runbook.)
- **3 stale scripts referenced only by dated docs**: `backfill_site_reference_database.py`
  (2026-05-16 audit only), `check_cosmos3_readiness.py` (June feasibility docs only),
  `prepare_strict_g1_kitchen_bundle.py`. Keep `rebind_quality_gap_ledger_digests.py`
  until the SC3 launch ledger closes — it maintains the live ledger. (An earlier draft
  listed `build_launch_readiness_packet.py` here — wrong: it is documented in the
  living `docs/architecture/command-safety-matrix.md` as the canonical local launch
  evidence packet and has a real test suite. It stays.)

---

## P1 — Deprecate-then-remove: dead by import graph, but documented operator surface

These were in the P0 list until review caught that living docs still instruct
operators to run them. They are still pivot leftovers with no live execution path,
but removing them is a two-step: decide/announce the deprecation, update the runbook
and command surface (pyproject entry, `source_governance_policy.json`), then delete.

1. **Lambda Labs failover chain — ~2,530 LOC incl. tests.**
   `lambda_provider_adapter.py` (1,729) + `robot_eval_provider_race_launcher.py`.
   Import-dead, but *not* reference-dead: `robot_eval_job_orchestrator.py` maps
   `lambda_cloud` → `blueprint-run-lambda-provider-adapter` and emits
   `blueprint-run-robot-eval-provider-race` into `gpu_provider_race_handoff.json` as
   the customer robot-eval failover path, and `docs/LIVE_PIPELINE_SETUP.md` documents
   both (failover run + Lambda teardown loop). Lambda is absent from the
   `gpu_render_providers` registry (runpod/vast/digitalocean/gcp/aws), so the product
   decision to drop it appears half-made. Finish it: remove the `lambda_cloud` mapping
   and race-handoff emission from the orchestrator, update LIVE_PIPELINE_SETUP.md,
   drop the two console scripts, then delete both modules — or consciously keep the
   failover lane and say so.
2. **First-GPU sample-video trio — 1,354 LOC.**
   `first_gpu_sample_video_stage.py` (795), `first_gpu_sample_video_preflight.py`
   (321), `first_gpu_candidate_audit.py` (238). Zero importers, but their commands are
   documented steps in `docs/FIRST_GPU_E2E_RUNBOOK.md` (still maintained — updated by
   #144) and rows in the command-safety matrix. Either the runbook's sample-video
   staging steps are still how operators seed a first capture (keep all three), or the
   runbook section is retired along with the modules in one change.
3. **UniFOLM infrastructure one-offs — 1,648 LOC.**
   `unitree_unifolm_gpu_image.py` (851), `unitree_unifolm_runpod_server.py` (797).
   Never-run backend infra with zero development since the baseline, but both are
   installed CLIs (`blueprint-build-unitree-unifolm-gpu-image`,
   `blueprint-launch-unitree-unifolm-runpod-server`) with CHANGELOG entries. Keep the
   small UniFOLM *policy command contract* modules (cataloged candidate in
   `docs/UNITREE_G1_POLICY_ENDPOINT_LANE.md`) so the candidate stays cheap to revive;
   deprecate and delete the infra shells + their console scripts.

## P0 — Doctrine contradictions in docs (small, high-leverage)

The current doctrine (PLATFORM_CONTEXT.md, 2026-07-15) is explicit: capture-first;
Task Evaluation Runs and Post-Training Data Packages are the sellable products; world
models are swappable internal support, **not** the product. Three places still say the
opposite — exactly the kind of stale doctrine that misleads agents:

1. `docs/superpowers/specs/2026-03-28-autonomous-org-design.md` — states
   "world-model-product-first" and site-specific world-model products as Pipeline
   output; also targets a `.paperclip.yaml` that doesn't exist here. Delete or archive
   with a superseded banner.
2. `docs/superpowers/plans/2026-03-28-autonomous-org-implementation.md` — same
   pre-pivot framing. Same treatment.
3. **`AUTONOMOUS_ORG.md` (root)** — one bullet still reads "world-model-product-first",
   contradicting PLATFORM_CONTEXT in a file agents are told to treat as an org anchor.
   Fix the bullet in place.

Also: `docs/IOS_SITE_GROUNDED_WORLD_MODEL_SPEC.md` — the body was already reframed to
the capture-first "Data Package Support Capture Spec", but the filename still says
world-model. Rename.

## P1 — Docs and scripts hygiene

- **Create `docs/archive/` and move the verified historical subset (~28 records)**:
  the June audits/spikes (arena proof-boundary, live-webapp forwarding, e2e-gpu
  readiness gap, lucky-engine spike, mujoco-live-product-path, COSMOS3 feasibility,
  PIPELINE_CURRENT_PROCESS_AUDIT, SITE_REFERENCE_GROUNDING, cpu-work-audit,
  last-24h launch audit), the G1 kitchen deep-audit pair (2026-07-10), the seven July
  handoff/spec snapshots (`fable-*`, `groot-oscar-release-reliability-hardening`,
  `isaac-startup-reliability-*`, `kitchen-random-task-e2e-cloud-handoff`), five
  completed June `goals/` records, and the three implemented June `superpowers/`
  designs. **Do not archive yet:** the launch-audit and beta-audit trees plus the
  100-beta audit are digest-bound by the current SC3 ledger; the 2026-07-02 SC3 goal is
  linked by the current protocol; and the KYC decision is referenced as current by the
  paid-marketplace gate and operator-evidence example. Move those only after their
  governing consumers are deliberately retired or migrated.
- **Keep** the dated-filename docs that are actually script-enforced contracts
  (BETA_CAPACITY/RETENTION/RESIDENCY, CAPTURE_TRUTH_BACKUP_DR + JSON twins) — the date
  is a version stamp, and `scripts/validate_beta_capacity_storage.py` etc. consume them.
- `docs/GPU_VM_RUNBOOK.md` self-labels legacy but is still linked from README — move to
  archive, fix the link.
- **Makefile**: `make test` help text claims "full CPU test suite" but pyproject
  `addopts` deselect slow/gpu — it's the fast lane. Fix the text (the full lane is
  `scripts/pytest_full.sh`). Makefile references zero scripts; either make it the
  command surface or delete it in favor of README/CLAUDE.md.
- **CLAUDE.md**: the "Read first" and evidence-checklist paths use
  `$HOME/workspace/...` which doesn't resolve in remote/CI environments (repo lives
  elsewhere). Use repo-relative paths for in-repo docs and mark cross-repo paths as
  environment-dependent.
- **Deps** (hygiene is otherwise good — exports are uv-frozen and CI-verified):
  pin `functions/requirements.txt` (currently range-based, outside the frozen graph);
  drop or annotate `torchvision` (zero direct imports; only transitively needed) and
  `pycollada` (zero direct imports); document `websockets` as uvicorn's WS backend.
- `docker-compose.yml` mounts `./data` and `./outputs` while the code writes `output/`
  — align the naming.

## P1 — Core-path defects (the product spine)

The spine (storage_trigger → `capture_orchestrator` → qualification →
evaluation_prep → cards → packages) is coherent and evidence-disciplined — schema
versions, claim boundaries, fail-closed rights/privacy gates (PIPE-01 verified real at
`evaluation_prep_stage.py:3817`). Defects:

1. **`evaluation_prep` runs twice in default `run_e2e`** — once inside lane
   `current` (stage 3) and again as stage 5 (CLI default `--run-evaluation-prep=True`).
   Idempotent but wasteful and confusing; make stage 5 a no-op when lane `current`
   already ran it.
2. **`agent_review` is a mandatory, non-skippable stage** of the operator CLI (stage 4,
   before evaluation_prep), and `qualification` writes alpha-readiness summaries,
   launch bundles, and buyer trust scores inline. Doctrine says readiness/review is
   secondary to the product core; the production storage-trigger path already skips
   agent review — give the CLI the same default and move readiness emission behind an
   explicit flag/edge.
3. **The primary sellable product routes through a function named "legacy"**:
   run_e2e's only Task-Eval entry is
   `robot_eval_evaluation_run_adapter.execute_legacy_robot_eval_request_as_evaluation_run`.
   Rename the adapter and its entry to the evaluation_run vocabulary — names are
   navigation for agents.
4. **Dataset manifest compatibility alias needs an expiry.** The canonical
   `robot_eval_dataset_manifest.json` is already accompanied by the requested
   one-release `real_site_robot_eval_dataset_manifest.json` legacy alias. Record the
   removal release and then delete the alias; this is no longer an unfixed duplicate
   writer design.
5. **Legacy lanes still routable** in `capture_orchestrator` (`scene_memory`,
   `retrieval_index`, `frame_alignment`, `synthesis_coverage_validation`,
   `cosmos_single_capture_smoke`). Gate them behind an explicit legacy flag or remove
   the routing.

## P1 — Swappable-backend seam violations

The world-model seam is real and live (substrate registry `wam_eval_substrate` →
env-command process boundary `wam_provider_runtime` → strategy catalog
`wam_backend_strategy` → executor registry `evaluation_run_execution`). Registered and
selectable: fixture, OSCAR, Cosmos3 (candidate), MuJoCo, Isaac, pybullet. Violations of
"keep world-model backends swappable":

1. `run_e2e.py:32,728` hard-imports the legacy Cosmos-Predict2.5 lane as a named
   pipeline stage (`cosmos_validation`) instead of routing through the seam. The
   catalog itself marks Cosmos-Predict2.5 "no longer under active development" —
   retire the stage or move it behind the substrate registry.
2. `evaluation_prep_stage.py:4254` hard-wires `synthesis/cosmos_training_export` into
   the Post-Training Data Package prep path (model-family-specific export inside the
   product core).
3. `native_runtime_backend.py` (2,970 LOC) hard-codes Cosmos-Predict2.5 repo paths
   (`/root/workspace/cosmos-predict2.5`) and a binary `cosmos_i2w | splat_only` split,
   bypassing the strategy catalog entirely. It is not unwired: `native_runtime_service`
   imports it, `scripts/start_native_runtime_vast.sh` starts it, and the command-safety
   matrix classifies that startup as live/runtime risk. Rewrite and split it behind the
   backend seam while preserving the hosted-runtime contract; deletion would break a
   documented operator surface.
4. `robot_eval_job_orchestrator.py:7973` auto-defaults the `isaac_sim` simulator
   command to a specific 5k-LOC G1/3DGS module in the core job path. Require the
   command through the existing simulator-command configuration instead of selecting
   a model-specific module inside the orchestrator.
5. `post_training_data_package.py:5507` references the OSCAR-specific
   `oscar_visual_augmentation_packet/model_backend_registry.json` filename in the
   package builder (mitigated by the packet emitting a per-run registry — rename to a
   neutral artifact name).
6. **Corrected on revalidation — Cosmos3 preference is not permanently asserted or
   impossible to activate.** The first-party scorer modules do not exist, but
   `wam_backend_strategy` also recognizes the live
   `wam_strict_action_consistency_scorer_client` when its external scorer URL is
   configured. Adapter/runtime, scorer, calibration-anchor, and run-gate checks derive
   the state; without all of them the catalog reports `aspirational`, not preferred.
   Tests cover both the default aspirational state and successful activation through
   an explicit scorer or configured external service. Keep the missing first-party
   scorer as a capability gap, not a false catalog-state bug.
7. The new **model-neutral evaluator layer** (#140–#143: `evaluator_evidence_profiles`,
   `evaluator_runtime_evidence`, `evaluator_qualification_workflow`,
   `policy_evaluation_contracts`, `decision_grade_ranking`) is actively developed but
   **not yet consumed by the orchestrator** — currently an unreachable island (it shows
   up in the dead-code scan). Wire it in or it becomes next quarter's dead cluster.

## P2 — Consolidations (duplication from pivots)

1. **WAM async runners**: `runpod_wam_async_runner.py` (3,717) vs
   `vast_wam_async_runner.py` (1,949) — parallel per-provider implementations of the
   same launch/poll/collect lifecycle behind one facade (`wam_compute_providers`,
   1,903), with cross-entanglement (the RunPod runner imports Vast modules). Extract
   the shared lifecycle against the provider-adapter contracts; ~1,500 LOC reduction.
   Note `wam_compute_providers` itself overlaps `wam_provider_runtime` (its design doc
   is a June superpowers spec) — decide which is the seam and fold the other in.
   **Started:** JSON-state loading, signed-URL file handling, and URL redaction now
   live in provider-neutral `wam_async_runner_common`; both runners retain their public
   and private call surfaces. Provider artifact transfer is shared as well; all 70
   async-runner tests pass. Poll, collection, and teardown lifecycle extraction remains.
2. **Single-consumer satellites**: fold `runpod_wam_launch_contract.py` (461) into its
   sole consumer; fold `vast_authorized_probe_runner.py` (548) into
   `vast_wam_authorized_runner`.
3. **Monolith splits** (all live, all too big to navigate):
   `robot_eval_job_orchestrator.py` (10,440), `unitree_groot_n17_sonic_vast_persistent_session.py`
   (10,101), `post_training_data_package.py` (6,678),
   `vast_provider_adapter.py` (6,739), `qualification.py` (5,560),
   `evaluation_prep_stage.py` (5,141), `single_g1_kitchen_episode_runpod.py` (4,341),
   `single_g1_kitchen_qualification_session.py` (4,248),
   `robot_eval_dataset.py` (4,959), `wam_fixture_evaluator.py` (4,605).
4. **`mujoco_g1_wam_vla_policy_endpoint_eval.py` (12,297 LOC)** — largest module in the
   repo, reachable only via its own console script + tests, zero importers. Decide:
   if the MuJoCo policy-endpoint eval lane is a current product lane, split and test
   it; if it was a one-time investigation, archive it. Do not leave a 12k-LOC
   single-file CLI in the flat namespace.
5. **Do NOT consolidate the four groot_oscar RunPod lanes** (serverless /
   persistent-carrier / storage-volume / carrier-volume). A previous hypothesis called
   them "generations"; the audit shows they are distinct *admitted spend lanes* under
   the canonical `paid_resource_allocator`, each with its own probe-kind. The only true
   dead generation there is the `runpod_model_volume` stub (P0 #8).

## P2 — Structure: from flat 461 to navigable subpackages

A flat directory where the product core is 15 files out of 461 is the single largest
navigability cost. Target structure (grounded in the actual run_e2e trace):

```
blueprint_pipeline/
  core/          common, safe_env, logging_utils, optional_dependencies,
                 security_controls, output_run_transaction, lane_resume
  capture/       local_capture, preflight_capture, materialization, capture_bridge,
                 ios_manifest, capture_orientation, temporal_alignment, source_metadata
  rights/        privacy_*, consent_*, proof_contracts, launch_provenance,
                 same_capture_lineage, secret_artifact_policy, success_claim_contracts
  geometry/      geometry_*, gaussian_splat_decode, splat_scene_*, object_index_*,
                 scene_semantics, scene_placement/, simready + importer lanes
  qualification/ qualification (split), canonical_site_package, launch_bundle
  cards/         robot_eval_dataset (split into site/task/scenario/eval cards),
                 eval_card_ids, scenario_variation_instantiator, scene_eval_autogen
  eval_runs/     evaluation_run*, robot_eval_* contracts/execution/adapters,
                 episode_spec, task_eval_run_report, arena_*
  packaging/     post_training_data_package (split), clip_curation_stage,
                 semantic_dedup_stage, lerobot_*, buyer_package_readout, rl_handoff
  hosted/        runtime_service_app, native_runtime_*, site_world_runtime_service_client,
                 video_to_world_*, live_pipeline_*
  providers/     gpu_render_providers, cloud_vm_render_providers, runpod_*, vast_*,
                 paid_*, production_gpu_*, spend guards
  policies/      policy adapters/runtimes (openvla, unitree_*, groot sonic, oscar_*
                 command adapters), wam_* seam + backends
  campaigns/     g1_*, groot_oscar_*, gear_sonic_*, kitchen_*, single_g1_* — quarantined
  readiness/     alpha_readiness, agent_runtime/, evaluator_*, *_readiness, launch gates
  webapp/        webapp_sync, status projections
  entrypoints/   run_e2e, capture_orchestrator, storage-trigger glue
```

Enforce direction mechanically with an import-linter contract: `core/capture/rights/
cards/packaging/eval_runs` must never import `readiness/`, `campaigns/`, or
`providers/` directly (registry/plugin seams instead — one already half-exists in
`capture_orchestrator`'s simulator plugin registry). Today the CLAUDE.md rule is
prose-only; make CI enforce it.

**Started:** the sellable artifact spine now has an AST-enforced dependency contract
that rejects direct campaign, readiness, and paid-provider imports. The first real
violation was removed by extracting G1 closure projection into a neutral aggregate-
closure contract; campaign modules now depend inward on that contract, while buyer
package generation no longer imports G1 campaign code. Physical subpackage moves and
broader layer coverage remain incremental follow-ups.

Do this **after** the P0 removals (moving dead code is wasted work) and incrementally —
one subpackage per PR with import shims, not a big-bang rename.

Related: 157 console scripts is its own navigability problem. After removals, collapse
the long tail of one-off CLIs into grouped `blueprint <family> <command>` entry points
or drop scripts nothing references.

## P2 — Series-A gaps (what's missing, not what's extra)

1. **Typed artifact contracts.** 2,463 `Dict[str, Any]` and 3,656 built-in
   `dict[str, Any]` occurrences; card/package/manifest schemas
   (`*.v0.1`/`v1`) exist as strings, not as schema definitions — a corrupted upstream
   artifact silently degrades downstream because everything reads
   `_read_optional_mapping(...) or {}`. Start with the sellable boundaries: pydantic
   (already used in `runtime_service_app`) or dataclass models + JSON Schema for
   site/task/scenario/eval cards, `evaluation_run.v1`, and the package export manifest,
   validated at stage boundaries.
2. **Central config.** 1,698 direct env reads (`os.getenv` + `os.environ.get`) in src;
   behavior flags are
   env-truthy strings; the one `PipelineConfig` dataclass carries only `gcs_root`.
   Introduce a typed settings object loaded once at each entry point.
3. **One true end-to-end test.** 489 Python files under `tests/` (~277k LOC) but they're per-module
   contract tests against the committed fixture; there is no single CI assertion
   "capture bundle in → cards + package tar.gz + webapp projection out". Add it (can
   run against `kitchen_task_min`); it becomes the regression net for all later
   restructuring. Also: the `gpu` pytest marker exists but zero tests carry it.
4. **No GPU-marked/live-provider integration lane for the paid path.** The earlier
   claim that the live paid giants had zero tests was wrong. Dedicated CPU/unit suites
   exist and are substantial: 80 tests each for the Vast persistent session and Vast
   adapter, 30 for the RunPod kitchen episode, 37 for kitchen qualification, and 68
   across ten `g1_microwave_*` test files. The actual gap is that the `gpu` marker has
   zero uses, so these suites do not prove real provider mutation, GPU runtime startup,
   artifact collection, or teardown against live infrastructure.
5. **Error semantics.** Distinguishing "artifact absent by design" from "failed" relies
   on convention (`status: not_requested/failed_closed`) with no shared result type.
   A small shared enum/result helper in `core/` would make fail-closed checks uniform.
6. **Observability.** Structured `log_event` exists in orchestrators only; no metrics,
   no run-duration/cost aggregation beyond ad-hoc spend ledgers; stage ledgers are
   per-capture JSON with no fleet view. Minimum: a run-summary artifact per pipeline
   invocation (stage timings, spend, outcome) and one place that aggregates them.
7. **A bare deploy produces contracts, not product.** Defaults are fixtures by design
   (`provisioner="fixture_local"`, `simulator="fixture"`, privacy disabled by default,
   hosted-session artifacts contract-only, delivery upload off). The claim-boundary
   honesty is good; what's missing is a single documented "production profile" (env +
   flags) that turns on the real path, so the difference between demo and production is
   one profile, not folklore.

---

## Do NOT remove — false positives and load-bearing surprises

Recorded so the next audit (or an eager agent) doesn't re-flag them:

- **`splat_backends.py` + `isaac_nurec_export.py`** — #150 extended the registry
  for the NVIDIA/SIGGRAPH integration; its 3dgrut exporter imports the NuRec adapter.
  Both direct test modules are restored and retained after the remediation rebase.
- **`frames_layout.py`** (212) — looks orphaned (only a comment in
  `materialization.py` mentions it), but it is the mandated cross-repo bundle-reader
  contract: `docs/CAPTURE_BRIDGE_CONTRACT.md` requires all frame readers to go through
  `blueprint_pipeline.frames_layout` (v1/v2 layouts, fail-closed on unknown packing),
  and the BlueprintCapture producer rollout of packed v2 frames is gated on this
  reader being deployed. Import graphs cannot see cross-repo contracts.
- **`cross_repo_first_gpu_readiness.py`** — another import-graph false positive. Its
  console command is documented in the root README, the current
  `FIRST_GPU_E2E_RUNBOOK.md`, and the command-safety matrix; it remains the read-only
  pre-spend audit that forbids GPU allocation while cross-repo evidence is incomplete.
  The first successful manipulation run is not yet proven, so the milestone has not
  concluded in the sense required to retire this guard.
- **`gear_sonic_container_smoke.py`** — documented as the container-side FABLE-004
  FK smoke and covered by `tests/test_gear_sonic_container_smoke.py`; absence from the
  import graph is expected for a `python -m` container command.
- **`oscar_action_control_contracts.py`** — authoritative evidence in the still-live
  SC3 quality-gap ledger; `scripts/rebind_quality_gap_ledger_digests.py` and the ledger's
  required test command name both the module and its dedicated test.
- **`groot_oscar_cached_footprint.py`** — a current operator command in
  `docs/runbooks/groot-oscar-thin-release.md`, used to bind image and model-cache disk
  footprint after exact-release verification.
- **`wam_strict_action_consistency_scorer_client.py`** — the configured service bridge
  for the DigitalOcean closed-loop job (`python -m` string reference), with dedicated
  transport tests. It does not implement a first-party Cosmos3 consistency scorer,
  but it legitimately satisfies the strategy's scorer-availability precondition when
  an external scorer URL is configured, and must remain.
- **`scripts/build_launch_readiness_packet.py`** — documented in the living
  command-safety matrix as the canonical local launch evidence packet, with a real
  test suite (`tests/test_launch_readiness_packet.py`).
- **`g1_microwave_finetune_worker.py`** (1,317) — looks orphaned to the import graph,
  but `g1_microwave_finetune_provider_bundle.py:31,95` bundles it **by filename** and
  ships it for execution on remote GPU providers.
- **`production_gpu_image_contract` / `release_candidate` / `reliability_qualification`
  / `runpod_autoscaler`** — runtime-unreachable, but their tests are the deliberate CI
  contract run by `.github/workflows/production-gpu-reliability.yml`.
- **`isaac_review_renderer_canary`, `isaac_worker_runtime_preflight`,
  `mujoco_worker_runtime_preflight`, `pytest_full_lane_evidence`** — invoked via
  `python -m` strings from worker startup scripts and CI, invisible to import graphs.
- **The four groot_oscar RunPod lanes** — admitted spend lanes, not dead generations
  (see P2 #5).
- **The campaign families** (`groot_oscar_*` ~31 modules, `single_g1_kitchen_*` +
  Vast qualification, `g1_kitchen_*` proof machinery, `g1_microwave_*`, `paid_*`,
  `production_gpu_*`) — 41 of the 50 post-baseline commits develop this lane; the
  G1 kitchen campaign is **not concluded** (no successful end-to-end manipulation run
  yet per the 7/10 deep audit). Cross-campaign imports make even the "proof" modules
  load-bearing (e.g. `buyer_package_readout` imports `g1_kitchen_attempt_closure`).
- **The evaluator qualification cluster** (#140–#143) — unreachable today because it's
  *new*, not because it's dead (see P1 seam violation #7).
- **`first_gpu_run_packet`, `first_gpu_e2e_readiness`, `owner_gpu_proof_runner`** —
  imported by shipped console scripts.
- **Dated-filename policy docs with JSON twins** — script-enforced contracts, the date
  is a version stamp.

## Suggested sequencing

1. **PR 1 — dead code removal** (P0 §1–§9): pure deletions + the model-volume helper
   move, including console-script/governance-ledger entries per the command-surface
   rule; `pytest` fast lane green is the gate. ~12.9k LOC src + tests.
2. **PR 2 — docs/scripts hygiene** (P0 doctrine + P1 hygiene): fix the three doctrine
   contradictions, create `docs/archive/`, delete `agent_skills/` + orphaned scripts,
   Makefile/CLAUDE.md/deps fixes. Fold in the deprecate-then-remove decisions (Lambda
   failover lane, first-GPU sample-video runbook steps, UniFOLM infra CLIs) — each is
   a small product call plus a runbook edit before its deletion lands here or in PR 1.
3. **PR 3 — core-path fixes** (P1): dedupe evaluation_prep, demote agent_review to
   opt-in on the CLI, rename the "legacy" eval-run adapter, single dataset manifest.
4. **PR 4 — seam enforcement** (P1 violations): retire/re-seam the Cosmos-Predict2.5
   stage + `native_runtime_backend`, config-driven simulator default, wire (or
   consciously park) the model-neutral evaluator layer.
5. **PR 5+ — structure** (P2): the end-to-end test first, then one subpackage move per
   PR with the import-linter contract added as each layer lands, then monolith splits
   and the WAM runner consolidation.

Rough net effect of PRs 1–2 alone: ~15k LOC of code and ~60 stale/contradictory docs
out of the tree, zero behavior change, and every remaining module either reachable,
CI-contracted, or explicitly quarantined.
