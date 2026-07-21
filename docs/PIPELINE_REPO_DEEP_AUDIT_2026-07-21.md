# Pipeline Repo Deep Audit — 2026-07-21

Status: point-in-time audit (actionable backlog). Supersedes nothing; complements
`PIPELINE_CURRENT_PROCESS_AUDIT_2026-06-06.md` (now stale) and the July launch ledgers.

Goal: make this repo navigable and trustworthy for AI agents and human engineers at a
Series-A bar — remove what successive pivots left behind, consolidate what drifted apart,
and name the gaps in the product core. Every removal below states **why** and the evidence.

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
- Import-graph reachability alone produces **false positives**. Three were caught and
  are explicitly excluded from removal lists (see "Do NOT remove" below).

## Headline numbers

| Metric | Value |
|---|---|
| Modules in `src/blueprint_pipeline` | 461 (~440k LOC), flat directory |
| Console-script entry points | 157 (a third of the package is CLI tails) |
| Unreachable from any entry point | 34 modules (~19.5k LOC) |
| Verified-dead removal shortlist | ~21.5k LOC src + associated tests |
| Campaign-specific code (`g1_*`/`groot_oscar_*`/`oscar_*`/`wam_*`/`unitree_*`/`isaac_*`/`mujoco_*`/`kitchen_*`) | ~195k LOC / 174 files (~46% of package) |
| `Dict[str, Any]` occurrences | 2,412 |
| `os.getenv` occurrences in business logic | ~1,971 |
| docs/ files | 147 (~55 are dated point-in-time records to archive) |
| scripts/ files | 94 (12 orphaned or stale) |

The single biggest structural fact: the product core (capture → qualification →
cards → packages → eval runs) is roughly **15 modules** out of 461. Everything else is
provider plumbing, campaign machinery, readiness/review layers, or dead pivots.

---

## P0 — Remove: verified dead code (~21.5k LOC src, plus tests)

Each row was verified: no importers outside the listed cluster, no console-script that
anything references, no CI workflow, no `python -m` string reference, no worker-bundle
inclusion. Delete the module(s) and their dedicated test files.

### 1. `robot_eval_team_closure` cluster — 2,343 LOC, zero tests

`robot_eval_team_closure.py` (1,420), `robot_eval_webapp_projection.py` (635),
`robot_eval_closure_common.py` (288).
**Why:** `robot_eval_closure_common` is imported *only* by the other two, which have
zero importers, zero tests, no console script, no CI/doc references. This is a
completed closure campaign's reporting code. (The similarly named
`live_robot_eval_closure` and `robot_eval_closure_decisions` are live — imported by
`robot_eval_job_orchestrator` — and stay.)
*Caveat:* confirm Blueprint-WebApp doesn't shell out to `robot_eval_webapp_projection`
before deleting that one file.

### 2. Lambda Labs provider chain — ~2,530 LOC incl. tests

`lambda_provider_adapter.py` (1,729) + `robot_eval_provider_race_launcher.py` + their
two test files.
**Why:** the race launcher is the adapter's *sole* consumer and itself has zero
consumers outside its own test. Superseded by `robot_eval_provider_launcher` +
the `gpu_render_providers` registry (which covers runpod/vast/digitalocean/gcp/aws —
Lambda was dropped as a provider). The adapter's legacy mutation CLI is already
disabled in-file.

### 3. `gpu_campaign_state_machine` + `gpu_campaign_provider_adapters` — 1,221 LOC

**Why:** zero src consumers, zero script/CI/doc references, test-only. Functionally
superseded by `production_gpu_campaign_control_plane` + `production_gpu_campaign_budget`
+ `paid_resource_allocator` — the current campaign control generation.

### 4. `real_policy`/`lerobot` policy-family cluster — 3,264 LOC

`real_policy_family_eval_harness.py` (852), `real_policy_closed_loop_rollout.py`,
`lerobot_policy_family.py`, `lerobot_torch_policy_adapter.py`.
**Why:** closed dead cluster — the members import only each other; the root
(`real_policy_family_eval_harness`) is referenced only by its own test. The live
LeRobot code path (`lerobot_episode_export`, `lerobot_export_validation`, used by
`post_training_data_package`) is unrelated and stays.

### 5. Splat/InteriorGS dead cluster — 2,329 LOC

`interiorgs_task_preflight.py` (970), `splat_scene_bootstrap.py`, `splat_occupancy.py`,
`splat_depth.py` (163), `splat_robot_composite.py` (115).
**Why:** closed cluster reachable from nothing; test-only. The *live* splat surface
(`gaussian_splat_decode`, `splat_scene_analysis`, `splat_scene_render`,
`synthesis/depth_splat`, `tools/splat_render`) is separate and stays.

### 6. `splat_backends` registry + `isaac_nurec_export` — 330 LOC

**Why:** `splat_backends.py` is a backend *registry* (splat_transform, spark,
threedgrut, particlefield_usd, isaac_nurec, artifixer) that nothing imports except its
own tests — a seam that was built and never wired. Either wire it into the live splat
render path deliberately, or delete it; a dead registry masquerading as a seam is worse
than no registry. Default: delete (the live path selects renderers without it).

### 7. First-GPU milestone leftovers — 4,040 LOC

`cross_repo_first_gpu_readiness.py` (2,686), `first_gpu_sample_video_stage.py` (795),
`first_gpu_sample_video_preflight.py` (321), `first_gpu_candidate_audit.py` (238).
**Why:** the "first GPU run" milestone concluded and was superseded by the
`production_gpu_*` reliability program; these four have zero consumers (the sample-video
chain only imports intra-cluster). **Keep** `first_gpu_run_packet` and
`first_gpu_e2e_readiness` — imported by the `production_handoff_readiness` console
script — and `owner_gpu_proof_runner` (imported by a shipped autoresearch evaluator).

### 8. `realistic_readiness_rehearsal` — 1,873 LOC

**Why:** zero importers (verified by hand — one audit pass initially called it an
"active eval lane" because it *imports* live modules, but nothing imports or invokes
*it*). Readiness-rehearsal one-off; readiness is doctrinally secondary anyway.

### 9. Small verified orphans — ~1,800 LOC

| Module | LOC | Why dead |
|---|---|---|
| `synthetic_2d_wam_seed.py` | 467 | test-only; abandoned 2D-seed WAM experiment |
| `reference_image_utils.py` | 534 | test-only, zero importers |
| `frames_layout.py` | 212 | unreachable; only a comment in `materialization.py` mentions it |
| `oscar_isaac_closed_loop_gpu_launch.py` | 170 | zero importers (superseded by the carrier/allocator launch path) |
| `gear_sonic_container_smoke.py` | 120 | zero refs, no console script |
| `oscar_action_control_contracts.py` | 109 | test-only |
| `groot_oscar_cached_footprint.py` | 109 | test-only; only vestigial member of the otherwise-live groot_oscar family |
| `wam_strict_action_consistency_scorer_client.py` | 88 | client for a scorer module that does not exist in the repo |

### 10. UniFOLM infrastructure one-offs — 1,648 LOC

`unitree_unifolm_gpu_image.py` (851), `unitree_unifolm_runpod_server.py` (797).
**Why:** per-provider serve one-offs for a backend that has never been run and has had
zero development since the baseline. Keep the small UniFOLM *policy command contract*
modules (cataloged candidate in `docs/UNITREE_G1_POLICY_ENDPOINT_LANE.md`) so the
candidate remains cheap to revive; delete the infra shells.

### 11. `groot_oscar_runpod_model_volume` disabled stub — ~250 of 466 LOC

**Why:** the module docstring says it is the "disabled legacy GPU preparation seam";
`run_model_volume` unconditionally fails with
`legacy_gpu_model_volume_preparation_disabled_use_storage_only_allocator`. The current
implementation is `groot_oscar_runpod_storage_volume` (#145). **But** its deadline
watchdog + compat-admission helpers are imported by `runpod_preflight`,
`runpod_storage_volume`, and `paid_provider_lane_lease` — move those helpers into
`groot_oscar_runpod_storage_volume` (or a watchdog module), then delete the stub.

### 12. Non-src removals

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
  `prepare_strict_g1_kitchen_bundle.py`, `build_launch_readiness_packet.py` (2026-07-09
  audit only). Keep `rebind_quality_gap_ledger_digests.py` until the SC3 launch ledger
  closes — it maintains the live ledger.

---

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

- **Create `docs/archive/` and move ~55 dated point-in-time records**: the
  already-SUPERSEDED-bannered trees (`specs/launch-audit-2026-07-02/` — 14 files,
  `beta-launch-audit-2026-07-03/`, `100_BETA_TESTER_LAUNCH_BLOCKER_AUDIT_2026-07-06.md`),
  the June audits/spikes (arena proof-boundary, live-webapp forwarding, e2e-gpu
  readiness gap, lucky-engine spike, mujoco-live-product-path, COSMOS3 feasibility,
  PIPELINE_CURRENT_PROCESS_AUDIT, SITE_REFERENCE_GROUNDING, cpu-work-audit,
  last-24h launch audit), the G1 kitchen deep-audit pair (2026-07-10), the seven July
  handoff/spec snapshots (`fable-*`, `groot-oscar-release-reliability-hardening`,
  `isaac-startup-reliability-*`, `kitchen-random-task-e2e-cloud-handoff`), all six
  `goals/` records, the KYC decision record, and the four implemented `superpowers/`
  June designs. **Exception:** the SC3 quality-gap ledger set (2026-07-09) is the
  currently authoritative launch ledger — archive only when launch closes.
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
3. **The primary sellable product routes through a module named "legacy"**:
   run_e2e's only Task-Eval entry is
   `robot_eval_evaluation_run_adapter.execute_legacy_robot_eval_request_as_evaluation_run`.
   Rename the adapter and its entry to the evaluation_run vocabulary — names are
   navigation for agents.
4. **Duplicate dataset manifest filenames** maintained in parallel
   (`robot_eval_dataset_manifest.json` + `real_site_robot_eval_dataset_manifest.json`).
   Pick one, alias the other for one release, delete.
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
   bypassing the strategy catalog entirely — and is nearly unwired (no deploy target,
   one env example). Retire or rewrite against the seam; as-is it's ~3k LOC of
   backend-specific hosted-runtime code nothing deploys.
4. `robot_eval_job_orchestrator.py:7973` auto-defaults the `isaac_sim` simulator
   command to a specific 5k-LOC G1/3DGS module in the core job path — make the default
   a config value, not a hard-coded module name.
5. `post_training_data_package.py:5507` references the OSCAR-specific
   `oscar_visual_augmentation_packet/model_backend_registry.json` filename in the
   package builder (mitigated by the packet emitting a per-run registry — rename to a
   neutral artifact name).
6. **Cosmos3 "preferred candidate" status can never activate**: `wam_backend_strategy`
   derives it mechanically, but the required scorer modules
   (`sc3_consistency_scorer`/`wam_consistency_scorer`) don't exist in the repo. Either
   build the scorers or stop cataloging cosmos3 as preferred — a permanently
   aspirational preferred backend is misleading state.
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
2. **Single-consumer satellites**: fold `runpod_wam_launch_contract.py` (461) into its
   sole consumer; fold `vast_authorized_probe_runner.py` (548) into
   `vast_wam_authorized_runner`.
3. **Monolith splits** (all live, all too big to navigate):
   `robot_eval_job_orchestrator.py` (10,440), `unitree_groot_n17_sonic_vast_persistent_session.py`
   (10,101, **zero tests**), `post_training_data_package.py` (6,678),
   `vast_provider_adapter.py` (6,739, zero tests), `qualification.py` (5,560),
   `evaluation_prep_stage.py` (5,141), `single_g1_kitchen_episode_runpod.py` (4,341,
   zero tests), `single_g1_kitchen_qualification_session.py` (4,248, zero tests),
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
   dead generation there is the `runpod_model_volume` stub (P0 #11).

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

Do this **after** the P0 removals (moving dead code is wasted work) and incrementally —
one subpackage per PR with import shims, not a big-bang rename.

Related: 157 console scripts is its own navigability problem. After removals, collapse
the long tail of one-off CLIs into grouped `blueprint <family> <command>` entry points
or drop scripts nothing references.

## P2 — Series-A gaps (what's missing, not what's extra)

1. **Typed artifact contracts.** 2,412 `Dict[str, Any]`s; card/package/manifest schemas
   (`*.v0.1`/`v1`) exist as strings, not as schema definitions — a corrupted upstream
   artifact silently degrades downstream because everything reads
   `_read_optional_mapping(...) or {}`. Start with the sellable boundaries: pydantic
   (already used in `runtime_service_app`) or dataclass models + JSON Schema for
   site/task/scenario/eval cards, `evaluation_run.v1`, and the package export manifest,
   validated at stage boundaries.
2. **Central config.** ~1,971 scattered `os.getenv` calls; behavior flags are
   env-truthy strings; the one `PipelineConfig` dataclass carries only `gcs_root`.
   Introduce a typed settings object loaded once at each entry point.
3. **One true end-to-end test.** 486 test files (~277k LOC) but they're per-module
   contract tests against the committed fixture; there is no single CI assertion
   "capture bundle in → cards + package tar.gz + webapp projection out". Add it (can
   run against `kitchen_task_min`); it becomes the regression net for all later
   restructuring. Also: the `gpu` pytest marker exists but zero tests carry it.
4. **Zero-test giants on the live paid path**: `unitree_groot_n17_sonic_vast_persistent_session`
   (10.1k), `vast_provider_adapter` (6.7k), `single_g1_kitchen_episode_runpod` (4.3k),
   `single_g1_kitchen_qualification_session` (4.2k), and the entire g1_microwave
   fine-tune group (6.9k LOC, 10 modules, zero test files). These spend real money on
   real GPUs; they are the most under-tested code in the repo.
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

1. **PR 1 — dead code removal** (P0 §1–§11): pure deletions + the model-volume helper
   move; `pytest` fast lane green is the gate. ~21.5k LOC src + tests.
2. **PR 2 — docs/scripts hygiene** (P0 doctrine + P1 hygiene): fix the three doctrine
   contradictions, create `docs/archive/`, delete `agent_skills/` + orphaned scripts,
   Makefile/CLAUDE.md/deps fixes.
3. **PR 3 — core-path fixes** (P1): dedupe evaluation_prep, demote agent_review to
   opt-in on the CLI, rename the "legacy" eval-run adapter, single dataset manifest.
4. **PR 4 — seam enforcement** (P1 violations): retire/re-seam the Cosmos-Predict2.5
   stage + `native_runtime_backend`, config-driven simulator default, wire (or
   consciously park) the model-neutral evaluator layer.
5. **PR 5+ — structure** (P2): the end-to-end test first, then one subpackage move per
   PR with the import-linter contract added as each layer lands, then monolith splits
   and the WAM runner consolidation.

Rough net effect of PRs 1–2 alone: ~25k LOC of code and ~60 stale/contradictory docs
out of the tree, zero behavior change, and every remaining module either reachable,
CI-contracted, or explicitly quarantined.
