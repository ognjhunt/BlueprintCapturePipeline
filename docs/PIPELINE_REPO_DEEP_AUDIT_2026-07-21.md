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
entrypoint settings, writes per-run timing/spend summaries and a fail-closed fleet
aggregator, documents a production profile, and adds one CPU end-to-end regression from
capture through cards, WebApp projection, and a signed/verified Post-Training Data Package
archive.

Verification for remediation commit `5bfc2091`: the complete default pytest lane
passed with **4,656 passed, 1 skipped, and 1,595 deselected**. The previously reported
failures were reproduced as host disk-exhaustion effects, then their affected storage,
materialization, package, product-spine, local-bundle, and governed-ledger modules were
rerun successfully before the clean full-suite result.

After the first P2 seams were added, the complete combined-tree default lane also passed
with **4,736 passed, 1 skipped, and 1,595 deselected in 1:09:41**. This is the newest
complete-suite result; the smaller count above records the earlier P0/P1 checkpoint.

After that verification, `origin/main` advanced through #150/#151, #148, and #153. The
remediation branch was first rebased onto `894583dd`, refreshed onto `1974536b`, and
then rebased again onto current main `95b10eb1`; the NVIDIA/SIGGRAPH integration makes
`splat_backends` and its `isaac_nurec_export` adapter live, so both modules and their
direct tests were restored. This is a post-audit reachability change, not evidence that
the original graph scan was fabricated. The safe removal estimate and P0 ledger below
have been reduced accordingly. Post-rebase validation is recorded separately from the
pre-rebase full-suite result.

Post-rebase validation on the reconciled tree includes: 64 NVIDIA/SIGGRAPH integration
and restored splat/NuRec tests, 6 additional SimReady asset tests, 84 selected core
overlap tests (artifact/evaluation contracts, `run_e2e`, simulation automation, and the
CPU product spine), all 198 slow robot-eval orchestrator tests, and all 12 governed SC3
ledger tests. A final 318-test seam/core/upstream-overlap selection also passed: 317 in
one run, while the product-spine archive test correctly failed closed at 100% disk and
then passed separately after only pytest-generated output was removed. These are
hermetic results; they do not claim live GPU/provider execution, ranking fidelity,
deployment, or paid-resource proof.

The first P2 seam slice now puts the hosted native runtime behind a typed strategy
catalog. `site_splat` is the provider-neutral default; the legacy Cosmos-Predict2.5
adapter requires explicit `cosmos_wam` selection, conflicting old/new settings fail
closed, and the runtime no longer discovers a model checkout from hard-coded workspace
paths. Strategy-aware readiness and refinement admission now live outside the runtime
store, reducing the grandfathered monolith below its no-growth budget. The stable
runtime-service contract is unchanged. Model-family-specific artifact discovery, frame
normalization, LoRA lookup, and synchronous inference now live in the explicit
`native_runtime_cosmos_adapter`; compatibility wrappers preserve the store surface and
32 focused fast/slow native-runtime tests pass. Pure rollout merging, buffer-depth
decisions, bounded chunk replacement, and playback transitions now live in
`native_runtime_rollout_state` with direct contract tests. The asynchronous generation,
session ownership, and media-production loop is the remaining incremental split, not a
safe deletion.

Live Vast revalidation was authorized with a hard `$15` cap. A read-only, account-wide
inventory query returned HTTP 200 with zero non-terminal instances, so observed spend
remained `$0`. No instance was launched: the allocator correctly requires a clean image
and release-evidence binding for the exact `origin/main` commit, while the retained
qualification image is bound to `fca4712e` plus a non-empty dirty-patch digest and the
newest clean registry image is bound to audit commit `2edb48b`, not current main
`95b10eb1`. The normal qualification watchdog contract is bounded to at most `$5.50`,
but budget authorization does not override source identity. A clean, immutable
`95b10eb1` image plus its generated thin-release evidence is the concrete prerequisite
for live GPU startup/teardown proof; startup would still not prove policy quality,
semantic task success, or ranking fidelity.

The canonical image build route is independently blocked by provider scope: the
`paid_resource_allocator cpu-build` execution plane admits only a verified native
Linux builder or DigitalOcean, while this revalidation is authorized for Vast only.
There is no approved Vast-native image-builder path, and adding an unreviewed nested
container/build route would bypass the build-plane admission contract. After this PR
merges, either authorize the existing DigitalOcean builder or provide an admitted
native Linux builder; then build the exact merged-main image before spending on Vast.

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
| Revalidated documented operator surface (keep/narrow) | ~5.5k LOC src |
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

## P1 — Revalidated operator surfaces: do not bulk-delete

These were in the P0 list until review caught living CLI/runbook or execution evidence.
The follow-up recheck resolved each disposition; none is a safe cluster deletion.

1. **Narrowed — Lambda is read-only compatibility, not a live fallback.** The generic
   `robot_eval_provider_race_launcher` serves RunPod/Vast failover and stays. Lambda's
   adapter retains dry-run and read-only inventory compatibility, while its mutation
   CLI is hard-disabled and no canonical allocator issues its grant. Startup policy and
   the orchestrator no longer advertise `lambda_cloud` as live; runbooks no longer
   instruct operators to call the disabled terminate mode.
2. **Keep — First-GPU sample-video trio (1,354 LOC).**
   `first_gpu_sample_video_stage.py` (795), `first_gpu_sample_video_preflight.py`
   (321), `first_gpu_candidate_audit.py` (238). Zero importers, but their commands are
   documented steps in `docs/FIRST_GPU_E2E_RUNBOOK.md` (still maintained — updated by
   #144) and rows in the command-safety matrix. Either the runbook's sample-video
   staging steps remain the maintained pre-success path. Since the milestone has not
   produced a successful manipulation run, all three stay.
3. **Keep — UniFOLM infrastructure (1,648 LOC).** The initial “never run” verdict was
   false: `UNITREE_G1_POLICY_ENDPOINT_LANE.md` records completed provider/model
   execution and replay evidence, and 53 dedicated tests cover the build/server pair.
   The old launch CLI is deliberately hard-disabled, while guarded poll/probe/delete
   compatibility and the policy command/provider-bundle contracts remain useful.

## P0 — Doctrine contradictions in docs (fixed)

The current doctrine (PLATFORM_CONTEXT.md, 2026-07-15) is explicit: capture-first;
Task Evaluation Runs and Post-Training Data Packages are the sellable products; world
models are swappable internal support, **not** the product. The three contradictory
surfaces found by the audit are fixed:

1. `docs/superpowers/specs/2026-03-28-autonomous-org-design.md` has an explicit
   superseded banner and historical-only status.
2. `docs/superpowers/plans/2026-03-28-autonomous-org-implementation.md` has the same
   fail-closed superseded banner.
3. `AUTONOMOUS_ORG.md` now states capture-first, Task-Evaluation-Run and
   Post-Training-Data-Package product-first doctrine, with world-model backends as
   replaceable support infrastructure.

The misleading `docs/IOS_SITE_GROUNDED_WORLD_MODEL_SPEC.md` path was also renamed to
`docs/IOS_DATA_PACKAGE_SUPPORT_CAPTURE_SPEC.md`, with references updated.

## P1 — Docs and scripts hygiene (fixed or explicitly retained)

- **Fixed — `docs/archive/` contains the verified historical subset (~28 records)**:
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
- **Fixed — the superseded GPU VM runbook is archived** at
  `docs/archive/runbooks/GPU_VM_RUNBOOK.md`; no living README or runbook points to the
  old path.
- **Fixed — Makefile command truth.** `make test` is labeled as the fast CPU lane and
  `scripts/pytest_full.sh` is named as the full lane.
- **Fixed — portable agent paths.** `CLAUDE.md` uses repo-relative in-repo paths and
  labels the Blueprint-WebApp evidence checklist as environment-dependent.
- **Fixed — dependency intent and pinning.** `functions/requirements.txt` is exactly
  pinned; `pycollada` and `torchvision` carry their indirect/upstream rationale; and
  `websockets` is documented as uvicorn's WebSocket backend. Frozen exports remain
  CI-verified.
- **Fixed — compose output naming.** All services mount `./output` at
  `/workspace/output`; capture input remains the intentional read-only `./data` mount.

## P1 — Core-path defects (the product spine)

The spine (storage_trigger → `capture_orchestrator` → qualification →
evaluation_prep → cards → packages) is coherent and evidence-disciplined — schema
versions, claim boundaries, fail-closed rights/privacy gates (PIPE-01 verified real at
`evaluation_prep_stage.py:3817`). Defects:

1. **Fixed — `evaluation_prep` no longer runs twice in default `run_e2e`.** The stage
   ledger detects when `current`, `all`, or another selected capture lane already ran
   evaluation prep, reuses that committed result, and records the later stage as
   satisfied rather than executing it again. Tests cover both reuse and standalone
   execution.
2. **Fixed — review and trust outputs are explicit support edges.** `run_e2e` defaults
   agent review off and records a typed `not_requested` outcome. Qualification now
   computes its core evidence/routing state without writing alpha-readiness, launch, or
   buyer-trust projections unless the typed
   `BLUEPRINT_EMIT_READINESS_SUPPORT_OUTPUTS` setting is explicitly true. The optional
   writer and WebApp projection live outside the qualification monolith; tests bind both
   the default no-output path and the admitted support path.
3. **Fixed — Task Evaluation uses neutral vocabulary.** The primary entry is now
   `execute_robot_eval_request_as_evaluation_run`. The old function name remains only
   as a documented one-release compatibility alias and is not used by `run_e2e`.
4. **Fixed for the compatibility window.** The canonical filename is
   `robot_eval_dataset_manifest.json`; the legacy
   `real_site_robot_eval_dataset_manifest.json` writer carries explicit compatibility
   metadata with `sunset_not_before: 2026-08-21`, and tests bind that deadline. Delete
   the alias after that date and one compatible release, not before.
5. **Fixed — legacy lanes require explicit admission.** `capture_orchestrator` and
   `run_e2e` reject `scene_memory`, `retrieval_index`, `frame_alignment`,
   `synthesis_coverage_validation`, and `cosmos_single_capture_smoke` unless their
   respective `--allow-legacy-*` flag is present. Current product lanes never set it
   implicitly.

## P1 — Swappable-backend seam violations

The world-model seam is real and live (substrate registry `wam_eval_substrate` →
env-command process boundary `wam_provider_runtime` → strategy catalog
`wam_backend_strategy` → executor registry `evaluation_run_execution`). Registered and
selectable: fixture, OSCAR, Cosmos3 (candidate), MuJoCo, Isaac, pybullet. Violations of
"keep world-model backends swappable":

1. **Fixed — retired Cosmos-Predict2.5 support is no longer a normal `run_e2e` stage.**
   Its import is lazy inside a compatibility adapter and execution requires both the
   legacy support flag and explicit legacy-lane admission. The current product path
   never loads it.
2. **Fixed — optional backend support discovery is registry-owned.** Evaluation prep
   never executes a Cosmos exporter: it only ingests explicitly pre-existing optional
   support artifacts and otherwise records `not_requested`. The compatibility paths
   for Cosmos training and zero-shot artifacts now live in the typed, neutral
   `backend_support_artifacts` registry rather than the 5k-LOC product stage. Existing
   compatibility output keys are unchanged; registry and prep integration tests cover
   missing and supplied artifacts.
3. **Partially fixed — native runtime selection is now behind a typed strategy seam.**
   `site_splat` is the neutral default, `cosmos_wam` is explicit, conflicting old/new
   settings fail closed, and hard-coded workspace checkout discovery is removed.
   Model-family-specific prebuilt-video, conditioning-frame, LoRA checkpoint, frame
   normalization, and synchronous inference logic now lives in the explicit
   `native_runtime_cosmos_adapter`, with the runtime store's existing methods retained
   as compatibility wrappers. This reduced `native_runtime_backend.py` from 2,955 to
   2,734 LOC without changing its service contract. Pure rollout/chunk/playback state
   transitions are also isolated in `native_runtime_rollout_state`. The remaining
   asynchronous Cosmos generation, session ownership, and media-production loop is
   still embedded in the live store and needs incremental extraction.
4. **Fixed — simulator commands are configuration-owned.** The orchestrator no longer
   defaults `isaac_sim` to a G1/3DGS module. Non-fixture execution requires an explicit
   admitted simulator command and tests bind that fail-closed behavior.
5. **Fixed with a compatibility alias.** Packages use
   `visual_augmentation_backend_registry.json`; the packet still emits and the package
   builder still recognizes `model_backend_registry.json` for one compatibility window.
6. **Corrected on revalidation — Cosmos3 preference is not permanently asserted or
   impossible to activate.** The first-party scorer modules do not exist, but
   `wam_backend_strategy` also recognizes the live
   `wam_strict_action_consistency_scorer_client` when its external scorer URL is
   configured. Adapter/runtime, scorer, calibration-anchor, and run-gate checks derive
   the state; without all of them the catalog reports `aspirational`, not preferred.
   Tests cover both the default aspirational state and successful activation through
   an explicit scorer or configured external service. Keep the missing first-party
   scorer as a capability gap, not a false catalog-state bug.
7. **Fixed — the model-neutral evaluator layer is consumed by the orchestrator.** Task
   Evaluation jobs build and commit `evaluator_qualification_workflow.json`, include it
   in the job artifact manifest, and preserve its evidence/claim boundaries. Dedicated
   workflow tests plus orchestrator integration assertions cover the wiring.

## P2 — Consolidations (duplication from pivots)

1. **WAM async runners**: `runpod_wam_async_runner.py` (3,682) vs
   `vast_wam_async_runner.py` (1,893) — parallel per-provider implementations of the
   same launch/poll/collect lifecycle behind one facade (`wam_compute_providers`,
   1,903), with cross-entanglement (the RunPod runner imports Vast modules). Extract
   the shared lifecycle against the provider-adapter contracts; ~1,500 LOC reduction.
   Note `wam_compute_providers` itself overlaps `wam_provider_runtime` (its design doc
   is a June superpowers spec) — decide which is the seam and fold the other in.
   **Started:** JSON-state loading, signed-URL file handling, and URL redaction now
   live in provider-neutral `wam_async_runner_common`; both runners retain their public
   and private call surfaces. Provider artifact transfer is shared, rejects non-HTTPS
   or credential-bearing URLs before I/O, and no longer needs the moved Bandit risk
   exceptions. Monotonic retry/deadline scheduling and paid-deadline wait capping are
   now shared as well, with provider-injected sleepers preserving deterministic tests.
   Terminal ZIP inspection, runtime-result summarization, and MP4 validation now live
   in `wam_provider_output`; RunPod, Vast async, the compute facade, and the Vast
   adapter all consume it, removing RunPod's import of the 6.5k-LOC Vast adapter.
   Explicit boolean `task_success` now survives summarization while provider-runtime
   completion remains separately labeled. Token creation and authenticated staging-URL
   construction also moved to `provider_bundle_staging_common`, so RunPod and the Vast
   probe/authorized/async lanes no longer import private token helpers from the Vast
   staging server module. The larger staging manifest/server wrapper is still
   Vast-named, and provider-specific poll state transitions and teardown execution
   still need incremental extraction.
2. **Corrected on revalidation — do not fold the alleged single-consumer satellites.**
   `runpod_wam_launch_contract.py` is a cohesive carrier-volume admission, pod-payload,
   watchdog-handoff, and secret-redaction boundary deliberately extracted from the
   already 3.7k-LOC async runner; folding it back would worsen the monolith. Likewise,
   `vast_authorized_probe_runner.py` is an independently executable generic Blueprint
   bundle probe with fourteen dedicated tests, while `vast_wam_authorized_runner` is a
   separate WAM-specific lane. Their real coupling bug was the WAM lane importing two
   private guard functions from the generic runner. Those spend and staging guards now
   live in neutral `vast_probe_guards`, and both lanes depend on that shared contract.
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
package generation no longer imports G1 campaign code. The first physical package slice
is also in place: common filesystem/contracts, structured logging, safe environment
loading, fail-closed identifier/path/URL security controls, optional-dependency messaging,
typed settings, shared stage-outcome semantics, legacy-lane admission, production lane
resume, and output-run transaction/commit semantics are canonical under
`blueprint_pipeline.core`, with compatibility imports at their old paths and a mechanical
rule preventing the core package from importing campaign, readiness, or provider code.
Moving the remaining core primitives and adding broader layer coverage remain incremental
follow-ups.

Do this **after** the P0 removals (moving dead code is wasted work) and incrementally —
one subpackage per PR with import shims, not a big-bang rename.

Related: 157 console scripts is its own navigability problem. After removals, collapse
the long tail of one-off CLIs into grouped `blueprint <family> <command>` entry points
or drop scripts nothing references.

## P2 — Series-A gaps (what's missing, not what's extra)

1. **Typed artifact contracts started at the sellable boundaries.** The audit snapshot
   contained 2,463 `Dict[str, Any]` and 3,656 built-in `dict[str, Any]` occurrences.
   `artifact_contracts` now defines machine-readable contracts and JSON Schemas for
   site/task/scenario/eval cards, `evaluation_run.v1`, and the package export manifest;
   the producing and consuming stage boundaries validate them fail closed. Card
   validation/serialization and dataset compatibility/path projection are now outside
   the 4.9k-LOC dataset builder, which is below its enforced no-growth budget. Expanding
   those types into the thousands of internal campaign/provider mappings remains a
   broad incremental migration.
2. **Central config started at product and paid-path entrypoints.** The audit snapshot
   had 1,698 direct env reads (`os.getenv` + `os.environ.get`) in src. Typed
   `PipelineSettings` is now loaded once by `capture_orchestrator`, `run_e2e`, and the
   robot-eval job CLI. GCS root and the simulator, GPU, Cosmos-training, live-agent, and
   sim-only beta flags use strict boolean parsing; CLI mutation flags require matching
   environment approval before pipeline/provider execution starts. Provider- and
   campaign-specific settings plus the remaining direct reads still need incremental
   migration.
3. **One true CPU end-to-end regression is implemented.** `test_product_spine_e2e.py`
   asserts capture bundle input through cards, WebApp projection, and a signed/verified
   Post-Training Data Package archive. It is the restructuring regression net requested
   here. The separate `gpu` marker still has zero uses, so this does not supply live
   provider evidence.
4. **No GPU-marked/live-provider integration lane for the paid path.** The earlier
   claim that the live paid giants had zero tests was wrong. Dedicated CPU/unit suites
   exist and are substantial: 80 tests each for the Vast persistent session and Vast
   adapter, 30 for the RunPod kitchen episode, 37 for kitchen qualification, and 68
   across ten `g1_microwave_*` test files. The actual gap is that the `gpu` marker has
   zero uses, so these suites do not prove real provider mutation, GPU runtime startup,
   artifact collection, or teardown against live infrastructure.
5. **Shared error semantics started.** `stage_outcome` now distinguishes produced,
   not-requested, unavailable, blocked, and failed outcomes, and `run_e2e` records those
   normalized kinds in its ledger and fleet summaries. Production lane-resume markers
   also bind an explicit typed `produced` outcome to the stored lane result and reject
   mismatched/non-produced outcomes while retaining compatibility with older v1 markers.
   Most older stages still emit ad-hoc status strings, so adoption across the product
   spine and provider adapters remains incomplete.
6. **Observability.** Structured `log_event` remains concentrated in orchestrators and
   there is no metrics backend, but the minimum audit target is implemented. Every
   `run_e2e` invocation writes `pipeline/run_summary.json` with stage timings, outcome,
   and spend-evidence fields. `run_summary_aggregation` discovers those summaries under
   a fleet root, validates every input before counting any of them, and aggregates
   outcomes, providers, lanes, stage duration coverage, requested budgets, and known GPU
   seconds. It keeps unknown GPU time distinct from zero and labels requested budget as
   not actual spend. A durable fleet store/dashboard and broader orchestrator coverage
   remain follow-ups, not blockers for this minimum filesystem view.
7. **Production profile implemented; deployment evidence remains external.** Defaults
   remain fixtures by design (`provisioner="fixture_local"`, `simulator="fixture"`,
   privacy disabled by default, hosted-session artifacts contract-only, delivery upload
   off). `configs/production_pipeline.env.example` and
   `docs/PRODUCTION_PIPELINE_PROFILE.md` now provide one non-secret profile for the real
   capture path, explicit local MuJoCo evaluation, neutral hosted rendering, fleet
   summary aggregation, and the paid-resource admission boundary. Loading it is not
   permission to spend and does not prove a deployment or live provider execution.

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
  yet per the 7/10 deep audit). Several proof modules remain load-bearing. The prior
  product-spine violation where `buyer_package_readout` imported
  `g1_kitchen_attempt_closure` has been removed through the neutral
  `attempt_closure_projection` contract; it is no longer a reason to preserve that
  dependency direction.
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
   Makefile/CLAUDE.md/deps fixes. Preserve the revalidated first-GPU, generic provider
   failover, and UniFOLM operator surfaces recorded above.
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
