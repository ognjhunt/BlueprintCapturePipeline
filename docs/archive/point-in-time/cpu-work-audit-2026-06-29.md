# Non-GPU / No-Spend Work Audit — BlueprintCapturePipeline

> Archived point-in-time work audit. Verify current dependencies and test lanes.

_Generated 2026-06-29 by a 15-dimension multi-agent audit (31 agents, ~2.8M tokens). Read-only: no code changed, nothing launched._

**Scope.** Everything that can be done and **validated on CPU with no GPU and no paid cloud spend**, to be finished before resuming the Isaac G1 "open the refrigerator" head-POV seed render on RunPod.

**How to use.** Each item is a self-contained **goal command** — copy its **Prompt** into a fresh Claude/Codex session; it carries its own context. The completeness critic confirmed **0 of 147 items require a GPU or paid spend**.

**Totals:** 147 commands — **39 P0**, **75 P1**, **33 P2** — plus 5 critic-added items.

> ⚠️ **Critic's key correction — read first.** The dramatic "59 collection ERRORs" framing in P0-02 describes only a *bare* Python env without Pillow. The **canonical `.venv` already collects all 2491 tests with zero errors and PIL present.** The *real* durable gap is that **`pxr` (usd-core) and `mujoco` are MISSING from the canonical `.venv`**, so the no-GPU dry-render / placement / POV gates that are supposed to catch the invisible-robot and pitched-down-crop bugs **silently skip green** instead of running. **Fix the dependency reconciliation first** (P0-20, P0-23, P1-42), then lock a zero-skip CPU baseline (P1-03), then run the real-USD dry-render regression (P0-06, P1-04), then add provenance markers so a CPU proxy PNG can never masquerade as a real render (CRIT-01).

---

## ⭐ Do these first — critic's highest-leverage order

1. [P0-20] Declare usd-core (pxr) as a no-GPU dependency extra and add it to dev
2. [P0-23] Declare mujoco as a CPU dependency extra and un-skip the MuJoCo lane
3. [P1-42] Reconcile the dev extra so trimesh is actually installed by pip install -e .[dev]
4. [P1-03] Establish a green CPU baseline command and assert no collection errors
5. [P0-02] Guard all module-level PIL imports so tests skip without Pillow
6. [P0-06] Make the two real-USD-gated local render-preview tests run on a synthetic USD fixture
7. [P1-04] Add CPU dry-render regression test for the G1 POV seed lane
8. [P0-21] Make the --dry-render CLI actually runnable end-to-end on .venv
9. [P1-64] Extend build_parity_bundle + namelist test before any module extraction
10. [P0-04] Assert robot_visual_mesh_missing fail-closed gate fires end-to-end with proxies
11. [P0-05] Diagnose offline which G1 asset exposes renderable Gprims and lock candidate ordering
12. [P0-25] Add a 2026-06-29 CHANGELOG entry covering today's 11 G1 render commits

## Critic's overall assessment

> Largely exhaustive for CPU/no-spend work, with strong coverage of test-suite health, the Isaac G1 render lane CPU logic, scene_placement robustness, provider/spend lifecycle, and doc/provenance discipline. Zero items actually require a GPU or paid spend (needs_gpu_remove is empty) — i31/i50/i112 correctly fence the only paid surface (live Gemini/VLM) behind mocks. The biggest structural issue is duplication, not omission: the PIL-guard one-liner is filed three times (i0/i7/i117) and the pxr/mujoco/trimesh dependency work is split across i77/i80/i81/i137/i138 under different dimensions. A material premise correction: the canonical .venv already collects all 2491 tests with ZERO errors and PIL present, so the dramatic '59 collection ERRORs' framing (i1) describes only a bare Pillow-less env, not the real interpreter — the genuine, durable gap is that pxr and mujoco are MISSING from that canonical venv, silently skipping the 10 dry-render/placement USD tests and the MuJoCo-parity lane. The single highest-leverage cluster is therefore dev-env/dependency reconciliation (i77/i80/i81/i137/i138 + the meta-test in missing_items): until pxr and mujoco are in the canonical interpreter, the no-GPU dry-render and placement/POV gates that are supposed to catch the invisible-robot and pitched-down-crop bugs BEFORE any cloud spend are not actually executing — they skip green. Fix that first, then lock a zero-skip CPU baseline (i4), then run the real-USD dry-render regression (i5/i78), then add the missing provenance marker on the preview PNG so a CPU proxy can never masquerade as a real render. The thinnest area is claim-boundary discipline on the locally-generated artifacts themselves (today's 11 render commits are CPU/hermetic-only and that boundary is under-documented)."

## Critic-added items (gaps not in the 147 — add these)

### [CRIT-01] Stamp an X-Blueprint-Render-Source provenance marker on the dry-render preview PNG  (P0)
- **Why:** The local --dry-render preview PNG (scripts/run_isaac_g1_kitchen_parity_eval.py, _draw_dry_render_preview ~line 6761) carries NO render-source marker, unlike the native_runtime placeholder path ([P1-10] Regression-test the placeholder_cosmos_pending render fallback label's X-Blueprint-Render-Source: placeholder_cosmos_pending). A cheap CPU proxy PNG can therefore be filed/screenshotted as if it were a real Isaac render — exactly the claim-boundary violation the project rules warn against. No item covers provenance on the dry-render artifact itself.
- **Validate (CPU):** CPU unit test: run the dry-render path on the synthetic/real USD, open the emitted PNG metadata (PIL Image.info / PngInfo) and assert it contains an explicit 'dry_render_preview' source tag and a 'NOT a rendered frame' note; assert the summary JSON carries the same marker.

### [CRIT-02] Add a meta-test asserting the canonical .venv collects with zero errors AND has PIL/pxr/mujoco present  (P0)
- **Why:** Ground truth: the canonical .venv collects 2491 tests with zero collection errors and PIL present, so the 59/7/8-error premises ([P0-02] Guard all module-level PIL imports so tests skip without Pillow/[P1-01] Guard fastapi/uvicorn/cv2 optional imports in service tests/[P1-02] Add a Python-version guard so 3.9 envs skip 3.10+ modules cleanly) only describe a bare env, NOT the real test interpreter. Nothing pins the contract that the canonical interpreter must have the full no-GPU stack, so a future venv rebuild could silently drop pxr/mujoco again (both currently missing) and re-skip the placement/dry-render tests with no failing signal.
- **Validate (CPU):** Add tests/conftest or a meta-test that imports PIL, pxr, mujoco, trimesh, boto3 and xfails/reports loudly which are absent; wire it so CI on the canonical venv fails (not skips) if pxr or mujoco is missing.

### [CRIT-03] Pin the exact 2026-06-29 working-tree state as the evidence boundary for today's render commits  (P0)
- **Why:** There are 11 commits dated 2026-06-29 plus a dirty working tree (4 modified files: CHANGELOG, runner, two test files). [P0-25] Add a 2026-06-29 CHANGELOG entry covering today's 11 G1 render commits adds a changelog entry but nothing records that the render-visibility work (proxies, material binding, POV widening) has NEVER been proven on a real GPU render this session — it is CPU-logic + hermetic tests only. Per the autonomous-loop evidence checklist this must be stated explicitly before any 'done'.
- **Validate (CPU):** Add an explicit 'render-seed proof boundary: CPU/hermetic only, no live GPU frame produced 2026-06-29' line to the new CHANGELOG entry and the READINESS_MATRIX G1 row ([P1-48] Add READINESS_MATRIX rows for G1 render lane, scene_placement, warm-serve, provider/spend safety); assert via test_external_alpha_launch_gate-style check that the matrix row's status is not 'ready'.

### [CRIT-04] Commit-or-stash gate before resuming GPU work (dirty-tree guard)  (P1)
- **Why:** The working tree is dirty right now (runner + 2 test files + changelog uncommitted). Firing a GPU render from a dirty tree means the cloud bundle (build_parity_bundle, [P1-64] Extend build_parity_bundle + namelist test before any module extraction) may not match committed source, destroying provenance of any frame produced. No item enforces a clean/known tree as a precondition to the paid launch path.
- **Validate (CPU):** Add a pre-launch assertion in the job path (or a CPU test over it) that records the git SHA + dirty flag into the launch manifest and refuses --allow-paid on a dirty tree unless an explicit override is passed.

### [CRIT-05] Snapshot-test the full dry-render artifact set (PNG + summary JSON) against the real KitchenRoom USD on CPU  (P0)
- **Why:** The real fixture exists on disk (output/.../KitchenRoom.usd) and MEMORY says the dry-render already matched a real fridge run, but no committed test exercises the whole _draw_dry_render_preview -> _dry_render_checks chain end-to-end on that USD. [P1-04] Add CPU dry-render regression test for the G1 POV seed lane/[P1-09] Make --dry-render bind the real G1 visual asset and run pitch/POV/visibility gates/[P0-21] Make the --dry-render CLI actually runnable end-to-end on .venv are adjacent but none asserts the concrete check booleans (placement-in-frame, POV-not-pitched-down, arm-visible) on the actual asset, which is the cheapest possible catch for the invisible-robot/pitched-crop bugs.
- **Validate (CPU):** pxr-gated CPU test: open KitchenRoom.usd, run the dry-render for 'open the refrigerator', assert _dry_render_checks returns all-True and the PNG/JSON are written; xfail-marker if pxr absent so it converts to a real run once [P0-20] Declare usd-core (pxr) as a no-GPU dependency extra and add it to dev lands.

## Critic notes — overlaps to merge & priority tweaks

**Overlapping commands (do once, don't double-execute):**
- [P0-01] Guard the new PIL import in the G1 parity runner test+[P0-03] Gate PIL-dependent seed-frame test with importorskip+[P0-33] Skip PIL tests when Pillow absent so the CPU suite goes green: identical 'guard the new PIL import in the G1 parity runner test with importorskip' — three dimensions filed the same P0 one-liner against tests/test_isaac_g1_kitchen_parity_runner.py + test_local_render_preview.py
- [P0-02] Guard all module-level PIL imports so tests skip without Pillow subsumes [P0-01] Guard the new PIL import in the G1 parity runner test/[P0-03] Gate PIL-dependent seed-frame test with importorskip/[P0-33] Skip PIL tests when Pillow absent so the CPU suite goes green: the broad 'guard all module-level PIL imports' work is the superset; the single-runner-test guards should be merged into it as the first step
- [P0-20] Declare usd-core (pxr) as a no-GPU dependency extra and add it to dev+[P0-38] Install/declare usd-core so dry-render USD geometry tests run instead of skipping: same 'declare usd-core (pxr) CPU extra so dry-render/placement USD tests stop skipping' — identical P0, just different file lists (pyproject vs requirements-geometry.txt)
- [P0-21] Make the --dry-render CLI actually runnable end-to-end on .venv+[P1-09] Make --dry-render bind the real G1 visual asset and run pitch/POV/visibility gates: both are 'make --dry-render bind the real G1 visual asset and actually run cold on the kitchen USD' — [P1-09] Make --dry-render bind the real G1 visual asset and run pitch/POV/visibility gates is the runner-side feature, [P0-21] Make the --dry-render CLI actually runnable end-to-end on .venv the venv-runnability proof; same end-to-end goal
- [P0-23] Declare mujoco as a CPU dependency extra and un-skip the MuJoCo lane+[P0-37] Sync local venv + declare missing CPU deps so the materialization test stops failing: same 'mujoco missing from venv blocks the CPU MuJoCo-parity / materialization path' — [P0-23] Declare mujoco as a CPU dependency extra and un-skip the MuJoCo lane declares the extra, [P0-37] Sync local venv + declare missing CPU deps so the materialization test stops failing syncs the venv; one work item
- [P1-42] Reconcile the dev extra so trimesh is actually installed by pip install -e .[dev]+[P0-37] Sync local venv + declare missing CPU deps so the materialization test stops failing: trimesh venv-vs-pyproject reconciliation overlaps with [P0-37] Sync local venv + declare missing CPU deps so the materialization test stops failing's venv-sync goal
- [P1-04] Add CPU dry-render regression test for the G1 POV seed lane+[P1-09] Make --dry-render bind the real G1 visual asset and run pitch/POV/visibility gates: both lock the CPU --dry-render G1 POV-seed regression behavior; merge into one dry-render regression test
- [P0-34] Block task gate when manipulation-POV produced zero frames+[P1-70] Cover empty-input behavior of qc_manipulation_pov_frames and qc_render_frames: both assert empty manipulation-POV frame set fails closed (blocker) — runner gate vs rubric helper, same invariant
- [P2-05] Correct stale module references in MEMORY.md: stale MEMORY.md references (swap_orchestrator.py/nurec_worker.py) overlaps with the broader doc-accuracy cluster ([P0-24] Fix the false 'no uncommitted state' provenance line in CHANGELOG/[P0-25] Add a 2026-06-29 CHANGELOG entry covering today's 11 G1 render commits/[P1-47] Document new modules in README and link the scene_placement README)

**Suggested priority adjustments:**
- [P0-24] Fix the false 'no uncommitted state' provenance line in CHANGELOG down P0->P2 or DROP: the premise looks mis-grounded. The 2026-06-28 CHANGELOG already states 'broad uncommitted docs, scripts, source, tests' (line 310) honestly; there is no false 'no uncommitted state' claim to fix. Verify before doing work.
- [P2-27] Inventory and narrow the highest-risk bare except-Exception swallows down P2->P3/defer: inventorying/narrowing 128 bare excepts (claim said 94; actual count is 128) in a 7233-line runner is large, low-leverage, and risks behavior change right before a render; not unblocking.
- [P2-26] Decompose the 1103-line run_scenarios god-function into named phase helpers down P2->P3: decomposing the 1100-line run_scenarios god-function is real debt but a refactor that risks the exact pre-close result-write ordering the render lane depends on — defer until after a green GPU frame.
- [P2-30] Untrack tools/splat_render/node_modules (1693 committed files, 61% of repo)/[P2-31] Remove stray tracked run-artifact wam_provider_output.json from repo root/[P1-75] Fix .gcloudignore to exclude 15GB+ of run artifacts and node_modules from deploys up consideration but keep P2: untracking node_modules / run-artifacts touches provenance hygiene (a Key Rule) but is not on the render-resume critical path.
- [P1-64] Extend build_parity_bundle + namelist test before any module extraction up P1->P0-adjacent: extend build_parity_bundle + namelist test is the guardrail EVERY module extraction ([P1-65] Extract CPU-pure camera/projection geometry into parity_geometry module-[P2-25] Extract image/IO leaf helpers (denoise, quality, arg-parser) into parity_io module) and the GPU bundle depend on; it should precede all extraction items and the next paid launch.
- [P1-10] Regression-test the placeholder_cosmos_pending render fallback label up P1->P0: the placeholder_cosmos_pending render-source marker test is the canonical claim-boundary defense and directly models the missing dry-render provenance gap above.

---

## Coverage by dimension

| Dimension | Commands |
|---|---|
| Test suite health | 7 |
| Isaac G1 render — CPU logic | 10 |
| TODO/FIXME sweep | 6 |
| Main 11-stage pipeline | 11 |
| scene_placement package | 17 |
| provider_race orchestrator | 12 |
| Spend guard & pod lifecycle | 14 |
| Dev env & deps | 9 |
| Docs / provenance / claims | 7 |
| Launch gates & readiness | 8 |
| Warm render transport / object store | 8 |
| scene_semantics (Gemini) | 8 |
| Code structure / tech debt | 9 |
| Visual QC rubrics | 11 |
| Catch-all / completeness | 10 |

---

## Index (all 147)

| ID | Pri | Dimension | Title | Effort |
|---|---|---|---|---|
| P0-01 | P0 | Test suite health | Guard the new PIL import in the G1 parity runner test | S |
| P0-02 | P0 | Test suite health | Guard all module-level PIL imports so tests skip without Pillow | M |
| P0-03 | P0 | Isaac G1 render — CPU logic | Gate PIL-dependent seed-frame test with importorskip | S |
| P0-04 | P0 | Isaac G1 render — CPU logic | Assert robot_visual_mesh_missing fail-closed gate fires end-to-end with proxies | M |
| P0-05 | P0 | Isaac G1 render — CPU logic | Diagnose offline which G1 asset exposes renderable Gprims and lock candidate ordering | M |
| P0-06 | P0 | TODO/FIXME sweep | Make the two real-USD-gated local render-preview tests run on a synthetic USD fixture | M |
| P0-07 | P0 | Main 11-stage pipeline | Add per-lane fault isolation test for run_capture_pipeline | M |
| P0-08 | P0 | Main 11-stage pipeline | Surface object-index failures instead of silently swallowing in qualification | M |
| P0-09 | P0 | Main 11-stage pipeline | Test the 2D-detection to bbox-proxy-mesh fallback contract (no pointCloudFile) | M |
| P0-10 | P0 | Main 11-stage pipeline | Add all-backends-skipped end-to-end test for run_object_index_stage | M |
| P0-11 | P0 | scene_placement package | Guard compute_stand_pose against non-finite and inverted target AABBs | S |
| P0-12 | P0 | scene_placement package | Normalize inverted/zero-size AABBs at SceneObject/index construction | S |
| P0-13 | P0 | scene_placement package | Add end-to-end CPU integration test for the perception backend chain | M |
| P0-14 | P0 | provider_race orchestrator | Extract a shared importable boot-marker check helper | M |
| P0-15 | P0 | provider_race orchestrator | Reconcile cold/allow_cold_fallback/warm_only between race_launch and the real provider | M |
| P0-16 | P0 | provider_race orchestrator | Preserve warm pods on race loss: stop() instead of terminate() | M |
| P0-17 | P0 | provider_race orchestrator | Wire race_launch into the G1 kitchen parity launch path | L |
| P0-18 | P0 | Spend guard & pod lifecycle | Fix spend guard reaping STOPPED warm-reuse pods | M |
| P0-19 | P0 | Spend guard & pod lifecycle | Terminate (not stop) timed-out RunPod renders to stop disk billing | M |
| P0-20 | P0 | Dev env & deps | Declare usd-core (pxr) as a no-GPU dependency extra and add it to dev | S |
| P0-21 | P0 | Dev env & deps | Make the --dry-render CLI actually runnable end-to-end on .venv | M |
| P0-22 | P0 | Dev env & deps | Move boto3/botocore out of the cloud-only extra so staging stops failing pre-pod | S |
| P0-23 | P0 | Dev env & deps | Declare mujoco as a CPU dependency extra and un-skip the MuJoCo lane | S |
| P0-24 | P0 | Docs / provenance / claims | Fix the false 'no uncommitted state' provenance line in CHANGELOG | S |
| P0-25 | P0 | Docs / provenance / claims | Add a 2026-06-29 CHANGELOG entry covering today's 11 G1 render commits | S |
| P0-26 | P0 | Docs / provenance / claims | Run all new-module CPU test suites green and cite them as local proof | S |
| P0-27 | P0 | Launch gates & readiness | Fix launch-gate bundle-readiness override that lets non-ready bundles pass | S |
| P0-28 | P0 | Launch gates & readiness | Add a hermetic test suite for run_paid_marketplace_launch_gate.py | M |
| P0-29 | P0 | Warm render transport / object store | Run-scope warm results so poll_result cannot return a prior run's colliding-id result | M |
| P0-30 | P0 | Warm render transport / object store | Surface presigned-URL expiry (401/403) instead of swallowing it as 'no job yet' | M |
| P0-31 | P0 | Warm render transport / object store | Bound SignedUrlJobSource.poll failures so a persistently-broken inbox fails fast instead of polling forever | S |
| P0-32 | P0 | Warm render transport / object store | Require an instance/session nonce in bootstrap/serve-ready markers so a stale marker can't satisfy the gate | M |
| P0-33 | P0 | Code structure / tech debt | Skip PIL tests when Pillow absent so the CPU suite goes green | S |
| P0-34 | P0 | Visual QC rubrics | Block task gate when manipulation-POV produced zero frames | S |
| P0-35 | P0 | Visual QC rubrics | Make generic render-QC parser fail closed on missing safety booleans | S |
| P0-36 | P0 | Visual QC rubrics | Stop _norm_severity from downgrading unknown severities to 'low' | S |
| P0-37 | P0 | Catch-all / completeness | Sync local venv + declare missing CPU deps so the materialization test stops failing | S |
| P0-38 | P0 | Catch-all / completeness | Install/declare usd-core so dry-render USD geometry tests run instead of skipping | S |
| P0-39 | P0 | Catch-all / completeness | Run full pytest suite to completion and establish a clean CPU-green baseline | M |
| P1-01 | P1 | Test suite health | Guard fastapi/uvicorn/cv2 optional imports in service tests | M |
| P1-02 | P1 | Test suite health | Add a Python-version guard so 3.9 envs skip 3.10+ modules cleanly | M |
| P1-03 | P1 | Test suite health | Establish a green CPU baseline command and assert no collection errors | M |
| P1-04 | P1 | Test suite health | Add CPU dry-render regression test for the G1 POV seed lane | M |
| P1-05 | P1 | Isaac G1 render — CPU logic | Test camera pitch-down cap and target-raising trig helpers | S |
| P1-06 | P1 | Isaac G1 render — CPU logic | Unit-test head-lens mount selection ranking and head-bounds scoring | M |
| P1-07 | P1 | Isaac G1 render — CPU logic | Test review-proxy geometry math and no-physics-API invariant | M |
| P1-08 | P1 | Isaac G1 render — CPU logic | Test render-visibility diagnostics across instanceable meshes | M |
| P1-09 | P1 | Isaac G1 render — CPU logic | Make --dry-render bind the real G1 visual asset and run pitch/POV/visibility gates | L |
| P1-10 | P1 | TODO/FIXME sweep | Regression-test the placeholder_cosmos_pending render fallback label | S |
| P1-11 | P1 | TODO/FIXME sweep | Test the retrieval-index .npy decode-failure placeholder-frame fallback | S |
| P1-12 | P1 | TODO/FIXME sweep | Verify the cosmos_lora_training missing_training_command blocked-manifest contract | S |
| P1-13 | P1 | TODO/FIXME sweep | Assert WorldLabsPreviewProvider overrides every StubPreviewProvider success shortcut | S |
| P1-14 | P1 | TODO/FIXME sweep | Replace ffmpeg-conditional skips in test_privacy_processing with a deterministic fixture/mock | M |
| P1-15 | P1 | Main 11-stage pipeline | Test object_index backend-report normalization for malformed/partial output | M |
| P1-16 | P1 | Main 11-stage pipeline | Add CPU coverage for retrieval frame-extraction and quality-gating with injected embedding model | M |
| P1-17 | P1 | Main 11-stage pipeline | Pin geometry-source dispatch swappability priority ladder | S |
| P1-18 | P1 | Main 11-stage pipeline | Validate swap_candidates min_volume boundary and exclude/force precedence | S |
| P1-19 | P1 | Main 11-stage pipeline | Test scene_semantics Gemini-success branch with a mocked genai client | S |
| P1-20 | P1 | scene_placement package | Extend build_scene_index factory to construct the multi-view (preferred) backend | S |
| P1-21 | P1 | scene_placement package | Extract broad-AABB false-positive guard into the scene_placement package | M |
| P1-22 | P1 | scene_placement package | Validate against fine obstacle_boxes() instead of grouped objects() in place_and_validate | M |
| P1-23 | P1 | scene_placement package | Harden _clean_label for suffix-only and index-only prim names | S |
| P1-24 | P1 | scene_placement package | Add usd-core round-trip test of the real USD walk on a synthetic .usda | M |
| P1-25 | P1 | scene_placement package | Detect multi-target tasks and pin a deterministic contract | M |
| P1-26 | P1 | scene_placement package | Close the validation rotation-frame blind spot with a geometric look-at cross-check | M |
| P1-27 | P1 | scene_placement package | Classify openable vs static targets and consume it in placement/validation | M |
| P1-28 | P1 | scene_placement package | Validate degenerate cameras and oversize unprojected boxes in perception_index | S |
| P1-29 | P1 | provider_race orchestrator | Add a --providers (multi) / --race CLI flag to both render jobs | S |
| P1-30 | P1 | provider_race orchestrator | Define how the race interacts with the serve=True warm-pool path | M |
| P1-31 | P1 | provider_race orchestrator | Surface + persist the circuit-breaker snapshot across jobs | M |
| P1-32 | P1 | provider_race orchestrator | Test the all-providers-tripped -> race-all fallback | S |
| P1-33 | P1 | provider_race orchestrator | Test booted_lost classification + breaker success feedback | S |
| P1-34 | P1 | provider_race orchestrator | Test marker_check-raising recovery and terminate()-raising resilience | S |
| P1-35 | P1 | Spend guard & pod lifecycle | Add job-level allow_paid=True test for G1 parity job | L |
| P1-36 | P1 | Spend guard & pod lifecycle | Test warm_only blocks instead of spending on a cold pod | S |
| P1-37 | P1 | Spend guard & pod lifecycle | Test marker-retry stops every flaky warm pod with no leak | S |
| P1-38 | P1 | Spend guard & pod lifecycle | Model mid-render node 404 teardown as already-gone | M |
| P1-39 | P1 | Spend guard & pod lifecycle | Test Vast teardown contract and stop==destroy hazard | M |
| P1-40 | P1 | Spend guard & pod lifecycle | Protect not-yet-recorded pods during launch/reap race | M |
| P1-41 | P1 | Spend guard & pod lifecycle | Add teardown/test safety net for warm serve not-ready timeout | M |
| P1-42 | P1 | Dev env & deps | Reconcile the dev extra so trimesh is actually installed by pip install -e .[dev] | S |
| P1-43 | P1 | Dev env & deps | Write one documented reproducible CPU dev-setup (DEV_SETUP.md + README) | M |
| P1-44 | P1 | Dev env & deps | Add a no-GPU env doctor that asserts the full CPU dependency set | M |
| P1-45 | P1 | Docs / provenance / claims | Soften the User-Facing warm-render-server CHANGELOG claim to hermetic-only | S |
| P1-46 | P1 | Docs / provenance / claims | Document the Isaac G1 kitchen-parity render lane in README.md | M |
| P1-47 | P1 | Docs / provenance / claims | Document new modules in README and link the scene_placement README | M |
| P1-48 | P1 | Docs / provenance / claims | Add READINESS_MATRIX rows for G1 render lane, scene_placement, warm-serve, provider/spend safety | M |
| P1-49 | P1 | Launch gates & readiness | Harden _build_readiness_decision so needs_more_evidence can block and human_review_required carries signal | M |
| P1-50 | P1 | Launch gates & readiness | Parametrize that any blocked capability check forces a non-ready readiness decision | M |
| P1-51 | P1 | Launch gates & readiness | Deduplicate hidden-zone and route-edge gate thresholds across orchestrator and qualification | S |
| P1-52 | P1 | Launch gates & readiness | Add main()-flow coverage and a stale-test-list guard for the external alpha launch gate | M |
| P1-53 | P1 | Launch gates & readiness | Add an end-to-end fixture test for run_agent_review's deterministic reviewer pipeline | M |
| P1-54 | P1 | Warm render transport / object store | Run-scope or drain the warm inbox key so a restarted pod can't re-claim an orphaned job | M |
| P1-55 | P1 | Warm render transport / object store | Isolate or clear /workspace/out warm_results per serve session so the output zip can't carry stale state | M |
| P1-56 | P1 | Warm render transport / object store | Add hermetic coverage for presign_warm_inbox_channel and _await_warm_serve_ready | M |
| P1-57 | P1 | scene_semantics (Gemini) | Reconcile Gemini model-cascade ID drift across the three mirrored modules | S |
| P1-58 | P1 | scene_semantics (Gemini) | Harden _extract_json_object against multi-object / nested-trailing-junk and markdown fences | S |
| P1-59 | P1 | scene_semantics (Gemini) | Add bounded retry/backoff for transient 429 / RESOURCE_EXHAUSTED before abandoning a cascade tier | M |
| P1-60 | P1 | scene_semantics (Gemini) | Delete uploaded Gemini video files after inference (File API retention/cost/privacy leak) | M |
| P1-61 | P1 | scene_semantics (Gemini) | Emit diagnostic logging on Gemini cascade exhaustion / empty-text / parse failure | S |
| P1-62 | P1 | scene_semantics (Gemini) | Reject bool/garbage confidence so it does not silently coerce to 1.0 | S |
| P1-63 | P1 | Code structure / tech debt | Remove dead function _run_placement_visual_qc | S |
| P1-64 | P1 | Code structure / tech debt | Extend build_parity_bundle + namelist test before any module extraction | S |
| P1-65 | P1 | Code structure / tech debt | Extract CPU-pure camera/projection geometry into parity_geometry module | M |
| P1-66 | P1 | Code structure / tech debt | Extract arm-reach/skeleton kinematics and de-duplicate inline vec3 helpers | M |
| P1-67 | P1 | Visual QC rubrics | Reflect hard boolean failures in worst_severity rollup | S |
| P1-68 | P1 | Visual QC rubrics | Add boundary and false-positive tests for the black-wedge POV detector | M |
| P1-69 | P1 | Visual QC rubrics | Test manipulation rubric gripper-key alias and canonical precedence | S |
| P1-70 | P1 | Visual QC rubrics | Cover empty-input behavior of qc_manipulation_pov_frames and qc_render_frames | S |
| P1-71 | P1 | Visual QC rubrics | Test runner visual-QC import-failure and cross-rubric fail-closed propagation | M |
| P1-72 | P1 | Visual QC rubrics | Add objective pixel cross-check for VLM-self-reported dark_region_fraction | M |
| P1-73 | P1 | Catch-all / completeness | Add a ruff lint gate to CI and clear the 25 existing violations | S |
| P1-74 | P1 | Catch-all / completeness | Commit uv.lock and switch CI sync to --frozen for reproducibility | S |
| P1-75 | P1 | Catch-all / completeness | Fix .gcloudignore to exclude 15GB+ of run artifacts and node_modules from deploys | S |
| P2-01 | P2 | Test suite health | Fix the webapp_sync parametrize KeyError collection error | S |
| P2-02 | P2 | Isaac G1 render — CPU logic | Test verify-cam 3/4 framing keeps robot and target in frame | S |
| P2-03 | P2 | Isaac G1 render — CPU logic | Test software denoise PIL fallback and _save_rgb denoise routing | M |
| P2-04 | P2 | Main 11-stage pipeline | Add real-library (trimesh/scipy) skippable coverage for object_geometry mesh + collision-hull | M |
| P2-05 | P2 | Main 11-stage pipeline | Correct stale module references in MEMORY.md | S |
| P2-06 | P2 | scene_placement package | Add ambiguity diagnostic to resolve_target_by_label on equal-rank ties | S |
| P2-07 | P2 | scene_placement package | Model a target-height/swing-aware close-reach envelope for standoff | M |
| P2-08 | P2 | scene_placement package | Make multi-view fusion robust at even-count rings (odd-preferring merge) | M |
| P2-09 | P2 | scene_placement package | Flag degenerate (zero-size) view-ring bounds instead of a 1mm orbit | S |
| P2-10 | P2 | scene_placement package | Lock the Gemini response-parsing contract and flag the one paid step | S |
| P2-11 | P2 | provider_race orchestrator | Test poll-budget discretization and boot-on-last-poll boundary | S |
| P2-12 | P2 | provider_race orchestrator | Document race_launch wiring status in CHANGELOG and module docstring | S |
| P2-13 | P2 | Spend guard & pod lifecycle | Pin cold-create capacity-500 behavior distinctly from flaky pod | S |
| P2-14 | P2 | Spend guard & pod lifecycle | Surface stopped-but-billing RunPod disk in burn estimate | M |
| P2-15 | P2 | Spend guard & pod lifecycle | String-test bootstrap /workspace/out wipe before runner_done | S |
| P2-16 | P2 | Spend guard & pod lifecycle | Add JSON + burn-threshold + watch mode to spend guard | M |
| P2-17 | P2 | Spend guard & pod lifecycle | Pin particlefield flaky-cold wait gap vs parity marker-retry | M |
| P2-18 | P2 | Dev env & deps | Document the canonical pytest interpreter and the BlueprintContracts sibling requirement | S |
| P2-19 | P2 | Dev env & deps | Remove the orphan output/runpod_launch_venv ad-hoc venv | S |
| P2-20 | P2 | Launch gates & readiness | Make launch-gate scripts require the project .venv interpreter instead of failing with a misleading PIL error | S |
| P2-21 | P2 | Warm render transport / object store | Make WarmPoolClient.poll_result use conditional GET / backoff instead of re-downloading the growing zip every interval | S |
| P2-22 | P2 | scene_semantics (Gemini) | Remove or repurpose the dead _extract_json_array helper | S |
| P2-23 | P2 | scene_semantics (Gemini) | Guard a typo'd SCENE_SEMANTICS_GEMINI_MODEL / CAPTURE_FIDELITY_GEMINI_MODEL override from silently blanking the cascade | S |
| P2-24 | P2 | Code structure / tech debt | Extract pure data/manifest/serialization helpers into parity_manifest module | M |
| P2-25 | P2 | Code structure / tech debt | Extract image/IO leaf helpers (denoise, quality, arg-parser) into parity_io module | M |
| P2-26 | P2 | Code structure / tech debt | Decompose the 1103-line run_scenarios god-function into named phase helpers | L |
| P2-27 | P2 | Code structure / tech debt | Inventory and narrow the highest-risk bare except-Exception swallows | M |
| P2-28 | P2 | Visual QC rubrics | Harden and test JSON/text extraction for cascade output shapes | S |
| P2-29 | P2 | Visual QC rubrics | Pin sample_frame_paths first/last guarantee and dedup-collision behavior | S |
| P2-30 | P2 | Catch-all / completeness | Untrack tools/splat_render/node_modules (1693 committed files, 61% of repo) | S |
| P2-31 | P2 | Catch-all / completeness | Remove stray tracked run-artifact wam_provider_output.json from repo root | S |
| P2-32 | P2 | Catch-all / completeness | Collapse legacy agent_skills/ duplicates into pointers to the canonical skillpack | S |
| P2-33 | P2 | Catch-all / completeness | Add a CPU secret-scan CI step to lock in the clean secrets posture | S |

---

# P0 — Do first (unblock the G1 render lane + green CPU suite)

## Test suite health

### [P0-01] Guard the new PIL import in the G1 parity runner test

- **Priority:** P0 · **Effort:** S · **Dimension:** Test suite health
- **Goal:** Stop the new POV-seed-quality test from hard-failing on CPU envs without Pillow by adding a module-aligned importorskip guard, matching the existing pattern.
- **Files:** `tests/test_isaac_g1_kitchen_parity_runner.py`, `tests/test_local_render_preview.py`
- **Validate (CPU):** python3 -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -q  (expect 0 failed, ~4 skipped) ; python3 -m py_compile tests/test_isaac_g1_kitchen_parity_runner.py

- **Context:** Active G1 'open the refrigerator' POV seed render lane. The modified scripts/run_isaac_g1_kitchen_parity_eval.py adds `_pov_seed_frame_quality` and visual-fallback binding; the new test validates the POV seed frame quality gate (rejects black-edge self-occlusion). Baseline measured now: tests/test_isaac_g1_kitchen_parity_runner.py = 88 passed, 3 skipped, 1 FAILED solely because of the unguarded PIL import at line 1898. Every other PIL-using test in this lane (test_local_render_preview.py) already guards with pytest.importorskip("PIL"). This is the single blocker keeping the modified G1 lane test file from being green on CPU.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), the uncommitted change to tests/test_isaac_g1_kitchen_parity_runner.py added a new test `test_pov_seed_frame_quality_rejects_black_edge_occlusion` (around line 1897-1913) that does `from PIL import Image, ImageDraw` directly inside the test body with no guard. On a CPU-only environment without Pillow installed (e.g. the local system python3 = 3.9.6), this raises `ModuleNotFoundError: No module named 'PIL'` and the test FAILS rather than skips. The repo already establishes the correct pattern in tests/test_local_render_preview.py: call `pytest.importorskip("PIL")` immediately before importing PIL (see lines 89-91 there).

Task: Add `pytest.importorskip("PIL")` as the FIRST line of `test_pov_seed_frame_quality_rejects_black_edge_occlusion`, immediately before the `from PIL import Image, ImageDraw` line, so the test SKIPS (not fails) when Pillow is absent. Confirm `pytest` is already imported at the top of the file (it is). Do not change any assertion logic or the `_pov_seed_frame_quality` helper behavior — this test exercises the active 'open the refrigerator' G1 POV seed lane and the quality gate must keep its FAIL-on-edge-occlusion semantics; render/preview outputs are simulator support, NOT policy-success claims, so keep the assertions intact.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (no capture data touched here); render outputs are simulator support NOT policy-success claims; do not weaken existing assertions. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: run `python3 -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -q` and confirm the previously-failing test now reports as skipped (1 skip added) with 0 failures, and `python3 -m py_compile tests/test_isaac_g1_kitchen_parity_runner.py`.
```

</details>

### [P0-02] Guard all module-level PIL imports so tests skip without Pillow

- **Priority:** P0 · **Effort:** M · **Dimension:** Test suite health
- **Goal:** Convert the ~27 test modules that import PIL at module top-level (causing 59 collection ERRORs) to skip cleanly when Pillow is absent, getting the full CPU suite to collect.
- **Files:** `tests/test_geometry_da3.py`, `tests/test_lightwheel_kitchen_isaac_scenarios.py`, `tests/test_g1_site_3dgs_mujoco_preview.py`, `tests/test_oscar_isaac_closed_loop_eval.py`, `tests/synthesis/conftest.py`, `tests/synthesis/test_synthesize.py`
- **Validate (CPU):** python3 -m pytest --co -q tests/ 2>&1 | grep -c 'No module named .PIL.'  (expect 0) ; python3 -m py_compile $(grep -rln 'importorskip("PIL")' tests/)

- **Context:** CPU test-suite health. Measured baseline: `python3 -m pytest --co tests/` = 1515 tests collected, 76 collection errors; 59 of those 76 are missing-PIL ModuleNotFoundErrors that cascade and abort collection of otherwise-CPU-safe modules (orchestrator, readiness, geometry, cosmos, wam, isaac scenario tests). Pillow is an optional/heavy dep not present in the bare CPU env; the fix mirrors the single already-correct module (test_local_render_preview.py). This unblocks meaningful CPU coverage of the broader suite around the active G1 render lane.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), running `python3 -m pytest --co tests/` on a CPU env without Pillow produces 76 collection errors, of which 59 are `ModuleNotFoundError: No module named 'PIL'`. The cause: many test modules do `from PIL import Image` (or `from PIL import Image, ImageDraw`) at module top-level with no guard, so the whole module ERRORs at import time instead of skipping. The repo's established pattern is `pytest.importorskip("PIL")` (see tests/test_local_render_preview.py line 90).

Task: For EVERY test module under tests/ that imports PIL at module level without a guard, insert a module-level `pytest.importorskip("PIL")` immediately before the PIL import (ensuring `import pytest` is present above it). The affected files include at least: tests/test_scene_wam_policy_episode_packet.py, tests/test_geometry_da3.py, tests/test_reference_image_utils.py, tests/test_cosmos_worker.py, tests/test_synthetic_2d_wam_seed.py, tests/test_cosmos_benchmark.py, tests/test_lightwheel_kitchen_isaac_scenarios.py, tests/test_retrieval_index_stage_coverage.py, tests/test_oscar_isaac_closed_loop_eval.py, tests/test_sim_only_provider_execution_planner.py, tests/test_robot_initial_observation.py, tests/test_unitree_groot_n17_sonic_vast_persistent_session.py, tests/test_wam_derived_observation_harness.py, tests/test_mujoco_g1_simulator_command.py, tests/test_native_runtime_backend_coverage.py, tests/test_unitree_groot_n17_sonic_policy_server_command.py, tests/test_wam_generated_video_review.py, tests/test_oscar_isaac_closed_loop_gpu_launch.py, tests/test_cosmos_inference.py, tests/test_simulator_beta_readiness.py, tests/test_g1_site_3dgs_mujoco_preview.py, tests/test_native_runtime_service.py, tests/test_mujoco_g1_simulator_command_coverage.py, tests/test_wam_auxiliary_observation.py, tests/synthesis/conftest.py, tests/synthesis/test_depth_splat.py, tests/synthesis/test_synthesize.py. Discover the full set first with: `grep -rln 'from PIL\|import PIL' tests/ | while read f; do grep -q 'importorskip("PIL")' "$f" || echo "$f"; done`. For tests/synthesis/conftest.py, the PIL imports are already inside functions — guard those call sites with `pytest.importorskip("PIL")` at the start of the enclosing function rather than at module top, so the conftest still imports.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support NOT policy-success claims; do NOT delete or weaken any test — they must SKIP (not vanish) when Pillow is missing and still RUN normally when Pillow is present. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: `python3 -m pytest --co -q tests/ 2>&1 | grep -c 'No module named .PIL.'` must print 0; rerun `grep -rln 'from PIL\|import PIL' tests/ | while read f; do grep -q 'importorskip("PIL")' "$f" || echo UNGUARDED:$f; done` and confirm only intentional in-function-guarded files remain; `python3 -m py_compile <each edited file>`.
```

</details>

## Isaac G1 render — CPU logic

### [P0-03] Gate PIL-dependent seed-frame test with importorskip

- **Priority:** P0 · **Effort:** S · **Dimension:** Isaac G1 render — CPU logic
- **Goal:** Stop the one environmental failure so a clean CPU-only run is all-green or all-skip.
- **Files:** `tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k seed_frame_quality -q  → should SKIP (PIL absent) or PASS (PIL present), never FAIL. Then python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -q  → expect 0 failed. python -m py_compile tests/test_isaac_g1_kitchen_parity_runner.py.

- **Context:** This is the only red in the active CPU validation lane for the G1 'open the refrigerator' POV seed work. tests/test_isaac_g1_kitchen_parity_runner.py line 1897-1913 is the failing test; the guard pattern already exists at tests/test_local_render_preview.py ~line 431. A hard PIL failure masks the green/red signal that the audit phase depends on.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), tests/test_isaac_g1_kitchen_parity_runner.py::test_pov_seed_frame_quality_rejects_black_edge_occlusion (starts at line 1897) imports PIL at the top of the test body (line 1898: `from PIL import Image, ImageDraw`) with NO guard. On any interpreter without Pillow this FAILS (raises ModuleNotFoundError) instead of skipping. Confirmed: `python3 -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -q` currently reports `1 failed, 74 passed` and the only failure is this PIL import. Fix: add `pytest.importorskip("PIL")` (and `import pytest` if not already imported at module top) at the very start of the test body, before the `from PIL import ...` line — mirror the pattern already used in tests/test_local_render_preview.py::test_dry_render_cli_runs_end_to_end_on_real_kitchen (around line 431) and other importorskip-guarded tests in this repo. Audit the same file for any OTHER test that imports PIL or cv2 unguarded in its body (e.g. denoise/seed-frame/preview helpers) and apply the same `pytest.importorskip` guard so the CPU suite never hard-fails purely on a missing optional image library. Do NOT change production code in scripts/run_isaac_g1_kitchen_parity_eval.py; this is test-hygiene only. Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims. After editing, run `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k seed_frame_quality` and `python -m py_compile tests/test_isaac_g1_kitchen_parity_runner.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P0-04] Assert robot_visual_mesh_missing fail-closed gate fires end-to-end with proxies

- **Priority:** P0 · **Effort:** M · **Dimension:** Isaac G1 render — CPU logic
- **Goal:** Prove offline that a 0-renderable-mesh G1 blocks the render and emits review proxies + diagnostics, not a PASS.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_local_render_preview.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** python -m pytest tests/test_local_render_preview.py -q  (pxr available under python3; gate with importorskip). Assert no proxy prim path startswith '/World/G1'. python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py.

- **Context:** This is the in-progress fail-closed task for the active 'open the refrigerator' G1 POV seed lane. Wiring exists at scripts/run_isaac_g1_kitchen_parity_eval.py lines 5659-5669 (orchestrator appends ROBOT_VISUAL_MESH_MISSING_BLOCKER under manipulation_cam/verify_cam), 6157-6189 (per-scenario proxy path /World/RobotReviewVisualProxies/...), and 6269-6284 (pov_geom merge appends ROBOT_VISUAL_MESH_MISSING_BLOCKER + ROBOT_REVIEW_VISUAL_PROXY_USED_BLOCKER and forces status=FAIL). Today the only orchestration-level proof is a source grep; a refactor could let an invisible-robot render pass as PASS. tests/test_isaac_g1_kitchen_parity_runner.py:312 covers fail-closed diagnostics but not the proxy-creation + outside-/World/G1 invariant together.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add a real-pxr regression that drives the robot-visual fail-closed chain end-to-end against a synthetic in-memory USD stage, so the exact failure that wasted a GPU render (arm-link projections present but ZERO renderable meshes) can never silently pass. Use pxr (available under python3) with `Usd.Stage.CreateInMemory()`. Build a `/World/G1` subtree of `UsdGeom.Xform`/typeless prims only — apply `UsdPhysics.ArticulationRootAPI` and `UsdPhysics.CollisionAPI` on appropriate prims, add named arm link Xforms (e.g. `left_shoulder_link`, `right_shoulder_link`, `left_elbow_link`, `right_elbow_link`, `left_wrist_link`, `right_wrist_link`, `left_hand_link`, `right_hand_link`) positioned with translate ops — but add NO `UsdGeom.Gprim`/`UsdGeom.Mesh` descendants. The functions under test in scripts/run_isaac_g1_kitchen_parity_eval.py: `_robot_render_visibility_diagnostics` (line 5183), `_robot_visual_geometry_missing` (line 2251), and `_create_robot_review_visual_proxies` (line 5343). Assert: (a) the diagnostics dict has `status=='FAIL'`, `blockers` contains `M.ROBOT_VISUAL_MESH_MISSING_BLOCKER` (the constant 'robot_visual_mesh_missing', defined line 145), and `gprim_count==0`; (b) `_robot_visual_geometry_missing(diag) is True`; (c) calling `_create_robot_review_visual_proxies(stage, '/World/G1', proxy_root_path='/World/RobotReviewVisualProxies/test', arm='both')` creates prims strictly UNDER the proxy_root_path and NOT under `/World/G1`, produces a created gprim/box count >= 10, and the returned dict carries a 'render aids, not proof'-style claim_boundary (the proxies are review-only). If `_bind_g1_with_visual_fallback` (line 2264) can be exercised with a vendored physics-only candidate, also assert it returns `visual_binding_status=='blocked_missing_renderable_robot_geometry'` (set at line 2326) and preserves `selected_nonvisual_candidate_reason` (line 2328) noting articulation/collision were detected; otherwise drive the three helpers directly. Prefer extending tests/test_local_render_preview.py (real pxr, gate with `pytest.importorskip('pxr')`). Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; the proxies must never imply physical geometry. Add/extend tests only — do not weaken the production gate. Run `python -m pytest tests/test_local_render_preview.py -q` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py tests/test_local_render_preview.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P0-05] Diagnose offline which G1 asset exposes renderable Gprims and lock candidate ordering

- **Priority:** P0 · **Effort:** M · **Dimension:** Isaac G1 render — CPU logic
- **Goal:** Determine before the next render which G1 USD path actually carries visible meshes, and pin the candidate order in a test.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k visual_asset_candidates -q  (pure pxr/string logic, no GPU). If assets vendored: the replay test reports a non-zero gprim_count for the correct VISUAL candidate; otherwise it skips cleanly. python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py.

- **Context:** Root cause of the wasted render in the active G1 POV lane: the worker composed /World/G1 from a physics-only asset with 0 renderable meshes. The candidate logic at scripts/run_isaac_g1_kitchen_parity_eval.py lines 2206-2217 tries .usda→.usd and Isaac/Robots/Unitree/G1→Unitree/G1, and diagnostics traverse instance proxies at line 5209, but nothing offline confirms WHICH path carries visible geometry. Existing coverage: tests/test_isaac_g1_kitchen_parity_runner.py:42.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add an offline pxr replay/test that determines which G1 visual asset path actually exposes renderable Gprims, so we never burn another paid render just to discover the asset path. Two parts: (1) Candidate-ordering test (no assets needed): assert `M._g1_visual_asset_candidates('Isaac/Robots/Unitree/G1/g1.usda')` (scripts/run_isaac_g1_kitchen_parity_eval.py line 2190) returns an ordered, deduped list that puts the `.usd` visual sibling AND the shorter `Unitree/G1/...` path AHEAD of the physics-only `.usda` fallback — extend the existing test at tests/test_isaac_g1_kitchen_parity_runner.py:42 (`test_g1_visual_asset_candidates_try_exact_then_visual_siblings`) to also cover the absolute `/Isaac/Robots/Unitree/G1/` rewrite (line 2208) and the `.usda`→`.usd` sibling of the short path (line 2212/2216-2217). (2) Replay harness (runs only if real Isaac G1 assets are vendored locally): write a small standalone helper/test that, for each candidate from `_g1_visual_asset_candidates`, opens the resolved asset with `Usd.Stage.Open`, composes payload/reference, and counts `UsdGeom.Gprim`/`UsdGeom.Mesh` descendants using the SAME `Usd.PrimRange(robot, Usd.TraverseInstanceProxies())` traversal that `_robot_render_visibility_diagnostics` uses (line 5209), reporting per-candidate `gprim_count`/`mesh_count`/`instanceable_count`. Guard the replay with a skip when the assets are absent (do not vendor or download anything). Document in the test docstring the expected composition (payload vs reference; meshes likely marked instanceable, requiring TraverseInstanceProxies to be counted). Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims. Run `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k visual_asset_candidates -q` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## TODO/FIXME sweep

### [P0-06] Make the two real-USD-gated local render-preview tests run on a synthetic USD fixture

- **Priority:** P0 · **Effort:** M · **Dimension:** TODO/FIXME sweep
- **Goal:** Replace the 'real kitchen USD not present' skips in test_local_render_preview.py with a tiny synthetic usd-core/pxr stage so the no-GPU camera/stance/arm-framing assertions execute in every clean checkout.
- **Files:** `tests/test_local_render_preview.py`, `tests/fixtures/`
- **Validate (CPU):** python3 -m pytest tests/test_local_render_preview.py -q -rs  (the two cases formerly at lines 148/434 now run — no 's' for them); python3 -m py_compile tests/test_local_render_preview.py; if available, run the CPU `--dry-render` entrypoint against the synthetic stage to confirm camera/stance parity

- **Context:** tests/test_local_render_preview.py:148 and :434 are the skip points. The local dry-render tool (MEMORY local-dry-render-tool.md) reproduces Isaac G1 stance/camera/arm framing in ~7s with no GPU and is the main iteration loop for the 'open the refrigerator' G1 POV seed lane during the GPU pause. usd-core/pxr is the expected authoring path. P0 because this directly unblocks/validates the active G1 render lane on CPU. Note this file already shows uncommitted modifications in git status, so re-read it fresh before editing.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), de-skip the two real-USD-gated local render-preview tests by authoring a synthetic USD fixture, so the CPU dry-render iteration loop for the active G1 'open the refrigerator' lane is continuously exercised.

Background: tests/test_local_render_preview.py has two cases (skips at lines 148 and 434) that call pytest.skip('real kitchen USD not present in this checkout'). In a clean checkout without the heavy real kitchen asset these cases never run, leaving the local --dry-render / render-preview geometry/camera/stance/arm-framing logic only partially validated. Per the MEMORY 'local-dry-render-tool' note, this no-GPU tool is the primary iteration loop while GPU work is paused, so its tests must not silently skip.

Task: Author a minimal synthetic USD stage that satisfies what these two tests need — enough prims/geometry and a camera so the render-preview's geometry/camera/stance/arm-framing assertions can run. Prefer authoring it in-process via pxr/usd-core (Usd.Stage.CreateInMemory or a tmp .usda) inside a pytest fixture; if usd-core is genuinely unavailable in the CPU env, fall back to committing a tiny hand-written .usda under tests/fixtures/ and load that. Read the two test bodies first to learn exactly which prims/camera/bounds/landmarks they assert against (e.g. a fridge-like box at a known pose, a floor, a head/POV camera) and build the smallest stage that makes those assertions meaningful. Wire the fixture so each previously-skipped case uses the synthetic stage instead of skipping. Keep the real-USD path working when the real asset IS present (gate on presence, prefer real, fall back to synthetic) rather than deleting the real-asset branch.

Constraints: Keep world-model/USD backends swappable (use the standard pxr/usd-core API; do not hardcode a single renderer). Protect raw-capture truth: the synthetic stage is clearly a TEST fixture and must never be mistaken for real captured geometry — name it and label it as synthetic. Render/preview outputs are simulator support for review, NOT policy-success or task-success claims; assertions should check camera/stance/arm-framing geometry, not 'the robot opened the fridge'. Add/extend tests and fixtures only. Run `python3 -m pytest tests/test_local_render_preview.py -q -rs` and confirm the two cases at lines 148/434 no longer report 's' (skipped), and `python3 -m py_compile tests/test_local_render_preview.py`. If the repo exposes a CPU `--dry-render` entrypoint, also run it once against the synthetic stage to confirm parity. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## Main 11-stage pipeline

### [P0-07] Add per-lane fault isolation test for run_capture_pipeline

- **Priority:** P0 · **Effort:** M · **Dimension:** Main 11-stage pipeline
- **Goal:** Prove a single failing lane in a multi-lane capture run does not discard already-completed lane results.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/capture_orchestrator.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_capture_orchestrator.py`
- **Validate (CPU):** .venv/bin/python -m py_compile src/blueprint_pipeline/capture_orchestrator.py && .venv/bin/python -m pytest tests/test_capture_orchestrator.py -q

- **Context:** The canonical orchestrator is src/blueprint_pipeline/capture_orchestrator.py:run_capture_pipeline (lines ~1462-1666) — NOT swap_orchestrator.py/nurec_worker.py, which do not exist. This is the orchestrator's most important resilience invariant for multi-lane runs (e.g. qualification + retrieval_index) and is currently unverified. While the active 'open the refrigerator' G1 POV seed lane runs on GPU, all of this orchestration logic is CPU-testable because stages are injectable/subprocess-shelled. Test file: tests/test_capture_orchestrator.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), harden and test the multi-lane fault-isolation behavior of run_capture_pipeline.

Background: src/blueprint_pipeline/capture_orchestrator.py defines run_capture_pipeline (around lines 1462-1666). It iterates the requested pipeline lanes in a bare for-loop with NO per-lane try/except. Any stage that raises (e.g. run_retrieval_index_stage, run_frame_alignment_stage, or the explicit `raise ValueError('Unsupported pipeline lane')` near line 1666) aborts the entire multi-lane run and discards every lane result already accumulated in the local `results` collection. Multi-lane requests like ['qualification','retrieval_index'] are a real product surface, so losing a completed qualification result because a later retrieval lane raised is a data-loss/availability hole.

Task:
1. Read run_capture_pipeline carefully to understand exactly how lanes are iterated and how `results` is built and returned, and where the dispatch raises.
2. Add a hermetic test in tests/test_capture_orchestrator.py that drives a 2-lane request where the FIRST lane succeeds and the SECOND lane's stage entry-point is monkeypatched to raise. Assert the desired resilience invariant: completed lanes are preserved AND/OR the failing lane is surfaced as a structured lane result, rather than the whole call dying with an uncaught exception. Match whichever contract the code already intends; if the code currently lets the exception escape, ALSO update run_capture_pipeline to wrap each lane in try/except that records a structured failure entry (status='failed' plus the lane name and error string) and continues, then assert that.
3. Keep the change minimal and do not alter lane ordering or the success-path return shape. Keep world-model backends swappable — do not hardcode any specific backend; the lane dispatch must stay generic. Render/eval outputs are simulator support, not policy-success claims — do not add any success-claim semantics. Protect provenance/rights/privacy/raw-capture-truth — do not log raw capture bytes or PII in the new failure entries (lane name + error string only).
4. Add/extend tests as above. Run the validation commands and ensure they pass.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P0-08] Surface object-index failures instead of silently swallowing in qualification

- **Priority:** P0 · **Effort:** M · **Dimension:** Main 11-stage pipeline
- **Goal:** Make a SAM3D/object-index crash a logged, recorded blocker rather than an invisible empty index.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/qualification.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_qualification_coverage_edges.py`
- **Validate (CPU):** .venv/bin/python -m py_compile src/blueprint_pipeline/qualification.py && .venv/bin/python -m pytest tests/test_qualification_coverage_edges.py -q

- **Context:** src/blueprint_pipeline/qualification.py is the quality stage. SAM3D/object-index is the single most failure-prone GPU stage, and silently swallowing its crash hides it from operators and from qualification gating, undermining readiness/provenance guarantees. This matters for the active 'open the refrigerator' G1 lane: if object indexing silently empties, the kitchen scene loses its detected objects with no trace. Test file: tests/test_qualification_coverage_edges.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), fix silent exception swallowing on the object-index stage inside run_qualification_pipeline.

Background: src/blueprint_pipeline/qualification.py wraps ensure_object_index_stage(...) at roughly lines 4238-4244 in `except Exception: stage_result = {}` with NO logging. The outer block (~4256-4261) also has a bare `except Exception:` that nulls manifest/object_index/grounding with no log_event. As a result a SAM3D/object-index crash becomes an invisible empty index — downstream consumers just see zero objects with no signal to operators or to qualification gating.

Task:
1. Read the two except blocks (~4238-4261) and the surrounding context, including how object_index_runtime_blockers and log_event are used elsewhere in this module so you match existing conventions.
2. Add a log_event (or the module's existing structured logging call) on BOTH except paths that records that the object-index stage failed, including the exception type/message. If the module already tracks a structured blocker list such as object_index_runtime_blockers, append a blocker entry there too.
3. Do NOT change the graceful-degradation outcome (object_count==0 path must still be reachable without crashing) — only make the failure observable. Keep readiness/review logic secondary to the product core; do not introduce a hard gate that aborts the pipeline.
4. Add a test in tests/test_qualification_coverage_edges.py that monkeypatches ensure_object_index_stage to raise, then asserts (a) the failure is recorded/logged (e.g. captured log_event call, or an entry in object_index_runtime_blockers) and (b) the run still reaches the object_count==0 path without raising.
5. Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (log only error type+message, never raw capture content); render/eval outputs are simulator support, not policy-success claims. Run validation and ensure it passes.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P0-09] Test the 2D-detection to bbox-proxy-mesh fallback contract (no pointCloudFile)

- **Priority:** P0 · **Effort:** M · **Dimension:** Main 11-stage pipeline
- **Goal:** Pin the cross-stage contract that 2D-only object index entries deterministically land on bbox_proxy_mesh / grounding_level='inferred'.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/object_index_stage.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/object_geometry_stage.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/swap_candidates.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_object_geometry_stage_coverage_edges.py`
- **Validate (CPU):** .venv/bin/python -m py_compile src/blueprint_pipeline/object_geometry_stage.py src/blueprint_pipeline/object_index_stage.py src/blueprint_pipeline/swap_candidates.py && .venv/bin/python -m pytest tests/test_object_geometry_stage_coverage_edges.py -q

- **Context:** This is a load-bearing cross-stage contract spanning object_index_stage.py -> object_geometry_stage.py -> swap_candidates.py. With GPU/splat work paused for the active 'open the refrigerator' G1 lane, every kitchen asset is built from bbox proxies, so the fallback chain must be provably correct on CPU. Test file: tests/test_object_geometry_stage_coverage_edges.py (already contains a fake-trimesh harness, e.g. test_object_geometry_fake_trimesh_branches).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add an integration-style test that pins the 2D-detection to bbox-proxy-mesh fallback contract across stages.

Background: src/blueprint_pipeline/object_index_stage.py emits per-object entries (around lines 1150-1192) from the 2D backends (yolo_world / grounding_dino / sam3) WITHOUT a `pointCloudFile` key — only the splat_analyzer (3D) backend produces point clouds. But both src/blueprint_pipeline/swap_candidates.py (line ~647) and src/blueprint_pipeline/object_geometry_stage.py (line ~463) read `entry.get('pointCloudFile')`. So in the no-GPU / 2D-only path that value is always None and the code silently falls back to bbox-proxy meshes (object_geometry_stage.py ~lines 918-920 `_box_mesh_from_bbox`, mesh_source='bbox_proxy_mesh'). With GPU/splat work paused, ALL geometry currently comes from these bbox proxies, and nothing asserts that fallback chain end-to-end — a silent key mismatch would degrade every asset with no test catching it.

Task:
1. Read object_geometry_stage.py around run_object_geometry_stage, the pointCloudFile read (~463), the bbox-proxy branch (~918-920), and how mesh_source and grounding_level are set. Look at tests/test_object_geometry_stage_coverage_edges.py to reuse the existing fake-trimesh stub harness.
2. Add a test in tests/test_object_geometry_stage_coverage_edges.py that builds a 2D-only object index (several entries with bbox but NO pointCloudFile key), runs run_object_geometry_stage with the fake trimesh stub, and asserts that for EVERY object: mesh_source == 'bbox_proxy_mesh' AND grounding_level == 'inferred'.
3. Do not change production behavior unless you find an actual bug in the fallback chain; this is primarily a characterization/contract test. If you do find a key-name mismatch bug, fix it minimally and note it.
4. Constraints: keep world-model backends swappable (the test must not assume any single detection backend beyond the entry shape); protect provenance/rights/privacy/raw-capture-truth; render/eval outputs are simulator support, not policy-success claims. Run validation and ensure it passes.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P0-10] Add all-backends-skipped end-to-end test for run_object_index_stage

- **Priority:** P0 · **Effort:** M · **Dimension:** Main 11-stage pipeline
- **Goal:** Pin the no-GPU default path: zero backends configured yields status='built', object_count==0, empty_index_cause=='backend_skipped', plus emitted artifacts.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/object_index_stage.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_object_index_stage.py`
- **Validate (CPU):** .venv/bin/python -m py_compile src/blueprint_pipeline/object_index_stage.py && .venv/bin/python -m pytest tests/test_object_index_stage.py -q

- **Context:** This is precisely the no-GPU default operating mode while cloud work is paused for the active 'open the refrigerator' G1 lane. Pinning the exact empty-index classification ('backend_skipped' vs 'zero_detections'/'all_filtered') guards qualification gating, which consumes empty_index_cause. File: src/blueprint_pipeline/object_index_stage.py; test file: tests/test_object_index_stage.py (has capture-synthesis helpers and adjacent empty-case tests ~883-963).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add a first-class no-GPU-default end-to-end test for run_object_index_stage.

Background: src/blueprint_pipeline/object_index_stage.py: when no OBJECT_INDEX_*_COMMAND env vars are set, _run_backend_command returns status='skipped' / reason='command_not_configured' (around lines 648-649) for every backend, and the empty_index_cause classification logic (around lines 1759-1767) should yield 'backend_skipped'. ffmpeg is available locally so _extract_keyframe_images works on a tiny synthetic video. Existing tests in tests/test_object_index_stage.py (around lines 883-963) cover legacy-reuse and a force-rebuild empty case, but there is NO test that runs the FULL stage with zero backends configured and asserts the exact empty-index classification and artifact emission.

Task:
1. Read run_object_index_stage end-to-end, the _run_backend_command skip path (~648-649), the empty_index_cause logic (~1759-1767), and how the build report and grounding-hints files are written. Reuse fixtures/helpers from tests/test_object_index_stage.py for synthesizing a small capture with a tiny video.
2. Add a test that: synthesizes a minimal kitchen capture with a tiny video, ensures all OBJECT_INDEX_*_COMMAND env vars are unset (use monkeypatch.delenv with raising=False, and guard against any leaking from the environment), runs run_object_index_stage(force_rebuild=True), and asserts status=='built', object_count==0, empty_index_cause=='backend_skipped', and that the build report path and grounding hints file actually exist on disk.
3. Do not weaken the empty_index_cause classification; if the run is mislabeled as 'zero_detections' or 'all_filtered' instead of 'backend_skipped', fix the classification minimally so 'no backend configured' is distinct.
4. Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (use synthetic capture data only); render/eval outputs are simulator support, not policy-success claims. Run validation and ensure it passes.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## scene_placement package

### [P0-11] Guard compute_stand_pose against non-finite and inverted target AABBs

- **Priority:** P0 · **Effort:** S · **Dimension:** scene_placement package
- **Goal:** Make compute_stand_pose reject NaN/inf and inverted (min>max) target boxes instead of silently emitting NaN-position / negative-standoff poses.
- **Files:** `src/blueprint_pipeline/scene_placement/placement.py`, `src/blueprint_pipeline/scene_placement/types.py`, `src/blueprint_pipeline/scene_placement/validation.py`, `tests/test_scene_placement.py`
- **Validate (CPU):** python -m pytest tests/test_scene_placement.py -k stand_pose -q ; python -m pytest tests/test_scene_placement.py tests/test_placement_validation.py -q ; python -m py_compile src/blueprint_pipeline/scene_placement/placement.py src/blueprint_pipeline/scene_placement/types.py ; plus a one-off repro: python3 -c "from src.blueprint_pipeline.scene_placement.types import SceneObject; from src.blueprint_pipeline.scene_placement.placement import compute_stand_pose; nan=float('nan'); o=SceneObject(id='t',label='faucet',bbox_min=(nan,nan,0),bbox_max=(nan,nan,1),centroid=(nan,nan,.5)); sp=compute_stand_pose(o,probe=lambda p,y:0); import math; assert math.isfinite(sp.position[0]) and not sp.clear, sp"

- **Context:** placement.py::compute_stand_pose is called by the runner's _scene_placement_stand_plan path (scripts/run_isaac_g1_kitchen_parity_eval.py) with USD-derived bounds, and its output feeds validate_stand_pose only AFTER the fact. A NaN pelvis pose or negative standoff flows downstream as a 'fine' placement because IEEE comparisons against NaN silently pass. This directly threatens the active 'open the refrigerator' G1 POV seed lane: a degenerate fridge AABB would otherwise park the pelvis at a NaN/garbage spot that no pre-render check in the solver catches. Files: src/blueprint_pipeline/scene_placement/placement.py (lines ~62-75 _half_extent_along, ~148-260 compute_stand_pose), src/blueprint_pipeline/scene_placement/validation.py (_all_finite/_obj_bbox_finite at lines ~102-113), src/blueprint_pipeline/scene_placement/types.py (SceneObject.size at lines ~43-52).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), harden src/blueprint_pipeline/scene_placement/placement.py::compute_stand_pose so a degenerate target cannot silently produce a garbage pose.

Confirmed problem (reproduced empirically): a SceneObject whose AABB is non-finite (e.g. NaN centroid) makes compute_stand_pose return position=(nan,nan,0.79), yaw=nan, clear=False, standoff_m=nan. An inverted box (bbox_min=(1,1,1), bbox_max=(0,0,0)) gives SceneObject.size()=(-1,-1,-1), so _half_extent_along returns a NEGATIVE half-extent and the standoff/start/ceiling math is corrupted (a clear=True pose with a wrong standoff). validate_stand_pose already gates non-finite input via _all_finite/_obj_bbox_finite in src/blueprint_pipeline/scene_placement/validation.py and proves the team wants explicit non-finite failures (see test_nan_position_fails_with_explicit_reason) — the SOLVER should not be the one place that swallows them.

What to do:
1. At the top of compute_stand_pose, add a guard mirroring validation.py's finiteness check: if any coordinate of target.bbox_min / bbox_max / centroid is non-finite, OR any axis is inverted (bbox_min[i] > bbox_max[i]), do NOT run the probe loop. Either (a) raise ValueError with an explicit message naming the target id and which condition failed, or (b) return a StandPose flagged clear=False with notes containing an explicit token like 'degenerate_target' and a finite, non-misleading position (e.g. the finite footprint center at z=floor_z+pelvis_height, or the centroid if finite). Pick ONE behavior and document it in the docstring; a returned StandPose must never have a NaN position or a negative standoff_m.
3. Reuse the existing finiteness helper if practical: import _all_finite / _obj_bbox_finite from validation, or add a tiny local _obj_bbox_finite in placement.py (keep the package dependency-light — stdlib only, no new third-party imports). Do not change the existing valid-input behavior.

Constraints: keep world-model backends swappable (do not couple placement to a concrete index); protect provenance/rights/privacy/raw-capture-truth (do not fabricate geometry — flag, do not invent a plausible box); render/placement outputs are simulator support, NOT policy-success claims; add/extend tests; run `python -m pytest tests/test_scene_placement.py tests/test_placement_validation.py` and `python -m py_compile src/blueprint_pipeline/scene_placement/placement.py src/blueprint_pipeline/scene_placement/types.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Extend tests in tests/test_scene_placement.py: (a) a target with a NaN coordinate yields the documented flagged-unclear pose (finite position, clear=False, 'degenerate_target' note) or raises ValueError — assert exactly the behavior you chose; (b) an inverted (min>max) finite AABB yields the same flagged/raised outcome rather than a clear=True pose with a corrupted standoff; (c) a normal ordered box still produces the same pose as before (regression guard). Drive everything with a synthetic SceneObject + a mock probe; no GPU.
```

</details>

### [P0-12] Normalize inverted/zero-size AABBs at SceneObject/index construction

- **Priority:** P0 · **Effort:** S · **Dimension:** scene_placement package
- **Goal:** Ensure no SceneObject can carry an inverted (min>max) box: normalize axes to (min,max) at construction so negative size() never poisons downstream placement/validation math.
- **Files:** `src/blueprint_pipeline/scene_placement/types.py`, `src/blueprint_pipeline/scene_placement/usd_index.py`, `src/blueprint_pipeline/scene_placement/perception_index.py`, `tests/test_scene_placement.py`
- **Validate (CPU):** python -m pytest tests/test_scene_placement.py -k 'bounds or inverted or size' -q ; python -m pytest tests/test_scene_placement.py -q ; python -m py_compile src/blueprint_pipeline/scene_placement/usd_index.py src/blueprint_pipeline/scene_placement/perception_index.py src/blueprint_pipeline/scene_placement/types.py ; python3 -c "from src.blueprint_pipeline.scene_placement.usd_index import _objects_from_bounds; o=_objects_from_bounds([('faucet',((1,1,1),(0,0,0)))])[0]; assert all(s>=0 for s in o.size()), o.size()"

- **Context:** Inverted AABBs are a real input: authored USD prims and especially perception corner-unprojection (finite center depth applied to far-offset corners) can produce min>max. The package is the single normalization seam between raw scene names/geometry and task placement, so canonicalizing here protects every downstream consumer at once and removes a class of silent sign-flip bugs that no current test catches. Files: src/blueprint_pipeline/scene_placement/usd_index.py (_objects_from_bounds ~139-186), src/blueprint_pipeline/scene_placement/perception_index.py (_aabb_from_points ~268), src/blueprint_pipeline/scene_placement/types.py (SceneObject ~23-72).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), make inverted/degenerate AABBs impossible to leak out of the scene_placement indices.

Confirmed problem: _objects_from_bounds([('faucet', ((1,1,1),(0,0,0)))]) in src/blueprint_pipeline/scene_placement/usd_index.py yields a SceneObject with size()=(-1,-1,-1). A negative extent flips _half_extent_along, _xy_box_gap, and standoff signs across placement.py and validation.py. validate_stand_pose handles NON-finite boxes but NOT inverted-finite ones (test_zero_size_target_has_defined_verdict only covers zero-size). Perception unprojection (_aabb_from_points in perception_index.py) can also emit inverted boxes from bad corner depths.

What to do (pick the construction-time seam, not the read-time accessors, so existing callers of size() keep working):
1. In src/blueprint_pipeline/scene_placement/usd_index.py::_objects_from_bounds, after computing bmin/bmax, normalize each axis: bmin_i = min(raw_min_i, raw_max_i), bmax_i = max(...). Recompute centroid from the normalized box. Add an inline comment that authored USD / perception can hand back inverted corners and we canonicalize here.
2. In src/blueprint_pipeline/scene_placement/perception_index.py::_aabb_from_points (and any other place a SceneObject AABB is assembled on the perception side), apply the same per-axis min/max canonicalization so a finite-but-inverted unprojected box is corrected, not propagated.
3. Leave NON-finite handling to the existing finiteness guards (do not silently 'fix' a NaN into 0). Add a short docstring note in types.py near SceneObject documenting the invariant that producers MUST emit bbox_min<=bbox_max per axis, and that the indices enforce it.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (canonicalizing min/max is lossless reordering, NOT fabrication — never invent a non-degenerate box from a zero-size one); render/placement outputs are simulator support, NOT policy-success claims; add/extend tests; run `python -m pytest tests/test_scene_placement.py` and `python -m py_compile` on the touched files. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Add tests in tests/test_scene_placement.py: (a) _objects_from_bounds with an inverted input box yields size() with all non-negative axes and a centroid equal to the equivalent ordered box; (b) feeding the resulting object into compute_stand_pose + validate_stand_pose behaves identically to the ordered-box equivalent; (c) a perception _aabb_from_points case where corner order is reversed still yields min<=max. Pure synthetic numbers, no GPU.
```

</details>

### [P0-13] Add end-to-end CPU integration test for the perception backend chain

- **Priority:** P0 · **Effort:** M · **Dimension:** scene_placement package
- **Goal:** Prove the perception units compose into a placement decision with a single hermetic test: view ring -> build views -> MultiView fused catalog -> resolve -> compute_stand_pose -> validate.
- **Files:** `src/blueprint_pipeline/scene_placement/perception_fusion.py`, `src/blueprint_pipeline/scene_placement/perception_views.py`, `src/blueprint_pipeline/scene_placement/perception_adapter.py`, `src/blueprint_pipeline/scene_placement/__init__.py`, `tests/test_perception_views.py`, `tests/test_perception_fusion.py`
- **Validate (CPU):** python -m pytest tests/test_perception_views.py -k 'full_chain or integration or placement' -q ; python -m pytest tests/test_perception_views.py tests/test_perception_fusion.py tests/test_scene_placement.py -q

- **Context:** The README markets MultiViewPerceptionSceneSpatialIndex as the 'preferred' raw-scene/splat path and the splat_analyzer parity story depends on it, yet PerceptionSceneSpatialIndex/MultiView/perception_adapter/perception_views are imported nowhere outside the package and its own unit tests (the runner only ever constructs UsdSceneSpatialIndex). A swappable backend with zero integration coverage is a latent break. Files: src/blueprint_pipeline/scene_placement/perception_fusion.py (MultiViewPerceptionSceneSpatialIndex, fuse_scene_objects), perception_views.py (view_ring_for_bounds, assemble_views), perception_adapter.py (build_perception_views, build_perception_views_from_frames), __init__.py (place_and_validate_robot_for_task ~159-191), tests/test_perception_views.py (test_full_chain_bounds_to_fused_object), tests/test_perception_fusion.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), add a hermetic CPU integration test that exercises the FULL perception placement chain end to end with injected fakes. Today every perception unit is tested in isolation and the chain is never assembled into a placement decision; a contract drift between perception_index/perception_fusion output and resolve_target/compute_stand_pose would pass all current per-module tests.

What to do:
1. In tests/test_perception_views.py (or a new tests/test_perception_placement_integration.py), build a test that starts from view_ring_for_bounds(bbox_min, bbox_max) for a synthetic kitchen-ish scene, runs build_perception_views_from_frames (or build_perception_views) with INJECTED fake detect/depth callables (deterministic 2D boxes + a constant/lookup depth provider so a known 'faucet' lands at a known world centroid), constructs MultiViewPerceptionSceneSpatialIndex from the assembled views, calls .objects() to get a fused catalog, then continues into place_and_validate_robot_for_task(index, 'turn on the faucet', probe=<mock clear probe>, floor_z=0.0) and asserts the returned StandPose is clear=True, faces the faucet centroid (yaw within tolerance), and the PlacementVerdict is ok with standoff in range.
2. Keep it 100% offline: no pxr, no google-genai (omit generate so the label fallback runs), no SAM3/DA3, no GPU. All detect/depth/probe are plain Python callables in the test.
3. Extend the existing test_full_chain_bounds_to_fused_object if it already builds the fused catalog — chain from its output into placement rather than duplicating setup.

Constraints: keep world-model backends swappable (the test must go through the package's public surface — build_perception_views*, MultiViewPerceptionSceneSpatialIndex, place_and_validate_robot_for_task — not private internals where avoidable); protect provenance/rights/privacy/raw-capture-truth (use synthetic detections, never real capture); render/placement outputs are simulator support, NOT policy-success claims; this is primarily a test addition; run `python -m pytest tests/test_perception_views.py tests/test_perception_fusion.py tests/test_scene_placement.py` and `python -m py_compile` on any touched source. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## provider_race orchestrator

### [P0-14] Extract a shared importable boot-marker check helper

- **Priority:** P0 · **Effort:** M · **Dimension:** provider_race orchestrator
- **Goal:** Factor the inline bootstrap.json-zip poll into one reusable function that race_launch can be handed as marker_check, so every future wiring site shares the canonical detection logic.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/provider_race.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_particlefield_render_job.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_provider_race.py`
- **Validate (CPU):** python -m pytest tests/test_provider_race.py -q ; python -m pytest tests/<new_helper_test>.py -q (test boot_marker_present with a monkeypatched urlopen returning an in-memory zip WITH bootstrap.json -> True, WITHOUT it -> False, and a urlopen that raises -> False, plus missing provider_output_get_url.txt -> False) ; python -m py_compile src/blueprint_pipeline/provider_race.py src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py

- **Context:** src/blueprint_pipeline/provider_race.py is a complete pure-orchestration racer whose marker_check seam (signature marker_check(provider, launch_result) -> bool, called at provider_race.py:272) is the single thing standing between it and being wired into the live launch path. Without a shared, importable helper every call site re-implements the zip/bootstrap.json poll and drifts (note bootstrap.json vs runner_done are different markers in the two files). This is the foundational seam for the active 'open the refrigerator' G1 POV render lane: the G1 job at isaac_g1_kitchen_parity_job.py:599 still does single-provider launch_with_marker_retry, and a clean importable marker_check is the prerequisite for racing RunPod+Vast there.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), the boot-marker detection logic is duplicated inline and is not importable, which blocks wiring src/blueprint_pipeline/provider_race.py:race_launch (its marker_check seam needs a production implementation).

The inline logic appears in two places:
- src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py lines ~400-409 inside launch_with_marker_retry: it reads (job_dir / 'provider_output_get_url.txt'), GETs that signed url via urllib.request.urlopen(get_url, timeout=60), opens the bytes as zipfile.ZipFile(io.BytesIO(data)), and checks whether 'bootstrap.json' is in namelist().
- src/blueprint_pipeline/isaac_particlefield_render_job.py lines ~317-336 inside watch_and_collect: similar GET-zip-extract pattern, but it keys off bootstrap.json phase == 'runner_done'.

Task: add a single reusable, pure helper (e.g. boot_marker_present(job_dir, *, get_url=None, marker_name='bootstrap.json', urlopen=urllib.request.urlopen) -> bool) in a sensible importable location — prefer adding it to src/blueprint_pipeline/provider_race.py as a module-level function OR a new small src/blueprint_pipeline/boot_marker.py if you judge provider_race must stay dependency-free of urllib (provider_race's docstring explicitly says it imports no provider/SDK/network — if you add urllib there, gate the import inside the function body so the module stays pure-orchestration at import time). The helper must: read provider_output_get_url.txt from the given job_dir when get_url is not passed, fetch the signed url, open the zip, return True iff marker_name is in the zip namelist, and return False (never raise) on any network/zip/missing-file error. Make urlopen injectable so it is testable without network.

Then refactor launch_with_marker_retry in isaac_g1_kitchen_parity_job.py to call the new helper (preserving its existing behavior exactly — same marker, same swallow-and-continue semantics).

Constraints: keep world-model backends swappable; do not change provider surface contracts; protect provenance/rights/privacy/raw-capture-truth (this only reads a boot heartbeat marker, not capture data); render/launch outputs are simulator support, NOT policy-success claims; add/extend tests. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation you must run and show green: python -m pytest tests/test_provider_race.py and any new test file you add; python -m py_compile on every file you touch.
```

</details>

### [P0-15] Reconcile cold/allow_cold_fallback/warm_only between race_launch and the real provider

- **Priority:** P0 · **Effort:** M · **Dimension:** provider_race orchestrator
- **Goal:** Make race_launch able to express warm-restart-then-cold and warm_only per provider so wiring it does not silently disable the cheap warm path and the warm_only guard.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/provider_race.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/gpu_render_providers.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_provider_race.py`
- **Validate (CPU):** python -m pytest tests/test_provider_race.py -q (with new kwargs-forwarding + warm_only-never-cold + cold-only-provider-tolerance tests) ; python -m py_compile src/blueprint_pipeline/provider_race.py src/blueprint_pipeline/gpu_render_providers.py

- **Context:** This is a correctness/cost gap that must be closed BEFORE race_launch is wired into the active G1 'open the refrigerator' lane, or wiring it will regress the warm-pod reuse strategy (forcing expensive cold ~10.7GB pulls) and break the --warm-only guard at isaac_g1_kitchen_parity_job.py:602. The racer's docstring (provider_race.py:21-26) deliberately constrains the provider surface to launch(job_dir, request, *, cold=False) + terminate(instance_id); reconciling the extra RunPod kwargs while staying provider-agnostic is the crux.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), src/blueprint_pipeline/provider_race.py:race_launch calls provider.launch(sub_dir, _resolve_request(request, provider), cold=cold) at provider_race.py:246 — it ONLY forwards cold. But the real RunPodRenderProvider.launch in src/blueprint_pipeline/gpu_render_providers.py (signature around line 162: launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True)) ALSO takes allow_cold_fallback, and the G1 job passes allow_cold_fallback=not warm_only at isaac_g1_kitchen_parity_job.py:602. If race_launch is wired without addressing this, warm-restart reuse (avoids a ~10.7GB image pull) and the warm_only guard silently stop working.

Task: extend race_launch so callers can express per-provider launch kwargs beyond cold without breaking its 'pure orchestration / only touches the agreed provider surface' design. Implement ONE of these (pick the cleanest and document the choice in the race_launch docstring): (a) allow the request callable / an optional launch_kwargs callable to supply per-provider keyword args that are forwarded to provider.launch via **kwargs; or (b) add an explicit allow_cold_fallback parameter (and pass it through) while tolerating providers whose launch does not accept it (inspect signature or try/except TypeError fallback to a plain cold= call). The Vast provider's launch may not accept allow_cold_fallback — your forwarding MUST NOT break a provider that only accepts cold=. Update the docstring's stated provider surface accordingly.

Constraints: keep world-model/provider backends swappable (do not hardcode RunPod-specific kwargs into the racer's core path — make it generic passthrough); render/launch outputs are simulator support NOT policy-success claims; protect provenance/rights/privacy/raw-capture-truth; add/extend tests. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: extend tests/test_provider_race.py — make FakeProvider.launch record the kwargs it received (cold + allow_cold_fallback or whatever passthrough you chose), and add tests asserting (1) a warm_only-style race never causes a cold create (the fallback flag is forwarded as expected), (2) a provider whose launch only accepts cold= still works (no TypeError). Run python -m pytest tests/test_provider_race.py and python -m py_compile on touched files.
```

</details>

### [P0-16] Preserve warm pods on race loss: stop() instead of terminate()

- **Priority:** P0 · **Effort:** M · **Dimension:** provider_race orchestrator
- **Goal:** Let race_launch honor the warm-restart vs cold distinction when tearing down losers so a merely-slower warm pod is stopped (preserved for reuse), not deleted.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/provider_race.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/gpu_render_providers.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_provider_race.py`
- **Validate (CPU):** python -m pytest tests/test_provider_race.py -k 'warm or stop' -q ; python -m pytest tests/test_provider_race.py -q ; python -m py_compile src/blueprint_pipeline/provider_race.py

- **Context:** Terminating a warm-restart pod that only lost the race defeats the warm-pool strategy (documented in MEMORY: avoiding the ~10.7GB cold image pull), forcing future cold pulls — a real cost regression hiding inside the otherwise-safe no-leak teardown. This directly affects the active G1 'open the refrigerator' render lane once race_launch is wired there, because that lane's whole point is reusing warm pods. The canonical stop-vs-terminate logic to mirror lives at isaac_g1_kitchen_parity_job.py:415-418.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), src/blueprint_pipeline/provider_race.py:race_launch unconditionally calls runnable[i].terminate(iid) on every launched loser (around provider_race.py:317). But the existing retry path deliberately distinguishes warm-restart pods from cold pods: in src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py:415-418, launch results whose mode starts with 'warm' are stopped via prov.stop(iid) (preserved for reuse) while cold pods are terminated (deleted). A warm-restart pod that merely LOSES a race must not be deleted.

Task: in race_launch's loser-teardown loop, when a launched loser's recorded launch mode indicates a warm pod (e.g. str(rec['mode'] or '').startswith('warm')) AND the provider exposes a stop() method, call provider.stop(iid) instead of provider.terminate(iid); fall back to terminate() for cold losers or providers without stop(). Record which path was taken in the contender record (e.g. rec['terminated'] vs a rec['stopped'] field, or a teardown_action field) so manifests can see it. Keep the existing exception-swallowing behavior (a raising stop()/terminate() must not sink the race; record a *_failed detail). Preserve the terminate_losers=False short-circuit.

Constraints: keep provider backends swappable — only call stop() when hasattr(provider, 'stop'); do not assume every provider has it. render/launch outputs are simulator support NOT policy-success claims. protect provenance/rights/privacy/raw-capture-truth. add/extend tests. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: in tests/test_provider_race.py add a FakeProvider variant that returns mode='warm_restart' on launch and records stop() calls separately from terminate(); assert a warm loser is stop()ed not terminate()d, and a cold loser (mode='fake_cold') is terminate()d not stop()ed. Add a test where stop() raises and assert the race still returns launched with a *_failed detail recorded. Run python -m pytest tests/test_provider_race.py and python -m py_compile on touched files.
```

</details>

### [P0-17] Wire race_launch into the G1 kitchen parity launch path

- **Priority:** P0 · **Effort:** L · **Dimension:** provider_race orchestrator
- **Goal:** When more than one provider is requested, route the G1 parity job through provider_race.race_launch instead of single-provider launch_with_marker_retry, forwarding the winner to watch_and_collect.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/provider_race.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/gpu_render_providers.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -q ; python -m pytest tests/test_provider_race.py -q ; python -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py src/blueprint_pipeline/provider_race.py

- **Context:** This is the highest-value wiring for the active 'open the refrigerator' G1 POV seed lane: that lane is what actually fires GPU renders, and the sequential RunPod->Vast failover (the documented motivating bug in provider_race.py:3-13) is still live in its launch path at isaac_g1_kitchen_parity_job.py:599. Wiring is entirely CPU/hermetic to build and validate — real launches are paid, but the orchestration, fake providers, and a stubbed watch_and_collect are fully mockable. Depends on the marker_check helper, the cold/fallback reconciliation, and the stop-vs-terminate teardown landing first (or being included here).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), wire src/blueprint_pipeline/provider_race.py:race_launch into run_isaac_g1_kitchen_parity_job in src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py. Today (line ~599) it always calls launch_with_marker_retry against a single provider built from the `provider` string; provider_race exists specifically to kill that sequential RunPod->Vast stall but has ZERO non-test callers.

Prerequisites (do these first or assume sibling tasks did them; if missing, implement minimally): (1) an importable boot-marker check helper to pass as marker_check; (2) race_launch able to express the warm/cold fallback semantics; (3) race_launch stop()-vs-terminate() loser teardown. Reference the canonical inline logic at isaac_g1_kitchen_parity_job.py:400-419.

Task: add an internal code path so that when the caller supplies MORE THAN ONE provider, run_isaac_g1_kitchen_parity_job builds the provider list (e.g. [get_render_provider('runpod', warm_candidates=warm_candidate_ids), get_render_provider('vast')]), constructs a request callable prov -> prov.build_request(spec, job_dir) (each provider has its own native body), and calls race_launch(providers, request_callable, marker_check=<helper bound to job_dir>, marker_timeout=marker_timeout, job_dir=job_dir, cold=cold, circuit_breaker=<optional>, ...). On a 'launched' result, hand res['winner_provider'] (the live object) and res['instance_id'] to the existing watch_and_collect(job_dir, render_out, launch['instance_id'], provider=res['winner_provider'], ...) at line ~626. Record the full race result (contenders, skipped, terminated_losers) into manifest['launch']. CRITICAL: keep the single-provider default path (and the serve=True branch) exactly as-is — only multi-provider requests take the race path. The allow_paid=False plan path must remain byte-for-byte unchanged in behavior (still returns status 'prepared' without launching).

Constraints: keep world-model/provider backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims (do not let manifest wording imply the robot succeeded at opening the fridge); add/extend tests; Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: add hermetic tests to tests/test_isaac_g1_kitchen_parity_runner.py: (a) a multi-provider call with allow_paid=False still returns status 'prepared' and never launches (assert no provider.launch invoked); (b) a multi-provider call with allow_paid=True, fake providers (modeled on FakeProvider in test_provider_race.py) and a monkeypatched watch_and_collect, asserts race_launch is invoked and the winning provider/instance_id is forwarded to watch_and_collect; (c) the single --provider path is byte-for-byte unchanged (still calls launch_with_marker_retry). Run python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_provider_race.py and python -m py_compile on touched files.
```

</details>

## Spend guard & pod lifecycle

### [P0-18] Fix spend guard reaping STOPPED warm-reuse pods

- **Priority:** P0 · **Effort:** M · **Dimension:** Spend guard & pod lifecycle
- **Goal:** Stop gpu_spend_guard from classifying deliberately-STOPPED warm-pool RunPod pods as reapable duds so --reap can never delete the warm pool.
- **Files:** `scripts/gpu_spend_guard.py`, `src/blueprint_pipeline/isaac_particlefield_render_job.py`, `tests/test_gpu_spend_guard.py`
- **Validate (CPU):** python3 -m pytest tests/test_gpu_spend_guard.py -k 'stopped or reap or protected' -q ; add a test asserting `_parse_runpod_pod({'desiredStatus':'STOPPED','runtime':None,...}, now=...)` with large age is NOT reapable; add a test asserting each DEFAULT_WARM_CANDIDATES id is protected/non-reapable even with empty process_cmdlines. Then full file: python3 -m pytest tests/test_gpu_spend_guard.py -q and python3 -m py_compile scripts/gpu_spend_guard.py

- **Context:** scripts/gpu_spend_guard.py is the per-minute cost watchdog that lists live RunPod/Vast pods and, with --reap, terminates orphaned not-booted duds. The warm-reuse cost strategy (warm pods kept STOPPED for cheap restart, ids in src/blueprint_pipeline/isaac_particlefield_render_job.py DEFAULT_WARM_CANDIDATES lines 36-39) is exactly what this bug would destroy: the tool meant to SAVE money would delete the user's warm pods, forcing expensive cold pulls afterward. This directly threatens the active 'open the refrigerator' G1 POV seed render lane, which depends on warm-restart pods to avoid the ~10.7GB cold image pull. The parse logic is at lines 140-178, RUNPOD_TERMINAL_STATUSES at line 54, is_reapable at lines 422-438, protected-id scan find_protected_pod_ids at lines 363-394.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), fix a confirmed critical spend bug in scripts/gpu_spend_guard.py: the guard mis-classifies deliberately STOPPED warm-reuse RunPod pods as reapable orphans and a `--reap` run with no live owning process would DELETE the user's entire warm pool.

Confirmed reproduction (run it yourself first): `_parse_runpod_pod({'id':'pwbu7wxsvxpr0x','name':'warm','desiredStatus':'STOPPED','runtime':None,'costPerHr':0.79,'createdAt':'2026-01-01T00:00:00Z'}, now=<large>)` yields `state='stopped', booted=False, live=True, age=<large>` and `is_reapable(...)` returns True. Root cause: RUNPOD_TERMINAL_STATUSES (scripts/gpu_spend_guard.py line 54) is only {EXITED, TERMINATED, TERMINATING}, so a `desiredStatus='STOPPED'` pod with `runtime=None` parses to `live=True, booted=False` (lines 140-178) and, once older than --max-boot-seconds, becomes a reap candidate (is_reapable, lines 422-438).

Required fix:
1. Make a STOPPED / paused / never-RUNNING RunPod pod NOT reapable. A pod is only an 'orphaned not-booted dud' when it is a genuinely-allocated, actively-booting pod (desiredStatus RUNNING, runtime still absent) past the boot threshold. Add a clear notion (e.g. a `stopped`/non-booting state classification, or a `reap_eligible` flag) so is_reapable only fires on allocated-and-booting pods, never on STOPPED ones. Keep the existing EXITED/TERMINATED behavior (already not live, already excluded).
2. Unconditionally protect the warm-candidate ids. Import or reference the warm-pool ids from src/blueprint_pipeline/isaac_particlefield_render_job.py DEFAULT_WARM_CANDIDATES (lines 36-39) — but DO NOT add a heavy import of the render-job module into the standalone guard if it would pull GPU/network code; prefer a small shared constant or a defensive local copy with a comment pointing back to the source of truth. Treat every warm-candidate id as protected/non-reapable even when no live owner process references it.

Constraints: keep world-model backends swappable (this is provider-lifecycle tooling, do not couple it to any one world-model backend); protect provenance/rights/privacy/raw-capture-truth; render and reap outputs are simulator/cost-control support, NOT policy-success claims; secrets remain file-based under ~/.blueprint-secrets and are never logged. Add/extend hermetic tests in tests/test_gpu_spend_guard.py (canned JSON, no network, no real secrets). Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done run: `python3 -m pytest tests/test_gpu_spend_guard.py -q` and `python3 -m py_compile scripts/gpu_spend_guard.py`.
```

</details>

### [P0-19] Terminate (not stop) timed-out RunPod renders to stop disk billing

- **Priority:** P0 · **Effort:** M · **Dimension:** Spend guard & pod lifecycle
- **Goal:** Escalate a watch_and_collect timeout that never reached runner_done to terminate() so a hung pod's 140GB+ container disk stops billing, even under preserve_instance=True.
- **Files:** `src/blueprint_pipeline/isaac_particlefield_render_job.py`, `src/blueprint_pipeline/gpu_render_providers.py`, `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `tests/test_gpu_render_providers.py`, `tests/test_isaac_particlefield_render_job.py`
- **Validate (CPU):** Add a test: drive watch_and_collect with a fake provider exposing both stop() and terminate(), monkeypatch job.time.time/sleep and job.urllib.request.urlopen to return a heartbeat zip whose bootstrap.json phase != 'runner_done' until max_seconds elapses; assert provider.terminate was called (not stop) and stop was NOT called, including with preserve_instance=True. Run: python3 -m pytest tests/test_gpu_render_providers.py -k 'timeout or terminate' -q ; then python3 -m pytest tests/test_gpu_render_providers.py tests/test_isaac_particlefield_render_job.py -q and python3 -m py_compile src/blueprint_pipeline/isaac_particlefield_render_job.py src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py

- **Context:** watch_and_collect is the shared poll/collect/teardown loop for both the splat render job and the G1 kitchen parity job (the active 'open the refrigerator' POV seed lane). A hung render is the most expensive failure mode: after the parent gives up, a merely-stopped RunPod pod keeps billing its 140GB+ container disk indefinitely. The teardown selection is at src/blueprint_pipeline/isaac_particlefield_render_job.py lines 351-360; preserve_instance is wired True for the parity job at src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py line 627; terminate-is-DELETE rationale is at src/blueprint_pipeline/gpu_render_providers.py lines 232-239. Existing fake-provider teardown tests live in tests/test_gpu_render_providers.py lines 246-444.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), fix a real billing leak in src/blueprint_pipeline/isaac_particlefield_render_job.py watch_and_collect (lines 296-364). When the poll loop reaches max_seconds WITHOUT ever observing bootstrap.phase=='runner_done' (done stays False) but at least one heartbeat/console/boot artifact arrived, `runner_started = bool(done or last or last_boot or last_console_tail)` (line 352) is True, so teardown uses `provider.stop(instance_id)` (line 357) instead of `provider.terminate(...)`. For RunPod a stopped pod still bills for its 140GB+ container disk (see RunPodRenderProvider.terminate docstring, src/blueprint_pipeline/gpu_render_providers.py lines 232-239). A render that hangs past max_seconds is precisely the runaway case and must be terminated, not preserved. The G1 parity job makes it worse by always passing preserve_instance=True (src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py line 627), which forces stop() even on a no-output timeout.

Required fix:
1. Distinguish a clean finish (done==True, runner_done seen) from a timeout (loop exhausted max_seconds with done==False). On a timeout where runner_done was never seen, escalate to provider.terminate(instance_id) — do NOT preserve. Preserve-for-warm-reuse via stop() should only apply to a run that actually reached runner_done (or, at minimum, never to a run that timed out without completing).
2. Make preserve_instance NOT force-preserve a never-completed (timed-out, no runner_done) run. preserve_instance should still allow stop() for a run that genuinely produced a runner_done marker, but a timeout with no completion is a terminate.
3. Add a deterministic field to the returned dict (e.g. teardown reason / timed_out flag) so callers and tests can assert the escalation happened.

Keep the existing covered behaviors green: completed run -> stop() preserve (test_watch_and_collect_stops_successful_pod_for_warm_reuse), blocked-but-runner_done -> stop() (test_watch_and_collect_stops_blocked_runner_pod_for_warm_reuse), no-output dud -> terminate (test_watch_and_collect_tears_down_via_provider), stale-before-runner_done ignored (test_watch_and_collect_ignores_stale_result_before_runner_done).

Constraints: keep world-model backends swappable (teardown stays provider-parameterized via provider.stop/terminate; do not hardcode RunPod); protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; secrets file-based and never logged. Add/extend hermetic tests in tests/test_gpu_render_providers.py and/or tests/test_isaac_particlefield_render_job.py using a fake provider + monkeypatched time/urlopen. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done run: `python3 -m pytest tests/test_gpu_render_providers.py tests/test_isaac_particlefield_render_job.py -q`, `python3 -m py_compile src/blueprint_pipeline/isaac_particlefield_render_job.py src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`.
```

</details>

## Dev env & deps

### [P0-20] Declare usd-core (pxr) as a no-GPU dependency extra and add it to dev

- **Priority:** P0 · **Effort:** S · **Dimension:** Dev env & deps
- **Goal:** Add a CPU-only usd-core extra providing pxr so USD-replay and dry-render tests stop silently skipping in the project venv.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/pyproject.toml`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_local_render_preview.py`, `$HOME/workspace/BlueprintCapturePipeline/scripts/run_isaac_g1_kitchen_parity_eval.py`, `$HOME/workspace/BlueprintCapturePipeline/uv.lock`
- **Validate (CPU):** `.venv/bin/python -c 'from pxr import Usd, UsdGeom, Gf; import PIL; print("ok")'` succeeds; then `.venv/bin/python -m pytest tests/test_local_render_preview.py -q` shows the 10 previously-skipped pxr tests now PASS (0 skips attributable to missing pxr). No GPU, no cloud.

- **Context:** pxr is the single most valuable no-GPU guard: it catches G1 placement/camera/POV-framing bugs without a GPU and is required by the active 'open the refrigerator' G1 POV seed lane's dry-render. Files: $HOME/workspace/BlueprintCapturePipeline/pyproject.toml (optional-dependencies block at line 37; dev extra at line 64), $HOME/workspace/BlueprintCapturePipeline/tests/test_local_render_preview.py, $HOME/workspace/BlueprintCapturePipeline/scripts/run_isaac_g1_kitchen_parity_eval.py (_open_stage_local at line 6698 imports pxr). Verified: .venv lacks pxr; only the unrelated system python3.9 has it but cannot import the project package or blueprint_contracts.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), pxr (OpenUSD Python bindings) is imported by 13+ tests and by the local dry-render path, but it is declared in NO dependency file (verified absent from pyproject.toml, uv.lock, requirements.txt, requirements-geometry.txt). The project interpreter at .venv/bin/python (Python 3.12) does NOT have pxr (`import pxr` -> ModuleNotFoundError), so tests/test_local_render_preview.py runs 7 passed / 10 skipped purely because pxr is missing. usd-core is the pip-installable, pure-CPU OpenUSD wheel that provides pxr.

Do this:
1. In pyproject.toml [project.optional-dependencies], add a new extra (name it `usd`) containing `usd-core>=24.0` (pick the latest version that resolves on Python 3.12). Also add `usd-core` to the `dev` extra so the canonical dev interpreter carries it.
2. Install it into .venv: `.venv/bin/pip install 'usd-core>=24.0'` (do NOT rebuild the venv from scratch; just add the wheel).
3. Regenerate uv.lock if the repo uses uv-managed locking, so the new extra is reproducible.

Constraints: this is a pure-CPU dependency change. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only. Keep world-model backends swappable (do not couple usd-core to any one backend). Protect provenance/rights/privacy/raw-capture-truth (no changes to capture data). Render/dry-render outputs are simulator support, NOT policy-success claims. Add/extend tests if you change any import-guard logic. Run `python -m py_compile` on any edited Python and `python -m pytest tests/test_local_render_preview.py`.
```

</details>

### [P0-21] Make the --dry-render CLI actually runnable end-to-end on .venv

- **Priority:** P0 · **Effort:** M · **Dimension:** Dev env & deps
- **Goal:** After pxr lands in .venv, prove the local --dry-render path runs cold on the real kitchen USD and writes its PNG + summary JSON with no GPU.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/scripts/run_isaac_g1_kitchen_parity_eval.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_local_render_preview.py`, `$HOME/workspace/BlueprintCapturePipeline/output/isaac_g1_dynamic_standing_contact_floor_asset/Collected_KitchenRoom/KitchenRoom.usd`
- **Validate (CPU):** `.venv/bin/python -m pytest 'tests/test_local_render_preview.py::test_dry_render_cli_runs_end_to_end_on_real_kitchen' -q` runs (does NOT skip) and passes; a manual `.venv/bin/python scripts/run_isaac_g1_kitchen_parity_eval.py --dry-render ...` (with valid scenarios + --kitchen-usd pointing at the real KitchenRoom.usd, --out-dir /tmp/dryrun) writes /tmp/dryrun/dry_render/dry_render_preview.png and dry_render_summary.json. No GPU.

- **Context:** The user's memory notes --dry-render is the ~7s local iterate loop before firing ONE paid cloud render, and it is the single biggest no-GPU capability currently broken by environment alone. The real kitchen asset is present at $HOME/workspace/BlueprintCapturePipeline/output/isaac_g1_dynamic_standing_contact_floor_asset/Collected_KitchenRoom/KitchenRoom.usd (592 bytes, a Collected stub). This directly supports the active 'open the refrigerator' G1 POV seed lane. Files: $HOME/workspace/BlueprintCapturePipeline/scripts/run_isaac_g1_kitchen_parity_eval.py, $HOME/workspace/BlueprintCapturePipeline/tests/test_local_render_preview.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), the local --dry-render path in scripts/run_isaac_g1_kitchen_parity_eval.py requires pxr AND PIL AND the blueprint_pipeline package in ONE interpreter. No current interpreter satisfies all three: .venv (3.12) has PIL+project+boto3 but no pxr; system python3.9 has pxr but no PIL and cannot import the project; homebrew 3.13 has neither. The canonical interpreter is .venv. Once usd-core is installed into .venv (see the usd-core extra task), the --dry-render branch (args.dry_render at scripts/run_isaac_g1_kitchen_parity_eval.py line 7164; _draw_dry_render_preview imports PIL at line 6780; _open_stage_local reads pxr at line 6698) should run end-to-end.

Do this:
1. Confirm .venv has pxr+PIL (install usd-core if not already done — do it via pip into .venv, do not rebuild).
2. Run the --dry-render CLI against the real KitchenRoom.usd and verify it writes dry_render_preview.png and dry_render_summary.json. The dry_render branch needs scenarios AND a kitchen_usd or it returns status=blocked with blocker missing_scenarios_or_kitchen_usd — inspect the CLI arg parser (lines ~7059-7134) and the branch (lines ~7155-7180) to assemble a valid invocation (it accepts a request file/args that supply scenarios plus --kitchen-usd). Use the existing real-kitchen test as the reference for how scenarios are supplied.
3. Un-skip tests/test_local_render_preview.py::test_dry_render_cli_runs_end_to_end_on_real_kitchen by ensuring the env it needs is present; do not weaken the test's assertions.

Constraints: Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only. Keep world-model backends swappable. Protect provenance/rights/privacy/raw-capture-truth. Dry-render outputs are simulator support, NOT policy-success claims — do not let any code label a dry-render as a task success. Add/extend tests for any code path you touch. Run `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py` and `python -m pytest tests/test_local_render_preview.py`.
```

</details>

### [P0-22] Move boto3/botocore out of the cloud-only extra so staging stops failing pre-pod

- **Priority:** P0 · **Effort:** S · **Dimension:** Dev env & deps
- **Goal:** Add boto3/botocore to the dev (and default-installable) dependency set so the object-store staging subprocess can run from the canonical interpreter and be exercised/mocked on CPU.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/pyproject.toml`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_particlefield_render_job.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/wam_provider_object_store.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/robot_eval_provider_input_setup.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `$HOME/workspace/BlueprintCapturePipeline/uv.lock`
- **Validate (CPU):** `.venv/bin/python -c 'import boto3, botocore; print(boto3.__version__)'` works; `.venv/bin/python -m pytest tests/test_wam_provider_object_store.py -q` passes (3 tests collect) and exercises staging with mocked S3. Optionally rebuild a throwaway venv with `pip install -e .[dev]` and assert `import boto3` succeeds — proving the documented command is now sufficient. No GPU, no spend.

- **Context:** This is the documented root cause of staging_failed without spending on GPU. Because stage_bundle uses sys.executable, the staging subprocess inherits whatever interpreter launched the job — half of the split-brain trap. Files: $HOME/workspace/BlueprintCapturePipeline/pyproject.toml, $HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_particlefield_render_job.py, $HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/wam_provider_object_store.py, $HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/robot_eval_provider_input_setup.py, $HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py. Relevant to the 'open the refrigerator' G1 lane because a boto3-less launcher kills the parity job before any render.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), stage_bundle in src/blueprint_pipeline/isaac_particlefield_render_job.py (lines ~194-213) shells out via sys.executable: `subprocess.run([sys.executable, '-m', 'blueprint_pipeline.wam_provider_object_store', ...])`, and wam_provider_object_store / robot_eval_provider_input_setup lazily `import boto3` (raising 'boto3 is required for s3://...'). boto3 is declared ONLY in the `cloud` extra of pyproject.toml (line ~40), NOT in core deps, NOT in `dev`, and NOT present in uv.lock — so the README-documented `uv sync --extra dev` does NOT install it. When the parity job runs under any interpreter lacking boto3, the object-store subprocess fails and the job appends staging_failed / kitchen_staging_failed (src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py lines ~543 and ~530) BEFORE any pod launches.

Do this:
1. Add `boto3>=1.34.0` and `botocore` to the `dev` extra in pyproject.toml (keep it in `cloud` too). The goal is that `uv sync --extra dev` reproduces boto3 in the canonical interpreter (.venv already happens to have boto3 1.43.36, but only by hand-assembly — make it declared).
2. Regenerate uv.lock so boto3 is locked.
3. Add or extend a test that exercises the staging code path with S3 mocked (use moto if available, else the repo's existing fakes/stubs in tests/test_wam_provider_object_store.py) — simulate an upload with NO real S3/R2 call.

Constraints: Do NOT launch any GPU or paid cloud pod; do NOT make any real S3/R2/network call; this is CPU/no-spend only. Keep world-model backends swappable (object-store provider must stay pluggable). Protect provenance/rights/privacy/raw-capture-truth (do not log or upload real capture bytes in the test). Render outputs are simulator support, NOT policy-success claims. Run `python -m py_compile` on edited files and `python -m pytest tests/test_wam_provider_object_store.py`.
```

</details>

### [P0-23] Declare mujoco as a CPU dependency extra and un-skip the MuJoCo lane

- **Priority:** P0 · **Effort:** S · **Dimension:** Dev env & deps
- **Goal:** Add a CPU-only mujoco extra so the preferred no-GPU MuJoCo-parity validation substrate can run locally instead of silently skipping.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/pyproject.toml`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/cpu_simulator_preflight.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/mujoco_worker_runtime_preflight.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/mujoco_g1_simulator_command.py`, `$HOME/workspace/BlueprintCapturePipeline/uv.lock`
- **Validate (CPU):** `.venv/bin/python -c 'import mujoco; print(mujoco.__version__)'` works; `.venv/bin/python -m pytest tests/test_mujoco_worker_runtime_preflight.py tests/test_mujoco_g1_simulator_command.py -q` runs the mujoco-gated tests instead of skipping for missing mujoco (tests still legitimately gated on BLUEPRINT_MUJOCO_G1_MODEL_ROOT may skip — that is expected). CPU-only.

- **Context:** CPU MuJoCo parity is the explicitly-preferred no-GPU validation substrate per the user's GPU-startup strategy and the MuJoCo-parity G1 eval lane. Without a declared mujoco, the runnable CPU simulator preflight and worker runtime preflight cannot execute their physics path without an undocumented manual install. Files: $HOME/workspace/BlueprintCapturePipeline/pyproject.toml, $HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/cpu_simulator_preflight.py, $HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/mujoco_worker_runtime_preflight.py, $HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/mujoco_g1_simulator_command.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), mujoco is imported lazily in src/blueprint_pipeline/cpu_simulator_preflight.py (line ~552), src/blueprint_pipeline/mujoco_worker_runtime_preflight.py (line ~81), and drives mujoco_g1_simulator_command / mujoco_g1_wam_vla_policy_endpoint_eval — all pure-CPU physics. Yet `mujoco` is declared in NO optional-dependency group (existing extras: cloud, runtime, llm, simulation-agents, retrieval, validation, dev) and is MISSING from .venv (`import mujoco` -> ModuleNotFoundError). 4 importorskip('mujoco') test files skip locally.

Do this:
1. In pyproject.toml, add a `sim` (or `mujoco`) extra containing `mujoco>=3.0`, and add `mujoco>=3.0` to the `dev` extra.
2. Install into .venv: `.venv/bin/pip install 'mujoco>=3.0'` (CPU wheel; do not rebuild the venv).
3. Regenerate uv.lock.
4. Note (do NOT implement here) that the physics tests additionally need the mujoco_menagerie G1 model root via BLUEPRINT_MUJOCO_G1_MODEL_ROOT — that asset fetch is a SEPARATE task; tests that need the model should remain gated on that env var, not on the mujoco import.

Constraints: Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only. Keep world-model backends swappable (MuJoCo is one parity backend, not the only one). Protect provenance/rights/privacy/raw-capture-truth. Render/sim outputs are simulator support, NOT policy-success claims. Run `python -m py_compile` on edited files and `python -m pytest tests/test_mujoco_worker_runtime_preflight.py tests/test_mujoco_g1_simulator_command.py`.
```

</details>

## Docs / provenance / claims

### [P0-24] Fix the false 'no uncommitted state' provenance line in CHANGELOG

- **Priority:** P0 · **Effort:** S · **Dimension:** Docs / provenance / claims
- **Goal:** Correct the 2026-06-28 CHANGELOG entry's self-claim that there is no uncommitted local state, which is false at write time.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/docs/CHANGELOG.md`
- **Validate (CPU):** Run `git status --short` (must show the 4 modified files the corrected line now enumerates). Run `git diff --stat scripts/run_isaac_g1_kitchen_parity_eval.py tests/test_isaac_g1_kitchen_parity_runner.py tests/test_local_render_preview.py docs/CHANGELOG.md`. Manually re-read docs/CHANGELOG.md lines ~60-72 to confirm the statement now matches the working tree. CPU only, no GPU/cloud.

- **Context:** The repo's CLAUDE.md/AGENTS.md make provenance and capture/state truth authoritative, and the autonomous-loop evidence checklist forbids false-completion / inaccurate state self-claims. A changelog asserting 'no uncommitted local state' while a 276-line script diff (the active 'open the refrigerator' G1 POV seed lane in scripts/run_isaac_g1_kitchen_parity_eval.py) sits uncommitted is exactly the overstatement the doctrine prohibits and will mislead the next agent about what has actually landed. Verified facts: `git status --short` shows the 4 files; the false line is at docs/CHANGELOG.md:69-70.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), the file docs/CHANGELOG.md has an entry dated `## 2026-06-28`. Its 'Future-Agent-Facing' section ends (lines ~69-70) with: 'Uncommitted local state: none found in the current checkout for June 28; this entry is based on committed history for the previous completed calendar day.' This statement is FALSE at the moment it is written. Run `git status --short` and you will see 4 modified-but-uncommitted files: docs/CHANGELOG.md itself, scripts/run_isaac_g1_kitchen_parity_eval.py (a large +276/-15 diff), tests/test_isaac_g1_kitchen_parity_runner.py (+88), and tests/test_local_render_preview.py (+16).

Task: Rewrite that 'Uncommitted local state' line so it truthfully enumerates the actual uncommitted working-tree changes (list the 4 files and, briefly, what each contains — e.g. the eval script gained G1 render-visibility / head-POV seed work, the two test files gained matching coverage, and the changelog itself is being edited). Do NOT touch the older 2026-06-27 / 2026-06-26 'Uncommitted local state' lines unless they are also factually wrong for their own dates (they are about prior dates — leave them). Keep the existing render-seed proof-boundary language intact.

Constraints: This is documentation-only. Keep world-model backends swappable; protect provenance, rights, privacy, and raw-capture truth; render outputs are simulator/review support, NOT policy-success, contact, safety, or deployment claims. Do NOT weaken or remove any existing proof-boundary caveat. Run `python3 -m py_compile` is not applicable to .md; instead re-read the edited section to confirm it matches `git status --short`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P0-25] Add a 2026-06-29 CHANGELOG entry covering today's 11 G1 render commits

- **Priority:** P0 · **Effort:** S · **Dimension:** Docs / provenance / claims
- **Goal:** Document the 11 commits dated today (2026-06-29) that the changelog currently skips, with the same render-seed proof boundary used in prior entries.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/docs/CHANGELOG.md`, `$HOME/workspace/BlueprintCapturePipeline/scripts/run_isaac_g1_kitchen_parity_eval.py`
- **Validate (CPU):** Run `git log --pretty='%ad %h %s' --date=short | grep 2026-06-29` and confirm every listed commit's theme is represented in the new entry. Run `grep -n '^## 2026' docs/CHANGELOG.md | head -2` and confirm the newest heading is now `## 2026-06-29`. Re-read the new entry to confirm the render-seed proof-boundary caveat is present. CPU only.

- **Context:** Today is 2026-06-29; a changelog whose newest entry is dated yesterday and silently skips today's 11 committed commits understates what changed and breaks the audit trail. These commits ARE the claim-sensitive G1 render-seed / head-POV lane the audit flags, so they need the explicit proof-boundary framing the rest of the changelog applies. Verified: `git log ... | grep 2026-06-29 | wc -l` = 11; newest `^## 2026` heading in docs/CHANGELOG.md is line 3 (`## 2026-06-28`). The proof-boundary language to mirror lives in scripts/run_isaac_g1_kitchen_parity_eval.py:799-815.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), the newest entry in docs/CHANGELOG.md is dated `## 2026-06-28`, but today is 2026-06-29 and `git log --pretty='%ad %h %s' --date=short | grep 2026-06-29` shows 11 commits authored today that are completely undocumented. They are (HEAD first): 8715581d 'Render G1 review proxies from link geometry', 6fd040fa 'Diagnose G1 render visibility', a6543a6b 'Bind visible robot material at G1 root', db7909c9 'Make robot visible in review renders', 5a581434 'Widen head POV and tighten arm visibility gate', c1917b38 'Limit G1 POV pitch and verify from side', a7c6f79a 'Clear warm RunPod output before reruns', 995821e9 'Keep G1 head lens behind forearms', 93bf2b3b 'Bound Isaac render step hangs', fbe72ecf 'Fix G1 head-forward manipulation seed', a6d749c5 'Refine G1 head POV seed framing'.

Task: Add a new `## 2026-06-29` entry at the TOP of docs/CHANGELOG.md (above the 2026-06-28 entry), following the existing entry format (User-Facing / Future-Agent-Facing sections as used in nearby entries). Summarize this work as G1 review-render visibility and head-POV manipulation-seed framing for the 'open the refrigerator' kitchen-parity lane. CRITICAL: reuse the existing render-seed proof-boundary framing — these are Isaac RTX review/render seeds and head-POV framing for review media; they are simulator/render SUPPORT, NOT manipulation success, object contact, physical reach, learned-policy success, safety, or deployment readiness. Mirror the proof-boundary wording already in scripts/run_isaac_g1_kitchen_parity_eval.py (e.g. lines ~799-815 and ~4087, ~5905) and in the prior CHANGELOG render-seed sentences. Include an accurate 'Uncommitted local state' note consistent with `git status --short` at the time you write it.

Constraints: Documentation-only. Keep world-model backends swappable; protect provenance/rights/privacy/raw-capture truth; do not overstate readiness. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P0-26] Run all new-module CPU test suites green and cite them as local proof

- **Priority:** P0 · **Effort:** S · **Dimension:** Docs / provenance / claims
- **Goal:** Confirm the no-GPU tests backing the new features pass locally and reference that green run as the durable CPU evidence for the doc/readiness updates.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/tests/test_scene_placement.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_placement_validation.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_perception_adapter.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_perception_fusion.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_perception_views.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_provider_race.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_render_lock.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_warm_render_server.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_gpu_spend_guard.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_local_render_preview.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** `python3 -m pytest tests/test_scene_placement.py tests/test_placement_validation.py tests/test_perception_adapter.py tests/test_perception_fusion.py tests/test_perception_views.py tests/test_provider_race.py tests/test_render_lock.py tests/test_warm_render_server.py tests/test_gpu_spend_guard.py tests/test_local_render_preview.py tests/test_isaac_g1_kitchen_parity_runner.py -q` exits 0 (all pass/skip, no failures/errors). `python3 -m pytest <paths> --collect-only -q` lists all 11 files. CPU only, no GPU, no network.

- **Context:** The audit goal is to finish and VALIDATE everything doable on CPU before any GPU spend. Grounding the doc/readiness claims in a concrete green local run confirms the --dry-render and scene_placement claims are real and reproducible at zero cloud cost. Verified locally: `python3 -m pytest` over test_scene_placement / test_provider_race / test_render_lock / test_warm_render_server / test_gpu_spend_guard / test_local_render_preview returned 169 passed, 3 skipped in 6.27s on Python 3.9.6. All 11 listed test files exist.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), validate that the no-GPU test coverage backing the new features passes locally, and cite it as the evidence for the doc/readiness updates. NOTE: this environment has NO `python` on PATH — use `python3` (Python 3.9.6). pytest config is in pyproject.toml.

Task:
1. Run `python3 -m pytest tests/test_scene_placement.py tests/test_placement_validation.py tests/test_perception_adapter.py tests/test_perception_fusion.py tests/test_perception_views.py tests/test_provider_race.py tests/test_render_lock.py tests/test_warm_render_server.py tests/test_gpu_spend_guard.py tests/test_local_render_preview.py tests/test_isaac_g1_kitchen_parity_runner.py -q` and capture the pass/skip counts. (A prior run of a subset returned 169 passed, 3 skipped in ~6s — the full set including placement_validation/perception/isaac_runner should also be green; investigate any failure before proceeding.)
2. Run `python3 -m pytest <those paths> --collect-only -q` to confirm discovery of every file.
3. The uncommitted +16 in tests/test_local_render_preview.py and +88 in tests/test_isaac_g1_kitchen_parity_runner.py MUST be included in the run.
4. Record the concrete pass counts and the exact command in the CHANGELOG (and/or README/READINESS_MATRIX) entries produced by the sibling doc tasks, as the CPU proof that these lanes work without GPU — replacing any vague 'covered by tests' prose with the real numbers.

Constraints: This is the CPU evidence-discipline step required by the autonomous-loop checklist — ground doc/readiness claims in a concrete green local run, not prose. Keep world-model backends swappable; protect provenance/rights/privacy/raw-capture truth; render outputs are simulator support, not policy success. If any test fails or errors, do NOT paper over it — report the failure. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## Launch gates & readiness

### [P0-27] Fix launch-gate bundle-readiness override that lets non-ready bundles pass

- **Priority:** P0 · **Effort:** S · **Dimension:** Launch gates & readiness
- **Goal:** Stop build_launch_gate_summary from forcing buyer_fulfillment_bundle_ready=True on mere file existence regardless of the bundle's own status.
- **Files:** `src/blueprint_pipeline/alpha_readiness.py`, `tests/test_alpha_readiness.py`
- **Validate (CPU):** .venv/bin/python -m pytest tests/test_alpha_readiness.py -q  &&  .venv/bin/python -m py_compile src/blueprint_pipeline/alpha_readiness.py  ;  new test must fail before the fix and pass after.

- **Context:** src/blueprint_pipeline/alpha_readiness.py build_launch_gate_summary is the buyer-fulfillment gate feeding the paid/external alpha launch verdict. Confirmed: lines 763-768 contain the override; the buyer_fulfillment_bundle_ready check is at lines 841-848. The existing test at tests/test_alpha_readiness.py:457 only exercises the ready-status happy path, so this false-pass is currently uncovered. Use .venv/bin/python (system python3.9 lacks PIL and other deps; .venv has them).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU/no-spend task. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Fix a genuine launch-gate logic bug in src/blueprint_pipeline/alpha_readiness.py, function build_launch_gate_summary. At lines ~763-766 it correctly computes:
  launchable_bundle_ready = bool(launchable_export_bundle and str(launchable_export_bundle.get('status') or '').strip().lower() in {'ready','launch_ready'})
but then lines ~767-768 unconditionally override it:
  if not launchable_bundle_ready and (eval_root / 'launchable_export_bundle.json').is_file():
      launchable_bundle_ready = True
This means a bundle the pipeline itself wrote with status='blocked', 'failed', or any non-ready value still passes the 'buyer_fulfillment_bundle_ready' stage check (the _check at lines ~841-848), which feeds all_stage_checks_passed and the external_beta_contract_ready / internal_only_contract_ready verdict. A bundle the pipeline marked not-ready would be reported as launch-ready — exactly the false-pass the evidence checklist exists to prevent.

Required change: remove the unconditional file-existence override so a present-but-non-ready bundle BLOCKS the gate. If there is a real legacy need to treat a status-less bundle as ready (bundles written before the status field existed), scope the fallback narrowly: only treat the on-disk bundle as ready when it has NO 'status' key at all (legacy), never when status is present and non-ready. Prefer the simplest correct fix; do not broaden behavior.

Constraints: keep world-model backends swappable; protect provenance, rights, privacy, and raw-capture truth; render/eval outputs are simulator support, NOT policy-success claims; do not weaken any other stage check. Add/extend tests. This change must satisfy the autonomous-loop evidence checklist at $HOME/workspace/Blueprint-WebApp/docs/autonomous-loop-evidence-checklist-2026-05-03.md before claiming done.

Add a regression test in tests/test_alpha_readiness.py that writes pipeline/evaluation_prep/launchable_export_bundle.json (the eval_root the function reads) with status='blocked', drives build_launch_gate_summary, and asserts checks['buyer_fulfillment_bundle_ready']['passed'] is False and that overall_status is 'blocked' (not a launchable verdict). Mirror the fixture setup from the existing happy-path test near tests/test_alpha_readiness.py:457. Add a second case where the on-disk bundle has NO status key, asserting whatever legacy behavior you chose, documented in a comment.

Then run: .venv/bin/python -m pytest tests/test_alpha_readiness.py -q and .venv/bin/python -m py_compile src/blueprint_pipeline/alpha_readiness.py
```

</details>

### [P0-28] Add a hermetic test suite for run_paid_marketplace_launch_gate.py

- **Priority:** P0 · **Effort:** M · **Dimension:** Launch gates & readiness
- **Goal:** Give the untested paid-marketplace launch gate unit coverage over its pure aggregation functions so a regression cannot silently upgrade a blocked source to contract-ready.
- **Files:** `scripts/run_paid_marketplace_launch_gate.py`, `tests/test_paid_marketplace_launch_gate.py`
- **Validate (CPU):** .venv/bin/python -m pytest tests/test_paid_marketplace_launch_gate.py -q  &&  .venv/bin/python -m py_compile scripts/run_paid_marketplace_launch_gate.py

- **Context:** scripts/run_paid_marketplace_launch_gate.py is the externally marketable paid-marketplace beta gate producing the per-source acceptance verdict and do-not-claim guardrails. Confirmed pure functions and line ranges: summarize_sources 253-327, build_claims 412-434, evidence_boundary 437-461, closeout_summary 464-491, should_skip 242-250, skip_evidence_class 87-98. The status-string concatenation across the three-source (iPhone/glasses/Android) matrix is precisely where a regression would silently upgrade a blocked source. Test loader pattern lives in tests/test_external_alpha_launch_gate.py:12. Use .venv/bin/python (system python3.9 lacks deps).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU/no-spend task. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

scripts/run_paid_marketplace_launch_gate.py (632 lines) has NO dedicated test file. Create tests/test_paid_marketplace_launch_gate.py covering its pure, side-effect-free functions. Import the module via importlib (mirror the loader in tests/test_external_alpha_launch_gate.py:12, _load_gate_module, which uses importlib.util.spec_from_file_location on the script path). Do NOT call run_command, run subprocesses, touch the network, or use a GPU — all target functions take in-memory CommandResult lists.

The CommandResult dataclass (defined at line 36) has fields in this order: id, label, repo, command, cwd, status, blocking, source_tags, exit_code=None, stdout_tail='', stderr_tail='', skip_reason=None, evidence_class=None, evidence_note=None. Build fixtures accordingly.

Cover at minimum:
- summarize_sources (lines 253-327): (a) all webapp+capture+pipeline results status='passed' and ios result status='passed' -> iPhone status 'external_beta_contract_ready' (no manual suffix), glasses/Android 'internal_only_contract_ready'; (b) ios result status='manual_required' -> iPhone status gets the '_manual_device_confirmation_required' suffix; (c) one blocking webapp result status='failed' -> iPhone/glasses/Android all 'blocked'; (d) Android result status='manual_required' with evidence_class='operator_toolchain_required' (and webapp/capture/pipeline passed) -> android_status 'internal_only_contract_ready_operator_toolchain_evidence_required' and the operator-toolchain automated_claim wording; (e) Android manual_required without that evidence_class -> '...manual_bundle_confirmation_required'.
- build_claims (lines 412-434): a blocking failed result -> justified == [] and not_justified contains the 'Do not claim the paid marketplace beta gate passes...' line; all-pass -> non-empty justified list.
- evidence_boundary (lines 437-461): results with evidence_class='operator_toolchain_required' appear in operator_toolchain_evidence with id/label/repo/reason/note keys.
- closeout_summary (lines 464-491): given a report dict with automated_checks (list of {'label','status'}) and manual_checks (list of {'id','status'}), automated_contracts_prove lists only passed labels and remaining_manual_evidence_ids lists only ids whose status startswith 'manual_'.
- should_skip (lines 242-250) and skip_evidence_class (lines 87-98): construct CommandSpec fixtures and assert the Android-SDK-missing and xcodebuild-missing branches return the expected reason / evidence-class tuple. For env-dependent branches, monkeypatch os.environ (e.g. delete ANDROID_HOME/ANDROID_SDK_ROOT) so the test is deterministic regardless of the runner's shell.

Constraints: keep world-model backends swappable; protect provenance, rights, privacy, and raw-capture truth; these gate outputs are contract/support claims, NOT live-payment or live-device success claims — do not assert any wording that implies otherwise. Tests must be fully hermetic. This work must satisfy $HOME/workspace/Blueprint-WebApp/docs/autonomous-loop-evidence-checklist-2026-05-03.md before claiming done.

Then run: .venv/bin/python -m pytest tests/test_paid_marketplace_launch_gate.py -q and .venv/bin/python -m py_compile scripts/run_paid_marketplace_launch_gate.py
```

</details>

## Warm render transport / object store

### [P0-29] Run-scope warm results so poll_result cannot return a prior run's colliding-id result

- **Priority:** P0 · **Effort:** M · **Dimension:** Warm render transport / object store
- **Goal:** Make WarmPoolClient.poll_result reject stale warm_results/<request_id>.json left in the cumulative output zip from a previous warm rerun.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/warm_render_server.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_warm_render_server.py`
- **Validate (CPU):** python -m pytest tests/test_warm_render_server.py -q (add a new test: build an in-memory zip whose warm_results/job-1.json is a STALE result lacking this session's token; construct a fresh WarmPoolClient with http_get=lambda u: <that zip bytes>; assert poll_result('job-1', timeout_s small, sleep=lambda s: None) returns None instead of the stale payload; keep existing test_warm_pool_client_poll_result_reads_from_output_zip green by stamping the matching token). Also run `python -m py_compile src/blueprint_pipeline/warm_render_server.py`.

- **Context:** This protects the active 'open the refrigerator' G1 POV seed lane: the warm pod stays RUNNING across reruns (run_isaac_g1_kitchen_parity_job serve mode in src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py:607-624), and a stale warm_results/job-1.json would make the control plane believe a fridge-POV render completed when the real render never ran — the worst failure on a billed lane because it masks a no-op. Key file: src/blueprint_pipeline/warm_render_server.py (WarmPoolClient.submit line 225, poll_result line 237; SignedUrlJobSource.publish_result line 203; serve_render_loop result round-trip line 91). The one-shot equivalent fix lives at src/blueprint_pipeline/wam_provider_object_store.py:281.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). This is a CPU-only, no-spend hermetic transport bug fix.

Problem: In src/blueprint_pipeline/warm_render_server.py, WarmPoolClient.poll_result (lines 237-252) downloads the warm pod's CUMULATIVE /workspace/out output zip and returns the JSON at key `warm_results/<request_id>.json` the instant that key exists. WarmPoolClient._seq resets to 0 on every new client (line 223) so request_ids default to job-1, job-2 again; SignedUrlJobSource.publish_result (lines 203-204) writes warm_results/<request_id>.json into the persistent out dir which the heartbeat keeps re-uploading and which is never cleared across warm reruns. So a fresh control-plane client submitting job-1 to a still-warm pod will, before the pod renders, find the PRIOR run's warm_results/job-1.json already in the uploaded zip and return it immediately as a falsely-completed render. This is the same stale-result class already fixed for the one-shot output_key (delete_object at src/blueprint_pipeline/wam_provider_object_store.py:281, tested at tests/test_wam_provider_object_store.py:109-111) but left unfixed for the warm channel.

Fix: add run-scoped result freshness. Concretely, give WarmPoolClient a per-session nonce/token (generated at construction, injectable for tests) that it stamps into the submitted job payload (alongside seq/request_id/scenario in WarmPoolClient.submit, line 225-230). Have SignedUrlJobSource carry that token through into the published result (serve_render_loop already round-trips request_id into the result at line 91; thread the submit token the same way so publish_result writes it). Then make poll_result REQUIRE the result it reads to carry a token matching THIS client's session (and/or a submit-seq >= the submitted seq); when the key exists but the token does not match, treat it as 'not ready yet' and keep waiting rather than returning the stale payload. Keep the injectable http_get/http_put/clock/sleep signature unchanged so existing tests still pass. Do not change the on-GPU render path semantics.

Constraints: keep world-model backends swappable (do not couple this to any specific object store or Isaac); the JobSource Protocol and injected render_one must stay the transport boundary. Protect provenance/rights/privacy/raw-capture-truth — do not log or persist raw presigned URLs or secrets; redaction behavior must be preserved. Remember render outputs are simulator support, NOT policy-success claims — the result token is a freshness guard, not a success signal. Add/extend tests. Run `python -m pytest tests/test_warm_render_server.py -q` and `python -m py_compile src/blueprint_pipeline/warm_render_server.py`.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P0-30] Surface presigned-URL expiry (401/403) instead of swallowing it as 'no job yet'

- **Priority:** P0 · **Effort:** M · **Dimension:** Warm render transport / object store
- **Goal:** Classify HTTP 401/403 on the inbox GET and output GET as expired/forbidden so an expired 12h presigned URL becomes a visible blocker, not an invisible warm-pod hang.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/warm_render_server.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/wam_provider_object_store.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_warm_render_server.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_wam_provider_object_store.py`
- **Validate (CPU):** python -m pytest tests/test_warm_render_server.py tests/test_wam_provider_object_store.py -q (add: inject http_get=lambda u: (_ for _ in ()).throw(urllib.error.HTTPError(u,403,'Forbidden',{},None)); assert SignedUrlJobSource.poll and WarmPoolClient.poll_result classify it distinctly — raise/surface expired-or-forbidden — instead of returning None; assert a 404 HTTPError still yields None. Assert presign manifests carry expires_at and an expiry warning when TTL is short). Also `python -m py_compile src/blueprint_pipeline/warm_render_server.py src/blueprint_pipeline/wam_provider_object_store.py`.

- **Context:** Named KNOWN issue (kitchen bundle presigned-URL 12h TTL handling). On the active 'open the refrigerator' warm lane this converts a recoverable credential refresh into a silent hang that wastes billed warm-pod time until a human notices — exactly the failure the warm strategy exists to avoid. Files: src/blueprint_pipeline/warm_render_server.py (poll 182, poll_result 243), src/blueprint_pipeline/wam_provider_object_store.py (presign 168/396), src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py (_await_warm_serve_ready 458). Test scaffolding already exists: tests/test_warm_render_server.py uses lambda http_get/http_put fakes; tests/test_wam_provider_object_store.py uses a FakeClient/SimpleNamespace boto3 stub.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU-only, no-spend hermetic fix.

Problem: presigned URLs are minted with ExpiresIn=12h (src/blueprint_pipeline/wam_provider_object_store.py:168 staging, :396 warm inbox). A warm pod is designed to stay alive across many reruns, so URLs can expire mid-life. Today every failure collapses to 'no job yet': SignedUrlJobSource.poll (src/blueprint_pipeline/warm_render_server.py:182-186) catches all exceptions and returns None; WarmPoolClient.poll_result (lines 243-250) catches all and retries to timeout; _await_warm_serve_ready (src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py:458-459) swallows it as 'mid-upload'. A 404 (genuinely empty inbox) is indistinguishable from a 401/403 (expired/forbidden URL), so an expired credential turns into an invisible hang that burns billed warm-pod time.

Fix (all hermetic, no network): the default _http_get_bytes/_http_put_bytes use urllib (warm_render_server.py:151-158) which raises urllib.error.HTTPError with a .code. (1) In SignedUrlJobSource.poll and WarmPoolClient.poll_result, distinguish HTTP status: a 404/empty body stays 'no job yet' (return None / keep waiting), but a 401 or 403 must be surfaced — raise a typed/sentinel error or set a classified state the caller can read (e.g. raise a small custom exception class like PresignedUrlExpired, or record last_error='presigned_url_expired_or_forbidden'). Catch only urllib.error.HTTPError to inspect .code; keep generic connection errors as 'no job yet'. (2) In presign manifests (stage_wam_provider_bundle_object_store and presign_warm_inbox_channel in wam_provider_object_store.py), record an `expires_at` (computed from expiration_seconds + generated_at / utc_now_iso) and an `expiry_warning` flag when the remaining TTL is short; do NOT log raw URLs. (3) In _await_warm_serve_ready, propagate a distinct 'presigned_url_expired_or_forbidden' reason on 401/403 rather than looping to serve_ready_timeout.

Constraints: keep world-model backends swappable; do not hardcode any one object store. Protect provenance/rights/privacy — never log or persist the raw presigned URL or query string; only redacted URLs and expires_at metadata. Render outputs are simulator support, NOT policy-success claims. Add/extend tests. Run `python -m pytest tests/test_warm_render_server.py tests/test_wam_provider_object_store.py -q` and `python -m py_compile src/blueprint_pipeline/warm_render_server.py src/blueprint_pipeline/wam_provider_object_store.py`.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P0-31] Bound SignedUrlJobSource.poll failures so a persistently-broken inbox fails fast instead of polling forever

- **Priority:** P0 · **Effort:** S · **Dimension:** Warm render transport / object store
- **Goal:** Add error classification + a consecutive-failure counter to SignedUrlJobSource.poll / serve loop so a forbidden/malformed inbox surfaces a blocker rather than looping silently on a billed pod.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/warm_render_server.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_warm_render_server.py`
- **Validate (CPU):** python -m pytest tests/test_warm_render_server.py -q (add: an http_get that always raises a 403 HTTPError; assert poll increments a failure counter and after N calls surfaces a classified state; drive serve_render_loop with that source and assert it logs the blocker and exits with the new exit_reason rather than only on idle_timeout. Assert a 404/empty inbox does NOT trip the counter). Also `python -m py_compile src/blueprint_pipeline/warm_render_server.py`.

- **Context:** Silent indefinite polling on a billed warm pod is the costliest, hardest-to-notice failure on the active 'open the refrigerator' lane. Distinguishing transient-empty from persistent-broken lets the pod fail fast and free the GPU. This also makes the TTL-expiry finding actionable end to end. File: src/blueprint_pipeline/warm_render_server.py (SignedUrlJobSource.poll 182, serve_render_loop 41). Test scaffolding: tests/test_warm_render_server.py already drives SignedUrlJobSource and serve_render_loop with lambda http_get fakes and a _FakeSource.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU-only, no-spend hermetic fix.

Problem: SignedUrlJobSource.poll (src/blueprint_pipeline/warm_render_server.py:182-201) wraps the GET in a bare `except` returning None (185-186) and json.loads in another bare `except` returning None (191-192). A 403 (expired URL), a 500, a truncated body, and a genuinely-absent 404 inbox all collapse to 'no job yet', so the pod polls forever with no error surfaced, no counter, no give-up; the control plane only learns via the pod's idle_timeout exit. serve_render_loop (lines 41-94) already has a `_log` hook but poll has no logger and no failure signal.

Fix: add a consecutive-failure counter inside SignedUrlJobSource (reset to 0 on any successful GET, including a clean empty/404 which is a normal 'no job'). Distinguish transient-empty (404/empty body) from persistent-broken (repeated 401/403/5xx or repeated malformed JSON). After N consecutive HARD failures (configurable, default e.g. 10), surface a classified blocker — expose it on the source (e.g. a property the serve loop checks) and have serve_render_loop emit it via `_log` and exit with a new exit_reason like 'inbox_unrecoverable' instead of spinning until idle_timeout. Keep the happy path and existing return-None-when-empty behavior identical so current tests pass; keep all I/O injected (no real network). This should compose with the 401/403 classification work — if that lands first, reuse the typed expiry error here.

Constraints: keep world-model backends swappable (JobSource Protocol stays the boundary). Protect provenance/rights/privacy — do not log raw URLs or secrets; log only classified reasons. Render outputs are simulator support, NOT policy-success claims. Add/extend tests. Run `python -m pytest tests/test_warm_render_server.py -q` and `python -m py_compile src/blueprint_pipeline/warm_render_server.py`.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P0-32] Require an instance/session nonce in bootstrap/serve-ready markers so a stale marker can't satisfy the gate

- **Priority:** P0 · **Effort:** M · **Dimension:** Warm render transport / object store
- **Goal:** Make launch_with_marker_retry and _await_warm_serve_ready verify the marker belongs to THIS launch's instance_id instead of accepting mere presence of bootstrap.json / warm_serve_ready.json in the reused cumulative zip.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_isaac_g1_kitchen_parity_job.py`
- **Validate (CPU):** python -m pytest tests/test_isaac_g1_kitchen_parity_job.py -q (extend the marker-retry tests near line 375: make the fake urlopen return a zip whose bootstrap.json has a WRONG/absent instance_id; assert launch_with_marker_retry does NOT set marker_seen and treats the pod as flaky; add a _await_warm_serve_ready test where warm_serve_ready.json carries the wrong instance_id and assert ready is False, and a matching-instance_id case returns ready True). Also `python -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`.

- **Context:** The marker protocol is the only guard between 'pod genuinely booted Isaac+scene for the fridge-POV render' and paying for a dud — the ~50% cold-flaky problem the retry logic catches. A stale marker on the reused output key defeats it on both the warm lane and cold retries, re-introducing false-completed runs on the active 'open the refrigerator' seed lane. Files: src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py (launch_with_marker_retry 401, _await_warm_serve_ready 424), output-key reuse at src/blueprint_pipeline/wam_provider_object_store.py:264. Existing tests at tests/test_isaac_g1_kitchen_parity_job.py:375-427 use _make_fake_provider() whose fake urlopen writes a zip with bootstrap.json (currently empty '{}').

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU-only, no-spend hermetic fix.

Problem: launch_with_marker_retry (src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py:401-409) treats the mere presence of 'bootstrap.json' in the downloaded cumulative output zip as proof THIS pod started, and _await_warm_serve_ready (lines 443-457) treats presence of 'warm_serve_ready.json' as serve-ready and reads bootstrap.json's `phase` but never checks the marker belongs to launch['instance_id']. Because the output key is reused per job_dir (not run-unique — src/blueprint_pipeline/wam_provider_object_store.py:264) and the out dir is never cleared, a NEW launch against an out dir still holding a prior session's bootstrap.json will see the marker immediately and declare marker_seen / ready without the new container booting — defeating the ~50% cold-flaky guard the retry logic exists for, and re-introducing false-completed-run on the warm lane.

Fix: thread the launch's instance_id (and/or a per-launch session nonce) into the marker check. The worker writes bootstrap.json / warm_serve_ready.json containing an `instance_id` (or session nonce) field; the poller must parse the marker JSON and ACCEPT it only when that field matches the expected launch['instance_id'] (or the nonce passed into this launch). In launch_with_marker_retry, after `iid = launch['instance_id']` (line 398), only set marker_seen when the parsed bootstrap.json's instance_id == iid. In _await_warm_serve_ready, only return ready when warm_serve_ready.json's instance_id == the instance_id argument. A marker with a wrong/absent instance_id must be ignored (keep polling), not accepted. Keep the existing fake-provider + monkeypatched urllib test pattern working; update the worker-side marker writer if it lives in this repo so emitted markers include instance_id (search for where bootstrap.json / warm_serve_ready.json are written; if the writer is in the GPU bundle and not importable here, document the required field in a comment and gate only on the poller side).

Constraints: keep world-model backends swappable; do not couple marker logic to a specific provider. Protect provenance/rights/privacy — markers carry only an opaque instance_id/nonce and phase, never secrets or raw URLs. Render outputs are simulator support, NOT policy-success claims — the marker proves the pod booted, not that any task succeeded. Add/extend tests. Run `python -m pytest tests/test_isaac_g1_kitchen_parity_job.py -q` and `python -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## Code structure / tech debt

### [P0-33] Skip PIL tests when Pillow absent so the CPU suite goes green

- **Priority:** P0 · **Effort:** S · **Dimension:** Code structure / tech debt
- **Goal:** Replace the hard PIL import in the runner test with pytest.importorskip so the CPU suite reports 0 failed instead of 1 false-red.
- **Files:** `tests/test_isaac_g1_kitchen_parity_runner.py`, `tests/test_local_render_preview.py`
- **Validate (CPU):** In an env without Pillow: `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -q` must show the PIL test SKIPPED (not FAILED) and exit 0 (target: `0 failed`). Also run `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_local_render_preview.py -q` and `python -m py_compile tests/test_isaac_g1_kitchen_parity_runner.py`.

- **Context:** This sits directly on the active 'open the refrigerator' G1 POV seed lane: `_pov_seed_frame_quality` (runner line 1523) is the CPU gate that rejects black-edge / occluded POV seed frames, and its test is exactly the one that hard-fails. Verified live: `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_local_render_preview.py -q` => `1 failed, 88 passed, 3 skipped`, failing only at `tests/test_isaac_g1_kitchen_parity_runner.py:1898` with `ModuleNotFoundError: No module named 'PIL'`. The PIL tests in `tests/test_local_render_preview.py` already skip (the `ss` markers in the run), proving the intended pattern. Project memory documents that silent optional-dep failures have bitten this team before, so making the skip explicit matters.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), fix the one falsely-failing test that breaks the CPU test suite. The test `test_pov_seed_frame_quality_rejects_black_edge_occlusion` in `tests/test_isaac_g1_kitchen_parity_runner.py` (around line 1898) does a hard `from PIL import Image, ImageDraw  # type: ignore`, which raises ModuleNotFoundError and FAILS (not SKIPS) on any environment where Pillow is not installed. On this machine that single test turns an otherwise-green run into `1 failed, 88 passed, 3 skipped`. The sibling file `tests/test_local_render_preview.py` already does this correctly with `pytest.importorskip("PIL")` (see lines 90, 145, 431) — mirror that pattern.

What to do:
1. In `tests/test_isaac_g1_kitchen_parity_runner.py`, add `pytest.importorskip("PIL")` at the start of `test_pov_seed_frame_quality_rejects_black_edge_occlusion` (before the `from PIL import ...` line) so the test SKIPS rather than FAILS when Pillow is unavailable. `pytest` is already imported in that file; confirm before adding an import.
2. Grep the whole `tests/` tree for any other bare `from PIL import` / `import PIL` / `import cv2` that is not already guarded by an `importorskip`, and add the same guard (`pytest.importorskip("PIL")` / `pytest.importorskip("cv2")`) to each such test so the CPU suite is green on a minimal env.
3. Do NOT change any assertion logic, threshold, or production code — this is purely making optional-dependency tests skip cleanly.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; do not weaken any existing assertion. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done, run `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_local_render_preview.py -q` and `python -m py_compile tests/test_isaac_g1_kitchen_parity_runner.py`.
```

</details>

## Visual QC rubrics

### [P0-34] Block task gate when manipulation-POV produced zero frames

- **Priority:** P0 · **Effort:** S · **Dimension:** Visual QC rubrics
- **Goal:** Make _run_task_visual_qc fail closed (blocker) when pov_frame_paths is empty instead of silently passing on placement alone.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** python3 -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k 'task_visual_qc' -q  &&  python3 -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py

- **Context:** This is in the active 'open the refrigerator' G1 POV seed lane: a seed image is only trustworthy as an OSCAR seed if it was POV-validated. The runner imports qc_manipulation_pov_frames/qc_robot_placement_frames from render_visual_qc and aggregates their statuses in _run_task_visual_qc (scripts/run_isaac_g1_kitchen_parity_eval.py:4136-4199). The runner test module loads the script via importlib.util.spec_from_file_location as `M` (tests/test_isaac_g1_kitchen_parity_runner.py:16-26) and the existing happy-path test at line 403 monkeypatches the two qc_* functions on the imported blueprint_pipeline.render_visual_qc module. A POV-less render currently ships as 'passed'.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), fix a fail-OPEN bug in the combined task visual-QC gate.

File: scripts/run_isaac_g1_kitchen_parity_eval.py, function _run_task_visual_qc (starts at line 4136). Currently pov_report is computed only `if pov_frame_paths` else None (lines 4172-4181), and the None branch is excluded from blocker aggregation (`if pov_report is not None and ...` at line 4185). This means a render that produced ZERO robot-POV frames passes the whole task gate on placement alone — the arm-visible / affordance-visible manipulation checks never run. This is the exact class of bug the gate exists to prevent: a missing render artifact must BLOCK, not pass, and it is the opposite of the fail-closed intent in the docstring.

Fix: when pov_frame_paths is empty, treat it as a blocker. Append a blocker code 'manipulation_pov_visual_qc_no_frames' (mirroring qc_robot_placement_frames / qc_manipulation_pov_frames which emit *_no_frames when their sample list is empty, see src/blueprint_pipeline/render_visual_qc.py lines 540-541 and 575-576). The combined report status must roll up to 'blocked' in that case. Keep the existing behavior when pov frames DO exist. Do not change placement handling.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; preserve the claim_boundary language. Add a hermetic test (no GPU, no network) in tests/test_isaac_g1_kitchen_parity_runner.py next to test_task_visual_qc_splits_verify_and_pov_rubrics (line 403) that monkeypatches blueprint_pipeline.render_visual_qc.qc_robot_placement_frames to return status='passed' and calls M._run_task_visual_qc([verify_path], [], target_label='refrigerator', task_description='open the refrigerator'); assert report['status']=='blocked' and 'manipulation_pov_visual_qc_no_frames' in report['blockers']. Run the tests and py_compile.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P0-35] Make generic render-QC parser fail closed on missing safety booleans

- **Priority:** P0 · **Effort:** S · **Dimension:** Visual QC rubrics
- **Goal:** parse_qc_verdict must not default missing coherent/robot_visible/background_consistent to clean/True when the model omits them.
- **Files:** `src/blueprint_pipeline/render_visual_qc.py`, `tests/test_render_visual_qc.py`
- **Validate (CPU):** python3 -m pytest tests/test_render_visual_qc.py -k 'parse or flag' -q  &&  python3 -m py_compile src/blueprint_pipeline/render_visual_qc.py

- **Context:** render_visual_qc.py is the trust gate between 'rendered' and 'trusted as an OSCAR seed' for the G1 'open the refrigerator' lane. The two pass/fail parsers (placement, manipulation-POV) already fail closed; the generic rubric is the outlier and a partial JSON from the model cascade (gemini-3-flash-preview -> ... -> gemini-2.5-pro, lines 29-34) would slip through as clean. verdict_is_flagged is the single decision function consumed by qc_render_frames (line 514).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), fix a fail-OPEN asymmetry in the generic render-QC parser.

File: src/blueprint_pipeline/render_visual_qc.py. parse_qc_verdict (lines 213-244) calls _as_bool(obj.get('coherent'), True), _as_bool(obj.get('robot_visible'), True), _as_bool(obj.get('background_consistent'), True). _as_bool (lines 190-197) returns the default (True) when the key is absent or None. So a reply that parses (some JSON exists, parsed=True) but OMITS robot_visible / coherent / background_consistent is scored as clean. By contrast parse_robot_placement_verdict (lines 264-267) and parse_manipulation_pov_verdict (lines 294-302) default every safety field to False (fail-closed). The generic render-QC rubric must also fail closed on missing critical booleans so a truncated or schema-drifted reply across the model cascade cannot pass as clean.

Fix (choose the minimally invasive option and keep it consistent): when the key is absent/None, set coherent/robot_visible/background_consistent to None (not True). Then in verdict_is_flagged (lines 316-335) ensure a None for any of these three flags the frame (currently it only flags on an explicit False at lines 324-329 — change to treat None as 'not affirmed' => flagged). Keep the existing explicit-True / explicit-False behavior unchanged so existing tests still pass. Do NOT loosen any other path.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims. Extend tests in tests/test_render_visual_qc.py: add a case parse_qc_verdict(json.dumps({'summary':'ok'})) (no safety booleans) and assert the three fields are None and verdict_is_flagged(...) is True; add a case where robot_visible is explicitly true and others present to confirm a genuinely-clean reply still passes (flagged False). Run tests and py_compile.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P0-36] Stop _norm_severity from downgrading unknown severities to 'low'

- **Priority:** P0 · **Effort:** S · **Dimension:** Visual QC rubrics
- **Goal:** An unrecognized non-empty severity (e.g. 'critical') must clamp UP to the floor/high, never down to 'low'.
- **Files:** `src/blueprint_pipeline/render_visual_qc.py`, `tests/test_render_visual_qc.py`
- **Validate (CPU):** python3 -m pytest tests/test_render_visual_qc.py -k 'sever or flag or parse' -q  &&  python3 -m py_compile src/blueprint_pipeline/render_visual_qc.py

- **Context:** _norm_severity feeds every severity comparison in verdict_is_flagged (lines 332-335) and worst_severity (lines 338-347). On the G1 refrigerator lane a model that calls a frame 'critical' but uses a word outside the four-rank vocabulary currently passes the medium floor unflagged — a silent inversion exactly where a fail-closed gate must not leak.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), fix a severity-inversion bug in the visual-QC normalizer.

File: src/blueprint_pipeline/render_visual_qc.py, _norm_severity (lines 208-210). It returns the value only if it is in _SEVERITY_RANK {none,low,medium,high} (line 36); any other NON-EMPTY string ('critical','severe','major', stray-text variants) maps to 'low'. So a model that escalates by reporting overall_severity='critical' is DOWNGRADED to 'low', falling below the default 'medium' floor (line 38) and going UNflagged. In a fail-closed gate an unrecognized but non-empty severity must clamp UP, not down.

Fix: change the unknown-but-non-empty branch to return the configured high/floor severity (return 'high' for any non-empty string not in _SEVERITY_RANK; keep empty/None -> 'none'). Optionally add an explicit synonym map ('critical'/'severe'/'major' -> 'high') before the fallback. Empty/missing stays 'none'. Do not change recognized values.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims. Add tests in tests/test_render_visual_qc.py: parse_qc_verdict(json.dumps({'overall_severity':'critical','anomalies':[]})) currently normalizes to 'low' and verdict_is_flagged()==False — assert overall_severity normalizes to 'high' and verdict_is_flagged(...) is True; also assert an anomaly with severity 'critical' is treated as high. Run tests and py_compile.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## Catch-all / completeness

### [P0-37] Sync local venv + declare missing CPU deps so the materialization test stops failing

- **Priority:** P0 · **Effort:** S · **Dimension:** Catch-all / completeness
- **Goal:** Make the local .venv match pyproject so test_mujoco_scene_scenario_packet exercises the real CPU mesh-materialization path instead of the trimesh-missing fallback.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/pyproject.toml`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/mujoco_scene_scenario_packet.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_mujoco_scene_scenario_packet.py`
- **Validate (CPU):** uv sync --extra dev; .venv/bin/python -c 'import trimesh'; .venv/bin/python -m pytest tests/test_mujoco_scene_scenario_packet.py -o addopts='' (must pass, 0 failures); python -m py_compile src/blueprint_pipeline/mujople_scene_scenario_packet.py 2>/dev/null || python -m py_compile src/blueprint_pipeline/mujoco_scene_scenario_packet.py

- **Context:** Verified live: `.venv/bin/python -c 'import trimesh'` -> ModuleNotFoundError; pyproject.toml declares trimesh>=4.4.0 in both `runtime` and `dev` extras; src/blueprint_pipeline/mujoco_scene_scenario_packet.py has the two branches at lines 1034 (blocked_missing_trimesh_runtime) and 1133 (blocked_no_visual_meshes_materialized); the test asserts the latter at lines 95 and 115. A green-CI/red-local split means the developer's primary validation surface is lying and masking the real mesh-materialization code path. trimesh/scipy/open3d are pure-CPU wheels.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), the local .venv is out of sync with pyproject.toml, which produces a REAL local test failure (not cosmetic).

Reproduce first: run `.venv/bin/python -m pytest tests/test_mujoco_scene_scenario_packet.py::test_build_mujoco_scene_packet_writes_warehouse_tasks_matrix_and_recording_plan -o addopts=''`. It fails because `.venv/bin/python -c 'import trimesh'` raises ModuleNotFoundError, so src/blueprint_pipeline/mujoco_scene_scenario_packet.py:1034 returns scene_materialization_status='blocked_missing_trimesh_runtime' while the test (tests/test_mujoco_scene_scenario_packet.py:95 and :115) asserts 'blocked_no_visual_meshes_materialized' (the branch at line 1133).

Do this:
1. Confirm which declared deps are missing from the venv: trimesh (declared in pyproject.toml [project.optional-dependencies] dev and runtime), and check scipy / open3d / google-genai. Note: `dev` currently includes trimesh+pycollada but NOT scipy/open3d/google-genai. If the code under test (or its import chain in mujoco_scene_scenario_packet.py) needs scipy/open3d/google-genai to reach the intended branch on a clean machine, add the missing ones to the `dev` extra in pyproject.toml so `uv sync --extra dev` produces a venv that runs this test correctly. Do not add GPU-only packages.
2. Run `uv sync --extra dev` (or `uv pip install -e '.[dev]'`) to install.
3. Re-run the failing test and the full module; both must pass.

Constraints: keep world-model backends swappable (do not hardcode a single mesh backend); protect provenance/rights/privacy/raw-capture-truth (do not alter what gets persisted); render/materialization outputs are simulator support, NOT policy-success claims. Add/extend tests only if you change pyproject extras semantics. Run `python -m pytest tests/test_mujoco_scene_scenario_packet.py -o addopts=''` and `python -m py_compile src/blueprint_pipeline/mujoco_scene_scenario_packet.py`.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P0-38] Install/declare usd-core so dry-render USD geometry tests run instead of skipping

- **Priority:** P0 · **Effort:** S · **Dimension:** Catch-all / completeness
- **Goal:** Make pxr (usd-core) available on CPU so the 10 dry-render preview tests that open the real KitchenRoom USD actually run, protecting the no-GPU placement/camera/POV-framing gate for the G1 'open the refrigerator' lane.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/pyproject.toml`, `$HOME/workspace/BlueprintCapturePipeline/requirements-geometry.txt`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_local_render_preview.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_scene_placement.py`, `$HOME/workspace/BlueprintCapturePipeline/scripts/run_isaac_g1_kitchen_parity_eval.py`
- **Validate (CPU):** pip install usd-core (or uv sync --extra dev); python -c 'from pxr import Usd, UsdGeom'; python -m pytest tests/test_local_render_preview.py -o addopts='' (10 previously-skipped tests now collected+passing, 0 skips for pxr); python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py

- **Context:** The dry-render tool (scripts/run_isaac_g1_kitchen_parity_eval.py:6671+, '--dry-render', NO GPU) is the user's stated mechanism to catch placement/camera bugs before firing a paid GPU render in the active 'open the refrigerator' G1 POV seed lane. Right now its highest-value validation (against real USD geometry) silently SKIPs locally because pxr is absent, so a placement regression would only surface on a real GPU run. usd-core ships a pure-CPU pip wheel.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), usd-core (pxr) is not installed and not declared anywhere (pyproject.toml, requirements.txt, requirements-geometry.txt). Verified: `.venv/bin/python -c 'import pxr'` -> ModuleNotFoundError, and tests/test_local_render_preview.py calls pytest.importorskip('pxr') at lines 89, 144, 206, 226, 255, 281, 313, 329 (10 skipped tests). tests/test_scene_placement.py has analogous pxr-gated skips. These are exactly the tests that open the REAL kitchen USD (assets/Collected_KitchenRoom/KitchenRoom.usd) via _open_usd_stage_plain in scripts/run_isaac_g1_kitchen_parity_eval.py (around line 6698) and validate placement/camera/POV-framing geometry.

Do this:
1. Add `usd-core` to a CPU dev/geometry dependency surface. Preferred: add it to the `dev` extra in pyproject.toml (so `uv sync --extra dev` installs it); also add it to requirements-geometry.txt for the geometry path. Pin a floor version that ships a macOS+Linux CPU wheel (e.g. usd-core>=24.0).
2. Install it (`uv sync --extra dev` or `pip install usd-core`).
3. Confirm `python -c 'from pxr import Usd'` works.
4. Run tests/test_local_render_preview.py and confirm the previously-skipped 10 tests now run (and pass against the committed KitchenRoom USD).

Constraints: keep world-model backends swappable (pxr is one geometry reader; do not make it a hard import at module top-level of production code paths that must run without it — keep the importorskip/guarded pattern); protect provenance/rights/privacy/raw-capture-truth; dry-render outputs are simulator support, NOT policy-success claims; extend tests if you add new guarded paths. Run `python -m pytest tests/test_local_render_preview.py tests/test_scene_placement.py -o addopts=''` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py`.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P0-39] Run full pytest suite to completion and establish a clean CPU-green baseline

- **Priority:** P0 · **Effort:** M · **Dimension:** Catch-all / completeness
- **Goal:** After the venv/usd-core fixes, run the entire ~2491-test suite to completion on CPU and record/triage any remaining failures so there is a trustworthy local baseline before GPU work resumes.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/pyproject.toml`, `$HOME/workspace/BlueprintCapturePipeline/tests`
- **Validate (CPU):** python -m pytest -q -o addopts='' runs to completion (no -x) with 0 failures; any skips are explicitly justified; the run reports final pass/skip counts.

- **Context:** Audit ran `python -m pytest -o addopts='' -x` and stopped at the first failure (test_mujoco_scene_scenario_packet) after 878 passed / 13 skipped / 1 failed — that first failure is env drift, which could be masking additional failures deeper in the run. The user wants EVERYTHING CPU-validatable green before resuming GPU work; a confirmed full-green baseline is the core completeness deliverable for this dimension.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), prerequisite work (sync venv + add trimesh deps; add usd-core) should already be applied — if not, do it first. No tests are marked GPU (grep for `mark.gpu` / `@pytest.mark.gpu` across tests/ returns nothing), so the whole suite is CPU-runnable.

Do this:
1. Run the full suite to completion WITHOUT -x: `python -m pytest -q -o addopts=''`. Budget time (~3-4 min per ~900 tests).
2. Record pass/skip/fail counts.
3. For every remaining failure, determine whether it is (a) more env drift (missing CPU dep -> fix by declaring/installing), (b) a genuine code bug (fix minimally), or (c) a test that legitimately needs hardware/network (mark it skip with a clear reason and a `pytest.importorskip`/`skipif`, do not silently delete). Do not paper over real failures by loosening assertions.
4. Produce a short summary of final counts and any skips with their reasons.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (never weaken provenance assertions to make a test pass); render outputs are simulator support, NOT policy-success claims; add/extend tests where you fix a bug. Run `python -m pytest -q -o addopts=''` and `python -m py_compile` on any file you edit.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

---

# P1 — Important hardening & correctness

## Test suite health

### [P1-01] Guard fastapi/uvicorn/cv2 optional imports in service tests

- **Priority:** P1 · **Effort:** M · **Dimension:** Test suite health
- **Goal:** Eliminate the remaining ~7 non-PIL collection errors (fastapi, uvicorn, cv2) by guarding those optional heavy imports so the modules skip on a bare CPU env.
- **Files:** `tests/test_operational_logging.py`
- **Validate (CPU):** python3 -m pytest --co -q tests/ 2>&1 | grep -cE "No module named '(fastapi|uvicorn|cv2)'"  (expect 0)

- **Context:** CPU test-suite health, second-largest bucket after PIL. From the measured 76 collection errors: fastapi (4, incl. tests/test_operational_logging.py), uvicorn (1), cv2 (2). These are optional server/vision deps absent on the bare CPU runner; guarding them lets the suite collect cleanly so the active G1 render lane's CPU runs are not drowned in unrelated collection noise.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), after PIL guards are added, `python3 -m pytest --co tests/` still reports collection errors from missing optional service/vision deps: ~4 from `No module named 'fastapi'`, ~1 from `No module named 'uvicorn'`, and ~2 from `No module named 'cv2'`. These come from test modules (e.g. tests/test_operational_logging.py for fastapi) and their imported source modules importing fastapi/uvicorn/cv2 at module top-level.

Task: For each affected TEST module, add a module-level `pytest.importorskip("fastapi")` / `importorskip("uvicorn")` / `importorskip("cv2")` (as appropriate) before the import chain that pulls the missing package, so the module SKIPS on a bare CPU env instead of ERRORing. Identify the exact set and the offending package per file with: `for f in $(python3 -m pytest --co tests/ 2>&1 | grep '^ERROR' | sed 's/ERROR //;s/ -.*//'); do echo "== $f =="; python3 -m pytest --co "$f" 2>&1 | grep -iE "No module named '(fastapi|uvicorn|cv2)'" | head -1; done`. If the missing import is in a SOURCE module under src/ that the test imports, prefer guarding at the test boundary (importorskip in the test) unless the source already has a try/except optional-import shim you can extend — do NOT add a hard runtime dependency. Keep imports lazy where a source module legitimately needs fastapi only for an HTTP-server entrypoint.

Constraints: keep world-model backends swappable (do not couple core orchestration to fastapi/cv2); protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support NOT policy-success claims; tests must SKIP when the dep is absent and RUN when present; do not delete tests. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: `python3 -m pytest --co -q tests/ 2>&1 | grep -cE "No module named '(fastapi|uvicorn|cv2)'"` must print 0; `python3 -m py_compile <each edited file>`.
```

</details>

### [P1-02] Add a Python-version guard so 3.9 envs skip 3.10+ modules cleanly

- **Priority:** P1 · **Effort:** M · **Dimension:** Test suite health
- **Goal:** Make the 8 PEP-604 union (`X | None`) and `tomllib` collection errors degrade to a clear skip/explanation on Python 3.9 instead of cryptic TypeErrors, while keeping requires-python>=3.10 authoritative.
- **Files:** `tests/test_sim_only_beta_deployment_parity_proof.py`, `tests/test_policy_autoresearch.py`, `scripts/run_sim_only_beta_deployment_parity_proof.py`, `src/blueprint_pipeline/runpod_provider_adapter.py`, `pyproject.toml`
- **Validate (CPU):** python3 -m pytest --co -q tests/ 2>&1 | grep -cE "unsupported operand|No module named 'tomllib'"  (expect 0) ; python3 -m py_compile src/blueprint_pipeline/runpod_provider_adapter.py

- **Context:** CPU test-suite health / portability. Measured: 2 union-pipe errors + 6 tomllib errors = 8 of the 76 collection errors, all rooted in the local interpreter being Python 3.9 while requires-python is >=3.10. scripts/run_sim_only_beta_deployment_parity_proof.py:30 evaluates a PEP-604 union in a module-level type alias (NOT a deferred annotation, so `from __future__ import annotations` does not save it). tomllib is 3.11+ stdlib. Making these degrade to explicit skips removes cryptic TypeErrors from CPU collection without lowering the project's stated 3.10+ floor.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), pyproject.toml declares `requires-python = ">=3.10"`, but the local CPU runner's system `python3` is 3.9.6. On 3.9, `python3 -m pytest --co tests/` produces: 2 errors of the form `TypeError: unsupported operand type(s) for |: ...` (from module-level PEP-604 unions evaluated at import time, e.g. scripts/run_sim_only_beta_deployment_parity_proof.py line 30 `JsonFetcher = Callable[[str, Mapping[str, str] | None, int], dict[str, Any]]`, imported by tests/test_sim_only_beta_deployment_parity_proof.py; and tests/test_policy_autoresearch.py line 27), and 6 errors `No module named 'tomllib'` (tomllib is stdlib only on 3.11+, used via src/blueprint_pipeline/runpod_provider_adapter.py).

Task, two parts:
1) Add a top-of-file guard `import sys; pytest.importorskip = pytest.importorskip` is NOT enough — instead, in EACH affected test module add a module-level skip: `import sys, pytest; if sys.version_info < (3, 10): pytest.skip("requires Python >= 3.10 (PEP 604 unions)", allow_module_level=True)` placed BEFORE the failing import. For the tomllib group, add `if sys.version_info < (3, 11): pytest.importorskip("tomllib")` (or a module-level skip with a clear reason) before the import chain that reaches tomllib. Discover the exact files with: `for f in $(python3 -m pytest --co tests/ 2>&1 | grep '^ERROR' | sed 's/ERROR //;s/ -.*//'); do echo "== $f =="; python3 -m pytest --co "$f" 2>&1 | grep -iE 'unsupported operand|tomllib' | head -1; done`.
2) For the tomllib SOURCE usage (src/blueprint_pipeline/runpod_provider_adapter.py), make the import portable WITHOUT changing behavior on 3.11+: use `try: import tomllib\nexcept ModuleNotFoundError: tomllib = None` or fall back to `tomli` if available, and only require it at the call site (raise a clear error there if both are missing). Do NOT add new hard dependencies and do NOT rewrite the PEP-604 unions in source (the project is 3.10+; rewriting to Optional[...] is acceptable only if it is purely an annotation and you keep `from __future__ import annotations` — prefer the test-side skip to avoid churn).

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support NOT policy-success claims; do not weaken assertions; the modules must RUN normally on Python >=3.10/3.11. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: on the 3.9 system python3, `python3 -m pytest --co -q tests/ 2>&1 | grep -cE "unsupported operand|No module named 'tomllib'"` must print 0 (they now skip with a clear reason); `python3 -m py_compile src/blueprint_pipeline/runpod_provider_adapter.py`; if a 3.10+ interpreter is available (e.g. python3.10/python3.11 on PATH) run `python3.11 -m py_compile <edited source/test files>` to confirm no regression. Do not install packages into the 3.11 env.
```

</details>

### [P1-03] Establish a green CPU baseline command and assert no collection errors

- **Priority:** P1 · **Effort:** M · **Dimension:** Test suite health
- **Goal:** After the import guards land, codify a single CPU-only invocation that collects the whole suite with zero ERRORs and add a meta-test that fails if new unguarded heavy imports regress collection.
- **Files:** `tests/test_collection_health.py`, `docs/CHANGELOG.md`
- **Validate (CPU):** python3 -m pytest tests/test_collection_health.py -q ; python3 -m pytest --co -q tests/ 2>&1 | grep -c '^ERROR '  (expect 0)

- **Context:** CPU test-suite health hardening. The suite currently has 76 collection errors purely from optional-dep / version gaps (measured: 1515 collected, 76 errors). A meta-test that asserts clean collection prevents future unguarded `from PIL import ...` / `import fastapi` additions (like the one just introduced in the G1 parity runner test) from silently breaking the CPU lane again. This protects the active 'open the refrigerator' G1 render lane's ability to be exercised on CPU via --dry-render and unit tests.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), once the PIL / fastapi / uvicorn / cv2 / version-guard fixes are in, the full suite should COLLECT with zero errors on a bare CPU env (missing Pillow/fastapi/cv2/tomllib). Add a lightweight guardrail so this does not silently regress.

Task: 1) Add a new test file tests/test_collection_health.py with a single test that runs `python3 -m pytest --co -q tests/` as a subprocess (using sys.executable and `-p no:cacheprovider`), captures stdout+stderr, and asserts there are 0 lines matching `^ERROR ` AND 0 occurrences of `No module named 'PIL'`, `No module named 'fastapi'`, `No module named 'cv2'`, `No module named 'tomllib'`, and `unsupported operand type`. To avoid infinite recursion, the test must collect a curated subset (e.g. pass an explicit list of directories excluding itself, or set an env var the meta-test checks and short-circuits on). Keep runtime under ~30s. 2) Document the canonical CPU baseline command in docs/CHANGELOG.md (append an entry, do not rewrite history): `python3 -m pytest -q tests/` should show 0 collection errors on CPU; optional heavy-dep tests skip.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support NOT policy-success claims; the meta-test must not itself require Pillow/fastapi/cv2; it must be deterministic and CPU-only. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: `python3 -m pytest tests/test_collection_health.py -q` passes; `python3 -m pytest --co -q tests/ 2>&1 | grep -c '^ERROR '` prints 0; `python3 -m py_compile tests/test_collection_health.py`.
```

</details>

### [P1-04] Add CPU dry-render regression test for the G1 POV seed lane

- **Priority:** P1 · **Effort:** M · **Dimension:** Test suite health
- **Goal:** Lock in the modified parity script's CPU --dry-render behavior with a no-GPU test so visual-fallback binding and POV-seed quality cannot silently regress.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`, `tests/test_local_render_preview.py`
- **Validate (CPU):** python3 -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_local_render_preview.py -q ; python3 -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py

- **Context:** Active G1 'open the refrigerator' POV seed render lane. The modified scripts/run_isaac_g1_kitchen_parity_eval.py (276-line diff) hardens robot-visibility diagnostics and adds a visual-fallback binding that prefers renderable-Gprim compositions while preserving articulation/collision candidates. The matching test additions in tests/test_isaac_g1_kitchen_parity_runner.py and tests/test_local_render_preview.py cover the new helpers, but there is no CPU regression test exercising the end-to-end --dry-render summary keys. A no-GPU dry-render test is the cheapest guard that the local iterate-then-fire-one-cloud-render workflow stays trustworthy.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), the uncommitted changes to scripts/run_isaac_g1_kitchen_parity_eval.py added: `_g1_visual_asset_candidates`, `_robot_visual_geometry_missing`, `_bind_g1_with_visual_fallback`, new blocker constants (ROBOT_VISUAL_MESH_MISSING_BLOCKER, ROBOT_REVIEW_VISUAL_PROXY_USED_BLOCKER), and stricter `_robot_render_visibility_diagnostics` (now traverses instance proxies and exposes `renderable_robot_geometry_present`). There is a local no-GPU `--dry-render` path used to reproduce the Isaac G1 stance/camera/arm framing without a GPU (see memory: local dry-render tool reproduces stance/camera/arm framing in ~7s).

Task: First, find the dry-render entrypoint and existing dry-render tests: `grep -rn 'dry.render\|dry_render\|--dry-render' scripts/run_isaac_g1_kitchen_parity_eval.py tests/ | head`. Then add CPU-only unit tests (extend tests/test_isaac_g1_kitchen_parity_runner.py or tests/test_local_render_preview.py, whichever already drives the dry-render helpers) that assert: (a) the `--dry-render` path runs to completion without importing isaacsim/torch (monkeypatch or rely on the no-GPU branch) and emits the expected POV-seed/stance summary keys; (b) `_robot_render_visibility_diagnostics` reports `renderable_robot_geometry_present=False` and `ROBOT_VISUAL_MESH_MISSING_BLOCKER` for an Xform-only robot subtree (a regression guard for the new fail-closed visual gate), and reports True when a Mesh/Gprim is present. Reuse pxr via `pytest.importorskip("pxr")`. Keep all new tests strictly CPU and fast (<5s).

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render/dry-render outputs are simulator support NOT policy-success claims — assert structural/visibility facts, never 'the robot opened the fridge'; tests must skip cleanly when pxr/PIL absent. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: `python3 -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_local_render_preview.py -q` passes (0 failed); `python3 -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py`. If the dry-render CLI is invocable headless, also run it once: `python3 scripts/run_isaac_g1_kitchen_parity_eval.py --dry-render` (or the discovered flag) and confirm exit 0 with no GPU/cloud calls.
```

</details>

## Isaac G1 render — CPU logic

### [P1-05] Test camera pitch-down cap and target-raising trig helpers

- **Priority:** P1 · **Effort:** S · **Dimension:** Isaac G1 render — CPU logic
- **Goal:** Pin the head-forward pitch cap math so a down-looking crop can't slip past the POV-geometry gate.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k pitch -q  (pure math, no pxr/GPU). python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py.

- **Context:** The head-forward pitch cap is the fix several recent commits churned on ('Limit G1 POV pitch', 'Widen head POV and tighten arm visibility gate') in the active 'open the refrigerator' POV seed lane. A wrong mount or low handle produces a down-looking or in-mesh seed. Constants at scripts/run_isaac_g1_kitchen_parity_eval.py:141-142 (MAX 26.0, HEAD_FORWARD 24.0); helpers at lines 1129/1140; gate appends 'manipulation_pov_camera_pitched_down_too_far' around line 1267. Only the composite selection has a test (line 292); the underlying trig has no isolated regression.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add direct unit tests for the pure-math pitch helpers in scripts/run_isaac_g1_kitchen_parity_eval.py: `_camera_pitch_down_deg(eye, target)` (line 1129) and `_target_raised_to_max_pitch_down(eye, target, max_pitch_down_deg)` (line 1140). Assert: (a) a head eye looking down at a low handle yields `_camera_pitch_down_deg` > MANIPULATION_POV_MAX_CAMERA_PITCH_DOWN_DEG (26.0, line 141); (b) `_target_raised_to_max_pitch_down` raises the look-at z to exactly `min_target_z` so the recomputed `_camera_pitch_down_deg(eye, raised_target)` equals the cap (within ~1e-6 deg), and is a no-op when the target z is already at/above that floor; (c) when horizontal distance is ~0 the guard at line 1144 returns the target unchanged. Also add a test for `_select_manipulation_camera_target_for_visible_arm` (line 1578) asserting it prefers pitch-limited candidates and rejects a 'pitch down to force the affordance in frame' workaround (the composite is partially covered at tests/test_isaac_g1_kitchen_parity_runner.py:292 `test_manipulation_camera_target_selection_rejects_downward_pitch_workaround`, but the raw trig helpers are untested). These are pure math — no pxr/PIL needed. Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; add/extend tests only. Run `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k pitch -q` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-06] Unit-test head-lens mount selection ranking and head-bounds scoring

- **Priority:** P1 · **Effort:** M · **Dimension:** Isaac G1 render — CPU logic
- **Goal:** Lock authored-camera-vs-head-vs-neck mount ranking and degenerate-box rejection so the POV eye sits on the head.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_local_render_preview.py`
- **Validate (CPU):** python -m pytest tests/test_local_render_preview.py -q  (in-memory pxr stage; importorskip pxr). python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py.

- **Context:** Head-lens/mount selection decides where the POV eye sits (scripts/run_isaac_g1_kitchen_parity_eval.py around lines 4569-4613, `_robot_mounted_manipulation_cam_pose` line 4569, `_robot_head_lens_eye_from_mount` line 4429) for the active 'open the refrigerator' seed. A wrong mount (neck below shoulders, or a tiny degenerate prim) yields a down-looking or in-mesh seed — the failure class the pitch caps try to catch downstream. These selectors currently have zero direct tests; only _average_arm_link_points and _robot_head_lens_eye_from_mount are exercised.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add real-pxr tests (use `Usd.Stage.CreateInMemory()`, gate with `pytest.importorskip('pxr')`) for the head-lens mount selectors in scripts/run_isaac_g1_kitchen_parity_eval.py: `_robot_authored_camera_mount` (line 4221), `_robot_link_mount` (line 4246), and `_robot_head_bounds_for_mount` (line 4356). Build `/World/G1` subtrees and assert: (a) when an authored `UsdGeom.Camera` exists alongside `*_head_link` and `*_neck_link` Xforms, `_robot_authored_camera_mount` returns `source=='authored_robot_camera'` with the camera's world translation, and `_robot_link_mount` would otherwise pick it (rank 0, mount_role 'camera_link'); (b) with NO camera, `_robot_link_mount` ranks `head_link` (rank 1, mount_role=='head_link') OVER `neck_link` (rank 2) — verify the preference tuple at lines 4253-4257 and the sorted selection at line 4269; (c) `_robot_head_bounds_for_mount` prefers a head/camera/face/neck-named prim, and (per the function's intent) rejects degenerate sub-~0.04m boxes / prefers higher center_z so the lens does not land below the shoulders. Position prims so head is above neck above shoulders to make the ranking observable. Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; add/extend tests only. Run `python -m pytest tests/test_local_render_preview.py -q` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-07] Test review-proxy geometry math and no-physics-API invariant

- **Priority:** P1 · **Effort:** M · **Dimension:** Isaac G1 render — CPU logic
- **Goal:** Guarantee review proxies derive arm/body boxes correctly, stay outside /World/G1, and never add collision/articulation APIs.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_local_render_preview.py`
- **Validate (CPU):** python -m pytest tests/test_local_render_preview.py -q  (in-memory pxr stage). Assert no HasAPI(UsdPhysics.*) on any proxy prim and every proxy path startswith proxy_root_path. python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py.

- **Context:** Proxies are the 'clearly-labeled fallback' half of the active fail-closed G1 task. If a proxy ever leaks a collision/articulation API or lands inside /World/G1 it would corrupt placement/collision validation and falsely imply physical geometry — violating the claim boundary. The arm-span math (scripts/run_isaac_g1_kitchen_parity_eval.py lines 5369-5436) determines whether the reviewer can SEE a reach pose for 'open the refrigerator'. None of these invariants are asserted today beyond a happy-path count at tests/test_isaac_g1_kitchen_parity_runner.py:328.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), extend coverage of `_create_robot_review_visual_proxies` (scripts/run_isaac_g1_kitchen_parity_eval.py line 5343) and `_add_robot_review_proxy_box` (line 5304) with real-pxr tests (`Usd.Stage.CreateInMemory()`, gate `pytest.importorskip('pxr')`). Build a `/World/G1` with a known world-space bbox and named arm link Xforms (shoulder/elbow/wrist/hand for left and right). Assert: (a) torso/pelvis/leg body boxes derive from bbox proportions (e.g. torso center at bmin_z + 0.58*size_z, sizes clamped to the min floors in the body-box block around lines 5384-5436); (b) arm boxes span shoulder→elbow→wrist→hand from `_robot_arm_link_points_by_arm`; (c) a side with no arm points records a `'{side}_arm_link_points_unavailable_for_review_proxy'` blocker but STILL builds body boxes; (d) EVERY created proxy prim has NO `UsdPhysics.CollisionAPI` and NO `UsdPhysics.ArticulationRootAPI` (assert `not prim.HasAPI(UsdPhysics.CollisionAPI)` etc.) and lives strictly under `proxy_root_path`, never under `/World/G1`; (e) when the bbox is unavailable the result records `'robot_bbox_unavailable_for_review_proxy'`. The existing test at tests/test_isaac_g1_kitchen_parity_runner.py:328 (`test_robot_review_visual_proxies_use_link_geometry_without_scene_coords`) only checks the happy-path count — go beyond it. Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; proxies must never imply physical geometry. Add/extend tests only. Run `python -m pytest tests/test_local_render_preview.py -q` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-08] Test render-visibility diagnostics across instanceable meshes

- **Priority:** P1 · **Effort:** M · **Dimension:** Isaac G1 render — CPU logic
- **Goal:** Prove an instanced-but-renderable G1 is NOT falsely flagged robot_visual_mesh_missing.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_local_render_preview.py`
- **Validate (CPU):** python -m pytest tests/test_local_render_preview.py -k instance -q  (in-memory pxr stage with an instanceable reference). python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py.

- **Context:** Production robot USDs frequently mark meshes instanceable. _robot_render_visibility_diagnostics (scripts/run_isaac_g1_kitchen_parity_eval.py line 5183) attempts `Usd.PrimRange(robot, Usd.TraverseInstanceProxies())` at line 5209 with a plain-PrimRange fallback at line 5212, but no test proves it counts geometry inside an actually-instanced subtree. If it missed instance-proxy geometry it would emit a false robot_visual_mesh_missing for the active 'open the refrigerator' lane and needlessly swap to proxies or block a renderable robot.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add a real-pxr test (`Usd.Stage.CreateInMemory()`, gate `pytest.importorskip('pxr')`) for `_robot_render_visibility_diagnostics` (scripts/run_isaac_g1_kitchen_parity_eval.py line 5183) that exercises the instance-proxy traversal path. Build a `/World/G1` whose renderable meshes live under an INSTANCEABLE reference/prototype (visible only as instance proxies — set the referencing prim instanceable). Assert: (a) the diagnostics return `gprim_count > 0` and `mesh_count > 0` and `traversed_instance_proxies == True` (the success branch of the try at lines 5208-5213), and `M.ROBOT_VISUAL_MESH_MISSING_BLOCKER` is NOT in blockers and status is 'PASS' (assuming materials/visibility ok); (b) `instanceable_prim_count > 0` (line 5219-5220); (c) a separate NON-instanced, mesh-less subtree still FAILS with the blocker present and `gprim_count == 0`. This proves an instanced-but-renderable G1 (the plausible real composition) is not falsely flagged. Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; add/extend tests only. Run `python -m pytest tests/test_local_render_preview.py -k instance -q` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-09] Make --dry-render bind the real G1 visual asset and run pitch/POV/visibility gates

- **Priority:** P1 · **Effort:** L · **Dimension:** Isaac G1 render — CPU logic
- **Goal:** Let the cheap local preview catch the invisible-robot and pitched-down-crop bugs before any cloud render.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_local_render_preview.py`
- **Validate (CPU):** python -m pytest tests/test_local_render_preview.py -k dry_render -q  (needs pxr+PIL; gate with importorskip). If a kitchen+G1 USD is vendored: python scripts/run_isaac_g1_kitchen_parity_eval.py --dry-render --g1-usd <path> against it produces a summary whose new checks reflect visibility/pitch. python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py.

- **Context:** The active strategy is 'iterate locally then fire ONE cloud render.' Today the local tool (scripts/run_isaac_g1_kitchen_parity_eval.py `_local_render_preview` line 6938, `_dry_render_checks` line 6746) validates stance/camera-projection but is blind to the two bugs recent commits churned on for the 'open the refrigerator' G1 POV seed: robot invisible (0 meshes) and head POV pitched down too far. The dry-render is already proven CPU-only (tests/test_isaac_g1_kitchen_parity_runner.py:427 test_dry_render_cli_runs_end_to_end_on_real_kitchen). Closing this lets the local preview reproduce the exact failure that wasted GPU time.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), extend the local no-GPU dry-render path so it can exercise the same robot-visibility and head-POV gates the GPU runner uses, closing the blind spot that let two recent bug classes (robot invisible / 0 renderable meshes, head POV pitched down too far) survive the cheap local check. Today `_local_render_preview` (scripts/run_isaac_g1_kitchen_parity_eval.py line 6938) uses the simple `manipulation_cam_pose` fallback (line 7003) plus a synthetic `nominal_g1_rest_offsets` skeleton (line 7005); it never binds the real G1 USD, never calls `_robot_mounted_manipulation_cam_pose`/`_robot_head_lens_eye_from_mount`, never runs `_manipulation_pov_geometry`, and has no robot_visual_mesh / pitch-cap / seed-frame check. Add an OPTIONAL path (e.g. when a `--g1-usd` argument is provided to the dry-render CLI) that: (1) binds the real G1 via `_bind_g1_with_visual_fallback` (line 2264) and records its `robot_render_diagnostics` into the dry-render summary; (2) computes the head-mounted manipulation camera via `_robot_mounted_manipulation_cam_pose` (line 4569) when a real robot is bound, instead of the nominal fallback; (3) runs `_manipulation_pov_geometry` (line 1360) against the projected USD arm links and applies the `_camera_pitch_down_deg` cap vs MANIPULATION_POV_MAX_CAMERA_PITCH_DOWN_DEG (26.0, line 141) / HEAD_FORWARD (24.0, line 142). Then extend `_dry_render_checks` (line 6746) to add booleans for `robot_visual_mesh_present`, `camera_pitch_within_cap`, and `pov_geometry_pass`, so the checklist surfaces invisible-robot and pitched-down-crop failures. Keep the existing no-robot fallback behavior fully intact and default (no --g1-usd → today's behavior, byte-for-byte). Add tests in tests/test_local_render_preview.py (gate pxr+PIL with `pytest.importorskip`) that drive the new path against a synthetic mesh-less /World/G1 and assert the new checks go False; and a renderable case where they go True. Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims (the dry-render claim_boundary at lines 6977-6980 must stay). Add/extend tests. Run `python -m pytest tests/test_local_render_preview.py -k dry_render -q` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## TODO/FIXME sweep

### [P1-10] Regression-test the placeholder_cosmos_pending render fallback label

- **Priority:** P1 · **Effort:** S · **Dimension:** TODO/FIXME sweep
- **Goal:** Add a CPU unit test proving the no-splat/no-Cosmos render branch returns a valid PNG carrying the explicit X-Blueprint-Render-Source: placeholder_cosmos_pending marker so a placeholder frame can never masquerade as a real render.
- **Files:** `src/blueprint_pipeline/native_runtime_backend.py`, `tests/test_native_runtime_backend_coverage.py`
- **Validate (CPU):** python3 -m pytest tests/test_native_runtime_backend_coverage.py -q  (new case asserts a valid PNG body and X-Blueprint-Render-Source == 'placeholder_cosmos_pending'); python3 -m py_compile src/blueprint_pipeline/native_runtime_backend.py tests/test_native_runtime_backend_coverage.py

- **Context:** src/blueprint_pipeline/native_runtime_backend.py lines ~2620-2721 hold the splat>Cosmos>placeholder dispatcher and the 'placeholder_cosmos_pending' header. tests/test_native_runtime_backend_coverage.py is the existing coverage suite for this module. This matters because the served render is the human-review artifact for the active 'open the refrigerator' G1 POV lane — a placeholder frame that loses its label could be mistaken for a real Cosmos/splat render in review, violating CLAUDE.md's raw-capture/provenance-truth rule.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), harden the provenance labeling of the native runtime render fallback.

Background: src/blueprint_pipeline/native_runtime_backend.py renders frames with a priority order of synthesized splat > Cosmos video > placeholder. When BOTH the synthesized splat and the Cosmos video frames are unavailable, the frame-priority dispatcher falls back to a placeholder PNG and sets the HTTP response header 'X-Blueprint-Render-Source' to 'placeholder_cosmos_pending' (see the literal at native_runtime_backend.py:2721 and the related 'render_source=placeholder_cosmos_pending' string at line ~2625; the dispatcher returns the headers dict near lines 2700-2721). This is a real branch and the kind of seam that could silently become the served output. There is currently no explicit test that drives this exact fallback and asserts the label.

Task: Add a deterministic CPU-only test to tests/test_native_runtime_backend_coverage.py that constructs a render session/request where there is NO synthesized splat and NO Cosmos video frame available, invokes the same code path the server uses to produce a frame response, and asserts BOTH: (a) the returned body is a valid, decodable PNG (check the PNG magic bytes b'\x89PNG\r\n\x1a\n' at minimum, or decode with the stdlib if a decoder is already imported in that test module), and (b) the response headers contain 'X-Blueprint-Render-Source' == 'placeholder_cosmos_pending'. Reuse existing fixtures/helpers in that test module; do not add new heavy dependencies. If the production code only sets the header in one place, assert against that exact code path rather than re-deriving it.

Constraints: Keep world-model/render backends swappable (do not hardcode a single backend; drive the public render entrypoint). Protect provenance/rights/privacy and raw-capture truth: the whole point is that placeholder output stays unambiguously labeled. Treat render outputs as simulator support, NOT as policy-success or task-success claims. Add/extend tests only; do not change production behavior unless a genuine labeling bug is found (if so, fix it minimally and note it). Run `python3 -m pytest tests/test_native_runtime_backend_coverage.py -q` and `python3 -m py_compile src/blueprint_pipeline/native_runtime_backend.py tests/test_native_runtime_backend_coverage.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-11] Test the retrieval-index .npy decode-failure placeholder-frame fallback

- **Priority:** P1 · **Effort:** S · **Dimension:** TODO/FIXME sweep
- **Goal:** Add a CPU test that drives a corrupt .npy through the retrieval-index frame extractor and asserts the _write_placeholder_frame fallback fires, produces a decodable image, and does not crash downstream embedding.
- **Files:** `src/blueprint_pipeline/retrieval_index_stage.py`, `tests/test_retrieval_index_stage_coverage.py`, `tests/test_retrieval_index_geometry_source.py`
- **Validate (CPU):** python3 -m pytest tests/test_retrieval_index_stage_coverage.py tests/test_retrieval_index_geometry_source.py -q  (new case writes a malformed .npy, asserts _write_placeholder_frame is invoked and the output is a decodable image); python3 -m py_compile src/blueprint_pipeline/retrieval_index_stage.py tests/test_retrieval_index_stage_coverage.py

- **Context:** src/blueprint_pipeline/retrieval_index_stage.py:1178-1182 (try/except fallback) and :1199-1210 (_write_placeholder_frame). tests/test_retrieval_index_stage_coverage.py and tests/test_retrieval_index_geometry_source.py are the existing CPU suites. Confirm the exact image format the writer emits before asserting — the input describes a 2x2 PPM written with a .jpg suffix, so assert against the real bytes, not an assumed JPEG. Matters because index quality directly affects object/scene retrieval used throughout the pipeline.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add visibility coverage for the silent placeholder-frame fallback in the retrieval index stage.

Background: src/blueprint_pipeline/retrieval_index_stage.py _extract_frame (around line 1178-1182) attempts to decode a .npy frame and, on ANY exception, falls back to _write_placeholder_frame(output_path) (defined near line 1199), which writes a tiny 2x2 PPM-style image with a .jpg suffix. A corrupt or odd-shaped .npy therefore silently yields a 2x2 grey frame that then feeds the retrieval/DINOv2 index, degrading retrieval quality with no signal. This fallback is currently untested for visibility.

Task: Add a deterministic CPU-only test to tests/test_retrieval_index_stage_coverage.py that writes a deliberately malformed .npy file (e.g. random/truncated bytes with a .npy suffix that numpy cannot load, or a valid-but-wrong-shape array if that is what trips the decoder), points _extract_frame at it, and asserts: (a) _write_placeholder_frame is invoked (assert via the returned bool / patched spy / by asserting the output file is the placeholder shape), and (b) the produced output file exists and is a readable/decodable image (verify by reading the header bytes the writer emits — confirm the exact format _write_placeholder_frame produces and assert against it). Optionally assert that the caller can distinguish a placeholder frame (e.g. by size/shape) so the index could flag it. Reuse existing helpers from tests/test_retrieval_index_geometry_source.py / tests/test_retrieval_index_stage_coverage.py where possible. If feasible without GPU, also assert that the existing downstream embedding helper does not raise on the placeholder frame; if embedding requires GPU/heavy models, stop at the decodable-image assertion and do NOT pull in GPU deps.

Constraints: Keep world-model/retrieval backends swappable. Protect provenance/raw-capture truth — a grey 2x2 placeholder silently entering the index is exactly the degradation this test surfaces. Render/index outputs are simulator/retrieval support, not policy-success claims. Add/extend tests only; do not change the fallback's runtime behavior unless you find a real bug (e.g. the placeholder being indistinguishable from real frames), in which case make the minimal fix and add a flag/marker. Run `python3 -m pytest tests/test_retrieval_index_stage_coverage.py tests/test_retrieval_index_geometry_source.py -q` and `python3 -m py_compile src/blueprint_pipeline/retrieval_index_stage.py tests/test_retrieval_index_stage_coverage.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-12] Verify the cosmos_lora_training missing_training_command blocked-manifest contract

- **Priority:** P1 · **Effort:** S · **Dimension:** TODO/FIXME sweep
- **Goal:** Add/confirm a CPU test asserting that with no COSMOS_TRAINING_COMMAND env and no training_command arg, the LoRA training stage writes status='blocked', reason='missing_training_command' with a non-empty blockers list — never success and never a silent skip.
- **Files:** `src/blueprint_pipeline/synthesis/cosmos_lora_training.py`, `tests/test_cosmos_lora_training.py`
- **Validate (CPU):** python3 -m pytest tests/test_cosmos_lora_training.py -q  (case with COSMOS_TRAINING_COMMAND unset asserts manifest['status']=='blocked' and manifest['reason']=='missing_training_command' and a non-empty blockers list); python3 -m py_compile src/blueprint_pipeline/synthesis/cosmos_lora_training.py tests/test_cosmos_lora_training.py

- **Context:** src/blueprint_pipeline/synthesis/cosmos_lora_training.py:81 (env read), :102/:119-120 (two blocked branches; confirm which reason maps to the missing-command case), :129 (operator-facing blocker string). tests/test_cosmos_lora_training.py is the existing CPU suite. This keeps the GPU training lane honest during the current GPU pause: the stage must explicitly mark itself blocked, never silently skip, so the autonomous-loop/readiness logic cannot mistake an unconfigured stage for a completed one.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), lock down the fail-closed contract of the Cosmos LoRA training stage.

Background: src/blueprint_pipeline/synthesis/cosmos_lora_training.py reads the training command from the COSMOS_TRAINING_COMMAND env var / a training_command argument (see the env read at line ~81). When no command is configured it returns a manifest with status='blocked' and reason='missing_training_command' (literals at lines ~119-120) plus a blocker string instructing the operator to set COSMOS_TRAINING_COMMAND using {trainer_config_path}/{output_dir} placeholders (line ~129). This is correct fail-closed behavior and the only thing preventing 'no GPU command configured' from looking like a successful (or silently skipped) training run.

Task: Add or confirm a deterministic CPU-only test in tests/test_cosmos_lora_training.py that runs the stage with COSMOS_TRAINING_COMMAND unset (use monkeypatch.delenv / monkeypatch.setenv to clear it) and with no training_command argument, and asserts ALL of: manifest['status'] == 'blocked', manifest['reason'] == 'missing_training_command', and that the blockers list (whatever the field is named — read the code to get the exact key) is present and non-empty. Also assert the stage does NOT raise and does NOT return a 'done'/'succeeded' status in this configuration. If such a case already exists, tighten it to assert the exact reason string and the non-empty blockers list rather than a generic 'blocked'. Do not invoke any real trainer command.

Constraints: Keep world-model/training backends swappable (the stage must stay a GPU-only slot that fail-closes on CPU, not be reimplemented). Protect provenance: a readiness gate must never be fooled into thinking LoRA training ran. Training/render outputs are simulator support, not policy-success claims. Add/extend tests only; do not weaken the blocked contract. Run `python3 -m pytest tests/test_cosmos_lora_training.py -q` and `python3 -m py_compile src/blueprint_pipeline/synthesis/cosmos_lora_training.py tests/test_cosmos_lora_training.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-13] Assert WorldLabsPreviewProvider overrides every StubPreviewProvider success shortcut

- **Priority:** P1 · **Effort:** S · **Dimension:** TODO/FIXME sweep
- **Goal:** Add an introspection regression test proving the live WorldLabsPreviewProvider does not inherit StubPreviewProvider.submit/poll verbatim, so it can never silently emit a fake status='succeeded'/cost=0 with a stub file:// artifact.
- **Files:** `src/blueprint_pipeline/provider_preview.py`, `tests/test_provider_preview_edges.py`, `tests/test_provider_preview_qa.py`
- **Validate (CPU):** python3 -m pytest tests/test_provider_preview_edges.py tests/test_provider_preview_qa.py -q  (new assertions: WorldLabsPreviewProvider.submit.__qualname__ != StubPreviewProvider.submit.__qualname__ and same for poll); python3 -m py_compile src/blueprint_pipeline/provider_preview.py tests/test_provider_preview_edges.py tests/test_provider_preview_qa.py

- **Context:** src/blueprint_pipeline/provider_preview.py: StubPreviewProvider class ~209-267 (success shortcut), WorldLabsPreviewProvider ~271 subclasses it and overrides submit ~664 / poll ~806; resolver default mapping for 'stub'/'stub_preview' ~853-854. Verified locally: WorldLabsPreviewProvider.submit/poll qualnames already differ from Stub, so this test passes today and acts as a regression lock. Matters because a fabricated 'succeeded' preview with a non-existent still is exactly the evidence-integrity failure CLAUDE.md's provenance rules forbid.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add a cheap regression guard ensuring the real preview provider never falls back to the hermetic Stub's success shortcut.

Background: src/blueprint_pipeline/provider_preview.py defines StubPreviewProvider (class at line ~209) as the resolver default for 'stub'/'stub_preview'. Its submit/poll (lines ~213-267) take a hard-coded shortcut that reports status='succeeded' with cost=0 and a stub file:// artifact — fine for hermetic tests, catastrophic if a live provider inherited it (it would fabricate evidence: report success with zero cost and a non-existent preview still). WorldLabsPreviewProvider (class at line ~271) subclasses StubPreviewProvider for code reuse and currently DOES override submit (line ~664) and poll (line ~806). We want to make that override a permanent, enforced invariant so a future refactor cannot accidentally delete the override and re-expose the Stub's success path.

Task: Add a deterministic CPU-only introspection test (no network) to tests/test_provider_preview_edges.py and/or tests/test_provider_preview_qa.py that asserts WorldLabsPreviewProvider.submit.__qualname__ != StubPreviewProvider.submit.__qualname__ AND WorldLabsPreviewProvider.poll.__qualname__ != StubPreviewProvider.poll.__qualname__ (equivalently assert 'submit' in WorldLabsPreviewProvider.__dict__ and 'poll' in WorldLabsPreviewProvider.__dict__). Add a clear comment in the test explaining WHY: the live provider must never emit the Stub's fabricated 'succeeded'/cost=0/file:// result. Also extend the test to document, via a short module-level or test docstring note, that StubPreviewProvider is the intentional hermetic default for 'stub'/'stub_preview' resolution.

Constraints: Keep preview/world-model backends swappable (resolver-driven; do not hardcode provider selection in the test beyond introspection). Protect provenance/rights: this guard exists precisely to prevent fabricated success evidence. Preview/render outputs are simulator support, not policy-success claims. Add/extend tests only; do not modify production classes unless the assertion fails (it should currently pass). Run `python3 -m pytest tests/test_provider_preview_edges.py tests/test_provider_preview_qa.py -q` and `python3 -m py_compile src/blueprint_pipeline/provider_preview.py tests/test_provider_preview_edges.py tests/test_provider_preview_qa.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-14] Replace ffmpeg-conditional skips in test_privacy_processing with a deterministic fixture/mock

- **Priority:** P1 · **Effort:** M · **Dimension:** TODO/FIXME sweep
- **Goal:** Make the privacy-processing logic-path tests run without a system ffmpeg by mocking the ffmpeg invocation or committing a tiny synthetic input, so the rights/PII gate is verified on CPU boxes that lack ffmpeg.
- **Files:** `tests/test_privacy_processing.py`, `src/blueprint_pipeline/privacy_processing.py`, `tests/fixtures/`
- **Validate (CPU):** python3 -m pytest tests/test_privacy_processing.py -q -rs  (cases formerly at lines 28/51 now run — no 's' for them); python3 -m py_compile tests/test_privacy_processing.py src/blueprint_pipeline/privacy_processing.py

- **Context:** tests/test_privacy_processing.py:26-28 and :51 (ffmpeg skips); :841-857 show the existing monkeypatch(shutil.which)+subprocess-mock pattern producing reasons 'ffmpeg_not_found' and 'ffmpeg_redaction_failed:3' — copy that approach. src/blueprint_pipeline/privacy_processing.py:117/119 hosts the 'runner_url_invalid_or_placeholder' guard. Matters because privacy/PII redaction is a rights-critical gate and must not be unverified on the GPU-paused CPU validation boxes.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), de-skip the ffmpeg-gated privacy-processing tests so the privacy/PII gate is exercised deterministically on CPU machines without a system ffmpeg binary.

Background: tests/test_privacy_processing.py skips at line 28 ('ffmpeg not installed', after shutil.which('ffmpeg') returns None at line ~26) and at line 51 ('ffmpeg test source failed'). On a CPU box without ffmpeg these privacy tests silently do not run. The same module already shows the right pattern elsewhere — later cases (around lines 841-857) monkeypatch pp.shutil.which to return a fake '/usr/bin/ffmpeg' and monkeypatch the subprocess call to assert outcomes like reason=='ffmpeg_not_found' and 'ffmpeg_redaction_failed:3'. The core privacy_processing logic (src/blueprint_pipeline/privacy_processing.py, including the 'runner_url_invalid_or_placeholder' guard around lines 117/119) should be reachable without a real ffmpeg.

Task: Convert the two skipping cases (lines 28 and 51) into deterministic tests. Prefer the existing monkeypatch approach: patch shutil.which to report ffmpeg present and patch the subprocess.run/invocation so the privacy_processing logic path runs against a controlled fake result (success and/or the relevant failure reason), OR commit a tiny synthetic input (a few small frame files / a minimal pre-generated artifact) under tests/fixtures/ that the logic can consume without invoking a real encoder. Read both case bodies first to determine exactly what they assert (the redacted-output path, the placeholder/invalid-runner-URL guard, reason strings) and reproduce those assertions deterministically. Keep a real-ffmpeg path if one is genuinely needed for an end-to-end smoke (gate it so it skips only when ffmpeg is absent), but ensure the LOGIC-PATH assertions for these two cases run unconditionally via the mock/fixture.

Constraints: Privacy processing is rights/PII-critical per CLAUDE.md ('Protect provenance, rights, privacy') — the gate must be verified, not skipped, on the CPU machines we now validate on. Keep backends swappable; mock at the subprocess/ffmpeg boundary, do not couple the test to a specific encoder build. Do not weaken or bypass the privacy guard (e.g. 'runner_url_invalid_or_placeholder') — assert it fires. Protect raw-capture truth: synthetic inputs must be labeled as test fixtures. Add/extend tests (and small fixtures) only. Run `python3 -m pytest tests/test_privacy_processing.py -q -rs` and confirm the two cases formerly at lines 28/51 no longer report 's' (skipped), and `python3 -m py_compile tests/test_privacy_processing.py src/blueprint_pipeline/privacy_processing.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## Main 11-stage pipeline

### [P1-15] Test object_index backend-report normalization for malformed/partial output

- **Priority:** P1 · **Effort:** M · **Dimension:** Main 11-stage pipeline
- **Goal:** Prove _run_backend_command degrades cleanly (skipped/failed) on bad JSON, stdout-only JSON, and payload status override.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/object_index_stage.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_object_index_stage.py`
- **Validate (CPU):** .venv/bin/python -m py_compile src/blueprint_pipeline/object_index_stage.py && .venv/bin/python -m pytest tests/test_object_index_stage.py -q

- **Context:** _run_backend_command in src/blueprint_pipeline/object_index_stage.py is the CPU boundary between the orchestrator and every GPU detection backend. Hardening its parsing keeps the no-GPU path used by the active 'open the refrigerator' G1 lane from crashing or mislabeling status. Test file: tests/test_object_index_stage.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add tests for backend-report normalization in _run_backend_command.

Background: src/blueprint_pipeline/object_index_stage.py:_run_backend_command (around lines 641-730) normalizes subprocess results: invalid command template (~665-666), empty command (~667-668), OSError launch failure (~672-678), missing output file falling back to parsing stdout JSON (~688-698), and reading backend_status/reason from the payload (~706-730). The existing test test_object_index_subprocess_backend_and_detection_helpers (tests/test_object_index_stage.py:~324) covers some helpers but not these malformed/partial cases. This is the pure-CPU boundary between the orchestrator and every GPU detection backend; robust parsing is what lets the no-GPU path degrade to 'skipped'/'failed' rather than crash or mislabel.

Task:
1. Read _run_backend_command (~641-730) and the existing helper test (~324) to match conventions for setting OBJECT_INDEX_*_COMMAND to a tiny `python -c ...` command.
2. Add tests covering: (a) a backend that writes INVALID JSON to OUTPUT_JSON -> assert status=='failed' and reason starts with 'invalid_output_json:'; (b) a backend that writes nothing to the output file but prints valid JSON on stdout -> assert it parses from stdout; (c) a backend whose payload sets backend_status='skipped' even with return code 0 -> assert the normalized report reflects status 'skipped' (payload overrides returncode). Prefer a small inline `python -c` for each so no real model is invoked.
3. Do not change production behavior unless you find a normalization bug; fix minimally if so.
4. Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render/eval outputs are simulator support, not policy-success claims. Run validation and ensure it passes.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-16] Add CPU coverage for retrieval frame-extraction and quality-gating with injected embedding model

- **Priority:** P1 · **Effort:** M · **Dimension:** Main 11-stage pipeline
- **Goal:** Run run_retrieval_index_stage fully on CPU via the embedding_model injection seam and pin the extraction/dedup/coverage flow and idempotent skip.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/retrieval_index_stage.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_retrieval_index_stage_coverage.py`
- **Validate (CPU):** .venv/bin/python -m py_compile src/blueprint_pipeline/retrieval_index_stage.py && .venv/bin/python -m pytest tests/test_retrieval_index_stage_coverage.py -q

- **Context:** Retrieval is the cross-session site-memory backbone and is fully runnable on CPU via the documented embedding_model injection seam. Pinning extraction+dedup+coverage keeps the highest-value pure-CPU stage validated while GPU embeddings are paused for the active 'open the refrigerator' G1 lane. File: src/blueprint_pipeline/retrieval_index_stage.py; test file: tests/test_retrieval_index_stage_coverage.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add a CPU-only full-run test for run_retrieval_index_stage using its embedding-model injection seam.

Background: src/blueprint_pipeline/retrieval_index_stage.py:run_retrieval_index_stage (starting ~line 73) is GPU-optional. _load_dinov3 (~1221) is the only torch/cuda path, and the function accepts an embedding_model injection ('inject for testing; loads DINOv2 if None'); _generate_embeddings (~1241-1268) honors an injected encoder. The CPU-side logic is the substance of the stage: skip gating (android_xr / world_model_candidate=false / no_site_id / already_indexed, ~lines 95-150), distance-gated dense frame extraction, ffmpeg frame materialization (~1140-1148), quality filtering, and site-index dedup. ffmpeg is available locally.

Task:
1. Read run_retrieval_index_stage's signature and the skip-gating block (~95-150), the embedding injection points, and where dense_index/site_index records and the coverage map are produced. Reuse helpers from tests/test_retrieval_index_stage_coverage.py.
2. Add a full-run test that passes embedding_model as a callable returning fixed float32 vectors (deterministic), runs the stage on a synthesized world-model-candidate capture, and asserts: the dense export manifest and site reference index rows are produced and the coverage map is non-empty, all WITHOUT importing torch (assert torch is not imported, e.g. via sys.modules check or by ensuring no real _load_dinov3 call).
3. Add a second assertion (same or sibling test): an idempotent re-run returns status=='skipped' with reason 'already_indexed'.
4. Constraints: keep world-model backends swappable (the injected embedding model must remain a clean seam — do not hardcode DINOv2/v3 specifics into the test); protect provenance/rights/privacy/raw-capture-truth (synthetic capture only, this is the cross-session site-memory backbone so be careful not to bake real site identifiers); render/eval outputs are simulator support, not policy-success claims. Run validation and ensure it passes.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-17] Pin geometry-source dispatch swappability priority ladder

- **Priority:** P1 · **Effort:** S · **Dimension:** Main 11-stage pipeline
- **Goal:** Add an explicit invariant test for resolve_geometry_source's backend selection priority order.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/geometry_sources.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_small_runtime_helper_coverage.py`
- **Validate (CPU):** .venv/bin/python -m py_compile src/blueprint_pipeline/geometry_sources.py && .venv/bin/python -m pytest tests/test_small_runtime_helper_coverage.py -q

- **Context:** src/blueprint_pipeline/geometry_sources.py implements the CLAUDE.md 'keep world-model backends swappable' rule. An explicit priority-ladder test prevents a future backend from silently shadowing arkit/arcore selection — important because the active 'open the refrigerator' G1 lane depends on correct geometry-source selection for the kitchen scene. Test file: tests/test_small_runtime_helper_coverage.py (existing per-source coverage ~305-408).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add a focused invariant test for the geometry-source dispatch priority ladder.

Background: src/blueprint_pipeline/geometry_sources.py:resolve_geometry_source and load_capture_geometry (around lines 86-124) are the world-model-backend swappability seam: they pick arkit vs arcore vs pipeline(video_to_world) geometry purely from which pose files exist plus descriptor hints. resolve_geometry_source applies a priority ladder (roughly lines 105-111): geometry_pose_path > arkit > arcore > descriptor top_level > quality_source > 'unknown'. Coverage exists in tests/test_small_runtime_helper_coverage.py (~305-408) for arkit/arcore/descriptor-source individually, but there is no single focused test asserting the FULL priority ladder as an explicit invariant. The CLAUDE.md rule 'keep world-model backends swappable' is directly implemented by this dispatch.

Task:
1. Read resolve_geometry_source (~86-124, ladder ~105-111) to confirm the exact precedence and the inputs (which pose files present, descriptor hints, quality_source).
2. Add a parametrized test in tests/test_small_runtime_helper_coverage.py that constructs contexts for each meaningful combination of present pose files + descriptor hints (including cases where a higher-priority source is present alongside lower-priority ones) and asserts resolve_geometry_source returns the expected source in strict priority order, ending with the 'unknown' fallback when nothing matches.
3. This is characterization; do not change production behavior unless you discover the ladder is actually mis-ordered, in which case fix minimally and document why.
4. Constraints: keep world-model backends swappable (the test must guard the ladder so a future backend like NuRec/splat cannot silently shadow arkit/arcore); protect provenance/rights/privacy/raw-capture-truth; render/eval outputs are simulator support, not policy-success claims. Run validation and ensure it passes.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-18] Validate swap_candidates min_volume boundary and exclude/force precedence

- **Priority:** P1 · **Effort:** S · **Dimension:** Main 11-stage pipeline
- **Goal:** Pin the non-forced volume filter boundary and exclude-vs-articulated keyword precedence in candidate selection.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/swap_candidates.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_swap_candidates_coverage.py`
- **Validate (CPU):** .venv/bin/python -m py_compile src/blueprint_pipeline/swap_candidates.py && .venv/bin/python -m pytest tests/test_swap_candidates_coverage.py -q

- **Context:** src/blueprint_pipeline/swap_candidates.py is the backend-swappable candidate-selection core. The min_volume boundary and exclude/force precedence determine which objects become separate sim assets — directly relevant to the active 'open the refrigerator' G1 lane, where the fridge must be selected as a manipulable/articulated asset. Pure-CPU branch decisions, cheap to pin. Test file: tests/test_swap_candidates_coverage.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add boundary tests for swap candidate selection.

Background: src/blueprint_pipeline/swap_candidates.py:select_swap_candidates (around lines 582-659) applies a min_volume_m3 filter (~623-625) that is bypassed when force_manipulable/force_articulated is set, and _classify_role (~547-571) excludes policy.exclude_keywords UNLESS forced. The current test in tests/test_swap_candidates_coverage.py (~line 287) covers force-bypass (tiny-forced), exclude (floor), and articulated roles, but does NOT assert (a) the non-forced volume boundary (an object just below vs just above policy.min_volume_m3 that is NOT force-flagged) nor (b) that an excluded keyword that ALSO matches an articulated keyword resolves correctly.

Task:
1. Read select_swap_candidates (~582-659), the min_volume filter (~623-625), and _classify_role (~547-571) to confirm exact comparison operators (e.g. strict < vs <=) and exclude/force precedence.
2. Add tests in tests/test_swap_candidates_coverage.py: (a) two non-force-flagged entries with volume just below and just above policy.min_volume_m3 — assert the below-threshold one is dropped and the above-threshold one is kept; (b) an entry whose label matches BOTH an exclude keyword and an articulated keyword, with and without a force flag — assert the documented precedence (exclude wins when not forced; force overrides exclude).
3. This is characterization of pure-CPU branch decisions; do not change production logic unless a real off-by-one/precedence bug surfaces, then fix minimally.
4. Constraints: keep world-model backends swappable (swap_candidates is the backend-swappable candidate-selection core — kitchen drawer + warehouse tote fixtures; keep tests backend-agnostic); protect provenance/rights/privacy/raw-capture-truth; render/eval outputs are simulator support, not policy-success claims. Run validation and ensure it passes.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-19] Test scene_semantics Gemini-success branch with a mocked genai client

- **Priority:** P1 · **Effort:** S · **Dimension:** Main 11-stage pipeline
- **Goal:** Cover the gemini-object-enumeration success path (underscore normalization, prompt_source, empty-objects fallback) with a fake genai, no paid call.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/scene_semantics.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_scene_semantics.py`
- **Validate (CPU):** .venv/bin/python -m py_compile src/blueprint_pipeline/scene_semantics.py && .venv/bin/python -m pytest tests/test_scene_semantics.py -q

- **Context:** src/blueprint_pipeline/scene_semantics.py object enumeration drives the SAM3D detection prompt bank — a regression in gemini-object-enumeration normalization silently weakens detection on every real capture, including the kitchen scene behind the active 'open the refrigerator' G1 lane. The parsing/normalization logic is fully mockable for free via the existing _fake_genai_module. Test file: tests/test_scene_semantics.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), extend scene_semantics tests to cover the Gemini SUCCESS branch using a mocked genai client (no real API call).

Background: src/blueprint_pipeline/scene_semantics.py:infer_scene_semantics (around lines 416-499) has three paths: explicit_hint (pure CPU, already tested), gemini video inference (~482-499), and local fallback (~501+). Existing tests in tests/test_scene_semantics.py (~188-228) monkeypatch _infer_with_gemini_video to exercise the failure/fallback and explicit-hint paths, and the test file already builds a _fake_genai_module helper. The SUCCESS branch is uncovered: a gemini_result with detected_objects should yield prompt_source=='gemini_object_enumeration', with underscore->space normalization (~473-474), and an empty-objects result should fall back to hardcoded prompts (~475-476). Object enumeration prompts drive the entire SAM3D detection prompt bank, so a normalization regression silently weakens detection on every real capture.

Task:
1. Read infer_scene_semantics (~416-499), focusing on the success branch (~473-476, 482-499) and how detection_prompts and prompt_source are set. Reuse the _fake_genai_module / fake _GeminiResult harness already in tests/test_scene_semantics.py.
2. Add a test that returns a fake _GeminiResult with detected_objects containing underscores (e.g. 'kitchen_drawer', 'trash_can') and asserts: detection_prompts contain the space-normalized strings ('kitchen drawer', 'trash can') and prompt_source=='gemini_object_enumeration'.
3. Add a second test where gemini_result has EMPTY detected_objects and assert the hardcoded-prompts fallback (~475-476) is taken (prompt_source reflects the fallback, prompts are the hardcoded bank).
4. Use ONLY the fake genai client — do NOT make a real Gemini call (it costs money and is non-deterministic). Do not change production behavior unless a normalization bug surfaces.
5. Constraints: keep world-model backends swappable; the genai client must stay injectable (do not hardcode a model); protect provenance/rights/privacy/raw-capture-truth; render/eval outputs are simulator support, not policy-success claims. Run validation and ensure it passes.

Do NOT launch any GPU or paid cloud pod, and do NOT make any paid Gemini/genai API call; this is CPU/no-spend only.
```

</details>

## scene_placement package

### [P1-20] Extend build_scene_index factory to construct the multi-view (preferred) backend

- **Priority:** P1 · **Effort:** S · **Dimension:** scene_placement package
- **Goal:** Let the swap-friendly factory build MultiViewPerceptionSceneSpatialIndex via backend='perception_multiview' so callers stop importing the concrete class.
- **Files:** `src/blueprint_pipeline/scene_placement/__init__.py`, `tests/test_scene_placement.py`
- **Validate (CPU):** python -m pytest tests/test_scene_placement.py -k 'factory or build_scene_index' -q ; python -m pytest tests/test_scene_placement.py -q ; python -m py_compile src/blueprint_pipeline/scene_placement/__init__.py

- **Context:** PLATFORM/CLAUDE rule: 'Keep world-model backends swappable.' The factory is the seam that enforces it; a missing dispatch for the preferred backend means every real caller bypasses the seam and re-couples to the concrete class. Files: src/blueprint_pipeline/scene_placement/__init__.py (build_scene_index ~109-123, imports ~18-35), tests/test_scene_placement.py (factory dispatch test).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), extend src/blueprint_pipeline/scene_placement/__init__.py::build_scene_index so the swappable factory can construct the backend the README calls 'preferred'.

Current state: build_scene_index only dispatches 'usd' -> UsdSceneSpatialIndex and 'perception' -> PerceptionSceneSpatialIndex (single-view). MultiViewPerceptionSceneSpatialIndex is documented as preferred but is unreachable through the factory, forcing callers to import the concrete class and bypass the seam that enforces the 'keep world-model backends swappable' rule.

What to do:
1. Add a 'perception_multiview' key that returns MultiViewPerceptionSceneSpatialIndex(**kw) (it is already imported at module top). Update the docstring to list the new backend and what kwargs it forwards (e.g. views=[...]).
2. Optionally add a 'usd_obstacles'/'usd_leaf' key only if it maps cleanly to existing UsdSceneSpatialIndex construction (e.g. a flag selecting obstacle_boxes-style fine boxes) — do NOT invent new index behavior; skip this if it requires non-trivial new code.
3. Keep the unknown-backend branch raising ValueError with an updated message listing all valid keys.

Constraints: keep world-model backends swappable (this change strengthens that — every backend must be reachable via the factory); protect provenance/rights/privacy/raw-capture-truth; render/placement outputs are simulator support, NOT policy-success claims; add/extend tests; run `python -m pytest tests/test_scene_placement.py` and `python -m py_compile src/blueprint_pipeline/scene_placement/__init__.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Extend tests/test_scene_placement.py::test_build_scene_index_factory_dispatch (or add a sibling): assert build_scene_index('perception_multiview', views=[...]) returns a MultiViewPerceptionSceneSpatialIndex instance, that 'usd' and 'perception' still dispatch correctly, and that an unknown backend still raises ValueError. Use minimal synthetic kwargs (e.g. views=[] if the constructor tolerates it, else a tiny fake view); no GPU.
```

</details>

### [P1-21] Extract broad-AABB false-positive guard into the scene_placement package

- **Priority:** P1 · **Effort:** M · **Dimension:** scene_placement package
- **Goal:** Move _broad_aabb_false_positive_clip_ids/_adjust_verdict_for_broad_aabb_false_positives out of the runner into validation.py as pure, parameterized, boundary-tested functions.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `src/blueprint_pipeline/scene_placement/validation.py`, `src/blueprint_pipeline/scene_placement/__init__.py`, `tests/test_placement_validation.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** python -m pytest tests/test_placement_validation.py -q ; python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k broad_aabb -q ; python -m py_compile src/blueprint_pipeline/scene_placement/validation.py scripts/run_isaac_g1_kitchen_parity_eval.py

- **Context:** This guard decides whether a placement is accepted or blocked and is the least-tested, most heuristic-laden piece, sitting outside the pure package where every other validation rule lives. The 'open the refrigerator' lane renders against coarse USD where a closed-door box can over-cover floor; magic 12x/4x with no boundary tests is exactly where a coarse-USD scene silently passes a real clip or rejects a valid pose. Files: scripts/run_isaac_g1_kitchen_parity_eval.py (lines 3783-3871, helpers _is_structural_or_target_obstacle / _scene_object_xy_size_area nearby, callers at ~3423 and ~4001), src/blueprint_pipeline/scene_placement/validation.py, src/blueprint_pipeline/scene_placement/__init__.py, tests/test_placement_validation.py, tests/test_isaac_g1_kitchen_parity_runner.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), relocate the broad-AABB false-positive clip-suppression logic from the runner into the pure scene_placement package so its edge cases can be pinned by the package's own validation suite.

Current state: scripts/run_isaac_g1_kitchen_parity_eval.py defines _broad_aabb_false_positive_clip_ids (lines ~3783-3828) and _adjust_verdict_for_broad_aabb_false_positives (~3831-3871). This guard can flip a PlacementVerdict from ok=False to ok=True (it strips 'clips:' failures), yet it lives runner-private with only ~2 runner-level tests, behind magic thresholds footprint_area*12.0 and half_extent*4.0.

What to do:
1. Add a pure function to src/blueprint_pipeline/scene_placement/validation.py, e.g. suppress_broad_aabb_clip_false_positives(verdict, obstacles, target, *, contact_count, footprint_half_extent, broad_area_factor=12.0, broad_span_factor=4.0, min_broad_area_m2=2.0, min_broad_span_m=1.0) -> (adjusted_verdict, suppressed: list[dict]). Promote 12.0 and 4.0 (and the max(2.0,...)/max(1.0,...) floors) to NAMED, DOCUMENTED keyword parameters with module-level DEFAULT_* constants and a docstring explaining the heuristic (zero PhysX contacts + broad non-structural AABB => coarse-USD occupancy false positive). It must depend only on SceneObject/PlacementVerdict and stdlib — no runner imports. Preserve the existing exemptions: structural/target obstacles are never suppressed, and contact_count != 0 disables suppression entirely.
2. Reduce the runner functions to thin wrappers that call the new package function (passing ROBOT_FOOTPRINT_HALF_EXTENT and the record-derived contact_count), so runner behavior is byte-for-byte unchanged. Keep _is_structural_or_target_obstacle / _scene_object_xy_size_area where they are, or pass small adapter callables — do not change their semantics.
3. Export the new function from __init__.py.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (suppression must remain conservative — never suppress a structural or target clip); render/placement outputs are simulator support, NOT policy-success claims — be explicit in the docstring that suppressing a coarse-USD clip is an occupancy-fidelity correction, not a claim the robot succeeded; add/extend tests; run `python -m pytest tests/test_placement_validation.py tests/test_isaac_g1_kitchen_parity_runner.py` and `python -m py_compile src/blueprint_pipeline/scene_placement/validation.py scripts/run_isaac_g1_kitchen_parity_eval.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Add pure tests in tests/test_placement_validation.py for the extracted function: a clip whose area/span is JUST below the threshold is NOT suppressed; JUST above IS suppressed; AT the threshold (document the inclusive/exclusive boundary and test it); a structural obstacle and the target obstacle are never suppressed; contact_count != 0 disables all suppression; multiple broad clips are all handled. Keep tests/test_isaac_g1_kitchen_parity_runner.py -k broad_aabb green to prove the wrapper preserves runner behavior. Pure synthetic boxes, no GPU.
```

</details>

### [P1-22] Validate against fine obstacle_boxes() instead of grouped objects() in place_and_validate

- **Priority:** P1 · **Effort:** M · **Dimension:** scene_placement package
- **Goal:** Make place_and_validate_robot_for_task pull obstacle_boxes() for clip validation when the index exposes it, so the in-package orchestrator is correct by construction.
- **Files:** `src/blueprint_pipeline/scene_placement/__init__.py`, `src/blueprint_pipeline/scene_placement/usd_index.py`, `tests/test_scene_placement.py`
- **Validate (CPU):** python -m pytest tests/test_scene_placement.py -k 'place_and_validate' -q ; python -m pytest tests/test_scene_placement.py -q ; python -m py_compile src/blueprint_pipeline/scene_placement/__init__.py

- **Context:** Validating against grouped objects is precisely the broad-AABB false positive the runner then suppresses with a heuristic guard. Doing the fine/grouped split inside the package's own orchestrator makes place_and_validate_robot_for_task correct by construction and reduces reliance on the runner-side suppression hack — complementary to extracting that guard. Files: src/blueprint_pipeline/scene_placement/__init__.py (place_and_validate_robot_for_task ~159-191), src/blueprint_pipeline/scene_placement/usd_index.py (objects ~266, obstacle_boxes ~281-301), src/blueprint_pipeline/scene_placement/validation.py (NOTE ~25-30), scripts/run_isaac_g1_kitchen_parity_eval.py (_placement_obstacles_for_stage ~3510-3534).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), fix place_and_validate_robot_for_task in src/blueprint_pipeline/scene_placement/__init__.py so clip validation runs against FINE obstacle boxes, not the grouped target catalog.

Current state: place_and_validate_robot_for_task enumerates index.objects() once and uses it for BOTH target resolution AND clip validation. validation.py's own NOTE (lines ~25-30) warns that the grouped catalog collapses a cabinet run into a broad AABB covering open aisle floor, and that obstacle_boxes() should be used for clipping. The runner already does this split via _placement_obstacles_for_stage (scripts/run_isaac_g1_kitchen_parity_eval.py ~3510-3534, which prefers index.obstacle_boxes()).

What to do:
1. In place_and_validate_robot_for_task, resolve the target from index.objects() (grouped is correct for resolution). For validation, if the index exposes a callable obstacle_boxes(), use its output as the obstacle catalog passed to validate_placement; otherwise fall back to objects() (so duck-typed fakes and the perception index without obstacle_boxes still work). Use getattr(index, 'obstacle_boxes', None) and callable(...) so the SceneSpatialIndex protocol is not broken.
2. Keep place_robot_for_task (non-validating) unchanged. Update the place_and_validate docstring to state that target resolution uses grouped objects() and clip validation uses obstacle_boxes() when available, mirroring the runner split.

Constraints: keep world-model backends swappable (probe via getattr, do NOT add obstacle_boxes to the required SceneSpatialIndex protocol — it stays optional); protect provenance/rights/privacy/raw-capture-truth; render/placement outputs are simulator support, NOT policy-success claims; add/extend tests; run `python -m pytest tests/test_scene_placement.py` and `python -m py_compile src/blueprint_pipeline/scene_placement/__init__.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Add a test in tests/test_scene_placement.py with a fake index exposing BOTH objects() (a broad grouped cabinet box covering the front-of-cabinet floor) and obstacle_boxes() (fine per-cabinet boxes that do NOT cover the stand spot): assert place_and_validate_robot_for_task resolves the target from the grouped catalog but the verdict is NOT flagged clipping (because validation consumed the fine boxes). Also assert a fake index WITHOUT obstacle_boxes still validates via objects() unchanged. Pure synthetic boxes, mock probe, no GPU.
```

</details>

### [P1-23] Harden _clean_label for suffix-only and index-only prim names

- **Priority:** P1 · **Effort:** S · **Dimension:** scene_placement package
- **Goal:** Make degenerate prim names ('_geo','_mesh','001') reduce to '' so they are skipped, instead of leaking bogus labels like 'geo'/'mesh'.
- **Files:** `src/blueprint_pipeline/scene_placement/usd_index.py`, `tests/test_scene_placement.py`
- **Validate (CPU):** python -m pytest tests/test_scene_placement.py -k clean_label -q ; python -m pytest tests/test_scene_placement.py -q ; python -m py_compile src/blueprint_pipeline/scene_placement/usd_index.py ; python3 -c "from src.blueprint_pipeline.scene_placement.usd_index import _clean_label as c; assert c('_geo')=='' and c('_mesh')=='' and c('Faucet_geo')=='faucet', (c('_geo'),c('Faucet_geo'))"

- **Context:** _clean_label is the sole normalizer between raw USD prim names and task matching, so its edge cases directly corrupt target resolution: a 'geo'/'mesh' object can win resolution and become a robot stand target. This matters for the 'open the refrigerator' lane where authored fridge sub-meshes (Fridge_geo, Door_mesh) are common. Files: src/blueprint_pipeline/scene_placement/usd_index.py (_clean_label ~67-103, _LABEL_STRIP_SUFFIXES, _objects_from_bounds ~139-186), tests/test_scene_placement.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), fix the _clean_label edge cases in src/blueprint_pipeline/scene_placement/usd_index.py.

Confirmed problem: _clean_label('_geo') returns 'geo', _clean_label('_mesh') returns 'mesh'. The structural-suffix strip is gated on len(label) > len(suffix), so a name that is ONLY a structural suffix (with or without a leading separator) strips just the underscore and keeps the bare suffix word as a bogus label. Index-only names already reduce correctly (_clean_label('001') -> ''), so the gap is specifically suffix-only names. A SceneObject labeled 'geo'/'mesh'/'link' is noise the resolver can match against and place a robot for.

What to do:
1. In _clean_label (lines ~67-103), change the structural-suffix handling so that a name whose ENTIRE meaningful content is a structural suffix reduces to '' (skip it), while real names with a trailing structural suffix still strip only the suffix ('Faucet_geo' -> 'faucet', 'Stove_link' -> 'stove'). Concretely: after normalizing separators and trimming, if the remaining token (case-insensitive) is itself one of the structural suffix words, return ''. Make sure stacked decorations still fully reduce ('Faucet_01_geo' -> 'faucet').
2. Do not regress the existing index-only behavior ('001' -> '', '01' -> '') or any currently-passing _clean_label test.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (dropping a meaningless structural-suffix label is not data loss — it is noise removal; do not invent a label for a nameless prim); render/placement outputs are simulator support, NOT policy-success claims; add/extend tests; run `python -m pytest tests/test_scene_placement.py` and `python -m py_compile src/blueprint_pipeline/scene_placement/usd_index.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Add cases in tests/test_scene_placement.py (clean_label tests): _clean_label('_geo') == '' , _clean_label('_mesh') == '' , _clean_label('link') == '' , and regression asserts _clean_label('Faucet_geo') == 'faucet', _clean_label('Faucet_01_geo') == 'faucet', _clean_label('001') == ''. Also assert _objects_from_bounds skips a prim named '_geo' (no SceneObject produced). Pure, no GPU.
```

</details>

### [P1-24] Add usd-core round-trip test of the real USD walk on a synthetic .usda

- **Priority:** P1 · **Effort:** M · **Dimension:** scene_placement package
- **Goal:** Cover the only pxr-touching code (BBoxCache/PrimRange/PruneChildren) with a CPU usd-core test that builds a tiny stage and runs UsdSceneSpatialIndex(stage=...).objects()/obstacle_boxes().
- **Files:** `src/blueprint_pipeline/scene_placement/usd_index.py`, `tests/test_scene_placement.py`
- **Validate (CPU):** python -m pytest tests/test_scene_placement.py -k 'usda or real_stage or pxr' -q ; python -m pytest tests/test_scene_placement.py -q ; python -m py_compile src/blueprint_pipeline/scene_placement/usd_index.py ; (test must self-skip via pytest.importorskip('pxr') where USD is absent)

- **Context:** The task notes 'All USD logic is usd-core testable' and pxr imports successfully on this machine. The pure helpers (_clean_label/_is_excluded/_objects_from_bounds) are heavily tested but the BBoxCache/PrimRange/PruneChildren glue is faked everywhere, so a USD-version drift in that glue would silently break object enumeration with all unit tests green. Files: src/blueprint_pipeline/scene_placement/usd_index.py (_walk_stage ~306-493, objects ~266, obstacle_boxes ~281-301, _drop_degenerate_boxes ~202 with 6.0 m cap), tests/test_scene_placement.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), add a CPU test that drives src/blueprint_pipeline/scene_placement/usd_index.py's _walk_stage against a REAL usd-core (pxr) stage. Today every _walk_stage test monkeypatches pxr, so the actual BBoxCache/PrimRange/PruneChildren/_subtree_has_gprim interaction — the only pxr-touching code and the part most likely to drift across USD versions — has zero coverage. pxr (usd-core) is available on CPU with no GPU/cloud.

What to do:
1. In tests/test_scene_placement.py (or a new tests/test_usd_index_real_stage.py), add a test guarded by pytest.importorskip('pxr') so it stays hermetic where USD is absent. In it, build a small in-memory stage with Usd.Stage.CreateInMemory(): a /World/Scene wrapper Xform; a multi-mesh sink (an Xform 'Sink' with two child Mesh gprims) to exercise submesh collapse; a wall ('EastWall' or 'Wall_North') to exercise shell exclusion; and a degenerate/empty leaf (an Xform with no gprim, or a gprim with a tiny/zero extent) to exercise the wrapper-non-emission and degenerate-box drop. Give the meshes real points/extent so BBoxCache returns finite bounds.
2. Run UsdSceneSpatialIndex(stage=stage).objects() and assert: the sink appears once (submeshes collapsed to one object, not two), the wall is excluded, the /World/Scene wrapper does not emit its own object, and any degenerate leaf is dropped. Then call obstacle_boxes() and assert it returns the finer boxes (and still drops degenerate/oversize ones via the 6.0 m cap).
3. Keep the synthetic stage construction minimal and well-commented; use UsdGeom.Mesh with explicit points or a Cube with a size so the bound is deterministic.

Constraints: keep world-model backends swappable (test through UsdSceneSpatialIndex's public objects()/obstacle_boxes(), not private _walk_stage internals where avoidable); protect provenance/rights/privacy/raw-capture-truth (synthetic stage only, no real capture .usd); render/placement outputs are simulator support, NOT policy-success claims; this is a test addition; run `python -m pytest tests/test_scene_placement.py` and `python -m py_compile` on touched files. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only (usd-core is a pure CPU wheel; pxr is already importable in this environment).
```

</details>

### [P1-25] Detect multi-target tasks and pin a deterministic contract

- **Priority:** P1 · **Effort:** M · **Dimension:** scene_placement package
- **Goal:** Stop silently dropping all-but-one target on conjunction tasks ('the faucet and the stove'); detect multi-noun intent and surface a deterministic primary + an ambiguity flag.
- **Files:** `src/blueprint_pipeline/scene_placement/target_resolver.py`, `tests/test_scene_placement.py`
- **Validate (CPU):** python -m pytest tests/test_scene_placement.py -k 'label_fallback or multi_target' -q ; python -m pytest tests/test_scene_placement.py -q ; python -m py_compile src/blueprint_pipeline/scene_placement/target_resolver.py

- **Context:** Free-form task strings will contain conjunctions; silently dropping all but one target produces a confidently-wrong placement with no diagnostic, and the package's whole value is task->target correctness. Pinning the contract (even if the chosen behavior is 'pick a deterministic primary + flag ambiguous') is load-bearing for trust in the dynamic-placement story. Files: src/blueprint_pipeline/scene_placement/target_resolver.py (_task_intent_tokens ~302-313, _SYNONYM_GROUPS ~59-80, resolve_target_by_label ~345-371, build_target_prompt ~134-170), tests/test_scene_placement.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), make multi-target task strings produce a documented, deterministic outcome in src/blueprint_pipeline/scene_placement/target_resolver.py instead of silently picking the longest token.

Current state: resolve_target / resolve_target_by_label return exactly one SceneObject, and build_target_prompt hard-codes 'Pick the SINGLE object'. A task like 'open the fridge and the microwave' or 'turn on the faucet and the stove' has more than one distinct matching synonym group in its intent tokens, but the resolver just returns one with no diagnostic.

What to do (decide the contract, then implement and PIN it with tests — do not leave it to longest-token tie-breaking):
1. Add a pure detector, e.g. detect_multi_target(task) -> bool (or a helper returning the list of distinct matched synonym groups). It tokenizes via the existing _task_intent_tokens, maps each content noun to its synonym group, and reports True when two or more DISTINCT fixture groups are present (so 'faucet'+'tap' is ONE target, but 'faucet'+'stove' is two). Reuse _synonyms_of / the synonym table; do not duplicate it.
2. Surface it without breaking the single-target API. Choose ONE: (a) add resolve_targets(task, objects) -> list[SceneObject] returning all matched primaries deterministically ordered, and keep resolve_target returning the first/primary; OR (b) keep returning one object but attach an explicit signal (e.g. a module-level helper callers invoke, or a documented note) so a caller can detect 'ambiguous_multi_target'. Document the chosen primary rule (e.g. first matched intent token in longest-first order) so it is deterministic, not incidental.
3. Do not change behavior for genuine single-target tasks (including synonym-only variation like 'tap'/'faucet').

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render/placement outputs are simulator support, NOT policy-success claims; add/extend tests; run `python -m pytest tests/test_scene_placement.py` and `python -m py_compile src/blueprint_pipeline/scene_placement/target_resolver.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Add tests in tests/test_scene_placement.py (label_fallback area): a conjunction task across two distinct fixtures sets the multi-target signal True and returns the documented deterministic primary; a single-fixture task (and a synonym-only variant) sets it False and resolves unchanged. Pure synthetic objects, no GPU.
```

</details>

### [P1-26] Close the validation rotation-frame blind spot with a geometric look-at cross-check

- **Priority:** P1 · **Effort:** M · **Dimension:** scene_placement package
- **Goal:** Convert validate_stand_pose's known sub-threshold pure-rotation blind spot into a detected failure via a render-free forward-axis look-at check.
- **Files:** `src/blueprint_pipeline/scene_placement/validation.py`, `tests/test_placement_validation.py`
- **Validate (CPU):** python -m pytest tests/test_placement_validation.py -k rotation -q ; python -m pytest tests/test_placement_validation.py -q ; python -m py_compile src/blueprint_pipeline/scene_placement/validation.py

- **Context:** The articulated 'open the refrigerator' head-POV seed depends on the robot actually FACING the door; a flipped/rotated frame under 30deg renders the back of the head or a side view yet currently validates as PASS. This is exactly the class of bug the validator exists to catch pre-render. Confidence medium because the fix must model the runner's actual forward/look-at convention. Files: src/blueprint_pipeline/scene_placement/validation.py (validate_stand_pose ~174-333, docstring caveat ~18-22, _angle_diff_deg ~90-99, DEFAULT_VALIDATION_MAX_FACING_ERROR_DEG ~46), scripts/run_isaac_g1_kitchen_parity_eval.py (head-POV forward pitch/look-at convention, e.g. MANIPULATION_POV_HEAD_FORWARD_PITCH_DOWN_DEG ~142), tests/test_placement_validation.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), add a geometric cross-check to validate_stand_pose in src/blueprint_pipeline/scene_placement/validation.py that catches a flipped/rotated frame convention which currently passes.

Current state: the docstring (lines ~18-22) admits a sub-threshold pure-ROTATION frame bug passes BOTH the intent and actual checks — only gross (> max_facing_error_deg, default 30) heading errors are caught, and test_subthreshold_rotation_frame_error_is_a_known_blind_spot pins it as a KNOWN gap. The facing check compares yaw to the bearing-to-target but does not verify the robot's forward axis is applied in the SAME frame the runner uses to place the root.

What to do:
1. Add a cheap, render-free check: given the pose yaw and the convention by which the runner derives the head/forward look-at direction, compute where the robot's forward axis points when that convention is applied, and assert the resulting look-at vector lands within tolerance of the target direction. Concretely, parameterize the forward-axis convention (e.g. forward_axis_sign or a yaw_offset_deg representing the frame mapping the runner uses), default it to the correct convention, and FAIL the verdict when applying a flipped/rotated convention sends the forward vector away from the target even though the raw yaw-vs-bearing diff is under tolerance. Add a named tolerance constant.
2. Add the failure to PlacementVerdict.failures with an explicit reason (e.g. 'forward_frame_mismatch') and document the new check in the docstring, replacing the 'known blind spot' caveat with the closed behavior (or narrowing it precisely if some residual case remains).
3. Do not regress existing facing tests; the correct-yaw correct-frame case must still PASS.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render/placement outputs are simulator support, NOT policy-success claims (this check reduces a render-time facing bug to a pre-render geometric failure — it is not a success guarantee); add/extend tests; run `python -m pytest tests/test_placement_validation.py` and `python -m py_compile src/blueprint_pipeline/scene_placement/validation.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Add a test in tests/test_placement_validation.py (rotation area): a pose with the CORRECT yaw-vs-bearing but a FLIPPED frame convention applied now FAILS the new check (forward_frame_mismatch), while the correct-convention pose still PASSES. Update or supersede test_subthreshold_rotation_frame_error_is_a_known_blind_spot to reflect the now-detected case. Pure trig, no GPU/render.
```

</details>

### [P1-27] Classify openable vs static targets and consume it in placement/validation

- **Priority:** P1 · **Effort:** M · **Dimension:** scene_placement package
- **Goal:** Add an openable classifier (fridge/oven/microwave/dishwasher/door/drawer/cabinet) and have placement/validation reason about door swing for the 'open the refrigerator' lane.
- **Files:** `src/blueprint_pipeline/scene_placement/target_resolver.py`, `src/blueprint_pipeline/scene_placement/types.py`, `src/blueprint_pipeline/scene_placement/placement.py`, `tests/test_scene_placement.py`
- **Validate (CPU):** python -m pytest tests/test_scene_placement.py -k 'openable or classify' -q ; python -m pytest tests/test_scene_placement.py -q ; python -m py_compile src/blueprint_pipeline/scene_placement/target_resolver.py src/blueprint_pipeline/scene_placement/placement.py src/blueprint_pipeline/scene_placement/types.py

- **Context:** The active lane is specifically 'open the refrigerator' — an articulated task. Without classifying openables, placement cannot reason about door swing or hinge side, and the broad-AABB guard cannot distinguish a closed-door box from a static counter. This is foundational label/geometry logic for the parity story. Files: src/blueprint_pipeline/scene_placement/target_resolver.py (_SYNONYM_GROUPS ~59-80), src/blueprint_pipeline/scene_placement/types.py (SceneObject.category ~38), src/blueprint_pipeline/scene_placement/placement.py (compute_stand_pose ~148-260), tests/test_scene_placement.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), introduce an articulated/openable classification for placement targets and consume it in the solver/validator.

Current state: nothing distinguishes an articulated/openable target (fridge, oven, microwave, dishwasher, door, drawer, cabinet) from a static fixture (faucet, sink). SceneObject.category exists but nothing populates or consumes an 'openable' flag, and the synonym table in target_resolver.py mixes openables and statics with no tag.

What to do:
1. Add a pure classifier, e.g. is_openable_target(obj) (or classify_target_kind(obj) -> 'openable'|'static') in target_resolver.py (or a small new module imported by it), keyed off label/category tokens: fridge/refrigerator/freezer, oven, microwave, dishwasher, door, drawer, cabinet/cupboard => openable; faucet/sink/tap/spout/counter => static. Reuse the existing synonym groups where possible; keep it stdlib-pure.
2. Consume the flag in placement.compute_stand_pose and/or validation: at minimum, when the target is openable, widen the standoff lower bound so the pelvis is far enough back that a swung door does not intersect the footprint (you may approximate the swing by augmenting the standoff for openables; the full swing-arc model is covered by a separate close-reach task — keep this one to the openable/static distinction and a conservative standoff bump). Expose it as an opt-in parameter so static-target behavior is unchanged by default.
3. Thread the classification onto SceneObject.category or extra when the index builds objects, OR compute it on demand in the resolver — pick the lighter option and document it.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (classification is label-derived inference, not asserted ground truth — keep it advisory); render/placement outputs are simulator support, NOT policy-success claims; add/extend tests; run `python -m pytest tests/test_scene_placement.py` and `python -m py_compile` on touched files. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Add tests in tests/test_scene_placement.py: fridge/drawer/door/oven/dishwasher classify openable; faucet/sink/counter classify static; an openable target produces a stand plan with a larger lower-bound standoff than the same-geometry static target. Pure synthetic objects, no GPU.
```

</details>

### [P1-28] Validate degenerate cameras and oversize unprojected boxes in perception_index

- **Priority:** P1 · **Effort:** S · **Dimension:** scene_placement package
- **Goal:** Reject eye==target cameras (zero forward) and cap/flag room-spanning unprojected AABBs so one bad detection cannot win target resolution.
- **Files:** `src/blueprint_pipeline/scene_placement/perception_index.py`, `tests/test_scene_placement.py`
- **Validate (CPU):** python -m pytest tests/test_scene_placement.py -k 'perception or unproject or camera_basis' -q ; python -m pytest tests/test_scene_placement.py -q ; python -m py_compile src/blueprint_pipeline/scene_placement/perception_index.py

- **Context:** Perception detections feed resolve_target/compute_stand_pose directly; a degenerate camera yields a basis with zero forward and nonsense world points, and a single bad box can produce a room-spanning 'object' that wins target resolution with no diagnostic. Files: src/blueprint_pipeline/scene_placement/perception_index.py (_normalize ~72, camera_basis ~152, unproject ~200, _sample_box_depth ~221, _aabb_from_points ~268), src/blueprint_pipeline/scene_placement/usd_index.py (_drop_degenerate_boxes ~202, 6.0 m cap, for parity), tests/test_scene_placement.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), add input validation to the perception unprojection path in src/blueprint_pipeline/scene_placement/perception_index.py.

Current state: _sample_box_depth drops non-finite/non-positive depths, but camera_basis/unproject do not guard a fully-degenerate camera (eye==target gives a zero forward vector; _normalize returns the zero vector unchanged rather than raising), and there is no perception-side analogue of the USD _drop_degenerate_boxes 6m cap — so a detection box whose corner unprojection mixes a finite center depth with the same depth on far-offset corners can produce a room-spanning AABB that wins target resolution.

What to do:
1. In camera_basis (or its callers), detect a degenerate camera where eye and target coincide (forward ~ zero after _normalize) and either raise ValueError with an explicit message or return a clearly-flagged result the caller rejects — do NOT silently proceed with a zero forward. Pick one and document it.
2. Add an oversize-AABB guard on the perception side mirroring usd_index._drop_degenerate_boxes: after building each world AABB, drop or flag boxes whose max extent exceeds a sane named cap (reuse the 6.0 m idea; expose as a constant/param). Keep finite-depth filtering as-is.
3. Do not change valid-input results.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (drop/flag implausible boxes, do not silently shrink real geometry to fit the cap — dropping a clearly-degenerate detection is fine, fabricating a plausible size is not); render/placement outputs are simulator support, NOT policy-success claims; add/extend tests; run `python -m pytest tests/test_scene_placement.py` and `python -m py_compile src/blueprint_pipeline/scene_placement/perception_index.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Add tests in tests/test_scene_placement.py: eye==target camera raises (or returns the documented flagged result); an unprojected AABB exceeding the cap is dropped/flagged and does not appear in objects(); a normal camera + box still unprojects correctly. Pure synthetic numbers, no GPU.
```

</details>

## provider_race orchestrator

### [P1-29] Add a --providers (multi) / --race CLI flag to both render jobs

- **Priority:** P1 · **Effort:** S · **Dimension:** provider_race orchestrator
- **Goal:** Expose a CLI surface so operators can request a multi-provider race, while keeping single --provider as the backward-compatible default.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_particlefield_render_job.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -q ; ls tests | grep -i particlefield (then pytest that file) ; python -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py src/blueprint_pipeline/isaac_particlefield_render_job.py

- **Context:** Even once race_launch is wired internally, operators have no way to request a race — the feature stays dormant. This is pure argument parsing, fully CPU/hermetic. It complements the G1 race wiring for the active 'open the refrigerator' lane by giving the operator the actual switch to flip. Pairs naturally with the internal wiring task but is independently testable.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), both render-job CLIs only expose a single backend: src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py:664 and src/blueprint_pipeline/isaac_particlefield_render_job.py:452 both define ap.add_argument('--provider', default='runpod', choices=['runpod','vast']). With only one provider selectable, the provider_race.race_launch path is unreachable from the command line.

Task: add a way to request multiple providers (e.g. ap.add_argument('--providers', nargs='+', choices=['runpod','vast']) OR a --race boolean flag that expands to ['runpod','vast']). When multiple providers are given, the CLI must route into the multi-provider race path of the run_* function (if the race wiring exists) or, if that wiring is not yet present, construct and pass the provider list through the function's parameters in a way that is forward-compatible. Keep the existing single --provider default working exactly as before (default stays runpod; passing a single --provider must behave identically to today). Validate the args sanely (e.g. --providers and a non-default --provider together should error or have a documented precedence). Do this for BOTH CLIs symmetrically.

Constraints: keep provider backends swappable; render outputs are simulator support NOT policy-success claims; add/extend tests at the argparse level; Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: add hermetic argparse-level tests (call main(argv) with --allow-paid omitted, or parse_args directly) asserting (1) '--providers runpod vast' parses to a two-provider plan/list; (2) the default (no flag) still resolves to a single runpod provider; (3) a single '--provider vast' is unchanged. Run python -m pytest on tests/test_isaac_g1_kitchen_parity_runner.py and the particlefield runner test (locate it under tests/), plus python -m py_compile on both job files.
```

</details>

### [P1-30] Define how the race interacts with the serve=True warm-pool path

- **Priority:** P1 · **Effort:** M · **Dimension:** provider_race orchestrator
- **Goal:** Make the interaction between race_launch and the serve=True warm-pool branch explicit so a race never tears down the warm pod the caller needs alive.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/provider_race.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k 'serve or warm' -q ; python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -q ; python -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py src/blueprint_pipeline/provider_race.py

- **Context:** If race_launch is naively wired into the serve path it would tear down the warm pod the caller needs alive, breaking the WarmPoolClient lifecycle — the warm-serve lane (used by the active G1 'open the refrigerator' render lane for low-latency repeated submissions) would silently break. The two markers differ: race/bootstrap.json (early heartbeat) vs warm_serve_ready.json (Isaac booted + scene loaded), so the interaction can't be left implicit. Best done after, or together with, the G1 race wiring.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), run_isaac_g1_kitchen_parity_job has a serve=True branch (src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py:607-624) that leaves the WINNING pod RUNNING, waits on a DIFFERENT readiness marker via _await_warm_serve_ready (it polls for 'warm_serve_ready.json', not the early 'bootstrap.json'), and deliberately does NOT call watch_and_collect or teardown — the warm pod must stay alive for the caller's WarmPoolClient. Meanwhile src/blueprint_pipeline/provider_race.py:race_launch always terminates losers and only knows whatever marker_check it is handed.

Task: decide and implement how serve mode and the race coexist. Two acceptable designs (pick one and document it in both files' docstrings/comments): (A) serve mode bypasses race_launch entirely and remains single-provider — in which case add an explicit guard/comment so a future multi-provider serve request raises a clear error or falls back to single-provider rather than silently racing and tearing down the warm pod; OR (B) serve mode supports racing: race to first serve-ready (pass a serve-ready marker_check that looks for warm_serve_ready.json), keep the winner RUNNING, and stop() — never terminate() — the losers (reuse the warm-pod-preserving teardown). If you choose (B), ensure the winning pod is NOT torn down and that _await_warm_serve_ready still runs against the winner.

Constraints: keep provider backends swappable; render outputs are simulator support NOT policy-success claims; protect provenance/rights/privacy/raw-capture-truth; the warm pod lifecycle (WarmPoolClient ownership) must not be broken; add/extend tests; Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: add hermetic tests asserting your chosen behavior. For (A): a serve=True multi-provider request does not call race_launch (or raises a clear documented error) and never terminates a winner. For (B): a serve-mode race keeps the winner running (winner never stop/terminate'd) and stop()s losers, using an injected serve-ready marker_check. Run python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py and python -m py_compile on touched files.
```

</details>

### [P1-31] Surface + persist the circuit-breaker snapshot across jobs

- **Priority:** P1 · **Effort:** M · **Dimension:** provider_race orchestrator
- **Goal:** Give ProviderCircuitBreaker cross-launch memory by persisting its state to JSON and emitting its snapshot (plus skipped providers) into the launch manifest.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/provider_race.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_provider_race.py`
- **Validate (CPU):** python -m pytest tests/test_provider_race.py -k 'persist or snapshot or reload' -q ; python -m pytest tests/test_provider_race.py -q ; python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -q ; python -m py_compile src/blueprint_pipeline/provider_race.py src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py

- **Context:** The circuit breaker's entire value is memory across launches so a chronically bad GPU pool stops being raced and paid for. Without persistence it resets every run and never trips in production. This protects spend on the active 'open the refrigerator' G1 render lane once racing is enabled. Depends on the G1 race wiring existing for the manifest-emission half; the persistence half is independently buildable and testable now.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), src/blueprint_pipeline/provider_race.py:ProviderCircuitBreaker.snapshot() emits per-provider lifetime + recent health, but nothing constructs a breaker, threads it through race_launch, records its snapshot into a manifest, or persists it across job invocations. As written, every job starts with an empty breaker — so a chronically bad pool is never actually skipped in production (the breaker is decorative). The module's stated purpose (provider_race.py:15-19) is memory ACROSS launches.

Task: (1) Add simple JSON persistence to ProviderCircuitBreaker: a to_dict()/from_dict() (or save(path)/load(path) classmethod) that round-trips the per-provider recent window (deques) and lifetime totals so trip state survives a reload. Keep it thread-safe (use the existing lock). Choose a sensible on-disk shape (deques become lists). (2) Make the breaker loadable from / saveable to a path keyed appropriately (e.g. one JSON file under a stable state dir or out_dir). (3) When the G1 job takes the race path, construct/restore a breaker, pass it to race_launch, and after the race write the race result's 'skipped' list AND breaker.snapshot() into manifest['launch'] (or a manifest['provider_race'] block), then persist the updated breaker. Do not break the existing no-breaker path (circuit_breaker=None must still work).

Constraints: keep provider backends swappable; render outputs are simulator support NOT policy-success claims; protect provenance/rights/privacy (the persisted file holds only provider names + success/dud counts — no capture data, no secrets); add/extend tests; Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: add hermetic tests (tmp_path only): (a) a breaker recorded into a tripped state, saved to a temp JSON and reloaded, still reports is_tripped()==True and preserves lifetime counts; (b) the wired G1 race path writes breaker.snapshot() + skipped into its manifest (use fake providers + stubbed watch_and_collect, allow_paid=True). Run python -m pytest tests/test_provider_race.py tests/test_isaac_g1_kitchen_parity_runner.py and python -m py_compile on touched files.
```

</details>

### [P1-32] Test the all-providers-tripped -> race-all fallback

- **Priority:** P1 · **Effort:** S · **Dimension:** provider_race orchestrator
- **Goal:** Add a hermetic test proving that when the breaker trips every provider, race_launch still races them all healthiest-first rather than dead-ending.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/tests/test_provider_race.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/provider_race.py`
- **Validate (CPU):** python -m pytest tests/test_provider_race.py -k all_tripped -q ; python -m pytest tests/test_provider_race.py -q ; python -m py_compile tests/test_provider_race.py

- **Context:** This is the explicit 'don't dead-end the job' safety branch — load-bearing for the active 'open the refrigerator' lane, since a transient bad streak across BOTH providers must still attempt a launch rather than returning blocked. It is currently untested, so a regression that returns no_providers when all are tripped would pass CI silently.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), src/blueprint_pipeline/provider_race.py:race_launch has a safety branch: when circuit_breaker.partition() yields an empty runnable list (every provider tripped), it falls through to runnable = circuit_breaker.order(providers) and races them all healthiest-first (provider_race.py:214-215) rather than returning no_providers. No test exercises this branch — the existing test_race_returns_fast_booter_and_terminates_the_rest only trips ONE of three providers.

Task: add a hermetic test to tests/test_provider_race.py (model fakes on the existing FakeProvider) that constructs a ProviderCircuitBreaker and records enough duds to trip EVERY provider, then calls race_launch with those providers. Assert: (1) all providers are still launched (none skipped) — e.g. assert res['skipped'] == [] and every fake's launch_calls >= 1; (2) a healthy fast booter among them still wins; (3) they were ordered healthiest-first (give the eventual winner the lowest dud-rate so order() puts it first, and assert the winner is that provider). Use the injected _NO_SLEEP and a small poll_interval as the existing tests do.

Constraints: fully hermetic (fake providers, injected sleep/monotonic), no GPU, no network. render outputs are simulator support NOT policy-success claims. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: python -m pytest tests/test_provider_race.py -k all_tripped (or your chosen test name) and the full tests/test_provider_race.py; python -m py_compile on the test file.
```

</details>

### [P1-33] Test booted_lost classification + breaker success feedback

- **Priority:** P1 · **Effort:** S · **Dimension:** provider_race orchestrator
- **Goal:** Add a deterministic test that a slower-but-healthy provider gets outcome 'booted_lost' and is recorded as a breaker SUCCESS, not penalized as a dud.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/tests/test_provider_race.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/provider_race.py`
- **Validate (CPU):** python -m pytest tests/test_provider_race.py -k booted_lost -q ; python -m pytest tests/test_provider_race.py -q ; python -m py_compile tests/test_provider_race.py

- **Context:** booted_lost -> success is load-bearing: it prevents a healthy-but-slower provider from being unfairly tripped as a dud over time. Without a test, a refactor could misclassify it and wrongly trip a healthy provider — directly harming provider availability for the active 'open the refrigerator' G1 render lane. Deterministic construction (controllable marker) is the key challenge.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), src/blueprint_pipeline/provider_race.py:race_launch classifies a provider that boots AFTER a winner is already chosen as outcome 'booted_lost' (provider_race.py:289) and feeds it to the breaker as a SUCCESS (provider_race.py:328). Existing tests assert the slower booter is terminated but never assert outcome=='booted_lost' nor that it records a breaker success.

Task: add a deterministic hermetic test to tests/test_provider_race.py. Make a winner that fires its marker first and a second provider that DOES eventually boot but later (e.g. marker_after high enough that the winner is chosen first, but with enough poll attempts that it still flips its marker before the loser's poll is aborted — OR use a controllable marker so the slow provider returns True on a later poll). Pass a ProviderCircuitBreaker. Assert: (1) the slow provider's contender record has outcome == 'booted_lost'; (2) breaker.snapshot()[slow_name]['success'] >= 1 and ['dud'] == 0 (it was NOT penalized); (3) the winner's outcome == 'won' and success >= 1. Make it deterministic — do not rely on thread scheduling races; control the marker timing so booted_lost is guaranteed.

Constraints: fully hermetic; injected sleep/monotonic; no GPU; no network. render outputs are simulator support NOT policy-success claims. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: python -m pytest tests/test_provider_race.py -k booted_lost and the full file; python -m py_compile on the test file.
```

</details>

### [P1-34] Test marker_check-raising recovery and terminate()-raising resilience

- **Priority:** P1 · **Effort:** S · **Dimension:** provider_race orchestrator
- **Goal:** Cover the two swallow-and-continue resilience paths: a marker_check that raises then recovers, and a loser terminate() that raises.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/tests/test_provider_race.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/provider_race.py`
- **Validate (CPU):** python -m pytest tests/test_provider_race.py -k 'marker_raises or terminate_raises' -q ; python -m pytest tests/test_provider_race.py -q ; python -m py_compile tests/test_provider_race.py

- **Context:** These are the resilience guarantees that keep a flaky boot probe or a flaky teardown API from sinking the whole race or leaking an exception to the caller — important for the active 'open the refrigerator' lane where real provider APIs are intermittently flaky. They are currently unverified, so the swallow-and-continue behavior could regress unnoticed.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), src/blueprint_pipeline/provider_race.py:race_launch has two untested resilience guarantees: (1) it swallows marker_check exceptions per poll (provider_race.py:275-276 — records 'marker_check_raised:...' reason but keeps polling); (2) it swallows terminate() exceptions during loser teardown (provider_race.py:318-319 — records {'status':'terminate_failed',...} but does not propagate).

Task: add two hermetic tests to tests/test_provider_race.py (subclass/extend FakeProvider and/or wrap _marker_check). (a) marker_check raises on the FIRST poll then returns True on the next: assert the provider still boots and wins, and its contender record carries a reason containing 'marker_check_raised'. (b) a launched loser whose terminate() raises: assert race_launch still returns status=='launched' (winner present) and the loser's contender record['terminated'] reflects the terminate_failed detail (status=='terminate_failed'). Use injected _NO_SLEEP and a small poll_interval; make both deterministic.

Constraints: fully hermetic; no GPU; no network; injected clocks. render outputs are simulator support NOT policy-success claims. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: python -m pytest tests/test_provider_race.py -k 'marker_raises or terminate_raises' (or your test names) and the full file; python -m py_compile on the test file.
```

</details>

## Spend guard & pod lifecycle

### [P1-35] Add job-level allow_paid=True test for G1 parity job

- **Priority:** P1 · **Effort:** L · **Dimension:** Spend guard & pod lifecycle
- **Goal:** Cover the full run_isaac_g1_kitchen_parity_job(allow_paid=True) orchestration (launch->watch->teardown->result parsing) end-to-end with a fake provider, no GPU.
- **Files:** `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `tests/test_isaac_g1_kitchen_parity_job.py`
- **Validate (CPU):** python3 -m pytest tests/test_isaac_g1_kitchen_parity_job.py -k 'allow_paid' -q (new tests, both happy and flaky paths) ; then full file python3 -m pytest tests/test_isaac_g1_kitchen_parity_job.py -q and python3 -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py

- **Context:** run_isaac_g1_kitchen_parity_job is the orchestration for the active 'open the refrigerator' G1 POV seed render lane — it is where stage/launch/watch/teardown/result-parsing are stitched together, and historically where most lifecycle bugs landed. Without a job-level fake-provider test, a regression in teardown selection, warm_only plumbing, or parity-result parsing ships silently. Anchors: provider availability gate src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py lines 589-592; launch wiring lines 599-602; watch_and_collect(preserve_instance=True) lines 625-627; result parse lines 636-640; harness build lines 642-645. Existing fakes/fixtures: _make_fake_provider (tests/test_isaac_g1_kitchen_parity_job.py lines 333-372), _SCENARIOS (lines 10-15), _fake_stage (lines 307-311).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add hermetic job-level tests for the full allow_paid orchestration in src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py run_isaac_g1_kitchen_parity_job (lines 465-648). Today tests/test_isaac_g1_kitchen_parity_job.py only covers build_request/bundle/launch_spec, the no-spend prepared plan (test_job_prepared_plan_without_spend), the no-scenarios/unknown-provider blocks, and launch_with_marker_retry in isolation. The stitched-together allow_paid path — provider availability gate (lines 589-592), launch_with_marker_retry wiring with allow_cold_fallback=not warm_only (lines 599-602), watch_and_collect(preserve_instance=True) (lines 625-627), parity_result parsing from isaac_g1_kitchen_parity_result.json (lines 636-640), and harness assembly (lines 642-645) — is never exercised against a fake provider.

Write tests that:
1. Monkeypatch J.stage_bundle to a stub that creates job_dir and writes provider_bundle_url.txt, provider_output_put_url.txt AND provider_output_get_url.txt (the watch/marker loops read the GET url). Reuse the _fake_stage shape from test_job_prepared_plan_without_spend (lines 305-319) but also write the get url.
2. Inject a fake provider by monkeypatching J.get_render_provider to return a fake exposing build_request/available/launch/stop/terminate (model it on _make_fake_provider, lines 333-372). Monkeypatch J.time.sleep/time and J.urllib.request.urlopen (and watch_and_collect's urlopen) so no real network/clock is used.
3. Happy path: fake provider launches, emits the early marker, and the output zip contains bootstrap.json phase 'runner_done' plus isaac_g1_kitchen_parity_result.json with status 'completed'; assert manifest status 'completed', a harness package is built, and teardown used stop() (preserve_instance=True path), NOT terminate.
4. Launch-failed-all-flaky path: fake never emits a marker; assert manifest blockers contains 'launch_failed_all_attempts_flaky' and status 'blocked'.

Constraints: keep world-model backends swappable (drive everything through the provider abstraction and get_render_provider; do not hit RunPod/Vast APIs); protect provenance/rights/privacy/raw-capture-truth; the harness 'evaluates video_rollout_fidelity_not_task_success' boundary must stay intact — render/harness outputs are simulator support, NOT policy-success claims (assert the claim_boundary / wam_evaluator semantics are not weakened); secrets file-based, never logged. This is added test coverage only — change product code only if a test reveals a real defect, and if so keep the fix minimal and provider-neutral. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done run: `python3 -m pytest tests/test_isaac_g1_kitchen_parity_job.py -q` and `python3 -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`.
```

</details>

### [P1-36] Test warm_only blocks instead of spending on a cold pod

- **Priority:** P1 · **Effort:** S · **Dimension:** Spend guard & pod lifecycle
- **Goal:** Prove warm_only=True never falls back to a cold create — at both the launch_with_marker_retry layer and the job wrapper.
- **Files:** `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `tests/test_isaac_g1_kitchen_parity_job.py`
- **Validate (CPU):** python3 -m pytest tests/test_isaac_g1_kitchen_parity_job.py -k 'warm_only' -q (new tests at both layers) ; then python3 -m pytest tests/test_isaac_g1_kitchen_parity_job.py tests/test_gpu_render_providers.py -q and python3 -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py src/blueprint_pipeline/gpu_render_providers.py

- **Context:** warm_only is the explicit 'never spend on a cold pod' switch for the spend-paused period that protects the warm-restart-first economics of the active G1 refrigerator render lane. If marker-retry or the job wrapper accidentally falls back to cold create when warm fails, the user's guarantee breaks and money is spent. Anchors: launch_with_marker_retry src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py lines 377-419 (cold-fallback wiring line 602); provider warm-only block src/blueprint_pipeline/gpu_render_providers.py lines 209-214; existing provider test tests/test_gpu_render_providers.py lines 134-162.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add hermetic tests proving the warm_only 'never spend on a cold pod' safety switch holds below the CLI-forwarding layer. warm_only flows as allow_cold_fallback=not warm_only into launch_with_marker_retry (src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py line 602) and then into prov.launch (src/blueprint_pipeline/gpu_render_providers.py lines 209-214), where RunPod returns blocked ['warm_restart_failed_cold_fallback_disabled']. tests/test_gpu_render_providers.py::test_runpod_warm_only_blocks_without_cold_create covers the provider and tests/test_isaac_g1_kitchen_parity_job.py::test_cli_forwards_warm_only covers arg forwarding, but nothing tests that (a) launch_with_marker_retry surfaces a warm-restart-failed-cold-disabled blocked launch WITHOUT entering the marker-poll loop and without any cold create, nor (b) run_isaac_g1_kitchen_parity_job(warm_only=True) returns a blocked manifest (not a cold spend) when warm candidates are all stale.

Write tests that:
1. launch_with_marker_retry layer: build a fake provider whose launch(...) returns {'status':'blocked','blockers':['warm_restart_failed_cold_fallback_disabled']} when allow_cold_fallback=False. Assert launch_with_marker_retry never enters the marker-poll loop (no urlopen calls), records the launch_call_failed attempt(s), and returns status 'blocked' across max_attempts. Reuse/extend _make_fake_provider (tests/test_isaac_g1_kitchen_parity_job.py lines 333-372) so its launch can be parameterized to honor allow_cold_fallback.
2. Job wrapper layer: with J.stage_bundle stubbed (write provider_*_url.txt incl. the get url), J.get_render_provider monkeypatched to that fake, and time/urlopen monkeypatched, call run_isaac_g1_kitchen_parity_job(..., allow_paid=True, warm_only=True) and assert the manifest is blocked with 'launch_failed_all_attempts_flaky' and that the fake's cold-create path was never invoked.

Constraints: keep world-model backends swappable (drive through the provider abstraction); protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; secrets file-based and never logged. Coverage-only — only touch product code if a test reveals a genuine warm_only leak, and keep any fix minimal and provider-neutral. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done run: `python3 -m pytest tests/test_isaac_g1_kitchen_parity_job.py -q` and `python3 -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py src/blueprint_pipeline/gpu_render_providers.py`.
```

</details>

### [P1-37] Test marker-retry stops every flaky warm pod with no leak

- **Priority:** P1 · **Effort:** S · **Dimension:** Spend guard & pod lifecycle
- **Goal:** Assert that across multiple warm-restart attempts each flaky warm pod is stop()'d exactly once by the right id and none is left running.
- **Files:** `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `tests/test_isaac_g1_kitchen_parity_job.py`
- **Validate (CPU):** python3 -m pytest tests/test_isaac_g1_kitchen_parity_job.py -k 'marker_retry' -q (extended warm-exhaustion + cold-exhaustion assertions) ; then python3 -m pytest tests/test_isaac_g1_kitchen_parity_job.py -q and python3 -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py

- **Context:** Warm pods that launch but never heartbeat are a subtle billing leak: if any attempt forgets to stop the pod, it keeps running. The current single-attempt test cannot catch a regression where one flaky warm pod across N attempts is left running. This guards the warm-restart economics of the active 'open the refrigerator' G1 render lane. Anchors: stop/terminate selection at src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py lines 415-418; existing tests tests/test_isaac_g1_kitchen_parity_job.py lines 389-427; fake provider lines 333-372.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), strengthen the marker-retry warm-exhaustion test for src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py launch_with_marker_retry (lines 377-419). It calls prov.stop(iid) for flaky warm restarts and prov.terminate(iid) for flaky cold pods (lines 415-418). The current test test_launch_with_marker_retry_stops_flaky_warm_restart (tests/test_isaac_g1_kitchen_parity_job.py lines 405-427) only runs max_attempts=1 and asserts terminated==[]; it does not exercise multiple warm attempts nor assert each flaky warm pod was stopped exactly once by the correct id.

Write/extend tests so that:
1. The fake provider records (mode, stop ids, terminate ids) and returns mode 'warm_restart' when cold=False. Run launch_with_marker_retry(..., cold=False, max_attempts=3, marker_timeout=2, poll=1) with a provider that launches but never heartbeats (marker=False). Assert fp.stopped == ['pod0','pod1','pod2'] (each warm pod stopped exactly once) and fp.terminated == [] (no warm pod deleted, none left running/billing), and the result is blocked with 'all_launch_attempts_flaky'.
2. Keep/confirm the cold counterpart: with cold=True the flaky pods are terminated (fp.terminated == ['pod0','pod1','pod2'], fp.stopped == []) — extend test_launch_with_marker_retry_terminates_all_flaky_pods (lines 389-402) if needed so the warm vs cold teardown asymmetry is pinned side by side.
Use the monkeypatched clock pattern already in those tests (J.time.time/J.time.sleep, J.urllib.request.urlopen = fp.urlopen). Extend _make_fake_provider (lines 333-372) to record stop ids and to set its launch 'mode' from the cold flag.

Constraints: keep world-model backends swappable (provider-neutral); protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; secrets file-based and never logged. Coverage-only — change product code only if a test reveals a real leak (e.g. a warm pod left un-stopped), and keep any fix minimal. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done run: `python3 -m pytest tests/test_isaac_g1_kitchen_parity_job.py -q` and `python3 -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`.
```

</details>

### [P1-38] Model mid-render node 404 teardown as already-gone

- **Priority:** P1 · **Effort:** M · **Dimension:** Spend guard & pod lifecycle
- **Goal:** Treat a 404 on stop()/terminate() (instance already vanished) as effectively-cleaned-up, and pin watch/marker-retry behavior when every poll 404s.
- **Files:** `src/blueprint_pipeline/gpu_render_providers.py`, `src/blueprint_pipeline/isaac_particlefield_render_job.py`, `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `tests/test_gpu_render_providers.py`
- **Validate (CPU):** python3 -m pytest tests/test_gpu_render_providers.py -k 'node_404 or already_gone or terminate or stop' -q (new 404 classification + watch-all-404 teardown tests) ; then python3 -m pytest tests/test_gpu_render_providers.py -q and python3 -m py_compile src/blueprint_pipeline/gpu_render_providers.py src/blueprint_pipeline/isaac_particlefield_render_job.py

- **Context:** Mid-render node death is a named real bug ('mid-render node 404 death'). Currently it surfaces as teardown_failed noise that could be misread as a leaked pod, while a transient 404 during marker wait shouldn't be conflated with a flaky-pod kill. Pinning the intended 'already gone == clean' semantics removes a class of false spend-alarm noise on the active G1 refrigerator render lane. Anchors: RunPod stop/terminate src/blueprint_pipeline/gpu_render_providers.py lines 225-239; Vast stop lines 340-352; caught GET-raises in the watch/marker loops at isaac_g1_kitchen_parity_job.py lines 408-409 and isaac_particlefield_render_job.py lines 319-320.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), model and pin the mid-render node-404 (instance vanishes) lifecycle case. When a node dies mid-flight, the output GET raises and is caught (src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py lines 408-409 and src/blueprint_pipeline/isaac_particlefield_render_job.py lines 319-320), but the eventual stop()/terminate() then hits a 404. Today RunPodRenderProvider.stop/terminate map any non-2xx to 'stop_failed'/'terminate_failed' with the http code (src/blueprint_pipeline/gpu_render_providers.py lines 225-239) and VastRenderProvider.stop likewise (lines 340-352) — a 404 'already gone' (which is actually success: nothing left billing) is conflated with a real teardown failure.

Required work:
1. In src/blueprint_pipeline/gpu_render_providers.py, classify a 404 on stop()/terminate() as effectively-gone success (e.g. status 'terminated'/'stopped' with a flag like already_gone=True, or a distinct 'already_gone' status) rather than 'stop_failed'/'terminate_failed', for BOTH RunPod and Vast. Keep other non-2xx codes as failures. Do not change the 2xx happy paths.
2. Add hermetic tests in tests/test_gpu_render_providers.py: monkeypatch the provider's HTTP call (_runpod_call for RunPod; the vast _api_json import for Vast) to return 404 on /stop and DELETE; assert the new effectively-gone classification. Keep fail-closed-without-key behavior intact (test_runpod_terminate_is_delete_and_fail_closed).
3. Add a test that drives watch_and_collect with urlopen always raising (node gone for the whole window): assert it still tears down exactly once and returns blocked, and that a 404 teardown is reported as already-gone, not a blocker that looks like a leaked pod.

Constraints: keep world-model backends swappable (apply the 404 semantics uniformly across providers, do not special-case one); protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; secrets file-based and never logged; the 404 reclassification must NOT cause a still-running pod to be reported as gone — only an actual 404 from the teardown call. Add/extend tests. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done run: `python3 -m pytest tests/test_gpu_render_providers.py -q` and `python3 -m py_compile src/blueprint_pipeline/gpu_render_providers.py src/blueprint_pipeline/isaac_particlefield_render_job.py src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`.
```

</details>

### [P1-39] Test Vast teardown contract and stop==destroy hazard

- **Priority:** P1 · **Effort:** M · **Dimension:** Spend guard & pod lifecycle
- **Goal:** Pin that Vast launch writes started_vast_instance_id.txt and that VastRenderProvider stop()/terminate() both DELETE (destroy), documenting that warm-preserve must never be used with Vast.
- **Files:** `src/blueprint_pipeline/gpu_render_providers.py`, `tests/test_gpu_render_providers.py`
- **Validate (CPU):** python3 -m pytest tests/test_gpu_render_providers.py -k 'vast' -q (new launch-writes-id + terminate-delegates-to-DELETE tests) ; then python3 -m pytest tests/test_gpu_render_providers.py -q and python3 -m py_compile src/blueprint_pipeline/gpu_render_providers.py

- **Context:** The stop-vs-terminate teardown contract is provider-specific and only RunPod is tested. On Vast, 'stop to preserve' is a no-op-to-destroy mismatch; if warm-reuse logic is later pointed at Vast it will quietly lose instances. The started_vast_instance_id.txt ownership write the spend guard depends on (scripts/gpu_spend_guard.py OWNER_ID_FILENAMES line 312, find_protected_pod_ids lines 363-394) is also untested against the live Vast launch path. Anchors: VastRenderProvider src/blueprint_pipeline/gpu_render_providers.py lines 244-352 (launch lines 279-338, stop lines 340-352), base terminate fallback lines 112-115; ownership-file write line 334.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add hermetic tests for the Vast provider lifecycle, which is currently untested where RunPod is tested. VastRenderProvider (src/blueprint_pipeline/gpu_render_providers.py lines 244-352) defines launch and stop (stop = DELETE /instances/{id}/) but inherits terminate() from the base class, which falls back to stop() (lines 112-115). For Vast that DELETE destroys the instance — so the watch_and_collect stop-vs-terminate 'preserve for warm reuse' distinction collapses: a stop() to preserve actually DESTROYS the Vast instance.

Write tests in tests/test_gpu_render_providers.py that:
1. Monkeypatch the vast_provider_adapter helpers imported inside VastRenderProvider.launch (_api_json, _offers_from_response, _select_offer) to canned offers + a create response containing an instance id. Assert launch writes started_vast_instance_id.txt into job_dir (the file the spend guard's ownership scan relies on — see find_protected_pod_ids / started_vast_instance_id.txt in scripts/gpu_spend_guard.py) and returns status 'launched' with mode 'vast_on_demand'.
2. Assert VastRenderProvider().terminate(iid) delegates to the same DELETE /instances/{iid}/ path that stop() uses (monkeypatch _api_json to record method+path), proving terminate==stop==destroy for Vast.
3. Add a documentation-style test or an explicit code comment in gpu_render_providers.py near VastRenderProvider making the 'stop()==destroy for Vast; do NOT use warm-preserve/stop() to keep a Vast instance' contract explicit, so future warm-reuse logic is not silently pointed at Vast.
Keep fail-closed-without-key behavior intact (test_vast_launch_fail_closed_without_key, test_vast_stop_fail_closed_without_key).

Constraints: keep world-model backends swappable (the stop/terminate contract is provider-specific by design; make the Vast semantics explicit rather than papering over them); protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; secrets file-based and never logged. Add tests; only touch product code for the clarifying comment unless a real defect surfaces. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done run: `python3 -m pytest tests/test_gpu_render_providers.py -q` and `python3 -m py_compile src/blueprint_pipeline/gpu_render_providers.py`.
```

</details>

### [P1-40] Protect not-yet-recorded pods during launch/reap race

- **Priority:** P1 · **Effort:** M · **Dimension:** Spend guard & pod lifecycle
- **Goal:** Close the spend-guard reap race between API pod-create and the started_pod_id.txt write so a concurrent --reap cannot delete a pod the orchestrator just paid to create.
- **Files:** `scripts/gpu_spend_guard.py`, `src/blueprint_pipeline/gpu_render_providers.py`, `tests/test_gpu_spend_guard.py`
- **Validate (CPU):** python3 -m pytest tests/test_gpu_spend_guard.py -k 'race or grace or prefix' -q (new race/grace test) ; then full file python3 -m pytest tests/test_gpu_spend_guard.py -q and python3 -m py_compile scripts/gpu_spend_guard.py

- **Context:** A --reap invocation racing an active launch could delete a pod the orchestrator just paid to create but hasn't recorded yet — the guard reaping live work, the exact failure mode the safety rails exist to prevent, on the active G1 refrigerator render lane. Anchors: find_protected_pod_ids scripts/gpu_spend_guard.py lines 363-394; is_reapable lines 422-438; id-file writes src/blueprint_pipeline/gpu_render_providers.py lines 206, 220, 334; pod name from RenderLaunchSpec.name (src/blueprint_pipeline/gpu_render_providers.py line 58) and build_render_launch_spec name='blueprint-isaac-splat-render' (src/blueprint_pipeline/isaac_particlefield_render_job.py line 244).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), address the spend-guard ownership-scan race in scripts/gpu_spend_guard.py find_protected_pod_ids (lines 363-394). It protects a pod only if an owner-id file exists AND a live process references the run. But the render job creates the pod (prov.launch) and writes started_pod_id.txt only AFTER a successful start (src/blueprint_pipeline/gpu_render_providers.py lines 206, 220; Vast line 334). Between the API create returning an id and the file being written, a concurrent `--reap` run can see a live, not-yet-booted pod with NO owner file and reap it. The ownership signal is also purely process-cmdline based, so a run whose launching shell already exited (but whose pod is still booting) is unprotected.

Required work:
1. Add a hermetic test that pins the CURRENT (unsafe) behavior: a pod present in the API JSON (live, not booted, past boot threshold) with a live owning process cmdline but NO started_pod_id.txt yet is currently reapable. Use the _make_started_pod_id_file / patched_guard patterns (tests/test_gpu_spend_guard.py lines 186-341).
2. Implement and test a safer behavior: add a grace so a not-yet-booted pod whose NAME matches the blueprint render prefix (the launch spec names pods 'blueprint-isaac-splat-render' / 'blueprint-isaac-...'; see RenderLaunchSpec.name and build_launch_spec) is protected while an owning render process is live, even before its id file lands — OR honor a younger min-age grace window so a just-created pod cannot be reaped before the file-write can complete. Pick the approach that is safest (errs toward keep) and keeps the existing reap tests green (the genuine unowned dud past threshold must still be reapable; healthy and owned pods still kept).

Constraints: keep world-model backends swappable (provider-neutral ownership logic; rely on pod-name prefix / id-file conventions, not provider APIs); protect provenance/rights/privacy/raw-capture-truth — erring toward NOT reaping live work; secrets file-based and never logged; reap decisions are cost-control, NOT policy-success claims. Add/extend tests in tests/test_gpu_spend_guard.py. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done run: `python3 -m pytest tests/test_gpu_spend_guard.py -q` and `python3 -m py_compile scripts/gpu_spend_guard.py`.
```

</details>

### [P1-41] Add teardown/test safety net for warm serve not-ready timeout

- **Priority:** P1 · **Effort:** M · **Dimension:** Spend guard & pod lifecycle
- **Goal:** Ensure run_isaac_g1_kitchen_parity_job serve mode does not silently leave a paid pod running on a ready-timeout, and cover the serve branch with a fake-provider test.
- **Files:** `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `tests/test_isaac_g1_kitchen_parity_job.py`
- **Validate (CPU):** python3 -m pytest tests/test_isaac_g1_kitchen_parity_job.py -k 'serve' -q (new ready + not-ready serve-branch tests) ; then python3 -m pytest tests/test_isaac_g1_kitchen_parity_job.py -q and python3 -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py

- **Context:** Serve mode is the warm-pool entry point and the highest standing-cost path for the active G1 refrigerator render lane. A not-ready timeout that silently leaves a paid pod running with no teardown is a direct spend leak, and it is completely untested. Anchors: serve branch src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py lines 607-624; _await_warm_serve_ready lines 424-462; inbox presign lines 547-555; existing no-spend test pattern test_job_prepared_plan_without_spend (tests/test_isaac_g1_kitchen_parity_job.py lines 305-319) and _make_fake_provider lines 333-372.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), close a spend leak and add the missing test for warm serve mode in src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py run_isaac_g1_kitchen_parity_job (serve branch lines 607-624) and _await_warm_serve_ready (lines 424-462). In serve mode the job launches ONE pod and intentionally returns with it RUNNING (status 'serving', or blocker 'warm_serve_not_ready'), relying on the caller's WarmPoolClient to tear it down. But on a ready timeout, _await_warm_serve_ready returns ready=False and the job returns WITHOUT tearing the pod down — the pod keeps billing. There is currently NO test of the serve branch at all (nothing references serve=True or _await_warm_serve_ready).

Required work:
1. Add hermetic tests for the serve branch: monkeypatch J.stage_bundle (write provider_*_url.txt incl. the output get url), monkeypatch the presign_warm_inbox_channel import used in the serve path (lines 547-555) to a stub returning a completed inbox with a warm_inbox_get_url_file, inject a fake provider via J.get_render_provider, and monkeypatch time/urlopen. Cover BOTH: (a) ready path — the output zip yields warm_serve_ready.json -> assert manifest status 'serving' and that the instance is left running (no teardown call); and (b) not-ready path — readiness never appears within serve_ready_timeout -> assert current behavior.
2. Decide and implement the safe not-ready contract: either tear the pod down on a not-ready timeout (terminate, since it never proved healthy), OR, if leaving it running is intentional for caller pickup, ensure the manifest CLEARLY records the still-running instance_id under a stable key so the caller/spend-guard can reclaim it (it already sets warm_serve.instance_id at lines 613-619 — make this unambiguous and assert it). Prefer terminate-on-not-ready unless a clear caller-handoff contract exists; document the choice in a code comment. Whichever you choose, the test must prove no pod is silently orphaned with no record.

Constraints: keep world-model backends swappable (provider-neutral teardown); protect provenance/rights/privacy/raw-capture-truth; the serve harness is simulator/runtime support, NOT a policy-success claim; secrets file-based and never logged. Add/extend tests. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done run: `python3 -m pytest tests/test_isaac_g1_kitchen_parity_job.py -q` and `python3 -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`.
```

</details>

## Dev env & deps

### [P1-42] Reconcile the dev extra so trimesh is actually installed by pip install -e .[dev]

- **Priority:** P1 · **Effort:** S · **Dimension:** Dev env & deps
- **Goal:** Fix the inconsistency where the dev extra declares trimesh but the assembled .venv lacks it, unblocking geometry/scenario-packet tests.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/pyproject.toml`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_robot_eval_job_orchestrator.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_mujoco_scene_scenario_packet.py`, `$HOME/workspace/BlueprintCapturePipeline/uv.lock`
- **Validate (CPU):** `.venv/bin/python -c 'import trimesh; print(trimesh.__version__)'` works; `.venv/bin/python -m pytest tests/test_mujoco_scene_scenario_packet.py -q` runs (does not skip for missing trimesh). Throwaway `pip install -e .[dev]` venv imports trimesh, proving the documented command is sufficient. CPU-only.

- **Context:** The mismatch between the declared dev extra (includes trimesh) and the actual .venv (lacks it) is direct evidence of the reproducibility gap being audited. Files: $HOME/workspace/BlueprintCapturePipeline/pyproject.toml, $HOME/workspace/BlueprintCapturePipeline/tests/test_robot_eval_job_orchestrator.py, $HOME/workspace/BlueprintCapturePipeline/tests/test_mujoco_scene_scenario_packet.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), trimesh is imported by tests/test_robot_eval_job_orchestrator.py and tests/test_mujoco_scene_scenario_packet.py (1 importorskip('trimesh') site) and IS declared in both the `runtime` and `dev` extras of pyproject.toml (lines ~45 and ~71). But .venv MISSES it (`import trimesh` -> ModuleNotFoundError), proving the venv was hand-assembled rather than reproduced from pyproject (it has cv2+PIL but not trimesh).

Do this:
1. Install trimesh into .venv: `.venv/bin/pip install 'trimesh>=4.4.0' 'pycollada>=0.8'` (CPU; do not rebuild).
2. Verify a clean reproduction: create a THROWAWAY venv elsewhere (e.g. /tmp/repro-venv) with `python3.12 -m venv` then `pip install -e .[dev]` against this repo, and assert `import trimesh` succeeds in it. This proves the documented dev command is self-sufficient. Delete the throwaway venv afterward.
3. If the throwaway repro reveals the dev extra is missing anything trimesh needs, add it to the dev extra and regenerate uv.lock.

Constraints: Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only. Keep world-model backends swappable. Protect provenance/rights/privacy/raw-capture-truth. Render outputs are simulator support, NOT policy-success claims. Run `python -m py_compile` on any edited Python and `python -m pytest tests/test_mujoco_scene_scenario_packet.py`.
```

</details>

### [P1-43] Write one documented reproducible CPU dev-setup (DEV_SETUP.md + README)

- **Priority:** P1 · **Effort:** M · **Dimension:** Dev env & deps
- **Goal:** Replace the two conflicting, incomplete setup paths with one canonical Python 3.12 command that installs the full no-GPU dep set and a one-line import probe to verify it.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/README.md`, `$HOME/workspace/BlueprintCapturePipeline/pyproject.toml`, `$HOME/workspace/BlueprintCapturePipeline/.python-version`, `$HOME/workspace/BlueprintCapturePipeline/uv.lock`
- **Validate (CPU):** From a clean checkout, run ONLY the documented command, then the probe `python -c 'import pxr, PIL, mujoco, trimesh, boto3, blueprint_pipeline, blueprint_contracts; print("full CPU env ok")'` succeeds; `python -m pytest -q` shows 0 skips attributable to missing pxr/mujoco/trimesh (compare skip count before/after). No GPU, no cloud.

- **Context:** The user explicitly wants ONE documented setup that makes pytest, --dry-render, and usd-core replay fully runnable locally. The current docs mislead by implying `[dev]` is enough. This task depends on the usd-core/mujoco/boto3 extras existing first. Files: $HOME/workspace/BlueprintCapturePipeline/README.md, $HOME/workspace/BlueprintCapturePipeline/pyproject.toml, $HOME/workspace/BlueprintCapturePipeline/.python-version, $HOME/workspace/BlueprintCapturePipeline/uv.lock.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), the README offers TWO conflicting setup paths — `uv sync --extra dev` (line ~399) and `python3 -m venv .venv && pip install -e .[dev]` (lines ~776-778) — and NEITHER installs pxr/usd-core, mujoco, or boto3, so following the docs literally reproduces the split-brain (.venv with no pxr). Also .python-version pins 3.12 while pyproject requires-python is >=3.10, and there are orphan 3.13 venvs.

Do this (AFTER the usd-core, mujoco, and boto3-in-dev extras land):
1. Add a docs/DEV_SETUP.md with ONE canonical block: use Python 3.12, then a single command — `uv sync --extra dev --extra usd --extra sim --extra cloud` (or the equivalent `pip install -e '.[dev,usd,sim,cloud]'`) — followed by a verification probe.
2. Update the README 'Local Development' section to point at DEV_SETUP.md and remove/deprecate the second conflicting path, OR make both paths install the same full set.
3. State explicitly that Gemini/LLM extras stay optional and mock-only (no live keys needed for CPU validation).
4. Note that .python-version pins 3.12 and that contributors must use 3.12 for the canonical env.

Constraints: Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only. No live LLM/cloud calls. Keep world-model backends swappable (do not document one backend as mandatory). Protect provenance/rights/privacy/raw-capture-truth. Render outputs are simulator support, NOT policy-success claims. Run `python -m py_compile` on any helper scripts you add.
```

</details>

### [P1-44] Add a no-GPU env doctor that asserts the full CPU dependency set

- **Priority:** P1 · **Effort:** M · **Dimension:** Dev env & deps
- **Goal:** Add a fail-fast CPU env doctor (console entry + test) that imports the full no-GPU stack and reports exactly which modules are missing from the current interpreter.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/cpu_simulator_preflight.py`, `$HOME/workspace/BlueprintCapturePipeline/pyproject.toml`, `$HOME/workspace/BlueprintCapturePipeline/tests/conftest.py`
- **Validate (CPU):** Run the doctor under .venv: before the usd-core/mujoco/trimesh extras land it FAILS listing pxr+mujoco+trimesh missing; after they land it PASSES. The new pytest (`.venv/bin/python -m pytest tests/test_cpu_env_doctor.py -q` or wherever placed) asserts the return shape and passes. CPU-only, no cloud.

- **Context:** A fail-fast doctor turns a confusing pre-pod failure into an actionable 'pxr missing in this interpreter' message and gives the paused-GPU effort a green/red gate for 'is my CPU env complete?' before anything is attempted — directly de-risking the 'open the refrigerator' G1 lane launch. Files: $HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/cpu_simulator_preflight.py, $HOME/workspace/BlueprintCapturePipeline/pyproject.toml, $HOME/workspace/BlueprintCapturePipeline/tests/conftest.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), the current dependency failure modes are silent (importorskip -> SKIP) or late (staging_failed mid-job). Add a tiny env doctor that codifies the manual probing from the dependency audit.

Do this:
1. Add a small function/module (e.g. extend src/blueprint_pipeline/cpu_simulator_preflight.py or add src/blueprint_pipeline/cpu_env_doctor.py) that attempts to import the full no-GPU stack — pxr, PIL, mujoco, trimesh, boto3, numpy, yaml, jsonschema, blueprint_pipeline, blueprint_contracts — and returns a structured result listing which are present/missing, plus sys.executable and sys.version. It must NOT raise on missing modules; it reports them.
2. Register a console entry point in pyproject.toml (e.g. `blueprint-check-cpu-env`).
3. Add a pytest in tests/ that imports the doctor and asserts its return shape (a dict/dataclass with the module->bool map, sys.executable, sys.version). The test must pass regardless of which optional modules happen to be installed (assert shape, not specific presence).

Constraints: Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only. Keep world-model backends swappable (the doctor checks imports, it does not pick a backend). Protect provenance/rights/privacy/raw-capture-truth. Render outputs are simulator support, NOT policy-success claims. Run `python -m py_compile` on the new/edited files and `python -m pytest` on the new test.
```

</details>

## Docs / provenance / claims

### [P1-45] Soften the User-Facing warm-render-server CHANGELOG claim to hermetic-only

- **Priority:** P1 · **Effort:** S · **Dimension:** Docs / provenance / claims
- **Goal:** Reword the User-Facing warm-render bullet so it no longer reads as proven live multi-job reuse, matching the module's hermetic-only validation.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/docs/CHANGELOG.md`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/warm_render_server.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_warm_render_server.py`
- **Validate (CPU):** Re-read docs/CHANGELOG.md lines ~18-24 against src/blueprint_pipeline/warm_render_server.py lines 1-8 and confirm the User-Facing line no longer asserts proven live multi-job reuse. Run `python3 -m pytest tests/test_warm_render_server.py -q` and confirm it passes and contains only hermetic (no live-GPU) coverage. CPU only.

- **Context:** Cloud is paused, so live multi-job reuse cannot be proven now — the claim must be scoped down rather than left implying live success. The unqualified phrasing at docs/CHANGELOG.md:20-21 contradicts the module docstring at src/blueprint_pipeline/warm_render_server.py:1-8 ('guarded on-GPU validation', 'hermetically testable', imports no isaacsim/pxr). This is part of the warm --serve lane that supports the active G1 kitchen-parity render flow.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), docs/CHANGELOG.md has a User-Facing bullet (lines ~20-21) stating the change adds 'a persistent warm render server that can serve multiple task requests after one Isaac scene load.' This is unqualified and reads as a proven live capability. But the module's own docstring at src/blueprint_pipeline/warm_render_server.py (lines 1-8) is explicit that the Isaac setup/render are INJECTED, the module imports NO isaacsim and NO pxr, the control flow is only 'hermetically testable', and the on-GPU path is the '(guarded) on-GPU validation' — i.e. multi-job reuse AFTER a real Isaac scene load has never been demonstrated live. tests/test_warm_render_server.py covers only the hermetic loop (no live-GPU test).

Task: Reword the User-Facing bullet so it says the warm-serve loop is IMPLEMENTED and HERMETICALLY TESTED, with live multi-job reuse after a real Isaac scene load STILL PENDING on-GPU proof. Keep it concise and consistent in tone with the surrounding User-Facing bullets. Do not delete the Future-Agent-Facing caveat (it is already correct) — just bring the User-Facing line into agreement with it and with the module docstring's 'guarded on-GPU validation' / 'hermetically testable' language.

Constraints: Documentation-only. PLATFORM_CONTEXT and the autonomous-loop checklist forbid presenting scaffolding/adapter success as proven runtime capability. Keep world-model backends swappable; protect provenance/rights/privacy/raw-capture truth; render outputs are simulator support, not policy/runtime success. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-46] Document the Isaac G1 kitchen-parity render lane in README.md

- **Priority:** P1 · **Effort:** M · **Dimension:** Docs / provenance / claims
- **Goal:** Add a README subsection naming scripts/run_isaac_g1_kitchen_parity_eval.py and the 'open the refrigerator' head-POV seed lane, with an explicit claim boundary.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/README.md`, `$HOME/workspace/BlueprintCapturePipeline/scripts/run_isaac_g1_kitchen_parity_eval.py`
- **Validate (CPU):** Run `grep -n -i 'run_isaac_g1_kitchen_parity\|head.POV\|render seed\|--dry-render' README.md` and confirm non-empty after the edit. Re-read the new subsection against scripts/run_isaac_g1_kitchen_parity_eval.py:799-815 to confirm the boundary wording is consistent (support not success). CPU only (doc edit).

- **Context:** README + AGENTS.md are the designated 'read first' product surface and where claim boundaries for lanes are stated. Leaving the entire active G1 render lane undocumented means its boundaries live only in code docstrings and an uncommitted changelog — an overstatement/understatement risk and a violation of the 'render seeds are support not success' rule. Verified: README.md has 0 mentions of the script; proof-boundary docstrings are at scripts/run_isaac_g1_kitchen_parity_eval.py:799-815.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), README.md (1984 lines) has ZERO mentions of scripts/run_isaac_g1_kitchen_parity_eval.py — confirm with `grep -c 'run_isaac_g1_kitchen_parity' README.md` (returns 0). This script is one of the most heavily edited files in recent history and owns the active G1 kitchen-parity render lane (dynamic placement, manipulation/head-POV seeds including the 'open the refrigerator' POV, warm --serve, and the local --dry-render preview). README.md has top-level sections including '## Entry Points' (line ~793) and '## Contract Boundary' (line ~1973).

Task: Add a concise README subsection (a good home is under '## Entry Points' or adjacent to it) that: (1) names the script scripts/run_isaac_g1_kitchen_parity_eval.py and states it runs a MuJoCo-parity G1 eval that renders review media; (2) describes the head-POV 'open the refrigerator' seed lane in one or two sentences; (3) states the explicit claim boundary that the render seed is simulator/render SUPPORT (review media), NOT policy success, manipulation/object contact, physical reach, safety, learned-policy success, deployment, or live robot readiness. Word the boundary to match the existing docstring proof-boundary at scripts/run_isaac_g1_kitchen_parity_eval.py lines ~799-815 and ~4087/~5905, and the CHANGELOG render-seed sentence. Mention the no-GPU `--dry-render` flag as the local CPU preview path.

Constraints: Documentation-only. Keep world-model backends swappable; protect provenance/rights/privacy/raw-capture truth; do not overstate readiness. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-47] Document new modules in README and link the scene_placement README

- **Priority:** P1 · **Effort:** M · **Dimension:** Docs / provenance / claims
- **Goal:** Add concise README entries (with claim boundaries) for scene_placement, warm_render_server, provider_race, render_lock, gpu_spend_guard, and the local --dry-render tool, and link the package README.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/README.md`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/scene_placement/README.md`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/warm_render_server.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/provider_race.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/render_lock.py`, `$HOME/workspace/BlueprintCapturePipeline/scripts/gpu_spend_guard.py`
- **Validate (CPU):** Run `grep -n -i 'scene_placement\|provider_race\|warm_render\|render_lock\|spend_guard\|dry.render' README.md` and confirm non-empty after the edit. Confirm the relative link to src/blueprint_pipeline/scene_placement/README.md resolves (file exists). Confirm each module's claim wording matches its source docstring (e.g. warm_render_server.py header 'guarded on-GPU validation'; scene_placement/README.md 'no isaacsim, torch, google-genai, or GPU'). CPU only.

- **Context:** These are the major new features the user explicitly listed. Undocumented spend-guard/provider-race code can be misread as proven live-provider capability, and an unlinked scene_placement README means the swappability contract is not discoverable from the top-level docs. scene_placement powers the dynamic stand-pose placement in the active G1 'open the refrigerator' lane. Verified: all six files exist; README grep returns empty; scene_placement/README.md documents the no-GPU import contract.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), README.md and docs/ (outside the uncommitted CHANGELOG) contain ZERO mentions of scene_placement, provider_race, warm_render_server, render_lock, gpu_spend_guard, or the local --dry-render tool — confirm with `grep -in -E 'scene_placement|provider_race|warm_render|render_lock|spend_guard|dry.render' README.md` (currently empty). The well-written package doc at src/blueprint_pipeline/scene_placement/README.md is also unlinked from any top-level doc.

Task: Add a concise README section (e.g. under or adjacent to '## Contract Boundary' at line ~1973, or '## Entry Points' at ~793) documenting each module WITH its claim boundary, and add a relative link to src/blueprint_pipeline/scene_placement/README.md. Use boundaries grounded in each file's own docstring:
- scene_placement (src/blueprint_pipeline/scene_placement/): pure, dependency-light, swappable placement — importing the package pulls in NO isaacsim/torch/google-genai/GPU; the heavy bits (USD pxr, Gemini, PhysX probe, SAM3/DA3) are injected behind hooks. CPU-hermetic.
- warm_render_server (src/blueprint_pipeline/warm_render_server.py): warm --serve loop is implemented and hermetically tested; on-GPU multi-job reuse after a real Isaac scene load is still unproven ('guarded on-GPU validation').
- provider_race (src/blueprint_pipeline/provider_race.py): races GPU launches across providers + circuit breaker; spend-safety/orchestration scaffolding, NOT proven live-provider execution.
- render_lock (src/blueprint_pipeline/render_lock.py): render concurrency/spend-safety scaffolding, not runtime/execution proof.
- gpu_spend_guard (scripts/gpu_spend_guard.py): spend-guard scaffolding, not runtime/execution proof.
- --dry-render: no-GPU local CPU preview of the Isaac G1 stance/camera/arm framing; CPU-hermetic.
Make clear the spend-guard / provider-race / render-lock code is spend-safety scaffolding and must NOT be read as proven live-provider capability.

Constraints: Documentation-only. PLATFORM_CONTEXT/WORLD_MODEL_STRATEGY require provider/model backends stay swappable AND that readiness not be overstated. Keep world-model backends swappable; protect provenance/rights/privacy/raw-capture truth. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-48] Add READINESS_MATRIX rows for G1 render lane, scene_placement, warm-serve, provider/spend safety

- **Priority:** P1 · **Effort:** M · **Dimension:** Docs / provenance / claims
- **Goal:** Give the new lanes explicit rows in the strict readiness matrix using its own status and blocker-class vocabulary, so readiness is never implied by omission.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/docs/READINESS_MATRIX.md`, `$HOME/workspace/BlueprintCapturePipeline/README.md`
- **Validate (CPU):** Run `grep -n -i 'render\|g1\|isaac\|placement\|warm\|provider_race\|spend_guard' docs/READINESS_MATRIX.md` and confirm non-empty after the edit. Confirm each new row's Status is one of `ready`/`partial`/`blocked` and each blocker uses a defined class (`live-provider`/`hardware`/etc.) from the matrix header. Re-read the rows to confirm no CPU-only lane is overstated and no GPU-dependent lane is marked `ready`. CPU only.

- **Context:** The whole point of READINESS_MATRIX is to fail-closed about unproven external runtime. Brand-new lanes that touch live GPU spend with zero matrix rows let readiness be implied by omission — the opposite of the strict contract the README promises. With cloud paused, marking the live/GPU-dependent lanes `partial` with a `live-provider`/`hardware` blocker and the CPU-hermetic lanes `ready` is the honest state. Verified: matrix grep returns nothing; its Status Rules and Blocker Class Rules are defined in the header (lines ~5-30); existing rows like `qualification` use `ready` and `zero-shot Cosmos` uses `blocked`.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), README.md (~line 1984) designates docs/READINESS_MATRIX.md as the strict source of truth for 'what is shipped in-repo versus what still depends on live GPU/runtime/model access.' But `grep -in -E 'render|g1|isaac|placement|kitchen|parity|warm' docs/READINESS_MATRIX.md` currently returns NOTHING — none of the new lanes have a row. The matrix's vocabulary (read its header) is: Status = `ready` (implemented on the canonical contract with local in-repo evidence), `partial` (substantial impl but external runtime/deployment/production-faithfulness proof missing), `blocked` (no truthful shipped path). Blocker classes include `live-provider` (missing non-payment provider execution proof) and `hardware` (missing real-device/hardware proof). The matrix is a markdown table: `| Surface | Status | What is true in repo | Blocking gap |`.

Task: Add new rows to the matrix table for:
- `scene_placement` → `ready` — pure/swappable CPU-hermetic placement (no isaacsim/torch/GPU on import; heavy bits injected), covered by local tests (tests/test_scene_placement.py + perception tests). Blocking gap: none in-repo for the pure placement contract; live USD/Gemini/PhysX/SAM3/DA3 behind injected hooks.
- local `--dry-render` preview → `ready` — no-GPU CPU preview of G1 stance/camera/arm framing, covered by tests/test_local_render_preview.py. Blocking gap: none in-repo.
- Isaac G1 kitchen-parity live render seed → `partial`, blocker class `hardware` (or `live-provider` as appropriate) — control flow + dry-render exist and are CPU-tested, but no in-repo GPU execution proof of the live render seed. State clearly it is render/review SUPPORT not policy/manipulation success.
- warm render server multi-job reuse → `partial`, blocker class `live-provider`/`hardware` — hermetically tested loop; live reuse after a real Isaac scene load unproven.
- provider_race / gpu_spend_guard / render_lock live behavior → `partial`, blocker class `live-provider` — spend-safety/orchestration scaffolding with hermetic tests; no in-repo live-provider execution proof.
Use ONLY the matrix's defined Status values and blocker classes; keep each 'What is true in repo' factual and each 'Blocking gap' honest about the missing live/GPU proof.

Constraints: Documentation-only. The matrix is intentionally fail-closed; do not mark anything `ready` that depends on unproven external GPU/runtime. Keep world-model backends swappable; protect provenance/rights/privacy/raw-capture truth; render outputs are simulator support, not policy success. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## Launch gates & readiness

### [P1-49] Harden _build_readiness_decision so needs_more_evidence can block and human_review_required carries signal

- **Priority:** P1 · **Effort:** M · **Dimension:** Launch gates & readiness
- **Goal:** Prevent a metric-not-ready scene whose capability checks are all needs_more_evidence from passing through as 'ready', and make human_review_required derived or documented as an invariant.
- **Files:** `src/blueprint_pipeline/qualification.py`, `tests/test_qualification_coverage_edges.py`
- **Validate (CPU):** .venv/bin/python -m pytest tests/test_qualification_coverage_edges.py tests/test_qualification_alpha.py -q  &&  .venv/bin/python -m py_compile src/blueprint_pipeline/qualification.py

- **Context:** _build_readiness_decision is the core readiness verdict the launch gates and webapp sync consume (derive_webapp_qualification_state maps it to qualified_ready/qualified_risky). A scene with insufficient geometry evidence reaching 'ready' is a false-pass at the most load-bearing gate. Confirmed: human_review_required hardcoded True at line 3956; needs_more_evidence -> medium at line 3923; downgrade triggers at 3959-3961. Existing edge tests at tests/test_qualification_coverage_edges.py:624-639 cover only the high-blocker and hidden-zone paths. Use .venv/bin/python.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU/no-spend task. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

In src/blueprint_pipeline/qualification.py, _build_readiness_decision (lines ~3941-3983) only downgrades an already-'ready' record on: unresolved HIGH-severity blockers, uncertainty_score > maximum_uncertainty_score (line ~3959), or hidden_zone_bound for the 'risky' path (line ~3961). But _build_blocker_register (lines ~3917-3931) assigns severity 'medium' to every non-blocked capability check (status 'needs_more_evidence' -> 'medium', line ~3923). So a scene whose capability checks are all 'needs_more_evidence' contributes only medium blockers and can still pass as 'ready' if the upstream readiness_state was 'ready'. Separately, human_review_required is hardcoded True (line ~3956) and never derived from evidence, so the field carries no signal.

Decide and implement the intended policy (keep it explicit and minimal):
(a) a configurable count/severity of 'needs_more_evidence' capability checks (or any 'needs_more_evidence' check combined with metric_ready being False) forces status away from 'ready' (to 'risky' or 'not_ready_yet' per existing semantics); and/or gate 'ready' on metric_ready / absence of evidence_gaps. Read how 'ready' is currently produced upstream (qualification_record['readiness_state']) and how metric_ready is available to this function — thread it in if needed without changing call signatures used elsewhere unless you also update all callers.
(b) either compute human_review_required from evidence (e.g. True whenever there are any blockers/evidence_gaps or status != 'ready'), or, if it is intentionally always-on, leave it True but add a clear comment documenting it as an invariant. Pick one and make the code self-explanatory.

Do not regress the existing high-blocker downgrade (tests/test_qualification_coverage_edges.py:624-639) or hidden-zone risky path. Keep world-model backends swappable; protect provenance, rights, privacy, and raw-capture truth; readiness logic is secondary support, NOT a policy-success claim. Add/extend tests. This must satisfy $HOME/workspace/Blueprint-WebApp/docs/autonomous-loop-evidence-checklist-2026-05-03.md before claiming done.

Add a test to tests/test_qualification_coverage_edges.py: a qualification_record with readiness_state='ready', a blocker_register containing ONLY medium 'needs_more_evidence' entries, and metric_ready False -> assert the decided status is NOT 'ready'. Add a second assertion on the human_review_required behavior you chose.

Then run: .venv/bin/python -m pytest tests/test_qualification_coverage_edges.py tests/test_qualification_alpha.py -q and .venv/bin/python -m py_compile src/blueprint_pipeline/qualification.py
```

</details>

### [P1-50] Parametrize that any blocked capability check forces a non-ready readiness decision

- **Priority:** P1 · **Effort:** M · **Dimension:** Launch gates & readiness
- **Goal:** Lock the safety invariant that a single 'blocked' capability check (high-severity blocker) downgrades the readiness decision to not_ready_yet, end-to-end across all seven check ids.
- **Files:** `src/blueprint_pipeline/qualification.py`, `tests/test_qualification_coverage_edges.py`
- **Validate (CPU):** .venv/bin/python -m pytest tests/test_qualification_coverage_edges.py -q  &&  .venv/bin/python -m py_compile src/blueprint_pipeline/qualification.py

- **Context:** These seven capability checks are the automation-gap gate; the single-blocked-check -> not_ready_yet mapping is the safety-relevant invariant (e.g. a route width below minimum_path_width_m must block). Confirmed: severity='high' on blocked at qualification.py:3923; not_ready_yet trigger at 3959; partial existing coverage at tests/test_qualification_coverage_edges.py:600-631. Use .venv/bin/python.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU/no-spend task. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

The chain in src/blueprint_pipeline/qualification.py encodes: a 'blocked' capability check -> high-severity blocker (_build_blocker_register line ~3923: severity='high' if status=='blocked') -> forces not_ready_yet (_build_readiness_decision line ~3959). The existing test at tests/test_qualification_coverage_edges.py:600-631 only checks one blocked entry with a hand-built blocker_register; it does not assert the invariant end-to-end across all capability check ids built by _build_capability_checks.

Add a parametrized test in tests/test_qualification_coverage_edges.py over each capability check id produced by _build_capability_checks (read the function to confirm the exact ids; per the audit they are: clearance_precheck, reach_envelope_precheck, workcell_occupancy_analysis, choke_point_detection, occlusion_analysis, route_viability_hypotheses, coexistence_fit). For each id, construct the minimal geometry_evidence / route_graph / scope_record (and any other inputs _build_capability_checks consumes) that drives THAT specific check to status 'blocked', then run the real chain _build_capability_checks -> _build_blocker_register -> _build_readiness_decision and assert the decided status == 'not_ready_yet'. Prefer driving the inputs through the real builders rather than hand-building the blocker_register, so the test exercises the actual mapping. If a particular check cannot be independently driven to 'blocked' via inputs, document why in a comment and assert via the blocker_register path for that id only.

Constraints: keep world-model backends swappable; protect provenance, rights, privacy, and raw-capture truth; readiness logic is support, NOT a policy-success claim. Do not modify production logic unless you find a check that cannot reach 'blocked' at all (in that case, surface it as a finding in your final message rather than silently changing thresholds). This must satisfy $HOME/workspace/Blueprint-WebApp/docs/autonomous-loop-evidence-checklist-2026-05-03.md before claiming done.

Then run: .venv/bin/python -m pytest tests/test_qualification_coverage_edges.py -q and .venv/bin/python -m py_compile src/blueprint_pipeline/qualification.py
```

</details>

### [P1-51] Deduplicate hidden-zone and route-edge gate thresholds across orchestrator and qualification

- **Priority:** P1 · **Effort:** S · **Dimension:** Launch gates & readiness
- **Goal:** Make the agent reviewer reference the shared hidden-zone constant from the qualification envelope and give the 0.7 route-edge cutoff a named constant, so the two surfaces cannot silently diverge.
- **Files:** `src/blueprint_pipeline/agent_runtime/orchestrator.py`, `src/blueprint_pipeline/qualification.py`, `tests/test_agent_runtime_orchestrator_coverage.py`
- **Validate (CPU):** .venv/bin/python -m pytest tests/test_agent_runtime_orchestrator_coverage.py tests/test_qualification_coverage_edges.py -q  &&  .venv/bin/python -m py_compile src/blueprint_pipeline/agent_runtime/orchestrator.py src/blueprint_pipeline/qualification.py

- **Context:** The hidden-zone bound drives both whether qualification downgrades a scene to 'risky' AND whether the agent review adds a high-severity evidence gap that becomes a recapture blocker. Divergence means the readiness report and the agent memo disagree about the same scene. Confirmed literals: orchestrator.py:482 (0.7) and 503 (0.35); shared constant qualification.py:3484 (maximum_hidden_zone_bound=0.35). Use .venv/bin/python.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU/no-spend task. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

src/blueprint_pipeline/agent_runtime/orchestrator.py _evidence_audit hardcodes the hidden-zone limit as a literal 0.35 (line ~503: if hidden_zone_bound > 0.35) and the low-confidence route-edge cutoff as a literal 0.7 (line ~482: if confidence < 0.7). The hidden-zone limit is already a named constant in src/blueprint_pipeline/qualification.py: _GENERIC_CAPABILITY_ENVELOPE['maximum_hidden_zone_bound'] = 0.35 (defined near line 3478/3484), consumed by _build_readiness_decision and _build_capability_checks. Two unsynchronized copies of the same physical threshold mean tuning one silently desyncs the readiness report from the agent memo.

Change:
1. Import/reference the shared hidden-zone constant in the orchestrator instead of the literal 0.35. Avoid creating an import cycle — check whether agent_runtime/orchestrator.py can import from qualification.py cleanly; if a direct import risks a cycle, hoist maximum_hidden_zone_bound into a small shared constants location both modules import, or expose a lightweight accessor. Keep the value identical (0.35).
2. Give the 0.7 route-edge confidence cutoff a named, documented module-level constant (e.g. MINIMUM_ROUTE_EDGE_CONFIDENCE = 0.7) with a one-line comment on what it gates.

Constraints: keep world-model backends swappable (do not couple modules more than needed; a shared constant is fine, a hard cross-module dependency chain is not); protect provenance, rights, privacy, and raw-capture truth; these thresholds gate review/readiness support, NOT policy-success claims. Add/extend tests. This must satisfy $HOME/workspace/Blueprint-WebApp/docs/autonomous-loop-evidence-checklist-2026-05-03.md before claiming done.

Add an assertion in tests/test_agent_runtime_orchestrator_coverage.py that the orchestrator's hidden-zone threshold equals qualification._GENERIC_CAPABILITY_ENVELOPE['maximum_hidden_zone_bound'] (so a future edit to one is caught), plus a test that an edge with confidence just below MINIMUM_ROUTE_EDGE_CONFIDENCE is flagged and one just at/above is not.

Then run: .venv/bin/python -m pytest tests/test_agent_runtime_orchestrator_coverage.py tests/test_qualification_coverage_edges.py -q and .venv/bin/python -m py_compile src/blueprint_pipeline/agent_runtime/orchestrator.py src/blueprint_pipeline/qualification.py
```

</details>

### [P1-52] Add main()-flow coverage and a stale-test-list guard for the external alpha launch gate

- **Priority:** P1 · **Effort:** M · **Dimension:** Launch gates & readiness
- **Goal:** Cover the untested main() control flow of run_external_alpha_launch_gate.py and assert every path in the hardcoded pipeline pytest list still exists on disk.
- **Files:** `scripts/run_external_alpha_launch_gate.py`, `tests/test_external_alpha_launch_gate.py`
- **Validate (CPU):** .venv/bin/python -m pytest tests/test_external_alpha_launch_gate.py -q  &&  .venv/bin/python -m py_compile scripts/run_external_alpha_launch_gate.py

- **Context:** The external alpha gate is the cross-repo entrypoint for the alpha launch decision; an untested main() is where a flag-handling regression (e.g. accidentally skipping the pipeline leg) would let the gate report 'passed' without running the contract suite. Confirmed: main() at lines 197-285, require-android raise/print at 270-273, staleness note at 229-233, hardcoded pytest literal at 91-102. Loader at tests/test_external_alpha_launch_gate.py:12. Use .venv/bin/python.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU/no-spend task. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only. iOS/Android/Xcode/npm legs MUST be stubbed via monkeypatch — never actually invoked.

tests/test_external_alpha_launch_gate.py covers _resolve_simulator_destination, _run timeout behavior, _android_skip_reason, and _pipeline_pytest_command, but NOT the main() control flow in scripts/run_external_alpha_launch_gate.py (lines ~197-285). Load the module via the existing _load_gate_module helper (tests/test_external_alpha_launch_gate.py:12).

Add tests that monkeypatch the side-effecting helpers — at minimum gate._run, gate._ensure_extract_frames_dependencies, gate._resolve_swift_packages, gate._resolve_simulator_destination, and gate.load_env_files — to record calls instead of executing, then drive main() (passing argv via parser if main reads sys.argv; otherwise monkeypatch sys.argv) and assert which legs execute under each flag combination:
- --skip-capture-cloud skips the npm-test leg; without it, the extract-frames npm leg runs.
- --skip-ios skips the xcodebuild leg; without it, _resolve_simulator_destination and the xcodebuild _run are invoked.
- --skip-android skips the android leg.
- --skip-pipeline skips the pipeline pytest leg; without it, _run is called with _pipeline_pytest_command() and contract_test_env().
- android: with android skip reason present and NOT --require-android, main prints manual_required and does NOT raise; with --require-android it raises RuntimeError (lines ~270-273). Monkeypatch _android_skip_reason to return a reason and assert both branches.
- desktop-vs-canonical staleness note (lines ~229-233): when a desktop repo path exists and differs from the canonical capture repo, the note is printed. Use monkeypatch + capsys.

Separately add a guard test that, with all run legs monkeypatched OR by calling gate._pipeline_pytest_command() directly, asserts every pytest path in that list resolves to an existing file under the pipeline repo root (e.g. (pipeline_repo / path).is_file()). The current list is a hardcoded literal (lines ~91-102): tests/test_alpha_readiness.py, tests/test_qualification_alpha.py, tests/test_site_world_packaging.py, tests/test_storage_trigger.py, tests/test_webapp_sync.py, tests/test_world_model_candidate_parity.py — a renamed/removed test would make the gate silently skip coverage.

Constraints: hermetic, no subprocess/GPU/cloud/xcode; keep world-model backends swappable; protect provenance, rights, privacy, raw-capture truth; gate output is a contract/support claim NOT a policy-success claim. This must satisfy $HOME/workspace/Blueprint-WebApp/docs/autonomous-loop-evidence-checklist-2026-05-03.md before claiming done.

Then run: .venv/bin/python -m pytest tests/test_external_alpha_launch_gate.py -q and .venv/bin/python -m py_compile scripts/run_external_alpha_launch_gate.py
```

</details>

### [P1-53] Add an end-to-end fixture test for run_agent_review's deterministic reviewer pipeline

- **Priority:** P1 · **Effort:** M · **Dimension:** Launch gates & readiness
- **Goal:** Assert cross-step data flow through the offline reviewer stack: an evidence gap from _evidence_audit propagates into agent_blocker_register.json, recapture_plan.json, and the memo.
- **Files:** `src/blueprint_pipeline/agent_runtime/orchestrator.py`, `src/blueprint_pipeline/agent_runtime/artifacts.py`, `tests/test_agent_runtime_orchestrator_coverage.py`
- **Validate (CPU):** .venv/bin/python -m pytest tests/test_agent_runtime_orchestrator_coverage.py -q  &&  .venv/bin/python -m py_compile src/blueprint_pipeline/agent_runtime/orchestrator.py

- **Context:** This deterministic path produces the product's reviewer output (evidence_audit.json, agent_blocker_register.json, recapture_plan.json, oem_handoff_summary.json, agent_review_memo.md) and is fully CPU-runnable now while GPU work is paused. A wiring regression (e.g. recapture_plan no longer consuming the enriched blocker register) would pass per-function unit tests but break the actual artifact a human reviewer reads. Confirmed: _evidence_audit hidden-zone gap emitted when hidden_zone_bound>0.35 (orchestrator.py:503); run_agent_review at lines 860-990; smoke test at tests/test_agent_runtime_orchestrator_coverage.py:340-378. Use .venv/bin/python.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU/no-spend task. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only. Use the offline fake provider so all local-builders fire with zero LLM/cloud calls.

src/blueprint_pipeline/agent_runtime/orchestrator.py run_agent_review (lines ~860-990) chains the full reviewer stack: intake_normalizer -> evidence_auditor -> blocker_taxonomist -> capability_envelope_writer -> standards_retriever -> site/workcell/route reviewers -> oem_handoff_writer -> recapture_planner -> memo. Individual local-builders are unit-tested in tests/test_agent_runtime_orchestrator_coverage.py and run_agent_review is smoke-tested there with a fake provider (lines ~340-378), but no test asserts cross-step wiring end-to-end.

Extend the fake-provider run_agent_review test (or add a new one) in tests/test_agent_runtime_orchestrator_coverage.py: build an on-disk capture fixture whose geometry_evidence has hidden_zone_bound > 0.35 (so _evidence_audit emits the hidden-zone high-severity gap with detail string 'Hidden-zone bound ... exceeds the readiness envelope.'). Run run_agent_review with the offline fake provider (override returns None so local-builders fire). Then assert the SAME gap detail string appears in: evidence_audit.json, agent_blocker_register.json, and at least one recapture_plan.json step — and ideally in agent_review_memo.md. Read src/blueprint_pipeline/agent_runtime/artifacts.py to confirm the exact artifact filenames and the on-disk layout run_agent_review writes, and how to load each. Assert by reading the written artifacts from the run output directory, not by re-deriving values in the test.

Constraints: deterministic, offline, no GPU/cloud/LLM; keep world-model backends swappable; protect provenance, rights, privacy, raw-capture truth; reviewer artifacts are human-review support, NOT policy-success claims. This must satisfy $HOME/workspace/Blueprint-WebApp/docs/autonomous-loop-evidence-checklist-2026-05-03.md before claiming done.

Then run: .venv/bin/python -m pytest tests/test_agent_runtime_orchestrator_coverage.py -q and .venv/bin/python -m py_compile src/blueprint_pipeline/agent_runtime/orchestrator.py
```

</details>

## Warm render transport / object store

### [P1-54] Run-scope or drain the warm inbox key so a restarted pod can't re-claim an orphaned job

- **Priority:** P1 · **Effort:** M · **Dimension:** Warm render transport / object store
- **Goal:** Make presign_warm_inbox_channel write a run-unique inbox key (or clear a pre-existing one) so a restarted pod resetting _last_seq=0 cannot re-claim an already-served job.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/wam_provider_object_store.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_wam_provider_object_store.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/warm_render_server.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_warm_render_server.py`
- **Validate (CPU):** python -m pytest tests/test_wam_provider_object_store.py tests/test_warm_render_server.py -q (add a presign_warm_inbox_channel test mirroring tests/test_wam_provider_object_store.py:37-121: assert it writes a run-unique inbox_key — or clears a pre-existing inbox object via delete_object — plus 0600 url files and status=completed; pair with a warm_render_server test driving SignedUrlJobSource with _last_seq reset to 0 and asserting a previously-served seq is NOT re-claimed when the inbox is run-scoped/drained). Also `python -m py_compile src/blueprint_pipeline/wam_provider_object_store.py src/blueprint_pipeline/warm_render_server.py`.

- **Context:** The inbox half of the same stale/uniqueness class as the fixed output-key bug. A re-claimed job on the active 'open the refrigerator' warm lane means a duplicate paid render or a wrong-scenario render attributed to a new request — silent cross-run contamination, currently entirely untested. Files: src/blueprint_pipeline/wam_provider_object_store.py (presign_warm_inbox_channel 392, _job_key_component 71, output-path stale clear 280), src/blueprint_pipeline/warm_render_server.py (SignedUrlJobSource dedup 193). The output-key clear pattern to mirror is the delete_object at line 281. Test scaffolding: tests/test_wam_provider_object_store.py FakeClient (add put_object/delete_object/generate_presigned_url) + SimpleNamespace boto3 stub.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU-only, no-spend hermetic fix.

Problem: presign_warm_inbox_channel (src/blueprint_pipeline/wam_provider_object_store.py:392-472) always put_objects {"seq":0} (lines 447-448) at a FIXED inbox_key derived only from job_dir (line 438 via _job_key_component — no run id/timestamp/nonce). SignedUrlJobSource dedups only by the monotonic seq it has seen locally (src/blueprint_pipeline/warm_render_server.py:193-201), and a pod whose process restarts resets _last_seq=0 (line 180), so it will RE-CLAIM and re-render an already-served job; two overlapping control planes both PUT seq=1.. to the same key with no idempotency/version guard, silently contaminating across runs.

Fix: make the inbox run-scoped. Add a session nonce / run id (injectable param defaulting to a generated value) into the inbox_key (alongside _job_key_component) so each serve session gets its own inbox object, mirroring the explicit stale-object clear the output path already does at wam_provider_object_store.py:280-283. Alternatively/additionally, before seeding {"seq":0}, delete_object the pre-existing inbox_key so a fresh session cannot inherit an orphaned job. Return the chosen inbox_key (already returned at line 467) and surface the run id in the status dict so the control plane and pod agree on the same key. Keep the file-based-secrets and 0600 sensitive-file behavior intact (lines 458-463); keep redacted-URL-only output.

Constraints: keep world-model backends swappable (do not hardcode any one object store; reuse the same DEFAULT_*_FILES secret resolution). Protect provenance/rights/privacy — never log or persist raw presigned URLs or secrets; only redacted URLs and the inbox_key. Render outputs are simulator support, NOT policy-success claims. Add/extend tests. Run `python -m pytest tests/test_wam_provider_object_store.py tests/test_warm_render_server.py -q` and `python -m py_compile src/blueprint_pipeline/wam_provider_object_store.py src/blueprint_pipeline/warm_render_server.py`.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-55] Isolate or clear /workspace/out warm_results per serve session so the output zip can't carry stale state

- **Priority:** P1 · **Effort:** M · **Dimension:** Warm render transport / object store
- **Goal:** Give each --serve session a clean results area (per-session out subdir or explicit clear of warm_results + stale markers) so the cumulative output zip stops growing unbounded and stops carrying stale results/markers.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/warm_render_server.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_warm_render_server.py`
- **Validate (CPU):** python -m pytest tests/test_warm_render_server.py -q (add: pre-populate tmp_path/out/warm_results with a stale json and tmp_path/out/bootstrap.json; construct the new serve-session entrypoint/flag; assert the stale result is cleared or isolated under a prior-session subdir and the new session's results land in a clean namespace; assert the default constructor still does NOT clear, keeping existing tests green). Also `python -m py_compile src/blueprint_pipeline/warm_render_server.py`.

- **Context:** Unbounded growth on the exact object the control plane polls turns the 'seconds per rerun' warm lane (the active 'open the refrigerator' seed lane) into progressively slower/costlier polls, and is the root enabler of both the stale-result and stale-marker correctness bugs. File: src/blueprint_pipeline/warm_render_server.py (SignedUrlJobSource.__init__ 171, publish_result 203; poll_result reads the zip at 246). Test scaffolding: tests/test_warm_render_server.py uses tmp_path + SignedUrlJobSource directly (see test_signed_url_job_source_publishes_result_into_out_dir at line 162).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU-only, no-spend hermetic fix.

Problem: SignedUrlJobSource.__init__ (src/blueprint_pipeline/warm_render_server.py:171-180) mkdirs out_dir/warm_results but never clears it, and the worker heartbeat re-uploads the WHOLE /workspace/out as a cumulative zip (consumed by WarmPoolClient.poll_result line 246 and _await_warm_serve_ready). Across reruns warm_results/, bootstrap.json, runner_console.log and per-scenario artifacts accumulate in the same out dir with no truncation or per-session subdir; no code resets out_dir at the start of a new --serve session. Effects: the upload zip grows every rerun (slower/larger polls), and stale warm_results / stale bootstrap.json are the root enablers of the staleness and marker bugs.

Fix: add a serve-session entrypoint or a flag to SignedUrlJobSource that, when starting a new serve session, either (a) writes results into a per-session subdir like out_dir/warm_results/<session_id>/ (and teaches publish_result + the result key used by poll_result to include that session_id), or (b) explicitly clears stale warm_results/*.json and stale bootstrap.json / warm_serve_ready.json at session start. Prefer the approach that composes with the run-scoped result token / instance_id marker work (consistent session_id everywhere). Keep the default constructor backward-compatible (no clearing unless a new serve-session flag/param is set) so existing tests pass. Pure filesystem — no network, no GPU.

Constraints: keep world-model backends swappable. Protect provenance/rights/privacy/raw-capture-truth — clearing must touch ONLY warm transport scratch (warm_results + serve markers), never raw capture inputs or provenance artifacts. Render outputs are simulator support, NOT policy-success claims. Add/extend tests. Run `python -m pytest tests/test_warm_render_server.py -q` and `python -m py_compile src/blueprint_pipeline/warm_render_server.py`.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-56] Add hermetic coverage for presign_warm_inbox_channel and _await_warm_serve_ready

- **Priority:** P1 · **Effort:** M · **Dimension:** Warm render transport / object store
- **Goal:** Cover the warm inbox presign + serve-ready orchestration (currently zero tests) so every robustness fix has a regression net and the happy path is verified.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/tests/test_wam_provider_object_store.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_isaac_g1_kitchen_parity_job.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/wam_provider_object_store.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`
- **Validate (CPU):** python -m pytest tests/test_wam_provider_object_store.py tests/test_isaac_g1_kitchen_parity_job.py -q (new presign_warm_inbox_channel completed+blocked tests pass; new _await_warm_serve_ready ready/timeout/bootstrap-phase-only tests pass). Also `python -m py_compile tests/test_wam_provider_object_store.py tests/test_isaac_g1_kitchen_parity_job.py`.

- **Context:** This transport is what the paused GPU 'open the refrigerator' lane resumes on; the happy path itself is currently unverified, so a refactor or any of the staleness/TTL/marker fixes could break inbox presign or serve-ready detection with no signal. These are pure-transport functions, fully CPU-testable. Files: src/blueprint_pipeline/wam_provider_object_store.py (presign_warm_inbox_channel 392), src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py (_await_warm_serve_ready 424). Reusable scaffolding: FakeClient/SimpleNamespace boto3 stub at tests/test_wam_provider_object_store.py:50-83; fake-provider urlopen-returns-zip pattern at tests/test_isaac_g1_kitchen_parity_job.py:360-372.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU-only, no-spend hermetic test-coverage task — adds tests only (no production behavior change required unless a test exposes a real defect, in which case note it).

Gap: presign_warm_inbox_channel (src/blueprint_pipeline/wam_provider_object_store.py:392-472) and _await_warm_serve_ready (src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py:424-462) are wired into run_isaac_g1_kitchen_parity_job serve mode (isaac_g1_kitchen_parity_job.py:546-624) but have ZERO tests — only launch_with_marker_retry is tested (tests/test_isaac_g1_kitchen_parity_job.py:375-427) and test_wam_provider_object_store.py covers only stage_wam_provider_bundle_object_store, not the inbox presign. So inbox key derivation, the 0600 url-file writes, blocker propagation, the {"seq":0} seed, and serve-ready polling (bootstrap phase tracking, warm_serve_ready detection, timeout) ship untested.

Add hermetic tests:
(1) presign_warm_inbox_channel — mirror tests/test_wam_provider_object_store.py:37-121: monkeypatch a SimpleNamespace boto3 stub whose FakeClient implements put_object(Bucket,Key,Body,ContentType), generate_presigned_url(...), and (for the run-scope/drain fix if present) delete_object; with valid file-based creds assert status=completed, the 0600 warm_inbox_get_url.txt / warm_inbox_put_url.txt files (mode 0600), inbox_key shape, the seq:0 seed body, and redacted-URL-only output; with NO creds assert status=blocked with the missing_object_store_* blockers (reuse the blocked-path assertion style at tests/test_wam_provider_object_store.py:11-34).
(2) _await_warm_serve_ready — reuse the fake-provider + monkeypatched urllib pattern at tests/test_isaac_g1_kitchen_parity_job.py:360-372: write provider_output_get_url.txt into a tmp job_dir, monkeypatch urllib.request.urlopen to return an in-memory zip; assert (a) warm_serve_ready.json present -> ready True with serve_detail/last_phase; (b) only bootstrap.json present -> ready False with reason serve_ready_timeout and last_phase tracked; (c) empty/absent output -> ready False timeout. Drive its clock/sleep via the existing time monkeypatch so it does not actually sleep.

Constraints: tests must be fully hermetic (fakes only, no boto3/no network/no GPU). Do not weaken provenance/privacy assertions — assert raw URLs/secrets do NOT leak into manifests/status dicts. Render outputs are simulator support, NOT policy-success claims. Run `python -m pytest tests/test_wam_provider_object_store.py tests/test_isaac_g1_kitchen_parity_job.py -q` and `python -m py_compile tests/test_wam_provider_object_store.py tests/test_isaac_g1_kitchen_parity_job.py`.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## scene_semantics (Gemini)

### [P1-57] Reconcile Gemini model-cascade ID drift across the three mirrored modules

- **Priority:** P1 · **Effort:** S · **Dimension:** scene_semantics (Gemini)
- **Goal:** Make the second-tier Gemini model id consistent across scene_semantics.py, render_visual_qc.py, and target_resolver.py and lock it with a cross-module test.
- **Files:** `src/blueprint_pipeline/scene_semantics.py`, `src/blueprint_pipeline/render_visual_qc.py`, `src/blueprint_pipeline/scene_placement/target_resolver.py`, `tests/test_scene_semantics.py`
- **Validate (CPU):** python3 -m pytest tests/test_scene_semantics.py -q (and the new cross-module test path if separate) — all green; python3 -m py_compile src/blueprint_pipeline/scene_semantics.py src/blueprint_pipeline/render_visual_qc.py src/blueprint_pipeline/scene_placement/target_resolver.py. Sanity grep: `grep -n 'gemini-3' src/blueprint_pipeline/scene_semantics.py src/blueprint_pipeline/render_visual_qc.py src/blueprint_pipeline/scene_placement/target_resolver.py` shows one canonical second-tier id.

- **Context:** scene_semantics.py is the Gemini room-classification + object-enumeration module that feeds SAM detection prompts; render_visual_qc.py and target_resolver.py (scene_placement) reuse the same cascade. The drift means every 'gemini-3-flash-preview' failure on the scene_semantics path may jump straight to gemini-2.5-flash if '3.1-pro-preview' is not a real served model, silently weakening the intended quality fallback that the 'open the refrigerator' G1 POV seed lane relies on for correct kitchen classification and object prompts. The inconsistency contradicts in-code comments claiming the modules 'mirror' each other.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), three modules each define a Gemini model cascade that is supposed to mirror the others, but the second-tier id has drifted:
- src/blueprint_pipeline/scene_semantics.py defines `_DEFAULT_MODEL_CASCADE` (around line 234) as ['gemini-3-flash-preview', 'gemini-3.1-pro-preview', 'gemini-2.5-flash', 'gemini-2.5-pro'].
- src/blueprint_pipeline/render_visual_qc.py (around line 30) uses 'gemini-3.1-pro-preview' as tier #2.
- src/blueprint_pipeline/scene_placement/target_resolver.py (around lines 42-43) uses 'gemini-3-pro-preview' as tier #2.
The documented canonical cascade (see `$HOME/.claude/projects/-Users-example-workspace-BlueprintCapturePipeline/memory/MEMORY.md` 'Model cascade' line) is gemini-3-flash-preview -> gemini-3-pro-preview -> gemini-2.5-flash -> gemini-2.5-pro, i.e. 'gemini-3-pro-preview' (no '.1'). An invalid model id silently fails generate_content (caught by the broad `except Exception: continue`) and falls through, so a wrong id degrades the cascade with NO error surfaced.

Do this:
1. Reconcile all three modules to ONE canonical second-tier id. Default to 'gemini-3-pro-preview' to match MEMORY.md and target_resolver.py UNLESS you find in-repo evidence (a comment, doc, or commit message) that 'gemini-3.1-pro-preview' is the intended id; if you keep '3.1', add a short code comment in scene_semantics.py and render_visual_qc.py explaining why it intentionally differs from the documented cascade, and note that confirming whether 'gemini-3.1-pro-preview' is a REAL served model id needs a real API key / Google docs (flag for human verification, do not call the API).
2. Add a regression test asserting scene_semantics._DEFAULT_MODEL_CASCADE equals the exact canonical list, plus a cross-module consistency test that imports the three cascade constants and asserts their second-tier entries match (or, if you intentionally diverge, that the divergence is exactly the documented one). Put the cross-module test in tests/test_scene_semantics.py or a new tests/test_gemini_cascade_consistency.py.

Constraints: keep world-model backends swappable (do not hardcode Gemini as the only path); protect provenance/rights/privacy/raw-capture-truth; render and review outputs are simulator-support signals, NOT policy-success claims; add/extend tests; run `python3 -m pytest <paths>` and `python3 -m py_compile <files>`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only (mock-only, no API key).
```

</details>

### [P1-58] Harden _extract_json_object against multi-object / nested-trailing-junk and markdown fences

- **Priority:** P1 · **Effort:** S · **Dimension:** scene_semantics (Gemini)
- **Goal:** Make _extract_json_object return the FIRST complete JSON object and strip markdown fences, instead of greedily matching to the last brace and returning {}.
- **Files:** `src/blueprint_pipeline/scene_semantics.py`, `tests/test_scene_semantics.py`
- **Validate (CPU):** Reproduce the bug first: `python3 -c "import sys; sys.path.insert(0,'src'); from blueprint_pipeline.scene_semantics import _extract_json_object as f; print(f('{\"a\":1} junk {\"b\":2}'))"` should print {} BEFORE the fix and {'a': 1} AFTER. Then: python3 -m pytest tests/test_scene_semantics.py -q (green); python3 -m py_compile src/blueprint_pipeline/scene_semantics.py.

- **Context:** `response_mime_type=application/json` usually yields clean JSON so the first json.loads wins, but the regex is the safety net when a fallback-tier model (commonly the 2.5 tiers) wraps output in prose or markdown. When _extract_json_object returns {}, infer_scene_semantics collapses to resolved_environment='default', environment_confidence=0.35, no detected objects, and generic local-fallback SAM prompts (scene_semantics.py around lines 391-411 and 505-516) with no error logged. For the 'open the refrigerator' G1 POV seed lane this means a kitchen capture can silently mis-classify as 'default' and lose kitchen-specific prompts, degrading downstream detection.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), `_extract_json_object(text)` in src/blueprint_pipeline/scene_semantics.py (around lines 184-199) does `json.loads(text)` then falls back to `re.search(r'\{.*\}', text, re.DOTALL)`. The greedy `.*` is buggy on two reproducible inputs:
- '{"a":1} junk {"b":2}' returns {} (the regex spans both braces, producing invalid JSON).
- 'prefix {"obj":{"k":1}} more {"z":9}' returns {} (matches through the last brace).
It also does NOT strip ```json ... ``` markdown fences, unlike `_extract_json_array` (around line 303) which already does.

Do this:
1. Strip markdown code fences first (mirror the `re.sub(r'```(?:json)?\s*', '', text)` logic from _extract_json_array).
2. After a plain `json.loads` attempt, replace the greedy regex with a brace-balanced scan that returns the FIRST complete top-level JSON object (track `{`/`}` depth, respecting string literals/escapes so braces inside strings don't break the count). Return {} only when no complete object is found.
3. Keep the existing contract: return a dict (or {} on failure), never raise.

Add unit tests in tests/test_scene_semantics.py asserting:
- _extract_json_object('{"a":1} junk {"b":2}') == {'a': 1}
- nested-with-trailing input returns the OUTER first object {'obj': {'k': 1}}
- a fenced object ('```json\n{"x":1}\n```') parses to {'x': 1}
- plain clean JSON still parses; pure-junk returns {}.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render/review outputs are simulator-support signals NOT policy-success claims; add/extend tests; run `python3 -m pytest tests/test_scene_semantics.py -q` and `python3 -m py_compile src/blueprint_pipeline/scene_semantics.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-59] Add bounded retry/backoff for transient 429 / RESOURCE_EXHAUSTED before abandoning a cascade tier

- **Priority:** P1 · **Effort:** M · **Dimension:** scene_semantics (Gemini)
- **Goal:** Retry a transient rate-limit/5xx once or twice with backoff on the SAME model before falling through to the next cascade tier, instead of treating every exception as a hard failure.
- **Files:** `src/blueprint_pipeline/scene_semantics.py`, `tests/test_scene_semantics.py`
- **Validate (CPU):** python3 -m pytest tests/test_scene_semantics.py -q (green, including the new retry tests); python3 -m py_compile src/blueprint_pipeline/scene_semantics.py. Tests must monkeypatch time.sleep to a spy so no real delay occurs.

- **Context:** MEMORY.md notes the prior two-call pattern 'hit free-tier rate limits' — throttling is a known recurring failure on this path. Today a transient 429 on the first/strongest model (gemini-3-flash-preview) silently skips the rest of the cascade and dumps a healthy capture into the 0.35-confidence local_auto_fallback with generic prompts, wasting the (already-uploaded) video and degrading the kitchen classification that the 'open the refrigerator' G1 POV seed lane depends on — when a 1-2s retry would likely have succeeded.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), both Gemini video helpers in src/blueprint_pipeline/scene_semantics.py wrap generate_content in `except Exception: continue`, advancing to the NEXT model on ANY error with no retry and no distinction between a transient rate-limit (429 / RESOURCE_EXHAUSTED) and a hard failure:
- _infer_with_gemini_video (around lines 373-385)
- _infer_capture_review_with_gemini_video (around lines 666-678)
The only existing sleep is the upload poll (around line 275).

Do this:
1. Add a small helper that classifies whether an exception looks transient (inspect the exception type name and message for '429', 'RESOURCE_EXHAUSTED', 'rate', 'quota', '503', '500', 'UNAVAILABLE', 'DEADLINE'). Be conservative and string-based so it works without importing google-specific exception classes (keep backends swappable / no hard google dep at import time).
2. On a transient error for a given model, retry that model up to a bounded number of attempts (e.g. 2 retries => 3 total) with backoff via time.sleep (e.g. 1s, 2s) BEFORE moving to the next tier. On a non-transient error or after retries are exhausted, fall through to the next model as today.
3. Make the retry count and base backoff overridable via env (e.g. SCENE_SEMANTICS_GEMINI_MAX_RETRIES, default small) and clamp to sane bounds.
4. Keep the existing return contracts (None on total failure) unchanged.

Add tests in tests/test_scene_semantics.py using a fake client whose generate_content raises a simulated 429 once then returns valid JSON; assert the result is returned (not skipped to a lower tier) and that time.sleep was invoked. Monkeypatch time.sleep to a no-op spy (the suite already patches semantics.time.sleep around line 333). Add a second test asserting a NON-transient exception is NOT retried (sleep not called for it). Reuse the fake-google-module pattern from test_scene_semantics_gemini_import_and_model_edges.

Constraints: keep world-model backends swappable (no hard google import for classification); protect provenance/rights/privacy/raw-capture-truth; render/review outputs are simulator-support signals NOT policy-success claims; add/extend tests; run `python3 -m pytest tests/test_scene_semantics.py -q` and `python3 -m py_compile src/blueprint_pipeline/scene_semantics.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only (mock-only, no real key, sleep monkeypatched).
```

</details>

### [P1-60] Delete uploaded Gemini video files after inference (File API retention/cost/privacy leak)

- **Priority:** P1 · **Effort:** M · **Dimension:** scene_semantics (Gemini)
- **Goal:** Best-effort delete the uploaded walkthrough video from the Gemini File API after inference completes (success OR cascade exhaustion) in both helpers.
- **Files:** `src/blueprint_pipeline/scene_semantics.py`, `tests/test_scene_semantics.py`
- **Validate (CPU):** Confirm the gap first: `grep -n '\.delete(' src/blueprint_pipeline/scene_semantics.py` returns nothing BEFORE; returns the cleanup call AFTER. Then python3 -m pytest tests/test_scene_semantics.py -q (green, including the delete-spy tests); python3 -m py_compile src/blueprint_pipeline/scene_semantics.py.

- **Context:** CLAUDE.md mandates 'Protect provenance, rights, privacy, and raw capture truth'. Raw walkthrough videos (including the kitchen captures feeding the 'open the refrigerator' G1 POV seed lane) currently persist on the Gemini File API longer than necessary and count against storage/retention; the double-upload compounds the exposure and cost. Best-effort post-inference deletion is the cheapest mitigation and is fully mock-validatable.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), _upload_gemini_video_file (src/blueprint_pipeline/scene_semantics.py around lines 258-279) uploads the raw walkthrough via client.files.upload, callers use the URI, but NOTHING ever calls client.files.delete — `grep -n '\.delete(' src/blueprint_pipeline/scene_semantics.py` returns nothing. Both _infer_with_gemini_video (around lines 373-413) and _infer_capture_review_with_gemini_video (around lines 666-694) leave the uploaded file on Google's File API to expire on Google's schedule, and infer_scene_semantics + infer_capture_fidelity_review each upload the SAME video independently (two uploads per capture).

Do this:
1. After generate_content finishes for a given uploaded file — on success AND on the None/exhaustion path — best-effort delete the uploaded file by name in a try/finally (wrap `client.files.delete(name=<uploaded.name>)` in try/except so a delete failure never breaks inference and never raises). Resolve the file name defensively (getattr(uploaded, 'name', None)); skip delete if no name.
2. Apply the same cleanup to both helpers. Do not change the success/None return contracts.
3. (Optional, only if low-risk) add a short comment noting the two-upload-per-capture pattern as a known follow-up; do NOT attempt to dedupe uploads across the two public functions in this task (out of scope).

Add tests in tests/test_scene_semantics.py: extend the fake client's `files` object with a `delete(name=...)` spy; assert delete is invoked with the uploaded file's name (a) after a successful inference and (b) after the cascade-exhaustion None path. Reuse the upload/poll mocking pattern from test_raw_video_review_polls_uploaded_file_until_active and the fake-google-module pattern from test_scene_semantics_gemini_import_and_model_edges. Monkeypatch time.sleep to no-op.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (this directly serves the raw-capture-privacy rule — minimize how long raw walkthrough video sits on a third-party endpoint); render/review outputs are simulator-support signals NOT policy-success claims; add/extend tests; run `python3 -m pytest tests/test_scene_semantics.py -q` and `python3 -m py_compile src/blueprint_pipeline/scene_semantics.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only (mock-only, no real key).
```

</details>

### [P1-61] Emit diagnostic logging on Gemini cascade exhaustion / empty-text / parse failure

- **Priority:** P1 · **Effort:** S · **Dimension:** scene_semantics (Gemini)
- **Goal:** Replace the silent `except/continue` and silent None returns with logger.warning at each distinct failure branch (upload failed, model raised, empty text, parse-empty, all-exhausted), including the model id.
- **Files:** `src/blueprint_pipeline/scene_semantics.py`, `tests/test_scene_semantics.py`
- **Validate (CPU):** python3 -m pytest tests/test_scene_semantics.py -q (green, including caplog assertions); python3 -m py_compile src/blueprint_pipeline/scene_semantics.py. Confirm warnings fire: the caplog tests assert at least one WARNING record per failure branch mentioning the model id / reason.

- **Context:** Because this is a paid external call, re-runs should be minimized. With no logs, an operator who sees the generic local_auto_fallback cannot tell whether to fix a missing key, wait out a 429, or fix a malformed prompt/response — and may waste a paid GPU/cloud run re-triggering the kitchen capture for the 'open the refrigerator' G1 POV seed lane. Observability here is the cheapest way to avoid repeat spend and is fully mock-testable via caplog.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), when every Gemini model fails, the two helpers in src/blueprint_pipeline/scene_semantics.py return None with NO logging, and infer_scene_semantics maps that to the single opaque string fallback_reason='gemini_video_unavailable_or_failed' (around line 501). Specifically the silent branches are: upload failure (helpers return None right after _upload_gemini_video_file, around lines 348-350 and 661-663), model raised (around lines 384-385 and 677-678 `except Exception: continue`), empty response text (around lines 388-389 and 681-682), parse-empty in the review helper (around lines 685-686), and the final `return None` (around lines 413 and 694). A logger is already configured at module top (`logger = logging.getLogger(__name__)`).

Do this:
1. Add a logger.warning at each distinct failure branch in BOTH helpers, naming the model id where applicable and the reason (e.g. 'upload_failed', 'model_raised', 'empty_response', 'parse_empty', 'cascade_exhausted'). Keep messages concise and free of any raw-capture PII — log the model id, the reason, and an exception class/message summary, but do NOT log video contents or file paths beyond the basename. For model_raised, include a short str(exc) summary.
2. Optionally enrich infer_scene_semantics so the fallback_reason it records distinguishes these cases where cheaply possible (e.g. keep 'raw_walkthrough_video_missing' as-is, but the inference-failure case can stay 'gemini_video_unavailable_or_failed' since the per-branch logs now carry the detail).

Add tests in tests/test_scene_semantics.py using pytest's `caplog` fixture: drive each failure branch (model raises, empty text, non-JSON parse-empty) via fake clients and assert a WARNING is logged that mentions the model id and/or the reason. Reuse the RaisingModelsClient / EmptyTextClient / NonJsonReviewClient patterns already present in test_scene_semantics_gemini_import_and_model_edges. Monkeypatch time.sleep to no-op.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (do NOT log raw video contents or full capture paths); render/review outputs are simulator-support signals NOT policy-success claims; add/extend tests; run `python3 -m pytest tests/test_scene_semantics.py -q` and `python3 -m py_compile src/blueprint_pipeline/scene_semantics.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only (mock-only, no real key).
```

</details>

### [P1-62] Reject bool/garbage confidence so it does not silently coerce to 1.0

- **Priority:** P1 · **Effort:** S · **Dimension:** scene_semantics (Gemini)
- **Goal:** Make confidence parsing in scene_semantics (and _bounded_score) reject non-numeric and bool JSON values and default to 0.0, so 'confidence': true does not masquerade as maximum confidence.
- **Files:** `src/blueprint_pipeline/scene_semantics.py`, `tests/test_scene_semantics.py`
- **Validate (CPU):** Reproduce: `python3 -c "print(float(True))"` prints 1.0. Then python3 -m pytest tests/test_scene_semantics.py -q (green, including the confidence:true -> 0.0 test and the _bounded_score(True) -> 0.0 test); python3 -m py_compile src/blueprint_pipeline/scene_semantics.py.

- **Context:** environment_confidence and the capture-review scores gate downstream behavior (local fallback uses 0.35; explicit hints use 1.0). A model erroneously emitting boolean confidence injects a spurious 1.0, over-trusting a weak classification and potentially skipping review — directly affecting whether a kitchen capture in the 'open the refrigerator' G1 POV seed lane is treated as high-confidence. Low likelihood but cheap to harden and trivially mockable; bool->1.0 is reproducible via `python3 -c 'print(float(True))'`.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), confidence parsing in src/blueprint_pipeline/scene_semantics.py (around lines 393-398) does `float(confidence_raw)` then clamps to [0,1]. Because bool is an int subclass in Python, JSON 'confidence': true coerces to 1.0 (float(True) == 1.0). The capture-review path shares this via `_bounded_score` (around lines 524-529).

Do this:
1. Add explicit type validation so bool and non-numeric values are rejected and default to 0.0. Concretely: treat the value as valid only if it is an int/float that is NOT a bool (e.g. `isinstance(v, (int, float)) and not isinstance(v, bool)`); otherwise default to 0.0. Apply this in the inline confidence parse AND inside _bounded_score (keep _bounded_score's `default` parameter behavior, just guard out bool/non-numeric before float()).
2. Keep the [0,1] clamp.

Add tests in tests/test_scene_semantics.py:
- A SuccessClient variant returning 'confidence': true; assert the resulting environment_confidence (or _GeminiResult.confidence) == 0.0, NOT 1.0.
- Direct unit assertions: _bounded_score(True) == 0.0, _bounded_score(False) == 0.0, _bounded_score(0.7) == 0.7, _bounded_score('x') == 0.0 (or the provided default).
Reuse the SuccessClient pattern from test_scene_semantics_gemini_import_and_model_edges.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render/review outputs are simulator-support signals NOT policy-success claims; add/extend tests; run `python3 -m pytest tests/test_scene_semantics.py -q` and `python3 -m py_compile src/blueprint_pipeline/scene_semantics.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## Code structure / tech debt

### [P1-63] Remove dead function _run_placement_visual_qc

- **Priority:** P1 · **Effort:** S · **Dimension:** Code structure / tech debt
- **Goal:** Delete the never-called _run_placement_visual_qc (runner lines 4106-4133) and confirm nothing referenced it.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`
- **Validate (CPU):** `grep -rn _run_placement_visual_qc scripts/ src/ tests/` returns nothing after deletion; `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py` passes; `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -q` stays green (0 failed once the PIL skip-guard task lands).

- **Context:** Part of the code-structure cleanup for the 7233-line `scripts/run_isaac_g1_kitchen_parity_eval.py`, which supports the active 'open the refrigerator' G1 POV render lane. The runner's Isaac-only section (1740-7233) carries dead and duplicated helpers; this is the one confirmed fully-dead function. Removing it shrinks the Isaac-only section and eliminates a misleading second QC entry point that falsely implies an alternate placement-QC wiring. Confirmed via grep that only the def line matches.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), remove dead code from `scripts/run_isaac_g1_kitchen_parity_eval.py`. The function `_run_placement_visual_qc` (lines 4106-4133) is defined but never called anywhere in `scripts/`, `src/`, or `tests/` — verified: `grep -rn _run_placement_visual_qc scripts/ src/ tests/` returns only its own `def` line. The live code path is `_run_task_visual_qc` (defined at line 4136, called at line 6474 inside `run_scenarios`), which performs placement QC inline. `_run_placement_visual_qc` is a stale duplicate of the `render_visual_qc` import-fallback block.

What to do:
1. Re-run `grep -rn _run_placement_visual_qc scripts/ src/ tests/` to re-confirm zero call sites (only the def). If any real caller exists, STOP and do not delete.
2. Delete the entire `_run_placement_visual_qc` function (def + body, lines ~4106-4133) and any now-orphaned blank lines.
3. Do NOT touch `_run_task_visual_qc` or its call at line 6474.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done, run `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py` and `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -q`.
```

</details>

### [P1-64] Extend build_parity_bundle + namelist test before any module extraction

- **Priority:** P1 · **Effort:** S · **Dimension:** Code structure / tech debt
- **Goal:** Make build_parity_bundle copy each newly extracted parity_* module into the GPU bundle and assert it in the namelist test, the guardrail every extraction depends on.
- **Files:** `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `tests/test_isaac_g1_kitchen_parity_job.py`, `scripts/run_isaac_g1_kitchen_parity_eval.py`
- **Validate (CPU):** `python -m pytest tests/test_isaac_g1_kitchen_parity_job.py -q` (the namelist test builds the real zip and asserts membership); for each extracted module verify its name appears in `zf.namelist()`; `python -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`.

- **Context:** `build_parity_bundle` (`src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py:200-224`) is the only thing that assembles the worker bundle, and `tests/test_isaac_g1_kitchen_parity_job.py:91-116` is the namelist guardrail that pins its contents. Confirmed: the test builds the real zip and asserts `isaac_g1_policy.py`, `render_visual_qc.py`, `scene_placement/*` are present. Every extraction task in this dimension (camera geometry, arm kinematics, manifest, io helpers) depends on this guardrail so the runner that drives the 'open the refrigerator' G1 render lane keeps importing its math on the GPU worker. This must land alongside (or just before) the first extraction.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), prepare the GPU-bundle guardrail that makes the parity-module extractions (parity_geometry / parity_kinematics / parity_manifest / parity_io) safe. The GPU worker has NO `blueprint_pipeline` on its path: it imports sibling modules flatly from `/workspace/bundle`. `build_parity_bundle` in `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py` (lines ~204-224) already flat-copies `run_isaac_g1_kitchen_parity_eval.py`, `isaac_g1_policy.py`, `render_visual_qc.py`, and `warm_render_server.py` into the bundle, and `tests/test_isaac_g1_kitchen_parity_job.py::test_build_parity_bundle_contains_runner_policy_request_and_assets` (lines 91-116) asserts each is in `zf.namelist()`. If an extracted module is NOT copied into the bundle, the runner silently falls back or crashes on the worker — this is the single highest-risk part of any extraction.

What to do:
1. In `build_parity_bundle`, for EACH parity module you (or a sibling task) extract into `src/blueprint_pipeline/`, add a flat write mirroring the existing `(bundle / "render_visual_qc.py").write_bytes(visual_qc.read_bytes())` lines (read from `_repo_root() / "src" / "blueprint_pipeline" / "<module>.py"`, write to `bundle / "<module>.py"`). Only copy modules that exist; do not add writes for modules not yet created.
2. In `tests/test_isaac_g1_kitchen_parity_job.py`, add an `assert "<module>.py" in names` line alongside lines 103-114 for each module copied.
3. Keep the runner's import of each extracted module as a bundle-first then `blueprint_pipeline`-fallback dual-try (`try: import parity_geometry as _pg  except Exception: from blueprint_pipeline import parity_geometry as _pg`), mirroring the policy import at runner lines 41-46, so it resolves on both the worker (flat bundle) and in tests (installed package).

If no parity_* module has been extracted yet, implement this as the pattern/scaffolding and add a self-check test asserting the existing four sibling modules (isaac_g1_policy, render_visual_qc, warm_render_server) are still in the namelist, so the guardrail is in place before extraction.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; worker behavior must stay byte-identical. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done, run `python -m pytest tests/test_isaac_g1_kitchen_parity_job.py -q` and `python -m py_compile src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`.
```

</details>

### [P1-65] Extract CPU-pure camera/projection geometry into parity_geometry module

- **Priority:** P1 · **Effort:** M · **Dimension:** Code structure / tech debt
- **Goal:** Move the pure trig/linear-algebra camera-math cluster into src/blueprint_pipeline/parity_geometry.py, re-import it back into the runner so all M.<name> callers stay unchanged.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `src/blueprint_pipeline/parity_geometry.py`, `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`, `tests/test_isaac_g1_kitchen_parity_job.py`
- **Validate (CPU):** `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_local_render_preview.py tests/test_isaac_g1_kitchen_parity_job.py -q` stays green (these assert M.project_point_to_pixel/look_at_quat/camera_aperture_for_fov/manipulation_cam_pose/follow_cam_pose/verify_cam_pose/scene_framing); new `import blueprint_pipeline.parity_geometry` test passes; `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py src/blueprint_pipeline/parity_geometry.py`.

- **Context:** This camera math is the projection/FOV/look-at backbone of the 'open the refrigerator' G1 POV-framing and skeleton-projection pipeline — its correctness gates every rendered POV frame. The repo already proves this extraction pattern: `isaac_g1_policy.py` and `render_visual_qc.py` live in `src/blueprint_pipeline/`, are imported by the runner with a bundle/repo dual-try (runner lines 41-46), and are flat-copied into the bundle by `build_parity_bundle` (`isaac_g1_kitchen_parity_job.py:204-218`). Confirmed line numbers via grep; tests already reference these names as `M.<name>` (e.g. runner test lines 132-133, 563, 1757, 1769, 1776), and the runner is loaded in tests via `importlib.util.spec_from_file_location` so module-level names must remain importable post-move.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), extract the CPU-pure camera/projection geometry out of the 7233-line `scripts/run_isaac_g1_kitchen_parity_eval.py` into a new importable module `src/blueprint_pipeline/parity_geometry.py`. These functions are pure trig/linear-algebra with NO `pxr`/Isaac dependency, yet several (notably `camera_aperture_for_fov` at line 5103) live deep in the Isaac-only section far from their siblings, so they can currently only be unit-tested by exec'ing the whole 330KB runner.

Move these (confirmed line numbers): `yaw_to_quat` (883), `_norm` (888), `_cross` (893), `look_at_quat` (897), `project_point_to_pixel` (937), `scene_framing` (~960), `follow_cam_pose` (~1001), `verify_cam_pose` (~1009), `manipulation_cam_pose` (~1030), `_weighted_xyz` (~1088), `_manipulation_camera_target_with_arm_context` (~1107), `_camera_pitch_down_deg` (~1129), `_target_raised_to_max_pitch_down` (~1140), `_manipulation_seed_arm_target_for_shoulder` (~1152), `_projection_dict` (~1169), and `camera_aperture_for_fov` (5103).

What to do:
1. Create `src/blueprint_pipeline/parity_geometry.py` containing these functions verbatim (preserve signatures, docstrings, numeric behavior). Move `_norm`/`_cross` too and have the camera functions use them from module scope.
2. In the runner, replace the moved defs with a dual-try re-import at module scope so the names stay module-level (tests access them as `M.project_point_to_pixel`, `M.look_at_quat`, `M.camera_aperture_for_fov`, `M.manipulation_cam_pose`, `M.follow_cam_pose`, `M.verify_cam_pose`, `M.scene_framing`): `try: from parity_geometry import *  except Exception: from blueprint_pipeline.parity_geometry import *` — OR explicit `from ... import yaw_to_quat, look_at_quat, ...`. Mirror the existing policy import fallback at runner lines 41-46. Verify the runner's internal callers (e.g. `_set_camera_fov` near line 5114 calling `camera_aperture_for_fov`) still resolve.
3. Add the bundle copy + namelist assertion for `parity_geometry.py` (see the build_parity_bundle guardrail task) so the GPU worker still imports the math.
4. Add a direct `import blueprint_pipeline.parity_geometry` test asserting a couple of known values (e.g. `project_point_to_pixel`, `camera_aperture_for_fov`) so the math is testable without exec'ing the runner.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; this is a behavior-preserving move (no logic changes); add/extend tests. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done, run `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_local_render_preview.py tests/test_isaac_g1_kitchen_parity_job.py -q` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py src/blueprint_pipeline/parity_geometry.py`.
```

</details>

### [P1-66] Extract arm-reach/skeleton kinematics and de-duplicate inline vec3 helpers

- **Priority:** P1 · **Effort:** M · **Dimension:** Code structure / tech debt
- **Goal:** Move the GPU-independent arm-reach/skeleton kinematics into parity_kinematics.py and collapse the triplicated vector primitives into shared module-level helpers.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `src/blueprint_pipeline/parity_kinematics.py`, `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`, `tests/test_isaac_g1_kitchen_parity_job.py`
- **Validate (CPU):** `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_local_render_preview.py tests/test_isaac_g1_kitchen_parity_job.py -q` stays green (reach/skeleton assertions pin numeric output); `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py src/blueprint_pipeline/parity_kinematics.py`.

- **Context:** This is the arm-reach skeleton that proves the G1 hand reaches the faucet/handle by step N in the active 'open the refrigerator' / faucet-manipulation POV lane (project memory: 9 arm landmarks reach the faucet by step 7). Existing reach/skeleton tests pin exact numeric output (runner test lines 461-584: `M.compute_arm_reach_skeleton`, `M.arm_reach_rotation` with reach fractions 0.0/0.5/1.0), so any drift fails loudly. The same 7 vector primitives are hand-redefined in at least 3 places (module-level `_norm`/`_cross` at runner lines 888/893 plus two closure sets), a copy-paste drift risk this consolidates. Best done after parity_geometry so the shared `_norm`/`_cross` have one home.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), extract the CPU-pure arm-reach/skeleton kinematics out of `scripts/run_isaac_g1_kitchen_parity_eval.py` into a new `src/blueprint_pipeline/parity_kinematics.py`, and remove duplicated inline vector helpers. These functions are self-documented as 'Pure geometry, GPU-independent' (see runner lines 2511 and 2572) but live in the Isaac-only section.

Move (confirmed): `compute_arm_reach_skeleton` (2511), `arm_reach_rotation` (2572), `_rest_skeleton_world` (~2483), `_project_skeleton` (~2450), `manipulation_ready_arm_joint_deltas` (~2351), `_apply_joint_deltas` (~2367), `_joint_targets_for_pose` (~2377), `_find_arm_link` (~2605), `_is_manipulation_arm_link_name` (~2614), `nominal_g1_rest_offsets` (~6723). If `parity_geometry.py` already holds `_norm`/`_cross`, import them from there instead of redefining; otherwise place shared `vec3` primitives (sub/add/scale/length/dot/cross/norm) once at module scope in `parity_kinematics.py`.

What to do:
1. Create `src/blueprint_pipeline/parity_kinematics.py` with these functions verbatim.
2. Collapse the duplicated nested closures — `sub/add/scale/length` defined at ~2537-2549 inside `compute_arm_reach_skeleton` and `sub/dot/cross/length/norm` defined at ~2579-2591 inside `arm_reach_rotation` — into the shared module-level vec3 helpers, reusing the existing module-level `_norm`/`_cross` rather than redefining `cross`/`norm` a third time. Numeric output must be identical.
3. Re-import the moved names back into the runner at module scope with the bundle/repo dual-try fallback (mirror runner lines 41-46) so tests still access `M.compute_arm_reach_skeleton`, `M.arm_reach_rotation`, `M.manipulation_ready_arm_joint_deltas`, `M._apply_joint_deltas`, `M._is_manipulation_arm_link_name`, `M._project_skeleton`, `M._rest_skeleton_world`, `M.skeleton_world_for_frame`.
4. Add the bundle copy + namelist assertion for `parity_kinematics.py`.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; behavior-preserving move only; add/extend tests. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done, run `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_local_render_preview.py tests/test_isaac_g1_kitchen_parity_job.py -q` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py src/blueprint_pipeline/parity_kinematics.py`.
```

</details>

## Visual QC rubrics

### [P1-67] Reflect hard boolean failures in worst_severity rollup

- **Priority:** P1 · **Effort:** S · **Dimension:** Visual QC rubrics
- **Goal:** worst_severity should report 'high' (not 'none') when a frame is flagged solely by a hard boolean (robot missing / incoherent / background inconsistent).
- **Files:** `src/blueprint_pipeline/render_visual_qc.py`, `tests/test_render_visual_qc.py`
- **Validate (CPU):** python3 -m pytest tests/test_render_visual_qc.py -k 'worst or aggregat or flag' -q  &&  python3 -m py_compile src/blueprint_pipeline/render_visual_qc.py

- **Context:** worst_severity is the single human-facing severity in RenderQCReport.to_dict (lines 488-497) used by reviewers triaging the G1 render lane. A robot-missing frame reading 'none' undersells a serious defect. Pairs naturally with the fail-closed parser change (None safety booleans) so do that first if both land together.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), fix a misleading human-facing rollup in the visual-QC report.

File: src/blueprint_pipeline/render_visual_qc.py. verdict_is_flagged (lines 316-335) correctly flags on hard booleans, but worst_severity (lines 338-347) and RenderQCReport.worst_severity (line 515 via qc_render_frames) only aggregate overall_severity + per-anomaly severity. A frame flagged solely because robot_visible is False/None, coherent is False/None, or background_consistent is False/None rolls up worst_severity='none'. Any downstream consumer or reviewer keying on worst_severity (rather than the flagged boolean) would under-prioritize a robot-missing frame.

Fix: map hard boolean failures into the rollup. In worst_severity, for each verdict, if coherent/robot_visible/background_consistent is False (or None after the fail-closed change), treat that verdict as at least 'high' when computing the max rank. Keep anomaly + overall_severity aggregation. Do not change verdict_is_flagged. If you prefer not to change semantics, instead update the worst_severity docstring to state it is anomaly-only AND add the boolean mapping anyway — the input audit recommends the mapping; implement the mapping.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims. Add a test in tests/test_render_visual_qc.py feeding a reply with robot_visible=False, overall_severity='none', empty anomalies: assert verdict_is_flagged is True (already holds) AND worst_severity([that_verdict])=='high' (currently 'none'); also assert a clean verdict still yields 'none'. Run tests and py_compile.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-68] Add boundary and false-positive tests for the black-wedge POV detector

- **Priority:** P1 · **Effort:** M · **Dimension:** Visual QC rubrics
- **Goal:** Lock the 0.38/0.46 edge thresholds, prove center-dark passes vs edge-wedge fails, and cover the unreadable-file FAIL path.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** python3 -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k 'pov_seed_frame_quality' -q  &&  python3 -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py

- **Context:** _pov_seed_frame_quality is the cheap pixel-statistics guard for the 'open the refrigerator' POV seed: it must distinguish a clipped near-field body part (edge wedge) from a legitimately dark fridge interior (center). The magic constants 0.38/0.46 have no boundary tests, so any tweak silently changes sensitivity. The detector returns max_edge_dark_fraction / max_lower_edge_dark_fraction (lines 1569-1570), which tests can assert against to pin the thresholds without hardcoding pixel math.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add hermetic coverage for the black-wedge / edge self-occlusion detector.

Target: scripts/run_isaac_g1_kitchen_parity_eval.py, _pov_seed_frame_quality (lines 1523-1575). It flags 'manipulation_pov_edge_self_occlusion' when edge_dark > 0.38 OR lower_edge_dark > 0.46, using a 16% edge band (line 1541) and a lower-region cutoff at 45% height (line 1542). Dark task objects centered in frame are allowed; edge occlusion is not. The only existing test (tests/test_isaac_g1_kitchen_parity_runner.py:1897, test_pov_seed_frame_quality_rejects_black_edge_occlusion) covers clean vs fully-black-right-edge.

This task is TEST-ONLY: do not change _pov_seed_frame_quality behavior unless a test proves an actual bug (if so, report it and fix minimally). Add PIL-synthesized tests (PIL is already used by the existing test) in tests/test_isaac_g1_kitchen_parity_runner.py via the module loaded as M:
  (a) a legitimately dark TASK object filling the center 40% of the frame -> status PASS (proves center-dark is allowed, like a refrigerator interior);
  (b) a left-edge black band wider than the 16% edge band -> status FAIL with 'manipulation_pov_edge_self_occlusion';
  (c) two boundary cases: an edge fill just BELOW 0.38 dark fraction -> PASS, and just ABOVE 0.38 -> FAIL (size the black region against the edge-band geometry so the computed dark_fraction straddles 0.38; assert via report['max_edge_dark_fraction']);
  (d) a non-image file path (write some text bytes to a .png) -> status FAIL with 'manipulation_pov_frame_unreadable' (lines 1533-1539).

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; tests must be deterministic and CPU-only. Run the tests and py_compile.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-69] Test manipulation rubric gripper-key alias and canonical precedence

- **Priority:** P1 · **Effort:** S · **Dimension:** Visual QC rubrics
- **Goal:** Pin the gripper_or_hand_visible / robot_arm_or_hand_visible alias fallback so a dropped canonical key cannot fail open.
- **Files:** `src/blueprint_pipeline/render_visual_qc.py`, `tests/test_render_visual_qc.py`
- **Validate (CPU):** python3 -m pytest tests/test_render_visual_qc.py -k 'manipulation_pov_verdict' -q  &&  python3 -m py_compile src/blueprint_pipeline/render_visual_qc.py

- **Context:** The arm-visible-beyond-gripper gate is the central correctness rule for the 'open the refrigerator' POV seed: it rejects an isolated fingertip with no arm context (see the prompt at render_visual_qc.py:116-144 and the fail-closed conjunction at line 305). An untested key-aliasing branch is precisely where a silent fail-open creeps in if the canonical key is renamed or dropped.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add regression tests for the manipulation-POV gripper-key alias.

Target: src/blueprint_pipeline/render_visual_qc.py, parse_manipulation_pov_verdict (lines 279-313). Gripper visibility is read at lines 296-299 via obj.get('gripper_or_hand_visible', obj.get('robot_arm_or_hand_visible')) — a legacy/alias key. The existing test (tests/test_render_visual_qc.py:232, test_parse_manipulation_pov_verdict_fails_closed) only exercises the canonical key. The alias branch — the only thing preventing a fail-closed false-negative if a model or older prompt emits robot_arm_or_hand_visible — is untested.

This is TEST-ONLY. Add tests in tests/test_render_visual_qc.py:
  (a) a reply that supplies ONLY robot_arm_or_hand_visible=true (no gripper_or_hand_visible) plus the other pass fields (pass/target_visible/robot_arm_visible_beyond_gripper/arm_reaching_target/not_mostly_dark_or_occluded all true) -> assert parsed True, gripper_or_hand_visible resolves True, and passed True;
  (b) a conflicting reply with gripper_or_hand_visible=false AND robot_arm_or_hand_visible=true -> assert the CANONICAL key wins (gripper_or_hand_visible resolves False, passed False), documenting precedence;
  (c) a reply missing BOTH keys -> assert gripper_or_hand_visible is False (fail-closed default at line 299) and passed False.
Note dict.get evaluates the second arg eagerly but that is harmless; just pin behavior. Do not change source unless a test reveals a real defect (then report and fix minimally).

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims. Run the tests and py_compile.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-70] Cover empty-input behavior of qc_manipulation_pov_frames and qc_render_frames

- **Priority:** P1 · **Effort:** S · **Dimension:** Visual QC rubrics
- **Goal:** Pin that an empty manipulation set blocks, and surface/fix the generic empty-set returning 'not flagged'.
- **Files:** `src/blueprint_pipeline/render_visual_qc.py`, `tests/test_render_visual_qc.py`
- **Validate (CPU):** python3 -m pytest tests/test_render_visual_qc.py -q  &&  python3 -m py_compile src/blueprint_pipeline/render_visual_qc.py

- **Context:** Empty-input is where gates leak: a render step that produced no frames should block, but the generic qc_render_frames returns a clean RenderQCReport while placement/manipulation guard against it (lines 540-541, 575-576). On the G1 refrigerator lane the generic rubric (qc_render_frames / qc_render_output_dir, lines 594-600) glob-finds robot_pov_*.png; a glob that matches nothing currently passes clean. Existing tests use _CLEAN_REPLY etc. defined at the top of tests/test_render_visual_qc.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), pin and harden empty-input behavior across the visual-QC aggregators.

File: src/blueprint_pipeline/render_visual_qc.py.
  1) qc_manipulation_pov_frames (lines 559-591) emits 'manipulation_pov_visual_qc_no_frames' + status='blocked' on empty input, but no test exercises the empty-list call directly. Add a test: qc_manipulation_pov_frames([], 'refrigerator', task_description='open the refrigerator', generate=lambda b,p: '') -> assert status=='blocked', frames_reviewed==0, and 'manipulation_pov_visual_qc_no_frames' in blockers.
  2) qc_render_frames (lines 500-518) with an empty list returns flagged=False (any([]) is False) — an empty render set is reported NOT flagged / clean. Given the module otherwise fails closed, this is a latent fail-open. Add a no-frames guard: when sample_frame_paths returns empty, set flagged=True and record an anomaly/marker (e.g. anomalies=[{'frame': None, 'category': 'other', 'description': 'no_frames', 'severity': 'high'}]) so an empty generic-QC set is flagged; keep worst_severity consistent. Update/extend tests in tests/test_render_visual_qc.py: qc_render_frames([], task, generate=lambda b,p: _CLEAN_REPLY) -> assert frames_reviewed==0 and flagged is True. Ensure the existing non-empty tests still pass.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims. Run the tests and py_compile.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-71] Test runner visual-QC import-failure and cross-rubric fail-closed propagation

- **Priority:** P1 · **Effort:** M · **Dimension:** Visual QC rubrics
- **Goal:** Cover the fail-closed import-error branches and placement/POV blocker propagation in _run_placement_visual_qc and _run_task_visual_qc.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** python3 -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k 'visual_qc' -q  &&  python3 -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py

- **Context:** The import-failure branch is a deliberate fail-closed safety net (if QC can't load, block). An untested net can silently invert under refactor and never be caught. Cross-rubric propagation is the core aggregation contract of the combined task gate for the 'open the refrigerator' G1 lane. The wrappers try `from render_visual_qc import ...` first (script-dir import) then fall back to blueprint_pipeline.render_visual_qc (lines 4113-4116, 4144-4153).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add hermetic tests for the runner-side visual-QC wrappers.

File: scripts/run_isaac_g1_kitchen_parity_eval.py. Both wrappers wrap the render_visual_qc import in try/except and fail closed: _run_placement_visual_qc (lines 4106-4133) returns status='blocked' with 'placement_visual_qc_import_failed' on import error; _run_task_visual_qc (lines 4136-4199) returns status='blocked' with 'visual_qc_import_failed' on import error. Only the happy path of _run_task_visual_qc is tested (tests/test_isaac_g1_kitchen_parity_runner.py:403). The import-failure branches and the cross-rubric blocker propagation are untested.

This is TEST-ONLY (do not change runner behavior unless a test reveals a real defect; then report + fix minimally). The runner module is loaded as `M` via importlib at tests/test_isaac_g1_kitchen_parity_runner.py:16-26; the QC functions are imported INSIDE the wrappers from blueprint_pipeline.render_visual_qc, so monkeypatch that module's attributes (as the existing test does at lines 431-432). Add tests:
  (a) placement-blocked: monkeypatch qc_robot_placement_frames to return status='blocked' with blockers=['placement_visual_qc_failed']; call M._run_task_visual_qc([verify],[pov],...) with qc_manipulation_pov_frames patched to 'passed'; assert combined status 'blocked' and the placement blocker propagates.
  (b) pov-blocked/placement-passed: placement 'passed', qc_manipulation_pov_frames 'blocked' -> assert combined 'blocked' with the pov blocker.
  (c) import-failure: monkeypatch the import to raise (e.g. set sys.modules['blueprint_pipeline.render_visual_qc'] to a module object whose attribute access raises, or use monkeypatch on builtins/import — simplest: insert a fake module into sys.modules that lacks the names and ALSO break the `from render_visual_qc import ...` path) and assert M._run_task_visual_qc(...) returns status='blocked' with 'visual_qc_import_failed'; do the same for M._run_placement_visual_qc asserting 'placement_visual_qc_import_failed'. If reliably forcing the import error is awkward, at minimum cover (a) and (b) and add a focused test for the import-failure return shape by calling the wrapper with the module temporarily removed from sys.modules and the script dir off sys.path.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims. Run the tests and py_compile.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-72] Add objective pixel cross-check for VLM-self-reported dark_region_fraction

- **Priority:** P1 · **Effort:** M · **Dimension:** Visual QC rubrics
- **Goal:** Compute a CPU dark-pixel fraction and take max(model, measured) before flagging so a model can't underreport a black frame.
- **Files:** `src/blueprint_pipeline/render_visual_qc.py`, `tests/test_render_visual_qc.py`
- **Validate (CPU):** python3 -m pytest tests/test_render_visual_qc.py -k 'flag or dark or review' -q  &&  python3 -m py_compile src/blueprint_pipeline/render_visual_qc.py

- **Context:** The 'void / under-lit basin' anomaly is a named rubric target (build_qc_prompt, render_visual_qc.py:54-86) for the refrigerator lane (a dark fridge interior vs an actually-broken black frame). Relying solely on the VLM's self-reported dark fraction means the same model that produced a bad frame can clear its own review. The runner already computes pixel dark fractions for the POV seed path (_pov_seed_frame_quality) but that signal is not fed back into the generic parse_qc_verdict path.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), close a self-reporting loophole in the generic dark-region check.

File: src/blueprint_pipeline/render_visual_qc.py. verdict_is_flagged compares verdict['dark_region_fraction'] >= DEFAULT_DARK_REGION_FLOOR (0.30) (lines 330, 40), but that fraction is whatever the VLM self-reports (_as_fraction clamps to [0,1], lines 200-205). There is NO objective pixel cross-check in the generic rubric, so a model reporting dark_region_fraction=0.0 on an actually-black frame passes. A cheap numpy/PIL dark-pixel measurement closes that with zero GPU cost (PIL/luma histogram math already exists in the runner at scripts/run_isaac_g1_kitchen_parity_eval.py:1507-1520, _fraction_from_histogram / _image_luma_extreme_fractions — mirror that approach here, do not import from the runner).

Implement: add a small helper (e.g. measured_dark_fraction(image_bytes)) that loads bytes via PIL as 'L' and returns the fraction of near-black pixels (luma in roughly [0,13], matching the runner's range(0,14)). In review_render_frame (lines 404-428), after parsing, set verdict['dark_region_fraction'] = max(parsed_value, measured) when image bytes are available; guard so missing PIL or undecodable bytes degrades gracefully (leave the model value, never crash — a decode failure should not silently clean the frame). Keep the floor at 0.30. Do not change placement/manipulation parsers.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; keep the module import-light (PIL import lazy/optional). Add tests in tests/test_render_visual_qc.py: (1) a boundary test asserting a verdict with dark_region_fraction exactly 0.30 flags; (2) build a synthetic all-black PNG via PIL, inject generate=lambda b,p: a reply with dark_region_fraction=0.0, call review_render_frame(black_png_bytes, task, generate=...), and assert flagged is True because measured override raised the fraction; (3) a clean mid-gray PNG with a clean reply stays flagged False. Run tests and py_compile.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## Catch-all / completeness

### [P1-73] Add a ruff lint gate to CI and clear the 25 existing violations

- **Priority:** P1 · **Effort:** S · **Dimension:** Catch-all / completeness
- **Goal:** Wire `ruff check` into .github/workflows/ci.yml and fix the 25 currently-ungated lint errors so dead imports/style drift can't accumulate.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/.github/workflows/ci.yml`, `$HOME/workspace/BlueprintCapturePipeline/pyproject.toml`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_simulator_beta_readiness.py`
- **Validate (CPU):** ruff check src tests scripts (exit 0, zero errors); grep 'ruff' .github/workflows/ci.yml shows the new step; python -m pytest -q -o addopts='' still passes after the fixes.

- **Context:** Lint is configured but unenforced, so unused imports and style violations accumulate silently. Adding ruff to CI is the single highest-leverage CPU-only quality gate available and costs nothing. Verified: ci.yml has only checkout/install-uv/uv-sync/pytest steps; ruff reports 'Found 25 errors. [*] 12 fixable'.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), CI (.github/workflows/ci.yml) runs only `uv run pytest -q` and never invokes ruff, even though ruff>=0.6.0 is a declared dev dependency and [tool.ruff] is configured in pyproject.toml. Verified live: `.venv/bin/ruff check src tests scripts` reports 25 errors (12 auto-fixable): 12x F401 unused-import, 6x E702 multiple-statements-semicolon, 3x F841 unused-variable, 2x E731 lambda-assignment, 2x E741 ambiguous-variable-name. Example: an unused `_isaac_gate` import in tests/test_simulator_beta_readiness.py.

Do this:
1. Run `ruff check --fix src tests scripts` to clear the 12 auto-fixable issues, then manually fix the remaining ~13 (rename/remove ambiguous vars, split semicolon statements, convert lambda-assignments to defs, drop unused imports). Do NOT suppress with blanket `# noqa` unless a specific line genuinely must keep the construct, in which case use a targeted `# noqa: <code>` with a reason.
2. Re-run `ruff check src tests scripts` and confirm zero errors.
3. Add a `Run ruff` step to ci.yml (after Install dependencies, before/after Run tests): `uv run ruff check src tests scripts`.
4. Confirm the test suite still passes after the fixes (a removed 'unused' import could be a real import side-effect — verify, don't assume).

Constraints: keep world-model backends swappable (don't delete imports that register a backend via side-effect — verify each F401 is truly unused); protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; extend tests if a fix changes behavior. Run `python -m pytest -q -o addopts=''` and `python -m py_compile` on edited files.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-74] Commit uv.lock and switch CI sync to --frozen for reproducibility

- **Priority:** P1 · **Effort:** S · **Dimension:** Catch-all / completeness
- **Goal:** Track the 695KB uv.lock and have CI run `uv sync --frozen --extra dev` so the CPU test environment is deterministic and can't silently drift.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/.gitignore`, `$HOME/workspace/BlueprintCapturePipeline/uv.lock`, `$HOME/workspace/BlueprintCapturePipeline/.github/workflows/ci.yml`
- **Validate (CPU):** grep -c 'uv.lock' .gitignore returns 0; git ls-files | grep -x uv.lock returns uv.lock; grep 'frozen\|locked' .github/workflows/ci.yml shows the new flag; uv sync --frozen --extra dev resolves with no errors; python -m pytest -q -o addopts='' passes.

- **Context:** Verified: `grep -n uv.lock .gitignore` -> line 81; uv.lock exists on disk (695157 bytes) but is untracked; ci.yml uses `uv sync --extra dev`. Committing the lock + `--frozen` makes the CPU-only test environment deterministic at zero cost and prevents transitive-bump drift between local and CI. Note ordering: if pyproject extras change in the venv-sync/usd-core tasks, regenerate uv.lock so the committed lock reflects them.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), uv.lock (695KB, present in the working tree) is gitignored (.gitignore line 81) so it is untracked, and CI runs `uv sync --extra dev` (.github/workflows/ci.yml) with no committed lockfile — every run re-resolves dependencies and is non-reproducible. This is the exact class of drift that produced the trimesh-missing failure.

Do this:
1. Remove the `uv.lock` line from .gitignore.
2. `git add uv.lock`.
3. Change the CI dependency step to `uv sync --frozen --extra dev` (or `--locked`) in ci.yml so the resolved environment matches the committed lock.
4. Run `uv sync --frozen --extra dev` locally to confirm the lock resolves cleanly (regenerate the lock first with `uv lock` if it is stale relative to pyproject — e.g. after the trimesh/usd-core/scipy additions — then commit the regenerated lock).

Constraints: keep world-model backends swappable (don't pin away optional backend extras); protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims. No test changes expected, but run `python -m pytest -q -o addopts=''` to confirm the frozen env still produces a green suite.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P1-75] Fix .gcloudignore to exclude 15GB+ of run artifacts and node_modules from deploys

- **Priority:** P1 · **Effort:** S · **Dimension:** Catch-all / completeness
- **Goal:** Add output/, robot_eval_jobs/, policy_endpoint_setups/, .local_runs/, local_runs_worldlabs/, and tools/splat_render/node_modules/ to .gcloudignore so a gcloud deploy doesn't upload large run data (which is provenance/raw-capture-adjacent).
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/.gcloudignore`, `$HOME/workspace/BlueprintCapturePipeline/.gitignore`, `$HOME/workspace/BlueprintCapturePipeline/main.py`, `$HOME/workspace/BlueprintCapturePipeline/functions/storage_trigger.py`
- **Validate (CPU):** For each of output/ robot_eval_jobs/ policy_endpoint_setups/ .local_runs/ tools/splat_render/node_modules/ : `grep -F '<path>' .gcloudignore` returns a match; optionally `gcloud meta list-files-for-upload . | grep -E 'output/|robot_eval_jobs/|node_modules/'` returns nothing; python -m py_compile main.py functions/storage_trigger.py succeeds.

- **Context:** Verified .gcloudignore contents: it lists .git, .venv, docs/, skillpacks/, tests/, local_runs_worldlabs/, local_runs/, runs/ — but omits output/, robot_eval_jobs/, policy_endpoint_setups/, .local_runs/, and tools/splat_render/node_modules/. An accidental gcloud deploy uploading 15GB+ of binary run artifacts is slow, may fail, and could push raw-capture-adjacent run data into a deploy bundle. The dirs are already gitignored; .gcloudignore should mirror that.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), .gcloudignore excludes tests/, docs/, skillpacks/, runs/, local_runs/, local_runs_worldlabs/ etc., but does NOT exclude output/ (~12GB, 128 run dirs), robot_eval_jobs/ (~2.8GB, 196 dirs), policy_endpoint_setups/, .local_runs/, or tools/splat_render/node_modules/. main.py is a Cloud Functions entrypoint (functions/storage_trigger.py), so `gcloud functions deploy` / `gcloud app deploy` from this root would attempt to upload all of it.

Do this:
1. Add these paths to .gcloudignore: `output/`, `robot_eval_jobs/`, `policy_endpoint_setups/`, `.local_runs/`, and `tools/splat_render/node_modules/`. (local_runs_worldlabs/ and local_runs/ and runs/ are already present — verify and don't duplicate.)
2. Run `gcloud meta list-files-for-upload .` (DRY — no deploy) and confirm none of those paths appear in the upload manifest. If gcloud is not installed/authenticated, instead grep the new .gcloudignore to confirm each path is listed and explain that the dry manifest check requires gcloud.

Constraints: mirror .gitignore intent (these dirs are already gitignored); protect provenance/rights/privacy/raw-capture-truth — these run dirs contain capture-adjacent outputs that must NOT be swept into a deploy bundle; render outputs are simulator support, NOT policy-success claims. This is an ignore-file edit only; no tests change. Run `python -m py_compile main.py functions/storage_trigger.py` to confirm the entrypoints still import-compile.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only (do NOT run an actual gcloud deploy).
```

</details>

---

# P2 — Cleanup / nice-to-have

## Test suite health

### [P2-01] Fix the webapp_sync parametrize KeyError collection error

- **Priority:** P2 · **Effort:** S · **Dimension:** Test suite health
- **Goal:** Resolve the lone `KeyError: 'file'` collection error in tests/test_webapp_sync.py so the whole suite collects cleanly.
- **Files:** `tests/test_webapp_sync.py`
- **Validate (CPU):** python3 -m pytest -p no:cacheprovider --co -q tests/test_webapp_sync.py 2>&1 | grep -c 'KeyError'  (expect 0) ; python3 -m py_compile tests/test_webapp_sync.py

- **Context:** CPU test-suite health, lowest-frequency error (1 of 76). Unlike the PIL/version buckets, this is a pytest parametrize id-generation crash specific to tests/test_webapp_sync.py, not a missing dependency. Fixing it removes the last non-dependency collection error and gets the suite to a fully clean collect on CPU.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), `python3 -m pytest --co tests/test_webapp_sync.py` fails collection with `KeyError: 'file'` raised inside pytest's `_pytest/python.py` parametrize machinery (`_resolve_parameter_set_ids` -> `make_unique_parameterset_ids` -> `file = self.__dict__['file']`). The repo uses `@pytest.mark.parametrize` at tests/test_webapp_sync.py lines ~285 and ~507. This is a pytest-internal id-generation crash, typically triggered when a parametrize `ids=` callable or an argvalue object lacks expected metadata, or when a custom collector/object is passed as an argvalue.

Task: Investigate `python3 -m pytest --co tests/test_webapp_sync.py 2>&1` to get the full traceback, then inspect the two parametrize decorators. Most likely fixes: provide explicit string `ids=[...]` for the offending parametrize so pytest does not try to derive ids from objects that lack a `file` attribute, OR replace any non-trivial argvalue objects with `pytest.param(value, id="...")`. Apply the minimal change that makes collection succeed without altering which cases run or their semantics. Confirm whether this reproduces only on the local pytest 8.4.2 / Python 3.9 combo; if it is purely a stale-cache artifact, clearing is not a code fix — the test must collect cleanly from a fresh `-p no:cacheprovider` run.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support NOT policy-success claims; do not drop or merge parametrized cases. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: `python3 -m pytest -p no:cacheprovider --co -q tests/test_webapp_sync.py 2>&1 | grep -c 'KeyError'` prints 0; `python3 -m pytest tests/test_webapp_sync.py -q` collects and runs (skips allowed if it needs absent optional deps, but no collection ERROR); `python3 -m py_compile tests/test_webapp_sync.py`.
```

</details>

## Isaac G1 render — CPU logic

### [P2-02] Test verify-cam 3/4 framing keeps robot and target in frame

- **Priority:** P2 · **Effort:** S · **Dimension:** Isaac G1 render — CPU logic
- **Goal:** Ensure the independent 3rd-person placement-QC camera frames both robot and affordance at a genuine 3/4 angle.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k verify_cam -q  (pure-math projection, no GPU). python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py.

- **Context:** The verify cam is the human reviewer's independent ground-truth check that placement is correct in the active 'open the refrigerator' G1 lane; recent commits added 'verify from side' framing. If its eye/target math drifts so the robot or workspace falls out of frame, reviewers lose the one independent visual placement signal. scripts/run_isaac_g1_kitchen_parity_eval.py verify_cam_pose line 1009 (side offset line 1024); only weakly tested at tests/test_isaac_g1_kitchen_parity_runner.py:1923.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), strengthen the test for `verify_cam_pose` (scripts/run_isaac_g1_kitchen_parity_eval.py line 1009), the 3rd-person 'prove where the robot stands' camera. The existing test (tests/test_isaac_g1_kitchen_parity_runner.py:1923 `test_verify_cam_pose_is_behind_robot_for_visual_placement_qc`) only checks the eye is behind+above the root and offset to the side. Add a test that, given a root_pose, yaw, and look_at, both (a) the robot torso point AND (b) the affordance/look_at project INSIDE the frame via `project_point_to_pixel` at the verify-cam vfov, and (c) the perpendicular side offset (the `side` param / line 1024 logic) yields a genuine 3/4 view — i.e. the eye is NOT collinear with the root→target ray (nonzero perpendicular component). This catches a verify cam that frames only the robot or only the target, or collapses to a straight-behind shot. Pure-math projection — no pxr/PIL/GPU. Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; add/extend tests only. Run `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k verify_cam -q` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P2-03] Test software denoise PIL fallback and _save_rgb denoise routing

- **Priority:** P2 · **Effort:** M · **Dimension:** Isaac G1 render — CPU logic
- **Goal:** Guarantee saved review frames still denoise when cv2 is absent, via the PIL MedianFilter+SMOOTH_MORE fallback.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k denoise -q  (importorskip PIL; monkeypatch to force cv2-absent fallback). python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py.

- **Context:** software_denoise defaults to True for saved frames (scripts/run_isaac_g1_kitchen_parity_eval.py lines ~5611 and ~7222) and is the only grain mitigation when a pod lacks RTX/NGX denoising (single-step RayTracedLighting leaves heavy noise). In the active 'open the refrigerator' review lane, if the cv2 path silently errors and the PIL fallback regresses, every saved review frame degrades. cv2 is confirmed absent in the CPU env, so the PIL fallback (lines 5075-5082) is the production path locally. Pure CPU image processing — fully validatable offline.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add a deterministic test for `_software_denoise_image` (scripts/run_isaac_g1_kitchen_parity_eval.py line 5063) and the denoise routing in `_save_rgb` (line 5087). The function uses `cv2.fastNlMeansDenoisingColored` when cv2 is present and falls back to the PIL `MedianFilter`+`SMOOTH_MORE` chain (lines 5075-5082) when absent. Test the PIL fallback specifically (cv2 is NOT installed in the CPU env, so the fallback is the live path): gate with `pytest.importorskip('PIL')`, and to be robust also force the fallback by monkeypatching the cv2 import to fail (so the test is deterministic whether or not cv2 is later installed). Assert: (a) output size and mode ('RGB') are preserved; (b) a synthetic salt-and-pepper input has fewer extreme (near-0 / near-255) pixels after denoise than before; (c) `_save_rgb(annot, out_path, software_denoise=True)` writes a PNG file and routes through the denoiser (e.g. verify the output differs from a `software_denoise=False` save of the same noisy input). Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; add/extend tests only. Run `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -k denoise -q` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## Main 11-stage pipeline

### [P2-04] Add real-library (trimesh/scipy) skippable coverage for object_geometry mesh + collision-hull

- **Priority:** P2 · **Effort:** M · **Dimension:** Main 11-stage pipeline
- **Goal:** Validate object_geometry's real geometry math against actual trimesh/scipy, not just the hand-rolled fake stub.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/object_geometry_stage.py`, `$HOME/workspace/BlueprintCapturePipeline/pyproject.toml`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_object_geometry_stage_coverage_edges.py`
- **Validate (CPU):** .venv/bin/pip install trimesh scipy && .venv/bin/python -m py_compile src/blueprint_pipeline/object_geometry_stage.py && .venv/bin/python -m pytest tests/test_object_geometry_stage_coverage_edges.py -q

- **Context:** All collision geometry and support-surface inference in src/blueprint_pipeline/object_geometry_stage.py is pure-CPU math currently unvalidated against the real library locally. With GPU/splat paused for the active 'open the refrigerator' G1 lane, every kitchen asset's collision geometry flows through this code, so a real-vs-fake divergence would only surface on a GPU image. pyproject.toml gets a dev/test extra; test file: tests/test_object_geometry_stage_coverage_edges.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add real-library coverage for object_geometry_stage geometry code so it is validated against actual trimesh/scipy, not only the fake stub.

Background: trimesh and scipy are NOT installed in .venv, so src/blueprint_pipeline/object_geometry_stage.py's real geometry code — _load_mesh_or_points (marching-cubes/convex-hull ~484-495), _collision_hull_meshes (~571+, with _kmeans2 splitting ~549-568), _support_surfaces, _normalize_mesh_to_local — is exercised ONLY through a hand-rolled fake trimesh in tests/test_object_geometry_stage_coverage_edges.py (test_object_geometry_fake_trimesh_branches). The fake can diverge from real trimesh semantics, and such a divergence would only surface on a GPU image, defeating CPU-first validation.

Task:
1. Add trimesh and scipy to the dev/test optional dependencies in pyproject.toml (a [project.optional-dependencies] dev/test extra — do NOT add them as hard runtime deps). Confirm they install on CPU with no GPU/CUDA wheels (these are pure-CPU libs).
2. Read _load_mesh_or_points (~484-495), _collision_hull_meshes / _kmeans2 (~549-568, 571+), _support_surfaces, and _normalize_mesh_to_local to understand inputs/outputs.
3. Add a test in tests/test_object_geometry_stage_coverage_edges.py guarded by @pytest.mark.skipif(trimesh is None or scipy is None, ...) that runs the stage (or the targeted helpers) against a small REAL point cloud / known box mesh, exports an actual GLB, and asserts plausible hull counts and support-surface counts for that known box. Make the test deterministic (fixed input geometry).
4. Keep the existing fake-trimesh test intact (it must still run when the libs are absent). Do not change production geometry behavior unless a real-vs-fake divergence reveals a bug, then fix minimally and note it.
5. Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render/eval outputs are simulator support, not policy-success claims. Run validation and ensure it passes.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only (trimesh and scipy are free CPU-only pip installs).
```

</details>

### [P2-05] Correct stale module references in MEMORY.md

- **Priority:** P2 · **Effort:** S · **Dimension:** Main 11-stage pipeline
- **Goal:** Replace non-existent swap_orchestrator.py / nurec_worker.py entries with the real orchestrator and geometry modules.
- **Files:** `$HOME/.claude/projects/-Users-example-workspace-BlueprintCapturePipeline/memory/MEMORY.md`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/capture_orchestrator.py`
- **Validate (CPU):** ls src/blueprint_pipeline/swap_orchestrator.py src/blueprint_pipeline/nurec_worker.py 2>&1 | grep -q 'No such file' && grep -n 'def run_capture_pipeline' src/blueprint_pipeline/capture_orchestrator.py && grep -n 'capture_orchestrator.py' "$HOME/.claude/projects/-Users-example-workspace-BlueprintCapturePipeline/memory/MEMORY.md"

- **Context:** Accurate entry-point docs are a prerequisite for reliable CPU-first validation. MEMORY.md currently points reviewers at non-existent files (swap_orchestrator.py / nurec_worker.py), which already misdirected this audit pass and risks tests being written against the wrong module. The real orchestrator is src/blueprint_pipeline/capture_orchestrator.py:run_capture_pipeline. File to edit: `$HOME/.claude/projects/-Users-example-workspace-BlueprintCapturePipeline/memory/MEMORY.md`.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), correct documentation drift in the project memory file.

Background: The memory file at `$HOME/.claude/projects/-Users-example-workspace-BlueprintCapturePipeline/memory/MEMORY.md` has a 'Key Files' section pointing to src/blueprint_pipeline/swap_orchestrator.py ('Main orchestrator') and src/blueprint_pipeline/nurec_worker.py ('NuRec reconstruction worker'). Neither file exists (verified: `ls` returns ENOENT for both). The real orchestrator is src/blueprint_pipeline/capture_orchestrator.py:run_capture_pipeline, and NuRec/geometry is handled by src/blueprint_pipeline/geometry_stage.py + src/blueprint_pipeline/geometry_sources.py (plus a synthesis/ package). This drift misdirects every audit/onboarding pass.

Task:
1. Confirm the facts: `ls src/blueprint_pipeline/swap_orchestrator.py src/blueprint_pipeline/nurec_worker.py` (both ENOENT) and `grep -n 'def run_capture_pipeline' src/blueprint_pipeline/capture_orchestrator.py` (present). Also confirm geometry_stage.py and geometry_sources.py exist.
2. Edit MEMORY.md 'Key Files': replace the swap_orchestrator.py line with `src/blueprint_pipeline/capture_orchestrator.py - Main orchestrator (run_capture_pipeline)` and replace the nurec_worker.py line with `src/blueprint_pipeline/geometry_stage.py + geometry_sources.py - NuRec/geometry reconstruction`. Leave the rest of MEMORY.md unchanged. Keep accurate, file-grounded wording.
3. This is a documentation-only change; no code or tests change. Do not invent modules — only reference files you verified exist.
4. Constraints: protect provenance/rights/privacy/raw-capture-truth (do not add speculative claims); keep the note factual.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## scene_placement package

### [P2-06] Add ambiguity diagnostic to resolve_target_by_label on equal-rank ties

- **Priority:** P2 · **Effort:** S · **Dimension:** scene_placement package
- **Goal:** When several objects match the winning intent token equally well, emit an ambiguity flag/note (and allow an optional disambiguator) instead of silently returning the sort-stable first.
- **Files:** `src/blueprint_pipeline/scene_placement/target_resolver.py`, `tests/test_scene_placement.py`
- **Validate (CPU):** python -m pytest tests/test_scene_placement.py -k 'label_fallback or ambig' -q ; python -m pytest tests/test_scene_placement.py -q ; python -m py_compile src/blueprint_pipeline/scene_placement/target_resolver.py

- **Context:** Authored and perception scenes routinely contain duplicate same-label fixtures (two sinks, a row of burners). A silent arbitrary pick can place the robot at the wrong instance with no trace. Confidence is medium because the exact surfacing mechanism is a design choice — pin it. Files: src/blueprint_pipeline/scene_placement/target_resolver.py (resolve_target_by_label ~345-371, build_target_prompt centroid note ~134-170), tests/test_scene_placement.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), give resolve_target_by_label in src/blueprint_pipeline/scene_placement/target_resolver.py a diagnostic when its pick is arbitrary.

Current state: when multiple objects share the best (rank, label-length, label) tuple for the winning intent token (e.g. two prims both labeled 'faucet' at different centroids), resolve_target_by_label returns ranked[0] purely by sort stability with no signal that the choice was arbitrary. The VLM-disambiguation-by-centroid promise in build_target_prompt is never exercised on this fallback path.

What to do (keep the single-SceneObject return type stable):
1. Detect the tie: after sorting, check whether more than one candidate shares the same winning (rank, len(label), label) key. If so, mark ambiguity.
2. Surface it without breaking callers — pick ONE: (a) accept an optional disambiguator centroid arg (resolve_target_by_label(task, objects, *, near_xy=None)) and, when present, break ties by nearest centroid to near_xy deterministically; AND/OR (b) expose the ambiguity via a thin companion helper (e.g. resolve_target_by_label_with_diagnostics(task, objects) -> (obj, note)) so the existing function stays drop-in. The winner must remain DETERMINISTIC (define the tie-break: nearest to near_xy if given, else current stable order) and you must be able to assert which one wins.
3. Document the behavior in the docstring.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render/placement outputs are simulator support, NOT policy-success claims; add/extend tests; run `python -m pytest tests/test_scene_placement.py` and `python -m py_compile src/blueprint_pipeline/scene_placement/target_resolver.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Add a test in tests/test_scene_placement.py (label_fallback area): two objects labeled 'faucet' at distinct centroids -> assert a deterministic winner and that the ambiguity flag/note is emitted; with a near_xy disambiguator the nearer one wins; a non-ambiguous single match emits no ambiguity flag. Pure synthetic objects, no GPU.
```

</details>

### [P2-07] Model a target-height/swing-aware close-reach envelope for standoff

- **Priority:** P2 · **Effort:** M · **Dimension:** scene_placement package
- **Goal:** Replace the flat standoff band with a reach envelope keyed off target min_z/max_z (and an openable swing arc) so low drawers, counter faucets, and fridge doors get appropriate bands.
- **Files:** `src/blueprint_pipeline/scene_placement/placement.py`, `src/blueprint_pipeline/scene_placement/validation.py`, `src/blueprint_pipeline/scene_placement/types.py`, `tests/test_scene_placement.py`, `tests/test_placement_validation.py`
- **Validate (CPU):** python -m pytest tests/test_scene_placement.py -k standoff -q ; python -m pytest tests/test_placement_validation.py -k standoff -q ; python -m py_compile src/blueprint_pipeline/scene_placement/placement.py src/blueprint_pipeline/scene_placement/validation.py

- **Context:** The G1 'open the refrigerator' task needs the pelvis far enough back that the swung door does not intersect the footprint, while faucet/drawer tasks have very different reach geometry. A flat standoff band either parks the robot inside the door swing or too far to reach a low handle — both render wrong and both are CPU-checkable from the AABB. Confidence medium: the exact reach constants are a modeling choice to be pinned by tests. Files: src/blueprint_pipeline/scene_placement/placement.py (compute_stand_pose ~148-260, standing_distance), src/blueprint_pipeline/scene_placement/validation.py (validate_stand_pose standoff ~311-323, DEFAULT_VALIDATION_STANDOFF_RANGE ~47), src/blueprint_pipeline/scene_placement/types.py (min_z/max_z ~65-71), tests/test_scene_placement.py, tests/test_placement_validation.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), make standoff height/articulation aware in src/blueprint_pipeline/scene_placement/placement.py and src/blueprint_pipeline/scene_placement/validation.py.

Current state: placement uses a single standing_distance (default 0.55) and validation a single standoff_range (default 0.4-1.2) regardless of whether the target is a low drawer, a counter-height faucet, an overhead cabinet, or a swing-out fridge door. SceneObject already exposes min_z()/max_z().

What to do:
1. Add a reach-envelope helper that derives a close-reach standoff band from the target's min_z/max_z (e.g. a low target near the floor and a high target each get a band suited to a standing G1's reach, vs a counter-height target). Keep it a pure function returning (lo, hi) given min_z, max_z, pelvis_height and conservative reach constants (named, documented).
2. Wire it as an OPT-IN: compute_stand_pose and validate_stand_pose accept the height-aware band (or a flag enabling it) without changing default behavior unless requested, so existing tests stay green. When an openable swing arc is provided/known (coordinate with the openable-classification task), push the standoff LOWER bound out so the pelvis clears the door swing.
3. Document that this is geometry-derived reach approximation, not a guarantee the arm can reach.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render/placement outputs are simulator support, NOT policy-success claims; add/extend tests; run `python -m pytest tests/test_scene_placement.py tests/test_placement_validation.py` and `python -m py_compile` on touched files. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Add tests: in tests/test_scene_placement.py and tests/test_placement_validation.py (standoff area), a LOW target (min_z near floor) and a HIGH target (max_z high) yield different accepted standoff bands; a swing-arc-augmented openable target pushes the lower bound out so a too-close pose now fails. Pure synthetic AABBs, no GPU.
```

</details>

### [P2-08] Make multi-view fusion robust at even-count rings (odd-preferring merge)

- **Priority:** P2 · **Effort:** M · **Dimension:** scene_placement package
- **Goal:** Stop the default even (8) view ring from running fusion in its non-robust mean-drift regime; add an odd-preferring or confidence-weighted merge so a single off-view outlier is rejected.
- **Files:** `src/blueprint_pipeline/scene_placement/perception_fusion.py`, `src/blueprint_pipeline/scene_placement/perception_views.py`, `tests/test_perception_fusion.py`
- **Validate (CPU):** python -m pytest tests/test_perception_fusion.py -k 'median or outlier' -q ; python -m pytest tests/test_perception_fusion.py tests/test_perception_views.py -q ; python -m py_compile src/blueprint_pipeline/scene_placement/perception_fusion.py

- **Context:** The default view ring is even (8), so fusion runs in its non-robust regime precisely when multi-view was supposed to suppress a bad-depth outlier — the perception story sells robustness the default path does not deliver. Confidence medium: the exact robust estimator is a design choice to pin with tests. Files: src/blueprint_pipeline/scene_placement/perception_fusion.py (_median ~225, fuse_scene_objects merge ~240-260), src/blueprint_pipeline/scene_placement/perception_views.py (generate_view_ring n_azimuths default ~49-53), tests/test_perception_fusion.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), make perception_fusion deliver the outlier resistance it advertises on the DEFAULT view ring.

Current state: perception_fusion._median averages the two middle elements at even member counts, so a typical 4/6/8-azimuth ring (generate_view_ring default n_azimuths=8 in perception_views.py) makes the merged box drift toward the mean and a single off-view can drag it. This is honestly documented (module/_median docstrings; pinned by test_even_count_median_degenerates_toward_mean and test_two_member_median_equals_mean), but the default path runs in its non-robust regime exactly when multi-view was supposed to suppress a bad-depth outlier.

What to do:
1. Add an odd-preferring merge in perception_fusion: when fusing an even-count cluster, force an odd count for the median (e.g. drop the single lowest-confidence member before taking the per-axis median) OR switch to a confidence-weighted median. Keep _median itself available (tests reference it) but route the fusion merge through the new robust path. Preserve identity selection (id/label/category) behavior.
2. Make the behavior explicit and documented; if you drop a member, prefer the lowest-confidence one deterministically (tie-break stable).
3. Do not regress fuse_scene_objects for odd counts or the existing fusion tests beyond the intentionally-updated even-count case.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (dropping the weakest member for a robust estimate is a documented fusion choice, not data fabrication); render/placement outputs are simulator support, NOT policy-success claims; add/extend tests; run `python -m pytest tests/test_perception_fusion.py tests/test_perception_views.py` and `python -m py_compile src/blueprint_pipeline/scene_placement/perception_fusion.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Add a test in tests/test_perception_fusion.py (median/outlier area): a 4-member cluster with one clear outlier -> the odd-preferring merge rejects the outlier (merged box close to the 3 good members) where plain _median drifts toward the mean. Update test_even_count_median_degenerates_toward_mean to reflect that the FUSION path no longer degenerates (while _median in isolation may still document the averaging behavior). Pure synthetic boxes, no GPU.
```

</details>

### [P2-09] Flag degenerate (zero-size) view-ring bounds instead of a 1mm orbit

- **Priority:** P2 · **Effort:** S · **Dimension:** scene_placement package
- **Goal:** Stop _bounds_center_radius from silently producing a ~1mm orbit on a zero-size target; raise or apply a documented minimum radius with a caller-visible flag.
- **Files:** `src/blueprint_pipeline/scene_placement/perception_views.py`, `tests/test_perception_views.py`
- **Validate (CPU):** python -m pytest tests/test_perception_views.py -k 'bounds or validates' -q ; python -m pytest tests/test_perception_views.py -q ; python -m py_compile src/blueprint_pipeline/scene_placement/perception_views.py

- **Context:** A degenerate or single-detection target (common from one bad SAM3 box) would generate a 1mm orbit; every view is the same near-degenerate frame so fusion gets no diversity, silently defeating the multi-view premise. Confidence medium: raise-vs-minimum-radius is a design choice to pin. Files: src/blueprint_pipeline/scene_placement/perception_views.py (_bounds_center_radius ~29-46, view_ring_for_bounds ~100-130), tests/test_perception_views.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), fix the degenerate view-ring case in src/blueprint_pipeline/scene_placement/perception_views.py.

Current state: _bounds_center_radius clamps a zero-size box to radius=1e-3, so view_ring_for_bounds on a degenerate/single-detection target orbits at ~1mm with all eyes nearly coincident with the target — every view sees the same near-degenerate frame and fusion gets zero angular diversity, defeating the multi-view premise with no signal.

What to do:
1. In _bounds_center_radius / view_ring_for_bounds, detect a degenerate (half_diag ~ 0) bound and either (a) raise ValueError with an explicit message, or (b) apply a DOCUMENTED minimum sane radius (a named constant, clearly larger than 1e-3 — sized so cameras are not inside the object) AND surface a flag/warning the caller can see. Pick one and document it. Do not silently keep the 1e-3 orbit.
2. Keep non-degenerate behavior unchanged.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render/placement outputs are simulator support, NOT policy-success claims; add/extend tests; run `python -m pytest tests/test_perception_views.py` and `python -m py_compile src/blueprint_pipeline/scene_placement/perception_views.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Add a test in tests/test_perception_views.py (bounds/validates area): a zero-size bbox either raises or yields a ring at the documented minimum radius (assert radius and that eyes are not coincident with the center) plus the flag — not a 1e-3 orbit. A normal bbox still produces the expected radius. Pure, no GPU.
```

</details>

### [P2-10] Lock the Gemini response-parsing contract and flag the one paid step

- **Priority:** P2 · **Effort:** S · **Dimension:** scene_placement package
- **Goal:** Add hermetic contract tests over the target_resolver response-parsing surface and document that a live VLM call is the only paid validation step.
- **Files:** `src/blueprint_pipeline/scene_placement/target_resolver.py`, `tests/test_scene_placement.py`, `src/blueprint_pipeline/scene_placement/README.md`
- **Validate (CPU):** python -m pytest tests/test_scene_placement.py -k 'resolve_target or extract or gemini' -q ; python -m pytest tests/test_scene_placement.py -q ; python -m py_compile src/blueprint_pipeline/scene_placement/target_resolver.py ; confirm no network/Gemini call occurs in the default run (tests inject fakes; any live test is skip-by-default)

- **Context:** The resolver is the 'hinge of the dynamic-placement pipeline'; the model call is injected so logic is testable, but the parsing contract (thinking-part filtering per repo notes, cascade fallthrough, hallucinated-id rejection) is where real Gemini replies break and is fully exercisable with synthetic objects at zero cost. Repo memory notes thinking parts must be filtered and thinking_config must be omitted. Files: src/blueprint_pipeline/scene_placement/target_resolver.py (_extract_response_text ~197-220, _extract_json_object ~175-194, resolve_target ~266-299, _gemini_resolve_text ~225-257), src/blueprint_pipeline/scene_placement/README.md, tests/test_scene_placement.py.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), lock the VLM response-parsing contract in src/blueprint_pipeline/scene_placement/target_resolver.py with zero-cost synthetic tests, and document the single paid step.

Current state: resolve_target's real Gemini call (_gemini_resolve_text) cannot be validated for free (needs GOOGLE_GENAI_API_KEY and costs money). The model call is correctly injected so the logic is testable, but the response-parsing surface — _extract_response_text (thinking-part filtering, candidates-only text) and _extract_json_object (code-fenced JSON, stray prose), plus hallucinated-id rejection in resolve_target — is where real Gemini replies break and is fully exercisable with synthetic response objects.

What to do:
1. Add contract tests in tests/test_scene_placement.py feeding FAKE google-genai-shaped response objects (simple stand-in classes with .text / .candidates[].content.parts[] and a part.thought=True flag) into _extract_response_text: assert thinking parts are skipped, a candidates-only answer is returned, and an empty/garbled response yields ''. Feed _extract_json_object with bare JSON, code-fenced JSON, and JSON embedded in prose; assert correct dict extraction and {} on unparseable input.
2. Add resolve_target tests with an INJECTED generate that returns: a target_id pointing at a near-miss/nonexistent id (assert it falls back to resolve_target_by_label, not a crash or a wrong pick), null target_id, and a valid id (assert exact object). All offline.
3. Add an explicit note in src/blueprint_pipeline/scene_placement/README.md (and the _gemini_resolve_text docstring) that live VLM resolution is the ONE genuinely paid step (one gemini-3-flash-preview call, needs GOOGLE_GENAI_API_KEY) and is skipped by default. If you add any live smoke test, mark it skip-by-default (e.g. require an env var) so the suite never spends money.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render/placement outputs are simulator support, NOT policy-success claims; add/extend tests; run `python -m pytest tests/test_scene_placement.py` and `python -m py_compile src/blueprint_pipeline/scene_placement/target_resolver.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only — and crucially, make NO live Gemini call in the default test run.
```

</details>

## provider_race orchestrator

### [P2-11] Test poll-budget discretization and boot-on-last-poll boundary

- **Priority:** P2 · **Effort:** S · **Dimension:** provider_race orchestrator
- **Goal:** Cover the attempts = ceil(marker_timeout/poll_interval) edge cases so an off-by-one can't reap a healthy pod one poll early or add a paid extra wait.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/tests/test_provider_race.py`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/provider_race.py`
- **Validate (CPU):** python -m pytest tests/test_provider_race.py -k 'poll_budget or boundary' -q ; python -m pytest tests/test_provider_race.py -q ; python -m py_compile tests/test_provider_race.py

- **Context:** The G1 job uses generous 900s marker_timeouts (isaac_g1_kitchen_parity_job.py default marker_timeout=900), so the discretization math directly governs both cost and the false-dud rate for the active 'open the refrigerator' render lane. An off-by-one would either reap a pod one poll too early (false dud -> wrongly trip a healthy provider) or do an extra paid wait. Currently untested.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), src/blueprint_pipeline/provider_race.py:race_launch computes attempts = 1 if poll_interval<=0 else max(1, math.ceil(marker_timeout/poll_interval)) (provider_race.py:233) and only sleeps between polls when attempt < attempts-1 (provider_race.py:277). These edges are untested.

Task: add hermetic tests to tests/test_provider_race.py asserting, via the contender record's 'polls' count and the injected sleep call count: (a) poll_interval <= 0 -> exactly one immediate marker check and zero sleeps (use a sleep spy/counter); (b) marker_timeout == 0 -> at least one attempt still happens (max(1, ...)); (c) BOUNDARY: a provider that first shows its marker on the FINAL allowed poll (marker_after == attempts) still wins and is not reaped one poll early. Use a sleep spy that counts calls, injected _NO_SLEEP-style; assert rec['polls'] equals the expected attempt count for the winner and that the boundary provider's outcome == 'won'.

Constraints: fully hermetic; no GPU; no network; injected sleep/monotonic. render outputs are simulator support NOT policy-success claims. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: python -m pytest tests/test_provider_race.py -k 'poll_budget or boundary' (or your names) and the full file; python -m py_compile on the test file.
```

</details>

### [P2-12] Document race_launch wiring status in CHANGELOG and module docstring

- **Priority:** P2 · **Effort:** S · **Dimension:** provider_race orchestrator
- **Goal:** Record, in checked-in docs, that provider_race is built-but-dormant (or, once wired, how to enable it) so the next operator doesn't assume the sequential-failover stall is already fixed.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/docs/CHANGELOG.md`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/provider_race.py`
- **Validate (CPU):** python -m pytest --collect-only -q ; python -m py_compile src/blueprint_pipeline/provider_race.py

- **Context:** Right now the dormant status is recorded only in private memory; a checked-in doc prevents someone from assuming the sequential RunPod->Vast failover stall (which still affects the active 'open the refrigerator' G1 render lane) is already fixed when it is not. Best done LAST so it reflects whatever wiring actually landed.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), docs/CHANGELOG.md (around line 36) lists src/blueprint_pipeline/provider_race.py as added, but NO checked-in doc states its wiring status. The only record that it is built-but-dormant lives in private agent memory ('16 hermetic tests; not yet wired into run_full').

Task: add a short, accurate wiring-status note. (1) In docs/CHANGELOG.md (Future-Agent-Facing section or near the provider_race line), add a sentence describing the CURRENT TRUE state: whether race_launch is wired into the launch path yet, and if not, the concrete steps to enable it (importable marker_check helper, cold/fallback reconciliation, stop-vs-terminate loser teardown, multi-provider CLI flag). IMPORTANT: first check the actual current state of src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py and isaac_particlefield_render_job.py — if a sibling task has since wired it, document 'enabled via --providers/--race'; if not, document 'present but dormant'. Do not assert it is wired unless the code shows it. (2) Add a brief 'Wiring status / how to enable' note to the module docstring of src/blueprint_pipeline/provider_race.py pointing at the launch path call sites.

Constraints: doc-only; keep claims accurate to the code as it stands when you run; render outputs are simulator support NOT policy-success claims; do not overstate readiness. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

Validation: re-read the edited files to confirm wording matches the actual code state; run python -m pytest --collect-only -q to confirm nothing broke; python -m py_compile src/blueprint_pipeline/provider_race.py.
```

</details>

## Spend guard & pod lifecycle

### [P2-13] Pin cold-create capacity-500 behavior distinctly from flaky pod

- **Priority:** P2 · **Effort:** S · **Dimension:** Spend guard & pod lifecycle
- **Goal:** Test that a RunPod cold-create 500 'not enough free GPUs' is retried across attempts with its error preserved, and document the gap that it currently collapses to all_launch_attempts_flaky.
- **Files:** `src/blueprint_pipeline/gpu_render_providers.py`, `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `tests/test_gpu_render_providers.py`
- **Validate (CPU):** python3 -m pytest tests/test_gpu_render_providers.py -k 'no_free_gpus or capacity' -q (new provider + job-layer assertions) ; then python3 -m pytest tests/test_gpu_render_providers.py tests/test_isaac_g1_kitchen_parity_job.py -q and python3 -m py_compile src/blueprint_pipeline/gpu_render_providers.py src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py

- **Context:** Capacity 500s are common and transient; conflating them with flaky-pod failures hides why a batch failed and blocks a smarter wait-and-retry. A test pins current behavior and documents the gap for the provider-race orchestrator already in memory. Anchors: cold create src/blueprint_pipeline/gpu_render_providers.py lines 215-223; marker-retry launch-call-failed handling src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py lines 388-397; existing provider monkeypatch pattern tests/test_gpu_render_providers.py lines 87-131.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), pin (and lightly improve the legibility of) the RunPod cold-create capacity-500 path. In src/blueprint_pipeline/gpu_render_providers.py RunPodRenderProvider.launch cold path (lines 215-223), a 500 'not enough free GPUs' POST /pods has no id, so launch returns blocked ['no_pod_started'] with the raw error buried in attempts. In src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py launch_with_marker_retry (lines 388-397), a launch that returns status != 'launched' is recorded as a 'launch_call_failed' attempt and the loop continues — so capacity 500s ARE retried, but with no backoff and no distinct surfaced blocker (they collapse to 'all_launch_attempts_flaky').

Write hermetic tests that:
1. Provider layer: monkeypatch _runpod_call so POST /pods returns (500, {'error':'not enough free GPUs'}); assert launch returns blocked ['no_pod_started'] AND that the 500 detail is preserved in attempts (the error string is not dropped). Use the existing fake_key/fake_call monkeypatch pattern (tests/test_gpu_render_providers.py lines 87-131).
2. Job/marker layer: assert launch_with_marker_retry records a 'launch_call_failed' attempt for each capacity-500 launch and never enters the marker-poll loop (no phantom pod), ending blocked.
Then make a SMALL, safe improvement: preserve the capacity-500 error detail in the attempts (and, if cheap and non-disruptive, surface a distinct marker like a 'capacity' hint in the attempt dict) so a no-capacity batch is distinguishable from a flaky-pod batch in logs. Do NOT add real backoff/sleep that could slow tests or change spend behavior; if you note that wait-and-retry belongs to the provider-race work, leave a short code comment pointing there rather than implementing it.

Constraints: keep world-model backends swappable (provider-neutral; no new RunPod-only coupling in the job layer); protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; secrets file-based and never logged. Add/extend tests. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done run: `python3 -m pytest tests/test_gpu_render_providers.py tests/test_isaac_g1_kitchen_parity_job.py -q` and `python3 -m py_compile src/blueprint_pipeline/gpu_render_providers.py src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`.
```

</details>

### [P2-14] Surface stopped-but-billing RunPod disk in burn estimate

- **Priority:** P2 · **Effort:** M · **Dimension:** Spend guard & pod lifecycle
- **Goal:** Make total_burn_per_hour/build_report report a distinct standing-disk cost line for STOPPED RunPod pods instead of silently over- or under-counting them.
- **Files:** `scripts/gpu_spend_guard.py`, `tests/test_gpu_spend_guard.py`
- **Validate (CPU):** python3 -m pytest tests/test_gpu_spend_guard.py -k 'burn or disk or stopped' -q (new stopped-disk-cost line test) ; then python3 -m pytest tests/test_gpu_spend_guard.py -q and python3 -m py_compile scripts/gpu_spend_guard.py

- **Context:** The per-minute spend-watch discipline the user wants as reusable tooling needs an HONEST burn number; silently dropping stopped-pod disk charges (the exact reason RunPodRenderProvider.terminate exists, src/blueprint_pipeline/gpu_render_providers.py lines 232-239) makes the watchdog under-report the standing cost of the warm pool the user is deliberately keeping for the G1 refrigerator render lane. Anchors: total_burn_per_hour scripts/gpu_spend_guard.py lines 233-234; build_report lines 481-521; _parse_runpod_pod lines 140-178.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), make the spend-guard burn estimate honest about stopped-but-billing RunPod container disk. total_burn_per_hour (scripts/gpu_spend_guard.py lines 233-234) sums cost_per_hr only over instances where i.live, and RunPod live excludes EXITED/TERMINATED/TERMINATING. A RunPod pod that is STOPPED keeps billing for its (140GB) container disk, but today the guard either counts it as full live compute burn (overstating) when its desiredStatus isn't terminal, or drops it once terminal (understating the persistent disk cost). build_report (lines 481-521) has no notion of standing storage cost.

Required work (coordinate with the STOPPED-state classification fix if that landed first — reuse the same notion of a stopped pod):
1. Distinguish, in the GpuInstance / report, live-compute burn from standing-disk cost for stopped pods. Add a way to compute/report a stopped-pod disk-billing line (you may estimate from container disk size when present in the pod JSON; if the size isn't available, still surface that a stopped pod carries standing cost). Keep total live-compute burn separate from standing-disk cost in build_report output.
2. Do not break the existing burn test (test_collect_instances_and_total_burn) — keep total_burn_per_hour's live-compute semantics, and add the disk accounting as an additional, clearly-labeled figure.
3. Add hermetic tests in tests/test_gpu_spend_guard.py feeding a STOPPED RunPod pod with a known container disk size; assert the report surfaces a distinct 'stopped (disk billing)' line and that the report separates live-compute burn from standing-disk cost.

Constraints: keep world-model backends swappable (this is provider cost tooling, provider-neutral where possible); protect provenance/rights/privacy/raw-capture-truth; secrets file-based and never logged; the burn numbers are cost-control telemetry, NOT policy-success claims. Add/extend tests with canned JSON only. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done run: `python3 -m pytest tests/test_gpu_spend_guard.py -q` and `python3 -m py_compile scripts/gpu_spend_guard.py`.
```

</details>

### [P2-15] String-test bootstrap /workspace/out wipe before runner_done

- **Priority:** P2 · **Effort:** S · **Dimension:** Spend guard & pod lifecycle
- **Goal:** Pin both halves of the stale-output defense: the bootstrap wipes /workspace/out at start AND runner_done is marked only after the runner exits.
- **Files:** `src/blueprint_pipeline/isaac_particlefield_render_job.py`, `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `tests/test_isaac_particlefield_render_job.py`
- **Validate (CPU):** python3 -m pytest tests/test_isaac_particlefield_render_job.py -k 'bootstrap_cleanup or runner_done or wipe' -q (new string-assertion test) ; then python3 -m pytest tests/test_isaac_particlefield_render_job.py tests/test_isaac_g1_kitchen_parity_job.py -q and python3 -m py_compile src/blueprint_pipeline/isaac_particlefield_render_job.py

- **Context:** Warm reuse + stale /workspace/out is the named recurring bug for the active G1 refrigerator render lane. The parent-side runner_done guard mitigates it, but the bootstrap-side cleanup is the first line of defense and is currently under-asserted for the particlefield job. Anchors: out-dir wipe src/blueprint_pipeline/isaac_particlefield_render_job.py lines 48-51; runner_done after subprocess.call lines 86-87; docker_start_cmd lines 123-133; the parity job already has a partial string test (tests/test_isaac_g1_kitchen_parity_job.py lines 137-140).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add string-level hermetic tests pinning the bootstrap-side defense against the classic warm-reuse stale-output bug. Both bootstraps wipe /workspace/out at start (src/blueprint_pipeline/isaac_particlefield_render_job.py lines 48-51; src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py BOOTSTRAP) and flip bootstrap.phase to runner_done ONLY after the runner subprocess exits (isaac_particlefield_render_job.py lines 86-87). watch_and_collect's parent-side guard (ignoring result json before runner_done) is already tested (test_watch_and_collect_ignores_stale_result_before_runner_done), but the bootstrap-side cleanup is not — a regression that removed the out-dir wipe would still pass current tests, re-introducing the stale-output bug where a prior warm-run's isaac_g1_kitchen_parity_result.json is uploaded and mistaken for the new run on a warm-restarted pod.

Write pure-string assertion tests (no execution, no network) in tests/test_isaac_particlefield_render_job.py (and optionally a mirror in tests/test_isaac_g1_kitchen_parity_job.py) that:
1. Assert the BOOTSTRAP / docker_start_cmd() body contains the out-dir clear loop: iterating pathlib.Path(OUT).iterdir() and unlink/rmtree of each entry (the existing parity test already checks 'pathlib.Path(OUT).iterdir()' and 'shutil.rmtree(p)' — extend the particlefield job with the same guard).
2. Assert that mark('runner_done', ...) / the runner_done marker is emitted only AFTER subprocess.call(cmd) in the script body (i.e. the runner_done text appears after the runner-invocation text), so a stale result cannot be flipped to runner_done before the runner actually runs.
These are static string assertions over docker_start_cmd()[1] / the BOOTSTRAP constant; do not run any container or python subprocess.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (stale-output prevention is part of raw-capture/run-result integrity); render outputs are simulator support, NOT policy-success claims; secrets file-based and never logged. Tests-only unless a missing wipe is found, in which case add the iterate+unlink loop to the affected bootstrap. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done run: `python3 -m pytest tests/test_isaac_particlefield_render_job.py tests/test_isaac_g1_kitchen_parity_job.py -q` and `python3 -m py_compile src/blueprint_pipeline/isaac_particlefield_render_job.py src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`.
```

</details>

### [P2-16] Add JSON + burn-threshold + watch mode to spend guard

- **Priority:** P2 · **Effort:** M · **Dimension:** Spend guard & pod lifecycle
- **Goal:** Make gpu_spend_guard scriptable: a --json machine-readable mode, a --max-burn-usd threshold that exits non-zero when exceeded, and an optional --watch-seconds loop.
- **Files:** `scripts/gpu_spend_guard.py`, `tests/test_gpu_spend_guard.py`
- **Validate (CPU):** python3 -m pytest tests/test_gpu_spend_guard.py -k 'json_or_threshold or json or threshold or watch' -q (new --json valid-JSON test + --max-burn-usd exit-2 test) ; then python3 -m pytest tests/test_gpu_spend_guard.py -q and python3 -m py_compile scripts/gpu_spend_guard.py

- **Context:** The user explicitly wants the per-minute spend watch assessed as reusable tooling. Today it is a one-shot text report with a fixed exit 0, which cannot drive an alert or a CI gate during the spend-paused G1 refrigerator render lane. build_report/total_burn_per_hour/collect_instances are already pure and testable (scripts/gpu_spend_guard.py lines 216-234, 481-521), so a --json flag + --max-burn-usd threshold + --watch-seconds loop are small, pure additions that don't touch the HTTP/terminate code. Existing CLI/fixture anchors: main lines 536-614, patched_guard fixture tests/test_gpu_spend_guard.py lines 300-341.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), turn scripts/gpu_spend_guard.py into a reusable per-minute spend-watch tool by adding pure, testable CLI surface around the existing pure functions (build_report, total_burn_per_hour, collect_instances). Today main (lines 536-614) prints a human report and always returns 0, with no machine-readable output, no threshold exit, and no interval loop.

Required work:
1. Add a --json flag: emit a machine-readable JSON object (schema_version, instances with id/provider/state/cost_per_hr/age/owned, total live-compute burn, reap candidates) instead of (or alongside) the text report. Build it from collect_instances + the existing pure helpers; never include secret values.
2. Add --max-burn-usd THRESHOLD: when the live burn exceeds the threshold, exit with code 2 (distinct from the normal 0). Keep the default (no threshold) behavior returning 0.
3. Add an optional --watch-seconds N loop wrapper that re-collects and re-reports every N seconds; factor the single-pass logic so it is callable in a loop. Keep it injection-friendly (the loop's sleep and clock indirected) so tests can run one or two iterations without real time passing. Do NOT add any network retry/backoff that spends.
4. Add hermetic tests in tests/test_gpu_spend_guard.py reusing the patched_guard fixture pattern (lines 300-341): call main(['--json', ...]) and assert valid JSON containing burn + candidates; call main(['--max-burn-usd', <below live burn>, ...]) and assert exit code 2; (optionally) test one watch iteration via a monkeypatched sleep that raises/stops after the first pass.

Constraints: keep world-model backends swappable (provider cost tooling, no world-model coupling); protect provenance/rights/privacy/raw-capture-truth; secrets file-based and never logged — assert no secret leaks into --json output; burn numbers are cost telemetry, NOT policy-success claims; do not change the network/terminate code paths. Add/extend tests. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done run: `python3 -m pytest tests/test_gpu_spend_guard.py -q` and `python3 -m py_compile scripts/gpu_spend_guard.py`.
```

</details>

### [P2-17] Pin particlefield flaky-cold wait gap vs parity marker-retry

- **Priority:** P2 · **Effort:** M · **Dimension:** Spend guard & pod lifecycle
- **Goal:** Document and test that run_isaac_particlefield_render_job pays for a dead cold pod up to max_seconds because it skips launch_with_marker_retry, motivating reuse of the marker-retry guard.
- **Files:** `src/blueprint_pipeline/isaac_particlefield_render_job.py`, `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `tests/test_isaac_particlefield_render_job.py`
- **Validate (CPU):** python3 -m pytest tests/test_isaac_particlefield_render_job.py -k 'flaky_cold or no_marker or dead_pod' -q (new behavior-pin test) ; then python3 -m pytest tests/test_isaac_particlefield_render_job.py -q and python3 -m py_compile src/blueprint_pipeline/isaac_particlefield_render_job.py

- **Context:** The splat render path can pay for a dead cold pod for up to max_seconds (1200s default) before terminate, because it skips the marker-retry the parity path uses — a real spend gap and an inconsistency that should be tested and ideally unified later (provider-race work in memory). Anchors: run_isaac_particlefield_render_job watch call src/blueprint_pipeline/isaac_particlefield_render_job.py lines 433-439; parity marker-retry src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py lines 599-602; launch_with_marker_retry lines 377-419; existing watch_and_collect tests tests/test_isaac_particlefield_render_job.py lines 103-161.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), pin the asymmetry where the splat render path lacks the flaky-cold marker-retry protection the parity path has. src/blueprint_pipeline/isaac_particlefield_render_job.py run_isaac_particlefield_render_job calls watch_and_collect directly (lines 433-439) WITHOUT launch_with_marker_retry, so a flaky cold pod (created + billing but the container never runs the bootstrap) is only reaped after the full max_seconds watch (default 1200s) — a long, expensive wait on a dead pod. The parity job uses launch_with_marker_retry (~150s marker window) at src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py lines 599-602. This gap is undocumented and untested.

Write a hermetic test in tests/test_isaac_particlefield_render_job.py that:
1. Drives run_isaac_particlefield_render_job(allow_paid=True) with J.stage_bundle stubbed (write provider_*_url.txt incl. the get url), J.get_render_provider monkeypatched to a fake provider that launches successfully but whose output zip never contains a runner_done / never heartbeats, and time/urlopen monkeypatched so no real wait occurs.
2. Asserts the CURRENT behavior: the job watches up to max_seconds (use a tiny max_seconds in the test) and then terminates the dead pod, pinning that there is no early marker-retry kill. Capture/assert the teardown call and the blocked status.
3. Adds a short code comment in run_isaac_particlefield_render_job noting the asymmetry and that reusing launch_with_marker_retry would shorten the dead-cold-pod wait — but DO NOT refactor the splat path to use marker-retry in this task unless trivially safe; the goal is to pin behavior and document the gap (the unify work can be a follow-on).

Constraints: keep world-model backends swappable (provider-neutral); protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; secrets file-based and never logged. Add/extend tests. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done run: `python3 -m pytest tests/test_isaac_particlefield_render_job.py -q` and `python3 -m py_compile src/blueprint_pipeline/isaac_particlefield_render_job.py`.
```

</details>

## Dev env & deps

### [P2-18] Document the canonical pytest interpreter and the BlueprintContracts sibling requirement

- **Priority:** P2 · **Effort:** S · **Dimension:** Dev env & deps
- **Goal:** Document that pytest MUST run from .venv and explain the dual blueprint_contracts source (pinned git commit vs sibling checkout) so contributors stop seeing false skips/errors.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/tests/conftest.py`, `$HOME/workspace/BlueprintCapturePipeline/pyproject.toml`, `$HOME/workspace/BlueprintCapturePipeline/README.md`
- **Validate (CPU):** `.venv/bin/python -m pytest --collect-only -q` collects ~2491 with no errors; the docs now name `.venv/bin/python -m pytest` as the canonical command. `.venv/bin/python -c 'import blueprint_contracts; print(blueprint_contracts.__file__)'` resolves from the pinned/installed source. No GPU, no cloud.

- **Context:** Reproducible pytest is the baseline CPU validation. The audit confirmed 2491 tests collect cleanly under .venv but the docs never say which interpreter — and the only interpreter with pxr (system 3.9) is precisely the one that cannot import the contracts/project. Files: $HOME/workspace/BlueprintCapturePipeline/tests/conftest.py, $HOME/workspace/BlueprintCapturePipeline/pyproject.toml, $HOME/workspace/BlueprintCapturePipeline/README.md.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), tests/conftest.py (lines ~7-25) injects src/ and a sibling ../BlueprintContracts/src onto sys.path. The sibling repo IS present (../BlueprintContracts exists) and blueprint_contracts is also pip-installed into .venv as a git dependency (pinned commit referenced in pyproject.toml around line 26). This dual path (sys.path sibling OR installed wheel) is undocumented: the only interpreter with pxr (system python3.9) has neither the wheel nor can resolve the project, so `python3 -m pytest` from system python collects nothing useful.

Do this:
1. In docs (DEV_SETUP.md if it exists, else README 'Local Development'), state clearly: run pytest as `.venv/bin/python -m pytest` (the project interpreter), NOT bare `python3 -m pytest`.
2. Document that the pinned blueprint-contracts git commit in pyproject.toml is the source of truth, and the sibling ../BlueprintContracts checkout is an optional dev override that conftest.py prepends to sys.path.
3. Do NOT change conftest behavior unless it is broken; this is primarily documentation. If you add anything, add a tiny test asserting blueprint_contracts imports from the expected source.

Constraints: Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only. Keep world-model backends swappable. Protect provenance/rights/privacy/raw-capture-truth. Render outputs are simulator support, NOT policy-success claims. Run `python -m py_compile tests/conftest.py` and `python -m pytest --collect-only -q`.
```

</details>

### [P2-19] Remove the orphan output/runpod_launch_venv ad-hoc venv

- **Priority:** P2 · **Effort:** S · **Dimension:** Dev env & deps
- **Goal:** Delete the unreferenced hand-built Python 3.13 boto3-only venv once boto3 is in the canonical interpreter, so it stops masquerading as a real environment.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/output/runpod_launch_venv/pyvenv.cfg`
- **Validate (CPU):** `grep -rn 'runpod_launch_venv' --include='*.py' --include='*.sh' --include='*.md' .` returns no code references; after the boto3-in-dev task, `.venv/bin/python -m blueprint_pipeline.wam_provider_object_store --help` works without the side venv; `ls output/runpod_launch_venv` reports no such directory after removal. No GPU, no cloud.

- **Context:** Orphan venvs that no code references are a reproducibility hazard: this one encodes an undocumented manual fix and shares boto3 1.43.36 with .venv, so it can be mistaken for the project environment. Removing it forces boto3 to be solved in pyproject. Depends on the boto3-in-dev task. Files: $HOME/workspace/BlueprintCapturePipeline/output/runpod_launch_venv/pyvenv.cfg (and the enclosing directory).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), output/runpod_launch_venv is a hand-created venv (pyvenv.cfg shows /opt/homebrew/opt/python@3.13, Python 3.13.5) containing ONLY boto3/botocore/jmespath/s3transfer/urllib3 and NO project code. It is gitignored and referenced by NO source/script (verified: `grep -rn 'runpod_launch_venv' --include='*.py' --include='*.sh' --include='*.md' .` returns zero code references). It is a manual workaround for the boto3-not-in-dev gap.

Do this:
1. Re-verify there are zero references: `grep -rn 'runpod_launch_venv' --include='*.py' --include='*.sh' --include='*.md' .` returns nothing.
2. Confirm boto3 is now available in .venv (depends on the boto3-in-dev task) so the side venv is unnecessary.
3. Delete the directory output/runpod_launch_venv. It is gitignored so this is a local-filesystem cleanup, not a tracked change — do NOT commit a deletion of an untracked path; just remove the on-disk directory.

Constraints: Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only. Keep world-model backends swappable. Protect provenance/rights/privacy/raw-capture-truth (the venv holds no capture data; verify before deleting). Render outputs are simulator support, NOT policy-success claims.
```

</details>

## Launch gates & readiness

### [P2-20] Make launch-gate scripts require the project .venv interpreter instead of failing with a misleading PIL error

- **Priority:** P2 · **Effort:** S · **Dimension:** Launch gates & readiness
- **Goal:** Replace the misleading 'No module named PIL' collection error on the alpha-readiness gate leg with an actionable interpreter/venv requirement message or guard.
- **Files:** `scripts/run_external_alpha_launch_gate.py`, `scripts/run_paid_marketplace_launch_gate.py`, `pyproject.toml`
- **Validate (CPU):** /usr/bin/python3 -m pytest tests/test_alpha_readiness.py --collect-only  (reproduces failure)  ;  .venv/bin/python -m pytest tests/test_alpha_readiness.py --collect-only  (collects)  ;  .venv/bin/python -m pytest tests/test_external_alpha_launch_gate.py -q  &&  .venv/bin/python -m py_compile scripts/run_external_alpha_launch_gate.py scripts/run_paid_marketplace_launch_gate.py

- **Context:** While GPU work is paused and the team leans on CPU gate validation, a misleading 'PIL not found' collection error on the alpha-readiness leg can be mistaken for a real gate regression or cause someone to skip that leg, eroding trust in the CPU gate. Confirmed: Pillow in pyproject.toml line ~29; .venv/bin/python has PIL and passes; bare system python3.9 fails at collection. This is doc/guard only — no production logic change. Use .venv/bin/python for validation.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU/no-spend task. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When the launch-gate pipeline pytest legs run under an interpreter that lacks runtime deps (e.g. system Xcode python3.9), they fail at COLLECTION time with ModuleNotFoundError: No module named 'PIL'. The import chain is tests/test_alpha_readiness.py -> capture_orchestrator -> robot_eval_job_orchestrator -> robot_eval_execution -> robot_initial_observation, which does 'from PIL import Image, ImageDraw'. Pillow IS declared in pyproject.toml (Pillow>=10.0.0) and present in the repo .venv (.venv/bin/python imports PIL and the tests pass), so this is a tooling/repro-hygiene gap, NOT a production defect: a contributor/CI invoking bare 'pytest' or sys.executable -m pytest outside the venv gets a collection error that looks like a real gate regression.

Add a lightweight, non-invasive guard or documentation so the interpreter requirement is explicit:
- Preferred: in scripts/run_external_alpha_launch_gate.py and scripts/run_paid_marketplace_launch_gate.py, before invoking the pipeline pytest leg, check that the interpreter that will run pytest can import the required runtime deps (e.g. import 'PIL'); if not, print a clear, actionable message naming the .venv requirement (e.g. 'Run this gate with the project .venv interpreter: .venv/bin/python -m scripts/...; system python lacks Pillow and other runtime deps') and exit non-zero, instead of letting pytest emit a confusing collection error. Keep it a guard/diagnostic only — do NOT change which tests run or alter gate pass/fail semantics for a correctly-provisioned environment.
- Also add a short note in pyproject.toml comments or the gate scripts' module docstrings documenting the .venv requirement.

Constraints: do not modify production import chains or robot_initial_observation; keep world-model backends swappable; protect provenance, rights, privacy, raw-capture truth; gate output is support, NOT a policy-success claim. This must satisfy $HOME/workspace/Blueprint-WebApp/docs/autonomous-loop-evidence-checklist-2026-05-03.md before claiming done.

Reproduce the problem first: /usr/bin/python3 -m pytest tests/test_alpha_readiness.py --collect-only (should fail with No module named 'PIL'); contrast: .venv/bin/python -m pytest tests/test_alpha_readiness.py --collect-only (should collect). Then verify your guard message fires under the missing-deps interpreter and does NOT fire under .venv.

Then run: .venv/bin/python -m pytest tests/test_external_alpha_launch_gate.py -q and .venv/bin/python -m py_compile scripts/run_external_alpha_launch_gate.py scripts/run_paid_marketplace_launch_gate.py
```

</details>

## Warm render transport / object store

### [P2-21] Make WarmPoolClient.poll_result use conditional GET / backoff instead of re-downloading the growing zip every interval

- **Priority:** P2 · **Effort:** S · **Dimension:** Warm render transport / object store
- **Goal:** Reduce poll_result egress/latency by adding If-None-Match / If-Modified-Since conditional GET (or at least exponential backoff) so it stops re-downloading the full cumulative output zip ~60x per job.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/warm_render_server.py`, `$HOME/workspace/BlueprintCapturePipeline/tests/test_warm_render_server.py`
- **Validate (CPU):** python -m pytest tests/test_warm_render_server.py -q (add: wrap http_get in a call counter; assert poll_result issues fewer full downloads via backoff, or sends an If-None-Match/conditional request and treats a 304-equivalent as 'keep waiting' without re-parsing; assert the existing reads-from-output-zip and times-out-when-absent tests still pass). Also `python -m py_compile src/blueprint_pipeline/warm_render_server.py`.

- **Context:** On the active 'open the refrigerator' warm lane the whole point is cheap, fast reruns; re-downloading a growing multi-artifact zip ~60x per job inflates poll latency and egress, eroding the warm-pod cost advantage. It compounds with the no-cleanup issue and is trivially fixable. File: src/blueprint_pipeline/warm_render_server.py (poll_result 237-252, _http_get_bytes 151). Test scaffolding: tests/test_warm_render_server.py wraps http_get in lambdas and drives clock/sleep (see test_warm_pool_client_poll_result_reads_from_output_zip line 184).

<details><summary>Prompt (copy into a fresh session)</summary>

```text
Repo: BlueprintCapturePipeline (cwd = repo root). CPU-only, no-spend hermetic efficiency fix.

Problem: WarmPoolClient.poll_result (src/blueprint_pipeline/warm_render_server.py:243-251) calls http_get(output_get_url) — a full download + full ZipFile parse of the cumulative /workspace/out zip — every interval_s (default 5s) until timeout (default 300s), i.e. up to ~60 full downloads per job. Combined with the never-cleared out dir, each poll re-downloads all accumulated artifacts. There is no ETag/Last-Modified conditional GET, no central-directory-only read, and no backoff.

Fix: reduce redundant transfer. Minimum: add exponential backoff to the poll interval (cap at a sane max) so a slow render does not trigger ~60 full downloads. Better: support a conditional GET — extend the injected http_get contract to optionally pass/receive validators (If-None-Match with the last ETag, or If-Modified-Since) and treat a 304 as 'unchanged, keep waiting' without re-parsing. Keep the default _http_get_bytes behavior working for callers that don't supply validators, and keep the existing http_get=lambda u: bytes test signature usable (e.g. make the conditional path opt-in / backward-compatible). Do not change the success semantics — once the keyed result with the right freshness token (if that work has landed) is present, return it.

Constraints: keep world-model backends swappable (the http_get injection point stays the transport boundary). Protect provenance/rights/privacy — no raw URL or secret logging. Render outputs are simulator support, NOT policy-success claims. Add/extend tests. Run `python -m pytest tests/test_warm_render_server.py -q` and `python -m py_compile src/blueprint_pipeline/warm_render_server.py`.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## scene_semantics (Gemini)

### [P2-22] Remove or repurpose the dead _extract_json_array helper

- **Priority:** P2 · **Effort:** S · **Dimension:** scene_semantics (Gemini)
- **Goal:** Either delete the orphaned _extract_json_array (and its 4 tests) or fold its fence-stripping into _extract_json_object so the more-robust logic is the one actually used in production.
- **Files:** `src/blueprint_pipeline/scene_semantics.py`, `tests/test_scene_semantics.py`
- **Validate (CPU):** Confirm orphan: `grep -rn '_extract_json_array' src/` shows only the definition BEFORE; shows nothing AFTER (path A or B both delete it). Then python3 -m pytest tests/test_scene_semantics.py -q (green); python3 -m py_compile src/blueprint_pipeline/scene_semantics.py.

- **Context:** Dead code with passing tests creates false confidence that JSON-array parsing is exercised in production when prod only ever calls _extract_json_object. Worse, the orphaned helper holds the MORE robust fence-stripping/envelope-unwrapping logic while the LESS robust _extract_json_object is what actually runs on the scene_semantics path feeding the 'open the refrigerator' G1 POV seed lane. This goal pairs with (and is partly subsumed by) the _extract_json_object hardening goal — coordinate to avoid duplicate edits.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), _extract_json_array (src/blueprint_pipeline/scene_semantics.py around lines 300-324) is defined and has 4 dedicated test assertions (tests/test_scene_semantics.py around lines 336-339) but is NEVER called in production — `grep -rn '_extract_json_array' src/` returns only the definition. Production paths parse an OBJECT (the combined prompt requests a JSON object with an 'objects' array), so _extract_json_object is the correct prod path and the array helper is orphaned.

Choose ONE of:
A) DELETE _extract_json_array and remove its 4 test assertions; keep the suite green. Prefer this if the 'Harden _extract_json_object' goal (fence-stripping + brace-balanced first-object extraction) has already landed, so no robustness is lost.
B) REPURPOSE: fold _extract_json_array's markdown-fence-stripping (the `re.sub(r'```(?:json)?\s*', '', text)` step) into _extract_json_object, then delete _extract_json_array and migrate/replace its tests with object-extraction coverage. Do NOT leave both: the goal is to eliminate the false confidence that array parsing is exercised in prod.

If the _extract_json_object hardening goal has NOT yet landed, take path B and implement the fence-stripping there as part of this task.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render/review outputs are simulator-support signals NOT policy-success claims; add/extend tests; run `python3 -m pytest tests/test_scene_semantics.py -q` and `python3 -m py_compile src/blueprint_pipeline/scene_semantics.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P2-23] Guard a typo'd SCENE_SEMANTICS_GEMINI_MODEL / CAPTURE_FIDELITY_GEMINI_MODEL override from silently blanking the cascade

- **Priority:** P2 · **Effort:** S · **Dimension:** scene_semantics (Gemini)
- **Goal:** When an explicit model override fails all attempts, log that the OVERRIDE (not the standard cascade) was the failing path, and optionally fall back to the default cascade.
- **Files:** `src/blueprint_pipeline/scene_semantics.py`, `tests/test_scene_semantics.py`
- **Validate (CPU):** python3 -m pytest tests/test_scene_semantics.py -q (green, including the bogus-override tests for both env vars); python3 -m py_compile src/blueprint_pipeline/scene_semantics.py. The override tests assert a WARNING naming the override value and the chosen behavior (None or cascade-fallback).

- **Context:** An operator pinning a model via SCENE_SEMANTICS_GEMINI_MODEL or CAPTURE_FIDELITY_GEMINI_MODEL can, with one typo, silently disable the entire multi-model resilience and get only the opaque local fallback — wasting the uploaded video and a debugging cycle, e.g. while iterating on the kitchen capture for the 'open the refrigerator' G1 POV seed lane. This pairs with the cascade-exhaustion logging goal to make the failure attributable to the override rather than the standard cascade.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In BlueprintCapturePipeline (cwd = repo root), both helpers in src/blueprint_pipeline/scene_semantics.py build `models_to_try = [override] if override else list(_DEFAULT_MODEL_CASCADE)` (around lines 345-346 for SCENE_SEMANTICS_GEMINI_MODEL, and around lines 652-653 for CAPTURE_FIDELITY_GEMINI_MODEL / SCENE_SEMANTICS_GEMINI_MODEL). The override is .strip()'d so whitespace-only correctly falls back to the cascade (good). But a single typo'd override (e.g. 'gemini-flash') REPLACES the entire cascade with one possibly-invalid model and, on failure, returns None with no fallback and no indication that the override was the cause.

Do this:
1. At minimum: when an explicit override is set and every attempt for it fails, log a logger.warning naming the override value and stating the standard cascade was bypassed (this pairs with the cascade-exhaustion logging goal). Do NOT log any raw-capture content.
2. Optionally (preferred if low-risk): if an explicit override exhausts all attempts, fall back to trying _DEFAULT_MODEL_CASCADE (excluding the already-tried override) before returning None, so one typo cannot fully disable multi-model resilience. Gate this behind clear, simple logic and keep the None contract when even the cascade fails. Make the fallback behavior easy to reason about; if you implement it, document it with a short comment.

Add tests in tests/test_scene_semantics.py: monkeypatch.setenv('SCENE_SEMANTICS_GEMINI_MODEL', 'bogus-model'); use a fake client that RAISES for 'bogus-model' but SUCCEEDS for a real cascade model; assert the implemented behavior (None today if you only log; or a successful fallback result if you implement fallback) AND assert a warning was logged naming the override. Add the analogous test for CAPTURE_FIDELITY_GEMINI_MODEL on the review helper. Reuse the fake-google-module + RaisingModelsClient/SuccessClient patterns; monkeypatch time.sleep to no-op.

Constraints: keep world-model backends swappable (the override mechanism itself is part of swappability — preserve it); protect provenance/rights/privacy/raw-capture-truth; render/review outputs are simulator-support signals NOT policy-success claims; add/extend tests; run `python3 -m pytest tests/test_scene_semantics.py -q` and `python3 -m py_compile src/blueprint_pipeline/scene_semantics.py`. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## Code structure / tech debt

### [P2-24] Extract pure data/manifest/serialization helpers into parity_manifest module

- **Priority:** P2 · **Effort:** M · **Dimension:** Code structure / tech debt
- **Goal:** Move the no-Isaac dict/sequence/SceneObject data-shaping and placement-validation-manifest helpers into src/blueprint_pipeline/parity_manifest.py.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `src/blueprint_pipeline/parity_manifest.py`, `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`, `tests/test_isaac_g1_kitchen_parity_job.py`
- **Validate (CPU):** `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_isaac_g1_kitchen_parity_job.py -q` stays green (covers _build_placement_validation_manifest, _find_standoff_fixtures, _placement_*); new `import blueprint_pipeline.parity_manifest` test passes; `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py src/blueprint_pipeline/parity_manifest.py`.

- **Context:** `_build_placement_validation_manifest` (confirmed at runner line 3932) is a ~161-line pure aggregator that produces the placement-pass contract the readiness gate consumes, buried in the Isaac-only section; it is already exercised by `M._build_placement_validation_manifest` (runner test lines 1151/1235/1279) by exec'ing the whole 330KB file. Extracting the serialization/manifest layer makes that contract testable in isolation and clarifies the stage-touching boundary for the 'open the refrigerator' placement-validation path. Confidence on exact line numbers is medium — re-grep each name before moving, and keep any pxr-stage-bound helper in the runner.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), extract the CPU-pure data-shaping / manifest / serialization helpers out of `scripts/run_isaac_g1_kitchen_parity_eval.py` into a new `src/blueprint_pipeline/parity_manifest.py`. These operate on plain dicts/sequences/SceneObject, not live USD stages.

Move (approx line numbers from audit; re-confirm by grep before moving): `_scene_object_to_dict` (~3752), `_scene_object_xy_size_area` (~3763), `_is_structural_or_target_obstacle` (~3774), `_placement_verdict_to_dict` (~3874), `_footprint_box_for_pose` (~3895), `_footprint_center_xy_from_bbox` (~3240), `_xy_distance` (~3249), `_find_standoff_fixtures` (~3903), `_build_placement_validation_manifest` (3932, confirmed), `_placement_validation_passed_manifest` (~4094), `_placement_visual_qc_target_label` (~4098), `_target_object_from_stance_plan` (~3710), `_synthesized_room_edge_shell_boxes` (~3662), `_safe_shell_obstacle_id` (~3704), `_is_support_contact` (~2956), `_vec3_to_list` (~2865), `_matrix4_to_rows` (~3099), `_xform_op_record` (~3106). If any of these touch a live `pxr` Usd stage object (not just plain matrices/dicts), leave it in the runner and note it.

What to do:
1. Create `src/blueprint_pipeline/parity_manifest.py` with the verbatim functions. If any helper imports from `parity_geometry`/`parity_kinematics`, import from there.
2. Re-import the moved names back into the runner at module scope via the bundle/repo dual-try fallback so existing tests still access `M._build_placement_validation_manifest`, `M._find_standoff_fixtures`, `M._xy_rect_overlap_and_gap`, `M._placement_*`.
3. Add the bundle copy + namelist assertion for `parity_manifest.py`.
4. Add a direct-import test for `blueprint_pipeline.parity_manifest` building a small placement-validation manifest from synthetic dict inputs and asserting the pass/fail contract.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; the placement-validation manifest is the contract the readiness gate consumes — render/placement outputs are simulator support, NOT policy-success claims; behavior-preserving move only; add/extend tests. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done, run `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_isaac_g1_kitchen_parity_job.py -q` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py src/blueprint_pipeline/parity_manifest.py`.
```

</details>

### [P2-25] Extract image/IO leaf helpers (denoise, quality, arg-parser) into parity_io module

- **Priority:** P2 · **Effort:** M · **Dimension:** Code structure / tech debt
- **Goal:** Move PIL/cv2/argparse leaf helpers into src/blueprint_pipeline/parity_io.py, leaving numpy/replicator-bound helpers in the runner.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `src/blueprint_pipeline/parity_io.py`, `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`, `tests/test_isaac_g1_kitchen_parity_job.py`
- **Validate (CPU):** `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_isaac_g1_kitchen_parity_job.py -k 'pov_seed_frame_quality or arg_parser or facing or dry_render or denoise' -q` stays green; `import blueprint_pipeline.parity_io` succeeds with no Pillow installed; `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py src/blueprint_pipeline/parity_io.py`.

- **Context:** `_pov_seed_frame_quality` (runner line 1523) and the luma/histogram trio are the CPU image-statistics gate for the 'open the refrigerator' POV seed frames — exactly the logic that should be CPU-unit-tested with synthetic images, and which currently has no clean skip when Pillow is missing (the same gap fixed by the P0 PIL-skip task). `build_arg_parser` (confirmed line 7056) is pure argparse already covered by `M.build_arg_parser`/`M.main` tests. Splitting these out removes ~150 lines from the runner body and gives the image QC a clean home. Keep numpy/Replicator-bound helpers in the runner. Confidence on exact line numbers is medium — re-grep before moving.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), extract CPU-only image/IO leaf helpers out of `scripts/run_isaac_g1_kitchen_parity_eval.py` into a new `src/blueprint_pipeline/parity_io.py`.

Move (re-confirm line numbers by grep): `_software_denoise_image` (~5063, cv2/PIL only), `_fraction_from_histogram` (~1507), `_image_luma_extreme_fractions` (~1514), `_pov_seed_frame_quality` (~1523, PIL only), `_render_step_watchdog_seconds` (~4988), `_write_render_step_timeout_result` (~4998), `_facing_error_deg` (~6733), `_dry_render_checks` (~6746), `build_arg_parser` (7056, confirmed). LEAVE in the runner: `_save_rgb` (numpy/annotator-bound) and `_replicator_step_with_watchdog` (Replicator-bound).

What to do:
1. Create `src/blueprint_pipeline/parity_io.py`. Import PIL/cv2 lazily inside the functions that need them (do NOT import PIL/cv2 at module top level) so `import blueprint_pipeline.parity_io` succeeds on a minimal CPU env without Pillow/OpenCV — match how the runner currently keeps imports lazy so the module loads without GPU.
2. Re-import the moved names back into the runner at module scope via the bundle/repo dual-try fallback so tests still access `M.build_arg_parser`, `M.main`, `M._pov_seed_frame_quality`, `M._facing_error_deg`, `M._dry_render_checks`.
3. Add the bundle copy + namelist assertion for `parity_io.py`.
4. Add CPU tests guarded with `pytest.importorskip("PIL")` (and `importorskip("cv2")` where relevant) that exercise `_software_denoise_image` and the luma/histogram quality trio on synthetic in-memory images, plus a no-dependency test of `build_arg_parser`.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; image-quality QC is simulator support, NOT a policy-success claim; behavior-preserving move only; new tests must SKIP (not fail) when Pillow/cv2 absent. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done, run `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_isaac_g1_kitchen_parity_job.py -k 'pov_seed_frame_quality or arg_parser or facing or dry_render or denoise' -q` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py src/blueprint_pipeline/parity_io.py`.
```

</details>

### [P2-26] Decompose the 1103-line run_scenarios god-function into named phase helpers

- **Priority:** P2 · **Effort:** L · **Dimension:** Code structure / tech debt
- **Goal:** Carve run_scenarios (runner lines 5593-6696) into sequential phase functions while preserving the exact pre-close result-write ordering and the run_scenarios signature/return.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_isaac_g1_kitchen_parity_job.py -q` stays green (test_runner_writes_result_before_simulation_app_close + source-marker tests pin ordering); `run_scenarios` signature/return unchanged so callers compile; `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py`.

- **Context:** `run_scenarios` is the single largest tech-debt hotspot in the runner that drives the active 'open the refrigerator' G1 POV render lane (1103 lines, confirmed def at 5593, called from `isaac_g1_kitchen_parity_job.py`). `test_runner_writes_result_before_simulation_app_close` (runner test line 33) and source-marker tests pin the result-before-close ordering, so the refactor must keep that call explicit and ordered. This is the highest-effort, medium-confidence item — best done LAST, after the leaf extractions (geometry/kinematics/manifest/io) have shrunk the function body and clarified the CPU-vs-GPU boundary. Tests load the runner via `importlib.util.spec_from_file_location`, so module-level names and ordering must be preserved.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), decompose the 1103-line `run_scenarios` function (lines 5593-6696, confirmed def at 5593) in `scripts/run_isaac_g1_kitchen_parity_eval.py`. It currently mixes boot, stage open, target resolution, stance planning, placement, camera setup, articulation, rendering, QC, manifest assembly, and result write in one body — impossible to unit-test a sub-phase, and the Isaac-vs-CPU boundary is invisible inside it.

What to do:
1. Carve the body into sequential, well-named phase functions, each taking an explicit context object (a dataclass or dict carrying the in-progress state), e.g. `_phase_resolve_and_plan_stance`, `_phase_place_and_validate`, `_phase_setup_cameras`, `_phase_render_frames`, `_phase_assemble_result`. Keep the EXACT ordering — in particular the pre-close result write that `test_runner_writes_result_before_simulation_app_close` (runner test line 33) pins, and any source-marker tests that read the runner source (`source = _RUNNER.read_text()` patterns in the test file).
2. Keep `run_scenarios`'s signature and return value byte-identical so `isaac_g1_kitchen_parity_job.py` and all other callers stay unchanged.
3. Prefer making the CPU-pure planning/manifest phases (`_phase_resolve_and_plan_stance`, `_phase_place_and_validate`, `_phase_assemble_result`) callable/mockable independently from the GPU render phase, so they can later be tested without Isaac.
4. Do NOT change numeric behavior, render outputs, or the result JSON schema.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; behavior-preserving refactor only; add/extend tests where a phase is now independently callable. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done, run `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py tests/test_isaac_g1_kitchen_parity_job.py -q` and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py`.
```

</details>

### [P2-27] Inventory and narrow the highest-risk bare except-Exception swallows

- **Priority:** P2 · **Effort:** M · **Dimension:** Code structure / tech debt
- **Goal:** Inventory all 94 bare except-Exception:#noqa:BLE001 blocks in the runner and narrow or add logging to the highest-risk silent swallows around stage/articulation reads.
- **Files:** `scripts/run_isaac_g1_kitchen_parity_eval.py`, `tests/test_isaac_g1_kitchen_parity_runner.py`
- **Validate (CPU):** `grep -c 'except Exception:  # noqa: BLE001' scripts/run_isaac_g1_kitchen_parity_eval.py` shows a reduced count vs 94; `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -q` stays green; `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py`.

- **Context:** 94 blanket excepts is a documented anti-pattern in this codebase — project memory records that silent Gemini/SAM failures (e.g. `thinking_config` causing silent failures, SAM3 triton failing silently with 0 detections) bit the team before. In the 'open the refrigerator' G1 render lane these swallows can hide exactly the CPU-detectable regressions this audit wants surfaced (e.g. a stage/articulation read returning None and the render silently degrading). The runner already has a `_log` helper (line 36) used for heartbeat-uploaded progress, so adding diagnostic logging is low-risk and high-value.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), reduce silent error-swallowing in `scripts/run_isaac_g1_kitchen_parity_eval.py`. The runner has 94 bare `except Exception:  # noqa: BLE001` blocks (verified: `grep -c 'except Exception:  # noqa: BLE001' scripts/run_isaac_g1_kitchen_parity_eval.py` => 94). Many are intentional import-fallbacks, but several around stage/articulation/USD reads silently degrade behavior and hide CPU-detectable regressions.

What to do:
1. Produce an inventory: list each `except Exception:  # noqa: BLE001` site with its enclosing function and a one-line classification — (a) legitimate optional-import fallback (leave as-is), (b) swallow that should at least `_log(...)` the exception before continuing, (c) swallow that should be narrowed to a specific exception type or re-raised.
2. For category (b), add a `_log(f"... degraded: {exc!r}")` (the runner already has a `_log` helper at line 36) inside the handler so failures are diagnosable from the heartbeat-uploaded console, without changing control flow.
3. For category (c), narrow the catch to the specific exception(s) actually expected (e.g. AttributeError/KeyError on a stage/articulation read) so genuinely unexpected errors propagate.
4. Prioritize handlers around stage open, articulation reads, and placement/QC — do NOT touch the policy/visual_qc/parity-module import fallbacks (those are deliberate dual-try blocks).
5. Keep changes behavior-preserving for the success path; only the failure path gains logging or a narrower catch.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims; add tests where a now-narrowed handler has testable behavior. Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.

When done, run `grep -c 'except Exception:  # noqa: BLE001' scripts/run_isaac_g1_kitchen_parity_eval.py` to confirm the count dropped, `python -m pytest tests/test_isaac_g1_kitchen_parity_runner.py -q`, and `python -m py_compile scripts/run_isaac_g1_kitchen_parity_eval.py`.
```

</details>

## Visual QC rubrics

### [P2-28] Harden and test JSON/text extraction for cascade output shapes

- **Priority:** P2 · **Effort:** S · **Dimension:** Visual QC rubrics
- **Goal:** Cover all-thinking responses and reasoning-brace preambles so extraction degrades to parsed=False (flagged), not a mis-parsed clean verdict.
- **Files:** `src/blueprint_pipeline/render_visual_qc.py`, `tests/test_render_visual_qc.py`
- **Validate (CPU):** python3 -m pytest tests/test_render_visual_qc.py -k 'extract or parse' -q  &&  python3 -m py_compile src/blueprint_pipeline/render_visual_qc.py

- **Context:** Robust extraction is what makes the gate trustworthy against the model cascade's varied output shapes (gemini-3-flash-preview -> gemini-2.5-pro, lines 29-34; thinking parts filtered per repo notes). An all-thinking response or a double-brace reasoning preamble must degrade to parsed=False (flagged), never to a mis-parsed clean verdict, on the G1 refrigerator seed lane.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add coverage (and a minimal robustness fix if needed) for the visual-QC model-output extractors.

File: src/blueprint_pipeline/render_visual_qc.py. _extract_json_object (lines 149-164) falls back to the FIRST greedy {.*} match (re.DOTALL); a reply with reasoning braces or multiple JSON objects can capture the wrong/merged span. extract_model_text (lines 167-187) returns the first non-thought part. Existing coverage (tests/test_render_visual_qc.py:176, test_extract_model_text_skips_thinking_parts) only covers single-thinking-part-then-answer.

Add tests in tests/test_render_visual_qc.py:
  (a) a response object whose ONLY part has thought=True (build a small fake like the existing R/C/P classes at lines 177-189) -> extract_model_text returns '' and parse_qc_verdict('')['parsed'] is False and verdict_is_flagged(parse_qc_verdict('')) is True;
  (b) response.text being whitespace-only with a valid candidate non-thought part -> extract_model_text returns that part's text (assert the candidate path is used when .text is blank);
  (c) raw text with a reasoning '{ ... }' preamble brace BEFORE the real JSON object -> assert the intended QC object is parsed (e.g. coherent/anomalies present). If (c) currently mis-parses (captures the reasoning brace), apply a minimal fix to _extract_json_object: prefer a json.loads of the LAST balanced {...} or scan candidate brace-spans and pick the first that json.loads to a dict containing an expected QC key; keep it dependency-light and do not break existing tests. If (c) already passes, leave the source unchanged and keep it as a regression test.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims. Run tests and py_compile.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P2-29] Pin sample_frame_paths first/last guarantee and dedup-collision behavior

- **Priority:** P2 · **Effort:** S · **Dimension:** Visual QC rubrics
- **Goal:** Parametrize tests proving first+last are always present and document/handle dedup collisions returning fewer than sample_n frames.
- **Files:** `src/blueprint_pipeline/render_visual_qc.py`, `tests/test_render_visual_qc.py`
- **Validate (CPU):** python3 -m pytest tests/test_render_visual_qc.py -k 'sample' -q  &&  python3 -m py_compile src/blueprint_pipeline/render_visual_qc.py

- **Context:** Frame sampling sets the gate's coverage and cost for the G1 refrigerator render lane; qc_render_frames/qc_robot_placement_frames/qc_manipulation_pov_frames all call sample_frame_paths (lines 504, 529, 564). The first and especially the last frame are most likely to show end-state manipulation defects, so a silent dedup dropping the last frame would be a coverage hole no current test catches.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), add coverage for the frame-sampling that decides which frames the paid VLM reviews.

File: src/blueprint_pipeline/render_visual_qc.py, sample_frame_paths (lines 352-362). It builds indices via round(i*(n-1)/(sample_n-1)) then sorts the UNIQUE set, so some n/sample_n combos return FEWER than sample_n frames (rounding collisions). The 'always includes first + last' claim (docstring line 353) is only tested for n=10,sample_n=3 (tests/test_render_visual_qc.py:107).

This is TEST-ONLY (do not change behavior unless a test reveals an actual dropped-last-frame bug; if found, report + fix minimally and keep existing tests green). Add a parametrized test asserting: for n in {2,3,5,7} and sample_n in {2,3,4} with n>sample_n, the result's first element equals paths[0] and last equals paths[-1] (the worst end-state defect is most likely in the last frame, so last must never be dropped); also add an explicit case documenting that len(result) may be < sample_n due to dedup collisions (assert len(result) <= sample_n and that no index repeats). Cover sample_n==2 returns exactly [first,last].

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims. Run tests and py_compile.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

## Catch-all / completeness

### [P2-30] Untrack tools/splat_render/node_modules (1693 committed files, 61% of repo)

- **Priority:** P2 · **Effort:** S · **Dimension:** Catch-all / completeness
- **Goal:** git rm --cached the vendored node_modules so it stops bloating clones, diffs, and code search, while keeping files on disk and restorable via npm ci.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/.gitignore`, `$HOME/workspace/BlueprintCapturePipeline/tools/splat_render/package-lock.json`
- **Validate (CPU):** git rm -r --cached tools/splat_render/node_modules; git status --short | grep -c '^D' (~1693 deletions staged); ls tools/splat_render/node_modules >/dev/null (files still on disk); git ls-files | grep -x tools/splat_render/package-lock.json (present); python -m pytest -q -o addopts='' unaffected.

- **Context:** Verified: `git ls-files | grep -c tools/splat_render/node_modules` = 1693; total tracked = 2786; both tools/splat_render/package.json and package-lock.json exist. Vendored node_modules dominates the file count, pollutes diffs/search, and can carry transitive-dependency CVEs that look like first-party code. .gitignore already declares intent to ignore it — the tracking is a pre-ignore leftover.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), 1693 files under tools/splat_render/node_modules/ are tracked out of 2786 total tracked files (61%). The path IS in .gitignore (last line: tools/splat_render/node_modules/) but the files were committed before the ignore rule was added (commit c77a52a2). A package.json AND package-lock.json both exist under tools/splat_render/, so `npm ci` can fully rebuild it.

Do this:
1. `git rm -r --cached tools/splat_render/node_modules` (keeps files on disk, stages 1693 deletions).
2. Confirm `git status` shows ~1693 deletions staged and the files still present on disk (`ls tools/splat_render/node_modules | head`).
3. Confirm tools/splat_render/package-lock.json is tracked so `npm ci` restores the exact tree; if a short README/build note in tools/splat_render/ documents setup, add one line noting `npm ci` is required after clone.
4. Do not modify the JS source under tools/splat_render outside node_modules.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims. No Python behavior changes — confirm with `python -m pytest -q -o addopts=''` (should be unaffected). This is a pure git index operation.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P2-31] Remove stray tracked run-artifact wam_provider_output.json from repo root

- **Priority:** P2 · **Effort:** S · **Dimension:** Catch-all / completeness
- **Goal:** Untrack the committed run dump at the repo root that leaks a machine-local absolute path and a transient 'blocked' state.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/wam_provider_output.json`, `$HOME/workspace/BlueprintCapturePipeline/.gitignore`
- **Validate (CPU):** grep -rl wam_provider_output.json src tests scripts (returns nothing referencing it as a fixture); git rm --cached wam_provider_output.json; git status --short shows it staged for deletion / untracked; grep -F 'wam_provider_output.json' .gitignore returns a match; python -m pytest -q -o addopts='' passes.

- **Context:** Verified: `git ls-files | grep wam_provider_output.json` -> tracked at root. The run dirs output/, runs/, robot_eval_jobs/ are correctly gitignored; this file is the lone committed run-dump leak. It leaks a machine-specific absolute path and a meaningless transient 'blocked' state, and sets a bad precedent for committing run dumps.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), wam_provider_output.json is tracked at the repo root (committed in 91ed947c). It is generated runtime output: schema oscar_wam_provider_command_adapter.v1, status='blocked' with blocker 'blocked_missing_BLUEPRINT_WAM_ROLLOUT_INPUT', and an absolute machine-local work_dir under $HOME/workspace/.../robot_eval_jobs/. It is output, not source.

Do this:
1. Confirm no source/test references it: `grep -rl wam_provider_output.json src tests scripts` should return nothing (it is a default output filename, not a fixture). If something writes to it as a default path, leave the code but ensure the default lands in an already-gitignored run dir.
2. `git rm --cached wam_provider_output.json` and add `wam_provider_output.json` (or `/wam_provider_output.json`) to .gitignore so it doesn't get re-added.
3. Confirm `git status` shows it untracked/removed-from-index.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (a leaked machine-local absolute path is exactly the kind of provenance/PII leak to remove); render outputs are simulator support, NOT policy-success claims. No code behavior change expected — run `python -m pytest -q -o addopts=''` to confirm nothing depended on the file.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P2-32] Collapse legacy agent_skills/ duplicates into pointers to the canonical skillpack

- **Priority:** P2 · **Effort:** S · **Dimension:** Catch-all / completeness
- **Goal:** Replace the 7 self-described-legacy markdown notes under agent_skills/ with one-line pointers (or delete them) so the only source of truth is skillpacks/industrial_readiness/skills/.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/agent_skills/README.md`, `$HOME/workspace/BlueprintCapturePipeline/agent_skills`, `$HOME/workspace/BlueprintCapturePipeline/skillpacks/industrial_readiness/skills`
- **Validate (CPU):** For each file, `diff agent_skills/<name>.md skillpacks/industrial_readiness/skills/<name>/SKILL.md` shows divergence (justifying removal); after the change either the files are git-removed or are <=2 lines pointing to the canonical path; grep -rl 'agent_skills/' src tests scripts returns nothing; python -m pytest -q -o addopts='' passes.

- **Context:** Verified: agent_skills/README.md declares the files legacy and names skillpacks/industrial_readiness/skills/ as canonical; that skillpack dir contains all the corresponding skills (blocker_taxonomist, capability_envelope_writer, evidence_auditor, intake_normalizer, readiness_report_writer, recapture_planner, standards_retriever, plus more). Stale duplicated docs drift from the canonical skillpack and confuse human and agent readers. This is the notes-drift-vs-reality item: the notes say 'legacy' but the files are still full-content duplicates.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), agent_skills/ contains a README.md plus 7 .md files (blocker_taxonomist.md, capability_envelope_writer.md, evidence_auditor.md, intake_normalizer.md, readiness_report_writer.md, recapture_planner.md, standards_retriever.md). agent_skills/README.md itself states these are 'legacy drafting notes' and that the canonical sources now live under skillpacks/industrial_readiness/skills/ (and skillpacks/blueprint_operating_system/skills/), synced into .claude/skills/ and .agents/skills/. So each skill exists in multiple places and agent_skills/ is self-declared stale.

Do this:
1. For each agent_skills/<name>.md, diff it against skillpacks/industrial_readiness/skills/<name>/SKILL.md to confirm it is a full-content duplicate that has diverged.
2. Decide per file: either `git rm` it, or replace its body with a one-line pointer like 'Moved. Canonical source: skillpacks/industrial_readiness/skills/<name>/SKILL.md'. Apply the SAME choice consistently across all 7 (prefer stubs if any external doc links to these paths; otherwise delete and keep only README.md as the pointer).
3. Update agent_skills/README.md if you delete the files so it doesn't reference removed paths.
4. Confirm nothing in code imports these (they are markdown): `grep -rl 'agent_skills/' src tests scripts` should return nothing.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth; render outputs are simulator support, NOT policy-success claims. These are docs, so pytest is unaffected — but run `python -m pytest -q -o addopts=''` to confirm no test reads these paths.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

### [P2-33] Add a CPU secret-scan CI step to lock in the clean secrets posture

- **Priority:** P2 · **Effort:** S · **Dimension:** Catch-all / completeness
- **Goal:** Add a gitleaks (or trufflehog) CI step so the verified-clean secrets posture stays clean and the two tracked .env.example files are guarded against ever gaining a real value.
- **Files:** `$HOME/workspace/BlueprintCapturePipeline/.github/workflows/ci.yml`, `$HOME/workspace/BlueprintCapturePipeline/configs/native_runtime_vast.env.example`, `$HOME/workspace/BlueprintCapturePipeline/deploy/systemd/pipeline-control-plane.env.example`, `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/model_access_env.py`
- **Validate (CPU):** git ls-files | grep -E '\.env' returns only the two .example files; `grep -riE 'sk-[A-Za-z0-9]{20}|hf_[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16}|-----BEGIN' $(git ls-files)` returns nothing; the new ci.yml step is present and (if gitleaks is installed) a local `gitleaks detect --no-banner` exits 0.

- **Context:** The dimension explicitly asks to verify no secrets in repo/logs and that ~/.blueprint-secrets is file-based — all confirmed clean. The one place to keep watching is the two tracked .env.example files, which must stay placeholder-only. A CPU-only gitleaks/trufflehog CI step provides ongoing assurance at zero spend and converts a one-time manual audit into a durable guard.

<details><summary>Prompt (copy into a fresh session)</summary>

```text
In the BlueprintCapturePipeline repo (cwd = repo root), the secrets posture is currently clean and verified: all working-tree .env* files are gitignored; the only tracked env files are configs/native_runtime_vast.env.example and deploy/systemd/pipeline-control-plane.env.example, which contain ONLY commented placeholders; ~/.blueprint-secrets is file-based (read in src/blueprint_pipeline/model_access_env.py:23-36 and runpod_wam_async_runner.py) and never committed. Goal: keep it that way with an automated CPU-only gate.

Do this:
1. Add a secret-scan step to .github/workflows/ci.yml using gitleaks (e.g. gitleaks/gitleaks-action) OR trufflehog filesystem mode — whichever is simplest to pin. Configure it to scan the working tree (and ideally git history) and fail on findings.
2. If gitleaks needs a config, add a minimal .gitleaks.toml that does NOT allowlist real secret patterns but DOES allow the two known placeholder .env.example files (which contain only commented `# KEY=` lines).
3. Verify locally if the tool is available: run a one-off scan and confirm zero findings against the current tree. If the tool is not installed locally, document the exact CI step and run the equivalent grep check below to prove the tree is clean now.

Constraints: keep world-model backends swappable; protect provenance/rights/privacy/raw-capture-truth (the scan is itself a privacy/secrets safeguard — do NOT broaden allowlists to mute real findings); render outputs are simulator support, NOT policy-success claims. No production code changes. Run `python -m py_compile` on any edited Python (none expected) and confirm CI yaml is valid.

Do NOT launch any GPU or paid cloud pod; this is CPU/no-spend only.
```

</details>

---
