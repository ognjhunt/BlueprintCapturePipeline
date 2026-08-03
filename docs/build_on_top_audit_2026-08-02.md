# Build-on-top audit — 2026-08-02

Status: research audit, verified against primary sources on 2026-08-02.
Trigger: the 506Lenox appearance regression (3,697,507-Gaussian Scaniverse
export silently decimated to ~250k and pushed through a hand-rolled SPZ
decode/render path that misread coordinates at ±238-unit bounds for a ~9-unit
room). The corrective appearance-fidelity contracts are in flight on
`codex/506-five-controller-isaac` (`appearance_fidelity.py`,
`appearance_fidelity_qualification.v1`, `appearance_render_route.v1`,
`appearance_presentation_derivative.v1`, `robot_appearance_composite_contract.v1`).

This document generalizes the lesson: **build from scratch as little as
possible; build on top of existing services and tools**, and identifies the
concrete subsystems to re-point. Evidence labels follow the 2026-08-01 routing
research convention: VF (verified in this repo), EC (external claim fetched
live from a primary source on 2026-08-02), INF (inference).

## 1. Decision rule (first principles)

Blueprint's differentiated core is not rendering, reconstruction, physics, or
GPU plumbing. It is:

1. rights-clean capture truth and provenance,
2. fail-closed contracts, gates, and claim ladders,
3. the routing/qualification kernel that decides which backend may support
   which claim (capability profiles, R0–R8 admission, abstention),
4. site-package orchestration and the customer-facing Task Evaluation Run.

Everything else is commodity. The rule for every subsystem:

- **Moat → custom.** Contracts, gates, routing, provenance, orchestration.
- **Commodity with a maintained upstream → wrap and qualify.** Pin the exact
  version/digest, register it in `docs/runtime_dependency_license_policy.json`
  (permissive-only allowlist, fail-closed), give it a
  `method_capability_profile.v1`, and route it like any other engine. Never
  reimplement its parser/renderer/solver.
- **Commodity without a usable upstream → smallest possible custom shim**, and
  a research-catalog entry so `measurement_research_monitoring` watches for an
  upstream to appear.

The 506 incident is what happens when a commodity subsystem (SPZ decode +
splat rasterization) is hand-rolled: the failure was silent, looked like a
provider-quality problem, and nearly demoted a good reconstruction. The
measurement-adapter fleet (MuJoCo, Drake, SAPIEN, Chrono, Newton, Isaac PhysX
as peer engines behind fail-closed capability profiles) is the in-repo proof
of the opposite pattern working. Isaac is one routed backend among several —
none of the candidates below make Isaac (or any single vendor) mandatory;
each is a lane the router may select when its capability profile and
qualification support the claim.

## 2. Ranked candidates

### C1 — Splat decode + appearance rendering: finish the upstream swap (act now)

- Custom today (VF): main's `gaussian_splat_decode.py` /
  `splat_scene_render.py` / `splat_backends.py` already shell out to
  `splat-transform` for decode — the remaining work is deleting the
  decimation default, landing the in-flight fidelity contracts, and
  retiring the **seven additional independent hand-rolled PLY
  parsers/writers** the sweep found replicating the same pattern
  (`scene_asset_preflight.py`, `external_pointcloud_initialization.py`,
  `reconstruction_geometry_compiler.py`, `local_reconstruction_adapters.py`,
  `g1_site_3dgs_mujoco_preview.py`, `geometry_stage.py`,
  `site_memory_utils.py`).
- Upstream (EC): `@playcanvas/splat-transform` — MIT, v3.2.0 (2026-07-31),
  reads `.ply/.spz/.sog/.ksplat/.splat`, NaN/opacity/box/sphere/floater
  filtering, statistical summaries, GPU render to lossless WebP, `--no-tty`
  (https://github.com/playcanvas/splat-transform). Already pinned at ^2.7.0 in
  `tools/splat_render/package.json` next to Spark.js + Playwright. Niantic's
  `spz` reference codec is MIT with C++/Python/WASM bindings
  (https://github.com/nianticlabs/spz); SPZ v4 announced 2026-05.
  For exact-camera renders with depth, `gsplat` (Apache-2.0, v1.5.3) renders
  arbitrary PLYs at full SH degree inference-only, with depth output and 3DGUT
  support (https://github.com/nerfstudio-project/gsplat) — NuRec itself
  renders via gsplat.
- Move: splat-transform/spz become the only decode oracle; `gsplat` becomes
  the qualified neutral renderer for depth-aware composites outside Isaac;
  the hand-rolled decode shrinks to a conformance test that asserts agreement
  with the upstream oracle (`gsplat_conformance.py` already points this way).
  Full-resolution source is appearance truth; only receipted, qualified
  removals (the in-flight `appearance_fidelity` contract).

### C2 — Robot-in-splat rendering, Isaac lane: use the native vendor path (act now)

- Custom today (VF): `scripts/run_isaac_splat_nurec_render.py` +
  `reconstruction_isaac_worker_bundle.py` build packages around a decimated
  splat; the sibling branch is adding depth-aware compositing by hand.
- Upstream (EC): `nv-tlabs/3dgrut` (Apache-2.0) officially converts generic
  3DGS PLY → Isaac-renderable USDZ (`ply_to_usd`, `transcode`, partitioning
  for >4 GiB); Isaac Sim 5.x renders it natively in RTX via
  `OmniNuRecVolumeAPI` on a `UsdVolVolume` prim — per-ray, so robot-vs-splat
  occlusion is handled in the renderer, no external compositor
  (https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_nurec.html,
  https://github.com/nv-tlabs/3dgrut). Isaac Sim 6.0 migrates the format to
  the OpenUSD **ParticleField** schema (NuRec USDZ deprecated) — this repo's
  `particlefield_usd.py` lane is already aligned with that direction.
- Known limits to encode as gate fields (EC): fp16 transcode default (fp32
  reportedly fails to render in 5.1.0), keep scenes normalized near origin,
  disable DLSS Frame Generation, splats are excluded from SDG AOVs — depth /
  semantic IDs for SDG must come from proxy meshes, and proxy-matte depth
  compositing uses proxy-mesh depth rather than splat depth.
- Non-Isaac lanes keep their own qualified renderers (C1); NRE `serve-grpc`
  (release_26.04) exists for external-simulator RGB service but is RGB-only
  (no depth), so it is not the compositing answer.

### C3 — Collision geometry from splats: stop hand-cooking (act now)

- Upstream (EC): `splat-transform --collision-mesh [smooth|faces]` emits a
  `.collision.glb` from a sparse voxel octree (`--voxel-size`,
  `--voxel-external-fill`, `--voxel-carve`)
  (https://developer.playcanvas.com/user-manual/splat-transform/collision/);
  `3dgrut add_mesh_to_usdz --input_usdz … --mesh_ply … --set_collision`
  embeds it, and `OmniNuRecVolumeAPI` carries up to 4 Proxy meshes used for
  shadows, physics, and SDG semantics. NVIDIA's own PhysicalAI-Robotics-NuRec
  USDZs ship aligned mesh+occupancy the same way.
- The mesh remains a *derived geometry candidate* until the existing collider
  qualification gates pass (per
  `docs/research/interactable_capture_approaches_research_2026-08-02.md`);
  what changes is that its *generation* is upstream, receipted, and pinned.
  Avoid SuGaR (research-only license) and `kaolin/non_commercial`.
- Direct in-repo payoff (VF): `g1_site_3dgs_mujoco_preview.py:377` currently
  ships ~10 hand-authored literal box proxies because a MuJoCo probe found
  "no ply/spz mesh decoder". The upstream route (splat-transform collision
  mesh → trimesh + CoACD/V-HACD convex decomposition) replaces
  furniture-shaped boxes with receipted geometry candidates.

### C4 — GPU provisioning/failover: SkyPilot owns the mechanics, guards keep the money truth (next)

- Custom today (VF): **~16,750 lines of hand-rolled `urllib.request`
  REST/GraphQL clients and lifecycle plumbing** — `vast_provider_adapter.py`
  (6,711), `gpu_render_providers.py` (3,316), `scripts/gpu_spend_guard.py`
  (2,227), `paid_resource_allocator.py` (2,209), `runpod_provider_adapter.py`
  (1,996), `lambda_provider_adapter.py` (1,729),
  `cloud_vm_render_providers.py` (775) — request shaping, auth, pagination,
  polling, retry/backoff, and bespoke HTTP-error classifiers per provider,
  against hardcoded endpoints; remote bootstrap scripts are emitted as Python
  source strings that themselves hand-roll urllib
  (`vast_provider_adapter.py:2772`). `provider_race.py` (1,012) adds its own
  circuit breaker. `safe_outbound_http.py` (542) exists largely to harden
  this surface. The `vastai` CLI is on disk but used only for a version probe.
- Two-step move, first step independent of SkyPilot (INF): (a) collapse the
  six HTTP stacks onto official SDKs (`vastai`, `runpod`) or one `httpx` +
  `tenacity` client (httpx is already an approved component in the license
  policy) — this is the single largest deletion available in the repo;
  (b) adopt SkyPilot for provisioning/failover lifecycle:
- Upstream (EC): SkyPilot v0.13.0 (Apache-2.0, 2026-07-22) — Vast.ai and
  RunPod are first-class in-tree backends (`skypilot[vast]`,
  `skypilot[runpod]`), spot on both; managed jobs give cost-ordered failover
  across regions then clouds with automatic teardown/reprovision — i.e.,
  `provider_race.py` productized (sequential-by-cost rather than parallel
  racing); autostop/autodown executes cluster-side; v0.13.0 adds lifecycle
  hooks (`resources.hooks`) and `resources.max_hourly_cost`; fully local
  operation (set `SKYPILOT_DISABLE_USAGE_COLLECTION=1`)
  (https://docs.skypilot.ai/en/latest/examples/auto-failover.html,
  https://github.com/skypilot-org/skypilot/releases/tag/v0.13.0).
- What must stay custom (VF+EC): the fail-closed spend/teardown *contracts* —
  `require_pre_spend_preflight`, the cumulative spend ledger,
  `pending_teardown.v1`, teardown proof with
  `status_source="provider_api"`, and orphan reaping — because SkyPilot has
  documented INIT-state `sky down` failures and `--purge` merely forgets
  local state (leaked instances are "the user's responsibility";
  https://github.com/skypilot-org/skypilot/issues/3029,
  https://github.com/skypilot-org/skypilot/issues/4590), and it has no
  cumulative cross-run budget. The guard layer re-attaches via the async SDK,
  `sky.status(refresh)`, `event_callback`, lifecycle hooks, plus direct
  provider list calls for reconciliation.
- Per-cloud constraints to encode in the capability profile (EC): RunPod is
  terminate-only (no stop) and overrides container entrypoints; Vast has no
  nested Docker, no post-launch port opening; neither supports multi-node;
  no Windows anywhere.

### C5 — 3DGS training: add a Linux open-trainer arm; treat the Windows lane as at-risk (next)

- Custom today (VF): `scripts/postshot_windows_worker/` exists solely because
  Jawset Postshot is Windows-only; the Teleport/Postshot bakeoff packets and
  frozen scorecard are already prepared.
- Findings (EC): Postshot is still Windows-10+/NVIDIA-only; CLI automation is
  gated to the Studio tier (€39/mo); the EULA allows one device per license
  with no concurrent use, requires an internet connection, and is silent on
  cloud/VM use — a procurement risk for exactly the lane we automated.
  gsplat's MCMC strategy at 3M Gaussians beats the INRIA reference on
  MipNeRF360 (PSNR 29.65 / SSIM 0.89, ~28 min on A100;
  https://docs.gsplat.studio/main/tests/eval.html); nerfstudio + gsplat are
  Apache-2.0, headless, Dockerized. Brush (Apache-2.0, single binary, v0.3.0)
  is the runner-up with bus-factor 1. LichtFeld Studio (GPL-3.0) and
  OpenSplat (AGPL-3.0) fail the permissive-only license allowlist. Scaniverse
  still has no programmatic export API (manual-import provider it remains).
  This also matches `docs/research/capture_reconstruction_sdk_decision_2026-07-30.md`,
  which already names COLMAP + Nerfstudio/gsplat the preferred next local
  reconstruction implementation.
- Move: register `splatfacto`/gsplat-MCMC as a third bakeoff arm on the
  existing frozen scorecard (runs on the ordinary Linux Vast lane — and on
  SkyPilot if C4 lands). If it reaches parity on our capture profile, the
  Windows worker is demoted to a licensed-tool exception lane or deleted.
  No new paid runs are implied by this doc; the arm rides the already-gated
  bakeoff budget.

### C6 — Presentation-only enhancement: two commercially licensed options now exist (when the lane is built)

- Prior audit (VF): `docs/research/reconstruction_enhancement_audit_2026-07-30.md`
  rejected Difix3D+ (non-commercial license) and gated ArtiFixer/Harmonizer.
- Update (EC): NVIDIA **Fixer** (`nv-tlabs/Fixer`, Apache-2.0 code) and
  **Harmonizer** (`NVIDIA/harmonizer`) both publish weights under the NVIDIA
  Open Model License with commercial use allowed; Harmonizer is already wired
  into NRE `serve-grpc --enable-harmonizer` and the `nurec-fixer` skill.
  Difix3D+ and ArtiFixer weights remain non-commercial → still rejected.
- The authority ceiling is unchanged and absolute: presentation derivatives
  (`appearance_presentation_derivative.v1`) never feed policy observations,
  target binding, geometry, collision, evaluation evidence, or ranking.

### C7 — Provenance interchange: C2PA at the export edge only (later)

- Findings (EC): C2PA spec v2.4; ISO/DIS 22144 in flight; c2pa-rs/-python are
  MIT/Apache and embed manifests in JPEG/PNG/HEIC/MP4/MOV/DNG with
  first-class custom assertions and sidecar/remote manifests; **no 3D format
  support exists** (glTF/USD/PLY/SPZ absent from the format bindings); EU AI
  Act Art. 50 (in force 2026-08-02) rewards stamped media. Trust-listed certs
  are purchasable (SSL.com, DigiCert); self-PKI validates but shows
  "unrecognized signer".
- Move: keep the internal ledger authoritative (it is stronger than C2PA's
  model); at the existing buyer-package export gate, embed a manifest in
  customer-facing image/video deliverables carrying one custom
  reverse-domain assertion with the ledger-entry digest + rights snapshot;
  sidecar `.c2pa` for 3D assets; server-side signing; no capture-time iOS
  signing yet.

### C8 — Job transport, retries, and watchdogs: use the queue we already pay for (later)

- Custom today (VF): `robot_eval_job_orchestrator.py` (10,501 lines) runs a
  filesystem inbox queue with hand-rolled claim/lease semantics; ~10 discrete
  `*_watchdog.py` modules; 27 modules define their own backoff/retry loops;
  `pyproject.toml` has zero retry/workflow dependencies — while
  `google-cloud-pubsub>=2.21.0` is already a paid core dependency with a
  1,818-line `pubsub_handoff_listener.py` in place.
- Move (INF): route job handoff through Pub/Sub / Cloud Tasks (already paid
  for), replace bespoke retry loops with `tenacity` and the circuit breaker
  with `pybreaker`; the Postshot worker's self-re-invoking watchdog daemon
  and its ~250-line inline PowerShell provisioning blob become a scheduler
  job plus a prebaked AMI/Packer image. Domain state machines (admission,
  evidence, gates) stay custom; only transport/retry mechanics move. A full
  workflow engine (Temporal) stays deliberately unadopted for now.

## 3. Inventory: custom-commodity code this doctrine retires or shrinks

Full-tree sweep of 2026-08-02 (736 modules / ~607k lines under
`src/blueprint_pipeline/`). Beyond the lane-level candidates above, these are
the code-level instances of the 506 anti-pattern, each with an upstream that
is in most cases *already a declared dependency*:

| # | Hand-rolled today (VF) | Where | Upstream to use |
| --- | --- | --- | --- |
| 1 | Provider REST/GraphQL clients, ~16.7k lines | `vast_provider_adapter.py`, `runpod_provider_adapter.py`, `lambda_provider_adapter.py`, `gpu_render_providers.py`, `paid_resource_allocator.py`, `scripts/gpu_spend_guard.py` | official `vastai`/`runpod` SDKs, `httpx`+`tenacity`; then SkyPilot (C4) |
| 2 | Seven independent PLY parsers/writers | see C1 list | `plyfile` (already installed in the perception GPU image), `trimesh` (already a dep), splat-transform oracle |
| 3 | Two dependency-free PNG encoders | `isaac_review_renderer_canary.py:256`, `object_geometry_stage.py:77` | Pillow (already a core dep; the fallback branch is dead weight) |
| 4 | GLB/glTF binary chunk walkers | `scene_asset_preflight.py:1068`, `mujoco_g1_simulator_command.py:187` | `trimesh` — `external_scene_collision_candidate.py:87` already does it right |
| 5 | USD text scraping + manual USDZ 64-byte zip alignment | `scene_asset_preflight.py:735`, `nurec_openusd_packaging.py:189` | `pxr.Usd`/`Sdf` dependency APIs, `UsdUtils.CreateNewUsdzPackage`/`usdzip` (usd-core already in the geometry extra) |
| 6 | SSIM implemented twice by hand | `heldout_appearance_evaluation_v2.py:94`, `reconstruction_heldout_evaluation.py:173` | `skimage.metrics`/`torchmetrics` — the same file already wraps LPIPS correctly with SHA-pinned weights |
| 7 | Corner detection + NCC matching + fundamental matrix | `pose_image_consistency.py` (the epipolar gate) | OpenCV (`opencv-python-headless` already a dep): `goodFeaturesToTrack`, `findFundamentalMat` |
| 8 | Equirect→perspective projector with manual bilinear | `equirectangular_virtual_rig.py` | `py360convert`, `cv2.remap`, or `ffmpeg -vf v360` (ffmpeg already wrapped in 30+ modules) |
| 9 | COLMAP text-model writer | `reconstruction_colmap_dataset.py:677` | `pycolmap` (execution side is already a clean CLI wrapper) |
| 10 | `.npy` header parser + restricted unpickler | `policy_ranking_wam_validity.py:61` | `numpy.lib.format.read_array(..., allow_pickle=False)` |
| 11 | Depth back-projection surface compiler | `arkit_depth_surface_compiler.py` | Open3D RGBD/TSDF integration |
| 12 | Filesystem job queue + bespoke retries/watchdogs | see C8 | Pub/Sub (already paid), `tenacity`, `pybreaker` |
| 13 | Hardcoded box collision proxies | `g1_site_3dgs_mujoco_preview.py:377` | C3 route (splat-transform → CoACD/V-HACD) |

Remediation order by liability × ease: (1) provider clients, (2) format I/O
(PLY/PNG/GLB/USD), (3) metrics + epipolar, (4) COLMAP writer + USD packaging,
(5) job transport. Each retirement follows the §5 adoption mechanics and the
no-hand-fixes rule: conformance test against the upstream oracle first, then
delete the custom path.

The in-repo template for all of this is
`measurement_newton_rigid_adapter.py`: exact upstream version pins verified
via `importlib.metadata`, real upstream APIs, self-digesting source, result
shaping delegated to the shared adapter contract — and nothing physical
hand-rolled.

## 4. Explicitly keep custom (audited, not drifted into)

- The routing kernel, capability profiles, R0–R8 admission, research catalog
  + release monitoring, abstention machinery.
- All fail-closed gates: consent/rights/takedown, claim ladders, spend
  preflight, teardown proofs, buyer-package readout, license policy.
- Site-package orchestration (`site_package_orchestrator.py`) and evaluation
  pack registry.
- Native ARKit capture (per the 2026-07-30 SDK decision and 2026-08-01
  scanning-SDK verdict: no embeddable 3DGS capture SDK exists; capture truth
  is the moat).
- Considered and not adopted now: generic workflow engines (Temporal/Prefect
  — the orchestrator encodes domain contracts, migration buys nothing yet),
  experiment trackers (W&B/MLflow — the evidence ledger is the product),
  dstack (self-hosted control plane, Vast spot missing).

## 5. Adoption mechanics (how every candidate enters)

Each adopted upstream follows the same admission path — this is the durable
form of "build on top" here:

1. Research-catalog entry (`measurement_method_research_catalog` /
   monitoring) so releases and license changes are diffed automatically.
2. Component registration in `docs/runtime_dependency_license_policy.json`
   (exact `name==version` → SPDX allowlist, fail-closed on new/changed
   components; every candidate above must land there before first
   execution). Soft spot worth fixing while here: license expressions are
   transcribed by hand — generate the observed column with `pip-licenses`
   or `cyclonedx-py` so review diffs only surface genuine changes.
3. A `method_capability_profile.v1` with fail-closed booleans and pinned
   digests; routing selects it per claim; unverified capabilities abstain.
4. A hermetic fast-lane conformance test asserting our shim agrees with the
   upstream oracle on committed fixtures (the PR #180/#181 pattern; no
   hand-fixes rule applies).

## 6. Sequencing

1. Now (free, local): C1 + C2 + C3 — they are one connected lane
   (spz/splat-transform decode → 3dgrut USDZ → native Isaac render → voxel
   collision candidate) and directly unblock a trustworthy 506 composite.
   Fold in inventory items 2–4 (PLY/PNG/GLB) as the conformance tests touch
   those files.
2. Next: C5 bakeoff arm (rides existing bakeoff gates); C4 step (a) SDK
   consolidation of the provider clients; then C4 step (b) SkyPilot spike
   behind `paid_lane_guard` (one bounded canary, teardown-proof reconciler
   first).
3. Then: C6 presentation lane with Fixer/Harmonizer under pinned digests;
   C7 C2PA stamping at the export gate; C8 job-transport migration;
   inventory items 5–11 opportunistically as their modules are next opened.
