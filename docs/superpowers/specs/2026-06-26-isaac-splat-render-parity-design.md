# Isaac Sim Splat-Render Parity — Design Spec

Date: 2026-06-26
Status: Approved (brainstorming → implementation)

## Goal

Make the **Isaac Sim lane actually render the real captured Gaussian-splat scene**
into its evaluation image/video artifacts. Today the MuJoCo lane cannot display
`.ply`/`.usd`/`.spz` at all (its own probe: MuJoCo "loads OBJ triangle meshes but
does not provide a PLY or SPZ mesh decoder"; it falls back to MJCF box proxies),
and the Isaac lane only *references the splat as USD metadata* and never renders it
(`camera_evidence_status: "blocked_until_isaac_or_splat_renderer_runs"`).

Parity flip: where MuJoCo emits primitive box proxies, the Isaac lane emits
**real renders of the captured environment**.

## Locked decisions

1. Target = **Gaussian splat rendering** (`.ply` / `.spz`), not textured-USD.
2. Win condition = **local-first, reproducible**, with **Isaac-on-GPU as the final proof**.

## Ground truth (verified)

- Production splat format = **PlayCanvas `splat-transform 0.1.3` compressed PLY**:
  3 elements — `chunk` (per-256-splat AABBs for position/scale/color),
  `vertex` (`packed_position`/`packed_rotation`/`packed_scale`/`packed_color` uint32),
  `sh` (45 × uchar higher-order SH). Example + pipeline-native `scene.ply` are both this.
  `.spz` = gzip-wrapped Niantic SPZ (same splat, alternate encoding).
- The web app already renders these via `@sparkjsdev/spark` (Spark.js / three.js) —
  a known-good reference renderer with a parametric camera (`ExactSiteSparkViewer.tsx`).
- Local tooling: Python 3.9 + numpy (no torch/PIL/plyfile); node 18; ffmpeg;
  Playwright Chromium already installed. No splat-decode code exists in the repo yet.
- Isaac's only *native* splat render path = NVIDIA **NuRec / RTX-3DGUT** (Isaac Sim 5.0+),
  GPU-only, requires the renderer/extension present in the worker image.

## Architecture

### Phase 1 — local, reproducible (no GPU)

1. **`gaussian_splat_decode.py`** — decode compressed PLY (and SPZ) → standard INRIA
   3DGS arrays and write a standard float PLY (`x,y,z, f_dc_0..2, f_rest_*, opacity,
   scale_0..2, rot_0..3`). Unpacking math **ported verbatim from the canonical
   Spark.js decoder** and cross-validated against it on real data. numpy-only.
2. **`splat_scene_render.py`** + a self-contained node harness — drive headless
   Chromium + Spark.js to load the splat and render from the eval's **6 camera poses**
   (`head_pov, torso, wrist, third_person, overhead, task_focus`) → per-camera PNGs →
   ffmpeg → MP4. Reuses the known-good renderer.
3. **Wire into `isaac_g1_site_3dgs_realistic_eval.py`** — replace the metadata-only
   `GaussianSplatVisualSource` + `blocked_until_..._renderer_runs` with **real rendered
   artifacts**, truthfully labeled `rendered_by: reference_spark_renderer` (NOT Isaac RTX).

### Phase 2 — Isaac-GPU final proof (gated) — VERIFIED MECHANISM

Authoritative path (confirmed against Isaac Sim 5.0/6.0 docs + NVIDIA `nv-tlabs/3dgrut`):
Isaac renders 3DGS via the RTX **NuRec / 3DGUT** path; an existing trained PLY is made
Isaac-renderable with 3dgrut's standalone transcoder (no retraining):

```
python -m threedgrut.export.scripts.transcode <std.ply> -o <out.usdz> --format nurec       # NuRec USDZ (5.0+)
python -m threedgrut.export.scripts.transcode <std.ply> -o <out.usd>  --format lightfield   # ParticleField (6.0 preferred)
```

The USDZ/USD loads via `add_reference_to_stage` / `omni.usd open_stage` and renders on the
RTX renderer. Built:

4. **`isaac_nurec_export.py`** — `convert_ply_to_isaac_usd()` wraps the 3dgrut transcode
   above; fail-closed (`threedgrut_unavailable` when 3dgrut is absent); location-agnostic.
5. **`scripts/run_isaac_splat_nurec_render.py`** — the GPU-worker runner: transcode (if
   needed) → open USDZ → author the eval's 6 cameras → RTX-render each (NuRec/3DGUT
   subframes) → PNG + MP4 → upload via the provider signed-PUT contract. Phase-logged like
   the existing `run_lightwheel_kitchen_isaac_scenarios.py`.
6. Run on RunPod reusing Codex's proven boot/teardown/spend harness (image
   `nvcr.io/nvidia/isaac-sim:6.0.0`). **Gates** (honest): (a) `3dgrut` must be importable on
   the worker (install in image or at runtime), (b) the live render needs a warm/cached pod
   (fresh on-demand still times out on the 10.6GB image pull), (c) real GPU spend.

Truth labeling: Phase-1 frames are `rendered_by: reference_spark_renderer`,
`rendered_by_isaac_rtx: false`; Phase-2 frames are `rendered_by_isaac_rtx: true`. Neither
proves physics, navigation, G1 control, or readiness.

## Component interfaces

- `decode_compressed_ply(path) -> SplatData` (numpy arrays: xyz, scales, quats, sh0/dc,
  shN, opacity, count). `write_standard_3dgs_ply(SplatData, out_path)`.
- `decode_spz(path) -> SplatData` (gunzip + documented SPZ layout). Secondary.
- `render_splat_views(splat_path, cameras, out_dir, *, width, height) -> RenderResult`
  (per-camera PNG paths + MP4 path + status/blockers; truthful `rendered_by`).
- `author_nurec_usd(standard_ply, out_usd) -> dict` (Phase 2).

## Data flow

compressed `.ply`/`.spz`
  → (Phase 1 render) Spark.js headless → per-camera PNG + MP4 → Isaac eval artifacts
  → (Phase 1 decode) standard 3DGS `.ply`
  → (Phase 2) NuRec USD → Isaac RTX-3DGUT on GPU → per-camera PNG + MP4 (Isaac-native proof)

## Success criteria

- **Phase 1:** from the example compressed PLY, the lane produces (a) a validated
  standard 3DGS PLY, (b) per-camera PNGs + MP4 that *visibly show the real interior*,
  (c) wired into the Isaac eval output with honest labeling, (d) fully reproducible via
  `pytest` + a CLI, zero GPU. Renders are non-blank and depict the captured scene.
- **Phase 2:** a RunPod Isaac run that loads the NuRec USD and RTX-renders the splat
  from the same cameras, with teardown/spend proof.

## Testing

- Decoder: cross-validate a sample of decoded splats against the canonical Spark.js
  decode (tiny node oracle) within tolerance; assert count, finite values, plausible
  AABB; round-trip header of the standard PLY. Run against the real `scene.ply`.
- Renderer: integration test asserts 6 non-blank PNGs (variance/entropy threshold) +
  a playable MP4 from the real scene; harness failure → explicit blocker, never a fake pass.
- Eval wiring: artifact-contract test that `camera_evidence_status` flips to rendered and
  labeling is truthful; absent-renderer path stays fail-closed.
- NuRec authoring: structure test on the emitted USD.

## Scope (YAGNI)

- **In:** displaying the real captured scene as images/video in the Isaac lane (both the
  local Spark proof render and the Isaac NuRec GPU render).
- **Out:** splat→collision/physics; rendering the G1 *inside* the splat (Phase 2 nicety);
  SPZ beyond a secondary decoder (compressed PLY is primary).

## Adaptability — swappable 3DGS backends

Per `WORLD_MODEL_STRATEGY_CONTEXT` (keep backends swappable), splat frameworks plug in
through `splat_backends.py`, a registry with a uniform fail-closed `run()` + `available()`
contract across four kinds: **decoder / renderer / exporter / enhancer**. Built-ins:

| backend | kind | what |
| --- | --- | --- |
| `splat_transform` | decoder | compressed PLY/SPZ → standard 3DGS PLY |
| `spark` | renderer | local headless three.js reference render |
| `threedgrut` | exporter | standard PLY → NuRec USDZ / ParticleField USD (needs ncore, GPU image) |
| `particlefield_usd` | exporter | standard PLY → `ParticleField3DGaussianSplat` USD in pure pxr (no ncore/3dgrut) |
| `isaac_nurec` | renderer | Isaac RTX/NuRec render (GPU worker) |
| `artifixer` | enhancer | NVIDIA ArtiFixer diffusion artifact-fix / novel-view frames (GPU) |

### Phase-2 clean path (verified)

Isaac Sim 6.0 RTX renders the OpenUSD **`ParticleField3DGaussianSplat`** schema (UsdVol)
natively — the preferred, non-deprecated splat schema. `particlefield_usd.py` authors it
**directly from our standard 3DGS PLY in pure Python/pxr** (conventions from 3dgrut:
`scales=exp`, `orientations=normalized quat (w,x,y,z)`, `opacities=sigmoid`, raw SH
coefficients, degree set to match). This avoids the entire ncore/3dgrut/NRE chain (NRE is
AV-rig-oriented and cannot render a standalone PLY from free cameras). A real 1.24M-splat
ParticleField USD has been authored + validated locally. Live render = ship that USD +
`cameras.json` to a **base Isaac 6.0** RunPod pod and RTX-render the 6 free cameras via the
runner — the only remaining GPU-gated step.

Adding a framework (a new renderer, exporter, or enhancer like ArtiFixer) =
`register_backend(SplatBackend(...))` with an availability probe and a fail-closed `run` —
no core change. `list_backends(kind)` reports what's installed/available in the current env.

## Task-aware framing (scene → views + robot start)

`splat_scene_analysis` recovers reusable geometry (up-axis, floor, footprint, free-space)
that feeds BOTH camera framing and robot placement. Both are optionally **task-aware**:
`suggest_robot_start(..., task_target=...)` biases the start to a standoff from the task's
focus and faces it; `derive_eval_cameras(..., focus_point=...)` aims `task_focus`/`wrist`
at the task region. Without a target, both default to scene-centered behavior.

## Productionized live render (reproducible for any capture)

The whole live Isaac render is a repo module, not scratch scripts:
**`isaac_particlefield_render_job.py`** (CLI `blueprint-render-splat-isaac`). One call runs:
capture splat → standard PLY (`gaussian_splat_decode`) → ParticleField USD
(`particlefield_usd`) → 6 free cameras (`splat_scene_analysis`) → bundle + stage to DO
Spaces signed URLs (`wam_provider_object_store`) → RunPod base-Isaac pod runs
`scripts/run_isaac_splat_nurec_render.py` via a hardened, diagnostics-streaming bootstrap
(warm-host restart first, else cold on-demand) → RTX frames + MP4 uploaded → poll +
teardown. Paid launches are gated behind `--allow-paid` (default: prepare + stage + return
a launchable plan, no spend). Hermetic tests cover the bundle/launch-request/bootstrap
construction; the runner was hardened against an adversarial pre-spend review
(camera-xform via `AddTransformOp`, ParticleField schema assertion, single-GPU, BasicWriter
+ pixel-variance gate). The bootstrap writes scripts to files (no stdin tricks) and emits a
system-python "container_bash_started" marker so "container never started" is distinguishable
from "Isaac failed".

**Live-render infra status (honest):** the render *code* is complete + locally verified;
the remaining gate is RunPod startup on the 10.7GB Isaac image — warm-pod restarts return
HTTP 200 but the container runtime frequently never attaches (`runtime: none`, no marker),
and fresh on-demand pods must pull 10.7GB before the container starts. This is the same
external infra issue the prior RunPod thread fought; it is not a code defect. A reusable
image with the renderer pre-warmed (or a slimmer image) is the durable fix.

## Provider-agnostic launch (RunPod and Vast)

The launch is abstracted behind one interface so the same render job runs on either GPU
provider without changing the bundle, the runner, or the watch loop:

- **`gpu_render_providers.py`** — a provider-neutral `RenderLaunchSpec` (image + env +
  bootstrap command + GPU sizing) and a `GpuRenderProvider` registry with two backends:
  - `runpod` — REST pods; warm-host restart first (no image pull), else cold on-demand create.
  - `vast` — search RT-capable offers under an hourly rate, then create an instance from the
    chosen ask with the bootstrap as an `args` onstart. Reuses the proven Vast API mechanics
    in `vast_provider_adapter` (`_search_payload` / `_select_offer` / `_create_payload` /
    `_api_json`) so offer selection lives in one place; `_select_offer` already drops
    non-RT GPUs and prefers RT-capable ones (required for splat RTX rendering).
- The **bundle/output transport is provider-neutral** by design — signed GET/PUT URLs ride
  inside `spec.env` and the bootstrap fetches/uploads through them — so only *launch* and
  *stop* differ per provider. `watch_and_collect` polls the same signed output URL and tears
  down via `provider.stop(instance_id)`.
- `run_isaac_particlefield_render_job(..., provider="runpod"|"vast")` and the CLI flag
  `--provider` select the backend; `--list-providers` reports which credentials are present.
  Secrets stay file-based (`~/.blueprint-secrets/runpod_api_key`, `vast_api_key`) and are
  never logged. Both launch/stop are fail-closed when their key is absent (no spend, no net).

Adding a third provider (e.g. Lambda) is a new `GpuRenderProvider` subclass implementing
`available` / `build_request` / `launch` / `stop` plus a registry entry — no change to the
render job, bundle, or runner. The chosen scope is **cold-create per render** (no warm pool);
a persistent warm worker can layer on later behind the same interface.

**MuJoCo:** the launch layer is also *renderer*-neutral — the bootstrap simply fetches a
bundle and runs whatever runner it contains. A MuJoCo variant is therefore a sibling bundle
(MuJoCo runner + its own bootstrap) launched through the identical RunPod/Vast providers;
no separate launch code is needed. (MuJoCo still cannot display the splat itself — that is
the parity gap this whole lane exists to close — so a MuJoCo "version" is only for the
box-proxy eval lane, built if/when needed.)

## Risks / honesty boundaries

- Phase 2's *live* GPU render depends on external RunPod infra + NuRec being present in
  the image (unverified) + real spend. All Phase-2 *code* is built and locally validated
  to the boundary; the live render is attempted but cannot be guaranteed "mistake-free"
  because it is not fully in-process. Proof labeling stays truthful at every step.
- Headless WebGL + Spark load is the one Phase-1 assumption validated first.
