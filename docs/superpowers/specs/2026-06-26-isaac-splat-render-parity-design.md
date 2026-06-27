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

### Phase 2 — Isaac-GPU final proof (gated)

4. **`isaac_nurec_usd.py`** — standard 3DGS PLY → **Isaac NuRec USD** asset (exact
   Isaac Sim 5.0 NuRec schema confirmed against source/docs at build time).
5. Reference the NuRec USD in the eval scene and run on RunPod with **RTX-3DGUT**,
   rendering the splat from the same cameras; reuse the existing RunPod launch /
   teardown / spend-proof harness. **Gated** on confirming the worker image contains NuRec.

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

## Risks / honesty boundaries

- Phase 2's *live* GPU render depends on external RunPod infra + NuRec being present in
  the image (unverified) + real spend. All Phase-2 *code* is built and locally validated
  to the boundary; the live render is attempted but cannot be guaranteed "mistake-free"
  because it is not fully in-process. Proof labeling stays truthful at every step.
- Headless WebGL + Spark load is the one Phase-1 assumption validated first.
