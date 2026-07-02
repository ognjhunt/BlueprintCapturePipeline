# G1 Textured Robot Render Noise Audit

Implements the "G1 Textured Robot Render Noise Audit Spec": makes the Isaac robot-POV
seed-frame render path auditable so Blueprint can decide, per task/backend, whether to use a
verified textured robot material, a simplified diffuse material, the white proxy, or a noisy
textured fallback — always with explicit quality labels.

Claim boundary: this is a simulator/render-quality audit. It does not prove physical robot
readiness, task success, contact correctness, policy quality, or WAM rank fidelity.

## What it isolates

Close robot-POV manipulation frames (e.g. kitchen/fridge) can be clean with the untextured
white proxy but grainy/dark/blotchy with textured G1 arms. Texture correctness is not assumed
to be the cause: the audit separates

- missing texture assets / worker asset resolution (`robot_material_resolution_manifest.json`)
- render budget / sample starvation (high-spp variants D/E)
- denoiser path failure (denoised-vs-raw regression at both budgets)
- PBR/specular material response (simplified-diffuse variant F)
- lighting underexposure (boost-light variant G)
- camera/pose/clipping (black-edge-wedge + arm-visibility gates on every variant)
- shader/cache cold-start variance (dedicated measured warmup before the first variant)

## Variant matrix (spec minimum)

| Variant | Robot material | Denoiser | Budget | Purpose |
| --- | --- | --- | --- | --- |
| A | white diffuse proxy | on | current default | known clean proxy baseline |
| B | textured/original | off | current default | raw textured-noise baseline |
| C | textured/original | on | current default | denoiser regression check |
| D | textured/original | off | high spp | test sample starvation |
| E | textured/original | on | high spp | test denoiser with enough samples |
| F | simplified diffuse (sampled base colors) | on | current default | PBR/specular map stability |
| G | textured/original + brighter task lighting | on | current default | shadow/underexposure |

Execution order on the worker is material-monotonic (`B,C,D,E,G,F,A`: authored materials
first, then simplified-diffuse overrides, then the white proxy) so authored materials never
need to be un-authored mid-run; scene, task, stance, camera, arm pose, and resolution are
identical across variants. Declared pass/fail comparisons each isolate exactly one variable
(`validate_variant_plan` enforces this); anything else is exploratory.

Audit render steps (warmup, per-variant settle, capture) run under a dedicated watchdog
(`PARITY_AUDIT_RENDER_STEP_WATCHDOG_SECONDS`, default 900s) instead of the generic 180s
realtime-step watchdog: audit steps are path traced at up to the high-spp budget and the
first warmup frame additionally pays cold shader compile, which exceeded 180s and killed
the 2026-07-02 GPU run at `audit:warmup:0` before any variant rendered.

## Dynamic path (no hardcoded coordinates)

The worker mode reuses the normal seed-render chain in
`scripts/run_isaac_g1_kitchen_parity_eval.py`:

```
task string -> scene-placement target resolution -> task stance plan (+ placement validation)
            -> root placement -> kinematic arms-forward pose -> robot-mounted head camera
            -> camera contract -> render variants
```

Kitchen/fridge is only the first regression case; any task/site that can produce a robot POV
seed frame can be audited.

## How to run

```bash
# 1. Prepare (no GPU spend) or launch the audit on RunPod/Vast:
python scripts/run_g1_render_noise_audit.py launch \
    --task "open the fridge door" \
    --out-dir output/g1_render_noise_audit_fridge \
    --kitchen-url "$STAGED_KITCHEN_URL" \
    --warm-candidate <stopped-pod-id> \
    --allow-paid

# 2. Re-analyze a collected run locally (no GPU):
python scripts/run_g1_render_noise_audit.py analyze --run-dir output/g1_render_noise_audit_fridge/render_output

# 3. Inspect the default variant plan:
python scripts/run_g1_render_noise_audit.py plan --out /tmp/plan.json
```

Equivalent lower-level entry points: `python -m blueprint_pipeline.isaac_g1_kitchen_parity_job
--render-noise-audit ...` (job) and `--render-noise-audit` on the GPU runner itself.

## Outputs

Worker (`render_noise_audit/` in the collected output):

- `audit_run_manifest.json` — task, target resolution, stance summary, placement validation,
  robot asset resolution, camera contract, pose-constant arm/end-effector visibility,
  variant plan + per-variant execution records, measured warmup timings, lighting inventory,
  GPU/driver/Isaac/image identity
- `robot_material_resolution_manifest.json` — every robot material, shader ids, texture asset
  refs with authored/resolved paths + existence, missing-ref and unbound-gprim counts
- `render_settings_manifest.json` — renderer, default/high spp, denoiser/firefly, resolution,
  lighting summary, runtime metadata
- `camera_contract.json` — source (authored/derived/fallback), pose, pitch, clip range, intrinsics
- `variants/<ID>/frame_raw.png` (+ `variant_manifest.json`, optional robot instance mask)

Analysis (worker-side best effort, always reproducible locally):

- `textured_robot_render_noise_audit_manifest.json` — per-variant frame stats
  (`mean_luma`, `std_luma`, `dark_pixel_ratio`, `edge_density`,
  `high_frequency_noise_estimate`, `black_edge_wedge_ratio`, center-crop stats), seed-frame
  gates, denoiser-regression checks, spec interpretation rules with a `primary_diagnosis`,
  and required-recorded-input coverage
- `render_noise_audit_frame_stats.json`, `render_noise_audit_contact_sheet.png`

## Material-mode honesty and WAM use

`blueprint_pipeline.g1_render_noise_audit` defines the only allowed robot material labels:

- `verified_textured` — textured variant AND the material resolution manifest proves texture
  refs exist with zero missing; never granted otherwise
- `textured_unverified` — textured render whose texture refs did not fully resolve; may never
  be presented as textured material fidelity
- `simplified_diffuse`, `white_proxy` — explicit visual proxies

`build_wam_seed_media_contract(...)` encodes the WAM-use gates: proxies are allowed for
short-term WAM conditioning only with the simplified-robot boundary recorded; noisy textured
frames are allowed only when visual smoke accepted them and the noisy/textured status is
recorded; `seed_frame_visual_quality_status` must be `completed`.
`normalize_legacy_robot_material_mode(...)` maps the pre-existing pipeline labels
(`neutral_matte_untextured_g1`, `preserve_authored_g1_materials_when_available`) onto these
modes; `kitchen_task_scaling_preflight` now records the normalized label alongside the legacy
one.

## 2026-07-02 fridge audit result (first GPU run of this harness)

The "open the fridge door" audit on RunPod (RTX 4090 class, 1280x960) completed all seven
variants and diagnosed **`render_budget_sample_starvation`**: default-budget 64-spp textured
variants B/C came back starved/black (`dark_pixel_ratio` 1.0) while the same scene/stance/
camera at 384 spp (D/E) was clean (E: `high_frequency_noise_estimate` 0.56 vs D raw 5.8),
at ~11 s/frame render cost. Secondary findings (all downstream of the starved baseline):
`pbr_specular_material_response` (F clean at the same budget), `lighting_underexposure`
(G clean at the same budget), and `white_proxy_bounded_workaround_available` (A clean).
The material-resolution manifest also proved the stock Isaac `G1/g1.usd` ships **zero
texture asset references** (four OmniPBR `DefaultMaterial`s, 12/81 gprims unbound), so
`verified_textured` is unreachable for this asset and textured renders must be labeled
`textured_unverified`.

Production change: `DEFAULT_PATH_TRACING_MIN/MAX_SAMPLES_PER_PIXEL` in
`scripts/run_isaac_g1_kitchen_parity_eval.py` raised from 64/128 to 384/512 — the
audit-proven clean budget for path-traced manipulation/verify review frames. Measured
warmup also showed the first path-traced frame pays ~252 s of cold shader compile, which is
why audit steps run under the dedicated 900 s watchdog documented above.

## Interpretation rules (priority order)

1. all variants show black wedges/missing arms → `camera_pose_clipping`
2. texture refs missing → `missing_texture_assets`
3. proxy baseline A fails → `proxy_baseline_failed`
4. high-sample raw clean but high-sample denoised dark/blotchy → `denoiser_path_failure`
5. high-sample textured clean, default-budget textured noisy → `render_budget_sample_starvation`
6. simplified diffuse clean while full PBR noisy → `pbr_specular_material_response`
7. brighter lighting clean while default noisy → `lighting_underexposure`
8. proxy passes while textured fails → `white_proxy_bounded_workaround_available`
   (bounded workaround; textured fidelity must not be claimed)
