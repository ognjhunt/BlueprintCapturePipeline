# Scene 839873 rendering root-cause audit (2026-09-02)

Independent investigation of why the exact configured NuRec appearance
(`sha256:9193a9de…`) renders in Isaac Sim 6.0.1 as a bright, white,
chromatically splattered scene through Blueprint's ParticleField path, and the
smallest upstream-aligned change that lets a Website Quick-10 consume only
proven observations.

ADP item: ADP-009D (Franka policy rehearsal on the sealed scene) via ADP-050
policy/runtime integration. Day gate unblocked: the next Scene 839873 render
probe and, after it, the first observation-integrity-gated Quick-10. Claim
ceiling of everything here: `diagnostic_policy_execution`. Nothing in this
document qualifies controls, physics, or sim-to-real transfer.

## Evidence boundary

What this session could and could not verify:

- **Verified in code, hermetically.** The current Blueprint conversion
  (`particlefield_usd.write_particlefield_usd_from_nurec`, NuRec tensors →
  temporary INRIA PLY → `usd-convert-gsplat 0.1.15`) was executed on a synthetic
  NuRec USDZ carrying the configured asset's pathologies (float16, 58.56% `+inf`
  density logits, albedo tail at 29.19) and compared attribute by attribute with
  NVIDIA 3DGRUT's real `threedgrut.export.scripts.transcode … --format lightfield`
  at revision `a37ef721012dea0f29c0fcfff2d525023b4e854a` (CPU, `ncore` stubbed;
  it is only used for camera export, which transcode disables).
- **Verified from primary sources.** 3DGRUT importer/adapter/LightField writer,
  PLY exporter/importer, NuRec template, SH rotation utilities, and CUDA radiance
  math (`gaussianParticles.cuh`); `usd-convert-gsplat` writer and PLY reader at
  `621017eb`; the Omniverse ParticleField colour-space statement; the NRE gRPC
  contract; Isaac Sim discussion #680.
- **Not available here.** The failing and reference PNGs, receipts, and the
  deterministic `.npz` sample named in the packet were not present in this
  session's upload. Every statement about the frames below relies on the
  packet's reported statistics. No GPU render was run.

## Executive verdict

1. **Blueprint's private NuRec-tensor conversion is not the root cause of the
   rainbow breakup.** On the same input it produces ParticleField attributes
   that are bit-identical to NVIDIA's direct transcode for positions and all 16
   SH coefficient triplets, and within float32 rounding (≤1.2e-7) for scales,
   orientations and opacities. The packet's decision-table row "native NRE good,
   direct transcode good, current Blueprint bad" is therefore falsified at the
   attribute level before any GPU is spent. The remaining differences between the
   two routes are prim-level metadata only: `projectionModeHint=perspective`,
   `sortingModeHint=cameraDistance`, `colorSpace:name=srgb_rec709_display` and a
   `rtx:post:tonemap:op=2` layer render setting authored by 3DGRUT and absent
   from the current asset. Of these, only the sorting hint can change pixels
   today (Omniverse states RTX does not yet honour `ColorSpaceAPI`).
2. **The leading unisolated cause is the configured appearance itself as seen
   from the policy cameras**: an Artifixer-edited radiance field with a large,
   never-render-qualified coefficient tail (max DC coefficient 29.19 → radiance
   8.7× display white before the view-dependent bands; 15.8–16.6% of Gaussians
   outside display range at the policy poses), authored without cameras, and
   viewed from wide-FOV robot poses that no packaged training rig covers.
   Whether that field is bad everywhere (appearance defect) or only away from
   its teacher views (camera-domain defect) is the question the A/B/C/D render
   settles; conversion arithmetic cannot.
3. **Three genuine code defects were confirmed and are fixed in this PR**: the
   Isaac launcher selected the scientific backend by a default argument and the
   policy canary worker relied on it; the pre-policy visual receipt hard-coded
   `candidate_policy_loaded=False` while the session loads each policy before
   its first episode; and a structural frame pass was the only thing between a
   render and a paid policy query. A fourth, production-only trap was found that
   would have wasted the next GPU: the runtime validator refuses the
   `projectionModeHint`/`sortingModeHint` that NVIDIA's own transcode authors,
   so variant C of the experiment would have been blocked at
   `native_task_arena_particlefield_nonstandard_render_hint` before rendering.
4. **The packet's premise that NuRec is a post-activation format is wrong.**
   3DGRUT's `transcode.py` and `NuRecUSDImporter.stores_preactivation == True`
   both treat `.nurec` state as pre-activation (logit densities, log scales),
   exactly as Blueprint's codec does. Nothing in the failure follows from an
   activation mismatch.

Rendering readiness stays **blocked** (score unchanged at 2/10). What changed is
where the next dollar goes: not into conversion arithmetic, but into one native
same-pose reference render and a camera-domain check.

## Ranked root causes

| Rank | Hypothesis | Confidence | Evidence for | Evidence against | Falsification test |
|---|---|---|---|---|---|
| 1 | The configured appearance's radiance is invalid or only valid near its Artifixer teacher views (large coefficient tail, no view coverage of policy poses) | High that it contributes; unknown whether sole cause | Albedo/specular differ from the retained PLY (0.01% / 22.12% exact-equal) while geometry is float16-exact; max DC 29.19 vs 5.89; 11.5% of DC and ~16% of degree-3 colours outside [0,1] at policy poses; appearance state `paused_ungraded`; USDZ ships no cameras or validation frames | Coherent 2D teacher outputs exist (but are not 3D renders); a 0.28×29 tail is common in unclamped 3DGS training | Variant A (native NRE at the three policy poses and three reference-target poses). Bad at both pose sets ⇒ appearance defect; good at reference, bad at policy ⇒ camera domain. |
| 2 | Wide-FOV, robot-parented policy cameras (especially the wrist) sit outside the trained view domain; near-camera Gaussians the NuRec renderer culls (near clip 0.2 m) dominate the ParticleField wrist frame | Medium | Wrist frame is a near-uniform white surface with chromatic edges; NuRec payload declares near clip 0.2 m, transmittance threshold 1e-4 and global Z order, none of which ParticleField carries; no packaged rig | Overview and external frames also break up and are not near-field | Variant B vs C at the wrist pose (native NuRec applies its own culling; ParticleField does not). Additionally re-render C with Gaussians within 0.2 m of the wrist camera removed as a diagnostic-only control. |
| 3 | Sorting / compositing semantics differ between NuRec's `3dgut-nrend` (global Z order, K-buffer 0, threshold 1e-4) and RTX ParticleField without a sorting hint, and 58.56% of Gaussians are fully opaque so ordering decides every pixel | Medium-low | Exact-opacity fields make per-pixel winner order visible as halos/streaks; the current asset authors no `sortingModeHint`; 3DGRUT authors `cameraDistance` | Ordering errors produce popping and halos, not saturated per-splat rainbow colours | Variant C (hint `cameraDistance`) vs D (no hint) on the same Isaac process; then `--render-order-hint zDepth` and `rayHitDistance`. |
| 4 | Higher-order SH basis, ordering, or direction convention mismatch | Low | Rainbow is view-dependent; PR #1555 changed layout | Hermetic parity: SH triplets are bit-identical to 3DGRUT's writer; 3DGRUT and INRIA share the basis constants (`sh_rotation.py`, `gaussianParticles.cuh`); RTX renders 3DGRUT LightFields per NVIDIA docs | Variant E: degree 0/1/2/3 from the official transcode (`transcode … -o ply` then `--max-sh-degree N --format lightfield`). Degree 0 coherent and degree ≥1 broken ⇒ SH handling; otherwise closed. |
| 5 | Pre/post activation, quaternion order, up-axis, or coordinate transform error in Blueprint's conversion | Very low | Custom seam at a private boundary | Parity report: positions exact, scales/opacities/orientations ≤1.2e-7; both routes wxyz; `usd-convert-gsplat up_axis` only sets stage metadata; the Volume transform is identity and both routes apply it to geometry only | Already falsified hermetically; re-confirm on the GPU by comparing C and D pixel-for-pixel (expected: identical up to sorting). |
| 6 | Colour pipeline (double sRGB, tonemapping) | Very low for the current asset | Earlier washout/clipping history | PR #1546/#1556 fixed; current path is material-free with `skipTonemapping` read back true; saturation now 0.005–0.06% | Confirm `gaussian_skip_tonemapping_enabled` in the launch receipt of the run that produced the frames; already recorded. |
| 7 | A different asset or backend rendered the failing frames than the one the receipts name | Very low | The policy worker's launch receipt could name a backend the stage never composed | The render-only probe passed the ParticleField path explicitly and the derived asset digest matched | Closed by this PR's typed backend contract; for the past run, compare the probe's `particlefield_runtime_asset_cache` digest to `sha256:1bfd4438…`. |

## Definitive judgment on equivalence

**Can Blueprint's private-tensor translation be treated as equivalent to
3DGRUT's direct transcode?** For the learned attributes of *this* asset class
(single `gaussians` node, identity Volume transform, degree 3, no PPISP, no
background), yes, and it is now pinned by test: the hermetic reference model in
`particlefield_upstream_parity.py` transcribes the upstream importer, adapter and
LightField writer at `a37ef72` and is exercised against `build_particlefield_arrays`
and against the full production converter output (`tests/test_particlefield_upstream_parity.py`).
The same reference model was checked in this session against the real upstream
transcode output: positions and SH exact, scales ≤1.5e-8, opacities ≤1.2e-7,
orientations exact up to sign.

For the *route* as a whole, no. The private path (a) does not author the
sorting and projection hints or the display colour space that NVIDIA's writer
authors, (b) has no upstream guarantee for future container versions, multi-node
payloads, non-identity transforms (which 3DGRUT keeps on the prim rather than
baking, precisely because baking would require rotating the SH bands), PPISP or
background layers, and (c) reinterprets a private format. It is therefore
recorded in the new backend contract as
`particlefield_blueprint_private_tensor_conversion`, `development_only=True`,
never a production default. The production route is the pinned direct transcode
wrapper (`isaac_nurec_export.transcode_nurec_usdz_to_particlefield`).

## Exact attributes, transforms and activations to compare

Per-attribute digests are produced by both
`validate_transcoded_particlefield` (direct transcode) and
`particlefield_upstream_parity.attribute_digests` (reference model), so the
comparison is a receipt diff, not a viewer session.

| Item | Upstream (3DGRUT `a37ef72`) | Blueprint current | Status |
|---|---|---|---|
| `positions` | float16 → float32, `p @ M_volume^T` | same, transform required identity | exact |
| `scales` | `exp(log_scale)` (float32) | `np.exp` then `usd-convert-gsplat` `math.exp` | ≤1.5e-8 |
| `orientations` | `normalize(wxyz)` after `q_volume * q` | normalize wxyz; `Gf.Quatf.Normalize` | exact up to sign |
| `opacities` | `sigmoid(logit)` then `clip[0,1]` | `_sigmoid` (`+inf`→1.0); converter `1/(1+exp(-x))` | ≤1.2e-7 |
| `radiance:sphericalHarmonicsCoefficients` | `[albedo, specular.reshape(N,15,3)]` → `(N·16, 3)` | one NuRec→INRIA transpose, converter INRIA→vec3 read | exact |
| `radiance:sphericalHarmonicsDegree` / `elementSize` | 3 / 16 | 3 / 16 | exact |
| Volume transform | applied to geometry only, never to SH | refused unless identity | consistent |
| `projectionModeHint` | `perspective` | not authored (validator refused it before this PR) | prim-level difference |
| `sortingModeHint` | `cameraDistance` (also `zDepth`, `rayHitDistance`) | not authored | prim-level difference; the only one that can move pixels today |
| `colorSpace:name` | `srgb_rec709_display` | not authored | metadata only until RTX honours `ColorSpaceAPI` |
| Layer `renderSettings` | `rtx:post:tonemap:op = 2` | none (runtime reads `skipTonemapping` instead) | no effect on ParticleField with tonemapping skipped |
| Material binding | none (unless PPISP) | none | consistent |
| NuRec-only render config not representable in ParticleField | near clip 0.2 m, transmittance threshold 1e-4, `global_z_order`, `ccm` exposure matrix | dropped | **semantic gap**; only variant B/C isolates it |

Two upstream facts the packet had inverted: NuRec is pre-activation, and
`--apply-coordinate-transform` must **not** be passed for a NuRec input (it is
the 3DGRUT-frame → Omniverse flip for fields trained in 3DGRUT's own frame; the
importer already applies the authored Volume transform).

## Code findings (file and line, at `origin/main` 418091d)

| Severity | Location | Finding | Change in this PR |
|---|---|---|---|
| Critical | `native_task_isaaclab_launch.py:291` (`appearance_render_path: str = NATIVE_TASK_ARENA_NUREC_RENDER_PATH`) | The launcher defaulted the backend name; `native_task_arena_policy_canary_worker.py:1077-1081` called it without the argument, so the policy session's receipt named `plain_nurec_volume` while the stage composed a ParticleField. The launcher only *records* the path, so this was a false receipt rather than a different picture, which is exactly why it must not default. | Argument required and validated against `NATIVE_TASK_ARENA_APPEARANCE_RENDER_PATHS`; every worker derives it from the sealed plan via `appearance_render_path_from_plan`; the policy canary worker seals a typed `appearance_render_backend` (new module `appearance_render_backend.py`) and binds it into the session receipt and result. |
| Critical | `native_task_camera_observability.py:790-862` | One overloaded `passed` covered blankness, clipping, near-black and duplicates only; it unlocked policy queries in `adp009d_policy_episode.py:656-664`. | Receipt v2 separates `frame_structure_passed`, `appearance_reference_parity_passed`, `human_visual_review_status`, `target_semantic_visibility_passed` (owned elsewhere) and `policy_observation_integrity_passed`; only the last unlocks a query. Parity and review come from a sealed `policy_observation_integrity_authority.v1` bound by digest to the session's backend. |
| High | `native_task_camera_observability.py:854` (`"candidate_policy_loaded": False`) | Hard-coded while `execute_paired_session` (`native_task_arena_policy_canary_session.py:582-602`) loads each policy before its first episode. | `candidate_policy_loaded` is a required keyword; the episode passes `True`; a new optional pre-load hook in `execute_paired_session` runs before the first `load_policy` and the worker uses it to refuse a session with zero client loads when the authority is missing, unbound, failed, or unreviewed. |
| High | `native_task_nurec_render_setup.py` `apply_display_referred_particlefield_material` | Refused any authored `projectionModeHint`/`sortingModeHint`, i.e. it would block NVIDIA's direct transcode output at runtime (found only by reading the code; would otherwise surface on the next GPU). | Accepts the upstream token sets; still refuses unknown tokens. |
| Medium | `isaac_nurec_export.py` | Wrapped `transcode` for PLY input only; no NuRec-USDZ entry, no pinned revision, no output validation. | `transcode_nurec_usdz_to_particlefield`: digest check before any subprocess, pinned revision, whitelist-only environment (no NGC/cloud tokens forwarded), scrubbed stdout/stderr tails, output validation of counts/degree/hints/colour space/material/transform with per-attribute digests, sealed receipt, and `nurec_state_reinterpreted_by_blueprint=False`. |
| Medium | `particlefield_usd.py:747-914` | Correct arithmetic, but a private reinterpretation at the least-understood boundary; its output self-check refuses hints. | Left as the typed development comparator; parity pinned by `tests/test_particlefield_upstream_parity.py`. |
| Low | `particlefield_usd.py` docstring | Still describes direct Blueprint authoring. | Not changed (documentation-only; follow-up). |

## Smallest implementation plan (this PR) and focused tests

Mapping to the packet's twelve required tests:

| # | Required test | Test |
|---|---|---|
| 1 | Direct-transcode command is digest-pinned and secret-free | `test_direct_transcode_command_is_pinned_and_carries_no_secrets`, `test_direct_transcode_forwards_only_the_environment_whitelist_and_scrubs_output` |
| 2 | Source identity mismatch fails before invoking a container | `test_direct_transcode_refuses_identity_mismatch_before_invoking_anything` |
| 3 | Wrong count / SH degree / hints / colour space fails | `test_direct_transcode_output_contract_drift_fails_closed` |
| 4 | Backend digest mismatch fails | `test_prepolicy_gate_unlocks_only_with_bound_parity_and_approved_review`, `test_preload_gate_requires_authority_bound_to_this_backend`, `test_unbound_failed_or_unreviewed_authority_keeps_the_episode_blocked` |
| 5 | Launcher receives the backend explicitly; no default | `test_launch_requires_an_explicit_known_appearance_render_path`, `test_every_arena_worker_derives_the_render_path_from_its_plan`, rehearsal asserts `isaac.appearance_render_paths` |
| 6 | Truthful loaded/query state | `test_prepolicy_gate_records_the_caller_supplied_policy_load_state`, `test_prestart_receipt_records_that_the_candidate_is_loaded_but_not_queried` |
| 7 | Failed gate ⇒ zero client loads and zero queries | `test_missing_observation_integrity_authority_blocks_before_any_policy_load` (real worker path), `test_blocked_preload_observation_gate_loads_zero_policies_and_still_closes` |
| 8 | Structural pass with failed parity stays blocked | `test_structural_pass_without_sealed_observation_integrity_blocks_before_any_query`, `test_structural_failure_keeps_integrity_false_even_with_a_perfect_authority` |
| 9 | Failing frames as fixture | Not possible here (frames absent). The receipt now carries per-view chromatic descriptors so the three PNGs can be added as a fixture with their expected descriptor values in the follow-up that lands them. |
| 10 | Human review required while the detector is diagnostic-only | Gate design; `test_chromatic_diagnostics_describe_scattered_saturated_splats_without_gating` |
| 11 | Manifest includes all three PNGs on success and failure | Unchanged behaviour of the existing retained-frames path; not re-tested here. |
| 12 | Portal test for blocked pre-policy runs | Blueprint-WebApp, out of this repository's scope. |

Not done in this PR, deliberately: no threshold for the chromatic descriptors
(uncalibrated), no NRE client, no change to the private converter's arithmetic
(it is correct), no portal work.

## Decisive single-GPU, no-policy experiment

One instance, retry 0, explicit spend cap and TTL, teardown and provider-zero
proven even if variant A fails. Prefer a 48 GB-class GPU so NRE and Isaac can
coexist; otherwise run A then B–E sequentially on one 24 GB instance without
reallocation.

Sealed inputs (digests recorded before anything starts):

- Source USDZ `sha256:9193a9de…` (exact, symlink-refused, re-hashed after every step).
- Policy camera poses and intrinsics from `native_task_arena_runtime_preflight.v1.json`;
  reference-target poses from `calibrated_source_cameras.json`. Both converted
  once into NuRec space (frame 0 identity) for NRE and kept in registered world
  for Isaac; the conversion matrix is part of the receipt.
- Pinned NRE image by digest (never `latest` in the receipt), Isaac Sim
  `nvcr.io/nvidia/isaac-sim:6.0.1@sha256:b1c542b2…`, 3DGRUT `a37ef721…`.

Variants and commands (paths are identifiers):

```bash
# A. Native NRE authority (six poses, one warm server, one batch)
docker run --gpus all --net=host --rm \
  -v /work:/workdir/output nvcr.io/nvidia/nre/nre@sha256:<pinned> \
  serve-grpc --renderer nrend --artifact-glob "/workdir/output/configured_appearance.usdz"
python nre_render_client.py --scene configured_appearance \
  --poses poses_nurec_space.json --intrinsics intrinsics.json \
  --resolution 320x180 --image-format PNG --out A/nrend/
# repeat once with --renderer default; record server/client versions

# C. Official direct transcode (no Blueprint tensor decode)
python -m threedgrut.export.scripts.transcode configured_appearance.usdz \
  -o C/lightfield.usdz --format lightfield            # sortingModeHint=cameraDistance
# Blueprint wrapper: transcode_nurec_usdz_to_particlefield(..., expected_source_sha256=...)

# E. SH-degree controls from the official route
python -m threedgrut.export.scripts.transcode configured_appearance.usdz -o E/scene.ply
for d in 0 1 2 3; do
  python -m threedgrut.export.scripts.transcode E/scene.ply -o E/deg$d.usdz \
    --format lightfield --max-sh-degree $d
done

# B and D inside the same Isaac 6.0.1 process, same cameras, robot composed:
#   B = original USDZ as plain_nurec_volume; D = sha256:1bfd4438… ParticleField.
#   Launch with appearance_render_path derived from the plan (now required).
```

Outputs per variant: lossless PNGs at 320×180 and at 1024 px width for each of
the six poses; PNG digests; the launch receipt with `nurec_renderer` readback;
per-attribute digests for C, D, and each E; the chromatic descriptors from
`measure_native_task_frame_chromatic_diagnostics` for every frame; masked
SSIM/LPIPS of B, C, D and E against A on the background (robot and replacement
object masked out).

Decision table (as in the packet, with one row corrected by the parity
result):

| A native NRE | B Isaac NuRec | C direct transcode | D current Blueprint | E degree 0 | Conclusion |
|---|---|---|---|---|---|
| bad at all six poses | any | any | any | any | Appearance defect. Stop renderer work; repair or regenerate the configured field. |
| good at reference poses, bad at policy poses | same pattern | same pattern | same pattern | mixed | Camera-domain or pose/intrinsics contract. Reposition cameras or extend view coverage; do not touch the converter. |
| good | good | bad | bad | good | ParticleField/RTX semantics (sorting, culling, near field). Use native NuRec for the canary; isolate with hints and the near-field control. |
| good | good | bad | bad | bad | ParticleField representation unsuitable for this source. Use native NuRec/NRE. |
| good | good | good | bad | any | Would contradict the hermetic parity; investigate asset identity, not arithmetic. |
| good | good | good | good | good | The failing run rendered something else; audit launch/backend identity. |

Stop conditions carry over unchanged: no policy starts if any required camera
fails human review; no silent camera changes; no Difix/Harmonizer outputs
counted as evidence; no post-hoc threshold tuning without a recorded
calibration set.

## Visual-quality rubric

A frame is **rejected** if any of the following holds; a frame is accepted only
if none holds *and* a named human approves the three-frame contact sheet.

1. Structural (already gated): blank, >50% near-black, saturated channels
   above the sealed floor, or duplicate digests across views.
2. Reference parity (new, required): masked-background SSIM against the
   same-pose native NRE render below the preregistered floor, or LPIPS above the
   ceiling. Until the floors are preregistered from a known-good/known-bad
   calibration set, the parity receipt may still be sealed only by a human who
   has viewed both images side by side.
3. Chromatic breakup (diagnostic now, gate after calibration): the receipt
   reports `rgb_spread_pixel_fraction` (share of pixels whose channel spread
   exceeds 64 levels) and `local_chroma_outlier_fraction` (share of pixels whose
   chroma departs from its 3×3 neighbourhood by more than 48 levels). The three
   failing frames reported spread fractions of 5.16%, 5.10% and 2.14%; a grey
   textured surface reports 0 for both; scattered saturated splats on that
   surface report both above 2%. Reference images with genuinely colourful
   objects will show spread but low *local outlier* values, which is why the
   second descriptor exists and why neither is a threshold yet.
4. Semantic (already gated separately): task-object pixels below the sealed
   floor in any required policy view.
5. Robot and replacement object visible and not occluded by the splat in the
   required views (human review item; no automatic proxy proposed).

## Explicit stop/go for the Website Quick-10

Go only when all hold on the exact backend the run will launch with:

- `appearance_render_backend.kind` is a production kind
  (`isaac_native_nurec`, `particlefield_3dgrut_transcode`, or `nre_native_grpc`),
  not the development comparator, and its `receipt_digest` is identical in the
  render probe, the sealed `policy_observation_integrity_authority.v1`, the
  activation receipt and the policy session result.
- `appearance_reference_parity.passed` is `True` against native NRE at the three
  policy poses, and `human_visual_review.status == "approved"` by a named
  reviewer of the contact sheet.
- The launch receipt reads back `gaussian_skip_tonemapping_enabled` not `False`.
- The semantic camera-observability gate passes for the required views.

Stop, with no policy client constructed, if any is missing. The worker now
enforces this before `load_policy`.

## Additional failures visible only in the source files

- The runtime validator's refusal of upstream hints (fixed above) would have
  blocked variant C at runtime.
- `usd-convert-gsplat 0.1.15` computes opacity with `math.exp(-x)` per vertex; a
  `-inf` logit raises `OverflowError` and the private route reports it as
  `upstream_usd_convert_gsplat_failed` with no attribute named. The configured
  field has `+inf` only, so this did not fire, but any future field with exact
  zero-opacity endpoints will. The direct transcode handles both signs.
- The configured payload's renderer config (`global_z_order=true`, near clip
  0.2 m, transmittance threshold 1e-4) is not 3DGRUT `a37ef72`'s template
  (`false`, 1e-8, 1e-3); it was adopted from a shipped NVIDIA package by the
  Blueprint NuRec writer. Its `+inf` logits could not have come through
  `nurec_volume_codec.build_state_dict`, which refuses non-finite arrays, so the
  artifixer bundle wrote the state directly. That provenance should be sealed
  before the field is repaired or regenerated.
- The shipped NuRec layer carries `rtx:post:registeredCompositing:invertToneMap`
  and an emissive `ccm` exposure matrix that only the native NuRec path honours;
  a ParticleField derived from it renders with neither. Variant B vs C measures
  the pixel effect.

## Primary sources (pinned)

- 3DGRUT `a37ef721012dea0f29c0fcfff2d525023b4e854a`:
  `threedgrut/export/scripts/transcode.py`, `importers/nurec_usd.py`,
  `adapter.py`, `usd/writers/lightfield.py`, `usd/writers/base.py`,
  `usd/exporter.py`, `usd/nurec/templates.py`, `formats/ply.py`,
  `importers/ply.py`, `sh_rotation.py`, `partition.py`,
  `threedgrt_tracer/include/3dgrt/kernels/cuda/gaussianParticles.cuh`
  (`radianceFromSpH`: `SH_C0 * c0 + … + 0.5`, clamped at 0).
  https://github.com/nv-tlabs/3dgrut/blob/a37ef721012dea0f29c0fcfff2d525023b4e854a/threedgrut/export/README.md
- `usd-convert-gsplat` `621017ebf78394488260c70ec4eadd70ff621131`:
  `source/python/usd_convert_gsplat/usd_writer.py`, `ply_reader.py`
  (numeric `f_rest_N` ordering; `up_axis` sets stage metadata only).
- Omniverse ParticleField colour space and tonemapping statement:
  https://docs.omniverse.nvidia.com/materials-and-rendering/latest/particle-fields.html
- NRE gRPC contract (poses in NuRec space, `--renderer nrend|default`, batch):
  https://github.com/NVIDIA/nurec-skills/blob/main/skills/nre/references/grpc-api.md
- Isaac Sim discussion #680 (engineer directs PLY conversion to `transcode`):
  https://github.com/isaac-sim/IsaacSim/discussions/680
