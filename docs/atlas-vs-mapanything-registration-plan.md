# Atlas vs MapAnything source registration: side-by-side plan

Status: plan only. Nothing here has run. The plan was requested in the
2026-10-19 beta launch audit. Running it is the founder's decision, because
`CLAUDE.md` freezes provider bakeoffs outside the beta journeys. The run needs:

- World Labs Atlas access;
- a reviewed Atlas price;
- consent to Atlas's terms recorded on the capture's sponsorship;
- the founder's approval to spend.

Beta journey step: concurrent scene preparation. A capture without a
scene-preparation sponsorship waits at `needs_input` with
`website_source_camera_depth_registration_required`. Captures with a
sponsorship get source registration from the pinned MapAnything worker. This
plan tests whether Atlas `images2PosedRGBD` could fill that gap. One measured
capture runs through both, and the results are compared against independent
measurements.

## The question, and what this plan cannot change

Question: on the same frames of one measured capture, is Atlas's camera and
depth registration as usable as the pinned MapAnything registration for source
registration and mask lifting?

What this plan cannot change:

- **Both arms stay estimates.** Every artifact keeps
  `claim_ceiling: development_only` and `metric_measurement_proven: false`.
  Neither arm qualifies a scene. Qualification still needs the existing
  `blueprint.site_task_simulation_qualification.v1` receipt, built from
  independent measurements (`docs/beta-site-task-simulation-qualification.md`).
- **Atlas supplies no metric scale.** Its depth is in "world units" with no
  stated scale. The beta's measurement requirement stands whatever the
  outcome.
- **No adoption here.** Routing unsponsored captures to Atlas would be a
  separate, reviewed change to `website_scene_handoff.py`, decided by the
  founder from this plan's report.
- **Agreement is not accuracy.** Where the two arms agree, that is not truth.
  Only the independent measurements are external truth, so the capture must
  have them.

## Inputs, frozen before either arm runs

1. **One website capture.** It needs a retained source video and a sponsorship
   that allows external disclosure, because Atlas receives site pixels. The
   CPU preparation writes `worker_inputs/geometry_inputs.json` (digest **D**)
   and `atlas_pose_inputs.json`, whose `binding.geometry_input_digest` must
   equal **D**.
2. **An independent measurement document**, `blueprint.site_task_measurement_samples.v1`,
   with at least five `scale` and five `registration` checks. Each check is a
   measured distance between two landmarks, by tape or laser, that are visible
   in at least two retained frames. A measurement producer who is not the
   analyst records it before either arm runs.
3. **Blind landmark annotations.** Each landmark gets a frame ID and a source
   pixel, annotated on the source frames before anyone sees either arm's
   output. Without them, the measured checks cannot be read from either arm.
4. **A frozen comparison profile.** The decision thresholds below are written
   down and reviewed before the run.

## The two arms

| | Arm M: MapAnything, pinned | Arm A: Atlas |
|---|---|---|
| Model | `facebook/map-anything-apache` at code revision `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9` | `POST /api/v2/tasks:images2PosedRGBD` (beta) |
| Entry | `run_sponsored_website_geometry`, which calls `dispatch_geometry`, with the release's `website_mapanything_runtime.v1` profile and worker image digest | `run_sponsored_website_atlas_pose` (see `docs/website-atlas-pose-operation.md`) |
| Frames | The frames in **D** | The same frames, in the same order |
| Pixels | 518-pixel upright model inputs | The same files, verified by digest before upload |
| Held-out pixels | Excluded | Excluded |
| Options | Fixed by the worker | `refinePoses: true`; grid 1280×720 or 720×1280, whichever keeps at least 95% of every view |
| Output | `world_from_camera` (OpenCV), `depth_m` in estimated meters, confidence and valid mask on the model-input grid | Position and quaternion with `coordinateSystem`; half-float linear-depth EXR, confidence and mask on the Atlas grid |
| Spend | One GPU rental, capped by the runtime profile | One operation, capped by the reviewed price and reserved as `world_labs_atlas` |

Run M on **D** directly. A sponsored M run on a task-mask-augmented input set
does not count, because its frames differ. If a retained M result already has
`binding.input_digest == D`, replaying it costs nothing.

## Reading both arms in one frame

Write the following analysis in one CPU-only script. It reads only the retained
bytes from both arms and makes no network calls. The script is a follow-up; see
below.

1. **Atlas conventions are read from the first real response, not assumed.**
   The docs leave these open:
   - the quaternion component order (the identity example `[0, 0, 0, 1]`
     suggests x, y, z, w);
   - whether the extrinsics are camera-to-world;
   - `rub` against `rdf`. For `rub`, OpenCV axes are `R · diag(1, −1, −1)`;
   - whether "linear depth" is z-depth or distance along the ray.

   Choose the reading that gives the lowest multi-view reprojection error, and
   record it. If no reading gives a coherent reconstruction, stop: Atlas's
   output is not usable as registration.
2. **Decode depth with OpenCV**, whose rows run top-down. The docs note that
   Three.js `EXRLoader` lists rows bottom-up, so it is not used. Half-float
   storage adds about 5e-4 relative error.
3. **Map pixels between grids.** The documented reframing scales a view to
   cover the grid and then center-crops it. A model-input pixel `(u, v)` on a
   `w × h` frame maps to the Atlas `W × H` grid as `s·(u + 0.5) − 0.5 − ox`,
   where `s = max(W/w, H/h)` and `ox = (s·w − W)/2` (and likewise for `v`).
   Composing that with the stored `source_to_geometry_pixels` maps source
   pixels to Atlas pixels.

## Measurements, per arm, side by side

| # | Measurement | Why it matters |
|---|---|---|
| 1 | **Pixel binding (Atlas only).** Resample our input with the transform in step 3, then compare it with Atlas's returned reframed RGB, using normalized cross-correlation and PSNR per frame | A stable, verified transform is what `website_atlas_native_geometry_binding_required` is waiting for. Without one, masks cannot be lifted onto Atlas depth. |
| 2 | **Relative pose agreement, A vs M.** Pairwise relative-rotation error in degrees, relative translation-direction error in degrees, and the ATE RMSE after a Sim(3) (Umeyama) alignment of camera centres, with the fitted scale | Shows whether the two estimates agree on the camera path. It is agreement, not accuracy. |
| 3 | **Multi-view depth consistency**, each arm on its own outputs. Warp depth from frame *i* into frame *j*, then report the median relative error and the share of co-visible pixels within 5% | Mask lifting and scene registration need depth that agrees across views |
| 4 | **Measured scale and shape.** Unproject each landmark pair with the arm's own depth and camera to get a distance `e_k`. Fit one scale `s*` to the measured `d_k` and report the residual RMS percentage. For M only, also report the raw error in its estimated meters | This is the only comparison against external truth. Atlas's world units have no metric meaning, so only the residual after fitting applies to it. |
| 5 | **Coverage.** Valid-pixel fraction and the confidence distribution. Neither arm's confidence is a calibrated probability | Gaps in coverage mean nothing to lift masks onto |
| 6 | **Cost and time.** Wall time and provider calls. M's cost comes from its GPU receipt; A's comes from the platform's Usage page, because Atlas operations carry no cost | Needed to compare running cost per capture |

## Decision rule, frozen before the run

Before either arm runs, the founder approves the thresholds in a comparison
profile shaped like `blueprint.site_task_fidelity_profile.v1`. There are no
defaults. The starting values below are for that review only:

- **Pixel binding** (measurement 1) holds on every frame: NCC ≥ 0.98 after the
  derived transform.
- **Shape** (measurement 4): A's residual after fitting is at most M's residual
  plus 2 percentage points, and within the profile's registration tolerance.
- **Depth consistency** (measurement 3): A's inlier share is at most 5 points
  below M's.
- **Pose agreement** (measurement 2): median relative-rotation error is at
  most 2°.

The run ends in one of three outcomes:

- **Atlas usable for registration.** All four hold. Next comes a separate,
  reviewed PR that routes unsponsored captures to Atlas for registration
  only, with the same measured-scale requirement and the same claim ceiling.
- **Keep MapAnything only.** Any of the four fails. Unsponsored captures keep
  waiting at `needs_input`.
- **Inconclusive.** M itself fails the measured checks, the Atlas conventions
  cannot be resolved, or there are too few measured checks. Neither arm is
  endorsed, and the capture or the measurements are redone.

## Budget, approvals and order

The founder approves each paid step separately before it runs:

1. **Access.** Atlas access with `images2PosedRGBD` enabled for the account,
   the platform's API root and a scoped key. Even a key check may cost credits,
   so it waits for approval.
2. **Price.** A reviewed price recorded as `BLUEPRINT_ATLAS_POSE_PRICE_REVIEW_JSON`.
3. **Privacy and consent.** Atlas receives unedited source frames, with people
   not removed, so two things are needed:
   - a decision on where uploaded frames are kept
     (`BLUEPRINT_ATLAS_PROVIDER_RETENTION`);
   - the capture's sponsorship recording consent to Atlas's terms. The WebApp
     refuses the `world_labs_atlas` reservation until it does.
4. **Spend.** One Atlas operation at or below the reviewed ceiling, plus one
   MapAnything rental at or below its profile cap, unless a retained M result
   for **D** exists. Both draw on the capture's preparation allowance
   (`upstream_max_spend_usd` and `max_paid_attempts` in its sponsorship), which
   is shared with SAM, image edits and Marble. Check the headroom first.

Order:

1. Choose the capture.
2. Freeze the measurements, the annotations and the profile.
3. Run the CPU preparation.
4. Run arm M, or replay a retained result.
5. Run arm A.
6. Run the offline analysis.
7. Write a one-page report and a JSON result, both `development_only`, for the
   founder's decision.

## Follow-ups this plan needs

- **The analysis script.** CPU-only, with hermetic tests on synthetic fixtures
  that have known poses and depth.
- **An operator entry point for arm A.** An allocator subcommand, or a
  controller hook that only an explicit selection reaches.
- **Atlas on full-resolution upright frames,** as a second run. Arm A uses the
  518-pixel model inputs so that both arms see identical pixels. Atlas
  upsamples them onto its 1280×720 grid, so input resolution may matter more
  to Atlas than to MapAnything. Treat this as a separate approved run.
