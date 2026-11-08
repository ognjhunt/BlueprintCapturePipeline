# Native Atlas splats

ADP-040 / day 28 appearance construction. World Labs' Atlas beta API added
`posedImages2Splats` on 2026-10-29: posed RGBD frames in, refined Gaussian
splats (SPZ, keyed by count) out. It replaces our depth-fused draft as the
final appearance of a website scene. The result is development-only: it does
not prove metric scale, source-geometry preservation, collision or physics.

## Where it runs

`atlas_scene_automation.run_automation` (the object-removal background flow)
keeps the existing steps: removal edits and review, fallback pose, up-front gap
completion for the planner's uncaptured regions, and geometry recovery. Those
steps still fuse a local draft, because gap detection projects it. When the
reviewed policy carries `refined_splat_cost_usd`, the refined splat follows
that draft directly. The later coverage-repair pass is skipped: it measures
gaps in the draft, which is no longer the final appearance.

1. `atlas_native_splat.prepare_splat_request` posts the views the final draft
   fused, each once, in World Labs' documented frame shape: `imageAsset`,
   `camera` and `depth.depthAsset` (linear EXR), nothing else. Views with the
   same camera (within 1e-6) and identical RGB are one posed view
   (`merge_duplicate_views`): a whole duplicate makes it whole, otherwise the
   admissions unite, and depth takes each partial duplicate's values where its
   review admitted them. The provenance row records the merged positions
   (`merged_duplicates`). Real frames go in whole. A generated or recovered view
   (one an earlier review admitted only in part, `keep_mask`) goes in whole only
   if the whole-view agreement review admitted it, and is left out otherwise.
   Frames on one grid go unchanged; mixed grids (portrait source beside
   landscape generated views) use the ray-preserving photograph tiler.
2. `run_native_splat` reserves the reviewed price, submits through the shared
   admitted task runner, and resumes a pending operation by ID (an operation
   takes about 20–30 minutes; each controller pass returns `processing`).
3. The largest returned count is kept. Its original SPZ is retained and decoded
   in Python (`atlas_depth_splat.decode_spz`; Atlas publishes SPZ v2 with
   degree-0 colour, bit-identical to `splat-transform` on a 2M-Gaussian Atlas
   splat) into a standard 3DGS PLY. Any other layout fails closed with the paid
   SPZ kept, so a retry never buys twice.
4. `atlas_refined_splat_manifest.json` binds the pose, the draft manifest, the
   generated-result digests, the request, every upload and the whole-view
   verdicts. Mesh registration keeps the draft's fusion grid as its tolerance.
   `splats2Mesh`, registration, the download bundle and scene handoff read
   either manifest.

## Request forms

World Labs documents a posed frame as image + camera + depth (`depthAsset`, or
`logDepthAsset` with `zMin`/`zMax`) only
([request/response reference](https://docs.atlas-beta.worldlabs.ai/api-reference/posed-images2-splats),
[guide](https://docs.atlas-beta.worldlabs.ai/posed-images-to-splats)).
`maskAsset` and `depth.confidenceAsset` are not documented. Both production
requests that sent them were accepted and then failed with `EXECUTION_FAILED`
(failed runs are free): capture-4b60431e (11 views, masks on 8, confidence on
all) and capture-447ae4f4 (22 views, 4 of them posted twice with conflicting
depth and masks). The undocumented fields are the likely cause. The splat
guidance treats generated views as ordinary whole frames, to be compared with
overlapping views before splatting: objects keep their position, orientation,
shape and appearance, or the splat shows ghosted or duplicated surfaces
([splat guidance](https://docs.atlas-beta.worldlabs.ai/splat-best-practices)).

Each form has its own root beside the draft:

- `documented_reviewed_whole` (`<draft root>-refined-whole`): real frames plus
  the generated views the whole-view review admitted, whole. If it admits none,
  no request is sent from this root.
- `documented_minimal` (`-refined-minimal`): real frames only. It follows when
  World Labs failed every attempt of the reviewed-whole form, or straight away
  when no generated view was admitted (the same request is never bought twice).
- `reviewed_masks_and_confidence` (`-refined`) and `reviewed_masks`
  (`-refined-masks`) are earlier releases' forms: a retained attempt is polled
  or adopted with exactly what it sent, never re-derived or bought again.

A completed splat in any root is adopted, earlier roots first, and no form is
newly bought behind a later live or completed one.

**Whole-view agreement review** (`atlas_whole_view_review`). The completion
review judges the generated set for the edit (`consistent_background`) and
names no per-view verdict, so it cannot admit one view whole. Before the
reviewed-whole form, one Gemini call reviews every candidate at once: each
merged generated view beside the up to three real frames that overlap it most
(sampled real depth projected into its camera, at least 2%), returning
`admit_whole` and a reason per view. A view no real frame overlaps is never
admitted. The call goes through `retained_gemini_call` (admitted by the WebApp
spend ledger, retained by its exact inputs, so a restart never buys it again)
and is quoted from its image and token count; above the policy's
`maximum_review_cost_usd`, or after an uncertain earlier call, nothing is
admitted and nothing more is bought. The verdicts are kept in the root
(`whole_view_review.json`), the input binding and each posted view's provenance.
The prompt is scene-agnostic and covered by `test_harness_prompts_scene_agnostic`.

**Candidates corrected where they contradict the capture** (`atlas_gap_composite`).
`atlasGenerate` documents no keep or mask input, so each generated view is a whole
new image, and completion's preservation pastes sparse cross-view-supported
context into it (42% and 32% of capture-447ae4f4's two gap views, 2026-10-30).
That speckled frame is what the review refused. Before the review,
`composited_views` now rebuilds each partly admitted view from the provider's own
image (`provider_rgb`, kept by `restrict_frame` when completion replaces `rgb`).
The real frames are rendered into its camera with per-pixel quality (photographs
resampled at a recorded centre; elsewhere the draft's smoothed depth splatted
over each pixel's footprint, nearest camera first, colour sampled back from its
frame). A capture pixel and its depth replace the provider's only where the
capture is well observed and the provider contradicts it; the admitted region,
everything seen poorly and everything never seen stay the provider's, and a
replaced region blends over 4 px into its agreeing neighbours.

The thresholds come from capture-447's retained frames:

| quantity | value | derivation |
|---|---|---|
| depth contradiction | relative difference > 0.0394 | q99 of provider vs capture on known-agree content: the provider's re-render at the two recorded centres it was given, against the capture rendered there from frames ≥ 0.5 units away (0.0394, 0.0305) |
| colour contradiction | > 80.9 after a 1.13 px blur | blur = median misregistration (optical flow) on that content; tolerance = larger of its q99 max-channel difference (78.1) and the render's own q99 error inside the gate (80.9) |
| stretch (view-angle and distance change) | < 1.850 | four photographs against their renders from frames ≥ 0.4 away, equal-count bins of 2,000 px: first bin whose q90 render depth error exceeds 0.0394 |
| incidence | ≥ 0.150 | same: last failing bin ends there |
| depth confidence | ≥ 0.635 | no bin fails; the lowest value calibrated |
| agreeing views | ≥ 3 | q90 render depth error 0.069 / 0.041 / 0.032 at 1 / 2 / 3 views |

The review judges, and the request posts, that frame and its depth; the record
carries `candidate_basis` and per-view `gap_composites` receipts (counts by
source and the thresholds), and the input binding repeats the receipt. The
candidate basis is written to `whole_view_candidates.json` before any review
call. A root an earlier release reviewed, or began to review (any receipt,
intent or lock without that record), keeps the provider's frames, so its call is
found again and never bought twice. The render reuses frames already in the same
request: nothing new is disclosed or bought. Offline on capture-447's retained
request, 3.2% and 0.5% of the two views became capture pixels; the foreground
tabletop, every selected gap pixel and the chairs (seen by the capture only at a
median stretch of 3.35 and 3.56; 98% and 100% provider pixels) stay the
provider's, and the pasted context is gone.
Whether the review now admits them needs one bounded review call.

**Upgrade of a finished job.** When the adopted splat is `documented_minimal`
and left generated views out, the result records
`refined_splat.left_out_admitted_view_ids` and whether the reviewed-whole form
was never tried (`gap_preserving_form_untried`: never sent and never reviewed).
Only that marker lets an operator retry (`retry_failed_job` or the door's
atlas-retry) reopen a `ready` job. A job that became ready before jobs recorded
the marker (capture-447ae4f4) has it derived from retained records
(`retained_minimal_marker`): the archived result names the adopted minimal splat
by manifest digest, and the views the failed masked form sent with a review mask
and the minimal form did not send are the ones left out. The retry records
`refined_splat_gap_preserving_form_untried` and `request_gap_preserving_upgrade`
names the reviewed-whole root. The next run reviews the generated views if
needed and sends that form at most once. A success supersedes the minimal splat
(its mesh root is keyed by the new digest, so a new mesh is bought and the
archive rebuilt). If the review admits nothing or World Labs fails it, the
minimal splat is adopted again unchanged, the marker clears and the job returns
to `ready`. Nothing reopens a ready job on its own.

Uploads to an account that does not retain them are purged about an hour after
creation. If a send is refused with 404 naming one of the attempt's own
uploads, nothing was bought, so the same admitted purchase uploads again under
a new idempotency key (`refresh_purged_uploads`, at most twice).

A failed or refused attempt's reservation is released once the attempt settles
(`atlas_spend_settlement`, a WebApp `world_labs_atlas` settlement at zero); a
completed one settles at its reviewed price.

The refined roots sit beside the draft (`<draft root>-refined*`), and the mesh
root carries the adopted refined digest, so a new draft gets its own purchase
and a legacy mesh is never adopted for a refined splat.

The plain pose-and-generate route keeps the draft for now. Preserved-capture
scenes buy the refined splat from their source pose (see "Source frames for a
preserved scene's splat" below).

New object-removal jobs with a reviewed refined-splat price clean every selected
real view. Selection retains the task/room cover and up to 30 distinct overlapping
views; when a slow camera movement yields fewer than 16, sharp, connected source
frames fill the set toward 16. Identical source-image digests are never repeated.
These are labelled `overlapping_support_view`, separately from distinct views.
If fewer usable frames exist, the manifest reports that shortfall without
inventing coverage. Excluded frames, quality thresholds, face screening and
the cumulative allowance still apply. Views without a removal target remain
unedited. Generated views address only the recorded uncaptured regions.

The input binding retains the minimum, the actual support count and whether it
was met. Existing jobs retain their original edit scope and selection, so this
change requires a fresh job; it never rewrites job 3's three-view inputs.

## Source frames for a preserved scene's splat

A preserved-capture scene (`no_manipulated_subject` or `synthetic_design`)
trains its refined splat on its source pose: the video selection plus the
capture's task-source views. Forward camera motion moves features little, so
the 12-px near-duplicate rule keeps few views. On capture-5b067498 (a 4K
forward-motion library aisle), 122 of 152 candidates were near duplicates, and
the `documented_minimal` splat trained on 41 frames (31 selected, 10
task-source). It came out soft.

When the policy carries `refined_splat_cost_usd`, a new selection keeps the
connected cover and distinct views, then fills toward 96 frames
(`SPLAT_FRAME_TARGET`, under the 128-image `SPLAT_POSE_INPUT_CAPACITY`) with
`add_time_spread_views`. The admitted still frames' span is cut into 96 equal
windows. Each step takes the sharpest candidate of the emptiest window that
passes the sharpness and clipping policy and overlaps a kept time neighbour by
the policy's inlier count (24 by default). The displacement rule does not apply,
an identical source image is never repeated, and with fewer usable frames the
set stays short. These views are labelled `time_spread_view`; the plan's
`frame_policy` and the input binding record `splat_frame_target`.

Pose is priced on the actual count, about $0.01 a frame (96 frames, about $1).
A budget that cannot pose the set refuses it
(`atlas_selection_budget_below_accurate_inputs`) and never poses fewer.

Gap completion runs on the same source pose before the splat, and its
`atlasGenerate` context shares the served model's rollout: `target_batches`
keeps the context plus two views per target within 64, so 60 photographs at
most. capture-5b067498's completion sent all 41 frames. A 106-frame pose would
have refused there (`atlas_chisel_source_rollout_capacity_insufficient`) and
stopped the job before its splat. The fill adds density, not viewpoints, so
preserved completion keeps the views selected before the fill (cover, distinct
and task-source) as its gap-survey photographs, generation context and joint
registration (`reference_frame_ids`, recorded as `registration_frame_ids`).
Every posed frame still feeds the gap geometry, the draft and the refined
splat. The completion planner's survey still holds every posed frame, so that
review costs more (capture-5b067498's, at 41 frames, cost $0.28).

Nothing else changes. Jobs that buy no refined splat keep
`MAX_POSE_INPUT_IMAGES` (64). Removal jobs keep the profile above. A captured
subject's plain route posts its whole selection as `atlasGenerate` context, and
keeps its selection. A retained selection resumes with the profile and the
`provider_input_capacity` it recorded. No Atlas pose request above 64 images
has run in production yet, so 128 is configured, not verified.

World Labs' limits, read 2026-11-08:

- `images2PosedRGBD`: `targetResolution` must be one of 1280x720, 720x1280,
  960x720 or 528x704. The page states no maximum image count
  ([images to posed images](https://docs.atlas-beta.worldlabs.ai/images-to-posed-images)).
- `posedImages2Splats`: "Provide 2 to 512 ordered posedFrames", and every
  camera grid must be one of the same four
  ([posed images to splats](https://docs.atlas-beta.worldlabs.ai/posed-images-to-splats));
  the reference gives the same 2 to 512
  ([request/response reference](https://docs.atlas-beta.worldlabs.ai/api-reference/posed-images2-splats)).
  The pinned schema carries `maxItems: 512`.

The pose grid is already the largest documented one, so the number and spread
of source frames is what Blueprint can change.

## Which posed views a refined splat is sent

capture-5b067498's refined splat (41 frames, `documented_minimal`, 1280x720)
came out blurry. capture-447ae4f4's kitchen, on the same path and form with 16
frames, came out clear. The library job's retained records showed why:

- The first ~3 s (10 of 41 views) had poses and depth that disagree. 39 of the
  draft's 287 cross-view pairs were over the 0.05 limit (worst 10.7), and every
  one involved views 000–074, 029 or 058. World Labs did not accept pose
  refinement (`poseRefinementAccepted` false). The only check was the pooled
  median (0.007), which hid them, and the request sent every view.
- 22 views stood at one point, panning. The pose then jumped 3.3 units in
  0.16 s into a straight dolly, which agreed within 1 % on its own.
- The soft opening views passed the reviewed absolute sharpness floor (450) at
  about half the clip's median.

Two changes apply to every job. Neither stops one: they filter and flag.

**Selection (before the pose).** Every view added after the reviewed cover
(distinct, time-spread and support views) must also be at least
`RELATIVE_SHARPNESS_FRACTION` (0.6) of the candidates' median sharpness
(`atlas_video_inputs.prepare_video_inputs`). The cover keeps the reviewed
floor, so a required view is never refused for it. A soft cover view is named in
the plan (`cover_frames_below_relative_sharpness_floor`), and the views the floor
kept out are named `below_relative_sharpness_floor`. A retained selection keeps
the profile it recorded, so an in-flight job's inputs never change.

**The splat request (after the pose, before the purchase).**
`atlas_splat_input_quality.select_splat_frames` reads the draft's own per-pair
agreement and the posed cameras, and records its decision once, in
`refined_splat_input_selection.json` beside the refined-splat roots. The new
request's input binding names it (`input_selection_digest`). It works in three steps:

1. **Consistency, per view.** It drops the worst view while any view's own
   median residual against its measured neighbours is over the limit, and then
   any group that agrees only with itself after every member disagreed with its
   other neighbours (029 and 058 matched each other at 0.007 and nothing else). A
   view with no measured pair stays and is named. Below 8 consistent views it
   flags and continues; below the provider's 2 it restores the least
   inconsistent views.
2. **Motion.** It splits the views in time at pose jumps (a step of a quarter of
   the scene depth at ten times the clip's median speed, or a fast 45° turn) and
   around no-parallax runs (within 2 % of the scene depth for at least 1 s). It
   keeps the moving segment covering most of the reviewed task route, then most
   baseline, plus every other moving segment on the same side of any jump. It
   keeps at most 4 views of each no-parallax run, spread widest in viewing
   direction. Measured pairs only compare nearest cameras, so they never compare
   views across a jump. A segment beyond a jump is left out unless measured pairs
   tie it to the kept side. A capture that never moved keeps every view.
3. **Held out.** When at least 24 views would still be sent, it holds back 2
   sharp, measured, interior views of the kept segment, never task-route views.
   When the splat returns, `heldout_render_score.json` beside its manifest records
   a render-vs-photo score at those views and at their nearest sent views. The
   render takes each pixel's nearest opaque Gaussian centre at 320 px width, a CPU
   stand-in for rasterising the splat. It compares views with each other and is no
   image-quality measurement. Held-out views more than 3 dB below the training
   views are flagged. It never blocks.

On the library job's retained records this keeps the 19 dolly views. It drops the
10 opening views, leaves out the 12 consistent pan views across the jump, and
flags `pose_refinement_not_accepted` and `task_route_frames_left_out` (views 000
and 199). 19 views are too few to hold any back. The kitchen's 16 views are sent
unchanged. A splat bought before this record existed keeps what it was sent.
The draft, gap completion and mesh registration still use every posed view. The
result's `appearance` carries `splat_input` (flags and counts) and
`heldout_quality`.

## Rebuilding a ready job's refined splat

Owner approval 2026-11-08, about 20:45Z: "fix the input first, then do the
rebuild". A refinement needs a new job root (`atlas_refinement_requires_new_job`),
and retained data is never deleted. So a rebuild runs beside the original
(`atlas_splat_rebuild`, `website_atlas_pipeline._run_splat_rebuild`):

- **Who qualifies.** A ready website job whose scene was kept as filmed, with
  Atlas's refined splat of the captured views and no generated geometry, bought
  before this release recorded an input selection, whose own pose shows the
  cause: refinement not accepted, or a view that disagrees with its measured
  neighbours. One rebuild per job.
- **How it starts.** Only through the door's existing
  `selected-handoff --atlas-retry`, after `--atlas-status` shows
  `job.splat_rebuild` (reasons, `expected_cost_usd`, `maximum_cost_usd`,
  `allowance_covers_maximum`). The retry refuses with
  `selected_handoff_atlas_splat_rebuild_allowance_insufficient` unless the job's
  remaining Atlas allowance covers the maximum. It writes
  `pipeline/atlas/splat_rebuild_request.json` and reopens the job.
- **What it buys**, in `pipeline/atlas/splat-rebuild/<parent digest hex>/`, on
  the next controller ticks, through the job's own allowance and paid admission:
  - a new selection under the current policy (96 time-spread views, relative
    sharpness floor);
  - a new source pose (about $0.01 a view);
  - the scale measurement on it: one segmentation call for the named references
    (`atlas_reference_masks`, on the upstream ledger), and a planner scale review
    only if they disagree;
  - then the refined splat ($5) and the mesh ($0.50), but only if World Labs
    accepted pose refinement, enough views agree and the allowance still covers
    both.

  It buys no completion and no new task review.
- **What it leaves.** The parent's result stays as
  `splat-rebuild/<hex>/parent_result.json`, and every earlier pose, splat, mesh and
  bundle stays where it is. A completed rebuild becomes the job's `result.json`
  (`splat_rebuild` names the parent), and the controller archives and delivers it
  as a new revision. A stop (refinement not accepted, too few consistent views,
  allowance, the new pose's draft refusing) is recorded in `outcome.json` and the
  job is ready again on its parent result. Nothing more is bought, and the stop is final.
- **Downstream is not automatic.** Scene preparation (levelling, the navigation
  scene, the job-page scene preview, G1 registration) reads the Atlas result only
  when its native preparation runs. A completed native preparation does not run
  again because the Atlas result changed. `--resume-preparation` replays only a
  held handoff, and `--reissue-preparation` needs the sponsor's renewed grant. The
  scene preview prefers the prepared base scene, so it keeps showing the old
  splat until preparation runs again.

## Reviewed runtime policy

The signed WebApp policy (`BLUEPRINT_WEBSITE_ATLAS_RUNTIME_POLICY_JSON`) is the
rate-card authority. These optional fields are accepted by the WebApp:

```json
{
  "rgbd_view_cost_usd": 0.05,
  "additional_rgbd_view_cost_usd": 0.01,
  "rgb_view_cost_usd": 0.04,
  "additional_rgb_view_cost_usd": 0.008,
  "refined_splat_cost_usd": 5.0
}
```

`refined_splat_cost_usd` selects the refined splat. The additional-frame rates
apply World Labs' 2026-10-29 pricing: the first frame of each request keeps its
rate, and each further frame in the same request costs the additional rate (30
RGBD frames in one request reserve $0.34). Policies without them keep their
original quotes and receipt identities. Update `rate_card_reference`,
`reviewed_by` and `reviewed_on` with the change.

Sources checked 2026-10-29:
[posed images to splats](https://docs.atlas-beta.worldlabs.ai/posed-images-to-splats),
[request/response reference](https://docs.atlas-beta.worldlabs.ai/api-reference/posed-images2-splats),
[splat guidance](https://docs.atlas-beta.worldlabs.ai/splat-best-practices),
[billing](https://docs.atlas-beta.worldlabs.ai/billing).
