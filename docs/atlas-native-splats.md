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

Preserved-capture scenes and the plain pose-and-generate route keep the draft
for now.

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
