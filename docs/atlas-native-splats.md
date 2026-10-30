# Native Atlas splats

ADP-040 / day 28 appearance construction and the existing beta preparation
journey. The completion artifact is a retained `posedImages2Splats` operation,
its original SPZ, decoded standard PLY, input binding and portable bundle.
These are development-only appearance, with unproven metric scale, output
geometry preservation, collision and physics.

The previous Atlas adapter could only fuse estimated depth into draft splats.
The new beta API reconstructs and refines splats directly from posed RGBD.
Use the existing allocator and durable task runner; do not submit a separate
untracked CLI request or replace the original capture.

## Flow

1. Reuse the admitted dense survey and its existing quality, static-state,
   privacy and overlap selection. The splat input selection grows beyond the
   minimal edit set to a target of 16–30 useful, distributed real views. That
   target is not an API requirement: fewer distinct qualified views remain
   usable and their coverage limitation is recorded. The API minimum is two.
2. Pass object-free frames through unchanged. Frames showing a removal target
   use the existing exact-view visibility check, composited edit and independent
   review. Missing masks are checked before editing. Existing approved edits
   can be adopted; the historical three-view run stays intact.
3. Pose the approved views with `images2PosedRGBD`. Keep each returned image,
   linear EXR depth and camera together. Mixed grids use the existing calibrated
   photograph tiler. Source observations never become generated evidence.
4. Submit one `posedImages2Splats` operation. It is the first reconstruction
   attempt; local depth fusion and the old up-front gap-generation campaign are
   not prerequisites. Atlas performs its own internal rendering and refinement.
   Do not count those internal generated views as additional real observations.
   Additional explicit generation must address gaps absent from the recording,
   through the existing source-first review/recovery seam.
5. Persist the operation ID before polling. Pending and uncertain submissions
   resume through the same task runner, reservation and idempotency key. A
   30-minute operation is supported; it does not require a held-open request.
   The existing preparation watchdog and withdrawal checks remain in force.
6. Retain all returned SPZ variants and select the highest published count for
   the final appearance. Verify byte counts, SPZ header/count and digests, then
   decode through the existing pinned `splat-transform` runtime. Decoder
   availability is checked before purchasing. The SPZ is kept unchanged.
   Optional `splats2Mesh` consumes this exact native splat; the legacy
   depth-disc reduction must not rewrite it.

`atlas_refined_splat_manifest.json` distinguishes native refined splats from
`atlas_depth_splat_manifest.json`. Export, mesh registration and scene handoff
read either contract. The download includes the native receipt, original SPZ
variants, selected SPZ/PLY and input binding. Native refinement does not prove
that the provider preserved source geometry.

## Reviewed runtime policy

The signed WebApp policy remains the rate-card authority. Add these fields to
the existing reviewed policy, retaining the same owner/source grant and total
allowance:

```json
{
  "rgbd_view_cost_usd": 0.05,
  "additional_rgbd_view_cost_usd": 0.01,
  "rgb_view_cost_usd": 0.04,
  "additional_rgb_view_cost_usd": 0.008,
  "refined_splat_cost_usd": 5.0
}
```

Also record the current `rate_card_reference`, `reviewed_by` and `reviewed_on`.
The `refined_splat_cost_usd` field selects the native path. Historical policies
and their completed receipts retain the legacy route and original quotes.
Native background runs use a separate `automation/native_splats_v1` root and
may adopt approved edits from the preceding background.

Each render request costs its first-frame rate plus its additional-frame rate
times the remaining output frames. For example, 30 RGBD output frames in one
request reserve $0.34. Starting another request charges another first frame.
Pose inputs, edits, reviews, native splats and optional meshing continue to
share the existing source-bound allowance. Reserve the final splat/mesh costs
before buying upstream work. Estimated reservations are not provider bills.

The current OpenAPI request schema does not expose the guide's optional draft
switch. This adapter uses the refined default rather than sending an
undocumented `refine` field.

The existing `website_atlas_operator --task-spec` entry point accepts
`posedImages2Splats` with the usual task-context, reviewed request digest,
retention decision and admission. A valid World Labs key alone does not replace
Blueprint source/spend authority. Keep ordinary worker routing and timer
ownership unchanged while another lane is advancing an existing capture.

Sources checked October 29, 2026:
[posed images to splats](https://docs.atlas-beta.worldlabs.ai/posed-images-to-splats),
[request/response reference](https://docs.atlas-beta.worldlabs.ai/api-reference/posed-images2-splats),
[splat guidance](https://docs.atlas-beta.worldlabs.ai/splat-best-practices),
[billing](https://docs.atlas-beta.worldlabs.ai/billing).
