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


1. `atlas_native_splat.prepare_splat_request` posts exactly the frames the
   draft fused, each with its camera, linear depth and an alpha mask of the
   pixels the draft used (finite depth, confidence > 0). Frames on one grid go
   unchanged; mixed grids (portrait source beside landscape generated views)
   use the ray-preserving photograph tiler, with the mask taken from the
   projected depth. Generated gap views therefore enter the refined splat only
   where their review admitted them.
2. `run_native_splat` reserves the reviewed price, submits through the shared
   admitted task runner, and resumes a pending operation by ID (an operation
   takes about 20–30 minutes; each controller pass returns `processing`).
3. The largest returned count is kept. Its original SPZ is retained and decoded
   in Python (`atlas_depth_splat.decode_spz`; Atlas publishes SPZ v2 with
   degree-0 colour, bit-identical to `splat-transform` on a 2M-Gaussian Atlas
   splat) into a standard 3DGS PLY. Any other layout fails closed with the paid
   SPZ kept, so a retry never buys twice.
4. `atlas_refined_splat_manifest.json` binds the pose, the draft manifest, the
   generated-result digests, the request and every upload. Mesh registration
   keeps the draft's fusion grid as its tolerance. `splats2Mesh`, registration,
   the download bundle and scene handoff read either manifest.

The refined root sits beside the draft (`<draft root>-refined`), and its mesh
root carries the refined digest, so a new draft gets its own purchase and a
legacy mesh is never adopted for a refined splat.

Preserved-capture scenes and the plain pose-and-generate route keep the draft
for now. Choosing more real views for the splat (16–30 overlapping frames) is a
separate change: it alters which views are edited and posed.

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
