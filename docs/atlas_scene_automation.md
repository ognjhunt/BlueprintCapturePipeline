# Atlas scene automation

This is the ADP-040 / Day 28 development-only capture preparation adapter. It
extends the existing source-video planner and canonical paid-resource allocator;
it does not create a separate agent service or replace original capture evidence.

The production Atlas controller runs the following for a new scene plan:

1. The configured Agents SDK model considers candidate frames, task objects,
   surrounding room/access space, static state, and reconstruction gaps. It
   proposes useful frames, object regions, scale references, and removal intent.
   There is no scene-wide frame quota. Vendor request/context and spending limits
   still apply. Old plans without an automation proposal remain read-only.
2. Bounded hosted web search can research public category/manufacturer dimensions
   using generic queries. At most two search calls are reserved at $0.01 each,
   plus model tokens, inside the existing review allowance. The existing safe
   public-page fetcher retains up to two pages, verifies exact short quotes and
   unit conversion, and labels unmatched references unconfirmed. A manufacturer's
   generic dimension does not identify the actual captured appliance.
3. Hash-checked original posed RGB/depth/confidence/cameras are registered to
   upright source pixels. Named assumed lengths produce an approximate scale
   and range; deterministic robust visible-point bounds produce scene dimensions
   in provider units, meters, and feet. Hosted SAM supplies object masks for
   object extents. Missing depth/alignment yields explicit gaps. Raw geometry
   remains unchanged; these are estimates of visible coverage.
4. SAM masks cover the removal targets and all selected posed views. An existing
   image editor produces one clean anchor, independently reviewed with Gemini.
   Atlas `atlasMasked` gets the clean anchor, same-index source cameras, and
   grayscale 255-keep / 0-fill masks. Overlapping cover-crop tiles preserve long
   images without letterbox padding or stretching; camera intrinsics follow the
   exact resize/crop. Requests use up to 16 context views with a common anchor,
   without imposing an overall input-frame ceiling. Depth requires translated
   source views.
5. Every Atlas batch is independently reviewed. Failed edits cannot enter splat
   or mesh reconstruction. The existing image editor is a bounded fallback for a
   failed visual review; its reviewed RGB must acquire new posed depth before
   reconstruction, rather than using original appliance depth under generated
   pixels. An uncertain submission is held until the operation list shows
   whether it bought anything.
6. Passed generated RGBD uses the shared depth-splat writer and existing
   `splats2Mesh` / registration / download paths. Generated backgrounds get a
   separate ZIP and explicit result/review lineage. Original reconstruction and
   its ZIP remain available. Native photometrically trained splat export remains
   subject to Atlas availability; this adapter produces a depth-fused draft.

All provider stages retain existing durable intents/receipts and reserve against
the same source-video allowance. Meta segmentation and Google review additionally
require their existing native sponsorship and accepted provider terms; Atlas
consent cannot grant another provider's disclosure. Forecasts cover the main
route, while each actual reservation remains the transaction authority. Fallback
work requires remaining allowance. There is no budget reset or automatic retry
of an uncertain purchase. A blocked auxiliary stage preserves the source result
and available size estimates.

Artifacts include `scene_measurements.json`, `background/object_measurements.json`,
SAM masks, edited anchor/review receipts, `tile_lineage.json`, Atlas result assets,
the generated splat manifest, registered mesh, separate downloads, and the final
`atlas_scene_automation.v1` result. Physical evidence, measured metric calibration,
and simulation qualification remain false.

For a zero-spend retained-run measurement replay:

```bash
python -m blueprint_pipeline.atlas_scene_automation \
  --pose-root /path/to/retained/pose \
  --inputs /path/to/atlas_video_inputs.json \
  --review /path/to/scene_review.json \
  --output-root /path/to/new/replay
```

This command does not load credentials or invoke providers. A manually supplied
replay reference remains labeled an assumption and is not a live agent result.
