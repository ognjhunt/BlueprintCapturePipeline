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
3. When original posed RGB/depth/confidence/cameras already exist, they are registered to
   upright source pixels. Named assumed lengths produce an approximate scale
   and range; deterministic robust visible-point bounds produce scene dimensions
   in provider units, meters, and feet. Hosted SAM supplies object masks for
   object extents. Missing depth/alignment yields explicit gaps. Raw geometry
   remains unchanged; these are estimates of visible coverage.
4. For object removal, the existing editor prepares a clean anchor and then the
   current review's task/context/additional views. SAM uses source RGB pixels;
   a new capture does not need an original-scene Atlas pose to start editing.
   Older completed editor batches can supply exact candidate bytes without
   buying their images again. Superseded views do not silently become inputs.
   New edits additionally receive a SAM overlay as the last visual reference,
   aligned to the first image's canvas. This identifies the target without
   using an API mask or pasting the fill through its silhouette; shadows,
   reflections and contents can extend beyond the highlighted region. The
   overlay and its checksum are retained with the image receipt. Completed
   earlier edits remain reusable; uncertain earlier intents still refuse.
5. Gemini independently reviews the complete selected edited set before Atlas
   receives any RGB. Rejected views cannot enter reconstruction. Accepted RGB
   acquires fresh Atlas cameras and depth, rather than putting original
   appliance depth under generated pixels. The earlier Atlas inpainting path
   remains an explicit alternative (`editor_first=False`), with its own masks
   and per-batch review; it is not the default object-removal path.
   With the existing `BLUEPRINT_WEBSITE_IMAGE_REPAIR_AGENT` option enabled,
   rejected editor views enter the existing targeted repair agent. It can select
   as many repairs as this scene needs, using the passed anchor as its fixed
   reference; every call stays on the same source allowance. A fresh independent
   review of the complete set decides whether reconstruction can proceed.
6. Passed generated RGBD uses the shared depth-splat writer and existing
   `splats2Mesh` / registration / download paths. Generated backgrounds get a
   separate ZIP and explicit result/review lineage. Original reconstruction and
   its ZIP remain available. Native photometrically trained splat export remains
   subject to Atlas availability; this adapter produces a depth-fused draft.

All provider stages retain existing durable intents/receipts and reserve against
the same source-video allowance. Meta segmentation and Google review additionally
need their own accepted provider terms; Atlas consent cannot grant another
provider's disclosure. An automatic preview grant carries that consent itself,
pinned to each provider's terms when the grant is made (Blueprint-WebApp #1097);
an explicit owner authorization still needs the native sponsorship. Forecasts cover the main
route, while each actual reservation remains the transaction authority. Fallback
work requires remaining allowance. There is no budget reset or automatic retry
of an uncertain purchase. A blocked auxiliary stage preserves the source result
and available size estimates.

Artifacts include `scene_measurements.json`, `background/object_measurements.json`,
SAM masks, edited anchor/review receipts, `tile_lineage.json`, Atlas result assets,
the generated splat manifest, registered mesh, separate downloads, and the final
`atlas_scene_automation.v1` result. Physical evidence, measured metric calibration,
and simulation qualification remain false.

`background/prepared_background.json` retains the accepted edited RGB set before
posing or meshing. Resuming this job reuses those exact images without repeating
segmentation, editing or review, even in another operator session. It verifies
the source video, source pose, input selection, task context, automation plan,
review binding and image checksums first. Changed or rejected artifacts cannot
be adopted. Keep this manifest, its referenced files and provider receipts in
the retained job; a new geometry job does not require new image generation.
The fresh background pose owns its RGB inputs under `fallback_pose-v2/inputs`.
The canonical pose-input loader validates that handoff before dispatch. A
refused older sibling-directory handoff remains intact; an older submission
intent requires reconciliation before a different pose can be bought.

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
