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
   New edits additionally receive a target-guide overlay as the last visual reference,
   aligned to the first image's canvas. This identifies the target without
   using an API mask or pasting the fill through its silhouette; shadows,
   reflections and contents can extend beyond the highlighted region. The
   overlay and its checksum are retained with the image receipt. Completed
   earlier edits remain reusable; uncertain earlier intents still refuse.
   New edits use a separate immutable backend row with custom output dimensions
   at the complete source aspect ratio (rounded to the API's 16-pixel grid).
   The previous resolution budget, model, quality and price cap are unchanged.
   No black letterbox border is added or cropped away. Camera-change repairs
   require this canvas and cannot adopt a completed legacy crop; uncertain
   legacy requests still refuse. Canvas policy and dimensions are retained in
   each new image receipt. Panoramas outside the provider's 1:3 to 3:1 aspect
   range refuse before a provider request rather than silently losing coverage.
   Provider failures retain the shared adapter's sanitized status, error code
   and request ID with the non-completed intent. They keep the full reservation
   and require reconciliation; restarting cannot purchase the uncertain call
   again. Raw error bodies, headers and secrets are never retained.
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
   Repairs restore the original source mask's pixel count rather than copying
   the completed edit's zero-remaining marker. Original image/mask digests,
   binary mask values and grid alignment are checked before buying a repair
   plan. Completed correction receipts remain reusable on restart; the original
   source cannot silently substitute for a requested correction. An unchanged
   correction retains the failed verdict without buying the same review again.
   Each repair plan has its own inference audit directory. Returned plans are
   saved before audit finalization. For a legacy lost output, only a validated
   SDK completion paired with the exact input reservation can establish a
   completed charge and permit an explicitly bound replacement plan. Its charge
   is retained; missing or uncertain completion evidence still blocks a retry.
   Before that review is bought, a deterministic camera check
   (`edited_view_camera.py`) compares each edited view with its original outside
   the dilated removal mask. A coherent transform supported by distributed
   shared features rejects a substantial camera shift without a paid review,
   with `camera_changed_frame_ids`. Low texture retention alone does not prove
   movement: repainted wood grain can leave very few matches even when cabinet
   edges remain aligned. Such views require the independent visual review,
   which must explicitly name camera changes and reject copied reference
   compositions. Minor texture changes and local distortion remain acceptable.
   Missing or invalid camera findings block and retain the completed answer;
   they cannot silently approve a view or become a reason to rebuy that review.
   Novel-camera generation is reviewed under its separate viewpoint contract.
   The repair agent re-edits views with a changed camera
   from their original without the anchor reference they copied, and a rejected
   anchor is re-edited once with its camera named. Views with too few features
   outside the mask, or whose retained mask is gone, stay with the model review.
   `python -m blueprint_pipeline.edited_view_camera <prepared_background.json>`
   re-checks an already approved set without provider calls; a moved or
   unverifiable camera exits nonzero and requires review.
6. The agent surveys missing surfaces across the captured scene and chooses
   named gaps and candidate cameras, including modest translated viewpoints.
   Additional scene-depth-relative viewpoints stay inside source-observed free
   space and survey both nearby surfaces and the upper surrounding scene.
   Their projected scene outline can expose open disocclusions between known
   boundaries; it is a planning hypothesis, not evidence of a wall. The agent
   selects supported continuations and leaves ambiguous voids unknown. Nothing
   outside the projected outline or capture-derived 3D bounds is added.
   Model input uses a compact table of every camera and region; full precision
   masks and survey receipts remain unchanged for selection and admission.
   `atlasGenerate` receives all approved cleaned photographs with aligned depth
   and cameras. Images/depth/cameras rotate and resize together; full photograph
   coverage survives. Returned RGBD passes independent review and source-priority
   admission before final splat construction. Known geometry and observed free
   space take precedence; native outputs remain retained. Prior passed Chisel
   outputs are adopted without buying their generation again. Generation is an
   estimate and neither a sparse preview nor a successful mesh export proves
   complete scene coverage.
7. Passed generated RGBD uses the shared depth-splat writer and existing
   `splats2Mesh` / registration / download paths. Generated backgrounds get a
   separate ZIP and explicit result/review lineage. Original reconstruction and
   its ZIP remain available. Native photometrically trained splat export remains
   subject to Atlas availability; this adapter produces a depth-fused draft.

   Before purchasing the final mesh, the harness re-surveys the exported splat
   from every source and approved generated view. Reviewed depth supplies a
   foreground-surface hypothesis: a floor or wall behind a missing work surface
   does not count as closing that hole. The same scene agent prioritizes material
   task/support/access gaps, aims cameras at them, and uses the original and
   approved generated RGBD together as Atlas references. Each accepted repair is
   fused and checked again. Small distortions are acceptable; the agent can stop
   when task coverage is sufficient. A repair that adds no exportable geometry
   stops with an explicit unresolved gap. Completed images are reused, all calls
   share the existing allowance, and this check does not qualify simulation or
   physical fidelity.

   Before another novel-view purchase, the agent also checks the original
   capture candidate gallery, including recorded views omitted from the current
   reconstruction. Each selected gap cites its source frames and distinguishes
   an observed surface, an occluded surface, an unresolved area, and a physical
   opening. Original views are individual full-frame previews; they remain
   decision evidence and never bypass cleaning into Atlas generation inputs.
   A useful unconditioned recorded view returns `source_recovery_required`
   before generation. The controller automatically reviews those exact source
   views, appends removal regions to frozen targets, reuses previously approved
   edits, cleans only the new views, and independently reviews the expanded set.
   It obtains new depth with shared cleaned anchors, then uses the existing
   shared-view depth fit to register the complete fresh posed set into the original
   coordinate frame. Shared IDs replace their earlier derived RGB, depth and
   cameras in both geometry and conditioning; added IDs enter exactly once.
   The next generation uses every current approved photograph, including the
   added sources, and their current depth. The fresh pose root supplies pixel
   verification and planner photographs while the original world frame remains
   the placement authority. Mixed capture identities or missing current frames
   refuse before generation. Original pose files, task-state selection, native
   generations and receipts remain unchanged; a versioned recovery report
   retains the refreshed bindings without overwriting an earlier report.
   A moved articulated target is acceptable only in this background-only stage
   when completely erased; retained fixtures must still agree. Uncertain
   visibility, changed fixed layout, failed cleanup or weak alignment stays an
   explicit held result. Rejected recorded views and their reasons accompany
   the next review. Every call remains on the existing source allowance.
   Physical openings are excluded from fill masks. Candidate sampling alone
   never proves absence from every video frame. Subsequent reviews also receive
   per-view requested pixels, available native depth, and admitted depth, so an
   attractive RGB fill cannot hide a failed 3D repair. Passed native generations
   and their verified fused pixels are adopted without another purchase.

   The post-export review uses full-frame geometry previews and records
   `environment_surfaces` for the complete connected local work fixture: its
   worktop, outer faces, ends, edges, supports, and operating floor as applicable.
   A `sufficient` decision requires geometry-preview evidence and cannot contain
   a material gap or unresolved required surface. Minor seams/speckles and real
   openings are acceptable; concealed interiors, distant scenery, and physical
   fit/force qualification remain separate unknowns. A narrow old task-extent
   summary does not excuse a large missing part of its support fixture. The
   earlier planner inputs and completed native repair receipts remain retained.
   If the full review exceeds its existing request cap, the planner fits its
   input before reserving money: every camera, gap and source view stays present,
   camera/gap metadata uses labeled tables, and derived overlays use labeled
   contact sheets. Original capture previews retain the largest affordable
   full-frame resolution. Response tokens stay inside the SDK limit and the
   same reviewed request cap; source images sent to Atlas are unchanged.
   Legacy affordable requests retain their exact input serialization and
   binding. An unfundable request still refuses locally rather than omitting
   observations or increasing the capture's cumulative allowance.

When a refinement adds source views, its parent prepared-background manifest
can seed the expanded set. Source video, task context and original RGB pixels
must match, and the accepted edits stay byte-identical. Only missing views are
edited; the complete expanded set receives a fresh independent review. The
parent manifest and assets remain retained, and reuse is recorded in the result.
If SAM returns no unambiguous instance, the image editor may use the agent's
validated source region as a location guide. Its receipt explicitly distinguishes
that region from segmentation. It is never a measurement/geometry mask, never
clips the generated image, and still requires the same independent image review.

All provider stages retain existing durable intents/receipts and reserve against
the same source-video allowance. Meta segmentation and Google review additionally
need their own accepted provider terms; Atlas consent cannot grant another
provider's disclosure. An automatic preview grant carries that consent itself,
pinned to each provider's terms when the grant is made (Blueprint-WebApp #1097);
an explicit owner authorization still needs the native sponsorship. Forecasts cover the main
route, while each actual reservation remains the transaction authority. Fallback
work requires remaining allowance. If only an otherwise admitted GCP cost report
has aged out while its monitor is paused or refreshing, the existing Atlas
controller returns `processing` at `paid_admission_wait`. A later controller tick
adopts completed stages and must pass the original fresh-report admission again.
The waiting state grants no spending authority and does not rewrite the report.
Exhausted budgets, changed policy, month rollover, missing evidence and additional
rights or funding blockers retain their refusal.
There is no budget reset or automatic retry
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
