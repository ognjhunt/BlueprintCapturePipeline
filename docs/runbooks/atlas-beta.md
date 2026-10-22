# Atlas beta source preparation

Production uses `blueprint_pipeline.website_atlas_pipeline`. Website uploads
reach it through the worker described in the next section. It is the capture's
only paid reconstruction lane: scene preparation waits for it and reads its
outputs (`docs/website-atlas-scene-source.md`), and never skips its own
sponsorship for it.
The worker loads current task/source consent and the runtime rate policy from
the signed WebApp `atlas-configuration` endpoint. The source-bound authorization
must explicitly include OpenAI planner disclosure. Planner and Atlas reservations
share the same dollar/request cap; legacy native/Marble authority is untouched.

`BLUEPRINT_WEBSITE_ATLAS_RUNTIME_POLICY_JSON` selects the role model (initially
`gpt-6.1-sol`) and a reviewed per-call allowance/rate card. The existing Agents SDK
uses structured output, disabled tracing and `store=False`. It reviews candidate
frames, proposes static intervals and required task/room anchors, and records
actual model, SDK, usage and cost. CPU checks validate source hashes, unsafe-frame
exclusion, coverage, quality, overlap and budget. Every first run then densifies the
reviewed still interval and reviews the denser survey again, without reading frozen
held-out pixels. Retained
plans and operation IDs survive restarts without repeat purchases. Unknown model
or provider submission outcomes hold until the operation list reconciles them
(`docs/website-atlas-pose-operation.md`).

After posing, the controller generates RGBD at both task and room anchors, retains
every asset, and writes one portable ZIP. Its terminal status remains `needs_input`
while the public API lacks splat creation or metric registration is unproven.
Use the same controller entrypoint for an explicit production-host test; it needs
only server-resident video, normal secret integration and capture/request identity:

```bash
python -m blueprint_pipeline.website_atlas_pipeline --source-video VIDEO \
  --output-root ROOT --request-id REQUEST --scene-id SCENE --capture-id CAPTURE
```

ADP-040 / Day 28, owner-directed Atlas integration. Extend the admitted
`website_atlas_pose` producer; do not submit Atlas through the Marble world API.
The source adapter remains interchangeable and all results are development-only.

## Automatic previews of website uploads

Owner decisions D1-D6 (2026-10-22): consented website uploads get an Atlas 3D
preview automatically, beside the existing preparation, under a Blueprint-paid
per-upload cap the WebApp grant enforces. It is development-only and unscaled.

- **Routing.** Intake reads `BLUEPRINT_WEBSITE_ATLAS_ROUTING` from
  `/etc/blueprint/pipeline-control-plane.env`: `off` (the default, and any
  unreadable value), `allowlist` (walkthrough IDs in
  `BLUEPRINT_WEBSITE_ATLAS_CAPTURE_IDS_JSON`) or `consented` (every walkthrough).
  A routed upload gets `pipeline/atlas/preview_job.json`; recording it never
  raises into intake. A malformed setting routes nothing and breaks nothing else.
- **Worker.** `blueprint-website-atlas-preview.timer` runs
  `python -m blueprint_pipeline.website_atlas_preview tick` as `blueprint`. Each
  tick moves each job one bounded step (one poll per provider stage) and keeps
  it in one plain state: `processing`, `ready`, `needs_input`, `failed` or
  `withdrawn`. A missing grant or terms waits as `needs_input` and is re-checked
  every 30 minutes; transport errors back off and fail after 8 tries; jobs end
  after 7 days. With routing off no job exists and the tick does nothing.
- **Secrets.** Only the worker reads `/etc/blueprint/website-atlas-preview.env`:
  `WORLDLABS_ATLAS_API_KEY` (or `_FILE`), `WORLDLABS_ATLAS_API_BASE_URL` and
  `BLUEPRINT_ATLAS_PROVIDER_RETENTION`. The key needs `assets.delete` besides the
  scopes in `docs/website-atlas-pose-operation.md`.
- **Privacy.** Every frame Atlas receives is face-screened first
  (`atlas_privacy_screen`, YuNet, refusing when the model or OpenCV detector is
  missing) and recorded in `atlas_view_selection.json`; the planner still excludes
  frames with people. The download bundle carries no input frames.
- **Archive and cleanup.** A finished preview is published to the dedicated B2
  store (never the legacy fallback) as `atlas-scene-bundle/sha256/<digest>/<capture>.zip`
  with a full readback; withdrawal deletion finds it by that name, also after the
  workspace is retired. Then every Atlas asset the job uploaded or received is
  deleted at World Labs. Operation records stay there for 30 days.
- **Withdrawal.** The tombstone is re-checked every tick and before every paid
  stage. A withdrawn job archives nothing, deletes its Atlas assets and tells the
  WebApp.
- **WebApp.** Each state change is pushed through the signed `atlas-preview`
  operation. The owner download is streamed from
  `GET /api/live-pipeline/website-atlas-previews/<capture>/bundle?request_id=&digest=`
  (WebApp client only), which serves a ready, unwithdrawn bundle of that exact digest.

## One video, even while people work

Owner direction (2026-10-22): a site sends one video. We ask for a still scene with
nobody in view, but the job must work when people keep working in it.

- **Transients are not a new scene state.** People, hands, carried items and moving
  equipment are transient. Frames showing them stay usable and are listed in
  `transient_frame_ids`, with each kind named by a segmentation prompt (`person` is
  always added). A still interval spans one state of the room and its doors,
  drawers and lids.
- **Unusable frames are left out, not fatal.** An unsafe frame inside an interval
  (someone hides most of the view or the task object, motion blur, another door
  state) is recorded in `excluded_frame_ids` and never becomes a candidate.
  Required and additional views can never be unsafe.
- **People are erased before Atlas reconstructs anything.** A transient in a view
  the editor processes (the required and additional views) sends the scene down
  the editor-first path: SAM masks every instance of each transient in those
  views, the editor erases them with the task objects, and the reviewer rejects
  any view still showing one. On the direct path, no frame showing a transient
  is posed (`atlas_video_inputs.takes_reviewed_background`).
- **A carried task object is described where it rests.** Scene preparation lifts a
  task object only from the views that agree on one resting place
  (`website_task_masks.resting_views`). A still object seen from opposite sides
  stays whole: views count as one place within the object's own visible extent.

## Accuracy-first input selection

Owner direction (2026-10-22): accuracy, not cost, decides the provider inputs.
After the first review, `accuracy_survey` adds candidates every 0.1 s inside the
reviewed still interval (coarser only when the interval is long, so the second
review stays under 300 images), and the planner reviews that denser survey before
any selection. A second review of about 80 images reserves about $0.56 worst case,
so the runtime policy's `maximum_review_cost_usd` must allow it (about $4 covers a
300-image review).
`prepare_video_inputs` then builds the connected cover of the reviewed task and
room anchors, bounded only by the reviewed dollars, and `add_distinct_views` grows
it: each step admits the sharp, unclipped candidate farthest in time from the kept
set that still overlaps a kept time neighbour (the cover's support rule) and moved
at least 12 preview pixels from each overlapping one (the near-duplicate rule).
Every posed view therefore overlaps another, earlier context is reached through
its own bridges, and a binding capacity keeps the widest connected spread. Added
views stop at the Atlas pose adapter's configured input capacity
(`website_atlas_pose.MAX_POSE_INPUT_IMAGES`, 64, provider data per the package
AGENTS.md); the cover is never cut by it. The reviewed dollars are checked only
after selection and refuse (`atlas_selection_budget_below_accurate_inputs`); they
never pose fewer frames. Atlas has not published a per-request image limit, so the
first 64-frame request settles whether it accepts that many.

## Review output gaps and refine source inputs

The initial cover connects reviewed task/room anchors and is then densified as
above. For an output-aware second pass, retain an
`atlas_reconstruction_feedback.v1` JSON with a canonical `digest`, `parent_root`,
`parent_result_digest`, `parent_selection_sha256`, and `previews` entries containing
`relative_path`, `sha256`, and `source_frame_id`. Previews are labeled derived
reconnaissance, must reference already-selected source cameras, and never become
new source observations. Include renderer provenance and diagnostic notes in the
feedback manifest. Keep image bytes next to the feedback file.

Run the same controller with a **new** `--output-root` and
`--refinement-feedback FEEDBACK.json`. It verifies the parent capture/context and
all preview hashes, densifies candidates inside the previously reviewed static
interval without reading frozen holdouts, and invokes the same sponsored agent.
The structured refinement names gaps and either retains the result or specifies
additional input IDs. Original inputs, static intervals, and quality thresholds
are preserved. New inputs require overlap and quality checks. Input additions
are independent of synthetic output camera count, avoiding unnecessary generation
charges. Every new operation shares the original WebApp dollar/request ledger;
neither the authorization nor cap is reset. Each revision retains its parent
result digest, review, costs, and independent provider operations.

An unsafe flag inside the selected interval still refuses privacy/state review
unless that unselected frame deterministically fails the retained blur/clipping
threshold. Such quality failures are recorded and never selected; they do not
invalidate the sharp static inputs. Required or additional unsafe inputs always
refuse. Never lower thresholds to admit them. Keep the pushed experimental branch
available until the canary is terminal: repository auto-delete-on-merge can remove
that ref and correctly stop subsequent admission before any provider purchase.

Inspect retained artifacts privately with the existing Spark/Three dependencies:

```bash
cd tools/splat_render
npm ci --ignore-scripts
node view_scene.mjs --splat SCENE.spz --mesh SUPPORTED.glb --cameras CAMERAS.json
```

Open the printed loopback URL. Orbit/pan/zoom, select captured cameras, and toggle
the splat, supported mesh, or wireframe. Camera rows use the renderer's existing
`{id, spec: {pos, target, up, fov}}` convention. Nothing uploads to another viewer
service. This is an inspection preview, not a qualified method render or metric
measurement. For a remote worker, forward the loopback port through SSH.

Read the current [quickstart](https://atlas-beta.worldlabs.ai/docs/quickstart.md),
[input guide](https://atlas-beta.worldlabs.ai/docs/images-to-posed-images),
[camera conventions](https://atlas-beta.worldlabs.ai/docs/cameras-and-posed-images),
and [billing](https://atlas-beta.worldlabs.ai/docs/billing). Beta schemas and
prices can change. The posing schema publishes a minimum of one image and no
maximum; this is not proof of unlimited server capacity.

1. Run `python -m blueprint_pipeline.atlas_video_inputs survey --source-video
   VIDEO --capture-id ID --output-root ROOT`. The existing decoder freezes a
   hidden holdout. Only candidate frames appear in `candidate-survey.jpg`.
   Initial survey size follows video duration and a per-run `--sampling-policy`;
   `--sample-count` is an explicit override, not a provider limit.
2. The harness inspects that sheet and records `atlas_scene_state_review.v1`
   with its `survey_digest`, `reviewed_by`, `scene_state`,
   `same_scene_state: true`, `moving_people_excluded: true`, and
   `static_intervals_seconds: [[start, stop]]`. Frames inside an interval that
   show people or other transients go in `transient_frame_ids` with a
   `transients` list; unusable ones go in `excluded_frame_ids`. `anchor_frame_id`
   selects the task's visible subject. `coverage_requirements` must include
   `task_object` and `surrounding_scene` roles, each with a reviewed `label`
   and `required_frame_ids`. Add separate groups for each needed object and
   interaction/support region. Do not combine different door/rack
   states. Image matching does not prove scene-state identity.
3. Run `python -m blueprint_pipeline.atlas_video_inputs prepare --survey-path
   ROOT/atlas_video_survey.json --review REVIEW --output-root RUN_ROOT`.
   Selection connects every required object/context anchor with overlap
   bridges and retains its decisions. A disconnected region blocks preparation
   rather than silently yielding only object close-ups. Per-run
   `selection_policy` controls sharpness, clipping and match thresholds.
   There is no fixed 24/64-frame input cap. `selection_budget` requires
   `maximum_pose_cost_usd`, current `input_cost_usd`, `rate_card_reference`,
   `reviewed_on` and `reviewed_by`; it determines the planning ceiling in images.
   The separate WebApp gate remains the actual authority to spend or upload.
   Atlas receives lossless native upright source images, rather than 518-pixel
   model inputs. Gaps remain explicit. Existing source-geometry
   orchestration uses this path when its root has an `atlas_scene_review.json`.
   For a temporal gap, run `python -m blueprint_pipeline.atlas_video_extension
   --source-video VIDEO --base-survey SURVEY --decoded-index INDEX --review
   REVIEW --output-root EXTENSION_ROOT`. It derives denser sampling from the
   input cadence, preserves every original hidden frame exclusion, and labels
   added frames as development-only training observations. Review that new
   candidate sheet and bind the updated survey digest before preparation.
4. Obtain the WebApp's explicit source-bound Atlas beta authorization. It
   records the owner-accepted retention/training terms and a separate test cap.
   A key, a rate estimate, or an old Marble sponsorship is insufficient.
5. Record the current rate card in `website_atlas_pose_price_review.v1` as
   documented in `docs/website-atlas-pose-operation.md`. Pin these options:
   `refinePoses: true`, `base64Outputs: false`. Bind the selected count and grid.
6. Run `python -m blueprint_pipeline.website_atlas_operator --output-root
   RUN_ROOT --task-context CONTEXT --price-review REVIEW --api-key-file KEY_FILE
   --provider-retention project_library_retention_approved`. Set the existing
   signed WebApp transport in the process environment. Keys never appear in
   arguments or logs. `WLT_API_KEY` and `WLT_API_BASE_URL` are also supported.
   `--experimental` requires a clean, remotely retained `codex/` commit;
   production retains its exact-main release gate. Repeat the same command to
   resume; it retains uploads, the idempotency key and the operation ID.

The output bundle keeps RGB, camera-to-world XYZW poses, depth and confidence
together on the returned grid. Do not substitute original RGB after posing.
Copy the retained run into authorized company object storage before closeout.

As of the 2026-10-21 documentation, the public API exposes images2PosedRGBD,
atlasGenerate, atlasMasked, atlasChisel, atlasTextToImage and splats2Mesh.
Posing returns cameras/depth, not a Gaussian splat. `splats2Mesh` needs an
existing splat. A splat-creation price on the platform is not a published API
endpoint. Preserve this distinction when planning the full appearance and mesh
package. Both need independent scale and collision/articulation validation.

## Generation, editing, and mesh API tasks

Use the same operator with `--task-spec SPEC.json` for `atlasGenerate`,
`atlasMasked`, `atlasChisel`, `atlasTextToImage`, or `splats2Mesh`. The spec
contains `task`, `request`, `source_video_digest`, and optional `uploads`.
Local uploads are a map of label to `{path, sha256}`; reference them in the
request as `{assetId: "upload:label"}`. Existing inputs use private project
asset IDs. Every task uses the same source-bound WebApp consent and aggregate
spend cap. A separate `website_atlas_task_price_review.v1` records `task`,
the canonical `request_digest`, `maximum_cost_usd`, `reviewed_on`,
`rate_card_reference`, and `reviewed_by`. Do not reuse a pose price review.

The pinned official request and response schemas are in `_atlas_schemas/`.
The adapter validates them before reservation and after completion, retains
upload intent and the operation ID, and downloads every named output with
byte-count and SHA-256 checks. An unknown POST outcome holds until the operation
list shows what it bought; rerunning never purchases a second operation. A known operation resumes without
another reservation. Billing stays `unreported` until provider usage is retained.

Generation uses explicit 1280×720 target cameras. The returned RGBD context can
retain its original grid. Do not distort portrait source observations to fill
landscape targets. Reuse `promptUsed` with `enhancePrompt: false` for later
views of the same scene. Chisel and masked edits require paired context and
target cameras; output depth needs translated targets. Their exact constraints
are documented in the corresponding task guides.

`splats2Mesh` consumes an existing `.spz` and returns a GLB. Its adapter refuses
an image masquerading as a mesh. There is no images-to-splat task in the live
2026-10-21 public OpenAPI contract. World Labs staff also report that Atlas
splat generation/export is not available yet; a visible Studio splat control
does not establish availability. Do not silently fall back to that control or
to the incompatible legacy Marble world API. Required generated RGBD views
are budgeted before selecting or purchasing pose inputs.

Export a completed pose run with `python -m blueprint_pipeline.atlas_result_bundle
--pose-root RUN_ROOT --output-root BUNDLE_ROOT`. This verifies the authoritative
receipt and emits a reproducible ZIP of native inputs, returned color/depth/
confidence, cameras, and evidence. The manifest explicitly records missing
splat/mesh and unproven metric scale; a downloadable RGBD package is not a
qualified simulation scene.
