# Atlas beta source preparation

ADP-040 / Day 28, owner-directed Atlas integration. Extend the admitted
`website_atlas_pose` producer; do not submit Atlas through the Marble world API.
The source adapter remains interchangeable and all results are development-only.

Read the current [quickstart](https://atlas-beta.worldlabs.ai/docs/quickstart.md),
[input guide](https://atlas-beta.worldlabs.ai/docs/images-to-posed-images),
[camera conventions](https://atlas-beta.worldlabs.ai/docs/cameras-and-posed-images),
and [billing](https://atlas-beta.worldlabs.ai/docs/billing). Beta schemas and
prices can change. The posing schema publishes a minimum of one image and no
maximum; this is not proof of unlimited server capacity.

1. Run `python -m blueprint_pipeline.atlas_video_inputs survey --source-video
   VIDEO --capture-id ID --output-root ROOT`. The existing decoder freezes a
   hidden holdout. Only candidate frames appear in `candidate-survey.jpg`.
2. The harness inspects that sheet and records `atlas_scene_state_review.v1`
   with its `survey_digest`, `reviewed_by`, `scene_state`,
   `same_scene_state: true`, `moving_people_excluded: true`, and
   `static_intervals_seconds: [[start, stop]]`. Optional `anchor_frame_id`
   selects the task's visible subject. Do not combine different door/rack
   states. Image matching does not prove scene-state identity.
3. Run `python -m blueprint_pipeline.atlas_video_inputs prepare --survey-path
   ROOT/atlas_video_survey.json --review REVIEW --output-root RUN_ROOT`.
   Selection rejects blur, clipped exposure, near duplicates and disconnected
   viewpoints. It chooses an overlapping chain around the reviewed anchor and
   retains its decisions. The ceiling defaults to 24, not a fixed input count.
   Atlas receives lossless native upright source images, rather than the old
   518-pixel MapAnything inputs. Gaps remain explicit. Existing source-geometry
   orchestration uses this path when its root has an `atlas_scene_review.json`.
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
