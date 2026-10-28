# The Atlas job as the website scene source

Beta journey step: concurrent scene preparation. Owner direction, 2026-10-22:
website scenes rely on World Labs Atlas only; MapAnything is no longer part of scene preparation.

Each routed website capture has one Atlas job (`website_atlas_preview`, under
`pipeline/atlas/`). It is the capture's only paid reconstruction lane. Scene
preparation (`website_scene_handoff`) reads what the job retained and buys
nothing.

For an owner task that explicitly adds an absent synthetic subject, the current
production planner returns `subject_origin: synthetic_design` and covers the
planned interaction areas as `task_region`. Its validated plan must keep every
captured object and contain no removal instruction or selected transient to
erase. The controller then reconstructs the selected full-field source views
through the existing padded pose, scale, depth-fusion and mesh stages. It does
not buy unconditional novel views. Uncaptured surfaces remain explicit gaps.
The same scene-source handoff reads this preserved environment directly; no
object-removal manifest is manufactured. Scale remains an estimate, and the
handoff does not qualify the task, robot, synthetic asset or native physics.
An uncertain origin or a conflicting edit plan refuses before reconstruction.

Tasks that do not move, alter or add an object use the explicit
`subject_origin: no_manipulated_subject` plan. Navigation and inspection can
cover their route, stopping area or inspection region as `task_region`, alongside
the wider scene, without inventing a task object. This uses the same preserved
environment reconstruction and source handoff. Every observed object stays in
place; removal, inpainting and transient-containing selected inputs are refused.
An unobserved manipulated object is still uncertain, not a task without a subject.
This enables beta scene preparation (ADP-040/050, day 28 dependency rehearsal);
it provides no navigation policy, metric registration, native physics admission
or successful episode evidence.

## What the job buys, in order

For captured manipulated subjects, the existing replacement route is unchanged:

1. Review the survey, select views and prepare their pose inputs. The exact
   input document the pose binds is kept as `pose/atlas_geometry_inputs.json`.
2. Remove the task objects from the selected views, review the edited set,
   and pose it (`automation/`). This gives the object-removed background splat
   and its supported mesh. The original object's depth never sits under
   generated pixels.
3. Once the background is complete, pose the selected **original** views in
   `pose/` (`source_pose` in `result.json`). Scale is estimated from them,
   using the review's named scale references (`pose/scene_measurements.json`).
   A refused purchase is recorded as `source_pose.status: needs_input` with its
   reason. The finished background is kept.

## What scene preparation reads

`website_atlas_scene_source.atlas_scene_source` returns `ready`, `pending` or
`needs_input` with a reason.

- **Scene:** the background's PLY splat and supported GLB mesh, with up axis
  `Y` (Atlas worlds are right-up-back) and provider `world_labs_atlas`.
- **Source geometry:** the original views' pose, as `website_source_geometry.v1`
  (`atlas_source_geometry.atlas_posed_view_geometry`).
  - Each native upright view is resized by Atlas's cover scale, so one
    geometry pixel is one Atlas pixel. Crop margins stay invalid.
  - Depth and cameras are scaled to estimated metres. The scale basis is
    `reference_estimate`, bound to the measurement file's digest.
  - Frames keep their decoded frame identity and an exact source-to-geometry
    pixel transform, including the phone's display rotation.
- **Alignment:** the background was posed separately, so it lives in another
  Atlas world. The cameras both poses share give the similarity between the
  two worlds. Rotation comes from the cameras' orientations, so a
  near-straight walk still fixes it; scale and translation come from their
  centres. This is the registration anchor (`source_world_alignment`). Without
  agreeing shared cameras, registration falls back to its search.
- **Task masks:** hosted SAM 3.1 tracks on the continuous source video's
  timeline, so its tracks name each posed view by decoded frame index. (The
  opt-in `BLUEPRINT_WEBSITE_VIEW_FIRST_SAM` mode chose views for a Marble world
  and is retired; its retained masks still bind.) They lift onto the Atlas geometry only when every
  geometry frame is the registry frame of that index, at the same presentation
  time and upright size. Only frames a track observed contribute, so a target
  no posed view observed fails rather than guessing.

## What it waits for

| Reason | Meaning |
|---|---|
| `website_atlas_job_required` | The capture was not routed to Atlas (`BLUEPRINT_WEBSITE_ATLAS_ROUTING`). |
| `website_atlas_job_processing` | The job is still running; the handoff returns `awaiting_reconstruction`. |
| `website_atlas_job_<state>` | The job ended `needs_input`, `failed` or `withdrawn`. |
| `website_atlas_background_unavailable` | Object removal or its review did not complete. |
| `website_atlas_background_mesh_unavailable` | No reviewed mesh price, so there is no collider. |
| `website_atlas_source_pose_unavailable` | The original views were not posed (see the job's reason). |
| `website_atlas_scale_estimate_unavailable` | No named scale reference gave an estimate. |

Jobs that finished before this change have no source pose and report
`website_atlas_source_pose_unavailable`.

Nothing here measures anything. Scale comes from named reference assumptions,
`metric_measurement_proven` stays false, and qualification still needs the
site's independent measurements (`docs/beta-site-task-simulation-qualification.md`).
