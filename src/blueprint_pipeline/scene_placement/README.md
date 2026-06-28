# scene_placement

Given a **task string** and **any scene**, decide **where a robot should stand** — with no
hardcoded coordinates. Replaces guessed faucet/standing coords with scene understanding +
task→object resolution + probe-based placement.

```
task ─────────────────────────────────────────────┐
                                                   ▼
scene ─▶ SceneSpatialIndex ─▶ object catalog ─▶ resolve_target ─▶ compute_stand_pose ─▶ StandPose
         (USD or perception)   (3D bboxes)       (task→object)     (probe open floor)    (stand + face)
```

Every stage is swappable and dependency-light: importing the package pulls in **no** isaacsim,
torch, google-genai, or GPU. The heavy bits (USD `pxr`, the Gemini call, the PhysX probe, SAM3/DA3
inference) are all lazy and/or injected. ~160 hermetic tests across 6 suites.

## Scene backends (the swappable `SceneSpatialIndex`)

| Backend | Use when | Source of truth |
|---|---|---|
| `UsdSceneSpatialIndex(usd_path=… or stage=…)` | the scene is a USD (sim / assembled NuRec) | exact world AABBs via `UsdGeom.BBoxCache` |
| `PerceptionSceneSpatialIndex(detections, depth_provider, camera)` | one rendered view of a raw scene/splat | SAM3 2D boxes + DA3 depth, unprojected to 3D |
| `MultiViewPerceptionSceneSpatialIndex(views)` | a raw scene/splat, multiple views (preferred) | the above, fused across views (1 box/object) |

`build_scene_index("usd" | "perception", **kw)` is the factory.

## Entry points

- `place_robot_for_task(index, task, *, probe, generate=None, **place_kw) -> StandPose` — the
  whole flow: enumerate objects → resolve the task's target → solve the stand pose. `generate` is
  the injectable Gemini text call; omit it to resolve by label (offline).
- `compute_stand_pose(target, *, probe, pelvis_height=0.79, standing_distance=0.55, …)` — placement
  only. Probes the four sides (opt-in diagonals) for the nearest **clear** floor and faces the target.
- `resolve_target(task, objects, *, generate)` / `resolve_target_by_label(task, objects)` — task→object.

`probe(pose, yaw) -> int` is a footprint-overlap hit count (`0` == clear); pass a PhysX overlap
query on GPU, or a stub in tests.

### Perception helpers (raw-scene path)

- `view_ring_for_bounds(bbox_min, bbox_max, *, n_azimuths=8, …)` → camera ring around the scene.
- `build_perception_views_from_frames(frames, cameras, *, detect, depth)` → run injected SAM3/DA3
  per frame and assemble the `views` list. `detect(frame)->records`, `depth(frame)->HxW meters`.
- `build_perception_views(cameras, sam3_records_per_view, depth_maps)` → if you already have outputs.
- `fuse_scene_objects(objects, *, merge_iou=0.25, merge_gap=None, min_views=1, max_spread=None)` →
  cluster same-object detections (complete-linkage + spread cap; `min_views≥2` drops false positives).

## Usage

USD scene (exact, the common case):

```python
from blueprint_pipeline.scene_placement import UsdSceneSpatialIndex, place_robot_for_task

index = UsdSceneSpatialIndex(usd_path="kitchen.usd")
pose = place_robot_for_task(index, "turn on the faucet", probe=physx_overlap_probe)
# -> pose.position (pelvis xyz), pose.yaw (faces the faucet), pose.clear
```

Raw scene / splat (multi-view perception):

```python
from blueprint_pipeline.scene_placement import (
    view_ring_for_bounds, build_perception_views_from_frames,
    MultiViewPerceptionSceneSpatialIndex, place_robot_for_task,
)

cams   = view_ring_for_bounds(scene_min, scene_max, n_azimuths=8, width=1280, height=960)
frames = [render(cam) for cam in cams]                       # GPU: your splat/USD renderer
views  = build_perception_views_from_frames(frames, cams,    # GPU: SAM3 + DA3 inference
                                            detect=sam3_detect, depth=da3_depth)
index  = MultiViewPerceptionSceneSpatialIndex(views, min_views=2)
pose   = place_robot_for_task(index, "open the drawer", probe=physx_overlap_probe)
```

## The GPU boundary

Only two things need a GPU to **execute** (everything else is pure and runs anywhere): rendering
the view-ring frames, and the SAM3/DA3 inference behind `detect`/`depth`. Those are injected, so the
package and all its tests run with no GPU. The USD backend needs only `pxr` (lazy; its one test skips
when `pxr` is absent).

## In the Isaac runner

`scripts/run_isaac_g1_kitchen_parity_eval.py` uses the USD backend: a scenario with only a task
description (no object id, no coords) resolves its target from the open kitchen stage, and the
manipulation camera + arm reach + fill light follow that resolved target (`effective_look_at`).
