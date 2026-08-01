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
inference) are all lazy and/or injected. 177 tests across 5 dedicated suites (test_scene_placement 94, test_placement_validation 43, test_perception_fusion 22, test_perception_views 7, test_perception_adapter 11).

## Scene backends (the swappable `SceneSpatialIndex`)

| Backend | Use when | Source of truth |
|---|---|---|
| `UsdSceneSpatialIndex(usd_path=… or stage=…)` | the scene is a USD (sim / assembled NuRec) | exact world AABBs via `UsdGeom.BBoxCache` |
| `PerceptionSceneSpatialIndex(detections, depth_provider, camera)` | diagnostic analysis of one rendered view | 2D boxes + a supplied depth array; not placement authority by itself |
| `MultiViewPerceptionSceneSpatialIndex(views)` | qualified multi-view metric placement support | only views carrying source-bound metric depth, pose, calibration, timing, and validation evidence are admitted by default |

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
- `build_perception_views_from_frames(frames, cameras, *, detect, depth, depth_evidence=None)` →
  run injected segmentation/depth callables per frame and assemble the `views` list.
  `depth_evidence(frame, depth_map)` must bind exact source/frame/depth and loaded-payload digests,
  decoded PTS plus its sync-map row, calibration,
  pose, metric scale, units, depth semantics, and validation metrics before placement is allowed.
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
views  = build_perception_views_from_frames(                 # GPU inference is injected
    frames, cams, detect=sam3_detect, depth=depth_model,
    depth_evidence=qualified_depth_evidence_for_frame,
)
index  = MultiViewPerceptionSceneSpatialIndex(views, min_views=2)
pose   = place_robot_for_task(index, "open the drawer", probe=physx_overlap_probe)
```

## The GPU boundary

Only rendering and model inference may need a GPU; those operations are injected. The legacy
`scripts/sam3_detect.py` remains a fail-closed object-index helper. The source-track lane instead
has an executable provider-neutral SAM 3.1 Object Multiplex adapter at
`blueprint_pipeline.sam31_source_track_provider_stage`. It uses Meta's official stateful
`build_sam3_multiplex_video_predictor()` API, not an assumed Transformers integration. The stage
requires ordered hash-bound JPEG derivatives of encoder-retained frames, exact decoded PTS and
camera records, a pinned official code revision and checkpoint digest, explicit checkpoint,
license-terms/use, privacy, trade-controls, and execution authorization, offline inference, and
persistent object IDs. It validates
every untrusted mask, score, and ID tensor before emitting the existing compact provider result
and a ready import request. Until the gated checkpoint is present in a pinned GPU runtime and a
real run is accepted, this is implemented adapter code rather than live provider evidence. Its
masks remain inferred 2D candidates. A model name or numeric depth array never establishes metric
scale. The USD backend needs only `pxr` (lazy; its one test skips when `pxr` is absent).

The existing Splat Analyzer adapter is a candidate generator only. Its upstream implementation
uses small box-center depth samples, an inferred depth extent, identity orientation, and distance
clustering. Blueprint therefore marks those boxes visualization-only and does not use them for
metric placement, collision, contact, or physics qualification.

Blueprint also includes a bounded deterministic NumPy contribution renderer for standard 3DGS
PLY inputs. It consumes the compact source-track artifact plus exact retained-frame cameras,
projects anisotropic Gaussians with the declared OpenCV pinhole convention, and emits the real
front-to-back `transmittance * alpha` rows accepted by the semantic lifting contract. This closes
the executable small-scene/reference path and provides a conformance oracle for a future GPU
adapter. It intentionally rejects unrectified cameras, stale splat/camera/mask bindings,
nonstandard compressed PLY inputs, excessive projected work, and oversized JSON views. It is not
the large-scene production transport, a semantic detector, collision geometry, or physics proof.

## In the Isaac runner

`scripts/run_isaac_g1_kitchen_parity_eval.py` uses the USD backend: a scenario with only a task
description (no object id, no coords) resolves its target from the open kitchen stage, and the
manipulation camera + arm reach + fill light follow that resolved target (`effective_look_at`).
