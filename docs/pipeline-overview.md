# Blueprint Capture Pipeline

This document describes the end-to-end pipeline for converting walkthrough video captures into SimReady 3D scenes for robotics simulation.

## Key Constraint: Meta Wearables DAT

**Meta Wearables DAT preview (as of Dec 2025) is primarily camera (and audio via Bluetooth), not a full VIO/depth stack.** The pipeline is designed to work well in **monocular RGB** conditions, treating metric scale as something that must be anchored during capture or calibrated post-hoc.

## Two-Deliverable Strategy

1. **Perception Twin** — Photoreal dense representation (Gaussian splats, mesh) for rendering and perception training.
2. **Sim Twin** — Object-centric assets with colliders, physics materials, and semantics for interaction in Isaac Sim.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      BlueprintCapturePipeline (this repo)               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [Capture (iOS/Meta DAT)]                                               │
│         │                                                                │
│         v                                                                │
│  [Upload to GCS] ─────> [Cloud Function Trigger]                        │
│         │                                                                │
│         v                                                                │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │              video2zeroscene Pipeline                         │       │
│  │                                                               │       │
│  │  Stage 0: Ingest ──> CaptureManifest + keyframes             │       │
│  │     │                                                         │       │
│  │     v                                                         │       │
│  │  Stage 1-early: SAM3 tracking ──> dynamic masks              │       │
│  │     │                                                         │       │
│  │     v                                                         │       │
│  │  Stage 2: SLAM (sensor-conditional)                          │       │
│  │     ├── RGB-only: WildGS-SLAM                                │       │
│  │     ├── RGB-D: SplaTAM                                       │       │
│  │     ├── Visual-Inertial: VIGS-SLAM                          │       │
│  │     └── iOS ARKit: Direct pose import                        │       │
│  │     │                                                         │       │
│  │     v                                                         │       │
│  │  Stage 3: Mesh ──> SuGaR extraction + decimation             │       │
│  │     │                                                         │       │
│  │     v                                                         │       │
│  │  Stage 4: Tracks ──> SAM3 concept segmentation               │       │
│  │     │                                                         │       │
│  │     v                                                         │       │
│  │  Stage 5: Lift ──> 2D tracks to 3D proposals                 │       │
│  │     │                                                         │       │
│  │     v                                                         │       │
│  │  Stage 6: Assetize ──> tiered object assets                  │       │
│  │     ├── Tier 1: Multi-view reconstruction                    │       │
│  │     ├── Tier 2: Proxy geometry (box/hull)                    │       │
│  │     └── Tier 3: Asset replacement (future)                   │       │
│  │     │                                                         │       │
│  │     v                                                         │       │
│  │  Stage 7: Export ──> ZeroScene bundle                        │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ ZeroScene handoff
                                    v
┌─────────────────────────────────────────────────────────────────────────┐
│                        BlueprintPipeline (downstream)                    │
├─────────────────────────────────────────────────────────────────────────┤
│  zeroscene_adapter ──> simready ──> usd_assembly ──> isaac_lab         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Sensor-Conditional SLAM Selection

| Sensor Type      | SLAM Backend   | When to Use                       |
|------------------|----------------|-----------------------------------|
| RGB-only         | WildGS-SLAM    | Meta glasses, generic cameras     |
| RGB-D            | SplaTAM        | iPhone LiDAR, RealSense           |
| Visual-Inertial  | VIGS-SLAM      | RGB + synchronized IMU            |
| iOS ARKit        | Direct import  | iOS with ARKit tracking enabled   |

## GCS Storage Layout

```
gs://bucket/captures/{capture_id}/
├── raw/
│   ├── video.mp4
│   ├── metadata.json
│   └── arkit/                    # iOS only
│       ├── poses.jsonl
│       └── intrinsics.json
├── stage0_ingest/
│   ├── capture_manifest.json
│   └── frame_index.json
├── stage1_frames/
│   └── frames/*.png
├── stage2_slam/
│   ├── poses/
│   └── gaussians/
├── stage3_mesh/
│   ├── environment_mesh.glb
│   └── environment_collision.glb
├── stage4_tracks/
│   ├── masks/
│   ├── tracks.json
│   └── annotations.json
├── stage5_proposals/
│   └── proposals.json
├── stage6_assets/
│   └── {object_id}/
├── zeroscene/                    # <- Handoff to BlueprintPipeline
│   ├── scene_info.json
│   ├── objects/
│   ├── background/
│   └── camera/
└── blueprint/                    # <- Output from BlueprintPipeline
    └── scene.usdc
```

## Cloud Deployment (GCP)

* **Runtime:** Cloud Run Jobs with GPU (NVIDIA L4, 24GB VRAM)
* **Storage:** GCS buckets with lifecycle rules
* **Messaging:** Pub/Sub for stage transitions; Cloud Tasks for job dispatch
* **Triggers:** Cloud Functions for Firebase Storage upload detection

## Capture requirements (iOS + Meta DAT)

* **Two-pass capture:**
  * **Structure pass:** Slow room walkthrough with wide parallax, full loop, avoid fast rotations.
  * **Object micro-scans:** Short sequences focused on manipulable objects (drawers, appliances, dishes, handles).
* **Scale anchor (mandatory):** Show an AprilTag/ArUco board, tape-measured segment, or A4 sheet in multiple locations for 2–3 seconds.
* **Stabilization:** Encourage 1080p or higher, minimal motion blur, and good lighting.
* **MockDeviceKit:** Enable for local QA and automated tests of the capture UX.

## Data model and ingress

* **Session manifest (JSON):**
  * `session_id`, `capture_start`, `device` metadata (DAT build, lenses, resolution, FPS).
  * `scale_anchor` observations (tag size, measured distance, or known object dimensions).
  * `clips[]` with GCS URIs, timestamps, and optional user notes.
* **Upload path:** `gs://<bucket>/sessions/<session_id>/raw/{clip.mp4,manifest.json}`.
* **Integrity:** Compute checksums client-side; server validates duration and metadata before enqueueing work.

## Processing stages

### 1) Frame extraction & masking

* **Decode video → frames** at ~3–5 fps for reconstruction; keep full-rate frames for object micro-scans when available.
* **SAM 3 (video) masks:**
  * Generate per-frame instance masks.
  * Tag dynamic classes (people/hands) for exclusion during reconstruction.
  * Persist masks as PNG + COCO-style JSON per clip.

### 2) Camera trajectory & dense reconstruction (Perception Twin backbone)

* **WildGS-SLAM**
  * Input: frames + dynamic masks (to ignore moving objects).
  * Output: camera poses (COLMAP-style), 3D Gaussian map, filtered point cloud.
  * Apply **scale calibration** using anchor observations (AprilTag detections or measured pixel length).
* **Quality checks:**
  * Reprojection error thresholds.
  * Coverage heatmap; flag low-parallax regions.

### 3) Mesh extraction

* **SuGaR** on WildGS Gaussians → watertight mesh for static environment.
* **Texture baking:**
  * Bake albedo/normal maps for USD rendering.
  * Keep both high-res render mesh and decimated collision mesh.

### 4) Object lifting & assetization (Sim Twin)

* **2D tracking:** Use SAM 3 prompts (text or point) to track target objects across frames.
* **3D lifting:** Project masks into 3D using camera poses + mesh/splats to produce object clusters and oriented bounding boxes (OBBs).
* **Tiered assetization:**
  * **Tier 1 (reconstruct):** If coverage is sufficient, run object-centric 3DGS + SuGaR to create textured meshes faithful to the captured object.
  * **Tier 2 (generate):** If coverage is poor, generate via **Hunyuan3D** (shape + texture) constrained to the OBB dimensions; align to scene scale.
* **Dynamics filtering:** Skip or replace highly reflective/transparent items unless adequate views exist.

### 5) USD authoring for simulation

* **Scene scale:** Set `metersPerUnit` and apply calibrated scale globally.
* **Environment (static):**
  * Render mesh (pretty) and simplified collision mesh (tri or voxelized SDF) for walls/floors.
* **Objects (dynamic):**
  * Collision: convex decomposition or primitive approximations (box/capsule). Avoid triangle-mesh colliders for rigid bodies.
  * Physics materials: friction/restitution bound via USD Physics schema.
  * Optional articulation: add joints only when confidently detected; otherwise keep static.
* **Packaging:**
  * `scene.usda/usdc` with references to per-object USDs.
  * Per-object USDs include render mesh, collision mesh, material bindings, mass/inertia estimates, and semantic labels.

## Cloud deployment (GCP)

* **Runtime:** Cloud Run Jobs with GPU (L4) for SAM 3, WildGS-SLAM, SuGaR, and Hunyuan3D.
* **Storage:** GCS buckets with lifecycle rules; signed URLs for client uploads.
* **Messaging:** Pub/Sub topics for stage transitions; dead-letter queues for failed jobs.
* **Observability:** Cloud Logging + Cloud Trace; per-stage metrics (reproj error, mask count, pose completeness, mesh watertightness).
* **Artifacts:**
  * `reconstruction/` — gaussians, poses, mesh, textures.
  * `objects/<id>/` — masks, crops, per-object 3DGS checkpoints, USDs.
  * `reports/` — QA metrics, thumbnails, coverage heatmaps.

## QA harness

* **Automated checks:**
  * Pose quality (median reprojection error, track length).
  * Scale sanity (anchor consistency across clips).
  * Mesh integrity (non-manifold count, watertightness).
  * Physics sanity (convex hull volume vs. OBB volume).
* **Isaac Sim smoke tests:**
  * Drop-test objects onto floor collider.
  * Stack-test with representative objects.
  * Robot reach test to ensure bounding boxes and colliders are reachable without tunneling.

## Roadmap (execution order)

1. Build capture ingest (manifest validation + GCS upload) and Cloud Run orchestration.
2. Add SAM 3 masking stage and WildGS-SLAM backbone with scale calibration.
3. Integrate SuGaR mesh extraction and texture baking.
4. Implement object lifting → per-object reconstruction; add Hunyuan3D fallback.
5. USD authoring with physics-ready assets; ship QA harness (reports + Isaac Sim tests).
