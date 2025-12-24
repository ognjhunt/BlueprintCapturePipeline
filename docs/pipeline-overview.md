# BlueprintCapture Pipeline Overview

## Phase 3: Video → Gaussian → DWM

This pipeline converts video walkthroughs into 3D Gaussian representations ready for DWM (Dexterous World Models) processing in BlueprintPipeline.

## Key Constraint: Meta Wearables DAT

**Meta Wearables DAT preview (as of Dec 2025) is primarily camera (and audio via Bluetooth), not a full VIO/depth stack.** The pipeline is designed to work well in **monocular RGB** conditions, treating metric scale as something that must be anchored during capture or calibrated post-hoc via ARKit (iOS) or scale anchors.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BlueprintCapturePipeline (this repo)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [Capture (iOS/Meta DAT)]                                                   │
│         │                                                                    │
│         v                                                                    │
│  [Upload to GCS] ─────> [Cloud Function Trigger]                            │
│         │                                                                    │
│         v                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                   CapturePipeline                             │           │
│  │                                                               │           │
│  │  Stage 0: Ingest ──> CaptureManifest + keyframes             │           │
│  │     │                                                         │           │
│  │     v                                                         │           │
│  │  Stage 1: SLAM (sensor-conditional)                          │           │
│  │     ├── RGB-only: WildGS-SLAM                                │           │
│  │     ├── RGB-D: SplaTAM                                       │           │
│  │     ├── Visual-Inertial: VIGS-SLAM                          │           │
│  │     └── iOS ARKit: Direct pose import                        │           │
│  │     │                                                         │           │
│  │     v                                                         │           │
│  │  Stage 2: Export ──> DWM-ready output                        │           │
│  │     • gaussians.ply                                          │           │
│  │     • camera/trajectory.json                                 │           │
│  │     • camera/intrinsics.json                                 │           │
│  │     • capture_info.json                                      │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ DWM-ready handoff
                                    v
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BlueprintPipeline (downstream)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  DWM processing ──> egocentric rollouts ──> training data                   │
└─────────────────────────────────────────────────────────────────────────────┘
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
├── ingest/
│   ├── capture_manifest.json
│   ├── frame_index.json
│   └── frames/*.png
├── slam/
│   ├── poses/
│   │   └── poses.json
│   └── gaussians/
│       └── point_cloud.ply
└── output/                       # <- DWM-ready output
    ├── gaussians.ply
    ├── camera/
    │   ├── intrinsics.json
    │   └── trajectory.json
    └── capture_info.json
```

## Cloud Deployment (GCP)

* **Runtime:** Cloud Run Jobs with GPU (NVIDIA L4, 24GB VRAM)
* **Storage:** GCS buckets with lifecycle rules
* **Messaging:** Pub/Sub for stage transitions; Cloud Tasks for job dispatch
* **Triggers:** Cloud Functions for Firebase Storage upload detection

## Capture Requirements

### Video Quality
* **Resolution:** 1080p or higher
* **Frame rate:** 30 fps minimum
* **Stabilization:** Minimal motion blur, good lighting
* **Movement:** Slow, smooth walkthrough with wide parallax

### Scale Calibration
For metric-scale output, use one of:
* **ARKit poses** (iOS) - Automatic metric scale
* **Scale anchors** - Show AprilTag/ArUco board with known size
* **Known objects** - Reference objects with known dimensions

## Processing Stages

### Stage 0: Ingest

**Purpose:** Prepare video for SLAM processing.

**Operations:**
1. Decode video clips
2. Extract frames at target FPS (default: 2 fps)
3. Quality filtering:
   - Blur detection (Laplacian variance)
   - Exposure quality
   - Parallax from previous frame
4. Generate CaptureManifest

**Configuration:**
```python
PipelineConfig(
    target_fps=2.0,           # Keyframes per second
    blur_threshold=100.0,     # Variance of Laplacian threshold
    min_parallax_threshold=0.1,
)
```

### Stage 1: SLAM Reconstruction

**Purpose:** Estimate camera poses and build 3D Gaussian representation.

**WildGS-SLAM (RGB-only):**
* Input: frames (optionally with dynamic masks)
* Output: camera poses, 3D Gaussian map
* Handles dynamic objects (people, hands)
* Apply scale calibration using anchor observations

**ARKit Direct (iOS):**
* Skip SLAM entirely
* Use ARKit poses directly (metric scale)
* Train 3DGS from known poses

**Quality metrics:**
* Registration rate (poses / keyframes)
* Reprojection error
* Scale confidence

### Stage 2: Export

**Purpose:** Package output for BlueprintPipeline/DWM handoff.

**Output files:**
* `gaussians.ply` - 3D Gaussian point cloud
* `camera/trajectory.json` - Camera poses per frame
* `camera/intrinsics.json` - Camera parameters
* `capture_info.json` - Metadata for handoff

## GPU Requirements

| Stage | GPU Memory | Notes |
|-------|------------|-------|
| Ingest | None | CPU-only frame extraction |
| SLAM | 8-16 GB | WildGS-SLAM needs ~16GB for large scenes |
| Export | None | File I/O only |

**Recommended:** NVIDIA L4 (24GB) or A100 for production workloads.

## Data Models

### CaptureManifest
```python
@dataclass
class CaptureManifest:
    capture_id: str
    capture_timestamp: str
    device_platform: str  # "ios", "meta_glasses", "android"
    sensor_type: SensorType  # RGB_ONLY, RGB_DEPTH, VISUAL_INERTIAL
    has_depth: bool
    has_imu: bool
    has_arkit_poses: bool
    intrinsics: CameraIntrinsics
    clips: List[Dict]
    scale_anchors: List[ScaleAnchorObservation]
```

### CameraPose
```python
@dataclass
class CameraPose:
    frame_id: str
    image_name: str
    rotation: Tuple[float, float, float, float]  # Quaternion (w, x, y, z)
    translation: Tuple[float, float, float]
    timestamp: float
```

### SLAMResult
```python
@dataclass
class SLAMResult:
    poses: List[CameraPose]
    gaussians_path: Path
    registration_rate: float
    scale_factor: float
    success: bool
```

## Integration with BlueprintPipeline

The output is designed for seamless handoff to BlueprintPipeline for DWM processing:

```python
# BlueprintCapturePipeline (this repo)
from blueprint_pipeline import run_capture_pipeline

result = run_capture_pipeline(
    video_paths=[Path("walkthrough.mp4")],
    output_dir=Path("capture_output"),
)

# Output structure:
# capture_output/
#   gaussians.ply           <- 3D Gaussians
#   camera/
#     intrinsics.json       <- Camera params
#     trajectory.json       <- Poses
#   capture_info.json       <- Metadata with dwm_ready flag

# BlueprintPipeline (downstream) reads this and generates DWM data
```

The `capture_info.json` contains the `dwm_ready` flag to confirm the output is complete and valid for DWM processing.
