# BlueprintCapture Pipeline

**Phase 3: Video → High-Quality 3D Gaussian → DWM-Ready Output**

A GPU-accelerated pipeline for converting video walkthroughs into 3D Gaussian representations, ready for DWM (Dexterous World Models) processing.

## What is This?

BlueprintCapture is the **capture pipeline** in the Blueprint system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Blueprint System                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: BlueprintPipeline ─── Image → SimReady 3D reconstruction          │
│  Phase 2: DWM Data Layer ────── Scene → egocentric rollouts + training data │
│  Phase 3: BlueprintCapture ──── Video → 3D Gaussian capture (THIS REPO)     │
│  Phase 4: AR Platform ───────── Digital twins → AR Cloud                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Flow

```
┌──────────────────┐     ┌──────────────────────────┐     ┌────────────────────┐
│  Video Capture   │────▶│  BlueprintCapturePipeline │────▶│  BlueprintPipeline  │
│ (Meta glasses,   │     │  (This Repo)              │     │  (DWM Processing)   │
│  iPhone, etc.)   │     │                           │     │                     │
└──────────────────┘     │  • Ingest video           │     │  • Generate DWM     │
                         │  • SLAM reconstruction    │     │    training data    │
                         │  • Export Gaussians +     │     │  • Egocentric       │
                         │    camera data            │     │    rollouts         │
                         └──────────────────────────┘     └────────────────────┘
```

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BlueprintCapture Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐          │
│  │   Stage 0    │───▶│     Stage 1       │───▶│     Stage 2      │          │
│  │   Ingest     │    │      SLAM         │    │     Export       │          │
│  │              │    │                   │    │                  │          │
│  │ • Extract    │    │ • Pose estimation │    │ • gaussians.ply  │          │
│  │   keyframes  │    │ • 3D Gaussian     │    │ • trajectory.json│          │
│  │ • Metadata   │    │   reconstruction  │    │ • intrinsics.json│          │
│  └──────────────┘    └───────────────────┘    └──────────────────┘          │
│                                                        │                     │
│                                                        ▼                     │
│                                               ┌──────────────────┐          │
│                                               │ DWM-Ready Output │          │
│                                               │ → BlueprintPipeline│         │
│                                               └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 0: Ingest
- Decode video from Meta glasses, iPhone, or generic cameras
- Extract keyframes at configurable FPS
- Quality filtering (blur detection, exposure)
- Generate CaptureManifest with device/sensor metadata

### Stage 1: SLAM Reconstruction
- **WildGS-SLAM** for RGB-only captures (Meta glasses)
- **SplaTAM** for RGB-D captures (iPhone LiDAR)
- **ARKit Direct** for iOS captures (uses ARKit poses directly)
- **COLMAP Fallback** when other methods unavailable
- Output: Camera poses + 3D Gaussian splats

### Stage 2: Export for DWM
Package the output for BlueprintPipeline/DWM processing:

```
output/
├── gaussians.ply           # 3D Gaussian point cloud
├── camera/
│   ├── intrinsics.json     # Camera parameters
│   └── trajectory.json     # Per-frame camera poses
└── capture_info.json       # Metadata for handoff
```

## Installation

```bash
# Basic install
pip install -e .

# With GPU support (recommended)
pip install -e ".[gpu]"

# Development
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from pathlib import Path
from blueprint_pipeline import CapturePipeline, run_capture_pipeline

# Simple usage
result = run_capture_pipeline(
    video_paths=[Path("walkthrough.mp4")],
    output_dir=Path("output"),
)

if result.dwm_ready:
    print(f"Success! Output at: {result.output_path}")
    print(f"  - Gaussians: {result.export_result.gaussians_path}")
    print(f"  - Trajectory: {result.export_result.trajectory_path}")

# Or with more control
from blueprint_pipeline import CapturePipeline, CaptureConfig

config = CaptureConfig(
    target_fps=2.0,  # Keyframes per second
    slam_backend=None,  # Auto-select based on sensors
)

pipeline = CapturePipeline(config)
result = pipeline.run(
    capture_id="kitchen_scan_001",
    video_paths=[Path("walkthrough.mp4")],
    output_dir=Path("output"),
    arkit_data_path=Path("arkit_data"),  # Optional: iOS ARKit poses
)
```

### CLI

```bash
# Run pipeline
blueprint-capture --manifest session.yaml

# With GCS
blueprint-capture --manifest session.yaml --gcs-bucket my-bucket
```

### iOS Capture with ARKit

When using iPhone with ARKit tracking, the pipeline can skip SLAM entirely and use metric-scale poses directly:

```python
result = run_capture_pipeline(
    video_paths=[Path("capture.mov")],
    output_dir=Path("output"),
    arkit_data_path=Path("arkit/"),  # Contains poses.jsonl
)
# result.slam_result.scale_factor == 1.0 (metric scale!)
```

## Output Format

The pipeline produces a DWM-ready output:

### `gaussians.ply`
Standard 3D Gaussian Splatting point cloud format:
- Position (x, y, z)
- Covariance (6 parameters)
- Spherical harmonics (colors)
- Opacity

### `camera/trajectory.json`
```json
{
  "poses": [
    {
      "frame_id": "frame_0001",
      "rotation": [0.9, 0.1, 0.0, 0.0],  // Quaternion (w, x, y, z)
      "translation": [0.0, 0.0, 0.0],
      "timestamp": 0.0
    }
  ],
  "coordinate_system": "colmap",
  "scale_factor": 1.0
}
```

### `camera/intrinsics.json`
```json
{
  "fx": 1500.0,
  "fy": 1500.0,
  "cx": 960.0,
  "cy": 540.0,
  "width": 1920,
  "height": 1080,
  "camera_model": "PINHOLE"
}
```

### `capture_info.json`
Metadata for BlueprintPipeline handoff, including:
- Capture ID and timestamp
- Device and sensor info
- Reconstruction metrics
- DWM readiness status

## Sensor Support

| Sensor Type | Device Examples | SLAM Backend | Notes |
|-------------|-----------------|--------------|-------|
| RGB-only | Meta Ray-Ban Stories | WildGS-SLAM | Handles dynamic objects |
| RGB-D | iPhone LiDAR | SplaTAM | Metric-scale depth |
| iOS ARKit | iPhone, iPad | Direct pose import | Skips SLAM entirely |
| Visual-Inertial | RGB + IMU | VIGS-SLAM | Better under motion blur |

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.1+ (for SLAM/reconstruction)
- 16GB+ GPU memory recommended
- 32GB+ system RAM

## Project Structure

```
BlueprintCapturePipeline/
├── src/blueprint_pipeline/
│   ├── __init__.py           # Package exports
│   ├── video2zeroscene/      # Main capture pipeline
│   │   ├── pipeline.py       # CapturePipeline
│   │   ├── ingest.py         # Video ingestion
│   │   ├── slam.py           # SLAM backends
│   │   ├── export.py         # DWM export
│   │   └── interfaces.py     # Data models
│   ├── jobs/                 # Cloud Run job implementations
│   ├── utils/                # Utilities (GCS, GPU, logging)
│   ├── orchestrator.py       # Pipeline orchestration
│   └── runner.py             # CLI entry point
├── docs/
│   ├── pipeline-overview.md
│   └── job-stubs.md
├── Dockerfile
└── pyproject.toml
```

## Docker

### Build
```bash
docker build -t blueprint-capture:latest .
```

### Run with GPU
```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/output:/workspace/output \
  blueprint-capture:latest \
  --manifest /workspace/data/session.yaml
```

## Cloud Run Jobs

Deploy as a Cloud Run job for scalable processing:

```bash
# Build and push
docker build -t gcr.io/$PROJECT_ID/blueprint-capture:latest .
docker push gcr.io/$PROJECT_ID/blueprint-capture:latest

# Create job
gcloud run jobs create blueprint-capture \
  --image gcr.io/$PROJECT_ID/blueprint-capture:latest \
  --region us-central1 \
  --cpu 4 --memory 16Gi \
  --gpu 1 --gpu-type nvidia-l4
```

## Related Projects

- [BlueprintPipeline](https://github.com/ognjhunt/BlueprintPipeline) - DWM processing and scene generation
- [BlueprintCapture iOS](https://github.com/ognjhunt/BlueprintCapture) - iOS capture app with ARKit
- [DWM (Dexterous World Models)](https://snuvclab.github.io/dwm/) - Visual world model for robotics

## License

MIT License - see [LICENSE](LICENSE) for details.
