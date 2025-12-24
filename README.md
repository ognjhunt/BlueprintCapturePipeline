# Blueprint Capture Pipeline

A GPU-accelerated pipeline for converting Meta smart glasses video captures into SimReady 3D scenes for robotics simulation.

## Overview

This pipeline transforms video from Meta smart glasses (captured via the [BlueprintCapture iOS app](https://github.com/ognjhunt/BlueprintCapture)) into two outputs:

1. **Perception Twin:** Dense, photorealistic reconstruction (3D Gaussian Splatting + textured mesh) for rendering and visual QA.
2. **Sim Twin:** Object-centric USD assets with clean colliders and physics materials for robotics simulation (e.g., NVIDIA Isaac Sim).

### 🎯 NEW: DWM (Dexterous World Models) Compatible

This pipeline now outputs **DWM-compatible** scene bundles that include:
- **Raw 3D Gaussians** (PLY format) for high-quality rendering
- **Camera trajectory** with metric-scale poses
- **Camera intrinsics** for view synthesis

DWM uses these to generate photorealistic egocentric interaction videos, enabling visual planning, action evaluation, and training data generation for robotics.

👉 See [DWM Compatibility Guide](docs/DWM_COMPATIBILITY.md) for details.

### 🎬 NEW: 3DGS Static Scene Renderer

This pipeline now includes a **full 3D Gaussian Splatting renderer** for generating static-scene videos from ZeroScene bundles:

```bash
# Render static scene for DWM
python scripts/render_static_scene.py output/zeroscene -o static_scene.mp4
```

Or use the Python API:
```python
from blueprint_pipeline.video2zeroscene.rendering import GaussianRenderer

renderer = GaussianRenderer.from_zeroscene("output/zeroscene")
frames = renderer.render_trajectory()
renderer.save_video(frames, "static_scene.mp4")
```

Supports multiple backends (diff-gaussian-rasterization, gsplat, CPU fallback).
👉 See [Rendering Guide](docs/RENDERING.md) for full documentation.

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Blueprint Capture Pipeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌───────────────┐    ┌────────────────┐                │
│  │    Frame     │───▶│ Reconstruction │───▶│     Mesh       │                │
│  │  Extraction  │    │  (WildGS-SLAM) │    │   Extraction   │                │
│  └──────────────┘    └───────────────┘    │    (SuGaR)     │                │
│         │                                  └────────────────┘                │
│         │                                          │                         │
│         │  SAM3 Masks                              │                         │
│         ▼                                          ▼                         │
│  ┌──────────────┐                          ┌────────────────┐                │
│  │    Object    │──────────────────────────│      USD       │                │
│  │ Assetization │                          │   Authoring    │                │
│  │ (Hunyuan3D)  │                          │  (Isaac Sim)   │                │
│  └──────────────┘                          └────────────────┘                │
│                                                    │                         │
│                                                    ▼                         │
│                                            ┌────────────────┐                │
│                                            │   scene.usdc   │                │
│                                            │  (SimReady)    │                │
│                                            └────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. Frame Extraction (`FrameExtractionJob`)
- Decode video clips from Meta smart glasses
- Extract frames at configurable FPS
- Run SAM 3 (Segment Anything Model) for object detection and tracking
- Generate dynamic masks (people, hands) for reconstruction

### 2. Reconstruction (`ReconstructionJob`)
- Run WildGS-SLAM for camera pose estimation and 3D Gaussian Splatting
- Apply scale calibration using fiducials/anchors
- Output: Gaussian splats, camera poses, point cloud

### 3. Mesh Extraction (`MeshExtractionJob`)
- Extract textured mesh from Gaussian splats using SuGaR
- Generate simplified collision mesh for physics simulation
- Bake multi-view textures onto mesh
- Export in USD format

### 4. Object Assetization (`ObjectAssetizationJob`)
- Lift SAM 3 2D tracks into 3D using camera poses
- **Tier 1:** Reconstruct objects from multi-view when coverage is good
- **Tier 2:** Generate objects using Hunyuan3D when coverage is poor
- Output: Individual object USD files with placement info

### 5. USD Authoring (`USDAuthoringJob`)
- Compose final scene from environment mesh and objects
- Add physics properties (rigid bodies, colliders, materials)
- Set up Isaac Sim-compatible hierarchy
- Validate SimReady compliance

## Installation

### Basic Installation
```bash
pip install -e .
```

### With GPU Support (Recommended)
```bash
pip install -e ".[gpu]"
```

### Individual Components
```bash
# Core dependencies
pip install -e ".[core]"

# Cloud Run / GCS support
pip install -e ".[cloud]"

# SAM 3 segmentation
pip install -e ".[sam]"

# USD authoring
pip install -e ".[usd]"

# 3DGS rendering (for DWM static-scene videos)
pip install -e ".[rendering]"
# For CUDA acceleration (recommended):
pip install gsplat  # Easy install
# OR for best quality/speed, build from source:
# https://github.com/graphdeco-inria/diff-gaussian-rasterization

# Hunyuan3D generation
pip install -e ".[generation]"
```

## Quick Start

### 1. Create a Session Manifest
```yaml
# session.yaml
session_id: kitchen_001
capture_start: "2024-01-15T10:30:00Z"
device:
  type: meta_ray_ban_stories
  firmware: "1.0.0"
scale_anchors:
  - type: fiducial
    anchor_id: aruco_42
    known_size_meters: 0.15
clips:
  - clip_id: structure_pass
    gcs_uri: gs://bucket/sessions/kitchen_001/video.mp4
    purpose: structure
```

### 2. Run the Pipeline

#### CLI
```bash
# Run full pipeline
python -m blueprint_pipeline.runner --manifest session.yaml

# Run single stage
python -m blueprint_pipeline.runner --manifest session.yaml --stage reconstruction

# With GCS bucket
python -m blueprint_pipeline.runner --manifest session.yaml --gcs-bucket my-bucket
```

#### Python API
```python
from blueprint_pipeline import (
    PipelineOrchestrator,
    SessionManifest,
)

# Load session
session = SessionManifest(
    session_id="kitchen_001",
    capture_start="2024-01-15T10:30:00Z",
    device={"type": "meta_ray_ban_stories"},
    clips=[...],
)

# Run pipeline
orchestrator = PipelineOrchestrator(gcs_bucket="my-bucket")
result = orchestrator.run_full_pipeline(session)

print(f"Pipeline {'succeeded' if result.success else 'failed'}")
print(f"Duration: {result.total_duration_seconds:.1f}s")
```

## Docker

### Build GPU Image
```bash
docker build -t blueprint-pipeline:latest .
```

### Run Locally with GPU
```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  blueprint-pipeline:latest \
  --manifest /workspace/data/session.yaml
```

### Docker Compose (Development)
```bash
# GPU-enabled development shell
docker-compose run --rm dev

# Run pipeline
docker-compose run --rm pipeline-gpu --manifest /workspace/configs/session.yaml
```

## Cloud Run Jobs Deployment

### Build and Push
```bash
# Set project
export PROJECT_ID=your-project
export REGION=us-central1

# Build and push
docker build -t gcr.io/$PROJECT_ID/blueprint-pipeline:latest .
docker push gcr.io/$PROJECT_ID/blueprint-pipeline:latest
```

### Create Cloud Run Job
```bash
gcloud run jobs create blueprint-pipeline \
  --image gcr.io/$PROJECT_ID/blueprint-pipeline:latest \
  --region $REGION \
  --cpu 4 \
  --memory 16Gi \
  --task-timeout 60m \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT_ID" \
  --gpu 1 \
  --gpu-type nvidia-l4
```

### Execute Job
```bash
gcloud run jobs execute blueprint-pipeline \
  --region $REGION \
  --set-env-vars "JOB_PAYLOAD={\"job_name\":\"frame-extraction\",...}"
```

## Project Structure

```
BlueprintCapturePipeline/
├── src/blueprint_pipeline/
│   ├── __init__.py           # Package exports
│   ├── models.py             # Data models (SessionManifest, etc.)
│   ├── pipeline.py           # Pipeline builder utilities
│   ├── orchestrator.py       # Pipeline orchestration
│   ├── runner.py             # CLI entry point
│   ├── jobs/
│   │   ├── base.py           # BaseJob, GPUJob classes
│   │   ├── frame_extraction.py
│   │   ├── reconstruction.py
│   │   ├── mesh.py
│   │   ├── object_assetization.py
│   │   └── usd_authoring.py
│   └── utils/
│       ├── gcs.py            # GCS client utilities
│       ├── gpu.py            # GPU detection/management
│       ├── io.py             # File I/O utilities
│       └── logging.py        # Logging/progress tracking
├── configs/
│   └── example_session.yaml
├── docs/
│   ├── pipeline-overview.md
│   └── job-stubs.md
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Configuration

### Job Parameters

Each job accepts configuration parameters:

```python
# Frame Extraction
parameters = {
    "target_fps": 2.0,           # Frames per second to extract
    "sam3_enabled": True,        # Run SAM 3 segmentation
    "mask_dynamics": True,       # Mask dynamic objects
}

# Reconstruction
parameters = {
    "slam_method": "wildgs",     # SLAM method
    "max_iterations": 30000,     # Training iterations
    "apply_scale_correction": True,
}

# Mesh Extraction
parameters = {
    "decimation_target": 500000, # Target face count
    "bake_textures": True,
    "texture_resolution": 4096,
}

# Object Assetization
parameters = {
    "coverage_threshold": 0.6,   # Min coverage for reconstruction
    "hunyuan_enabled": True,     # Enable AI generation fallback
    "max_objects": 50,
}

# USD Authoring
parameters = {
    "convex_decomposition": True,
    "enable_rigid_body": True,
    "meters_per_unit": 1.0,
}
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.1+ (for full pipeline)
- 16GB+ GPU memory recommended
- 32GB+ system RAM

## Dependencies

- **PyTorch 2.x** - Deep learning framework
- **SAM 2** - Segment Anything Model for object segmentation
- **Open3D** - 3D geometry processing
- **OpenUSD** - Universal Scene Description
- **Hunyuan3D** - Image-to-3D generation (optional)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [BlueprintCapture](https://github.com/ognjhunt/BlueprintCapture) - iOS capture app
- [BlueprintPipeline](https://github.com/ognjhunt/BlueprintPipeline) - Original pipeline design
- [Meta Wearables DAT](https://github.com/facebook/meta-wearables-dat-ios) - Meta glasses SDK
