# BlueprintCapturePipeline Setup Guide

Complete setup instructions for the video-to-3DGS pipeline.

## Quick Start

### Local Development (CPU)

```bash
# 1. Clone the repository
git clone https://github.com/ognjhunt/BlueprintCapturePipeline.git
cd BlueprintCapturePipeline

# 2. Install core dependencies
pip install opencv-python scipy torch torchvision

# 3. Install COLMAP (required for camera pose estimation)
# Ubuntu/Debian:
sudo apt install colmap
# macOS:
brew install colmap

# 4. Install the package
pip install -e .

# 5. Verify installation
python -c "from blueprint_pipeline.video2zeroscene.pipeline import CapturePipeline; print('✅ Ready!')"
```

### Local Development (GPU)

For 10-100x faster 3DGS training:

```bash
# 1. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install CUDA rasterizer
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
pip install git+https://github.com/camenduru/simple-knn.git

# 3. Verify CUDA is working
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Cloud Deployment

```bash
# Run the deployment script
./scripts/deploy_cloud.sh
```

See [Cloud Deployment](#cloud-deployment) section for details.

---

## Detailed Installation

### 1. System Dependencies

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y \
    python3.11 python3.11-dev python3-pip \
    colmap \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0
```

#### macOS

```bash
brew install python@3.11 colmap ffmpeg
```

#### Windows

Use WSL2 with Ubuntu, or install:
- Python 3.11 from python.org
- COLMAP from https://github.com/colmap/colmap/releases
- FFmpeg from https://ffmpeg.org/download.html

### 2. Python Dependencies

#### Core (Required)

```bash
pip install opencv-python scipy numpy Pillow pyyaml
```

#### Reconstruction

```bash
pip install torch torchvision plyfile tqdm
```

#### Cloud

```bash
pip install google-cloud-storage google-cloud-run google-cloud-logging
```

### 3. COLMAP

COLMAP provides Structure-from-Motion for camera pose estimation.

**Verify installation:**
```bash
colmap --help | head -5
# Expected: COLMAP 3.x -- Structure-from-Motion and Multi-View Stereo
```

**If COLMAP is missing, reconstruction falls back to placeholder poses.**

### 4. CUDA Acceleration (Optional)

For GPU-accelerated 3DGS (10-100x faster):

```bash
# Requires CUDA 12.1+ installed
export CUDA_HOME=/usr/local/cuda

# Install CUDA rasterizer
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git

# Install simple-knn
pip install git+https://github.com/camenduru/simple-knn.git
```

**Without CUDA:** Uses slower Python rasterizer (works, but ~100x slower).

---

## Cloud Deployment

### Prerequisites

1. Google Cloud Project with billing enabled
2. `gcloud` CLI installed and authenticated
3. Enable required APIs:
   ```bash
   gcloud services enable \
       cloudfunctions.googleapis.com \
       cloudbuild.googleapis.com \
       run.googleapis.com \
       pubsub.googleapis.com \
       storage.googleapis.com
   ```

### Deploy Everything

```bash
export PROJECT_ID=your-project-id
export REGION=us-central1
./scripts/deploy_cloud.sh
```

### Manual Deployment

#### 1. Cloud Function

```bash
cd functions

gcloud functions deploy storage_trigger \
    --gen2 \
    --runtime python311 \
    --trigger-resource your-bucket.appspot.com \
    --trigger-event google.storage.object.finalize \
    --entry-point on_storage_finalize \
    --region us-central1 \
    --memory 512MB \
    --timeout 60s
```

#### 2. Pub/Sub

```bash
# Create topic
gcloud pubsub topics create pipeline-trigger

# Create subscription (push to Cloud Run)
gcloud pubsub subscriptions create pipeline-job-trigger \
    --topic=pipeline-trigger \
    --push-endpoint="https://your-cloud-run-url/trigger"
```

#### 3. Cloud Run Job

```bash
# Build container
gcloud builds submit --tag gcr.io/$PROJECT_ID/blueprint-pipeline:latest

# Create job with GPU
gcloud run jobs create blueprint-pipeline-job \
    --image gcr.io/$PROJECT_ID/blueprint-pipeline:latest \
    --region us-central1 \
    --cpu 8 \
    --memory 32Gi \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --task-timeout 60m
```

---

## Usage

### Local Processing

```bash
# Process a local video
python -m blueprint_pipeline.runner \
    --manifest configs/example_session.yaml \
    --output output/

# Process iOS upload format
python -m blueprint_pipeline.runner \
    --ios-upload /path/to/capture/raw/
```

### Cloud Processing

Upload to GCS to trigger automatic processing:

```bash
gsutil cp -r /path/to/capture gs://your-bucket/scenes/my-scene/iphone/2024-12-26T12:00:00-xyz/raw/
```

Required files:
- `manifest.json` - Capture metadata
- `walkthrough.mov` - Video file

Optional files:
- `arkit/poses.jsonl` - ARKit camera poses (skips SLAM)
- `arkit/intrinsics.json` - Camera parameters
- `arkit/depth/*.png` - Depth maps

---

## Configuration

### Session Manifest

```yaml
session_id: my-capture-001
capture_start: "2024-12-26T12:00:00Z"

device:
  platform: iOS
  model: iPhone 15 Pro
  resolution: 1920x1080
  fps: 30

sensor:
  type: rgb_only  # or rgb_depth, visual_inertial
  has_arkit_poses: true

scale_anchors:
  - anchor_type: aruco_board
    size_meters: 0.05
    notes: "5cm ArUco markers in scene corners"

clips:
  - uri: walkthrough.mov
    fps: 30

pipeline_config:
  target_fps: 2.0
  blur_threshold: 100.0
  max_keyframes: 200
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_PROJECT_ID` | `blueprint-8c1ca` | GCP project ID |
| `PIPELINE_REGION` | `us-central1` | GCP region |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID |
| `JOB_PAYLOAD` | - | Cloud Run job input (JSON) |

---

## Verification

### Check Installation

```bash
python -c "
import cv2
import torch
from blueprint_pipeline.video2zeroscene.pipeline import CapturePipeline
from blueprint_pipeline.reconstruction.gaussian_splatting import GaussianModel

print('✅ OpenCV:', cv2.__version__)
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA:', 'Available' if torch.cuda.is_available() else 'Not available')
print('✅ All imports successful!')
"
```

### Check COLMAP

```bash
colmap feature_extractor --help | head -5
```

### Check CUDA Rasterizer

```bash
python -c "
try:
    from diff_gaussian_rasterization import GaussianRasterizer
    print('✅ CUDA rasterizer available (10-100x faster)')
except ImportError:
    print('⚠️ CUDA rasterizer not available (using Python fallback)')
"
```

---

## Troubleshooting

### COLMAP Not Found

```
Error: COLMAP not found
```

**Solution:** Install COLMAP or add to PATH:
```bash
# Ubuntu
sudo apt install colmap

# Or build from source
git clone https://github.com/colmap/colmap.git
cd colmap && mkdir build && cd build
cmake .. && make -j8 && sudo make install
```

### CUDA Error

```
OSError: CUDA_HOME environment variable is not set
```

**Solution:**
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Import Errors

```
ModuleNotFoundError: No module named 'blueprint_pipeline'
```

**Solution:**
```bash
pip install -e .
```

### Cloud Function Not Triggering

1. Check function logs: `gcloud functions logs read storage_trigger`
2. Verify bucket name in trigger
3. Ensure required files exist (manifest.json, walkthrough.mov)

---

## Next Steps

- Read [ARKit Integration](ARKIT_INTEGRATION.md) for iOS capture details
- Read [Scale Calibration](SCALE_CALIBRATION.md) for metric accuracy
- Check [examples/](../examples/) for sample manifests
