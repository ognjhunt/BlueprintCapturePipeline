# 3D Gaussian Splatting Renderer

This document describes the 3DGS rendering module for generating static-scene videos from ZeroScene bundles for use with DWM (Dexterous World Models).

## Overview

The rendering module provides GPU-accelerated 3D Gaussian Splatting rendering with multiple backend options:

| Backend | Speed | Quality | Requirements |
|---------|-------|---------|--------------|
| `diff-gaussian-rasterization` | Fastest | Best | CUDA + custom build |
| `gsplat` | Fast | Great | CUDA + pip install |
| `cpu-numpy` | Slow | Good | None (always works) |

## Installation

### Basic (CPU-only)
```bash
pip install -e ".[rendering]"
```

### With CUDA Acceleration

**Option 1: gsplat (Recommended for ease of install)**
```bash
pip install gsplat
pip install -e ".[rendering]"
```

**Option 2: diff-gaussian-rasterization (Best quality/speed)**
```bash
# Clone and install the official 3DGS rasterizer
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization
cd diff-gaussian-rasterization
pip install .

# Then install the pipeline
pip install -e ".[rendering]"
```

## Quick Start

### Command Line

```bash
# Render static scene video from ZeroScene bundle
python scripts/render_static_scene.py output/zeroscene -o static_scene.mp4

# Or use the installed command
render-static-scene output/zeroscene -o static_scene.mp4
```

### Python API

```python
from blueprint_pipeline.video2zeroscene.rendering import GaussianRenderer

# Load from ZeroScene bundle
renderer = GaussianRenderer.from_zeroscene("output/zeroscene")

# Render all frames along trajectory
frames = renderer.render_trajectory()

# Save as video
renderer.save_video(frames, "static_scene.mp4", fps=30)
```

## Detailed Usage

### Loading Assets

```python
from blueprint_pipeline.video2zeroscene.rendering import (
    GaussianRenderer,
    GaussianModel,
    CameraTrajectory,
    RenderSettings,
)

# Option 1: Load from ZeroScene bundle (easiest)
renderer = GaussianRenderer.from_zeroscene("output/zeroscene")

# Option 2: Load from individual files
renderer = GaussianRenderer(
    gaussians_path="gaussians.ply",
    trajectory_path="trajectory.json",
    intrinsics_path="intrinsics.json",
)

# Option 3: Load model and trajectory separately
model = GaussianModel()
model.load_ply("gaussians.ply")

trajectory = CameraTrajectory.from_files(
    "intrinsics.json",
    "trajectory.json",
)

renderer = GaussianRenderer(model=model, trajectory=trajectory)
```

### Rendering Options

```python
from blueprint_pipeline.video2zeroscene.rendering import RenderSettings

settings = RenderSettings(
    background_color=(1.0, 1.0, 1.0),  # White background
    scaling_modifier=1.0,               # Scale Gaussians
    sh_degree=-1,                       # Auto-select from model
    min_opacity=0.01,                   # Filter low-opacity Gaussians
    max_sh_degree=3,                    # Maximum SH degree to use
)

renderer = GaussianRenderer.from_zeroscene(
    "output/zeroscene",
    settings=settings,
)
```

### Rendering Individual Frames

```python
from blueprint_pipeline.video2zeroscene.rendering import Camera

# Render a specific camera from the trajectory
output = renderer.render_frame(renderer.trajectory[0])
print(f"Rendered {output.n_rendered} Gaussians")

# Access the rendered image
image = output.to_uint8()  # (H, W, 3) uint8
```

### Novel View Synthesis

```python
# Render from a custom viewpoint
output = renderer.render_novel_view(
    position=(0, 1.5, -2),      # Camera position
    look_at=(0, 0, 0),          # Look at point
    up=(0, 1, 0),               # Up vector
    fov=60.0,                   # Field of view
    width=1920,
    height=1080,
)
```

### Progress Tracking

```python
def progress_callback(current, total):
    print(f"Rendering: {current}/{total} ({100*current/total:.1f}%)")

frames = renderer.render_trajectory(progress_callback=progress_callback)
```

### Saving Output

```python
# Save as video
renderer.save_video(frames, "output.mp4", fps=30)

# Save as image sequence
renderer.save_frames(frames, "frames/", prefix="frame", format="png")
```

## Camera Utilities

The module includes utilities for camera manipulation:

```python
from blueprint_pipeline.video2zeroscene.rendering import (
    Camera,
    CameraTrajectory,
    quaternion_to_matrix,
    matrix_to_quaternion,
    focal_to_fov,
    fov_to_focal,
)

# Load trajectory from ZeroScene
trajectory = CameraTrajectory.from_zeroscene("output/zeroscene")

# Get scene bounds from camera positions
bounds_min, bounds_max = trajectory.get_bounds()

# Interpolate between poses
camera_at_t = trajectory.interpolate(0.5)  # t in [0, 1]

# Create camera from pose dict (ZeroScene format)
pose = {
    "rotation": [1, 0, 0, 0],     # Quaternion (w, x, y, z)
    "translation": [0, 0, 0],     # Position
    "timestamp": 0.0,
}
intrinsics = {
    "fx": 1500, "fy": 1500,
    "cx": 960, "cy": 720,
    "width": 1920, "height": 1440,
}
camera = Camera.from_pose_dict(pose, intrinsics)
```

## Gaussian Model Access

```python
from blueprint_pipeline.video2zeroscene.rendering import GaussianModel

model = GaussianModel()
model.load_ply("gaussians.ply")

print(f"Loaded {model.num_gaussians:,} Gaussians")
print(f"SH degree: {model.sh_degree}")
print(f"Center: {model.center}")
print(f"Extent: {model.extent}")

# Access raw parameters
xyz = model.xyz           # (N, 3) positions
opacities = model.opacities  # (N,) opacities
scales = model.scales     # (N, 3) scales
rotations = model.rotations  # (N, 4) quaternions

# Get colors (with view-dependent effects)
colors = model.get_colors(viewdirs)  # (N, 3) RGB

# Filter Gaussians
filtered = model.filter_by_opacity(min_opacity=0.1)
cropped = model.filter_by_bounds(bounds_min, bounds_max)

# Save modified model
filtered.save_ply("filtered_gaussians.ply")
```

## Backend Selection

```python
from blueprint_pipeline.video2zeroscene.rendering.rasterizer import (
    available_backends,
    get_best_rasterizer,
)

# Check available backends
print(f"Available: {available_backends()}")

# Get best available
rasterizer = get_best_rasterizer(device="cuda")
print(f"Using: {rasterizer.name}")

# Force specific backend
rasterizer = get_best_rasterizer(preferred="cpu-numpy")
```

## CLI Reference

```bash
render-static-scene [OPTIONS] ZEROSCENE_PATH

Arguments:
  ZEROSCENE_PATH          Path to zeroscene/ directory

Options:
  -o, --output PATH       Output video path (e.g., static_scene.mp4)
  --output-dir PATH       Output directory for image sequence
  --format {png,jpg}      Image format for sequence (default: png)
  --fps FLOAT             Video frame rate (default: 30)
  --backend NAME          Rasterizer backend (auto-select if not specified)
  --device {cuda,cpu}     Device to use (default: cuda)
  --background R G B      Background color (default: 0 0 0 = black)
  --sh-degree INT         SH degree (-1 = auto, 0-3 for manual)
  --scale FLOAT           Gaussian scale modifier (default: 1.0)
  --min-opacity FLOAT     Minimum opacity filter (default: 0.0)
  --info                  Print scene info and exit
  --list-backends         List available backends and exit
  -v, --verbose           Verbose output

Examples:
  # Basic rendering
  render-static-scene output/zeroscene -o video.mp4

  # CPU-only rendering
  render-static-scene output/zeroscene -o video.mp4 --backend cpu-numpy --device cpu

  # White background, higher FPS
  render-static-scene output/zeroscene -o video.mp4 --background 1 1 1 --fps 60

  # Save as image sequence
  render-static-scene output/zeroscene --output-dir frames/ --format png
```

## Performance Tips

1. **Use GPU backend**: The CUDA backends are 10-100x faster than CPU
2. **Install gsplat**: Easier to install than diff-gaussian-rasterization
3. **Filter low-opacity**: Set `--min-opacity 0.01` to skip invisible Gaussians
4. **Lower SH degree**: Use `--sh-degree 1` for faster rendering (less quality)
5. **Reduce scale**: Use `--scale 0.8` for smaller Gaussians (faster, less overlap)

## File Formats

### Camera Intrinsics (intrinsics.json)
```json
{
  "fx": 1500.0,
  "fy": 1500.0,
  "cx": 960.0,
  "cy": 720.0,
  "width": 1920,
  "height": 1440
}
```

### Camera Trajectory (trajectory.json)
```json
[
  {
    "frame_id": "frame_0001",
    "rotation": [1.0, 0.0, 0.0, 0.0],
    "translation": [0.0, 0.0, 0.0],
    "timestamp": 0.0
  }
]
```

### Gaussians PLY Format (3DGS Standard)
```
ply
format binary_little_endian 1.0
element vertex N
property float x
property float y
property float z
property float opacity
property float scale_0/1/2
property float rot_0/1/2/3
property float f_dc_0/1/2
property float f_rest_0/1/2/...
end_header
<binary data>
```

## Integration with DWM

The output video from this renderer is the "static-scene conditioning video" for DWM:

```python
# 1. Render static scene
from blueprint_pipeline.video2zeroscene.rendering import render_zeroscene

render_zeroscene(
    zeroscene_path="output/zeroscene",
    output_path="static_scene.mp4",
)

# 2. (Future) Add hand trajectories with HaMeR
# hand_frames = hamer_extract(original_video)

# 3. (Future) Generate interaction videos with DWM
# from dwm import DexterousWorldModel
# model = DexterousWorldModel.load_pretrained()
# interaction = model.generate(
#     static_scene_video=static_frames,
#     hand_video=hand_frames,
#     text_prompt="open the drawer"
# )
```

## Troubleshooting

### "No rasterizer backend available"
Install at least one CUDA backend or ensure NumPy/SciPy are available:
```bash
pip install gsplat  # Easiest CUDA option
# OR
pip install numpy scipy  # CPU fallback
```

### "CUDA out of memory"
- Reduce image resolution in intrinsics
- Use `--min-opacity 0.1` to filter Gaussians
- Use `--backend cpu-numpy` for CPU rendering

### "Slow rendering"
- Ensure CUDA backend is being used (check `--list-backends`)
- Install gsplat or diff-gaussian-rasterization
- Reduce SH degree with `--sh-degree 1`

### "Black/corrupted output"
- Check that gaussians.ply exists and is valid
- Verify camera trajectory is not empty
- Try with `--verbose` to see detailed errors
