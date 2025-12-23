# DWM (Dexterous World Models) Compatibility

## Overview

This pipeline now outputs **DWM-compatible** scene bundles that can be used for training and running Dexterous World Models ([DWM paper](https://arxiv.org/html/2512.17907v1), [project page](https://snuvclab.github.io/dwm/)).

DWM is a scene-action-conditioned video diffusion model that generates realistic egocentric interaction videos by predicting how a static 3D scene changes when hands manipulate objects.

## What DWM Is

DWM turns static 3D reconstructions ("digital twins") into interactive visual simulations:

- **Input**: Static 3D scene + hand motion trajectory
- **Output**: Photorealistic egocentric video showing plausible interaction (e.g., door opens, drawer slides, objects move)
- **Key Innovation**: Works without explicit articulation modeling (no need for hinges, joints, etc.)

## Why DWM Matters for Blueprint Vision

DWM aligns perfectly with Blueprint's flywheel:

1. **Phase 1-2**: We already generate 3D scenes from video → DWM lets us turn those into **interactive training data**
2. **Phase 3**: Real-world captures → train better DWM models → generate location-specific interaction data
3. **Robotics customers**: DWM-generated data = cheaper alternative to expensive egocentric capture
4. **Pre-training use case**: Generate diverse interaction videos for vision model pre-training

## What DWM Needs

To use our scene bundles with DWM, you need:

### 1. Static 3D Scene (Renderable)
- **Format**: 3D Gaussian Splatting (3DGS) PLY file
- **Our output**: `zeroscene/background/gaussians.ply`
- **Why Gaussians**: DWM paper explicitly used 3DGS for real-world evaluation; Gaussians render higher quality novel views than meshes

### 2. Camera Trajectory
- **Format**: Per-frame camera poses (rotation + translation)
- **Our output**: `zeroscene/camera/trajectory.json`
- **Contains**:
  ```json
  [
    {
      "frame_id": "frame_0001",
      "rotation": [qw, qx, qy, qz],  // Quaternion
      "translation": [tx, ty, tz],    // World coordinates
      "timestamp": 1.234
    },
    ...
  ]
  ```

### 3. Camera Intrinsics
- **Format**: Pinhole camera parameters
- **Our output**: `zeroscene/camera/intrinsics.json`
- **Contains**:
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

### 4. (For DWM) Hand Mesh Trajectory
- **Not yet included** in this pipeline
- DWM needs rendered hand meshes aligned with camera trajectory
- Options:
  - Extract from video using HaMeR (hand mesh estimator)
  - Mocap / gloves during capture
  - Synthetic hand trajectories

## ZeroScene Bundle Structure (DWM-Compatible)

After running the video2zeroscene pipeline, you'll get:

```
zeroscene/
├── scene_info.json              # Metadata including dwm_compatible flag
├── background/
│   ├── gaussians.ply            # ← 3D Gaussians for DWM rendering
│   ├── mesh.glb                 # Extracted mesh (for physics sim)
│   ├── collision.glb            # Collision mesh
│   └── info.json                # has_gaussians: true
├── camera/
│   ├── intrinsics.json          # ← Camera intrinsics
│   └── trajectory.json          # ← Camera poses
└── objects/
    └── obj_XXXX/                # Segmented objects (optional for DWM)
        ├── mesh.glb
        ├── pose.json
        └── ...
```

### Scene Info Metadata

The `scene_info.json` includes DWM compatibility indicators:

```json
{
  "capture_id": "kitchen_001",
  "has_gaussians": true,
  "gaussians_format": "3dgs_ply",
  "dwm_compatible": true,
  "scale_factor": 1.0,
  "meters_per_unit": 1.0,
  ...
}
```

## Using the Output with DWM

### Step 1: Check Compatibility

```python
import json
from pathlib import Path

scene_info = json.loads(Path("zeroscene/scene_info.json").read_text())
if scene_info["dwm_compatible"]:
    print("✓ Scene is DWM-compatible!")
    print(f"  Gaussians: {scene_info['has_gaussians']}")
    print(f"  Format: {scene_info['gaussians_format']}")
```

### Step 2: Load Assets

```python
gaussians_path = Path("zeroscene/background/gaussians.ply")
trajectory_path = Path("zeroscene/camera/trajectory.json")
intrinsics_path = Path("zeroscene/camera/intrinsics.json")

# Load trajectory
trajectory = json.loads(trajectory_path.read_text())

# Load intrinsics
intrinsics = json.loads(intrinsics_path.read_text())
```

### Step 3: Render Static Scene Video

You need a 3DGS renderer to create the "static-scene conditioning video":

```python
from gaussian_renderer import GaussianRenderer  # Your 3DGS renderer

renderer = GaussianRenderer(gaussians_path, intrinsics)

static_frames = []
for pose in trajectory:
    # Render from this camera pose
    frame = renderer.render(
        rotation=pose["rotation"],
        translation=pose["translation"]
    )
    static_frames.append(frame)
```

### Step 4: Add Hand Motion (External)

For DWM, you also need to render hand meshes:

```python
# Option 1: Extract from video using HaMeR
from hamer import HaMeR
estimator = HaMeR()
hand_meshes = estimator.estimate_from_video("capture.mp4")

# Option 2: Use mocap data
hand_meshes = load_mocap_hands("hands.json")

# Render hand-only video
hand_frames = []
for pose, hand_mesh in zip(trajectory, hand_meshes):
    hand_frame = render_hand_mesh(hand_mesh, pose, intrinsics)
    hand_frames.append(hand_frame)
```

### Step 5: Run DWM Inference

```python
from dwm import DexterousWorldModel  # When DWM code is released

model = DexterousWorldModel.load_pretrained()

# Generate interaction video
interaction_video = model.generate(
    static_scene_video=static_frames,
    hand_video=hand_frames,
    text_prompt="open the cabinet door"  # Optional
)
```

## DWM vs. Physics Simulation

### What DWM Provides
- ✓ Photorealistic interaction videos
- ✓ Works without explicit articulation (no joint modeling)
- ✓ Faster generation than real-world capture
- ✓ Good for visual planning / imagination

### What DWM Does NOT Provide
- ✗ Physically accurate forces/torques
- ✗ Updated 3D scene state (object poses, articulation angles)
- ✗ Contact constraints for robot control
- ✗ Long-horizon physical consistency guarantees

### When to Use DWM
- **Pre-training visual encoders**: Generate lots of interaction videos
- **Data augmentation**: Expand limited real-world ego data
- **Visual planning**: "What would it look like if I did X?"
- **Action ranking**: Simulate multiple candidates, pick best

### When to Use Physics Sim (Isaac Sim)
- **Robot policy training** with contact/force requirements
- **Manipulation planning** needing guaranteed feasibility
- **Long-horizon rollouts** (DWM is video-based, finite length)
- **Precise state estimation** (joint angles, object velocities)

### Hybrid Approach
1. Use DWM to generate **visual diversity** (many scenarios, viewpoints, actions)
2. Use physics sim for **final policy training** (accurate dynamics, contact)
3. Combine: DWM for pre-training → Isaac Sim for fine-tuning

## Pipeline Output Guarantees

When you run the video2zeroscene pipeline:

### ✓ Guaranteed Outputs
1. **3D Gaussians** in standard PLY format (renderable)
2. **Camera trajectory** with metric scale (if ARKit or scale anchors used)
3. **Camera intrinsics** matching the input video
4. **Temporal alignment** (trajectory indices match video frames)

### ⚠️ Quality Depends On
1. **Camera motion**: Smooth motion → better SLAM → better Gaussians
2. **Dynamic masking**: SAM3 must detect hands/people for clean reconstruction
3. **Scale calibration**: Needs ArUco markers or ARKit for metric scale
4. **Lighting**: Well-lit scenes → better feature matching → better poses

### 🔧 Potential Issues
1. **SLAM failure**: Low-texture environments may fail reconstruction
   - Mitigation: Use ARKit poses (iOS) or add visual features to scene
2. **Gaussian artifacts**: Fast motion or specular surfaces may create "floaters"
   - Mitigation: Slower capture, dynamic masking, SuGaR mesh extraction
3. **Scale drift**: Without anchors, scale is arbitrary
   - Mitigation: Place ArUco markers in scene before capture

## Roadmap: DWM Integration

### Current Status (Phase 1)
- ✓ Export 3D Gaussians in DWM-compatible format
- ✓ Export camera trajectory and intrinsics
- ✓ Metric scale calibration (ARKit / anchors)

### Next Steps (Phase 2)
- [ ] Integrate HaMeR for automatic hand mesh extraction
- [ ] Export hand trajectory alongside camera trajectory
- [ ] Validate DWM rendering pipeline with real captures

### Future (Phase 3)
- [ ] DWM training data generation service
  - Input: ZeroScene bundle + interaction clips
  - Output: (static scene, hand motion, interaction video) triplets
- [ ] DWM fine-tuning on Blueprint-captured data
- [ ] Location-specific DWM models for deployed robots

## Example: End-to-End DWM Workflow

```bash
# 1. Capture video with Meta glasses + BlueprintCapture iOS app
# → Saves to GCS with ARKit poses

# 2. Run video2zeroscene pipeline
python -m blueprint_pipeline.video2zeroscene \
  --video capture.mp4 \
  --arkit-data arkit/ \
  --output zeroscene/

# 3. Verify DWM compatibility
python -c "
import json
info = json.load(open('zeroscene/scene_info.json'))
print(f\"DWM compatible: {info['dwm_compatible']}\")
print(f\"Gaussians: {info['has_gaussians']}\")
"

# 4. Extract hand meshes from video (when HaMeR integrated)
python -m blueprint_pipeline.extract_hands \
  --video capture.mp4 \
  --trajectory zeroscene/camera/trajectory.json \
  --output zeroscene/hands/

# 5. Run DWM (when code released)
python -m dwm.generate \
  --gaussians zeroscene/background/gaussians.ply \
  --trajectory zeroscene/camera/trajectory.json \
  --hands zeroscene/hands/ \
  --output interaction.mp4
```

## FAQ

**Q: Can I use the extracted mesh instead of Gaussians?**

A: Technically yes, but quality will be lower. DWM was designed for 3DGS because Gaussians render higher-quality novel views than meshes. The mesh is better for physics simulation.

**Q: Do I need articulated objects in my scene?**

A: No! DWM's key innovation is that it learns interaction dynamics from data, without explicit joint modeling. A monolithic static scan is sufficient.

**Q: What if SLAM fails and I don't have Gaussians?**

A: The scene won't be DWM-compatible. Check `scene_info.json` → `dwm_compatible: false`. Solutions:
- Use ARKit poses (iOS) for better tracking
- Improve capture technique (slower motion, better lighting)
- Add visual features to the environment

**Q: Can I use this for robot training?**

A: DWM is best for **visual pre-training** and **data augmentation**. For robot policy training with contact/force requirements, use physics simulation (Isaac Sim) or real-world rollouts. DWM does not output forces, contact points, or physically grounded state updates.

**Q: How do I get hand motion trajectories?**

A: Options:
1. Vision-based: Run HaMeR (hand mesh estimator) on your video
2. Mocap: Capture with gloves/markers
3. Synthetic: Generate hand trajectories programmatically for specific tasks

**Q: What's the typical Gaussian file size?**

A: 50-500 MB for room-scale scenes, depending on resolution and training iterations. Larger spaces (grocery store) may be 1-2 GB.

## References

- **DWM Paper**: [Dexterous World Models (arXiv)](https://arxiv.org/html/2512.17907v1)
- **DWM Project Page**: [https://snuvclab.github.io/dwm/](https://snuvclab.github.io/dwm/)
- **3D Gaussian Splatting**: [3DGS Paper (SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **WildGS-SLAM**: RGB-only SLAM backend we use for reconstruction
- **HaMeR**: Hand mesh estimator for extracting hand trajectories
