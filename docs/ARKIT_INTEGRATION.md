# ARKit Integration Guide

This guide explains how to capture and upload ARKit data from iOS devices for high-quality 3D reconstruction with metric scale.

## Overview

When capturing video on iOS devices (iPhone/iPad with LiDAR), you can also capture ARKit pose data. This provides:

- **Metric scale**: Real-world measurements in meters (no scale calibration needed)
- **Accurate poses**: ARKit's visual-inertial odometry is highly accurate
- **Faster processing**: Skip SLAM reconstruction entirely
- **Better quality**: ARKit poses often outperform pure visual SLAM

## Data Format

### Directory Structure

Upload the following files to GCS:

```
gs://bucket/scenes/{scene_id}/iphone/{timestamp}/raw/
├── manifest.json          # Required: iOS capture metadata
├── walkthrough.mov        # Required: Main video file
├── arkit/
│   ├── poses.jsonl        # Camera poses (one JSON per line)
│   ├── intrinsics.json    # Camera intrinsics
│   ├── frames.jsonl       # Frame metadata (optional)
│   └── depth/             # Depth maps if LiDAR available (optional)
│       ├── 0001.png
│       ├── 0002.png
│       └── ...
└── motion.jsonl           # IMU data (optional)
```

### manifest.json

```json
{
  "scene_id": "ChIJ9_QNuFHkrIkR3YlZInIh5Ow",
  "video_uri": "walkthrough.mov",
  "device_model": "iPhone 15 Pro",
  "os_version": "17.2",
  "fps_source": 30,
  "width": 1920,
  "height": 1080,
  "capture_start_epoch_ms": 1702144245000,
  "has_lidar": true,
  "scale_hint_m_per_unit": 1.0,
  "intended_space_type": "indoor"
}
```

### arkit/poses.jsonl

Each line is a JSON object representing the camera pose at a specific frame:

```json
{"frame_id": 0, "timestamp": 0.0, "transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
{"frame_id": 1, "timestamp": 0.033, "transform": [[0.999,0.01,0.02,0.1],[...],[...],[0,0,0,1]]}
```

**Transform format**: 4x4 camera-to-world transformation matrix (row-major):
```
[R_00, R_01, R_02, T_x]
[R_10, R_11, R_12, T_y]
[R_20, R_21, R_22, T_z]
[0,    0,    0,    1  ]
```

Where:
- R is the 3x3 rotation matrix (camera-to-world)
- T is the 3D translation vector (camera position in world coordinates)

### arkit/intrinsics.json

```json
{
  "fx": 1488.2412109375,
  "fy": 1488.2412109375,
  "cx": 960.0,
  "cy": 540.0,
  "width": 1920,
  "height": 1080,
  "camera_model": "PINHOLE"
}
```

## iOS App Implementation

### Swift Code Example

```swift
import ARKit
import AVFoundation

class ARKitCapture: NSObject, ARSessionDelegate {
    private var session: ARSession!
    private var poses: [(Int, TimeInterval, simd_float4x4)] = []

    func startCapture() {
        session = ARSession()
        session.delegate = self

        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth]  // Enable LiDAR depth

        session.run(config)
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Save camera pose
        let transform = frame.camera.transform
        let timestamp = frame.timestamp
        poses.append((poses.count, timestamp, transform))

        // Optionally save depth map
        if let depthMap = frame.sceneDepth?.depthMap {
            saveDepthMap(depthMap, frameId: poses.count)
        }
    }

    func exportPoses(to url: URL) {
        var lines: [String] = []

        for (frameId, timestamp, transform) in poses {
            let matrix = [
                [transform.columns.0.x, transform.columns.0.y, transform.columns.0.z, transform.columns.0.w],
                [transform.columns.1.x, transform.columns.1.y, transform.columns.1.z, transform.columns.1.w],
                [transform.columns.2.x, transform.columns.2.y, transform.columns.2.z, transform.columns.2.w],
                [transform.columns.3.x, transform.columns.3.y, transform.columns.3.z, transform.columns.3.w]
            ]

            let data: [String: Any] = [
                "frame_id": frameId,
                "timestamp": timestamp,
                "transform": matrix
            ]

            if let jsonData = try? JSONSerialization.data(withJSONObject: data),
               let jsonString = String(data: jsonData, encoding: .utf8) {
                lines.append(jsonString)
            }
        }

        let content = lines.joined(separator: "\n")
        try? content.write(to: url, atomically: true, encoding: .utf8)
    }

    func exportIntrinsics(from frame: ARFrame, to url: URL) {
        let intrinsics = frame.camera.intrinsics
        let resolution = frame.camera.imageResolution

        let data: [String: Any] = [
            "fx": intrinsics[0][0],
            "fy": intrinsics[1][1],
            "cx": intrinsics[2][0],
            "cy": intrinsics[2][1],
            "width": Int(resolution.width),
            "height": Int(resolution.height),
            "camera_model": "PINHOLE"
        ]

        if let jsonData = try? JSONSerialization.data(withJSONObject: data, options: .prettyPrinted) {
            try? jsonData.write(to: url)
        }
    }
}
```

## Pipeline Processing

When ARKit data is detected, the pipeline:

1. **Loads poses.jsonl** - Parses camera-to-world transforms
2. **Converts to COLMAP convention** - Inverts to world-to-camera
3. **Extracts intrinsics** - Uses arkit/intrinsics.json if available
4. **Skips SLAM** - No reconstruction needed, poses are already metric
5. **Trains 3DGS** - Uses video frames + ARKit poses directly

### Code Path

```python
# In jobs/reconstruction.py
if file_status.get("arkit/poses.jsonl"):
    # Load ARKit poses directly
    poses = load_arkit_poses(raw_prefix / "arkit/poses.jsonl")
    intrinsics = load_arkit_intrinsics(raw_prefix / "arkit/intrinsics.json")

    # Skip SLAM - ARKit provides metric scale
    scale_factor = 1.0
    scale_confidence = 1.0
else:
    # Fall back to COLMAP SLAM
    poses, intrinsics = run_colmap(frames_dir)
```

## Benefits

| Feature | Without ARKit | With ARKit |
|---------|--------------|------------|
| Scale accuracy | ~10-50% error | <1% error |
| Processing time | 10-30 min | 2-5 min |
| Pose accuracy | Good | Excellent |
| LiDAR depth | Not used | Used for dense reconstruction |
| Robustness | Fails on textureless scenes | Works everywhere |

## Troubleshooting

### Poses Not Loading

Check that `poses.jsonl` uses the correct format:
```bash
head -1 arkit/poses.jsonl | jq .
```

Expected output:
```json
{
  "frame_id": 0,
  "timestamp": 0.0,
  "transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
}
```

### Frame Mismatch

If video frames don't match pose count:
- ARKit runs at 60Hz, video at 30Hz → use frame_id to align
- Timestamps can be used for interpolation

### Coordinate System

ARKit uses:
- **Y-up** coordinate system
- **Right-handed** convention
- **Camera-to-world** transforms (we convert to world-to-camera)

The pipeline handles this conversion automatically.
