# Scale Calibration with ArUco Markers

This guide explains how to use ArUco markers for metric scale calibration in 3D reconstruction.

## Overview

When processing RGB-only video (no ARKit poses, no LiDAR), the reconstruction is in an arbitrary scale. To achieve real-world metric scale, you can:

1. **Use ArUco markers** - Place printed markers of known size in the scene
2. **Use AprilTag markers** - Alternative marker system
3. **Manual scale hints** - Specify known object dimensions

## ArUco Markers

### What Are ArUco Markers?

ArUco markers are square fiducial markers with a unique binary pattern. They're:
- Easy to detect with computer vision
- Robust to partial occlusion
- Provide 6-DOF pose estimation
- Available in different dictionary sizes

### Supported Dictionaries

| Dictionary | Marker Count | Best For |
|-----------|--------------|----------|
| DICT_4X4_50 | 50 markers | Small scenes |
| DICT_5X5_100 | 100 markers | Medium scenes |
| DICT_6X6_250 | 250 markers | Large scenes |
| DICT_7X7_1000 | 1000 markers | Complex scenes |

### Generating Markers

#### Using OpenCV (Python)

```python
import cv2

# Generate a 5cm ArUco marker (ID 0)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
marker_image = cv2.aruco.generateImageMarker(dictionary, 0, 200)  # 200 pixels

# Save
cv2.imwrite("aruco_marker_0.png", marker_image)
```

#### Using Online Generator

Visit: https://chev.me/arucogen/

1. Select dictionary (5x5, 100 markers)
2. Enter marker ID (0)
3. Set marker size (50mm = 5cm)
4. Download and print

### Printing Guidelines

1. **Print at exact size** - Verify with ruler (e.g., exactly 5cm × 5cm)
2. **Use matte paper** - Avoid glossy surfaces (causes reflections)
3. **Include white border** - Maintain the white margin around the black square
4. **Keep flat** - Don't bend or fold the marker

### Placement in Scene

```
Recommended placement:
┌────────────────────────────────────────┐
│                                        │
│    ┌───┐                    ┌───┐     │
│    │ A │                    │ B │     │
│    └───┘                    └───┘     │
│                                        │
│              SCENE                     │
│                                        │
│    ┌───┐                    ┌───┐     │
│    │ C │                    │ D │     │
│    └───┘                    └───┘     │
│                                        │
└────────────────────────────────────────┘
```

**Guidelines:**
- Place 2-4 markers visible from the main capture viewpoints
- Position at different depths for better scale estimation
- Avoid markers on moving objects
- Keep markers vertical or horizontal (not at extreme angles)

## Configuration

### Session Manifest

Specify scale anchors in your session manifest:

```yaml
session_id: sample-session-001
sensor:
  type: rgb_only
scale_anchors:
  - anchor_type: aruco_board
    size_meters: 0.05  # 5cm marker
    dictionary: DICT_5X5_100
    marker_ids: [0, 1, 2, 3]  # Optional: specific markers to look for
    notes: "Four 5cm ArUco markers placed in corners"

  # Alternative: single marker
  - anchor_type: aruco_single
    size_meters: 0.10  # 10cm marker
    dictionary: DICT_4X4_50
    marker_id: 0
```

### iOS Manifest

When using the iOS app, include in `manifest.json`:

```json
{
  "scene_id": "my-scene",
  "scale_anchors": [
    {
      "anchor_type": "aruco_board",
      "size_meters": 0.05,
      "dictionary": "DICT_5X5_100"
    }
  ]
}
```

## How It Works

### Detection Pipeline

1. **Extract frames** - Sample keyframes from video
2. **Detect markers** - Use OpenCV ArUco detector
3. **Compute pixel size** - Measure marker side length in pixels
4. **Calculate scale** - `scale = size_meters / size_pixels`
5. **Aggregate** - Average across multiple observations
6. **Apply** - Scale all poses and Gaussians

### Code Example

```python
import cv2
import numpy as np

def detect_aruco_scale(image, marker_size_meters, dictionary_id=cv2.aruco.DICT_5X5_100):
    """Detect ArUco markers and compute scale factor."""

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(image)

    if ids is None:
        return None, 0.0

    # Compute average pixel size
    pixel_sizes = []
    for corner in corners:
        # corners shape: (1, 4, 2)
        pts = corner[0]
        # Compute side lengths
        side1 = np.linalg.norm(pts[0] - pts[1])
        side2 = np.linalg.norm(pts[1] - pts[2])
        side3 = np.linalg.norm(pts[2] - pts[3])
        side4 = np.linalg.norm(pts[3] - pts[0])
        avg_side = (side1 + side2 + side3 + side4) / 4
        pixel_sizes.append(avg_side)

    avg_pixel_size = np.mean(pixel_sizes)
    scale_factor = marker_size_meters / avg_pixel_size
    confidence = min(1.0, len(pixel_sizes) / 4.0)  # Higher with more markers

    return scale_factor, confidence
```

## Scale Confidence

The pipeline computes a confidence score (0-1) based on:

| Factor | Weight |
|--------|--------|
| Number of markers detected | 30% |
| Detection consistency across frames | 30% |
| Marker visibility (not partially occluded) | 20% |
| Marker orientation (facing camera) | 20% |

**Confidence thresholds:**
- `> 0.8` - High confidence, use as-is
- `0.5 - 0.8` - Medium confidence, acceptable
- `< 0.5` - Low confidence, may need verification

## Alternative: AprilTag

AprilTag markers are similar to ArUco but with:
- Better detection at extreme angles
- More robust to motion blur
- Slightly slower detection

### Using AprilTag

```yaml
scale_anchors:
  - anchor_type: apriltag
    size_meters: 0.05
    family: tag36h11
```

Requires: `pip install apriltag`

## Troubleshooting

### Markers Not Detected

1. **Check lighting** - Avoid harsh shadows or reflections
2. **Verify size** - Marker must be >20 pixels in image
3. **Check print quality** - Crisp black/white, no gray areas
4. **Check white border** - Must have white margin

### Inconsistent Scale

1. **Use multiple markers** - Average reduces noise
2. **Check marker flatness** - Curved surface distorts size
3. **Verify measurement** - Print size matches configured size

### Scale Still Wrong

Try:
1. Manual measurement of known object
2. ARKit poses (for iOS devices)
3. Structure-from-Motion with known camera focal length

## Best Practices

1. **Always include markers** - Even if you have ARKit, markers provide verification
2. **Use standard sizes** - 5cm or 10cm are easy to work with
3. **Print multiple** - Have spares for different scenes
4. **Document placement** - Note where markers are in the capture
5. **Remove after capture** - Markers can be masked out in post-processing
