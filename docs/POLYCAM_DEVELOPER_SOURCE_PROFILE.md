# Polycam Developer Mode Raw Source Profile

Status: local deterministic adapter; provider-derived support only.

`blueprint-adapt-polycam-developer-raw` binds one user-managed Polycam
Space/LiDAR Developer Mode raw ZIP to `polycam_developer_source_profile.v1`.
It performs no Polycam network request and does not extract the archive. It
hashes the original ZIP and every regular member before binding semantic roles.

## Inputs

The command accepts the untouched raw ZIP and a
`polycam_developer_source_declaration.v1` JSON document. The declaration binds
site/capture identity, Polycam capture identity, Apple device identity, provider
app/export identity, provider-declared meter units, and exact member paths for:

- full-resolution RGB frames and/or source video;
- per-frame timestamps;
- camera intrinsics and extrinsics;
- depth and confidence;
- mesh geometry, mesh metadata, and metric-unit metadata; and
- capture, device, and provider identity metadata.

The command also requires `--source-commit-sha` (or
`BLUEPRINT_SOURCE_COMMIT`) so the receipt is bound to an immutable adapter
revision without invoking a shell or discovering repository state implicitly.

One member may support more than one metadata role. For example, one camera JSON
record may carry timestamp, intrinsics, and extrinsics. The adapter does not
guess that meaning from an unverified filename. The declaration makes the
binding explicit and its digest makes replay detectable.

Minimal shape:

```json
{
  "schema_version": "polycam_developer_source_declaration.v1",
  "source_profile": "polycam_developer_mode_lidar_raw_zip",
  "provider_identity": "polycam",
  "source_capture_identity": "site-capture-001",
  "provider_capture_identity": "polycam-capture-001",
  "provider_app_version": "declared-version",
  "provider_export_timestamp": "2026-08-03T06:02:03Z",
  "layout_profile": "observed-layout-v1",
  "capture_mode": "space_lidar",
  "developer_mode_enabled": true,
  "blueprint_remote_upload_performed": false,
  "device_identity": {
    "manufacturer": "Apple",
    "model": "iPhone Pro model",
    "operating_system": "declared iOS version",
    "lidar_capable": true
  },
  "metric_units": {"length_unit": "meter", "scale_to_meters": 1.0},
  "semantic_bindings": {
    "source_rgb_frames": ["keyframes/images/000001.jpg"],
    "source_video": [],
    "frame_timestamps": ["keyframes/cameras/000001.json"],
    "camera_intrinsics": ["keyframes/cameras/000001.json"],
    "camera_extrinsics": ["keyframes/cameras/000001.json"],
    "depth": ["keyframes/depth/000001.png"],
    "confidence": ["keyframes/confidence/000001.png"],
    "mesh_geometry": ["mesh/raw_mesh.glb"],
    "mesh_info": ["mesh/mesh_info.json"],
    "metric_units": ["mesh/mesh_info.json"],
    "capture_identity": ["metadata/capture.json"],
    "device_identity": ["metadata/capture.json"],
    "provider_identity": ["metadata/capture.json"]
  }
}
```

Member paths in this example illustrate a declaration shape; they are not a
promise that every Polycam app/export version uses those names. Record the
observed layout as a versioned declaration and bind its exact paths.

## Safety and abstention

The adapter rejects source-archive symlinks, unsafe/traversing member names,
duplicate members, symlink members, encryption, excessive member/count/total
sizes, and excessive compression ratios. It streams each member for SHA-256
without materializing it on disk.

A safe archive with a missing or unresolved semantic lane produces
`status: abstained`, an ordered blocker list, and `smallest_missing_measurement`.
It does not silently infer a route from incomplete metadata.

## Claim boundary

An admitted profile proves deterministic byte inventory and declared semantic
binding only. It does not establish:

- Blueprint Raw Contract 3.2 truth, encoder-attempt evidence, or retained-frame
  evidence;
- independent metric scale or metric geometry;
- collision/contact validity or Isaac compatibility;
- reconstruction quality, task success, physical success, deployment
  readiness, or policy ranking.

Provider-declared meter units are retained as support and must pass an
independent scale gate before metric, reach, placement, or collision claims.
The original ZIP remains the authoritative artifact for replay of this adapter.
