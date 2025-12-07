# BlueprintCapturePipeline

This repository tracks the design of a Cloud Run–based pipeline that turns Meta smart glasses captures (via the BlueprintCapture iOS app) into two outputs:

1. **Perception Twin:** Dense, photorealistic reconstruction (Gaussians + mesh) for rendering and visual QA.
2. **Sim Twin:** Object-centric USD assets with clean colliders and physics materials for robotics simulation (e.g., Isaac Sim).

See [`docs/pipeline-overview.md`](docs/pipeline-overview.md) for the proposed architecture, capture requirements, processing stages (WildGS-SLAM, SuGaR, SAM 3, Hunyuan3D), and QA harness.

## GPU job stubs

The `src/blueprint_pipeline` package contains lightweight stubs for each Cloud Run GPU job. They do not execute ML models yet; instead they define the input/output contract and default parameters for orchestration:

* `FrameExtractionJob` — decode clips, sample frames, and run SAM 3 masking.
* `ReconstructionJob` — WildGS-SLAM with dynamic masking and scale anchors.
* `MeshExtractionJob` — SuGaR mesh + optional collision mesh and texture baking.
* `ObjectAssetizationJob` — SAM 3 lifting + object-centric 3DGS or Hunyuan3D fallback.
* `USDAuthoringJob` — package environment and objects into USD with physics metadata.

Use `build_default_pipeline(session_manifest)` to generate `JobPayload` JSON payloads for Cloud Run Jobs. An example manifest is provided at [`configs/example_session.yaml`](configs/example_session.yaml).
