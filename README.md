# BlueprintCapturePipeline

This repository tracks the design of a Cloud Run–based pipeline that turns Meta smart glasses captures (via the BlueprintCapture iOS app) into two outputs:

1. **Perception Twin:** Dense, photorealistic reconstruction (Gaussians + mesh) for rendering and visual QA.
2. **Sim Twin:** Object-centric USD assets with clean colliders and physics materials for robotics simulation (e.g., Isaac Sim).

See [`docs/pipeline-overview.md`](docs/pipeline-overview.md) for the proposed architecture, capture requirements, processing stages (WildGS-SLAM, SuGaR, SAM 3, Hunyuan3D), and QA harness.
