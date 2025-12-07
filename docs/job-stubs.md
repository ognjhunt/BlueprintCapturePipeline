# GPU job stubs

These stubs define the contract (inputs/outputs/parameters) for each GPU-heavy stage. They intentionally avoid pulling heavy
frameworks so they can be used in orchestration tests and CI before models are wired in.

## How to use

```python
from pathlib import Path
import json
yaml = __import__("yaml")  # keep dependency light

from blueprint_pipeline import build_default_pipeline
from blueprint_pipeline.models import SessionManifest, ScaleAnchor, Clip

with open("configs/example_session.yaml", "r", encoding="utf-8") as f:
    raw = yaml.safe_load(f)

manifest = SessionManifest(
    session_id=raw["session_id"],
    capture_start=raw["capture_start"],
    device=raw["device"],
    scale_anchors=[ScaleAnchor(**anchor) for anchor in raw.get("scale_anchors", [])],
    clips=[Clip(**clip) for clip in raw.get("clips", [])],
    user_notes=raw.get("user_notes"),
)

for payload in build_default_pipeline(manifest):
    print(json.dumps(payload.as_json(), indent=2))
```

Each payload maps cleanly onto a Cloud Run Job invocation with structured parameters for the downstream binary or container.

## Stage summaries

| Stage | Inputs | Outputs | Notes |
| --- | --- | --- | --- |
| frame-extraction | raw clip URIs | sampled frames + SAM 3 masks | configurable target FPS, dynamic mask toggle |
| reconstruction | frames + masks | camera poses, Gaussian splats, reprojection report | enforces scale anchors by default |
| mesh-extraction | Gaussian splats | render mesh, collision mesh, baked textures | toggles for collision mesh + texture baking |
| object-assetization | frames, masks, poses, env mesh | per-object USDs + QA report | reconstruct when coverage is high; Hunyuan3D fallback |
| usd-authoring | env mesh, collision mesh, object USDs | scene USD + authoring report | metersPerUnit + convex decomposition flags |

See [`docs/pipeline-overview.md`](./pipeline-overview.md) for the broader roadmap and capture guidance.
