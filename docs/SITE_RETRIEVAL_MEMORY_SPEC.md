# Site Retrieval Memory Spec

## What To Tackle Next And Why

Phase 1 fixed the contract. The raw bundle now has stable `site_id`, structured topology, deterministic `world_model_candidate`, and rich per-frame quality data. That was the foundation work.

The right next thing to build is the **site retrieval memory** — the layer that turns Blueprint captures into a queryable, pose-aware site knowledge base.

This is the right next step because:

- It uses **existing qualifying captures** — no iOS changes required to start
- It produces a **tangible artifact** that buyers and ops can see and reason about
- It is **purely pipeline-side** — no App Store review cycles
- It **directly enables retrieval-grounded generation experiments** with existing models/providers
- Phase 2 iOS UX improvements (anchor prompts, ARKit enforcement) will improve the quality of what goes into this index but do not block its construction

The site retrieval memory is the product. Everything else — generation conditioning, route-conditioned previews, cross-temporal pairing — is downstream of having a good retrieval index.

---

## What This Is Not

This spec does not cover:

- Generation conditioning (Plücker ray embeddings, depth splatting, model integration) — those are Phase 4
- Cross-session coordinate frame alignment — addressed as a design consideration below; full implementation is a follow-on
- Phase 2 iOS UX (anchor prompts, ARKit enforcement) — runs in parallel, feeds future index quality

---

## The Coordinate Frame Problem

Before the design, this problem must be understood clearly because it shapes the index schema.

Every ARKit session creates its own world coordinate frame. `T_world_camera` from capture A is NOT comparable to `T_world_camera` from capture B even if both captured the same warehouse. Each session's world origin is wherever the device was when tracking initialized.

**Consequence:** Spatial nearest-neighbor search across captures from different sessions does not work with raw ARKit poses. You cannot ask "which frame from any capture shows the dock threshold nearest to this query pose" without first aligning the sessions.

**How this spec handles it:**

Within a single capture session, ARKit poses are fully consistent and spatial lookup works correctly.

For the Phase 3A index, each reference record carries a `coordinate_frame_session_id` — the ARKit session UUID. Records with different `coordinate_frame_session_id` values are in different unaligned frames. The index also stores an optional `site_frame_transform` per record: when populated, it transforms that record's pose from its session frame into a canonical site frame. Initially null; populated by the alignment stage described below.

This design is correct and future-proof: the index starts with per-session frames and upgrades to a common site frame when alignment runs. Retrieval implementations must check `site_frame_transform` and use site-frame coordinates when available, per-session coordinates otherwise.

**Phase 3B (alignment, not in this spec) will add:**

- An entry anchor localization stage that computes the relative transform between sessions using the shared entry anchor viewpoint (Phase 2 UX provides the anchor hold data)
- Alternatively, a visual place recognition stage using DINOv2 embeddings to match overlapping frames across sessions and compute relative transforms via essential matrix or PnP
- A pipeline stage that patches `site_frame_transform` for all existing index records for a site after each new session is aligned

Do not let the alignment problem block Phase 3A. A per-session spatial index over a single good capture is still useful for:

- Retrieval within the trajectory of that capture
- Visual similarity search across any capture (embedding similarity is coordinate-frame independent)
- Coverage visualization of individual captures
- Retrieval-grounded generation for the single-capture case

---

## What We Are Building

### Per-Capture Dense Export

For each `world_model_candidate == true` capture, the pipeline materializes:

```
scenes/{scene_id}/captures/{capture_id}/world_model_export/
  dense_index.jsonl           ← one record per exported frame, with pose + quality + URIs
  dense_pose_alignment.json   ← summary of pose alignment quality for this export
  frames/
    {frame_id}.jpg            ← full-resolution privacy-safe frames
  embeddings/
    {frame_id}.bin            ← float32 DINOv2 embedding vector
```

### Site Retrieval Index

For each site with at least one qualifying capture, the pipeline materializes and maintains:

```
sites/{site_id}/reference_memory/
  site_reference_manifest.json  ← metadata, capture count, last updated, coverage summary
  site_reference_index.jsonl    ← append-only; one record per reference frame across all captures
  coverage/
    coverage_map.json           ← 2D occupancy grid in site/session coordinate space
  thumbnails/
    {reference_id}.jpg          ← 256px privacy-safe thumbnails for inspection
```

---

## Dense Frame Export Stage

### When It Runs

After `world_model_candidate == true` is confirmed for a capture, after the privacy processing stage has completed.

Gating conditions:

- `descriptor.world_model_candidate == true`
- Privacy-processed video exists (or raw video if privacy skipped for this capture)
- `arkit/frames.jsonl` exists with per-frame quality fields (Phase 1 addition)
- `arkit/poses.jsonl` exists with valid pose rows

### Frame Selection Policy

Naive fixed-rate extraction (e.g., 10fps) produces redundant frames when the camera is stationary and too-sparse frames during fast movement. Use a **distance-gated** extraction policy:

```
min_travel_distance_m = 0.07   # ~7cm per frame
max_frame_gap_sec = 0.5        # always include at least every 0.5 sec regardless
min_travel_fallback_fps = 2    # fallback if pose gap is too large to compute distance
```

Algorithm:
1. Load `arkit/poses.jsonl` into a time-sorted list
2. Load `arkit/frames.jsonl` into a dict keyed by `frame_id` for quality lookup
3. For each pose in the sequence:
   - Compute Euclidean distance from the last exported pose (translation component of `T_world_camera`)
   - Include this frame if: (a) distance >= `min_travel_distance_m`, or (b) time since last included frame >= `max_frame_gap_sec`
4. For each selected frame, apply quality gate:
   - `tracking_state == "normal"` (from Phase 1 frames.jsonl extension)
   - `sharpness_score >= 40.0` (Laplacian variance; empirically calibrated — start here, tune)
   - `world_mapping_status` in `{"mapped", "extending"}` (prefer; do not hard-gate on this)
   - Skip frames where `relocalization_event == true`
5. For consecutive selected frames with < 2cm travel (camera stationary during pan), keep every 4th frame only to avoid embedding duplicates

This yields approximately 8-15 frames per second of walkthrough at normal walking speed.

### Frame Extraction

The bridge already uses FFmpeg for extraction. The dense export stage uses the same pattern but operating on the privacy-processed video (preferred) or raw video with a lightweight person-presence filter.

```python
ffmpeg_args = [
    "-hide_banner", "-loglevel", "error", "-y",
    "-i", privacy_processed_video_path,
    "-vf", f"select=eq(n\\,{frame_number})",    # extract specific frame by index
    "-vframes", "1",
    "-q:v", "1",                                 # high quality JPEG (scale 1-31, 1 = best)
    output_path,
]
```

Extract at native video resolution. Do not downscale — downstream embedding and generation need full resolution.

### Privacy Handling

Two tiers:

**Tier 1 (preferred):** Extract from the privacy-processed video. The pipeline already runs SAM3 + VIP on the walkthrough video. Extract dense frames from that output video. This gives zero additional privacy cost and guarantees people-free frames. Limitation: if the privacy video is lower quality due to VIP inpainting artifacts, frames near inpainted regions will have visual noise. This is acceptable for a retrieval substrate.

**Tier 2 (fallback):** If the privacy-processed video is not yet available when the dense export stage runs, use the raw video with a depth-based person-presence flag:

- For each candidate frame, check if the paired depth map contains a large near-foreground region (any connected region > 200px² with depth < 1.5m that is not the floor plane)
- Mark such frames `privacy_filtered: true` and exclude them from the retrieval index
- Include them in `dense_index.jsonl` as `included_in_index: false` for audit

Do not include any raw (non-privacy-processed) frames in the retrieval index or in embedding generation when Tier 2 is used.

### Dense Index Record Schema

Each row of `dense_index.jsonl`:

```json
{
  "frame_id": "000247",
  "pass_id": "uuid",
  "capture_id": "uuid",
  "scene_id": "string",
  "site_id": "string",
  "coordinate_frame_session_id": "arkit-session-uuid",
  "t_capture_sec": 8.23,
  "T_world_camera": [[r00, r01, r02, tx], [r10, r11, r12, ty], [r20, r21, r22, tz], [0, 0, 0, 1]],
  "intrinsics": {"fx": 1462.3, "fy": 1462.3, "cx": 960.0, "cy": 720.0, "width": 1920, "height": 1440},
  "depth_uri": "gs://bucket/scenes/.../raw/arkit/depth/000247.png",
  "confidence_uri": "gs://bucket/scenes/.../raw/arkit/confidence/000247.png",
  "frame_uri": "gs://bucket/scenes/.../world_model_export/frames/000247.jpg",
  "embedding_uri": "gs://bucket/scenes/.../world_model_export/embeddings/000247.bin",
  "privacy_source": "privacy_processed_video",
  "included_in_index": true,
  "quality": {
    "tracking_state": "normal",
    "world_mapping_status": "mapped",
    "sharpness_score": 142.7,
    "relocalization_event": false,
    "travel_from_prev_m": 0.082
  },
  "anchor_observations": [],
  "zone_id": null
}
```

### Pose Alignment Summary

`dense_pose_alignment.json`:

```json
{
  "schema_version": "v1",
  "capture_id": "uuid",
  "total_frames_extracted": 1247,
  "frames_included_in_index": 1089,
  "frames_excluded_quality": 98,
  "frames_excluded_privacy": 60,
  "pose_match_rate": 0.98,
  "p95_pose_gap_m": 0.11,
  "total_path_length_m": 94.3,
  "session_duration_sec": 312.4,
  "coordinate_frame_session_id": "arkit-session-uuid",
  "site_frame_transform": null,
  "generated_at": "2026-03-17T..."
}
```

---

## SWM Stack Alignment

Before embedding: understand what SWM (arXiv:2603.15583) actually uses for retrieval, because Blueprint should adopt the same primitives where possible.

**SWM retrieval is pose-based, not embedding-based.** SWM retrieves K=5 reference frames during training and K=1 during inference by nearest viewpoint distance (spatial NN), then warps them into the target viewpoint via depth-based forward splatting:

```
Render(Unproj(x_ref, d_ref), c_ref→target)
```

Camera poses are encoded as **6-channel Plücker ray maps** from extrinsics + intrinsics, projected into latent space via a convolutional encoder and added as residuals to both video tokens and reference tokens. SWM does not use DINO or CLIP for retrieval.

**Blueprint's spatial retrieval path (T_world_camera nearest neighbors, coverage map, depth URIs) is the exact SWM-equivalent approach.** The `T_world_camera` + `intrinsics` + `depth_uri` fields on every reference record are what enable this.

**DINOv3 embeddings serve a Blueprint-specific need SWM didn't have:** cross-session visual similarity retrieval before ARKit coordinate frame alignment runs. SWM worked on pre-aligned street-view imagery across the same city; Blueprint captures have per-session ARKit world frames that aren't aligned until Phase 3B. Until `site_frame_transform` is populated, embedding-based NN is the only cross-session retrieval available.

**Phase 4 generation conditioning** will adopt the SWM generation stack:
- Plücker ray maps (6-channel) from per-reference `T_world_camera` + `intrinsics`
- Depth-based forward splatting of reference frames to target viewpoint using `depth_uri`
- K=5 references as conditioning during training, K=1 at inference
- Fine-tuned Cosmos-Predict2.5-2B on Blueprint captures

---

## Embedding Generation

### Model

**DINOv3 ViT-L/16** (1024-dimensional CLS token embeddings).

DINOv3 (February 2026, arXiv:2508.10104, Meta Research) supersedes DINOv2. Trained on 1.7 billion images (vs. 142M for DINOv2) with Gram anchoring — a new training objective that prevents feature degradation. +6 mIoU on ADE20K semantic segmentation over DINOv2.

Rationale over DINOv2:
- Significantly better dense feature quality — critical for geometric scene matching
- Larger embedding dimension (1024 vs 768) captures more discriminative spatial structure
- Gram anchoring improves feature consistency across viewpoints
- Publicly available via HuggingFace: `facebook/dinov3-vitl16-pretrain-lvd1689m`

Do not use CLIP. CLIP is optimized for cross-modal image-text alignment; it is weaker than DINO-family for geometric scene matching without text queries.

### Preprocessing

DINOv3 uses `AutoImageProcessor` from HuggingFace Transformers — no manual resize/normalize needed:

```python
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")

inputs = processor(images=pil_images, return_tensors="pt")
outputs = model(**inputs)
# CLS token per image:
embeddings = outputs.last_hidden_state[:, 0, :]  # [N, 1024]
```

### Embedding Storage

Each embedding stored as a flat float32 binary file: 1024 × 4 bytes = 4096 bytes per frame.

```python
embedding = outputs.last_hidden_state[0, 0].cpu().numpy().astype(np.float32)
embedding.tofile(output_path)
```

Load:

```python
embedding = np.fromfile(path, dtype=np.float32)   # shape [1024]
```

### Batch Processing

Process in batches of 32 on GPU. For a 5-minute capture at 12fps → ~3600 frames, batch processing takes ~18 seconds on a T4. Run inline in the pipeline stage.

---

## Site Retrieval Index Stage

### Site Reference Index Schema

Each row of `site_reference_index.jsonl` is a reference record. The index is **append-only**: new captures append new records; records are never deleted or modified except when `site_frame_transform` is patched by the alignment stage.

```json
{
  "reference_id": "uuid",
  "site_id": "stable-site-id",
  "capture_id": "uuid",
  "scene_id": "string",
  "pass_id": "uuid",
  "pass_index": 1,
  "capture_session_id": "uuid",
  "coordinate_frame_session_id": "arkit-session-uuid",
  "site_frame_transform": null,
  "frame_id": "000247",
  "t_capture_sec": 8.23,
  "T_world_camera": [[...]],
  "intrinsics": {"fx": 1462.3, "fy": 1462.3, "cx": 960.0, "cy": 720.0},
  "depth_uri": "gs://...",
  "confidence_uri": "gs://...",
  "embedding_uri": "gs://...",
  "thumbnail_uri": "gs://...",
  "quality": {
    "tracking_state": "normal",
    "sharpness_score": 142.7,
    "world_mapping_status": "mapped"
  },
  "anchor_observations": [],
  "zone_id": null,
  "captured_at": "2026-03-17T...",
  "indexed_at": "2026-03-17T..."
}
```

### Site Reference Manifest

`site_reference_manifest.json` is rewritten (not appended) each time the index is updated:

```json
{
  "schema_version": "v1",
  "site_id": "stable-site-id",
  "total_reference_frames": 4821,
  "capture_count": 3,
  "captures": [
    {
      "capture_id": "uuid",
      "scene_id": "string",
      "captured_at": "2026-03-17T...",
      "frame_count": 1089,
      "coordinate_frame_session_id": "arkit-session-uuid",
      "site_frame_aligned": false,
      "path_length_m": 94.3
    }
  ],
  "coverage_summary": {
    "covered_area_m2": 312.4,
    "cells_total": 1249,
    "cells_with_coverage": 812,
    "cells_with_dense_coverage": 631,
    "coverage_fraction": 0.65
  },
  "last_updated": "2026-03-17T...",
  "site_frame_established": false
}
```

### Coverage Map

The coverage map operates in the coordinate frame of whichever session is treated as the reference frame (initially: the first qualifying capture for the site).

Grid parameters:
- Cell size: 0.5m × 0.5m
- Plane: XZ (ARKit Y is up)
- Stored as a sparse dict: only cells with nonzero count are stored

```json
{
  "schema_version": "v1",
  "site_id": "stable-site-id",
  "coordinate_frame_session_id": "arkit-session-uuid-of-reference-session",
  "cell_size_m": 0.5,
  "origin_x": -12.4,
  "origin_z": -8.1,
  "grid_width": 62,
  "grid_depth": 41,
  "cells": {
    "14,22": {"frame_count": 12, "capture_ids": ["uuid1"], "mean_sharpness": 138.2},
    "15,22": {"frame_count": 9, "capture_ids": ["uuid1"], "mean_sharpness": 142.7}
  },
  "coverage_summary": {
    "covered_area_m2": 203.0,
    "dense_area_m2": 157.0,
    "dense_threshold_frames_per_cell": 5
  }
}
```

### Thumbnails

For each reference frame, write a 256-wide JPEG thumbnail to `thumbnails/{reference_id}.jpg`. Used by ops/buyer surfaces to inspect what a retrieval candidate looks like without loading the full-resolution frame.

---

## Pipeline Stage Implementation

### New File: `retrieval_index_stage.py`

Location: `src/blueprint_pipeline/retrieval_index_stage.py`

```python
def run_retrieval_index_stage(
    *,
    capture_root: str | Path,
    force_rebuild: bool = False,
    embedding_model: Optional[Any] = None,  # inject for testing; loads DINOv2 if None
) -> Dict[str, Any]:
    """
    For a world_model_candidate capture:
    1. Extract dense frames at distance-gated intervals
    2. Filter by frame quality
    3. Generate DINOv2 embeddings
    4. Write per-capture world_model_export/
    5. Append to site-level reference memory index
    6. Recompute coverage map
    Returns stage result with status, frame counts, and output URIs.
    """
```

Internal structure:

```
retrieval_index_stage.py
  run_retrieval_index_stage()            ← main entry
  _extract_dense_frames()                ← distance-gated extraction
  _filter_frames_by_quality()            ← tracking state, sharpness, privacy
  _generate_embeddings()                 ← DINOv2 batch inference
  _write_dense_export()                  ← writes world_model_export/
  _append_to_site_reference_index()      ← reads existing index, appends new records
  _update_coverage_map()                 ← recomputes coverage cells
  _write_site_manifest()                 ← overwrites manifest with updated counts
```

### Integration: New Lane

Add `retrieval_index` as a new lane in `capture_orchestrator.py`:

```python
_SUPPORTED_LANES = {
    "qualification", "scene_memory", "evaluation_prep",
    "retrieval_index",   # ← NEW
    "all"
}
```

Trigger condition (in orchestrator):

```python
if "retrieval_index" in selected_lanes:
    descriptor = load_descriptor(capture_root)
    if descriptor.get("quality", {}).get("world_model_candidate"):
        retrieval_result = run_retrieval_index_stage(
            capture_root=capture_root,
            force_rebuild=parse_bool(os.getenv("RETRIEVAL_INDEX_FORCE_REBUILD"), False),
        )
        results.append({"lane": "retrieval_index", **retrieval_result})
    else:
        results.append({
            "lane": "retrieval_index",
            "status": "skipped",
            "reason": "world_model_candidate=false",
        })
```

Add `retrieval_index` to the `all` lane expansion so it runs automatically for qualifying captures.

### Idempotency

Check for `world_model_export/dense_index.jsonl` existence. If present and `force_rebuild == False`, skip extraction and embedding and go directly to index append (in case the site index was lost or needs rebuilding from existing per-capture exports).

Check for existing records in `site_reference_index.jsonl` with `capture_id == this_capture_id`. If present and `force_rebuild == False`, skip the append entirely.

### Dependencies

```
pip install torch torchvision    # DINOv2
pip install numpy                # already present
# No new GCS client needed — existing google-cloud-storage client is used
```

DINOv2 model download: `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")` caches to `~/.cache/torch/hub/`. In Cloud Run, use a warmed image or pre-download in the Dockerfile:

```dockerfile
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"
```

---

## What Phase 2 iOS Provides (Running In Parallel)

These iOS changes are NOT blockers for Phase 3A but directly improve the retrieval index:

| iOS Phase 2 Addition | Benefit To Index |
|---|---|
| Entry anchor hold at capture start | Enables session-to-session coordinate alignment (Phase 3B) |
| ARKit hard-requirement enforcement | Eliminates low-quality captures from entering the index |
| `route_anchors.json` / `checkpoint_events.json` | Populates `anchor_observations` in reference records |
| On-device downgrade to `qualification_only` | Stops low-ARKit captures from claiming `site_world_candidate` |

The index schema already has `anchor_observations` and `zone_id` fields. They start as null/empty and get populated as Phase 2 UX ships.

---

## What This Unblocks

Once the site retrieval index exists:

### Retrieval-Grounded Generation (Phase 4A)

Given a target camera pose in site coordinates:
1. Query the site index for the K nearest reference frames by spatial distance (within-session) or embedding similarity (cross-session before alignment)
2. Pass the K retrieved frames (with their poses, depth maps, and intrinsics) to a generation model as conditioning
3. Generate a short video segment from the target pose

For the zero-FT path, the generation model can be:
- **Lightweight warp/blend:** Use depth-based forward splatting to reproject the nearest reference frame into the target viewpoint. No model needed. Produces correct geometry, visual quality depends on coverage density. This is the right v1 product — fast, trustworthy, no model cost.
- **Provider-conditioned:** Pass retrieved views to World Labs or another provider as additional conditioning context
- **SWM-style:** Full retrieval-conditioned diffusion model (longer term)

### Coverage-Based Recapture Guidance

The coverage map immediately enables: "this site has 65% coverage — aisle 3 and the dock approach are dark zones." Surfaceable to ops and buyers in the web app.

### Cross-Temporal Comparison

With multiple captures of the same site, the index enables: "compare this zone across 3 captures" — useful for change detection, recapture validation, and showing buyers that the site is actively maintained.

### Site World-Model Readiness Score

A simple aggregate from the coverage map: `covered_area_m2 / estimated_total_navigable_area_m2`. Can be surfaced in the marketplace as a readiness signal for buyers.

---

## Storage Structure Summary

```
gs://bucket/
  sites/
    {site_id}/
      reference_memory/
        site_reference_manifest.json
        site_reference_index.jsonl
        coverage/
          coverage_map.json
        thumbnails/
          {reference_id}.jpg

  scenes/
    {scene_id}/
      captures/
        {capture_id}/
          world_model_export/
            dense_index.jsonl
            dense_pose_alignment.json
            frames/
              {frame_id}.jpg
            embeddings/
              {frame_id}.bin
```

---

## Critical Code References

| Location | Purpose |
|---|---|
| `src/blueprint_pipeline/retrieval_index_stage.py` | New stage (to create) |
| `src/blueprint_pipeline/capture_orchestrator.py` | Add `retrieval_index` lane |
| `src/blueprint_pipeline/materialization.py` | `world_model_candidate` already canonical after Phase 1 |
| `src/blueprint_pipeline/qualification.py` | Reads `world_model_candidate` — no changes needed |
| `cloud/extract-frames/src/index.ts` | Existing 5fps extraction pattern to reference for FFmpeg args |
| `BlueprintCapture/BlueprintCapture/VideoCaptureManager.swift:61` | `ARFrameLogEntry` fields (tracking_state, sharpness_score) used for frame quality filtering |

---

## Verification

1. **Single capture:** Run the stage on a known `world_model_candidate == true` capture. Confirm `world_model_export/dense_index.jsonl` is written with ~8-15 records per second of walkthrough. Confirm embeddings are 768-dimensional float32. Confirm `site_reference_index.jsonl` exists at `sites/{site_id}/reference_memory/`.

2. **Quality filtering:** Check that frames with `tracking_state == "limited"` in `arkit/frames.jsonl` are excluded. Check that frames with `sharpness_score < 40` are excluded.

3. **Retrieval smoke test:** Load the index, compute a query embedding from a test image of the same site, do nearest-neighbor search. Confirm the top-1 result is visually similar to the query.

4. **Idempotency:** Run the stage twice on the same capture. Confirm the site index has the same number of records (no duplicates).

5. **Coverage map:** Load `coverage_map.json`, visualize the 2D grid. Confirm it roughly outlines the walkthrough path.

6. **Multi-capture append:** Run the stage on a second capture of the same `site_id`. Confirm `site_reference_index.jsonl` grows by the second capture's frame count. Confirm `site_reference_manifest.json` shows `capture_count: 2`. Confirm `coordinate_frame_session_id` differs between the two capture batches and `site_frame_aligned: false`.

---

## Phasing

### Phase 3A (this spec — build now)

- Dense frame extraction with distance gating
- Frame quality filtering using Phase 1 `arkit/frames.jsonl` fields
- DINOv2 embedding generation
- Per-capture `world_model_export/` materialization
- Site-level `reference_memory/` index (append-only, per-session coordinates)
- Coverage map computation
- `retrieval_index` lane in orchestrator
- Thumbnail generation

### Phase 3B (after Phase 2 iOS ships entry anchor hold)

- Entry anchor visual localization: extract embedding at anchor hold moment, match across sessions to compute relative transform
- `site_frame_transform` patch stage: update existing index records with aligned poses
- True spatial nearest-neighbor lookup across all sessions of a site
- Site frame establishment and `site_frame_established: true` in manifest

### Phase 3C (route graph)

- Materialize a route graph from anchor observations and captured poses
- Nodes: defined anchors from `route_anchors.json`
- Edges: annotated with coverage density and mean sharpness along path
- Enables: "can we generate from node A to node B?" with a coverage confidence score
- Enables: operator-facing recapture plan ("cover edge 3→4, it has < 3 frames/m²")
