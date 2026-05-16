# Site Reference Database v1 Contract

Status: Active local contract for SWM-style site grounding.

Owner repo: `BlueprintCapturePipeline`

## Purpose

The Site Reference Database is Blueprint's canonical derived site-memory layer for retrieval-grounded world-model work. It groups privacy-safe reference frames, geometry references, semantic references, visibility cells, route anchors, lookahead support, and validation/readiness summaries under a stable `site_id`.

Raw capture remains authoritative. This contract must not rewrite raw capture, rights, privacy, provenance, or upstream request truth from `BlueprintCapture` or `Blueprint-WebApp`.

This is not a model-backend contract. It does not require SWM, Cosmos, World Labs, Sana-WM, `video_to_world`, Render, Firebase, Stripe, Notion, or live provider calls.

## Audit Matrix

| Artifact | Current path | Owner repo | Authority level | Storage class | Gap |
|---|---|---|---|---|---|
| Raw capture bundle | `scenes/{scene_id}/captures/{capture_id}/raw/` | `BlueprintCapture` | Authoritative raw evidence | Object/file storage | Already authoritative; site-reference records must point back to it and never replace it. |
| Raw manifest, provenance, rights, context | `raw/manifest.json`, `raw/provenance.json`, `raw/rights_consent.json`, `raw/capture_context.json` | `BlueprintCapture` | Authoritative capture, provenance, privacy, and rights source | Object/file storage JSON | Pipeline must preserve missing or unknown rights as blockers/unknowns, not infer clearance. |
| ARKit or modality signals | `raw/arkit/*`, `raw/arcore/*`, `raw/glasses/*`, `raw/motion.jsonl` | `BlueprintCapture` | Authoritative sensor/source observations when present | Object/file storage, JSONL, images, mesh/depth files | Derived indices must keep pose/intrinsics/depth/confidence lineage explicit. |
| Bridge descriptor and QA | `scenes/{scene_id}/captures/{capture_id}/capture_descriptor.json`, `qa_report.json`, `pipeline_handoff.json` | `BlueprintCapture` bridge plus Pipeline materialization | Derived handoff / compatibility | Object/file storage JSON | Useful routing input; not raw truth. |
| Canonical package | `pipeline/site_package/canonical_site_package.json` | `BlueprintCapturePipeline` | Derived product package | Object/file storage JSON | Should reference site-reference outputs when packaged, but not embed dense records. |
| Geometry lane | `pipeline/geometry/**` | `BlueprintCapturePipeline` | Derived geometry conditioning | Object/file storage JSON, JSONL, depth/confidence blobs | Existing contract is derived and fail-closed; site DB consumes only readiness-labeled outputs. |
| Per-capture world-model export | `world_model_export/dense_index.jsonl`, `dense_export_manifest.json`, `dense_pose_alignment.json` | `BlueprintCapturePipeline` | Derived dense export | Object/file storage JSONL and blobs | Existing export needs a canonical site-level contract and WebApp-safe summary projection. |
| Site reference manifest | `sites/{site_id}/reference_memory/site_reference_manifest.json` | `BlueprintCapturePipeline` | Derived site-memory family index | Object/file storage JSON | Now canonicalized by this v1 contract. |
| Site reference index | `sites/{site_id}/reference_memory/site_reference_index.jsonl` | `BlueprintCapturePipeline` | Derived per-reference index | Object/file storage JSONL | May point to dense frame/depth/embedding blobs; must not be copied into Firestore/WebApp documents. |
| Reference thumbnails / embeddings | `sites/{site_id}/reference_memory/thumbnails/`, capture `world_model_export/embeddings/` | `BlueprintCapturePipeline` | Derived blobs | Object/file storage binary/image | Store only as blob URIs in index records, never in WebApp summary docs. |
| Coverage and retrieval indices | `coverage/coverage_map.json`, `indices/*.json`, `site_overlap_graph.json` | `BlueprintCapturePipeline` | Derived support indices | Object/file storage JSON | WebApp may receive counts, scores, and artifact URIs only. |
| Reference selection and future anchors | `synthesis` manifests such as reference-selection and future-anchor regrounding outputs | `BlueprintCapturePipeline` | Derived synthesis support | Object/file storage JSON | Must use real reference records only; no invented future context. |
| Splat and Plucker outputs | Synthesis output directories and maps | `BlueprintCapturePipeline` | Derived geometry/rendering support | Object/file storage image/video/binary | Summary projection may cite family URIs and readiness, not dense maps. |
| Validation/readiness outputs | `retrieval_validation.json`, `evaluation_prep/*readiness*.json`, alpha summaries | `BlueprintCapturePipeline` | Derived readiness support | Object/file storage JSON | Readiness cannot imply provider/live/hosted availability unless owning proof exists. |
| Hosted/package summaries | `server/routes/internal-pipeline.ts`, site-world/session summaries in `Blueprint-WebApp` | `Blueprint-WebApp` | Summary/status projection only | Firestore/WebApp documents | Should store artifact URIs, counts, blockers, scores, and states only. |

## Canonical Storage Layout

```text
sites/{site_id}/reference_memory/
  site_reference_manifest.json              required summary/family index
  site_reference_index.jsonl                required per-reference JSONL index
  site_reference_summary_projection.json    required WebApp/Firestore-safe projection
  retrieval_validation.json                 required readiness/support validation
  coverage/
    coverage_map.json                       derived coverage cells and scores
  indices/
    manifest.json                           index-family summary
    visual_index.json                       embedding URI rows only
    geometry_index.json                     geometry fingerprint and visibility rows
    anchor_inverted_index.json              anchor to reference/chunk ids
    zone_index.json                         zone to reference/chunk ids
  thumbnails/
    {reference_id}.jpg                      blob, URI only in JSON
```

Per-capture dense exports remain under:

```text
scenes/{scene_id}/captures/{capture_id}/world_model_export/
  dense_index.jsonl
  dense_export_manifest.json
  dense_pose_alignment.json
  frames/
  embeddings/
```

Heavy files remain in object/file storage:

- frames and thumbnails
- embedding vectors
- depth and confidence maps
- splat maps and rendered images/videos
- Plucker ray maps
- provider-native model outputs

Firestore/WebApp must not store dense arrays, per-pixel maps, embeddings, binary blobs, or full per-reference record bodies.

## `site_reference_manifest.json`

Required fields:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Must be `site_reference_database.v1`. |
| `site_id` | string | Stable Blueprint site identity. |
| `authority_level` | string | Must be `derived_site_reference_manifest`. |
| `storage_class` | string | Must be `object_storage_manifest`. |
| `raw_capture_authority` | object | States that raw capture/provenance/rights are authoritative upstream. |
| `total_reference_frames` | integer | Count of rows in `site_reference_index.jsonl`. |
| `capture_count` | integer | Count of captures represented in the index. |
| `chunk_count` | integer | Count of route/visibility chunks represented in the index. |
| `captures` | array | Per-capture summary: no dense rows. |
| `coverage_summary` | object | Coverage cell/area rollup. |
| `readiness` | object | Local retrieval-readiness state, blockers, and scores. |
| `artifact_uris` | object | Family-level URIs for manifest, index, validation, coverage, and index-family files. |
| `last_updated` | string | UTC ISO-8601 timestamp. |

`captures[]` entries must include:

- `capture_id`
- `scene_id`
- `captured_at`
- `frame_count`
- `chunk_count`
- `coordinate_frame_session_id`
- `site_frame_aligned`
- `path_length_m`

## `site_reference_index.jsonl`

Each line is one reference record. Required record fields:

| Field | Type | Meaning |
|---|---|---|
| `reference_id` | string | Stable derived reference id. |
| `site_id` | string | Stable site id shared across captures. |
| `scene_id` | string | Raw scene identity. |
| `capture_id` | string | Raw capture identity. |
| `authority_level` | string | Must be `derived_reference_record`. |
| `storage_class` | string | Must be `jsonl_reference_record`. |
| `capture_session_id` | string | Session-level grouping for repeated passes. |
| `coordinate_frame_session_id` | string | Source coordinate-frame segment. |
| `pass_id` | string or null | Route/pass identity when known. |
| `pass_index` | integer or null | Pass order when known. |
| `chunk_id` | string or null | Route chunk id. |
| `chunk_order` | integer or null | Chunk order. |
| `frame_id` | string | Source frame id. |
| `frame_index` | integer | Source frame index. |
| `t_capture_sec` | number | Capture-relative timestamp. |
| `T_world_camera` | 4x4 matrix | Source-session camera pose. |
| `T_site_camera` | 4x4 matrix or null | Site-frame pose after alignment. |
| `intrinsics` | object | Camera intrinsics. |
| `depth_uri` | string or null | Object/file URI for depth blob. |
| `confidence_uri` | string or null | Object/file URI for confidence blob. |
| `embedding_uri` | string or null | Object/file URI for embedding blob. |
| `frame_uri` | string or null | Object/file URI for privacy-safe reference image. |
| `thumbnail_uri` | string or null | Object/file URI for thumbnail. |
| `privacy_source` | string | Source of the reference image. |
| `geometry_source` | string | `arkit`, `video_to_world`, or other explicit derived source. |
| `provenance_lineage` | object | Raw/descriptor/geometry derivation pointers. |
| `privacy_lineage` | object | Privacy-safe source and blockers/status. |
| `rights_lineage` | object | Rights status and source, conservative when unknown. |
| `quality` | object | Tracking, sharpness, pose, and frame-quality support signals. |
| `retrieval_signals` | object | Anchor/staticness/checkpoint/geometry support signals. |
| `visibility_cells` | array | Grid cells visible from this reference. |
| `zone_id` | string or null | Site zone grouping. |
| `anchor_observations` | array | Route/checkpoint anchors observed. |
| `captured_at` | string | Capture timestamp. |
| `indexed_at` | string | Index timestamp. |

Dense values must stay out of JSONL. The index stores URIs and compact metadata only.

## Geometry And Synthesis Outputs

Geometric reference outputs are derived from raw capture and geometry lanes:

- `geometry_fingerprint`
- `visibility_cells`
- `T_site_camera`
- `site_frame_transform`
- depth/confidence artifact URIs
- future splat output URIs
- Plucker map URIs

Splat, Plucker, generated preview, and video outputs must be stored as object/file blobs with URI references. The only summary fields safe for WebApp are readiness state, blockers, counts, validation scores, and family-level artifact URIs.

## Semantic References, Zones, Anchors, And Chunks

Semantic payloads are support records attached to reference IDs, chunks, zones, and anchors. Required grouping concepts:

- `zone_id`: stable area/room/floor grouping when known.
- `anchor_observations`: anchors seen by the frame.
- `chunk_id`: contiguous route segment.
- `visibility_cells`: grid cells visible from the pose/depth record.
- `retrieval_signals`: compact quality and anchor signals.

Semantic labels can help retrieval and buyer review, but they must not claim rights clearance, operational readiness, or real-world facts that are not present in raw or reviewed records.

## Retrieval Query Inputs And Outputs

Retrieval queries must accept:

- `site_id`
- target `T_world_camera` or `T_site_camera`
- target intrinsics
- optional query embedding URI/vector
- retrieval mode: `spatial`, `embedding`, or `hybrid`
- optional anchor, zone, chunk, distance, and quality constraints

Retrieval outputs must include:

- selected `reference_id` values
- selected `frame_id` and `capture_id`
- compact retrieval scores or distances
- blocker/empty-result reasons
- no dense frame, depth, confidence, embedding, splat, or Plucker payloads

## Lookahead / Future-Anchor Outputs

Lookahead support must be built only from real future references already in the site reference index. Required summary fields:

- `future_anchor_context_id`
- `target_frame_id`
- `future_anchor_reference_ids`
- `future_anchor_frame_ids`
- `future_anchor_count`
- `future_anchor_candidates`
- `status`
- `reason`

No generated or predicted future observation may be stored as capture truth.

## Validation And Readiness

`retrieval_validation.json` and summary projections must distinguish:

- local contract readiness
- privacy-safe source readiness
- geometry/source readiness
- retrieval coverage readiness
- site-frame alignment readiness
- live/provider/hosted readiness blockers

Allowed readiness states:

- `ready`
- `degraded`
- `blocked`
- `not_available`

Local Site Reference Database readiness is not Operational Launch Ready. Hosted sessions, provider execution, payment, payout, rights clearance, and buyer access still require proof from the systems that own them.

## WebApp / Firestore Summary Projection

WebApp and Firestore may store only:

- `schema_version`
- `site_id`
- `authority_level`
- `storage_class`
- `artifact_uris`
- `readiness`
- `counts`
- `scores`
- `blockers`
- `last_updated`

The projection must not include:

- per-reference records
- frame pixels or thumbnails
- depth/confidence arrays or maps
- embedding vectors
- splat maps
- Plucker maps
- full camera-pose matrices
- dense `visibility_cells` arrays

Family-level artifact URIs are allowed. Per-record dense blob URIs such as `embedding_uri`, `depth_uri`, `confidence_uri`, `frame_uri`, and `thumbnail_uri` are not allowed in Firestore/WebApp summary documents.

## Local Enforcement

Executable local enforcement lives in:

- `src/blueprint_pipeline/site_reference_database.py`
- `tests/test_site_reference_database_contract.py`

The helper validates required manifest and record fields, builds a WebApp-safe summary projection, and rejects summary payloads that accidentally include dense per-record fields.
