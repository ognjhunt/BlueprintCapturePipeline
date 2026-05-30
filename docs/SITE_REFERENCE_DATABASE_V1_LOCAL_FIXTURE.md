# Site Reference Database v1 Local Fixture

This fixture proves the local Site Reference Database v1 path from a staged capture bundle to the WebApp-safe summary projection.

It is intentionally local-only:

- no World Labs, Cosmos, SWM, video_to_world, storage, Firebase, or WebApp calls
- local `local_sfm` geometry only, marked degraded for non-ARKit readiness
- deterministic local embeddings, not a model checkpoint
- dense frame, depth, confidence, embedding, pose, and geometry fields stay in object-storage artifacts
- WebApp receives only `site_reference_summary_projection.json`

## Run

```bash
python scripts/build_site_reference_database_fixture.py \
  --output-root output/site-reference-database-v1-fixture \
  --json-output output/site-reference-database-v1-fixture/summary.json
```

The command rebuilds:

```text
output/site-reference-database-v1-fixture/site_reference_database_v1_fixture/
  source_bundle/raw/
  storage/local-blueprint-fixture/scenes/site-reference-fixture-scene/captures/site-reference-fixture-capture/
  storage/local-blueprint-fixture/sites/site-reference-fixture-site/reference_memory/
```

## Proof Artifacts

The generated reference-memory directory contains:

- `site_reference_index.jsonl`
- `site_reference_manifest.json`
- `retrieval_validation.json`
- `site_reference_summary_projection.json`
- `coverage/coverage_map.json`
- `indices/manifest.json`
- `site_overlap_graph.json`

`site_reference_index.jsonl` is allowed to contain dense object-storage URIs such as `depth_uri`, `confidence_uri`, `embedding_uri`, and `frame_uri`.

`site_reference_summary_projection.json` is the WebApp-safe projection. It must not contain dense per-record fields such as `T_world_camera`, `intrinsics`, `visibility_cells`, `geometry_fingerprint`, `depth_uri`, `confidence_uri`, `embedding_uri`, `frame_uri`, or `thumbnail_uri`.

## Expected Readiness

The fixture should report:

- local contract ready
- summary projection safe
- retrieval query ready
- non-ARKit geometry degraded
- SWM/world-model readiness blocked
- operational live provider and hosted-session readiness blocked

That distinction is the point of the fixture: it proves local contract continuity without turning local degraded geometry into live provider proof.
