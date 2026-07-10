# Site Reference Grounding Audit — 2026-05-16

This audit covers the local/offline per-site SWM-style grounding path only. It does not claim live provider, hosted-session, payment, payout, rights clearance, privacy clearance, customer traction, or operational launch success.

| Area | Current state | Evidence | Remaining blocker |
| --- | --- | --- | --- |
| ARKit path | ARKit/iPhone captures keep the existing pose/intrinsics/depth path. Stable `site_id` is still required before `site_world_candidate`; missing identity downgrades with `missing_site_id`. | `$HOME/workspace/BlueprintCapture/BlueprintCapture/Services/CaptureBundleSupport.swift`; `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/materialization.py`; `$HOME/workspace/BlueprintCapturePipeline/tests/test_world_model_candidate_parity.py` | Existing captures without upstream identity need review/backfill packets, not generated IDs. |
| Meta/raw video evidence | Meta glasses and other non-ARKit video captures are preserved as raw evidence with stable `site_id`, session/route/pass identity, source device, original video URI, media metadata, frame timestamp URI, stream metadata URI, privacy lineage, and rights/provenance lineage. They are not promoted from metadata alone. | `$HOME/workspace/BlueprintCapture/cloud/extract-frames/src/index.ts`; `$HOME/workspace/BlueprintCapture/cloud/extract-frames/src/index.test.ts`; `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/materialization.py`; `$HOME/workspace/BlueprintCapturePipeline/tests/test_world_model_candidate_parity.py` | Raw video without stable identity or derived-rights lineage remains review/blocked. |
| Local geometry diagnostics | `provider=local_sfm` currently writes explicitly synthetic diagnostics with `geometry_source=fallback_geometry`, `fallback_used=true`, `provider_native_result=false`, `contract_ready_for_world_model=false`, `ready_for_world_model=false`, and blockers such as `synthetic_geometry_not_capture_truth`, `provider_native_geometry_missing`, `scale_not_proven`, and `site_frame_not_proven`. Retrieval must reject this path. | `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/geometry_stage.py`; `$HOME/workspace/BlueprintCapturePipeline/scripts/run_geometry_lane.py`; `$HOME/workspace/BlueprintCapturePipeline/tests/test_geometry_stage.py`; `$HOME/workspace/BlueprintCapturePipeline/tests/test_retrieval_index_geometry_source.py` | A real local SfM runner still needs capture-derived pose/intrinsics/depth proof before any non-ARKit reference indexing. |
| Provider-native geometry | `provider=video_to_world` is gated by `VIDEO_TO_WORLD_URL` and `VIDEO_TO_WORLD_RUNNER_TOKEN`. Missing env writes provider blocker fields and does not call a live provider. Only non-fallback `video_to_world` proof can set `provider_native_result=true`, `geometry_live_ready=true`, and `ready_for_world_model=true`. | `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/geometry_stage.py`; `$HOME/workspace/BlueprintCapturePipeline/tests/test_geometry_stage.py` | Live provider credentials/backend remain required; no live provider call was made in this session. |
| Site Reference DB readiness | Site reference DB v1 manifest/index/summary projection can ingest ARKit captures and non-ARKit captures only when reference-indexable geometry is provider-native `video_to_world` proof or another future capture-derived geometry source with explicit truth labels. Validation separately reports raw evidence, retrieval readiness, non-ARKit geometry state, and SWM-style world-model readiness. | `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/retrieval_index_stage.py`; `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/site_reference_database.py`; `$HOME/workspace/BlueprintCapturePipeline/tests/test_retrieval_index_geometry_source.py`; `$HOME/workspace/BlueprintCapturePipeline/tests/test_site_reference_database_contract.py` | SWM-style readiness remains blocked for synthetic fallback geometry and any non-provider-native rows. |
| Backfill/reporting | Backfill discovers Meta/raw video captures with stable site identity and raw evidence but no geometry proof, reports `status=geometry_required`, blocker `non_arkit_geometry_missing`, the exact expected `geometry_summary.json` path, the local geometry command, and provider env blockers. | `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/site_reference_backfill.py`; `$HOME/workspace/BlueprintCapturePipeline/scripts/backfill_site_reference_database.py`; `$HOME/workspace/BlueprintCapturePipeline/tests/test_site_reference_backfill.py` | Backfill does not invent site IDs for old captures. |
| Alignment readiness | Alignment patches `site_frame_transform` and `T_site_camera` only from overlap/anchor/geometry evidence, updates validation, and refreshes the WebApp-safe summary projection. Weak overlap stays degraded. | `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/frame_alignment_stage.py`; `$HOME/workspace/BlueprintCapturePipeline/tests/test_frame_alignment_stage.py` | Scale or overlap gaps remain degraded until supported by evidence. |
| Runtime adapter readiness | Native runtime consumes Site Reference DB v1 and reports `non_arkit_geometry_state`. Provider-native rows can promote the geometry gate while model backend readiness stays separate; synthetic diagnostics must remain blocked. | `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/native_runtime_backend.py`; `$HOME/workspace/BlueprintCapturePipeline/tests/test_native_runtime_service.py` | Cosmos/SWM-like model backend still requires real local packages/model/checkpoint or provider runtime. |
| Hosted/live/provider readiness | Operational/live/provider/hosted readiness remains explicitly blocked in local validation. This work does not claim hosted-session success, live SWM provider success, payment, payout, rights clearance, privacy clearance, customer traction, or operational launch success. | `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/retrieval_index_stage.py`; `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/native_runtime_backend.py` | Needs live provider/runtime/hosted proof from the owning systems. |
| WebApp projection | No WebApp code change was required. The Pipeline summary projection remains WebApp/Firestore-safe and excludes dense frames, depth, confidence, embeddings, Plucker maps, splats, trajectories, and full reference rows. | `$HOME/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/site_reference_database.py`; `$HOME/workspace/BlueprintCapturePipeline/tests/test_site_reference_database_contract.py` | None for summary shape; WebApp should consume only the summary/status/URI projection. |

## Local Commands

Dry-run review/backfill report:

```bash
blueprint-backfill-site-reference-db /path/to/local/storage --report-path /path/to/backfill_report.json
```

Execute eligible captures:

```bash
blueprint-backfill-site-reference-db /path/to/local/storage --execute --report-path /path/to/backfill_report.json
```

Run local synthetic geometry diagnostics for a Meta/raw-video capture:

```bash
python3 scripts/run_geometry_lane.py --capture-root /path/to/capture --provider local_sfm --model local-sfm-offline
```

Run provider-native proof only when env and credentials are present:

```bash
VIDEO_TO_WORLD_URL=<provider-url> VIDEO_TO_WORLD_RUNNER_TOKEN=<token> \
python3 scripts/run_geometry_lane.py --capture-root /path/to/capture --provider video_to_world --model video_to_world-default
```
