# Same-Capture Lineage Packet

`same_capture_lineage_packet.v1` is the repo-local proof surface that tells a
downstream agent what happened to one `capture_id` across Capture, bridge,
Pipeline, WebApp upstream linkage, and Paperclip issue ownership.

The packet is generated from a local capture root only. It does not prove live
hardware, hosted runtime access, provider execution, payments, payouts, buyer
access, or public launch readiness.

## Local Command

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_same_capture_lineage.py \
  --capture-root /path/to/scenes/<scene_id>/captures/<capture_id> \
  --paperclip-issue-id PC-123
```

Use `--write /path/to/same_capture_lineage_packet.json` to persist the packet.
When omitted, the script prints JSON to stdout and exits non-zero if the chain is
blocked.

## Required Schema

- `schema_version`: `same_capture_lineage_packet.v1`
- `capture_id` and `scene_id`: the same ids across path, raw bundle, bridge
  outputs, Pipeline artifacts, and WebApp sync payload when present
- `raw_bundle`: local `raw/manifest.json`, `raw/capture_context.json`, and
  `raw/capture_upload_complete.json` status
- `bridge_handoff`: `capture_descriptor.json`, `qa_report.json`, and
  `pipeline_handoff.json` status for the same capture
- `pipeline_result`: local Pipeline handoff, qualification summary, completion
  marker, and geometry summary/blockers
- `webapp_upstream_ids`: real `site_submission_id`, `request_id`,
  `buyer_request_id`, and `capture_job_id`; missing or placeholder ids block
  hosted-review and launch claims
- `paperclip_issue`: durable issue id for human/agent follow-up correlation
- `claims`: repo-local claim gates; `launch_claim_allowed` remains `false`
  because launch proof requires live owners outside this packet
- `repo_blockers`: fixable repo-local blockers
- `remaining_hardware_gaps` and `remaining_runtime_gaps`: non-repo evidence that
  must stay separate in closeouts

## Claim Boundaries

- A capture upload alone is not WebApp, hosted-session, provider, payout, buyer,
  or launch proof.
- Missing WebApp upstream ids block hosted-review and launch claims.
- `fallback_geometry` and any fallback-used geometry can be advisory only; it
  cannot satisfy `world_model_ready_claim_allowed`.
- Android XR `android_xr_glasses` / `android_xr_video_only` stays video-only.
  It cannot become public-ready without physical hardware proof, same-capture
  downstream proof, WebApp upstream ids, and a future explicit XR geometry
  contract.
