# Platform Context

<!-- SHARED_PLATFORM_CONTEXT_START -->
## Shared Platform Doctrine

### System Framing

- `BlueprintCapture` is the contributor evidence-capture tool inside Blueprint's three-sided marketplace.
- `BlueprintCapturePipeline` is the authoritative qualification, privacy, provenance, and downstream-routing service.
- `Blueprint-WebApp` is the marketplace and operating system that ingests pipeline outputs and exposes buyer, ops, preview, and hosted-session surfaces.
- `BlueprintValidation` remains optional downstream infrastructure for benchmarking, runtime-backed demos, and deeper robot evaluation after qualification.

This platform is qualification-first.

### Three-Sided Marketplace

- **Capturers** gather evidence packages from real sites.
- **Robot teams** are the primary buyers of trusted qualification outputs, previews, and deeper downstream work.
- **Site operators** control access, consent, rights, and commercialization boundaries for their facilities.

### Truth Hierarchy

- qualification records, readiness decisions, provenance, and rights/compliance outputs are authoritative
- privacy-safe derived media, World Labs previews, scene-memory bundles, and hosted/runtime artifacts are downstream products
- downstream products do not rewrite qualification truth

### Product Stack

1. primary product: qualification record / readiness decision / buyer-safe evidence bundle
2. secondary product: site-specific world models grounded in captured locations for preview, simulation, and hosted demos
3. third product: fallback provider previews, hosted runtime prep, and deeper evaluation packages
4. fourth product: managed tuning, training data, licensing, and deployment support
<!-- SHARED_PLATFORM_CONTEXT_END -->

This repo is the authoritative middle of the product.

## What This Repo Owns

`BlueprintCapturePipeline` turns a finalized capture bundle into:

- qualification artifacts and readiness decisions
- buyer trust, rights/compliance, and recapture outputs
- privacy-safe walkthrough media via on-demand GPU privacy services (SAM3, VIP)
- internal world-model conditioning substrates and optional site-memory / evaluation artifacts
- World Labs request, operation, and world manifests when fallback preview is requested
- optional scene-memory and evaluation/runtime-prep artifacts when those lanes are explicitly requested
- enforced sync back into `Blueprint-WebApp` with a durable `webapp_sync_result.json` artifact

Today, this repo is not only a qualification engine. It is the production bridge from capture evidence through privacy redaction, internal world-model conditioning, optional fallback provider preview, and automatic web app surfacing.

## Upstream Contract

The canonical upstream contract is the raw bundle uploaded by `BlueprintCapture`:

```text
scenes/{scene_id}/captures/{capture_id}/raw/
  manifest.json
  intake_packet.json
  capture_context.json
  capture_upload_complete.json
  task_hypothesis.json
  walkthrough.mov
  motion.jsonl
  arkit/...
```

Compatible triggers the repo accepts today:

- bridge-produced Pub/Sub handoff payloads on `blueprint-capture-pipeline-handoff` (primary path in cloud mode)
- materialized `capture_descriptor.json`
- raw upload completion via `raw/capture_upload_complete.json` (storage trigger, secondary; standdown when bridge handoff is primary)

## Default Runtime Behavior

Product direction:

- the end goal is a site-specific SWM-style world model grounded in Blueprint capture evidence and surfaced through the web app
- World Labs is a fallback / bridge provider for preview generation, not the long-term product truth

Current shipped runtime behavior:

1. materialize the bundle and descriptor
2. run qualification and capture-fidelity analysis (Gemini fidelity review included)
3. call the SAM3 GPU service to detect and remove people; VIP inpaints the removed regions
4. preserve ARKit depth when present, otherwise generate depth conditioning
5. prepare internal world-model conditioning inputs and retrieval/geometry substrates
6. when fallback preview is requested, prepare World Labs-compliant privacy-safe video input and submit/poll World Labs Marble 0.1-mini
7. write preview manifests and sync artifact URIs into the web app (fail-closed, up to 5 retries when sync is required)

Important boundaries:

- `preview_simulation` currently maps most directly to provider-preview behavior, and it is not automatically normalized into `deeper_evaluation` by this repo's lane-routing code
- hosted/runtime launch artifacts come from `scene_memory` and especially `evaluation_prep`
- the internal retrieval-memory / alignment / splat-or-Cosmos synthesis path is the intended native direction, but today it remains opt-in / partial rather than the default production preview path

## Downstream Outputs

The main outputs this repo writes today are:

- qualification summaries and readiness decisions
- buyer trust, capture quality, and rights/compliance summaries
- privacy manifests, verification reports, privacy-safe walkthrough media, and depth manifests
- internal retrieval-memory, geometry, and synthesis-conditioning artifacts when those lanes run
- provider run manifests and preview manifests
- World Labs request / operation / world manifests
- `webapp_sync_result.json` — durable record of the sync attempt, status, and response
- optional `scene_memory/*`
- optional `evaluation_prep/*` including `site_world_spec.json`, `site_world_registration.json`, and `site_world_health.json`

## WebApp Boundary

This repo can push pipeline attachment metadata into `Blueprint-WebApp` through the internal sync endpoint.

That sync is currently:

- authenticated by shared token
- enforced: `PIPELINE_SYNC_REQUIRED=true` means sync failure blocks pipeline completion
- retried up to 5 times with 1000ms backoff before failing

Pipeline completion is gated on successful WebApp attachment sync.

## Operational Reality

What is implemented and live today:

- qualification with Gemini fidelity review
- on-demand GPU privacy redaction via Cloud Run services (`sam3-detect`, `vip-inpaint`, `deepprivacy2-anonymize`) — idle at zero cost, spin up only during pipeline runs
- World Labs Marble 0.1-mini fallback preview generation from privacy-safe video
- automatic public site-world surfacing in the web app on every successful run, usually via attached pipeline/provider artifacts
- optional hosted-runtime prep in deeper lanes (`scene_memory`, `evaluation_prep`)
- retrieval-memory, frame-alignment, and splat/Cosmos validation lanes exist in-repo, but they are opt-in and not the default preview product
- non-ARKit geometry staging exists for glasses / Android style captures via `video_to_world`, but it still depends on live GPU services and is not equivalent to a shipped site-faithful SWM runtime
- iPhone LiDAR remains the strongest and most complete path in-repo; glasses remain internal-experimental for site-faithful world-model claims, Android is contract-supported but not yet covered by a dedicated alpha-readiness profile in this repo
- local raw-bundle materialization is not fully caught up to the newer Capture-side sidecars / modality variants, so bridge-produced descriptors are the more faithful upstream handoff today

What is not guaranteed by the default preview lane:

- native site-specific SWM-style runtime output
- runtime launchability
- `site_world_spec.json`
- `site_world_registration.json`
- `site_world_health.json`
- zero-shot Cosmos execution
- WebApp-hosted internal SWM-style demo output from the retrieval/synthesis lanes

Those belong to `evaluation_prep`, not to preview alone.

## Practical Rule For Agents In This Repo

When changing this repo, optimize for:

1. grounded qualification outputs
2. fail-closed privacy behavior
3. explicit separation between preview generation and hosted-runtime prep
4. durable WebApp handoff records
5. optional downstream lanes that never rewrite qualification truth
