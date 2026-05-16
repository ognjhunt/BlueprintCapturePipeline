# iOS Site-Grounded World Model Capture Spec

## Purpose

This document explains, in one place, what Blueprint is building, how the current iPhone capture path works today, why that is already close to a Seoul World Model style setup, and what minimum contract changes are required to make Blueprint iPhone captures usable for a site-grounded retrieval world model.

This is written as a handoff document for a new agent or engineer. It assumes no prior context beyond access to:

- `/Users/nijelhunt_1/workspace/BlueprintCapture`
- `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline`

The emphasis is iOS only for now.

## Platform Context

Blueprint is not trying to build a generic unconstrained world model that invents arbitrary worlds.

Blueprint is building a capture-first, world-model-product-first system around real sites:

- `BlueprintCapture` captures raw evidence from real facilities.
- `BlueprintCapturePipeline` turns that evidence into site-specific world-model packages, hosted-session artifacts, provider adapter inputs, and optional trust / review outputs.
- `Blueprint-WebApp` is the operational and buyer-facing surface where packages, hosted access, licensing, and support artifacts appear.

The product doctrine is:

- raw capture, rights, privacy, provenance, timestamps, poses, and device metadata are authoritative evidence
- canonical site packages are the primary downstream product contract
- qualification and readiness records are support artifacts for trust, review, pricing, and launch gating
- generated previews, hosted-review surfaces, and provider worlds are downstream projections
- derived outputs must never overwrite capture truth or package truth

That matters because the target system is not "train a cool model from videos." The target system is:

1. capture a real site
2. preserve trustworthy sensor evidence
3. derive a site-grounded memory / world representation
4. support preview, simulation, runtime prep, and eventually stronger interactive site worlds

This is already conceptually aligned with Seoul World Model:

- SWM grounds generation in a real city using a retrieval database, camera trajectory, and geometry
- Blueprint wants to ground world artifacts in a real facility using walkthrough video, poses, depth, and repeated captures

The key difference is:

- SWM is a generative model architecture built on top of Cosmos Predict
- Blueprint today is primarily the capture, packaging, and support-trust pipeline around grounded world generation

## Why This Spec Exists

The current iPhone pipeline already captures much more than plain video:

- RGB walkthrough video
- ARKit poses
- camera intrinsics
- per-frame depth
- per-frame confidence
- mesh exports
- IMU / motion logs

That is already enough to support a prototype site-grounded retrieval world model workflow.

However, the current contract still carries qualification-era naming and preview-generation assumptions. Those records remain compatibility and support artifacts; they were not designed specifically for:

- repeated same-site capture over time
- site-local retrieval databases
- cross-temporal pairing
- lookahead-anchor selection
- training-ready aligned datasets

So the question is not "do we have enough raw signal?" The answer there is mostly yes.

The real question is:

"What minimum contract changes are needed so these captures become a reliable, reusable site-memory substrate for a retrieval-grounded world model?"

This spec answers that question.

## Current iPhone Capture Path

### Capture App Today

The iPhone app currently records:

- `walkthrough.mov`
- `motion.jsonl`
- `arkit/frames.jsonl`
- `arkit/poses.jsonl`
- `arkit/intrinsics.json`
- `arkit/depth/*.png`
- `arkit/confidence/*.png`
- `arkit/meshes/*.obj`
- `manifest.json`

Important code references:

- ARSession recorder is used to capture video plus poses, depth, and intrinsics in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/VideoCaptureManager.swift`
- ARKit artifact directories are created in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/VideoCaptureManager.swift:744`
- per-frame ARKit logging is written in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/VideoCaptureManager.swift:960`
- pose rows include `frame_id`, `t_device_sec`, and `T_world_camera` in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/VideoCaptureManager.swift:1034`
- motion logging is written in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/VideoCaptureManager.swift:923`

### Finalized Raw Bundle Today

The finalizer patches and emits:

- `manifest.json`
- `intake_packet.json`
- `capture_context.json`
- `capture_upload_complete.json`
- `task_hypothesis.json`

Important code references:

- evidence inspection and sensor availability are computed in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/Services/CaptureBundleSupport.swift:222`
- world-model-related manifest fields are written in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/Services/CaptureBundleSupport.swift:580`
- final raw metadata and context files are written in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/Services/CaptureBundleSupport.swift:689`

### Cloud Bridge Today

The bridge:

- waits for upload completion
- extracts JPEG frames from the walkthrough at 5 fps
- loads ARKit poses
- matches frames to poses by frame ID or nearest timestamp
- computes pose alignment quality
- emits `capture_descriptor.json`, `qa_report.json`, and `pipeline_handoff.json`

Important code references:

- normalized scene-memory fields are read from raw manifest in `/Users/nijelhunt_1/workspace/BlueprintCapture/cloud/extract-frames/src/index.ts:468`
- actual sensor availability is computed in `/Users/nijelhunt_1/workspace/BlueprintCapture/cloud/extract-frames/src/index.ts:667`
- frame-to-pose alignment is performed in `/Users/nijelhunt_1/workspace/BlueprintCapture/cloud/extract-frames/src/index.ts:732`
- iPhone quality gating uses pose match rate and p95 pose delta in `/Users/nijelhunt_1/workspace/BlueprintCapture/cloud/extract-frames/src/bridge.ts:241`

### Pipeline Today

The pipeline preserves:

- raw video URI
- ARKit pose/intrinsics/depth/confidence URIs
- ARKit frames URI
- motion log URI
- world-model fitness summaries
- scene-memory bundles
- presentation-world bundles
- evaluation-prep `site_world_spec.json`

Important code references:

- raw bundle materialization into descriptor in `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/materialization.py:660`
- current local scene-memory candidate inference in `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/materialization.py:767`
- scene-memory readiness logic in `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/qualification.py:831`
- evaluation-prep canonical `site_world_spec.json` generation in `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/evaluation_prep_stage.py`

## Why This Is Already Close To An SWM-Like Setup

An SWM-style site-grounded retrieval model needs four things:

1. a place-specific visual memory
2. camera poses / trajectory information
3. depth or geometry
4. enough repeated coverage to retrieve nearby views during rollout

Blueprint iPhone captures already provide:

- place-specific visual content: walkthrough video and ARKit keyframes
- poses: `arkit/poses.jsonl`
- geometry cues: depth, confidence, meshes, and downstream geometry lanes
- motion-aligned timing: `t_device_sec` and `motion.jsonl`

So the raw foundation is not missing.

What is missing is the structure that turns those signals into a reusable site-memory system.

## Current Gaps

### 1. No Stable Site Identity For Open iPhone Captures

Problem:

- if there is no target or reservation, `jobId` falls back to a new UUID in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/CaptureFlowViewModel.swift:441`
- scene identity is then derived from target, reservation, or jobId in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/Services/CaptureBundleSupport.swift:267`

Why this matters:

- repeated captures of the same facility can become separate unrelated scenes
- cross-temporal pairing becomes weak
- a same-site retrieval database becomes fragmented
- "open capture" becomes much less useful for building site memory

Minimum change:

- add a stable `site_id` contract that survives recaptures and does not depend on one upload job

### 2. No Structured Site Location / Site Frame Metadata

Problem:

- the app has no explicit raw-bundle fields for structured `place_id`, address components, lat/lng, floor, room, or local coordinate origin
- `captureContextHint` is only free text in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/Services/CaptureUploadService.swift:201`

Why this matters:

- retrieval wants structured grouping and filtering, not a text hint
- multiple captures in one building need floor/zone separation
- same-site matching should not rely on string similarity
- local route graphs and revisit anchors need a common site frame

Minimum change:

- add structured site identity and site-location fields to raw manifest and capture context

### 3. `world_model_candidate` Is Unset By Default On iOS

Problem:

- iOS capture metadata initializes `continuityScore` as `nil` in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/CaptureFlowViewModel.swift:486`
- `world_model_candidate` is only true if continuity score exists and is `>= 0.5` in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/Services/CaptureBundleSupport.swift:317`
- the cloud bridge trusts the raw manifest field and blocks runtime build eligibility when it is false in `/Users/nijelhunt_1/workspace/BlueprintCapture/cloud/extract-frames/src/index.ts:918`

Why this matters:

- production path says "not world-model-ready" even when raw ARKit evidence is strong
- iOS app and cloud bridge disagree on what "candidate" means

Minimum change:

- stop using a nullable operator-entered continuity score as the sole source of truth for world-model candidacy
- compute candidate eligibility deterministically from evidence and capture mode

### 4. Local And Cloud Behavior Disagree

Problem:

- local materialization infers `world_model_candidate` from evidence tier in `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/src/blueprint_pipeline/materialization.py:767`
- production cloud bridge trusts raw manifest `scene_memory_capture.world_model_candidate` in `/Users/nijelhunt_1/workspace/BlueprintCapture/cloud/extract-frames/src/index.ts:468`

Why this matters:

- local testing can appear healthier than production
- engineers and agents get false confidence about readiness

Minimum change:

- define one canonical eligibility rule and use it in both bridge and local pipeline code

### 5. ARKit Is Still Best-Effort Instead Of Required For The World-Model Lane

Problem:

- the app can disable ARKit on the next recording after a camera conflict in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/VideoCaptureManager.swift:315` and `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/VideoCaptureManager.swift:640`

Why this matters:

- SWM-style conditioning depends on reliable pose and geometry
- "graceful fallback" is fine for qualification-only uploads
- it is not fine for a site-grounded retrieval world model lane

Minimum change:

- add a capture mode where ARKit pose/intrinsics/depth are hard requirements
- if they fail, downgrade requested outputs to qualification-only

### 6. On-Device Coverage Signals Are Too Weak

Problem:

- the current coverage estimate is only a rough mesh-anchor heuristic in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/Services/CaptureQualityMonitor.swift:204`

Why this matters:

- mesh-anchor count is not enough to know if important viewpoints were seen
- retrieval-grounded generation needs coverage of turns, thresholds, side views, and revisit points

Minimum change:

- compute and persist better coverage signals tied to route completion and anchor observations

### 7. No Explicit Revisit / Cross-Temporal Structure

Problem:

- iPhone uploads include a text `coveragePlan`, but no `pass_id`, revisit anchors, loop closures, or completed checkpoints by default in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/CaptureFlowViewModel.swift:471`

Why this matters:

- SWM-style cross-temporal pairing needs repeated views of the same place at different times
- current captures are mostly single-pass recordings with weak revisit structure

Minimum change:

- add explicit capture-session, route, pass, and anchor contracts

### 8. Bridge Outputs Are QA-Oriented, Not Training-Oriented

Problem:

- the bridge extracts frames at only 5 fps in `/Users/nijelhunt_1/workspace/BlueprintCapture/cloud/extract-frames/src/index.ts:682`

Why this matters:

- 5 fps is fine for QA and preview surfaces
- it is not enough for a denser aligned training export or high-quality retrieval index

Minimum change:

- keep the QA index, but add a second dense aligned export for world-model work

### 9. Per-Frame AR Quality Is Not Rich Enough

Problem:

- current AR frame rows contain transform, intrinsics, resolution, and paths for depth/confidence in `/Users/nijelhunt_1/workspace/BlueprintCapture/BlueprintCapture/VideoCaptureManager.swift:1012`
- they do not appear to persist:
  - tracking state
  - relocalization / limited tracking reasons
  - world-mapping confidence
  - frame sharpness / blur score
  - exposure quality summary

Why this matters:

- world-model training and retrieval indexing need frame filtering
- low-trust frames should not seed the retrieval database

Minimum change:

- extend `arkit/frames.jsonl` with frame-quality and tracking-health fields

### 10. No Site Retrieval Index Or Future-Anchor Index Exists Yet

Problem:

- current system preserves raw evidence, but does not build a privacy-safe site-local retrieval memory

Why this matters:

- SWM works because generation can retrieve nearby views and future anchors
- Blueprint currently has the ingredients, but not the retrieval layer

Minimum change:

- materialize a privacy-safe site reference index from captures after upload

## Minimum Contract Changes

This section defines the minimum viable additions, not the final ideal system.

### A. Add Stable Site Identity Fields

Add the following required fields to the raw contract for iPhone world-model-eligible captures:

```json
{
  "site_identity": {
    "site_id": "stable blueprint site id",
    "site_id_source": "buyer_request|site_submission|place_match|manual_review",
    "place_id": "provider place id or null",
    "site_name": "string or null",
    "address_full": "string or null",
    "address_structured": {
      "street": "string or null",
      "city": "string or null",
      "region": "string or null",
      "postal_code": "string or null",
      "country": "string or null"
    },
    "geo": {
      "latitude": 0.0,
      "longitude": 0.0,
      "accuracy_m": 0.0
    },
    "building_id": "string or null",
    "floor_id": "string or null",
    "room_id": "string or null",
    "zone_id": "string or null"
  }
}
```

Rules:

- `site_id` must survive recapture and not depend on one upload UUID
- `site_id` must be identical for repeated captures of the same facility scope
- `floor_id`, `room_id`, and `zone_id` are nullable but should be present when known

### B. Add Capture Session / Route / Pass Identity

Add the following required fields for world-model-eligible captures:

```json
{
  "capture_topology": {
    "capture_session_id": "uuid",
    "route_id": "uuid",
    "pass_id": "uuid",
    "pass_index": 1,
    "capture_mode": "qualification_only|site_world_candidate",
    "intended_pass_role": "primary|revisit|loop_closure|critical_zone_revisit",
    "entry_anchor_id": "anchor_001",
    "return_anchor_id": "anchor_001"
  }
}
```

Rules:

- multiple passes in one site visit share `capture_session_id`
- distinct traversals share `route_id` if they represent the same intended path
- each recording gets its own `pass_id`

### C. Add Structured Anchor And Checkpoint Metadata

Replace purely textual `coveragePlan` semantics with structured anchor/checkpoint records:

```json
{
  "route_anchors": [
    {
      "anchor_id": "anchor_entry",
      "anchor_type": "entry|junction|threshold|dock_turn|handoff_point|restricted_boundary|critical_zone",
      "label": "Main loading dock threshold",
      "expected_observation": "pause_and_pan",
      "required_in_primary_pass": true,
      "required_in_revisit_pass": false
    }
  ],
  "checkpoint_events": [
    {
      "anchor_id": "anchor_entry",
      "pass_id": "uuid",
      "t_capture_sec": 12.4,
      "completed": true
    }
  ]
}
```

Why:

- this makes cross-temporal pairing and retrieval grouping possible
- it also supports loop-closure scoring

### D. Add A Dedicated `site_world_candidate` Capture Mode

Add a new capture mode in app metadata and raw manifest:

```json
{
  "capture_mode": {
    "requested_mode": "qualification_only|site_world_candidate",
    "resolved_mode": "qualification_only|site_world_candidate",
    "downgrade_reason": null
  }
}
```

Rules:

- `site_world_candidate` requires:
  - ARKit world tracking enabled
  - valid intrinsics
  - valid pose log
  - usable depth coverage
- if requirements fail, app or bridge must downgrade to `qualification_only`
- downgraded captures must not claim world-model candidacy

### E. Make `world_model_candidate` Deterministic

Do not set this from a nullable manual continuity score.

Replace it with a computed field derived from:

- `capture_mode.resolved_mode == "site_world_candidate"`
- valid ARKit pose log
- valid intrinsics
- minimum depth coverage
- minimum pose-to-video alignment quality
- minimum intake completeness
- rights allow derived scene generation

Canonical rule:

```text
world_model_candidate =
  capture_mode == site_world_candidate
  AND arkit_poses_valid
  AND arkit_intrinsics_valid
  AND depth_coverage_ok
  AND pose_alignment_ok
  AND intake_complete
  AND derived_scene_generation_allowed
```

This rule must be shared between:

- iOS finalizer
- cloud bridge
- local materialization
- pipeline readiness logic

### F. Extend `arkit/frames.jsonl`

Add the following per-frame fields:

```json
{
  "frameIndex": 0,
  "frame_id": "000001",
  "timestamp": 123.456,
  "t_device_sec": 0.033,
  "cameraTransform": [...],
  "intrinsics": [...],
  "imageResolution": [1920, 1440],
  "sceneDepthFile": "arkit/depth/000001.png",
  "confidenceFile": "arkit/confidence/000001.png",
  "tracking_state": "normal|limited|not_available",
  "tracking_reason": "initializing|excessive_motion|insufficient_features|relocalizing|null",
  "world_mapping_status": "not_available|limited|extending|mapped|null",
  "exposure_duration_s": 0.0,
  "iso": 0.0,
  "sharpness_score": 0.0,
  "blur_score": 0.0,
  "relocalization_event": false,
  "anchor_observations": ["anchor_entry"]
}
```

This is the minimum needed for:

- filtering unstable frames
- building retrieval indexes from high-trust frames
- constructing revisit pairs

### G. Add A Dense World-Model Export Lane

Keep the current 5 fps QA extraction.

Add a second export family for world-model work:

```text
scenes/{scene_id}/captures/{capture_id}/world_model_export/
  dense_frames/
  dense_index.jsonl
  dense_pose_alignment.json
  keyframes/
  retrieval_candidates.json
```

Requirements:

- denser than 5 fps
- each exported frame includes pose, depth pointer, confidence pointer, and frame-quality fields
- retrieval candidates should be privacy-safe and tagged with anchor / zone / pass metadata

### H. Add A Privacy-Safe Site Retrieval Index

This is the minimum new downstream artifact needed to support an SWM-like approach:

```text
sites/{site_id}/reference_memory/
  site_reference_index.json
  embeddings/
  thumbnails/
```

Minimum record shape:

```json
{
  "reference_id": "uuid",
  "site_id": "stable site id",
  "capture_id": "capture id",
  "pass_id": "pass id",
  "frame_id": "000123",
  "t_capture_sec": 45.2,
  "zone_id": "dock_a",
  "anchor_ids": ["anchor_threshold_1"],
  "T_world_camera": [[...]],
  "intrinsics": {"fx": 0, "fy": 0, "cx": 0, "cy": 0},
  "depth_uri": "gs://...",
  "confidence_uri": "gs://...",
  "embedding_uri": "gs://...",
  "privacy_safe_image_uri": "gs://...",
  "quality": {
    "tracking_state": "normal",
    "sharpness_score": 0.91,
    "blur_score": 0.04
  }
}
```

This is the artifact that will eventually enable:

- nearest-view retrieval
- same-site cross-temporal pairing
- future-anchor selection for lookahead

## Minimum iOS UX / Workflow Changes

These are required because contract changes alone will not create the right data.

### Required New Capture Prompts For `site_world_candidate`

The app should guide the operator through a stricter protocol:

1. Start at entry anchor and hold still for 2-3 seconds.
2. Do a left-right look sweep.
3. Walk primary route steadily.
4. Pause and pan at each required anchor.
5. Revisit critical zones from the reverse direction or a second angle.
6. Return to the entry anchor for loop closure when feasible.

This creates:

- repeated observations
- anchor-aligned revisits
- better retrieval coverage
- better pose graph consistency

### Required On-Device Checks

Before allowing `site_world_candidate` upload, app should validate:

- ARKit tracking was mostly normal
- pose log exists
- intrinsics exist
- depth coverage exceeds threshold
- minimum anchor coverage achieved
- at least one revisit or loop closure occurred

If not, the app should either:

- ask for recapture immediately
- or downgrade to `qualification_only`

## Canonical Eligibility Rules

For iPhone captures, the minimum rules should be:

### Qualification-Only Eligible

- walkthrough video exists
- manifest exists
- intake minimally complete or review-intake path is requested

### Site-World-Candidate Eligible

- all qualification requirements
- valid ARKit pose log
- valid intrinsics
- minimum depth coverage
- acceptable pose alignment quality
- structured `site_id`
- structured capture topology
- anchor coverage meets threshold
- rights allow derived scene generation

## Recommended File / Schema Additions

Add to raw bundle:

```text
raw/
  manifest.json
  capture_context.json
  route_anchors.json
  checkpoint_events.json
  site_identity.json
```

Add to descriptor:

- `site_identity`
- `capture_topology`
- `capture_mode`
- `anchor_coverage_summary`
- `loop_closure_summary`
- `world_model_candidate_reasoning`

## Rollout Plan

### Phase 1: Contract Alignment

- define stable `site_id`
- unify `world_model_candidate` logic
- add capture mode and topology fields
- extend `frames.jsonl`

### Phase 2: iOS Workflow Enforcement

- add `site_world_candidate` capture mode
- add anchor prompts and revisit prompts
- add downgrade logic when ARKit evidence is weak

### Phase 3: Retrieval Artifact Materialization

- build privacy-safe site reference index
- export denser aligned frame datasets
- support recapture linkage under the same `site_id`

### Phase 4: Model-Facing Integration

- use site reference index for retrieval
- support cross-temporal pairing from multiple passes and recaptures
- support lookahead-anchor retrieval along route graphs

## Final Bottom Line

Minimum conclusion:

- Blueprint iPhone capture already has the raw sensor foundation for a site-grounded retrieval world model.
- The current blockers are mostly contract and workflow structure, not lack of core data.
- The minimum viable changes are:
  - stable `site_id`
  - structured site and route metadata
  - deterministic world-model candidacy
  - required ARKit mode for the site-world lane
  - explicit revisit / anchor structure
  - richer per-frame quality logs
  - privacy-safe site retrieval index
  - dense aligned export separate from QA frames

If those changes are made, Blueprint iPhone captures become a credible substrate for an indoor SWM-like approach rather than only a support/trust walkthrough pipeline.
