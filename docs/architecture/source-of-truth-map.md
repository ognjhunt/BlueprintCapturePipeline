# Source Of Truth Map

This repo has multiple useful outputs. They are not equal sources of authority.
Use this map before changing package, runtime, provider, WebApp sync, or launch
gate behavior.

## Truth Hierarchy

| Layer | Examples | Authority |
| --- | --- | --- |
| Raw capture truth | `raw/manifest.json`, `walkthrough.mov`, ARKit/ARCore poses, depth, confidence, device metadata, timestamps, rights/consent metadata | Authoritative evidence of what was captured |
| Materialized descriptor truth | `capture_descriptor.json`, `qa_report.json` | Normalized bridge view of raw truth and local QA |
| Canonical package truth | `pipeline/site_package/canonical_site_package.json` | Product package contract grounded in raw/descriptor evidence |
| Support/trust artifacts | `qualification_summary.json`, `buyer_trust_score.json`, `rights_and_compliance_summary.json`, `world_model_fit_summary.json` | Review, pricing, trust, and readiness support |
| Provider adapter truth | `pipeline/site_package/provider_adapter_inputs/*.json`, `worldlabs_request_manifest.json` | Provider-specific projection from canonical package |
| Provider output truth | `provider_run_manifest.json`, `preview_manifest.json`, `worldlabs_operation_manifest.json`, `worldlabs_world_manifest.json` | Generated preview/run status only |
| Hosted-review/runtime projection | `pipeline/evaluation_prep/*`, native runtime session state, WebApp hosted-review state | Delivery/runtime projection, gated by live config and proof |
| Gate snapshots | `output/*launch_gate*`, local gate markdown/json | Point-in-time evidence snapshot, not standing authority |

## Raw Capture Truth Vs Derived Package Truth

Raw capture truth answers:

- what media and sensor files were uploaded
- what timestamps, poses, intrinsics, depth, confidence, motion logs, and metadata exist
- what rights, privacy, consent, provenance, and capture restrictions were declared
- what the device and capture modality actually provided

Derived package truth answers:

- which capture-grounded inputs are package-ready
- which privacy-safe media and geometry artifacts are available
- which package fields are missing or blocked
- which provider adapters can be built from the canonical package
- which generated/derived artifacts exist for review or hosted access

The package must not invent raw truth. If a field is missing from raw capture or
the descriptor, the package should expose missing fields or blockers instead of
smoothing them over.

## Canonical Package Vs Provider Preview Vs Hosted Review

The canonical package is the provider-agnostic product contract:

```text
pipeline/site_package/canonical_site_package.json
```

Provider adapter inputs are derived from the canonical package:

```text
pipeline/site_package/provider_adapter_inputs/world_labs_marble.json
```

Provider preview manifests are generated-provider state:

```text
pipeline/worldlabs_request_manifest.json
pipeline/provider_run_manifest.json
pipeline/preview_manifest.json
pipeline/worldlabs_operation_manifest.json
pipeline/worldlabs_world_manifest.json
```

Hosted review and runtime records are delivery projections:

```text
pipeline/evaluation_prep/site_world_spec.json
pipeline/evaluation_prep/site_world_registration.json
pipeline/evaluation_prep/site_world_health.json
pipeline/webapp_sync_result.json
```

Do not treat provider output or hosted-review projection as more authoritative
than the canonical package. Do not treat the canonical package as more
authoritative than raw capture evidence.

## Fallback Geometry Labels And Blockers

Geometry summary is the first file to read:

```text
pipeline/geometry/geometry_summary.json
```

Live site-faithful geometry requires:

```text
geometry_source=video_to_world
fallback_used=false
provider_native_result=true
ready_for_world_model=true
geometry_live_ready=true
site_frame_available=true
scale_resolved=true
```

`local_sfm` geometry is allowed for offline/degraded reference-media indexing
when stable site identity and rights/privacy lineage exist. It must keep
`provider_native_result=false`, `ready_for_world_model=false`, and
`geometry_live_ready=false`.

Fallback geometry is allowed for local development and contract-shape debugging,
but it must remain blocked for retrieval indexing, alpha readiness, launchable
export packaging, buyer runtime launch, and site-faithful claims.

Important fallback labels include:

```text
geometry_source=fallback_geometry
fallback_used=true
fallback_kind=internal_synthetic_geometry
fallback_kind=local_da3_synthetic_depth
geometry_live_ready=false
site_faithful_market_ready=false
```

If a downstream output says it is ready while geometry still has fallback labels,
the downstream output is suspect until the code path is inspected.

## Operator, Toolchain, Manual, And Live Evidence Boundaries

Automated local tests can prove contract shape and fail-closed behavior. They
cannot prove live operations by themselves.

Operator evidence means a human or device action happened and was recorded, for
example real-device discovery, reservation, upload completion, or authenticated
buyer artifact access.

Toolchain evidence means required local or CI tooling is available, for example
Android SDK paths, Python packages, ffmpeg, or ML model packages.

Manual evidence means a review or decision record exists, for example a finance
review owner, KYC provider decision, or background-check provider decision.

Live evidence means an external service actually executed, for example Stripe
live payment/payout, World Labs API generation, privacy GPU inference, live
`video_to_world`, or WebApp buyer-access verification.

Do not convert mocked Stripe tests, missing SDK checks, fallback geometry, raw
World Labs bypass, or automated contracts into live launch proof.

## `output/` Artifacts Are Snapshots

Files under `output/` are point-in-time evidence snapshots from local commands.
They are useful for review, but they are not the source of authority.

Before using an `output/` file as evidence:

- rerun the command that generated it, or say that it may be stale
- compare `git status --short --branch` and `git diff --stat`
- inspect whether the command mutated `output/` files
- trace any launch claim back to current code, raw artifacts, gate output, and
  live/manual evidence when required
