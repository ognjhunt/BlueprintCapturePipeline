# Cross-Repo Readiness Matrix

This matrix is intentionally strict. A row is only `ready` when the capability is implemented in-repo, on the canonical contract, and does not still depend on unproven external runtime access.

## Status Rules

- `ready`: implemented on the canonical contract with local evidence in these repos
- `partial`: substantial implementation exists, but external runtime, deployment, or production-faithfulness proof is still missing
- `blocked`: the repos do not currently provide this capability as a truthful shipped path

## Blocker Class Rules

Use the same blocker classes as the paid-marketplace gate when converting this
matrix into closeout work:

- `repo`: failing or missing in-repo contract implementation
- `toolchain`: missing local/CI SDK, package, binary, or runner needed to verify
- `human/operator`: missing decision, owner, review, or authenticated operator action
- `live-provider`: missing non-payment provider execution or provider account proof
- `hardware`: missing real-device capture, upload, or hardware proof
- `payment`: missing Stripe/payment/payout/finance-provider proof

## Matrix

| Surface | Status | What is true in repo | Blocking gap |
| --- | --- | --- | --- |
| `BlueprintCapture` | `partial` | Produces raw bundles, capture context, task/intake sidecars, and cloud bridge handoff for iPhone, glasses, and Android. Android is now normalized to `android` instead of `android_phone`. | It does not produce SWM-style output or runtime site worlds itself. |
| `BlueprintCapturePipeline` | `partial` | Runs qualification, privacy preparation, geometry staging, retrieval memory, presentation/evaluation-prep packaging, and WebApp sync. | Live `video_to_world`, live Cosmos, and production runtime deployment still require real GPU services and credentials. |
| `Isaac/G1 kitchen-parity render seed` | `partial` | Local CPU dry-render, task-aware `scene_placement`, bundle namelist checks, dirty-tree paid-launch guard, and focused unit tests exist for the G1 kitchen/fridge/faucet review lane. Render-seed proof boundary: CPU/hermetic only; no live GPU frame was produced on 2026-06-29. | Live Isaac GPU frame, provider closure, artifact upload proof, teardown/spend proof, and review-quality media are still missing before this lane can be called ready. |
| `scene_placement` | `partial` | Pure placement package, USD/perception indexes, multi-target diagnostics, openable standoff hints, geometric validation, and CPU tests are implemented without hardcoded scene coordinates. | It remains placement support evidence. It does not prove physical manipulation, real robot readiness, safety, or live render quality by itself. |
| `warm-serve render transport` | `partial` | Warm-serve control loop and signed warm-inbox scaffolding are implemented and hermetically tested. | Live multi-job reuse after one real Isaac scene load is still unproven; provider runtime, output, teardown, and spend evidence are required. |
| `provider/spend safety` | `partial` | Provider race, render lock, spend guard, bundle manifest, and dirty-tree paid-launch guard exist as CPU-validated safety scaffolding. | Safety scaffolding is not provider execution proof; paid runs still require clean git evidence, live provider closure, cost-control, upload, and teardown artifacts. |
| `Blueprint-WebApp` | `partial` | Ingests qualification and site-world artifacts and can display/runtime-launch them when truthful artifacts exist. | It is a consumer and operating layer, not the producer of SWM-style synthesis or canonical site-world generation. |
| `iPhone` | `partial` | Canonical `iphone` capture source, ARKit-backed `iphone_arkit_lidar`, and the strongest path into metric evidence and site-world candidacy exist. | Repo evidence still does not prove production captures remain site-faithful through zero-shot Cosmos or downstream synthesis. |
| `glasses` | `partial` | Canonical `glasses` source, `glasses_video_only`, and scaffolded-video promotion into validated metric evidence are implemented. This is not a public Google/Meta hardware promise. | Public Google/Meta smart-glasses support still requires an approved repeat-walk assignment, hardware proof, launch proof, downstream capture/package proof, and live video-to-world/runtime availability. |
| `Android` | `partial` | Canonical `android` source is now aligned across producer and pipeline. Android now shares the same video-only and scaffolded-video readiness semantics as glasses. | Still depends on live video-to-world/runtime services; no in-repo proof of production site-faithful synthesis or public Google/Meta smart-glasses support. |
| `Android XR video-only` | `partial` | Pipeline preserves `capture_profile_id=android_xr_glasses` and `capture_modality=android_xr_video_only` as glasses/video provenance, strips geometry/provider/hosted/payout readiness claims, resolves default downstream lanes to qualification only, and skips retrieval geometry with `android_xr_video_only_requires_explicit_geometry_contract`. | Physical Android XR hardware proof, bridge handoff proof, same-capture Pipeline proof, WebApp upstream ids, and a new explicit XR geometry profile are still required before any pose/depth/geospatial/world-model/provider/payout/hosted/launch readiness claim. |
| `qualification` | `ready` | Qualification artifacts, readiness decisions, provenance/trust outputs, and WebApp sync are implemented and covered by local tests. | None inside these repos. |
| `site-world packaging` | `ready` | `evaluation_prep/site_world_spec.json`, `site_world_registration.json`, and `site_world_health.json` are produced and validated in local tests. | None for packaging itself. Live runtime registration remains a separate deployment concern. |
| `SWM-style synthesis` | `partial` | Retrieval memory, geometry conditioning, Plucker-ray conditioning hooks, splat-first synthesis, and Cosmos adapter surfaces exist. | The repos do not yet prove end-to-end SWM-style, site-faithful output on production captures. |
| `zero-shot Cosmos` | `blocked` | Adapter code exists for Cosmos Image2World invocation, and the default Hugging Face model id is now corrected to `nvidia/Cosmos-Predict2.5-2B`. | Installed runtime, actual inference execution, and site-faithfulness proof on production captures are still missing. |
| `fine-tuned SWM` | `blocked` | Training-aligned design notes and adapter placeholders exist. | No fine-tuned model, no training pipeline, and no shipped inference/runtime path exist in these repos. |

## Contract Notes

Android is no longer a separate contract class from glasses. The canonical source values are:

- `iphone`
- `glasses`
- `android`

For non-ARKit video capture, the readiness classes are now shared:

- video-only capture stays `pre_screen_video`
- scaffolded video with validated scale and pose coverage promotes to `video_with_validated_scaffolding`

Android XR projected glasses are the current exception to generic non-ARKit promotion. The current
contract is `capture_profile_id=android_xr_glasses` plus
`capture_modality=android_xr_video_only`; it remains video-first even when requested outputs ask for
scene memory or preview simulation. A future XR geometry lane must use a new explicit
profile/modality and bridge sidecar contract rather than reusing `android_xr_video_only`.

The remaining Android-versus-glasses differences are producer provenance and eventual hardware quality, not downstream contract shape.
