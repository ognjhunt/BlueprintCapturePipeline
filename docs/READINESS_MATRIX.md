# Cross-Repo Readiness Matrix

This matrix is intentionally strict. A row is only `ready` when the capability is implemented in-repo, on the canonical contract, and does not still depend on unproven external runtime access.

## Status Rules

- `ready`: implemented on the canonical contract with local evidence in these repos
- `partial`: substantial implementation exists, but external runtime, deployment, or production-faithfulness proof is still missing
- `blocked`: the repos do not currently provide this capability as a truthful shipped path

## Matrix

| Surface | Status | What is true in repo | Blocking gap |
| --- | --- | --- | --- |
| `BlueprintCapture` | `partial` | Produces raw bundles, capture context, task/intake sidecars, and cloud bridge handoff for iPhone, glasses, and Android. Android is now normalized to `android` instead of `android_phone`. | It does not produce SWM-style output or runtime site worlds itself. |
| `BlueprintCapturePipeline` | `partial` | Runs qualification, privacy preparation, geometry staging, retrieval memory, presentation/evaluation-prep packaging, and WebApp sync. | Live `video_to_world`, live Cosmos, and production runtime deployment still require real GPU services and credentials. |
| `Blueprint-WebApp` | `partial` | Ingests qualification and site-world artifacts and can display/runtime-launch them when truthful artifacts exist. | It is a consumer and operating layer, not the producer of SWM-style synthesis or canonical site-world generation. |
| `iPhone` | `partial` | Canonical `iphone` capture source, ARKit-backed `iphone_arkit_lidar`, and the strongest path into metric evidence and site-world candidacy exist. | Repo evidence still does not prove production captures remain site-faithful through zero-shot Cosmos or downstream synthesis. |
| `glasses` | `partial` | Canonical `glasses` source, `glasses_video_only`, and scaffolded-video promotion into validated metric evidence are implemented. | Still depends on video-to-world/runtime availability for live downstream world-model execution. |
| `Android` | `partial` | Canonical `android` source is now aligned across producer and pipeline. Android now shares the same video-only and scaffolded-video readiness semantics as glasses. | Still depends on live video-to-world/runtime services; no in-repo proof of production site-faithful synthesis. |
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

The remaining Android-versus-glasses differences are producer provenance and eventual hardware quality, not downstream contract shape.
