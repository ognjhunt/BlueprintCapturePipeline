# Privacy Runner Services

The production default preview path is:

`BlueprintCapture upload -> storage trigger -> materialize -> qualification -> privacy/final_walkthrough.mov -> World Labs generate/poll -> WebApp sync -> catalog launch`

Only the privacy-safe walkthrough may be used for World Labs. Raw-video fallback is not allowed.

Exception for temporary internal demos only:

- If `BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS=true`, the pipeline may prepare a World Labs input clip from the raw walkthrough when privacy processing is unavailable.
- This bypass path must be treated as non-production and unredacted.
- The prepared input is auto-trimmed/compressed to World Labs upload limits before submission.

## Services

Deploy four GPU-backed Cloud Run services:

- `sam3-detect`
- `vip-inpaint`
- `deepprivacy2-anonymize`
- `video-to-world`

The main `blueprint-pipeline` Cloud Run job remains CPU-only.

## HTTP Contract

All services accept `POST /run` with `Authorization: Bearer $PRIVACY_RUNNER_TOKEN`.

Shared input fields:

- `input_video_uri` or `input_video_path`
- `output_json_uri` or `output_json_path`

`sam3-detect` request fields:

- `masks_prefix_uri` or `masks_dir_path`
- `prompt`
- `stage_name`
- `sam3_weights_path`

`sam3-detect` response fields:

- `status`
- `people_detected`
- `people_count`
- `mask_paths`

`vip-inpaint` request fields:

- `masks_prefix_uri` or `masks_dir_path`
- `output_video_uri` or `output_video_path`
- `arkit_depth_prefix_uri`
- `arkit_confidence_prefix_uri`
- `depth_manifest_uri`
- `confidence_manifest_uri`
- `preferred_depth_source`
- `depth_generation_only`
- `depth_output_prefix_uri`
- `confidence_output_prefix_uri`
- `output_depth_manifest_uri`
- `output_confidence_manifest_uri`
- `vip_model_path`
- `depth_anything_model_path`

`vip-inpaint` response fields:

- `status`
- `output_video`
- `output_video_uri`
- `depth_source`
- `depth_manifest_uri`
- `confidence_manifest_uri`

`deepprivacy2-anonymize` request fields:

- `output_video_uri` or `output_video_path`
- `deepprivacy2_model_path`

`deepprivacy2-anonymize` response fields:

- `status`
- `output_video`
- `output_video_uri`
- `face_anonymized_segments`

## Storage Behavior

The services support both mounted GCS paths and direct `gs://` URIs.

- If the bucket is mounted at `GCS_ROOT`, the service reads and writes directly through the mount.
- If the object is not mounted locally, the service downloads inputs and uploads outputs with `google-cloud-storage`.
- The main pipeline still re-materializes remote outputs locally before verification or final ffmpeg steps.

## video_to_world Runtime

The geometry path uses a dedicated `video-to-world` service with:

- `VIDEO_TO_WORLD_RUNNER_TOKEN`
- `VIDEO_TO_WORLD_PIPELINE_PRESET` or `VIDEO_TO_WORLD_COMMAND_TEMPLATE`
- `VIDEO_TO_WORLD_COMMAND_TIMEOUT_SECONDS`

Supported presets:

- `preprocess_only`
- `preprocess_plus_alignment` (default deployment preset)
- `full_fast`
- `full_extensive`

The normalized geometry contract treats this service boundary as the live proof
path. If local development falls back to synthetic geometry or local DA3 helper
outputs, the geometry artifacts must remain labeled fallback/internal and must not
set `ready_for_world_model`, `geometry_live_ready`, or site-faithful launch flags.

If `VIDEO_TO_WORLD_COMMAND_TEMPLATE` is set, it overrides the preset. Template substitutions:

- `{INPUT_VIDEO}`
- `{GEOMETRY_ROOT}`
- `{SCENE_ROOT}`
- `{RESULT_JSON}`
- `{DYNAMIC_MASK_MANIFEST}`

## Depth Behavior

Depth behavior is now mandatory and persistent:

- use ARKit depth and confidence when present
- otherwise run Depth Anything 3 for every non-ARKit capture, even when SAM3 finds no people
- persist `depth_manifest.json` and `confidence_manifest.json` plus the per-frame artifact prefixes
- reuse those manifests during VIP inpainting for non-ARKit captures
- always return `depth_source = "arkit"` or `depth_source = "depth_anything"`

No separate service is required. `vip-inpaint` accepts a `depth_generation_only` request mode for DA3 artifact generation, and standard VIP requests can consume the resulting manifests.

## Model Paths

Each model env var may be:

- a local filesystem path
- a `gs://` URI
- an `https://` URL

Supported env vars:

- `SAM3_WEIGHTS_PATH`
- `VIP_MODEL_PATH`
- `DEPTH_ANYTHING_MODEL_PATH`
- `DEEPPRIVACY2_MODEL_PATH`

The services materialize remote weights locally at request time when necessary. Do not commit secrets; use deployment env vars or secret injection for tokens.

## Runtime Notes

- `sam3-detect` installs Meta's `sam3` package and expects either `SAM3_WEIGHTS_PATH` or Hugging Face access to `facebook/sam3`.
- `vip-inpaint` uses the repo-managed depth-guided inpainting backend, bundles Depth Anything 3, and supports both depth-only generation and depth-guided inpainting.
- `deepprivacy2-anonymize` shells into a checked-out `deep_privacy2` repo and uses `configs/anonymizers/face.py`.

## Deployment Caveats

Local tests validate the service contract and storage behavior, not live GPU inference.

External deployment validation is still required for:

- actual SAM3 checkpoint access and GPU memory sizing
- the final Cloud Run build/install compatibility of DeepPrivacy2 plus DensePose dependencies
- any optional proprietary VIP model drop referenced by `VIP_MODEL_PATH`
