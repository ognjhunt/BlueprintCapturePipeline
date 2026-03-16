# Privacy Runner Services

The production preview path is:

`BlueprintCapture upload -> storage trigger -> materialize -> qualification -> privacy/final_walkthrough.mov -> World Labs generate/poll -> WebApp sync -> catalog launch`

Only the privacy-safe walkthrough may be used for World Labs. Raw-video fallback is not allowed.

## Services

Deploy three GPU-backed Cloud Run services:

- `sam3-detect`
- `vip-inpaint`
- `deepprivacy2-anonymize`

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
- `preferred_depth_source`
- `vip_model_path`
- `depth_anything_model_path`

`vip-inpaint` response fields:

- `status`
- `output_video`
- `output_video_uri`
- `depth_source`

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

## Depth Behavior

`vip-inpaint` must select depth automatically:

- use ARKit depth and confidence when present
- otherwise use Depth Anything
- always return `depth_source = "arkit"` or `depth_source = "depth_anything"`

No separate Depth Anything service is used.

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
- `vip-inpaint` uses the repo-managed depth-guided inpainting backend and bundles Depth Anything 3 for non-ARKit captures.
- `deepprivacy2-anonymize` shells into a checked-out `deep_privacy2` repo and uses `configs/anonymizers/face.py`.

## Deployment Caveats

Local tests validate the service contract and storage behavior, not live GPU inference.

External deployment validation is still required for:

- actual SAM3 checkpoint access and GPU memory sizing
- the final Cloud Run build/install compatibility of DeepPrivacy2 plus DensePose dependencies
- any optional proprietary VIP model drop referenced by `VIP_MODEL_PATH`
