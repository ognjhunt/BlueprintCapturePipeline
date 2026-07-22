# Privacy Runner Services

The production default preview path is:

`BlueprintCapture upload -> storage trigger -> materialize -> qualification -> privacy/final_walkthrough.mov -> World Labs generate/poll -> WebApp sync -> catalog launch`

Only the privacy-safe walkthrough, or an audited derivative of it, may be used
for World Labs in production. Raw-video fallback is not allowed.

SAM3, VIP/depth, and DeepPrivacy2 are optional implementations for producing or
checking that privacy-safe walkthrough. They are not required for a production
handoff when `privacy/final_walkthrough.*` already exists with an audit manifest
that proves the World Labs input derives from it.

Exception for temporary internal demos only:

- If `BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS=true`, the pipeline may prepare a World Labs input clip from the raw walkthrough when privacy processing is unavailable.
- This bypass path must be treated as non-production and unredacted.
- The prepared input is auto-trimmed/compressed to World Labs upload limits before submission.

## Optional Services

Deploy only the GPU-backed services needed by the current privacy strategy:

- `sam3-detect`
- `vip-inpaint`
- `deepprivacy2-anonymize`
- `video-to-world`

The main `blueprint-pipeline` Cloud Run job remains CPU-only.

## HTTP Contract

All services accept `POST /run` with `Authorization: Bearer $PRIVACY_RUNNER_TOKEN`.
Terraform deploys these GPU-backed services as private Cloud Run services. The
pipeline job is granted `roles/run.invoker`; public `allUsers` or
`allAuthenticatedUsers` invoker bindings are not valid for these runners.

When `BLUEPRINT_CLOUD_RUN_IAM_AUTH_ENABLED=true`, clients also add
`X-Serverless-Authorization: Bearer <Google ID token>` for the target service
URL. `Authorization` remains reserved for the runner token that the service
validates after Cloud Run IAM accepts the request.

Shared input fields:

- `input_video_uri` or `input_video_path`
- `output_json_uri` or `output_json_path`

For local or owner-managed deployments, each privacy runner also has a command
template hook. The command must write the requested output JSON path.

- `PRIVACY_SAM3_COMMAND` or legacy `SAM3_COMMAND`
- `PRIVACY_VIP_COMMAND` or legacy `VIP_COMMAND`
- `PRIVACY_DEPTH_ANYTHING_COMMAND` or legacy `DEPTH_ANYTHING_COMMAND`
- `PRIVACY_DEEPPRIVACY2_COMMAND` or legacy `DEEPPRIVACY2_COMMAND`

## Runner Environment Variables

The orchestrator in `src/blueprint_pipeline/privacy_processing.py` resolves each
runner from these env vars. The `PRIVACY_`-prefixed name always wins; the bare
(legacy) name is consulted only as a fallback. The HTTP `*_URL` runners have no
legacy bare spelling. Every variable below is grep-verifiable in
`privacy_processing.py`.

| Runner | PRIVACY_-prefixed | Bare / legacy | Purpose |
| --- | --- | --- | --- |
| SAM3 detect | `PRIVACY_SAM3_URL` | (none) | HTTP `POST /run` endpoint |
| SAM3 detect | `PRIVACY_SAM3_COMMAND` | `SAM3_COMMAND` | local/owner command template |
| SAM3 detect | `PRIVACY_SAM3_TIMEOUT_SECONDS` | (none) | request timeout (default 3600) |
| VIP inpaint | `PRIVACY_VIP_URL` | (none) | HTTP `POST /run` endpoint |
| VIP inpaint | `PRIVACY_VIP_COMMAND` | `VIP_COMMAND` | local/owner command template |
| VIP inpaint | `PRIVACY_VIP_TIMEOUT_SECONDS` | (none) | request timeout (default 7200) |
| Depth Anything | `PRIVACY_DEPTH_ANYTHING_URL` | (none) | HTTP endpoint; falls back to `PRIVACY_VIP_URL` |
| Depth Anything | `PRIVACY_DEPTH_ANYTHING_COMMAND` | `DEPTH_ANYTHING_COMMAND` | local/owner command template |
| Depth Anything | `PRIVACY_DEPTH_ANYTHING_TIMEOUT_SECONDS` | (none) | request timeout (default 7200) |
| DeepPrivacy2 | `PRIVACY_DEEPPRIVACY2_URL` | (none) | HTTP `POST /run` endpoint |
| DeepPrivacy2 | `PRIVACY_DEEPPRIVACY2_COMMAND` | `DEEPPRIVACY2_COMMAND` | local/owner command template |
| DeepPrivacy2 | `PRIVACY_DEEPPRIVACY2_TIMEOUT_SECONDS` | (none) | request timeout (default 7200) |

Pipeline-level controls (read directly by `privacy_processing.py`):

| Variable | Purpose |
| --- | --- |
| `PRIVACY_PIPELINE_ENABLED` | gate that turns privacy post-processing on |
| `PRIVACY_FAIL_CLOSED` | fail-closed policy flag, defaults to true |
| `PRIVACY_RUNNER_TOKEN` | bearer token added as `Authorization` on runner HTTP calls |
| `PRIVACY_RUNNER_LOCAL_PATH_ROOT` | optional containment root required before HTTP requests may use any local input/output path; `gs://` destinations do not require it |
| `BLUEPRINT_CLOUD_RUN_IAM_AUTH_ENABLED` | add `X-Serverless-Authorization` ID-token auth for private Cloud Run runners |
| `PRIVACY_LOCAL_FULL_FRAME_REDACTION_ENABLED` | enable the local, full-frame redaction proof path (legacy alias `BLUEPRINT_PRIVACY_LOCAL_FULL_FRAME_REDACTION`) |

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
- `VIDEO_TO_WORLD_RUNNER_LOCAL_PATH_ROOT` when an HTTP request uses local
  filesystem inputs or outputs; all such paths must resolve beneath this root
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

Depth artifacts are optional but persistent when they are available:

- use ARKit depth and confidence when present
- otherwise run Depth Anything 3 only when the depth runner is configured for
  the lane
- persist `depth_manifest.json` and `confidence_manifest.json` plus the per-frame artifact prefixes
- reuse those manifests during VIP inpainting for non-ARKit captures
- return `depth_source = "arkit"` or `depth_source = "depth_anything"` when
  depth artifacts are produced

No separate service is required. `vip-inpaint` accepts a `depth_generation_only`
request mode for DA3 artifact generation, and standard VIP requests can consume
the resulting manifests. If no depth runner is configured, production must rely
on the existing privacy audit rather than inventing depth proof.

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

- `sam3-detect` installs Meta's `sam3` package and expects either
  `SAM3_WEIGHTS_PATH` or Hugging Face access to `facebook/sam3`.
- `vip-inpaint` uses the repo-managed depth-guided inpainting backend, bundles
  Depth Anything 3 when configured, and supports both depth-only generation and
  depth-guided inpainting.
- `deepprivacy2-anonymize` shells into a checked-out DeepPrivacy2 repo located
  at `${DEEPPRIVACY2_REPO_DIR:-/opt/deepprivacy2}` and runs its `anonymize.py`
  against the anonymizer config at
  `${DEEPPRIVACY2_REPO_DIR:-/opt/deepprivacy2}/configs/anonymizers/face.py`.
  This config ships inside that checked-out repo; it is not a file in this
  pipeline repository.

## Deployment Caveats

Local tests validate the service/command contract and storage behavior, not live
GPU inference.

External deployment validation is still required for:

- actual SAM3 checkpoint access and GPU memory sizing, when SAM3 is used
- the final Cloud Run build/install compatibility of DeepPrivacy2 plus DensePose
  dependencies, when DeepPrivacy2 is used
- any optional proprietary VIP model drop referenced by `VIP_MODEL_PATH`
