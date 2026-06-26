# OSCAR Visual Augmentation Packet

`oscar_visual_augmentation_packet` prepares model-derived visual variants for
Post-Training Data Packages and visual distribution-shift evaluation. It keeps
motion and camera geometry fixed, requires camera and skeleton provenance, and
lets replaceable video backends such as OSCAR, Cosmos, or future WAM providers
generate realistic appearance variants.

The packet is support data. It does not prove contact physics, object drop
behavior, real robot readiness, deployment approval, safety validation, or
real-world task success.

## Inputs

Required inputs:

- first-frame visual context
- skeleton-conditioning video
- camera provenance such as calibration, policy-observation, or camera-profile
  manifest
- skeleton provenance such as projected G1 skeleton trace/manifest

Optional inputs:

- generated videos from a model backend, provided as model-derived references
- custom variant specs JSON with a `variants` list
- an OSCAR input package manifest, used to inherit first-frame, skeleton-video,
  and projected-skeleton paths

## Command

Build the request packet:

```bash
blueprint-build-oscar-visual-augmentation-packet \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --first-frame /path/to/first_frame.png \
  --skeleton-video /path/to/skeleton_conditioning.mp4 \
  --camera-provenance /path/to/camera_calibration_quality_gate.json \
  --skeleton-provenance /path/to/g1_projected_skeleton_trace.jsonl
```

With no `--output-dir`, the packet is written under:

```text
<job-dir>/oscar_visual_augmentation_packet/
```

Run generation for each variant with a visual-augmentation backend command:

```bash
export BLUEPRINT_OSCAR_VISUAL_AUGMENTATION_COMMAND="/path/to/oscar_visual_backend"

blueprint-run-oscar-visual-augmentation-generation \
  --packet-manifest <job-dir>/oscar_visual_augmentation_packet/oscar_visual_augmentation_packet_manifest.json \
  --backend-id oscar_wam \
  --backend-mode auto
```

The command receives:

- `BLUEPRINT_VISUAL_AUGMENTATION_REQUEST`
- `BLUEPRINT_VISUAL_AUGMENTATION_OUTPUT`
- `BLUEPRINT_VISUAL_AUGMENTATION_OUTPUT_VIDEO`
- `BLUEPRINT_VISUAL_AUGMENTATION_PACKET`
- `BLUEPRINT_VISUAL_AUGMENTATION_VARIANT_ID`

The backend should write the requested video and a JSON result that explicitly
marks whether the output is a learned/model-derived result. The runner only
labels an output model-derived when the backend result says so, for example with
`model_derived=true` plus a truth boundary such as
`generated_video_is_model_output=true`.

For local plumbing tests, use the fixture backend explicitly:

```bash
blueprint-run-oscar-visual-augmentation-generation \
  --packet-manifest <job-dir>/oscar_visual_augmentation_packet/oscar_visual_augmentation_packet_manifest.json \
  --backend-mode fixture \
  --allow-fixture-backend
```

Fixture videos are decodable MP4s for exercising the artifact and QA path only.
They are not OSCAR/Cosmos outputs and are not training data.

The Post-Training Data Package exporter discovers this packet automatically when
it exists under the job directory and indexes it as model-derived support.

## Artifacts

- `oscar_visual_augmentation_packet_manifest.json`
- `visual_augmentation_variant_requests.jsonl`
- `model_backend_registry.json`
- `visual_distribution_shift_eval_protocol.json`
- `claim_boundary.json`

After generation, the runner also writes:

- `generation_requests/*.json`
- `backend_results/*.json`
- `generated_videos/*.mp4`
- `visual_augmentation_generation_run_manifest.json`
- `visual_augmentation_generation_results.jsonl`
- `visual_augmentation_generation_qa_manifest.json`
- `visual_augmentation_training_readiness_manifest.json`
- `visual_augmentation_training_dataset_manifest.json`
- `exports/visual_augmentation/episodes.jsonl`

## Docker Image Check

The reusable OSCAR GPU image checked on June 26, 2026 is:

```text
docker.io/nijelhunt/blueprint-oscar-wam:20260622-cu128-shim
```

Registry metadata shows a linux/amd64 image with
`BLUEPRINT_OSCAR_WAM_SOURCE_ROOT=/opt/oscar-public`; its build history includes
cloning `https://github.com/wuzy2115/oscar-public.git` and running the OSCAR
image healthcheck. The older `20260621-cu128-shim` tag referenced by some tests
was not present in the registry during that check.

`BLUEPRINT_OSCAR_WAM_COMMAND` and `BLUEPRINT_OSCAR_WAM_PROVIDER_COMMAND` are
older WAM-rollout command contracts. They should be wrapped before use as a
visual-augmentation backend because this packet runner sends
`BLUEPRINT_VISUAL_AUGMENTATION_*` request variables.

## Claim Boundary

Generated videos in this packet must stay labeled as model-derived support
assets. They can support:

- visual robustness stress tests
- visual distribution-shift review
- training-data augmentation experiments
- policy-comparison support inside a bounded evaluator

They cannot support:

- contact or object-drop physics claims
- physical robot readiness
- deployment approval
- safety validation
- real-world success or sim-to-real calibration claims without accepted
  real-world anchors
