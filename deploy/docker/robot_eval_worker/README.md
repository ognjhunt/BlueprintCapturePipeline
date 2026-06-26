# Robot Eval Worker Images

These Dockerfiles are prepared-worker surfaces for queued robot-eval jobs. They
install the Blueprint Pipeline package and run `blueprint-run-robot-eval-worker`,
which loads `BLUEPRINT_EVAL_MANIFEST_URI`, runs `blueprint-run-robot-eval-job`,
and writes `worker_runtime_manifest.json`.

Build and publish versioned image refs before using a live provider launcher.
The job orchestrator intentionally blocks RunPod/Vast/GCP provider requests
until the selected simulator has a configured image ref:

- `BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF` for Isaac Sim
- `BLUEPRINT_ISAAC_ARENA_EVAL_WORKER_IMAGE_REF` for Isaac Lab/Arena
- `BLUEPRINT_MUJOCO_EVAL_WORKER_IMAGE_REF` for MuJoCo
- `BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF` as a generic fallback

Tags such as `latest`, `local`, `dev`, and `test` are not accepted as versioned
live-provider refs. Use a dated/versioned tag or an immutable digest.

They do not provision GPUs, call RunPod/Vast/GCP, send customer messages, or
upgrade proof by themselves. Live GPU/provider execution still requires the
orchestrator gates such as `BLUEPRINT_ALLOW_GPU_PROVISIONING=true`,
`--allow-gpu-provisioning`, `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true`, and
`--allow-simulator-execution`.
After the orchestrator writes a `request_manifest_ready`
`gpu_provider_launch_request.json`, use `blueprint-run-gpu-provider-launcher`
as the separate owner-controlled provider handoff. It requires
`BLUEPRINT_ALLOW_GPU_PROVIDER_LAUNCH=true`, `--allow-provider-launch`, and a
provider adapter command from `BLUEPRINT_GPU_PROVIDER_LAUNCH_COMMAND` or
`--provider-launch-command`. The launcher records
`gpu_provider_launcher_result.json` and stdout/stderr logs, does not store the
raw command or provider secret values, and does not prove simulator execution or
generated-world rank fidelity.
For RunPod, point the provider adapter command at
`blueprint-run-runpod-provider-adapter`. It defaults to a dry-run
`runpod_provider_adapter_result.json` request-shape artifact and requires
`BLUEPRINT_ALLOW_RUNPOD_API_CALLS=true`, `RUNPOD_API_KEY`, and
`--allow-runpod-api-call` before it can submit a serverless `/run` job or create
an on-demand Pod.

Worker manifest input supports local paths, `file://`, `http://`, `https://`,
`gs://`, `s3://`, and `r2://`. Live RunPod/Vast/GCP workers require a remote
manifest URI in `BLUEPRINT_EVAL_MANIFEST_URI`; a local path is accepted only for
local fixture/development workers because provider workers cannot fetch files
from the orchestrator host. Artifact output supports local paths, `file://`,
`gs://`, `s3://`, and `r2://`. S3-compatible storage uses `boto3`; R2 requires
`BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL` or `R2_ENDPOINT_URL`. Do not place access
keys in manifests or artifacts; use provider-native environment/secret injection.
For live/non-fixture provider jobs, provide `artifact_output_uri` in the worker
manifest or pass `--artifact-output-uri`; otherwise the worker writes a blocked
`worker_runtime_manifest.json` and does not run the orchestrator. Local fixture
jobs can enforce the same finalizer rule with `artifact_output_uri_required=true`
or `--require-artifact-output-uri`.
Live provider jobs must use the strict queued manifest envelope:
`schema_version: "robot_eval_worker_manifest.v1"` with an embedded `job_request`
object. A raw `robot_eval_job_request.v1` JSON is accepted only for local fixture
use, not as a provider worker manifest.
Live non-fixture worker manifests must also carry `runtime_preflight_contract`.
The worker entrypoint rejects provider manifests that do not require pre-scene
runtime preflight, a `worker_runtime_preflight.json` result artifact, and failed
preflight blocking scene load. Isaac contracts include NVIDIA inventory, driver,
Vulkan/RTX, headless launch, blank scene, and test-frame checks; MuJoCo keeps the
lower-cost import/headless/EGL-when-rendering/rollout checks.
The worker writes `worker_runtime_preflight.json` before delegating to the job
orchestrator. If `--allow-simulator-execution` is set for a non-fixture
simulator, the manifest or environment must provide a runtime preflight command
using `runtime_preflight_command`, `runtime_preflight_commands.<simulator>`, or
`BLUEPRINT_RUNTIME_PREFLIGHT_COMMAND`; missing, failing, or timed-out preflight
blocks scene work. Preflight stdout/stderr are persisted as
`worker_runtime_preflight.stdout.log` and
`worker_runtime_preflight.stderr.log` after known secret env values are
redacted; early preflight failures still copy worker-level failure artifacts to
`artifact_output_uri` when configured.
`blueprint-run-robot-eval-job` writes this payload as `worker_manifest.json` in
the job directory so provider launchers have an exact manifest to upload and use
as `BLUEPRINT_EVAL_MANIFEST_URI`. The provider launch request stays blocked for
live providers until that fetch URI is configured.
After the job artifact directory is copied, the worker writes the final
`worker_runtime_manifest.json` into the job directory and uploads/copies it to
the artifact output destination as `worker_runtime_manifest.json`.
Provider launch artifacts also carry `idle_timeout_seconds`,
`hard_timeout_seconds`, and `external_watchdog_ttl_seconds`; the watchdog TTL
must exceed the hard timeout so an external launcher can terminate a stuck worker
even if the finalizer never runs.

## MuJoCo Worker

```bash
BLUEPRINT_MUJOCO_EVAL_WORKER_IMAGE_REF="registry.example/blueprint/mujoco-eval-worker:2026-06-14-matrix-batch" \
BLUEPRINT_MUJOCO_WORKER_PLATFORM=linux/amd64 \
./scripts/build_push_mujoco_worker_image.sh
```

The MuJoCo image is the cheaper CPU/low-cost GPU lane. It installs MuJoCo and
uses `MUJOCO_GL=egl` for headless rendering when a renderer is needed.
The helper refuses `latest`, `local`, `dev`, and `test` tags. Set
`BLUEPRINT_ALLOW_MUJOCO_WORKER_IMAGE_PUSH=true` only when Docker is authenticated
to the target registry and the image ref is intended to be provider-fetchable.
The default image intentionally does not install Torch or an official locomotion
policy runtime. Use a separate policy-worker image before claiming balanced G1
walking-controller or training-grade rollout proof.

## Isaac Worker

```bash
BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF="registry.example/blueprint/isaac-eval-worker:2026-06-26" \
BLUEPRINT_ISAAC_WORKER_PLATFORM=linux/amd64 \
./scripts/build_push_isaac_worker_image.sh
```

Set `BLUEPRINT_ALLOW_ISAAC_WORKER_IMAGE_PUSH=true` only after Docker is
authenticated to the target registry and the image ref is intended to be
provider-fetchable. The helper refuses `latest`, `local`, `dev`, and `test`
tags. To use a different pinned Isaac base image, set
`BLUEPRINT_ISAAC_SIM_BASE_IMAGE`.

NVIDIA's current Isaac Sim 6.0 container docs use
`nvcr.io/nvidia/isaac-sim:6.0.0`, rootless user `1234:1234`, `--gpus all`, and
host-mounted cache directories for `.cache`, compute cache, config, data, logs,
package cache, and Omniverse hub cache. Keep those cache mounts warm outside the
customer job path.

Example run shape:

```bash
docker run --rm --gpus all --network=host \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e BLUEPRINT_EVAL_MANIFEST_URI=/work/worker_manifest.json \
  -v "$PWD:/work" \
  -v "$HOME/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw" \
  -v "$HOME/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
  -v "$HOME/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
  -v "$HOME/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw" \
  -v "$HOME/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw" \
  -v "$HOME/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw" \
  -v "$HOME/.cache/ov/hub:/var/cache/hub:rw" \
  blueprint/isaac-eval-worker:local
```

Before treating a run as simulator proof, the owner system still needs provider
runtime evidence: GPU model, driver, Vulkan/RTX preflight, scene load trace,
spawn trace, policy/action trace, logs, POV/screenshot/video artifacts, and the
job-level proof boundary.
