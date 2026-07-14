# GR00T + OSCAR cached foundation and thin release

This is the durable successor to the 47.1 GB sealed-image packaging. It does
not change the workload or claim that a new host can avoid acquiring Isaac,
CUDA, robot runtimes, and models once.

## Release units

The worker is an exact tuple:

1. digest-pinned foundation image built from `Foundation.Dockerfile`;
2. digest-pinned thin Blueprint release built from `Release.Dockerfile`;
3. `groot_oscar_external_model_cache.v1` manifest digest on a provider volume;
4. GPU serving class.

The foundation contains Isaac 6, the shared PyTorch/CUDA robot environment,
OSCAR runtime source, installed GR00T runtime, WBC runtime binaries/configs,
TensorRT runtime libraries, and pinned Isaac G1 assets. WBC compilers, object
files, source Git data, and build caches do not enter the final stage. GR00T
and OSCAR share `/opt/robot-venv`; `pip check` and the GR00T import gate must
pass before publication.

Checkpoints are absent from both OCI images. The release entrypoint verifies
every declared model file by size and SHA-256 while offline, verifies the exact
repository revisions and manifest self-digest, and only then links the verified
GEAR-SONIC model assets into the WBC runtime tree.

## 1. Build the stable foundation

The build plane is not the RunPod serve plane. A normal RunPod Pod is not an
approved Docker builder: its container-disk ceiling, volume semantics, and
privilege model do not satisfy this build. Use a verified native linux/amd64
Docker host with at least 120 GiB free, `docker buildx`, file-based registry
push credentials, a launch-bound independently verified SSH host key, a
two-hour-or-shorter hard TTL, and an independent teardown watchdog. The pure
`groot_oscar_build_plane_admission.v1` gate must say `admitted` before any paid
builder API mutation. Do not discover builder capabilities by renting pods.

The currently known profile is
`digitalocean-s-8vcpu-16gb-amd-ubuntu-24-04-v1`: 8 vCPU, 16 GiB RAM, 320 GB
disk, and a catalog ceiling of $0.16667/hour. Re-query the live size catalog,
verify zero other builder-tagged droplets, generate a launch-bound Ed25519 host
key locally, and bind its SHA-256 fingerprint into the admission record before
creation. `accept-new`, trust-on-first-use, or deleting a stale `known_hosts`
entry cannot satisfy the gate. Existing unrelated production droplets do not
count as builders and must never be repurposed.

```bash
BLUEPRINT_GROOT_OSCAR_FOUNDATION_IMAGE_REF='<registry>/blueprint-groot-oscar-foundation:<version>' \
BLUEPRINT_ALLOW_GROOT_OSCAR_FOUNDATION_IMAGE_PUSH=true \
./scripts/build_push_groot_oscar_foundation_image.sh
```

Resolve the pushed image to `repository@sha256:<digest>`. Cache that exact
digest on every GPU host; never configure the release builder with a tag.

## 2. Prepare the provider model volume

Mount the provider volume at `/models`. Preparation is asynchronous and may
use the network and a file-based Hugging Face token. Customer requests may not
run this step.

```bash
BLUEPRINT_GROOT_OSCAR_MODEL_CACHE=/models/blueprint-groot-oscar-v1 \
BLUEPRINT_GROOT_OSCAR_MODEL_CACHE_HF_TOKEN_FILE="$HOME/.blueprint-secrets/hf_token" \
./scripts/prepare_groot_oscar_model_cache.sh
```

The second command in the script re-reads and hashes every model byte under
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Preserve the emitted
`manifest_digest` as part of worker readiness evidence. File presence, total
directory size, or a hash of size metadata is not accepted as model proof.

RunPod exposes its network volume at `/workspace`, so prepare the same cache at
`/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1` and set the
provider request cache path `groot_oscar_models` to that exact directory. VM
providers mount the host cache root `/var/lib/blueprint/models` read-only at
`/models`; cloud-init refuses to start the thin worker if its manifest is absent.

## 3. Build the thin release

The source tree must be clean. The foundation must be digest-pinned. On push,
the builder inspects both registry manifests, proves that the release extends
the exact foundation layers, and limits only the new compressed release layers
to 2 GiB by default.

```bash
BLUEPRINT_GROOT_OSCAR_FOUNDATION_IMAGE_REF='<registry>/foundation@sha256:<digest>' \
BLUEPRINT_GROOT_OSCAR_RELEASE_IMAGE_REF='<registry>/blueprint-groot-oscar-eval:<version>' \
BLUEPRINT_ALLOW_GROOT_OSCAR_RELEASE_IMAGE_PUSH=true \
./scripts/build_push_groot_oscar_release_image.sh
```

The desired operational target is hundreds of MB; 2 GiB is a hard release
budget, not a claim that the first build already achieves the desired target.
The registry delta evidence is authoritative.

After the exact release is present on a worker host, record `docker image
inspect` evidence containing `local_uncompressed_size_bytes`, verify the real
model volume, and audit the combined on-disk footprint:

```bash
python -m blueprint_pipeline.groot_oscar_cached_footprint \
  --image-evidence <local-image-evidence.json> \
  --model-cache-verification <model-cache-verification.json> \
  --expected-release-ref '<registry>/release@sha256:<digest>' \
  --out groot_oscar_cached_worker_footprint.json
```

This is the 30 GiB target gate. Registry-compressed image bytes cannot be
substituted for local Docker image size, and unhashed model-directory sizes
cannot be substituted for the verified model-cache total.

## 4. Warm-worker admission

Start the release with the already-populated volume mounted at
`/models/blueprint-groot-oscar-v1`. The entrypoint fails before worker code if
the manifest is missing, a revision changed, a file was added/removed, or any
byte digest differs. `production_gpu_worker_agent` may set
`models_cached_offline=true` only from that verifier's manifest digest.

Serve customers only from ready warm workers. A new host must asynchronously
acquire the stable foundation and model volume once, run the same runtime and
model gates, load Isaac/scene/policy, and only then register as ready. Thin
release size alone is not cold-start or task-success proof.

## Proof boundaries

- Local Dockerfile/tests prove contract structure, not an amd64 image build.
- Registry diagnostics prove exact compressed layers and release delta, not
  provider startup latency.
- Model-cache verification proves exact offline bytes, not successful policy
  inference.
- Warm readiness proves same-session runtime health, not semantic task success.
