# GR00T + OSCAR cached foundation and thin release

This is the durable successor to the 47.1 GB sealed-image packaging. It does
not change the workload or claim that a new host can avoid acquiring Isaac,
CUDA, robot runtimes, and models once.

## Release units

The worker is an exact tuple:

1. digest-pinned foundation image built from `Foundation.Dockerfile`;
2. digest-pinned thin Blueprint release built from `Release.Dockerfile`;
3. `groot_oscar_external_model_cache.v2` manifest digest on a provider volume;
4. GPU serving class.

The foundation contains Isaac 6, the shared PyTorch/CUDA robot environment,
OSCAR runtime source, installed GR00T runtime, WBC runtime binaries/configs,
TensorRT runtime libraries, and pinned Isaac G1 assets. WBC compilers, object
files, C/C++ source/build trees, ONNX Runtime headers, source Git data, and
build caches do not enter the final stage. The WBC copy is an explicit runtime
allowlist: the production executable, G1 runtime assets, setup script, required
reference data, the ZMQ Python client surface, Unitree runtime libraries, and
ONNX Runtime shared objects. GR00T and OSCAR deliberately use isolated
environments: the pinned GR00T release declares Python 3.10 with Torch 2.7.1,
while the pinned OSCAR release is verified with Torch 2.10.0 and its public
inference requirements declare that exact version. Treating those stacks as
compatible silently downgraded OSCAR and skipped its real
`requirements_minimal.txt`. Both environments now run `pip check`, and both
the GR00T policy import and OSCAR inference import must pass before publication.
This gives up speculative Torch deduplication rather than shipping an invalid
shared environment; the foundation remains host-cached and the release layer
remains thin.

The OSCAR environment is installed from a fully resolved, hash-checked
Linux/Python 3.10 lock that includes the Blueprint runtime dependencies. GR00T
uses the immutable upstream `uv.lock` with `uv sync --frozen`, followed by a
no-dependency install of the pinned source tree. Neither environment performs
an unconstrained dependency solve on the paid builder.

Checkpoints are absent from both OCI images. The release entrypoint verifies
every declared model file by size and SHA-256 while offline, verifies the exact
repository revisions and manifest self-digest, and only then links the verified
GEAR-SONIC model assets into the WBC runtime tree.

Before allocating a builder, run the network-free architecture gate and the
read-only live prerequisite gate:

```bash
python scripts/verify_groot_oscar_thin_architecture.py
python scripts/verify_groot_oscar_live_prerequisites.py \
  --live \
  --output output/groot_oscar_live_prerequisites.json
```

CI runs both gates on every push and pull request. The live gate downloads and
hashes only the small pinned bootstrap archives, reads NVIDIA's Ubuntu 24.04
package index, and inspects immutable GitHub and Hugging Face revision metadata;
it also checks the declared Isaac asset URLs and byte sizes without downloading
those assets. For the public NGC Isaac base, it obtains an ephemeral anonymous
pull token, downloads only the manifest list, recomputes its pinned SHA-256, and
requires a Linux/amd64 child manifest. It does not pull a container layer or
download model weights. It fails before paid allocation if a base manifest,
exact TensorRT package, source commit, model revision, or required model
filename has disappeared or drifted. The
network-free gate rejects reintroduced WBC build trees, an unproven shared
GR00T/OSCAR environment, unpinned bootstrap downloads, model acquisition in the release
Dockerfile, incomplete critical model-file contracts, and mutable
foundation/release seams before paid infrastructure is used.

The canonical `paid_resource_allocator cpu-build` command runs the live gate
before starting its detached supervisor, and the supervisor runs it again
immediately before the provider adapter can create a droplet. A blocked or
missing evidence file therefore fails closed with
`provider_mutation_attempted: false`; passing CI is not a substitute for the
launch-time recheck.

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

Use `python -m blueprint_pipeline.paid_resource_allocator cpu-build` for this
profile. It is the only supported CPU builder allocator; the old
provider-specific `run` and `launch` entrypoints
are hard-disabled. It is dry by default and requires both an admitted spend JSON
and `--allow-paid`. The allocator rechecks the live size catalog and builder
inventory, provisions the exact launch-bound host key, then records the actual
filesystem mount/free bytes, Linux architecture, Docker daemon, and Buildx
responses from a probe running on the allocated host. Requested configuration
and provider catalog rows cannot substitute for that evidence. It transfers
registry credentials as files
over strict-key SSH, runs the checksum-bound packet, retrieves JSON evidence,
and deletes the droplet in `finally`. The `launch` command starts that
supervisor in a new OS session so a terminal or orchestration timeout cannot
interrupt the build; `run` remains the foreground implementation entrypoint.
A detached watchdog independently deletes the same droplet at the authorized
deadline.

The old standalone foundation build script is hard-disabled. Prepare the
checksum-bound thin-build packet, then run the canonical `cpu-build` command;
that single allocation builds foundation and release in the same bounded
session.

Resolve the pushed image to `repository@sha256:<digest>`. Cache that exact
digest on every GPU host; never configure the release builder with a tag.

## 2. Prepare the provider model volume

Mount the provider volume at `/models`. Preparation is asynchronous and may
use the network and a file-based Hugging Face token. Customer requests may not
run this step.

For RunPod, use the canonical guarded allocator rather than creating or
populating a volume in the console:

```bash
python -m blueprint_pipeline.paid_resource_allocator model-volume \
  --output-dir <evidence-dir> \
  --data-center-id '<capacity-verified-dc>' \
  --volume-size-gib 50 \
  --storage-hourly-rate-usd '<provider-verified-rate>' \
  --storage-ttl-seconds 14400 \
  --max-storage-spend-usd 0.05 \
  --builder-evidence <digitalocean-builder-evidence.json> \
  --builder-spend <digitalocean-builder-spend.json> \
  --login-private-key <digitalocean-login-private-key> \
  --host-private-key <pinned-builder-host-private-key> \
  --ssh-key-id <digitalocean-ssh-key-id> \
  --allow-paid
```

For the bounded persistent carrier lane, the same allocator can populate the
verified model cache and an allowlisted runtime archive onto one fresh 120 GiB
volume without allocating a GPU. Both images must be immutable digest refs.
The source release supplies the exact `/opt` runtime roots. The separately
deployable RunPod carrier includes the compatible CUDA 12.6/TensorRT and
Isaac system-link surface, but it does not contain the Blueprint runtime roots
or model cache:

```bash
python -m blueprint_pipeline.paid_resource_allocator model-volume \
  --output-dir <persistent-volume-evidence-dir> \
  --data-center-id '<capacity-verified-s3-dc>' \
  --volume-size-gib 120 \
  --runtime-source-release-image-ref '<release>@sha256:<digest>' \
  --runtime-source-release-evidence <groot_oscar_thin_remote_build_result.json> \
  --carrier-image-ref 'docker.io/nijelhunt/blueprint-groot-oscar-carrier@sha256:7e1bd59fda79a038f7154b8e2fbebe172c28d2a2995da4fd15a4f5a5d9e0c9a3' \
  --storage-hourly-rate-usd '<provider-verified-rate>' \
  --storage-ttl-seconds 28800 \
  --max-storage-spend-usd '<bounded-eight-hour-volume-cost>' \
  --builder-evidence <digitalocean-builder-evidence.json> \
  --builder-spend <digitalocean-builder-spend.json> \
  --login-private-key <digitalocean-login-private-key> \
  --host-private-key <pinned-builder-host-private-key> \
  --ssh-key-id <digitalocean-ssh-key-id> \
  --allow-paid
```

Runtime-bearing volumes require at least eight hours of bounded retention and
must also cover the builder TTL, transfer margin, and the 18,600-second GPU
watchdog. Before any detached supervisor or provider mutation, the allocator
requires the completed thin-release build result to bind the exact source
digest, Linux/amd64 platform, CUDA 12.8 contract, externalized-model claim, and
thin release delta. A sealed or mismatched image ref is rejected at the CLI or
pre-allocation evidence gate. The allocator pulls both images on the admitted CPU builder, records
their resolved repo digests, copies only the declared runtime roots from the
source release, rejects unsafe archive members, uploads the runtime archive and
manifest beside the model cache through RunPod S3, fully redownload-verifies all
bytes, and emits `carrier_volume_admission.json`. The GPU lane will not accept a
plain model-cache result in place of this combined admission.

Before upload, the CPU builder scans every ELF file across all declared runtime
roots for unresolved linkage inside the exact carrier. It also runs GR00T,
OSCAR, Isaac, and serverless import matrices and records every failed check in
one compatibility audit instead of stopping at the first missing dependency.
These checks do not prove a CUDA driver, Isaac rendering, policy execution, or
semantic task success.

Use the current provider-confirmed total hourly price for the selected volume
as `--storage-hourly-rate-usd`; the allocator rejects a TTL whose maximum cost
exceeds `--max-storage-spend-usd`. The admitted builder evidence and spend
files bind this operation to the one authorized DigitalOcean CPU builder.

It arms an independent name-scoped watchdog before creating anything, creates
one storage-only RunPod volume and no GPU Pod, prepares and fully hashes the
pinned allowlist on the admitted DigitalOcean CPU builder, uploads and
redownload-verifies it through RunPod S3, and emits the verified volume ID for
the GPU canary while the original deadline watchdog remains responsible for
the volume. On failure it deletes the volume too. Hugging Face and RunPod S3
credentials remain file-backed and raw secret values are never persisted.

The second command in the script re-reads and hashes every model byte under
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Preserve the emitted
`manifest_digest` as part of worker readiness evidence. File presence, total
directory size, or a hash of size metadata is not accepted as model proof.
Preparation downloads only the pinned runtime allowlist into a sibling staging
directory. It removes downloader metadata, builds and verifies the manifest,
then atomically replaces the active cache. A failed or interrupted preparation
leaves the previous cache in place; customer-serving workers never consume a
partially downloaded replacement.

RunPod exposes its network volume at `/workspace`, so prepare the same cache at
`/workspace/.blueprint-model-cache/blueprint-groot-oscar-v1` and set the
provider request cache path `groot_oscar_models` to that exact directory. VM
providers mount the host cache root `/var/lib/blueprint/models` read-only at
`/models`; cloud-init refuses to start the thin worker if its manifest is absent.
The RunPod readiness artifact must be produced while that exact volume is
mounted, and must bind both its provider ID and mounted cache root:

```bash
python -m blueprint_pipeline.groot_oscar_model_cache verify \
  --root /workspace/.blueprint-model-cache/blueprint-groot-oscar-v1 \
  --provider-volume-id '<existing-volume-id>' \
  --out model-cache-verification.json
```

The canary admission rejects verification from a different volume or path.
It also rejects a provider volume whose reported capacity is smaller than the
cache's byte total established by full offline file verification.
It also injects the admitted `model_manifest_digest` into the container as
`BLUEPRINT_GROOT_OSCAR_EXPECTED_MODEL_MANIFEST_DIGEST`; the thin entrypoint
compares that exact digest before worker code starts.

## 3. Build the thin release

The source tree must be clean. The foundation must be digest-pinned. On push,
the builder inspects both registry manifests, proves that the release extends
the exact foundation layers, and limits only the new compressed release layers
to 2 GiB by default.

The old standalone release and monolithic closed-loop image scripts are
hard-disabled. Use the same canonical `cpu-build` packet flow so a
release build cannot bypass live capability or paid-resource admission.

The desired operational target is hundreds of MB; 2 GiB is a hard release
budget, not a claim that the first build already achieves the desired target.
The registry delta evidence is authoritative.

For the canonical automated path, dispatch
`.github/workflows/groot-oscar-thin-release.yml` with an exact 40-character
commit SHA and two versioned registry tags. The job only targets the
`blueprint-large-docker` native Linux/x86_64 runner class, requires 120 GiB
free before building, and requires the provisioner-owned
`/root/blueprint-builder-ready` file to exist on that live host. It packages
the clean commit into a byte-inventoried build
context, builds and pushes foundation first, feeds its resolved digest into the
thin release build, and uploads both registry diagnostics plus the thin-layer
contract. It cannot run on RunPod. A tag or a workflow success without the
`groot_oscar_thin_remote_build_result.v1` artifact and exact
`release_image_ref` digest is not release evidence.

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

## 4. RunPod canary preflight and admission

RunPod is the GPU serve/canary plane only. Before a create call, start the
name-bound watchdog in a separate process and wait until its evidence says
`armed`. It enumerates and deletes every matching canary resource at the hard
deadline, so even an ambiguous create response cannot orphan spend:

```bash
deadline="$(( $(date +%s) + 480 ))"
python -m blueprint_pipeline.groot_oscar_runpod_watchdog \
  --out-dir <canary-dir> \
  --pod-name-prefix blueprint-groot-oscar-canary-<attempt> \
  --deadline-epoch "$deadline" &
```

Then run the read-only provider preflight with the same name prefix:

```bash
python -m blueprint_pipeline.groot_oscar_runpod_preflight \
  --network-volume-id '<existing-volume-id>' \
  --model-cache-path /workspace/.blueprint-model-cache/blueprint-groot-oscar-v1 \
  --gpu-type-id 'NVIDIA A40' \
  --required-cuda-version 12.8 \
  --name-prefix blueprint-groot-oscar-canary-<attempt> \
  --watchdog-evidence <canary-dir>/groot_oscar_runpod_canary_watchdog.json \
  --model-volume-watchdog-handoff <model-volume-dir>/watchdog_handoff.json \
  --max-spend-usd 1.00 \
  --paid-mutation-authorized \
  --out <canary-dir>/runpod_preflight.json
```

The command makes only read-only calls. It requires RunPod's exact network
volume response (ID, size, and datacenter), a one-GPU stock row with an hourly
rate, provider-confirmed account-global zero billable resources, and a live independent
watchdog whose deadline is already counting down. It binds the canary create
request to the volume datacenter through `dataCenterIds` and to the image CUDA
family through `allowedCudaVersions`. The stock and price query itself is
filtered by both that exact volume datacenter and the release's required CUDA
version; global stock or stock on an incompatible CUDA host cannot satisfy
admission. Unknown stock, a missing volume, a
datacenter mismatch, an unverified cache, a tag instead of a digest, an absent
rate, a TTL above 30 minutes, a TTL whose maximum cost exceeds the cap, or an
unarmed watchdog all reject before allocation.

Run the admission/launcher first without `--execute` to inspect the exact bound
request. Add `--execute` only for the single authorized canary; the generic
adapter remains behind its separate `BLUEPRINT_ALLOW_RUNPOD_API_CALLS=true`
gate:

```bash
python -m blueprint_pipeline.paid_resource_allocator gpu-canary \
  --provider-launch-request <provider-launch-request.json> \
  --release-evidence <groot_oscar_thin_remote_build_result.json> \
  --model-cache-evidence <model-cache-verification.json> \
  --preflight-bundle <canary-dir>/runpod_preflight.json \
  --admission-out <canary-dir>/runpod_admission.json \
  --bound-request-out <canary-dir>/bound_provider_request.json \
  --adapter-output <canary-dir>/runpod_adapter_result.json \
  --pod-name blueprint-groot-oscar-canary-<attempt> \
  --probe-kind strict-policy-smoke \
  --campaign-budget-ledger <durable-campaign-budget.json> \
  --campaign-initial-spent-usd 14.557003 \
  --campaign-initial-used-gpu-seconds 15624 \
  --campaign-total-spend-cap-usd 20.00 \
  --campaign-wall-cap-seconds 21000 \
  --campaign-reservation-seconds 480 \
  --future-campaign-allowance-seconds 3500 \
  --authorize-reduced-canary-timeout \
  --campaign-max-hourly-rate-usd '<capacity-verified-rate-at-or-below-3.50>'
```

The `14.557003` USD and `15624` GPU-second values are the latest conservative
reconciled campaign baselines, not a fresh allowance. The USD baseline includes
the latest cache builder's measured cost and the retained volume's full bounded
pre-handoff exposure. The strict policy probe
reserves at most 480 GPU-seconds while preserving a separately bounded
3,500-second future campaign allowance; the combined remaining plan is 3,980
seconds against the authorized 21,000-second cumulative campaign ceiling. The
strict work/startup deadline is 420 seconds and the independent watchdog is 480
seconds, preserving 60 seconds for teardown and control-plane closure. The
explicit reduced-timeout flag records the operator's authorization for this
staged plan; 21,000 cumulative seconds is not permission for a single
cap-sized job.
After inspecting the dry-run evidence, rerun this exact command with
`--execute`; omitting any ledger identity/rate or staged-plan authorization
argument fails before provider mutation.

After startup-only evidence passes, the same canonical command may run the
fixed learned-action probe by adding `--probe-kind strict-policy-smoke`. That
mode ignores request-supplied commands, hard-bounds the probe to three fresh
GR00T/SONIC actions, and retains the same budget, handoff, pending-teardown, and
watchdog contracts. Its PASS is policy-model execution evidence only, not Isaac
task success or authority to run general paid episodes.

For the exact persistent carrier campaign, arm a fresh 18,600-second watchdog
and generate a fresh read-only preflight using the carrier volume's ID,
datacenter, handoff, and the same pod-name prefix. Then inspect this allocator
command without `--execute` before authorizing the one paid run:

```bash
python -m blueprint_pipeline.paid_resource_allocator gpu-canary \
  --probe-kind persistent-policy-wam-loop \
  --provider-launch-request <provider-launch-request.json> \
  --release-evidence <groot_oscar_thin_remote_build_result.json> \
  --model-cache-evidence <model-cache-verification.json> \
  --carrier-volume-admission <persistent-volume-evidence-dir>/carrier_volume_admission.json \
  --preflight-bundle <persistent-canary-dir>/runpod_preflight.json \
  --policy-observation <fresh-policy-observation.json> \
  --persistent-job-dir <persistent-canary-dir>/job \
  --admission-out <persistent-canary-dir>/runpod_admission.json \
  --bound-request-out <persistent-canary-dir>/bound_provider_request.json \
  --adapter-output <persistent-canary-dir>/persistent_carrier_result.json \
  --pod-name blueprint-groot-oscar-canary-persistent-<attempt> \
  --campaign-budget-ledger <durable-campaign-budget.json> \
  --campaign-initial-spent-usd '<current-reconciled-spend>' \
  --campaign-initial-used-gpu-seconds '<current-reconciled-gpu-seconds>' \
  --campaign-total-spend-cap-usd 20.00 \
  --campaign-wall-cap-seconds 36000 \
  --campaign-max-hourly-rate-usd '<capacity-verified-non-h100-rate>'
```

After the dry-run admission and bound request are reviewed, rerun the exact
command with `--authorize-persistent-carrier-campaign --execute`. This lane
rejects H100, binds a 240 GiB container disk and the exact 120 GiB volume, and
reserves one 18,600-second watchdog window against the unchanged USD 20 total
cap. Completion requires one Pod to produce exactly five fresh policy calls,
four learned WAM generations, five distinct actions, no replay or resume, and
21 media files, followed by verified GPU teardown and provider-lane return.
Those facts prove the technical loop only; semantic task success, physical
robot readiness, and generated-world rank fidelity remain separate claims.

The launcher refuses to rewrite a tag into an admitted digest or silently pick
another GPU. On live submission, before the provider adapter is reachable, it
reruns every mutable read-only check and writes
`runpod_preflight_launch_refresh.json`. This refresh revalidates the volume,
datacenter/CUDA-filtered capacity, account-global zero billable resources, and the live
watchdog process with its remaining deadline. The watchdog PID is accepted only
when its live process argv (Linux `/proc` or the fail-closed macOS `ps -ww`
fallback) is the canonical name-bound watchdog with the
exact admitted pod prefix and deadline; any changed, forged, or expired evidence
rejects without a create call. It then records `warm_serve_pod.json` beside the
watchdog evidence so the independent watchdog can terminate the exact pod and
prove inventory absence at its hard deadline.

## 5. Warm-worker admission

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
