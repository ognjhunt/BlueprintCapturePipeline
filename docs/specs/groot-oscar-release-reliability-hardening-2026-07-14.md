# GR00T + OSCAR release reliability hardening

Status: next-release implementation. It does not modify or upgrade evidence for
the immutable `515b095f` / `sha256:75bc686e...` campaign.

## Incident evidence and claim boundary

The July 13–14 G4 campaign passed allocation, immutable pull, CUDA/RTX, review
rendering, and sealed image health, then stopped before smoke. Preserved logs
proved three independent runtime blockers:

1. GR00T's tokenizer path called Hugging Face `model_info` for
   `nvidia/Cosmos-Reason2-2B` while `HF_HUB_OFFLINE=1` was correctly enforced.
2. GEAR-SONIC could not find `policy/release/model_decoder.onnx` (and therefore
   could not initialize its control policy).
3. the persistent Isaac task executor could not wrap `/World/G1` because the
   staged kitchen did not contain that robot prim.

Those are image/bundle closure failures. The strict canary remains runtime and
review-media proof only; it is not simulator stepping, learned action, semantic
task success, buyer acceptance, deployment approval, or physical-robot proof.

## Official build and supply-chain evidence

`scripts/build_push_groot_oscar_closed_loop_image.sh` now treats an authorized
release build as the single BuildKit push and enables both BuildKit SBOM and
max-mode provenance attestations. The independently retained SPDX document is
always produced from `syft registry:<immutable-digest-ref>`; daemon auto-source
selection is impossible. The build fails closed when the SPDX JSON, digest
binding, layer inventory, registry diagnostic, or runtime-user smoke is absent.

Disk admission accounts for unpacked build scratch, registry-scan scratch, and
a fixed reserve. Its defaults reflect the observed 46 GiB compressed / 176 GiB
unpacked closure and therefore require about 293 GiB, rounded up to the release
runner's 300 GiB gate.

The dedicated workflow runs only for actual worker-closure paths or a manual
release request. Documentation and provider-control-plane changes do not build
the GPU image. Evidence is retained for 90 days.

## Pinned GCP host image

`infra/gcp/g4_host_image/g4-host.pkr.hcl` requires an exact date-pinned Ubuntu
image, an exact NVIDIA vGPU driver URL plus SHA-256, an exact NVIDIA
container-toolkit package version, one exact worker digest, and its protected-main
source SHA. The worker closure is preloaded only into Docker's content-addressed
store; its files are never independently installed onto the host. The image
configures Docker's NVIDIA runtime and installs a boot-time self-test. The
self-test records driver, GPU, Docker, toolkit, source, and locally resolved
worker-digest identity. Application, model, task, capture, and policy files are
forbidden outside the immutable worker-image or hashed job-bundle contracts.

The Packer manifest is the immutable machine-image identity input for campaign
configuration. A host image is not considered ready until a real G4 boot has
returned a passing `blueprint_g4_host_self_test.v1` artifact.

## Regional mirror and image size

`build_regional_mirror_plan` accepts only an immutable source digest. It emits
`crane copy` operations, recurring storage exposure, and a mandatory
`registry_mirror_equivalence.v1` post-copy gate. Equivalence compares every
platform child manifest digest; matching tags are never evidence. The plan uses
no idle compute and retains only active release closures. At 46 GiB and a
hypothetical $0.10/GiB-month, two regional copies expose about $9.20/month;
operators must replace that assumption with the provider's current quoted rate
before mutation.

Every official build writes `groot_oscar_image_layer_report.v1`, sorted by
compressed layer size. Optimization work must remove build-only dependencies,
deduplicate checkpoint/framework bytes, and preserve offline execution.
External model/asset blobs are permitted only when immutable hashes/revisions
are verified before startup; hidden runtime downloads are forbidden.

The `fca4712e` registry diagnostic measured 47,101,357,226 compressed bytes.
Five layers explain essentially the whole closure: 10,585,790,213 bytes for
Isaac Sim; 14,083,497,680 for sealed model checkpoints; 10,367,336,848 for the
GR00T Python/CUDA runtime; 7,223,826,406 for WBC plus the build-time CUDA
toolchain; and 4,322,370,426 for the OSCAR Python/CUDA runtime. The build log
proves that upstream WBC dependency setup installed a CUDA 12.4 compiler and
development runtime into the final carrier, while OSCAR installed Torch 2.10
with CUDA 12.8 wheels and GR00T installed Torch 2.7.1 with a different CUDA
12.8 wheel set. This is the first-principles source of the 46.8 GB closure.

The next-candidate Dockerfile now moves WBC compilation to a disposable
Isaac-based builder stage and copies only its runtime tree, ONNX Runtime,
`libcudart`, and pinned TensorRT/runtime dependencies. The final stage runs
`ldd` fail-closed against `g1_deploy_onnx_ref`; nvcc, CUDA headers/static
development archives, WBC git objects, and TensorRT development packages are
not copied into the carrier. The two Python environments may share only
byte-identical CUDA artifacts after both policy services pass real-GPU ABI
tests; their incompatible Torch versions must not be symlinked together merely
to improve size. Until this candidate is built and tested, the official build
enforces ceilings of 48,000,000,000 total compressed bytes and 15,000,000,000
bytes for any one layer, preventing silent growth above the measured release.

Images at or above 20 GiB are not eligible for an on-demand paid cold pull.
Their immutable campaign config must include `preloaded_worker_image.v1`
evidence binding the exact digest to a runtime-health-tested host-image/cache
identity before allocation. The allocated host must re-prove that the digest
was already local and that no cold pull happened during the paid campaign.
Registry resolution by itself is explicitly insufficient. This converts image
delivery from an opaque provider wait into a release prerequisite.

## Provider-neutral campaign state machine

`CampaignMachine.run()` is the repository-owned control-plane interface. Its
immutable config contains source/image identity, spend exposure, maximum
provider lifetime, independent seeds, and every stage deadline. Atomic
checkpoints make stages idempotent and resumable. Inventory is checked before
allocation, a duplicate allocation blocks before mutation, and budget admission
uses worst-case rather than expected spend.

The provider seam has concrete GCP and AWS adapter roles plus hermetic test
operations. All failure paths call the one recorded teardown owner. Provider
absence and final inventory zero are both required; an ambiguous delete remains
blocked and can never release billing proof.

Smoke is a hard stage barrier. A non-passing smoke stage prevents the episode
stage from being called. Stage results remain separate from semantic task and
buyer/public proof.

## Startup timing and SLOs

The required monotonic milestone set is:

1. VM allocation
2. driver readiness
3. container runtime readiness
4. image pull
5. container start
6. health
7. Isaac startup
8. policy readiness
9. first simulator step
10. first learned action
11. first frame
12. artifact upload

`evaluate_release_slos` fails on missing/out-of-order timing or when cached
policy readiness exceeds the initial five-minute target. The release contract
also requires failure classification within three minutes, forbids opaque waits,
and records that documentation/control-plane-only changes do not rebuild the
GPU image. Targets become claims only after a real campaign measures them.

## Same-allocation canary continuation

The optional `same_allocation_canary_handoff.v1` schema binds source SHA, image
digest, allocation key/id, launch nonce, the single teardown owner, runtime
health, valid review media, and the original provider-start epoch. The
continuing controller must find and adopt that exact allocation in live provider
inventory; it never allocates a new VM, and its lifetime cap continues from the
original paid allocation rather than resetting when the controller resumes.
Continuation requires that the allocation is still owned and no teardown has
been requested. The state machine may skip only its duplicate canary stage;
smoke and all task-evidence gates remain mandatory. A canary that has already
been torn down can never be reused.

## Release acceptance

- Hermetic unit/contract tests pass for budget refusal, duplicate allocation,
  smoke fail-closed behavior, resume ownership, teardown ambiguity, registry
  source selection, disk admission, mirror equivalence, layer reporting, timing,
  SLOs, GCP/AWS adapter parity, and canary handoff.
- Shell/Packer files parse and no secret value or signed URL is persisted.
- A future image closure must add build-time service preflights that reproduce
  the three July campaign failures before another paid run is authorized.
- Live G4 host-image readiness, regional registry copies, cached-readiness SLO,
  fixed GR00T/GEAR/Isaac startup, smoke, episodes, and semantic success remain
  unproven until separately executed and retrieved.
