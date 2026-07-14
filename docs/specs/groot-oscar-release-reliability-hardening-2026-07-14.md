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
image, an exact NVIDIA vGPU driver URL plus SHA-256, and an exact NVIDIA
container-toolkit package version. The image configures Docker's NVIDIA runtime
and installs a boot-time self-test. The self-test records driver, GPU, Docker,
and toolkit identity. Application, model, task, capture, and policy bytes are
explicitly forbidden from the host image and remain in the worker-image or
hashed job-bundle contracts.

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
digest, allocation key/id, launch nonce, runtime health, and valid review media.
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
