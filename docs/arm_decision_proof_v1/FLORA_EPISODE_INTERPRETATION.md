# FLORA episode interpretation

FLORA is an optional research interpreter for **ADP-009D / Day 28**. It grades
already sealed episode videos and compares its judgments with retained
deterministic simulator scores. It cannot change those scores, policy ranking,
promotion, or the Day-28 acceptance gate. This extends
[`EPISODE_INTERPRETATION.md`](EPISODE_INTERPRETATION.md) through the existing
interpreter, rights, receipt, delivery, and paid-resource contracts.

The completion artifact is a digest-bound `flora_episode_benchmark_report.v1`,
its authenticated episode receipts and raw model outputs, runtime evidence,
and provider teardown and billing receipts. Fixture tests and input preflight
are not model execution. The bounded additions implement FLORA's three batch
formats, upstream preprocessing, and immutable model/source verification.

## Pinned release and claim limits

The implementation pins [`nvidia/Flora-Rewarder` at
`7de68d22c4a9decec5ddb7f86ddcaee8f82d3367`](https://huggingface.co/nvidia/Flora-Rewarder/tree/7de68d22c4a9decec5ddb7f86ddcaee8f82d3367).
It verifies source, processor files, and eleven weight shards by SHA-256 before
inference. The release's research/noncommercial terms do not authorize a
commercial service rollout. Retain exact model-term and video-disclosure
authority with any external experiment.

Three upstream modes run once each over the same cohort:

| Mode | Retained output | Blueprint interpretation |
| --- | --- | --- |
| Terminal | Score 1–5, rationale, token probabilities, sampled-frame count | Apparent completion only at score 5 |
| Dense | Per-frame quality 1–3, failure/suboptimal spans and rationale | Any failure-quality span prevents an apparent pass |
| Rubric | Frozen rubric, sampled answers, selected answer and score | Full rubric completion required for apparent pass |

The combined apparent outcome requires all three modes to pass. Terminal and
rubric failure together produce apparent incompletion; conflicting judgments
or invalid outputs produce uncertainty/abstention. This mapping is an explicit,
uncalibrated integration rule, not a validated probability of success. Invalid
dense clips and invalid rubric values are rejected instead of inheriting
upstream fallback/clamping behavior.

Only the frozen instruction, robot/camera descriptions, rubric, and exact
review videos enter model jobs. Policy identities, reference labels, simulator
state, contacts, and event ledgers are withheld. Lossless policy frames remain
required and verified source evidence; derived videos are what FLORA sees.
Sampling can miss events, and visual agreement is not physical truth.

## Prepare and run a retained comparison

Keep output separate from the source run. Use the latest sealed result with
recorded score corrections explicitly bound. When the original sealed run has
a later published correction, pass `--corrected-reference` to `prepare` with
its exact `score_correction.json` or `website_ingest.v2.json`. The benchmark
validates that packet and its episode receipts against the original run,
preserves original score bytes, and labels both reference identities. Never
edit an old score receipt to make it usable. Missing media, traces, or confirmed
task contracts remain explicit gaps.

Create a `policy_canary_episode_interpreter_profile.flora.v1` with the exact
task-contract digest, instruction, camera inventory, and rubric. Maximum step
scores must sum to one; allowed levels use at most two decimal places. Seal
`profile_digest` with
`decision_evidence_contracts.canonical_digest(profile, digest_field="profile_digest")`.
`validate_flora_interpreter_profile` is the executable schema. Runtime paths,
generation seed, tensor-parallel size, and rubric sample count are sealed.
External profiles also identify the actual `compute_provider_id`; the model
vendor does not substitute for the recipient of disclosed videos.

```bash
python -m blueprint_pipeline.flora_episode_benchmark prepare \
  --session-result /path/to/sealed-session-result.json \
  --evidence-root /path/to/retained-evidence \
  --profile /path/to/profile.json \
  --output-dir /path/to/flora-input

python -m blueprint_pipeline.flora_episode_benchmark preflight \
  --manifest /path/to/flora-input/manifest.json \
  --profile /path/to/flora-input/profile.json \
  --rights-root /path/to/flora-input/rights \
  --output-dir /path/to/flora-output
```

Preparation copies source evidence and verifies result, episode, frame, video,
and contract digests. It records selected episodes including gaps. Rights
attestations use the existing episode-interpreter format, named
`<input-bundle-sha256-without-prefix>.json`. Each authorizes the exact recipient,
runtime, model, input digest, and artifact roles. A prior model's attestation
does not automatically authorize FLORA or a different compute provider.

On already admitted compute, replace `preflight` with `run`. Runtime checks
source/model bytes, dependencies, CUDA devices, and every episode's decoding
and preprocessing before inference. It retains raw JSONL, subprocess logs,
prompt/settings identities, input digests, and per-episode metadata. Replay
reads sealed receipts and report. An interrupted attempt without complete
receipts is ambiguous and cannot automatically retry.

After collecting provider outputs, use `run --raw-artifact-root /path/to/raw`
to rebase raw evidence onto its retained local directory. Replay verifies every
raw file and its hash, and binds native model rows back to the receipt.
Closeout supports the same relocation with `BLUEPRINT_FLORA_RAW_ARTIFACT_ROOT`.
Missing or changed raw evidence creates a separate integrity abstention; it
does not rewrite an old receipt or trigger another inference attempt.

Package the portable input tree using `python -m
blueprint_pipeline.flora_provider_bundle --help`. For a rented GPU use only
the canonical allocator:

```bash
python -m blueprint_pipeline.paid_resource_allocator gpu-canary \
  --provider vast --probe-kind flora-episode-benchmark \
  --expected-source-commit <immutable-commit> \
  --experimental-branch-diagnostic \
  --flora-bundle /path/to/flora.zip \
  --flora-bundle-receipt /path/to/flora.receipt.json \
  --flora-attempt-authority /path/to/attempt-authority.json \
  --flora-job-dir /path/to/flora-job \
  --admission-out /path/to/flora-job/admission.json \
  --adapter-output /path/to/flora-job/result.json
```

The default is dry run. `--execute` requires a clean immutable source commit,
exact bundle and digest-pinned image, current rights/terms authority, explicit
aggregate budget and expiry, live capacity/spend checks, independent watchdog,
and teardown. The bounded lane allows one allocation, exactly two admitted
GPUs, and no automatic paid retry. The worker downloads the pinned public
model onto admitted compute. Fleet limits are ceilings, not task authorization.
The attempt authority must set `provider_output_archive_max_bytes` to
1,000,000,000 and reserve at least 3,000,000,000 provider upload bytes for the
shared transport's maximum of three PUT attempts. The worker enforces the
archive limit before upload. Download admission must include the pinned image,
model, dependency wheels, input bundle, and a bounded transfer retry reserve.

The worker uses the shared `flora_dependency_bootstrap` helper to install only
the four hash-pinned add-ons required by the pinned vLLM base image:
Transformers 5.8.1, qwen-vl-utils 0.0.14, Decord 0.6.0, and PyAV 16.1.0. The
approved PyAV artifact is the CPython 3.12 Linux x86_64 wheel (41,174,337 bytes),
verified against [PyPI's release metadata](https://pypi.org/pypi/av/16.1.0/json).
The CPU diagnostic and worker both require this exact PyAV version. The helper leaves
Torch and vLLM unchanged and retains an installation receipt even on failure.
Use that same helper with `preflight_flora_cpu_imports` to rehearse imports and
the exact videos' preprocessing on existing CPU compute before renting GPUs.
This rehearsal always reports `execute_ready: false`; the GPU worker must
still verify the weight shards, devices, and complete runtime preflight.
CPU preprocessing honors the profile timeout up to 1,800 seconds; the GPU
diagnostic keeps its 300-second cap. Each processed episode emits a JSON progress
line and updates `diagnostic.json`. A partial checkpoint remains incomplete and
cannot establish readiness, including when the process times out.

## Read the comparison

JSON and Markdown reports retain episode/candidate comparisons, coverage,
abstentions, missing references, confusion counts, false-pass and false-fail
rates, and descriptive Wilson intervals. Error-rate denominators include only
classified episodes with the corresponding available reference. Inspect
coverage with those rates. A cohort containing only failed reference episodes
cannot estimate false-fail rate. Correlated scenario cells weaken the
independence assumption behind the intervals.

Reports distinguish real model execution from hermetic fixtures. Fixture
reports establish wiring only. A real twenty-episode simulator comparison
remains a development diagnostic; it does not qualify deployment accuracy,
a new site, or held-out physical performance.

## Verification

Focused FLORA tests cover strict upstream parsing, pinned runtime and
preprocessing, blinded jobs, immutable staging, authentic receipt/report
binding, cohort batching, fixture-versus-model evidence, replay, rights, and
bounded allocation. Existing policy-canary lifecycle and provider import
closure tests protect shared worker/client and sealed-bundle behavior before
a paid attempt. Hermetic tests are never FLORA grading results.
