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
Both admission checks use a 240-second retry window for a fresh billing snapshot
when the sole blocker is an expired spend lock. Probes retain their own network
timeouts. The lock's age limit and all
other gates stay unchanged; the initial check refreshes live capacity and
authority evidence on each retry. This wait cannot allocate a resource or
authorize another paid attempt.
Resource labels use `blueprint-task-evaluation-flora-` so the deployed spend
guard can verify the independent watchdog's ownership record.

After model download, the worker emits count-only `BLUEPRINT_FLORA_PROGRESS`
records from the exact cohort's preflight and newline-terminated native outputs.
The transport advances its liveness clock only when that stage/count increases;
duplicate markers, reordered or truncated log tails, and SSH chatter cannot
substitute for progress. These records always say `benchmark_complete=false`.
Only the authenticated final receipts and report determine execution completion.
Observation errors preserve the benchmark's result and remain subject to the
unchanged no-progress, batch-timeout, and independent resource deadlines.

The bundle also ships the existing pinned `rfc8785` 0.1.4 contract dependency
and its license through `stage_policy_contract_dependencies`. Its four files
are digest-bound with the sealed source; this adds no provider package download.
Before dependency installation or model download, the worker runs the actual
`flora_episode_benchmark preflight` CLI in its interpreter against that sealed
source. Socket access is disabled. The full corrected-reference, media, episode,
and rights validation must reproduce the bundle's exact preflight record,
including all selected episodes, before execution may continue.

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

The 2026-10-21 paid attempt exposed a process defect before inference: the
corrected-reference loader lazily imported `rfc8785`, which was absent from the
worker image and FLORA bundle. The previous CPU probe exercised imports and
video preprocessing without executing that loader. The original traceback is
retained in `attempts/attempt_001/immutable_execution/benchmark.log` for the
`retained20-ce8bfa13fa58` job. The bundled dependency and complete offline input
preflight repair this missing coverage; neither the repair nor a CPU replay is
evidence of a successful FLORA inference run.

The additional authorized attempt at `a92944f73137` was refused by Vast with
HTTP 400 `no_such_ask` before allocation: the selected offer had become
unavailable. Authenticated inventory confirmed no instance for its exact label;
no model download or inference occurred. Its single-use authority remains
consumed, and the zero-retry limit was preserved. Staged-object cleanup now
uses the same strict definite-refusal and verified-absence predicate as the
watchdog. Ambiguous create outcomes still retain remote results and recovery
access. The adapter's transfer estimate is a reservation, not a posted charge.

## Observed retained-run test, 2026-10-21

The third attempt executed source `c340519c96d8102a97e1db118ce2b1dcffff363c`
on two H200 GPUs. It passed the exact twenty-episode input and GPU media
preflights and generated twenty terminal, twenty dense, and fifteen rubric
records. The outer adapter then reported `no_log_progress_timeout` after
1,830 seconds without new stdout, even though the rubric JSONL was advancing.
It destroyed the instance at 05:33:12 UTC. This happened before the worker's
shared batch timeout and independent resource deadline.

The completed native records were recovered from digest-verified observation
snapshots. No final output archive, authenticated episode receipts, or complete
benchmark report was collected. Native diagnostics must remain labeled partial;
do not manufacture the absent completion artifacts or label five unobserved
rubric episodes as model failures. The artifact-count monitoring change above
encodes this observed failure and passed focused transport, worker lifecycle,
and sealed-import checks; it was not deployed into the frozen live attempt.

Against the separately sealed corrected references (seven simulator successes
and thirteen failures), terminal mode rated nine reference failures as complete
and missed three reference successes. Dense mode found a failure span in
nineteen episodes, including six eventual reference successes. Process mistakes
can coexist with final success, so dense labels are not terminal outcomes.
Several dense rationales describe unsuccessful picking or grasping despite the
frozen instruction to push the mug onto the target.

Of the fifteen rubric records, fourteen passed strict normalization. The GROOT
bounded-physics episode had only eight of nine samples parsed and was invalid
with `flora_rubric_samples_missing_or_invalid`. For the fifteen episodes with
raw records in all three modes, the unchanged combined rule produced zero pass
predictions, four correct incomplete predictions, one incorrect incomplete
prediction, and ten abstentions, including the invalid rubric record. The five
episodes without rubric records remain unobserved in the combined comparison.

These partial findings do not establish an advantage over the existing
deterministic scorer or the optional Argus/OpenAI interpreter; the latter was
not a paired comparator in this test.

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
