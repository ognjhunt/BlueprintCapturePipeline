# NVIDIA SIGGRAPH 2026 Experimental Integration Runbook

Date: 2026-07-21

This runbook covers optional, replaceable NVIDIA experiments. None of these
commands upgrades simulator, task-success, policy-ranking, deployment, or
real-world claims. Keep all NVIDIA prerelease dependencies outside the core
Blueprint environment and use only privacy-safe, pipeline-derived inputs.

## Source and Version Pins

Refresh these pins after SIGGRAPH closes before an external attempt:

| Component | Current reviewed pin | License boundary |
| --- | --- | --- |
| ovrtx | wheel `0.4.0.346409`, source `4b9a5fe6f8becf6c5ff031e167cd4201054a96ce` | NVIDIA proprietary SDK license |
| ovstage | wheel `0.1.0.346039`, internal ovrtx dependency | NVIDIA proprietary SDK license |
| ovphysx | wheel `0.4.13`, source `b4b286abff6f2b3debd1d1acb120dc428765cf2e` | BSD source plus NVIDIA binary terms |
| usd-convert-gsplat | package `0.1.15`, source `621017ebf78394488260c70ec4eadd70ff621131` | Apache-2.0 and CC-BY-4.0 |
| SimReady Foundation | validator `2026.04.1`, source `4d9f3bb2897bc16ff99978774138c68f4e30ecbf`; profile version remains an independent required pin | Apache-2.0 |
| Cosmos 3 Edge | `nvidia/Cosmos3-Edge`, exact checkpoint and code revisions required | OpenMDW-1.1 |

Complete `docs/nvidia_siggraph_post_conference_source_review.template.json` on
or after July 24, validate every component row, then pass that review to
`python -m blueprint_pipeline.nvidia_siggraph_policy`. A missing, early, or incomplete
refresh blocks activation.

## SimReady Advisory Validation

Prepare a disposable environment from a pinned checkout:

```bash
scripts/setup_simready_validator_env.sh \
  /path/to/simready-foundation \
  /path/to/disposable-simready-venv \
  <exact-source-revision>
```

Invoke the Blueprint adapter with the official CLI wrapper as its worker. The
worker command must include placeholders for `{input}` and `{output}`; the
adapter also supplies the requested profile and profile version.

```bash
python -m blueprint_pipeline.external_simready_validation \
  --capture-root <capture-root> \
  --input-usd <privacy-safe-pipeline-usd> \
  --validator-command '/path/to/disposable-simready-venv/bin/python <repo>/scripts/run_simready_validator_worker.py --source-root /path/to/simready-foundation --input {input} --output {output} --profile {profile} --profile-version {profile_version} --expected-validator-version <version>' \
  --profile Prop-Robotics-Neutral \
  --profile-version <profile-version> \
  --validator-version <validator-version> \
  --validator-source-revision <exact-source-revision> \
  --license-compatible \
  --repeat-runs 2 \
  --source-manifest-id <manifest-id>
```

The adapter prohibits transformations, verifies that the input digest did not
change, and emits request, result, and claim-boundary artifacts under
`pipeline/simready/`.

Before promoting any rule beyond advisory use, create a frozen calibration
manifest containing known-valid and intentionally malformed cases plus signed
expert review, then run:

```bash
python -m blueprint_pipeline.simready_rule_calibration \
  --manifest <calibration-manifest.json> \
  --evidence-root <immutable-evidence-root> \
  --output <capture-root>/pipeline/simready/rule_calibration.json \
  --authorize-rule <rule-id> \
  --human-promotion-approval-id <approval-id>
```

Only rules with zero observed false positives and false negatives across the
frozen reviewed corpus can be authorized.

## ovrtx and ovphysx Preflights

These workers require a supported Linux system. ovrtx additionally requires an
RTX-capable GPU. Create isolated environments; never target `.venv`:

```bash
scripts/setup_omniverse_library_envs.sh \
  /path/to/disposable-ovrtx-venv \
  /path/to/disposable-ovphysx-venv
```

Both the explicit CLI flag and environment gate are required:

```bash
export BLUEPRINT_ALLOW_OMNIVERSE_EXTERNAL_PREFLIGHT=true
```

An ovrtx configuration should pin image dimensions, camera intrinsics, warmup
frames, delta time, render-product paths for lidar/radar, and semantic labels.
Set `episode_mode` and `episode_sample_time_seconds` for an animated episode.
Blueprint automatically adds ParticleField, dynamic-update, semantic-map,
visibility, lidar, and radar checks when the scene or request requires them.

```bash
python -m blueprint_pipeline.omniverse_library_preflight ovrtx \
  --capture-root <capture-root> \
  --input-usd <privacy-safe-pipeline-usd> \
  --worker-command '/path/to/disposable-ovrtx-venv/bin/python <repo>/scripts/run_ovrtx_preflight_worker.py --input {input} --output {output} --output-dir {output_dir} --config {config} --mode {mode} --source-revision {source_revision} --modality rgb --modality depth --modality semantic_segmentation --modality semantic_id_map' \
  --component-version 0.4.0.346409 \
  --source-revision 4b9a5fe6f8becf6c5ff031e167cd4201054a96ce \
  --license-id NVIDIA-Proprietary \
  --license-compatible \
  --configuration <ovrtx-config.json> \
  --required-modality rgb \
  --required-modality depth \
  --required-modality semantic_segmentation \
  --required-modality semantic_id_map \
  --allow-external-preflight
```

The ovphysx configuration should pin fixed time step, number of steps, gravity,
mass/friction bounds, snapshot steps, and any articulation prim path:

```bash
python -m blueprint_pipeline.omniverse_library_preflight ovphysx \
  --capture-root <capture-root> \
  --input-usd <privacy-safe-pipeline-usd> \
  --worker-command '/path/to/disposable-ovphysx-venv/bin/python <repo>/scripts/run_ovphysx_preflight_worker.py --input {input} --output {output} --output-dir {output_dir} --config {config} --mode {mode} --source-revision {source_revision}' \
  --component-version 0.4.13 \
  --source-revision b4b286abff6f2b3debd1d1acb120dc428765cf2e \
  --license-id BSD-3-Clause-AND-NVIDIA-Binary \
  --license-compatible \
  --configuration <ovphysx-config.json> \
  --allow-external-preflight
```

Each adapter runs cold and warm attempts, validates component/config identities,
contains outputs under the capture pipeline root, and requires repeatable output
digests. Compare both results to an accepted same-scene Isaac baseline with
`python -m blueprint_pipeline.omniverse_library_preflight benchmark`. Retain a library only when the
comparison demonstrates a useful failure class or a measured runtime/cost gain.
The stronger
`python -m blueprint_pipeline.omniverse_library_preflight benchmark-suite --manifest <suite.json>`
requires valid and negative fixtures, exact scene-digest equality across the
library and Isaac evidence, cold/warm CPU and GPU memory, and matching detection
of the expected failure class.

## Gaussian-Splat Conformance Oracle

Install the official converter from a pinned checkout into a disposable
environment with `scripts/setup_gsplat_conformance_env.sh`. Then run:

```bash
python -m blueprint_pipeline.gsplat_conformance \
  --capture-root <capture-root> \
  --source-ply <privacy-safe-pipeline-ply> \
  --converter-command '/path/to/converter-venv/bin/python <repo>/scripts/run_usd_convert_gsplat_worker.py --input {input} --oracle-output {oracle_output} --report {output} --expected-version {converter_version} --source-revision {source_revision}' \
  --converter-version 0.1.15 \
  --source-revision 621017ebf78394488260c70ec4eadd70ff621131 \
  --license-compatible
```

This compares units, up axis, extent, positions, scales, quaternions, opacity,
and spherical-harmonic attributes. It does not replace Blueprint's writer or
prove that Isaac/ovrtx rendered the result correctly.

## Cosmos 3 Edge

Cosmos 3 Edge has a separate experimental adapter. Build the official
`cosmos-framework` source at an exact revision in an isolated Linux/CUDA
environment:

```bash
scripts/setup_cosmos3_edge_env.sh \
  /path/to/cosmos-source \
  /path/to/disposable-cosmos-edge-venv \
  <exact-code-revision> \
  cu128-train
```

Prepare a frozen manifest of privacy-safe cells with mode-specific official CLI
inputs, a local checkpoint with a recorded SHA-256 digest, and a pinned runtime
configuration. Both the CLI flag and this gate are mandatory:

```bash
export BLUEPRINT_ALLOW_COSMOS3_EDGE_EXPERIMENT=true
```

```bash
python -m blueprint_pipeline.cosmos3_edge_experiment \
  --capture-root <capture-root> \
  --frozen-manifest <edge-frozen-benchmark.json> \
  --worker-command '/path/to/disposable-cosmos-edge-venv/bin/python <repo>/scripts/run_cosmos3_edge_worker.py --input {input} --output {output} --output-dir {output_dir} --config {config} --mode {mode} --cell-id {cell_id} --model-revision {model_revision} --code-revision {code_revision} --checkpoint {checkpoint} --checkpoint-sha256 {checkpoint_sha256} --configuration-sha256 {configuration_sha256}' \
  --model-revision <exact-model-revision> \
  --code-revision <exact-code-revision> \
  --checkpoint <local-checkpoint> \
  --configuration <edge-configuration.json> \
  --license-compatible \
  --repeat-runs 2 \
  --allow-experiment
```

The harness always requests forward dynamics, inverse dynamics, and reasoning
as separate attempts and emits the existing evaluator-compatible runtime
evidence. It rejects Unitree G1 7D inverse/policy configuration because the
released model card does not list that action encoding. A completed attempt is
still generated-model evidence, not forward/inverse consistency, task success,
rank fidelity, or physical correctness.

After the configured Blueprint evaluator emits a validated runtime receipt and
a frozen accepted-anchor scorecard, build the separate qualification:

```bash
python -m blueprint_pipeline.cosmos3_edge_qualification \
  --attempt-manifest <capture-root>/pipeline/cosmos3_edge_experiment/attempt_manifest.json \
  --evaluator-runtime-receipt <validated-runtime-receipt.json> \
  --scorecard <frozen-blueprint-scorecard.json> \
  --expected-evaluator-id <configured-evaluator-id> \
  --output <capture-root>/pipeline/cosmos3_edge_experiment/qualification.json
```

Qualification measures grounding, abstention, ranking, and calibrated failure
detection on Blueprint data. It never inherits Nano results and never changes a
default without a separate owner decision.

## Deferred Asset-Conditioning Reviews

Use `python -m blueprint_pipeline.nvidia_asset_conditioning_review` only for a named buyer need.
CAD-to-SimReady requires evidence for import, minimum USD validation, material
proposal, physics proposal, conformance, and report stages. Content Agents and
SimReady Blender require a human approval receipt. All variants preserve the
original digest and mark candidate physics/material/semantic data as proposals,
not capture or physical truth.

## Paid Resources and Closeout

Do not launch provider-specific builders or canaries directly. Any paid CPU,
model-volume, or GPU allocation must use:

```bash
python -m blueprint_pipeline.paid_resource_allocator cpu-build <arguments>
python -m blueprint_pipeline.paid_resource_allocator model-volume <arguments>
python -m blueprint_pipeline.paid_resource_allocator gpu-canary <arguments>
```

An external attempt is incomplete until the exact allocation and the global
provider inventory both prove zero active resources. Record spend separately
from runtime success, semantic output quality, evaluator evidence, and ranking.
Pass the admission context into every external command with
`--resource-context`. After teardown, create the immutable closeout:

```bash
python -m blueprint_pipeline.nvidia_experiment_resource \
  --resource-context <admitted-resource-context.json> \
  --teardown-evidence <provider-teardown-evidence.json> \
  --output <capture-root>/pipeline/nvidia_experiment_resource_closeout.json
```

The closeout rejects missing allocation identity, nonzero exact-attempt or
global inventory, ongoing hourly burn, and unreconciled billing.
Re-run or finalize the applicable adapter with both `--resource-context` and
`--resource-closeout`; the shared loader verifies the context digest plus
provider, allocation, and attempt IDs before its stop-rule evaluation can treat
teardown as proven.

## Completion Evidence

After the final targeted and full test runs, record a
`nvidia_siggraph_2026_verification_receipt.v1` containing the exact command and
zero exit status. Then run
`python -m blueprint_pipeline.nvidia_siggraph_completion` to map every phase and
deferred lane to its implementation evidence. The matrix deliberately keeps
official NVIDIA execution, same-scene Isaac comparison, Edge rank fidelity,
and any provider execution unproven until their own evidence exists.

## Promotion Rules

Keep every lane advisory until all applicable policy-registry stop rules pass.
In particular, a successful process exit or generated image/video never proves
simulator parity, contact correctness, task success, policy quality, ranking,
deployment readiness, or real-world correlation.
