# NVIDIA SIGGRAPH 2026 Stack Impact

Date: 2026-07-21

Research status: point-in-time assessment while SIGGRAPH 2026 is still in
progress. The conference runs July 19-23, 2026. Recheck release notes, licenses,
API versions, and repositories after the conference closes before starting a
production integration.

Implementation status: Blueprint-owned request/result schemas, isolated-worker
adapters, policy gates, experiment harnesses, and tests are implemented in this
repository. No NVIDIA package was installed into the core environment, no
checkpoint was downloaded, no provider or paid GPU was launched, and no
external NVIDIA runtime or Blueprint qualification claim is proven. See
`NVIDIA_SIGGRAPH_2026_IMPLEMENTATION_RUNBOOK.md` for the exact canary boundary.

## Decision Summary

The SIGGRAPH announcements justify bounded experiments, not a wholesale stack
change.

The best near-term improvement is NVIDIA's open SimReady Foundation validator.
It can add profile-aware OpenUSD checks to Blueprint's existing local SimReady
lane, whose current validation is intentionally limited to package completeness
and coarse review outputs. The validator should run in an isolated environment,
produce versioned evidence, and begin as an advisory gate.

The next most useful experiments are:

1. evaluate `ovrtx` as a lightweight sensor-preflight worker for camera, lidar,
   radar, segmentation, and visibility checks before a full Isaac Sim run;
2. evaluate `ovphysx` as a Kit-free physics/contact smoke test before a full
   simulator launch;
3. qualify Cosmos 3 Edge as a separate experimental WAM/reasoner profile on the
   same frozen Blueprint cells used for other evaluator candidates; and
4. compare NVIDIA's `usd-convert-gsplat` output with Blueprint's existing
   ParticleField authoring as an interoperability check, not an immediate
   replacement.

The CAD-to-SimReady agent skill, Content Agents, and SimReady Blender blueprint
are useful references for buyer-supplied CAD and technical-artist workflows.
They should not become the authoritative path for capture-derived geometry or
physical properties. ArtiFixer, MotionBricks, ARDY, Cosmos-Dreams, and other
research demonstrations remain watch-list items.

## What Was Actually New at SIGGRAPH

The supplied analysis identifies NVIDIA's direction correctly but overstates
the novelty of the library-first architecture.

NVIDIA announced the standalone/headless Omniverse library strategy at GTC on
April 8, 2026. That announcement already described `ovrtx`, `ovphysx`,
`ovstorage`, C APIs, existing-application ownership of the main loop, and
agent/MCP workflows. The SIGGRAPH update is the productization around that
architecture: Omniverse tools in NVIDIA Agent Toolkit, released or more visible
SimReady skills, the Blender reference workflow, and an expanded set of partner
integrations.

The SIGGRAPH-specific changes that matter are therefore:

- deterministic Omniverse operations are packaged as Agent Toolkit tools and
  reusable skills;
- the SimReady Foundation validator and CAD-to-SimReady workflow are available
  as concrete open implementations;
- the SimReady Blender example demonstrates a host-owned authoring workflow
  backed by a separate NVIDIA library worker;
- Cosmos 3 Edge is now an openly available 4B model instead of a future variant;
  and
- NVIDIA's research program adds useful but mostly preproduction work in scene
  repair, motion generation, neural simulation, and reconstruction.

This is a meaningful distribution and integration change. It is not evidence
that simulation preparation, sim-to-real validation, or physical correctness
has been solved.

Primary sources:

- [NVIDIA SIGGRAPH newsroom release](https://nvidianews.nvidia.com/news/nvidia-agent-toolkit-expands-with-new-omniverse-libraries-putting-ai-agents-to-work-building-simulation-ready-worlds)
- [April 2026 library-first Omniverse announcement](https://developer.nvidia.com/blog/integrate-physical-ai-capabilities-into-existing-apps-with-nvidia-omniverse-libraries/)
- [NVIDIA SIGGRAPH 2026 recap](https://blogs.nvidia.com/blog/siggraph-news-2026/)
- [NVIDIA SIGGRAPH event page](https://www.nvidia.com/en-us/events/siggraph/)

## Capability and Maturity Matrix

| Capability | Verified current state | Blueprint overlap or gap | Recommendation |
| --- | --- | --- | --- |
| SimReady Foundation | Apache-2.0 Python library and CLI, reviewed at version `2026.04.1` and source revision `4d9f3bb`, with versioned profiles, requirements, features, validation, and transformations. Python range is `>=3.10,<3.13`. | Current `simready_validation.json` checks local inputs and output existence but does not establish profile conformance. | Implemented as an isolated advisory validator adapter; official CLI execution remains pending. |
| `ovrtx` | Prerelease 0.4 C/Python sensor-rendering SDK for camera, lidar, radar, ultrasonic, semantic outputs, Gaussian-splat ParticleFields, and visual preflight. The host owns simulation advancement and output handling. | Blueprint has Isaac RGB/segmentation review canaries, ParticleField episodes, and camera/FK evidence, but full Isaac startup is expensive and its proof remains simulator-specific. | Implemented as a gated isolated sidecar contract. External Linux/RTX qualification remains pending; measure startup, memory, output repeatability, and agreement with accepted Isaac fixtures. |
| `ovphysx` | Prerelease 0.4.13 standalone USD physics library with C/Python bindings, CPU/GPU paths, cloning, and DLPack; no Kit dependency. | Blueprint has simulator request/result manifests and owner GPU handoffs, but no lightweight Omniverse-native physics preflight. | Implemented as a gated smoke-test contract. It cannot replace Isaac runtime, sensor, policy, or task proof. |
| `ovstage` | Prerelease shared CPU/GPU scene-state layer used across Omniverse libraries; ovrtx 0.4 deprecates renderer-owned stage APIs in favor of it. | Existing Blueprint manifests and USD assets already provide durable interchange. A new in-memory substrate adds value only inside an adopted Omniverse worker. | Permit as an internal ovrtx/combined-worker dependency; do not expose it as an independent product or durable evidence store. |
| CAD-to-SimReady skill | Apache-licensed orchestration guidance for conversion, minimum USD validation, material and physics proposal agents, conformance, and reports. It expects Docker/GPU-oriented infrastructure and Python 3.12 for the documented workflow. | Useful for customer CAD/robot/prop inputs. It is not a reconstruction pipeline for raw Blueprint capture truth. | Reuse its staged-validation pattern and report schema ideas; do not make it the primary pipeline. |
| Content Agents | Material/physics agents are beta; texture/validation agents include research-preview components. Documented deployment is Linux/WSL-oriented and can require large RTX GPU capacity plus hosted model credentials. | Could reduce technical-art work on buyer-supplied assets. Suggested mass, friction, materials, semantics, or colliders would remain unverified proposals. | Later opt-in enrichment lane with human approval and immutable before/after evidence. |
| SimReady Blender | Public Blender 5.1 example using an external NVIDIA library worker. It covers a limited subset and is not a supported production renderer or full Isaac workflow. | Helpful interactive review for technical artists; no advantage for headless package assembly. | Developer tool only; do not place on the critical path. |
| Cosmos 3 Edge | Released July 20, 2026 as an OpenMDW-1.1 4B multimodal model with reasoning, forward/inverse dynamics, and policy-style modes. NVIDIA's listed action encodings do not include Unitree G1 7D control. The model card also states that it lacks explicit physical laws, can produce unrealistic contacts, and is not safety certified. | Existing SC3-Eval configuration and command adapter are deliberately Cosmos3-Nano-specific. Blueprint has no Edge runtime receipt or rank-fidelity measurement. | Implemented as a separate experimental contract. Never inherit Nano qualification or present Edge as a drop-in G1 policy/inverse-dynamics backend. |
| `usd-convert-gsplat` | New, small open converter, reviewed at package `0.1.15` and source revision `621017e`, for PLY/SPZ Gaussian splats to `ParticleField3DGaussianSplat` USD. | Blueprint already decodes SPZ, authors ParticleField USD, and can render it through Isaac with spend and proof gates. | Implemented as an optional conformance oracle/golden comparison; replace nothing until measured. |
| ArtiFixer | SIGGRAPH 2026 research code/model for completing and repairing 3D Gaussian-splat scenes, with a large model and CUDA workflow. | Blueprint already has a fail-closed ArtiFixer backend. Its generated unseen content is derived support, not capture evidence or collision truth. | Retain as optional research backend; evaluate with held-out real views only. |
| MotionBricks, ARDY, GPC, Newton research | Research systems and conference demonstrations, not one uniform production release. | Possible future humanoid motion priors or solver improvements, but no direct upgrade to current task/eval package fidelity. | Watch list. Require a stable runtime, license, fixtures, and a measurable Blueprint gap first. |

Technical sources:

- [Omniverse libraries and agentic workflows](https://developer.nvidia.com/omniverse)
- [`simready-foundation`](https://github.com/NVIDIA/simready-foundation)
- [`ovrtx` integration guide](https://developer.nvidia.com/blog/integrate-nvidia-omniverse-rtx-sensor-simulation-into-existing-apps/)
- [`ovrtx` repository](https://github.com/NVIDIA-Omniverse/ovrtx)
- [`ovphysx` repository](https://github.com/NVIDIA-Omniverse/PhysX/tree/main/ovphysx)
- [`ovstage` repository](https://github.com/NVIDIA-Omniverse/ovstage)
- [CAD-to-SimReady skill](https://github.com/NVIDIA/skills/blob/main/skills/omniverse-cad-to-simready/SKILL.md)
- [Content Agents](https://github.com/NVIDIA-Omniverse/content-agents)
- [SimReady Blender example](https://github.com/NVIDIA-Omniverse/omniverse-labs/tree/main/projects/ov-blender-example)
- [Cosmos 3 Edge model card](https://huggingface.co/nvidia/Cosmos3-Edge)
- [`usd-convert-gsplat`](https://github.com/NVIDIA-Omniverse/usd-convert-gsplat)
- [ArtiFixer project](https://research.nvidia.com/labs/sil/projects/artifixer/)
- [ArtiFixer code](https://github.com/nv-tlabs/artifixer)
- [ARDY research project](https://research.nvidia.com/labs/sil/projects/ardy/)

## Repository Implementation

The recommendation is now represented by replaceable, fail-closed Blueprint
contracts rather than direct dependencies in the core environment:

- `external_simready_validation.py` and `run_simready_validator_worker.py`
  provide advisory, no-transformation SimReady validation with pinned profile,
  source, executable, input, output, report identity, repeated-run stability,
  and a preserved local-validation baseline;
- `simready_rule_calibration.py` requires frozen valid/invalid cases, expert
  labels, and explicit human authorization before any selected rule may block;
- `omniverse_library_preflight.py` plus the ovrtx and ovphysx worker scripts
  provide explicit dual gating, cold/warm receipts, output containment,
  repeatability checks, pinned runtime/GPU identity, CPU/GPU memory, scene-derived
  required checks, and valid-plus-negative same-scene Isaac comparisons;
- `cosmos3_edge_experiment.py` provides a distinct three-mode Edge experiment
  family, repeated-run output-stability evidence, an official-framework worker,
  and evaluator receipt schema without changing the Nano adapter;
- `cosmos3_edge_qualification.py` requires a validated Blueprint evaluator,
  accepted anchors, grounding, abstention, rank correlation, and calibrated
  failure detection while still prohibiting an automatic default change;
- `gsplat_conformance.py` treats NVIDIA's converter as a pinned advisory oracle
  for Blueprint's existing ParticleField writer;
- `artifixer_heldout_evaluation.py` requires frozen real held-out views before an
  ArtiFixer result can move beyond pending research support; and
- `nvidia_siggraph_policy.py` records activation gates, stop rules, and the
  mandatory structured post-conference refresh;
- `nvidia_asset_conditioning_review.py` records immutable, proposal-only CAD,
  Content Agent, and Blender evidence; and
- `nvidia_experiment_resource.py` binds paid attempts to the shared allocator
  and exact-attempt/global-zero teardown and billing evidence.

These implementations prove contract behavior under local fixtures. They do
not prove that an NVIDIA binary, RTX GPU, model checkpoint, or paid-provider
attempt has run successfully.

## Fit With Blueprint's Product Strategy

The announcements strengthen the existing strategy; they do not justify
reversing it.

Blueprint should continue to own:

- raw media, provenance, rights, privacy state, poses, intrinsics, depth, and
  capture identity;
- capture-derived geometry and site-reference memory;
- task/scenario/evaluator contracts and package assembly;
- immutable request, result, runtime, model, and teardown receipts; and
- acceptance criteria that distinguish support evidence from simulator,
  ranking, task-success, deployment, or real-world proof.

NVIDIA components can sit behind replaceable adapters that consume derived
package inputs and emit candidate support artifacts. OpenUSD is a useful
exchange substrate, but it should not become the source of capture truth. An
agent-generated collider, mass, friction value, semantic label, repaired splat,
or unseen surface is a proposal until a deterministic check or authorized
review accepts it.

The practical architecture is:

```text
raw Blueprint capture and provenance
  -> privacy-safe and capture-derived package evidence
  -> replaceable USD / SimReady conditioning adapter
  -> deterministic validation and candidate-enrichment reports
  -> optional sensor / physics preflight workers
  -> accepted simulator request and runtime receipts
  -> WAM / policy / evaluator evidence
  -> calibration, ranking, and task-success gates
```

The Agent Toolkit is optional orchestration around these steps. Blueprint's
durable contracts should remain usable from a CLI, queue worker, or another
orchestrator without NVIDIA Agent Toolkit.

## Recommended Experiment 1: External SimReady Validation

### Why this is the first choice

`src/blueprint_pipeline/simready_assets.py` currently produces useful review
artifacts and an honest proof boundary. Its validation covers presence of
object geometry, task anchors, robot profiles, site reference, and expected
outputs. It does not validate a declared SimReady profile, USD schemas,
physical-property requirements, articulation requirements, or feature
conformance.

SimReady Foundation directly targets that gap and matches Blueprint's supported
Python version range. Its documentation nevertheless recommends isolation
because `omniverse-asset-validator` and `usd-core` can conflict with an
application environment. The integration should therefore be an external
worker or container, not a new import in core orchestration.

### Proposed contract

Add an optional adapter only after implementation is explicitly authorized. It
should consume a frozen USD path and write:

```text
pipeline/simready/external_validation_request.json
pipeline/simready/external_validation_result.json
pipeline/simready/external_validation_claim_boundary.json
```

The request should record:

- input path, content SHA-256, source manifest IDs, and capture/package ID;
- exact requested profile and profile version;
- validator package version, source URL, and executable identity;
- declared transformations, with transformations disabled by default; and
- timeout, resource class, and network posture.

The result should record:

- process exit code and normalized status;
- every rule ID, severity, object path, message, and suggested action;
- exact validator/profile versions actually loaded;
- stdout/stderr digests and raw-report path;
- before/after digests if an explicitly authorized transformation ran; and
- an explicit `external_validator_ran` boolean.

Start in report-only mode on a fixture corpus and real non-sensitive package
samples. Promote selected deterministic rules to a blocking CPU/pre-GPU gate
only after false-positive/false-negative review. Never translate a validator
pass into simulator-load, physics/contact, policy, ranking, or deployment
success.

### Acceptance gate

- identical input and pinned validator/profile versions produce stable
  normalized results;
- an intentionally malformed fixture fails the expected rules;
- a known-valid fixture passes its declared profile;
- package conflicts cannot mutate the core Blueprint environment;
- no network or GPU is required for the base validation path;
- every generated or modified asset preserves the original and receives a new
  digest and provenance edge; and
- the existing local validation remains available when the external worker is
  absent.

## Recommended Experiment 2: `ovrtx` Sensor Preflight

`ovrtx` could answer bounded questions earlier than a complete Isaac episode:

- does the USD load into NVIDIA's sensor stack;
- can accepted camera intrinsics/extrinsics produce nonempty RGB, depth, and
  segmentation outputs;
- can a lidar/radar configuration produce structured output;
- are target and robot-support regions visible; and
- are material/sensor interactions obviously pathological in a frozen scene.

It is also directly relevant to episode rendering: ovrtx 0.4 can consume
OpenUSD scene state through ovstage, apply a USD time sample from the host loop,
and render `ParticleField3DGaussianSplat` content. Blueprint therefore requires
a nonconstant RGB result for ParticleField episodes, a dynamic-update check for
time-sampled episodes, and semantic visibility checks when robot/target labels
are configured. This makes ovrtx a plausible fast review renderer and sensor
preflight, not a replacement for the episode's simulator or evaluator.

It should be a replaceable Linux/CUDA sidecar. Do not add prerelease wheels to
the core environment. Pin the wheel/source, Python version, CUDA/driver, GPU,
sensor configuration, shaders, scene digest, and output digests. The first
shader compilation can be material, so measure warm and cold start separately.

Suggested artifact family:

```text
pipeline/sensor_preflight/ovrtx_request.json
pipeline/sensor_preflight/ovrtx_result.json
pipeline/sensor_preflight/ovrtx_runtime_receipt.json
pipeline/sensor_preflight/ovrtx_claim_boundary.json
```

An `ovrtx` pass means only that the pinned worker produced the requested sensor
outputs from the pinned USD. It does not prove Isaac scene parity, policy
execution, physics, ranking, real sensor correlation, or task success.

Proceed only if a small benchmark demonstrates a useful reduction in time or
cost relative to the existing Isaac review canary while preserving required
sensor metadata. Otherwise the extra runtime is not worth maintaining.

## Recommended Experiment 3: `ovphysx` Physics Smoke Test

`ovphysx` is interesting because it exposes USD physics without launching Kit.
That could make it a cheap pre-Isaac check for:

- scene load and schema compatibility;
- gravity and basic rigid-body integration;
- collider presence and gross penetration;
- joint existence, limits, and simple articulation motion;
- mass/friction fields being present and within declared bounds; and
- deterministic state snapshots from a short fixed-step run.

Use the same external-worker pattern and evidence discipline as `ovrtx`.
Compare it with accepted Isaac fixtures before using failures as a blocking
gate. Differences in solver defaults, plugins, sensors, articulation behavior,
or Isaac integration can make a standalone PhysX pass diverge from the final
simulator.

An `ovphysx` pass is not Isaac execution proof, contact-task success, controller
proof, generated-world rank fidelity, or deployment evidence.

## Recommended Experiment 4: Cosmos 3 Edge

Cosmos 3 Edge materially changes the availability assessment in the June 1
feasibility note: the 4B model is now released. It does not change the proof
standard.

The model card is unusually useful here. It describes forward dynamics,
inverse dynamics, multimodal reasoning, and a policy derivative, but it also
states that the model has no explicit physics engine, 3D representation,
object-permanence guarantee, or contact law and can generate unrealistic
collisions or morphing. Those limitations align with Blueprint's current
separation between WAM rollout evidence and physics/task/ranking evidence.

Do not alter `cosmos3_wam_command_adapter.py` merely to accept an Edge model
name. That adapter is intentionally pinned to Cosmos3-Nano and a declared
SC3-Eval recipe. Edge has different scale, runtime, modes, and calibration
status. It needs a separate model profile or adapter that still emits the
existing evaluator runtime receipts and attempt manifests.

The first Edge benchmark should:

- use frozen capture-derived input cells already used for WAM candidates;
- pin model and code revisions, precision, resolution, frame count, action
  encoding, decoding parameters, and GPU identity;
- test forward prediction, inverse-action inference, and reasoning separately;
- include deterministic fixture/error paths before any paid run;
- compare wall time, peak VRAM, output stability, grounding, and abstention;
- score only with the configured Blueprint evaluator and accepted anchors;
- keep upstream NVIDIA latency/benchmark numbers labeled as upstream; and
- make no default-model change until rank fidelity and failure calibration are
  measured on Blueprint data.

The released model card's supported action encodings include camera, autonomous
vehicle, egocentric, Franka, Agibot, UR, Google Robot, WidowX, and UMI variants,
but not Unitree G1 7D actions. The implemented harness rejects that combination.
Forward visual prediction and multimodal reasoning remain valid candidates;
G1 inverse-dynamics or policy use requires an explicit compatible encoding or a
separately qualified adapter.

Potential payoff is a lower-cost local WAM/reasoning option. It is not yet a
reason to replace Cosmos3-Nano as the SC3-Eval recipe base, OSCAR, a structured
simulator, or real evaluation evidence.

## Existing Work That Should Not Be Rebuilt

Blueprint already has:

- replaceable WAM/evaluator adapters and runtime receipts;
- explicit generated-video, forward/inverse consistency, and ranking gates;
- fail-closed paid-resource admission and provider teardown requirements;
- local SimReady review artifacts for Isaac, MuJoCo, and PyBullet;
- a CPU/pre-GPU simulation automation lane;
- camera/FK and Isaac render canaries;
- SPZ/PLY decoding and direct `ParticleField3DGaussianSplat` USD authoring;
- a 3DGRUT/NuRec export path and Isaac ParticleField render jobs; and
- a fail-closed ArtiFixer backend.

Consequently:

- do not replace existing ParticleField authoring with `usd-convert-gsplat`
  without conformance and render comparisons;
- do not add a second generic agent framework merely because Agent Toolkit can
  call the same deterministic tools;
- do not treat Content Agents as a source of authoritative physical metadata;
- do not send raw or privacy-sensitive capture bundles to hosted agent/model
  services by default;
- do not expose `ovstage` independently until shared live state between at least
  two adopted Omniverse libraries has demonstrated value; permitting it as the
  internal dependency of a pinned ovrtx worker does not make it a product
  contract; and
- do not treat Blender example output as headless pipeline or simulator proof.

## Phased Recommendation

### Phase 0: no-cost design and fixture work

Repository status: implemented and locally testable; post-conference source
refresh remains outstanding until July 24, 2026 or later.

- define the external-validator request/result/claim schemas;
- select malformed and known-valid USD fixtures;
- capture baseline results from Blueprint's current local validation;
- add a post-SIGGRAPH source/version/license refresh checkpoint; and
- define the exact evidence needed to promote an advisory rule to a gate.

### Phase 1: isolated SimReady validator canary

Repository status: adapter, schemas, setup script, worker, repeated fixture
tests, local-baseline receipt, and expert-calibration contract are implemented.
An official-validator execution and actual expert-reviewed Blueprint calibration
are not yet proven.

- run in a disposable Python environment or container;
- prohibit transformations for the first pass;
- normalize findings into Blueprint-owned artifacts;
- test stable output and negative fixtures;
- compare findings with expert review; and
- remain advisory until calibrated.

### Phase 2: one-worker sensor/physics bake-off

Repository status: ovrtx/ovphysx workers, pinned runtime expectations,
cold/warm CPU/GPU memory evidence, and a valid-plus-negative benchmark-suite
contract are implemented. Linux/RTX and official ovphysx runtime attempts,
followed by the same-scene Isaac comparison, remain unproven.

- benchmark `ovrtx`, `ovphysx`, and the existing Isaac preflight on the same USD
  fixtures;
- record cold/warm startup, GPU/CPU memory, runtime, determinism, and failure
  coverage;
- retain only an experiment that catches a meaningful class of failures faster
  or more cheaply than the existing path; and
- allocate paid resources only through Blueprint's shared paid-resource
  allocator and prove global provider zero after the attempt.

### Phase 3: Cosmos 3 Edge evaluator canary

Repository status: the separate Edge experiment, official-framework worker,
repeated-run stability, evaluator receipt, and Blueprint qualification contracts
are implemented. No checkpoint download, GPU attempt, or rank-fidelity result
has occurred.

- create a distinct Edge model profile and runtime command;
- use the existing WAM attempt and evaluator receipt contracts;
- run the smallest frozen benchmark that can falsify the value proposition;
- measure Blueprint-specific correlation/ranking behavior; and
- promote nothing based only on NVIDIA's model-card examples or upstream
  benchmarks.

### Later or no action

- Content Agents: the proposal-review contract is implemented; wait for a
  specific buyer CAD/asset-conditioning need before running it.
- Blender blueprint: the proposal-review contract is implemented; use only if a
  technical-artist review workflow is needed.
- `ovstage`: wait for a multi-library live worker.
- ArtiFixer: retain the existing optional adapter and require held-out-view
  evaluation.
- MotionBricks/ARDY/Cosmos-Dreams: monitor; no current critical-path gap.

## Stop Rules

Stop an integration experiment if any of the following is true:

- the component cannot be version-pinned or its license is incompatible with
  the intended distribution path;
- prerelease API churn prevents stable normalized receipts;
- it requires raw/unredacted capture upload outside the accepted privacy and
  rights contract;
- its dependency set mutates or conflicts with the core package environment;
- it cannot preserve input/output digests and transformation provenance;
- a pass cannot be kept semantically separate from simulator, task-success,
  ranking, deployment, and real-world claims;
- it does not catch a useful failure class earlier or more cheaply than the
  current pipeline; or
- paid-resource admission or exact-attempt/global teardown proof cannot be
  enforced.

## Bottom Line

NVIDIA is making its simulation capabilities easier to embed, automate, and
compose. For Blueprint, that is valuable mainly at the conditioning and
preflight seams. It does not replace the capture-first substrate, the
replaceable model/evaluator boundary, or the evidence ledger.

The actionable change is to evaluate deterministic NVIDIA validators and
lightweight simulation libraries behind Blueprint-owned contracts. The model
change worth testing is Cosmos 3 Edge, but only as a distinct experimental
candidate. The rest of the announcement is better treated as a supplier and
ecosystem signal until stable releases demonstrate a measurable improvement on
Blueprint fixtures.
