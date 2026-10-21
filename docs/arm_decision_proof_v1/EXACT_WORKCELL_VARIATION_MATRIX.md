# Exact-Workcell Variation Matrix

Status: production contract for the ADP-009D rehearsal and later Task Evaluation Runs
Primary implementation: `src/blueprint_pipeline/exact_workcell_variation_inputs.py`,
`src/blueprint_pipeline/exact_workcell_variation_matrix.py`, and
`src/blueprint_pipeline/exact_workcell_variation_runtime.py`

## Purpose

Blueprint evaluates policies for the workcell in which they are intended to
operate. The primary matrix therefore holds the workcell, task, embodiment, and
canonical task object fixed. It samples uncertainty inside that exact deployment
envelope; it does not test lookalike objects or silently broaden the task family.

The default baseline is exactly 100 policy-neutral cells per workcell/task:

- one immutable canonical anchor;
- low and high one-factor cells for every admitted dimension;
- a deterministic pairwise covering array;
- agent-proposed targeted interactions, after deterministic validation;
- bounded composed qualification cells;
- 20 held-out composed cells at the 100-cell default.

The matrix is compiled once. A later schedule binds exactly two frozen policy
identities without changing the matrix. Both policies, the zero-action negative,
and the deterministic scripted positive receive the identical cell IDs, reset
digests, and seeds. At the 100-cell default this creates 400 planned episodes:
100 per policy and 100 per control.

The number 100 is a baseline generation target, not an automatic statistical
adequacy claim. An executable schedule must also bind the preregistered
experiment, minimum decision-relevant difference, and power-analysis digests;
its justified trials per candidate must equal the matrix cell count. If the
power plan requires more than 100 cells, the matrix must be regenerated at that
larger count before outcomes are observed. If it supports fewer, Blueprint
still runs the frozen 100-cell baseline rather than stopping early.

## Why Object Cousins Are Not In The Primary Matrix

An object cousin answers a different question: whether behavior survives an
intentional object-identity distribution shift. The primary Task Evaluation Run
answers how two policies compare in the exact intended workcell and task.

Accordingly:

- `matrix_kind=exact_workcell_primary` rejects any object cousin;
- every cell binds the same canonical object asset ID and digest;
- cousin suites, when separately authorized, keep separate artifacts, metrics,
  and claim ceilings;
- cousin results cannot alter the primary exact-workcell score.

This preserves the historical ADP cousin machinery without making cousins a
requirement for the exact-site product.

## Dynamic Inputs

The autonomous builder merges three typed, digest-bound contracts:

1. scene variation contract: registered cameras, lighting, support/workspace,
   canonical task object, and measured scene tolerances;
2. task variation contract: reset distribution, task-state tolerances, and
   owner-approved operational uncertainty;
3. embodiment variation contract: robot registration, joint limits, camera
   calibration, control timing, and embodiment-specific tolerances.

Each dimension must name its source contract, application target, unit, nominal
value, admitted range or categorical values, and measurement-authority digest.
Malformed, non-finite, missing, or identity-changing inputs fail closed. If 100
cells cannot cover every dimension's one-factor probes and required pairwise
coverage, the compiler abstains rather than dropping coverage silently.

## Agent / LLM Role

An agent receives only the admitted dimension names and bounds. It may propose:

- relative dimension priority;
- two-to-four-dimension targeted interactions;
- a short scientific rationale.

The agent cannot widen a bound, invent a measurement, change the workcell/task/
embodiment/object identity, inspect policy outcomes, add cousins, authorize a
cell, or change a proof boolean. Its prompt and raw response are digest-bound.
Deterministic code validates the proposal and materializes every cell. If no
agent is available, deterministic one-factor, pairwise, and composed generation
still produces the baseline autonomously and records that fallback explicitly.

`AgentsSDKVariationProposalAgent` connects this bounded role to Blueprint's
canonical OpenAI Agents SDK harness with strict structured output and no tools.
Its production default is `gpt-6-luna` at `reasoning.effort=max`; both values
are retained in the proposal identity and inference reservation.
The shared harness retains its explicit live-inference enablement, conservative
cost reservation, model identity, usage, and audit requirements. Tests inject a
network-free invoker; production injects the already gated SDK invoker. Neither
path grants the model scenario or proof authority.

## Isaac / EvaluationRun Integration

The compiler reuses the existing simulator and Evaluation Run machinery. It
emits the `exact_workcell_variation_matrix@1` task-scenario adapter, whose 100
scenario rows carry the exact condition digest, reset digest, seed, and
partition. Candidate-specific `EvaluationRunSpec` objects reference the same
published matrix URI and digest; no parallel simulator orchestration framework
is introduced.

Runtime adapters remain responsible for applying each typed dimension to the
correct Isaac Lab manager/event surface and independently reading it back within
tolerance before policy execution. A matrix or schedule is preparation evidence,
not proof that Isaac ran, a policy acted, the task succeeded, or the ranking is
valid.

## Execution availability

The matrix adapter registered in `evaluation_run_contract.py` is a compilation
adapter. It does not register an execution implementation or allocate a runtime.
`exact_workcell_variation_runtime` publishes the frozen schedule and independently
joins retained episode artifacts; neither operation executes robot policies.

The native paired-policy canary currently executes ten cells at the
`diagnostic_policy_execution` ceiling. It is not the 100-cell experiment above.
Its scene resolver declares its actual parameter application/readback profile in
`policy_canary_scene_resolution.NATIVE_SCENARIO_PARAMETERS` and refuses unknown
parameters during static preflight, before provider allocation or policy queries.
A task-family template or a generated exact-workcell dimension cannot widen that
profile by passing a new parameter name.

`exact_workcell_native_runtime.py` now provides the native target adapter for
supported scalar terms. `prepare_native_runtime_cell` verifies the frozen
matrix/cell and immutable source plan, checks scene/task/object/embodiment
identities, and calls the existing scene-plan setters. It requires absolute
native nominal values and declared units; it does not interpret a dimension
name ending in `delta` as an offset. Use `native_runtime_embodiment_digest` to
bind the source robot, cameras and cadence in the request. Unavailable targets,
categorical targets and articulated link friction without an exact selector
remain blocked.

For a new native site, `build_native_site_task_variation_request` reuses the
existing site-task bound generator and explicitly converts its bounded offsets
to absolute values using that immutable source plan's object pose and camera
calibration. It preflights the canonical anchor through the same native setters.
The old delta-shaped request is not passed directly to absolute native targets;
nonzero site coordinates require no site-specific source edits. A missing
friction measurement remains omitted, and changed source calibration requires
new matching source/configuration identities.

After building and resetting that prepared plan, `read_native_runtime_cell`
uses the existing native readback adapter and returns the digest-bound
`exact_workcell_reset_application.v1` consumed by the collector. The caller must
bind its prepared plan digest before construction. This receipt proves only
parameter application/readback. Full scientific reset, episodes, controls,
scoring, observation media and provider teardown retain their existing owners.

The full native execution port below supplies the supported fixed-arm adapter.
Every additional runtime must likewise consume exact cell, reset and episode
bindings, apply and independently read every admitted target, run both controls
and frozen policies on the same cells, and retain the media/lifecycle/scorer
artifacts consumed by `collect_evaluation_evidence`.
An unavailable target must identify the missing native application/readback
capability. Increasing the canary cell count or relabeling its result is not this
integration. Existing execution authority, paid-resource admission, controls holds
and physical-proof requirements remain unchanged.

## Immutable Artifacts

The create-only publisher writes and fully reads back:

- `exact_workcell_variation_request.v1.json`;
- `exact_workcell_evaluation_schedule_request.v1.json`;
- `exact_workcell_variation_matrix.v1.json`;
- `exact_workcell_isaac_lab_event_plan.v1.json`;
- `exact_workcell_evaluation_schedule.v1.json`;
- `exact_workcell_variation_validation.v1.json`;
- `exact_workcell_variation_publication.v1.json`.

Schemas live under `docs/schemas/`. Every matrix, cell, reset, schedule request,
episode binding, schedule, and publication receipt is digest-bound.

## Episode Completion Rule

The schedule sets `retry_cap=0`, prohibits automatic retry and early success
stopping, and requires each episode to reach the complete planned policy-control
duration or a legitimate terminal scientific/safety result. Runtime media and
receipt requirements remain those of the Arm Decision Proof doctrine: exact
lossless policy-input frames, digest-bound frame manifest, review media,
actions, commanded and observed state, independent grading, terminal receipt,
teardown, and provider-zero where paid infrastructure is used.

## Claim Ceiling

Generating or publishing the matrix proves only deterministic preparation. It
does not prove simulator execution, task success, policy ranking, physical
transfer, deployment readiness, safety, or multi-site generalization.

## Full native execution port

`exact_workcell_evaluation_session.execute_evaluation_schedule` executes the
frozen schedule, and `exact_workcell_native_session.NativeCellProcessRunner`
binds it to the existing native `CellRuntime`, task-neutral control loop and
DROID policy episode loop. The existing EvaluationRun execution registry names
this adapter `exact_workcell_native_droid`. It accepts the fixed-arm rigid-task
profile; it does not widen the Quick-10 canary, add a provider allocation route,
or declare another embodiment/task executable.

The admitted provider host calls the existing `execute_evaluation_run` API with
its compiled immutable `evaluation_run.v1`, `allow_execution=True`, and ephemeral
context containing `native_exact_workcell_config` plus
`assert_native_execution_allowed`. That callback is the host's existing live
permission/hold/deadline/spend recheck, not a new boolean approval. A generic
execution gate or JSON CLI request without that runtime authority stays blocked.
The spec's runtime `execution_adapter_id` is `exact_workcell_native_droid`; its
task-scenario pack is the existing `evaluation_run_task_scenario_pack` projection
of the published matrix. Config binds the compiled spec digest, exact source
scene/plan/capture and job identity, frozen candidate execution specs, full
policy-readiness/scenario-suite authorities, observation-integrity authority,
retained provider policy identity root, native packet root, and independently
qualified per-cell construction results. Its evidence root is the run output's
`native_exact_workcell` directory. These are retained input references and
existing authority, never placeholders synthesized by the execution driver.

Each cell runs in a fresh Isaac interpreter. Both controls precede both policies;
all four share the resolved cell, seed, native parameter readback and independently
measured reset. Per-cell policy admissions preserve the frozen model/interface
identities while binding actual construction and controls. Native camera, device,
gripper, observed checkpoint identity, exact policy inputs, media, independent
scoring and complete-duration rules remain those of the existing runners.

Before an episode effect, the parent fsyncs its source-bound intent. Retained
receipt bytes and sidecars are hashed before acknowledgement. A lost receipt
acknowledgement adopts the retained sidecar; an ambiguous effect without one
returns `reconciliation_required` and never purchases another episode. A changed
capture/configuration cannot reuse the directory. The existing collector validates
the union against every scheduled row and refuses missing media, controls, source
bindings and reset parity. A count of 400 alone remains insufficient evidence.
The enclosing managed allocator still owns watchdog, billing, teardown and
provider-zero receipts; this driver grants no new spending authority.

For the existing native provider bundle, pass
`exact_workcell_native_bundle.full_matrix_native_runtime_sources()` as the
`runtime_module_sources`/`runtime_extra_files` arguments to
`build_native_task_arena_bundle`. This resolves the complete local import
closure, including full rights/controls/media validators and the shared
immutable writer, and packages their exact bytes through the existing bundle
contract. The full session snapshots and hashes all supplied runtime inputs;
changing a policy endpoint, authority, construction result or packet reference
cannot resume the same retained session. This source staging function allocates
and dispatches nothing.

The control-plane default execution registry includes this native adapter.
Inside the native-only provider bundle, construct the existing
`EvaluationRunExecutionRegistry([ExactWorkcellNativeEvaluationRunExecutor()])`
and pass it as `execution_registry` to `execute_evaluation_run`; other runtime
families have their own bundles and dependencies.

Offline tests exercise software orchestration/recovery, native adapter inputs and
the actual deterministic control loop. Synthetic episode receipts deliberately
remain blocked by the real collector. They do not establish a live GPU run,
real policy outcomes, physical transfer or a deployment recommendation. A live
full comparison still requires the exact qualified capture-derived inputs,
reviewed policy authorities and an existing bounded allocation whose holds permit
it; no new job is automatically dispatched by registering this adapter.
