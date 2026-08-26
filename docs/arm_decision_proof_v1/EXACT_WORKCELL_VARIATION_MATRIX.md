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
