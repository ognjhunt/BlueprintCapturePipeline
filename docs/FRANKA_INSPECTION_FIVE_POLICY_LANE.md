# Franka Inspection Five-Policy Lane

This lane freezes one task and refuses to reshape it around available models.
The authoritative contract is produced by
`frozen_franka_inspection_contract()` in
`blueprint_pipeline.franka_inspection_learned_policy_lane`.

The task is `franka_marked_work_surface_inspection_v1`. It uses one DROID-style
Franka Panda, one fixed external RGB camera, one wrist RGB camera, live joint
and gripper state, a fixed inspection prompt, 224x224 uint8 images, and a 15 Hz
control loop. Every control step captures a new observation, invokes the learned
policy, retains the complete native 10x8 or 15x8 output, converts native row zero
through the frozen DROID delta/gripper mapping, applies one simulator action,
and retains the resulting contacts, collisions, and observation. Open-loop
chunk reuse and recorded-action replay are forbidden.

The frozen metric is the fraction of marked target-surface samples visible in
the live wrist camera with qualified depth and occlusion. All candidates use the
same scene, placement, route, target, reset seed, joint state, gripper state,
camera extrinsics, target state, metric, and control rate.

## Candidate audit

The smallest technically compatible five-checkpoint set is the five official
OpenPI RoboArena DROID baselines:

- `paligemma_binning_droid`;
- `paligemma_fast_droid`;
- `paligemma_fast_specialist_droid`;
- `paligemma_vq_droid`; and
- `paligemma_diffusion_droid`.

They are five different GCS object sets. Each is pinned by the complete object
generation-manifest digest and the OpenPI source revision
`15a9616a00943ada6c20a0f158e3adb39df2ccac`. The pinned OpenPI DROID transform
accepts the exact two RGB views, seven joint positions, gripper position, and
prompt used here; the five official configs emit 8-D DROID actions with native
horizons of 10 or 15.

The terminal artifact retains a per-candidate audit across immutable identity,
provenance, embodiment, observation, action, rights/license, hostability, and
runtime dependencies. Technical compatibility does not grant execution rights.
The OpenPI repository
is Apache-2.0 and describes the models as open source, but the separately hosted
checkpoint objects expose no checkpoint-specific license or grant in their GCS
object metadata. The pre-existing rights ledger also records those rights as
ambiguous. Consequently every candidate remains `blocked`, no real candidate
identity is admitted for execution, and no fleet authorization is issued.

## Runtime and compiler boundary

`IdentityBoundLearnedPolicyAdapter` wraps any DROID-compatible local or remote
backend without adding provider-specific observation fields.
`execute_learned_policy_attempt()` requires that identity-bound policy client
and an injected simulator, a matched-reset authorization, and a
fresh inference call for every control step. It emits
`learned_policy_attempt_receipt.v1` with inline evidence and derived digests for:

- initial and terminal observations;
- the full observation trace;
- complete native policy outputs;
- normalized actions and exact simulator actions;
- contacts and collisions; and
- the task metric.

`learned_policy_execution_bundle.v1` is the real compiler carrier. Its builder
rejects fixtures, rights-blocked candidates, unproven learned actions, mismatched
resets, incomplete evidence, and nonmatching authorization. The new-site Task
Evaluation Run recomputes the embedded evidence and refuses parallel
caller-supplied candidate, attempt, or metric placeholders.

Hermetic fakes test mechanics only. They set `fixture_or_fake=true`, never set
`learned_policy_action_proven=true`, and cannot enter a real execution bundle.

## Current terminal result

The committed terminal artifact is
`docs/evidence/franka_inspection_five_policy_terminal_2026-08-03.json`. It
contains five immutable proposed identities but zero admitted identities, zero
real identity-bound queries, and zero matched-reset attempts. At that artifact's
observation time no GPU or provider spend was authorized or performed. A later
operational authorization does not mutate that immutable result or create the
missing third-party checkpoint execution grant. No spend was performed.

The next valid transition requires attributable checkpoint-specific execution
rights for the selected checkpoint objects. A later, separate explicit budget
and TTL authorization would then permit the canonical `paid_resource_allocator`
path to perform the five identity-bound queries and matched-reset attempts.
