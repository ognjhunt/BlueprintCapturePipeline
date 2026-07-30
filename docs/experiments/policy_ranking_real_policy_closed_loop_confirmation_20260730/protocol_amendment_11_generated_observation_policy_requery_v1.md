# Protocol amendment 11: generated-observation policy re-query

Frozen prospectively before any current-reference Ctrl-World output was generated.

## Problem

The initial OpenPI current-reference canary accepts an exposed public frame-zero
engineering observation. A valid second policy query must instead accept exactly
the three generated Ctrl-World camera views and the commanded state from the
preceding action prefix. Reusing the frame-zero-only schema would obscure that
provenance boundary.

## Additive contract

The existing `ctrl_world_public_initial_observation.v1` schema remains supported
without modification. The additive
`openpi_current_reference_generated_observation.v1` variant is admitted only
when it binds:

- one validated `blueprint_ctrl_world_current_reference` result;
- its staged WAM request digest and seed;
- the immediately preceding exact OpenPI policy-query receipt;
- the same candidate policy identity required for the next query;
- the final generated frame from all three released Ctrl-World views;
- joint, gripper, and Cartesian state propagated from the executed command
  prefix.

The generated observation must declare `visual_source=wam_prediction`,
`state_source=commanded_prefix_kinematics`, no physical future RGB, no recorded
future state, no outcome access, and no confirmation eligibility. The provider
WAM result, request receipt, prior policy receipt, copied view bytes, and state
arrays are all hash-bound.

The policy input bundle and policy loader use one shared boundary validator for
both variants. Generated observations therefore enter the existing OpenPI
current-reference execution path; this amendment does not create a parallel
closed-loop harness.

## History rule

For each standalone policy request, the current generated image and commanded
state may seed the policy-side 24-slot observation history by repetition, which
matches the released Ctrl-World initialization behavior. The complete
cross-transition history remains owned by Blueprint's closed-loop transition
adapter and is not replaced by this policy-request seed.

## Claim boundary

Passing this contract proves only that an attributable generated observation can
be submitted for a real same-policy re-query. It does not prove WAM causal
qualification, episode coherence, ranking fidelity, physical success, blind
confirmation, captured-site transfer, or economics.
