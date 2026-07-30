# Protocol Amendment 8: Ctrl-World Causal-Control Construction

Frozen prospectively: 2026-07-30T12:46:03-0500

## Native control inventory

The Blueprint Ctrl-World current-reference causal matrix uses exactly six
conditions in this order:

1. the candidate policy's own complete native action;
2. valid no-motion joint velocities with an explicit current gripper hold;
3. a deterministic shuffle of the candidate's first eight executed rows;
4. reversal of those first eight rows;
5. a one-row circular shift of those first eight rows; and
6. the complete native action from a different frozen real policy request.

The deterministic shuffle seed is `20260730`, and its frozen first-eight order
is `[3, 5, 6, 0, 1, 4, 7, 2]`. The constructor rejects every other shuffle
seed. This order is neither identity, reversal, nor a circular shift. Tail rows
after the first eight are preserved for the own-action-derived controls. A
ten-row or fifteen-row action remains in its native shape; the already frozen
released Ctrl-World padding rule is applied later by the registered action
adapter.

The policy swap requires a distinct real request identity. A synthetic policy
swap is forbidden. All six first-eight-row hashes must be pairwise distinct or
the selected policy/window is ineligible for causal qualification. The builder
does not adapt controls after observing a WAM result.

## Execution and claim boundary

Each condition is adapted through the same exact released Dynamics checkpoint
and Franka FK, then evaluated at seeds 0 and 1 for all three registered views.
The construction accesses no outcome label. It does not change the causal or
reliability thresholds in `protocol_v1.json`, and generating the matrix does not
itself prove causal response, policy ranking, or physical success.
