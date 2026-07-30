# Protocol Amendment 16: first WAM policy and seed selection

Status: prospectively frozen before any Ctrl-World current-reference WAM output

Date: 2026-07-30

## Observed admission state

The identity-bound policy canary completed one real frozen checkpoint query for
each registered OpenPI DROID policy. All three native outputs passed portable
artifact validation. No Ctrl-World WAM output, generated frame, policy re-query,
or judge output existed when this amendment was written. The numeric contents
of the native action arrays were not inspected for this selection.

## Frozen first-generation selection

The first Blueprint Ctrl-World current-reference WAM request uses:

- policy: `pi05_droid`;
- policy output: the complete native `15x8` action from the completed
  identity-bound canary;
- task: `Move the banana to the right`;
- Ctrl-World task type: `pickplace`;
- released gripper maximum: `0.75`;
- executed prefix: exactly the first eight native rows;
- WAM seed: `0`;
- query index: `0`;
- WAM request count: exactly one;
- generated observations: five frames for each of the three registered DROID
  camera views.

`pi05_droid` is selected because its complete native 15-row output matches the
released Ctrl-World `pi05` action-adapter input length without repeat-padding.
This is a contract-based choice made before WAM output inspection, not a choice
based on action magnitude, trajectory appearance, or expected success. Seed 0
is the first seed in the already frozen causal seed inventory `[0, 1]`.

## Claim and progression boundary

This first request is only the registered one-action WAM generation gate. It
does not establish a policy-to-WAM-to-policy loop, causal qualification, episode
coherence, ranking, abstention quality, physical agreement, captured-site
transfer, or economics. Its output may advance only if the provider artifact is
attributable, complete, generated-only, three-view, hash-valid, and passes the
registered immediate reliability checks. The next paid progression remains one
three-interaction `pi05_droid` to Ctrl-World to the same `pi05_droid` episode.

No threshold is changed by this amendment.
