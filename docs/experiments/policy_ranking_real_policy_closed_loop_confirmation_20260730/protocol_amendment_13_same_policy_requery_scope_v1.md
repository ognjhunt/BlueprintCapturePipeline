# Protocol amendment 13: same-policy re-query scope

Frozen prospectively before any current-reference policy or WAM output existed.

## Scope correction

The initial frame-zero identity canary continues to query exactly
`pi0_droid`, `pi0_fast_droid`, and `pi05_droid` once each. A generated
observation bound by Protocol Amendment 11 instead carries the exact preceding
policy identity and may query only that same policy.

The generated-observation input bundle therefore declares:

- purpose `label_free_current_reference_same_policy_requery`;
- exactly one policy ID;
- the generated-observation schema;
- the same candidate policy ID bound by the prior policy receipt.

The existing current-reference policy canary, source overlay, checkpoint
verification, paid allocator, watchdog, provider transport, terminal archive,
and teardown path are reused. The policy runtime loads and queries only the
declared same policy. It must preserve the complete native output and emit an
identity-bound query receipt exactly as in the initial canary.

Legacy all-three frame-zero bundles and outputs remain accepted unchanged. The
new single-policy mode is additive and is admitted only for a validated
generated-observation contract.

## Reason

Loading all three policies on every generated observation would spend more GPU
time and blur the required feedback chain. Restricting later interactions to the
identity that produced the preceding action makes the sequence explicit:

`same policy receipt -> one WAM request -> generated observation -> same policy re-query`.

## Claim boundary

A successful single-policy re-query proves one additional real learned-policy
inference on WAM-generated views and commanded state. It does not by itself
prove a three-interaction episode, complete horizon, WAM causal qualification,
ranking fidelity, physical success, blind confirmation, transfer, or economics.
