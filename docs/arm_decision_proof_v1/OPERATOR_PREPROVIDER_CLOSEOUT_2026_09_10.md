# Operator canary closeout without provider allocation

ADP-009's development-only two-policy rehearsal needs terminal Website records
even when the allocator finds no compatible offer. This closes the day-7 public
rehearsal result-delivery seam; it does not prove the two-candidate matrix or a
physical outcome. The completion artifact is a signed Website acknowledgement
bound to the immutable operator registration and original no-allocation receipt.

The existing pre-provider endpoint payload requires an activation ID. An operator
registration has no activation, so fabricating one or an episode result would
misrepresent the failed attempt. The smallest change is a separate typed payload
on the existing Pipeline-HMAC endpoint, with a reusable producer helper:
`blueprint_pipeline.operator_policy_canary_preprovider_closeout`.

`build_operator_preprovider_publication` verifies the registration digest, native
closeout digest, no-create/no-side-effect facts, zero charge, cancelled watchdog,
cleanup, provider-zero and no scientific execution or automatic retry. It retains
the complete original UTF-8 receipt bytes and their SHA256 and size. This matters
because native Python JSON records `0.0`, while JavaScript canonical JSON records
`0`; the native receipt must not be rewritten to cross the transport boundary.
The outer payload uses the existing cross-runtime canonical digest.

The v1 source receipt explicitly retains the legacy allocation predicate's
`false` result and missing mutation-counter explanation. Direct adapter facts
prove this bounded refusal; the helper never changes the old predicate result.
The Website validates the exact source structure and registration/tenant scope,
closes only a still-pending operator registration, and shares the existing terminal
record key so delivery and refusal cannot replace each other. No source offering,
activation, episode, or result-delivery object is created.

`sync_operator_preprovider_closeout` sends one signed request and checks every
returned identity and the terminal notification receipt. A disabled email remains
an explicit failed notification; acknowledgement does not mean inbox delivery.
Exact retries remain idempotent on the Website. Transport failure is returned to
the caller and never triggers a provider retry.

Validation: `python -m pytest -q
tests/test_operator_policy_canary_preprovider_closeout.py
tests/test_task_evaluation_policy_canary_webapp_sync.py` covers original byte/float
preservation, digest changes, incorrect run binding, allocation uncertainty,
execution/retry rejection, signed publication and mismatched acknowledgement.
Changed-file Ruff and `git diff --check` cover local code hygiene. The retained
attempt-v24 payload was independently built by Python and accepted by the Website
parser with identical payload digest
`sha256:0c96fc8875971ea8ca694c24b66cc56af68a586340621eff2a400d06334f8b06`.
This offline match is not a live Website publication claim.
