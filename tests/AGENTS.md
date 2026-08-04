# Agent Guide For `tests/`

Tests should pin package/runtime truth boundaries without calling live providers,
GPU services, Stripe, WebApp mutation, or deployment commands.

Arm Decision Proof v1 is the sole active program. New tests must protect an ADP
contract or preserve compatibility needed by it. Existing captures, fixtures,
and scenes must retain `development_only` claim ceilings and may never be used to
assert partner capture, physical fidelity, or customer-value proof.

Use `PYTHONDONTWRITEBYTECODE=1` when avoiding local cache churn. Prefer focused
pytest targets for touched modules, then broader gates only when launch contracts
or cross-repo behavior changed.

Do not weaken fail-closed tests around privacy-safe media, fallback geometry,
WebApp upstream links, provider credentials, or operator/live evidence.
