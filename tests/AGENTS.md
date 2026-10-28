# Agent Guide For `tests/`

Tests should pin package/runtime truth boundaries without calling live providers,
GPU services, Stripe, WebApp mutation, or deployment commands.

Two programs are active: the free beta's job-first automation service and Arm
Decision Proof v1. New tests must protect a beta or ADP contract, or preserve
compatibility needed by one. Existing captures, fixtures,
and scenes must retain `development_only` claim ceilings and may never be used to
assert partner capture, physical fidelity, or customer-value proof.

Use `PYTHONDONTWRITEBYTECODE=1` when avoiding local cache churn. Prefer focused
pytest targets for touched modules, then broader gates only when launch contracts
or cross-repo behavior changed.

Do not weaken fail-closed tests that protect privacy, rights or money
(privacy-safe media, provider credentials, spend). Tests that pin other gates
may be changed or deleted along with the gate.
