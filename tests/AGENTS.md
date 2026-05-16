# Agent Guide For `tests/`

Tests should pin package/runtime truth boundaries without calling live providers,
GPU services, Stripe, WebApp mutation, or deployment commands.

Use `PYTHONDONTWRITEBYTECODE=1` when avoiding local cache churn. Prefer focused
pytest targets for touched modules, then broader gates only when launch contracts
or cross-repo behavior changed.

Do not weaken fail-closed tests around privacy-safe media, fallback geometry,
WebApp upstream links, provider credentials, or operator/live evidence.
