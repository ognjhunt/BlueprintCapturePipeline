# Agent Guide For `src/`

Read the root `AGENTS.md`, `PLATFORM_CONTEXT.md`, and
`WORLD_MODEL_STRATEGY_CONTEXT.md` first.

`src/` contains runtime code. Preserve behavior unless the user explicitly asks
for an implementation change. Keep model providers replaceable behind stable
capture, package, adapter, runtime, and sync contracts.

Do not edit env files, secrets, provider tokens, GPU runner config, or live
deployment config from this tree. Prefer targeted tests for touched modules.
