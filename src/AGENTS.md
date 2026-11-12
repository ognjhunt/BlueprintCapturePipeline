# Agent Guide For `src/`

Read the root `AGENTS.md` first.

`src/` contains runtime code. Keep model providers replaceable behind stable
capture, package, adapter, runtime, and sync contracts. Beta work may use any
task family or robot form Blueprint can run: arms, mobile manipulators,
wheeled humanoids and humanoids.

Do not edit env files, secrets, provider tokens, GPU runner config, or live
deployment config from this tree. Prefer targeted tests for touched modules.
