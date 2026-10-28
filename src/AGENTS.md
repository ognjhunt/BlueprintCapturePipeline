# Agent Guide For `src/`

Read the root `AGENTS.md` first.

`src/` contains runtime code. Keep model providers replaceable behind stable
capture, package, adapter, runtime, and sync contracts. Beta work may use any
task family or robot form Blueprint can run; Arm Decision Proof keeps its own
protocol in `docs/arm_decision_proof_v1/`.

Do not edit env files, secrets, provider tokens, GPU runner config, or live
deployment config from this tree. Prefer targeted tests for touched modules.
