# Agent Guide For `src/`

Read the root `AGENTS.md`, `PLATFORM_CONTEXT.md`, and
`WORLD_MODEL_STRATEGY_CONTEXT.md` first.

Arm Decision Proof v1 is the sole active program. A source change must name the
ADP backlog item and day gate it unblocks. Existing modules outside that path are
compatibility code, not invitations to extend their former lanes.

`src/` contains runtime code. Preserve behavior unless the user explicitly asks
for an implementation change. Keep model providers replaceable behind stable
capture, package, adapter, runtime, and sync contracts.

Prefer thin changes to the existing Decision/Evidence Router, EvaluationRunSpec,
two-candidate matrix, runtime receipt, sealing, statistics, and Physical Outcome
Join paths. Do not start humanoid, deformable, world-model, provider-bakeoff,
post-training, or universal-runtime work without an observed ADP blocker.

Do not edit env files, secrets, provider tokens, GPU runner config, or live
deployment config from this tree. Prefer targeted tests for touched modules.
