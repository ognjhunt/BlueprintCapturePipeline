# Claude Opus 5.5 local Agents SDK authoring

For a **new, separately authorized website scene** whose signed stage-3
configuration selects `authoring_model_provider: anthropic`, the CPU authoring
driver selects `execute_claude_sdk_agent_authoring`. OpenAI scenes keep their
existing model route. A prior Astra scene or its session cannot be relabeled as
Claude evidence. The SDK path has not been merged, deployed, or used on a paid
scene as of this branch.

The local OpenAI Agents SDK owns a `SQLiteSession` and the confined
observe/CAD/Blender tools. A `Model` adapter translates source images, strict
tools, JSON output schemas, and signed Claude thinking blocks. Each translated
Messages request is written before provider dispatch. The full response is
written before Runner may execute a tool. The existing Anthropic invoker checks
the signed scene authority and scoped `ANTHROPIC_API_KEY_FILE`, reserves the
published full 1M input window plus at most 12,000 output tokens, and receipts
actual provider usage. Unknown outcomes retain their reservation; they cannot
be replayed as a new paid request. Independent physical and visual reviews use
the same $7 model ledger. Compute remains separately quoted at $6 per attempt.
The website's new-scene sponsorship remains fixed at $25 ($5 preparation,
$20 simulation); this $13 CPU-stage quote is inside the simulation authority.

A read-only inspector matches provider requests and responses to signed
reservation/completion records, verifies the closed SQLite conversation and
local tool results, restores the CAD/render candidate, and verifies independent
review receipts and final artifacts. A completed render can resume after a
process interruption without repeating paid author turns or local CAD/Blender
work. Unknown local tool outcomes fail closed. The enclosing scene stage still
refuses cross-attempt Claude adoption; a failed attempt requires a new bounded
controller decision, not an ad hoc replay.

Hermetic tests exercise the actual SDK tool loop and a fake transport through
the signed future-scene stage handler, including a second handler pass with no
duplicate provider or tool calls. They prove the $7 reservation logic with fake
usage, not that real CAD authoring will fit $7. A no-inference Models API
preflight using the protected local key confirmed the model ID, 1M context,
128K max output and required capabilities on 2026-09-23. It made no billed
Messages call. A real signed, capped attempt is still required to observe
billing, native import, and policy results. The control-plane host needs its
scoped Anthropic secret installed before such an attempt.

Official references: [Opus 5.5 model and price](https://platform.claude.com/docs/en/models/opus-5-5/overview),
[Models API](https://platform.claude.com/docs/en/api/http/models/retrieve).
