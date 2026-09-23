# Claude Opus 5.5 authoring boundary

Status: optional future-scene preparation only. No live controller routing or
provider invocation is enabled by this change.

The local `ClaudeOpusAuthoringInvoker` can make one bounded Claude Messages
request for an image interpretation, CAD subtask, or independent review. It
requires a caller that verifies signed Anthropic disclosure authority and
provider terms, an `ANTHROPIC_API_KEY_FILE` reference, and a shared durable
inference reservation ledger. It reserves the cost of the model's entire
published one-million-token input window plus the configured maximum output
before dispatch. The unknown-outcome reservation stays charged; it is never
silently replayed. Model output never approves its own geometry or physics.

This does **not** replace the website capture's current persistent CAD/Blender
agent. That path uses the OpenAI Agents SDK's local SQLite conversation and
four confined CAD/Blender tools. Opus 5.5 returns signed `thinking` blocks
with tool calls. Anthropic requires those blocks to be replayed unmodified
with tool results. The current SQLite item format drops them. Routing a
Claude model through that format would corrupt the conversation.

To make a future scene's Claude tool loop runnable, build a native Messages
transcript journal that atomically retains every provider response block,
including signed thinking, before executing a local tool. Bind each tool call
and tool result to its message ID and immutable transcript digest; on resume,
prove every call/result pair and refuse unknown dispatch outcomes. Keep the
existing confined CAD/Blender tool implementations and independent validators.
Each Messages turn must pass through the same per-request worst-case ledger.
Then extend retained-phase validation and the controller's signed provider
authority, scoped secret wiring, model selection, and website provenance.
Run a no-network lifecycle rehearsal before any separately authorized scene.

Claude Managed Agents does not meet this run's hard $7 per-attempt guard by
itself: its session budget is checked between model requests and Anthropic
allows the final request to finish above the cap. Its hosted sandbox and
tool behavior would also need their own qualification. It is therefore not
the path used here.

Official reference: [Opus 5.5 model and pricing](https://platform.claude.com/docs/en/models/opus-5-5/overview),
[thinking and tool-use continuity](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5),
[Managed Agents session budgets](https://platform.claude.com/docs/en/managed-agents/sessions).
