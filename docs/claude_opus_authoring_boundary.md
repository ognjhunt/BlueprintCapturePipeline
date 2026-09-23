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

The optional `ClaudeNativeToolLoop` journals each raw model turn and its local
tool result before the next turn. It replays Opus 5.5's signed `thinking`
blocks unmodified with the matching tool result. It invokes the existing four
confined CAD/Blender tool definitions and pauses on a valid render for the
existing independent review. Unknown model or tool outcomes fail closed. A
new process verifies the transcript, tool receipts, CAD/Blender artifacts and
independent-review receipts, then restores completed asset state without
repeating those operations. In-flight provider or tool outcomes remain blocked
until independently reconciled. A fake-provider lifecycle test
exercises original-image interpretation, failed CAD repair, successful CAD,
Blender render, interruption after render, restoration, and independent
physics/appearance reviews without an Anthropic call.
`inspect_completed_claude_authoring` provides a read-only retained-result
gate; the default OpenAI retained-result validator still rejects Claude
unless that explicit gate has passed.

This does **not** replace the website capture's current persistent CAD/Blender
agent. That path uses the OpenAI Agents SDK's local SQLite conversation. Its
item format drops Claude's signed thinking blocks, so direct model swapping
would corrupt the conversation.

To enable a future scene, the website and controller must sign Anthropic into
the *new* scene's provider list and terms decision, bind a separate Anthropic
stage cap within the owner and per-attempt limits, and transport that authority
into the CPU child. Today `website_task_preparation` emits only `vast/openai`,
`task_evaluation_scene_execution_scope` rejects `anthropic`, and the paid
configuration authority/bundle and CPU stage-3 driver carry only the OpenAI
external-service cap, key scope, and cost gate. The driver also labels final
CAD/Blender receipts as GPT-6 Astra. An environment flag or a local key cannot
override these signed boundaries. The current scene has no Anthropic
authorization and cannot be switched in place. A future route also needs an
explicitly authorized provider canary and verification that the package and
retained-stage validators accept Claude-labelled evidence.

Claude Managed Agents does not meet this run's hard $7 per-attempt guard by
itself: its session budget is checked between model requests and Anthropic
allows the final request to finish above the cap. Its hosted sandbox and
tool behavior would also need their own qualification. It is therefore not
the path used here.

Official reference: [Opus 5.5 model and pricing](https://platform.claude.com/docs/en/models/opus-5-5/overview),
[thinking and tool-use continuity](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5),
[Managed Agents session budgets](https://platform.claude.com/docs/en/managed-agents/sessions).
