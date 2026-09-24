# Claude Opus 5.5 asset-authoring adapter

`task_object_claude_model.ClaudeMessagesModel` maps one stateless Anthropic
Messages response into the local OpenAI Agents SDK `Model` interface. It keeps
the SDK's SQLite session, confined tools, original-frame image bytes,
structured-output schema, tool-call IDs, signed thinking blocks and provider
token usage. It refuses remote image URLs, hidden server conversations,
unrecognized SDK items, incomplete responses and parallel tool calls.

`claude_opus_sdk_bridge.ClaudeSDKMessageClient` supplies this model from the
existing signed Anthropic invoker. Requests and complete responses are
journaled before local tool execution. The bridge binds each journaled turn to
its signed reservation and completion, and read-only inspection ties those
turns to SQLite and local tool outcomes. An unresolved provider call or local
tool outcome cannot be retried as fresh spend.

The stage driver selects this path only when a fresh website scene carries the
signed Anthropic provider choice, terms and $7 model cap. OpenAI scenes retain
their current route. The provider secret comes only from the scoped
`ANTHROPIC_API_KEY_FILE` reference. The model may author geometry and
appearance; independent reviewers and deterministic native import checks keep
their own authority. No Claude model output alone qualifies a scene or policy.

This branch is not deployed. A fake-provider stage rehearsal and a live,
no-inference Models API check have passed. Real Messages billing, native import,
scene placement and GPU policy outcomes are unproven until a separately
identified authorized scene runs under the controller.
