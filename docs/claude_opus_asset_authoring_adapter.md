# Claude Opus 5.5 asset-authoring adapter (not yet a paid profile)

`task_object_claude_model.ClaudeMessagesModel` maps one stateless native
Anthropic Messages response into the local OpenAI Agents SDK `Model` interface.
It retains the SDK's SQLite session, local function tools, original-frame image
bytes, structured output schema, tool-call IDs, signed thinking blocks and
provider token usage. It refuses remote image URLs, hidden server conversations,
unrecognized SDK items, incomplete responses and parallel tool calls. A scoped
Anthropic client must be supplied by the caller; the adapter disables client
transport retries.

The adapter is intentionally **not selected** by the CPU authoring worker. The
current worker reserves and receipts every authoring/model-review call as
`gpt-6-astra`/OpenAI and uses OpenAI-specific prices. Routing a Claude call
through that ledger would misstate the provider, price and spend. Before a
future separately identified scene can select this profile:

1. Resolve its secret from the existing scoped `_FILE` mechanism and build a
   zero-retry Anthropic client. No ambient key fallback.
2. Make the reservation, completion and retained-record schemas carry the
   actual provider and model. Use a conservative Claude Opus 5.5 input/output
   quote, including image and cache uncertainty, under the same per-attempt
   and cumulative caps. An unknown bill keeps its full reservation.
3. Route independent physical-property and visual reviews with correctly
   labeled/provider-priced calls, or explicitly keep those reviews on OpenAI
   with separately accounted receipts. Preserve the existing independent
   acceptance, native-import and policy-action checks.
4. Exercise the actual installed Anthropic SDK request shape in a no-spend
   transport fixture, especially `output_config.format`, strict local tools,
   image/tool-result content and signed thinking continuity. Then complete the
   focused changed-contract tests and normal paid-resource admission before a
   bounded provider canary. Do not adopt an Astra session into Claude.

The hermetic adapter tests make no provider calls. This branch does not affect
the active drawer run and cannot be used to claim Claude authoring is ready.
