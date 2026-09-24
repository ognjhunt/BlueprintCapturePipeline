# Claude Opus 5.5 Agents SDK reconciliation

This is an opt-in, no-network integration proof on a future-scene branch. It
does not select Claude in the paid worker or alter the current Astra scene.

The local OpenAI Agents SDK `Model` adapter translates source images, strict
tools, JSON output schemas and signed Claude thinking blocks. Its client bridge
submits each translated Messages request to the existing Claude invoker. That
invoker verifies signed Anthropic authority and the scoped `_FILE` secret,
reserves the published full one-million-token input window plus 12,000 output
tokens before dispatch, receipts actual provider usage, and retains the full
quote for an uncertain outcome. Independent physical and visual reviews use
the same provider-specific ledger.

The hermetic test performs observe → CAD → render with a real Agents SDK
`SQLiteSession`, then two independent review calls. Five sequential completed
calls fit the existing $7 Anthropic cap under its deliberately small fake usage
because each completion releases the unused portion of its $4.664 worst-case
reservation. An unresolved first call retains $4.664 and blocks the next
request. The exact translated message, tool-schema and image request is checked
before the fake send; its local submitted-input estimate is below 80,000 tokens,
but the **signed reservation still uses the full one-million-token window**.
This test proves admission behavior for its fake receipts, not that a real CAD
run's five calls will cost the same or necessarily fit $7.

The bridge is not ready for website selection. Before replacing the native
loop in the future-scene stage, it needs a durable request/response journal
around each Agents SDK call, a read-only completed-session inspector that binds
SQLite history to Anthropic reservation/tool/review receipts, safe process and
cross-attempt recovery, and exact fake-transport stage-driver coverage. A
provider canary after separate scene authority must verify the actual Claude
request shape and billing. Until then, #2141's native route and this SDK bridge
remain separate and unmerged.
