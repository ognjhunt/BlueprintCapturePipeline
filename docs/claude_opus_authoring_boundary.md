# Claude Opus 5.5 authoring boundary

Status: optional future-scene website and controller route, implemented but not
deployed or provider tested. The current signed drawer scene remains on OpenAI.

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

This does **not** swap Claude into the website capture's current OpenAI Agents
SDK SQLite conversation. Its item format drops Claude's signed thinking blocks.
The opt-in route uses its own durable Messages transcript and the same local
CAD/Blender tools. The website must record an exact `anthropic:` provider terms
reference in the fresh scene's owner consent and select
`authoring_provider: anthropic`. The website then signs Anthropic into that scene's execution scope,
requests an Anthropic-only network and `_FILE` secret scope, and quotes $7 for
Anthropic plus $6 for compute under the existing $20 simulation authority.
The paid authority and CPU stage enforce those separate caps; absent authority,
terms, or a scoped key file fails closed. Default scenes still use OpenAI.

This has only hermetic fake-provider coverage. A new authorized scene and a
provider canary are still needed to prove Anthropic authentication, exact API
behavior, successful CAD/Blender output, native qualification, and billing.
One operational boundary remains: a completed native session can be inspected
after a process interruption, but the enclosing scene stage currently refuses
cross-attempt Claude adoption. An attempt that fails after CPU authoring must
be reconciled rather than blindly replayed. Do not interpret the code or tests
as a completed drawer policy run or an approved mid-scene model switch.

Claude Managed Agents does not meet this run's hard $7 per-attempt guard by
itself: its session budget is checked between model requests and Anthropic
allows the final request to finish above the cap. Its hosted sandbox and
tool behavior would also need their own qualification. It is therefore not
the path used here.

Official reference: [Opus 5.5 model and pricing](https://platform.claude.com/docs/en/models/opus-5-5/overview),
[thinking and tool-use continuity](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5),
[Managed Agents session budgets](https://platform.claude.com/docs/en/managed-agents/sessions).
