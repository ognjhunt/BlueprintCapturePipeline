# Claude Opus 5.5 authoring boundary

Status: signed future-scene website and controller route implemented on an
unmerged branch. Existing Astra/OpenAI scene records and sessions are not
migrated. A newly authorized scene must explicitly select Anthropic and carry
an exact `anthropic:` provider-terms reference in its owner consent.

The website binds that selection into the scene's execution scope, permits only
`api.anthropic.com` for CPU authoring, and passes a scoped `_FILE` secret
reference. Stage 3 reserves at most $7 of Anthropic model exposure and $6 of
compute per attempt, under the fixed $25 new-scene development sponsorship
($5 preparation, $20 simulation). Reservation happens before each model call,
including independent physical and visual review. The full 1M input context
and bounded output are quoted at published US-only rates; an unknown bill
retains its full reservation. Provider terms, rights, expiry and scene identity
must match the signed grant.

The local OpenAI Agents SDK session authors through confined CAD and Blender
tools. A durable request/response journal preserves exact provider content,
including signed thinking, before any local tool executes. A read-only
inspector checks SQLite, provider and tool receipts, CAD/Blender artifacts,
independent reviews and the final asset. It can reuse a completed candidate
after an interruption. It fails closed on incomplete calls or tool outcomes.
Cross-attempt Claude stage adoption is not yet qualified; the controller must
not duplicate a prior paid request when an attempt fails.

Fake transport has exercised stage 3 and duplicate-free same-stage replay. A
protected local key successfully retrieved model metadata without an inference
call; this proves account/model availability but not Messages billing. The
control-plane host still needs a scoped Anthropic secret. A future capped
provider attempt must verify actual cost, CAD/Blender completion, native USD
qualification and the later GPU policy episode. None of these tests claims
that the drawer task succeeded.

Claude Managed Agents is outside this path because its session budget check
does not enforce this stage's per-request hard reservation. The local SDK
retains Blueprint's allocator, ledger, teardown and review boundaries.

Official references: [Opus 5.5 model and pricing](https://platform.claude.com/docs/en/models/opus-5-5/overview),
[Models API](https://platform.claude.com/docs/en/api/http/models/retrieve).
