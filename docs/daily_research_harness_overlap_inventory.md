# Daily research: managed harness overlap inventory

Design only, inspected 2026-10-04. Use the Agents API's managed Codex harness for
agent execution, context, planning and continuation. Keep Blueprint's evidence,
CRM, spend, provenance and draft permissions in application-owned contracts.
The inspected runner **already uses managed sessions**; there is no second local
model loop or local context-compaction implementation to delete. The opportunity
is to consolidate phase-specific choreography, not discard the durable ledger.

This supports **ADP-010 partner discovery, partner-phase day-7 admission**
([backlog:365–375][adp], [stop gate:299–300][day7]). That clock starts with the
partner-proof phase. A research result does not secure a partner, two candidates,
task truth, holdout authority or rights. The observed blocker is repeated
application lifecycle machinery around an already managed agent. The completion
artifact here is this source-pinned responsibility map and retirement criteria;
the smallest reversible change is one note. Existing infrastructure is sufficient
for execution but lacked this explicit ownership inventory. No migration,
parser change, release, activation, paid call or deployment is authorized here.

## Inspected sources and release boundary

Read-only ref checks at 17:13–17:19 UTC:

| Surface | Exact inspected revision / status |
| --- | --- |
| Pipeline `main`; isolated note base | `4474b9c18d464e376955d39dbc9ab73d016c74d2` |
| Pipeline research integration, PR [#2549](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/2549) | `aae6c5ad8d8b035682a1190eeedcebb7705cffcb`; open into main; [#2584](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/2584) and [#2585](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/2585) merged into this integration |
| WebApp `main` | `353b0e3715ff59a9bdf7a984c861ece8302306f4`; [package receipt][web-main-receipt] pins Pipeline `42ca18ec568f64b2dde0895a77c5d982e8a1bf6e` |
| WebApp draft [#848](https://github.com/ognjhunt/Blueprint-WebApp/pull/848) | `05f3b3ec72bfb126be12692b4aaf82a1ece402f6`; [candidate receipt][web-receipt] pins `aae6c5ad8d8b035682a1190eeedcebb7705cffcb` |

Every Pipeline code link below names **the integration SHA**, not moving main.
The candidate archive was inspected without installing or rebuilding it: 54
manifested source files match all 54 Pipeline source blobs byte-for-byte;
1,013,760 bytes; archive SHA256
`cf7f5086f580e49f81934e466f063536002dc2246bac368568748f47c98fd7b8`;
manifest SHA256
`c869553058e740aac200bcfd6ad8f844cf365f7f99a9a532385c17f3b36d7410`.

At that snapshot, [WebApp's integration note:10–24][web-integration] already
described the candidate as merged Pipeline main, contrary to the inspected
#2549 state. Treat that prose as release intent. Main's package receipt and
#848's reported prior live pin agree on `42ca18ec`, but neither proves current
Render deployment or Firestore control state. Those live systems were not read.
Later branch advancement does not change this inventory's source identity.

At the 17:27 UTC checkpoint, both main refs and Pipeline integration were
unchanged; #2549 remained open. WebApp #848 remained draft and advanced to
`3b550ab512414e437d1ad58a46381b71959bba9d`. Its
[receipt](https://github.com/ognjhunt/Blueprint-WebApp/blob/3b550ab512414e437d1ad58a46381b71959bba9d/vendor/daily-research/receipt.json)
still pins the same archive/source. The
[comparison](https://github.com/ognjhunt/Blueprint-WebApp/compare/05f3b3ec72bfb126be12692b4aaf82a1ece402f6...3b550ab512414e437d1ad58a46381b71959bba9d)
changes only contact evidence/resolution/visibility and a contact-refresh test;
the inspected package, installer and history host are unchanged. Contact-parser
behavior is outside this inventory.

## What the official API actually supplies

Official pages were read on 2026-10-04, including their recovery qualifications:

- [Overview](https://developers.openai.com/api/docs/guides/agents-api/overview):
  managed sessions, orchestration, context compaction and recovery; configured
  tools and execution environment remain application choices.
- [Architecture](https://developers.openai.com/api/docs/guides/agents-api/architecture):
  the harness runs the model/tool loop. The application submits work, receives
  events and handles function tools. Self-hosted compute lifecycle remains its
  responsibility. A tool-only session can use no environment.
- [Sessions](https://developers.openai.com/api/docs/guides/agents-api/sessions)
  and [events/items](https://developers.openai.com/api/docs/guides/agents-api/sessions/events):
  later input continues an idle session or steers an active turn. Saved items
  survive stream disconnects; streams do not replay missed events. Reconcile
  current state and paginate retained items. Closing a stream does not cancel
  work; explicit cancellation retains prior work.
- [Functions](https://developers.openai.com/api/docs/guides/agents-api/tools/functions):
  handle currently pending `required_actions`, return the same turn/call IDs,
  retain side-effect results durably, and check uncertain effects before rerun.
  This supports the adapter; it does not replace Blueprint's write claims.
- [Errors/recovery](https://developers.openai.com/api/docs/guides/agents-api/errors):
  check saved work and completed effects, bound retries, and continue unfinished
  work only after appropriate state checks. Failed sessions or expired hosted
  environments can require a new session. No unconditional restart or
  exactly-once external execution guarantee follows.
- [Web search](https://developers.openai.com/api/docs/guides/agents-api/tools/web-search)
  and [computer use](https://developers.openai.com/api/docs/guides/agents-api/tools/computer-use):
  configured search and browser tools let the agent choose queries/navigation.
  Browser activity and optional screenshots describe operations, not a
  Blueprint-specific attestation of an exact visible contact address.
- [Manage sessions](https://developers.openai.com/api/docs/guides/agents-api/sessions/manage)
  and [sandbox lifecycle](https://developers.openai.com/api/docs/guides/agents-api/environments/lifecycle):
  deleting a session may clean up asynchronously; session, environment,
  application retention and billing reconciliation are distinct concerns.

The proposed simplifications below are architectural inferences from those
capabilities, not promises that the API implements Blueprint admission policy.

## Responsibility inventory

**A** = candidate consolidation/migration of harness-level choreography;
**B** = Blueprint business/evidence rule that stays;
**C** = necessary API, storage or transport glue. Mixed entries explicitly split
the mechanism. No whole file is designated for deletion.

| Runtime mechanism and exact source | Classification and ownership |
| --- | --- |
| SDK session, turn/item pagination, artifact download: [runner:519–650][provider] | **C.** Already calls managed Agents API. Keep bounded pagination, exact resource binding and error handling; no custom model loop here. |
| Daily wake, per-date durable create intent, restart reconciliation: [runner:988–1182][start], [render:627–679][scheduler], [bridge:478–510][create] | **B/C.** Latest eligible date, overlap protection, intent before one claimed create, unchanged payload and unique saved-session reconciliation. The harness does not own the company schedule or authorize another paid create after an unknown response. |
| Root-turn observation and repeated phase observers: [runner:1207–1300][observe], [consumer:638–739][qa-observe], [render:76–181][workflow] | **A1/C** for duplicated observer/phase plumbing: consolidate around managed state/events with a small reconciliation adapter. **B** for exact turn/config/evidence binding, original deadlines, stop checks and terminal guard decisions. Events cannot replace those checks. |
| CRM identities, frozen knowledge/history/contact input: [runner:1020–1122][inputs], [history:37–83][history], [WebApp host:83–150][web-history] | **B/C.** Dated company evidence and access scope, not context-window compaction. Keep input hashes, grants, freshness, unknowns and untrusted-data boundary; the harness manages conversational context. |
| Output schema, evidence eligibility, exact duplicate checks and review packets: [runner:326–516][validation], [runner:1333–1399][packet], [runner:1456–1522][review] | **B.** Preserve raw discovery versus verified support versus commercial qualification; version-pinned assessments, CRM recheck, packet digests, inventory retention and no silent promotion. |
| Validation-repair turn choreography: [recovery:390–604][repair] | **A2** for local corrective-turn planning/observation: potentially let the existing agent use precise validation feedback inside managed execution. **B/C** for original failure, every revision/input hash, located exclusions, no-progress/attempt limits, shared budget, unchanged authority and uncertain-input no-resend. Validation still decides acceptance. |
| QA submission and corrective-turn choreography: [consumer:428–519][qa], [consumer:550–636][qa-correct] | **A3** for repetitive format-correction prompts and phase tracking. **B/C** for source/semantic-duplicate review, copied-placeholder refusal, exact candidate/packet/CRM bindings, at most two corrections, original deadline and durable event admission. A managed completed turn does not mean QA passed. |
| Typed transient QA input retry / resume: [qa_retry:69–120][retry-bind], [qa_retry:151–205][retry], [bridge:697–742][retry-gates] | **B/C.** Same immutable event/key, bounded typed-503 retry slots, complete saved-state/effect checks, Retry-After and fresh admission. Do not replace with blanket SDK retries or infer that idle means the earlier input was never accepted. |
| Offline normalization, quarantine, replay and best valid subset: [recovery:16–51][normalize], [recovery:127–168][replay], [recovery:606–743][recover], [runner:1401–1454][recover-output] | **B.** Evidence-preserving derivation with zero provider mutations; retain original failures and exclusions. Managed recovery does not decide which malformed scientific/business evidence can be accepted. |
| Pending function-call dispatch and saved results: [search:458–611][dispatch], [firestore:213–229][tool-fence] | **C** for `required_actions` → handler → tool-result routing. **B** for exact turn/call/request hashes, attempt before paid execution, immutable result bytes, scoped tools and fresh lease/control/deadline checks. Return retained results after restart; unknown execution is observation/error, not a new paid attempt. |
| General research, queries, alternatives and source access: [search:189–284][search-tools], [search:307–424][source] | Query/branch choice already belongs to the agent. **C** for replaceable search/read tools; **B** for complete receipts, source dates, redirects, raw digest, access/resource limits and explicit gaps. Generic navigation can use managed tools, but native search/browser output is not a drop-in replacement for this evidence receipt. No automatic new-provider fallback. |
| Optional paid expansion: [expansion:235–407][expansion], [firestore:134–189][allocation] | **B/C.** One daily paid claim independent of model call IDs; original-ID reads and ACK recovery; native cap/schema/authentication and verified all-in allocation. Managed session budgets do not supply this accounting producer or authorize duplicate paid starts. |
| Firestore blobs, claims, work projection and generation lease: [bridge:59–113][lease], [bridge:165–318][put], [bridge:650–695][gates], [bridge:1396–1426][heartbeat], [firestore:25–132][pipe] | **B** for durable company state, lease generation/expiry, fresh authority, immutable evidence and one-use claims. **C** for private Python/Node pipe and heartbeat. Session orchestration does not fence two application workers writing CRM or spending. |
| Same-session agent-owned publication continuation: [publication:48–229][publication], [consumer:403–426][publish-select] | **A4** for another phase-specific observer/continuation state machine. The agent already chooses destinations/presentation through tools. **B/C** for QA prerequisite, permitted destinations, retained input/decision, shared deadline, exact session/tool bindings and truthful completion only with readbacks. Existing sessions keep their creation-time tools. |
| CRM/report transport, retries and paginated effects: [publisher:32–156][plans], [publisher:170–299][sink], [bridge:743–932][agent-gates], [bridge:1043–1168][publish] | **B/C.** Current verification, duplicate/ID checks, exact plan/request digests, single claims, ordered batches, complete readbacks and unknown-write observation. A definitive rejected request may permit a changed presentation only after proven absence; an unknown write cannot. These are external-effect contracts, not harness duplication. |
| Contact gaps, bounded attempts and native proof reconciliation: [contact_research:24–75][contact-input], [contact_research:96–223][contact-proof] | **B/C.** Public allowlisted tasks, same daily session/budget, stable CRM/site identity, no sends, two-attempt ceiling, digest-bound source/QA receipts and quarantine. The module does not create a separate agent. Its static receipt requirement stays until the separate WebApp contact-proof design supplies an independently verified compatible replacement. |
| Stop, cancellation and deadlines: [runner:1184–1205][cancel], [consumer:521–548][qa-cancel], [recovery:648–657][repair-cancel], [publication:25–45][pub-cancel], [firestore:191–211][delete-fence] | **C** for API cancel/delete calls. **B** for operator authority, bound resource/key, durable attempts, unknown outcomes, unchanged absolute deadlines and no further side effects after revocation. Cancellation acknowledgement is not cleanup or a billing-stop receipt. |
| Export/archive, cleanup and deletion reconciliation: [render:302–469][export], [render:472–623][cleanup], [bridge:521–649][cleanup-gates] | **B/C.** Exact terminal inventories, archive/readback before durable delete claim, approved retention policy, authenticated absence checks and observation after uncertain deletion. The code explicitly leaves billing-stop verification false. API-managed cleanup does not replace company evidence retention. |
| Immutable export and WebApp installation/release pin: [standalone:12–86][package], [WebApp installer:12–49][installer], [render:50–73][pin] | **B/C.** Source-file allowlist, per-file/archive hashes and source identity. Preserve `manifest.source_commit == Firestore control.source_commit` before enabled creation; package installation itself is not activation or live-release proof. |

## Candidate retirement, risks and proof

All **A** candidates share this release invariant: new rows may use a new
reviewed profile only when the installed manifest and control pin the same exact
source commit. Preserve old row bytes, digests, original tool configuration and
consumed claims; no empty ledger, retrospective profile upgrade or rewritten
historical assessment. Drain active research, repair, QA, correction and
publication work, or route each in-flight row to its original compatible reader.
Source and WebApp package changes would be a later separately authorized release.

| Candidate | Migration risk and retained boundary | Retirement order and verification required before deletion |
| --- | --- | --- |
| **A1: shared observation adapter** | Losing a missed event or mistaking idle/terminal state can collect the wrong turn, recreate an in-flight row or drop its evidence. Retain session/turn IDs, payload/evidence hashes, complete pagination, create claim, original cancellation/deadline and manifest/control pin. | **1.** First consolidate read-only observation; keep durable states/readers. Prove stream loss, repeated events, pagination, restart during create/terminal collection, expired environment and concurrent lease loss against existing runner/render/store fixtures. No extra session or charged action. |
| **A2: agent-led validation repair** | Existing repair rows carry inputs, baseline turns and revisions that exports/contact proof depend on. Recomputed digests or a repeated repair input could charge again or change which evidence was accepted. Retain each revision, original raw hash, attempt/no-progress limits, one-use claim and old validator/profile under the same release-pin rule. | **2.** Only for new compatible rows, expose validation feedback to the existing agent after A1; keep deterministic acceptance/exclusion. Prove malformed JSON, repeated feedback, partial valid inventory, placeholder rejection, lost POST reply and stop/expiry produce the same retained evidence and no replay. Then retire only redundant dispatch/observation code. |
| **A3: agent-led QA formatting repair** | Removing the correction phase can bypass substantive source/duplicate review or forget an in-flight correction/retry claim. Preserve packet/CRM/assessment/raw hashes, versioned evaluator, original correction key, prior review, typed retry admission and two-attempt/deadline limits; enforce the common release pin. | **3.** Reuse validation feedback after A2 while retaining the protected QA decision. Prove copied placeholders, strings posing as booleans, null expiry, changed CRM/authority, accepted-but-unacknowledged inputs and restart across correction turns remain blocked/unresolved as appropriate. No new session or duplicate paid event. |
| **A4: shared publication continuation** | In-flight rows may have partial Notion batches, consumed CRM plans or uncertain writes. Replanning/rekeying can duplicate paid/tool effects or CRM rows; compacting away evidence can break receipt digests. Preserve exact input/tool results, plan/body/claim hashes, ordered batch history, destinations and readbacks; creation-time tools and source/control pin remain binding. | **4, last.** Consolidate observer plumbing only after A1–A3; preserve application publication transport. Prove write-before-crash, unknown reply, partial ordered readback, changed authority, definitive rejection with proven absence, duplicate CRM identity and missing publication tools all reconcile without another uncertain write. Retire legacy readers only after every old row is terminal and portable exports round-trip. |

Use synthetic saved sessions and fake tool/sink/store responses for these later
proofs. Existing reference coverage includes `test_daily_research_runner.py`,
`test_daily_research_render.py`, `test_daily_research_recovery.py`,
`test_daily_research_repair_gaps.py`, `test_daily_research_consumer.py`,
`test_daily_research_qa_retry.py`, `test_daily_research_agent_publication.py`,
`daily_research_store.test.mjs`, `daily_research_publisher.test.mjs`,
`daily_contact_research.test.mjs` and the standalone/cleanup suites at the pinned
source. These are verification targets, not a claim of a completed migration.

This note's checks are source/ref inspection, candidate archive/blob equivalence,
link/line verification, independent classification review and public-content
sanitization. No runtime suite is needed for a prose-only change. The public note
contains code contracts and public commit/package hashes only; it excludes
prospect names, private resource/run IDs, retained run bytes, mailboxes and
credentials. The separate contact-proof note owns any visibility-receipt design.

[adp]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/docs/arm_decision_proof_v1/IMPLEMENTATION_BACKLOG.md#L365-L375
[day7]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/docs/arm_decision_proof_v1/README.md#L299-L300
[web-main-receipt]: https://github.com/ognjhunt/Blueprint-WebApp/blob/353b0e3715ff59a9bdf7a984c861ece8302306f4/vendor/daily-research/receipt.json#L1-L6
[web-receipt]: https://github.com/ognjhunt/Blueprint-WebApp/blob/05f3b3ec72bfb126be12692b4aaf82a1ece402f6/vendor/daily-research/receipt.json#L1-L6
[web-integration]: https://github.com/ognjhunt/Blueprint-WebApp/blob/05f3b3ec72bfb126be12692b4aaf82a1ece402f6/docs/daily-research-render-integration.md#L10-L24
[provider]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/runner.py#L519-L650
[start]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/runner.py#L988-L1182
[scheduler]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/render.py#L627-L679
[create]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/firestore_bridge.mjs#L478-L510
[observe]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/runner.py#L1207-L1300
[qa-observe]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/consumer.py#L638-L739
[workflow]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/render.py#L76-L181
[inputs]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/runner.py#L1020-L1122
[history]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/history.py#L37-L83
[web-history]: https://github.com/ognjhunt/Blueprint-WebApp/blob/05f3b3ec72bfb126be12692b4aaf82a1ece402f6/server/research-learning/research-worker-host.ts#L83-L150
[validation]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/runner.py#L326-L516
[packet]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/runner.py#L1333-L1399
[review]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/runner.py#L1456-L1522
[repair]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/recovery.py#L390-L604
[qa]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/consumer.py#L428-L519
[qa-correct]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/consumer.py#L550-L636
[retry-bind]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/qa_retry.py#L69-L120
[retry]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/qa_retry.py#L151-L205
[retry-gates]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/firestore_bridge.mjs#L697-L742
[normalize]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/recovery.py#L16-L51
[replay]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/recovery.py#L127-L168
[recover]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/recovery.py#L606-L743
[recover-output]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/runner.py#L1401-L1454
[dispatch]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/search.py#L458-L611
[tool-fence]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/firestore.py#L213-L229
[search-tools]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/search.py#L189-L284
[source]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/search.py#L307-L424
[expansion]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/expansion.py#L235-L407
[allocation]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/firestore.py#L134-L189
[lease]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/firestore_bridge.mjs#L59-L113
[put]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/firestore_bridge.mjs#L165-L318
[gates]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/firestore_bridge.mjs#L650-L695
[heartbeat]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/firestore_bridge.mjs#L1396-L1426
[pipe]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/firestore.py#L25-L132
[publication]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/publication.py#L48-L229
[publish-select]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/consumer.py#L403-L426
[plans]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/publisher.mjs#L32-L156
[sink]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/publisher.mjs#L170-L299
[agent-gates]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/firestore_bridge.mjs#L743-L932
[publish]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/firestore_bridge.mjs#L1043-L1168
[contact-input]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/contact_research.mjs#L24-L75
[contact-proof]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/contact_research.mjs#L96-L223
[cancel]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/runner.py#L1184-L1205
[qa-cancel]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/consumer.py#L521-L548
[repair-cancel]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/recovery.py#L648-L657
[pub-cancel]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/publication.py#L25-L45
[delete-fence]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/firestore.py#L191-L211
[export]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/render.py#L302-L469
[cleanup]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/render.py#L472-L623
[cleanup-gates]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/firestore_bridge.mjs#L521-L649
[package]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/standalone.py#L12-L86
[installer]: https://github.com/ognjhunt/Blueprint-WebApp/blob/05f3b3ec72bfb126be12692b4aaf82a1ece402f6/scripts/install-daily-research.py#L12-L49
[pin]: https://github.com/ognjhunt/BlueprintCapturePipeline/blob/aae6c5ad8d8b035682a1190eeedcebb7705cffcb/tools/daily_research/render.py#L50-L73
