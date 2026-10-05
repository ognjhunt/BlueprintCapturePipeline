# Private bounded adaptive comparison v1

This opt-in protocol compares the same frozen 20 public cases in Parallel
Fast/Advanced and Perplexity Fast/standard. It uses `gpt-6.1-sol` exclusively,
with the existing Default project `proj_F2tFJuxLaovJru8RrtXRaqNj`. It is isolated
from production and from the earlier single-search diagnostic protocol.

For each case and arm, the controller receives one search, checks relevance and
coverage, optionally requests one distinct entity-first search, then synthesizes
from retained evidence. Both arms have the same opportunities and downstream
context/output limits. There are no transport retries, recounts, third searches,
native model search, provider/model substitutions, or direct page fetches.
Parallel Extract is documented, but a comparable direct Perplexity Search page
retrieval route was not verified; this revision compares search alone.

The initial keywords are frozen, 3–6 words, entity first, at most 200 characters.
Parallel receives those keywords separately from the case-specific objective.
Perplexity has no objective field, so its natural-language query contains those
same keywords followed by the same objective. Generic grading instructions are
supplied to the controller only. Native character/token controls differ; the
controller additionally clips evidence to the same character limits. Any search
warning retains an unscorable contract diagnostic and stops the whole phase
before another arm/case can dispatch. Transport/admission security failures likewise stop the phase; unsafe individual
citation sources are quarantined with their raw evidence retained.
All controller and reviewer developer instructions supply the trusted date
2026-09-30; retrieved text cannot change it.

The fresh execution owner's audit established that all 20 earlier Parallel
keyword inputs were truncated to 200 characters. The two pilot inputs were
1435/1398 characters, with the entity at offset 724. The old objective remained
intact. This invalidates interpretation of that request contract as a clean
keyword comparison. Identical irrelevant A/C results remain unexplained; the
false July 17 clock in one answer was absent from retrieval. Neither finding is
used as ground truth or exposed through provider labels to the reviewer.

The audit/reconciliation checkpoint was 112 completed dispatches, 37 diagnostic
answers, no extra calls, and $2.285510 reserved. This is a parent-reported
checkpoint, not locally inspected billing. Earlier receipts, raw responses,
scope, grants, and holds remain intact. Adaptive results use
`protocols/bounded_adaptive_v1/` under the **same** existing aggregate run root.
They must not be pooled with diagnostic scores.

## Conservative admission

| New work | Maximum calls | Reserved USD |
| --- | ---: | ---: |
| Raw searches, two per 80 cells | 160 | 0.480000 |
| Search extras allowance | included above | 0.500000 |
| Sol coverage: 1,500 input / 512 output | 80 | 0.709600 |
| Sol synthesis: 3,000 input / 1,216 output | 80 | 1.572800 |
| Controller input-count allowance, $0.02 each | 160 | 3.200000 |
| Independent four-arm review: 6,000 input / 2,048 output | 20 | 0.709600 |
| Reviewer input-count allowance, $0.02 each | 20 | 0.400000 |
| **Maximum increment** | **520 HTTP dispatches** | **7.572000** |

All Sol input is reserved at the $2.50/M cache-write rate, output at $10/M,
without relying on cache hits. With $2.285510 of retained prior exposure, the
total reservation bound is **$9.857510**, leaving $0.142490. Every dispatch
rechecks the full matrix plus independent review against the shared $10 ledger
under its existing exclusive lock. Other/prior exposure must be at most
**$2.428000**. The runner refuses a full matrix that no longer fits; it never
silently selects fewer cases or releases an earlier hold.

The official input-count route is used, with an exact measured input cap and
bounded output settings. Over-cap inputs or incomplete outputs produce explicit
unknowns without another call. The count endpoint tariff is not stated in the
official docs, so $0.02 per request is an allowance, not a verified price.
Search extras likewise have an explicit allowance. Taxes, invoice adjustments,
and actual provider billing remain unreconciled: this is a conservative
reservation plan conditional on those allowances, not a guaranteed invoice
ceiling. Parent/live owner must resolve any charge that would exceed them before
execution. Ledger reservations never claim actual billed spend.

## Fresh-owner commands

Only task `01a0f3b8-6abe-775b-bfea-5102185b80ce`, published catalog
`6abd5bd3b9f081a18f7946316d12ad95`, executes live work. Reuse its existing
aggregate root and `live_access.json`; creating another run root is not a resume.
No credential values belong in commands, receipts, chat, or committed files.
The old implementation task performs only mock verification.

### Current four-step continuation

The corrected first search and its count/assessment succeeded. The accepted
follow-up then encountered benign tracking/anchor and numeric pagination syntax.
The parser now keeps bounded unsigned numeric pagination on any public citation,
removes known bounded UTM tracking and safe anchors from display citations, and
quarantines unsupported/unsafe individual sources. Userinfo, credentials,
redirect targets, signed/unrecognized queries, duplicate keys and malformed
values cannot be stripped into an accepted source. No citation is requested.
Original URL/evidence bytes remain in the retained raw envelope. Display changes
or quarantine retain raw URL digests and ranks in durable citation audits;
quarantined titles/excerpts never reach Sol. Real provider warnings and malformed
response contracts still stop the whole phase.

The parent supplied the exact ten URLs and four-step checkpoint. The offline
URL audit accepts all ten in all four modes, strips only Tracxn's tracking/anchor
for display, preserves Chef pagination and both HTTP URLs, and confers no vendor
authority. See `retained_case01_followup_url_audit.json`. This task-input audit
is not Library materialization or independent evidence of vendor claims.

The existing effective scope is used and has four paid completed steps. **Do not
use unused-scope adoption or create another run root.** Check out the reviewed
remote commit, then authorize this network-free continuation:

```sh
EVAL_ROOT='/workspace/provider-eval-private-live-20260930'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_adaptive_v1.runner \
  --output "$EVAL_ROOT" --access-receipt "$EVAL_ROOT/live_access.json" --phase pilot \
  --continue-retained-attempt 31b8682404eb1d9b171ad98f13e584ce7ebb66e8a01a523d5d44c668e80d9fa3 \
  --retained-envelope-sha256 b2573b5a447069c9dff1bdde5b669c1aefb1f64a901ea96e5fe2b8fba7d89f12 \
  --expected-adaptive-scope-sha256 09680b135d5bd7dded1ee8cbcdb120bf894f4277eb5ed3c8cc8223bcdb2f19c0 \
  --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce
```

The free command verifies the original plan, pinned prior code/first continuation,
both failed receipts, and exactly four completed dependent requests/envelopes:
search1, assess_count, assess, search2. It reconstructs and verifies the original
assessment input, the 775 count/usage agreement and the retained decision's exact
follow-up query. Original first-search model inputs remain byte-equivalent.
Changes to those dependencies refuse continuation before a paid stage; no recount,
assessment rerun, search retry, reservation, ledger transition or release occurs.

`followup_citation_continuation.json` binds the new code and all four retained
steps. Both `receipts/01_parallel_fast.json` and
`continued_receipts/01_parallel_fast.json`, original first continuation, raw
files, scope, allocation binding, journal prefix and holds remain immutable.
Resume writes `followup_receipts/01_parallel_fast.json`; pilot gating and the
isolated reviewer resolve this same explicit sidecar.

Then resume, without continuation flags:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_adaptive_v1.runner \
  --output "$EVAL_ROOT" --access-receipt "$EVAL_ROOT/live_access.json" --phase pilot --execute \
  --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce
```

All four completed steps are reused. The first new request is **synthesis input
counting**. At most **44 new pilot HTTP requests** remain and at most **$0.609120**
new pilot reservation. The parent-reported checkpoint is 233 events, 116 completed,
zero uncertain and $2.322630 reserved. Of the unchanged $7.572000 adaptive envelope,
$0.037120 is already held, leaving at most **$7.534880** new reservation including
remaining cases and isolated review. The proposed aggregate reservation bound is
still **$9.857510**; actual bills/taxes remain unreconciled.

### Earlier unused-preflight setup

From the repository root, replace the following path with the owner's existing
aggregate root. If it has the old `63ad365` preflight scope, run the explicit
adoption command below first; otherwise use this network-free preflight:

```sh
EVAL_ROOT='/absolute/path/to/existing-aggregate-run'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_adaptive_v1.runner \
  --output "$EVAL_ROOT" --access-receipt "$EVAL_ROOT/live_access.json"
```

The live owner's earlier network-free preflight created an unused scope with
digest `d3d1430a7ddc6f8840f6d3c6ca115cacc5b47eae36b17583c5759e84aff0fbb6`.
For that existing root only, explicitly adopt the reviewed phase-control patch
before executing. This command makes no HTTP calls:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_adaptive_v1.runner \
  --output "$EVAL_ROOT" --access-receipt "$EVAL_ROOT/live_access.json" --phase pilot \
  --adopt-unused-scope-sha256 d3d1430a7ddc6f8840f6d3c6ca115cacc5b47eae36b17583c5759e84aff0fbb6 \
  --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce
```

Adoption requires the exact owner/digest and pinned previous reviewed code hash,
unchanged non-code scope fields, no adaptive reservations (even released ones),
and no adaptive raw/receipt/review artifacts. It preserves the original scope
bytes and every prior journal entry, recording only `unused_scope_adoption.json`.
Ordinary subsequent admission resolves that immutable receipt. Adoption cannot
be combined with `--execute` or used after an adaptive attempt.

After the parent's retained-response audit and immutable reviewed-commit gate,
execute or resume the two-case pilot (cases 1–2, eight cells):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_adaptive_v1.runner \
  --output "$EVAL_ROOT" --access-receipt "$EVAL_ROOT/live_access.json" --phase pilot \
  --execute --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce
```

The runner defaults to `pilot`, never to the full 80 cells. Maximum pilot
search/controller reservation is **$0.646240**, with at most 48 HTTP dispatches.
Full-plan admission still preserves the complete 20-case matrix and independent
review budget. After the owner reviews the pilot, select cases 3–20 explicitly:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_adaptive_v1.runner \
  --output "$EVAL_ROOT" --access-receipt "$EVAL_ROOT/live_access.json" --phase remaining \
  --execute --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce
```

Both remaining preflight and execution verify all eight pilot receipts against
their retained responses, without initiating missing research. Missing, altered
or protocol-warning pilot evidence blocks remaining before dispatch.

Accepted steps are verified and adopted without a second reservation or request.
An uncertain submission stops execution, retains its entire hold, and requires
owner reconciliation. It is never automatically retried. Code, original scope,
public bytes, access receipt, query sequence, and adaptive scope are frozen.

Independent grading of the completed 20-case matrix runs in a separate process.
The parent must supply the
actual local oracle/spec mapping, never a synthetic fixture:

```json
{"spec": "the actual frozen reviewer/spec.md text", "cases": {"BP-EVAL-01": "the actual case oracle", "BP-EVAL-02": "...all 20 case IDs..."}}
```

This wrapper is an integration contract; it does not claim that the unavailable
Library oracle was transferred or that its original schema was inspected here.
The parent preserves original oracle/spec hashes while mapping their real
content. The reviewer pins the resulting spec and per-case oracle hashes.
The controller never reads this file, and raw search providers never receive it.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m experiments.provider_eval_adaptive_v1.reviewer \
  --output "$EVAL_ROOT" --access-receipt "$EVAL_ROOT/live_access.json" \
  --reviewer-bundle /absolute/private/path/to/actual-reviewer-bundle.json \
  --execute --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce
```

The reviewer verifies the four receipts against retained responses and cannot
initiate missing research. It sees anonymous arm numbers, frozen truth, full
answers, and bounded evidence excerpts; partial excerpts are a stated limitation.
It separately assesses source entailment, primary provenance, dated citations,
contradictions and appropriate unknowns. Contract stops are unscorable;
unsupported claims/unknowns remain visible. Its judgment needs parent
adjudication against the oracle; no controller self-grading or automatic winner.

Raw request/response envelopes include latency and remain digest-bound to
completed journal attempts. Case receipts retain answers, sources, operational
coverage, explicit stops and attempt digests. Reviews retain oracle/spec digests.
The exact supplied public bytes retain SHA256
`cac8d7a31aea2ad1c2e5a47ea37e434abcaae4b4910ba4afc8f81dd11c404ff3`.
Library ZIP transfer remains unresolved; actual public inputs came from the
parent-message route, not a successful Library materialization.

## Official request and pricing references

- [Parallel search best practices](https://docs.parallel.ai/search/best-practices)
- [Parallel Search reference](https://docs.parallel.ai/api-reference/search/search)
- [Parallel pricing](https://docs.parallel.ai/getting-started/pricing)
- [Perplexity Search reference](https://docs.perplexity.ai/api-reference/search-post)
- [Perplexity pricing](https://docs.perplexity.ai/docs/getting-started/pricing)
- [OpenAI token counting](https://developers.openai.com/api/docs/guides/token-counting)
- [GPT-6.1 Sol model and pricing](https://developers.openai.com/api/docs/models/gpt-6.1-sol)

Tests use synthetic evidence/oracles only; no test fixture is a paid case input.
