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
before another arm/case can dispatch. Security failures likewise stop the phase.
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
