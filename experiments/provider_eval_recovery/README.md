# Private provider comparison recovery

Runnable now: an offline 20-case × four-mode replay, immutable per-cell request,
response and cost receipts, a bounded reservation/reconciliation journal, and a
separate deterministic reviewer process. No provider won: no live retrieval or
Sol inference was executed, and no real company case is integrated yet.

The 20 cases in `synthetic/` are entirely invented companies and hypothetical
sites. Their answers, sources, tokens and latencies are fixtures. Synthetic grade
accuracy is a lifecycle check, not a provider benchmark.

## Run locally

From the repository root (Python standard library only):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.provider_eval_recovery.test_harness -v
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.provider_eval_recovery.harness --output /workspace/provider-eval-offline-example
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.provider_eval_recovery.reviewer --output /workspace/provider-eval-offline-example
```

Repeat the replay against the same output directory to verify no resend. Each
run binds code, public input, fixture, rubric and oracle hashes. A changed plan
requires a new output directory. The provider/controller process hashes the
reviewer files but never parses the oracle; only the separate reviewer command
reads it. The wire adapters cannot load reviewer files, secrets or CRM.

The output contains `plan.json`, `journal.jsonl`, `raw/`, `receipts/`, `pending/`
when applicable, invocation summaries, and separate `reviews/` and review
summaries. Successful receipts retain public request envelopes, raw and normalized
sources, answer, controller usage, every attempt ID, synthetic provider latency
and actual-versus-simulated costs. Output files use private permissions.

## Source integration status

The supplied Library reference is
`libfile_01ae219642688191a6063e9d3fba9c0b`, version 0,
`Blueprint-20-Case-Provider-Eval-v1.zip`, 26,019 bytes, SHA256
`294f778b32670ee6ae07412a53c1627923f1004afeea501f32d3202e8b9cf736`.
The current Library skill's supported resolved-reference preparation succeeded,
but its current bundled materialization helper returned exactly
`library file transfer failed: download failed`. No readable ZIP was produced in
the executor; its expected hash is unverified. No alternate private URL or proxy
bypass was attempted.

`import_source.py` accepts **already materialized local bytes**, verifies the
specified count/hash, rejects traversal, links, duplicate names and oversized
expansion, and retains exact `public/inputs.json`, `reviewer/oracle.json` and
`reviewer/spec.md` bytes in separate partitions. It consumes the bytes it hashed.
It does not infer the unavailable real-case schema or silently mark integration
complete. Once transfer works, inspect the supplied manifest/README/spec, preserve
the reviewed prompts and oracle unchanged, map public inputs explicitly and
verify that no privateCRM or oracle fields reach a provider. Current replay
intentionally rejects a real bundle until that schema/spec review is completed.

## Approved cumulative budget

The user approved **$10 one-time total** across pilot plus remaining18, starting
with a two-case pilot under $1. Evidence:
`Sentinel_741e7736ff8c8191a085c8e2ece40576: yes i approve`.
No subscription/top-up, new credentials, grants or security changes are approved.
OpenAI must reuse Default project `proj_F2tFJuxLaovJru8RrtXRaqNj` and
`gpt-6.1-sol`.

All20 use one raw search per case/mode: 80 base calls, at most160 attempts
only after proven nonacceptance (pilot has exactly8 attempts and no retry).
At most3 Sol calls per cell, each capped at6000 total input and2048 total
output including reasoning. Full raw provider base is $0.24, conservative retry
ceiling $0.48, outer Sol uncached ceiling $7.7952, and maximum cache-write premium
$0.72. Subtotal **$8.9952**, plus **$0.50 provider-extras reserve**, is **$9.4952**,
leaving $0.5048 for tax/FX under the approved $10 cap. Any required token count
or additional fee outside that remainder stops execution; do not shrink the
reviewed rubric or substitute models to force admission.

Pilot: 2 cases ×4 modes ×1 search =8 provider attempts, $0.024 base. At most24
Sol calls: uncached $0.77952, cache-write premium $0.072. Subtotal $0.87552;
per-attempt extras reserve adds $0.025, for **$0.90052**, under the $1 cap.
Pilot and remaining18 must share one journal/scope and adopt pilot cells exactly
once. Repeat invocations neither rebudget nor rerun accepted or ambiguous calls.
Use `--phase pilot`, then `--phase remaining` against the same output directory
for the offline rehearsal. Those flags do not execute live providers.

TaskCore/Pro are excluded from this approval and remain a separately labeled,
unimplemented deeper experiment. Current actual external experiment spend is
**$0.00**; coding-agent session billing is not available as a provider invoice.
See [SECURE_SETUP.md](SECURE_SETUP.md) for exact hosts, headers and current
binding booleans. Secure access and the real-case schema remain blockers.

## Exact secure existing-account access needs

- Parallel: an existing authorized API account/key permitted for Search API,
  injected as `PARALLEL_API_KEY` into a future approved process from the
  established secret store; retain account/key identifiers and rate limits,
  not key values, in the private admission receipt.
- Perplexity: an existing authorized API organization/key permitted for Search
  API, injected as `PERPLEXITY_API_KEY`; existing API billing/credits must cover
  the approved cap. Consumer subscription access alone must not be assumed to
  provide API billing or key authorization.
- Outer controller/reviewer: an existing authorized OpenAI project credential
  with **`gpt-6.1-sol`** permission and Standard-tier billing, supplied through
  the canonical secret integration as `OPENAI_API_KEY` or that integration's
  equivalent. No model fallback chain. Reviewer runs with only its rubric,
  oracle and retained output, with no provider/CRM tool access.
- Human approval must identify the one-time pilot or main inclusive spend cap,
  approved public-input disclosure and provider retention/training terms, and
  the existing account/project identities. No key values in chat, Git, logs or
  receipts; no new keys/accounts, transfers, payment or subscription setup is
  performed by this recovery. Secure access remains pending; the cumulative $10 budget is approved.

The offline replay remains network-free. `live_http.py` now supplies minimal
HTTP seams for all three providers, validated with in-memory mocks and guarded
by the existing paid-resource admission grant, exact host/path/header checks,
frozen public inputs, pinned model/token-counter and cumulative budget journal.
It has no live runner CLI: real-case schema/rubric integration and owner-configured
existing access are still required. No live HTTP execution has occurred. There
is no publication or support-email prerequisite for private internal testing.

## Frozen comparison and limitations

Raw modes are Parallel Fast/Advanced and Perplexity Fast/standard. Each receives
the same prompt/scenario/query text, query order, one-call/attempt/time bounds,
maximum retained sources and a common 12,000-character evidence cap across
all admitted rounds. Wire excerpt caps use different native units and are explicitly
not equivalent: Parallel characters versus Perplexity tokens. Parallel GA Search
does not expose a result-count control in the current reference; the harness
retains at most ten after retrieval. The common research objective is included
in each query, rather than allowing provider-specific prompt optimization.
Future equal-treatment execution needs paired ordering or seeded alternation,
the same cutoff/time window, controller prompt/token caps, and retained exact
public source bytes. Model identity is pinned, but this offline scaffold does
not implement real planning or synthesis inference.

The journal reserves before a hypothetical dispatch and fsyncs each event.
Timeout, missing ID, malformed output, or a crash after reservation keeps full
charge exposure. An uncertain call cannot resend. Reconciliation is bounded at
two observations; unresolved exposure stays open. A fully retained response can
be adopted without resending, including its recorded latency. Only fixture-proven
nonacceptance enables one retry; that proof must never be treated as a live
provider guarantee. No undocumented idempotency header, synchronous result
lookup or Sonar asynchronous API is invented. Local attempt hashes are local
identity, not provider idempotency. Torn journals fail closed. Completed evidence
is recomputed and checked on replay rather than trusting modified receipts.

The separate reviewer validates unique matrix identities and receipt content against
frozen fixtures, retained raw digests and the journal before writing any grades.
Scaffold grading reports citation resolution, primary-source status, synthetic
literal entailment, supported/contradicted/unknown correctness and unsupported
assertions. Unknowns remain in the denominator. A matching URL alone cannot
validate a claim. Real semantic entailment, freshness, missing-company ambiguity
and explicitly hypothetical site scenarios require the supplied spec plus an
isolated Sol/human adjudicator. No live source fetch, injection-immunity claim,
quality ranking or site-readiness conclusion follows from these mocks. Citation
normalization currently rejects query/fragment-bearing URLs conservatively;
real-case integration must review that policy without leaking credential URLs.

## Current official documentation

Verified on 2026-09-30; the adapters are data envelopes, not paid calls.

- [Parallel Search reference](https://docs.parallel.ai/api-reference/search/search)
  documents `POST /v1/search`, `mode`, `objective`, `search_queries`,
  `max_chars_total`, `client_model`, and `results[].excerpts`.
- [Parallel modes](https://docs.parallel.ai/search/modes) list Fast at
  $1/1,000 requests and Advanced at $5/1,000.
- [Parallel pricing](https://docs.parallel.ai/getting-started/pricing) includes
  ten search results in the base price and Task Core/Pro at $25/$100 per 1,000
  successful runs. Deeper Task requests remain a separate experiment.
- [Perplexity Search reference](https://docs.perplexity.ai/api-reference/search-post)
  documents `POST /search`, `query`, `search_type: fast|web`, `max_results`,
  `max_tokens`, and returned snippet sources.
- [Perplexity pricing](https://docs.perplexity.ai/docs/getting-started/pricing)
  lists Search Fast at $1/1,000 and standard Search at $5/1,000 successful
  requests; Search does not add model-token charges.
- [Perplexity migration](https://docs.perplexity.ai/docs/sonar/quickstart)
  says Sonar asynchronous requests are no longer supported. The raw comparison
  uses Search; a separate future agent experiment must use current Agent API.
- [GPT6.1Sol model/rates](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
  lists Standard short-context rates per million tokens: input $2, cached input
  $0.10, cache write $2.50, output $10. Reasoning is billed within output.

## Scope and publication

This is the user's explicitly authorized private testing exception to the
repository's Arm Decision Proof focus rule. It claims no ADP backlog completion
or day-7/14/21/28/35/42 gate. The smallest reversible surface is this isolated
`experiments/` directory; production code, CLI packaging, runner #2486 and
snapshot #2487 are untouched. The repository's gstack link is broken in this
executor; independent GPT6.1Sol review applies the review discipline directly.

Recovery branch: `codex/private-provider-eval-recovery-20260930`, based on
`1f771e61`. All-state GitHub searches for `Parallel Perplexity` and `perplexity`
found no comparison PR. #2486 and #2487 are separate existing draft PRs. GitHub
reports this origin is **public**. Do not push private source bundles, receipts
or this private test to that origin. `DRAFT_PR.md` is a local reviewable draft;
remote draft creation requires an authorized private repository/destination.
No merge, deployment, schedule or production change is authorized here.
