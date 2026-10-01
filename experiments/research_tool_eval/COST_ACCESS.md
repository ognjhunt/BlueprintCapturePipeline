# One-time cost and access checklist

No spending has been authorized for execution. The mock run costs $0.
All numbers below are estimates from official documentation checked September 30,
2026; recheck tariffs, account contract and service tier immediately before an
approved evaluation. The source facts remain frozen at September 30 even if a
later run refreshes prices or finds new evidence.

| Stage | Provider requests | Sol answers | Provider cost | Sol token-envelope cost | Combined estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 20 × four raw arms | 80 | 80 | $0.240 | $2.560 | **$2.800** |
| Add six hard cases × Core/Pro | 12 | 12 | $0.750 | $0.384 | **+$1.134** |
| Both stages | 92 | 92 | $0.990 | $2.944 | **$3.934** |

Sol standard short-context rates are $2/M input and $10/M output per current
[OpenAI pricing](https://developers.openai.com/api/docs/pricing). Each answer
reserves 8,000 input and 1,600 output tokens including reasoning ($0.032).
Typical 4,000 input/800 output would produce $1.52 for raw only or $2.462 for both
stages. Cache savings are not assumed. Tokens from initial/final extra model
turns, hosted environment use or hidden tools must be included by the actual
existing researcher adapter; these are currently unverified.

**Proposed parent-approved dispatch budget: $10 total**, raw-only or both stages
as explicitly selected. This is a software admission/reservation budget, not a
hard provider-enforced all-in billing cap. Unknowns include incremental hosted
Agents API compute, tax/region/tier/contract differences and uncertain paid work.
No uncapped cost component may be silently assigned zero. Live enabling requires
a bounded-accounting decision for these unknowns or a revised estimate/cap.

The full protocol uses 184 logical requests (92 provider + 92 controller).
Worst-case proven-unaccepted retries allow 368 dispatch attempts; unsuccessful
acceptance must be positively established before retry. Task result observation
is separately bounded at 12 × 24 = 288 GET calls and does not submit new tasks.
Whether all observations are unbilled must be verified with the provider/account
contract before admission. Synchronous uncertain work pauses its cell for
reconciliation. No automatic replay expands the paid work count.

Parallel's [current FAQ](https://docs.parallel.ai/resources/faqs) says app/org
monthly spend limits **notify only** and do not block requests. Perplexity's
[project billing guide](https://docs.perplexity.ai/docs/getting-started/projects)
describes prepaid credits and blocking when depleted, with optional auto reload.
Neither establishes an enforced shared cap across Parallel, Perplexity and Sol.
No billing settings or top-ups were changed here.

| Target | Required capability/scope | Secure parent route | Gate/cost/terms |
| --- | --- | --- | --- |
| Stripe Projects catalog | Read-only provider catalog search | Existing CLI/session on a trusted machine; current [Linux installation guidance](https://docs.stripe.com/cli/install), CLI ≥1.40.0 and Projects plugin | CLI absent here. No catalog result is available; membership of Parallel/Perplexity is **unknown**, not absent. Do not initialize a project, accept terms or authenticate as part of this experiment. |
| Parallel Search | Only Search calls for frozen public cases; `x-api-key` auth | After catalog-route decision, authorized account owner at [Parallel Platform](https://platform.parallel.ai), keys via Settings, secret installed through canonical secret integration | No credential verified. Review customer terms, retention/training and billing; least app/account scope supported by provider. No account creation or purchase performed. |
| Parallel Core/Pro | Only Task create/result reads; no private MCP/connector inputs | Same existing authorized account and secure secret mechanism | Independent empty per-run/per-cell app memory scopes; confirm retention and default/index-partner behavior. This is not a documented memory-disable switch. Extra connectors cannot be added. |
| Perplexity Search | Search endpoint only for frozen public cases; bearer auth | Authorized project admin at [API keys](https://console.perplexity.ai/project/keys); canonical secret integration | No credential verified. Review provider terms, retention/training and credit/top-up policy. Do not buy credits, enable auto reload or create an account without parent/user authorization. Search-only key restriction support remains unverified. |
| Existing OpenAI Sol researcher | Existing project/model and Agents API turn/function/result access; no new agent edit | Parent's existing secure Default credential/launcher; if replacement is needed, explicit human decision through OpenAI Platform secure provisioning | Credential/launcher absent on this VM. The existing access grant, model snapshot/tier, turn cancellation and full cost/usage contract must be verified. No new credentials or hosted session created. |

The Stripe Projects skill was read before attempting catalog work. Stripe CLI
is absent, and the documentation connector returned an authentication gate
without results. That route stopped; no login/retry, project init, provider link,
legal acceptance or credential handling occurred. Following parent-controlled
installation/access, the exact read-only catalog checks are:

```sh
stripe --version
stripe projects search parallel --json
stripe projects search perplexity --json
```

Do not treat a failed/unauthenticated lookup as an empty catalog. If the Projects
plugin must be installed, the parent must choose the trusted environment and
review that setup step. Do not force a Stripe integration when unsupported.
No direct signup is proposed as a way around the pending catalog/access decision.

Current Parallel documentation needs explicit reconciliation:
[Task schema](https://docs.parallel.ai/api-reference/tasks/create-task-run)
says omitting `memory_scope_key` uses personal memory if available;
[Memory guide](https://docs.parallel.ai/resources/memory) says personal memory
does not alter API results and retrieval is explicit. The adapter sets documented
isolated application scope keys, unique per run/case/processor and stable on
resume. This avoids the omitted default, but does not prove disabling storage or
all contextual access. Confirm scope emptiness/isolation and accepted retention
terms before live Task runs. Never delete existing account memory to make an
evaluation work.

Parent paid-readiness checklist:

1. Review and sign off the 20 questions, dated primary facts, critical error rules
   and six-case hard subset. Verify all local read hashes with `validate
   --require-caches`; recheck or revise stale/unavailable facts explicitly.
2. Choose raw-only versus raw plus Task. Resolve the catalog route and secure
   existing provider access; never send a secret into chat or repository files.
3. Approve an explicit USD limit and unknown-cost handling. Bind the paidrun to
   an immutable clean commit and input hashes, with the repository's canonical
   fail-closed paid-resource admission receipt. An editable approval file is not
   a substitute for that authority.
4. Implement/review the small live boundary for the existing Agents API
   researcher: tokenizer checks, exact model/snapshot, total output/reasoning cap,
   no extra tools, verified environment cost, durable session/turn IDs and actual
   cancellation/reconciliation. Keep current live agent settings unchanged.
5. Verify transport TLS/auth/redaction, byte/time limits, provider acceptance
   evidence, invoice/usage mapping and the Task memory/connector conditions.
6. Repeat the focused hermetic checks. Admit the selected frozen run only;
   review blinded fields/claims and attach authoritative costs before selection.

These are the remaining execution blockers, not completed setup. No merge,
deployment, schedule or paidrun is authorized by this branch or a draft PR.
