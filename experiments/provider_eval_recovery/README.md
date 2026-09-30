# Private Parallel / Perplexity comparison

The live CLI integrates the exact 20 provider-safe public cases, compares Parallel
Fast/Advanced with Perplexity Fast/standard, and runs identical `gpt-6.1-sol`
synthesis over retained evidence. No actual provider call has occurred in this
implementation task and no provider winner is established. Only fresh execution
owner `01a0f3b8-6abe-775b-bfea-5102185b80ce` may execute paid requests.

## Live handoff

Run from the repository root with `PYTHONPATH=src`. Python standard library only;
the repository virtualenv also works. Use **one canonical private output directory**
for pilot and remaining phases. Do not use another directory to reset exposure.

The fresh owner must write its sanitized access receipt from verified metadata;
`ACCESS_RECEIPT.example.json` gives the exact schema. Set `journal_root` to the
chosen absolute directory. All three keys must already be securely bound in that
fresh executor. The receipt binds catalog version, owner, model metadata HTTP 200,
Default project, allowed hosts, approved budget, initial zero spend and exact
counting method. It contains no key values. Its initial zero-spend assertion is
historical: subsequent invocations reconcile the same cumulative journal.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m experiments.provider_eval_recovery.live_runner --phase pilot --output /workspace/provider-eval-private-live-20260930 --access-receipt /workspace/provider-eval-access.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m experiments.provider_eval_recovery.live_runner --phase pilot --output /workspace/provider-eval-private-live-20260930 --access-receipt /workspace/provider-eval-access.json --execute --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m experiments.provider_eval_recovery.live_runner --phase remaining --output /workspace/provider-eval-private-live-20260930 --access-receipt /workspace/provider-eval-access.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m experiments.provider_eval_recovery.live_runner --phase remaining --output /workspace/provider-eval-private-live-20260930 --access-receipt /workspace/provider-eval-access.json --execute --execution-owner-task-id 01a0f3b8-6abe-775b-bfea-5102185b80ce
```

Without `--execute`, preflight checks the deterministic gates and retained pilot
integrity without networking. Remaining18 cannot start until all eight pilot
receipts match exactly three completed, hash-verified search/count/synthesis
attempts each. Accepted requests are adopted on resume. Uncertain requests stop
with their full reservation; no automatic resend or retry exists in this CLI.
An uncertain transport failure or oversize count requires offline reconciliation;
do not reset the journal or switch execution owners.

Parallel uses `POST https://api.parallel.ai/v1/search`, `x-api-key`. Perplexity uses
`POST https://api.perplexity.ai/search`, Bearer. OpenAI uses
`POST https://api.openai.com/v1/responses` and `/v1/responses/input_tokens`, Bearer
plus `OpenAI-Project: proj_F2tFJuxLaovJru8RrtXRaqNj`. Responses Write is required;
Models Read is needed only for the owner's already completed model metadata check.
The ordinary configured HTTPS proxy is preserved; redirects and alternate hosts
are refused. No security binding, key, account or grant is created by the code.

The receipt can state that Parallel header forwarding is verified. If it is
still **unverified**, explicitly set `parallel_header_pilot_probe_authorized:true`
and `parallel_x_api_key_supported:"unverified"`: the first admitted pilot search
itself tests forwarding, with its cost reserved before dispatch. A known failed
or denied capability (`false`) remains blocked. No extra authentication search or
proxy bypass is performed. Do not claim proxy support from a mock test.

## Counting, costs and receipts

Official OpenAI counting accepts the same model, exact input and reasoning
configuration and includes request formatting tokens. No local tokenizer guess
or fallback is used. A failed count or count above6,000 prevents synthesis.
Sol is pinned to `gpt-6.1-sol`, default Standard billing, low reasoning effort,
2,048 maximum output tokens including reasoning, `store:false`, no tools.
The actual returned model/status/tier/token usage is checked, including equality
between counted and used input. Sources have identical downstream ten-source and
12,000-character budgets; native provider token/character controls differ.

Pilot: **8 searches +8 counting requests +8 Sol responses**; full20:
**80 searches +80 counting requests +80 Sol responses**, with no automatic retries.
The journal retains maximum cache-write inference pricing: $0.03548 per Sol call,
$0.02 per count request and $0.003125 extras per search. Pilot reserves
**$0.49284**, full20 retrieval+synthesis **$4.92840**. The unused $10 headroom can
cover one separately isolated reviewer Sol call and count per cell, bringing the
planned total to **$9.36680**; allowing one extra raw attempt per cell yields
**$9.85680**. Reviewer execution and raw retries are not automatically enabled.
These figures include outer Sol inference and maximum cache-write rates.

**Counting-endpoint pricing is not stated in the official counting docs.**
The $0.02/count is a retained allowance, not a verified tariff or a claim that
counting is free. Actual provider invoices, counting fees and tax/FX remain to be
reconciled by the execution owner against the $10 approved total. The admission
journal hard-stops forecast exposure above $10 and pilot exposure above $1;
it is not an account-level provider billing cap. If observed rates/extras exceed
these allowances, stop before remaining18 and retain all prior spend. No
subscription, automatic top-up or repeat campaign is authorized.

Outputs are private: `live_access.json`, `live_scope.json`, hash-chained
`live_journal.jsonl`, immutable `live_raw/` and `live_receipts/`. Each case/mode
receipt contains raw and normalized dated citations, counted tokens, raw Sol
usage/answer, latency for every HTTP request, reservation components and verified
attempt identities. Full reservations remain held until billing reconciliation.
Citation extraction is structural, not semantic grading.

## Case and reviewer provenance

`real_public/inputs.parent-message.json` contains all20 actual supplied company
questions, explicitly hypothetical site scenarios, as_of2026-09-30. Its exact
SHA256 is `cac8d7a31aea2ad1c2e5a47ea37e434abcaae4b4910ba4afc8f81dd11c404ff3`,
matching the parent's original public file. No synthetic cases enter paid testing.

Library ZIP `libfile_01ae219642688191a6063e9d3fba9c0b`, version0,26,019bytes,
SHA256 `294f778b32670ee6ae07412a53c1627923f1004afeea501f32d3202e8b9cf736`
could not be transferred after two supported bounded attempts: exactly
`library file transfer failed: download failed`. Its ZIP hash remains unverified.
The public file came through parent-message input, not Library materialization.

The actual reviewer oracle/spec remain **parent-side** and are never read by the
live runner or disclosed to retrieval. `live_rubric.md` pins the recovery protocol,
explicitly distinct from that original specification. The parent must complete
isolated source entailment, citation, primary-source, contradiction and unknown
grading against its real frozen oracle before naming a winner. The synthetic
reviewer below validates lifecycle behavior only. TaskCore/Pro remain excluded
from this experiment and would require a separate deeper experiment label.

## Hermetic validation

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest experiments.provider_eval_recovery.test_harness experiments.provider_eval_recovery.test_live_http experiments.provider_eval_recovery.test_public_inputs experiments.provider_eval_recovery.test_live_runner -q
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.provider_eval_recovery.harness --phase pilot --output /workspace/provider-eval-offline-example
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.provider_eval_recovery.harness --phase remaining --output /workspace/provider-eval-offline-example
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.provider_eval_recovery.reviewer --output /workspace/provider-eval-offline-example
```

52 tests include the full mocked live pilot+remaining80cells, no redispatch on
resume, uncertain spend holds, exact frozen inputs, counted token refusal, forged
pilot receipt refusal, oracle isolation and budget/account/model guards. Independent
Sol review and repository sentinel results are recorded in `VALIDATION.md`.

Official sources checked2026-09-30:
[Parallel Search](https://docs.parallel.ai/api-reference/search/search),
[Parallel pricing](https://docs.parallel.ai/getting-started/pricing),
[Perplexity Search](https://docs.perplexity.ai/api-reference/search-post),
[Perplexity pricing](https://docs.perplexity.ai/docs/getting-started/pricing),
[Sol pricing](https://developers.openai.com/api/docs/models/gpt-6.1-sol),
[OpenAI input count](https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens/methods/count),
[OpenAI counting guide](https://developers.openai.com/api/docs/guides/token-counting).
No obsolete Sonar async route is used.
