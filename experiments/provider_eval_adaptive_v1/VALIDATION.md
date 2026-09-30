# Adaptive revision verification

This revision is isolated under `experiments/provider_eval_adaptive_v1`. Existing
recovery Python files, diagnostic scope/journal, production sources, runner2486
and snapshot2487 are untouched. The user's explicit private-comparison scope and
adaptive-fix instruction authorize this reversible experiment.

The verification command exercises the actual public 20 cases with synthetic
HTTP responses and synthetic reviewer truth, never paid substitutes:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest experiments.provider_eval_adaptive_v1.test_adaptive -q
.venv/bin/ruff check experiments/provider_eval_adaptive_v1
git diff --check
```

Coverage includes all 80 cells using two searches, 20 isolated four-arm reviews,
exact 520-dispatch worst-case reservation arithmetic, shared cumulative cap,
free preflight, accepted-step replay without redispatch, uncertain holds with no
retry, third-search rejection, retained controller authorization for a second
search, exact trusted-date prompts, no oracle in controller/search requests,
input-count stops without recount/inference, warning stops before quality
interpretation, secure-access boundary preservation, full-result followup
evidence inclusion and malformed controller JSON as a per-case unknown.

Independent `gpt-6.1-sol` review reproduced and verified fixes for three findings:
followup sources crowded out by ten initial results; malformed coverage JSON
halting the matrix; and omitted legacy secure-access receipt checks. After the
fixes, **11 focused non-full tests passed independently**, with no remaining
must-fix finding. The final full suite passed **12 tests in 119.140 seconds**;
Ruff and diff checks passed. Both CLI help entry points loaded successfully.

Implementation executor provider calls: **0**. Implementation external spend:
**$0**. Live artifacts/billing are owned by the fresh execution task and were
not inspected here. Parent-reported checkpoint: 112 completed dispatches,
37 diagnostic answers, $2.285510 reserved, final billing unreconciled.

Actual grading still requires the parent's frozen local oracle/spec mapping and
adjudication. The model reviewer cannot establish ground truth or certify full
pages beyond retained excerpts. The reservation plan includes an unpublished
count-fee allowance and search extras; actual invoices/taxes remain unresolved.

## Pilot phase and unused-preflight-scope repair

The runner now defaults to cases 1–2 (`--phase pilot`); cases 3–20 require
`--phase remaining` and offline validation of all eight retained pilot receipts.
Protocol warnings stop the whole phase after retaining the affected receipt.
Security/uncertain failures continue to stop immediately with their original hold.
The full $7.572000 adaptive admission envelope is unchanged.

The owner's unused preflight scope may explicitly adopt this reviewed code when
its exact old digest and pinned `63ad365` code fingerprint match, all non-code
fields are unchanged, and there are zero adaptive reservations/artifacts. The
original scope remains byte-for-byte intact; a separate immutable adoption
receipt binds the effective scope. No diagnostic journal event is rewritten,
released, or duplicated. Parent-reported baseline is 225 events, 112 completed
attempts, zero uncertain attempts, $2.285510 reserved and 37 diagnostic answers.

**17 focused non-full tests pass**, independently replayed by GPT6.1Sol with no
must-fix findings. They cover default pilot selection, whole-phase warning and
uncertainty stops, zero-call scope adoption preserving original scope/journal
bytes, wrong owner/digest/execute/non-code drift, prior released adaptive attempts,
adaptive artifacts, and missing/tampered pilot rejection before remaining calls.
The complete pilot→remaining→review regression passed: **18 tests in 117.018
seconds**, Ruff and diff checks clean. CLI help confirms both phase choices and
the explicit unused-scope adoption flag. This patch made no provider calls.

## Accepted Chef pagination continuation

Commands and repository files were verified available after the implementation
executor disconnect. The parent supplied the effective scope, accepted attempt,
envelope digest and ten public URLs as task text; no Library transfer is claimed.
An offline audit accepted all ten URLs in all four modes with exactly two Chef
numeric homepage pagination exceptions. Three HTTP URLs were preserved, and no
lookalike domain was promoted to verified vendor authority.

The accepted-search continuation binds the original effective c6d09 plan/code,
original request, exact completed attempt/envelope, canonical preserved failed
receipt, patched code and normalized evidence. Creation requires the sole owner,
expected scope digest and exactly one adaptive attempt. It does not change any
journal state or reservation, and it does not repeat the search. The original
failed receipt remains immutable; an explicitly scoped sidecar holds the resumed
receipt and is used by pilot validation and the separate reviewer.

**27 hermetic tests passed in 126.673 seconds**, with Ruff and diff checks clean.
Continuation regressions include zero-call/idempotent authorization, bytewise
preservation of scope/adoption/raw/failure artifacts, journal-prefix retention,
first new request as token counting, 47 mocked pilot requests instead of 48,
resume without redispatch, sidecar reviewer integration, wrong owner/scope/hash,
extra released attempts, real provider warnings and authorization/raw tampering.
Source guards refuse userinfo, fragments, unknown queries, credential/redirect
parameters, duplicate keys and non-decimal values; local parser failures are
reported separately from provider input warnings and stop the entire phase.

Independent GPT6.1Sol review passed all **26 focused non-full tests**, verified
the c6d09 fingerprint and exact ten-URL audit, and found no remaining must-fix
issue. The complete 27-test result above passed in this implementation executor.

The parent's runtime checkpoint is 227 events, 113 completed attempts, zero
uncertain attempts and $2.289635 reserved; the prior 225-event prefix and 37
diagnostic answers are preserved. Those runtime files were not materialized in
this executor. Implementation provider calls and spend remain zero.
