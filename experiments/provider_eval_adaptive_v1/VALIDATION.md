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
