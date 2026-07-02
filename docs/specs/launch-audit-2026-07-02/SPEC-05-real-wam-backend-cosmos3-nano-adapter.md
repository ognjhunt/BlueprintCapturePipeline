# SPEC-05: Real WAM backend for beta — Cosmos3-Nano adapter + fixture truth labeling

- Status: Proposed
- Priority: **P0 — launch blocker** (for any model-backed Task Evaluation Run claim)
- Area: `src/blueprint_pipeline/wam_eval_substrate.py`, `wam_backend_strategy.py`, `wam_fixture_evaluator.py`, new `cosmos3_wam_command_adapter.py`, `native_runtime_backend.py`
- Papers: SC3-Eval (arXiv 2606.18610) — Cosmos3-Nano backbone; OSCAR (arXiv 2606.04463) — Cosmos-Predict2.5-2B

## Problem

The strategy doc names Cosmos 3 the "preferred configured candidate **when a real
adapter, checkpoint/provider runtime, explicit run gates, consistency scorer, and
calibration anchors exist**." Of those five preconditions, only the run gates are real
code today:

1. **No Cosmos3-Nano adapter exists.** `cosmos3_wam` is a catalog row with
   `local_available: False`, `command_surface: "blocked_until_provider_adapter_configured"`
   (`wam_eval_substrate.py:145-155`, `wam_backend_strategy.py:179-212`). The only
   executable cosmos adapter runs **Cosmos-Predict2.5-2B** (OSCAR's base model):
   `oscar_cosmos_wam_command_adapter.py:26-27,364`, `synthesis/cosmos_inference.py:51`.
2. **The default evaluation substrate is a deterministic keyword fixture, not a model.**
   `wam_eval_substrate.py:205` sets `default_primary_substrate: "fixture_wam"`, and
   `wam_fixture_evaluator.py:638-716` computes `predicted_success` from keyword matching
   of scenario text against policy `capabilities`, with a hand-tuned uncertainty formula.
   It is honestly flagged internally (`fixture_evaluator_only`), but it emits
   `predicted_success` / `uncertainty_score` / `failure_mode_ids` fields that look
   model-derived to any consumer that drops the flag.
3. **Mislabeling hazard:** an operator wiring `BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND` to
   the existing OSCAR/Predict2.5 adapter would produce artifacts labeled `cosmos3_wam`
   while running a different model family.
4. **Runtime metadata overstates the render source:** `native_runtime_backend.py:854-870`
   hard-codes `model_family: "cosmos_swm_native"` and
   `state_guarantees.render_source: "cosmos_zero_ft_native"` even when the session falls
   back to `splat_only` truthful preview (`:411,744-747,1162`).

## Why this blocks beta

Task Evaluation Runs are the product. If beta buyers receive fixture-derived
success predictions presented as model-backed evaluation — or `cosmos3_wam`-labeled
artifacts produced by a different backbone — that is fake readiness under our own
doctrine, and any correlation claims are indefensible. Conversely, launching with the
fixture *clearly labeled as a smoke substrate* is fine; the gap is the missing real
backend plus the labeling hazards.

## Proposed fix

1. **Build `cosmos3_wam_command_adapter.py`** mirroring the OSCAR adapter pattern
   (operator-supplied checkpoint/repo, trusted output schema `cosmos3_wam_command_adapter.v1`,
   `learned_wam_model_ran` set only when source+checkpoint verify — same honesty
   mechanics as `oscar_wam_command_adapter.py:1743-1752`). Keep it strictly behind the
   existing swappable command boundary; per SC3-Eval, target Cosmos3-Nano weights with the
   80/10/10 forward/cross-view/inverse training mixture noted as the upstream recipe.
2. **Backbone identity verification:** the provider runtime must verify the adapter's
   self-reported `base_model` against the substrate label and hard-fail on mismatch, so
   `cosmos3_wam` artifacts can never be produced by a Predict2.5 command (and vice versa).
3. **Fixture provenance propagation:** make `fixture_evaluator_only` a required
   pass-through field on every artifact schema downstream of WAM eval (package manifests,
   webapp sync payloads). Block MMRV/Spearman/Pearson correlation claims and buyer-facing
   "predicted success" displays for fixture runs at the schema level, not by convention.
4. **Truthful runtime identity:** `runtime_info.model_family` / `render_source` must
   reflect the actual selected path (`splat_only` vs `cosmos_i2w` vs unconfigured), not a
   hard-coded cosmos string.
5. Until (1) is deployed with a real checkpoint, update `wam_backend_strategy.py` so
   `cosmos3_wam` is explicitly `aspirational: true` and the strategy-doc preconditions are
   machine-checked (adapter present, scorer present per SPEC-06, anchors present) before
   the "preferred candidate" flag can be set.

## Acceptance criteria

- [ ] A Cosmos3-Nano command adapter exists with a trusted schema and checkpoint verification; wiring it to a wrong-family checkpoint fails closed.
- [ ] `fixture_evaluator_only` provably survives into every exported artifact and webapp payload (schema-required field + regression test).
- [ ] Fixture runs cannot emit correlation metrics or unlabeled `predicted_success` to buyer surfaces.
- [ ] `runtime_info` reports the real render path per session.
- [ ] The strategy "preferred candidate" state is derived from machine-checked preconditions, not asserted.
