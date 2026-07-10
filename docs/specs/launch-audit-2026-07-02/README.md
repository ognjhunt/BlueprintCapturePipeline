# Launch / Beta Readiness Audit — 2026-07-02

> [!WARNING]
> **SUPERSEDED FOR CURRENT LAUNCH STATUS.** This file is historical evidence, not a current completion or launch decision.
> Use the [current 107-gap ledger](/docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json) and the [July 9 source audit](/docs/PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md). Do not infer current status from “proposed,” “implemented,” or “fixed” wording below.

Full-codebase and recent-run audit of `BlueprintCapturePipeline` (companion audit for
`Blueprint-WebApp` lives in that repo at `docs/specs/launch-audit-2026-07-02/`).

Reference papers the service is based on:

- **OSCAR** — arXiv 2606.04463: action-conditioned world model fine-tuned from
  Cosmos-Predict2.5-2B; skeleton-rendering action conditioning; strict data-curation
  pipeline (frame-count/static-camera/action/visibility filters → SigLIP semantic dedup →
  VLM captioning); MoGe-v2/CtRNet-X camera calibration; PSNR/SSIM/LPIPS/tLPIPS/FVD/FID and
  MMRV/Spearman/Pearson real-world correlation.
- **SC3-Eval** — arXiv 2606.18610: policy evaluation on a Cosmos3-Nano backbone; joint
  forward+inverse dynamics training (80/10/10 mixture with cross-view inpainting);
  test-time consistency (commanded-vs-recovered action error terminates rollouts);
  prediction/execution horizon decoupling (predict 24, execute 16); per-dimension
  action normalization; Pearson r + MMRV.

## What recent runs actually show

- Every `ops/city-launch-runs/` run (Austin 2026-05-06, Durham 2026-05-11, audit-city
  2026-05-05) is `blocked` with 33–38 blockers; no `ready_to_market_*` proof exists
  anywhere, and all artifacts are ~2 months stale.
- The sim-only robot-eval beta path is blocked per `docs/last_24h_launch_audit_2026-06-26.md`
  and `READINESS_MATRIX.md` — no live GPU frame has ever been produced.
- CI on `main` is green today, but the passing legs are hermetic contract suites; the CPU
  visibility/placement gates silently skip (missing `pxr`/`mujoco`), and the launch-gate
  scripts can report green without touching any live path.
- The default WAM evaluation substrate is a deterministic keyword fixture, not a model.

## Specs (this repo)

| Spec | Title | Priority | Status |
|------|-------|----------|--------|
| [SPEC-01](SPEC-01-geometry-truth-no-fabricated-fallbacks.md) | Stop fabricating geometry (fallback + `local_sfm` relabel) | P0 | **Implemented** |
| [SPEC-02](SPEC-02-oscar-grade-clip-curation-filters.md) | OSCAR-grade clip curation filters | P0 | **Implemented** (`clip_curation_stage.py`; orchestrator wiring optional per bundle) |
| [SPEC-03](SPEC-03-semantic-dedup-stage.md) | Semantic dedup stage (embedding clustering + trajectory RMS) | P0 | **Implemented** (`semantic_dedup_stage.py`; production SigLIP/DINOv3 provider injectable) |
| [SPEC-04](SPEC-04-action-normalization-and-validation.md) | Action normalization & validation for SC3-style eval | P0 | **Implemented** (`action_normalization.py`; sc3 protocol + export fail-closed) |
| [SPEC-05](SPEC-05-real-wam-backend-cosmos3-nano-adapter.md) | Real WAM backend: Cosmos3-Nano adapter + fixture truth labeling | P0 | **Implemented** (adapter + backbone verification + fixture claim boundaries; needs real checkpoint to leave `aspirational`) |
| [SPEC-06](SPEC-06-sc3-consistency-scorer.md) | SC3 forward/inverse consistency scorer + uncertainty gating | P1 | Proposed |
| [SPEC-07](SPEC-07-camera-calibration-estimation-and-validation.md) | Camera calibration: estimate, validate, fail closed | P1 | Partially implemented (export paths fail closed; estimator lane still proposed) |
| [SPEC-08](SPEC-08-temporal-alignment-correctness.md) | Temporal alignment: single time base, canonical frame ids | P1 | Proposed |
| [SPEC-09](SPEC-09-immutable-raw-capture-artifacts.md) | Immutable raw capture artifacts (provenance) | P1 | Proposed (append-only truth flags shipped with SPEC-01) |
| [SPEC-10](SPEC-10-enrichment-validation-and-clip-captioning.md) | Validate LLM enrichment + add OSCAR-style clip captioning | P1 | Proposed |
| [SPEC-11](SPEC-11-launch-gate-hardening.md) | Launch-gate hardening (exit codes, skip-to-green, self-attested proofs) | P1 | Proposed |
| [SPEC-12](SPEC-12-cpu-safety-gate-environment.md) | Make CPU safety gates actually run (`pxr`/`mujoco` in canonical env) | P1 | Proposed |
| [SPEC-13](SPEC-13-city-launch-run-refresh.md) | Re-run city-launch evidence; fix Austin backend quota | P1 | Proposed |

Severity: **P0** = must fix before external beta; **P1** = must fix before paid launch /
strongly recommended before beta; **P2** = cleanup.

## Positive findings (verified, no action needed)

- The WAM adapter boundary is genuinely swappable (single command interface, trusted
  output schemas, catalog-driven substrates) — matches WORLD_MODEL_STRATEGY_CONTEXT.md.
- Run gates (`BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER`, `BLUEPRINT_ALLOW_LOCAL_WAM_MODEL`,
  GPU provider gates) are real and fail closed.
- No hard-coded credentials; secrets are file-based and redacted in artifacts.
- `launch_proof_policy` forces production flags in the strict direction;
  `launch_provenance` implements a real dirty-tree paid-launch block.
- The city-launch harness honestly refuses to certify readiness (50 proof fields,
  `contract_only` proofs rejected).
