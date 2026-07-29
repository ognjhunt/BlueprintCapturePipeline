# Decision/Evidence Router implementation ledger

Status values are `verified`, `implemented`, `incomplete`, `contradicted`, and
`intentionally_deferred`. This is a living ledger for the controlled migration;
it is not completion evidence by itself.

## Baseline

| Item | Status | Evidence |
| --- | --- | --- |
| Starting source | verified | Dedicated linked worktree and branch `codex/decision-evidence-router-20260729` created from `origin/main` at `ac8148a1429ca3f704bfdec2c26cc6f2d1cdb49c`. |
| Upstream synchronization | verified | Worktree fast-forwarded without conflict to `b98a2a11` after `origin/main` advanced; no user-owned primary-checkout state was touched. |
| User-owned dirty work | verified | Primary checkout was 17 commits behind with a modified `docs/CHANGELOG.md`; it was not stashed, reset, copied, or edited. |
| Nested instructions | verified | No nested `AGENTS.md` files exist below the repository root. |
| Required doctrine read | verified | Root `AGENTS.md`, `PLATFORM_CONTEXT.md`, `WORLD_MODEL_STRATEGY_CONTEXT.md`, `VISION.md`, `docs/DOCTRINE_PRECEDENCE.md`, `README.md`, `pyproject.toml`, `docs/architecture/ai-onboarding-map.md`, and `AUTONOMOUS_ORG.md` read at the starting SHA. |
| Focused baseline | verified | Initial host lane reported 72 passed and one environment-only import failure (`defusedxml` absent). `python -m pip install -e '.[dev]'` installed the declared environment; affected Python 3.12 lanes now pass. |

## Architectural inventory

| Item | Status | Current evidence and migration decision |
| --- | --- | --- |
| Canonical leaf execution contract | verified | `src/blueprint_pipeline/evaluation_run_contract.py` defines `evaluation_run.v1` with the six required replaceable components and is consumed by `evaluation_run_execution.py`, robot-eval, and G1 kitchen adapters. It remains the stable leaf boundary. |
| Duplicate `EvaluationRunSpec` name | implemented | Static pack type renamed `LegacyEvaluationPackSpec`; the old alias remains, and `legacy_evaluation_pack_to_leaf_spec` performs tested explicit translation to canonical `evaluation_run.v1`. |
| Leaf execution | verified | `evaluation_run_execution.py` resolves exactly one preselected execution adapter and binds output to the leaf spec digest. The router must sit above this module and may compile zero, one, or many leaves. |
| Existing reusable proof machinery | verified | `robot_eval_calibration.py`, `buyer_claim_ceiling.py`, `proof_contracts.py`, `policy_ranking_thesis.py`, Task/Scenario/Eval cards, provider envelopes, and WAM/classical-sim disagreement artifacts are present. They remain lower-level evidence inputs and compatibility surfaces. |
| Current product doctrine | implemented | Canonical living docs, README, scenario suite, CLI help, and default robot-eval/Arena orchestration now expose Task Evaluation Run as the product. Legacy builders are deprecated and require explicit evidence-export opt-in. |
| Claim-level router | implemented | Versioned request/testbed/method/qualification/plan/result/decision/outcome contracts, deterministic router, conditional escalation, stable adapters, aggregation, and learning loop are implemented. |
| Current policy-ranking verdict | verified | The frozen repo result remains `thesis_not_supported`; router work must preserve that exact three-verdict vocabulary and cannot qualify its own evaluator. |

## Finite terminal gates

| Gate | Status | Required proof |
| --- | --- | --- |
| One-product doctrine and default surfaces | implemented | Canonical living docs, CLI help, manifests, and scenario suites expose Task Evaluation Run as the sole product; legacy machinery is internal/deprecated and absent by default. |
| Legacy compatibility | implemented | Tested translators cover EvaluationRunSpec pack definitions, legacy data-export requests, Policy Improvement inputs, and WAM/classical-sim artifacts without granting qualification. |
| Versioned router contracts | implemented | Maintained testbed, request, method, qualification, plan, normalized result, decision, and physical-outcome contracts have deterministic serialization plus checked JSON Schemas. |
| Deterministic routing | implemented | Tests cover exact qualification/scope, rights/privacy/provider restrictions, inputs, availability, reproducibility, budget/latency, false-safe/coverage, dominance, correlation, conditional escalation, and abstention. |
| Stable method adapters | implemented | All seven required families are representable; analytic/capture and accepted physical evidence are non-leaf, physical execution is forbidden, and availability is not qualification. |
| Leaf compilation/execution | implemented | Router emits zero/one/many canonical leaves, preserves non-leaf methods, and normalized results bind request/plan/profile/qualification/leaf/testbed digests. |
| Vertical slice | implemented | Checked rigid-object fixture proves multi-method planning, two leaves plus analytic/capture steps, unqualified world-model rejection, ranking/deployment abstention, disagreement, partial decision, ceilings, and later immutable learning. |
| Learning loop | implemented | Exact prediction/outcome join is append-only; duplicate and calibration/held-out leakage fail closed; new testbed/qualification versions keep transfer disabled. |
| WebApp handoff | verified | Versioned schemas, state machine, translations, redaction, compatibility policy, checksum manifest, and decision/abstention/partial fixtures pass `python -m blueprint_pipeline.decision_evidence_handoff`. |
| Focused verification | verified | 160 router/contract/legacy tests and 33 WAM/default-orchestration tests pass; Ruff, source governance, WebApp handoff verification, and `git diff --check` pass. |
| Fast lane | verified | The single local bare lane ran all 5,597 selected tests: 5,596 passed and the quality-gap ledger alone detected changed source digests. The repository rebind tool refreshed 13 digest bindings and the exact failing governance test then passed. On router implementation candidate `3a77f3ab`, hosted run `30478926371` then selected 5,599 tests and passed 5,591 with 8 skipped and 1,664 deselected. The local bare lane was not repeated. |
| Verification/publication | verified | Router implementation run `30478926884` executed `scripts/pytest_full.sh`: 7,263 passed in 954.94 seconds and the fail-closed CPU evidence builder reported `status=passed`. All 16 hosted checks passed; protected-main PR #243 merged as `98484a591ec5c29e957248b65390b7c62a662b1a`. After fetch, the primary checkout proved `HEAD == main == origin/main`, matching tree `bf034335032af9f7400308b6c7a4ecb3c493fe96`, and `0 0` divergence while preserving the user-owned `docs/CHANGELOG.md` edit. |
| Live/provider/physical execution | intentionally_deferred | Explicitly unauthorized and unnecessary for the hermetic control-plane slice. This does not block sim-only functionality and cannot upgrade physical/deployment/safety claims. |

## Closeout

The control-plane migration was published through protected-main PR #243. The
exact tested candidate was `3a77f3ab14504d112603b816c20560ab12d5b6b2`; the
squash merge is `98484a591ec5c29e957248b65390b7c62a662b1a`. No paid
compute, provider execution, production deployment, live robot operation, or
physical-success claim was performed. Those remain evidence-dependent claim
upgrade lanes rather than repo-local implementation gaps.
