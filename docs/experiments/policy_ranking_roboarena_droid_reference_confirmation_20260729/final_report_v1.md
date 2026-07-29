# Cosmos3 DROID reference confirmation terminal report

Overall verdict: `inconclusive`.

The corrected native-Cosmos deployment returned a valid recorded-action clip and a matched valid no-motion clip. It did not show the action-specific separation needed to admit the larger Phase-B experiment. Blueprint correctly abstained. This is useful diagnosis of the frozen stack, not powered evidence supporting or falsifying the whole replaceable Blueprint thesis.

## Component verdicts

### cosmos_wam_qualification — inconclusive

Allocation 4 ran `nvidia/Cosmos3-Nano` on NVIDIA's published DROID reference observation with the pinned native `16x10` forward-dynamics action contract. Both responses were structurally valid `640x528`, 17-frame, 15 Hz videos. The recorded rollout had whole-video tier-1 motion mean `0.298105`; valid no-motion had `0.377772`. Recorded action/motion timing correlation was `-0.460018`. In the separately decoded wrist view, no-motion residual flow (`1.066360`) exceeded recorded (`0.773836`). The left and right exterior views were nearly static in both conditions.

The session-level gate had one eligible window where three were required and returned `timing_evidence_insufficient`. The development thresholds had not been prospectively frozen for this experiment, so this single unfavorable pair cannot be called a powered falsification. Blueprint abstained, assigned no Cosmos qualification credit, made no evaluator call, and did not admit untouched data.

### frozen_benchmark_calibration — not supported for the frozen GPT-5 mini Phase-A stack

The completed earlier Phase-A known-answer reproduction remains controlling. `gpt-5-mini-2025-08-07` evaluated 441 generated episodes across 63 sessions and seven policies. It achieved Spearman `0.357143`, Kendall tau-b `0.238095`, and policy pairwise accuracy `0.619048` with clustered interval `[0.428571, 0.857143]`. Selective coverage was `0.050182` and selective pairwise accuracy `0.600000`; 54 episodes were abstained. The frozen stack failed the registered gates.

The full policy vectors are retained in `phase_a_analysis_report_v4.json` in the prior external evidence root. The predicted top policy was the true top policy, but the overall ordering and selective-usefulness gates still failed. Gemini 3.6 Flash later achieved Spearman `0.75`, Kendall `0.619048`, policy pairwise accuracy `0.809524`, and direct within-session accuracy `0.721925` with clustered interval `[0.676303, 0.762410]`. It was post-unseal diagnostic evidence and never abstained, so it cannot admit Phase B.

### captured_site_transfer — inconclusive

Phase C was not admitted or run. No site-specific accuracy claim is made because no independently attributable physical outcomes exist for the same captured site, task, embodiment, and candidate policies.

### economics_and_speed — inconclusive

The current DROID reference campaign used an estimated `$0.6470414695` of GPU compute: allocation 1 `$0.387084`, allocation 2 `$0.1217340257`, allocation 3 `$0`, and allocation 4 `$0.1382234438`. Allocation 4 took `250.052` seconds wall time, including `145.042` seconds to load the model; the paired scientific runtime was `153.316` seconds. It used two WAM calls and no evaluator calls. Storage and transfer were not invoiced separately.

The prior calibration campaign recorded `$6.909436375`, giving `$7.5564778445` in combined known provider cost across the two separately reported campaigns. No useful whole-stack policy ranking completed, so cost per completed ranking, physical/digital ratios, speed ratios, break-even policy count, and "substantially cheaper/faster" claims are unavailable.

## Phase A and Phase B claim ceilings

Phase A was a complete but non-independent known-answer reproduction. It tested the judging, aggregation, ranking, calibration, and abstention instrument; it did not test live policy↔WAM execution or generalization.

The current campaign did not execute Phase B. It achieved only the preceding fallback admission screen: one published DROID reference observation, one recorded action chunk, one valid no-motion chunk, and native Cosmos output. No policy endpoint was queried, no policy was re-queried on a predicted observation, and no full terminal episode or multi-policy ranking was produced. Its claim ceiling is deployment and causal-screen diagnosis.

## WAM arms and baselines

Historical OSCAR Experiment 2 remains `thesis_not_supported` for that frozen OSCAR-derived stack. On 49 held-out clusters, visible skeleton excess action correlation was `0.296703`; skeleton-masked scene correlation was `0.039976` with interval `[0.012196, 0.067883]`, and the validity lower bound was `0.387755` versus `0.8` required. That is intended-motion evidence, not useful scene dynamics.

No new skeleton-only, OSCAR purpose-built WAM, visible-skeleton, scene-masked, or Cosmos-plus-skeleton hybrid full episode ran here. Native Cosmos is reported separately above. Cosmos Reasoner was an evaluator attempt in the earlier campaign, not a WAM, and receives no native-Cosmos qualification credit.

## Identity, protocols, evidence, and tests

Current execution identities were Cosmos source `0299468993d8bcd8f6a95b0d8427b1221fccfced`, cosmos-framework `9726697a83315540c6885baefd2fe353d9c74920`, vLLM-Omni `1c6e7313394923000215a3299f4f79ede3873ecc`, `nvidia/Cosmos3-Nano` revision `411f42a8fdfb8c5b2583cb8786e0938f49796eaa`, `nvidia/Cosmos3-DROID` revision `5c11a20accb11497270a5247a7f1e66ad04c956c`, and image digest `sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587`.

Protocol digests v1 through v5 are respectively `2e68f9c7d56a354fd0cb58a706155a672789f1fe726e1ef83df40f529fd4b38e`, `35039583984c045dcfb6e31116b63cd3b90d72181ce7fd64a37f268e1fc0d1dc`, `c200d6c36c10387f5357457500705b8c5db52c1964f9432f4c0d3ab0569b02d3`, `806b76ed1a2cc8b07d9da2746349c908bb7b2a59f88ccba7cba2b76f62fe8f45`, and `6acc514931c378c2d49d4fdaffdafc4e31bb715c21f3841e9606a5d146305984`. Earlier protocols remain immutable; amendments record infrastructure, geometry, and provider-retry changes without rewriting scientific history.

Reusable changes add view-attributable paired active/null analysis and exact object-store cleanup with absence proof. The affected focused lane passed `35/35`. Final repository-wide and hosted publication checks are recorded only on the eventual publication SHA.

Review media are indexed by `review_gallery_manifest_v1.json` under `$HOME/workspace/policy-ranking-roboarena-droid-reference-confirmation-evidence-20260729/review_media_v2`.

## Provider zero

Authenticated RunPod task and global inventories were empty after allocation 4, the watchdog exited, and continuing burn was false. All six exact bundle/output objects across staging v1-v3 were deleted and absence-confirmed. Local signed-URL files were removed. No persistent evaluator resource was created. This proves current provider state, not invoice settlement or scientific validity.

## Cheapest scientifically valid next experiment

Freeze the view-specific active/null, timing, collapse, and session-level thresholds before seeing new outputs. Then run the full control set—recorded, valid no-motion, shuffled, reversed, shifted, and real policy-swapped—on at least the prospectively justified independent DROID session count (the prior target was 17), with multiple windows/seeds. Only if that native Cosmos arm passes should Blueprint spend on a genuinely new labeled multi-policy snapshot and the complete policy→WAM→same-policy loop.

In plain English: the GPU and model plumbing finally worked, but the generated future did not reliably follow the real action better than "do nothing." Blueprint caught that and stopped before paying a judge or pretending it had ranked robot policies. We learned that the safety/evidence harness is doing its job; we did not prove that Cosmos is a trustworthy WAM, that the full loop works, that captured-site rankings are accurate, or that Blueprint is cheaper or faster than physical testing.
