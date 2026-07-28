# RoboArena full-stack calibration terminal report

Overall verdict: `inconclusive`.

The campaign produced a complete known-answer evaluator reproduction and a useful post-unseal judge diagnostic, but it did not produce the disjoint closed-loop WAM experiment required by the thesis. Evidence that is missing, confounded, or diagnostic-only is not combined into a stronger claim.

## Component verdicts

### cosmos_wam_qualification — inconclusive

No native Cosmos WAM arm passed the registered causal and reliability program. The immutable historical Cosmos3 follow-up remains underpowered and confounded: ten scientific outputs were valid; all eight active rows differed from the same-seed no-motion output, but only one of eight rejected the strongest temporal placebo and zero of four active conditions passed both-seed robustness. It used one independent DROID session, short clips, malformed zero-rot6d control, and a synthetic policy-swapped trace.

The later Cosmos3-Nano Reasoner work evaluated videos; it was not a WAM arm. V4 failed before model load because of an invalid architecture override. V5 loaded the model and returned seven HTTP 200 responses, but zero rows satisfied the structured-output contract. V6 carried the corrected schema transport and failed before model load with CUDA error 803 from a host/container runtime incompatibility. V4–V6 cost an estimated `$0.733574` in total and produced no scientific ranking or WAM qualification credit. A reusable pre-download CUDA admission probe now fails future Vast bundles closed on this class of incompatibility.

### frozen_benchmark_calibration — not supported for the frozen GPT-5 mini Phase-A stack

Phase A completed all `441` requests covering `63` public sessions and `7` policies with `gpt-5-mini-2025-08-07`. Policy identity, published outcomes, and physical pixels were excluded from evaluator payloads; predictions were frozen before labels were unsealed.

The result failed the registered gate set:

- Spearman rho `0.357143` versus required `0.70`.
- Kendall tau-b `0.238095` versus required `0.50`.
- Policy pairwise ordering accuracy `0.619048` versus required `0.70`; clustered interval `[0.428571, 0.857143]`.
- True top policy was ranked first, so the top-two gate passed.
- Selective coverage `0.050182` versus required `0.50`.
- Selective pairwise accuracy `0.600000` versus required `0.75`.
- Exact small-n p-values were `0.444444`, `0.561905`, and `0.280952` for Spearman, Kendall, and pairwise ordering respectively.
- The evaluator emitted `54` episode abstentions. The uncertainty-aware risk rule passed, but selective use was too sparse and inaccurate to pass the combined gate.

The claim ceiling is a non-independent known-answer reproduction because these public sessions were already used during method development.

Gemini 3.6 Flash was a promising post-unseal diagnostic challenger. On the complete direct-pair matrix it achieved Spearman `0.75`, Kendall `0.619048`, policy pairwise accuracy `0.809524`, and direct within-session pair accuracy `0.721925` with a session-clustered 95% interval `[0.676303, 0.762410]`. Exact p-values were `0.066270`, `0.069048`, and `0.034524`. It never abstained, and the answer key was already exposed, so it cannot establish calibrated selective use or admit Phase B.

### captured_site_transfer — inconclusive

Phase C was not admitted. No new captured-site stack was run, and no independently published or independently attributable physical outcomes existed for the same site, task, embodiment, and policies. Therefore no captured-site accuracy claim is made.

### economics_and_speed — inconclusive

Known conservative provider cost, excluding storage/transfer for which no invoice line was available, was `$6.909436375`: `$6.175862375` for API evaluator work and `$0.733574` for Reasoner GPU attempts. The Phase-A matrix cost `$2.53707425` and ran for `2001.05` seconds; its usefulness gates failed. The Gemini full matrix cost `$3.14775225`; submission-to-local-collection wall time was approximately `1514` seconds, but it did not establish abstention.

The `441` published comparison videos total `12,873.6` seconds (`3.576` hours) of footage. That is only a sequential playback lower bound. It excludes robot setup, resets, failures, operator/safety labor, robot cost, site preparation, and parallelism. No defensible physical monetary baseline or end-to-end Blueprint WAM time exists, so no speed ratio, cost ratio, or break-even policy count is claimed.

## Phase B design actually achieved

Only an availability audit was completed. No newer disjoint labeled RoboArena snapshot was available. Seven public OpenPI checkpoint locations were verified, but their outputs are `10x8` or `15x8` joint-command chunks rather than the frozen Cosmos `16x10` Cartesian/rot6d contract. The required observation-bound joint integration, forward kinematics, DROID/OpenCV pose differencing, rot6d encoding, and horizon adapter was not built or frozen. Because Phase A also failed its gates, confirmatory Phase B was not admitted. The highest available fallback was exposed-session reuse for descriptive sensitivity, and it was not executed because it could not change the claim ceiling.

## WAM-arm findings and immutable baselines

Historical Experiment 2 remains `thesis_not_supported` for its frozen OSCAR-derived stack. On 49 held-out session clusters, the visible skeleton-overlay signal had mean excess action correlation `0.296703`, while the skeleton-masked scene had `0.039976`, 95% interval `[0.012196, 0.067883]`, and a clustered lower validity bound `0.387755` versus required `0.8`. The held-out judge matrix stopped at `43/686` and was not scored. This supports intended-trajectory visibility, not useful scene dynamics.

The current campaign did not execute new skeleton-only, OSCAR purpose-built WAM, visible-skeleton, scene-masked, native Cosmos WAM, or Cosmos-plus-skeleton hybrid full episodes. No hybrid finding is claimed. Reasoner evaluator results are kept separate from native Cosmos WAM evidence.

## Identity, protocol, and evidence

Protocol v1 digest `eab9e7868bcc7cbd774c940c781e8c3a8faac3270cbc942f1248966ba037f683` remains immutable superseded history. Protocol v2 digest is `6b41ea618ec290f1c080573e093d26cae03d14a0ecb06b3bb2a4bd016e469066`; it governs integer prefixes and uses `16` steps (`16/15` seconds) derived from the pinned upstream contract because a live label-free endpoint pilot was unavailable. The source/model/data freeze is `source_revision_license_freeze_v2.json`. Key identities include OSCAR code `4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb`, RoboArena data `7931db81f3f6a48a3245427f7213a4c461f92ccc`, Cosmos3-Nano `411f42a8fdfb8c5b2583cb8786e0938f49796eaa`, vLLM-Omni `9c1b7504b178afcf541867c1a2d30db48c69cda8`, and Phase-A evaluator `gpt-5-mini-2025-08-07`.

Raw structured responses, frozen predictions, label-unseal ledger, crop audit, collapse report, Gemini analysis, Reasoner ledgers, and provider receipts remain under `/Users/nijelhunt_1/workspace/policy-ranking-roboarena-full-stack-calibration-evidence-20260728`. Review media locations and hashes are indexed by `review_gallery_manifest_v1.json`.

## Provider state and publication

All three task Vast instances were destroyed, watchdogs and tunnels were stopped, task staging secret files were deleted, and authenticated Vast inventory was empty after V6. Cancellation was requested for the GPT-5.4 mini batch; it had zero completed requests, zero recorded token usage, no output or error file, and its input file was deleted. The terminal provider receipt records whether the provider advanced the administrative batch object from `cancelling` to `cancelled`. Provider-zero proves current resource state, not invoice settlement or scientific validity. Exact merged-main publication and final hosted-check bindings are recorded after merge in the external evidence store so this report does not pretend a commit can contain its own SHA.

## Cheapest valid next experiment

Do not rerun judges on the exposed snapshot and do not run another short Reasoner pilot. Freeze the current evaluator and reliability contracts, then run them once on a genuinely new disjoint labeled RoboArena/DROID snapshot with runnable frozen policies and the prospectively powered independent session count. That single experiment can test generalization, closed-loop WAM causality, ranking, and abstention without paying again for diagnostics that cannot change the verdict.

In plain English: Blueprint successfully processed and judged a complete public robot-policy benchmark, but its frozen GPT-5 mini judge did not rank the policies accurately enough. Gemini ranked them much better, but only after the answers had already been exposed and without ever declining an uncertain comparison. We never obtained the new labeled robot data and compatible policy loop needed to test the complete product, and we did not validate captured-site accuracy or a real physical cost comparison. The experiment improved the reusable harness and identified the next valid test; it did not prove or conclusively disprove the overall Blueprint thesis.
