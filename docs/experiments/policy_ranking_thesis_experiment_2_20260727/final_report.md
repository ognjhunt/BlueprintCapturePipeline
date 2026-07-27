# Blueprint policy-ranking thesis Experiment 2

Experiment 2 produces a defensible negative verdict for the frozen system. It does not claim that every future policy-ranking system must fail.

## frozen_benchmark_calibration

Verdict: `not_supported`

The historical frozen calibration produced temporal pairwise accuracy `0.672566` with clustered 95% CI `[0.548387, 0.777778]`, compared with endpoint-only accuracy `0.659292`. The improvement was `0.013274`, below the registered `0.05` margin. It selected the correct top policy, but useful abstention failed: only 1 of 113 informative pairs survived the selective rule, coverage `0.00885` versus the required `0.25`. Historical action-following pass rate was `0.10204` versus `0.8` required.

Experiment 2 then ran the preregistered label-free causal diagnostic on all 49 held-out session clusters (343 policy rows). Development-only variance implied minimum detectable excess `0.031976`, below the registered useful margin `0.05`, so the held-out diagnostic was sufficiently powered for that margin.

The visible skeleton-overlay region showed mean excess action correlation `0.296703`, 95% CI `[0.271149, 0.323570]`. After masking that annotation, the residual generated scene showed only `0.039976`, 95% CI `[0.012196, 0.067883]`, with one-sided `p=0.762824` against the `0.05` margin. Its validity pass rate was `0.440233`; the session-clustered-bootstrap lower 95% bound was `0.387755`, below the registered `0.8`. All three causal gates failed. The evidence says the annotation follows the action, while the generated scene does not retain the required useful action-conditioned signal once that annotation is removed.

The exact GPT-5 ranking replication was stopped when the user reduced the combined model-API cap to `$10`. It closed at 43/686 accepted requests, with `$1.30073` recorded estimated cost and a conservative `$1.48073` upper bound including two unresolved in-flight claims. Held-out outcome labels remained sealed. No partial ranking, confidence intervals, Kendall statistic, or risk/coverage curve was computed.

## captured_site_transfer

Verdict: `not_supported`

The retained InteriorGS evidence proves that a captured 3DGS room could be ingested and used as a visual background for 12 contract-valid OpenPI/MuJoCo episodes under the prior experiment's stack. It does not prove transfer of the frozen OSCAR WAM/evaluator or policy cohort. Those episodes produced zero nonzero-transport outcomes, zero containment outcomes, no useful total ranking, and only one ordered pair under the overlap rule. The abstention followed the frozen contract, but abstention correctness was not independently calibrated and no site-specific physical labels exist.

The preregistered futility rule prohibited a new GPU transfer campaign after the powered causal gates failed. No new GPU, builder, or physical robot was funded or operated.

Confidence is high for failure of this frozen causal gate: the design targeted 80% power at one-sided alpha `0.05`, and its development-estimated minimum detectable excess (`0.031976`) was finer than the registered useful effect (`0.05`). Confidence is low for universal generalization because the ranking matrix, site-specific physical outcomes, and other WAM/model families remain unmeasured.

## Economics and scope

Experiment-2 incremental paid cost is at most `$1.48073`; Gemini, GPU/build, storage, and physical-robot spend were `$0`. Historical build plus two-scene GPU execution cost `$0.267538` and took about `2310.31` serial compute seconds, but it used a different stack and is not new Experiment-2 transfer evidence.

The physical counterfactual has only a `6620.13`-second held-out action-execution lower bound. It excludes setup, resets, labor, safety, scheduling, retries, and monetary cost. Therefore “substantially faster and cheaper” is not proven, even though the cloud work was inexpensive.

The user selected GPT-5 mini for the next valid experiment. Live read-only discovery confirmed the exact `gpt-5-mini-2025-08-07` snapshot. At the observed token rate, a token-identical 686-call projection is about `$4.15`, but the model was not added to this frozen holdout: the governing protocol explicitly requires a disjoint split or new benchmark snapshot for new evaluator arms. A valid Gemini credential supported model discovery, but billing/data-treatment admission was incomplete. Any mini, nano, or Gemini comparison needs a newly generated disjoint rollout pool.

The next WAM candidate is the general `nvidia/Cosmos3-Nano` model in forward-dynamics mode, with candidate action trajectories supplied through replaceable embodiment adapters and GPT-5 mini kept independent as the evaluator. The DROID policy checkpoint is not the neutral WAM because it generates its own DROID actions. A single H100 80GB is the preferred bounded canary hardware; no allocation or new spend is authorized by this decision. DROID is only the first benchmarked calibration anchor: warehouse, mobile-manipulator, dual-arm, forklift/AMR, and humanoid use remain prospective transfer lanes that must abstain until their own causal, predicate, and outcome evidence exists.

Procedurally, the immutable evidence system preserved 135 hash-chained events and an evidence manifest. Two deviations are disclosed: a broad search displayed policy names and free-text action feedback after execution began, and one accepted canary prediction was printed before matrix closure. Exact outcome fields were not displayed, no prediction/outcome join occurred, and no thresholds or inputs changed; nevertheless pristine blinding is not claimed.

## Overall verdict

`thesis_not_supported`

The frozen rule assigns a negative verdict when a sufficiently powered held-out result fails causal action signal or captured transfer. Here the registered causal test was powered and all action-following gates failed; captured-site transfer also failed to demonstrate the frozen stack, a useful ordering, or calibrated abstention. The incomplete rank-fidelity and economic measurements remain explicitly unmeasured and are not promoted into stronger claims.
