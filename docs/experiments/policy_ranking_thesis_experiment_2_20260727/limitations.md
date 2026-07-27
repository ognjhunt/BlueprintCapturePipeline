# Experiment 2 limitations

- The primary causal diagnostic compares recorded joint-action magnitude with 2D generated-pixel motion. It is a falsification test, not a positive test of 3D dynamics or task success.
- Alternative action sequences were not regenerated through the WAM. Zero, shuffle, reversal, shift, and within-session swaps are signal-analysis placebos, not counterfactual videos.
- The color mask is deliberately conservative but heuristic. It may mask some scene pixels or retain some antialiased skeleton pixels. The very large overlay-versus-residual gap survives that caveat but should not be interpreted as a pixel-perfect decomposition.
- The released OSCAR rollout dataset lacks explicit license metadata and remains internal research evidence. No restricted source media are redistributed.
- The seven policies share a DROID-compatible Franka embodiment, but the exact command normalization, eighth-channel gripper convention, and full OSCAR generation-time synchronization contract are not attributable from the released snapshot.
- The 49 held-out sessions are the complete remaining released clusters, but they are not evidence that the 441-rollout release is an unbiased sample of the paper's 455-rollout pool or of all DROID tasks.
- GPT-5 API cost is recomputed from provider token usage and the frozen published rate, not reconciled to an invoice line item. The API exposes no persistent paid compute resource to tear down.
- The Experiment-2 held-out risk/coverage curve is unavailable because the prediction matrix closed at 43/686. The labels stayed sealed, and completed rows were not selected or scored post hoc. Historical calibration selective coverage was only 1/113 informative pairs (`0.00885`), below the frozen `0.25` requirement.
- The captured-site component uses immutable historical evidence because the Experiment-2 futility rule stopped a new GPU campaign. Historical OpenPI/MuJoCo execution is not the same WAM/evaluator stack as the benchmark.
- No site-specific physical outcome labels exist for InteriorGS. The experiment cannot claim physical success, physical failure, or site-specific ranking accuracy.
- The only attributable physical timing is action execution at 15 Hz. It excludes setup, resets, labor, safety, scheduling, hardware cost, and retries, so the full economic comparison is unmeasured.
- A broad metadata search created a documented potential free-text outcome-context exposure after provider execution began. Exact outcome fields were not displayed and no scientific choice remained mutable, but the replication is not described as perfectly blinded.
- One accepted temporal-arm canary prediction was printed before the provider matrix closed. No inputs, thresholds, or execution choices changed afterward, but zero selective prediction inspection cannot be claimed.
- The user reduced the combined model-API cap to `$10` during the run. The GPT-5 matrix stopped at 43/686, two in-flight claims received a conservative `$0.18` reserve, and no held-out ranking or risk/coverage result was computed.
- Gemini was not admitted. A valid private credential supported model discovery, but paid billing, paid-service data treatment, exact pricing/rate limits, and a synthetic-frame canary were not all verified. Gemini also could not replace the frozen GPT-5 arm or use an already-consumed held-out partition.

These limitations narrow the verdict to the frozen evidence system, evaluator, released OSCAR rollouts, DROID/RoboArena cohort, and historical captured-site stack. They do not establish a universal impossibility result for future WAMs or policies.
