# Experiment 2 limitations

- The primary causal diagnostic compares recorded joint-action magnitude with 2D generated-pixel motion. It is a falsification test, not a positive test of 3D dynamics or task success.
- Alternative action sequences were not regenerated through the WAM. Zero, shuffle, reversal, shift, and within-session swaps are signal-analysis placebos, not counterfactual videos.
- The color mask is deliberately conservative but heuristic. It may mask some scene pixels or retain some antialiased skeleton pixels. The very large overlay-versus-residual gap survives that caveat but should not be interpreted as a pixel-perfect decomposition.
- The released OSCAR rollout dataset lacks explicit license metadata and remains internal research evidence. No restricted source media are redistributed.
- The seven policies share a DROID-compatible Franka embodiment, but the exact command normalization, eighth-channel gripper convention, and full OSCAR generation-time synchronization contract are not attributable from the released snapshot.
- The 49 held-out sessions are the complete remaining released clusters, but they are not evidence that the 441-rollout release is an unbiased sample of the paper's 455-rollout pool or of all DROID tasks.
- GPT-5 API cost is recomputed from provider token usage and the frozen published rate, not reconciled to an invoice line item. The API exposes no persistent paid compute resource to tear down.
- The risk/coverage curve is complete and descriptive. The protocol required uncertainty/error association but did not freeze a unique association statistic; the post-freeze confidence-gap diagnostic cannot be used to rescue a failed registered gate.
- The captured-site component uses immutable historical evidence because the Experiment-2 futility rule stopped a new GPU campaign. Historical OpenPI/MuJoCo execution is not the same WAM/evaluator stack as the benchmark.
- No site-specific physical outcome labels exist for InteriorGS. The experiment cannot claim physical success, physical failure, or site-specific ranking accuracy.
- The only attributable physical timing is action execution at 15 Hz. It excludes setup, resets, labor, safety, scheduling, hardware cost, and retries, so the full economic comparison is unmeasured.
- A broad metadata search created a documented potential free-text outcome-context exposure after provider execution began. Exact outcome fields were not displayed and no scientific choice remained mutable, but the replication is not described as perfectly blinded.
- Gemini was not admitted: the previously exposed credential was treated as compromised and paid billing/data-treatment conditions were not reverified. This does not block the original-arm replication but leaves the cheaper-comparator question incomplete.

These limitations narrow the verdict to the frozen evidence system, evaluator, released OSCAR rollouts, DROID/RoboArena cohort, and historical captured-site stack. They do not establish a universal impossibility result for future WAMs or policies.
