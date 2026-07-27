# Experiment 2 unresolved contradictions

These contradictions are retained as evidence. None is resolved by redefining generated media, simulator output, or prior execution as physical success.

1. **Visible action annotation versus scene response.** On 49 held-out session clusters, the colored skeleton-overlay region has mean excess temporal correlation of `0.296703` over the strongest placebo, while the overlay-masked generated scene has only `0.039976` and its 95% clustered interval starts at `0.012196`, below the frozen `0.05` useful margin. This is consistent with the annotation tracking the recorded trajectory while the rest of the generated scene does not provide the required useful signal.

2. **OSCAR action-conditioning claim versus attributable adapter semantics.** The public OSCAR demo documents skeleton construction from realized joint angles, end-effector pose, and gripper openness. The released evidence does not document a complete counterfactual policy-command adapter for a new captured site. The public rollout can test temporal consistency, but it cannot establish that Blueprint can vary a candidate policy and causally generate the corresponding site outcome.

3. **Released pool versus paper pool.** The OSCAR paper describes 455 policy rollouts (65 sessions by seven policies), while the released complete pool contains 441 (63 by seven). The selection mechanism for the missing 14 rollouts is undocumented, so representativeness of the frozen released pool cannot be assumed.

4. **Historical scene ingestion versus frozen-stack transfer.** The prior experiment executed 24 OpenPI/MuJoCo episodes across Warehouse and InteriorGS, but the benchmark lane used released OSCAR rollouts plus a GPT visual evaluator. The historical execution therefore proves scene/runtime ingestion under a different stack, not portability of the benchmark mechanism.

5. **Contract-compliant abstention versus useful abstention.** The historical captured-site lane correctly followed its frozen overlap rule and abstained from a total order, but had no independent site outcome labels or transfer-specific power analysis. Contract compliance does not prove that abstention predicts error or is commercially useful.

6. **Low cloud execution cost versus the economic thesis.** API and historical GPU costs are attributable, but the physical counterfactual has only an action-time lower bound. Robot/lab cost, setup, reset, operator, safety, scheduling, and retry burdens are unmeasured. No defensible monetary or full wall-clock “substantially faster and cheaper” comparison exists.

7. **Procedural lock versus pristine blinding.** After the immutable provider matrix began, a broad provenance search displayed policy names and some free-text action feedback from the dataset tree before completion. It did not display the exact `binary_success`, `partial_success`, or `preference` fields and could not alter frozen requests or deterministic metrics, but pristine metadata sealing cannot be claimed.

8. **Cheaper models versus frozen-arm identity.** GPT-5 mini and nano are materially cheaper than GPT-5, and a valid Gemini credential was found, but swapping models after the exact GPT-5 matrix began would change the scientific arm. With no unused disjoint OSCAR holdout left, cheaper models are successor-experiment candidates rather than repairs to this frozen replication.

9. **Low partial API spend versus a complete ranking result.** The run stayed below the user-authorized `$10` cap, but only 43/686 requests completed. Low spend proves bounded execution cost, not held-out rank fidelity or a complete faster-than-physical comparison.

Evidence class: limitations and contradiction accounting. This document does not itself prove ranking fidelity, captured-site transfer, economics, or physical performance.
