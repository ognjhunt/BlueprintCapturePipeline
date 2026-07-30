# Blueprint specialized WAM successor terminal report

Overall verdict: `inconclusive`

The specialized successor proved that Blueprint can load the exact public
Policy-DROID endpoint and execute exact public OSCAR and Ctrl-World replay
interfaces. It did not produce a causally qualified closed-loop policy ranking
on genuinely disjoint physical outcomes. The existing frozen tested stack
therefore remains immutably `thesis_not_supported`; this successor is
`inconclusive`, not an upgrade or rewrite of that result.

## Component verdicts

### cosmos_wam_qualification — inconclusive

The previously completed powered native-Cosmos screen remains the result for
its frozen configuration: 0/51 causal windows passed, 0/17 sessions were
reliable, and 46/51 active windows were `static_under_command`. Blueprint's
abstention correctly rejected that arm. This does not falsify every Cosmos
configuration.

The new Policy-DROID canary loaded
`nvidia/Cosmos3-Edge-Policy-DROID@3ea407af3e156c0af3b4bb6edd85842cc9a58777`
under cosmos-framework
`2f603cb114ff8b335e116060444d0b6caee3a85e`. It returned its native `32x8`
absolute-joint-plus-gripper chunk; Blueprint hashed all 32 rows, derived the
first `16x8` WAM prefix, and advanced exactly the first `8x8` rows. This proves
the candidate-policy endpoint contract only. No WAM ran in that canary.

The exact OSCAR public canary loaded
`zywu2115/OSCAR-2B@c9781ffa7dd8556d862d7d9f338a2ea008a58ca6` with source
`4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb`. It generated one coherent
81-frame, `640x480`, 14 Hz H.264 rollout without future physical RGB. That is an
open-loop recorded-action replay. OSCAR's video generation is autoregressive,
but the candidate policy is not re-queried on OSCAR-predicted observations.
Public OSCAR therefore is not closed-loop policy evaluation in the sense
registered here.

The exact Ctrl-World replay canary loaded source
`99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d` and checkpoint
`yjguo/Ctrl-World@8cf814693f411962dc866a2ddb5b785afd17a93a`. It generated five
frames for each of three views from one recorded 7-D Cartesian action
interaction. It did not re-query a policy. The public hard-coded `success`
field was ignored, future physical comparison pixels were removed, and no task
success was assigned. Exact reproduction of the published Ctrl-World
policy-in-loop result remains blocked because the release does not identify
the exact openpi revision or pi-policy checkpoint revisions used for it.

The separately registered Cosmos3 plus OSCAR-skeleton hybrid is technically
blocked. The pinned public Cosmos3 interface exposes numeric action
conditioning but no checkpoint-trained camera-aligned skeleton channel.
Blueprint did not paint a skeleton into RGB and pretend it was a valid adapter,
and no OSCAR-generated world video was fed to Cosmos.

No successor WAM passed the required own-action-versus-placebo, temporal,
seed-robustness, scene-masked, and collapse gates. WAM qualification therefore
remains inconclusive.

### frozen_benchmark_calibration — inconclusive

The prior frozen GPT-5 mini known-answer reproduction remains negative for its
tested stack. Across 63 sessions, seven policies, and 441 generated episodes,
`gpt-5-mini-2025-08-07` achieved Spearman `0.357143`, Kendall tau-b
`0.238095`, and policy pairwise accuracy `0.619048` with session-clustered 95%
interval `[0.428571, 0.857143]`. Selective coverage was `0.050182`, selective
pairwise accuracy was `0.600000`, and expected calibration error was
`0.228798`. The registered gates failed; that frozen stack remains
`thesis_not_supported`.

The successor registered an OSCAR-method-inspired full GPT-5 direct-pair arm,
not an exact reproduction of OSCAR's unpublished private judge code. The
inventory contained 63 sessions times 21 within-session policy pairs, or 1,323
comparisons. `gpt-5-2025-08-07` produced 355 valid unique results from 368
submitted pairs. Seven rows exhausted output tokens and six rows returned
provider `insufficient_quota`; 968 valid comparisons remain missing. All
submitted batches are terminal and no active batch remains. The comparison
graph is only 26.83% complete, so Blueprint did not run Bradley-Terry, publish
a partial policy vector, or award ranking credit.

Gemini 3.6 Flash remains a separate judge. Its already-unsealed 441-pair
diagnostic produced Bradley-Terry strengths:

- `pi0_fast_droid`: `1.321791`
- `paligemma_diffusion_droid`: `1.301127`
- `paligemma_fast_droid`: `1.295887`
- `paligemma_fast_specialist_droid`: `1.211237`
- `paligemma_vq_droid`: `0.820198`
- `pi0_droid`: `0.816895`
- `paligemma_binning_droid`: `0.232864`

Against the independently published physical ordering vector, that diagnostic
had Spearman `0.75`, Kendall tau-b `0.619048`, policy pairwise accuracy
`0.809524`, and direct within-session pair accuracy `0.721925` with clustered
95% interval `[0.676303, 0.762410]`. The predicted top policy was the physical
top policy, `pi0_fast_droid`. Exact small-n p-values were `0.066270`,
`0.069048`, and `0.034524` for Spearman, Kendall, and policy pairwise accuracy.
It made zero abstentions, so selective calibration was not established. The
labels had already been unsealed, making the result diagnostic only.

The prospectively frozen Gemini arm required the remaining 882 comparisons.
Its minimal one-pair Files-to-Batch transport canary returned
`FAILED_PRECONDITION` without creating a generation row or incurring new cost.
The exact provider precondition was not disclosed. Blueprint did not retry the
same unchanged transport, aggregate the partial graph, or mix judge arms.

Because neither new judge completed its own frozen graph, the successor has no
new full policy score vector, Spearman, Kendall, pairwise accuracy, risk curve,
calibration error, MMRV, top-policy rank, or abstention result. Those metrics are
not zero; they are not measured.

### captured_site_transfer — inconclusive

Captured-site execution was not admitted because no successor WAM and judge
combination passed causal, ranking, and abstention gates. No independently
attributable physical outcomes exist for the same captured site, task,
embodiment, and candidate policies. A working policy endpoint, skeleton
renderer, generated episode, or site adapter would prove technical execution
only, so none was promoted to transfer evidence.

### economics_and_speed — inconclusive

The campaign recorded `$8.418512` in evaluator API cost and `$3.986999` in GPU
cost, for `$12.405511` combined. No separate storage or transfer invoice line
was available. This remained inside the `$25` OpenAI, `$50` GPU, `$10`
storage/transfer, and `$100` total ceilings.

The Policy-DROID endpoint loaded in `55.006` seconds and inferred in `19.634`
seconds. OSCAR inference took `168.247` seconds within `527.470` provider-live
seconds and cost `$0.195793` for the successful canary. Ctrl-World's public
replay took `46.280` seconds within `455.655` provider-live seconds; the final
allocation cost `$0.194544`, while the whole Ctrl-World attempt series cost
`$0.446329`.

These are bounded technical execution measurements. No useful complete ranking
finished and no attributable physical monetary or wall-clock baseline was
available. Cost per useful ranking, speed ratio, cost ratio, break-even policy
count, and the claim "substantially faster and cheaper" are therefore
unavailable.

## Phase-B design and claim ceiling

The initially selected 17-session candidate cohort was disjoint from evaluator
development at selection time, but an outcome-inspection incident unsealed it.
It is permanently limited to diagnostic or sensitivity work. Exact public
Ctrl-World closed-loop reproduction was also unavailable for the reasons above.
No genuinely disjoint labeled snapshot plus runnable policy endpoints and a
qualified WAM was found.

Accordingly, Phase B was not measured. No complete terminal episode with
policy re-query, no multi-policy ranking, and no independent confirmation was
produced. The successor claim ceiling is specialized-architecture technical
compatibility plus incomplete post-unseal diagnostics.

## Skeleton findings

The new deterministic skeleton-only video passed as intended-motion
visualization only. The visible-skeleton artifact passed as an explicitly
attributed review composite, not native OSCAR output. The skeleton-region
occlusion retained the generated scene outside the mask, but the projected
centerlines were approximate, portions left frame, and the occlusion did not
remove the full rendered robot. It therefore cannot isolate all robot pixels or
prove scene consequences.

Historical Experiment 2 remains the quantitative reference: visible-skeleton
response was approximately `0.296703`, skeleton-masked scene response was
approximately `0.039976`, and the validity lower bound was approximately
`0.387755` against a `0.80` gate. These findings are not combined with the new
technical canaries to manufacture stronger evidence.

## Identities, protocols, and evidence

The candidate policy identity is
`nvidia/Cosmos3-Edge-Policy-DROID@3ea407af3e156c0af3b4bb6edd85842cc9a58777`
with cosmos-framework
`2f603cb114ff8b335e116060444d0b6caee3a85e`. OSCAR source and checkpoint are
`4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb` and
`c9781ffa7dd8556d862d7d9f338a2ea008a58ca6`. Ctrl-World source and checkpoint
are `99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d` and
`8cf814693f411962dc866a2ddb5b785afd17a93a`. The Ctrl-World runtime image is
`docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2`
with torch `2.7.1+cu128`.

The public OSCAR rollout snapshot is revision
`db5edfaef285c15d0a41d5115177a983c08b4f5f`; 63 complete sessions times seven
policies yielded 441 episodes. Blueprint did not invent the paper's two missing
sessions. RoboArena input availability was audited at dataset revision
`7931db81f3f6a48a3245427f7213a4c461f92ccc`.

The base protocol digest is
`76a2e0d32804c8c928c435828a18a5fbbc3eb95f3746b59c117e08beb41b60e6`.
The governing amendment-v15 digest is
`cd0952f97a5bc7139ce88a849fa99b7a442f5164a85b48e3f0bac752fc4304ae`,
source-freeze-v10 is
`e43a961e0cafe6fd8129ab552a64e9da4fff00ca57797b7087b2f502b3063f81`,
and the complete-graph judge protocol is
`4636d2e4a6b834f24f86f5073ef831e47c843925a11c1ad761d27663f8194954`.
Earlier freezes and amendments remain immutable.

The machine-readable result is `terminal_verdict_v1.json`; arm-level evidence
is in `final_evidence_matrix_v1.json`; review media are indexed by
`review_gallery_manifest_v1.json`. Large media remain in the external evidence
store and are linked by local path and content hash without credentials or
signed URLs.

## Abstention and failure layers

Abstention behaved correctly for the frozen native-Cosmos arm: it rejected an
unreliable action response. The new OSCAR and Ctrl-World canaries never reached
a causal or ranking gate, so they receive no calibrated abstention claim. The
historical GPT-5 mini stack abstained on 54/441 episodes but retained only
5.02% selective pair coverage and 0.60 selective accuracy, failing selective
usefulness. Gemini had zero abstentions and therefore did not establish
calibrated selective use.

Failures were kept attributable: native Cosmos failed at causal response;
OSCAR closed loop failed at the observation/state interface; the Cosmos plus
OSCAR-skeleton hybrid failed at the conditioning interface; exact Ctrl-World
closed loop failed at unpublished policy identities; GPT-5 stopped at provider
quota; Gemini stopped at batch transport; Phase B failed at independent data
availability; captured-site transfer and economics were consequently not
admitted.

## Provider and publication state

Vast allocation 12 was destroyed, its watchdog reached provider-terminal, all
four exact staged objects were deleted with absence proof, and its task/global
inventory was zero at closure. Its provider-zero evidence hash is
`bc903e9456dda1ef9256223e94f22452ae3eedcde1dda0c08707b3aabe8078e6`.
No experiment GPU or continuing experiment spend remains.

A later read-only account audit observed one unrelated shared-account L40S
instance, `46278054`, labeled
`blueprint-native-warehouse-camera-v13-faf0664f-a1`. This experiment did not
modify or stop it. The experiment is task-scoped provider-zero; global shared
account zero must be rechecked after that independent work closes.

Protected-main publication, hosted checks, exact-final local tests, merge SHA,
and final parity are recorded after this report commit is published. They are
not inferred from a pre-merge branch.

## Cheapest scientifically valid next experiment

Do not buy more open-loop replay canaries. Acquire a genuinely new,
independently administered blind DROID snapshot with matching physical outcomes
and runnable frozen `pi0_droid` and `pi0_fast_droid` endpoints. Freeze exact
policy revisions, then execute Ctrl-World's policy-in-loop interface under a
separately attributable reproducible environment with own-action, no-motion,
shuffled, reversed, shifted, and real policy-swapped controls. Use sessions as
the independent units. Only after causal and abstention gates pass should
Blueprint pay to complete both independent judge graphs and join labels.

In plain English: we proved the public model plumbing works. OSCAR and the
Ctrl-World canary both generated plausible robot futures, but both tests replayed
recorded actions; neither let the robot policy see the generated future and act
again. Native Cosmos failed to respond reliably to its actions, the proposed
hybrid had no valid public input channel, and the judge runs did not finish.
Blueprint abstained instead of turning those artifacts into a policy-ranking
claim. Nothing in this successor independently confirmed useful policy ordering,
captured-site transfer, or a cheaper/faster replacement for physical testing.
