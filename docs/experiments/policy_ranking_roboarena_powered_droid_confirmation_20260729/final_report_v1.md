# Powered DROID causal confirmation — final report

## Overall verdict

`thesis_not_supported`

This verdict applies to the frozen tested stack. The completed Phase-A GPT-5
mini calibration missed its registered ranking and selective-use gates, and the
adequately powered native-Cosmos causal screen passed zero windows and zero
sessions. Captured-site transfer and completed-ranking economics remain
unmeasured; those missing components do not turn the two direct necessary-gate
failures into support.

## Component verdicts

- `cosmos_wam_qualification`: **not supported**. Native
  `nvidia/Cosmos3-Nano` failed every registered causal and reliability gate;
  Blueprint correctly abstained.
- `frozen_benchmark_calibration`: **not supported**. The 441-episode GPT-5
  mini known-answer reproduction missed every required aggregate/selective
  endpoint. The stronger Gemini result is post-unseal diagnostic evidence and
  never abstained.
- `captured_site_transfer`: **inconclusive / not measured**. Prerequisites did
  not pass and no matching independently attributable site outcomes were
  available.
- `economics_and_speed`: **inconclusive / not measured for a useful ranking**.
  Digital causal-screen cost and runtime are measured below, but no ranking
  passed usefulness gates and no defensible physical-evaluation anchor was
  joined.

## What executed

The replacement RTX PRO 6000 Blackwell allocation loaded pinned
`nvidia/Cosmos3-Nano` revision
`411f42a8fdfb8c5b2583cb8786e0938f49796eaa`, passed the structured canary,
and generated the complete untouched matrix. The matrix contains 17 independent
sessions, three windows per session, recorded actions plus valid no-motion,
shuffled, reversed, shifted, and real policy-swapped controls, two seeds, and
three separately analyzed views.

- Provider attempts: `613` = one canary plus `612` matrix requests.
- Valid matrix responses/videos: `612/612`.
- Model load: `440.230` seconds.
- Inner provider runtime: `3,207.550` seconds.
- Durable output archive: `200,594,982` bytes, SHA-256
  `889e997ee61b26c8ca4cef50a5c31b84504f280314bdd70ee5ff6537a799c190`.

Vast removed the completed container between outer log polls, so the outer
adapter initially recorded `vast_heartbeat_container_missing`. The already
uploaded archive was fresh for this allocation, intact, and carried a completed
runtime result. This operational false negative did not lose or alter scientific
rows. Reusable fail-closed recovery landed in commit `eb043e85`; it rejects
stale callbacks and does not promote transport completion to scientific
validity.

## Native Cosmos causal result

The result is decisively negative under the frozen gates:

- Passed windows: `0/51`.
- Reliable sessions: `0/17`.
- Session-clustered causal-validity estimate: `0.0`, 95% interval `[0.0, 0.0]`
  from 10,000 session-clustered bootstrap replicates.
- Reliable-session estimate: `0.0`, 95% interval `[0.0, 0.0]`.
- `static_under_command`: `46/51` windows.

Median correct-action effect divided by cross-seed noise was `0.524` wrist,
`0.488` left, and `0.495` right, below the frozen minimum `1.0`. Median
same-seed recorded-versus-zero scene distance was `0.00406`, `0.00392`, and
`0.00440`, below the frozen minimum `0.01`. Some per-view timing correlations
looked favorable in isolation, but the correct-action scene effect was small,
often no larger than no-motion, and consistently weaker than ordinary seed
variation. Those weaker signals cannot be combined into a pass.

Blueprint abstained on the unreliable arm. That abstention was correct. No VLM
evaluator or policy-ranking score was allowed to erase the causal failure.

## Phase A and rank evidence

The completed non-independent Phase-A reproduction used
`gpt-5-mini-2025-08-07` over 63 sessions, seven policies, and 441 episodes. It
obtained Spearman `0.357143`, Kendall tau-b `0.238095`, policy pairwise accuracy
`0.619048` with clustered interval `[0.428571, 0.857143]`, selective coverage
`0.050182`, and selective pairwise accuracy `0.600000`. It did not pass.

Gemini 3.6 Flash later achieved Spearman `0.75`, Kendall `0.619048`, policy
pairwise accuracy `0.809524`, and direct within-session pair accuracy `0.721925`
with session-clustered interval `[0.676303, 0.762410]`. That matrix was already
unsealed and Gemini never abstained, so it remains diagnostic and cannot admit
Phase B.

No rank metric exists for the new Cosmos matrix because these are short
open-loop causal-control windows, not complete terminal episodes. Computing a
leaderboard from them would violate the protocol.

## Phase B design actually achieved

The achieved fallback is frozen replay of real candidate-policy traces. It is
open-loop WAM qualification only. A live policy → WAM → same-policy loop, the
joint/state observation adapter, complete terminal episodes, evaluator ranking,
and independently labeled new snapshot were not executed. The native arm failed
the admission gate, so full Phase B was not run.

## OSCAR, skeleton, and hybrid findings

Historical OSCAR Experiment 2 remains `thesis_not_supported` for its separate
frozen stack. Across 49 held-out session clusters, visible skeleton-overlay
signal had mean excess action correlation `0.296703`; skeleton-masked scene
signal was `0.039976`, 95% interval `[0.012196, 0.067883]`, with clustered lower
validity bound `0.387755` against the required `0.8`. This is intended-trajectory
evidence, not useful scene-dynamics evidence.

No fresh skeleton-only, OSCAR purpose-built WAM, or Cosmos + OSCAR-skeleton
hybrid arm was run in this powered experiment. Native Cosmos remained separate
and consumed no OSCAR-generated frames.

## Cost, time, and provider zero

Two GPU allocations consumed `4,615.574` live seconds and an estimated
`$2.226112` total. The first transport-defect attempt cost `$0.477329`; the
complete replacement cost `$1.748783`. This stage made zero evaluator and zero
policy API calls. Storage/transfer cost was not separately attributable.

Authenticated Vast inventory after teardown showed zero live instances and
`$0/hour` continuing burn. Provider zero does not prove invoice settlement or
scientific validity.

Terminal object-store closure then deleted the four exact bundle/output objects
recorded across the two allocations, received HTTP 404 absence confirmation for
all four, removed all six signed-URL files, and confirmed that no task watchdog,
spend guard, or paid allocator process remained. The machine-readable receipt is
`provider_zero_and_object_closure_v1.json`.

No “substantially faster” or “substantially cheaper” claim is allowed: the
experiment did not complete a useful ranking or join a defensible exhaustive
physical-evaluation cost/time anchor.

## Review media and immutable evidence

The full local evidence root is the operator-bound
`${BLUEPRINT_POWERED_DROID_EVIDENCE_ROOT}`.
The non-cherry-picked gallery is under `live_run_v5/review_gallery_v2`; it
contains all six action conditions, both seeds, all three separated camera
views, and the original composites. All 48 gallery videos passed hash and
ffprobe validation. Gallery manifest SHA-256 is
`1d1ee91486448d2db90acff1a7e0b5d3305b8fd047df669fe59a110f0437bee4`.

The analyzer output is `live_run_v5/powered_droid_analysis_v1.json`, file
SHA-256 `87a8a5ced3c6ca189a0edc2a989e0983493b876e5253caf819eca9400ee542ce`.
The frozen protocol manifest SHA-256 is
`6a7486b8eda01057934106206c5ecf3de808d19f3b3124a4a829c7a144f4c689`.

## Unresolved limitations and cheapest valid next experiment

The sessions are disjoint but come from the same published RoboArena snapshot;
this is not new-snapshot independent confirmation. The tested prompt is the
official label-free single-space DROID forward-dynamics input, the clips are
short causal windows, and no closed-loop state adapter exists. The result does
not prove all possible Cosmos adapters or prompts fail.

The cheapest scientifically valid next experiment is to freeze a separate
Cosmos + OSCAR-skeleton adapter and skeleton-only baseline, select 17 untouched
label-sealed sessions, and repeat only the causal/control matrix before spending
on an evaluator. Visible-skeleton and skeleton-masked scene outputs must be
scored separately. Do not rerun native Cosmos unchanged, and do not rank unless
the new arm passes.

In plain English: the machinery successfully ran hundreds of controlled robot
future predictions and safely preserved them. But Cosmos usually produced
almost the same weak/static future whether it received the correct actions or
the wrong ones, and random-seed variation was larger than the action effect.
Blueprint noticed that and refused to rank policies. This proves the harness can
detect and abstain from an unreliable WAM; it does not prove the closed-loop
product, captured-site accuracy, or the speed/cost thesis.
