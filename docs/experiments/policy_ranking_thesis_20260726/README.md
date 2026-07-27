# Blueprint policy-ranking thesis experiment

Status: complete. Final verdict: `thesis_not_supported`.

The exact-main Vast campaign completed 24 learned-policy episodes and then
abstained from a total ordering in both transfer scenes. Frozen RoboArena
calibration did not pass the registered selective-abstention or action-following
gates. See `final_verdict.md` and `final_verdict.json` for the decision and
claim boundaries.

## Claim under test

Given a captured real-site representation, a real robot task, and multiple candidate policies, Blueprint can predict a useful policy ordering substantially faster and more cheaply than exhaustive physical evaluation while abstaining when the prediction is not trustworthy.

The protocol keeps two results separate:

1. **Frozen benchmark calibration** compares a fixed WAM/evaluator against previously collected independent RoboArena real-policy outcomes.
2. **Captured-site transfer** retains a previously unseen 3DGS site as its visual source, adds only local task/robot interaction layers, and emits a prospective externally calibrated ranking. It cannot prove physical success at that site without site-specific outcomes.

## Current embodiment and task

The proof-stage embodiment is a DROID-compatible Franka Panda. The benchmark pool uses the seven policies shared by all 63 complete sessions in the public OSCAR rollout release. The transfer task is rigid pick-and-place: pick up a spray can and place it inside a marked tray.

The initially proposed Floppy Bagel checkpoint ladder was rejected before scoring because only nine sessions contain any pair of the five checkpoints and none contains more than two. That coverage cannot support the registered seven-way comparison.

## Frozen identities

- Protocol v2: `8ed8a7b4da73120b1806147a9b2de8b88bc76eaf1bbe2e582a1080212552b13c`
- Preserved protocol v1: `fadd4e3b701d6c51f4df7deb275d17668f501334a17bb948e0c6860f6665ad1d`
- Power analysis: `38d2646ed03db9f121b753a00b8789b83019a0b4accae84478854a72b8cd778e`
- RoboArena snapshot: `7931db81f3f6a48a3245427f7213a4c461f92ccc`
- Label-blind rollout index: `7861da1a77eb93a271c2b4bd1cd825d9efa4708f71642f7b01790df6c1169f20`
- OpenAI judge configuration: `3c282af98cd968a32fa130fc0d717b7aa4d471f50ff5b7b204e2cff508671314`
- OSCAR released rollout revision: `db5edfaef285c15d0a41d5115177a983c08b4f5f`
- Primary InteriorGS captured-site hybrid bundle: `c19eac1ccc9a2ed0564d6a0a9e5b3ea2e8dd751bc9534ae59bf11c498f8d4645`
- NVIDIA controlled-scene bundle: `fa7ddacdfafadcc9208665ba3c8b3fd5e0fed90a42f9f83ad4532ba7f72c6a95`
- Local articulated Franka feasibility oracle: `076769dbf6fb7858d0f6f9cebefa1efb509e0fbc9d3ea13e6225c4b08217a4f3`
- OpenPI DROID-to-MuJoCo bridge contract: `c187106f29386c362563057e46959cf64382cb4809e732a90253efa574f586a6`
- Native-square captured-site policy camera contract: `f4a2a6c02de426922b4d52290ec758f06f41e5381871acc724ab9c186c800c0e`
- Frozen captured-site ranking aggregator: `captured_site_policy_ranking.v1` with three preregistered can-position variants and strict interval separation
- Frozen OpenPI PolaRiS checkpoint inventory: canonical SHA-256 `492e...`; 47,286,181,297 bytes across four exact checkpoint generations

## Completed evidence

- Indexed 441 released OSCAR rollout videos across 63 sessions and seven policies without copying benchmark labels or evaluator PII.
- Materialized the seven-session pilot: 49 WAM videos and 49 attributable RoboArena action/proprioception files.
- Ran a label-blind 49-row action/motion diagnostic. Of 47 nonconstant rows, action-magnitude versus generated-pixel-motion Pearson correlation ranges from 0.0534 to 0.8717 with median 0.4881. This is a contradiction check, not proof of 3-D action following or success.
- Built a 98-request pilot inventory: 49 frozen 32-frame temporal judgments and 49 first/last-frame baselines. Only the generated left half is cropped locally; benchmark labels, PII, and physical-video pixels are excluded. Three superseded technical configurations are preserved and excluded from ranking; the last produced 44 accepted rows and four incomplete rows before termination showed that 4,096 shared reasoning/output tokens were still insufficient.
- Completed the one-digest v2 pilot with 98 of 98 accepted judgments, no failed requests or blockers, $3.0161225 conservative metered cost, and 420.179 seconds recorded for the completion invocation. Predictions remain label-blind; this is evaluator execution evidence, not benchmark calibration or ranking success.
- Joined the pilot partition only after prediction freeze. The amended primary `binary_then_partial` basis yields 0.6308 pairwise accuracy and positive rank correlation, but the full temporal judge is not better than chance at the registered lower-confidence bound, selective coverage is 0.0231, and action-following pass rate is 0.1633. The pilot therefore does not pass the registered gate set.
- Completed 98 of 98 label-blind calibration judgments under the same evaluator digest with zero failed requests or blockers, $2.89694 conservative metered cost, and 474.981 seconds wall time. The subsequent registered calibration join yields 0.6726 primary pairwise accuracy, but selective coverage is 0.00885 and action-following pass rate is 0.1020; the complete calibration gate set does not pass.
- The first held-out provider attempt failed closed under rate limiting after 121 of 686 judgments. Before a later no-key preflight replaced the output file, the partial artifact was observed at 565 rate-limit blockers, $3.6246275 conservative metered cost, 3,115.916 seconds wall time, and canonical run SHA-256 `26af01424e6f997ba9193edc7202230b60ee0a1ad2b92ad64ed69cd88f380595`. The judgment rows are no longer preserved locally, so this attempt cannot support held-out metrics or ranking.
- Repeated the exact frozen held-out inventory with concurrency reduced from eight to two. Attempt 002 terminated with 0 of 686 accepted rows, 686 final `RateLimitError` blockers, zero metered usage cost, 2,199.172 seconds wall time, and run SHA-256 `23e1eabcb8e50200905f78f9fb5b5cfb032c854a45c40411a8726942a23a1041`. Labels remain sealed; no held-out metric exists.
- Added a session-cluster power analysis before calibration or held-out unseal. With all 49 remaining released sessions, approximately 0.678 pairwise accuracy is needed for 80% power at one-sided alpha 0.05 under the conservative model. Small effects and wide confidence intervals remain inconclusive; adjacent pairs received no favorable exemption.
- Retained the Voxel51 playroom as a rights/metric-scale-limited fallback rather than the primary captured site.
- Promoted user-authorized InteriorGS scene `0787_841244` to the primary captured-site lane. Its metric 3DGS remains separate from a local table proxy, NVIDIA spray can, Blueprint tray, and Franka layer; the ready hybrid bundle does not rebuild the room as USD.
- Rendered four 1920x1440 task views directly from all 630,898 InteriorGS splats in 41.4 seconds on the Mac's local Metal GPU, with no cloud GPU. Two external views and a physically plausible link-derived initial Franka wrist view pass the static observability gate. A hand-occluded mount, a self-occluded camera-proxy render, and an unrealistically distant forearm view remain recorded as rejected or nonprimary candidates.
- After user review rejected the clarity of the 4:3 presentation, rendered eight native-square camera candidates and froze a higher centered external view plus a raised link-mounted wrist view before any captured-site policy outcome. The selected pair was rerendered at 1536x1536 and materialized at the exact OpenPI DROID 224x224 input size from all 630,898 splats. At 224x224 the can/tray occupy 1.38%/10.87% of the external view and 2.75%/11.93% of the wrist view. Candidate and final rendering took 17.95 and 8.77 seconds on local Metal at $0 provider cost. The report explicitly retains the remaining static-proxy and dynamic-compositing limitations.
- Ran the pinned Menagerie Franka in local MuJoCo 3.9.0 as the preregistered scripted scene-feasibility oracle. It grasped the rigid can, lifted it 0.1012 m, transported it, released it, and settled at `[0.4131, 0.3167, 0.1199]` m inside the tray at 0.00054 m/s. The live hand pose produced distinct mounted-camera transforms across eight phases. This is neither a learned-policy result nor NVIDIA warehouse, WAM, captured-site, or physical evidence.
- Pinned the public OpenPI DROID runtime semantics into a fail-closed simulator adapter: one 224x224 external image plus one 224x224 wrist image, seven joint positions, one gripper position, 10x8 action chunks, 15 Hz control, an eight-action horizon, normalized joint-velocity clipping, and the public 0.2-rad maximum joint-delta mapping. Unit gates passed before the later exact-main checkpoint execution.
- Executed the initial joint-velocity seam end to end in local MuJoCo with the real textured Menagerie Panda, a fixed external camera, and a wrist camera recomputed from the live hand transform before every query. The literal zero control correctly failed. The unchanged scripted positive control also failed: 168 actions/21 queries remained contract-valid but lifted the can only 0.00133 m, missed containment, and ended unstable. Seventeen preserved engineering variants also all failed containment and never exceeded 0.00480 m lift. Because the separate continuous scripted oracle lifted 0.1012 m and passed, this was a controller-translation contradiction, not scene infeasibility. It blocked that joint-velocity lane; the separately frozen absolute-joint lane later passed its positive control and was the only lane admitted to GPU execution.
- Audited the bridge against pinned OpenPI and DROID sources. Corrected DROID's `0=open, 1=closed` gripper convention, immediate nonblocking desired-joint updates, and the published 1 kHz inner/15 Hz outer controller rates. The joint-velocity positive still fails and remains disqualified; those corrections did not erase the contradiction.
- Before any learned-policy outcome, froze a v2 transfer cohort of four public OpenPI PolaRiS DROID joint-position checkpoints totaling 47,286,181,297 bytes. OpenPI's source converts their absolute joint actions to deltas for training and restores absolute actions at output specifically to support simulation. The new joint-position positive passed the same deterministic task predicates with 0.10949 m lift, containment, and 0.00038 m/s final speed; the stationary negative was correctly rejected. This admitted only the joint-position learned-policy lane. The later exact-main campaign downloaded, verified, loaded, and executed all four checkpoints without changing this cohort.
- Replaced the static arm overlay with a dynamic hybrid compositor: the frozen captured-site 3DGS derivative remains the background while live MuJoCo segmentation supplies the actual articulated Panda, can, and tray at every external-camera frame; the wrist camera is rendered live from the hand transform. The stationary control failed as expected, while the absolute-joint positive completed 168 steps with 0.10949 m lift, containment, 0.00038 m/s final speed, and 5,895--6,731 dynamic foreground pixels. The background used for the private GPU lane is an explicitly decimated 300,000-splat derivative, not the full 630,898-splat source.
- Froze the captured-site score before learned execution. Per episode it combines lift progress, then gated transport, containment, and stability; a policy is ordered above another only when its three-variant minimum is strictly greater than the other's maximum. Overlap forces abstention. A secondary visual judge cannot override deterministic state and any contradiction forces abstention.
- Implemented a one-shot GPU worker for exactly four learned policies, two separately labeled scene lanes, and three frozen variants per scene. Each 47.3 GB checkpoint is downloaded, verified, and loaded once before its Warehouse and InteriorGS episodes; rankings and claim boundaries remain separate. The worker requires a JAX GPU, clears caches between policies, and never exposes a physical-robot endpoint. A local CPU attempt failed closed at `jax_gpu_device_not_present` before checkpoint download.
- Built and round-trip verified a private 104,716-byte v2 GPU input bundle containing only the 224x224 InteriorGS-derived and NVIDIA-USD-derived backgrounds plus their rights/hash manifest; no raw 3DGS or full warehouse asset is included. Signed input/output URLs are forwarded only through secret-named environment variables and are excluded from persisted provider artifacts.
- Added provider-neutral OpenPI paid admission with Vast as the frozen default and RunPod as an explicit fallback. A live read-only Vast probe on 2026-07-26 verified zero global billable resources and qualifying 45+ GB offers, including an advisory A40 around $0.28/hour. Admission reserves the frozen $0.75/hour ceiling instead of trusting that transient offer. This performed zero mutations and is not a reservation.
- Wired the frozen OpenPI campaign into the canonical `paid_resource_allocator gpu-canary` entrypoint. Its execute path refreshes provider capacity and global inventory, reserves worst-case USD/GPU-wall budget, acquires an exclusive provider-lane lease, confirms a separately running name-scoped teardown watchdog, opens a pending-teardown record, and only then grants one creation call. The exact-main Vast launch completed all 24 episodes, validated the signed output, destroyed the exact instance, and proved prefix/global absence. A post-delete Vast HTTP-200/`instances: null` classification bug delayed only the terminal receipt; the tested fix and provider-zero recovery closed the record, released the lease, and settled the budget.
- Preserved a pre-execution camera contradiction found in review: the earlier MuJoCo free-camera conversion followed wrist position and forward direction but discarded the mounted `up` vector, so it did not preserve optical-axis roll. The learned lane now uses a mocap-mounted fixed camera with the full link rotation; a native MuJoCo check reproduced the requested camera matrix to `2.64e-16` maximum absolute error. The earlier deterministic positive/negative control outcomes remain object-state evidence, but their saved wrist frames are superseded and are not GPU-policy inputs.
- The `linux/amd64` worker Dockerfile passed BuildKit static checks with no warnings and every build stage completed under local x86 emulation. The later exact protected-main CPU builder published immutable digest `f8f4dc01...`; the Vast target then passed native JAX GPU and MuJoCo execution on an RTX 6000 Ada. The earlier canceled branch-built laptop load remains validation-only evidence and is not the release identity.
- Materialized the NVIDIA Warehouse sorting-area crate-packing-table workcell and SimReady spray can without downloading the full warehouse. A 224x224 camera-bound USD render feeds the same live MuJoCo Panda/can/tray compositor used for captured-site transfer. Through the frozen absolute-joint-position seam, the 168-action scripted positive lifted the can 0.10954 m, placed it in the tray, and settled at 0.000055 m/s; the 160-action stationary negative failed lift and containment. The later 12 learned-policy Warehouse episodes all failed transport/containment and the total ranking abstained. Isaac physics was not used, and the warehouse is not a benchmark or physical answer key.

## Final execution

- Published the exact merged-main `linux/amd64` image at immutable digest
  `sha256:f8f4dc01704b9051f564262b43238666f3fb915c6d145f0345fb21f1f7bf0e40`.
- Ran four public OpenPI PolaRiS DROID joint-position checkpoints across the
  InteriorGS and NVIDIA Warehouse scene lanes, with three frozen variants per
  scene, on a single Vast RTX 6000 Ada.
- Completed and validated all 24 episodes. Both scene aggregators abstained from
  a total ranking; each emitted one strictly separated diagnostic pair.
- Proved exact-instance, prefix, and global Vast absence; settled 991 GPU-seconds
  at a conservative $0.206458 and released the paid-lane lease.
- Preserved the post-delete Vast HTTP-200/`instances: null` adapter bug and its
  tested correction. The bug affected only terminal control-plane evidence; it
  did not alter episode outputs or rankings.

## Closed experiment

The v2 pilot and calibration judge calls and their partition-scoped label joins are complete. Both stayed below their $9.00 per-run caps, used `store=false`, and persisted all predictions before their respective label joins. The first pilot join failed closed on the label shape; that diagnostic remains preserved, and the attributable label-basis amendment was recorded before calibration unseal. Neither pilot nor calibration passes the complete registered gate set. Held-out attempt 001 lost its partial rows after a no-key preflight; exact attempt 002 was rate-limited on all 686 requests. Held-out labels remain sealed. No held-out claim may proceed without a complete, preserved one-digest prediction matrix under the frozen arms and decision rules.

The captured-site campaign is no longer awaiting GPU execution. Its abstention
and failed task predicates are terminal evidence for this frozen experiment.
Any attempt to change policies, task, evaluator, thresholds, camera, or sample
count is a new experiment and must not overwrite this result.

## Proof boundaries

- Published OSCAR metrics are independent external evidence, not a Blueprint result.
- Generated video is not physical success.
- NVIDIA warehouse execution is a required closed-loop bridge, not a benchmark or physical answer key.
- The 441 released OSCAR rollouts may be an author-selected subset of the paper's 455; the selection process is undocumented.
- OSCAR rollout media remain internal-only because no explicit dataset license metadata was found.
- The InteriorGS hybrid bundle is a prospective transfer surface, not site-specific validation. Its terms restrict this proof to internal non-commercial research, prohibit raw redistribution, and require citation.
- The dynamic hybrid observations prove live articulated robot/task-layer compositing against a captured 3DGS-derived background and live wrist rendering. They do not prove 3DGS depth occlusion, full captured-site collision geometry, learned-policy execution, ranking fidelity, or physical success.
- Blueprint has not operated, commissioned, rented, or purchased a physical robot in this goal.
- The final verdict applies to this frozen pipeline and evidence. It does not
  prove physical failure of any policy at either transfer site.

See `evidence_matrix.md`, `quality_gap_ledger.md`, `rights_and_access_matrix.md`, and `cost_report.md` for the live evidence boundary.
