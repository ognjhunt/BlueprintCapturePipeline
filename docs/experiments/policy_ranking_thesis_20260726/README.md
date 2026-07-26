# Blueprint policy-ranking thesis experiment

Status: active and deliberately incomplete. No thesis verdict has been issued.

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
- Captured-site hybrid bundle: `993c5f4b8de85db58971fcc1c147573388af57cf51e0f354439e008303b1ed60`
- NVIDIA controlled-scene bundle: `274f68e9c219915906c5f01d11b09d0f440419b9a1be3ef964ca8047fd3007aa`

## Completed evidence

- Indexed 441 released OSCAR rollout videos across 63 sessions and seven policies without copying benchmark labels or evaluator PII.
- Materialized the seven-session pilot: 49 WAM videos and 49 attributable RoboArena action/proprioception files.
- Ran a label-blind 49-row action/motion diagnostic. Of 47 nonconstant rows, action-magnitude versus generated-pixel-motion Pearson correlation ranges from 0.0534 to 0.8717 with median 0.4881. This is a contradiction check, not proof of 3-D action following or success.
- Built a 98-request pilot inventory: 49 frozen 32-frame temporal judgments and 49 first/last-frame baselines. Only the generated left half is cropped locally; benchmark labels, PII, and physical-video pixels are excluded. Three superseded technical configurations are preserved and excluded from ranking; the last produced 44 accepted rows and four incomplete rows before termination showed that 4,096 shared reasoning/output tokens were still insufficient.
- Completed the one-digest v2 pilot with 98 of 98 accepted judgments, no failed requests or blockers, $3.0161225 conservative metered cost, and 420.179 seconds recorded for the completion invocation. Predictions remain label-blind; this is evaluator execution evidence, not benchmark calibration or ranking success.
- Joined the pilot partition only after prediction freeze. The amended primary `binary_then_partial` basis yields 0.6308 pairwise accuracy and positive rank correlation, but the full temporal judge is not better than chance at the registered lower-confidence bound, selective coverage is 0.0231, and action-following pass rate is 0.1633. The pilot therefore does not pass the registered gate set.
- Completed 98 of 98 label-blind calibration judgments under the same evaluator digest with zero failed requests or blockers, $2.89694 conservative metered cost, and 474.981 seconds wall time. The subsequent registered calibration join yields 0.6726 primary pairwise accuracy, but selective coverage is 0.00885 and action-following pass rate is 0.1020; the complete calibration gate set does not pass.
- Added a session-cluster power analysis before calibration or held-out unseal. With all 49 remaining released sessions, approximately 0.678 pairwise accuracy is needed for 80% power at one-sided alpha 0.05 under the conservative model. Small effects and wide confidence intervals remain inconclusive; adjacent pairs received no favorable exemption.
- Ingested the public Voxel51 playroom 3DGS and rendered a 20,000-splat local preview with separately hashed work surface, spray can, tray, and Franka proxies. A single task-focus camera rule was frozen before ranking and the rerender has six of six nonblank views. This proves visual hybrid composition only; its wrist view is still fixed rather than mounted to the articulated robot.
- Defined the equivalent NVIDIA warehouse control scene without downloading the full warehouse and without treating simulation as the answer key. Protocol v2 makes this Blueprint-operated closed-loop bridge mandatory before the captured-site claim because Lane A uses author-generated open-loop replays.

## Next irreversible gates

The v2 pilot and calibration judge calls and their partition-scoped label joins are complete. Both stayed below their $9.00 per-run caps, used `store=false`, and persisted all predictions before their respective label joins. The first pilot join failed closed on the label shape; that diagnostic remains preserved, and the attributable label-basis amendment was recorded before calibration unseal. Neither pilot nor calibration passes the complete registered gate set. Held-out predictions and labels remain unopened; proceeding must follow the frozen stop/decision rules without changing predictions, thresholds, partitions, or the evaluator digest.

Captured-site ranking begins only after at least four policy weights/action adapters and the action-conditioned WAM path pass fail-closed preflight. Any new paid GPU is separately admitted through `blueprint_pipeline.paid_resource_allocator`.

## Proof boundaries

- Published OSCAR metrics are independent external evidence, not a Blueprint result.
- Generated video is not physical success.
- NVIDIA warehouse execution is a required closed-loop bridge, not a benchmark or physical answer key.
- The 441 released OSCAR rollouts may be an author-selected subset of the paper's 455; the selection process is undocumented.
- OSCAR rollout media remain internal-only because no explicit dataset license metadata was found.
- The playroom hybrid bundle is a prospective transfer surface, not site-specific validation.
- Blueprint has not operated, commissioned, rented, or purchased a physical robot in this goal.
- Until both components and the economic comparison are measured, the only scientifically valid status is “not yet adjudicated,” not one of the final verdict labels.

See `evidence_matrix.md`, `quality_gap_ledger.md`, `rights_and_access_matrix.md`, and `cost_report.md` for the live evidence boundary.
