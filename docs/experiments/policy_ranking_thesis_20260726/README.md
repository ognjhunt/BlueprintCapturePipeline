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

- Protocol: `fadd4e3b701d6c51f4df7deb275d17668f501334a17bb948e0c6860f6665ad1d`
- RoboArena snapshot: `7931db81f3f6a48a3245427f7213a4c461f92ccc`
- Label-blind rollout index: `7861da1a77eb93a271c2b4bd1cd825d9efa4708f71642f7b01790df6c1169f20`
- OpenAI judge configuration: `42d119e46e3e97ac4b492bd948058a75ad01bc34ee9372be5d8747c788fee349`
- OSCAR released rollout revision: `db5edfaef285c15d0a41d5115177a983c08b4f5f`
- Captured-site hybrid bundle: `c3cc38b40acdaf906fb2b48c71341b9cdf7d6c57c1aa99ec0004077216a5c56e`
- NVIDIA controlled-scene bundle: `274f68e9c219915906c5f01d11b09d0f440419b9a1be3ef964ca8047fd3007aa`

## Completed evidence

- Indexed 441 released OSCAR rollout videos across 63 sessions and seven policies without copying benchmark labels or evaluator PII.
- Materialized the seven-session pilot: 49 WAM videos and 49 attributable RoboArena action/proprioception files.
- Ran a label-blind 49-row action/motion diagnostic. Of 47 nonconstant rows, action-magnitude versus generated-pixel-motion Pearson correlation ranges from 0.0534 to 0.8717 with median 0.4881. This is a contradiction check, not proof of 3-D action following or success.
- Built a 98-request pilot inventory: 49 frozen 32-frame temporal judgments and 49 first/last-frame baselines. Only the generated left half is cropped locally; benchmark labels, PII, and physical-video pixels are excluded. No provider call has occurred.
- Ingested the public Voxel51 playroom 3DGS and rendered a 20,000-splat local preview with separately hashed work surface, spray can, tray, and Franka proxies. Five of six views are nonblank. This proves visual hybrid composition only.
- Defined the equivalent NVIDIA warehouse control scene without downloading the full warehouse and without treating simulation as the answer key.

## Next irreversible gates

The first gate is the explicitly approved pilot judge call. Its conservative cost upper bound is $1.0912125 and the command hard-caps admitted cost at $2.00. Results are resumable and use `store=false`. Only after all 98 predictions are persisted may pilot labels be joined. The one manually opened pilot session is logged and may not change gates.

If the pilot execution is technically valid, calibration and then held-out partitions use the same evaluator digest. Captured-site ranking begins only after at least four policy weights/action adapters and the action-conditioned WAM path pass fail-closed preflight. Any new paid GPU is separately admitted through `blueprint_pipeline.paid_resource_allocator`.

## Proof boundaries

- Published OSCAR metrics are independent external evidence, not a Blueprint result.
- Generated video is not physical success.
- NVIDIA warehouse execution would be a controlled diagnostic, not a benchmark answer key.
- The playroom hybrid bundle is a prospective transfer surface, not site-specific validation.
- Blueprint has not operated, commissioned, rented, or purchased a physical robot in this goal.
- Until both components and the economic comparison are measured, the only scientifically valid status is “not yet adjudicated,” not one of the final verdict labels.

See `evidence_matrix.md`, `quality_gap_ledger.md`, `rights_and_access_matrix.md`, and `cost_report.md` for the live evidence boundary.
