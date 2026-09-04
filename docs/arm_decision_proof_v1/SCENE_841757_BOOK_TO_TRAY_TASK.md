# Scene 841757 book-to-tray task freeze

## Decision

The leading fresh-scene development-only rehearsal is SAGE/InteriorGS scene
`841757` (`0839_841757`). The movable source object is label instance `115`, an
open book resting on label instance `85`, the long TV cabinet. The task is to
pick up the book, place it fully inside a separately authored blue document
tray on the same cabinet, release it, let it settle, and retreat.

This is a selection and preparation freeze, not evidence that the task is
reachable, that the appearance is qualified, or that either learned policy can
complete it. The CPU placement/IK inventory, native Isaac import, calibrated
camera coverage, controls, and one paired learned-policy diagnostic cell remain
required before Quick-10.

## Immutable source inventory

| Input | Exact source | SHA-256 |
| --- | --- | --- |
| InteriorGS appearance | `0839_841757/3dgs_compressed.ply` | `22150c41ca85d2f9f3b89d070a767d9f2daa8d39abd14f464ddb85657e8a8897` |
| InteriorGS labels | `0839_841757/labels.json` | `8e157200a7b1b29af7575f0e21a78d754e81c5316d6eebe594d090af8cb6692e` |
| InteriorGS structure | `0839_841757/structure.json` | `36a35122e8028ed2c3a8e8bfdf3aaa35a5c87927bdc3605d41623100d920857f` |
| SAGE collision USD | `Collision_Mesh/841757/841757_collision.usd` | `2ba60fb13f8704d7885df54233a25e1bdbf4800ee54deb5d3bc9a6301cd06dbd` |
| SAGE scene USDZ | `InteriorGS_usdz/841757.usdz` | `d7de3a7939c935f2a015c57aebc1e8e7a6f85ec2ba61c89b2cd318c9924e4956` |

Publisher-pinned sources:

- InteriorGS scene revision `334dfeea4e0241033b4e5de97c01bc7c9c080530`:
  <https://huggingface.co/datasets/spatialverse/InteriorGS/tree/334dfeea4e0241033b4e5de97c01bc7c9c080530/0839_841757>
- SAGE collision revision `3ba75cc7887b62bf84211d5db08adfa64d691597`:
  <https://huggingface.co/datasets/spatialverse/SAGE-3D_Collision_Mesh/blob/3ba75cc7887b62bf84211d5db08adfa64d691597/Collision_Mesh/841757/841757_collision.usd>
- SAGE USDZ revision `d0b6cdc0aa052d38d743339bb629799ae81e7966`:
  <https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz/blob/d0b6cdc0aa052d38d743339bb629799ae81e7966/InteriorGS_usdz/841757.usdz>

Use remains private and `development_only` under the dataset's
non-commercial research/education terms. Do not redistribute raw dataset
bytes or imply commercial/deployment rights.

## Observed source geometry

Instance `115` has an oriented source box approximately
`0.2953 x 0.3977 x 0.0211 m`, with Z from `0.2755` to `0.29664 m`. It is thin,
visible in the retained room survey, and had a unique SAGE collider match with
measured IoU `0.864` during candidate screening.

Support instance `85` spans approximately `0.375 x 5.1686 x 0.275 m` and had a
unique SAGE collider match with measured IoU `0.980`. The low support height is
a deliberate reachability risk: it is a reason to run the CPU base/reset/IK
inventory, not a reason to guess that the DROID-compatible Franka can reach.

## Destination object to qualify

Author one rigid, open-top blue document tray through the harness-approved CAD
skills. Freeze the following design intent, then let deterministic fit and
native validators determine the final admitted dimensions:

- fit the `0.2953 x 0.3977 m` book with positive containment margin;
- remain fully supported by the cabinet's approximately `0.375 m` short axis;
- use a thin base and low walls so the gripper can enter and retreat;
- expose simple convex collision parts and one exact bottom support rigid body;
- use a non-reflective, camera-visible blue material distinct from the book and
  cabinet;
- encode a shrunk interior scoring volume that requires the whole book, not
  only its center, to be inside;
- retain authoring script, parameters, kernel/export versions, USD, rights,
  mass, friction, collider decomposition, and renderer/native qualification.

A reasonable initial CAD proposal is an outer footprint near
`0.33 x 0.48 m` with thin approximately `5 mm` lips and a usable interior near
`0.32 x 0.47 m`. This leaves about `22.5 mm` of cabinet support margin on each
short side when centered. These are proposal dimensions only. They are not a
qualified asset, pose, or success boundary.

The production authoring agent uses `gpt-5.6-sol` with the exact pinned `cad`
skill source. It receives the owner-authored metric constraints and writes the
STEP-first build123d generator; it is not allowed to browse the web or alter
the dimensions. The production executor statically rejects file, process,
environment, dynamic-import, and network access in generated source, executes
the pinned CAD skill with retry zero, and reopens the STEP for deterministic
measurement. Its result remains pending visual, static, native-import, and
scene-placement qualification and cannot self-grade.

## Success contract to confirm before launch

The site owner must explicitly accept or change each clause. The proposed
strict contract is:

1. the book is acquired without a forbidden robot/background collision;
2. the book is lifted above a task-authored minimum height;
3. no drop event is permitted at any time;
4. the book is transported into the tray without leaving the admitted robot
   workspace or task containment region;
5. during the terminal settle window, the full book footprint is inside the
   tray's shrunk interior bounds and its support height is valid;
6. the book is supported by the tray's exact bottom rigid body;
7. the gripper releases the book and retreats by the authored clearance;
8. force, collision, retry, and regrasp limits are explicit rather than inferred
   by a learned interpreter.

The learned episode interpreter may explain the trajectory and flag possible
events, but deterministic simulator state and the confirmed site contract own
the score. Conflicts are visible and fail closed.

## Gated execution order

1. Generate and inspect the digest-bound InteriorGS preselection review pack.
2. Confirm book instance `115`, support `85`, tray design, placement, task
   wording, cameras, workspace, reset distribution, and success clauses.
3. Execute the fresh-scene segmentation/excision/ArtiFixer path with required
   grading and, if applicable, one bounded selective re-edit/retrain/regrade.
4. Author and independently qualify the tray and book replacement as SimReady
   rigid assets.
5. Solve CPU base/reset/IK inventory for the entire pick/place/release/retreat
   phase plan; refuse the task if no exact feasible inventory exists.
6. Let the production v3 progression run the executor-owned rigid-destination
   qualification probe and derive the placement receipt from its native
   measurements; do not stage the receipt by hand.
7. Run a no-policy render/camera/task-contract probe and deterministic
   zero-action plus scripted-positive controls.
8. Run one paired diagnostic cell with the two frozen DROID-compatible policies.
   Continue only when observations, actions, motion, task-directed progress,
   scoring readbacks, and evidence delivery are valid.
9. Run the ten aligned Quick-10 cells, close billing/provider-zero, stream the
   full result archive, invoke deterministic scoring and independent learned
   interpretation, and verify the Website exposes all twenty learned-policy
   episodes plus controls and digest-bound downloads.

## Tray qualification path

The tray has no source object, so its Isaac-native import qualification and its
task geometry are produced by the production scene-configuration run itself:
the `scene_configuration` request declares the tray under `task.destination`
without `native_import_qualification` or `geometry`, the recipe binds the same
tray as `supplemental_destination`, stage 4 re-derives the static
qualification from the exact bytes, stage 5 settles the tray natively in the
same Isaac session as the book replacement, and publication derives the
whole-book containment geometry and completes the revision's destination.
Only that published destination is admissible for the later
`destination_qualification` probe and Quick-10 episodes.

## Production admission safeguards

New passive-destination CAD requests use `task_evaluation_passive_destination_cad_request.v2`.
They must specify minimum interior clearance on all three axes. A draft that fits
XY but is shallower than its minimum Z is refused before model invocation.
These proposal-level checks do not replace orientation-aware eight-corner
containment, full trajectory IK, native contacts, or scene placement qualification.

The static USD gate resolves the physics-purpose material binding of every
collider and verifies that every collider belongs to the single intended rigid
body. An unbound material with plausible coefficients is insufficient. Destination
outputs distinguish the owning support rigid-body path from the exact bottom
collision-prim path; native qualification must establish the actual support contact.

The preselection command accepts `--production-runtime-root` to validate the
sealed renderer runtime independently from the executing Python source checkout.
It remains reconnaissance only, not a calibrated method-input or camera
qualification substitute. These safeguards do not claim that the supplemental
CAD-to-Content-Agents construction sequence has been executed or qualified.
