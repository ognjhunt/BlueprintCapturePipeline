# Scene 841757 G1 movement goal decision packet

**Status:** proposed for task-owner review. This is a development simulation
goal on the retained book-task site. It is not a navigation authority receipt,
an executed policy result, or a claim of obstacle avoidance.

## Blocker Title

Confirm the marked-area destination for the G1 movement pair on scene 841757.

## Blocker Id

`human-blocker:g1-841757-movement-goal-20260925`

## Why This Is Blocked

ADP-050 Day 28 compatibility has a materialized movement scene packet and a
proposed yellow marker, but the movement goal changes task truth. The site/task
owner must confirm that this is the destination to evaluate before
`seal_g1_navigation_goal_authority` can bind it to the exact scene plan. An
agent cannot infer that authorization from static geometry or a packet digest.

## Recommended Answer

Confirm the proposed development-only goal below if it matches the intended
movement task for the same 841757 captured site:

| Field | Proposed value |
| --- | --- |
| Robot root at reset | `[-2.703953, -3.441138, 0.79]` m |
| Yellow goal center | `[-3.703953, -3.441138, 0.0]` m |
| Planar start-to-goal distance | 1.0 m west along world X |
| Visible marker | Non-colliding flat yellow disc, radius 0.25 m |
| Arrival radius | Root XY within 0.20 m of goal center |
| Terminal hold | 5 scorer samples inside arrival radius |
| Height bound | Root height drift no greater than 0.15 m |
| Policy instruction | “Avoid obstacles and move to the yellow marked area.” |

The existing manipulation task is still “pick up the open book, place it over
the green target marker on the tabletop, release it, and move the gripper
clear.” The yellow movement marker is a separate side objective in the **same
scene**, not a replacement for the green book target.

## Alternatives

- Provide a corrected yellow goal center and acceptance criterion. The authoring
  manifest and scene plan must then be regenerated and reviewed before sealing
  authority.
- Hold movement evaluation while continuing packet-only manipulation checks.

## Downside / Risk

The proposed location has a static footprint-corridor check, but floor support,
camera visibility, dynamic contacts, and policy motion have not been observed
in Isaac. The current navigation scorer tests arrival and hold; it does not
score the instruction's obstacle-avoidance clause. An approval would authorize
only the named goal definition for development simulation.

## Exact Response Needed

Reply **confirm**, **correct**, or **hold** for this exact proposed center,
radius, root-height bound, and terminal hold on the retained 841757 task/site.
For a correction, supply the intended measured goal center and criterion.
Name the confirming site/task team and human reviewer so the authority receipt
can truthfully record who made the decision.

## Execution Owner After Reply

`pipeline-codex`, after the named site/task owner decides the goal.

## Immediate Next Action After Reply

For **confirm**, seal `native_g1_navigation_goal_authority.v1` against scene-plan
digest `sha256:57d4f9ff69225c1c8bc5754157df09a62983dae5ec42b52785cef5146fc3e293`
using the real reviewer and team id; then revalidate the no-spend worker request.
For **correct**, regenerate the proposed movement authoring and scene packet
and return the new exact plan for review. For **hold**, keep movement disabled.

## Deadline / Checkpoint

Before launching either `HSI_vision_navi` candidate on 841757. Revisit when
the site/task owner replies.

## Evidence

- Proposed source:
  `docs/arm_decision_proof_v1/manifests/g1_841757_book_movement_authoring.proposed.v1.json`,
  authoring digest
  `sha256:41ad27d02ed7b2e39c4817d39da07b5ca316fef00ffa23fde5a80fb5fdb98584`.
- The locally materialized movement packet retains the 841757 scene, book
  manipulation task, G1 reset pose, and yellow goal. Its packet receipt was
  `sha256:635f56c4ae3418f56d2a8bfb1648f9a0efdf08e51a73fb61637f28ac1788087d`.
  This receipt establishes packet construction only.
- The development static footprint check reported 21 of 21 sampled points on
  the straight 1 m corridor clear of 1,386 retained collision boxes. This
  does not establish floor support or a successful robot episode.
- `docs/g1_shared_scene_asset_setup.md` describes the goal-authority and
  scorer boundaries. The separate model-rights decision is recorded in
  `docs/g1_841757_model_rights_review_2026-09-25.md`.

## Channel Target

Durable review route: `ohstnhunt@gmail.com`, with the blocker id in the
subject. This packet does not send email or assert that a reply watcher is
active. The decision may also be made in the current Codex thread and
recorded in the owning run artifact.

## Non-Scope

This decision does not authorize checkpoint download, SONIC use, GPU spend,
physical G1 operation, public video, a qualified policy ranking, or a claim
that the robot avoided obstacles. Those require their own evidence and gates.
