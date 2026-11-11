# Plan: one fast, validated simulation proof lane

Written 2026-11-11. It follows the
[first-principles audit](../audits/2026-11-11-first-principles-audit.md) and
the owner's direction of the same day: simulation is the core proof in the
site's answer, and it must be accurate, cheap and quick. This is a plan. Only
step 1 has code on a branch; nothing here has been deployed or paid for.

## The outcome

For one kind of job, a site's video becomes a ranked comparison of robot
control policies, with the robot's reach and arm path checked in the same
scene, in about an hour and for a few dollars, with no operator. The ranking is
worth what published tests of the same method are worth, and Blueprint keeps
its own running record of how often it is right.

The lane is one kind of job on purpose: one solid object, picked up and put
down within one arm's reach, on the Franka research arm (the DROID setup). It
is the only setup with several public policies to rank and with published
agreement between simulated and real rankings.

## What the published method needs

Read from the papers on 2026-11-11.

- The simulated ranking of two policies matched the real ranking in 8 of 9
  comparisons in one direct test (arXiv 2610.17521). It fell to 7 of 9 with
  holes and specks in the scan, and to 6 of 9 with 30% of the images removed at
  random.
- Simulated and real scores correlate at about 0.9 on average, 0.81 in the
  worst scene (PolaRiS, arXiv 2512.16881), and from 0.77 to 0.96 depending on
  the reconstruction (arXiv 2609.23910).
- Those results needed four things:
  1. a scan with true scale (a printed scale board or markers in view);
  2. a calibrated robot controller;
  3. policies given a little simulated training (the PolaRiS checkpoints);
  4. a graded task-progress score, not only pass or fail.
- PolaRiS is open source (MIT). It reconstructs a scene from a 2–5 minute scan
  in about 30 minutes on one GPU and supports the DROID Franka arm.

## Where this repository stands against those four

| Condition | Today | Evidence |
|---|---|---|
| True scale | Estimated from reference objects by a model. No scale board or marker detection. Site-entered dimensions exist in the WebApp catalog and are not read on the website path. | `website_atlas_scene_source.py:38`; no `charuco`/`aruco` in `src/`; `contracts/pipeline/site-task-measurement-catalog.v1.json` (WebApp) |
| Calibrated controller | The DROID profile and the vendored RoboLab loop are in use. A gripper-state fault was fixed on 11-05. | `beta_policy_screening_plan.py:39-52` |
| Co-trained policies | In place: the four PolaRiS checkpoints are enabled. | `config/beta_policy_candidates.v1.json` |
| Graded score | Absent. Only pass or fail is published, though each episode already records closest approach, contact, lift, carry distance and placement. | `adp_rigid_task_scoring.py:314`; `task_predicates.py:436-466`; `beta_policy_screening_result.py:158-180` |

Two more facts shape the plan.

- **The one finished job could not have succeeded.** On capture-447 the
  robot was placed where the tray's near rim was in reach, but the grasp point
  at the put-down was 1.086 m away against a 0.990 m reach. The code computes
  that distance and does not use it (`robot_arm_path_clearance.py`,
  `task_contact_beyond`). The run cost about $27 and scored 0 of 39.
- **When nothing fails, a job costs about $34 and each policy takes about two
  hours.** $17 is a language model authoring the object, about $9.50 is
  reconstruction, and each policy's ten episodes cost $1.40–$1.70 on a GPU for
  about four minutes of simulated time (tracker lines 425-447; cost model).

## Steps, in order of value for effort

Each step is useful by itself and is proved before the next one starts.

1. **Place the robot where it can do the job, or hold before spending.**
   A stance counts as in reach only when the grasp point is within reach at
   both the pick-up and the put-down. When no stance qualifies, the existing
   admission stop holds the job before any paid run. On the branch
   `claude/fervent-noether-5qvel0` with unit tests; before it merges, replay
   capture-447's next step on the branch, as `AGENTS.md` requires.
2. **Score progress, not only success.** Add a 0-to-1 progress score per
   episode: reached, touched, grasped, lifted, carried, placed. Compute it
   first offline from retained traces with
   `task_evaluation_policy_canary_rescore.py`, at no GPU cost, starting with
   capture-447's 39 episodes. If it ranks the four policies, publish it beside
   the counts. Missing today: the final distance to the target, and sealed
   thresholds for each stage.
3. **Run a known-answer control in every scene.** A scripted pick and place on
   the same arm and stance, using the simulator's own positions, scored by the
   same contract. If it fails, the scene is at fault and no policy is blamed.
   The WebApp already reads `known_good_control_*` flags; nothing here writes
   them. It needs a scripted row for the DROID profile and a flag writer.
4. **Check the scale against something measured.** Read the site's entered
   distance or a reference sheet from the WebApp's measurement catalog and
   compare it with the estimated scale. Disagreement is flagged on the result;
   agreement lets the result say the scale was checked.
5. **Make a box a box.** For cartons, totes and trays, build the object from
   its estimated dimensions instead of asking a language model to author it.
   That removes most of the $17 for the commonest objects.
6. **Cut the two hours.** Ten 24-second episodes should not take two hours.
   Measure where the time goes (image pull, model load, start-up, rendering)
   and remove the largest part: a warm image, and all policies for one scene on
   one GPU session.
7. **More, and more varied, episodes.** Twenty to thirty per policy needs two
   shards under today's four-hour limit. Do this after step 6, so it stays
   inside the per-candidate cap. Raising any cap is the owner's decision.
8. **Adopt the published reconstruction where a scan has true scale.** For
   captures with a scale board or phone depth, reconstruct with the published
   method (structure from motion, surface splats, a fused mesh, USD) instead
   of the current eight paid sub-steps. This is the largest change and comes
   last, when steps 1–7 show the lane ranks policies.

## How we will know it is right

- **A free check first.** Public real-robot rankings exist for these same
  policies (the RoboArena network). Run the lane on a comparable tabletop task
  and compare its ranking with the public one. One study did this and matched
  all six pairs of four policies (arXiv 2604.09860).
- **Then paired trials.** Arm Decision Proof exists to compare a sealed
  simulated decision with physical trials. One DROID arm and one task would
  give Blueprint its own number. It has no partner yet.
- **Always, a running record.** Every result states the published accuracy,
  its conditions, and which of them held for that job. No page claims more.

## What this lets us stop

Not before the lane has ranked policies on three jobs.

| Area | Files | Lines |
|---|---|---|
| `task_evaluation_scene_configuration_*` (object authoring and scene set-up) | 84 | 46,215 |
| `atlas_*` and `website_atlas_*` (reconstruction sub-steps, levelling, scale review) | 61 | 23,107 |
| `task_object_artisan*` | 9 | 1,820 |

These counts are by name and are an upper bound.

## What needs the owner

- A paid run to try steps 1–3 on capture-447 again (about $6 for four policies
  at today's prices), and any change to a spend value.
- Whether to ask sites for a scale reference in the video, or to rely on the
  phone app's depth.
- A real arm for paired trials, or a partner who has one.
- Stopping the humanoid and navigation work while this lane is built. Two
  other sessions are driving those jobs today.
