# Plan: one fast, validated simulation proof lane

Written 2026-11-11. It follows the
[first-principles audit](../audits/2026-11-11-first-principles-audit.md) and
the owner's direction of the same day: simulation is the core proof in the
site's answer, and it must be accurate, cheap and quick. This is a plan.
Step 1 is merged (#3560). Step 2 is being built. The section "What the host
showed on 2026-11-11" records what was measured that day and corrects three
statements this plan first made.

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
| Graded score | A grader exists (`policy_episode_graded_report.py`: contact, moved, lifted, carried, placed, released) but is wired only to the older operator publication. Website screening publishes pass or fail only. Step 2 wires the grader in. | `policy_episode_graded_report.py`; `adp_rigid_task_scoring.py:314`; `task_predicates.py:395-466`; `beta_policy_screening_result.py:87-184` |

Two more facts shape the plan.

- **The one finished job could not have succeeded.** On capture-447 the
  robot was placed where the tray's near rim was in reach, but the grasp point
  at the put-down was 1.086 m away against a 0.990 m reach. The code computes
  that distance and does not use it (`robot_arm_path_clearance.py`,
  `task_contact_beyond`). The run cost about $27 and scored 0 of 39.
- **Each policy takes about two hours and costs $1.40–$1.70 on a GPU** for
  about four minutes of simulated time. The finished job spent $26.63 in all.
  This plan first said a language model authoring the object costs $17; that
  was an older run. The recorded authoring run cost $4.21 of a $12 cap
  (tracker; cost model).

## Steps, in order of value for effort

Each step is useful by itself and is proved before the next one starts.

1. **Place the robot where it can do the job, or hold before spending.**
   Merged as #3560. Not deployed when this was written.
   - The rule: a stance counts as in reach only when the grasp contact is
     within the arm's bound at both the pick-up and the put-down. Stances that
     reach the nominal put-down rank first.
   - It holds only on proof. The job is held before any paid run only when no
     stance can reach the nearest put-down the success contract accepts, using
     a distance that can only under-estimate, for any turn the contract allows.
     An assumed contact, a missing object box or an unknown destination
     orientation is flagged and the job continues.
   - Intake, candidate feasibility, the furniture proposal and the robot-fit
     reports follow the same rule, so none says "fits" for a job the admission
     stop would refuse.
   - `STANCE_SEARCH_VERSION` is 4. A job prepared earlier whose station passes
     is left alone; one whose station fails is searched again once, on CPU.
   - An independent review found five faults in the first version and one in
     the second, and passed the third: no over-estimate in 205,000 sampled
     poses, 710 related tests passing.
   - The documented replay could not run: the job's launch run was moved off
     the host on 2026-11-06, and nothing was restored. Instead the merged
     stance search was run on 2026-11-11 on the job's real retained inputs (the
     team half's task request, object box and scene collider, and the DROID
     profile), offline and read-only. A check first showed the rebuild matches
     the recorded search: 947 clear stances, the same refusal counts, and the
     old choice at the same rank, 54.
   - What the new search does on that scene: it proves 945 of the 947 clear
     stances out of reach and chooses one of the two left. From there the
     pick-up grasp contact is 0.9901 m against a 0.9903 m bound, the task's own
     put-down point is 1.076 m away (beyond), and the nearest put-down the
     success contract accepts is 0.978 m (within). No hold fires. The job
     continues with two flags: `robot_reach_nominal_contact_beyond_bound` and
     `scripted_carry_path_not_found`.
   - This corrects the pull request, which said the recorded job would be
     held. That was true of the recorded station, not of a fresh search.
   - What it means: the rule holds only on proof, so a scene the arm can
     barely reach still goes to a paid run it will almost certainly fail. See
     "Two rules this suggests" below.
   - The proof rests on the task's sealed grasp contact: a policy that holds
     the object somewhere else is not modelled.
2. **Score progress, not only success.** In progress on 2026-11-11.
   - The existing grader gives each episode ordered stages; a stage counts
     only when every earlier one did. Website screening will publish, per
     policy, how many episodes reached each stage, plus the closest the grasp
     point came to the object (publication schema v4, optional, with a
     downgrade to v3). The WebApp says how far each policy got when none
     completed the job. No winner is named on progress.
   - Checked first on real data, at no cost. The grader was run on the 39
     retained episodes of capture-447:

     | Policy | Graded | Touched | Moved | Lifted | Closest approach, median |
     |---|---|---|---|---|---|
     | pi05_droid | 9 | 1 | 0 | 0 | 4.3 cm |
     | groot_n17_droid | 10 | 2 | 1 | 0 | 6.5 cm |
     | molmoact2_droid | 10 | 0 | 0 | 0 | 8.3 cm |
     | pi0_droid_jointpos_100k_polaris | 10 | 0 | 0 | 0 | 20.0 cm |

     The stages separate two of the four policies. The closest approach
     separates all four, so it is published too. The recorded scores of that
     job carry no closest-approach measurement; it was recomputed from the
     recorded traces with the scorer's own function.
   - Thresholds are fixed by a published grader version. Nothing is tuned per
     job.
3. **Run a known-answer control in every scene.** Partly built by another
   change (#3556): `beta_policy_screening_known_good_control.py` plans a
   scripted control row and writes the `known_good_control_*` flags the WebApp
   reads. This plan first said nothing writes them; that was wrong. For the
   DROID arm it can only write `known_good_control_unavailable:embodiment`,
   because no scripted row targets the DROID profile. What is missing is a
   scripted runtime for that profile and a GPU run to show it succeeds. The
   older native control gate stays paused (owner, 2026-09-21).
4. **Check the scale against something measured.** Waiting on an input. The
   site's own measurements do not reach the website path (the WebApp keeps
   them out of the task context on purpose, so its digest does not move), and
   since 2026-11-11 the product no longer asks a site to measure anything.
   This step needs a scale reference in the video first.
5. **Make a box a box.** Smaller than first thought. A primitive path
   already exists (`website_synthetic_subject.py`, authored by the no-spend
   `provided_mesh_rigid_authoring` adapter) and is used today only for a
   designed object. Routing an observed carton or tote to it would save the
   authoring run, which cost $4.21 on the recorded job, not $17. A solid box is
   not a tray: an open container needs walls. Needs a GPU run to prove.
6. **Cut the two hours.** Measured on the four recorded runs (2026-11-11):
   - One GPU instance per policy. From launch to settled: 101 to 160 minutes
     (from the provider phase log and the screening ledger, read through the
     operator door).
   - From instance creation to the first episode: 11 to 19 minutes (same
     source). No timestamps are recorded inside that span (image pull, model
     load, environment start).
   - 39 timed episodes (the fortieth was blocked and has no timing). One
     24-second episode (360 control steps at 15 Hz, then 16 settle steps)
     takes 4.3 to 8.5 minutes of wall time; the median per policy is 258 to
     489 s.
   - Instrumented, per episode, as the range of the four policies' means:
     reset and initial state 47 to 89 s, environment step including render
     46 to 70 s, saving media 36 to 54 s, policy inference 9 to 14 s (one
     policy: 62 s), object state sampling 2.7 to 5.5 s, settle steps 2 to 3 s,
     everything else under 1 s. Together 144 to 281 s.
   - Those items explain 53 to 58% of a policy's mean episode wall time; the
     other 42 to 47% (41 to 50% for single episodes) falls between the
     instrumented intervals and is not explained by the record.
   - The runtime's coarser `env_step_s` reads 87 to 133 s. For every policy it
     equals, within 0.2 s, the step, settle, media, state-sampling and
     observation-read buckets added together; the 46 to 70 s above is the step
     alone.
   - Source: each episode's `episode.performance_diagnostics.timings_seconds`
     (`time.monotonic` buckets), `episode.evaluation_backend.timing.env_step_s`
     and the row's `telemetry.wall_time_ns`, in the four retained
     `native_task_arena_policy_canary_session_result.v1.json` files.
   - So the first change is to record where the unexplained time goes. Then:
     one GPU session for all policies of a scene, a warm image, and cheaper
     resets and media. `native_task_arena_policy_canary_worker.py` is two lines
     under its size cap, so timing has to be added from outside it.
7. **More, and more varied, episodes.** Twenty to thirty per policy needs two
   shards under today's four-hour limit. Do this after step 6, so it stays
   inside the per-candidate cap. Raising any cap is the owner's decision.
8. **Adopt the published reconstruction where a scan has true scale.** For
   captures with a scale board or phone depth, reconstruct with the published
   method (structure from motion, surface splats, a fused mesh, USD) instead
   of the current eight paid sub-steps. This is the largest change and comes
   last, when steps 1–7 show the lane ranks policies.

## Two rules this suggests

Not built. Each changes what is paid for, so each is the owner's call.

- **When the scene is flagged as probably out of reach, pay for one policy
  first.** If no stance reaches the task's put-down point and the scripted
  carry finds no path, run one policy, read its progress, and run the others
  only if it touched the object. That saves the other runs only when the first
  policy did not touch the object, and on the recorded job two of the four
  policies touched it.
- **When the job's two spots are further apart than the research arm can
  span, test the skill within its reach.** Reach is the robot's question, and
  the fit check and the scene check answer it for the site's pick. What the
  policy comparison can add is whether a policy can pick this object up from
  this surface and put it down again. Place the arm where it reaches the
  pick-up, set the put-down at a distance it can reach on the same surface,
  and say so on the page.

## What the host showed on 2026-11-11

Read through the operator door, read-only.

- The host ran `a0a6360` (#3558). Three merged commits were not live,
  #3560 among them. Nothing was in flight and no paid lock was held.
- Month to date: $146.28 of the $200 monthly limit. One full job costs about
  $27, so there is room for one, not two.
- Disk was at the critical level: `/mnt/blueprint-work` 92% used, below its
  free-space floor, with reclaim ineffective.
- The recorded job's launch run had been moved to cold storage, so the
  documented downstream replay cannot run on it without a restore.

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

- A paid run on a scene the research arm can comfortably reach (about $6 for
  four policies at today's prices, plus about $20 of preparation for a new
  scene), and any change to a spend value. The recorded scene is at the edge
  of the arm's reach, so it is a poor test of whether policies can be ranked.
- Whether to ask sites for a scale reference in the video. The phone capture
  app is no longer in use (owner, 2026-11-11), so its depth data is not
  available; the site films on its own phone in the browser.
- A real arm for paired trials, or a partner who has one.
- Stopping the humanoid and navigation work while this lane is built. Two
  other sessions are driving those jobs today.

## Scripted reference check on a fitting body

**What the code change does.** Each new website campaign already plans one
scripted reference row (`beta_policy_screening_known_good_control`):
Blueprint's own pick-and-place script, three nominal tries, reference only,
never a candidate or part of the estimate, never a gate. It checks only
Blueprint's own simulation: whether this scene, task and scorer can register a
success at all. It is not a test of any robot, the one the website names
included, and is never described as one. Until now it ran only
on the body of a selected learned candidate, and every learned candidate a rigid
job selects today is on the DROID Franka, which has no scripted body, so the row always said
`known_good_control_unavailable:embodiment`. Now, when no selected candidate's
robot has a scripted body, the row takes a registered scripted body for a model
the job's own robot fit table says fits (`fits` or `fits_spec`; `unknown` only
if none of those has a body; never `doesnt_fit`). Within that, the body nearest
the candidates' own robot comes first: for the DROID Franka, the Franka FR3
with the same Robotiq 2F-85 gripper (a different asset from the DROID arm, the
nearest one registered). Then the website's picks, if it sends them (step 2),
then the table's order. The row records the spec-catalog model
(`for_model_id`), which rule chose it (`chosen_by`) and, when the body stands at
the scene's robot station rather than its own, the code `scene_packet_station`.

**Money: it runs last, on what is left over.** It is planned when its cap fits
what the selected candidates leave of the plan budget and the site's allowance.
At run time nothing that funds, waitlists, refills or closes a learned
candidate counts it while it is unreserved, so every learned row ends exactly
as it would without it (a test runs the same queue with and without the row
and compares every learned row's end and money). It is reserved only after
every learned row of its campaign has ended, through the ordinary reservation,
so the plan budget, the site limit, the job ceiling and the account limit
bound it. If its cap does not fit then, it ends at once as
`known_good_control_unavailable:budget`; it never waits on money, so the
campaign always completes. Once reserved, its money counts against the site
limit like any row's. The publication carries the check's outcome flag and its
`episodes_succeeded` of `episodes_scored` on the row `blueprint_scripted_<body>`.

**It is not inert.** The host registry (`/etc/blueprint/team-evaluation-registry.json`,
unchanged since 2026-10-31) registers all 24 scripted bodies, and the effective
catalog of the newest campaign (2026-11-10) lists every one enabled, the UR10e
(`universal_robots_ur10e_robotiq_2f85_sim`) among them, with its model file
present. So the first new rigid job after a deploy can plan this row.

**Steps to the first real check (by default on the FR3 body).**

1. *Host operation:* merge, then deploy in the daily window (D-08). Nothing to
   prepare: the FR3 body (`franka_research_3_robotiq_2f85_sim`) is registered,
   enabled and its model is on the host, as is every other scripted body.
2. *WebApp change (optional):* the check's body can follow the robots the
   website names, which only picks which fitting body the simulation check
   runs on. The WebApp's
   non-creating Atlas configuration read (the one that already returns
   `job_ceiling` to screening planning) may add `preferred_robot_model_ids`: up
   to 3 distinct spec-catalog model ids, the answer's pick first, then its
   runner-up. Among fitting models (`fits` and `fits_spec` together) a
   preferred one goes first; it never beats a fitting model with an `unknown`
   one and never picks `doesnt_fit`. The marker records `preferred: true|false`.
   An unusable value is ignored and logged. The body nearest the candidates'
   robot (the FR3 today) still comes before the website's picks. Without
   either, the fit table's order applies.
3. *Host operation (none to start):* a new rigid pick-and-place website job.
   Its scene handoff must carry a robot fit table bound to the job; capture-447's
   handoff has none, so a re-plan of that job gets no fallback body.
4. *Money:* the scripted body is quoted at the per-candidate cap, $4.00
   (`DEFAULT_CANDIDATE_CAP_USD`; $0.80/h for 4 h, $0.80 of headroom). The
   reference row lowers that to its own cap, $3.00 (`REFERENCE_CAP_USD`, owner
   approval 2026-11-11), by the plan's existing rule for fitting a quote to a
   smaller amount: it keeps at least half the original headroom, $0.40, so its
   hard time limit is 3.25 h ($0.80/h x 3.25 h = $2.60). That fits the $3.00 a
   typical rigid job leaves after three learned policies at $4.00 of the $15
   plan. No other cap, budget or limit changed. On 2026-11-11 the month stood
   at $147.16 of $200, forecast to reach the limit on 2026-11-15.
5. What one scripted run costs and how long it takes has never been measured.
   Learned runs measured 11–19 min to the first episode and 4.3–8.5 min per
   episode; three tries at those rates would take about 25–45 min of the
   3.25 h limit, if a scripted run behaves like a learned one.

Not needed for the registered bodies, but for any new one: an import needs 24 GiB free on
`/mnt/blueprint-work` above its floor; on 2026-11-11 it had 8.2 GiB, so new
imports are refused until about 16 GiB is freed (host operation). The stuck
grant `public-robot-imports-20261030-01` spent nothing, expired on 2026-10-31 and is
re-read every pass because the queue only clears completed grants; it blocks nothing here.
