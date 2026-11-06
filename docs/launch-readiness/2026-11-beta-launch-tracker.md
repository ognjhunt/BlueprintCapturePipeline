# Beta launch tracker (November 2026)

Goal: make Blueprint ready for its invited beta, then its product launch. Every
item keeps the ID from the 2026-11-02 goal brief. Times are UTC unless marked.

Status: **done** (verified, with evidence) · **in progress** · **owner** (waits
on an owner decision or action) · **next deploy** (merged, ships with the next
batched Pipeline deploy) · **not started**.

Last updated: 2026-11-06 15:00Z (library-aisle run section). Earlier header as of 2026-11-03 14:49Z: Pipeline live: `333b1f74a` (11-03 14:29Z). Door: `2ace7c2d6` (11-03 03:49Z). WebApp live: `58677f751` (11-03 04:03Z, verified at the exact SHA), which includes #1203. Next Pipeline deploy: see the checklist below.

## Library-aisle navigation run (2026-11-06)

The next production job, driven by a cloud session: a navigation task, chosen as the forcing case for "any job" (owner direction 2026-11-06). Its 14-step evidence matrix and the task-kind gate inventory live in WebApp `docs/library-aisle-navigation-e2e.md` on the session branch.

- **Clip and rights.** Pexels 854533, "Video Inside A Library", uploaded by Pixabay and dated 2016-10-30.
  - The 4K rendition measured 49,023,833 bytes, sha256 `3f1c177f586ca2b7b29a4c7e1e7bfca52afe159e4aac270408dd5c405c76e1c0`, 17.49 s, 3840×2160, 25 fps, 437 frames.
  - The page's license line reads "License: Free", linked to Pexels' Creative Commons page. It does not print "CC0".
  - Pexels terms §4 say CC0 items are marked "CC0" on the page, and §5 says other items are marked "Pexels License"; this page shows neither marker.
  - Terms §8 forbid data extraction "for all unauthorised purposes, including … machine learning".
  - The owner approved the clip with that evidence. The owner also believes capture-447's clip came from Pexels; if it was under the standard Pexels License, the same §8 question applies to it. Noted, not acted on.
- **Money, owner approval before 14:44Z ("yes i approve everything").** Hard ceiling $115, expected $15–35:
  - Atlas $40. The job's automatic grant is $25 / 32 requests.
  - Upstream $5.
  - Scene $40. Intake set $20; it is raised only when a stage needs it.
  - Preparation attempts 16 → 32, one-shot. Applied at 14:53Z, amendment `sha256:99ab7973…`.
  - Screening $30, the existing site limit.
  - Cloud Run $0.
  - A $30 Vast top-up is the owner's action and is still pending; credit is $7.07 against the $11 floor.
  - Standing approvals reconfirmed: merge when CI is green, deploy within D-08, GPU screening runs up to $4 each, site limit $30.
- **Intake.** Public site, ops email, 14:44–14:51:48Z. Request `capture-5b067498-c8b3-4549-9da9-ded4c391db1f`, capture `walkthrough-capture-5b067498…`, state `processing_ready`, eligibility `unscreened`. The Atlas preview was queued at 14:44:36Z.
  - The site country was entered as "United States". The clip's real filming location is unknown; the form requires a country.
- **Expected stop on today's code.** From reading the code, not from a run: the job buys the Atlas planner review, then holds with `website_task_operation_coverage_unresolved` before any World Labs purchase (`website_early_task_hold.hold_after_review`). The hold keeps the job, and a later deploy reopens it without buying the review again.
- **What the live job did on `9dd4e0dc4`** (JSON and journals read through the door). It stopped twice, and neither stop came from the task kind:
  - **Atlas preview, failed.** One worker tick ran 14:45:29Z to about 15:50Z, and the job records its start time.
    - Review 1 cost $0.079 and proposed frames outside the 29-frame survey.
    - The worker then decoded a 153-frame extended survey as 4K PNGs: about 11 MB and about 30 s per frame, with host load 8.4 on 4 CPUs.
    - Review 2 cost $0.141. Review 3 cost $0.120 and selected 7 frames, then cited an eighth (`decoded-000000300`) as evidence.
    - `proposal_review` failed the job with `atlas_automation_frames_not_selected`. Planner spend: $0.339, settled.
    - Every review read the job correctly: `subject_origin: no_manipulated_subject`, one `navigation` step.
  - **Clean plate (qualification lane), stopped for reconciliation.**
    - Gemini's removal plan found 4 targets (left, right and end shelves, and the floor), with 0 to remove and 0 people.
    - Hosted SAM 3.1 runs one request per target:
      - 15:01Z: target 0 answered.
      - 15:06:57Z: its 1.8 MB response timed out in the mask parser (fixed 300 s).
      - 15:29:14Z: target 1 named the first pass's video upload, and the provider answered HTTP 404.
      - 15:42:20Z: `meta_sam_submission_requires_reconciliation`.
  - The host copy's sha256 equals the source (`3f1c177f…`).
- **Fixes, all general.** Each case in this job is the test, not a special case.
  - #3419 (merged 15:46Z): one predicate scorer, the task compiler, and behaviour measurements.
  - WebApp #1226 (merged): no task-kind gate on spend or team evaluation, and an expired owner row raises no ceiling.
  - #3423 (merged 16:46Z as `02668f5`):
    - route by requirement, and compile a locomotion job from its preserved scene;
    - a hold that names each missing piece (`website_task_runnable_policy_missing`);
    - G2 prompt;
    - G7 ceiling re-read before every paid launch;
    - G4 base height;
    - re-dispatch a SAM prompt the provider refused, and scale the parser's time limit;
    - a review-cited frame becomes an input;
    - Codex review fixes: the launch money check counts the owning queue and the whole ceiling, compiled specs dispatch to the predicate scorer, and a forceless named contact is unscored;
    - the owner's Bandit sign-off and the [51] DeepInfra download fix.
  - Replay before merge (AGENTS.md), on `6429be8`, paying nothing. Inputs were the job's JSON records and this session's own download of the source clip; no frames were copied off the host. The failed Atlas job is retried in the overlay and advances from `queued` into `plan`, with `would_buy: []`. It then stops at `missing_input`, the survey's decoded frame `decoded-000000000.png`. The next planner review is a new attempt, because its inputs now include the cited frames.
  - Why capture-447 did not hit either stop, from its retained records:
    - Its two completed Atlas reviews cited only the 2 frames each selected.
    - Its clip was 1080p (a 5.8 MB upload). SAM answered 76 KB and 232 KB, both parsed within 35 s, and its second prompt went out 28 s after the upload.
    - Here the clip is 4K (a 30.8 MB upload). The first answer was 1.8 MB and decoded to 518 MB, with a peak of 4.6 GB of memory. The second prompt went out 28 min after the upload. The 404 record holds only the status; an expired upload is the likeliest cause.
  - It goes live with the batched deploy (D-08), which the G1 session requests once main's CI attests the merge. After it, the job resumes through `door selected-handoff --atlas-retry`, an operator requeue within its existing allowance.
- **Owner decisions, 16:05Z** ("Bandit OK as recommended", "switch G3", "yes retire the 7", "yes standard - rights are okay"):
  - Bandit: applied in #3423 (above).
  - G3: the arm rule becomes "the object's centre within a tolerance" in a new surface-target schema, so earlier results still score as before (next item).
  - Parked deliveries: see below.
  - Rights: clips under the standard Pexels License are cleared for this work. That covers capture-447's clip and the alternates, Pexels 4294436 (warehouse) and 3773483 (kitchen). The §8 note above is settled by this ruling.
- **G3, owner switched it on 11-06 ("switch G3").** The arm placement success rule was "every corner of the object inside the disc", which forced a disc bigger than the object (capture-447: 76 cm across on a 1.3 m table, partly under the tray's start). It is now "the object's centre ends within the disc's radius". Every website region sealed from now on is `task_evaluation_surface_target.v2` (`success_rule: subject_centre_within_radius`):
  - The disc is a 5 cm tolerance. The owner chose the rule, not the 5 cm: that is the harness default (`WEBSITE_TARGET_TOLERANCE_M`).
  - The disc is kept wholly off the start footprint of the largest admitted replacement (`website_target_reach`).
  - That replacement, centred on the disc at any yaw, must also lie on the surface.
  - A container keeps whole-volume containment.
  - Targets sealed before stay v1 and rescore exactly as before.
  - The WebApp must accept v2 before a Pipeline release emits it (WebApp #1228).
- **Second Bandit packet.** #3422, from another session, lists the 36 expired entries without assessing them. This session's packet assesses all 96. They agree on the gate result.
- **Bandit triage (owner).** Review packet: [`2026-11-06-bandit-triage-review.md`](2026-11-06-bandit-triage-review.md).
  - Verdicts across the 96 entries: 77 keep, 18 tighten, 1 fix now (`wam_compute_providers.py` `_deepinfra_download_file`), 0 orphaned, 0 new MEDIUM or HIGH.
  - The latest allowed expiry for an entry signed today is 2027-02-04.
  - Signed off by the owner at 16:05Z and applied in #3423.
- **Parked deliveries (owner, 11-06: "yes retire the 7").** The 7 parked preparation deliveries are owner test captures from 10-28/29 whose status read cannot succeed since WebApp #1113. The door operation `retire-parked-delivery` retires them once it is deployed and the door is upgraded (`upgrade-door`). It is general: any delivery parked for a day can be retired, with the owner's approval on its record. Nothing is deleted, and a newer revision starts a new delivery.

## Deploys on 2026-11-05 (capture-447 screening)

- Two cloud sessions share the `cloud-scene-runs` door token today. Both merge and deploy on capture-447, so check `door requests` and `deploys.recent_receipts` before deploying.
- `2f688aa98` (14:48Z request, live 15:22Z): $30 site allowance, harness retries 6, offer-taken retry (#3395); website scene configuration from its own tick (#3196).
- `c5e03634a` (live about 16:27Z, the other session): screening review videos (#3396). WebApp #1218 had been live since 16:11Z.
- `63fa4d739` (16:48Z request, live 17:01Z, this session). It shipped as a money fix while pi05 was running: without a door hold on the screening timer (`hold_unit_profile_refused`), the next candidate (cosmos3, same OpenPI client) would have launched on the bug within a tick. It carried:
  - the OpenPI websocket `ping_timeout` fix (#3399);
  - withdrawal deletion of screening work (#3397);
  - every run keeps its review videos, plus the backfill (#3398);
  - the Cloud Run team-scene lane, inert until configured (#3331).

  The re-plan froze 63fa4d739 for cosmos3, molmoact2 and pi0_droid_jointpos_100k_polaris. WebApp #1220 (practice-run cache privacy fixes) had been live since 16:30Z.
- `64e25c2cd` (live about 17:25Z, the other session): #3401, harness retry limit 7.
- `3aa16eda8` (21:16Z request, live 21:37Z, this session). Deployed as a money fix while pi0_droid_jointpos_100k_polaris was running (it closed at 21:27Z); cosmos3 and molmoact2 were waiting for GPU capacity with no machine allocated. It carried:
  - #3404: DROID policies on the RoboLab path had been fed their gripper state inverted. The adapter measured Robotiq inner-finger body origins (0 m open, 0.083 m closed) against the probe's pad-centre endpoints, so open read as closed. Rigid release was scored the same way. The fix binds the servo's pad readback. Plans now carry `screening_harness_revision` 1, and robolab.v1 results under revision 0 are not counted: each such candidate is planned again once, inside the site's allowance (owner approval 2026-11-05).
  - #3403: a simulator start-up crash on the rented machine is a machine fault, never a harness retry.
  - #3402: this tracker's earlier entry.

  capture-447 results made before it are not counted: pi05 9 scored / 0 succeeded, groot 10 / 0 (two runs moved the tray), pi0_droid_jointpos_100k_polaris 10 / 0. Site spend at deploy: $16.18 of $30.
- Merged after it, not yet deployed: #3405 (the same pad readback in the controls gate, controls worker, policy worker and `controlled_native_isaac`; no beta screening path).
- After `3aa16eda8` went live, cosmos3 launched at 21:42Z on a bundle it had staged at 21:34Z, on `64e25c2cd`, while the deploy was building. Its plan predates revision 1, so its result does not count; it failed to load the policy ($0.60). #3408 closes that gap. The revision-1 re-plan (`screening-acf7fbcd…`, 22:16Z) ran on the fixed gripper:
  - pi05: 9 scored, 0 succeeded, one void episode (joint command out of bounds); $1.39.
  - groot: 10 scored, 0 succeeded; $1.51.
  - The run data shows the pad readback working: 0.085 m open, 0 m closed. In 18 of 19 episodes the tray never moved and had no robot contact; the gripper closes over or near the tray's rim, typically about 7 cm above it (measured 11-06, below). Every run carries `droid_wrist_task_pixels_below_threshold`.
  - cosmos3 failed in provisioning, before any policy ran: the cuDNN wheel download hit the total time limit on its machine ($0.54). As a harness failure, it held molmoact2 and the three other pi0 variants (`beta_screening_held_after_launch_failure`).

  Site spend: $20.22 of $30.
- `9dd4e0dc4` (06 03:06Z request, live 03:19:56Z, this session). Nothing was preparing, launching or running at any site, and the last deploy was `3aa16eda8`. It releases the held candidates, which are planned again at the new commit. It carried:
  - #3405;
  - #3408: preparation compares the running commit before it reuses a staged bundle, so a bundle staged during a deploy's build never launches the old code after the switch;
  - #3406: this tracker's previous entry.

  The Atlas $130 row expires 11-06 10:16Z; every paid screening run must start before then.
- The re-plan (`screening-050fe4a7…`, 03:27Z) launched cosmos3 alone at 03:29Z. Its run failed at 03:55Z with `policy_canary_session_setup_failed:policy_load` ($1.45). That is the same check as its 21:42Z run: `cosmos_droid_client_dependencies.py:57`, `cosmos_policy_client_dependency_precedence_invalid`. cosmos3 now has 8 harness failures at this site, past `HARNESS_RETRY_LIMIT` (7), so it is not planned again. Its failure held molmoact2 and the three pi0 variants. Site spend: $21.67 of $30.
- `5e826deba` (06 03:57Z request): **blocked**, `deploy_scene_retirement_runtime_unproven` (`ci_finished_without_attestation`). main's CI for it failed the Bandit gate. 36 entries in `docs/bandit_triage.json` expired on 11-05 (`triage_expired`), and 60 more expire 11-12; their owner is @ognjhunt. **owner**: until they are re-reviewed, no new main commit gets a source attestation, so none can deploy. 9dd4e0dc's CI passed at 11-05 21:59Z, before the expiry.
- `304e8aa97` (06 04:05Z request, live 04:18:11Z, this session). This is the last main commit with an attestation that is not live. Nothing was preparing, launching or running at any site, and the last deploy was this session's. The changed commit re-plans the four held candidates without cosmos3. It rolls back #3408 only, which guards bundles staged during a deploy's build; none was staged. #3408 returns with the next attested deploy.
- The re-plan (`screening-b67f815a…`, 04:24Z) launched molmoact2 at 04:26Z. It ran all ten episodes, and its 3.36 GB output was promoted and verified in B2. Streaming the needed members back then stopped after 146 of them on one `provider_output_http_failed`, so the run sealed blocked (`beta_screening_native_result_missing_or_invalid`, $1.90; settled 07:09Z). It counts as molmoact2's second harness failure and held the two pi0 variants. Nothing retried that stream: #3411 (merged, not deployed) now runs it again on transient transport codes, up to three passes. The five score receipts that arrived (episodes 0–4) are not counted; all five read `never_moved`, with no contact and the gripper never closed. The 60 review videos were retained. Site spend: $23.57 of $30.
- `9dd4e0dc4` again (06 07:14Z request, live 07:28:48Z, this session). main's head still has no attestation (Bandit triage, above). Nothing was preparing, launching or running at any site, and the last deploy was this session's 304e8aa at 04:18Z. The changed commit re-plans the held candidates and restores #3408.
- The re-plan (`screening-f3d50fae…`, 07:35:38Z) launched molmoact2's harness retry first, at 07:37:54Z ($4 reserved; $27.57 of $30 committed). The two pi0 variants wait for its settlement, and pi0_fast is waitlisted on budget. The run should settle around 10:20Z, after the Atlas row expires at 10:16Z. The owner does not want that row extended. Nothing in the coordinator checks it: the pi0 quotes stay valid until 11-07 07:34Z. **owner**: if a candidate can still launch at 10:14Z, this session holds `blueprint-beta-policy-screening.timer` through the door (at most 24 h). That pauses the trigger, not the running job, so no paid run starts after the cutoff. Releasing the hold settles molmoact2 and lets the next pi0 launch.
- At 10:14Z the door refused that hold with `hold_unit_profile_refused`: this door profile holds only `blueprint-agent-run-dispatcher.timer`. Nothing else was tried. The owner was notified at 10:14Z.
- molmoact2's retry completed at 10:17Z: 10 scored, 0 succeeded ($1.69; GPU torn down 09:44Z). In all ten episodes the tray never moved and had no robot contact. The gripper stayed open in seven; in the other three it closed 4–10 cm above the tray's rim. Flags: `droid_wrist_task_pixels_below_threshold`, `policy_canary_task_semantic_visibility_failed`, and composition/occlusion review on the external and overview cameras. Site spend: $25.26 of $30.
- pi0_droid_jointpos_100k_polaris launched at 10:23:01Z ($4 cap, attempt `screen-f7511ceb…`), seven minutes after the Atlas row expired. **owner**: it started after the cutoff the owner set, and this session could not stop it. It is the last run the $30 site limit admits: after it settles, $4 more would exceed $30, so pi0_droid_jointpos_polaris and pi0_fast_droid_jointpos_polaris stay waitlisted.
- That first attempt found no GPU offer and closed at 10:32Z with no machine ($0). The second (`screen-ace7e697…`) launched at 10:39Z and ran all ten episodes on one Vast GPU, torn down at 12:36Z. It settled at 12:58Z: 10 scored, 0 succeeded ($1.37). In all ten episodes the tray never moved and had no robot contact. The gripper closed away from the tray, a median 27 cm from it. The flags are molmoact2's. 60 review videos were retained.
- capture-447's screening is finished. The website acknowledged the final publication (`screening-f3d50fae…`, revision 18) at 12:57Z, and its outcome is `estimate`: molmoact2 and pi0_droid_jointpos_100k_polaris each scored 0 of 10, likely between 0% and 28% (Wilson 95%). pi05 (0 of 9, 0–30%) and groot (0 of 10) were scored in the earlier run on the same fixed gripper. cosmos3 has no score: it hit 8 harness failures on `policy_load`, at `cosmos_droid_client_dependencies.py:57`. Across the 39 counted episodes the tray moved in one: groot pushed it 0.18 m without a grasp (`moved_below_success_contract`). pi0_droid_jointpos_polaris and pi0_fast_droid_jointpos_polaris are deferred with `beta_screening_site_budget_exhausted`. Site spend is $26.63 of $30 with nothing reserved, and the $3.37 left is less than one $4 run, so nothing else starts at this site.
- #3416 (merged 13:54Z, not deployed) fixes the cosmos3 `policy_load` check. Isaac Kit's fast importer resolved Kit's own older websockets ahead of `sys.path`, so the pinned 16.0 client lost inside the Kit worker; the same install passed in the bare `python.sh` provisioning process. A finder at `sys.meta_path[0]` is now the only source of the pinned names, for Cosmos and OpenWAM. cosmos3 is out of retries at capture-447, so a re-screen there is an owner money decision.
- `a5b47bd11` (06 15:00:59Z request, the 15:00Z window, this session): **blocked** at 15:08:34Z, `deploy_scene_retirement_runtime_unproven` (`ci_finished_without_attestation`). Nothing was paused (`provider_mutation_performed: false`); `9dd4e0dc4` stays live. Before it, nothing held a paid-launch lock, no activation hold was set, nothing was launching or running, and no other deploy was pending. It would have carried #3407, #3409–#3418, and #3411 and #3416 among them. main's CI for `3d1cd4dd2` and `a5b47bd11` fails only the Bandit gate (its "Scan and enforce expiring triage" step), so the provenance job that attests is skipped. On `3d1cd4dd2` that gate lists 36 blockers, all `triage_expired` (high 0, medium 96), and no new finding; the 36 expired entries are B108 ×18, B103 ×11, B102 ×4, B104 ×2 and B608 ×1. `9dd4e0dc4` is still the newest attested commit, so `--latest-attested` has nothing newer to ship.
- Open after capture-447:
  - **owner**: Bandit triage re-review: the 36 entries in `docs/bandit_triage.json` that expired 11-05 (60 more expire 11-12), listed for review in [`../audits/2026-11-06-bandit-triage-expired.md`](../audits/2026-11-06-bandit-triage-expired.md). Until it is done, no main commit gets an attestation, so #3411 and #3416 cannot deploy.
- Wrist-camera study (11-06, from the 39 counted episodes; JSON read through the door only). The wrist camera is not the main reason the policies fail:
  - **At reset**:
    - The robot base sits at table height with the tray centre 0.92 m straight ahead; the home grasp point is 0.36 m ahead and 0.34 m up.
    - From there, the wrist camera sees the tray only in the bottom strip of its image, behind the gripper's own fingers. It saw 0, 86 or 94 tray pixels against a minimum of 120, except at start position 02 (1,943 px, centred).
    - At the policy's 224 px input, 94 px is about 7 × 0.5 px.
    - The external camera sees the tray at every start (16,390–18,056 px).
  - **During the rollout**: every policy moved the arm toward the tray, ending its closest approach 0.21–0.48 m nearer than at reset. Projecting the tray's measured box through each recorded wrist-camera pose puts at least 120 px of it outside the gripper's region in 22–44 of 45 policy steps per episode.
    - This is geometry, not a rendered count: per-frame segmentation is not recorded.
  - **The failure is the grasp**:
    - Of their gripper closes, groot made 13 of 14 over the tray (11 within 4 cm of its rim), pi05 9 of 58 and molmoact2 2 of 3. They close a median 7 cm above the rim (2.3–14.5 cm), on air.
    - groot's 0.18 m push (episode 02) came from a close about 1 cm from the tray.
    - molmoact2 kept the gripper open in 7 of 10 episodes.
    - pi0_droid_jointpos_100k_polaris went to the tray once: in episode 02 its grasp point closed 2.4 cm above the tray's box, over its footprint (step 264). In the other nine its nearest approach was 13–31 cm, and its 33 closes were a median 27 cm from the tray.
    - pi0's and groot's closest approaches came at start position 02, the one start where the wrist camera sees the tray. pi05's and molmoact2's did not. That is one start, not a measured effect.
  - **Correction**: the score's `gripper_closed` is evaluated only over the final settle window, and only for tasks that must end closed, so it is always false here. Earlier entries that read it as "the gripper never closed" are corrected above. The 5 uncounted molmoact2 episodes of 04:26Z did keep the gripper fully open (measured).
  - **Also seen, not measured as causes**:
    - The prompt the policies get is "Pick up and place the wooden tray on the dining table on the wooden dining table."
    - The destination marker is a policy-visible flat green disc of 0.39 m radius. Its edge comes within 4–7 cm of the tray's centre, and 31–38% of the tray's footprint sits over it.
    - The robot base is 1.3 cm above the surface the tray rests on.
  - **Reach** (second pass, 11-06; JSON only). The place is likely beyond the arm, and the pipeline flagged it before any paid run:
    - The scene's CPU check `native_task_robot_arm_path_clearance.v1.json` is `flagged` in all four counted attempts (plan `ccebae51…`): `scripted_carry_path_not_found`, advisory `robot_rigid_pick_place_unreachable:carry`. The scripted pick-and-place found no path past sample 15 of 42.
    - At the place, the task's grasp contact is 1.086 m from the shoulder, beyond the arm's reach bound of 0.990 m. At the pick it is 0.970 m.
    - Some point of the tray is inside the bound at both ends (0.735 m at the pick, 0.729 m at the place), so a policy holding the near rim is not ruled out. The bound is a triangle-inequality maximum, so being inside it does not prove a point is reachable.
    - That flag is not in the screening ledger or any run summary, so it was not visible beside the results.
    - Measured in the runs: the destination marker's centre is 1.04 m from the robot base, horizontally, and the nearest point of the 10 cm destination box is 0.97 m. No policy took its grasp point more than 0.81 m from the base in any of the 39 episodes.
    - The gripper opens to 0.104 m between its pads, and the tray's narrowest side is 0.313 m, so the tray can only be held by its rim.
  - **Placement numbers**, for the check below:
    - At the DROID home pose, the centre of the wrist image looks at the table 0.43 m ahead of the base.
    - A tray-top point 0.45 m ahead appears at row 391 of 720, above the gripper's own pixels, which start at row 432. At 0.55 m it appears at row 518. Beyond about 0.75 m it is below the frame.
    - The pick and the place are 0.45 m apart. A stance 0.5 m from the midpoint of that line, facing it, would put both 0.55 m from the base, if the table allows it.
  - **Next**, before any new paid screening:
    - Check the scene planner's robot placement, so the tray starts nearer the middle of the policy's view and the place is within reach. Run the free CPU arm-path check on the new plan and record its result beside the run. A `flagged` result is evidence, not a stop: the code keeps it advisory (`rigid_carry_verdict`), because a policy may hold the tray in a way the scripted carry did not try.
    - Check the prompt text.
    - Check the marker size.
    - Carry the scene's arm-path flags into the screening run summary, beside `observation_quality_flags`, as a flag and not a gate.

## G1 humanoid on website screening (audit, 2026-11-06)

This session owns the G1 rows (branch `claude/modest-hopper-102z18` in both repos). It read the code at `7f8728b` and read the host through the door between 14:26Z and 14:50Z. Nothing was launched, registered or paid.

**Where it stands.** The four HumanoidArena G1 rows have never run on the website path. In all 24 campaigns in the ledger (all capture-447), every G1 row is `unavailable` with `g1_beta_scene_packet_missing` and `runtime_candidate_binding_missing`. Both codes have one cause: the site has no G1 registration. The first is raised at `beta_policy_screening_capabilities.py:292-293`; with no binding, the plan then adds the second (`beta_policy_screening_plan.py:469-470`).

**What is built** (code and CPU tests only):
- planning;
- the automatic packet recipe;
- the one-shot paid session;
- the independent watchdog and scoped closeout;
- the controller's re-score;
- the `learned_policy` summary.

`test_beta_policy_screening_g1.py`, `test_beta_policy_screening_g1_prepare.py` and `test_operator_door_g1_registration.py` give 69 passed and 1 skipped (the skip needs an unprivileged fixture owner).

**What stops it, in order:**
1. **Nothing registers a website site.**
   - The only writer of `site_screening_candidates` is the door installer (`deploy/operator-door/operator_door/native_g1_registration.py:67-77`), and it needs a person's `authorization_reference`.
   - The natural automatic hook is `team_scene_packet.record_team_scene` (`team_scene_packet.py:1268-1291`). It runs as `blueprint`, which cannot write the root-owned `/etc/blueprint/team-evaluation-registry.json`.
2. **New: the shared G1 runtime source is gone from the host.**
   - The shared registry (`g1-team-campaign-registry.json`, `sha256:16fa2ace…`) points at `/mnt/blueprint-work/g1-cuda-runtime-source-7c1f528-20260926/native_task_runtime_source_packet.v1.json`, and the door returns `not_found` for that directory.
   - Only its build record remains: `g1-cuda-runtime-source-7c1f528-20260926-build.json`, a 4,638,507,022-byte packet with sha `8128c868…`.
   - The two G1 retirement records (09-27 and 10-23) do not list it, so how it was removed is unknown.
   - Until it is rebuilt or restored, every registration ends `g1_shared_source_verification_failed`, the manual door install included (`native_g1_registration_contract.py:449-466`).
   - The publisher source (receipt `4170b783…`) and the 38,195,671-byte robot USD are still present.
3. **No money check for G1 at planning.**
   - `candidate_gpu_blockers` skips G1 (`beta_policy_screening_gpu.py:110,112`), and no candidate gets a rate check before launch.
   - G1 takes the profile's shared cap, rate and TTL. The site overlay accepts only `native_g1` per row, so there is no G1-specific setting.
4. **Rights:** see the owner decisions below.
5. **Only rigid pick-and-place can reach G1.**
   - Mobile and navigation jobs are held before the scene build (`website_early_task_hold.py:94-114`).
   - On the website, mobile wording is more than a hint: `websiteSceneSponsorship.ts:240` gives such a job preparation spend but no execution grant (`website_scene_site_not_qualified`).
6. **A scored G1 row would be ranked.**
   - `website_candidate_selection.py:82-84` does not look at `ranking_eligible`, so a completed G1 row could be shown as chosen in simulation.
   - The WebApp's `choice_blocker` enum (`betaCandidateSelection.ts:38`) has no development-only value.
   - The task page shows no development-only boundary on any row.
7. **One generic error after the run.** `beta_policy_screening_launch.py:568-575` turns every `g1_beta_result_*` reason into `g1_beta_native_result_missing_or_invalid`.

**Corrections to the 11-06 map.** The rest of it checked out.
- The fit precheck maps `articulated_open_close` too (`website_robot_fit_precheck.py:68`).
- The site overlay compares 4 of the 6 source fields (`team_candidate_registry_reader.py:116-119`); `capabilities.py:287` checks all six.
- The registration contract and installer live in `deploy/operator-door/operator_door/`, not `src/`. A change to them needs a door upgrade.
- In `beta_policy_screening_g1_launch.py`, the gpu-canary arguments are at lines 64-68 and `join_result` at 78-107.
- The launch credit check needs the cap plus $1 (`provider_credit_admission.py:87`), which is $5 for a $4 cap. The $7.07 balance admits one G1 launch; $11 is the page line, not a launch floor.
- At planning the G1 quote is just the cap.
  - The 200 GB disk and the transfer quote are set when the bundle is prepared.
  - When an offer is chosen, compute plus 200 GB of storage must fit the hourly rate, and rate × minutes plus transfer must fit the cap (`vast_provider_adapter.py:1553-1555, 1816-1819`).
- 841757 is fixed in three places on the beta path:
  - the model-terms review receipt (`beta_policy_screening_g1.py:219`, `native_g1_beta_authority.py:38-43, 80`);
  - the authorization basis string (`native_g1_beta_authority.py:60`);
  - the tokenizer fallback folder (`native_g1_beta_tokenizer.py:19`).
- The ADP team campaigns have 11 dispatch records under `/mnt/blueprint-work/pipeline-control-plane/native-g1-team-campaign-work/`. None completed. Besides the blockers already listed they include `vast_cuda_runtime_host_image_incompatible` and `vast_paid_launch_lock_busy`.
- `g1_provider_manipulation_pair_blocked` comes from the ADP pair runner (`native_g1_provider_runtime.py:450`): a pair fails unless both of its candidates complete. It cannot occur on the beta path:
  - the beta runtime never imports that runner (`native_g1_beta_bundle.py:23, 54, 60`);
  - the beta lane skips the policy-query checks (`adp_isaac_lab_arena_vast.py:1169`).

**The 09-27 DP trial, measured.** Read from `controller-fe6061cc-blueprint-run-20260927d3`, JSON and logs only.
- **Machine and money:**
  - one RTX A6000 at $0.5485/h, live 4,988 s (83 min);
  - Vast's posted charge was $0.903: GPU $0.675, disk $0.097, bandwidth $0.131;
  - provider zero was confirmed;
  - its limits were the ADP lane's: $5 cap, $1.25/h, 4 h.
- **Setup:** about 12 min, from instance start at 23:17:24Z to the first worker at 23:29:05Z. It downloaded all four checkpoints (about 20.8 GB), 4.4 GB of runtime dependencies and the 339 MB bundle.
- **DP cell:**
  - the worker started at 23:29:05Z and reached its episode about 6.8 min later (cold Kit);
  - the next worker started at 00:18:48Z, so one 150-query episode (60 s simulated) with its media took at most 50 min;
  - the per-query records carry no wall-clock time, so the episode alone is not measured.
- **π0.5 cell:**
  - with Kit warm it reached its episode in about 1.5 min;
  - the run's output was 2.39 GB.
- **Why DP failed.** This was not a measurement of the policy:
  - Robot–background contact forces of 104–8,677 N appear in steps 1–9 (0.02–0.18 s), with the contact pair not identified.
  - From step 8 to the end the robot was outside the workspace.
  - The book never moved (translation 1.4 × 10⁻⁸ m, lift 0.9 mm), and task contact force stayed 0.
  - At its second query the policy already saw a state far outside its training range (raw values −37 to +22).
  - The retreat clearance reads −17.6 km.
  - That is a physics blow-up at spawn. The score's `collision_or_containment_failure` does not show it.
- **Prompt mismatch:** the policy got the site's sentence ("Pick up the open book…"). The checkpoint's trained prompt, which the server logs as its default route, is "Move the box from the table onto the shelf."
- **π0.5 result:**
  - its weights loaded ("All keys loaded successfully!");
  - it received its first query with 12.5% of state values outside the normalization range;
  - it returned no action before the worker's timeout;
  - the cause is not proven.

**Estimated cost and time per G1 beta session:**
- Setup: about 10–12 min.
- Cells: at most 50 min each, measured once; the code caps a cell at 55 min (`native_g1_beta_runtime.py:314`).
- A 4 h session therefore fits about 4 cells, not the 10 the plan lists. The session also stops at the first cell that does not complete (L319-322).
- At today's A6000 prices ($0.46–0.60/h with 200 GB disk), a full 4 h session costs about $1.9–2.4 plus about $0.1 transfer, under the $4 cap.

**Vast today** (14:32Z; free public offer search, verified on-demand, single GPU; nothing created):
- 48 GB RT parts the lane accepts:
  - RTX A6000: 4 offers at $0.40–0.51/h base, 2 of them with ≥ 200 GB disk;
  - RTX 6000 Ada: 2 offers at $0.51–0.53/h, both under 200 GB disk;
  - L40 and L40S: none.
- RTX PRO 6000 starts at $0.97/h.
- $0.80/h rents, but the pool is thin.

**Tokenizer folder:**
- The door refuses the folder and every receipt with "tokenizer" in its name (`secret_name_refused`).
- The only door code that checks those six files is inside `install-public-trlc-policy`, which also installs TRLC, so it is not a read-only check.
- Planning is the sanctioned check: once π0.5 has a registration, its row shows `g1_beta_selected_pi_tokenizer_unavailable`, `…_invalid`, or nothing.
- Not verified today.

**Host:**
- Disk headroom is `ok`, with 38.7 GB available above the floor and no refused roles.
- `/var/lib/blueprint` is 86.7% used and `/mnt/blueprint-work` 82.8%.
- A G1 collection reserves 4 × 4 GiB plus the bundle (`native_g1_beta_paid.py:97-104`).
- Rebuilding the runtime source needs about 4.6 GB more.

**Where to watch a G1 run.** All reads are JSON or text through the door.
- Queue: `/var/lib/blueprint/pipeline-control-plane/beta-policy-screening/`, holding `screening_ledger.v1.json`, `plans/`, `publications/<campaign>.json` and `attempts/<id>.attempt_dir.json`.
- Attempt `A`: `/mnt/blueprint-work/beta-policy-screening-work/attempts/screen-<id>/`:
  - `runtime_inputs/session_authority.json`;
  - `bundle/native_g1_beta_provider_bundle.v1.json`;
  - `prepared.json`, `paid_admission.json` and `allocator_result.json`;
  - `provider_absence/<digest>.json` (the scoped closeout).
- Provider run: `A/allocator/attempts/attempt_001/`:
  - `vast_provider_run/vast_onstart_container.log`, `vast_budget_ledger.json` and `vast_runtime_phase_log.jsonl`;
  - `independent_vast_watchdog/groot_oscar_runpod_canary_watchdog.json` (a legacy name);
  - `immutable_execution/native_g1_beta_session_result.v1.json`;
  - per cell: `cells/<cell_id>/episode/native_g1_shared_scene_episode_trace.v1.json` and `query_checkpoints/`.
- Read behaviour from the trace and the score's `event_ledger` (contacts, forces, workspace steps, object motion), not from the outcome string.

**Owner decisions, asked 11-06** (each with this session's recommendation):
1. **Rights scope.**
   - The 09-25 approval covers internal development simulation of scene 841757 only. Its non-scope excludes a production profile.
   - Recommendation: decide model by model.
     - DP: HumanoidArena source MIT, ModelScope label Apache 2.0, SONIC weights under the NVIDIA Open Model License. If you are satisfied with those terms for private customer-site screening (a private count summary, no media, no redistribution), approve DP.
     - π0.5: hold it. Its config says `license: null`, its `pi05_base` terms are inherited, and its tokenizer is the gated Gemma one.
   - **Until you answer, no G1 launch for a customer site.**
2. **Who registers a site.** Recommendation: one standing approval, recorded once and scoped by decision 1, so the service registers each pick-and-place site after its scene packet completes.
   - Registration grants no spend.
   - Planning and launch still re-check consent live.
   - The alternative keeps a one-tap door request per site.
3. **Money.**
   - Recommendation:
     - keep the $4 cap, $0.80/h and a 4 h TTL for G1;
     - plan 4 cells per session instead of 10;
     - G1 shares the $30 site budget, with no increase.
   - Top up Vast to clear the $11 page; $7.07 covers one $4 launch at a time.
   - For the first proof, capture-447 has $3.37 left, less than one G1 run. Recommendation: the next new pick-and-place site rather than raising capture-447.
4. **Navigation goal authority.** Recommendation: yes.
   - Let the site's confirmed task brief plus a machine-composed authority replace the named `human_reviewer` / `confirmed_by_team_id`.
   - It would be recorded as a disclosed assumption, still `development_only`.
   - That requirement protects task truth, not privacy, rights or money.

**Owner answers, 11-06 15:07Z (09:07 CST), in this session:**
1. **Rights.** The owner's words: "sept 25 is out of date - we're being far too strict/have far too many rules - g1 should be an option to choose to eval for all sites/tasks".
   - This session reads it as: all four G1 candidates, DP and π0.5, are approved for private screening on every site and task.
   - The authority's `authorization_basis` is now `owner_approved_g1_screening_for_all_sites_and_tasks_2026_11_06`.
   - The 841757 receipts still pin the exact model artifacts.
   - Public media, model redistribution, ranking, qualification and physical claims stay false; the owner did not change them.
2. **Registration:** approved. One standing approval; the service registers each site by itself.
3. **Money:** approved.
   - $4 cap, $0.80/h and 4 h for G1, with 4 cells per session.
   - G1 shares the $30 site budget, with no increase.
   - The owner tops up Vast.
   - The first proof is the next new pick-and-place site.
4. **Navigation goal:** approved. The site's confirmed task brief replaces the named human reviewer, recorded as a disclosed assumption and still `development_only`.
- **Runtime source:** "approve the rest" is read as approving the byte-identical restore, this session's stated preference.
- **"All tasks" is not yet true in code.** The G1 packet recipe and the runtime support only `rigid_pick_place`. Navigation is Phase D of the plan, and other task families (articulated, pouring and so on) have no G1 recipe yet.

**Next, without waiting on the decisions:**
- the G1 money check at planning;
- the development-only choice blocker (WebApp first, then the Pipeline);
- the runbook's stale $15;
- a plan to rebuild the runtime source from its build record.

**Progress, 11-06:**
- WebApp [#1225](https://github.com/ognjhunt/Blueprint-WebApp/pull/1225), merged 14:55Z. The website accepts `development_only_reference` and shows such a row as a development check, never ranked or chosen. Nothing emits the value until the Pipeline change below deploys.
- In the Pipeline PR from this branch:
  - the selection marks scored G1 rows `development_only_reference`;
  - planning checks G1 against its own GPU lane;
  - the G1 runbook and screening docstrings say $30;
  - `docs/G1_WEBSITE_ENABLEMENT_PLAN.md` holds the plan: restore or rebuild the runtime source, automatic registration, money, a spawn check, and navigation.
  - It ships with the first attested deploy after the owner's Bandit review.
- Automatic G1 registration (owner answer 2): the screening worker registers each pick-and-place site itself into a service-owned overlay from the pinned `config/native_g1_shared_inputs.v1.json`; planning stays the verifier. It registers nothing until the runtime source is restored and the change is deployed.
- [#3420](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3420) merged (92de2d9): the owner's decisions, the `restore-g1-runtime-source` door operation, and the Bandit triage renewal. The owner chose to renew all 96 entries; each now reads reviewed 2026-11-06, expiring 2027-02-04.
- [#3424](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3424) merged (3f2ceb6), on the owner's "make g1 eligible to be chosen as the winner too". A scored G1 row now competes like any learned policy and can be chosen; the `development_only_reference` marking from #3420 is gone. The claim ceiling stays `development_only`.
- Door and deploy, from the door's request log:
  - 16:31:42Z: `door-upgrade` to 92de2d9.
  - 17:04:34Z: `deploy` of a14db1c, the receipt at 17:19:40Z. It ran outside the 15:00Z window, on the owner's explicit "do all this", with no paid stage active, 2 h after the 15:00Z deploy.
  - 17:20:18Z: `door-upgrade` to a14db1c.
- Runtime source restore:
  - The plan (17:21:04Z) passed: the object is present, its sha and size match, and 34 GB is free against 4.9 GB needed.
  - The apply (17:21:38Z) refused, because the service user cannot create the target directory under the root-owned `/mnt/blueprint-work`. Nothing was copied.
  - The fix (the door script creates only the pinned directory, as root) is in #3426 and needs a door upgrade.
- [#3426](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3426) (open) holds automatic registration plus four more changes:
  - **Privacy:** consent is read live before every registration row, and a refusal removes the row. Withdrawal deletion now deletes the site's registration files and G1 packets (`website_capture_withdrawal_native_g1`).
  - **Main's reds, carried so its full suite can pass:**
    - #3427's `success_rule` is now admitted by the retirement observer.
    - #3428's hold split, ported.
    - Withdrawal deletion is now its own command (`python -m blueprint_pipeline.website_capture_withdrawal_deletion`), so the intake no longer reaches the hot lane. That had been red in the nightly since 11-04.
- **Privacy finding, decided by the owner:** withdrawal deletion had never deleted a site's compiled team scene halves (`task-evaluation-inputs/team-scene-halves`). These hold collision and appearance data derived from the capture, and earlier receipts were silent about them.
  - The owner decided "yes delete the team scene halves on withdrawal too". Withdrawal now deletes the site's halves and every record pointing at them (`website_capture_withdrawal_team_scene`): the team registry's derived rows, the team scene packet queue's records, and the probe queue's entries.
  - Probe copies: each robot-free probe run keeps a sealed bundle built from the half. The owner chose "Delete once settled", so withdrawal deletes a run's bundle and output once its money is final, and keeps the records settlement replays from (`website_capture_withdrawal_rigid_probe`).
  - Unsettled probe money: a funded run not yet settled when the deletion ran could never settle afterwards, because the scene-file deletion removes the capture's task context that settlement reads. The owner decided "make the deletion wait until probe runs settle", so any job whose money may still settle is now `unresolved` (`rigid_probe_money_not_final`) and the apply refuses until the runner settles it. A held settlement keeps the deletion waiting.
- Spawn check merged ([#3429](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3429), 8dc96be). Before funding, Codex's review added a live consent read ahead of measuring an older packet.
- 8dc96be (#3426, #3428, #3429) deploy requested by the library-aisle session at 22:20:23Z, after main's attestation at 22:19Z. dfd62ff got no attestation because of an order-dependent pinned-client test, fixed in #3430.
- 8dc96be became active at 22:32:27Z (deployed, `commit_proven`), with no paid stage active.
- Door and restore, from the door's request log:
  - 22:32:56Z: `door-upgrade` to 8dc96be, finished 22:33:17Z.
  - 22:33:44Z: restore plan, `planned`. The object was present with sha and size matching; 32.3 GB free against 4.9 GB needed.
  - 22:34:20Z: restore `--apply`, `restored` at 22:37:38Z. The packet matches `sha256:8128c868…` (4,638,507,022 bytes) and the receipt matches `sha256:b303f301…` (44,943 bytes). No provider mutation.
- First automatic registration, capture-447 at 22:40:34Z:
  - DP and π0.5 manipulation: `registered`.
  - Both navigation rows: `g1_current_navigation_goal_authority_missing`, as expected for a pick-and-place site.
  - The site has $3.37 of $30 left, below one $4 G1 run, so its G1 quotes defer with `site_screening_budget_insufficient`. The first paid G1 proof stays the next new pick-and-place site.
- First spawn check on the real 38 MB Arena USD, on capture-447's G1 packet:
  - `clear`: 71,464 scene samples, no overlap, no unreadable collider, the first searched stance kept.
  - 7 bodies have no collider and are ignored: `d435_link`, two IMUs, `mid360_link`, `pelvis`, and two waist links.
  - The left sole is 1.4 mm above the floor. The right sole has no upward-facing scene surface within 10 cm of it: a hole or steep patch in the scanned floor.
  - The check measures overlap, not support, so this is neither a gate nor a flag today. A follow-up should flag an unsupported foot (`setup_quality_flags`, not a gate).
- Scene retirement leaves the G1 overlay row and `native-g1/` packets in place. That is storage, not privacy, because withdrawal now deletes both. It is a follow-up.

## Operating notes (2026-11-02)

- A second operator (the cloud session) uses the shared `cloud-scene-runs` door token. It deployed `b631b23fa` at 14:05:57Z and `ed6d0befb` at 15:18Z. This session makes no Pipeline deploys while it drives capture-447; merged fixes ship with the next deploy. At 15:32Z no paid launch lock was held.
- 17:23–18:54Z, the cloud session (same door token) deployed `470bf0cc9` (#3277, 17:37Z) and `b04975ca7` (#3284, 18:38Z). Both are B-04 money fixes, which D-08 allows outside the window. It upgraded the door to `c5c3a12ee` (18:39Z). At 18:54Z it applied a $90, 16-attempt execution budget to capture-447's new job, recorded as approved by the owner in that session. The job was admitted at 19:36Z on Vast ($0.80/h, $31 hard cap, 7.5 h TTL) and has been in its CPU prestage since 19:47Z; no GPU was live at 21:04Z.
- 22:31Z, the cloud session (same door token) deployed `e86f4e545`, `main` through #3292. That deploy shipped everything this session had queued for the 11-03 window (E-02, D-05, the G-02 retry cap, #3285–#3289). The door was not upgraded, so `door status` still lacks the #3288 sections.
- 01:40Z, from the cloud session's WebApp [#1198](https://github.com/ognjhunt/Blueprint-WebApp/pull/1198) (draft): capture-447's ARTISAN task object cost $4.21 of its $12 cap; stages 5–6 ran on one Vast GPU (about $1.16), and teardown reached provider zero. Website publication waits for #3297 (merged as `ceddd0220`) to deploy. That deploy will also carry this session's #3294, #3296 and #3298 (see the checklist below).
- Five `footage_before_account` emails, the first ever, went to the owner at 14:05–14:08Z about old guest-uploaded test jobs. Each job gets one. The email promises "We email you when that changes", but nothing will change for those jobs (wording follow-up under F-03).

## Next Pipeline deploy (2026-11-03, 15:00 UTC window)

Deploys through the shared door token on 11-03: `ceddd0220` (01:59Z), `2ace7c2d6` (03:48Z; the door was upgraded
to it at 03:49Z), `6cec9f9b9` (04:15Z), `6e9141fe5` (04:44Z), `d2b17f8c9` (11:03Z), `9ea18d41e` (13:48Z) and `333b1f74a` (14:29Z). They shipped
[#3298](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3298) (doctrine), [#3303](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3303) and [#3305](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3305) (the geometry-only route and its
fit check, off by default), [#3307](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3307) (status v5: the robot fit table for the site) and
[#3309](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3309) (E-02), [#3318](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3318) (the team packet reads the room collider once) and
[#3320](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3320) (fit step 3). Another operator requested `333b1f74a` at 14:15Z, before the window, so
this session did not stack a second deploy. `333b1f74a` is `main` through [#3321](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3321); the next
deploy carries what merges after it.

Before: follow `AGENTS.md` "Beta Deploy Policy". At 11:04Z no paid launch lock was held.

Checked at 14:30Z: `active_release` is `333b1f74a` with its receipt, no paid launch lock is held, the spend
guard passed, no failed unit is new, and the live `robot_arm_chains.v1.json` is byte-identical to `main`'s.
Checked at 11:04Z: `active_release` was `d2b17f8c9` with its receipt. Done since 04:15Z: the door upgrade (`door
status` now shows `capacity.provider_spend`, `capacity.stall` and `capacity.reclaim_outlook`), the post-deploy
check, and E-02 (the listener reuses its stored verdict on about 9 passes in 10; see its row).

Still to do:
1. G-02: run the withdrawal drill on a test job.

## Owner decisions (2026-11-02)

The owner delegated every decision to the recommendations in the brief, except
where an action strictly needs the owner. That covers D-03, D-04, D-05, D-07,
D-08, D-12, D-13, D-15, D-16, D-17, D-19, D-20, D-21, D-22 and D-23.

about 15:45Z, the owner, on the Privacy/Terms correction and the doctrine line: "change it so it doesn't say this anymore" (G-04).

About 16:35Z, the owner: the corrected Privacy/Terms take effect today (G-04), and no volume growth for now (D-10).

Money and data-deletion items still need an explicit, specific approval:

| Ask | Amount or scope | Status |
|---|---|---|
| D-02: raise capture-447's job ceiling | $70 → $110 (Render `BLUEPRINT_WEBSITE_ATLAS_AUTHORIZATIONS_JSON` row) | owner |
| D-10: grow the work volume | +200 GiB (about +$20/month). **Declined for now** (owner, about 16:35Z: "NO - dont grow rn"). Reclaim instead (E-01). At the floor, launches refuse at $0 and resume by themselves (C-02) | owner |
| D-10: evict the SAM-3.1 preparation cache | 22 directories, 10.1 GB, idle 47–51 days, rebuildable. `#2380` requires a separate owner decision | owner |
| D-09: Vast auto-recharge | owner account setting; suggested: recharge to $50 below $25 | owner |
| D-01 / D-06 / B-04 defaults | amounts come from the B-03 cost model | not started |
| D-11: host resize | E-02 removed the listener's CPU waste (see its row). Load at 10:53Z: 3.88 / 2.88 / 3.00 on 4 CPUs. Resizing to 8 vCPU is the owner's call | owner |

Facts and actions only the owner has: counsel status before 2026-11-15 (G-01),
the on-call backup (D-24), the invitee allowlist (D-13), credential rotation
(G-05), and repo visibility (both repos are public; making them private moves CI
onto paid Actions minutes).

## Owner requests beyond the brief (2026-11-02)

The owner's goal: a place where anyone can post a job for a robot, and Blueprint connects them with robots that
can do it at the best price. It starts with a quick, cheap check of whether a robot model fits a physical job.

| Item | Status | Evidence / next step |
|---|---|---|
| Introduction by reply in place of booking | done (live) | See F-09. |
| Spec-feasibility tier, step 1: robot spec catalog and a fit table per job | live | Pipeline [#3294](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3294), live in `ceddd0220` (01:59Z): 22 robot models from their spec sheets, each figure with its source; 17 sampled figures were checked again at the source. At the website handoff, on the CPU only, each model gets fits, fits on figures, doesn't fit or unknown, with reasons. It gates nothing. On the test room with a 1 kg item, the Franka Panda fits (path-checked) and 9 more arms fit on figures. |
| Step 2: Atlas geometry-only mode (skips object removal, the refined splat, ARTISAN and the GPU scene build) | built, test job next | Owner, 11-03 about 02:20Z: "Yes, start step 2". [#3303](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3303) adds the route behind `BLUEPRINT_WEBSITE_ATLAS_ROUTE`: the job keeps the scene as filmed and buys only the pose, its scale review and the $0.50 mesh. [#3305](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3305) cuts the filmed object out of the collider, fills the counter it hid at the height of the counter around it, writes the capability verdict and the fit table, and stops before anything paid; the capture stage and the status sync accept that end. On the test scene the real check returned `clear`, with 7 models fitting on spec. [#3307](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3307) (status v5) and WebApp [#1203](https://github.com/ognjhunt/Blueprint-WebApp/pull/1203) (live 04:03Z) show the site a "Robots that may fit this job" card for both routes, send no needs-attention email for a geometry-only end, and word that end as a result. A replay of an old job (c33427c3) cannot spend its $0.50: a finished job keeps its route. Owner, 11-03 about 04:05Z: one fresh test job instead (about $1.50 of Atlas, plus the usual per-upload model steps under the existing $5 upstream cap). Order: `6cec9f9b9` live, then the owner adds `BLUEPRINT_WEBSITE_ATLAS_ROUTE=geometry_only` to `/etc/blueprint/website-atlas-preview.env` (read by both Atlas units, no deploy), uploads one short video, and removes the line after the run. 11-03 14:49Z: steps 1–3 are live; the owner will run the test job later today (switch on, upload one video, switch off once the job records its route). |
| Step 3: a URDF-chain reach reader for the recipe arms | live | Owner, 11-03 about 12:50Z: "Yes, start step 3". [#3320](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3320) reads each catalog arm's kinematic chain from the URDF the pipeline prepared from its pinned sources (13 arms; UR10e, CRX-10iA/L, GP20 and GoFa 12/127 were prepared for this). The fit table now solves those arms with the tool pointing down at the grasp and the place across the scale band, instead of a sphere of the published reach. It fails an arm only on the chain's own reach bound, or when no pose was found and the published reach fails too; otherwise it is unknown. Pointing down at a 0.1 m object, the CRX-10iA reaches 1.05 m (published 1.249 m) and the Kinova Gen3 0.70 m (published 0.902 m), and the GP12, iiwa 14 and M-710iC/45M cannot reach within 0.40, 0.45 and 0.55 m of their own base. On the step 2 replay job, no robot's overall verdict changed; the M-710iC/45M's reach went from pass to unknown and the GoFa 5/95's from unknown to pass. No collision, scene or self, is checked, so a pass stays a fit on paper. Live in `333b1f74a` (14:29Z); the host's chain catalog is byte-identical to `main`'s. |

## A. Prove one website job end to end

| ID | Status | Evidence / next step |
|---|---|---|
| A-01 | owner | **Driven by the cloud session (branch prefix `bold-darwin-iketye`)**. It deployed `b631b23fa` at 14:05:57Z and `ed6d0befb` at 15:18Z (the active release at 15:32Z), and merged [#3259](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3259), which sizes the CPU prestage by what it writes. This session stood down on capture-447 at 14:45Z to avoid two drivers. History: the export-only relaunch was refused before allocation at $0 (13:39:42Z, `scene_configuration_provider_output_disk_budget_exceeded`: needs 35.19 GB on the work volume, had 8.06 GB). The scene intent was then revoked at 14:08:33Z. Root cause (cloud session, WebApp #1182): the brief-reading lane, idle since 10-24, drained its backlog from 14:03Z, and one reading folded an inferred answer into the unconfirmed brief. That moved the task-context digest the scene grant is bound to (`website_scene_sponsorship_changed` → revoke). |
| A-02 | not started | Runs in the same GPU session after stage 3. The 10-31 destination gates are fixed on main (#3136; `ef276779a` for the stage-5 bind tolerance; `ae0e4ae2b`). |
| A-03 | owner | WebApp #1168 merged 14:04Z (deploy pending CI). capture-447 still needs the D-02 ceiling row ($110) before its screening plan can fit. |
| A-04 | not started | Host has never created the team-scene-packet status or rigid-probe roots. |
| A-05 | not started | Pre-flight (read-only):<br>• The 10-26 canary's 8.4 GB output failed its single signed PUT (probably the 5 GB single-upload limit). The automatic pinned-SSH fallback (`vast_provider_output_recovery`, 900 s) covers the screening lane, which writes about 4.2 GB per candidate. Multipart upload remains a SHOULD.<br>• Fixed: candidate order. [#3260](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3260) (D-20) funds π0.5-DROID, Cosmos3-Edge-DROID and GR00T-N1.7-DROID first when there is no intake ranking.<br>• Fixed: the job ceiling's native over-count. WebApp [#1184](https://github.com/ognjhunt/Blueprint-WebApp/pull/1184) bounds it by the budget the Pipeline enforces; it had counted about $361 against a $70 cap.<br>• Still open: ingest needs a confirmed `siteTaskBriefs/<job>`. Root has 4.58 GB of headroom for screening attempt folders (13:19Z). |
| A-06 | not started | After A-05 and C-01…C-07. |

## B. Velocity, cost model, money correctness

| ID | Status | Evidence / next step |
|---|---|---|
| B-01 | in progress | Phase 1 merged: [#3256](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3256), `python -m blueprint_pipeline.task_evaluation_downstream_replay` replays completion → readiness → team packet → probe enqueue → screening plan offline, with paid calls, network and WebApp stubbed (18 tests). Phase 2 (stages 3–6) needs a Linux mount namespace (absolute `/workspace` paths); the cloud session's #3253 covers the stage-3/4 ARTISAN export replay. |
| B-02 | not started | |
| B-03 | done | [`2026-11-cost-model.md`](2026-11-cost-model.md): capture-447 measured spend $25.08 vs about $83 held across caps (scene holds 2.8× actual). Modelled job: p50 about $27 (plus unread upstream), p90 about $44, $59 with screening. Recommended defaults are listed in section 7; they need owner money approval (D-01/D-06). Found by code reading, not runtime-tested: `max_paid_attempts` never binds website scene-configuration reservations. |
| B-04 | owner | Accounting policy change; needs owner approval once implemented. 11-03: the cloud session's Pipeline [#3315](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3315) and WebApp [#1204](https://github.com/ognjhunt/Blueprint-WebApp/pull/1204) (open since 06:01–06:05Z) report the hold that a settled attempt keeps, and count it in the WebApp's job ceiling. For capture-447 the Pipeline commits $39.63, where the ceiling counted $18. |
| B-05 | done | [#3265](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3265) merged: a "Spend limits" table in `AGENTS.md` lists 18 money ceilings with the file that sets each. Every code value was checked against `main`. Values that live only in Render or on the host are marked unverified. The B-03 defaults are not applied; they wait for the owner (D-01/D-06). |
| B-06 | not started | |
| B-07 | not started | `door status` 13:19Z: internal ledger $165.55; billing reconciled DigitalOcean $325.38 + Vast $455.88. The spend guard's RunPod inventory call succeeds (no errors in its journal over the last hour), so the audit's "1,065 failing RunPod calls a day" does not reproduce there; its source is unknown. |
| B-08 | done (live) | [#3279](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3279) is live in `b04975ca7`. The capacity summary at 21:01Z reads all three providers: OpenAI $27.90 month-to-date, all on 11-02 (Costs API, so the Admin key is on the host); World Labs $0 (no Atlas attempts this month); GCP `blueprint-8c1ca` $1.43 of $100. None is near its line, so no page. `door status` drops the section until the next `door-upgrade` ([#3288](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3288)). Owner facts still open: OpenAI and World Labs auto-recharge settings, and whether the Gemini key bills to `blueprint-8c1ca`. |

## C. Unattended operation

| ID | Status | Evidence / next step |
|---|---|---|
| C-01 | not started | #3196 merges after capture-447 settles. The configured-controls timer, path and service are all inactive (13:30Z). |
| C-02 | done (live) | [#3251](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3251) is in the active release `ed6d0befb`: a $0 pre-allocation disk refusal of a website scene resumes like credit and auth refusals, bounded by the 6-recovery ceiling. The new defaults from B-03 still need owner money approval (D-01/D-06). |
| C-03 | owner | WebApp #1180 merged: the sponsorship TTL may be up to 7 days. The Pipeline refuses any grant window over one day (`load_website_scene_sponsorship`, `validate_preparation_authority`) and any intent staged more than a day after consent until the website job window change ships (Pipeline `website_job_window.py`, WebApp `websiteJobWindow.ts`; 7 days, owner-approved 2026-11-03). **Owner action, in this order:** deploy that Pipeline change, then the WebApp change, then set `ttl_seconds: 604800` in Render `BLUEPRINT_WEBSITE_SCENE_SPONSORSHIP_JSON` (or leave it out for the 7-day default). Set earlier, it stops new jobs (the deployed Pipeline refuses a 7-day grant) and jobs in flight (the deployed WebApp refuses their grants as `website_scene_sponsorship_changed`). |
| C-04 | done (live) | WebApp [#1190](https://github.com/ognjhunt/Blueprint-WebApp/pull/1190) has been live since 16:27Z, and Pipeline [#3266](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3266) since the 17:37Z deploy. Checked at 21:15Z: a revoked job now carries a sealed `outcome.json` with exactly the 9 fields (for example `scene-929717a8…`: `withdrawn`, `privacy_rights`, `authority_revoked`, terminal). No outcome email has been sent and none is pending (Firestore `captureOutbox`, `taskEvaluationSceneIntakes`). The WebApp emails only for jobs it still follows, so old ended test jobs send nothing. |
| C-05 | not started | |
| C-06 | in progress | Deploy policy written ([#3272](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3272), D-08): one batched deploy of `main` a day in a 15:00 UTC window; held while an external job is in a paid stage (held paid-launch lock, activation hold, or screening launching or running); outside the window only money, privacy, rights or stuck-paid-stage fixes. The door already holds deploys during a live scene activation. Not started: a door-enforced paid-stage hold, release succession reusing unchanged preparations, and automatic keep-last-3 release retirement. |
| C-07 | done (live) | [#3257](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3257) is in the active release `ed6d0befb`: a Vast balance under $11 (two launches' $5 compute plus the $1 reserve) pages hourly, with the balance in the alert. Still open: a credit reservation across concurrent launches, needed once dispatcher concurrency is above 1. Auto-recharge is the owner's Vast setting (D-09). |

## D. Observability and operations

| ID | Status | Evidence / next step |
|---|---|---|
| D-01 | done (live) | #3226 is in the active release `ed6d0befb`: stall paging on the capacity alert route (D-12 approved by delegation). Follow-up: page daily on cap exhaustion. |
| D-02 | in progress | WebApp [#1176](https://github.com/ognjhunt/Blueprint-WebApp/pull/1176) merged: the external uptime probe also checks the door `/healthz`. Pending: the WebApp deploy, then a manual deploy of the `blueprint-uptime-probe` Render cron (`autoDeploy: false`). Capacity-tick freshness is a follow-up. |
| D-03 | done (live) | WebApp [#1200](https://github.com/ognjhunt/Blueprint-WebApp/pull/1200) is live in `a2bc22f9a` (02:14Z). For each job with footage, the ops console reads Blueprint's scene the way the job page does (`loadWebsiteSceneProgression`). It names five new stages (Scene preparation, Screening, Estimate, Held, Ended), each with the next owner. Money and setup waits (spend approval, setup repair) and a published estimate appear as attention notes. Team runs and withdrawal still come first. The `unclaimed` line no longer says a scene is built only for a claimed site. |
| D-04 | done (live) | [#3267](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3267) is live: the preflight summary is published again (generated 20:34Z for `b04975ca7`). It lists 186 blocker, 430 warning and 14 info findings, and 17 units have unknown coverage. Most blockers are path checks: `path_read_only` 98, `path_unreadable` 52, `path_permission_denied` 29. The rest: 5 `unit_trigger_inactive`, 1 `activation_intent_missing_for_active_release`, 1 `controls_robot_binding_unresolvable`. The live chain still ran activation and dispatch for capture-447 at 19:06–19:36Z, so most findings look like classifier gaps. Triage is a follow-up; until then the blocker count is not a health signal. |
| D-05 | done (live) | [#3286](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3286) is live in `e86f4e545` (22:31Z). At 01:11Z the 7 old deliveries were parked with typed reasons, 0 were due, the queue was not overloaded and no stall alert was raised. Root cause, from the host's files: WebApp [#1113](https://github.com/ognjhunt/Blueprint-WebApp/pull/1113) (10-27) changed the digest of unconfirmed job-brief context, so six old test captures can never match the one they saved. Follow-up: accept the older context digest. |
| D-06 | done | Each inactive trigger has a recorded decision in [`beta-operations.md` section 7](../runbooks/beta-operations.md#7-triggers-that-are-off-on-purpose-d-06), checked on the host at 01:58Z on 11-03:<br>• The billing reconciler timer is not needed, because the spend guard pulls the reconciler (E-02).<br>• The launch supervisor spends OpenAI money, so it waits for an owner decision.<br>• The terminal resource release watcher stays off. Its queue has been empty since 08-12; arm it if anything writes there again.<br>• The capture reconstruction dispatcher is a legacy queue that is empty; it is retired in H-03.<br>Not done: `reset-failed` on the transient failed units (O-24), which waits for a deploy window. |
| D-07 | owner | Runbooks done: [#3265](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3265) adds `docs/runbooks/beta-operations.md` (stuck job, Vast credit, full work volume, door or host down, deploy rollback). Still needs the on-call backup's name (D-24). |
| D-08 | not started | |
| D-09 | not started | |

## E. Capacity, infrastructure, security

| ID | Status | Evidence / next step |
|---|---|---|
| E-01 | done (live) | Live in `b04975ca7`. The first timer run (18:38:35Z) retired the 10-31 claim, and a later GC tick removed its 14.8 GB directory. capture-447's new activation (19:07Z) took about the same space, so `/mnt/blueprint-work` still has 30.75 GB free. That job's running CPU prestage reserves 18.3 GB, which leaves 0 GB above the 15.8 GB floor (21:01Z). Other launches refuse at $0 until it ends, then resume by themselves (C-02). `/var/lib/blueprint` is 92.6% used, 3.7 GB above its floor. The GC still keeps the result registry (17.1 GB, 29 runs), 13 activation pins (6.7 GB) and hot evidence (3.1 GB). Volume growth stays declined (D-10). Orphan scratch (26 GB in the earlier survey) needs the owner's OK. The GC's stranded-rows phase moves off root first (follow-up). 11-03 11:04Z: `/var/lib/blueprint` is 93.5% used (10.7 GB free) and `/mnt/blueprint-work` 87.8% (38.6 GB free). `door status` raises `reclaim_ineffective` (page) and names the result registry, hot and pinned data as the top retained reasons. |
| E-02 | done (live) | [#3278](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3278) (22:31Z) cut the billing reconciler to 389 CPU-s an hour (before: 1,279–1,511). [#3296](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3296), live in `ceddd0220` (01:59Z), stopped the catalog pass's WebApp reads, so the listener's guard could store its catalog verdict. Most passes still ran in full, because the verdict bound `gpu_spend_guard/latest.json`, which the spend guard rewrites every 1–2 minutes. [#3309](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3309), live in `6e9141fe5` (04:44Z), binds a resolved path by its type and link target only. Checked at 10:54Z: since that deploy, 212 of 243 listener passes logged `reused_phases=launch_profile_catalog,launch_input_residency`. The 31 full passes logged `expired` (21), `absent` (8), `path_changed` (1) and `entry_stale_or_invalid` (1). In the hour to 10:54Z the listener used 358 CPU-s over 35 runs (median 7.4 s a run). Before #3296 it used about 950 CPU-s an hour (median 28.6 s a run, 00:34–01:13Z). Host load at 10:53Z: 3.88 / 2.88 / 3.00 on 4 CPUs (01:15Z: 4.16 / 4.04 / 4.41). A resize (D-11) stays with the owner. |
| E-03 | not started | |
| E-04 | owner | Credential rotation (G-05). |
| E-05 | not started | |
| E-06 | in progress | The Full Test Lane run at 18:23Z (`5aa04e713`) failed two tests in shard 1. The import leak from #3277 is fixed by [#3287](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3287) (19:04Z). The bundle CLI contract (`stat_only_toolchain_host_validation`, added by eccf83d9c) is fixed by [#3289](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3289). The earlier two failures were fixed by #3261 and #3263. The door now runs `c5c3a12ee` (18:39Z), so the G1 `stance_reach` contract is live. A manual Full Test Lane run on `314a0c01e` passed at 21:21Z. Next: 7 green nightlies and a documented beta deploy gate. |
| E-07 | not started | |

## F. Website: result delivery, truthfulness, journey

| ID | Status | Evidence / next step |
|---|---|---|
| F-01 | done | WebApp [#1179](https://github.com/ognjhunt/Blueprint-WebApp/pull/1179) is live: one deduplicated email per terminal screening outcome (estimate, reference_only, held, infeasible, no_match). It re-checks the page at dispatch, is cancelled on consent withdrawal, and carries the simulation disclaimer. No backfill (0 production campaigns). |
| F-02 | done | The job-evidence index is READY. Created `creatorCaptures(creator_id, created_at desc)` (READY 14:10Z) for the website's capturer page. 16 other declared indexes serve ops, growth and payout routes and were never deployed (H-03). |
| F-03 | done | WebApp [#1177](https://github.com/ognjhunt/Blueprint-WebApp/pull/1177) merged and deployed (`ba27cd6e4`, 14:33Z): one sentence per outcome on both the headline and the body; reference-only runs are not called estimates; no raw codes, budgets or operator text reach sites; exhausted allowances now read as ended (the prefix bug). Follow-up: the `footage_before_account` email promises "We email you when that changes". |
| F-04 | in progress | WebApp #1181 merged: the default intake cap is 5 a day. **Owner checks:** Render must not pin `BLUEPRINT_INTAKE_DAILY_LIMIT=50`. The allowlist needs the invitee list. App Check `monitor` needs `VITE_FIREBASE_APPCHECK_SITE_KEY`. |
| F-05 | done | [#1199](https://github.com/ognjhunt/Blueprint-WebApp/pull/1199) (02:05Z) and [#1202](https://github.com/ognjhunt/Blueprint-WebApp/pull/1202) (live 02:32Z, verified at the exact SHA `25a11a196`). The workspace Results tab shows Blueprint's own results for the owner's job: an owner-only `GET /api/workspace/tasks/:id/results` returns the same site-redacted runs the job page shows, rendered with the job page's component. |
| F-06 | done (live) | Pipeline [#3281](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3281) is live in `b04975ca7`: the owner observation accepts exactly `2026-09-18.v1` and `2026-11-02.v2`. WebApp [#1193](https://github.com/ognjhunt/Blueprint-WebApp/pull/1193) merged 21:10Z (`fa335b441`) and deployed at 21:23Z, verified at the exact SHA; the live site bundle carries `2026-11-02.v2`. The site now attests v2 and shows it verbatim where it attests: "I am authorized to record this site and to let Blueprint use the recording to draft the job brief, build a simulation of the work area and run candidate robot policies in it." It restates Terms `2026-11-02.1` section 3. |
| F-07 | done (mitigated) | WebApp #1068 (10-18) accepts footage only for a saved job, so a new guest upload cannot strand mid-run. Old guest jobs get one `footage_before_account` email; the first five were sent 14:05–14:08Z, all to the owner's test jobs. |
| F-08 | in progress | WebApp [#1185](https://github.com/ognjhunt/Blueprint-WebApp/pull/1185) documents the 10 website-beta settings in `render.optional.env.example`. Readiness checks are a follow-up. |
| F-09 | in progress | WebApp [#1195](https://github.com/ognjhunt/Blueprint-WebApp/pull/1195) (live since 11-02 23:10Z) removed the pilot booking and proposal features. An estimate's result email now offers an introduction to a deployment company by reply; replies reach hello@tryblueprint.io, and a person makes any introduction (D-17). [#1196](https://github.com/ognjhunt/Blueprint-WebApp/pull/1196) (live 02:04Z): the robot-team page and the cohort guide now say every candidate is first screened with up to 10 episodes, and a full evaluation of 200 hidden episodes runs only with Blueprint's approval. Doctrine: the shared vision now describes the introduction by reply (Pipeline [#3298](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3298), WebApp [#1197](https://github.com/ognjhunt/Blueprint-WebApp/pull/1197), BlueprintCapture [#94](https://github.com/ognjhunt/BlueprintCapture/pull/94)). Checked and kept: "We usually reply within an hour" holds on measurement (all 111 outbox emails, 09-19 to 11-02: `task_received` 19 of 19 within 60 s; `video_received` 18 of 23 within 1 h, median 24 s; the 10 failed sends were all 09-19 to 09-22). "Policies served through APIs" is a supported screening path (hosted HTTPS policies). "Simulation of your site" is the owner's 10-28 doctrine wording. Left: "booking records" in `Privacy.tsx` needs a policy version, so it goes with the next privacy revision (G-01 packet). |
| F-10, F-11 | not started | |

## G. Privacy, rights, legal

| ID | Status | Evidence / next step |
|---|---|---|
| G-01 | owner | Counsel status before 2026-11-15 is still open. Add to the counsel packet: Terms §12 and the Privacy Policy's "Changes" section promise an email to account holders before a material change. On 2026-11-02 the owner made the corrected documents effective at once and no one was emailed (G-04). |
| G-02 | in progress | WebApp [#1178](https://github.com/ognjhunt/Blueprint-WebApp/pull/1178) and [#1194](https://github.com/ognjhunt/Blueprint-WebApp/pull/1194) (a 409 for refusals that will not clear) are live. Pipeline [#3262](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3262) is live in `b04975ca7`. The retry cap [#3283](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3283) (3 fresh attempts per candidate, backoff of 1, 2 and 4 min) is live in `e86f4e545` (22:31Z). Not done: the withdrawal drill on a test job (screening is idle, with no campaign). |
| G-03 | superseded | D-16 (owner, 14:15Z): no blurring or person screening. The residual-person check in #1980 is moot; its stage-reuse part can be revisited separately. |
| G-04 | done | Owner, about 15:45Z and 16:35Z: correct the text, effective today. WebApp [#1183](https://github.com/ognjhunt/Blueprint-WebApp/pull/1183) (the corrected wording) and [#1192](https://github.com/ognjhunt/Blueprint-WebApp/pull/1192) (versions `2026-11-02.1`, effective at once) are live in `282ba6452`, verified in the site bundle. People are not removed from footage, faces are blurred only in the frames sent to World Labs, and Meta is listed. An acceptance of the earlier November 15 pairs no longer counts; the identical `2026-11-15.4`/`.5` text does. The previous versions stay frozen at `/terms/previous` and `/privacy/previous`. Doctrine line replaced ([#3269](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3269), WebApp [#1188](https://github.com/ognjhunt/Blueprint-WebApp/pull/1188)). |
| G-05 | owner | Rotation is the owner's action. Both repos are public (verified 13:30Z). |

## H. Cleanup and consistency

| ID | Status | Evidence / next step |
|---|---|---|
| H-01 | done | Pipeline [#3268](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3268) archives four obsolete launch gates (one moved; three keep their paths with a banner because tests read them). WebApp [#1187](https://github.com/ognjhunt/Blueprint-WebApp/pull/1187) brings the beta guides and README to the 10-28 model. Found false but needing code or an owner call: robot-team copy says 200 episodes per evaluation (screening plans at most 10 per candidate); `main` had the delivery-proposal screen and booking route (removed by WebApp #1195, F-09); `docs/beta/ROBOT_TEAM_SUPPLY_AGREEMENTS.md` describes the old evaluate-and-deliver rule. BlueprintCapture had not received the owner's 10-28 platform doctrine; BlueprintCapture [#94](https://github.com/ognjhunt/BlueprintCapture/pull/94) synced it (11-03). |
| H-02 | in progress | 126 PRs and 3 issues closed (D-23). Also: WebApp #1142 was carried as #1178; #1103 was closed as superseded (main no longer caps Atlas requests, per the owner's 10-30 decision). 11-03 04:30Z: the other 56 open PRs in the three repos (all but the cloud session's) were checked against `main` by content. None is launch-critical ([`2026-11-03-open-pr-triage.md`](2026-11-03-open-pr-triage.md)). 10:50–10:52Z, on the owner's OK: the 33 PRs that the table marks Close were closed (Pipeline 13, WebApp 4, BlueprintCapture 16), and none was merged. The 23 deferred PRs stay open. Dependabot deleted the branches of 5 of its 6 closed PRs, as it does on close; their commits stay under each PR. DoD item 8 now waits only on the cloud session's #3196 and WebApp #1175. Its newer PRs (Pipeline #3315 and WebApp #1204, B-04) are also its to merge. |
| H-03, H-04 | not started | |
