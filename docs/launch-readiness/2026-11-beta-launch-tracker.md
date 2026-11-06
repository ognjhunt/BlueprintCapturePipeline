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
- **Bandit triage (owner).** Review packet: [`2026-11-06-bandit-triage-review.md`](2026-11-06-bandit-triage-review.md).
  - Verdicts across the 96 entries: 77 keep, 18 tighten, 1 fix now (`wam_compute_providers.py` `_deepinfra_download_file`), 0 orphaned, 0 new MEDIUM or HIGH.
  - The latest allowed expiry for an entry signed today is 2027-02-04.
  - Nothing in `docs/bandit_triage.json` is changed until the owner signs off.

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
- Open after capture-447:
  - **owner**: Bandit triage re-review. Until it is done, no main commit gets an attestation, so #3411 cannot deploy.
  - The cosmos3 `policy_load` precedence check.
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
    - pi0_droid_jointpos_100k_polaris never went to the tray: its 33 closes were a median 27 cm from it.
    - pi0's and groot's closest approaches came at start position 02, the one start where the wrist camera sees the tray. pi05's and molmoact2's did not. That is one start, not a measured effect.
  - **Correction**: the score's `gripper_closed` is evaluated only over the final settle window, and only for tasks that must end closed, so it is always false here. Earlier entries that read it as "the gripper never closed" are corrected above. The 5 uncounted molmoact2 episodes of 04:26Z did keep the gripper fully open (measured).
  - **Also seen, not measured as causes**:
    - The prompt the policies get is "Pick up and place the wooden tray on the dining table on the wooden dining table."
    - The destination marker is a policy-visible flat green disc of 0.39 m radius. Its edge comes within 4–7 cm of the tray's centre, and 31–38% of the tray's footprint sits over it.
    - The robot base is 1.3 cm above the surface the tray rests on.
  - **Next**, before any new paid screening:
    - Check the scene planner's robot placement, so the tray starts nearer the middle of the policy's view.
    - Check the prompt text.
    - Check the marker size.

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
