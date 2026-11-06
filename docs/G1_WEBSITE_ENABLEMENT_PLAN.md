# G1 humanoid on any website job: plan

Status: plan, 2026-11-06. The audit behind it is in the beta launch tracker,
section "G1 humanoid on website screening (audit, 2026-11-06)". Pieces marked
**owner** wait on an owner decision that the tracker lists. Nothing here
widens a cap, grant, TTL, retry limit or allowance, or reuses the 841757
rights approval outside its scope.

The guarantees that stay unchanged in every step:

- the `development_only` claim ceiling and `ranking_eligible=false`;
- one allocation per session, with no retry inside it;
- the independent watchdog and scoped closeout;
- the controller's own re-score;
- live consent rechecks at planning and at launch;
- pinned inventory and review digests.

## 1. Shared inputs owned by the controller (blocks everything)

The shared registry's `unitree_g1` runtime source packet is gone from the host,
so every registration fails (`g1_shared_source_verification_failed`). Its
identity is known from the build record
`/mnt/blueprint-work/g1-cuda-runtime-source-7c1f528-20260926-build.json`:

- `native_task_runtime_source_packet.v1`, packet sha `8128c868…`, 4,638,507,022 bytes;
- IsaacLab `ffff603e…` (tree `2f82f1af…`) plus Arena;
- 81 dependency wheels, profile `unitree_g1`.

Two ways back, both CPU only and both needing host access the door does not
offer, so each needs the owner:

- **Restore (byte-identical).** The 09-27 run retained a content-addressed
  object-store copy for reuse (key `561630a4…`). Copy it back with its receipt,
  and the existing registry binding verifies unchanged. Cost: 4.6 GB on
  `/mnt/blueprint-work` (82.8% used).
- **Rebuild.** Run
  `python -m blueprint_pipeline.native_task_runtime_source_packet --runtime-profile unitree_g1 --output-dir <dir> --isaaclab-repo <IsaacLab at ffff603e> --arena-repo <Arena at its pinned commit> --dependency-wheel-dir <the 81 wheels>`.
  The digest changes, so the new receipt must be pinned.

Then stop reading the historical ADP registry. Add a controller-owned manifest,
`config/native_g1_shared_inputs.v1.json`, that pins:

- the publisher source receipt (`4170b783…`, HumanoidArena `68479287…`);
- the runtime source receipt;
- the Arena robot USD (`sha256:a7a2bab7…`, 38,195,671 bytes).

`native_g1_registration_install.py` (L62-77) and the automatic path (§2) read
that manifest. The registration contract keeps verifying every file it names.
The contract lives under `deploy/operator-door/`, so this part ships with a
door upgrade.

## 2. Automatic registration for pick-and-place sites (**owner**: decision 2)

- **Hook.** When `team_scene_packet.record_team_scene` marks a website scene
  `completed` (`team_scene_packet.py:1268-1291`) and the task is
  `rigid_pick_place`, the service runs
  `beta_policy_screening_g1_prepare.prepare_native_g1_beta_packet` on that
  site's base packet. It writes under the site's own
  `native-g1/<base>/<source>/` directory.
- **Overlay.** The result goes to a service-owned file, the same row shape as
  `site_screening_candidates.<site_id>` (exact six-field `source`, one
  `native_g1` record per candidate). `team_candidate_registry_reader._site_screening_profile`
  reads it when the root-owned registry has no row for the site. The service
  runs as `blueprint`, which cannot write `/etc/blueprint`.
- **Independent check.** Planning stays the verifier.
  `beta_policy_screening_g1.candidate_execution_spec` already re-derives:
  - the packet derivation, assets and task binding;
  - the robot asset and head camera;
  - the runtime source.

  `compose_beta_authority` re-checks consent live. A changed context gets a new
  packet directory and a new catalog identity, so the site re-plans by itself.
- **Authorization.** The owner's choice is one of:
  - a standing reference recorded once in config, which the overlay copies;
  - per site, the existing door request kept as a one-tap step.

  The candidate list is limited to what decision 1 approves (DP only if π0.5
  stays held).

## 3. Money (**owner**: decision 3)

- **Done here:** planning checks G1 against its own lane's GPU filters (48,000
  MB, RT only) instead of skipping it.
- **To add once the numbers are approved:**
  - G1's own cap, TTL and cells per session, read from the profile and kept
    inside the existing bounds (cap ≤ $15, rate × TTL ≤ cap, 1,800–14,400 s);
  - quotes listing the cells that fit;
  - a plan that says when G1 does not fit the site's remainder. The existing
    `site_screening_budget_insufficient` deferral already holds it back.
- **Measured basis:**
  - setup about 12 min, at most 50 min per cell, so about 4 cells in 4 h;
  - A6000 at $0.40–0.60/h today;
  - one 83-minute session cost $0.90 on 09-27.

## 4. Task page

- **Done here:**
  - the selection marks a scored G1 row `development_only_reference`, so it is
    listed but never ranked or chosen;
  - the WebApp labels it a development check ([#1225](https://github.com/ognjhunt/Blueprint-WebApp/pull/1225)).
- **Still open:** the specific `g1_beta_result_*` reason should reach the ledger
  instead of the generic `g1_beta_native_result_missing_or_invalid`
  (`beta_policy_screening_launch.py:568-575`).

## 5. Before the first paid G1 run

- **Spawn check.** The 09-27 DP trial was a physics blow-up at spawn:
  - robot–background contact forces of 104–8,677 N in the first 0.18 s;
  - then the robot left the workspace, with the book untouched.

  Add a free CPU check to the packet recipe: G1's collision bodies at the
  prepared stance must clear the scene's colliders. Record the result beside
  the run as a flag, not a gate.
- **Prompt.** The manipulation checkpoints were trained on HOI_pp_box ("Move
  the box from the table onto the shelf."). The worker sends the site's own
  sentence. π0.5 reads language. Whether DP does is not verified. Record which
  prompt each run used.
- **Proof site.** Capture-447 has $3.37 of $30 left, less than one $4 run. Its
  place is also beyond a fixed arm's reach. A new pick-and-place site is the
  better first proof, unless the owner raises capture-447's allowance.

## 6. Navigation (plan; **owner**: decision 4)

Mobile jobs stop before the scene build today. To make them reach G1:

1. **Task profile.** Add a navigation profile to `task_profile_contracts.py`
   (the robot reaches a goal region and stops), with matching website
   preparation and coverage.
   - The WebApp's `siteTaskScope.ts` classifies mobile wording today, and
     `websiteSceneSponsorship.ts:240` gives such a job preparation spend but no
     execution grant. Both change for navigation.
   - The mobile regex runs before the pick-and-place verbs, so a pick-and-place
     brief that says "deliver" reads as mobile. Fix that in the same change.
2. **Drop the rigid dependency.**
   - `native_g1_navigation_goal.py` L40, L81 and L87-90;
   - `beta_policy_screening_g1.py` L194;
   - the registration contract, L323-328 and L479-481.
3. **Author the goal automatically.** In
   `beta_policy_screening_g1_prepare.py`, write `g1_navigation_goal` and the
   yellow marker from the measured goal region: a flat yellow disc, 0.1–0.5 m
   radius (L41-52). Record it as a disclosed assumption. Flip
   `navigation_goal_added` in both the recipe (L230) and the verifier
   (contract L393). With decision 4, the confirmed task brief plus a
   machine-composed authority replaces `human_reviewer` /
   `confirmed_by_team_id` (L114-118).
4. **Catalog.** Move the two `_vision_navi` rows to the new task kind. The
   catalog revision and digests change.
5. **Scoring.** Score obstacle contact and clearance; it is
   `obstacle_clearance_scored: False` today.
   - The trace already carries robot–scene contact forces per step. Those can
     be scored now.
   - Contact pairs come back unidentified (`contact_pair_identity_missing`),
     and clearance needs a distance measurement the trace does not have yet.
6. **Instruction.** The checkpoints' instruction is fixed: "Avoid obstacles
   and move to the yellow marked area." Record it as the checkpoints' trained
   prompt, and render the goal so it matches.
7. **First proof.** The CC0 library-aisle site (Pexels 854533, "navigate to a
   stopping point before the end bookshelf without touching the shelves"),
   once the parallel session submits it.

Order:
- 1 → 2 → 5 (spawn check) → the first manipulation proof;
- 3 and 4 in parallel with them;
- then 6.
