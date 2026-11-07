# G1 navigation from a compiled task spec

The owner asked on 2026-11-06 at about 18:00Z to "build the missing navigation piece", meaning a website navigation job should reach a scored G1 run. Today such a job holds at the scene handoff with `website_task_runnable_policy_missing` (#3423). This doc lists what that takes, in order, and who does each part. The G1 session reviews it, and the split was agreed between the two sessions at 17:58Z.

## Where a navigation job stops today

1. **Handoff.** `website_scene_handoff.py:361-380` compiles `compiled_task_spec.json`. Both `_vision_navi` rows then lack `pipeline:navigation_scene_packet_from_compiled_spec`, so the job holds. If a policy were runnable, `website_locomotion_task_execution_route_missing` would be raised.
2. **Everything after the handoff assumes a rigid object.** Each assumption below needs a navigation variant; none can simply be skipped:
   - `team_scene_packet.py:916`, `team_scene_task_kind_unsupported`;
   - `robot_free_probe_gate.py:66`;
   - `native_task_runtime_contract.py:57,267-269`;
   - `native_task_arena_scene_plan.py:862-865`;
   - `native_task_arena_runtime.py:1769`;
   - `native_task_arena_readback.py:1067,1097`;
   - G1 prepare `beta_policy_screening_g1_prepare.py:162-179` (task binding, rigid contract, subject bounds, object-centred stance);
   - the door's registration contract (L324, L350, L393 `navigation_goal_added`, L430-434, L479-481);
   - `beta_policy_screening_capabilities.py:291`;
   - `beta_policy_screening_plan.py:457`;
   - `beta_policy_screening_g1.py:135,196,199`;
   - `native_g1_navigation_goal.py:40,81,87-93`, whose authority needs a named human and the rigid contract's team;
   - `native_g1_beta_navigation.py`, which is cell-bound to a rigid base contract.
3. **The G1 runtime itself is not the limit.** It loads the site's own scene packet (`native_g1_beta_bundle.py:105-106`, `native_task_arena_runtime.py:1378,1639-1718`). HumanoidArena supplies only the policy server and the SONIC controller.
4. **Obstacle contact is not measured on the navigation path.** The G1 navigation sampler records only `step_index` and `root_position_world_m` (`native_g1_shared_scene_episode.py:415-420`). The per-step `robot_scene_contact_peak_force_n` that the compiled `no_contact` predicate needs is produced only by the rigid readback. Contacts are unattributed (`contact_pair_identity_missing`): a shelf cannot be told from the floor.

## Plan

| PR | What | Owner | State |
|---|---|---|---|
| A1 | Navigation goal and authority from the compiled spec. `native_g1_navigation_goal` accepts the `goal_region_navigation` kind. A machine-composed authority (owner decision 4, now `v3`) binds the site's brief, the compiled spec and the goal, recorded as a disclosed assumption, `development_only`. Whether the site confirmed its brief is recomputed from its sealed context and recorded in the authority's `site_brief_confirmed`, `status` and `authority_basis`; the job holds until the site confirms its brief on the website (a confirmed brief counts as consent, owner 2026-11-07). The authority only records it. `website_navigation_packet` builds the robot-free base navigation packet request: the scene's appearance and collision, the start, the goal, the yellow marker and the trained prompt. The handoff writes it, and the hold then names only the pieces still missing. | this session | merged in #3428 (513c93d) |
| A2 | The scene's Isaac files: the ParticleField appearance USD and the collision USD (Z up, metres), authored on the controller with no spend. It reuses `prepare_native_appearance` and `normalize_runtime_collision` with the base scene's own declared frame, the one the compiler uses (`website_navigation_scene_assets`). The handoff writes `navigation_scene_assets.json`; a failure is that job's missing piece. | this session | merged in #3428 (513c93d) |
| B | The arena runtime runs a scene with no task object, behind the `goal_region_navigation` kind. The rigid and articulated paths are unchanged. This covers the scene plan, runtime contract, spawn and readback. The G1 navigation sampler records per-step robot–scene contact force. | this session; the G1 session agreed, on condition that the rigid and articulated paths are unchanged, no simulator sensor is added and the bundle digest changes are listed | PR open, G1 session reviewing |
| C | G1 lane. Prepare and the execution spec use the compiled start as the stance; the navigation cells perturb the robot's start, not an object. The registration contract, the catalog's two `_vision_navi` rows (`supported_task_kinds`), capabilities and plan accept the kind. Registration of the rows comes from the A1 packet. | this session, after the G1 session's spawn-clearance PR and #3426 | both merged (#3429, #3426); in review: the lane is built, and a navigation site registers once its record names a source. The source is the scene intent of the job's website navigation intake (`website_navigation_intake`, owner 2026-11-07: $0 reserved, no construction allowance), bound by the intake receipt beside the record |
| D | Paid G1 navigation sessions on the library-aisle site, one for each `_vision_navi` row (owner, 2026-11-07, in the library-aisle session: "two"; confirmed to the G1 session at 13:43Z as $4 each, $8 per site): each $4 cap, 4 cells, inside the $30 site limit (owner, 2026-11-06). No code change: nothing limits a navigation site to one funded row (see "Funding both navigation rows"). | controller | after A1–C deploy |

## Prerequisites outside code

- **The site confirms its brief.** The library-aisle job's `website_task_context.json` reads `confirmed: false`. A confirmed brief counts as consent (owner, 2026-11-07), so the job holds by name until the site confirms its brief on the website: at the live consent recheck, or at the WebApp's `task_confirmed`. The goal authority only records the flag (`site_brief_confirmed: false`, status `composed_for_development_simulation_site_brief_unconfirmed`, basis `machine_composed_goal_site_brief_unconfirmed_disclosed`, which cites no owner decision).
- **The Vast top-up** (owner): credit was $7.07 against the $11 page line.
- **G1 runtime source restore and registration** (G1 session; #3426 fixes the restore's permission error).

## The G1's development-run allowance

The owner decided on 2026-11-07; each record quotes the owner's words with the session and UTC time (`CHECKPOINT_TERMS_DECISION`, `VERIFIED_EPISODE_DECISION`). The allowance had been deferred in #3443 because it had no bound and no owner decision on its `reviewed_checkpoint_terms` stand-in. `task_evaluation_embodiment_registry.G1_NAVIGATION_DEVELOPMENT_RUN` is declared on the G1 row. It is general: it applies to every site and scene, it names no site, and it covers the `goal_region_navigation` task profile only. It has no count or expiry, because its two bounds are the claim ceiling and each checkpoint's own review:

- **`reviewed_checkpoint_terms`** is stood in for by each navigation checkpoint's own human development rights review: `approved_for_development_simulation`, with `checkpoint_terms_reviewed: true`, in `docs/arm_decision_proof_v1/manifests/g1_841757_<candidate>_development_rights_review.v1.json`. The owner, 2026-11-07 about 13:20Z, in the library-aisle session, answering its question 3 ("may each navigation checkpoint's development rights review stand in for its checkpoint-terms review?"): "yes - make sure this is the case for all future runs/scenes". The owner confirmed it to the G1 session at 13:43Z for every G1 navigation run on all sites and scenes ("Yes, both confirmed"). It holds per checkpoint. `g1_navigation_checkpoint_development_rights_review` reads the review through the same sha256-pinned reader every run's authority uses (`native_g1_beta_authority.reviewed_model_terms`). A manipulation checkpoint, a checkpoint with no pinned review, or a changed review file keeps the item missing. Every G1 run validates the review again:
  - `native_g1_beta_authority.validate_beta_authority` refuses an authority unless its embedded review matches the pinned file sha256 and review digest, says `approved_for_development_simulation` and has both term reviews `true` (`native_g1_beta_authority.py:194-200`). It also refuses any scope other than `development_only` and `ranking_eligible: false` (`:189`).
  - The execution spec composes the authority from the file and pins the file's record (`beta_policy_screening_g1.py:297-301`, through `native_g1_beta_authority_composer.py:32,36`). Its validator checks the authority again and requires the pinned record to match the embedded review (`beta_policy_screening_g1.py:144-148`).
  - The session authority checks it again (`native_g1_beta_session.py:146`).
  - The provider bundle and runtime require the session's authority to equal the spec's (`native_g1_beta_bundle.py:73-77`, `native_g1_beta_runtime.py:46-50`).
  - The G1 worker checks it twice, at preflight and against the scene plan (`native_g1_development_worker.py:137,313`).
- **`verified_site_episode`** is stood in for on a development run. The owner, 2026-11-07 00:05Z, in the library-aisle session, on how the library job's first G1 navigation run counts: "Dev check, then counts". It was made general in the owner's 13:43Z confirmation to the G1 session: every site and scene, results development-only, and the first run to pass every check counts as G1's verified navigation episode. The run proceeds as a development-only check, and no result can claim more than `development_only`. The spec (`beta_policy_screening_g1.py:136`), the session (`native_g1_beta_session.py:126-134`) and the authority's scope (`native_g1_beta_authority.py:119-133`) each fix the ceiling, with ranking and scene promotion off.
  - Nothing in this change writes a verified site episode or removes the item from the G1's `missing`.
  - Promoting the first run that passes every check to G1's first verified site episode is future work. It needs that run's receipt.

What the allowance changes:

- `website_locomotion_task.runnable_options` reads it for a walk with no manipulation, and the handoff hold records it.
- Each G1 navigation option drops the two `embodiment:` items, carries `claim_ceiling: development_only`, and records each item under `development_run_allowance.stood_in` as "stood in by the development-run allowance (development_only)", never satisfied, with its basis and decision. The hold's reason says so too.
- The hold still names `pipeline:navigation_source_identity` (until that job's own website intake receipt exists, `website_navigation_intake`) and any catalog gaps. A website refusal of the intake (an unconfirmed brief is `website_task_context_not_confirmed`) holds the job by that name.
- The body is still listed `unavailable` with both items.
- Manipulation (`rigid_pick_place`) options, routing and every other robot are unchanged.

### Funding both navigation rows

The owner's "two" (2026-11-07, confirmed to the G1 session at 13:43Z) needs no code change. Nothing limits a navigation site to one funded row:

- Registration lists both `_vision_navi` rows (`config/native_g1_shared_inputs.v1.json:21-22`) and registers each one (`native_g1_site_registration.py:517-522`).
- Capabilities bind each catalog row on its own (`beta_policy_screening_capabilities.py:279`).
- The plan selects rows in order while each cap fits the remaining budget, with no count per kind (`beta_policy_screening_plan.py:671-735`).
- The coordinator funds a waitlisted row whenever its cap fits both the site limit and its plan's budget (`beta_policy_screening_coordinator.py:919`).
- Two $4 rows fit the $30 site limit (`SITE_LIMIT`, `SITE_BUDGET_USD`). No cap changes.

## Claims

Everything stays `development_only`:
- one allocation, no retry, the independent watchdog and scoped closeout;
- live consent rechecks at planning and launch;
- pinned digests.

The instruction is the checkpoints' trained prompt, "Avoid obstacles and move to the yellow marked area.", recorded as such, never as the site's sentence. Obstacle contact is scored only once the sampler measures it. A walking robot always touches the floor, and the room is one collider, so the measurement must count contact on robot bodies other than the feet (the sensors are per robot body). Until that is measured, `no_obstacle_contact` stays unscored and is never read as clear. The first run is checked before any contact result is published.
