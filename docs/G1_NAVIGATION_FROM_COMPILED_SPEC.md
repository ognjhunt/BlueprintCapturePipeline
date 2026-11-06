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
| A1 | Navigation goal and authority from the compiled spec. `native_g1_navigation_goal` accepts the `goal_region_navigation` kind. A machine-composed authority `v2` (owner decision 4) binds the site's **confirmed** brief, the compiled spec and the goal, recorded as a disclosed assumption, `development_only`. `website_navigation_packet` builds the robot-free base navigation packet request: the scene's appearance and collision, the start, the goal, the yellow marker and the trained prompt. The handoff writes it, and the hold then names only the pieces still missing. | this session | in progress |
| B | The arena runtime runs a scene with no task object, behind the `goal_region_navigation` kind. The rigid and articulated paths are unchanged. This covers the scene plan, runtime contract, spawn and readback. The G1 navigation sampler records per-step robot–scene contact force. | this session, if the G1 session agrees (asked 18:05Z) | waiting |
| C | G1 lane. Prepare and the execution spec use the compiled start as the stance; the navigation cells perturb the robot's start, not an object. The registration contract, the catalog's two `_vision_navi` rows (`supported_task_kinds`), capabilities and plan accept the kind. Registration of the rows comes from the A1 packet. | this session, after the G1 session's spawn-clearance PR and #3426 | blocked on those |
| D | One paid G1 navigation session on the library-aisle site: $4 cap, 4 cells, inside the $30 site limit (owner decision 3). | controller | after A1–C deploy |

## Prerequisites outside code

- **The site confirms its brief.** The library-aisle job's `website_task_context.json` reads `confirmed: false`. Decision 4 rests the goal authority on a *confirmed* brief, so the authority refuses an unconfirmed one.
- **The Vast top-up** (owner): credit was $7.07 against the $11 page line.
- **G1 runtime source restore and registration** (G1 session; #3426 fixes the restore's permission error).

## Claims

Everything stays `development_only`:
- one allocation, no retry, the independent watchdog and scoped closeout;
- live consent rechecks at planning and launch;
- pinned digests.

The instruction is the checkpoints' trained prompt, "Avoid obstacles and move to the yellow marked area.", recorded as such, never as the site's sentence. Obstacle contact is scored only once the sampler measures it. A walking robot always touches the floor, and the room is one collider, so the measurement must count contact on robot bodies other than the feet (the sensors are per robot body). Until that is measured, `no_obstacle_contact` stays unscored and is never read as clear. The first run is checked before any contact result is published.
