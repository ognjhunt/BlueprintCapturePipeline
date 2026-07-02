# Robot profiles and task files: new robots and new tasks as data

Two JSON surfaces make the kitchen task-scaling lane extensible without code
changes: a **task spec file** (what to do) and a **robot profile** (who does
it). Placement, stance validation, reach gating, and the dry-render preview
all consume both.

## New task = a JSON entry

```json
[
  {
    "task_id": "microwave_door",
    "description": "Stand at the microwave and open the microwave door.",
    "required_target_terms": ["microwave", "door", "handle"]
  }
]
```

```bash
python -m blueprint_pipeline.kitchen_task_scaling_preflight \
  --out-dir output/my_preflight \
  --kitchen-usd <Collected_KitchenRoom/KitchenRoom.usd> \
  --task-file my_tasks.json --task microwave_door
```

Required keys: `task_id`, `description`, `required_target_terms`. Optional:
`scenario_id`, `zone`, `preferred_stance_distance_m`,
`stance_distance_candidates_m`. Leave the stance distances out unless you have
a tuned ladder — absent hints let the runner derive a robot-footprint-scaled
candidate ladder, so the same task file works for differently sized robots.
File entries merge over the built-in tasks by `task_id` (same id = override).

## New robot = a JSON profile

```json
{
  "robot_id": "my_tall_humanoid",
  "pelvis_height_m": 1.05,
  "footprint_half_extent_xyz": [0.15, 0.28, 0.85],
  "arm_span_m": 0.62,
  "shoulder_lateral_offset_m": 0.20,
  "shoulder_above_root_m": 0.42,
  "standing_distance_m": 0.60,
  "standoff_range_m": [0.4, 1.4]
}
```

Pass `--robot-profile-json my_bot.json` (or `--robot-id <registered id>`) to
`kitchen_task_scaling_preflight` or directly to
`scripts/run_isaac_g1_kitchen_parity_eval.py`. The runner's
`apply_robot_profile()` rescales every robot-scale constant — footprint,
pelvis height, shoulder geometry, the seed reach envelope, and the close-reach
standoff ceiling (which scales with `arm_span_m`). Manifests record the honest
`robot_profile_id`. Unknown JSON keys fail loudly (typo protection).

Full field list and defaults: `src/blueprint_pipeline/scene_placement/robot_profile.py`
(`RobotProfile`). Programmatic registration: `register_robot_profile()`.

## Semantics worth knowing

- Placement adapts to size: a bigger footprint stands farther out and gets a
  higher pelvis; stances that would clip furniture are rejected per-robot.
- The reach gate is honest per-robot: a robot whose nearest legal stance
  leaves the affordance beyond `arm_span_m + max_effector_to_affordance_m`
  fails `rendered seed arm can plausibly reach affordance` — that is a real
  "this robot cannot do this task here" answer, not a bug.
- Omitting a profile everywhere keeps byte-identical G1 behavior.

Proven examples (2026-07-02): the microwave task above passes all 10 local
preflight gates for the G1 and a tall-humanoid profile with different
auto-computed stances; an intentionally oversized 0.7 m-wide profile is
rejected by the reach gate after 96/98 stance candidates clip.
