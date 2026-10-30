# Single-arm UR and FR3 Robotiq scripted simulation configurations

The exact new bodies are `universal_robots_ur5e_robotiq_2f85_sim` and
`universal_robots_ur10e_robotiq_2f85_sim`. Bare UR5e/UR10e and the prior dual UR5e
remain separate configurations. The existing single
`franka_research_3_robotiq_2f85_sim` gains an owned scripted candidate using its
existing model and seven-axis arm controller. There is no learned or vendor
policy claim for these three candidates.

## Sources and mounting

Both UR arm packages are pinned to Universal Robots' public description commit
`d4042c1f5123c86645cc5dbd650b4f461f697ee9`. Each uses its own `ur_type` and exact
source package subset. Robotiq is the same retained PickNik source at
`a74d007d8f2f06dc6a503ad21038ba869d4999a6`. Arm and tool bytes and their BSD-3-Clause
licenses remain separate, verified inventories; the generated assembly has its
own schema and digest. No source recipe, composition module or profile mapping
for the original 21 configurations is changed.

The UR source's existing wrist/flange/tool0 transforms are preserved. The pinned
UR-to-Robotiq adapter attaches at `tool0` with its authored 11 mm offset. The
published generic UR example supplies the nominal grasp frame 134 mm along the
tool Z axis with a pi yaw. This is an explicit simulation assembly, not a claim
of a manufacturer's calibrated installation, Dream endpoint compatibility or
collision clearance. Gripper actuation retains the source mimic mapping across
six named joints; closure is 0 to 0.7929 radians with the corresponding mimic
signs. Contacts bind the two declared finger-tip links.

## Controller and activation

The new `blueprint_scripted_pick_place_robotiq.v1` policy profile reuses the
unchanged owned pose servo with exact six/seven-axis and Robotiq source checks.
Its subclass re-derives the authored target plan for every sealed evaluation
cell. The existing scripted interface and typed evidence remain unchanged:
scripted controller calls never count as learned queries or prove task success.
The independent task scorer determines the outcome.

The three stable `blueprint_scripted_<body>` candidate IDs are visible before
installation with `prepared_robot_policy_registration_missing`. Existing fixed
`install-scripted-robot-baselines` registration enumerates all thirteen bodies,
verifies each actual published import and tool binding, and preserves an
individual reason when a body is absent or invalid. No caller-supplied endpoint,
model, source or arbitrary controller is accepted. Newly prepared source bodies
still need a separately authorized asset preparation batch; this code does not
modify or extend the active 21-body batch, its grant or its budget. No native
run is triggered by installing code or registering candidates.

The original ten scripted modules, policy identities and valid bindings remain
unchanged. The runtime configuration reader recognizes only the exact prior
21-body generator hash and reconstructs the complete original payload from
verified retained source/model bytes, including explicit GoFa modeled dynamics.
Changed cameras, resets, tooling, gains, source bytes and new-body attempts to
borrow that predecessor identity are refused. This is exact compatibility,
not acceptance of stale arbitrary hashes.

## Evidence and limits

`evidence/ur-single-source-expansion-2026-10-30.json` records CPU expansion from
retained pinned publisher sources for both UR variants, exact joint/tool
mapping and positive authored inertias. No network or checkpoint was used.
Software regressions exercise full source composition, synthetic-USD import
binding, exact previous-generation readback, all thirteen service/privileged
installer bindings, candidate replacement and publication, and sealed provider
bundles. The actual generic episode loop runs explicit Cartesian fake physics
for each new six/seven-axis profile and independently returns a failed score;
it establishes rebinding, action dimensions and evidence typing only.

Native Isaac import/reset/contact/episode execution and physical calibration
remain unverified for the new UR variants. Reachability, self-collision and
successful grasping must be established by their individual native results;
source expansion and script arrival do not establish those outcomes.
