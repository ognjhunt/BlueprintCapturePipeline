# Pinned robot source recipes

The beta's existing job-to-evidence preparation can resolve public source
material for exact catalog bodies. This is source preparation, not a new
admission or review stage. The recipe data lives in
`src/blueprint_pipeline/_catalog_data/embodiment_asset_recipes.v1.json`.
Every source is pinned to an immutable repository commit. No compiled USD,
checkpoint digest, customer calibration or qualified physics is invented.
Prioritize the RobotTeam companies and their task-specific configurations;
see the [company/policy access map](../research/deployment-company-policy-priorities-2026-10-29.md).
An imported source remains `source_asset_not_qualified`; the existing runtime,
rights, policy, independent grading and resource authority still decide whether
an episode can run. No provider or robot was operated to collect these recipes.

`include_paths` selects repository-relative files or directories, with their
license notices. `package_roots` records explicit package URI/Xacro aliases;
it grants no filesystem escape. `import_arguments` records source expansion
arguments, not deployment settings. A missing reference, Git LFS pointer,
unavailable importer or absent physical parameter remains a named limitation.
`source_arm_count` distinguishes a combined station from a single arm that still
needs composition; the recipe's `arm_count` is the requested complete body.

The 16 catalog identities retain their exact variants. Five legacy compatibility
identities and the existing DROID, G1 and Atlas entries have explicit lookup
results too. DROID uses `existing_runtime`: preserve its already pinned Arena
asset, camera, reset and action contract instead of substituting a generic Panda.
`source_available` means public source material exists; it does not mean
Blueprint has approved every intended use or executed the imported robot.

## Prepare every available source

```bash
python -m blueprint_pipeline.deployment_embodiment_assets \
  --all-available --destination /path/to/robot-sources
```

The parent directory must exist. The command attempts every public recipe and
prints one result per catalog entry: `source_prepared`, `preparation_failed`,
`source_unavailable`, or `existing_runtime`. One failed source does not stop
later robots. A retry verifies and reuses intact bundles; it never overwrites
changed retained files. Exit status 1 means at least one preparation failed;
unavailable source entries still report their specific reasons. Downloads use
the shared HTTPS transport, including the managed cloud's required proxy,
with fixed GitHub origins, bounded response sizes and redirects disabled.

Source preparation does not allocate compute or execute policies. Each robot
still needs its applicable tool, controller, observation/action mapping, scene
placement and reset. The task planner should enumerate compatible exact
robot–policy configurations, execute within the existing budget, and retain an
individual result or reason for every remaining candidate.

The 2026-10-29 preparation audit corrected three source issues: Axol now uses
its articulated standalone model; Kinova uses the complete ROS Kortex model
with its bundled Robotiq meshes; both xArm recipes retain the end-tool meshes.
The importer also distinguishes URDF mesh paths from embedded MuJoCo settings,
ignores Gazebo plugin declarations that the Isaac URDF importer does not load,
and ignores inactive package references in XML comments during Xacro expansion.
Retained publisher bytes and notices remain unchanged.

The completed audit downloaded and verified 1,135 files (423,261,298 bytes)
across all 19 source-backed configurations, then expanded Xacro where needed
and checked every referenced model dependency. All 19 passed those checks.
The batch command returned 19 prepared sources, four unavailable complete
sources with reasons, and the existing DROID runtime entry. These are
configuration counts, including single/dual variants, not 19 distinct robot
families or executed policies. The iiwa14 MJCF additionally completed 100 CPU
MuJoCo steps with finite state. Isaac was not installed in the audit environment,
so no Isaac conversion, task episode, or policy-compatibility claim follows
from this audit. Yaskawa's absent inertials remain explicit; chosen simulation
estimates can support exploratory execution without becoming measured dynamics.

Primary sources inspected on 2026-10-21:

- [Universal Robots description and license](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/d4042c1f5123c86645cc5dbd650b4f461f697ee9/README.md): exact UR5e/UR10e source is BSD-3-Clause. The publisher's newer-model mesh exceptions are not generalized to these recipes. An arm source does not supply a task tool.
- [YAM station source](https://github.com/i2rt-robotics/i2rt/blob/120c3c81400171174604e503943f8d1ebc891058/i2rt/robot_models/station/yam_station_linear_4310_d405/README.md): manufacturer MIT source composes two YAM v1 arms, linear_4310 grippers and three D405 frames. Full precision authored extrinsics exist; customer site calibration and camera intrinsics remain separate.
- [TRLC-DK1 follower](https://github.com/robot-learning-co/trlc-dk1/blob/12f5368aefd0381461f2c7ffbb5611b4e8c90de9/urdf/follower/TRLC-DK1-Follower.urdf): six arm joints and two prismatic fingers. Repository Apache-2.0 and inherited notices do not grant rights to separately reserved external CAD exports.
- [Axol standalone URDF](https://github.com/almond-bot/axol/blob/3ccc69cf3adaea09845456148d6a4f16ef059f27/almond_axol/kinematics/urdf/axol_standalone.urdf), rechecked 2026-10-29: manufacturer MIT source has fourteen revolute arm joints and four prismatic finger joints, with one mimic per hand, inertials and collision meshes. All 60 referenced STL paths exist at the pinned commit under `meshes/jelly` and `meshes/standalone`; retain those directories and the root MIT license. The old `axol.urdf` remains fixed-gripper, but is not the selected recipe. `axol_jelly`, `axol_owl` and `axol_ox_cart` are separate body variants; the separate Mink IK model has additional inherited mesh notices and is not this physical-contact source.
- [Official FANUC descriptions](https://github.com/FANUC-CORPORATION/fanuc_description/tree/fb40c9803a826ba68c7c8e28ba904a25efa7fcd2): exact CRX-10iA, CRX-10iA/L and M-710iC/45M packages declare Apache-2.0, with physical parameters in their macros. This route is distinct from NVIDIA's USD asset agreements and from hardware-driver access.
- [ROS-Industrial GP12](https://github.com/ros-industrial/motoman/blob/846fbbc57c60223f12bf4795ea6404982d57709e/motoman_gp12_support/package.xml): BSD-3-Clause package declaration and exact kinematic/mesh source. Its macro lacks inertial elements. GP20HL in the same repository cannot substitute for GP20.
- [ROS-Industrial GP25-20 / European GP20](https://github.com/ros-industrial/motoman/blob/846fbbc57c60223f12bf4795ea6404982d57709e/motoman_gp25_support/package.xml): the package explicitly identifies `YR-1-06VX25-20-A0*` and the GP20 operating manual. [PR 643](https://github.com/ros-industrial/motoman/pull/643) added that model; the recipe retains `gp25_20.xacro`, its macro, shared GP25 lower-arm meshes, GP25-20 upper-arm meshes, material package and BSD-3-Clause package declaration. Its macro has no inertials or application tool. This is not the default GP25 or GP20HL.
- [Manufacturer PiPER source](https://github.com/agilexrobotics/Agilex-College/tree/c2688be41e1bc99a9addd555f237a16bc839d936/isaac_sim/agx_arm_IsaacLab/IsaacLab_Data_Collection/source/agx_teleop/agx_description/agx_arm_urdf): standard PiPER plus gripper Xacro has six arm joints and two prismatic fingers; the local MIT notice is retained. H/L/X and Revo-hand variants are separate.
- [Manufacturer FR3 source](https://github.com/frankarobotics/franka_description/blob/7aeeddc449edf8d62b594f9e36a81da53e7796f9/robots/fr3/fr3.urdf.xacro): Apache-2.0 FR3 and Franka Hand; compose two explicit instances. Panda, FR3v2 and an example FR3-Duo mounting are not silently substituted.
- [Manufacturer Gen3 plus Robotiq 2F-85](https://github.com/Kinovarobotics/ros_kortex/blob/c3280d8a6a6b5d96590d51538e3f265710a6c77d/kortex_description/robots/gen3_robotiq_2f_85.xacro), rechecked 2026-10-29: exact seven-DOF, vision-enabled Gen3 configuration in the ROS1 source includes complete gripper macros and meshes. All 18 distinct arm-variant/gripper mesh paths referenced by those macros exist at the pin; the eight used gripper DAE files have no external content references. Retain the root BSD-3-Clause and bundled [Robotiq BSD-2-Clause notice](https://github.com/Kinovarobotics/ros_kortex/blob/c3280d8a6a6b5d96590d51538e3f265710a6c77d/kortex_description/grippers/robotiq_2f_85/LICENSE). The previously selected ROS2 wrapper depends on an external `robotiq_description` package and did not provide a self-contained gripper source. A commented historical include in the ROS1 macro is not an active package dependency.
- [Manufacturer dual xArm](https://github.com/xArm-Developer/xarm_ros2/blob/ac736c92b02de7ca7e0be6f737d77d46707464d8/xarm_description/urdf/dual_xarm_device.urdf.xacro): BSD-3-Clause; both DOFs are explicitly seven and both grippers enabled. Default arm spacing is an example requiring an explicit mount binding.
- [Menagerie iiwa14 conversion](https://github.com/google-deepmind/mujoco_menagerie/blob/cda5236f98383f6fbaa4820c6b18dba892cd087f/kuka_iiwa_14/iiwa14.xml): exact MJCF conversion retains its model-specific Drake BSD-3-Clause notice. No end effector or manufacturer-supported Isaac integration is inferred.
- [Official Unitree models](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/g1_29dof_with_hand.urdf): BSD-3-Clause source for G1 with articulated hands. Exact Dex3/SONIC asset revision, checkpoint rights and a real site episode remain unavailable under the existing G1 record.
- [Standard Bots Core URDF](https://github.com/standardbots/ros2-realtime-api/blob/8a2164e041d3a4e75b2eeffe6351f3c89b292f5c/src/robot_urdfs/core.urdf), rechecked 2026-10-29: official public URDF includes inertials and references 16 visual/collision meshes under `package://standard_bots_description/meshes/sbot/ro1`. The pinned repository contains neither those meshes nor an explicit reuse license. A community copy contains RO1 meshes, but its [description package declares Proprietary](https://github.com/murra133/pickNplace/blob/de306a95fec0890220420ad1ad784b100743a9b0/standard_bots_description/package.xml), despite the repository's root MIT notice. The [manufacturer's RO1 product URL](https://standardbots.com/ro1) now names Core; bind the exact generation before using a model across those catalog identities. Missing mesh closure and reuse terms keep the complete source recipe unavailable; the URDF itself is public. [External robot control](https://github.com/standardbots/ros2-realtime-api/blob/8a2164e041d3a4e75b2eeffe6351f3c89b292f5c/README.md) is a separate API enablement concern.
- [ROS-Industrial GoFa variants](https://github.com/ros-industrial/abb/tree/45f4769d826cf3ac62a65495f2db67b78b0c81df/abb_crb15000_support): Apache-2.0 source distinguishes 5/0.95 and 12/1.27; the generic legacy ID cannot choose a variant.
- [Atlas manufacturer information](https://bostondynamics.com/products/atlas/): product information does not establish an exact Blueprint-accessible licensed asset, controller or policy API. Historical Atlas and Spot assets are not substitutes.

For all multi-arm recipes, independent tool control, collision-aware mounting,
calibrated cameras and actual environment stepping remain necessary. The existing
independent task scorer must judge object/task state; a controller's success
response or an imported robot is never an executed result. Existing fixed-pair
Arm Decision Proof definitions and its physical-validation authority are unchanged.

## Source audit for the 16 configurations

This matrix describes the pinned source, not a completed import or task run.
For expanded Xacro and imported USD, the actual read-back names remain the
authority; prefix templates below must never be guessed into a runnable manifest.

| Configuration | Source and tool facts | Remaining asset/configuration work |
| --- | --- | --- |
| UR5e, one arm | Exact six-DOF manufacturer arm Xacro, BSD-3-Clause. | No application gripper in the arm source; bind a licensed tool, TCP, controller and site mount. |
| UR5e, two arms | Exact manufacturer arm source plus separately pinned BSD-3-Clause PickNik Robotiq2F85 tools and UR adapters; both publisher licenses are retained. | Generic composition has distinct prefixes, independent tools and explicit simulated mounts/TCPs; site calibration and Dream hardware/policy correspondence remain unestablished. |
| TRLC-DK1, two followers | Apache-2.0 repository; follower `joint1` through `joint6`, `gripper_left` and `gripper_right` are independent prismatic joints; terminal frame `tool0`. | Compose two followers with recorded mounts; retain per-asset notices, both finger-pair mappings and camera calibration. Reserved external CAD is not automatically licensed by the repository. |
| Almond Axol | MIT standalone torso, fourteen revolute arm joints, four prismatic finger joints and two mimic couplings; source collision geometry and optical camera frames. | Bind independent `left_finger_1_joint` / `right_finger_1_joint` commands and their mimics, read opening from actual geometry, apply custom drive gains and bind cameras/reset. Non-arm masses, inertia assumptions and some limits are publisher estimates, not measured robot validation. |
| FANUC CRX-10iA/L | Official Apache-2.0 Xacro, `J1` through `J6`, inertials on seven physical links. | Example top-level mount is 0.75 m above `world`; replace it with the recorded mount. `ee_link` is a frame, not a gripper. |
| FANUC M-710iC/45M | Official Apache-2.0 Xacro, `J1` through `J6`, inertials on seven physical links. | Exact application tool, payload, TCP and mount. This official description route has different terms from NVIDIA's USD distribution. |
| Yaskawa GP12 | BSD-3-Clause package declaration; `joint_1_s`, `joint_2_l`, `joint_3_u`, `joint_4_r`, `joint_5_b`, `joint_6_t`; `tool0`. | Source has no inertials or application tool. Do not synthesize measured dynamics from default density. |
| Yaskawa GP20 | Exact GP25-20 variant, BSD-3-Clause package declaration; same six joint-name suffixes as GP12 and terminal `tool0`. | Confirm `YR-1-06VX25-20-A0*`, supply qualified inertials and tool, and retain mesh notices. |
| YAM, two arms | MIT station source; `left_joint1..6` and `right_joint1..6`; each side has two prismatic `joint7/8` fingers. `left_base` is the root, `right_base_joint` supplies the authored second-arm mount. | Source camera meshes/frames are not calibrated simulated cameras. Bind finger commands, TCP, camera intrinsics, controller limits and reset. |
| PiPER, two arms | Local MIT notice; exact standard PiPER Xacro includes six arm joints and two prismatic fingers. | Resolve recorded package aliases, compose two instances with distinct prefixes and explicit mounts, then bind both tool channels. |
| Standard Bots RO1 | Official public Core URDF is pinned; it refers to RO1 meshes, and the manufacturer now names Core at its RO1 product URL. | Bind the exact RO1/Core generation and supply the 16 referenced meshes with reuse rights. Official repository has no mesh closure or explicit license; a community description package is marked Proprietary. |
| Standard Bots Core | Official public `core.urdf` includes inertials and 16 RO1 visual/collision mesh references. | Supply missing mesh closure and explicit reuse rights, then bind exact generation, application tool and controller. Public URDF availability does not establish a complete licensed source recipe or runtime integration. |
| FR3, two arms | Apache-2.0 manufacturer FR3 plus Franka Hand. Hand macro has `finger_joint1/2`, with the second mimicking the first, and a `hand_tcp` frame after prefix expansion. | Two explicit FR3 instances and exact mounts; preserve mimic behavior and independent hand commands. FR3-Duo example transforms are a separate mount. |
| Gen3, two arms | BSD-3-Clause seven-DOF vision Gen3 plus complete bundled Robotiq 2F-85 in the pinned ROS1 source; retain the gripper's BSD-2-Clause notice. | Expand exact source and read back arm/gripper names, then mount two instances and bind both hands/TCPs. The bare NVIDIA seven-DOF arm and an ROS2 wrapper with missing external gripper meshes do not supply this complete source configuration. |
| xArm7, two arms | BSD-3-Clause dual Xacro with both seven-DOF arms and grippers explicitly enabled, prefixes `L_` and `R_`. | Its default translations are `(0,0,0)` and `(0,1,0)` m; replace those example transforms with explicit configuration values and verify both grippers. |
| iiwa14, one arm | Model-specific BSD-3-Clause Menagerie MJCF; registered bare-arm native joint/reset/controller/camera configuration. | Actual MJCF-to-USD import and native dynamics/episode evidence remain required. Grasping additionally needs a licensed application tool/TCP and compatible policy. |

## Relocatable prepared assets

`deployment_embodiment_asset_import` uses the already installed official Isaac
converter; it installs nothing and allocates no worker. Composed USD layers are
flattened. Flattening alone does not inline textures or other content, so any
remaining publisher content is retained inside the prepared directory with its
path hierarchy and notices. Runtime references are relative and checked before
publication and again at the actual published location. OBJ/MTL, COLLADA and
glTF sidecars are checked as content dependencies too; unbound URLs, absolute
runtime references, path escapes and symlinks refuse preparation.

The existing `asset-import.json` now contains `dependency_closure`:

```json
{
  "schema_version": "embodiment_asset_dependency_closure.v1",
  "model_path": "robot.usda",
  "files": [{"path": "robot.usda", "sha256": "sha256:<actual bytes>", "size_bytes": 123}],
  "closure_digest": "sha256:<actual canonical closure>"
}
```

Those illustrative digests and sizes are not usable artifacts. The real receipt
has measured values. `prepared_directory_self_contained` means the root and
its retained closure can move together; `single_file_self_contained` is true
only when no additional content file is needed. The original source bundle
remains provenance, but is not needed to resolve the prepared runtime's paths.
An operator can recheck saved bytes with
`verify_prepared_asset_directory(path)`; it verifies the receipt, complete file
inventory, model identity and current local content references with bounded
reads, without network resolution or Isaac startup. The existing public
one-file native preflight remains one-file-only; a prepared directory uses the
operator-owned dependency-closure staging path. Neither verifier supplies
missing gripper actuation, calibration, rights admission or executed evidence.

## Explicit bimanual source composition

`deployment_embodiment_station` prepares a station URDF from retained publisher
source. `two_instances` copies one actual arm description with distinct link,
joint, mimic, transmission, actuator and material names. All collision geometry,
inertials, axes, limits and existing finger joints remain source-derived.
`remount_existing_pair` preserves the names and structure of an authored pair,
changing only its declared fixed base mounts. This covers the YAM station and
an expanded dual xArm. Axol's standalone source already supplies both arms and
actuated fingers; preserve that body's source frames when binding a station.
Both modes require two explicit SI mounting transforms. There is no
arm-spacing default, inferred physical installation or placeholder gripper.

For example, the following parameters preserve the YAM station's published
left-base coordinate origin and its explicit `right_base_joint` translation
of `(0, -0.61, 0)` metres. The station coordinate basis is a simulation assembly
choice. It is not an observed customer workcell pose or camera calibration.

```json
{
  "schema_version": "embodiment_station_parameters.v1",
  "embodiment_id": "i2rt_yam_bimanual",
  "composition_mode": "remount_existing_pair",
  "station_root_link": "blueprint_station",
  "station_from_source_root": {"translation_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
  "arms": [
    {"name": "left", "prefix": null, "source_base_link": "left_base", "source_mount_joint": null,
     "station_from_base": {"translation_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}},
    {"name": "right", "prefix": null, "source_base_link": "right_base", "source_mount_joint": "right_base_joint",
     "station_from_base": {"translation_m": [0, -0.61, 0], "quaternion_wxyz": [1, 0, 0, 0]}}
  ]
}
```

For single-arm sources (PiPER, UR5e, FR3, Gen3 and TRLC), use `two_instances`,
omit `station_from_source_root`, set each actual source root link, set
`source_mount_joint` to null, and supply distinct nonempty prefixes plus both
actual chosen simulation mounts. Source roots must come from the expanded
URDF; an informal product-family name is not a link binding. The existing
dual xArm source instead uses `remount_existing_pair` with actual fixed mount
joint names read from its expanded URDF, and explicit source-root placement.
Every chosen tool still needs its real source asset; the UR5e arm file alone
does not acquire a gripper by being duplicated.

Save exact parameters to a JSON file, then prepare the source without Isaac:

```bash
python -m blueprint_pipeline.deployment_embodiment_station \
  --source-bundle /path/to/yam-source-bundle \
  --parameters /path/to/explicit-station-parameters.json \
  --output /path/to/new-station-source
```

The output retains `robot.urdf`, a zero-argument `station.py` generator,
`parameters.json`, the complete source closure and `station-source.json` with
source, generator, kernel, parameters and output identities. Xacro uses the
existing guarded expander before composition; an absent Xacro runtime is a
named missing dependency. A source supplied only as MJCF is not silently
converted or fabricated by this URDF composer.

The installed official importer can perform the same explicit composition
before conversion by adding `--station-parameters <json>` to its existing
command. Its `station_composition` receipt retains that full source assembly;
`composed_arm_count` is separate from original `source_arm_count`. A retry with
different mounts refuses reuse of the old preparation. The resulting USD
still needs the normal actual gripper/controller/IK/camera/reset bindings and
independently scored simulator episode before it is runnable evidence.

## KUKA iiwa14 native configuration

`deployment_embodiment_runtime_configuration` now accepts the exact pinned
`kuka_lbr_iiwa_14` MJCF source. It resolves nested MJCF default classes and
retains the seven hinge names, radian limits, axes, body transforms, authored
mass/inertia, affine PD values (2000/200), and publisher `home` keyframe.
The original MJCF remains the input to Isaac's `MjcfConverter`; no replacement
URDF or estimated inertia is generated. A source-model digest and metadata
generator digest bind the configuration alongside the existing source bundle
and generated-file inventory. Existing URDF configuration receipts keep their
original model/hash fields.

The manifest uses native joint-position control, `link7` as the bare body
frame, no grippers, and no Cartesian arm/tool grouping. The source's
`attachment_site` at 0.045 m remains source metadata, not a declared installed
tool center point. The 1 rad/s velocity cap and overhead/wrist camera rig are
explicit simulation settings; the source PD mapping is not measured dynamics.
The configuration retains `execution_authorized:false` and
`policy_compatibility_established:false`. Native imported-USD structure and
dependency checks still run before the resulting binding can be retained.

CPU verification against the retained MJCF and MuJoCo confirms the source
joint/body/inertial/reset metadata. Development fixtures cover the existing
MJCF converter boundary and real USD structure readback. These checks do not
claim a GPU import, a policy execution, or task success.


## Dual UR5e with public Robotiq2F85 tools

`universal_robots_ur5e_bimanual` now combines the existing official UR5e source
pin with PickNik's public Robotiq2F85 and actual UR adapter at
[`a74d007d8f2f06dc6a503ad21038ba869d4999a6`](https://github.com/PickNikRobotics/ros2_robotiq_gripper/tree/a74d007d8f2f06dc6a503ad21038ba869d4999a6).
Both original source bundles and their BSD-3-Clause notices are retained
separately. A composite receipt binds both immutable pins, each source inventory,
all notices, the generated assembly and its generator hash. It does not rewrite
publisher bytes or relabel generated assembly XML as publisher source.

The source UR adapter supplies its visual/collision geometry and 11 mm tool-side
offset. The gripper retains its six moving joints, authored inertias, five signed
mimic relationships, source 0.7929 rad close setting and finger-tip contact links.
The explicit grasp frame is 134 mm along gripper-base Z with a pi yaw, following
[PickNik's pinned UR example](https://github.com/PickNikRobotics/moveit_pro_ur_ws/blob/bab77dd0862bd6dea828ec41fb60b39d22c6f2c0/src/ur_mock/description/picknik_ur.xacro).
This assembly omits that example's physical camera spacer; its cameras and dual
base spacing are declared simulation settings. The two instances receive
independent link/joint prefixes and gripper commands.

The actual retained source closure contains 55 files (13,447,471 bytes). Xacro
expansion and every referenced mesh/texture dependency verified locally. CPU
MuJoCo controller isolation moved both arms and all twelve gripper joints with
zero untargeted-tool movement and repeatable resets. It retained source
inertias and collision geometry, omitted visuals, and disabled contacts and
gravity. No Isaac import or policy/task result is claimed by these checks.
This generic public assembly does not establish Dream's hardware, calibration
or policy compatibility. PickNik maintains the tool package; its README says
it is not maintained or sponsored by Robotiq.

## Single FR3 with Robotiq2F85 simulation tool

`franka_research_3_robotiq_2f85_sim` is a separate single-arm assembly for
experiments with the public OpenWAM Franka EEF checkpoint. It uses official
[FR3 source](https://github.com/frankarobotics/franka_description/tree/7aeeddc449edf8d62b594f9e36a81da53e7796f9)
with `hand=false, ee_id=none` and the separately pinned
[PickNik Robotiq2F85 source](https://github.com/PickNikRobotics/ros2_robotiq_gripper/tree/a74d007d8f2f06dc6a503ad21038ba869d4999a6).
Retain the FR3 Apache-2.0 license and NOTICE and the Robotiq package BSD-3-Clause
license. Source files remain unchanged in independent component bundles. The
generated wrapper and its assumptions have their own sealed identity.

The [checkpoint card](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Real-Single-Arm-Franka/blob/e75522eb38f9b4148d5ecb985ab80c490ddb3854/README.md)
and paper identify FR3 and a Robotiq parallel gripper, without exact SKU, mount
or TCP calibration. This variant selects 2F85, fixes its base directly to
`fr3_link8` with zero translation/rotation, and retains the nominal 134mm/yaw-pi
`grasp_link` convention from the existing pinned public PickNik example. It
does not model an unpublished adapter or camera spacer. Those choices and
the generic overhead/wrist camera rig are simulation assumptions, not a
publisher hardware match. The existing dual-FR3/Franka Hand and Panda/DROID
identities are unchanged. Policy observation/calibration adaptation is separate.

The retained 2026-10-30 source audit prepared 71 files / 31,773,064 bytes from
verified cached publisher files and expanded all dependencies. It found seven
FR3 arm joints, six actuated/mimicked Robotiq joints, one tool channel and
authored positive inertias on all 13 moving links. A CPU MuJoCo 3.11 controller
isolation moved the arm and all six tool joints and repeated reset, with
contacts/gravity disabled. This establishes neither Isaac import, task contact,
GPU policy inference nor task success. The new 19th preparation option does not
alter any previously sealed 18-body grant.
