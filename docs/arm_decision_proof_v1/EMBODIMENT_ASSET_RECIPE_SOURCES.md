# Pinned robot source recipes

The beta's existing job-to-evidence preparation can resolve public source
material for exact catalog bodies. This is source preparation, not a new
admission or review stage. The recipe data lives in
`src/blueprint_pipeline/_catalog_data/embodiment_asset_recipes.v1.json`.
Every source is pinned to an immutable repository commit. No compiled USD,
checkpoint digest, customer calibration or qualified physics is invented.
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

Primary sources inspected on 2026-10-21:

- [Universal Robots description and license](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/d4042c1f5123c86645cc5dbd650b4f461f697ee9/README.md): exact UR5e/UR10e source is BSD-3-Clause. The publisher's newer-model mesh exceptions are not generalized to these recipes. An arm source does not supply a task tool.
- [YAM station source](https://github.com/i2rt-robotics/i2rt/blob/120c3c81400171174604e503943f8d1ebc891058/i2rt/robot_models/station/yam_station_linear_4310_d405/README.md): manufacturer MIT source composes two YAM v1 arms, linear_4310 grippers and three D405 frames. Full precision authored extrinsics exist; customer site calibration and camera intrinsics remain separate.
- [TRLC-DK1 follower](https://github.com/robot-learning-co/trlc-dk1/blob/12f5368aefd0381461f2c7ffbb5611b4e8c90de9/urdf/follower/TRLC-DK1-Follower.urdf): six arm joints and two prismatic fingers. Repository Apache-2.0 and inherited notices do not grant rights to separately reserved external CAD exports.
- [Axol URDF](https://github.com/almond-bot/axol/blob/cd199a2f4c847c3a10594c6c2ed2250b4b4be675/almond_axol/kinematics/urdf/axol.urdf): manufacturer MIT source has fourteen arm joints, but both gripper joints are fixed. The small referenced STL blobs contain actual binary geometry; their size alone is not a Git LFS indication.
- [Official FANUC descriptions](https://github.com/FANUC-CORPORATION/fanuc_description/tree/fb40c9803a826ba68c7c8e28ba904a25efa7fcd2): exact CRX-10iA, CRX-10iA/L and M-710iC/45M packages declare Apache-2.0, with physical parameters in their macros. This route is distinct from NVIDIA's USD asset agreements and from hardware-driver access.
- [ROS-Industrial GP12](https://github.com/ros-industrial/motoman/blob/846fbbc57c60223f12bf4795ea6404982d57709e/motoman_gp12_support/package.xml): BSD-3-Clause package declaration and exact kinematic/mesh source. Its macro lacks inertial elements. GP20HL in the same repository cannot substitute for GP20.
- [ROS-Industrial GP25-20 / European GP20](https://github.com/ros-industrial/motoman/blob/846fbbc57c60223f12bf4795ea6404982d57709e/motoman_gp25_support/package.xml): the package explicitly identifies `YR-1-06VX25-20-A0*` and the GP20 operating manual. [PR 643](https://github.com/ros-industrial/motoman/pull/643) added that model; the recipe retains `gp25_20.xacro`, its macro, shared GP25 lower-arm meshes, GP25-20 upper-arm meshes, material package and BSD-3-Clause package declaration. Its macro has no inertials or application tool. This is not the default GP25 or GP20HL.
- [Manufacturer PiPER source](https://github.com/agilexrobotics/Agilex-College/tree/c2688be41e1bc99a9addd555f237a16bc839d936/isaac_sim/agx_arm_IsaacLab/IsaacLab_Data_Collection/source/agx_teleop/agx_description/agx_arm_urdf): standard PiPER plus gripper Xacro has six arm joints and two prismatic fingers; the local MIT notice is retained. H/L/X and Revo-hand variants are separate.
- [Manufacturer FR3 source](https://github.com/frankarobotics/franka_description/blob/7aeeddc449edf8d62b594f9e36a81da53e7796f9/robots/fr3/fr3.urdf.xacro): Apache-2.0 FR3 and Franka Hand; compose two explicit instances. Panda, FR3v2 and an example FR3-Duo mounting are not silently substituted.
- [Manufacturer Gen3 plus Robotiq 2F-85](https://github.com/Kinovarobotics/ros2_kortex/blob/b2aa46b29ae8fadd5b293c57e3500925c61eb8e8/kortex_description/robots/gen3_robotiq_2f_85.xacro): pinned seven-DOF, vision-enabled Gen3 configuration; retain both Kortex and bundled gripper BSD-3-Clause notices.
- [Manufacturer dual xArm](https://github.com/xArm-Developer/xarm_ros2/blob/ac736c92b02de7ca7e0be6f737d77d46707464d8/xarm_description/urdf/dual_xarm_device.urdf.xacro): BSD-3-Clause; both DOFs are explicitly seven and both grippers enabled. Default arm spacing is an example requiring an explicit mount binding.
- [Menagerie iiwa14 conversion](https://github.com/google-deepmind/mujoco_menagerie/blob/cda5236f98383f6fbaa4820c6b18dba892cd087f/kuka_iiwa_14/iiwa14.xml): exact MJCF conversion retains its model-specific Drake BSD-3-Clause notice. No end effector or manufacturer-supported Isaac integration is inferred.
- [Official Unitree models](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/g1_29dof_with_hand.urdf): BSD-3-Clause source for G1 with articulated hands. Exact Dex3/SONIC asset revision, checkpoint rights and a real site episode remain unavailable under the existing G1 record.
- [Standard Bots resources](https://apihelp.standardbots.com/6-additional-resources.html): manufacturer links URDF resources through Notion; exact immutable RO1/Core artifacts and reuse licenses were not verified. [External control](https://apihelp.standardbots.com/5-external-control-beta.html) requires separately enabled beta access.
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
| UR5e, two arms | Same exact arm source for each instance. | Distinct joint/link prefixes, two actual tools, explicit mount transforms and independent tool commands. |
| TRLC-DK1, two followers | Apache-2.0 repository; follower `joint1` through `joint6`, `gripper_left` and `gripper_right` are independent prismatic joints; terminal frame `tool0`. | Compose two followers with recorded mounts; retain per-asset notices, both finger-pair mappings and camera calibration. Reserved external CAD is not automatically licensed by the repository. |
| Almond Axol | MIT, an authored dual-arm body with seven revolute joints per side and source mounting transforms. | `left_gripper_0` and `right_gripper_0` are fixed joints to whole gripper meshes. Actual articulated gripper geometry, joints, limits and dynamics are required; renaming the fixed joints does not supply them. |
| FANUC CRX-10iA/L | Official Apache-2.0 Xacro, `J1` through `J6`, inertials on seven physical links. | Example top-level mount is 0.75 m above `world`; replace it with the recorded mount. `ee_link` is a frame, not a gripper. |
| FANUC M-710iC/45M | Official Apache-2.0 Xacro, `J1` through `J6`, inertials on seven physical links. | Exact application tool, payload, TCP and mount. This official description route has different terms from NVIDIA's USD distribution. |
| Yaskawa GP12 | BSD-3-Clause package declaration; `joint_1_s`, `joint_2_l`, `joint_3_u`, `joint_4_r`, `joint_5_b`, `joint_6_t`; `tool0`. | Source has no inertials or application tool. Do not synthesize measured dynamics from default density. |
| Yaskawa GP20 | Exact GP25-20 variant, BSD-3-Clause package declaration; same six joint-name suffixes as GP12 and terminal `tool0`. | Confirm `YR-1-06VX25-20-A0*`, supply qualified inertials and tool, and retain mesh notices. |
| YAM, two arms | MIT station source; `left_joint1..6` and `right_joint1..6`; each side has two prismatic `joint7/8` fingers. `left_base` is the root, `right_base_joint` supplies the authored second-arm mount. | Source camera meshes/frames are not calibrated simulated cameras. Bind finger commands, TCP, camera intrinsics, controller limits and reset. |
| PiPER, two arms | Local MIT notice; exact standard PiPER Xacro includes six arm joints and two prismatic fingers. | Resolve recorded package aliases, compose two instances with distinct prefixes and explicit mounts, then bind both tool channels. |
| Standard Bots RO1 | Official manual supplies a URDF resource link. | Exact downloadable model/mesh closure, immutable version and reuse terms remain unverified. Do not label it a confirmed private file solely because a web reader cannot open the Notion link. |
| Standard Bots Core | Official manual supplies a URDF resource link. | Same artifact/access gaps, plus exact Core identity. RO1 geometry is not substituted for Core. |
| FR3, two arms | Apache-2.0 manufacturer FR3 plus Franka Hand. Hand macro has `finger_joint1/2`, with the second mimicking the first, and a `hand_tcp` frame after prefix expansion. | Two explicit FR3 instances and exact mounts; preserve mimic behavior and independent hand commands. FR3-Duo example transforms are a separate mount. |
| Gen3, two arms | BSD-3-Clause seven-DOF vision Gen3 plus bundled Robotiq 2F-85; retain the gripper's own notice. | Expand exact source and read back arm/gripper names, then mount two instances and bind both hands/TCPs. The bare NVIDIA seven-DOF arm is not this complete tool configuration. |
| xArm7, two arms | BSD-3-Clause dual Xacro with both seven-DOF arms and grippers explicitly enabled, prefixes `L_` and `R_`. | Its default translations are `(0,0,0)` and `(0,1,0)` m; replace those example transforms with explicit configuration values and verify both grippers. |
| iiwa14, one arm | Model-specific BSD-3-Clause Menagerie MJCF conversion. | A licensed application tool/TCP, actual MJCF-to-USD import and controller/reset validation remain required. |

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
an expanded dual xArm; it cannot turn Axol's fixed gripper meshes into actuated
hands. Both modes require two explicit SI mounting transforms. There is no
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
