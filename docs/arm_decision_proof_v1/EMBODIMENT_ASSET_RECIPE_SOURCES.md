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
