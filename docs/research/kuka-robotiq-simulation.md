# Source-specific iiwa14 + Robotiq simulation assembly

`kuka_lbr_iiwa_14_robotiq_2f85_sim` adds one seven-axis, single-arm scripted
pick/place candidate. The original bare iiwa14 and the preceding 30 runtime
profiles, 36 recipes, 111 catalog rows and 20 scripted policy identities remain
unchanged. The catalog row retains an individual missing-registration reason
until its imported model is registered. Preparation remains capped at 24 bodies
per separately funded batch; this addition launches nothing.

The exact arm source is [Menagerie cda5236](https://github.com/google-deepmind/mujoco_menagerie/tree/cda5236f98383f6fbaa4820c6b18dba892cd087f/kuka_iiwa_14)
(BSD3, retaining Drake attribution). The licensed tool is [PickNik a74d007](https://github.com/PickNikRobotics/ros2_robotiq_gripper/tree/a74d007d8f2f06dc6a503ad21038ba869d4999a6/robotiq_description)
(BSD3). Both component inventories, notices and original source bytes remain
separately verifiable. An exact-source bridge handles only this pinned MJCF,
not arbitrary MJCF input. An identity fixed mount places the tool at the
source `attachment_site` 45 mm beyond `link7`; the nominal grasp frame adds
134 mm and yaw pi. No adapter geometry, hardware fit or vendor policy is claimed.

The bridge retains all 8 arm body inertias and COMs, including nonidentity
inertial orientations, 7 joint frames/ranges, 15 visual instances over 13 unchanged
OBJ assets, 46 source collision spheres, 7 source contact exclusions, and the
source home reset `[0,0.785398,0,-1.5708,0,0,0]` radians. The tool retains its 9
inertial bodies and 6 independent/mimic jaw joints. Authored combined mass is
31.53086427304 kg (30.61 kg arm plus 0.92086427304 kg tool); these are source model
values, not measurements. MJCF specular/shininess has no exact standard URDF
representation; source RGBA and geometry are retained and that difference is
explicitly recorded.

The source XML contains no motor force or speed caps, despite a broader README
statement. Generated simulation arm effort caps are approximately
`[316.1600,210.4772,136.1158,74.3140,37.6520,10.9337,5.8750]` Nm. They use an
orientation-independent triangle-inequality bound on descendant authored
masses, COM radii and inertia, a hypothetical 1 kg payload at the nominal grasp,
a 2x gravity moment margin and 1 rad/s² acceleration allowance. No payload mass
is added to the body. Arm speed is explicitly capped at 1 rad/s; source PD gains
2000/200 and authored tool 50 Nm/0.5 rad/s caps are retained. These settings are
unvalidated simulation assumptions, not manufacturer motor ratings or physical
feasibility evidence.

The official URDF importer keeps fixed links unmerged. A separate post-import
step applies and reads back the 17 full inertias, 13 force-drive caps, angular
speed limits, exact 46 spheres, seven exclusions and enabled self-collisions.
USD angular speed is authored in degrees/s and converted back to radians/s,
as demonstrated by [IsaacSim6.0.1 articulation.py](https://github.com/isaac-sim/IsaacSim/blob/045ca8b59622b99a408092124377c66346e8d9c2/source/extensions/isaacsim.core.experimental.prims/python/impl/articulation.py#L3023).
Imported arm collision approximations and extra authored arm filters are
replaced by those exact source spheres/exclusions; tool mesh treatment stays
with the recorded importer. Default engine adjacent-joint filtering remains
an explicit engine difference. The existing v2 task path enables self-collision
but disables gravity for fixed arms; this is not full MuJoCo dynamics parity.
The separate native controller smoke disables collisions and gravity and must
never be interpreted as task-contact validation.

The new controller has a separate source identity and reuses the unchanged
scripted pose servo with 7 joint DOFs and the source-bound Robotiq grasp frame.
It reads privileged simulator state and authored task targets. It evaluates no
learned/vendor policy. Local MuJoCo FK/inertia/sphere checks, USD serialization
and tamper tests, sealed bundle tests and generic loop tests are software proof.
An actual Isaac import/controller/camera/task result remains to be obtained on
a separately bounded native batch; this change supplies none of those results.

The final retained real-source CPU replay used 49 files (10,867,312 bytes).
Across zero, home and a nontrivial joint pose, maximum arm transform and world
inertia error versus MuJoCo 3.11 was 1.23e-15. The import/binding replay used an
explicit CPU USD/Physx schema double; it supplies no installed Isaac proof.
