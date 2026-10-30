# GP12 and GP20 with Robotiq2F85: generic simulation assemblies

The distinct `yaskawa_gp12_robotiq_2f85_sim` and
`yaskawa_gp20_robotiq_2f85_sim` bodies combine retained ROS-Industrial arm
geometry with the licensed PickNik Robotiq2F85. They have a named six-joint
arm, six source gripper joints, two contact links, an explicit reset and two
simulated cameras. The registered controller is Blueprint's scripted
simulator-state pick/place baseline. It is not a learned or vendor policy.

The recorded Unchained LUNA deployment uses vacuum tooling and a private policy.
These independent generic parallel-gripper assemblies do not claim that cell's
tooling, policy, calibration or deployment compatibility. Existing bare-arm
catalog entries and earlier body/controller identities remain unchanged.

## Exact source and explicit mount

Both arm recipes retain ros-industrial/motoman commit
`846fbbc57c60223f12bf4795ea6404982d57709e` and the package BSD-3-Clause
declaration covering configuration and 3D models. GP20 means the package's
**GP25-20, YR-1-06VX25-20-A0***, not GP20HL or the default GP25. The tool is
PickNikRobotics/ros2_robotiq_gripper commit
`a74d007d8f2f06dc6a503ad21038ba869d4999a6`, BSD-3-Clause.

The generated entry adds an ideal identity fixed attachment at source `tool0`.
It retains GP12's 100 mm wrist translation and literal flange rotation, and
GP20's 105 mm translation and full flange rotation. No UR-specific 11 mm
adapter or invented mechanical mounting geometry is reused. The nominal grasp
frame is tool-base +134 mm Z/yaw pi, with the existing source geometry basis
and explicit absence of measured hardware calibration. Source frames, ranges,
velocities, effort limits and all gripper mimic relationships are preserved.

## Dynamics estimates, not manufacturer link dynamics

Neither retained arm supplies inertias. A separate generated simulation model
uses transformed collision-mesh axis-aligned boxes in each link frame, with one
effective density over the stationary base and six moving links. Their masses
sum to the selected nominal whole-arm mass; each box gives a positive tensor
about its estimated center. This does not identify the hollow structure,
actuator distribution, true center of mass or factory inertia tensors.

| Body | Nominal arm mass | Primary evidence and limits |
| --- | --- | --- |
| GP12 | 150 kg | Current published GP12-family mass, explicitly assumed applicable to the retained generic geometry; exact source SKU/revision unknown. |
| GP20 | 250 kg | Yaskawa exact YR-1-06VX25-20-A00 flyer matches retained variant, main chain dimensions, six ranges and speeds. |

Both primary PDFs actually say **C-03-2024** despite older dates in their URLs:

- [GP12](https://www.yaskawa.de/Global%20Assets/Downloads/Brochures_Catalogues/Robotics/MOTOMAN_Robots/GP-Series/Flyer_Robot_GP12_E_11.2020.pdf), A-No.206024, p2, SHA256 `6f10791025a69e2bd69428ae16ececa7969489332dba0fde5e89ae2d497315cc`.
- [GP20](https://www.yaskawa.co.uk/Global%20Assets/Downloads/Brochures_Catalogues/Robotics/MOTOMAN_Robots/GP-Series/Flyer_Robot_GP20_E_01.2022.pdf), A-No.195934, p2, SHA256 `019509fb9125413f10b6cfb24222b18140a61385263bdc29768c30ddfd4f039a`.

GP12's flyer contradicts itself: protection labels use `06VX12` and the footer
uses `25VX12`. Its U-axis upper limit is +155 degrees, while source is
**2.6179 rad (about 150 degrees), which remains unchanged**. Older 130 kg claims
lack a verified primary SKU/revision match and are not substituted or averaged.
The official [Yaskawa Singapore table](https://www.yaskawa.com.sg/product/robotics/gp-series)
corroborates 150 kg at family level only.

The tool's nine authored inertias add **0.92086427304 kg separately**; they do
not enter the arm-mass normalization. Source effort limits remain unchanged and
are labeled source model values, not verified manufacturer motor torques. The
flyers' allowable wrist moment/inertia describe payload constraints, not robot
link tensors. PD gains are declared simulation choices: saturation at 0.1 rad
using the source effort cap and critical damping for estimated downstream arm,
tool and nominal payload inertia with an explicit conservative lever assumption.
The existing native fixed-arm runtime's gravity, speed cap, reset and camera
choices remain simulation assumptions rather than site calibration.

Import retains original sources and separately generated model bytes. It applies
and reads back all seven arm masses, centers, diagonal tensors and identity
principal axes; all nine full tool tensors; and all twelve force-drive effort
caps in the final USD. Stage length/mass units and exact body/joint names are
checked. Runtime PD gains are carried in the bound native manifest. This CPU
readback is a software contract, not proof of Isaac execution or physical
performance.

## Registration, budget and evidence

Catalog rows remain individually unavailable until the existing fixed operator
registration operation finds a verified retained preparation publication and
model. The protected installer reconstructs the exact new scripted source
inventory and tool binding, preserves unrelated registry fields, and reports a
reason for each missing or failed body. Registration allocates no compute.

Normal screening selects compatible rigid pick/place candidates through the
existing allocator and site budget. The controller uses authored targets and
live simulator TCP state; the independent scorer remains separate. Evidence is
`scripted_simulator_state_baseline`, with scripted invocation/counts and zero
learned queries. Neither import success nor controller motion implies task
success, vendor endorsement, physical readiness or ADP evidence.

## Offline validation

Tests exercise complete source reconstruction, original-to-composed forward
kinematics at three poses, literal source limits, box mass/tensor derivation,
separate tool mass, saved USD application/readback and tampering, full predecessor
metadata reconstruction, fixed privileged registration, and actual generic
scripted-loop/scorer and sealed bundle paths on development-only doubles.
Retained real meshes were also replayed locally: 56 source files for GP12,
55 for GP20, zero source-link FK discrepancy, and complete USD readback using
the importer double. No Isaac/GPU run, native task success, paid allocation or
live registration is asserted by this change.
