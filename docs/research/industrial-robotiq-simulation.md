# Five industrial Robotiq simulation assemblies

These separate body IDs add a generic scripted rigid pick/place route to three
public FANUC arms and both exact ABB GoFa variants. They do not assert vendor
policy equivalence, an available trained checkpoint, hardware fit, measured
calibration, physical feasibility, or a successful native task episode.

| New body ID | Preserved arm frame | Arm mass basis | Combined arm + source tool mass |
| --- | --- | --- | --- |
| `fanuc_crx_10ia_robotiq_2f85_sim` | `fanuc_flange` | Authored URDF, 38.027 kg | 38.94786427304 kg |
| `fanuc_crx_10ia_l_robotiq_2f85_sim` | `fanuc_flange` | Authored URDF, 38.607 kg | 39.52786427304 kg |
| `fanuc_m710ic_45m_robotiq_2f85_sim` | `fanuc_flange` | Authored URDF, 570.613 kg | 571.53386427304 kg |
| `abb_gofa_crb15000_5_95_robotiq_2f85_sim` | `tool0` | Explicit geometry model anchored to nominal 28 kg | 28.92086427304 kg |
| `abb_gofa_crb15000_12_127_robotiq_2f85_sim` | `tool0` | Explicit geometry model anchored to nominal 48 kg | 48.92086427304 kg |

The FANUC mass values are sums of the publisher's source model, not independent
physical measurements. The Robotiq mass is the sum of nine authored URDF
inertial bodies, 0.92086427304 kg. There is no adapter mass or geometry: each tool
uses an explicitly declared ideal fixed simulation mount at the named frame.

## Exact sources and frames

- [Official FANUC description](https://github.com/fanuc-corporation/fanuc_description/tree/fb40c9803a826ba68c7c8e28ba904a25efa7fcd2), Apache-2.0. The full source J6-to-flange translation remains 0.160 m for CRX and 0.175 m for M710. The subsequent `flange` to `fanuc_flange` rotation remains RPY `(pi, -pi/2, 0)`. The example pedestal is replaced only in the generated assembly by a declared zero world-to-base mount; original source bytes are retained.
- [ROS-Industrial ABB description](https://github.com/ros-industrial/abb/tree/45f4769d826cf3ac62a65495f2db67b78b0c81df), Apache-2.0 for both support and shared resources. The full `flange` to `tool0` rotation remains RPY `(0, pi/2, 0)`. The original absent inertias and zero effort placeholders remain in retained source bytes.
- [PickNik Robotiq description](https://github.com/PickNikRobotics/ros2_robotiq_gripper/tree/a74d007d8f2f06dc6a503ad21038ba869d4999a6), BSD-3-Clause. Its nine inertial bodies, six moving jaw joints, limits and mimic signs are retained. The source closure also contains the licensed UR adapter file, but these five assemblies never include or instantiate it. In particular, they do not add the UR-specific 11 mm spacer.
- [Nominal grasp-frame reference](https://github.com/PickNikRobotics/moveit_pro_ur_ws/blob/bab77dd0862bd6dea828ec41fb60b39d22c6f2c0/src/ur_mock/description/picknik_ur.xacro): gripper base +0.134 m Z and yaw pi. This is a simulation convention, not publisher calibration for these five arms.

The unchanged GoFa arm model retains its official ABB nominal mass reference,
3HAC077389-001 Rev Y, section 2.2, page 42; retained PDF SHA-256
`b502ddd2e7cdf12066c708f5e590b6768651771cbb738fe35f27f1561e407fa5`.
The composite wrapper keeps that bare-arm report separately and adds the
source tool dynamics. It does not renormalize arm-plus-tool to 28/48 kg.
The arm drive estimates retain the original nominal-payload allowance and
simulation-only provenance. They are not identified manufacturer motor limits.

Import preserves fixed joints. The saved USD readback checks the seven modeled
arm masses/inertias and force limits, plus all nine source tool masses, COMs,
full body-frame inertia tensors and six force-drive caps. It compares full
tensors reconstructed from USD principal axes, allowing equivalent eigenbases.
Missing or merged tool bodies, changed inertia, changed effort, or acceleration
rather than force drives produce individual verification failures.

## Runtime and evidence boundary

Each manifest has one six-DoF arm, one six-joint gripper closure channel, a
source-derived TCP, bounded joint limits and reset positions, and overhead and
wrist cameras. The new controller profile is
`blueprint_scripted_pick_place_industrial_robotiq.v1`: the existing simulator-state
pose servo with a separate sealed source identity and exact tool contract.
It consumes the authored rigid-task targets and live TCP/joint observations.
The independent simulator scorer still determines task outcomes. This is not
learned-policy inference or evidence of vendor-trained performance.

The fixed preparation and scripted-registration installers admit the five
new IDs. Prepared-body, source/import, and tool-profile failures remain
individual candidate reasons. The preparation inventory is now 28 bodies;
the existing per-grant limit remains 24 and all money bounds are unchanged.
No current batch grant or live model registry is modified by this change.

All original 23 profiles, 29 recipes, 104 catalog rows, and 13 scripted source
identities are preserved. The exact corrected 537a62c generator hash is accepted
only for its original 23 IDs, after existing byte closure and complete current
metadata reconstruction. Unknown generators, stale pre-correction jaw mappings,
new IDs borrowing old generator hashes, and resealed metadata changes remain
refused.

Software verification uses exact pinned source XML with explicitly synthetic
meshes/SDK doubles, plus independent replay against retained actual public
source closures. Actual-source FK checks preserve all arm/link poses at three
joint configurations (maximum observed difference 2.22e-16), and all authored
or modeled inertia tensors have positive, triangle-valid eigenvalues. Full
source-to-fake-USD-to-manifest-to-scripted-policy checks are software evidence.
Native Isaac import, controller, camera and scored task execution remain to be
observed on a separately admitted GPU run. No native or physical proof is
claimed here. KUKA MJCF composition and Yaskawa missing-dynamics modeling remain
separate follow-ups.

Test fixtures retain upstream Xacro whitespace and line endings so their
SHA-256 values remain exact; do not silently reformat these source artifacts.
