# Public fixed arms, batch 1: deployment-company bodies

This batch adds public, license-checked robot bodies that named deployment
companies use. Each body is pinned to an exact public commit and keeps its
publisher license and notice bytes. Three of the bodies also get a complete
simulation configuration, with the Robotiq 2F-85 as an **explicit stand-in
tool**, in the same way as the existing FANUC, Yaskawa, ABB and KUKA Robotiq
configurations. None of this is a company's policy, tool, calibration or cell.
The companies' own policies are private, and their catalog rows stay
unavailable with that reason.

| Body ID | Source (exact commit) | License | Inertias | Added |
| --- | --- | --- | --- | --- |
| `fanuc_m10id_12` | [fanuc_description fb40c98](https://github.com/fanuc-corporation/fanuc_description/tree/fb40c9803a826ba68c7c8e28ba904a25efa7fcd2/fanuc_m10_description), `robot/m10_12_14d.urdf.xacro` (macro header: M-10iD/12) | Apache-2.0 | authored, 145.055 kg total | recipe, bare profile |
| `fanuc_m10id_12_robotiq_2f85_sim` | same arm + PickNik Robotiq a74d007 | Apache-2.0 + BSD-3-Clause | authored arm + tool | recipe, profile, scripted configuration |
| `yaskawa_hc20` | [ros-industrial/motoman 846fbbc](https://github.com/ros-industrial/motoman/tree/846fbbc57c60223f12bf4795ea6404982d57709e/motoman_hc20_support), `urdf/hc20.xacro` | BSD-3-Clause (package.xml) | none in source | recipe, bare profile |
| `yaskawa_hc20_robotiq_2f85_sim` | same arm + PickNik Robotiq a74d007 | BSD-3-Clause | **estimated** arm + authored tool | recipe, profile, scripted configuration |
| `agile_robots_diana_7` | [AgileRobots-APAC/diana_description 9278666](https://github.com/AgileRobots-APAC/diana_description/tree/92786660df762ff4cb803ab508d28a935ae87e96), `robots/diana7/diana7.urdf.xacro` | Apache-2.0 (LICENSE, package.xml, NOTICE) | authored, 26.83 kg total; link0 looks like a placeholder | recipe, bare profile |
| `agile_robots_diana_7_robotiq_2f85_sim` | same arm + PickNik Robotiq a74d007 | Apache-2.0 + BSD-3-Clause | authored arm + tool | recipe, profile, scripted configuration |
| `kinova_jaco2_j2n6s300` | [Kinovarobotics/kinova-ros 924781d](https://github.com/Kinovarobotics/kinova-ros/tree/924781d3dfe241b2b94b3c72a804b80d3658cf02/kinova_description), `urdf/j2n6s300_standalone.xacro` | BSD-3-Clause (root LICENSE; package.xml says "BSD") | authored | recipe only (reason below) |
| `kuka_kr20_r3100`, `kuka_kr30_r2100`, `kuka_kr50_r2100`, `kuka_kr50_r2500`, `kuka_kr70_r2100` | [kuka-ros/kuka_robot_descriptions 5ddbc41](https://github.com/kuka-ros/kuka_robot_descriptions/tree/5ddbc41ce8726fd680bcc19562d78bcabfbdf96d/kuka_iontec_support), `urdf/<model>.urdf.xacro` | Apache-2.0 | authored | recipe, bare profile |

Masses are sums of publisher source values, not measurements.

## Deployment companies and what is not known

| Company | Public association | Catalog row | Explicit gaps |
| --- | --- | --- | --- |
| sewts | [FANUC case study](https://www.fanuc.eu/eu-en/news/intelligent-robots-commercial-laundries): VELUM textile cells use FANUC M-10iD/12 robots | `sewts_fanuc_m10id_12_velum` | private policy; textile gripper and vision not modeled; deformable towel handling is not modeled as rigid pick/place |
| Formic | [Georgia Nut case study](https://www.casestudies.com/company/formic-technologies/case-study/georgia-nut-and-formic-deploy-first-robot-in-companys-history): Yaskawa Motoman HC20 palletizer | `formic_yaskawa_hc20_palletizing` | private controller; HC20 variant (DT, XP or DTP) unconfirmed; palletizing tool not modeled |
| Agile Robots | Manufacturer of the Diana 7 | `agile_robots_diana_7_controller` | private controller/policy; publisher status of the new repository unconfirmed |
| Diligent Robotics | Press reports say Moxi uses a Kinova arm and a Robotiq gripper ([The Robot Report](https://www.therobotreport.com/diligent-robotics-raises-seed-funding-launches-moxi-robot/)) | `diligent_moxi_kinova_jaco2` | private policy; Kinova variant and tool unconfirmed (the reported Robotiq gripper is not the source three-finger hand); mobile base not modeled |
| Contoro Robotics | [KUKA case study](https://www.kuka.com/en-us/industries/solutions-database/2026/04/robots-unload-truck-trailers) names a KR IONTEC arm for trailer unloading | `contoro_kuka_kr_iontec_adaptai` (configuration `kuka_kr_iontec_variant_unconfirmed`) | private AdaptAI policy; the KR IONTEC variant is not public; suction tool and mobile platform not modeled |

The company associations were checked through web search results. The FANUC,
casestudies.com, Robot Report and KUKA pages themselves were not fetched in
this change, so their exact wording is unverified here. Each company row is
disabled, has no policy digest and no packaged policy route, and states that a
Blueprint scripted baseline on a stand-in configuration is Blueprint's own
baseline, never the company's policy.

## Stand-in assemblies

The three `_robotiq_2f85_sim` bodies use a new composition module,
`deployment_embodiment_public_arm_composition.py`. The existing industrial,
Yaskawa and KUKA composition modules keep their hashes. Each assembly includes
the exact arm source and adds an identity fixed mount at a named source frame:

- M-10iD/12 at `fanuc_flange`, keeping the source J6-to-flange 75 mm and the
  `flange` to `fanuc_flange` rotation (pi, -pi/2, 0).
- HC20 at `tool0`, keeping the source 200 mm flange offset and rotation.
- Diana 7 at `diana7_link8`, the source flange frame, with the default robot
  arguments: no hand, no Gazebo tags and no self-collision capsules.

There is no adapter geometry and no UR-specific 11 mm spacer. The grasp frame
is the existing simulation convention: gripper base +134 mm Z, yaw pi. Recipes
mark the tool as `simulation_stand_in_not_deployed_tool`, and composition
refuses a changed tool role, arm pin, tool pin, mount or hardware claim.

## HC20 dynamics are estimates

The HC20 macro has no inertial elements. As for GP12 and GP20, a separately
generated simulation model (`deployment_embodiment_hc20_model.py`) uses
transformed collision-mesh boxes in each link frame. One effective density is
normalized to a nominal whole-arm mass, and the source effort limits set PD
gains. The retained source bytes keep their missing inertias, and the
configuration reports `estimated_inertias_added: true` with
`claim_ceiling: unvalidated_simulation_estimate`.

The only primary mass document found is the Yaskawa **HC20DTP** flyer
(YR-1-06VXCP20-B00, B-11-2023, A-No. 220876, page 2: weight 140 kg;
[download](https://yaskawa.de/_downloads/download_d17096), SHA-256
`89bb45594f5a4a1f4c602a6efe5a86c331f7934deaaf3356b299eb1c3ae2658f`). The
retained package models HC20 and says HC20DT and HC20XP share it. HC20DTP has
its own link 5 and 6. The 140 kg anchor is therefore recorded with
`exact_source_variant_established: false`. The flyer's S-axis range (+/-210
degrees) also differs from the retained source (+/-180 degrees); the source
limit is kept. The payload allowance is the flyer's 20 kg. The tool's nine
authored inertias (0.92086427304 kg) are added separately and are not part of
the arm normalization. None of these values are manufacturer link dynamics or
motor torques.

## Why the Jaco2 has no runtime profile

The j2n6s300 source has six arm joints (four continuous) and its own
three-finger hand: three proximal finger joints (0 to 1.51 rad) and three
distal joints (0 to 2 rad), all independent, with no mimic. The runtime
configuration must bind every source joint, and the existing gripper channel
needs a parallel-jaw width and open/closed mapping. That mapping would be new
grasp semantics the source does not provide. The existing scripted runtimes
only drive the Robotiq 2F-85 jaw mapping. So the body is recorded with the
hand in its recipe and an explicit reason, and no scripted configuration is
added. The reported Moxi tool is a Robotiq gripper in any case, and the Moxi
base cannot use the public Fetch description (CC-BY-NC-SA, non-commercial).

## KR IONTEC family

All five public KR IONTEC models are bare bodies only. The entry Xacro also
declares a ros2_control block. With its default `mode=hardware` the
`kuka_gazebo` macro is never included. The importer resolves `$(find ...)`
aliases textually, so the two small `kuka_gazebo` files from the same commit
are retained for that resolution. No scripted configuration is added, because
Contoro's variant is unconfirmed and its suction tool is not modeled.

## Not added

- Universal Robots UR20: its meshes are under Universal Robots' own terms,
  not BSD. This is a rights stop.
- Galaxea R1: the license is unclear (exporter-template package.xml, no
  LICENSE).
- Fetch: CC-BY-NC-SA-4.0, non-commercial.

## Identities and limits

All 37 previous recipes, 112 catalog rows, the scripted registration profile
and 20 model, composition and scripted-controller module hashes are pinned in
`tests/fixtures/public_fixed_arms/predecessor_identities.json` and remain
unchanged. The exact 867cee8 runtime generator is accepted for its original
31 profiles only, after the existing complete metadata reconstruction. The
scripted install set grows from 21 to 24 bodies, which is the existing
one-grant limit. No preparation batch, grant, registry entry, import, paid
allocation or provider call is made by this change.

## Verification

Tests use the exact public text bytes listed in
`tests/fixtures/public_fixed_arms/inventory.json` with explicitly synthetic
meshes and USD/SDK doubles. They cover source materialization, Xacro
expansion, bare and stand-in configurations, preserved source forward
kinematics at three poses, the ideal mount and grasp frame, HC20 estimate
derivation and USD readback tampering, predecessor reuse and identity
preservation. This is software evidence only: no Isaac import, native
episode, GPU run or physical result is claimed.
