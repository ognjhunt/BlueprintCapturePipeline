# Company-led robot and policy integration priorities

Owner direction, 2026-10-29: focus on the deployment companies in RobotTeam,
the embodiments they actually use, and whether their policies are public,
available as weights, callable through an API, or require company access.
Preserve the goal: select every compatible robot–policy configuration for a
task, execute what fits the budget, and return a result or reason for each
remaining candidate.

## Registry basis and limits

The canonical company list is WebApp Firestore `robotTeams`, consumed through
`read_robot_registry` / `listMatchableRobotTeams`. Its source-backed capability
fields, team identity, tasks and declared integration should drive task-level
priority. The local Pipeline checkout has no live Firestore registry connection.
This pass inspected WebApp source at
`7ecc256bb77c5ab15250aa0eeb68087e34efa868` and the existing Pipeline
`deployment_embodiment_support.v1.json` company associations. Those associations
are the starting list below, not a fresh export of all live registry records.

The [WebApp seed](https://github.com/ognjhunt/Blueprint-WebApp/blob/7ecc256bb77c5ab15250aa0eeb68087e34efa868/server/data/robotTeamSeed.ts)
also contains OEM prospects such as UR, Franka, Agility, Boston Dynamics,
Figure, Apptronik and Locus. A seed prospect or public deployment example does
not establish a Blueprint partnership, a supplied policy or willingness to
deploy. Do not replace the live list with the seed or an NVIDIA asset picker.

## Company, body, policy and access

| Recorded company association | Exact body focus in current mapping | Published policy/interface evidence | What Blueprint can progress; specific access gap |
| --- | --- | --- | --- |
| Dream Labs | Two-arm YAM, PiPER, UR5e, FR3, Kinova Gen3 and xArm7 | [Dream-1](https://dreamlabs.ai/research/dream-1/) is a public hosted API for bimanual parallel-gripper control. The page shows selected simulations and hardware demonstrations, not our task results. | Public sources for these configurations passed preparation checks. Start with the already authored YAM station, then independently bind each other body, tool and camera rig. Discover the account's actual model ID before binding it. Owner reports creating and saving a key; this session has not authenticated it. |
| Almond Robotics | Axol with two arms and articulated fingers | [Pi API beta](https://www.pi.website/blog/api-beta) demonstrates Almond's task-adapted model on Axol. The API provides managed fine-tuning and hosted inference; access is a beta application. | Axol's public articulated model is prepared. OpenPI provides an independent development path; Almond's exact trained model was not found as public weights in this pass. Its Pi API deployment/model access would be needed to evaluate that exact policy. |
| Extend Robotics | UR5e in the current mapping; broader bodies remain separately identified | Its [published workflow](https://www.extendrobotics.com/post/extend-robotics-bridges-simulation-and-reality-with-intuitive-xr-interface-accelerating-real-world) supports GR00T, ACT and Pi0 training/inference, plus AMAS teleoperation and data collection. | Integrate an exact compatible checkpoint through our existing model/container/endpoint path. Public framework availability does not establish that Extend's trained task checkpoints are public or that it uses a particular newer checkpoint version. |
| The Robot Learning Company | Two TRLC-DK1 followers | The official [DK1 repository](https://github.com/robot-learning-co/trlc-dk1) supplies a LeRobot hardware plugin, single-arm teleoperation and bimanual recording examples. | Public robot sources are prepared. A LeRobot-compatible trained checkpoint, normalization and camera/action mapping are still needed for a learned-policy evaluation. The inspected README supplies no ready-made company policy API or general task checkpoint. |
| Jacobi Robotics + Delta Technology | FANUC M-710iC/45M palletizing cell | The [deployment case](https://jacobirobotics.com/case-study/deployment-with-delta-tech) names this exact robot and OmniPalletizer. The [Jacobi platform](https://jacobirobotics.com/platform) documents proprietary planning with Python/C++ APIs, cloud and on-prem options. | Integrate the planner/trajectory contract and actual palletizing tool. API availability is established; account/license entitlement and access to the specific OmniPalletizer deployment remain separate. A generic motion-library example does not reproduce the whole deployed application. |
| GrayMatter Robotics | FANUC CRX-10iA/L with a finishing tool | The [FANUC case study](https://www.fanucamerica.com/case-studies/crx-cobot-ai-solve-sanding-challenges-for-performance-composites) identifies Scan&Sand / Scan&Polish, GMR-AI and proprietary software. | Prepare exact robot, compliance/tool/contact configuration. No public callable GMR-AI policy or weights were established in inspected primary sources. Keep a company-provided controller/endpoint route visible; an independently built finishing baseline must carry its own identity. |
| Unchained Robotics | Yaskawa GP12 / GP20 MalocherBot palletizing with vacuum tooling | The [product page](https://unchainedrobotics.de/en/products/turnkey-solutions/unchained-robotics-malocherbot-palettieren) specifies both robots, vacuum gripping and LUNA software. | Prepare robot, vacuum tooling and explicit simulation dynamics estimates where OEM inertias are absent. No public LUNA inference API or policy weights were established in this pass. Keep controller access separate from public robot assets. |
| Standard Bots + Path Machining | Exact RO1/Core generation and machine-tending tool | Standard Bots publishes a [ROS2/REST/Python control interface](https://github.com/standardbots/ros2-realtime-api), enabled in the robot app. [Path Machining](https://pathmachining.com/resources/capabilities-deep-dives/standard-bots-automated-machine-tending/) documents a Standard Bots tending cell. [Praxis-1](https://runway.com/research/introducing-praxis-1) shows RO1 under partner access, with public weights promised later. | Hardware control API access is not a hosted learned-policy endpoint. Complete licensed meshes remain unresolved. Exact generation, control access and trained policy access each need their own binding. Do not treat a general Runway media API account as Praxis access. |

Public-source preparation is already useful for these companies; missing
private policies should not prevent preparing their bodies or evaluating an
independent public baseline under its correct identity. Unknown access remains
unknown rather than being described as impossible or silently counted as ready.

## Candidate and routing behavior

For each task, begin with relevant company registry entries and preserve their
source provenance. Enumerate exact company/body/tool/policy/checkpoint or API
deployment combinations; a shared robot brand or model family is insufficient
to assert policy compatibility. Preserve every considered configuration in the
result, including setup failures, access/rights limits, incompatibility and
budget-deferred candidates.

Use company/task fit to prioritize preparation and budget scheduling. Do not
turn that preference into an exclusive allowlist that removes other compatible
candidates. Open baseline evaluations remain attributable to their own policy,
not the commercial company whose robot model they share. Marketing evidence,
a public repository and API discovery do not become scored task outcomes.

For executable candidates, keep the existing privacy/rights checks and paid
allocator with the per-site ceiling. Other uncertainty should be recorded and
resolved through the smallest useful supported experiment, with explicit
simulation assumptions rather than new approval stages.

## Dream credential state and next execution step

The owner reports `DREAM_API_KEY` saved in the Codex Cloud personal vault for
all configurations. A current environment-status read and a boolean process
check on 2026-10-29 did not find that variable in this already-running instance.
The key value was neither requested nor printed. Apply the updated environment
configuration or use a fresh instance before account discovery.

A cloud-session variable does not automatically install a production worker
credential. The existing worker factory reads `BLUEPRINT_TEAM_POLICY_BEARER_FILE`
through its protected credential delivery path. Keep that flow; do not embed
keys in catalog JSON, source files, screenshots or retained execution evidence.
Account model discovery is a separate read-only step; inference remains within
the existing disclosure and provider-spend authorization.
