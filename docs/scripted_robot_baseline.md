# Blueprint scripted robot baseline

This route evaluates a Blueprint-owned pick/place script on an imported robot
in an existing compiled simulation task. It is a **scripted simulator-state
baseline**, not a learned policy, vendor controller, deployment-company policy,
or physical validation result. A source robot description alone is not a run.

The fixed interface is `blueprint_scripted_pick_place.v1`. A registered body has
candidate ID `blueprint_scripted_<embodiment_id>`, catalog policy kind `scripted`,
and a source-code inventory digest. No checkpoint, server, credential, hosted
inference or model download exists on this route. Legacy execution fields named
`checkpoint_digest` bind that code inventory; `checkpoint_identity_basis`
explicitly says there are no weights. The sealed provider bundle binds the
complete deployed runtime as it does for other candidates.

## What runs

The controller reads the compiled rigid task's sealed interaction affordance,
grasp orientation, subject reset pose and destination. It derives the same
authored construction-phase geometry used by the existing task planner. It
does not reuse achieved Franka joint trajectories, construction success
verdicts, contact-gate verdicts, or task scorer outputs.

The script approaches the grasp point with the gripper open, closes, lifts,
translates through the authored waypoints, places, releases and retreats.
Authored grasp-frame targets are transformed through the explicit body/tool
binding into each actual TCP and the measured arm-base frame. Among independent
arms, the nearest measured initial TCP to the pregrasp target selects the
active arm; this is a deterministic initial heuristic, not a reachability
claim. The other arms hold their measured reset TCP and gripper closure.
Coordinated bimanual manipulation and bare-arm grasping are not implemented.

The existing native controller reads actual COM Jacobians every tick, shifts
them to the declared TCP, applies bounded damped least-squares control and
commands independent gripper drives. A phase advances on measured pose arrival
plus a 0.35-second dwell, or a five-second timeout. These are initial simulation
control assumptions. A phase timeout is retained and is never task success.
The simulator's sealed action-step budget and allocator cap still bound runs.
Each scenario rematerializes its own authored targets after its normal reset.

The script receives arm-base world poses through an explicit privileged input.
Learned and hosted policies retain their existing observations. Camera frames
are retained for review; the script does not use RGB perception. Exact consumed
state and task-plan digests are retained as in-process request snapshots, with
no claim that network serialization occurred.

## Results and coverage

`execution_evidence_kind=scripted_simulator_state_baseline` and
`action_source=scripted_simulator_state` follow the candidate through planning,
bundle admission, episode execution, closeout and publication. Scripted
episodes have `learned_policy_rollout_count=0`, a separate
`scripted_rollout_count`, and `candidate_policy_queried=false`. An actual
controller invocation is recorded separately. The unchanged native object and
contact scorer determines task outcome; script arrival never overrides it.
`success_claim_contracts` continues to exclude scripted actions from learned
policy and physical proof.

Unsupported task strategies, missing task affordances or unsupported tool
bindings produce individual candidate reasons. Other candidates continue, and
the same per-site USD15 budget decides what executes. Registered compatibility
and passing software tests do not establish that an Isaac episode has run.

The audited tool bindings cover dual YAM, PiPER, UR5e/Robotiq, FR3, Gen3, xArm7
and TRLC; Axol; and single Gen3 and xArm7. That is ten configurations covering
eight prioritized catalog bodies plus two single-arm variants. The source
TRLC and Axol open/closed corrections must be present in prepared bindings.
Per-tool receipts distinguish authored nominal TCPs from approximate mesh pad
centres and retain unknown physical calibration and moving-pad limitations.

## Registration and remaining operational step

`scripted_robot_registration.registration_from_prepared_inputs` constructs
the ordinary reusable `robot_policy` registration from verified imported model
bytes, their dependency/source closure, the exact body manifest and an existing
scoped rights record. It only retains the manifest. It does not change the
operator registry or launch resources.

Production installation must use a narrow authenticated registration operation
bound to an exact retained preparation publication, fixed Blueprint script
profile/body IDs, the installed release's code inventory and the expected
registry digest. Reuse the existing service-user validation plus privileged
compare-and-replace/audit path. A Blueprint-owned MIT script is not a grant for
third-party robot assets: the importer retains those source licenses. Capture
consent, source revocation checks and site budget authority remain unchanged.

The implementation tests use explicit development fixtures and fake physics
to exercise transforms, measured TCP progression, motor mapping, independent
scoring and sealed transport plumbing. They are not native GPU results.
