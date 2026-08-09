# ADP-009D overnight results — 2026-08-08

Status: learned-policy P0-P4 canaries completed; the frozen canonical
scripted-positive control remains open after v97, so no further learned-policy
run is admitted. The convergence-aware v4 control plan is published and awaits
one controls-only canary. Broader ADP qualification also remains open.

Scope: ADP-009D, public-scene day-28 Franka construction rehearsal. Every
result in this document remains `development_only`. Simulator state is the
task-scoring source; review media is derived evidence for human inspection.
Nothing here is physical evidence, partner-site qualification, or a claim that
one policy is generally better.

## What landed

The branch advanced from the overnight handoff at `54f70c8e0` to
`521f6537a`. The changes fall into six evidence-bearing groups:

- **P0, interpretable action delivery:** `c9103d311`, `b9a3b30ac`,
  `7b901349a`. Episode receipts retain reset/end joints, maximum joint motion,
  command magnitudes, executed/clamped rows, `arm_moved`, and
  `actions_reached_robot`. DROID joint velocities are clipped and integrated
  against observed joint state rather than misread as absolute positions.
  Outcomes without action-delivery evidence cannot enter the ordering.
- **P1, bounded and identity-bound official GR00T:** `dca08f185`,
  `85a4facc2`, `6d2a5ba49`, `9fcfdd4a2`, `11219f4a0`, `e2db2fa1d`,
  `5c45d73c2`, `81b8c8c9d`, `b6406923a`, `475290824`, `17697f72c`,
  `3fc6a25b7`. One blocking readiness attempt can no longer starve its
  deadline; failed server logs survive; the exact official source,
  checkpoint, embodiment, separately gated Cosmos backbone, offline alias,
  processor, client wire dependencies, and portable H.264 media path are
  digest-bound. Gated-backbone authority is explicit and fail-closed.
- **Complete, reviewable episodes and cycle-time evidence:** `c061fc1b9`,
  `5e01b5d38`, `d163ea63d`, `2196412e5`, `03d314fc3`. Every scored episode
  retains exact lossless policy-input frames, their manifest, a terminal
  observation, and a macOS-compatible review video. Timings distinguish
  environment construction, camera/observation work, inference, simulator
  stepping/render, scoring, and media persistence. A completed evaluation
  query is distinguished from a readiness query.
- **Reproducible admissible episode starts:** `4f4bbdfbf`, `4231d9315`,
  `7400ae233`, `cf0562a32`, `a5d05fa5d`, `3dcb3bc73`. The gripper command is
  calibrated to its physical stroke; the approach is expressed with the
  pinned IsaacLab XYZW convention; starts fail closed unless the can is
  visible, not border-clipped, and the arm/can state can be replayed within
  tolerance before every episode.
- **Both policy views now prove the sealed can is observable:** `589f798d0`,
  `51fd02555`. Selection and restore gates require semantic evidence in both
  wrist and external views. The external task camera is authored in the Arena
  DROID `CameraCfg.offset` before prim spawn, so its sensor buffer and
  render-authoritative USD transform agree. A post-spawn metadata-only pose
  update is no longer accepted as a camera fix.
- **Frozen scenario matrix and fail-closed native controls:** `46e5d2e99`,
  `2086690c1`, `a350153b5`, `adee2091a`, `db9559137`, `3127e461c`,
  `04db82770`, `47c83b41d`, `b97c13afd`, `8aeeb1203`, `5b1bb7e11`,
  `521f6537a`. The
  policy-neutral 128-cell matrix
  binds identical cells and seeds to both candidates and both controls. Every
  episode retains external, wrist, and overview media. The positive control
  now targets the measured finger-midpoint frame, gates every IK phase on
  arrival, and expresses the raw world-aligned PhysX Jacobian in the yawed
  robot root before pairing it with root-frame Cartesian errors. A blocked
  control can seal its IK receipt without recursively including the receipt's
  own digest. Wrist calibration is derived from the live gripper-base pose and
  the measured rigid mount when Isaac's sensor/USD pose remains frozen, and the
  aim target is the observed can centre rather than its support-plane root.
  The aim solver now accounts for the camera's rotational swing around its
  non-zero mount offset. Every bundle carries the frozen task destination
  required by the task-centred overview, and paid requests fail closed unless
  diagnostic-only, controls, or learned-policy mode is explicit. Commits
  `0191571a0` and `d14252947` additionally quarantine a host that exits before
  producing a terminal bundle receipt and make camera aiming orientation-first:
  every servo step resolves the rigid-mount look-at from the current live body
  pose while holding that live position, rather than pulling the gripper back
  toward an unreachable reset translation. The v4 control plan replaces fixed
  IK phase lengths with bounded convergence: motion phases have a 240-step hard
  ceiling, grasp and release retain a 30-step minimum dwell, and phase arrival
  requires three consecutive samples within 0.02 m. Receipts distinguish
  `stable_arrival` from `maximum_steps_exhausted` and retain the executed
  minimum, maximum, stability count, and terminal error.

Other supporting changes are `a78e70dde` and `f98622f67` for the frames-only
Aura comparison, `790b95eec` for explicit concurrent-GPU authority without
weakening provider inventory checks, and `3dcb3bc73` for restore replay.

The required pre-commit gates passed before every landed change:

```text
PYTHONPATH="$PWD/src" .venv/bin/pytest tests/ -q -k "adp009d or droid or episode or nurec or aura"
.venv/bin/ruff check src/ tests/
```

The latest convergence-aware control correction passed `973 passed, 1 skipped,
9052 deselected` in the required filtered lane. Ruff was clean.

## Scientific findings through v84

### P0 — closed as a harness ambiguity

The original `never_moved` x3 did not support a policy verdict. v63 proved
that actions reached the arm but also exposed that the released DROID joint
velocities were being interpreted as absolute positions. After the mapping
fix, v65 completed three π0.5 episodes with interpretable delivery and
deterministic outcomes `moved`, `grasped`, `moved`. This proves the episode
path can deliver a learned policy's actions and affect the task object. It does
not establish task completion: none of the three episodes placed the can in
the destination, and the later camera correction means v65 must not be treated
as directly paired with v84/v85.

### P1 — closed as a runnable official-runtime path

The exact NVIDIA GR00T N1.7 DROID runtime now provisions and serves from the
pinned official source and checkpoint with the exact DROID embodiment. The
separately gated `nvidia/Cosmos-Reason2-2B` dependency is materialized only
under explicit authority and used offline by the worker. v82 and v84 each
completed three GR00T episodes with 60 policy queries and 520 environment
steps per episode. This closes the old startup hang; it is not a performance
claim.

### P2 — float32 Aura is an operational null, not a quality result

v70 exercised the float32 NuRec candidate under a held-constant frames-only
diagnostic. The candidate appearance was absent in both cameras while depth and
semantic digests remained equal, so the receipt explicitly records
`quality_winner: null` and rejects v4 only as a drop-in. It does not establish
that float32 is visually worse. The 42.5 MB v3 asset remains the working
candidate instead of spending more runs to qualify the 87.3 MB v4 payload; no
additional CCM sweep was justified.

### P3 — the dominant time is runtime construction and stepping/render

The v84 timing receipt records 39.44 s for environment build, 7.86 s camera
warmup, 9.19 s gripper probe, 28.68 s approach, and 419.70 s across the three
GR00T policy episodes. Provisioning and cold environment construction still
dominate wall clock outside the episode loop; within episodes, simulator
stepping/render is materially larger than inference. Same-instance
comma-separated candidates are therefore the supported P4 optimization.
Persistent reuse across nominal runs was not adopted because each paid run
still requires a provider-zero teardown proof.

### Camera-evidence correction

v82 produced three scored GR00T episodes, but the external view contained only
about 315 semantic can pixels (`0.55%` of the image). That was too weak for a
two-policy comparison and matches the human observation that the v2 SimReady
can appeared absent. v83 added a dual-view gate and correctly failed before
policy inference with `external_task_camera_object_not_visible`. Its diagnostic
showed the sensor buffer reporting the requested new camera pose while the USD
camera prim and render remained at the old pose.

v84 authored the camera before spawn and closed that harness fault. The buffer
and USD world positions agree within floating-point tolerance. At the selected
step 169, the exact can occupies 3,052 wrist pixels (`5.30%`) and 806 external
pixels (`1.40%`), with both bounding boxes inside the required margin. Four
restore receipts (preflight plus three episodes) replayed 177 steps each with
maximum arm error `0.000907 rad` and object-position error about `5.54 um`.

v84 then completed three GR00T episodes. All delivered actions, all moved the
arm, and all scored `never_moved`; none moved the can. Per-episode maximum
absolute joint deltas were approximately `[2.672, 1.481, 1.117, 0.290, 1.196,
0.630, 0.573]`, `[0.810, 0.691, 0.241, 0.373, 1.091, 1.535, 1.814]`, and
`[0.866, 0.990, 0.411, 0.370, 0.461, 1.858, 2.037]` rad. Joint-limit clamping
occurred in 32, 1, and 5 action rows respectively. The result is an
interpretable canonical-cell null, not evidence of harness non-delivery.

### P4 — completed as a tied canonical-cell null

v85 ran π0.5 and GR00T together on one L40S from the same immutable bundle and
the same dual-view-admissible selected state. Seven restore operations
(preflight plus six episodes) produced one identical restore digest. Every
restore replayed 177 steps, held maximum arm error to `0.000907 rad`, held
object-position error below `5.59 um`, and retained 3,049 wrist / 806 external
semantic can pixels.

Both candidates completed three scored episodes with no failures, 60 policy
queries and 520 environment steps per episode, interpretable action delivery,
observed arm motion, and complete policy-input media. The deterministic result
was:

| Candidate | Outcomes | Mean rung | Interpretation |
| --- | --- | ---: | --- |
| `pi05_droid` | `never_moved` x3 | `0.0` | Actions reached the robot and the arm moved; the can did not move. |
| `groot_n17_droid` | `never_moved` x3 | `0.0` | Actions reached the robot and the arm moved; the can did not move. |

The comparison is a tie (`leader: null`, `tied: true`) with receipt digest
`sha256:15385d341dbedf49f75e1b2bd52e52290b1ad841f93ebf9723b2aea14b8e24fc`.
The serialized `ranking` array is only a stable ordering of tied rows; it must
not be read as GR00T beating π0.5. The receipt correctly records
`supports_policy_ranking: false`.

π0.5 maximum per-joint motion across the three episodes was approximately
`[0.721, 2.401, 0.645, 3.002, 1.187, 1.512, 1.315]`,
`[0.800, 2.571, 1.010, 3.002, 1.588, 1.512, 0.959]`, and
`[0.680, 2.296, 0.579, 2.849, 1.526, 1.512, 0.713]` rad; 18, 27, and 14
action rows contained a joint-limit clamp. GR00T's corresponding motion was
`[2.013, 1.834, 1.209, 0.192, 1.181, 1.519, 0.661]`,
`[2.127, 1.464, 0.994, 0.406, 0.835, 1.127, 1.016]`, and
`[1.421, 1.701, 0.631, 0.117, 1.174, 1.457, 2.341]` rad; 26, 27, and 31
action rows contained a clamp. These are retained diagnostics, not a smoothness
or safety verdict.

Episode totals were 117.83, 120.35, and 126.03 s for π0.5 and 121.80,
122.74, and 128.70 s for GR00T. Policy inference was only 5.27-5.53 s for π0.5
and 6.33-6.54 s for GR00T; environment stepping/render consumed 74.23-83.04 s.
This confirms that reducing inference latency would not materially solve the
current cycle-time bottleneck.

## Paid-run ledger

The conservative retained v1-v62 total is `$10.998299`. Retained v63-v97
ledgers add `$8.224745`, for `$19.223044` total and `$5.776956` unspent under
the `$25` cap. v85's provider API did not expose a final billed value, so its
ledger uses the adapter's conservative observed-runtime estimate of
`$0.433506`. Zero-cost inventory and launch-lock blocks are included because
they are evidence that concurrency failed closed.

| Run | Cost | Returned evidence / null |
| --- | ---: | --- |
| v63 P0 action evidence | `$0.367485` | Actions reached the arm; extreme clamping exposed the velocity-as-position harness fault. No policy verdict. |
| v64 velocity mapping | `$0.183501` | No scored episodes; retained blocked null while stabilizing the corrected mapping. |
| v65 velocity mapping retry | `$0.239484` | Completed π0.5: `moved`, `grasped`, `moved`; actions delivered, arm moved, no place success. |
| v66 GR00T bounded | `$0.027707` | Interrupted before terminal runtime; retained provider-output gap. |
| v67 GR00T bounded retry | `$0.439947` | Provisioned but no scored episodes; diagnostic retained. |
| v68 GR00T identity adapter | `$0.215095` | Exact runtime identity advanced; no scored episodes. |
| v69 GR00T pipless identity | `$0.133393` | Interrupted before terminal runtime; retained diagnostic gap. |
| v70 Aura float32 frames | `$0.258382` | Intentional frames-only stop; v4 appearance was absent, so `quality_winner: null` and v3 retained as the working drop-in. |
| v71 authorized backbone | `$0.238858` | Worker terminated before runtime result; gated dependency path diagnosed. |
| v72 offline alias | `$0.158508` | Worker terminated before runtime result; offline alias advanced but not yet sufficient. |
| v73 offline processor | `$0.259683` | Worker failed before runtime result; processor/backbone mismatch diagnosed. |
| v74 GR00T ffmpeg media | `$0.208022` | Policy path reached; episode did not qualify, portable-media defect exposed. |
| v75 wrist observable | `$0.194714` | Restore failed because the can was not wrist-visible; no episode scored. |
| v76 GR00T framed wrist | `$0.162530` | No safe wrist-observable start; border framing gate worked. |
| v76 π0.5 inventory attempt 1 | `$0` | Existing active instance blocked launch. |
| v76 π0.5 inventory attempt 2 | `$0` | Existing active instance again blocked launch. |
| v76 π0.5 launch-lock attempt | `$0` | Global paid-launch lock blocked concurrent mutation. |
| v76 π0.5 CUDA stamp | `$0.246983` | Episode failed on CUDA tensor to NumPy conversion; encoded host-copy fix followed. |
| v76 π0.5 prefix-fix attempt | `$0.049148` | Interrupted before completion; prefix handling evidence retained. |
| v77 π0.5 evidence parity | `$0.622001` | Completed policy work but comparison remained blocked under the then-current evidence profile. |
| v78 GR00T inventory attempt | `$0` | Existing active instance blocked launch. |
| v78 π0.5 fast render | `$0.231701` | No scored episodes; render-interval experiment exposed stale/evaluation-path mismatch. |
| v79 GR00T XYZW wrist | `$0.150819` | No safe wrist-observable start; pose convention alone was insufficient. |
| v79 π0.5 fast render | `$0.321983` | Policy queried; fast-render diagnostic retained, not promoted as an evaluation. |
| v80 GR00T camera aim | `$0.269420` | Restore joint mismatch blocked all episodes. |
| v80 π0.5 build-time interval | `$0.234155` | Policy queried; timing evidence retained, no ranking claim. |
| v81 GR00T restore budget | `$0` | Admission failed closed before allocation. |
| v82 GR00T restore budget | `$0.279730` | Three scored `never_moved` episodes, but external-can visibility was inadequate for comparison. |
| v83 dual-camera gate | `$0.318888` | Failed before inference: external camera metadata moved but render-authoritative prim did not. |
| v84 spawned dual camera | `$0.236308` | Completed three interpretable GR00T `never_moved` episodes with both views gated and exact can visible. |
| v85 two-policy dual camera | `$0.433506` estimated | Completed six interpretable episodes with complete media: both candidates `never_moved` x3; tied canonical-cell null, no winner. |
| v86 canonical controls + overview | `$0.100146` estimated | Blocked before both controls: the review-only overview camera had no valid metric-depth AOV. No policy/control outcome; encoded review-camera depth fix followed. |
| v87 canonical controls + overview retry | `$0.194701` estimated | All three camera warmups passed and the controls runner started, but the first control could not seal its required review video because the controls-only image lacked `ffmpeg`/`ffprobe`. No sealed control receipt or control outcome; encoded base media-toolchain preflight followed. |
| v88 canonical controls + media preflight | `$0.136833` estimated | Both controls sealed complete external/wrist/overview media. Zero-action passed as `never_moved`; scripted positive failed as `never_moved`. The arm moved up to 0.81 rad while the finger midpoint remained at least 0.39 m from the can, exposing a guessed IK-body/tool-frame transform. The stock overview also retained zero task semantic pixels. No policy verdict. |
| v89 measured grasp frame + overview gate | `$0.234044` | The task-centered overview passed with 111 exact-can semantic pixels inside the frame margin and both controls sealed all six videos. Zero-action passed as `never_moved`; scripted positive again failed as `never_moved`. The measured tool offset reduced the descend error, but holding the camera-aimed body orientation made the pregrasp body pose unreachable. No policy verdict. |
| v90 task orientation + phase gate | `$0.108621` | Zero-action passed. The scripted positive correctly aborted after the 80-step pregrasp instead of advancing or closing, with `scripted_control_phase_not_reached:pregrasp:error_m=0.331908` and `never_moved`. Its intended mostly world -Y motion appeared mostly as world +X, exposing a world-Jacobian/root-error frame mismatch. All six videos were retained. No candidate was provisioned or queried; no policy verdict. |
| v91 Jacobian root-frame gate | `$0.106024` estimated | The corrected approach completed waypoints -1 and 0 and then blocked at waypoint 1, but the runtime crashed while sealing the diagnostic receipt because its local digest helper rejected `digest_field`. Four diagnostic frames per external/wrist/overview camera show the arm moving toward the exact can and the wrist reacquiring it. No control receipt or control outcome was retained, no candidate was provisioned or queried, and no policy verdict can be drawn. Commit `47c83b41d` fixes and hermetically tests the receipt closeout before any retry. |
| v92 control receipt closeout | `$0.138032` estimated | The closeout fix worked and retained the exact failure: `no_safe_wrist_observable_episode_start` plus `scenario_controls_receipt_missing`. The corrected Jacobian binding was sealed with both row blocks rotated world-to-root. The rendered wrist view moved and observed up to 5,771 exact-can pixels, but both Isaac's sensor pose and direct USD pose stayed frozen while Fabric moved the articulation; the can was also clipped at the top because the aim targeted its support-plane root. No control ran, no candidate was provisioned or queried, and no policy verdict exists. Commit `b97c13afd` derives wrist calibration from the live rigid mount and aims at the observed can centre. |
| v93 live wrist mount + visual centre | `$0.130268` | Live body-derived wrist calibration worked and travelled 0.065720 m while the stale USD pose remained fixed. The exact can occupied up to 9,359 pixels, but every admitted wrist box still touched the top edge (`y_min=0`), so all 149 samples correctly failed the five-percent margin gate. No control ran and no candidate was provisioned or queried. The retained mount geometry proves the one-shot look-at left an 8.675955-degree optical-axis residual because rotating the body also swings the offset camera. Commit `8aeeb1203` replaces it with a bounded fixed-point rigid-mount aim solver. |
| v94 rigid-mount aim attempt | `$0.149134` | Blocked before Isaac startup and before either control. The launch omitted the explicit controls mode, and the diagnostic bundle then lacked the task-destination receipt that the task-centred overview loads unconditionally. No frames or control outcome exist. Commit `5b1bb7e11` ships that frozen receipt in every bundle and rejects ambiguous paid requests unless diagnostic-only, controls, or a policy mode is explicit. |
| v95 explicit rigid-mount controls | `$0.087379` | The bundle correctly bound controls-only mode, the canonical scenario, the frozen control plan, and the task destination. RTX 6000 Ada instance `47225330` on machine `137572` passed heartbeat, GPU, and Isaac smoke, then the provider host exited during dependency installation before any terminal bundle marker or output ZIP. No Isaac receipt, frame, or control outcome exists. Commit `0191571a0` classifies this as `provider_instance_exited_before_bundle_terminal_marker` and records the machine for future exclusion. No candidate was provisioned or queried. |
| v96 quarantined-host rigid-mount controls | `$0.238731` | Excluding machine `137572`, L40S instance `47226744` returned a valid native receipt with controls explicitly requested and no candidate queried. The fixed-point solver converged in eight iterations to a `0.000002832` degree predicted residual, but the fixed reset-position plus large target orientation was not a reachable six-DoF pose: reset arrival error was `0.048708 m` and translation-fallback errors were `0.182-0.239 m`. All 240 object holds were safe, yet all 166 wrist-visible samples remained clipped at the top edge, so `no_safe_wrist_observable_episode_start` and `scenario_controls_receipt_missing` correctly blocked both controls. Commit `d14252947` replaces the fixed-position aim with a live, orientation-priority servo. No policy outcome exists. |
| v97 live-orientation-priority controls | `$0.118881` | L40S instance `47229363` returned both controls with complete external/wrist/overview media and no candidate queried. The zero-action negative passed as `never_moved`. The scripted positive advanced through the safe wrist start and moved monotonically toward pregrasp for all 80 fixed steps, reducing grasp-frame error from `0.320352 m` to `0.146733 m`, but correctly stopped before descend/grasp with `scripted_control_phase_not_reached:pregrasp:error_m=0.146733`. Both Jacobian row blocks were world-to-root, rank remained six, the final live wrist-target position error was `0.002877 m`, and joint-limit margins remained healthy. This rules out wrong-axis motion, dead action delivery, and the previous wrist-start fault; it identifies the fixed phase budget as the next harness defect. No policy outcome exists. Commit `521f6537a` replaces fixed IK phase lengths with bounded stable convergence. |

All completed paid attempts were followed by an API ownership/zero check. v86
through v95 were each launched from provider zero as the sole active instance;
after each automatic teardown a fresh Vast API query returned `active: 0 []`.
v96 and v97 ran under exact concurrent-instance authority while a separately owned
Aura/InteriorGS instance `47226054` was active. Only v96's instance `47226744`
and v97's instance `47229363` were torn down; each post-teardown inventory proved
this goal owned zero instances and that the one explicitly allowed external
instance remained. It was not modified or charged to this ledger.

## Scenario-family and control-harness progress after v85

Commit `46e5d2e99` froze a checked, policy-neutral 128-cell scenario suite:
16 cells in each of canonical, placement/approach, illumination,
camera/sensor, physics, visual/material cousin, geometric cousin, and held-out
composed families. Every resolved cell binds identical seeds and parameters
for the zero-action negative, deterministic scripted positive, `pi05_droid`,
and `groot_n17_droid`. The suite discloses the earlier canonical canaries rather
than pretending they occurred after the freeze. Its maximum is 512 episodes
and 19 GPU-hours; this is an upper bound, not authority to launch all cells.

Commit `2086690c1` added the reusable control execution path through the
same native eight-dimensional action seam as the learned candidates. It:

- realizes zero joint velocity as a hold of the observed absolute joints;
- runs a fixed differential-IK pregrasp, descend, grasp, lift, transport,
  place, release, retreat, and settle program;
- scores both controls only from deterministic simulator state;
- blocks candidate execution when the negative completes the task, the
  positive fails to place, media is incomplete, or the shipped plan/instance
  binding changes; and
- retains action, state, contact-gap, calibration, timestamp, manifest, and
  portable H.264 evidence without querying a learned policy.

The manipulation visual-evidence profile now requires a third fixed overview
camera for every new episode. External and wrist remain the only policy-input
views; overview is explicitly review-only and cannot grade the episode. All
three calibrated lossless streams are sampled throughout motion, while the
manifest separately counts exact policy-input frames and review samples. This
is a reusable episode-evidence contract, not a one-off screenshot workaround.

v86 was the first controls-only paid canary. It failed before the controls at
the new overview camera's `camera_metric_depth_invalid:external_camera_2` gate.
The rendered RGB had already passed the degenerate-frame check, but Arena's
second exterior review camera does not provide a valid metric-depth AOV. Metric
depth is required for the external/wrist policy views, not for a review-only
camera that cannot reach policy input or scoring. The encoded fix therefore
keeps the overview's RGB, semantics, calibration, timestamps, lossless frames,
and video mandatory while representing metric depth as explicitly not required.
It does not relax either policy camera's depth gate.

v87 confirmed that fix: external, wrist, and overview RGB warmups passed and
the canonical controls runner began. The first control then failed while
sealing evidence because `ffmpeg` and `ffprobe` were absent. Learned-policy
provisioning had installed ffmpeg incidentally, but a controls-only bundle
correctly skips all policy provisioning. The encoded correction makes both
tools an explicit base episode-media dependency, checks or installs them before
any policy provisioning or simulator startup, writes a typed toolchain status,
and fails closed before paid task execution if either remains unavailable.

v88 passed the base media preflight and sealed both control episodes with all
six videos decode-validated. The zero-action negative passed as `never_moved`.
The scripted positive failed as `never_moved`, so the canonical cell remains
blocked and no learned-policy result may be interpreted. Its joint trace
proves this was not a dead action seam: the arm changed by as much as 0.81 rad,
but the finger midpoint stayed at least 0.39 m from the can and the can's
maximum horizontal displacement was only 2.3 micrometres. The control plan had
commanded the `panda_hand` body as though it were the finger midpoint, using a
guessed scalar vertical offset and a hard-coded world quaternion.

The retained overview stream exposed a separate evidence defect: Arena's stock
second external camera faced mostly away from the task and measured zero robot
or can semantic pixels. It is a valid RGB stream but not a valid overview of
the manipulation. The encoded corrections therefore:

- express every pick/place waypoint in the probe-calibrated finger-midpoint
  grasp frame, including can-centre grasp/place heights;
- resolve each IK body target from the live full three-dimensional
  body-to-finger-midpoint transform while holding the current controlled-body
  orientation, eliminating asset-specific guessed offsets;
- version that changed plan as `adp009d_control_plan.v2`; and
- place the review camera farther back on the proven task-camera ray centered
  on the start/destination envelope, copy the proven orientation at the
  render-authoritative pre-spawn seam, and fail closed before controls unless
  the exact can has at least 80 semantic pixels inside the frame margin.

v88 is the first interpretable paid control pair, but its positive result is a
task/control-harness failure rather than simulator task success. The six videos
and lossless manifests are retained under its immutable execution output. No
learned policy was queried.

v89 closed the overview ambiguity. The exact can occupied 111 semantic pixels
with bounding box `[150, 107, 156, 124]`, centroid approximately
`[0.4804, 0.6460]`, and passed the five-percent frame-margin gate. Both controls
again sealed external, wrist, and overview videos. The zero-action negative
passed; the scripted positive remained `never_moved`, with only micrometric can
motion. Its semantic grasp-frame terminal errors were approximately 0.287 m at
pregrasp and 0.343 m after descend. The retained reset/tool geometry explains
why: applying the measured world-space offset while holding the wrist-camera
aim orientation places the pregrasp controlled-body target about 0.93 m from
the Franka base, outside the frozen 0.855 m reach, although the finger target
itself is reachable. This is a control-frame construction fault, not a policy
result.

The next encoded correction versions the plan as `adp009d_control_plan.v3` and:

- measures the complete body-to-finger-midpoint offset in the controlled
  body's local frame;
- applies that offset at the horizontal-support top-down task orientation,
  rather than preserving the camera-observability orientation;
- keeps the target at the semantic finger midpoint for every pick/place phase;
  and
- records each phase's target, achieved grasp-frame position, terminal error,
  and 0.02 m arrival tolerance, aborting with the typed
  `scripted_control_phase_not_reached` blocker before the gripper closes or the
  program advances when convergence is not established.

v90 executed that correction and made the remaining failure interpretable. It
aborted the scripted positive at pregrasp after 80 steps with a 0.331908 m
finger-midpoint error. The target required approximately `[+0.090, -0.326,
+0.153]` m in world coordinates, while the observed tool motion was
approximately `[+0.208, -0.029, +0.065]` m. The exact pinned Arena source
confirms the absolute action binds ordered `panda_joint1` through
`panda_joint7`; action-column ordering was not the fault.

The exact pinned IsaacLab source at `e57379c634b42db5a0fe9f754341be6e2a7c7c43`
establishes the actual frame contract: `root_view.get_jacobians()` is
world-aligned, while the differential-IK pose error is expressed in the robot
root. IsaacLab's own task-space action rotates both the linear and angular
Jacobian row blocks by the inverse root rotation. ADP-009D's robot root is
yawed -90 degrees; the runtime had transformed the target/current poses into
that root but had passed the raw world Jacobian unchanged. That predicts the
wrong-axis response v90 recorded.

Commit `04db82770` applies the pinned implementation's world-to-root rotation
to both Jacobian row blocks in the reusable approach and scripted-control
paths. It also fails closed unless arm action indices resolve exactly to
`panda_joint1` through `panda_joint7`, retains body/joint/Jacobian/frame
bindings, adds controlled-body pose to every state sample, and retains bounded
per-step target, error, Jacobian norm/rank, and joint-delta diagnostics. The
hermetic -90-degree regression maps v90's world error to the expected root
error.

v91 exercised that correction without provisioning or querying either learned
candidate. The approach completed waypoints -1 and 0, the arm visibly moved
toward the exact SimReady can, and the wrist view reacquired the can before
waypoint 1 blocked. The runtime then lost the underlying blocker by raising
`_canonical_digest() got an unexpected keyword argument 'digest_field'` while
closing the new IK receipt. Consequently v91 retained no sealed control receipt
and establishes neither control outcome. Commit `47c83b41d` makes the bundled
runtime digest contract match the repository contract, excludes the self-digest
field without mutating the receipt, and covers the exact blocked-control shape
hermetically. One controls-only canonical canary remains required.

v92 proved that closeout and separated the next two harness faults. The
corrected root-frame binding was sealed, the approach arm moved, and the wrist
render observed up to 5,771 exact-can pixels. The render changed with the live
gripper, but the sensor-reported and direct USD camera positions both remained
byte-stable while Fabric drove the articulation, making the calibration stale.
The exact-can mask also remained clipped against the top edge because the aim
target used the can's root on the support plane rather than the centre of its
observed 0.169 m height. Commit `b97c13afd` measures the rigid camera offset at
reset, composes every retained wrist calibration from the live gripper-base
body, and targets the observed can centre. The episode adapter carries the same
live pose callback into every control frame. This is locally proven and still
requires a controls-only canonical canary.

v93 proved the live calibration path but not the episode-start gate. The wrist
pose derived from the live gripper-base body travelled 0.065720 m and saw up to
9,359 exact-can pixels. Nevertheless, every wrist mask touched the top image
edge and correctly failed the unchanged five-percent margin requirement. The
retained reset body, mount offset, camera quaternion, and target reproduce the
cause locally: a one-shot look-at leaves an 8.675955-degree residual after the
offset camera swings around the rotated body. Commit `8aeeb1203` iterates the
rigid transform and optical-axis correction to a bounded, recorded residual;
the exact v93 geometry converges below `1e-5` degrees in its hermetic test.

v94 did not exercise that correction. The paid request accidentally selected
neither controls nor a policy candidate, and the diagnostic bundle omitted the
task-destination JSON that the overview camera requires before the runtime
chooses its execution mode. It failed before Isaac startup, retained no frames,
and cost `$0.149134`. Commit `5b1bb7e11` closes both paid-launch ambiguities:
every bundle ships the frozen destination, and the canonical allocator rejects
a paid ADP-009D request unless diagnostic-only, controls, or at least one
candidate is explicit. No retry is admitted without the explicit controls flag
and frozen scenario instance.

v95 carried the correct explicit controls and canonical scenario bindings, but
provider machine `137572` exited during dependency installation after passing
the heartbeat, GPU, and Isaac smoke gates. It produced no output ZIP or native
receipt. That null is a provider-host failure rather than evidence about the
camera aim or controls. Commit `0191571a0` makes an instance that disappears
before a terminal bundle marker a typed blocker and automatically adds its
machine to the run's avoidlist.

v96 excluded that machine and returned the first native receipt for the bounded
fixed-point solver. The mathematical solution was exact, but the Franka could
not realize the old command because it simultaneously fixed the reset body
position and demanded a large body rotation. The first arrival retained a
`0.048708 m` position error, and the three translation fallbacks remained
`0.182-0.239 m` away. Although all 240 object-hold samples were safe and the
can reached 11,316 wrist pixels, every visible wrist mask still touched the top
edge; both controls correctly remained blocked. The external, overview, and
wrist contact sheets are retained in the run's `review/` directory.

Commit `d14252947` encodes the measured correction without weakening the
five-percent visibility margin or object-safety gates. During the camera-aim
waypoint, each IK step now recomputes the rigid-mount orientation from the live
body pose and holds that current live translation. The receipt distinguishes
this orientation-priority command from the preregistered reset position and
retains bounded solver updates.

v97 exercised that correction and reached the controls for the first time from
a live-aimed, non-border-clipped wrist start. The scripted arm reduced pregrasp
error on every one of its 80 steps and ended with no joint-limit margin below
`0.683 rad`; the target remained inside the frozen `0.855 m` base-reach bound.
Its last 20 steps reduced error by about `0.002048 m/step`, projecting arrival
roughly 62 steps beyond the old ceiling. Because it had not reached the 0.02 m
tolerance, the phase gate correctly prevented descend and gripper closure.

Commit `521f6537a` encodes that observed correction as
`adp009d_control_plan.v4`. Each IK phase now runs until three consecutive
in-tolerance samples or its hard maximum, whichever comes first; grasp and
release still dwell for at least 30 steps. Hermetic coverage proves a
v97-like slow phase can run past 80 steps then stop early, a transient arrival
does not pass, a nonconverging phase exhausts exactly 240 steps, and the legacy
v3 plan is rejected. This requires one controls-only canonical canary before
either learned candidate may run again.

## What remains open

- The P4 result is intentionally underpowered and single-cell. Its comparison
  receipt remains `supports_policy_ranking: false` until the native scripted
  positive succeeds on the canonical cell, both controls run on every scored
  cell, and a paired sample size for stated power is executed. The tied null
  means this rehearsal does not yet change the next scarce physical-test
  decision.
- The noncanonical suite cells are digest-resolved, but their lighting, camera,
  physics, placement, and cousin parameters still need a native scene
  application receipt before those cells may execute. A canonical control run
  cannot establish that broader application path.
- The wider Arm Decision Proof still requires one fresh Raw V3.2 unseen partner
  workcell capture; qualified metric registration, task physics, and
  observation-domain match; prospective preregistration of exactly two frozen
  candidates; and held-out physical adjudication of both the decision and one
  predicted failure boundary.

## Single next action

From provider ownership zero at clean immutable commit `521f6537a`, run only the
checked-in canonical scenario's zero-action negative and deterministic
scripted-positive control pair, explicitly binding `--adp009d-controls` and
`docs/arm_decision_proof_v1/manifests/adp009d_canonical_scenario_instance.v1.json`.
Do not query either learned policy. Verify that
the wrist aim selects a non-border-clipped start while retaining live
body-derived camera calibration, that the IK binding reports world-to-root
rotation for both Jacobian row blocks, and that every IK phase records either
stable arrival or maximum exhaustion. If the scripted positive does not place the exact
SimReady can, retain the overview/external/wrist videos and typed receipt, tear
down to provider ownership zero with no unexpected instances, and fix the
harness locally before any learned-policy spend.
