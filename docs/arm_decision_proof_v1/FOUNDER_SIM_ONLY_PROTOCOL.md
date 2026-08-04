# Founder Sim-Only Decision Test

Status: **frozen and exact-digest founder approved; runtime not started**
Claim ceiling: **development-only simulation**
Protocol ID:
`adp-founder-sim-arena-droid-pi05-vs-groot-n17-v1`

This is the deliberately thinner stage requested by Blueprint's founder. It
tests the decision harness against an existing simulated workcell before any
partner or physical phase. It is not ADP-010 partner admission, sim-to-real
validation, deployment evidence, or a physical allocation decision.

## Frozen Test

- Robot: the DROID embodiment: Franka Panda with Robotiq 2F-85 gripper.
- Scene: Isaac Lab-Arena's existing `pick_and_place_maple_table` environment.
- Task: pick up the registered Rubik's cube and place it in the registered YCB
  bowl.
- Baseline: `pi05_droid_jointpos_polaris`, the byte-inventoried OpenPI π0.5
  DROID joint-position checkpoint.
- Alternative: `nvidia/GR00T-N1.7-DROID`, pinned to NVIDIA Isaac-GR00T source
  revision `b9955401d50c92a29258732e3ad6ccd579f1bdc0` and Hugging Face
  checkpoint revision `05e7cc97e40dbd33b0890c35cc0214fcb0547ab5`.
- Primary metric: deterministic binary simulator-state task success. No
  secondary metric is registered.
- Conditions: one frozen scene/object/light configuration; each repetition uses
  a paired deterministic Arena relation-solver seed.
- Design: fixed two-candidate independent-proportion comparison, 30 percentage
  point minimum relevant difference, two-sided alpha 0.05, power 0.80,
  conservative planning variance 0.5.
- Schedule: the calculation requires 44 trials per candidate and 88 total.
  Failed, invalid, timed-out, interrupted, and missing trials remain failures in
  the denominator.
- Evidence: every episode must retain the exact lossless policy-input images,
  terminal image, frame manifest, review video, and independent simulator-state
  grader provenance.

## Frozen Simulation Stack

The candidate comparison will run natively in Isaac, not in the local MuJoCo
proxy:

- environment and evaluation substrate: Isaac Lab-Arena at exact revision
  `3c19a3a9e45fc2cc1b64ab8a43047ecac9c0ad4d`;
- Arena lock digest:
  `35001404fa10d3f591d326d7a36b15c0b35cf307b754edea87310f719ec439da`;
- task framework: Isaac Lab revision
  `af1bab4dc173ba69b08fab779c14ead61d13fd33` using a manager-based RL
  environment;
- simulator: Isaac Sim `6.0.0.1`;
- physics: PhysX;
- rendering and policy observations: Isaac RTX;
- embodiment and action interface: Arena `droid_abs_joint_pos`;
- Blueprint integration: a thin schedule, receipt, and media adapter around
  Arena's existing environment and remote-policy seams. Blueprint will not fork
  Arena or build a parallel task framework.

Isaac Lab-Arena is alpha software. Pinning it does not qualify it: native
zero-action, scripted positive, camera/action/reset parity, media, and runtime
identity canaries must pass before either model is allowed to run.

MuJoCo remains a cheap local control oracle only. It uses the exact pinned
Menagerie Franka at revision
`71f066ad0be9cd271f7ed58c030243ef157af9f4`; its outcome is not candidate,
native-Isaac, or sim-to-real evidence.

The protocol is generated deterministically by
[`adp_founder_sim_protocol.py`](../../src/blueprint_pipeline/adp_founder_sim_protocol.py),
and the GR00T ZMQ/DROID translation is isolated in
[`groot_n17_droid_policy_runtime.py`](../../src/blueprint_pipeline/groot_n17_droid_policy_runtime.py).

## Immutable Digests

- Protocol digest:
  `sha256:3940d30009c4d9fdadac1ba4419eb2f04f8b22fd0273131f31d0112e1e2f6ad2`
- Executable schedule digest:
  `sha256:b1c9fb7ee06c7f48014225868318e80e1652a69c1d6687347b5c2687a8ab34c6`
- Frozen non-executable Arena worker-request digest:
  `sha256:dd1d79206299a0d6954ee91daef6d168e42e1d054abafec33284f30348ad2650`

Any candidate, model revision, scene, reset, metric, schedule, controller,
evidence, or claim change creates a different digest and requires a new
approval.

## Founder Approval

One human approval is sufficient because this phase has no partner and no
physical holdout. Approval must quote the exact protocol digest after review:

> I approve protocol
> `adp-founder-sim-arena-droid-pi05-vs-groot-n17-v1` with digest
> `sha256:3940d30009c4d9fdadac1ba4419eb2f04f8b22fd0273131f31d0112e1e2f6ad2`
> in my role as Blueprint founder and simulation task owner. I approve only the
> development-only simulation phase; this does not authorize physical robot
> motion, deployment, publication, or uncapped paid compute.

Blueprint's founder supplied that exact statement in the Codex task on
2026-08-04. The deterministic receipt is
[`founder_sim_approval_receipt.json`](manifests/founder_sim_approval_receipt.json),
digest
`sha256:94ae151d8a7dca5a86bef37ef32233b13b9aa0a2ba0a73eba9d2c0dcdb69c9e9`.
The resulting non-paid execution admission is
[`founder_sim_execution_admission.json`](manifests/founder_sim_execution_admission.json),
digest
`sha256:5f586feea7211996edac22413d2a320d7ba79a231e3983fe64f964f9d595b4fa`.

The exact 88 logical Arena jobs are compiled by
[`adp_isaac_lab_arena_request.py`](../../src/blueprint_pipeline/adp_isaac_lab_arena_request.py).
That artifact binds environment, policy adapter, candidate identity, seed,
execution order, cameras, and evidence overlay, but remains explicitly
unauthorized for execution and paid compute.
The generated JSON artifacts are retained under
[`prospective_protocol`](../../output/arm_decision_proof_v1/prospective_protocol/).

## Runtime Preconditions Still Required

Founder protocol approval does not itself spend money or start 88 rollouts.
Before production simulation:

1. Materialize and byte-verify both checkpoints. GR00T must emit a worker
   receipt binding the exact checkpoint files and environment lock.
2. Materialize the exact Arena/Isaac Lab/Isaac Sim lock and byte-manifest the
   registered maple table, Rubik's cube, bowl, HDR, and DROID assets.
3. Pass native zero-action negative, replay-or-scripted positive, and
   camera/action/reset parity canaries in the built-in Arena environment.
4. Run one media-complete dry-run episode for each policy adapter; model outcome
   is ignored at this gate.
5. Use the canonical paid-resource allocator with a quoted price, hard spend
   cap, TTL, watchdog, teardown, and provider-zero verification.

The worker-side byte receipt and admission boundary are implemented in
[`adp_isaac_lab_arena_materialization.py`](../../src/blueprint_pipeline/adp_isaac_lab_arena_materialization.py).
It requires clean exact-revision Arena, Isaac Lab, OpenPI, and GR00T checkouts;
the frozen Arena lock; byte-complete inventories for the runtime lock, five
registered environment/embodiment groups, and both checkpoints. Passing it
authorizes only native control canaries, never candidate jobs or paid compute.

No production simulation, capture, reconstruction, physical trial, upload, or
paid allocation was started while freezing this protocol.

The reusable local harness-control command has now passed. Its receipt used schema
`franka_droid_control_preflight.v1` and digest
`sha256:30cc9636bc4a90b023f89e7aca65c6ebd66229462562902e89ad4b5ef48c1106`.
The scripted control completed 168 actions, lifted the can by
`0.10951612958149567` metres, placed it in the tray, and succeeded. The
stationary control completed 24 actions, produced zero lift, and failed. Both
episodes satisfied the lossless-frame, terminal-frame, manifest, video, and
independent-grader metadata contract. The retained evidence is under
[`prospective_control_preflight`](../../output/arm_decision_proof_v1/prospective_control_preflight/).
Because this proxy uses a different can-to-tray task and stock Menagerie
gripper, it proves only the reusable runner/media contracts—not the Arena task,
DROID gripper, model adapters, or native physics.

## Scenario and Cousin Policy

This first protocol uses one scene, one object, one destination, one HDR, and
one lighting value. The 44 reset seeds produce repeatable collision-free object
placements through Arena's pinned relation solver; they are repetitions, not
scene cousins. Each adjacent randomized two-trial block contains one baseline
and one alternative episode under the same reset and seed.

Object swaps, HDR/background changes, lighting, cameras, mass, friction, and
other cousins are forbidden in this digest. After this baseline is executable,
a separate robustness protocol may use Isaac Lab-Arena's variation system to
freeze such cousins. An environment-generation agent may propose variants only
before approval; it may not change a frozen matrix or react to candidate
outcomes.

## SimReady Asset Generation Decision

No new asset generation is needed for this test: the exact Arena environment
already registers the maple table, Rubik's cube, YCB bowl, HDR, DROID
Franka-plus-Robotiq embodiment, task, success metric, and both policy adapters.

Blueprint does have a provider-neutral SimReady request/draft/preflight seam in
[`simready_asset_lane.py`](../../src/blueprint_pipeline/simready_asset_lane.py).
Lightwheel SimReadyGen is an observed generator source—the sink asset used in
the 2026-08-03 Isaac canary came from that service. The retained Isaac canary
attempts did not produce a passing terminal validation artifact, so the sink
remains generated support evidence rather than a qualified asset.

NVIDIA USD Content Agents should be added and tested only when a selected task
has a real asset gap. The minimum useful canary is one small source USD through
Physics Agent and Validation Agent, followed by an independent Isaac
drop/contact/stability test. Material, texture, and joint agents are added only
when that asset's task requires those properties. The current repository plans
the provider and detects the optional `validation-agent` executable, but it
does not yet install or invoke a live Content Agents adapter.
