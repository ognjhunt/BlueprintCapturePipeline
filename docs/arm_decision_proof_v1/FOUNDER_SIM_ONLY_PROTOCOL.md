# Founder Sim-Only Decision Test

Status: **v3 correction frozen; exact-digest founder reapproval pending**
Claim ceiling: **development-only simulation**
Protocol ID:
`adp-founder-sim-arena-droid-pi05-vs-groot-n16-v3`

The approved v1 and v2 protocols are retained as superseded audit artifacts.
A pre-candidate execution audit found that v2 correctly pinned every runnable
GR00T identity to N1.6 but retained one stale descriptive `N1.7` label in its
shared-interface map. Native-control infrastructure was attempted under v2,
but no candidate policy was queried and no candidate outcome was accessed.
V3 changes only that label and amendment provenance; the task, candidates,
stack, statistical design, resets, schedule, and evidence rules are unchanged.

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
- Alternative: `nvidia/GR00T-N1.6-DROID`, pinned to Arena's NVIDIA
  Isaac-GR00T submodule revision
  `e29d8fc50b0e4745120ae3fb72447986fe638aa6` and Hugging Face checkpoint
  revision `ae3ebe8d288971ac53aa30c756ea5cba0f52611b`.
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
- simulator: Isaac Sim `6.0.1.0`, using the linux/amd64 image digest
  `sha256:b1c542b2ecc549b3d1ebb78c25664aa3bacba1709e6ad8e0a68e09426d57dedb`;
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
[`groot_n16_arena_policy_runtime.py`](../../src/blueprint_pipeline/groot_n16_arena_policy_runtime.py).

## Immutable Digests

- Protocol digest:
  `sha256:c9aac12d5643a788ef3195e5f959cc73677bd0f51f3583dd36dd4861d4e12924`
- Executable schedule digest:
  `sha256:f8c4b35234a70c37c04f2e95c1d9792585aa56ca02d647e83fde411447a47005`
- Frozen non-executable Arena worker-request digest:
  `sha256:f5c19a42190052161f6a6e999760542b840f60d845607e11db123f8d552dba9c`

Any candidate, model revision, scene, reset, metric, schedule, controller,
evidence, or claim change creates a different digest and requires a new
approval.

## Founder Approval

One human approval is sufficient because this phase has no partner and no
physical holdout. Approval must quote the exact protocol digest after review:

> I approve protocol
> `adp-founder-sim-arena-droid-pi05-vs-groot-n16-v3` with digest
> `sha256:c9aac12d5643a788ef3195e5f959cc73677bd0f51f3583dd36dd4861d4e12924`
> as Blueprint founder and simulation task owner.

That exact v3 statement has not yet been supplied. The immutable v2 approval
and execution-admission receipts remain retained as superseded provenance; they
do not approve the v3 digest. Missing v3 approval blocks all further paid or
candidate execution.

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

The canonical capped Vast control lane is implemented by
[`adp_isaac_lab_arena_vast.py`](../../src/blueprint_pipeline/adp_isaac_lab_arena_vast.py).
It stages only public, digest-bound inputs and the approval receipt, uses the
exact Isaac image, permits zero retries, and requires the shared price, TTL,
watchdog, teardown, output-return, and provider-zero contracts. Its first
execution target is only the Arena zero-action negative control.

No candidate production simulation, capture, reconstruction, or physical trial
was started while freezing this correction. Prior v2 native-control attempts
are retained as failed precursor evidence and queried no candidate policy.

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
