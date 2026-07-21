# MuJoCo-Parity G1 Eval on Isaac Sim + GPU — Design Spec

> Archived implemented design snapshot.

Date: 2026-06-27
Status: Approved (staged build in progress)

## Goal

Run **exactly what the MuJoCo lane runs** — a Unitree G1 walk-to-target navigation eval
(policy + per-step trace + WAM evaluation + harness + MP4) — but **executed inside Isaac Sim
on a real GPU**, using the sim-ready Lightwheel kitchen USD and the official Isaac G1 USD.

Parity flip: the MuJoCo lane runs a deterministic collision-aware walk-to-target preview
controller and renders MP4s; the Isaac lane runs the **same controller** against the same
scenarios, RTX-rendered on GPU, producing the same task-outcome contract.

## Locked decisions

1. **Staged policy.** Stage A = the deterministic walk-to-target controller (literal MuJoCo
   parity, guaranteed to run, de-risks the whole Isaac+GPU+render+WAM lane). Stage B = swap in
   the GR00T N1.7 SONIC VLA (`LucaFrat/groot-bs16`, embodiment `UNITREE_G1_SONIC`) as a
   pluggable policy, no harness change.
2. **Provider-agnostic launch** (RunPod or Vast) via `gpu_render_providers` (already built).
3. **Honesty boundary.** Stage A is a *kinematic* navigation preview (root placed along a
   collision-checked path), RTX-rendered — same semantics as MuJoCo's preview controller. It is
   not dynamic locomotion and not a learned policy until Stage B.

## Ground truth (verified in repo)

- The Lightwheel kitchen is a **sim-ready, decomposed USD**: `Collected_KitchenRoom/KitchenRoom.usd`
  with per-object sim-ready USDs (`Sink054`, `Dishwasher054`, `Stovetop012`, cabinets, …).
- The official Isaac G1 USD (`Isaac/Robots/Unitree/G1/g1.usd`) is sim-ready (PhysX
  ArticulationAPI / JointAPI / CollisionAPI). Codex's existing runner already verifies these
  APIs bind — it is a **binding probe**, not an eval (no stepping, no controller, no MP4).
- Navigation scenarios already exist (`entry_to_sink`, `narrow_passage_to_sink`, …) with spawn
  + target poses and success criteria — the same walk-to-target shape MuJoCo uses.
- The MuJoCo controller is a deterministic collision-aware preview
  (`mujoco_g1_simulator_command.py`): interpolate waypoints → propose candidate root poses
  (direct → lateral redirects → stop/relocation) → probe each for scene collision → accept the
  first collision-free → compute task outcome. **`POLICY_ID = blueprint_default_walk_to_target_smoke_policy`.**
- WAM = world-action-model (OSCAR/COSMOS) **video-rollout fidelity** evaluator
  (`oscar_cosmos_wam_evaluator.py` + `wam_derived_observation_harness.py`) — it grades whether a
  generated rollout video is plausible, **not** task success. Task success is the deterministic
  outcome contract above.

## Architecture

### Component 1 — pluggable policy (`isaac_g1_policy.py`) — BUILT

A pure, sim-agnostic policy module. `DeterministicWalkToTargetPolicy` is a **verbatim port** of
the MuJoCo controller math (route interpolation, candidate generation, `policy_action`
labelling, and the `compute_task_outcome` contract). The only sim-specific piece — probing a
candidate pose for scene collision — is injected by the host via `StepContext.probe_collision`
(MuJoCo: `mj_forward` + contacts; Isaac: PhysX overlap). `Groot17SonicPolicy` is the Stage-B
slot (GPU-only, fail-closed off-GPU). Parity is proven by a test that asserts the port is
**byte-identical** to the MuJoCo source functions (`_interpolate_route`, `_candidate_pose_specs`,
`_attempt_task_outcome`).

### Component 2 — Isaac GPU runner (`scripts/run_isaac_g1_kitchen_parity_eval.py`) — GPU-only

Runs inside `/isaac-sim/python.sh` on the worker. Self-contained (ships the policy module in
the bundle). Per scenario:
1. Boot `SimulationApp(headless)`, open kitchen USD, `add_reference_to_stage` the G1 USD, verify
   ArticulationRootAPI + CollisionAPI (reuse codex's binding probe).
2. Add a PhysicsScene + ground; create two RTX cameras (overview, robot-POV) as render products.
3. Run the controller loop: each step the policy proposes candidates; the runner probes each via
   a **PhysX overlap query** of the robot footprint at the candidate pose against scene
   colliders, returns the scene-hit count; the policy accepts the first collision-free pose;
   the runner kinematically places the G1 root there, steps, and RTX-renders the capture frames.
4. Assemble overview.mp4 + robot_pov.mp4 (ffmpeg), write the MuJoCo-schema per-step trace JSONL,
   compute the per-scenario outcome via `compute_task_outcome`.
5. Emit `isaac_g1_kitchen_parity_result.json` and upload the out dir via the signed-PUT contract.

Testable, non-Isaac helpers (scenario parsing, collision-summary assembly, result shaping, MP4
command) are factored out and unit-tested; the Isaac-API calls are lazily imported and
**GPU-unverified** (same honesty boundary as the splat runner).

### Component 3 — job orchestration (`isaac_g1_kitchen_parity_job.py`) — testable

One productionized job mirroring the splat render job: resolve kitchen + G1 assets → build the
bundle (kitchen USD refs + G1 USD ref + scenarios + the runner + the policy module) → stage to
the object store (signed GET/PUT) → launch on the chosen provider (`gpu_render_providers`,
RunPod or Vast, cold-create) → watch + collect → run the WAM evaluator on the returned traces
and emit the harness artifact with the honest claim boundary. Paid launches gated behind
`allow_paid`; hermetic tests cover bundle/plan/trace-adaptation.

## Data flow

```
scenarios (spawn→target) + kitchen USD + G1 USD
  → bundle + stage (signed URLs)
  → provider GPU pod runs run_isaac_g1_kitchen_parity_eval.py
      → per scenario: DeterministicWalkToTargetPolicy + PhysX collision probe + kinematic place + RTX render
      → overview.mp4 + robot_pov.mp4 + trace.jsonl + outcome (compute_task_outcome)
  → collect → WAM video-fidelity eval + harness artifact (honest claim boundary)
```

## Success criteria

- **Stage A:** on the GPU, the kitchen + G1 load, the controller runs the navigation scenarios,
  and we get per-scenario RTX MP4s (overview + POV) + traces + a parity outcome JSON whose
  task-outcome contract is identical in shape to MuJoCo's, plus a WAM + harness artifact.
  Truthfully labeled: kinematic preview parity, RTX-rendered on Isaac, not dynamic locomotion.
- **Stage B:** the same harness with `Groot17SonicPolicy` driving the G1 closed-loop
  (per-step camera+state → action), producing the analogous MP4 + outcome.

## Honest constraints / risks

- **Isaac Sim runs only on Linux+NVIDIA** — the runner is authored + its non-Isaac helpers
  hermetically tested locally; the rollout is proven only on the GPU worker.
- **The infra wall is the gating risk.** Codex hit `provider_auth_blocked` / `runtime_blocked`
  on the same Isaac image. Provider-agnosticism (Vast as well as RunPod) is the new lever, but
  getting Isaac to boot remains the uncertainty. Stage B needs an even larger image (3B
  checkpoint + the `gr00t` inference stack).
- **The PhysX overlap probe** is the trickiest Isaac-API piece and will likely need on-GPU
  iteration (footprint extent, hit filtering for robot/ground).
- **Real GPU spend**, and Stage B carries real model-integration uncertainty (checkpoint
  compatibility, embodiment obs/action wiring) — which is exactly why the deterministic stage
  goes first.

## Scope (YAGNI)

- **In:** the G1 walk-to-target navigation eval on Isaac GPU (Stage A), then GR00T (Stage B);
  RTX MP4 + trace + outcome + WAM + harness; provider-agnostic launch.
- **Out:** manipulation tasks (e.g. turning the sink handle — needs an articulated faucet asset
  and a manipulation policy, neither of which exists); rendering the splat *as physics*.
