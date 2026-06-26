# MuJoCo G1 Live-Product Path - 2026-06-12

## Decision

Use MuJoCo as the primary simulator lane for the next cheap serious live-product
proof, with Unitree G1 sourced from MuJoCo Menagerie and Pipeline ingestion kept
fail-closed.

This does not remove Isaac Sim or Isaac Lab-Arena. It demotes them to richer
follow-on lanes until the cheaper MuJoCo path proves the live request, real
asset, default task execution, and simulator POV surfaces without requiring an
RT-core GPU rental first.

## Source Basis

- MuJoCo is free and open source: https://mujoco.org/
- MuJoCo Python bindings install through `pip install mujoco` and include the
  MuJoCo library: https://mujoco.readthedocs.io/en/stable/python.html
- MuJoCo Menagerie is the Google DeepMind curated model collection and includes
  `unitree_g1` with 29 DoF and BSD-3-Clause licensing:
  https://github.com/google-deepmind/mujoco_menagerie
- Unitree's own public G1 description package also names G1 URDF/MJCF variants:
  https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/README.md

## What This Lane Can Prove

When `blueprint-build-first-gpu-run-packet --simulator mujoco` is used, the
packet now selects:

- robot asset: `Unitree G1`
- asset URI/path:
  `output/external_assets/mujoco_menagerie/unitree_g1/g1.xml`
- asset source: `google_deepmind_mujoco_menagerie`
- asset class: `humanoid_mjcf`
- generated owner command:
  `run_mujoco_unitree_g1_smoke.sh`

The generated MuJoCo smoke loads the staged World Labs GLB as converted OBJ
support, loads the real Menagerie G1 MJCF and mesh assets, runs the built-in
default `walk_to_target` smoke, captures MuJoCo renderer frames, and writes the
owner proof traces expected by `blueprint-run-owner-gpu-proof`.

Accepted owner-runtime MuJoCo proof can set:

- `owner_gpu_simulator_execution_proven=true`
- `mujoco_g1_asset_execution_proven=true`
- `owner_gpu_default_policy_execution_proven=true`
- `owner_gpu_sim_robot_pov_evidence_proven=true`

The local CPU MuJoCo smoke can set only:

- `local_mujoco_g1_asset_execution_proven=true`

## What Remains Blocked

The MuJoCo path still cannot prove:

- live WebApp forwarding or upstream truth without real non-placeholder
  `site_submission_id`, `request_id`, `buyer_request_id`, and `capture_job_id`
- robot-team policy quality beyond the default smoke policy
- owner-run POV
- physical generated-world rank fidelity
- safety/contact validity
- signed customer delivery
- public claim upgrades

The correct next live-product order is:

1. Keep running the local MuJoCo G1 smoke until it passes against the staged
   capture root and Menagerie asset cache.
2. Generate a MuJoCo packet with `--simulator mujoco`.
3. Stage a real WebApp `robot_eval_job_request.v1` through
   `blueprint-intake-live-pipeline-inputs`.
4. Rerun cross-repo readiness. Do not rent or allocate owner runtime while the
   WebApp truth or staged request gates are blocked.
5. Run the generated MuJoCo owner command only after the launch order allows
   owner simulator proof.
6. Run closure audits and keep real robot POV, robot-team policy, safety/contact,
   and delivery gates separate.

## Stop Rule

If any generated artifact claims generated-world rank fidelity, generated-world policy-evaluation evidence, live
forwarding, or customer delivery from only local MuJoCo or owner-runtime MuJoCo
evidence, treat the artifact as invalid and fix the proof boundary before
continuing.
