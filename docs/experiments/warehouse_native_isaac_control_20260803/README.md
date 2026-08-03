# Native-Isaac warehouse control experiment

This experiment asks one bounded question: does Blueprint's downstream
Isaac/controller/evaluation path work when capture-derived geometry is removed
and the task instead uses an NVIDIA-authored SimReady control workcell?

The only affirmative claim permitted by the protocol is:

> NVIDIA-authored SimReady control scene; native Isaac physics; scripted
> controllers; single workcell; simulation-only.

The protocol explicitly does not qualify Blueprint capture, ARKitScenes
collisions, a customer site, a learned policy, sim-to-real transfer, physical
success, deployment readiness, or safety.

## Pre-execution ledger

| Area | State before this change | This experiment |
| --- | --- | --- |
| Pinned NVIDIA workcell closure | Reusable: revision `c7fe115cb79c7ddbd0532630d7768b5736b0ecc4`; 168-file closure; manifest digest `3c2839e9847ae13686075b45fc25d67968f6b887ca142b185d3a7b11277b6c0f` | Revalidate every member before bundling |
| Isaac runtime | Reusable: Isaac Sim 6.0.1 digest `sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9` and native-camera startup fixes | Build and attest an experiment-commit image |
| Prior warehouse result | Reusable only as negative scope evidence: visual USD plus MuJoCo, `isaac_physics_claimed: false` | No hybrid or fallback backend is permitted |
| Scene physics | Missing task-specific contact/support qualification | Settle, contact, overlap, penetration, support, collision, and material evidence |
| Franka/reset | Missing native articulation task and reproducibility proof | Official Franka, fixed base/joints, five reset cycles and hashes |
| Positive control | Missing | Frozen can-to-tray positive controller, executed first |
| Controller comparison | Existing definitions were not native-Isaac evidence | Five frozen deterministic variants, matched reset and seed, no discarded attempts |
| Evidence/ranking | Missing native contact/action/object traces and claim envelope | Immutable frames/traces/index, deterministic tie-preserving ranking, Decision Envelope |
| Paid lifecycle | Reusable canonical allocator, watchdog, teardown, and provider-zero contracts | One smallest adequate GPU, four-hour TTL, total incremental cap USD 10 |

## Frozen inputs

- Spec: `../policy_ranking_thesis_20260726/nvidia_warehouse_native_control_spec_v1.json`
- Workcell: NVIDIA heavy-duty packing-table physics asset from the exact pinned
  warehouse closure. This bounded surface avoids unrelated warehouse clutter.
- Task object: pinned NVIDIA SimReady spray can. Source collision schemas are
  retained; native PhysX rigid-body and mass schemas are applied at its root.
- Target: a deterministic collision-authored marked tray composed in the same
  meter/Z-up stage.
- Robot: official Isaac Sim Franka articulation.
- Seed: `260803`.
- Positive control and all five controller variants are frozen in the spec
  before any native-runtime outcome is observed.

The evidence packet and terminal adjudication are added only after the paid run
has terminated and provider-zero has been independently proven.
