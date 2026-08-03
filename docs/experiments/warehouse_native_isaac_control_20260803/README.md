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

## Terminal adjudication

Primary verdict: `warehouse_native_isaac_downstream_fail`.

The bounded campaign proved that the digest-pinned Isaac Sim 6.0.1 image
started on NVIDIA GPUs, loaded the official Franka extension and NVIDIA assets,
entered native PhysX parsing, and reached `World.reset()` after the articulation
compatibility fixes. It did **not** prove a native manipulation episode. The
final immutable worker failed at `RtxCamera.__init__` / `_sensor_base.__init__`
with `ValueError` before frames, settle evidence, five reset cycles, the positive
control, or any of the five frozen controller runs.

This is an independent downstream integration failure, not an ARKitScenes or
capture-derived-geometry result. It therefore does not localize the current
blockage primarily to capture-derived collision geometry. The exact claim-level
Decision Envelope abstains.

| Required gate | Result | Observed boundary |
| --- | --- | --- |
| A. Scene and physics qualification | **Fail / incomplete** | Isaac 6 and PhysX startup were observed. PhysX rejected dynamic triangle meshes on three spray-can children and substituted convex hulls. Frozen settle, overlap, support, penetration, and contact measurements were not emitted. |
| B. Franka and reproducible reset | **Fail / incomplete** | Official Franka composition and one `World.reset()` path were reached. Five complete reset cycles and their state hashes/deviations were not executed. |
| C. Positive control | **Not executed** | Camera/evidence initialization failed first; the success predicate was not weakened. |
| D. Five controllers | **Not executed** | All five definitions stayed frozen; there were no silent retries or discarded episodes. |
| E. Camera and trace evidence | **Fail / partial** | Runtime, GPU, input, traceback, allocator, spend, watchdog, and teardown evidence were preserved. No frames or task traces exist. |
| F. Ranking and Decision Envelope | **Abstain** | Zero controller outcomes means no ordering. The Decision Envelope denies the downstream-success claim and all broader claims. |

Every paid reservation is retained in the external evidence roots. Seven
non-zero GPU reservations consumed 6,739 allocator-accounted GPU-seconds and
USD 1.403959. Seven completed DigitalOcean image builds account for an
additional USD 0.152480313982699329, for a total recorded incremental compute
charge/upper-bound of USD 1.556439313982699329. This is allocator accounting,
not a provider invoice.

The final attempt used Vast instance `46736649`, machine `51579`, and one RTX
A6000. Its independent watchdog stopped the instance and proved both exact-ID
absence and provider-global zero. A fresh read-only inventory also found zero
Vast and zero RunPod resources. All seven DigitalOcean build droplets have
delete-204 / inspect-404 evidence. The exact two staged object-store keys were
deleted and independently returned 404; signed URL files were removed.

Authoritative machine-readable records:

- `terminal_adjudication.json`
- `decision_envelope.json`
- `provider_zero.json`
- `evidence_index.json`

The smallest missing measurement is an unsanitized Isaac 6 RTX-camera failure
message plus a camera-prim validity/uniqueness probe that permits the existing
sensor wrapper to initialize, followed by the already frozen settle/contact,
five-reset, positive-control, and five-controller sequence. No additional GPU
attempt was made after the terminal failure.
