# Second-scene harness failure-prevention audit — 2026-08-09

Status: **implemented and locally qualified; native refrigerator qualification
is still in progress**.

This audit covers the reusable failures exposed while moving ADP-009D from the
`840313` rigid canned-object fixture to the `840796` articulated refrigerator
task. It unblocks the day-14 public-scene construction and native-control gate.
The observed completion artifact is a clean, pushed implementation commit plus
a focused hermetic regression for every invariant below. It does not itself
qualify the refrigerator, controls, learned policies, or physical behavior.

## First-principles findings

The failures did not come from one defective refrigerator. They came from
contracts that let a later and more expensive stage discover facts that the
earliest responsible stage could have rejected locally:

1. input scope was not always closed before Aura or research-agent execution;
2. a bundle plan could exist without proving that its exact entrypoint and
   complete import closure were runnable;
3. research-preview agent output was allowed to look like a prerequisite even
   when deterministic construction already owned the required physics;
4. static USD validity did not prove the APIs, driver, articulation
   construction, or material bindings that Isaac would actually read;
5. provider admission used an incomplete hourly-price interpretation and did
   not distinguish pre-execution provider nulls from scientific attempts;
6. progress written only to an internal log was invisible to the provider
   watchdog.

The stable correction is therefore a sequence of fail-closed gates. Each gate
owns one question and emits typed evidence; no later stage may infer that an
earlier gate passed merely because an artifact exists.

## Landed invariants

| Failure class | Reusable invariant now enforced | Hermetic coverage | Commit |
|---|---|---|---|
| Aura received bytes outside its declared scene-derived scope | Every Aura input is enumerated and scope-admitted before any upload or execution; undeclared bytes produce a typed blocker. | Both admitted and out-of-scope fixture bundles. | `e1f645d53` |
| A paid GPU discovered a malformed or incomplete runtime bundle | The exact archive is extracted and its real entrypoint is rehearsed locally up to the explicit GPU boundary before admission. | Exact-entrypoint success plus missing-module and changed-byte failures. | `c4f42dd89` |
| Scene fixtures or research agents were treated as qualification authorities | Fixtures are development-only, and Joint/Material/Texture Agent results are optional enrichment rather than native-gate prerequisites. | Original `840313` and new `840796` fixtures; agent-null admission cases. | `025440885` |
| A provider/container failure was confused with a scientific run | Attempt classification records whether the provider entrypoint started and whether the scientific attempt was consumed; automatic requeue remains disabled. | Pre-entrypoint exit, returned provider null, and scientific-attempt fixtures. | `76be286ae` |
| A flat neutral-gray USD could reach policy-visible cameras | Every policy-visible render Gprim must resolve an authored render material; collision-only and neutral fallback assets fail closed. | Original rigid fixture plus materialized articulated refrigerator and missing-binding mutations. | `c54e8454d` |
| Static material validation was mistaken for native renderer proof | The blank-stage Isaac diagnostic reads back material bindings and retains calibrated material/review frames; static admission remains a separate field. | Typed request, camera-role separation, material-path mismatch, and retained-frame contracts. | `6d2940885` |
| Generated bundles were not bound to the code being diagnosed | Diagnostic bundles bind the clean current commit and reject stale or caller-asserted implementation identities. | Changed-commit and dirty-checkout cases. | `d6f5c707c` |
| The articulated bundle fell through to the legacy canned-object/WAM transport | `adp009d_articulated_native` has an explicit native Isaac transport profile and cannot invoke learned policies or controls in its blank-stage diagnostic. | Complete transport closure and forbidden mixed-mode tests. | `6d1c51cfe` |
| Vast admission compared only compute price with the hourly cap | Offer selection and post-create validation use the all-in hourly rate, including storage surcharge, and tear down if observed rate exceeds the cap. | Under-cap offer, compute-only false pass, and post-create over-cap fixtures. | `d278143f8` |
| Isaac 6 removed the legacy dynamic-control module used by the diagnostic | The runtime uses `SingleArticulation` and performs one aggregate capability probe so all missing runtime modules/symbols are reported together. | Complete Isaac-6 symbol closure and aggregate missing-symbol cases. | `250ae942b` |
| Isaac 6 started on an unsupported driver and failed inside Vulkan | The articulated native lane admits only a proven Isaac-6 driver branch (`>=580.65.06`) and retains driver identity. | Boundary versions and provider-offer filtering. | `0d32f82c1` |
| PhysX discarded joints because the fixed cabinet was a kinematic articulation link | Fixed bases are anchored to world with a `UsdPhysics.FixedJoint`; articulation links remain non-kinematic. | Blank USD verifies empty `body0`, cabinet `body1`, and `kinematicEnabled=false`. | `d42c1362d` |
| Useful runtime phases appeared only after the process exited | Provider stdout is streamed through `tee` with `pipefail`, while the runtime's real exit status is preserved. | Entrypoint source contract verifies live stream, log retention, and `PIPESTATUS[0]`. | `d42c1362d` |

## Agent-output boundary

The refrigerator's required physics and appearance do not depend on research
agents succeeding:

- the two refrigerator revolute joints, rigid bodies, collision shapes, mass,
  inertia, friction, restitution, limits, and fixed-base anchor are
  deterministic USD authoring outputs;
- the observed exterior color and the explicitly generated/unobserved interior
  material are authored render-material outputs;
- the Joint Agent may propose or improve topology, and the Material/Texture
  Agents may enrich appearance, but their output must pass the same static and
  native gates before substitution;
- the neutral-gray silhouette renderer remains valid only for coverage audits.
  Its frames are never policy observations or appearance qualification.

Therefore an agent null cannot stop the must-have construction path, and an
agent success cannot bypass deterministic validation.

## Paid native diagnostics retained after the original v3 abstention

| Run | Exact result | Cost (USD) | Generic correction |
|---|---|---:|---|
| v4 | Articulated bundle fell through to the legacy transport. | 0.043681 | Explicit articulated-native transport (`6d1c51cfe`). |
| v5 | Provider instance exited before the bundle started. | 0.012404 | Typed pre-execution provider null; no scientific retry claim. |
| v6 | Isaac reported `No module named omni.isaac.dynamic_control`. | 0.058192 | Isaac-6 articulation API and aggregate dependency probe (`250ae942b`). |
| v7 | Driver 535 produced Vulkan/GPU-foundation failure. | 0.083108 | Proven driver floor (`0d32f82c1`). |
| v8 | Isaac reached the asset, but PhysX rejected kinematic articulation links and no articulation initialized. | 0.041064 | World fixed-joint anchor and live phase telemetry (`d42c1362d`). |

The five retained runs total **USD 0.238449**. Each run tore down and was
followed by API-confirmed provider zero. No automatic retry occurred; every
subsequent run followed a focused code change, hermetic test, clean immutable
commit, and fresh bounded authority already supplied by the user.

## Current gate and claim ceiling

The v9 exact bundle rehearsal passed locally at commit
`d42c1362db6f8793930a11808afacede2cdc1038`, bundle digest
`sha256:657d6257094ecfbc9e60a99788cc3bbb810f9743ce4a33a8a6538ca0853a373f`.
Its paid launch is held until an independently running Content Agents job tears
down and the Vast API again reports provider zero.

Until v9 completes, the strongest honest claim remains: the materialized
refrigerator is statically admitted and its native diagnostic is locally
rehearsed. Native articulation motion, reset, rendered appearance, task-scene
composition, controls, learned-policy outcomes, physical equivalence, and
deployment readiness remain unresolved.

