# Lightwheel SimReadyGen sink: first-party Isaac articulation evidence (2026-08-04)

First Blueprint-owned execution evidence for a `lightwheel-simready` generated
asset (catalog entry `lightwheel-simready`,
`src/blueprint_pipeline/measurement_method_research_catalog.py`). One
vendor-generated kitchen-sink asset was exercised in Isaac Sim 6.0.1 with PhysX
on a rented Vast L40 through the guarded canary on branch
`codex/lightwheel-sink-isaac-canary-20260803`.

This is a single-sample development observation about one generated asset. It
upgrades nothing past the entry's `C2` ceiling and creates no qualification;
vendor claims stay EC.

## Exact identities

| Item | Value |
| --- | --- |
| Source asset (`model.usd`, unmodified throughout) | `sha256:41ea847fe8e7304a081c2bcfa70cbdce80875167a660f92372249154804c8945` |
| Texture manifest (5 files) | `sha256:523f547025c360390a4d1f366650309c52f1131cd65b7d41134fc4e6008f57c9` |
| Derivative test wrapper (physics scene + articulation root + world anchor) | `sha256:5842b9c7ae5f5de886efd91705b456d2e755fcd70e6a10e5b262774decebdc1f` |
| Input bundle | `sha256:ed969b9e8bc0d774c301384c6b007999dbca34dfe1cd8a4280aff9ec86755bc6` |
| Worker source commit | `6501797e174d1b44c902aa507f68edc55ed25803` |
| Runtime image | `nvcr.io/nvidia/isaac-sim:6.0.1@sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9` |
| Vast instance (torn down, provider-zero verified) | `46757942` (NVIDIA L40, driver 570.211.01) |
| Uploaded runtime result (interim, `lightwheel_sink_franka_stage_incomplete`) | file sha256 `16961971ea2e6e022d4dc5a68cf0e7c0b5d502919b6149efe4b2b41b86919387` |

Local evidence root:
`output/lightwheel_sink_isaac_canary_20260803_attempt10/` (runtime result,
adapter/teardown/provider-zero receipts, decoded frames).

## What the generated asset demonstrably supports (PhysX, measured)

- The authored revolute handle joint is real and driveable: position-drive
  targets 0/30/60/90/120 degrees each reached within 0.002 degrees; the
  authored 0-120 degree limits held.
- Contact interaction works: a kinematic capsule pushing the passive handle
  rotated it **93.2 degrees through contact alone** (65 contact-report events,
  maximum penetration 0.24 mm, sub-millimeter contact resolution).
- With the wrapper's world anchor, sink base displacement stayed 0.000 m under
  drive and contact loads.
- The 4096x4096 base-color texture and OmniPBR material bind and render under
  RTX (basin/counter textured correctly in the captured 640x480 frames).

## Defects and limitations observed in the generated asset

- The asset requests `convexDecomposition` on five full-resolution meshes;
  PhysX VHACD cooking of those stalled parse for 25+ minutes (burned the entire
  first GPU attempt). The canary runs a session-layer downgrade to
  `convexHull`; the source layer is untouched (digests verified before/after
  every run).
- Ships unanchored (dynamic base, would be shoved by a robot), with no
  `ArticulationRootAPI` and no physics scene — all supplied by the derivative
  wrapper.
- Handle/faucet submesh renders near-black: metallic/roughness/normal maps are
  1x1 constants and the base color there is dark; only the basin/counter carry
  the real texture.
- Mass/inertia/friction/drive values remain generator estimates
  (`validated=False`); nothing here identifies them.

## Not established

- Franka articulated-robot contact: four runs (instances 46754204, 46755516,
  46755835, 46756087/46757942) ended in an identical clean native `exit 0`
  ("Simulation App Shutting Down", no Python exception, no faulthandler dump)
  at the instant of first Franka-sink contact. Reproducible Isaac Sim 6.0.1
  defect signature in this articulation-vs-articulation configuration —
  attributable to the runtime, not the asset. Untried mitigations: PhysX GPU
  pipeline, collision-group filtering (basin vs robot), sink as
  jointed rigid bodies instead of a `SingleArticulation`.
- Sim-to-real validity of any generated physical parameter.
