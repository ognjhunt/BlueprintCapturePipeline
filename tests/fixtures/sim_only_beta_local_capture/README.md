# Sim-Only Beta Local Gate Fixture

This fixture is a synthetic capture-root input for `scripts/run_sim_only_beta_local_gate.py`.
It exists so a fresh clone can regenerate a local sim-only gate report without a
real customer capture.

Boundaries:

- Synthetic fixture only; not a real site, customer capture, or upload proof.
- `fallback_allowed_for_beta_release=false`.
- No public claim upgrade, generated-world rank-fidelity claim, production
  forwarding proof, provider proof, payment proof, or delivery proof.
- The fixture includes a text-encoded tiny GLB scene and a minimal fixture MJCF
  smoke asset. These are local gate inputs only, not MuJoCo Menagerie fidelity
  evidence or real Unitree G1 readiness evidence.
- Generated gate artifacts are written under `output/`, not into this fixture.
- CI regenerates the local report via `.github/workflows/sim-only-local-gate.yml`
  and uploads only compact JSON evidence, not the full generated media tree.
