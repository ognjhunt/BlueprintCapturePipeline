# Native USD reset readback repair

This repairs an ADP-009D Day-28 rehearsal precursor. It does not qualify a
controls-omitted diagnostic run or change a scene, policy, reset, or action limit.

The frozen `277a01c668d74193c6b99b8467e24f1e9cf6e594` attempt retained
`TypeError` gaps for physics, lighting, collider, and scene-asset reset channels.
Its robot, object, contact, and camera channels were retained. Identical missing
channels do not prove reset equality; the original receipts remain unverified.

CPU reproduction identifies the serialization problem: OpenUSD Gf vectors and
matrices support indexed iteration without a `__iter__` attribute. The reader
now uses Python's sequence protocol and recursively converts nested Vt arrays.
Quaternions retain named real and imaginary components. Unsupported values and
nonfinite measurements still fail instead of becoming strings or plan values.

The exact retained book asset also exposes an unauthored
`physics:centerOfMass` fallback of three negative infinities. This is an
[OpenUSD MassAPI schema default](https://openusd.org/release/api/class_usd_physics_mass_a_p_i.html),
not a measured center of mass. Only that exact resolved schema fallback is
retained as an explicit nonnumeric configuration record. An authored nonfinite
value is still refused. Measured PhysX mass and inertia remain separate.

Verification:

- `pytest tests/test_native_usd_reset_serialization.py tests/test_policy_scientific_reset.py -q`
  protects numeric fidelity, strict rejection, all eight native reset channels,
  and detection of a changed light in a real in-memory USD stage: 25 passed.
- CPU readback of the retained book and all 166 source collision prims succeeds
  with their original bytes; no model, GPU, or provider call is involved.
- Bulk numeric-buffer conversion preserves the exact canonical digest of the
  largest 63,382-vertex collider. Full collision-asset value conversion took
  5.56 seconds, down from 27.67 seconds with per-element Gf iteration, on the
  same CPU. This is a readback timing, not simulator performance evidence.
- Changed-file Ruff and `git diff --check` pass.

The next paid attempt must bind this repair to its immutable source and run the
required canary lifecycle and provider import-closure suites before allocation.
Full native reset equality on that next host remains to be measured.
