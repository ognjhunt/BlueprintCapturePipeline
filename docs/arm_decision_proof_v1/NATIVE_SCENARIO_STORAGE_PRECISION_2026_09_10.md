# Native scenario storage precision and refusal evidence

ADP-009's day-7 development rehearsal must distinguish a correctly stored native
reset from an unapplied scenario. V25's translated-placement cell02 stopped both
policies before the first observation with
`scientific_reset_scenario_application_mismatch`. The failure did not retain its
expected and observed values, so its actual native Y remains unrecorded.

An independent CPU replay of all ten frozen scenario plans found a deterministic
precision defect sufficient to reject even a correctly reset float32 pose:

| Cell | Requested expected Y, meters | Correct float32 stored Y | Raw difference |
| --- | --- | --- | --- |
| 02 | -3.461138 | -3.4611380100250244 | 1.0025024366e-8 |
| 08 | -3.426138 | -3.426137924194336 | 7.5805663968e-8 |

The resolved plan set application tolerance to zero. Comparing the float64
decimal request directly with a float32 native tensor can never pass for these
values. Separately, the acos-based quaternion angle lost precision near equality:
it could accept one representable step of rotation or report a nonzero angle for
identical orientations.

The smallest change compares exact-zero root pose applications with the exact
expected value encoded in the observed native float32 storage. The original
requested expected value and raw error remain in the record; a separate
`native_storage_comparison` records dtype, encoded expected value and residual.
No distance or angular tolerance is added. One ULP of actual native drift still
refuses. Existing positive physical tolerances retain their original comparisons.
Quaternion distance uses a stable chord/atan2 calculation, with each raw
orientation normalized once and quaternion sign equivalence preserved.

`ScientificResetScenarioMismatch` now carries the measured channels to the
worker. The worker seals them into the original failure path before re-raising;
the snapshot is incomplete with an explicit scenario gap, and the policy is not
queried. This changes evidence retention, not the refusing predicate. Original
V25 failures are not rewritten, and the running V25 source is not patched.

Process defect: the pre-paid lifecycle rehearsal covered per-cell orchestration
but did not exercise exact-zero typed native root-pose comparisons for the frozen
translation variants. The new regressions exercise both translations, both yaw
angles, q/-q equivalence, one-ULP errors, unapplied translations, float64 behavior,
positive tolerance behavior and real worker failure sealing before policy query.
The independent ten-cell CPU replay confirms supported applications without
claiming that native execution has rerun. The existing unsupported friction and
material-cousin coverage gaps in cells06/07/09 remain explicit and are not promoted
to applied scenario evidence.

Validation:

- `python -m pytest -q tests/test_native_scenario_storage_precision.py
  tests/test_policy_scenario_failure_evidence.py tests/test_native_task_arena_readback.py
  tests/test_policy_scientific_reset.py`: 47 passed; protects numeric comparisons
  and retained refusal measurements.
- `python -m pytest -q -m 'not gpu'
  tests/test_native_task_arena_policy_canary_lifecycle_rehearsal.py
  tests/test_provider_runtime_import_closure.py`: 20 passed; protects real
  per-cell/client/close orchestration and sealed provider-bundle imports.
- Changed-file Ruff and `git diff --check` passed.

These are CPU verification results. Corrected native execution and any Website
publication must be verified separately against their exact immutable identities.
