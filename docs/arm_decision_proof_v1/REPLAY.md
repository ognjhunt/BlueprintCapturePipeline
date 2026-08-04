# Arm Decision Proof v1 Replay

From the repository root, the one-command reconstruction is:

```bash
PYTHONPATH=src .venv/bin/python -m blueprint_pipeline.arm_decision_proof
```

It consumes the admitted manifest, the tracked immutable execution directory,
and the separate published physical-outcome artifact at their canonical paths.
If the execution input is absent, the command fails closed and prints this exact
restore instruction; it never launches a new run or accepts newer inputs:

```bash
git restore --source=HEAD -- docs/arm_decision_proof_v1/immutable_execution
```

The tracked package is the observed attempt-13 result, not a fixture: SIMPLER
commit `06accaca93535902d408da4855f21cece12bceb7`, its
`ManiSkill2_real2sim` submodule at
`ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682`, two genuine RT-1 checkpoints,
six normalized traces, runtime-lock digest
`sha256:3ecd6028b9ae4f24a5303b44df5150aa58c11e8b18655c15da355fc0f0b776a3`,
and execution digest
`sha256:2bc388722999f62e2d1955c73ba3e935beb4a092229d784c11404ad5aef9e97a`.
The canary used Vast machine `41950`, cost an estimated `$0.025857`, had zero
internal retries, and retains teardown, provider-zero, and staged-object absence
receipts. Physical-reference values were not uploaded to or read by the worker.

The command seals the development decision before opening the separately tracked
published outcomes, then exactly joins all six candidate-condition cells. Two
consecutive runs produced the same artifact-index digest
`sha256:6e2ada17343a816b1842d5ef08d69a98d4d002ce29f980c496ef87b16c63f2a3`.

The result can qualify only Blueprint's bounded retrospective external-reference
harness. It remains `development_only`; it is not prospective validation,
deployment readiness, safety evidence, customer value, a digital twin, general
sim-to-real fidelity, general policy ranking, or rank correlation.
