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
consecutive post-upgrade replays produced the same artifact-index digest
`sha256:e009662e90c3d9966d31ccf56e209097c0df223b2332542b86e7823f15db48f2`.
These replays reconstruct the evidence package; they do not rerun the simulator.

The six admitted historical episodes use execution schema v1 and did not retain
observation pixels or video. The evidence package exposes that limitation as
`legacy_execution_missing_required_media` and reports zero human-review media
coverage. Execution schema v2 is forward-only: every newly executed episode must
retain every lossless policy-input image, a frame manifest, a terminal image,
and a derived review video, with digests linked through the episode receipt and
evidence matrix. A completed v2 episode missing any required visual artifact is
invalid and cannot qualify a run.

The result can qualify only Blueprint's bounded retrospective external-reference
harness. It remains `development_only`; it is not prospective validation,
deployment readiness, safety evidence, customer value, a digital twin, general
sim-to-real fidelity, general policy ranking, or rank correlation.
