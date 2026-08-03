# CI Python Interpreter Matrix

Status: canonical interpreter policy for Pipeline launch evidence.

The machine-readable source of truth is
`docs/CI_PYTHON_INTERPRETER_MATRIX.json`. Keep this document, `pyproject.toml`,
`uv.lock`, and all GitHub Actions `python-version` declarations aligned through
`scripts/validate_python_interpreter_matrix.py`.

## Canonical Launch Evidence

Canonical Pipeline launch evidence is Python `3.12`.

The following checks must run on Python `3.12` before their output can be used
as launch or deploy proof:

| Check | Workflow | Evidence role |
| --- | --- | --- |
| `CI / Impacted test and sentinel gate` | `.github/workflows/ci.yml` | Bounded impacted tests plus contract, security, paid-resource, and release-policy sentinels. |
| `Full Test Lane / Full pytest lane on CPU runner` | `.github/workflows/full-test-lane.yml` | Full CPU pytest lane required for deploy evidence. |
| `Sim-Only Local Gate / Regenerate sim-only local gate artifact` | `.github/workflows/sim-only-local-gate.yml` | Sim-only local gate artifact regeneration. |

## Compatibility Interpreters

`BlueprintCapturePipeline` supports package installation and local development
on Python `3.10`, `3.11`, and `3.12`; `pyproject.toml` and `uv.lock` must keep
`requires-python = ">=3.10,<3.13"`.

The `Python Compatibility` workflow installs the frozen lock independently on
all three advertised interpreters and runs the bounded grounding, public-claim,
and SC3 protocol contract suite declared in the machine-readable matrix. Those
jobs prove package/contract compatibility only; the canonical launch lanes in
the table above must still pass on Python `3.12` for the exact release commit.

Python `3.13` is not launch-proof evidence. If a local paid-gate subcommand or
operator run uses Python `3.13`, record it as non-canonical and rerun the gate
under Python `3.12` before making a launch-readiness claim.

## Validation

Run:

```bash
python scripts/validate_python_interpreter_matrix.py --assert-current
```

`--assert-current` is required in CI and launch-gate environments so a shell
using Python `3.13` fails instead of producing mixed-version proof.
Compatibility jobs intentionally omit `--assert-current`; the validator still
requires their workflow matrix to cover every advertised package interpreter.
