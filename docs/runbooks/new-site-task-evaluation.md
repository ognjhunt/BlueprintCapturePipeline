# New-site Task Evaluation Run

Use this path to exercise a previously unseen capture and one explicit task
without granting an agent, provider, or simulator authority to weaken the
measurement contract.

## What the command does

`blueprint_pipeline.new_site_task_evaluation` joins capture intake and
content-addressed materialization to the observed-site compiler, deterministic
task-requirement compiler, production measurement router, controlled local
development evaluator, Decision Envelope, and optional redacted WebApp
projection. Every transition is digest-bound. A production method remains
unavailable until an exact approved R7/R8 measurement qualification exists.

The committed `new_site_loading_bay_v1` data is a newly introduced
**fixture-only capture contract**. Its marker payloads and declared measurements
test the E2E control plane; they are not a customer capture, independent
metrology, physical evidence, or method qualification. Replace the entire
fixture directory with a newly collected bundle and observed evidence before
using this command on a real site. Do not copy fixture measurements into a real
site package.

## Required input directory

The directory passed to `--fixture-root` contains:

- `capture_intake_envelope.json`, whose `original_files` list binds every raw
  file by relative path, byte length, and SHA-256 and whose governance block
  records rights, consent, privacy, retention, revocation, and provider limits;
- `raw/observations.json`, with at least two retained RGB/depth frame bindings,
  integer nanosecond timestamps, 4x4 camera-to-site transforms, coordinate-frame
  convention, metric-scale observation, and an observed-volume bound;
- the raw files referenced by the intake envelope and observation manifest;
- `site_evidence_complete.json`, containing only artifacts actually observed or
  independently measured and bound to the exact capture-envelope digest;
- `task_spec.json`, with the exact site, task, claim, robot, sensor, target
  region, restrictions, and zero-spend development adapter identity.

The site-evidence file may carry raw/capture QA, metric scale, geometry status,
qualified colliders, robot/site registration, articulation, material, and sensor
calibration. Omit anything not measured. Never use `status=passed` or
`observed_or_independently_measured=true` merely to make the route pass: the
operator is responsible for the referenced evidence and its digest.

## Run it

Create and use a checkout-local environment:

```bash
uv venv --python 3.12 .venv
uv sync --frozen --extra dev
```

If an emergency diagnostic temporarily reuses another checkout's environment,
set `PYTHONPATH="$PWD/src"` so imports still resolve to this checkout.

Run the complete fixture lane:

```bash
.venv/bin/python -m blueprint_pipeline.new_site_task_evaluation \
  --fixture-root tests/fixtures/new_site_loading_bay_v1 \
  --state-root output/new-site-task-evaluation/complete
```

Run the intentional sensor-calibration abstention:

```bash
.venv/bin/python -m blueprint_pipeline.new_site_task_evaluation \
  --fixture-root tests/fixtures/new_site_loading_bay_v1 \
  --site-artifacts site_evidence_incomplete.json \
  --state-root output/new-site-task-evaluation/incomplete
```

The first lane returns a complete, zero-spend **development-only** Task
Evaluation Run for captured visibility. Its separately retained production plan
still abstains at `qualification_benchmark` because no R7 entry exists. The
second lane performs no evaluator execution and names `sensor_calibration` as
the smallest next measurement, covering calibration and timing.

Result files are created exclusively under
`<state-root>/runs/<run-id>/<result-sha256>.json`. Re-running identical inputs is
idempotent; a same-digest content collision fails closed. Use
`--no-webapp-projection` when no projection should be emitted.

## Operator checks

Before accepting an output, confirm:

- every value in `digest_joins` is `true`;
- the production plan is unchanged and either deterministically selected or
  contains an exact abstention;
- `supervisor_proposals` is non-authoritative and every proposal is
  `shadow_only` with `proof_effect=none`;
- `paid_compute_authorized`, `provider_execution_authorized`, and
  `physical_robot_run_authorized` remain false for this command;
- no R7 catalog entry was created;
- physical success, deployment, safety, and comparative policy ranking remain
  unclaimed (`thesis_not_supported`).

This path does not allocate GPU capacity. If a future task genuinely requires a
paid evaluator, stop at its abstention and use only the canonical
`python -m blueprint_pipeline.paid_resource_allocator gpu-canary ...` lifecycle
with an explicit cap, TTL, independent watchdog, immutable inputs/results, exact
teardown, object cleanup, and provider-zero verification.

## Downstream five-policy run

This command is capture admission, explicit-task routing, and development
evidence—not the five-policy Task Evaluation Run compiler. After independently
qualified reconstruction, target binding, placement, engine routing, and five
matched-reset learned-policy executions exist, pass those digest-bound artifacts
to `python -m blueprint_pipeline.new_site_task_evaluation_run`. The downstream
compiler ranks only supported receipts and remains fail-closed; this development
result cannot be used as a policy attempt or comparative-ranking receipt.
