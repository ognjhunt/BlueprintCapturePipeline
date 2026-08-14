# Test-suite audit and first optimization (2026-08-14)

## Scope and gate

- Active program: ADP-009D, day-14 production-promotion validation.
- Existing backlog contract: PIPE-05 risk-based test lanes.
- Completion artifact: a full-lane workflow that keeps exact node-ID and
  zero-skip evidence while emitting actionable duration/parametrization data.
- Existing insufficiency: the exhaustive CPU lane runs serially and retains raw
  JUnit timing, but it does not summarize the data needed to shard or retire
  redundant cases.
- Smallest reversible change: four file-isolated pytest workers plus a telemetry
  report. Set the worker count back to one to revert execution parallelism; no
  test is deleted or excluded.

## Evidence audited

Canonical production-promotion run
[31814199890](https://github.com/ognjhunt/BlueprintCapturePipeline/actions/runs/31814199890)
at commit `ac6e43be357fe95e44786fa355e3b884c3965d3c` supplied the exact collection,
JUnit, and CPU evidence used here.

| Measure | Observed |
|---|---:|
| Tests passed | 12,769 |
| Test files | 1,117 |
| Parametrized cases | 1,616 (12.66%) |
| Summed testcase time | 1,802.205 s |
| Full pytest step wall time | 1,828 s |
| Entire job wall time | 2,059 s |
| Slowest test file | 376.787 s / 192 cases |
| Top 20 files' share of testcase time | 50.0% |

The 12,769 count is therefore not mostly pytest parametrization. Parametrized
cases are real contributors to collection size, but the ten largest parameter
families together take less than six seconds. Deleting cartesian cases by count
would not materially improve this lane and could remove boundary coverage.

Runtime is concentrated elsewhere. `tests/test_robot_eval_job_orchestrator.py`
alone accounts for 20.9% of summed testcase time. The next four slowest files are
`test_oscar_cosmos_wam_evaluator.py`, `test_measurement_newton_rigid_adapter.py`,
`test_task_evaluation_supervisor.py`, and `test_vast_provider_adapter.py`.

The PR-gate description also needs precision. The selector's local default is
120 seconds, but hosted CI permits 480 seconds inside a ten-minute job. A recent
representative PR run, [31840559625](https://github.com/ognjhunt/BlueprintCapturePipeline/actions/runs/31840559625),
completed end-to-end in 144 seconds; its pytest step took 84 seconds. Also, the
full lane is not limited to nightly and promotion dispatches: `ci.yml` invokes it
for changes the selector classifies as cross-cutting.

## Implemented optimization

The full execution now uses four pytest-xdist workers with `--dist loadfile`.
Keeping every file on one worker avoids concurrent execution within a file and
preserves the repository's file-scoped assumptions. Exact planned/executed
collection and JUnit identity are canonicalized by node ID, so worker completion
order cannot change or weaken release evidence. Only one designated xdist worker
writes the collection manifest.

Using the observed file timings, a four-way longest-processing-time estimate is
450.553 seconds of testcase work per worker, compared with 1,802.205 seconds
serially. This is a projection, not a passing claim: dependency setup, collection,
worker startup, resource contention, and the 376.787-second largest file bound the
actual result. The next promotion/nightly run must validate wall time and test
isolation.

PR run [31844499960](https://github.com/ognjhunt/BlueprintCapturePipeline/actions/runs/31844499960)
provided that hosted validation. The pytest step completed in 593 seconds and the
entire job in 791 seconds, down from 1,828 and 2,059 seconds respectively. Exact
collection, CPU evidence, telemetry generation, and artifact upload all passed.
The 67.6% pytest-step reduction came with an 18.9% increase in summed per-case
durations, so future regression comparisons must compare equal worker counts and
must not interpret concurrent case timings as serial cost.

Every full run now uploads `test-suite-telemetry.json` with:

- total/file/parametrized case counts;
- summed case duration, reported suite wall time, and the slowest-file bound;
- the 100 slowest cases and files;
- the 100 largest parameter families with duration;
- a deterministic four-way file assignment estimate; and
- an explicit statement that line coverage was not collected.

## Deliberately deferred

No tests or parameter rows are removed in this change. Timing data says that
count-based trimming would target the wrong work, and the repository has no
line/branch coverage artifact from the exhaustive lane with which to prove
redundancy.

The cross-cutting PR fallback also remains in place for now. Removing it before
the impact selector has transitive dependency coverage would trade latency for
false-green risk. The next safe sequence is:

1. retain timing telemetry across nightly runs and flag unstable/duplicated slow
   cases;
2. add a commit-bound test-to-source coverage manifest for the deterministic CPU
   lane;
3. validate affected-test recall against several weeks of full-lane failures;
4. replace the cross-cutting PR full run with a fail-closed manifest decision;
5. allow descendant-commit artifact reuse only when every changed executable
   path is covered by that manifest and all required invariant sentinels are green;
6. use boundary or pairwise reduction only for a parameter family whose retained
   cases preserve the relevant branch and invariant coverage.

Timing history now uses the first serial promotion and first green four-worker run
as commit-bound seeds. Successful full runs append a bounded 30-observation
history artifact. Equal-worker wall-time regressions and repeatedly unstable slow
tests/files are warnings; malformed history or a greater-than-two-percent test or
test-file count contraction blocks. A transient history-download failure falls
back to the checked-in seed rather than blocking a production proof for GitHub API
availability.

The existing exact-SHA production artifact reuse remains valid: deployment
provenance already downloads and revalidates the green full-lane artifact for the
same commit. Reuse across a descendant deployment commit is not yet proven and
remains disallowed.
