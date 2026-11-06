# Measurement research monitoring runbook

This runbook operates the proposal-only monitoring lanes for the task/site
measurement router. A monitoring report may request research or
requalification work. It cannot approve a candidate, advance R0-R8, suspend a
production route, authorize paid/provider execution, or establish any claim.

## Monthly operating sequence

1. Collect release observations for the lightweight release watcher from
   checked primary repositories or an operator-maintained JSON file.
2. Run the release watcher and archive its digest-bound report.
3. Reduce broader primary sources to
   `measurement_primary_source_observation.v1` through an approved fetch
   adapter. Discard raw page prose, embedded instructions, cookies, tokens,
   and non-allowlisted headers.
4. Build the immutable current snapshot and diff it against the previous
   snapshot.
5. Review version/access/source alerts, benchmark recommendations, and the
   regression plan. Apply any admission or requalification action separately,
   with the human roles required by `measurement_research_admission.v1`.

Lightweight release watcher:

```bash
python scripts/measurement_research_monitor.py \
  --observations /path/to/release_observations.json \
  --admissions /path/to/admission_records.json \
  --observed-on 2026-08-02 \
  --output output/measurement-monitor/release-2026-08-02.json
```

The repeatable `--fetch-github METHOD_ID=OWNER/REPOSITORY` option may replace
or augment the observations file. Network results remain untrusted release
metadata and do not become qualification evidence.

Primary-source snapshot and diff:

```bash
python -m blueprint_pipeline.measurement_research_monitor \
  --observations /path/to/sanitized_source_observations.json \
  --previous-snapshot /path/to/previous_snapshot.json \
  --observed-at 2026-08-02T12:00:00+00:00 \
  --output output/measurement-monitor/snapshot-2026-08-02.json
```

The input object contains an `observations` list of already validated
`measurement_primary_source_observation.v1` rows. The output contains both the
new immutable snapshot and its report. For the first run, omit
`--previous-snapshot`.

## Scheduled repository lane and deployment boundary

`build_monthly_monitor_schedule` and `monitor_is_due` provide deterministic
cadence and due-state contracts. The checked
`.github/workflows/measurement-research-monitor.yml` workflow runs on the first
day of each month (and on manual dispatch), checks public GitHub release
metadata for the five catalog candidates with official GitHub release feeds,
runs the bounded routing/monitoring regression suite, creates exact isolated
Drake 1.55 and PyChrono 10.0 development environments, executes their bounded
development regressions, and retains the release report plus environment
receipts for 90 days. The adapter results remain development-only and cannot
advance admission. The workflow has read-only repository permission and
contains no paid-resource, provider-execution, admission-advance, or
trigger-application step.

The workflow becomes operational only after this branch is merged into a
GitHub default branch with Actions enabled. It does not cover non-GitHub or
credentialed primary sources. Add those only after the observation store,
approved source adapters, retention policy, alert recipient, and operator
ownership are configured. Every scheduler invocation must remain read-only
with respect to the routing catalog and R0-R8 admission records.

## Required review outcomes

- A version, adapter, driver, source, license, access, or regression change is
  a requalification proposal, not an automatically applied trigger.
- A benchmark recommendation is an R4 design input, not permission to execute
  paid compute, provider calls, or a robot.
- A completed qualification-split report is at most an R5 evidence candidate.
  Independent R6 approval and R7 catalog admission remain separate signed
  actions.
- Missing source access, labels, evaluator identity, or physical measurements
  results in abstention and the smallest missing action.
