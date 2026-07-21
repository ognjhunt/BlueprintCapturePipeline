# Blueprint Benchmark Protocol

`blueprint_benchmark_spec.v1` is Blueprint's benchmark-grade policy evaluation
contract. It standardizes evaluation discipline without making any external
benchmark or simulator a required product dependency.

## What the contract guarantees

- A frozen `train` / `dev` / `public_test` / `hidden_test` split.
- Seen and unseen labels for task, scene, object, camera, lighting, and
  embodiment generalization.
- Hidden scenario identifiers and initial conditions stay in private artifacts;
  the public card contains only counts and the committed split digest.
- Every public baseline has an exact checkpoint digest, adapter-code digest,
  public source, license, and reproducible runner pinned by source revision or
  container digest.
- The execution plan expands a fixed rollout count into immutable attempt IDs.
  Every scheduled attempt must appear exactly once in the result set. Failed or
  abstained attempts remain in the ledger and cannot be replaced or cherry-picked.
- Every completed episode carries full success, partial progress, efficiency,
  safety/intervention, and abstention state plus digest-bound video, action trace,
  and evaluator output references.
- Every reported aggregate carries a 95% percentile-bootstrap interval. Reports
  include per-policy, split, and seen/unseen-axis breakouts.
- An optional external-reference input compares exact checkpoint matches using
  Pearson, Spearman, Kendall tau-b, pairwise ordering accuracy, and MMRV, each
  with confidence intervals.

The JSON Schemas are:

- [`schemas/blueprint_benchmark_spec.schema.json`](schemas/blueprint_benchmark_spec.schema.json)
- [`schemas/blueprint_benchmark_report.schema.json`](schemas/blueprint_benchmark_report.schema.json)

## Robot-eval job option

`robot_eval_job_request.v1` can select the protocol with a
`blueprint_benchmark_protocol_request.v1` object:

```json
{
  "schema_version": "blueprint_benchmark_protocol_request.v1",
  "mode": "benchmark_grade",
  "benchmark_spec_uri": "private/benchmarks/drawer/spec.json",
  "benchmark_spec_sha256": "<64 lowercase hex characters>",
  "frozen_hidden_splits_required": true,
  "fixed_rollouts_required": true,
  "confidence_intervals_required": true,
  "exact_checkpoint_digests_required": true,
  "private_split_material_allowed_in_webapp": false,
  "scheduler_owner": "BlueprintCapturePipeline"
}
```

The orchestrator resolves only a staged local artifact beneath the capture root,
checks the exact spec digest, and writes the protocol under the job's
`benchmark_protocol/` directory. HTTP and cloud URIs must first be staged by an
authenticated owner-system adapter; the benchmark compiler does not download
URLs or resolve customer credentials. A request with no result artifact remains
`planned`. A supplied `benchmark_results_uri` is validated against the immutable
execution plan before a report can become `complete`.

The `standard` mode preserves the existing operational evaluation path. It does
not imply benchmark-grade reporting.

## Compile a protocol

```bash
python -m blueprint_pipeline.benchmark_protocol compile \
  --spec benchmark_spec.json \
  --output-dir output/benchmark
```

Compilation writes:

```text
benchmark_card.json
public_baseline_registry.json
benchmark_split_manifest.private.json
benchmark_execution_plan.private.json
evaluation_run_task_scenario_pack.private.json
webapp_benchmark_projection.json
```

The private artifacts are written with owner-only permissions where the host
filesystem supports POSIX modes. The public card never includes hidden scenario
IDs or seeds. `evaluation_run_task_scenario_pack.private.json` binds the plan to
the provider-neutral Evaluation Run interface through the
`benchmark_task_scenario_pack@1` adapter.

## Produce a report

```bash
python -m blueprint_pipeline.benchmark_protocol report \
  --spec benchmark_spec.json \
  --plan output/benchmark/benchmark_execution_plan.private.json \
  --results benchmark_results.json \
  --output-dir output/benchmark \
  --external-reference optional_external_reference.json
```

The command exits `2` when attempt coverage, binding, evidence, or result fields
are incomplete. It still writes a blocked report so the missing evidence stays
inspectable.

Reporting also writes `benchmark_evidence_index.private.json` with each scheduled
attempt's video, action trace, evaluator output, and content digest references.
It retains scenario IDs and seeds, so it is owner-only and never enters the
WebApp projection. The public report and WebApp projection expose only its
content digest and completeness counts.

## External comparison boundary

The external reference format requires a source artifact digest, task-mapping
digest, site-alignment declaration, independent acceptance, and exact checkpoint
digests. Unaccepted/provisional rows produce a blocked comparison. Its
`measurement_scope` is deliberately different for:

- `same_site_real_robot_rank_fidelity`
- `cross_site_real_robot_rank_concordance`
- `cross_evaluator_concordance`

Different-site agreement is not site-specific validation. Simulator or world-
model agreement is not real-world validation. Even a measured external report
does not automatically authorize a public accuracy claim; customer/site-specific
and physical-readiness claims require their separate owner-system evidence.

## WebApp projection

`webapp_benchmark_projection.json` is the only benchmark artifact intended for
direct buyer-surface projection. It contains public card summaries, aggregate
metrics and confidence intervals, safe split and seen/unseen breakdowns, and the
external comparison report. It cannot contain hidden scenario identifiers. The
WebApp validates this contract again before persisting or displaying it.

Private/closed policy executables and credentials remain in the existing sealed
policy-package/runtime path. Benchmark artifacts expose policy IDs and exact
checkpoint hashes for matching, not policy bytes, API tokens, container secrets,
or private source code.
