# New-site Task Evaluation Run

The capture-admission development command that used to open this runbook,
`blueprint_pipeline.new_site_task_evaluation`, was removed on 2026-11-05
because no production path reached it (see git history). The committed
`tests/fixtures/new_site_loading_bay_v1` fixture stays: Arm Decision Proof docs
cite it as the capture-to-testbed compiler shape. It is fixture-only data, not
a customer capture, independent metrology or method qualification.

This path does not allocate GPU capacity. If a task genuinely requires a paid
evaluator, use only the canonical
`python -m blueprint_pipeline.paid_resource_allocator gpu-canary ...` lifecycle
with an explicit cap, TTL, independent watchdog, immutable inputs/results, exact
teardown, object cleanup, and provider-zero verification.

## Five-policy run

After independently qualified reconstruction, target binding, placement,
engine routing, and five matched-reset learned-policy executions exist, pass
those digest-bound artifacts to
`python -m blueprint_pipeline.new_site_task_evaluation_run`. The compiler ranks
only supported receipts and remains fail-closed; fixture or development results
cannot be used as a policy attempt or comparative-ranking receipt.

For new work, use request v2 and freeze a three-to-five-scenario inspection
pack before execution. Validate the pack and authorization, then call
`build_matrix_execution_plan` or `execute_policy_scenario_matrix` from
`blueprint_pipeline.new_site_task_evaluation_matrix`. Persist the returned
packet before compiling it with the existing module entrypoint:

```bash
python -m blueprint_pipeline.new_site_task_evaluation_run \
  --request <request-v2-with-matrix-execution-packet.json> \
  --output <run-v2.json>
```

Operational sequence:

1. Freeze the metric and aggregation rule before every attempt.
2. Admit only three to five task-valid scenarios. Include nominal and a bounded
   placement/observation perturbation; admit occlusion only with exact support.
3. Bind the same five learned-policy identities to the scenario-pack digest.
4. Execute every policy in every scenario from that scenario's reset and seed.
5. Persist failed or missing cells; never trim the matrix before compilation.
6. Compile and inspect cell abstentions, paired scenario coverage, uncertainty,
   ties, excluded scenarios, unsupported metrics, and catastrophic failures.
7. Replay the same request and require the same execution-plan and run digests.

Do not pass evaluator-only target/distractor state to a policy endpoint. Do not
move a scripted controller into the learned-policy candidate set. Do not add a
geometry, material, lighting, sensor, or noise variant until its qualification
and evidence ceiling are bound inside the frozen scenario.

The committed hermetic example is replayable from
`docs/examples/new_site_task_evaluation_request.v2.example.json`. The retained
ARKitScenes packet is intentionally pre-execution and abstained; resolve its
ordered blockers rather than copying fixture qualifications into it.
