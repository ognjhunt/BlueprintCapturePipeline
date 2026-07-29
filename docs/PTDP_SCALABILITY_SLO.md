# Legacy evidence-export scalability SLO

`PTDP` in file names, commands, schemas, and release-scope identifiers below is
a compatibility identifier for the historical export implementation. It is not
a customer-facing product. The export is available only as a rights-cleared
evidence use inside a Task Evaluation Run.

The declared package ceiling is 100,000 clips per run. The implementation must
fail before materialization when clip count, estimated output quota, free-space
headroom, or cancellation checks fail. Semantic dedup uses deterministic bounded
angular LSH; JSONL output streams through same-filesystem temporary files;
videos use content-addressed objects and persistent resumable cache entries; and
the final archive is cancellable, temporary, independently verified, and only
then atomically published.

`scripts/benchmark_ptdp_scalability.py` executes the 100,000-item bounded
primitive workload with these fail-closed SLOs:

- ANN wall time: at most 120 seconds;
- streaming JSONL wall time: at most 30 seconds;
- measured process RSS growth: at most 1 GiB;
- JSONL output: at most 128 MiB;
- sparse ANN comparisons: at most 10 million;
- the injected exact duplicate must be recovered; and
- no pairwise similarity matrix may be materialized.

Run it with:

```bash
python scripts/benchmark_ptdp_scalability.py \
  --clip-count 100000 \
  --output output/benchmarks/ptdp-scalability.json
```

This synthetic benchmark validates bounded index and writer behavior. It does
not claim that a maximum-size representative real-media evidence-export run, buyer package
quality, or production hardware SLO has passed. Those require a release-bound
real-media workload and retained evidence at the exact code/image digest.
