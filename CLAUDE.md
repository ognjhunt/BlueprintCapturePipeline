# BlueprintCapturePipeline Claude Guide

Read [`AGENTS.md`](AGENTS.md); it applies to every agent.

In short: video in, site simulation, Blueprint runs the candidates, the site
gets a bounded estimate or "held". The only hard stops are privacy, rights
and money; everything else is flagged and keeps going. Don't make things up.
Arm Decision Proof v1 is a separate evidence program with its own protocol in
`docs/arm_decision_proof_v1/`.

```bash
python -m blueprint_pipeline.impacted_test_selection
ruff check <changed files>
```
