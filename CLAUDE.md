# BlueprintCapturePipeline Claude Guide

Read [`AGENTS.md`](AGENTS.md); it applies to every agent.

In short: a site films a job once; Blueprint turns the video into a job packet
that approved deployment companies can evaluate without a visit, and the site
learns who will take the job and on what evidence. This repo is the packet
factory: video in, a simulation of the site with the job's object in it, and
the success rule. The simulation is a means, not the product. Our own test of
openly available robot software is an early trial inside the packet, never the
site's answer; a scripted run only checks the simulation. The only hard stops
are privacy, rights and money; everything else is flagged and keeps going.
Don't make things up. Arm Decision Proof v1 is a separate evidence program
with its own protocol in `docs/arm_decision_proof_v1/`.

```bash
python -m blueprint_pipeline.impacted_test_selection
ruff check <changed files>
```
