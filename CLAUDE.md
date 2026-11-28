# BlueprintCapturePipeline Claude Guide

Read [`AGENTS.md`](AGENTS.md); it applies to every agent.

In short: a site films a job once; Blueprint turns the video into a job packet
that approved deployment companies can evaluate without a visit, and the site
learns who will take the job and on what evidence. This repo is the packet
factory: video in, the job's sound-free clip out, and, when a deployment
company asks for it (owner, 2026-11-28), a simulation of the site with the
job's object in it and the success rule. The simulation is a means, not the
product. Blueprint no longer runs robot software in the simulation: the owner
dropped the early trial on 2026-11-14 and the packet has no trial section. A
scripted run only checks the simulation. The only hard stops are privacy,
rights and money; everything else is flagged and keeps going. Don't make
things up.

```bash
python -m blueprint_pipeline.impacted_test_selection
ruff check <changed files>
```
