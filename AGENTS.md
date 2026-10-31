# BlueprintCapturePipeline Agent Guide

This guide applies to everyone working here, human or agent. `CLAUDE.md` is a
pointer to it.

## Mission

A site sends a video of one recurring job. This repo turns it into a
simulation of the site, runs the candidate robot policies and embodiments in it,
and returns a bounded estimate of what works, or "held — we can't simulate this
kind of job yet". Blueprint runs the candidates itself, then works alongside a
deployment company on the real site. WebApp owns the customer side.

Shared doctrine is generated from `doctrine/` (see `doctrine/README.md`).

## The Only Hard Stops

Fail closed only for:

- **Privacy:** people in captures are screened, and consent is respected.
- **Rights:** capture permissions, revocation, and dataset or model licenses.
- **Money:** the per-site budget (USD 15 for screening candidates), the Google
  Cloud monthly ceiling (USD 100), and the paid-resource allocator below. Never
  start paid compute or provider calls outside it.

Everything else keeps going. If an input is missing, ambiguous or unverified,
flag it, make the best supported guess, and continue the same job. Fixing a
failure means fixing its cause; do not add a gate, approval step, review stage
or readiness check unless it protects privacy, rights or money. Delete checks
that protect none of the three.

Ask the owner first before contacting outside companies, sending email, or
deleting retained data.

## Two Truths

- Don't make things up: no invented results, captures, approvals or readiness.
  Unknown stays unknown.
- Simulation results are bounded estimates; they never certify physical
  performance or safety.

## Arm Decision Proof v1

`arm-decision-proof-v1` is a separate evidence program with its own protocol in
`docs/arm_decision_proof_v1/`. That protocol applies only to ADP work. Beta
results never enter ADP evidence, and only ADP may claim physical validation.

## Sibling Checkouts

`BlueprintCapture`, `Blueprint-WebApp` and this repo are sibling checkouts;
where they live depends on the environment. If a sibling you need is missing,
say so.

## Cloud Sessions

Cloud sessions reach the control-plane host through the operator door
(`python3 scripts/operator_door.py`, see `docs/OPERATOR_DOOR.md`), never SSH.
Setup lives in the WebApp checkout's `docs/runbooks/cloud-scene-runs.md`.

## Commands

Paid resource allocation (the only supported launchers):

```bash
python -m blueprint_pipeline.paid_resource_allocator cpu-build <arguments>
python -m blueprint_pipeline.paid_resource_allocator model-volume <arguments>
python -m blueprint_pipeline.paid_resource_allocator gpu-canary <arguments>
python -m blueprint_pipeline.paid_resource_allocator provider-reconstruction <arguments>
python -m blueprint_pipeline.paid_resource_allocator remote-cpu-job <arguments>
```

Install and test:

```bash
python -m pip install -e .[dev]
python -m blueprint_pipeline.impacted_test_selection  # changed tests, capped at 120s
ruff check <changed Python files>
scripts/pytest_fast.sh                                # broader, slower
```

Before merging or deploying a fix for a stopped production job, replay the job's
next step on the fix branch (it pays for nothing) and include its first stop in
the PR. Deploy to find out only what a replay cannot show, such as provider
behavior:

```bash
python -m blueprint_pipeline.task_evaluation_stage_replay --child <id> --isolate
python -m blueprint_pipeline.atlas_preview_replay --job <pulled preview_job.json>
```

Entrypoints:

```bash
python main.py
python -m blueprint_pipeline.capture_orchestrator
python -m blueprint_pipeline.runtime_service_app
```
