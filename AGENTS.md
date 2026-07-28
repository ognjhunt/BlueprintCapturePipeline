# BlueprintCapturePipeline Agent Guide

This guide is canonical for everyone working in this repo: human engineers and
any coding agent (Claude, Codex, or other). Harness-specific entry files (for
example `CLAUDE.md`) are thin summaries that defer to this file. If a summary
drifts from this guide, this guide wins; if docs disagree with each other, use
[`docs/DOCTRINE_PRECEDENCE.md`](docs/DOCTRINE_PRECEDENCE.md).

## Mission

`BlueprintCapturePipeline` turns raw capture bundles into site/task/scenario/eval artifacts, Task Evaluation Run artifacts, Policy Improvement Run artifacts, Post-Training Data Package artifacts, hosted-session artifacts, generated/model-derived support assets, and optional trust or review outputs.

## Read First

All paths are repo-root-relative:

1. [`PLATFORM_CONTEXT.md`](PLATFORM_CONTEXT.md) — what is true and sellable today
2. [`WORLD_MODEL_STRATEGY_CONTEXT.md`](WORLD_MODEL_STRATEGY_CONTEXT.md) — model-backend posture and build priorities
3. [`VISION.md`](VISION.md) — the long-horizon ladder; direction and bets, never overrides the two above
4. [`docs/DOCTRINE_PRECEDENCE.md`](docs/DOCTRINE_PRECEDENCE.md) — how to resolve documentation conflicts
5. [`README.md`](README.md) and [`pyproject.toml`](pyproject.toml)
6. [`docs/architecture/ai-onboarding-map.md`](docs/architecture/ai-onboarding-map.md)

Org context (roles, agent lanes, who owns what): [`AUTONOMOUS_ORG.md`](AUTONOMOUS_ORG.md).

## Sibling-Checkout Convention

`BlueprintCapture`, `Blueprint-WebApp`, and this repo are sibling checkouts whose
location is environment-dependent. Some docs write sibling paths using the
conventional layout `$HOME/workspace/<repo>`; read that as "the local checkout of
`<repo>`, wherever it lives in your environment", not as a literal path. If a
referenced sibling repo is not present in your environment, say so and treat the
dependent step as blocked instead of guessing.

## Product Rules

- Keep model backends replaceable behind stable capture, evaluation, and data-package contracts.
- Optimize for Task Evaluation Runs, Policy Improvement Runs, Post-Training Data Packages, hosted outputs, and support artifacts, not one permanent provider or world-model product.
- Preserve rights, privacy, provenance, and capture truth through the pipeline.
- Treat readiness and review *outputs* (qualification summaries, trust scores, readiness matrices) as optional support layers. Do not confuse that posture with the module historically named for it: `src/blueprint_pipeline/site_package_orchestrator.py` (formerly `qualification.py`) is the core capture→package orchestration spine, not a secondary readiness module.
- Do not make downstream generated artifacts appear more authoritative than raw capture evidence.

## Repo Map

- `src/blueprint_pipeline/`: core orchestration, runtime services, stages, and adapters
  - `site_package_orchestrator.py` (formerly `qualification.py`): the capture→site-package orchestration spine
  - `robot_eval_job_orchestrator.py` and the WAM/simulator evaluator modules: the robot-evaluation engine lane
- `tests/`: pipeline, synthesis, runtime, and contract coverage
- `docs/`: contracts, runbooks, and launch gates
- `scripts/`: environment setup and runtime launch helpers
- `skillpacks/`: reusable operational skill content
- `autoresearch/`: eval targets and scoring harness

## Working Rules

- Prefer changes that strengthen package quality, hosted runtime reliability, and contract stability.
- Preserve raw bundle truth and downstream compatibility with other Blueprint repos.
- Do not hardwire the company to one model family, checkpoint, or provider.
- Keep cross-repo contracts explicit when changing bundle, runtime, or sync behavior.
- Never resolve a failure by hand or by one-off workaround. Every fix must land
  as code on main with a hermetic fast-lane test pinning the contract and,
  where a paid path exists, a fail-closed gate in front of it. A manual action
  taken to save a live run is a stopgap; the same session must land the
  encoded equivalent (precedents: PR #180 builder swap, PR #181 compute-cap
  ceiling — each replaced a repeatedly hand-applied workaround).
- Keep WAM rollout execution, generated-video success labels, and forward/inverse
  episode-consistency scoring separate. The WAM/evaluator may prepare
  `wam_episode_consistency_request.json` and normalize an external scorer result,
  but it must not claim forward/inverse consistency from WAM execution alone.
- For Paperclip/autonomous-loop closeouts, apply the Blueprint-WebApp
  `docs/autonomous-loop-evidence-checklist-2026-05-03.md` (sibling checkout; see
  the sibling-checkout convention above) before claiming `done`, `blocked`, or
  `awaiting_human_decision`.

## Commands

Paid resource allocation:

```bash
python -m blueprint_pipeline.paid_resource_allocator cpu-build <arguments>
python -m blueprint_pipeline.paid_resource_allocator model-volume <arguments>
python -m blueprint_pipeline.paid_resource_allocator gpu-canary <arguments>
```

These are the only supported CPU-build, model-volume, and GPU-canary allocation commands.
Provider-specific builder/canary modules are adapters and must not be invoked
as launchers. Every new paid-resource path must pass the shared fail-closed
admission seam and the CI bypass verifier.

Install:

```bash
python -m pip install -e .[dev]
```

Run tests:

```bash
pytest                    # fast lane (<90s): slow/gpu tests deselected via addopts
scripts/pytest_full.sh    # full suite including slow/gpu tests (equivalent: pytest -m '')
```

Test lanes (PIPE-05): heavy subprocess/Isaac/render/module-entrypoint tests are tagged
`@pytest.mark.slow` (and `gpu`); bare `pytest` deselects them, so it is the hermetic
pre-push gate. The success-claim contract truth tests always run against the committed
fixture in `tests/fixtures/kitchen_task_min/`; set `BLUEPRINT_TEST_LOCAL_ARTIFACTS=1`
to additionally sweep real `output/kitchen_task_scaling_preflight_*` artifacts.

Targeted launch checks:

```bash
python scripts/run_external_alpha_launch_gate.py
python -m blueprint_pipeline.run_e2e --capture-root <path-to-staged-capture> --provider openai
```

Common entrypoints:

```bash
python main.py
python -m blueprint_pipeline.capture_orchestrator
python -m blueprint_pipeline.runtime_service_app
```

## Slash-Skill Workflows

A repo-local gstack install lives at `.agents/skills/gstack` for agents whose
harness supports slash-skill workflows. Prefer `/investigate`, `/review`,
`/codex`, and `/cso` for cross-repo failures, security-sensitive work, and final
review. Agents without slash-skill support should apply the same discipline
manually: investigate cross-repo failures before patching, and route
security-sensitive work through review.
