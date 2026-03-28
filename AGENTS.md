# BlueprintCapturePipeline Agent Guide

## Mission

`BlueprintCapturePipeline` turns raw capture bundles into site-specific world-model packages, hosted-session artifacts, and optional trust or review outputs.

This repo must stay aligned with:

- `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/PLATFORM_CONTEXT.md`
- `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/WORLD_MODEL_STRATEGY_CONTEXT.md`

## Read First

1. `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/PLATFORM_CONTEXT.md`
2. `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/WORLD_MODEL_STRATEGY_CONTEXT.md`
3. `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/README.md`
4. `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/pyproject.toml`

## Product Rules

- Keep model backends replaceable behind stable capture and packaging contracts.
- Optimize for site-specific world-model products and hosted outputs, not one permanent provider.
- Preserve rights, privacy, provenance, and capture truth through the pipeline.
- Treat readiness and review outputs as optional support layers.
- Do not make downstream generated artifacts appear more authoritative than raw capture evidence.

## Repo Map

- `src/blueprint_pipeline/`: core orchestration, runtime services, stages, and adapters
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

## Commands

Install:

```bash
python -m pip install -e .[dev]
```

Run tests:

```bash
pytest
```

Targeted launch checks:

```bash
python scripts/run_external_alpha_launch_gate.py
python -m blueprint_pipeline.run_e2e
```

Common entrypoints:

```bash
python main.py
python -m blueprint_pipeline.capture_orchestrator
python -m blueprint_pipeline.runtime_service_app
```
