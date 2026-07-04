# BlueprintCapturePipeline Claude Guide

Read first:

1. `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/PLATFORM_CONTEXT.md`
2. `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/WORLD_MODEL_STRATEGY_CONTEXT.md`
3. `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/AGENTS.md`

Key rules:

- Keep world-model backends swappable.
- Protect provenance, rights, privacy, and raw capture truth.
- Optimize for strong site-specific packages and hosted runtime outputs.
- Keep readiness and review logic secondary to the product core.
- Before claiming Paperclip/autonomous-loop `done`, `blocked`, or `awaiting_human_decision`, apply `/Users/nijelhunt_1/workspace/Blueprint-WebApp/docs/autonomous-loop-evidence-checklist-2026-05-03.md`.

Key commands:

```bash
pytest                    # fast lane (<90s): slow/gpu tests deselected via addopts
scripts/pytest_full.sh    # full suite including slow/gpu tests (equivalent: pytest -m '')
python -m blueprint_pipeline.run_e2e
python scripts/run_external_alpha_launch_gate.py
```

Test lanes (PIPE-05): heavy subprocess/Isaac/render/module-entrypoint tests are tagged
`@pytest.mark.slow` (and `gpu`); bare `pytest` deselects them, so it is the hermetic
pre-push gate. The success-claim contract truth tests always run against the committed
fixture in `tests/fixtures/kitchen_task_min/`; set `BLUEPRINT_TEST_LOCAL_ARTIFACTS=1`
to additionally sweep real `output/kitchen_task_scaling_preflight_*` artifacts.

## gstack

- Use the repo-local gstack install at `.agents/skills/gstack` when you need slash-skill workflows.
- Prefer `/investigate`, `/review`, `/codex`, and `/cso` for cross-repo failures, security-sensitive work, and final review.
