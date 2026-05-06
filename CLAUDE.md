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
pytest
python -m blueprint_pipeline.run_e2e
python scripts/run_external_alpha_launch_gate.py
```

## gstack

- Use the repo-local gstack install at `.agents/skills/gstack` when you need slash-skill workflows.
- Prefer `/investigate`, `/review`, `/codex`, and `/cso` for cross-repo failures, security-sensitive work, and final review.
