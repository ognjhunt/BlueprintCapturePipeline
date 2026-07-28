# BlueprintCapturePipeline Claude Guide

[`AGENTS.md`](AGENTS.md) is the canonical working guide for all agents (Claude,
Codex, or other) and human engineers. Everything there is binding; this file is
the Claude-harness entry summary and must not drift ahead of it.

Read first (repo-root-relative):

1. `PLATFORM_CONTEXT.md`
2. `WORLD_MODEL_STRATEGY_CONTEXT.md`
3. `VISION.md` (long-horizon direction and bets; never overrides the two above)
4. `AGENTS.md`
5. `docs/DOCTRINE_PRECEDENCE.md` (when docs disagree)

Key rules (full text and precedents in `AGENTS.md`):

- Keep world-model backends swappable behind stable contracts.
- Protect provenance, rights, privacy, and raw capture truth.
- Optimize for Task Evaluation Runs, Policy Improvement Runs, Post-Training
  Data Packages, and hosted runtime outputs.
- Readiness and review *outputs* are optional support layers. The module
  historically named for them — now
  `src/blueprint_pipeline/site_package_orchestrator.py` (formerly
  `qualification.py`) — is the core capture→package orchestrator, not a
  secondary readiness module.
- Never resolve a failure by hand or by one-off workaround: every fix lands as
  code on main with a hermetic fast-lane test pinning the contract and, where a
  paid path exists, a fail-closed gate (precedents: PR #180, PR #181).
- Before claiming Paperclip/autonomous-loop `done`, `blocked`, or
  `awaiting_human_decision`, apply the Blueprint-WebApp
  `docs/autonomous-loop-evidence-checklist-2026-05-03.md` (sibling checkout;
  path is environment-dependent — see the sibling-checkout convention in
  `AGENTS.md`).

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
