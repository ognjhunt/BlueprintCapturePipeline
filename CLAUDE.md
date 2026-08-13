# BlueprintCapturePipeline Claude Guide

[`AGENTS.md`](AGENTS.md) is the canonical working guide for all agents (Claude,
Codex, or other) and human engineers. Everything there is binding; this file is
the Claude-harness entry summary and must not drift ahead of it.

Read first (repo-root-relative):

1. `docs/arm_decision_proof_v1/north_star_contract.json`
2. `docs/arm_decision_proof_v1/README.md`
3. `docs/arm_decision_proof_v1/IMPLEMENTATION_BACKLOG.md`
4. `docs/arm_decision_proof_v1/PUBLIC_EVIDENCE_LADDER.md`
5. `docs/arm_decision_proof_v1/PUBLIC_REFERENCE_SUBSTRATE.md` (historical ADP-008 decision)
6. `PLATFORM_CONTEXT.md`
7. `WORLD_MODEL_STRATEGY_CONTEXT.md`
8. `AGENTS.md`
9. `docs/DOCTRINE_PRECEDENCE.md` (when docs disagree)

Key rules (full text and precedents in `AGENTS.md`):

- Arm Decision Proof v1 is the sole active program: one partner, site, fixed
  arm, rigid-object task, and two real frozen candidates, prospectively sealed
  and physically adjudicated.
- Every task must name the ADP backlog item and day gate it unblocks. Existing
  captures/scenes may exercise downstream seams only as `development_only`.
- ADP-008 is observed complete. Complete ADP-009 with **artifixer3D+ with
  `gpt-image-2` as the appearance path**; targeted ScanNet++ real measured
  transfer after access; one exact SimReady USD; and a bounded NVIDIA USD
  Content Agents comparison before the fresh-site phase. Paper-only methods and
  unrecorded rights fail closed.
- **Inpaint360GS and AuraFusion360 are retired as appearance methods**
  (2026-08-13): a real artifixer3D + `gpt-image-2` run produced materially
  better results, so neither is needed as a primary adapter or a quality
  challenger. Their lanes, bundles, and allocator branches stay in the tree; no
  launch profile will be built for them and no further rights work is required.
  See `docs/arm_decision_proof_v1/LIVE_LANE_REACHABILITY.md`.
- Humanoid, deformable, five-policy/general-ranking, world-model, provider
  bakeoff, post-training, multi-site, and unrelated product work is frozen.
- Keep world-model backends swappable behind stable contracts.
- Protect provenance, rights, privacy, and raw capture truth.
- Optimize for the single customer-facing Task Evaluation Run. Treat the
  maintained Site-Task Testbed as its reusable substrate and any rights-cleared
  evaluation or post-training export as an evidence use inside the run.
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
python -m blueprint_pipeline.impacted_test_selection  # changed tests + sentinels, hard-capped at 120s
ruff check <changed files>            # default build loop: changed-file lint only
scripts/pytest_fast.sh                # bounded repository integration diagnostic
scripts/pytest_full.sh                # explicit promotion/scheduled/cross-cutting only
python -m blueprint_pipeline.run_e2e
python scripts/run_external_alpha_launch_gate.py
python scripts/agent_workspace_gc.py   # reap stale agent scratch clones (dry-run; delete needs --apply --ack reap-agent-scratch)
```

Test lanes (PIPE-05): heavy subprocess/Isaac/render/module-entrypoint tests are tagged
`@pytest.mark.slow` (and `gpu`). Bare `pytest` deselects those markers but still has
no guaranteed wall-time, so it is not the default build-loop or ordinary-PR gate.
Use the risk-based verification contract in `AGENTS.md`. The success-claim contract
truth tests always run against the committed fixture in
`tests/fixtures/kitchen_task_min/`; set `BLUEPRINT_TEST_LOCAL_ARTIFACTS=1` to
additionally sweep real `output/kitchen_task_scaling_preflight_*` artifacts.

## gstack

- Use the repo-local gstack install at `.agents/skills/gstack` when you need slash-skill workflows.
- Prefer `/investigate`, `/review`, `/codex`, and `/cso` for cross-repo failures, security-sensitive work, and final review.
