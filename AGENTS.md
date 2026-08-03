# BlueprintCapturePipeline Agent Guide

This guide is canonical for everyone working in this repo: human engineers and
any coding agent (Claude, Codex, or other). Harness-specific entry files (for
example `CLAUDE.md`) are thin summaries that defer to this file. If a summary
drifts from this guide, this guide wins; if docs disagree with each other, use
[`docs/DOCTRINE_PRECEDENCE.md`](docs/DOCTRINE_PRECEDENCE.md).

## Mission

`BlueprintCapturePipeline` turns raw capture bundles into maintained Site-Task
Testbeds and claim-level **Task Evaluation Runs**. A run routes each claim to
qualified evidence, returns a decision or explicit abstention, and may expose
rights-cleared evidence for evaluation or post-training use. Generated,
simulation, legacy export, hosted-session, and trust artifacts are supporting
machinery inside that one product, not separate products.

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

- Keep geometry, capture observations, simulators, learned evaluators, providers,
  and physical evidence replaceable behind stable capture and evaluation contracts.
- Optimize for Task Evaluation Runs over maintained Site-Task Testbeds. Legacy
  Policy Improvement Run and Post-Training Data Package contracts remain
  compatibility/internal evidence machinery, not products or default outputs.
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
- Never resolve a failure only by hand or by a one-off workaround. Use
  risk-based verification:
  - **Experimental/canary lane:** before paid mutation, bind the run to a clean
    immutable commit (a protected experiment branch or `main`) and immutable
    input hashes; run focused hermetic tests for every changed scientific,
    launch, spend, watchdog, teardown, and provider-zero contract; and require
    the canonical fail-closed paid-resource gate. A repository-wide fast lane,
    hosted-check completion, and merge to `main` are not prerequisites for the
    canary. Preserve failures and publish the encoded fix before a production
    release or terminal scientific claim.
  - **Build loop (target: under 2 minutes):** run only the deterministic tests,
    schema checks, replay fixtures, and changed-file lint that cover the edited
    surface. Do not run a repository-wide lane merely because a change is ready
    to commit.
  - **PR gate (target: under 10 minutes):** gate ordinary pull requests with
    impacted tests plus the small always-on contract, security, and paid-resource
    sentinel set. The PR description or check output must record why each command
    covers a changed claim or risk.
  - **Repository fast lane:** this is a bounded integration diagnostic, not the
    default build-loop or ordinary-PR command. Every multi-minute, subprocess,
    simulator, render, module-entrypoint, or external-runtime test belongs in a
    slower lane, and CI must enforce the lane's wall-time budget. A marker
    expression alone is not proof that the lane is fast.
  - **Full suite:** run only for an explicit production/deployment promotion, a
    scheduled integration run, or a recorded dependency-boundary analysis that
    finds the change cross-cutting. Do not require it for an ordinary PR or merely
    because a commit is called a release candidate. Run the smallest deterministic
    set that covers the changed contracts; hosted impacted checks gate PRs.
  - **GPU tests:** run only when the changed path reaches a qualified GPU gate or
    an explicit promotion requires that gate. A `gpu` marker by itself neither
    authorizes paid execution nor makes GPU coverage relevant.
  - **Failure handling:** rerun one isolated, apparently unrelated failure only in
    isolation and diagnose it. Do not automatically restart a broad or full suite.
  Every reported verification command must name the claim or risk it protects;
  "run everything" is not evidence by itself. For non-paid commands expected to
  exceed two minutes, run them in the background or CI and report only start,
  meaningful milestones or failures, and the final result. Paid runs retain their
  stricter monitoring, spend, watchdog, and teardown requirements.
  A manual action taken to save a live run remains a stopgap; encode and focus-
  test the equivalent in the same session (precedents: PR #180 builder swap,
  PR #181 compute-cap ceiling).
- Keep WAM rollout execution, generated-video success labels, and forward/inverse
  episode-consistency scoring separate. The WAM/evaluator may prepare
  `wam_episode_consistency_request.json` and normalize an external scorer result,
  but it must not claim forward/inverse consistency from WAM execution alone.
- For Paperclip/autonomous-loop closeouts, apply the Blueprint-WebApp
  `docs/autonomous-loop-evidence-checklist-2026-05-03.md` (sibling checkout; see
  the sibling-checkout convention above) before claiming `done`, `blocked`, or
  `awaiting_human_decision`.
- Disk hygiene for agent scratch (2026-08-02 audit: ~40 GB of session clones
  accumulated in six days and filled the disk): put throwaway clones under
  `/private/tmp` or a date-stamped `~/workspace/<purpose>-YYYYMMDD` name,
  prefer `git worktree add` against an existing checkout over a fresh clone,
  and delete your scratch dirs at session end. The reaper is
  `python scripts/agent_workspace_gc.py` (dry-run by default; deletion
  requires `--apply --ack reap-agent-scratch`); it only removes clean, pushed
  scratch dirs idle beyond the age window and always keeps primaries,
  evidence/inputs/dataset names, and dirty/unpushed/no-remote clones. Repo
  `output/` and `robot_eval_jobs/` stay governed by
  `scripts/manage_output_artifact_retention.py`; `~/.claude` is bounded by
  Claude Code's built-in `cleanupPeriodDays` cleanup.

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
python -m blueprint_pipeline.impacted_test_selection  # changed tests + sentinels, hard-capped at 120s
ruff check <changed Python files>          # build loop: changed-file lint only
scripts/pytest_fast.sh                     # bounded repository integration diagnostic
scripts/pytest_full.sh                     # explicit promotion/scheduled/cross-cutting only
```

Test lanes (PIPE-05): heavy subprocess/Isaac/render/module-entrypoint tests are tagged
`@pytest.mark.slow` (and `gpu`). Bare `pytest` currently deselects those markers,
but it still selects the repository-wide non-slow collection and has no guaranteed
wall-time; do not use it as the default build-loop or ordinary-PR gate. Experimental
canaries use the focused hermetic tests required by the canary-lane rule above. The
success-claim contract truth tests always run against the committed
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
