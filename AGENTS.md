# BlueprintCapturePipeline Agent Guide

This guide is canonical for everyone working in this repo: human engineers and
any coding agent (Claude, Codex, or other). Harness-specific entry files (for
example `CLAUDE.md`) are thin summaries that defer to this file. If a summary
drifts from this guide, this guide wins; if docs disagree with each other, use
[`docs/DOCTRINE_PRECEDENCE.md`](docs/DOCTRINE_PRECEDENCE.md).

## Mission

`BlueprintCapturePipeline` has one active mission: deliver **Arm Decision Proof
v1** (`arm-decision-proof-v1`). Qualify the reusable service first on exact,
rights-cleared public datasets—including one metric 3DGS/collision object
removal, released-code inpainting, and exact SimReady USD replacement—then take
one fresh capture of one previously unseen fixed-arm workcell. Blueprint must
prospectively choose or eliminate one of exactly two frozen policy/configuration
candidates for the next scarce physical-test budget, or explicitly abstain, then
adjudicate that decision and one predicted failure boundary with held-out
physical trials.

The one customer-facing product remains a **Task Evaluation Run**. The maintained
Site-Task Testbed is its reusable substrate; the candidate Minimum Sufficient
Evaluation Replica is a construction method; the Physical Outcome Join is the
proof. SiteBench may name the bounded case study, not a second product.

## Read First

All paths are repo-root-relative:

1. [`docs/arm_decision_proof_v1/north_star_contract.json`](docs/arm_decision_proof_v1/north_star_contract.json)
2. [`docs/arm_decision_proof_v1/README.md`](docs/arm_decision_proof_v1/README.md)
3. [`docs/arm_decision_proof_v1/IMPLEMENTATION_BACKLOG.md`](docs/arm_decision_proof_v1/IMPLEMENTATION_BACKLOG.md)
4. [`docs/arm_decision_proof_v1/PUBLIC_EVIDENCE_LADDER.md`](docs/arm_decision_proof_v1/PUBLIC_EVIDENCE_LADDER.md)
5. [`docs/arm_decision_proof_v1/PUBLIC_REFERENCE_SUBSTRATE.md`](docs/arm_decision_proof_v1/PUBLIC_REFERENCE_SUBSTRATE.md) — historical ADP-008 decision
6. [`PLATFORM_CONTEXT.md`](PLATFORM_CONTEXT.md) — current product and proof doctrine
7. [`WORLD_MODEL_STRATEGY_CONTEXT.md`](WORLD_MODEL_STRATEGY_CONTEXT.md) — backend admission and build priorities
8. [`docs/DOCTRINE_PRECEDENCE.md`](docs/DOCTRINE_PRECEDENCE.md) — how to resolve documentation conflicts
9. [`README.md`](README.md), [`pyproject.toml`](pyproject.toml), and [`docs/architecture/ai-onboarding-map.md`](docs/architecture/ai-onboarding-map.md)

Org context (roles, agent lanes, who owns what): [`AUTONOMOUS_ORG.md`](AUTONOMOUS_ORG.md).

## Sibling-Checkout Convention

`BlueprintCapture`, `Blueprint-WebApp`, and this repo are sibling checkouts whose
location is environment-dependent. Some docs write sibling paths using the
conventional layout `$HOME/workspace/<repo>`; read that as "the local checkout of
`<repo>`, wherever it lives in your environment", not as a literal path. If a
referenced sibling repo is not present in your environment, say so and treat the
dependent step as blocked instead of guessing.

## Product Rules

- Treat `arm-decision-proof-v1` as the sole active program. Historical docs,
  schemas, providers, evaluators, and runtime lanes are compatibility or paused
  material unless a recorded Arm Decision Proof blocker requires the smallest
  possible dependency.
- Before accepting work, name the ADP backlog item and day-7/day-14/day-21/day-28,
  day-35, or day-42 gate it unblocks, the observed completion artifact, why existing
  infrastructure is insufficient, and the smallest reversible change. Missing
  any answer means the work is out of focus.
- Keep geometry, capture observations, simulators, learned evaluators, providers,
  and physical evidence replaceable behind stable capture and evaluation contracts.
- Optimize for the one prospective two-candidate fixed-arm Task Evaluation Run.
  Legacy five-policy, Policy Improvement Run, Post-Training Data Package,
  humanoid, world-model, and provider contracts remain compatibility/internal
  machinery, not active products or default outputs.
- Preserve rights, privacy, provenance, and capture truth through the pipeline.
- Treat readiness and review *outputs* (qualification summaries, trust scores, readiness matrices) as optional support layers. Do not confuse that posture with the module historically named for it: `src/blueprint_pipeline/site_package_orchestrator.py` (formerly `qualification.py`) is the core capture→package orchestration spine, not a secondary readiness module.
- Do not make downstream generated artifacts appear more authoritative than raw capture evidence.
- Use existing captures, fixtures, OpenUSD scenes, and SimReady candidates to
  exercise downstream compiler/runtime/receipt/replay/sealing/outcome-join paths
  now. Keep them `development_only`; they cannot qualify partner capture, owner
  task truth, registration, task physics, observation-domain match, sim-to-real
  fidelity, or partner value. Never copy fixture data into qualified evidence.
- ADP-008 is observed complete. Until ADP-009 passes, direct nearly all
  engineering work to exact public-scene/method admission, metric registration,
  one exact InteriorGS/matching-SAGE-3D object removal through frozen render-
  derived inputs, unchanged Inpaint360GS author reproduction, an InFusion
  primary adapter and AuraFusion360 quality challenger, exact SimReady USD
  replacement, bounded NVIDIA USD Content Agents authoring comparison, targeted ScanNet++
  real measured transfer after access, hybrid Isaac qualification,
  abstention, media, and replay. Fresh capture feature work is forbidden unless
  a measured blocker proves the smallest missing measurement; the next
  construction phase uses the existing Raw V3.2 path for one fresh
  clean-background/object-present workcell capture.
- Do not start or expand humanoid/G1, locomotion, deformables, insertion/force
  tasks, five-policy/general-ranking campaigns, world-model/evaluator research,
  reconstruction/provider bakeoffs, universal runtimes, dynamic-scene research,
  post-training products, multi-site generalization, or unrelated WebApp/growth
  work without an observed ADP blocker and explicit scope change.

## Repo Map

- `src/blueprint_pipeline/`: core orchestration, runtime services, stages, and adapters
  - `site_package_orchestrator.py` (formerly `qualification.py`): the capture→site-package orchestration spine
  - Decision/Evidence Router, EvaluationRunSpec, scenario matrix, runtime
    adapters, episode receipts, rank-fidelity statistics, and Physical Outcome
    Join: the reusable Arm Decision Proof spine
  - legacy robot-evaluation, WAM, humanoid, and provider modules: compatibility
    unless an active-program blocker explicitly requires them
- `tests/`: pipeline, synthesis, runtime, and contract coverage
- `docs/arm_decision_proof_v1/`: sole active program, partner packet, backlog, and master goal
- `docs/`: stable dependency contracts, compatibility docs, and historical evidence
- `scripts/`: environment setup and runtime launch helpers
- `skillpacks/`: reusable operational skill content
- `autoresearch/`: eval targets and scoring harness

## Working Rules

- Work the Arm Decision Proof critical path in order. Prefer changes that turn an
  existing development-only seam into a replayable, fail-closed precursor of the
  partner proof; do not optimize unrelated platform breadth.
- For splat survey, inspection, or scene-understanding work, first use and extend
  the existing `scene_placement/interiorgs_index.py`, `splat_scene_analysis.py`,
  `scene_placement/perception_views.py`, and object-index Splat Analyzer seams.
  Survey the full known room topology before target close-ups, keep model-derived
  Splat Analyzer boxes candidate-only, and report unseen or uncaptured regions
  explicitly; moving a virtual camera cannot recover missing source observations.
- Keep splat survey previews separate from method inputs. A preview may use the
  complete splat for reconnaissance, but an ADP method input must additionally
  bind the exact renderer and version, calibrated camera pose and intrinsics,
  source-splat digest and retained count, output dimensions and supersampling,
  color/alpha settings, image digests, and a renderer-fidelity qualification.
  Do not call a browser preview, an unrecorded screenshot, or a camera plan a
  maximum-quality or evaluation-authorized render. Render SAGE collision geometry
  only for alignment/debug evidence; InteriorGS remains the appearance source.
- Do not impose a blanket local-only or no-external-provider rule. External CAD,
  model, reconstruction, and agent services may be proposed or used when the
  exact input bytes are rights-admitted for that disclosure, provider retention
  and training terms are accepted by an authorized human, secrets stay in the
  canonical secret integration, and any spend or upload has the required
  authority. Third-party dataset nonredistribution terms still fail closed;
  user preference cannot waive a publisher's license or another owner's rights.
- Choose the partner and its actual stack before choosing a permanent robot,
  simulator, provider, or reconstruction backend. Build from scratch as little
  as possible and use thin adapters.
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
- Compatibility work must preserve prior proof boundaries. In particular,
  generated-video or simulator execution never becomes physical truth, and a
  candidate policy or provider never grades itself.
- Every newly executed policy-evaluation episode must retain the exact lossless
  observation frames shown to the policy, a digest-bound frame manifest, and a
  derived human-review video. Completed episodes without all three are invalid;
  failures before the first observation must retain an explicit typed media gap.
  The receipt must identify whether success came from deterministic simulator
  state, a human, or a learned evaluator. A policy may never grade itself, and a
  review video remains derived visual evidence rather than physical truth.
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
python -m blueprint_pipeline.paid_resource_allocator provider-reconstruction <arguments>
```

These are the only supported CPU-build, model-volume, GPU-canary, and paid
provider-reconstruction allocation commands.
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
