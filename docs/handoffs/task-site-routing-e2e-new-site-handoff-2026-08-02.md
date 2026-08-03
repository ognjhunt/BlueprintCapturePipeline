# Task/site routing and new-site E2E handoff — 2026-08-02

## Primary objective

Make Blueprint able to accept a newly captured site plus a task, compile the
site/task-specific evidence requirements, select only qualified evaluation
methods, execute the permitted evaluation path, and return a digest-bound Task
Evaluation Run or a precise abstention naming the smallest missing measurement.

The user wants everything that can be implemented now completed. Paid GPU use
needed for this objective is authorized, but every allocation must still use the
canonical allocator, spend/TTL caps, independent watchdog, teardown, and
provider-zero checks.

## Continue from this branch

- Repository: `ognjhunt/BlueprintCapturePipeline`
- Branch: `codex/routing-integration-20260802`
- Starting commit before this handoff document: `3fa2382bcb7c2cd6abc996e5ae73592b7404f531`
- The branch is based on current `origin/main` (`69e673d02`) and was 67 commits
  ahead / 0 behind before this handoff document.
- Supersedes the older `codex/measurement-isaac-vast-20260802` branch.
- No pull request exists yet.

The original implementation report was supplied as a session attachment and is
intentionally not referenced by a workstation-local path.

## What is implemented

- Six task/site/method/qualification/decision/abstention contract families,
  controlled taxonomies, immutable digests, capture evidence auditing,
  fail-closed abstention, no unqualified fallback, and evidence packaging.
- Deterministic hard filters, qualification-scope containment, route
  explanations, rights and local-only gates, composite routes, and an agent
  layer that can propose but cannot authorize.
- R0-R8 governance, requalification triggers, research catalog, primary-source
  snapshot/diff machinery, monthly scheduled monitoring, release alerts,
  benchmark recommendations, and bounded regression automation.
- Executable development adapters and benchmark harnesses for the report's
  geometry/contact, observation, cloth, cable, granular, tactile, and
  world-model lanes. These remain development or candidate evidence until real
  held-out site/task qualification exists.
- Task-scoped raw RGB/depth/LiDAR/optional-event pairing and the full observation
  challenge matrix for transparent, reflective, dark, small, thin, and occluded
  targets across controlled, natural, and adverse lighting.
- Safe, canonical paid GPU paths for Isaac, exact-source DLO-Lab, and
  exact-source Chrono DEM, including typed signed transport, independent
  watchdogs, retry cap zero, spend/TTL bounds, teardown, and provider-zero
  evidence.

## Real runtime evidence completed

- Isaac Sim 6.0.1 / PhysX / RTX executed on Vast. The later run returned bound
  RGB, depth, and semantic outputs. This proves the development runtime path,
  not captured-site physical validity.
- DLO-Lab exact-source CUDA executed two cable cases on Vast, repeated each case
  deterministically, stayed inside the synthetic development envelope, and
  cleaned up completely.
- Chrono 10.0.0 exact-source DEM CUDA built and executed two granular cases on
  an RTX 6000 Ada with observed GPU contacts/forces and deterministic replays.
  Both cases exceeded the preregistered synthetic behavior envelope. The
  runtime is proven; granular accuracy and qualification are not.
- Latest live provider check returned zero Vast instances.

Evidence is under `docs/evidence/`, especially:

- `measurement_isaac_physx_rtx_development_canary_2026-08-02.json`
- `measurement_isaac_rtx_multimodal_development_canary_2026-08-02.json`
- `measurement_dlo_lab_vast_attempt10_2026-08-02.json`
- `measurement_chrono_dem_vast_attempt_006_2026-08-02.json`

## Verification completed on the integration branch

- 137 affected DLO, Chrono, allocator, admission, watchdog, and provider tests
  passed.
- 213 measurement-focused adapter, benchmark, routing, governance, and monitor
  tests passed.
- After integrating sensor pairing and the newer Isaac/RTX lane, 128 focused
  tests passed.
- All 12 quality-gap-ledger tests passed after rebinding exactly two changed
  authoritative digests.
- The older full-suite attempt was intentionally stopped because the shared
  virtual environment was importing the original checkout instead of the
  temporary integration worktree. Do not count that run as final evidence.

When using a virtual environment owned by another checkout, force the current
source explicitly:

```bash
PYTHONPATH="$PWD/src" /path/to/venv/bin/pytest ...
```

Prefer creating/installing a checkout-local environment before the final suite.

## What is not proven and must not be claimed

- No simulator, provider, tactile method, deformable method, or world model has
  a real site/task-scoped R7 production catalog admission.
- No completed physical Capture-to-Geometry-and-Contact,
  Capture-to-Observation, or Capture-to-Deformation qualification campaign
  exists.
- No independent held-out R5 result plus accountable R6 human qualification
  decision exists for these methods.
- No valid policy-ranking result exists; the verdict remains
  `thesis_not_supported`.
- No physical task success, deployment readiness, or safety claim exists.
- The Chrono synthetic envelope misses require scientific analysis and a new
  preregistered development test if another run is justified. Never tune to the
  observed cases and relabel that as qualification.
- EDEM/Rocky execution still requires commercial access/license and real
  characterized granular materials.

## Highest-priority next work: make a new captured site run end to end

Work from first principles. Define the minimum valid product loop before adding
more engine breadth:

1. Select or create one genuinely new capture fixture that was not used to tune
   the routing implementation. Record raw bundle hashes and capture mode.
2. Run capture ingestion/materialization and prove scale, coordinate frames,
   timestamps, rights/privacy, observed volume, and sensor calibration status.
3. Compile one explicit task at that site into task measurement requirements.
4. Build the site evidence profile from observed evidence only. Missing
   colliders, joints, materials, registration, or calibration must remain
   missing.
5. Ask the agentic supervisor for proposals, then pass them through the
   deterministic router. Verify that the agent cannot lower requirements,
   authorize spend, forge qualification, or silently substitute a method.
6. Exercise both outcomes:
   - a permitted development/evaluation route with a complete evidence bundle;
   - a deliberate missing-evidence case that abstains and names the smallest
     next measurement.
7. Feed the permitted result into the Task Evaluation Run path and verify exact
   digest joins through the API/output package. If the WebApp is in scope,
   verify the redacted projection without upgrading claims.
8. Add the new-site E2E fixture/test so future captures follow the same contract
   rather than relying on manual steps.
9. Run the focused tests that cover the changed claims and risks. Run the
   repository fast lane only if dependency-boundary analysis makes that bounded
   integration diagnostic relevant. Reconcile any quality-ledger digest changes
   with `scripts/rebind_quality_gap_ledger_digests.py` only after inspecting the
   exact changed artifacts.
10. Open a protected-main PR and let hosted impacted checks gate it. Run the full
    suite only for an explicit production/deployment promotion, a scheduled
    integration run, or recorded cross-cutting dependency impact, per `AGENTS.md`.

The important acceptance criterion is not “some simulator produced output.” It
is: a new raw capture and task flowed through evidence auditing, deterministic
routing, controlled execution or abstention, and a digest-bound Task Evaluation
Run without any unsupported claim upgrade.

## Read first in the next session

1. `AGENTS.md`
2. `PLATFORM_CONTEXT.md`
3. `WORLD_MODEL_STRATEGY_CONTEXT.md`
4. `docs/DOCTRINE_PRECEDENCE.md`
5. `docs/architecture/task-site-measurement-routing.md`
6. `docs/architecture/decision-evidence-router-implementation-ledger.md`
7. This handoff

Before editing, fetch the remote branch, inspect all worktrees/writers, confirm
the checkout is clean, verify `HEAD`, and check that Vast still has zero active
instances.
