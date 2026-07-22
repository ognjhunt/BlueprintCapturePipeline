# Fable execution instruction

> Archived cloud-agent handoff snapshot. Not a current operator instruction.

Start from `origin/main` at `d1220f788acb3b5d263af36297ed58443772469a`. Do not
reimplement the merged startup modules. First wire the supervisor, review canary, asset
gate, phase trace, and teardown transaction into the real default launcher; complete
P0-3 build/live validation; reconcile the duplicate stance configurators; then close
kitchen gaps A-F. The arm visual-center, RGB integrity, and strict scorer repairs
described as dirty-local are absent from main and must be carefully ported rather than
overwriting newer main changes. Finish with green CI and live provider proof.

For live validation, Fable will additionally need the selected scenario/task artifacts,
attempt 023/025 logs and media, the content-addressed kitchen archive, image registry
access, and file-mounted provider/object-store secrets. Code-only work does not require
your local machine.

# Handoff 1: Isaac startup reliability

Date: 2026-07-10

Repository: `BlueprintCapturePipeline`

Baseline: `origin/main` at `d1220f788acb3b5d263af36297ed58443772469a`

## Scope

Make Isaac GPU worker startup, image validation, asset transfer, renderer canaries,
monitoring, spending, and teardown reliable by default.

This is simulator-support evidence only. It does not prove kitchen task completion,
learned-policy quality, physical Unitree G1 readiness, or real-world safety.

## Important baseline warning

Start from the exact `origin/main` SHA above.

Do not overwrite newer `main` files with older dirty-worktree copies. Some later
kitchen fixes exist only in a local dirty checkout based on an older `main`; port
their behavior and tests carefully onto the current baseline.

## What merged to main

PR #67 added implementations for:

- Atomic canary-to-worker startup supervisor.
- Persistent cross-run provider-machine quarantine.
- Separate fast-startup and review-renderer canaries.
- Honest RunPod capacity confidence fields.
- Provider phase and heartbeat trace.
- Startup spending reconciliation.
- Content-addressed kitchen asset gate.
- Bounded adaptive stance configurator.
- Image build/publish script improvements.

Relevant modules include:

- `isaac_startup_supervisor.py`
- `machine_quarantine_registry.py`
- `isaac_review_renderer_canary.py`
- `provider_phase_trace.py`
- `startup_spend_reconciliation.py`
- `kitchen_asset_startup_gate.py`
- `adaptive_task_stance_configurator.py`
- `stance_configuration_agent.py`

Clean-main focused verification passed locally:

- 399 startup/reliability tests.
- 467 kitchen/OSCAR/stance tests.

## Remaining startup gaps

### P0: Production wiring is incomplete

Several new components exist but are not automatically invoked by the normal parity
launch path.

Specifically:

- `run_startup_supervisor()` has no production caller.
- The review-renderer canary is a standalone CLI.
- The kitchen asset gate is a standalone CLI.
- Provider phase trace and cumulative spending are primarily used by the standalone
  supervisor.
- Therefore future parity runs do not automatically receive the complete atomic
  startup transaction.

Required fix:

1. Make the real provider launcher call the supervisor before the full job.
2. Supply real DO and RunPod provider adapters, inventory callback, marker callback and
   canary callback.
3. On a passing canary, promote the same warm allocation to the full job.
4. Do not launch a second cold worker after a passing canary.
5. On every non-promoted terminal path, delete the allocation and verify provider
   `not_found`.
6. Run kitchen asset readiness before Isaac scene or policy startup.
7. Run the review-renderer canary after the fast canary.
8. Persist automatic phase/heartbeat and spending artifacts.

Acceptance:

- At most one billable GPU resource.
- No manual handoff between canary and full job.
- No stopped pod or attached volume remains.
- Every retry contributes to the goal-level spending ledger.
- Every artifact carries the run ID, attempt ID and nonce.

### P0: Worker overlay was prepared but not published or live-proven

The image build script was improved, but no new immutable worker digest was built,
pushed and proven.

Required fix:

1. Build `deploy/docker/robot_eval_worker/isaac/Dockerfile`.
2. Use the pinned Isaac Sim 6 base digest.
3. Publish linux/amd64 under a versioned tag.
4. Resolve the pushed image to an immutable `@sha256:` reference.
5. Generate registry manifest diagnostic v2.
6. Run the fast-startup canary against that digest.
7. Run the 480x640 review-renderer canary against that digest.
8. Only then update the configured production image reference.

Do not bake kitchen assets, provider secrets or policy credentials into the image.

### P0: Duplicate stance systems need reconciliation

There are now two overlapping implementations:

- `adaptive_task_stance_configurator.py`
- The runner-integrated `stance_configuration_agent.py` and
  `_adaptive_task_stance_search`

Required fix:

- Select one production abstraction.
- Keep one deterministic gate authority.
- Preserve measured rejection feedback and bounded retries.
- Remove or clearly separate unused duplicate paths.
- An LLM/agent may propose a candidate but may never modify thresholds, waive gates or
  fabricate measurements.

The primary kitchen placement defect is incorrect geometry measurement, not lack of an
agent.

### P1: CI on latest main is not green

Latest `main` GitHub results:

- Python Compatibility: passed.
- CodeQL: passed.
- Full Test Lane: failed before test collection because isolated `flash-attn` build
  could not import `torch`.
- Main CI test lane: 1 failure, 3,119 passed. The failure is a stale quality-ledger
  digest for `tests/test_spend_admission_lock.py`.
- Sim-Only Local Gate: blocked at
  `forwarding_preflight_before_route_proof_command_failed`.
- Source governance also reports module/literal/script budget failures.
- Supply-chain gate reports missing license review for `defusedxml==0.7.1`.
- Container-production evidence is blocked.

Required fix:

- Correct the `flash-attn`/torch installation order or UV build-dependency
  configuration.
- Rebind the exact quality-ledger digest after the final source/test modification.
- Diagnose and repair the forwarding preflight command.
- Reconcile governance budgets without blindly increasing limits.
- Add the missing dependency license review.
- Repair container-production evidence generation as applicable.
- Finish with required GitHub checks green.

## Existing live startup evidence

Reference image:

`docker.io/nijelhunt/blueprint-isaac-eval-worker@sha256:435f6ffa1ddb6cfbf72681e30f212d92ab7826420ea026f613e4a4f4c4679acd`

Observed outcomes:

- RunPod L40S at $0.99/hour: Isaac started, but RTX failed on host driver
  `570.124.06`, inside Isaac Sim 6's rejected R570 interval.
- RunPod A40 at $0.44/hour: catalog indicated stock, but create failed before
  allocation.
- RunPod RTX A6000 at $0.49/hour: startup and RTX pixel canary passed on driver
  `570.211.01`; allocation was subsequently deleted.
- H100/H200 are excluded from the Isaac RTX review-rendering lane. They may be valid
  compute workers, but do not satisfy the review-renderer capability contract.

Do not generalize one machine's driver result to every machine of the same GPU model.

## Startup completion criteria

Startup reliability is complete only when:

- The new immutable image digest is built and published.
- Fast and review canaries pass live on that digest.
- The supervisor is the default provider launch path.
- The full job reuses the passing warm allocation.
- The kitchen archive is checksum-verified provider-side before Isaac startup.
- Automatic phase/heartbeat evidence exists.
- Spending is reconciled across all attempts.
- Every terminal attempt ends in API-confirmed teardown.
- Focused and broad CI lanes pass.
