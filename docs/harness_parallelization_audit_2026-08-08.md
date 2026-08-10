# Harness Parallelization Audit and Decision — 2026-08-08

Scope: audit of the 2026-08-08 Codex response claiming the customer pipeline
should be "a dependency graph, not one long serial script", followed by the
general (non-run-specific) implementation decision. ADP anchor: the reusable
evaluation harness required by ADP-009D ("implement this as the reusable ADP
simulator evaluation harness used by future runs, not as a scene-specific
script") and the one-command day-28 rehearsal gate; the same seams carry
ADP-060 matrix execution and the customer Task Evaluation Run.

## Claim-by-claim audit

### 1. "The customer pipeline is one long serial script" — CONFIRMED

Every orchestration layer is strictly sequential, verified in code:

- `run_e2e.py` — 8 ledgered stages in a fixed tuple (`_RUN_E2E_STAGE_ORDER`,
  run_e2e.py:58), executed one `_run_stage` closure call at a time.
- `capture_orchestrator.py` — serial lane loop over `_LANE_ORDER`
  (capture_orchestrator.py:59, loop at :1459).
- `site_package_orchestrator.py` (`run_qualification_pipeline`, :4383) — a
  12-phase spine tracked only by a mutable `stage = "..."` string; no stage
  registry, no checkpointing inside the spine, all-or-nothing on failure.
- `decision_evidence_execution.execute_evidence_plan` — before this change, a
  flat serial loop over `plan["execution_order"]`; the router even models
  wall-clock as `sum` of step latencies (`projected_latency_seconds`,
  decision_evidence_router.py:802-805).
- `canonical_3dgs_pipeline.execute_canonical_3dgs_plan` — one arm at a time
  despite per-arm isolated `run_root`s.
- Remote provider runners are literal straight-line scripts
  (`scripts/adp_aura_interiorgs_provider_runner.py:343`,
  `scripts/adp_content_agents_provider_runner.py:274-454`).

Repo-wide, only 8 of ~840 modules use any concurrency primitive, all in
paused `policy_ranking_*` / `groot_oscar_*` research lanes. There was no DAG,
no topological sort, and no scheduler abstraction anywhere.

### 2. The four-branch decomposition (appearance ∥ physics ∥ robot ∥ evidence) — DIRECTIONALLY RIGHT, edges refined

The true dependency structure found in code is finer-grained than the four
branches, and some of it is hard-serial:

- Independent today, no code between them: the two enrichment LLM calls
  (site_package_orchestrator.py, weakness summary + recapture instructions)
  and the four post-readiness builders (world-model fit, payout, provenance,
  opportunity handoff).
- Independent by construction: canonical 3DGS arms (per-arm `run_root`),
  cross-claim evidence steps in an evidence plan, and per-candidate/per-cell
  scenario work once cells are admitted.
- Hard-serial chain that must never be parallelized: scene graph → route
  graph → geometry evidence → capability checks → blocker register →
  readiness decision → human actions (site_package_orchestrator.py:4835-4875,
  each consumes the prior).
- Hidden serialization barrier: `capture_descriptor.json` is re-read and
  rewritten four times mid-spine (:4824, :5183, :5263); any broad spine
  parallelization must first make descriptor writes append-only or batched.

### 3. "Setup/warming dominated task computation" — TRUE FOR THE LANE CODEX RAN; NOT TRUE TRANCHE-WIDE

Evidence from `BlueprintValidation/data/adp009a_tranche1_20260804`:

- Joint-agent `840796_v7_execute`: ~21 min paid GPU, 0 s task compute —
  `joint_agent_inference_executed: false`; ~15 of those minutes were the
  fixed 900 s renderer readiness poll (`seq 1 180; sleep 5`) in the provider
  entrypoint, ended by `joint_agent_local_ovrtx_renderer_not_ready`.
- The two long Aura runs are the opposite: 87% and 86% of paid instance time
  was recorded task compute (e.g. 9088 s of stage compute in 10429 s of
  instance life).
- Tranche-wide: 58.5 paid GPU-hours; 41% attributable to recorded task steps
  (61% restricted to runs that recorded any step durations).

So "eliminate repeated setup" is real but second-order to (a) not paying for
fixed-length readiness polls that fail closed anyway and (b) the ~69 runs
that spent instance time without recording any compute. Pre-baked images and
cached foundations already exist for the GR00T/OSCAR lane (sealed image +
`--reuse-foundation-exact`); the Aura/Joint-Agent lanes install per-run and
remain the gap.

### 4. "Aura and Joint Agent already ran on two GPUs in parallel" — TRUE, BUT HUMAN-SCHEDULED

The concurrency was real (instances 47226054 + 47232529 overlapped, proven by
`vast_prelaunch_inventory_guard.json`), but it was two manually started lane
processes 90 minutes apart. What the harness encodes is *authorization and
admission*, not scheduling: `public_scene_execution_authority.v2`
(`maximum_concurrent_paid_instances == 2`,
`concurrent_paid_compute_authorized`), `--adp-allowed-active-vast-instance-id`
fail-closed binding in `paid_resource_allocator.py`, and the prelaunch
inventory guard. The retrofit cost was visible: the teardown watchdog
fail-closed on the authorized sibling GPU and needed a same-night patch
("Close watchdogs with authorized sibling GPUs").

### 5. "What must remain serial" — CONFIRMED, and now enforced as graph edges

Outcome-blind freeze/seal before outcome release
(`arm_decision_proof.py:1165` seal precedes `:1178` release, with early-access
detection), paid-runtime canary → teardown-proof → object-cleanup ordering
(:1090-:375), controls before scored policy cells (north-star
`simulation_evaluation_contract.controls`), rights → privacy → delivery gates
in the spine (:4414 → :4725 → :4752 → :5033 → :5496), and
teardown/provider-zero before a paid resource is closed. In the new scheduler
these orderings are dependency edges and gate stages — scheduling policy
cannot drop them.

## Decision: what was implemented (general harness, nothing run-specific)

One reusable primitive plus three wirings, all default-serial and
byte-identical to prior behavior until explicitly widened:

1. **`src/blueprint_pipeline/core/stage_graph.py`** — deterministic
   dependency-graph stage scheduler. Contract: declared edges are the only
   ordering authority; `max_concurrency=1` reproduces exact sequential
   behavior; **paid stages never overlap each other without
   `paid_concurrency_authorized=True`** (mirrors the allocator's explicit
   concurrent-instance authority); failures fail closed with typed
   `blocked_by_dependency_failure:<id>` reasons and no automatic retry;
   execution rows are emitted in declared order regardless of completion
   order, with timing/completion order as non-digestable observability
   (`manifest(include_timing=False)` is byte-stable). Custom `serial_group`s
   serialize resource-sharing stages (e.g. one GPU device).

2. **`decision_evidence_execution.execute_evidence_plan`** — the Task
   Evaluation Run evidence seam now derives its true dependency graph from
   the plan (cross-claim steps independent; conditional escalations depend on
   all earlier same-claim steps) and accepts
   `max_concurrency`/`paid_concurrency_authorized`. Serial escalation-gate
   semantics are reproduced exactly under concurrency; paid steps are
   detected by positive `expected_cost_usd`; results and the execution
   manifest are proven byte-identical serial vs concurrent; step bindings are
   validated before any adapter executes (fail-closed before spend).

3. **`site_package_orchestrator`** — the two independent enrichment LLM calls
   run as a two-stage graph, opt-in via
   `BLUEPRINT_SITE_PACKAGE_STAGE_CONCURRENCY` (bounded, fail-closed to serial
   on invalid values). Failure still aborts the spine with the original
   exception.

4. **`canonical_3dgs_pipeline.execute_canonical_3dgs_plan`** — arms execute
   through the stage graph as paid stages: serial by default, overlapping
   only with `max_concurrency > 1` **and** `paid_concurrency_authorized=True`;
   campaign results always keep plan arm order.

Tests: `tests/test_stage_graph.py` (scheduler contract, including proof of
real overlap, proof paid stages do not overlap unauthorized, and declared-
order determinism under reversed completion),
`tests/test_decision_evidence_execution_concurrency.py` (serial/concurrent
byte-equality, escalation gating under concurrency, paid gating, exception
transparency), plus spine and canonical-arm wiring tests appended to
`tests/test_qualification_coverage_edges.py` and
`tests/test_canonical_3dgs_pipeline.py`.

## Follow-ups (ranked, not implemented here)

1. Convert `run_e2e._run_stage` from a closure into stage-graph nodes with an
   explicit edge set (stages 4/5 do not feed 6; only stage 3 does), keeping
   the ledger resume semantics; then give `run_qualification_pipeline` a real
   stage registry and per-phase checkpointing so a webapp-sync failure stops
   re-running Gemini/privacy/ffmpeg from scratch.
2. Overlap the WorldLabs Marble poll (up to 20 min at
   `provider_preview.py:158`) by producing the adapter input earlier and
   polling while later CPU stages run; requires moving preview launch ahead
   of the post-package builders.
3. Replace fixed-length remote readiness polls (the 900 s renderer poll) with
   event-driven readiness plus fail-fast, and extend pre-baked-image coverage
   to the Aura/Joint-Agent lanes — this is the measured setup-cost gap, not a
   scheduling gap.
4. Lane-loop graph in `capture_orchestrator` — low value for the default
   `current` lane set (a pure chain), only pays off when optional lanes
   (retrieval_index, frame_alignment) are requested.

## Claim ceiling

This work parallelizes *orchestration* under existing authority models. It
does not change any scientific ordering: seal-before-release, controls per
scored cell, rights/privacy gates, and teardown-before-closure remain
enforced, now as explicit graph edges instead of implicit code position.
