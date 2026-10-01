# Control-plane concurrency acceptance: reviewed TDD plan

ADP-009D / day 28. Observed blocker: Plan 13c has no joined scene-chain
load artifact. Completion artifact: an immutable JSON report and a summary
in `docs/BETA_CAPACITY_COST_STORAGE_MODEL_2026-07-08.md`, bound to the source
commit, fixture digests, host, measured concurrency, and all stage outcomes.
Every input and result remains `development_only`.

The requested `superpowers:writing-plans` skill is unavailable in the installed
catalog and filesystem. This document receives independent specification and
quality review before implementation instead. Reviews must assess the concrete
chain below, not accept a collection of existing tests as the load proof.

## Acceptance boundary

The owner concurrency decision is pending. Require `--expected-beta-concurrency`
as a CLI argument and test twice that value. An exploratory run may choose a
clearly recorded provisional value; it does not close the owner-sized gate.
Require an explicit `--maximum-retained-gib` threshold and keep observed results
separate from modeled capacity and costs.

Use actual intake, scene progression, preparation, compilation, activation,
provider-output ingestion, result joining and cleanup entrypoints in one chain.
Replace only external provider/object-store execution with deterministic local
fixtures and bounded local transport. Do not replace admission, queue claiming,
reference validation, compilation, output digest checks, pins, reservations,
result validation or retirement with unconditional successes. A missing joined
stage, swallowed blocker, fabricated completion or manually emptied directory
makes the report fail.

The implementation must join these production transitions: signed
`stage_scene_intent` -> `process_scene_intents` factory/publication and sealed
preparation request -> `run_preparation_service` /
`process_launch_preparation_queue` sealed references and pending construction
envelope -> `process_scene_intents` configuration activation link ->
`process_launch_activation_queue` canonical no-allocation configuration launch
preparation -> external configuration fixture through real publication and
finalization -> real configured-controls readiness/episode preparation ->
`process_launch_preparation_queue` sealed compilation envelope ->
`process_episode_compilation_queue` real compiler result and packet ->
`stage_configured_controls_activation` initial native construction activation
-> `process_launch_activation_queue` canonical no-allocation native launch
preparation -> fixture policy execution ->
`provider_output_range_ingestion.ingest_selected_members` selected members ->
`task_evaluation_policy_canary_result_projection.build_policy_canary_result_projection`
validated terminal result
projection -> `control_plane_storage_gc.run_storage_gc` authorized retirement.
Keep both activation receipts and their exact preparation bindings. The first
activation must consume a pending/processing construction envelope. A terminal
construction envelope cannot be reopened, have its preparation status rewritten,
or be accepted by a synthetic successful preparer. Native controls activation
requires real predecessor lineage; start with initial construction. This ordering
correction was independently checked against the production worker on October 1.
Fixture scene-configuration output enters the existing publication validator
at the provider return boundary; policy fixture output enters the existing
archive/member ingestion boundary. The fixture must bind each returned object
to the preceding stage's scene/run/attempt and source identity. Confirm exact
projection API and all artifact contracts in slice 2 before accepting the
joined single-scene test. An unsupported transition is a failed gate.

Scene preparation/configuration and policy execution are explicit external
fixture boundaries. Each must record its fixture identity and bytes; it must
advance the same validated production result/publication contracts that the
next stage consumes. Run the real native episode compiler on a closed fixture.
Include the streamed needed-set ingestion path, with unrelated provider output
members remaining in the fake object store, and measure actual transfer and
materialization bytes. Do not interpret fake provider execution as robotics,
Cloud Run, paid settlement or rights/provenance proof.

## Isolation and measurements

Run concurrent scenes in separate child processes so mutable test seams and
CPU measurements do not leak across scenes. Share one physical work-volume
root, reservation ledger, pins and content stores across all children. Block
outbound networking in children and remove provider credentials from their
environment. The parent creates a unique, owned, non-symlink root. Existing
production paths and queues are never eligible for harness cleanup.

Use three separately owned roots: control-plane state/stores on the work
volume, fixture object storage, and ephemeral worker files. Only the first
root counts toward host retention acceptance; report fixture/worker allocation
separately. Do not put entire simulated provider archives in the measured host
tree and later label their deletion as a streaming saving. Use finite per-child
and global deadlines; timed-out or crashed children retain a failed receipt and
their artifacts, and make the aggregate exit nonzero.

Measure allocated blocks, not logical size, with the production inode-aware
usage implementation. Sample the owned tree and filesystem free bytes at a
bounded interval during the entire run. Record per-stage wall/CPU seconds and
allocated-byte peak/delta; derive p95 using a documented nearest-rank rule.
Retain the aggregate baseline, maximum and final samples, peak active scenes,
each source/result digest, queue terminal states, residual leases/pins and
actual capacity refusals. Attribute the owned-tree result separately from
unrelated host filesystem changes.

Terminal acceptance requires every requested scene to finish every stage,
actual overlapping execution, zero unexpected capacity refusals, all leases
and transient pins released, and retained owned bytes within the explicit
threshold. Preserve failed scene artifacts; never erase failures to pass.
Successful output evidence may remain, counted in retained bytes. The report
is written outside transient stage directories before authorized retirement.
Require a start barrier and a measured representative storage-bearing stage
milestone with exactly N active scenes. Merely two overlapping scenes cannot
pass an N-scene run. Report how long the milestone held and actual peak
concurrency. Disk-flat acceptance additionally requires
`final_control_plane_allocated - baseline_control_plane_allocated <= X GiB`;
the retained-byte ceiling alone cannot pass this check. Record the explicit
threshold in every report.

## TDD slices

1. Write failing tests for required CLI parameters, owner/provisional labels,
   bounds, shared-root ownership, outbound network denial and allocation-aware
   p95/peak reporting. Implement the orchestrator/measurement layer only.
2. Add a single-scene integration test asserting actual intake through terminal
   results with all intermediate contracts joined by digests. Corrupt each
   boundary in turn; the harness must fail with the original typed blocker.
   Implement fixture providers and one joined scene path. Inspect fixture
   validators and compiler outputs before adding concurrency.
3. Add RED tests for two simultaneous scenes sharing the real disk ledger and
   content store; prove overlap, absence of duplicate external work, streamed
   subset size, teardown and preserved terminal evidence. Implement process
   orchestration without disabling admission or content-store verification.
4. Add insufficient-capacity and failed-cleanup cases that produce a nonzero
   exit and a failed report; prove neither can be converted to success by a
   fixture or summary writer. Add crash/restart reconciliation where production
   stages retain a resumable queue.
5. Run targeted tests, script/module governance, impacted tests/sentinels, then
   independent specification and different-model quality review on the exact
   candidate. Merge one PR through protected exact-head gates. Run the script
   on the live host at the promoted SHA with paid providers disabled. Publish
   the actual report and size quotas only from its measured boundary.

The storage margin table already reflects 256 MiB for preparation/compilation
at the current main. Do not replace the distinct scene-configuration 512 MiB
margin, which belongs to a different reservation contract.

## Outstanding decisions and independent proofs

The concurrency parameter awaits the owner. Plans 11/12 remain with their
active owner and cannot be made safe by this harness. Actual Cloud Run IAM,
environment preflight, three shadow passes per closure class, settlement and
transport-expiry teardown have their own live gates; passing this no-paid
load test cannot stand in for them.

## Fixture retirement boundary clarification (October 1)

Independent specification review and a different quality review approved the
plan boundary only: the short benchmark cannot prove the six-hour automatic
terminal-pin reconciliation. After all producers stop and join, verify owned
queues, requests, readers and leases, terminate future fixture execution through
an explicit development_only owner-cancellation record, and validate every
owned pin and its dependency cascade. Then release enumerated fixture pins
through the real release_storage_pin API and run production GC with recorded
zero-age parameters and strict namespace translation through require_storage_class.
Report this as **fixture-owner release followed by production GC**, excluding
automatic terminal reconciliation and normal retention timing from acceptance.
Never invent launch consumption or settlement, change clocks, manually clear
directories or retire failed artifacts. Automatic terminal retention remains a
separate unproven gate until its real age and proof requirements are observed.
