# Seven-piece automation audit and implementation handoff

Audit date: 2026-09-06. Scope: ADP-009D, day-28 development-only two-candidate
Franka rehearsal. Completion artifact: an authenticated owner intent progresses
through retained construction, controls, policy media/scoring, publication and
resource closure without hand-authored intermediate records. No physical-trial
or qualified-ranking claim follows from this work.

Existing infrastructure has the individual producers and validators. The remaining
work connects their production configuration and terminal state. Prefer small
installer, adapter, and reconciliation changes over new execution machinery.

## Verified baseline and limits

- Pipeline #1720 merged as `2ef95d1eb66c497882f03c5acb1d33b5cb129691`.
  It installs the completed-mesh/splat preparation service and drives its actual
  preparation consumers. The first deployment exposed unreadable unrelated
  historical release metadata during startup.
- #1722 merged as `987e61a40651b655a36db938e1b55a8a3b9410d6` fixes that readback.
  Its focused regression and hosted impacted checks passed. The candidate was
  also replayed as the host service account before deployment.
- The 17:31–17:40 UTC host audit was against deployed `2ef95d1e`. Below, host
  findings refer to that snapshot; recheck them after any subsequent deployment.
- Website task intake now validates destination pose and deterministic success
  fields, and binds an optional same-owner collision mesh. These are code
  findings, not proof of a fresh authenticated production submission.
- Billing reconciliation was producing fresh output. Capacity reported disk
  utilization `0.8141`, Vast credit about `$20.60065`, and `alert_posted=false`.
  Account credit is neither a reservation nor proof the entire chain is funded.
- Paid holds remained intentional. This audit does not authorize arming either
  dispatcher, enabling execution, allocating resources, or increasing a budget.

The seven pieces are not all operationally complete. In particular, installing
#1720 does **not** automatically make legacy public scene 841757 progress:
the installed source list is `mesh` and `gaussian_splat`, and activation is off.

## A — Correct the public-scene versus owner-source handoff

Priority: P0 for the claimed 841757 restart; otherwise an explicit scope boundary.

Evidence: `task_evaluation_scene_preparation_installation.py` installs
`supported_source_kinds=["mesh", "gaussian_splat"]`, local owned-queue submission,
and `activation_enabled=false`. `task_evaluation_scene_progression.py` refuses
unsupported source kinds and returns `construction_prepared` before activation.
An old preparation row alone is not a persistent owner intent.

Required implementation:

1. Choose and record the actual next input: an authenticated completed asset, or
   a rights-admitted public-scene persistent intent for 841757.
2. If the latter is required, add the smallest public-scene adapter/provisioner
   using the existing public source factory. Retain source rights and input hashes;
   do not relabel InteriorGS/SAGE as an uploaded mesh or manufacture owner consent.
3. Preserve no-spend preparation as a separate phase. Connect prepared output to
   the existing activation worker only through explicit bounded authority.
4. Keep a typed reason when the required intent/source is absent. Do not report
   an idle service as a successful scene re-preparation.

Acceptance: real worker rehearsal produces a fresh exact-release preparation
from the chosen source; restart and a compatible release change preserve intent
identity and completed-prefix evidence. A public source under a mesh-only config
must fail with the existing typed source refusal. No allocation occurs.

## B — Install automatic-controls configuration and assets

Priority: P0 before hands-off construction-to-controls operation.

Evidence: the host controls progression service had no
`BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG`, and its expected
configuration file was absent. In `task_evaluation_controls_autoprovision.py`,
`progression_owner_scope()` skips provisioning when that variable is absent.
Thus merged provisioning code can coexist with a running worker that never uses it.

Required implementation:

1. Extend the canonical installer/deployer to materialize a validated controls
   config, intent-store root, trusted clients, and the existing content catalog.
2. Bind retained Franka USD, camera template and runtime bytes by digest; use
   `resolve_robot_catalog` to bind the active release without mutating source content.
3. Ensure service-user readability and required writable roots under the real
   systemd sandbox. Preserve operator-owned configuration; reject conflicts.
4. Require owner-mode preflight to report missing config/assets as blockers.

Acceptance: run the actual autoprovisioner against output from the completed-scene
factory and a retained construction result. It must install a digest-bound controls
intent, be idempotent on the next tick, and reject changed robot/camera/runtime
bytes, expired consent and a mismatched construction result. Rehearse the actual
`_preparation_context` consumer; mocked provisioner success is insufficient.

## C — Wire owner-only selection and authority readback into both dispatchers

Priority: P0 before any dispatcher arming.

Evidence: launch and policy services had neither the scene-intake root nor a
controls-autoprovision config. `task_evaluation_scene_policy_binding.scene_store()`
requires one of these to resolve ownership. Both also lacked
`BLUEPRINT_TASK_EVALUATION_DISPATCH_OWNER_SCOPE`; the implementation defaults to
`all_authorized`, not `persistent_owner_only`.

Required implementation:

1. Canonically install the intended scope and shared owner-store configuration
   in every authority consumer, including both dispatchers and controls progression.
2. Add preflight validation of effective environment (including EnvironmentFile
   precedence), store identity, trusted clients, and sandbox access.
3. Maintain the existing exact profile, standing grant, release, spend, expiry,
   revocation and provider-zero checks. Queue filtering itself is not authority.

Acceptance: with legacy and owner rows together, owner-only mode selects only
the correctly bound owner row. Missing store, forged owner fields, changed profile,
revoked/expired intent and stale release refuse before allocator entry. Deploying
the wiring preserves dry-run, execute-off and existing holds.

## D — Prove the real owner policy handoff

Priority: P0 before paid Quick-10; integration proof remains missing.

Evidence: `task_evaluation_scene_policy_binding.py` checks the owner's two
checkpoint identities and binds a policy attempt. Those checks depend on C.
The host snapshot did not demonstrate a newly configured owner scene traversing
controls, activation, policy envelope and actual dispatcher admission.

Required implementation/verification:

1. Add a hermetic chain test that uses actual construction/controls producers,
   activation worker, `policy_dispatch_envelope`, dispatcher admission and policy
   binding validation. Stub external transport/runtime, not the contract consumers.
2. Carry the exact two owner checkpoint IDs/digests through every handoff; reject
   a legacy default pair, changed checkpoint, third candidate or mismatched attempt.
3. Preserve the diagnostic canary claim ceiling. A scored evaluation separately
   requires the reusable scenario matrix, identical cells/seeds and both controls.

Acceptance: before any paid attempt, run
`tests/test_native_task_arena_policy_canary_lifecycle_rehearsal.py` and
`tests/test_provider_runtime_import_closure.py` for real episode orchestration and
sealed-bundle imports. Retain lossless policy frames, their manifest and derived
review video; deterministic scoring cannot be supplied by the policy itself.

## E — Reconcile terminal results into the persistent owner status

Priority: P1 implementation gap; required before claiming end-to-end completion.

Evidence: `_advance_intent` in `task_evaluation_scene_progression.py` recognizes
already-completed progress but does not emit completion from downstream controls,
policy or publication receipts. Its successful tail emits `awaiting_execution`.
Preparation recovery exists; it is not a complete terminal owner-result join.

Required implementation:

1. Add an idempotent reconciler over existing retained downstream receipts, bound
   to owner intent, attempt, exact inputs and release. Do not infer completion from
   a process exit code or an activation record.
2. Publish truthful phase/status, deterministic result provenance, media references,
   billing/teardown/provider-zero status and authenticated Website readback.
3. Preserve failed children and typed pre-observation media gaps. Distinguish
   completed-unqualified diagnostics, blocked runs and unfinished resource closure.
4. On an authorized successor, adopt compatible completed stages. Never rerun
   completed GPU work for identical inputs or reset retry/spend limits on deploy.

Acceptance: replay success, failed child, stale receipt, duplicate tick, restart,
changed release, ambiguous create and incomplete teardown. A terminal policy result
must update the correct owner status; absent closure must remain explicit. Verify
the Website receipt authenticates and matches the same intent/result digest.

## F — Repair preflight's option-value serialization

Priority: P0 before interpreting chain preflight as admission evidence.

Evidence: the host preflight at 17:27:16 UTC had nine blockers. Two were actual
probe failures for intake and spend guard: argparse reported
`argument --read-only-paths: expected one argument`. `systemd_run_command()` passes
the value as a separate argv element; systemd path values may start with `-`.

Required implementation: serialize arbitrary path values as
`--read-only-paths=<value>` and `--read-write-paths=<value>` (or an equally robust
structured transport). Preserve systemd optional-path semantics. Test the real
parser with empty, leading-dash, multiple-path and space-containing values.

Acceptance: both failed probes execute under their actual service sandbox and
produce structured findings. Reclassify the three old-release path findings by
their source: old `safe.directory` declarations are not necessarily active code
paths. Keep the four intentional activation/dispatcher holds visible; never silence
all nine blockers to obtain a green report.

## G — Alert delivery and full-chain funding visibility

Priority: P2 operational improvement; not a replacement for hard spend gates.

Evidence: capacity already collects credit, applies a threshold and creates alerts.
The host had no configured alert webhook and `alert_posted=false` despite a disk
warning. Fresh billing is working; external notification delivery is not proven.

Required implementation: expose an explicit unconfigured/delivery-failed state,
support an authorized notification destination, and retain deduplicated delivery
receipts. Report remaining bounded costs by provider alongside liabilities and
available credit. Do not describe a current balance as reserved whole-run funding.

Acceptance: a fake notification sink tests threshold, deduplication and failure;
an authorized destination needs one delivery/readback receipt. Do not send a real
message until the user authorizes that destination. Leave allocator hard caps,
fresh-credit admission, billing and teardown independent of notification success.

## Handoff order and final evidence

Implement F and the no-spend configuration in B/C, then resolve A for the chosen
source and rehearse D/E. Run focused tests for each changed consumer and installer;
ordinary PRs use impacted checks. Avoid a broad test rerun without a changed risk.

Before seeking a paid decision, publish one current-release no-spend packet:
effective service configuration, actual owner intent/preparation, controls intent,
two-candidate policy binding, completed-prefix replay, categorized preflight,
resource ownership/provider-zero, fresh billing, bounded per-provider exposure and
the exact proposed arming/execute changes. Keep paid arming and the paid flip as
separate deliberate actions. A fresh authenticated Website-to-Pipeline submission
and terminal readback are still required production integration evidence.
