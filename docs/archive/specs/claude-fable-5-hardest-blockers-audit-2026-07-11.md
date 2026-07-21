# Claude Fable 5 handoff: hardest blockers from the last 48 hours

> Archived cloud-agent handoff snapshot. Not a current operator instruction.

Audit snapshot: 2026-07-11 06:58 CDT
Repository: `BlueprintCapturePipeline`
Authoritative baseline: `df030e45a4f8046c08d507df645eea31d91ea32f`
Branch state at snapshot: clean `main`, `HEAD == origin/main`, `0 ahead / 0 behind`
Audit-output state: this handoff is the only new untracked file; it is not part of
the authoritative source baseline unless intentionally reviewed and committed.

## Execution instruction

Start from `df030e45`. Do not reimplement the startup supervisor, quarantine,
phase-trace, spend-reconciliation, attempt-lineage, closure, controller-adapter,
persistent-task, review-media, or strict-scorer modules already merged in that
commit. Audit their integration and close the defects below.

Do not begin a paid provider attempt until FABLE-001 through FABLE-007 are fixed,
the exact current-source images are published by immutable digest, all required CI
checks pass, and an operator explicitly supplies a total spend cap and paid-run
authorization. This lane is simulation-only. Physical G1 readiness, deployment
approval, field safety, and real-world task success are outside this closure.

The first paragraph of
`docs/specs/kitchen-random-task-e2e-cloud-handoff-2026-07-10.md` is historical:
the dirty-local repairs it says are absent from `main` were merged in `df030e45`.
Use that file for the selected task and historical evidence, not as the current
source-state instruction.

## Executive verdict

The last 48 hours produced a very large amount of useful fail-closed scaffolding,
but not a closed live task. The current honest status remains:

`local_contracts_advanced_live_end_to_end_task_success_not_proven`

The most difficult blocker is no longer simply “find an available GPU.” The new
code contains several integration gaps that can either stop the first strict
episode before GR00T runs or allow incomplete/foreign worker evidence to be
rebound into a passing closure. Fix those before spending again.

The highest-risk findings are:

1. The host closure trusts worker-supplied `passed` proof rows and overwrites
   their identity instead of independently validating the leaf evidence.
2. The “full episode” media horizon is inferred from the frames that happened to
   arrive, so equally truncated camera streams can pass.
3. Initial G1 proprioception groups each leg with the token `left` or `right`,
   which also selects arm and hand DOFs and can fail before the initial GR00T
   query.
4. GEAR-SONIC controller output is positionally zipped onto MuJoCo joints without
   an explicit controller-to-model-to-Isaac joint-order contract.
5. Launch and closure do not compare attempt, image, source, and worker identities;
   the finalizer can rebind stale evidence to the current host identity.
6. Relative task success is evaluated against the immediately preceding step,
   not an immutable episode baseline.
7. `df030e45` is hosted-CI red on Bandit, `main` has no branch protection or
   ruleset, and the 107-row release ledger has no trusted commit/release
   attestation.
8. No current-source sealed GR00T + OSCAR image or fresh same-allocation live
   episode has passed. The local generic Isaac image is healthy but source-bound
   to the pre-commit `d1220f78 + dirty patch` identity.

## What changed in the last 24–48 hours

### Main-line commits

| Commit | Local time | Main result | Proof ceiling |
|---|---:|---|---|
| `cfe742ad` | Jul 9 07:36 | Published pre-beta remediation | Repository remediation, not live proof |
| `7a462e07` | Jul 9 13:25 | Published beta launch hardening | Policy/gate contracts, not deployment proof |
| `95a11a0f` | Jul 10 08:11 | Hardened launch quality and runtime contracts | Local/CI controls |
| `40859bd9` | Jul 10 11:40 | Hardened Isaac GPU startup reliability | Startup components, no task episode |
| `2def1d68` | Jul 10 11:48 | Merged PR 66: stance search + graded trace consistency | Graded trace support, not strict action recovery |
| `d1220f78` | Jul 10 13:09 | Merged PR 67: startup P0/P1 work packages | Startup reliability baseline |
| `df030e45` | Jul 11 06:44 | Merged G1 kitchen audit remediation/runtime hardening | Current code baseline; live closure still absent |

`df030e45` alone changed 116 files with 13,549 additions and 3,159
deletions. Across the reachable 48-hour history, the change surface is much
larger, so passing unit tests are necessary but not sufficient evidence of
cross-module correctness.

Four July 10 stash commits named as alternate startup stacks are not main-line
commits and must not be treated as additional merged proof.

### Material progress to preserve

- Atomic startup supervision, machine quarantine, fast/review canaries, kitchen
  asset gates, phase tracing, cumulative spend reconciliation, and teardown
  accounting are now present.
- Random-task selection generations, attempt identity, bundle compatibility,
  append-only run indexing, and buyer-readout closure projection now exist.
- Visual-mesh/instance-proxy geometry, RGB integrity rejection, ordered review
  media, an official GEAR-SONIC adapter, persistent Isaac task service, and
  strict action-aware scorer contracts were added.
- Local source governance passes: `status=passed modules=360 clis=147`.
- Ruff passes on the changed Python surface.
- `tests/test_quality_gap_ledger.py`: 12 passed.
- The exact current local all-marker suite completed with 4,733 passed and 3
  skipped in 21m13s. The newly identified integration defects remain because
  their paths are untested or, in the media case, encoded as expected behavior.
- Sim-Only Local Gate, Python Compatibility, CodeQL, source governance, typed
  contracts, container contracts, SBOM/license, dependency security, Ruff, and
  the normal CI test job passed on `df030e45` at the audit snapshot.
- A local linux/amd64 generic Isaac image exists at
  `sha256:c237ce2b34e4fda289089a3b37e21d558ae127b36f6fe6b7271cac0bd97d1057`;
  its static `/isaac-sim/python.sh` healthcheck passes.

### Current release state

- GitHub CI run `29151463039` is red only on the Bandit policy job.
- Full Test Lane run `29151463046` executed 4,735 passing tests and one skip,
  then failed its evidence gate with `cpu_full_junit_skipped:1`. The skipped
  test was
  `tests/test_lerobot_export_validation.py::test_native_parquet_without_pyarrow_fails_closed`:
  the CI environment had `pyarrow`, so the test declared its hermetic
  missing-dependency branch unreachable.
- The authoritative quality ledger reports 107 total rows: 91 partial, 16 open,
  and 0 closed.
- `main` is not protected, and the repository has no applicable ruleset.
- The current changelog/remediation prose still contains “uncommitted” and
  “pending rerun” language that became stale when `df030e45` was pushed.

## Observed failure evidence

### Live geometry and media

Attempt 023:

`output/kitchen_random_task_e2e_20260710T131557Z/attempt_023_do_rtx6000ada_microwave_seed/`

- Result: `blocked`.
- Exact top-level blockers:
  `manipulation_pov_geometry_failed`, `placement_validation_failed`.

Attempt 025:

`output/kitchen_random_task_e2e_20260710T131557Z/attempt_025_do_rtx6000ada_microwave_seed_provider_rate_fix/`

- Result: `blocked`.
- Exact top-level blockers:
  `manipulation_pov_geometry_failed`,
  `third_person_verify_frame_not_saved_or_corrupt`,
  `placement_validation_failed`.
- Geometry reported shoulder-to-affordance `1.1952 m` and nearest
  effector-to-affordance `1.2929 m`, with no active arm chain in the failed
  frames. Those measurements came from the pre-fix link-origin path and must not
  be used to judge the new visual-mesh implementation.
- `verify_0000.integrity_rejected.json` records
  `rgb_frame_blank_black`, `rgb_frame_flat_corrupt`, `dark_fraction=1.0`, and
  `luma_std=0.0`.

### Provider capacity

Attempts 027, 028, and 029 all failed before a RunPod pod ID with:

- `runpod_secure_cloud_create_capacity_unavailable`
- `provider_capacity_unavailable_before_instance_created`
- `launch_failed_provider_capacity_unavailable`

The provider response was HTTP 500: “This machine does not have the resources
to deploy your pod.” Catalog stock was therefore advisory, not reservation
proof. These failures created no provider allocation and no GPU spend.

### Task/evaluator truth

- Four audited task-success judge artifacts were `not_proven`.
- Completed structural loop/evaluator artifacts still reported
  `manipulation_success_proven=false`.
- Historical local-CV or VLM consistency labels are review/support evidence;
  neither is strict numeric forward/inverse action recovery.
- No post-`df030e45` `g1_kitchen_attempt_closure.v1` live artifact exists.

## Ranked work packages

## FABLE-001 — Reconstruct closure from trusted leaf evidence

Severity: P0 / critical
Category: proof integrity / false-positive prevention
Model-solvable: yes

### Evidence

- `src/blueprint_pipeline/g1_kitchen_digitalocean_closure.py:204` copies
  `worker_manifest.g1_kitchen_proof_rows`.
- The finalizer overwrites every row's `identity_binding` with host-calculated
  identity and adds only a canonical in-memory worker-manifest hash.
- `src/blueprint_pipeline/g1_kitchen_attempt_closure.py:190` validates status,
  blockers, and identity shape. Except for teardown/final inventory, it does not
  validate row-specific leaf schemas, hashes, or attestations.

### Required implementation

1. Add a validator registry keyed by proof-row ID and allowed schema version.
2. Reconstruct every row from collected leaf artifacts. Do not accept a worker
   boolean as the row verdict.
3. Validate artifact bytes, SHA-256, size, schema, run ID, attempt ID, nonce,
   source commit/dirty hash, image digest, bundle/task/selection digests,
   provider allocation, stage/session identity, and action chronology.
4. Verify controller, policy, task-transition, scorer, and semantic-review
   attestations against pinned public-key fingerprints.
5. Hash the actual collected worker-manifest bytes and retain its path. Do not
   hash only a re-serialized in-memory object.
6. Compare worker-emitted identity to host identity. Never inject or overwrite a
   missing/mismatched worker identity.
7. Make buyer/readout projection consume only the independently validated
   closure.

### Acceptance tests

- Forged `status=passed` row with no leaf artifact blocks.
- Mismatched nonce, source, image, stage, task, or allocation blocks.
- Tampered artifact, invalid signature, unknown schema, missing leaf, and
  cross-attempt replay block.
- A worker row with no identity remains blocked; the host may not repair it.
- A complete signed fixture closes and buyer projection preserves the same
  verified digests.

## FABLE-002 — Bind the full media horizon to the attempt

Severity: P0 / critical
Category: review evidence / false-positive prevention
Model-solvable: yes

### Evidence

`g1_kitchen_digitalocean_closure._collected_media_rows()` sets
`expected_frame_count` from the overview/robot-POV files it discovers. The
existing unit test explicitly accepts one frame per camera and passes `1` as the
expected horizon. If an intended 20-step episode uploads only frame zero for
both cameras, equal truncation can pass.

### Required implementation

1. Read expected steps and terminal step from the immutable attempt/task/executor
   manifest, never from collected files.
2. Require exact contiguous indices `0..N-1` for every required camera.
3. Reject gaps, duplicates, extras, stale hashes, reordering, and equal-stream
   truncation.
4. Bind camera frames to the action SHA, transition measurement, stage/session,
   and timestamp for the same step.
5. Require semantic-review request/response coverage for the identical ordered
   set.
6. Bind scenario count to the request, not discovered directories.

### Acceptance tests

- One-of-N frames per camera blocks.
- N overview plus N-1 POV blocks.
- Duplicate pixels/hash at two steps, out-of-order timestamps, or a frame from
  another attempt blocks.
- Exact N/N media plus complete semantic review passes.

## FABLE-003 — Replace substring proprioception grouping

Severity: P0 / critical
Category: live startup / policy input correctness
Model-solvable: yes

### Evidence

`src/blueprint_pipeline/isaac_runtime_task_backend.py:182` groups the left leg
with `group("left", limit=6)` and the right leg with `group("right", limit=6)`.
A normal G1 also has left/right shoulder, elbow, wrist, and hand joints, so the
selection can exceed six and raise
`persistent_isaac_initial_proprio_group_dimension_mismatch:left` before the
initial GR00T call. Current tests bypass `initial_policy_state()`.

### Required implementation

1. Define one explicit ordered canonical DOF map for left leg, right leg, waist,
   both arms, and both hands.
2. Resolve aliases deliberately; reject missing, duplicate, ambiguous, or
   unexpected required joints.
3. Emit the complete observed inventory, resolved map, dimensions, and mapping
   digest in the initial-state artifact.
4. Include velocities/other modalities only if the pinned GR00T schema requires
   them; do not synthesize missing live state.

### Acceptance tests

- Realistic G1 29-DOF and hand-extended inventories map deterministically.
- Extra arm/hand DOFs do not enter leg vectors.
- Missing, duplicate, alias-collision, and left/right swap cases block.
- Initial-state artifact passes the `UNITREE_G1_SONIC_STATE_DIMS` contract.

## FABLE-004 — Make controller/FK mapping named and end-to-end tested

Severity: P0 / critical
Category: controller semantics / state integrity
Model-solvable: yes; live/container acceptance also required

### Evidence

`gear_sonic_official_zmq_executor._official_mujoco_fk()` sorts all non-free
MuJoCo joints by qpos address and zips `[29 body, 7 left hand, 7 right hand]`
onto the first 43. There is no controller-provided joint-order schema or
permutation validation. Valid-length values can therefore be assigned to the
wrong semantic joints. The tests inject a fake transport and fake FK solver;
the real ZMQ topology and pinned XML are untested.

### Required implementation

1. Pin the protocol-v4 body and hand joint names/order.
2. Require controller output to include schema version, ordered joint names,
   and mapping digest.
3. Compare exact joint sets/order against both the pinned MuJoCo XML and the
   live Isaac articulation before applying targets.
4. Reject positional-only, duplicate, unknown, missing, or permuted mappings.
5. Emit controller revision/hash, model hash, mapping digest, and applied Isaac
   DOF mapping with each result.
6. Add real ZMQ tests for bind/connect direction, slow joiner, token matching,
   stale replies, timeout, and concurrent attempts.
7. Add a container smoke using the pinned WBC tree and real model XML.

### Acceptance tests

- Two asymmetric perturbations change the intended distinct joints and produce
  distinct FK/Isaac states.
- Left/right and adjacent-joint permutations block.
- A protocol-v4 result traverses policy -> ZMQ controller -> MuJoCo FK -> Isaac
  with one action SHA and mapping digest.

## FABLE-005 — Enforce attempt/image/source identity before allocation

Severity: P0 / critical
Category: reproducibility / replay prevention
Model-solvable: yes

### Evidence

- The paid launcher reads the attempt manifest but only requires a launch nonce.
- Closure takes source identity from the attempt and image digest from the
  caller-supplied image ref without comparing them to attempt/image evidence.
- `validate_bundle_compatibility()` is not an obligatory paid-launch gate.
- The sealed Groot/OSCAR healthcheck emits `configured_g1_usd_exists`; the
  worker-evidence validator requires `configured_g1_asset_binding_valid`, so an
  otherwise valid sealed healthcheck payload is contract-incompatible.
- The local generic image embeds source commit `d1220f78` plus a dirty-patch
  hash. The current clean repository identity is `df030e45`; content similarity
  does not satisfy the current identity contract.

### Required implementation

1. Before any allocation, require:
   attempt image digest == launch digest == registry digest == worker evidence
   digest.
2. Require attempt source commit/dirty hash == image source identity == bundle
   source identity.
3. Validate the actual payload and transport bundle schemas/hashes before spend.
4. Require worker-emitted run/attempt/nonce/source/image/task/bundle identity;
   compare rather than overwrite.
5. Define one shared image runtime-metadata schema. Preserve both
   `configured_g1_usd_exists` and `configured_g1_asset_binding_valid` as distinct
   claims.
6. Test the real sealed-healthcheck payload through the real evidence assembler.

### Acceptance tests

- Tag-only launch, stale image, old commit, wrong patch hash, swapped bundle,
  wrong nonce, or foreign worker result blocks before allocation.
- Current clean-source digest-pinned image and compatible bundle pass pre-spend.
- The sealed healthcheck artifact is accepted without hand-shaped fixture data.

## FABLE-006 — Evaluate relative success from an episode baseline

Severity: P0/high
Category: simulator semantics / false-negative prevention
Model-solvable: yes

### Evidence

The backend measures immediate `before`/`after` around one action, and the
service/evaluator applies the `increase_at_least` threshold to that step pair.
Two valid `+0.20 rad` actions fail a `+0.35 rad` episode criterion even though
the microwave door moved `+0.40 rad` overall.

### Required implementation

1. Capture one signed baseline after stage settle and before action zero.
2. Emit `episode_initial_value`, `step_before`, `step_after`, `step_delta`, and
   `episode_delta` for every transition.
3. Relative change criteria use current minus episode initial; per-step delta is
   diagnostic only.
4. Absolute target criteria remain separate.
5. Bind baseline to attempt, nonce, session, stage fingerprint, target prim, and
   task-contract hash.

### Acceptance tests

- `+0.20 +0.20` satisfies the `+0.35` episode criterion only after step two.
- Oscillation/regression uses current-vs-initial truth, not accumulated absolute
  motion.
- Stage/session restart, changed prim, or baseline tampering blocks.

## FABLE-007 — Restore release integrity

Severity: P0 for publish / high for runtime
Category: CI, SAST, branch protection, release evidence
Model-solvable: code/CI portion; repository-admin action also required

### Current Bandit blockers

Six orphaned triage fingerprints and eight untriaged findings:

- B310:
  - `scripts/run_isaac_g1_kitchen_parity_eval.py:2489`
  - `src/blueprint_pipeline/gpu_render_providers.py:186`
  - `src/blueprint_pipeline/gpu_render_providers.py:208`
  - `src/blueprint_pipeline/gpu_render_providers.py:865`
  - `src/blueprint_pipeline/isaac_persistent_task_completion_client.py:39`
  - `src/blueprint_pipeline/wam_strict_action_consistency_scorer_client.py:45`
- B615:
  - `src/blueprint_pipeline/retrieval_index_stage.py:1252`
  - `src/blueprint_pipeline/retrieval_index_stage.py:1257`

### Required implementation

1. Review reachability; fix genuine URL/download boundaries. Do not blanket
   `# nosec` the findings.
2. Centralize safe outbound HTTP behavior: allowed scheme/host, redirect policy,
   timeouts, maximum response size, and loopback-only exceptions where intended.
3. Keep presigned PUT support while rejecting unsupported schemes and redirect
   escape.
4. Make model retrieval explicitly gated, revision-pinned, remote-code-disabled,
   and preferably prefetch/cache-bound for release lanes.
5. Remove or deliberately re-review orphaned triage entries and regenerate exact
   fingerprints only after source is frozen.
6. Make the parquet missing-dependency test hermetic by controlling import
   availability inside the test; do not skip merely because CI installed
   `pyarrow` through another dependency.
7. Enable branch/ruleset protection: required PR, CI, Full Test Lane, review
   resolution, no direct/force push, and a time-bounded independently approved
   break-glass procedure.
8. Build an external signed command/release attestation bound to `GITHUB_SHA`,
   workflow run IDs, retained artifact hashes, and release ID. Avoid a circular
   “commit evidence back into the commit it attests” design.
9. Recompute the 107-row ledger from those attestations. Do not set rows closed
   merely because local tests pass.

### Acceptance tests

- Bandit: 0 high, 0 untriaged medium, 0 orphaned fingerprints.
- Full Test Lane: complete collection, 0 failures, and 0 skips under its
  fail-closed evidence policy.
- All required hosted checks pass on the same SHA.
- Negative branch tests prove red checks cannot merge and direct/force push is
  denied.
- Ledger command/release attestation verifies cryptographically and is bound to
  the current commit/release artifacts.

## FABLE-008 — Build and publish the current sealed image

Severity: P0 for live attempt
Category: image/runtime reproducibility
Model-solvable: partially; operator/build infrastructure required

### Current facts

- The generic local Isaac image is linux/amd64 and statically healthy, but is
  identity-bound to the pre-commit source state.
- The configured generic Isaac ref file is tag-only and dated July 2.
- The configured sealed-image ref is digest-pinned but dated July 7 and does not
  contain `df030e45`.
- No current local sealed GR00T + OSCAR image is present.
- The sealed build requires 120 GiB free; the data volume had roughly 12 GiB
  free at the snapshot.
- The configured remote amd64 builder is blocked by an SSH host-key change.
  Verify the new fingerprint independently; do not delete the known-host entry
  blindly.
- File-mounted provider, registry, Hugging Face, and object-store credentials
  exist and are mode `0600`. Explicit push/paid gates remain unset.

### Required implementation

1. Obtain adequate build storage or a trusted native linux/amd64 builder.
2. Independently verify the remote builder's new SSH host key.
3. Build from clean `df030e45` with pinned Isaac base, GR00T/WBC revisions, model
   assets, and checkpoints.
4. Pass the real sealed-image healthcheck and write source/image manifests.
5. Push a versioned tag and resolve it to a registry `@sha256:` reference.
6. Run fast and 640x480 review canaries against that exact digest on one
   allocation, then prove teardown and zero inventory.

### Acceptance

- Registry digest, source identity, Dockerfile/base/controller/model/checkpoint
  hashes, and runtime metadata all agree.
- Both canaries pass on the same provider allocation and launch nonce.
- No task-success claim is made from image/canary proof alone.

## FABLE-009 — Close one live microwave episode

Severity: P0 for the kitchen objective
Category: geometry, policy, controller, persistent simulation, review
Model-solvable: partially; paid provider execution required

### Frozen task

- Task: `microwave_door`
- Target: `/root/Microwave017`
- Affordance: `/root/Microwave017/Microwave017_Door`
- Reference stance: `[-1.229635, 1.471274, 0.84]`
- Reference yaw: `3.141593`
- Success: episode-relative door articulation increase of at least `0.35 rad`

Do not reroll merely because the task is difficult.

### Required sequence

1. Run exact-source pre-spend identity/bundle validation.
2. Allocate one eligible RTX worker.
3. On that allocation: fast canary -> asset gate -> review canary -> scene load.
4. Measure live visual-mesh shoulder/wrist/palm/fingertip and affordance geometry.
5. Pass independent stance, collision, clearance, facing, reach, and POV gates.
6. Capture signed episode baseline.
7. Query fresh GR00T on the initial real observation.
8. Apply each named action through the official controller/FK map to the same
   Isaac stage/timeline.
9. Measure the same target articulation after every action.
10. Save and admit the full ordered overview and robot-POV horizon.
11. Run strict forward and inverse scorer plus separate semantic review.
12. Stop only because the registered task criterion passed or a truthful
    blocker/timeout occurred.
13. Upload artifacts, terminate, and prove zero inventory/spend.

### Completion definition

All FABLE-001 closure rows pass under one immutable identity. Startup, renderer,
structural loop, generated media, or marker completion alone is insufficient.

## FABLE-010 — Implement real scorer and semantic-review services

Severity: high
Category: evaluator validity
Model-solvable: service implementation yes; calibration/review inputs required

The repository now has strict request/result validators and HTTPS clients. It
does not contain a calibrated external action-recovery scorer or a configured
full-episode semantic-review backend.

Required deliverables:

1. Independent forward and inverse action-recovery service.
2. Numeric recovered action, per-dimension error/uncertainty, timing, units,
   termination chunk, controller/generated-state hashes, evidence refs, and
   calibration identity.
3. Signed response with model/code/calibration digests and non-replayed runtime
   ID.
4. Held-out positive/negative calibration including action-agnostic visual
   motion, swapped actions, replayed motion, wrong dimension/unit/timing, and
   threshold edge cases.
5. Separate semantic-review service over the exact ordered frame set, with
   explicit abstention and per-frame hashes/visibility/occupancy/coherence.
6. Semantic review must never determine articulation success; strict consistency
   must never substitute for semantic task review.

## FABLE-011 — Make provider retries resumable but authoritative

Severity: high
Category: capacity, cost, teardown
Model-solvable: orchestration; live capacity is external

Required behavior:

- Catalog inventory remains advisory; create/reservation is authoritative.
- Create-without-ID is no allocation/no spend.
- Retry one provider at a time within one cumulative time/dollar cap.
- Preserve the identical task/attempt input identity across capacity retries;
  allocate a new attempt ID only when the execution attempt changes.
- Reuse a passing warm allocation for the full job; do not cold-launch again
  after canary success.
- Track compute, container disk, persistent volume, and network-volume cost
  separately.
- Every non-promoted branch terminates and verifies absence. Unknown inventory
  is blocked, not zero.

External inputs required before live execution:

- explicit paid-run authorization;
- total goal spend cap and one-resource limit;
- explicit provider API and image-push gates;
- exact current image digest;
- scorer URL/token and semantic-review command;
- FK/task attestation signing keys with trusted public-key pins.

## FABLE-012 — Run the SC3-style fidelity study separately

Severity: critical only for a public rank-fidelity claim
Category: research/calibration
Model-solvable: study tooling; data, policies, compute, and human review required

This is `SC3-22` / `EVID-01`, not part of the one-task kitchen runtime closure.
Do not block the sim-only product lane on physical validation or pretend one
microwave episode proves evaluator fidelity.

Minimum study:

- at least seven independent policies/checkpoints;
- explicit inclusion/exclusion criteria;
- matched tasks/conditions/replicates;
- locked InD and OOD splits;
- raw per-cell outputs, failures, and abstentions;
- independent human-label protocol and adjudication;
- correct Pearson, Spearman, and MMRV;
- hierarchical/bootstrap confidence intervals;
- coverage-vs-abstention curves;
- exact code/model/data/config digests;
- frozen report that distinguishes paper results from Blueprint measurements.

## Broader public-launch evidence not closed by code

The 16 open ledger rows are not one undifferentiated blocker list.

| Scope | Rows | Required external proof |
|---|---|---|
| Repository governance | `REL-02` | Protected branch/ruleset and negative tests |
| Scientific claim | `SC3-22`, `EVID-01` | Frozen fidelity study |
| Capture/package | `EVID-02`, `EVID-04` | Clean real capture E2Es and privacy corpus |
| Live sim/provider | `EVID-03` | Authenticated exact-image canary with teardown |
| Live operations | `EVID-05`–`EVID-08`, `EVID-13` | Restore drill, load/SLO, incident drills, cloud/IaC readback |
| Paid/legal/device | `EVID-09`–`EVID-12` | Settlement, entitlement, legal/rights, device-path proof |
| Physical-only | `EVID-14` | Real-robot trials only if that claim is made |

`EVID-14` is explicitly nonblocking for this sim-only lane.

## File and artifact map

### Start here

- `docs/G1_KITCHEN_RUN_DEEP_AUDIT_2026-07-10.md`
- `docs/G1_KITCHEN_RUN_DEEP_AUDIT_REMEDIATION_2026-07-10.md`
- `docs/specs/kitchen-random-task-e2e-cloud-handoff-2026-07-10.md`
- `docs/specs/isaac-startup-reliability-cloud-handoff-2026-07-10.md`
- `docs/PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md`
- `docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json`

### Critical implementation

- `src/blueprint_pipeline/g1_kitchen_digitalocean_closure.py`
- `src/blueprint_pipeline/g1_kitchen_attempt_closure.py`
- `src/blueprint_pipeline/g1_kitchen_bundle_compatibility.py`
- `src/blueprint_pipeline/g1_kitchen_worker_image_evidence.py`
- `src/blueprint_pipeline/isaac_runtime_task_backend.py`
- `src/blueprint_pipeline/isaac_persistent_task_executor_service.py`
- `src/blueprint_pipeline/gear_sonic_official_zmq_executor.py`
- `src/blueprint_pipeline/gear_sonic_controller_fk_adapter.py`
- `src/blueprint_pipeline/isaac_review_media.py`
- `src/blueprint_pipeline/g1_kitchen_semantic_review.py`
- `src/blueprint_pipeline/wam_action_consistency_contract.py`
- `src/blueprint_pipeline/wam_strict_action_consistency_scorer_client.py`
- `src/blueprint_pipeline/groot_oscar_digitalocean_closed_loop_job.py`
- `scripts/prepare_strict_g1_kitchen_bundle.py`
- `scripts/build_push_isaac_worker_image.sh`
- `scripts/build_push_groot_oscar_closed_loop_image.sh`

### Tests that must be strengthened

- `tests/test_g1_kitchen_digitalocean_closure.py`
- `tests/test_g1_kitchen_attempt_closure.py`
- `tests/test_g1_kitchen_bundle_compatibility.py`
- `tests/test_g1_kitchen_worker_image_evidence.py`
- `tests/test_isaac_persistent_task_executor.py`
- `tests/test_gear_sonic_official_zmq_executor.py`
- `tests/test_gear_sonic_controller_fk_adapter.py`
- `tests/test_isaac_review_media.py`
- `tests/test_wam_action_consistency_contract.py`
- `tests/test_release_quality_governance.py`

## Required verification order

```bash
git fetch origin --prune
git status --short --branch
git rev-parse HEAD origin/main
git rev-list --left-right --count HEAD...origin/main

python scripts/verify_source_governance.py
python -m ruff check src/blueprint_pipeline scripts tests

python -m pytest -q -o addopts='' \
  tests/test_g1_kitchen_digitalocean_closure.py \
  tests/test_g1_kitchen_attempt_closure.py \
  tests/test_g1_kitchen_bundle_compatibility.py \
  tests/test_g1_kitchen_worker_image_evidence.py \
  tests/test_isaac_persistent_task_executor.py \
  tests/test_gear_sonic_official_zmq_executor.py \
  tests/test_gear_sonic_controller_fk_adapter.py \
  tests/test_isaac_review_media.py \
  tests/test_wam_action_consistency_contract.py \
  tests/test_release_quality_governance.py

bash scripts/pytest_full.sh
```

Then require hosted CI and Full Test Lane on the same commit. Image build,
registry inspection, provider canaries, and a live task follow only after the
code/CI gates and explicit operator authorizations pass.

## Deliverables expected from Fable

1. One patch series closing FABLE-001 through FABLE-007 with negative regression
   tests first.
2. A compatibility/readiness report proving whether the current sealed build can
   proceed without paid provider allocation.
3. A current clean-source immutable image manifest and registry digest, or an
   exact named external blocker.
4. Same-allocation fast/review canary artifacts with teardown and zero inventory.
5. One attempt-bound microwave episode closure, whether passed or truthfully
   blocked, containing all leaf evidence and no inferred proof upgrades.
6. A separate scorer/calibration plan and a separate SC3 fidelity-study plan.
7. Updated changelog/remediation/handoff documents that remove stale
   “uncommitted” state and preserve the sim-only claim boundary.

## Stop rules

- Never run paid compute without explicit authorization and a total spend cap.
- Never weaken a fail-closed gate merely to obtain green output.
- Never treat local tests, image build, provider marker, renderer completion,
  generated video, or semantic labels as task success.
- Never treat task success as rank-fidelity, physical readiness, safety, or
  deployment approval.
- Never silently reroll the selected microwave task.
- Never delete an SSH known-host entry until the replacement fingerprint is
  independently verified.
- Never expose or commit raw secret values.
