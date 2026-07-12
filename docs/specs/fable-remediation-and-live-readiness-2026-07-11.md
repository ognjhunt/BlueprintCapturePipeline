# FABLE remediation and live-run readiness — 2026-07-11

Source audit: `docs/specs/claude-fable-5-hardest-blockers-audit-2026-07-11.md`
Baseline: `df030e45a4f8046c08d507df645eea31d91ea32f` (clean `main` at start of work)
Work state at this report: published as PR #68 on
`codex/fable-001-007-closure`; closure is evaluated against the latest PR head,
not the original baseline or an earlier check run.

Honest status: `fable_001_through_007_code_and_local_gates_closed_live_episode_not_run`.
Nothing below claims a live episode, task success, rank fidelity, or physical
robot readiness.

## 1. What was fixed (FABLE-001..007, all test-first)

| Package | Core change | Key files |
|---|---|---|
| FABLE-001 | The sealed worker now emits the proof contract the host consumes: exact-byte Ed25519 leaves for startup, policy action sequence, controller/FK, task transitions, live geometry, strict scorer, judge, and terminal horizon. Closure reconstructs every verdict from those leaves against pinned per-role keys; it compares rather than injects worker identity, correlates the exact action sequence across roles, verifies the real collected manifest bytes, and exposes only verified digests to buyer readout. | `g1_kitchen_leaf_evidence.py`, `g1_kitchen_startup_proof.py`, `g1_kitchen_worker_proof_emission.py`, `g1_kitchen_proof_row_validation.py`, `g1_kitchen_digitalocean_closure.py` |
| FABLE-002 | A signed terminal leaf carries planned cap, executed count, terminal step, reason, task-completed flag, scenario count, stage/session, and exact action sequence. The host compares it with the immutable launch request; it never shrinks the horizon to arrived files. Required camera indices are exactly 0..N-1, and every renderer sidecar is bound to action SHA, stage, session, timestamps, attempt, and nonce. Semantic review is a separate pinned detached signature over the identical request/frame set. | `isaac_review_media.py`, `isaac_task_review_renderer.py`, `g1_kitchen_semantic_review.py`, `g1_kitchen_digitalocean_closure.py` |
| FABLE-003 | Explicit ordered canonical G1 DOF map (legs 6+6, waist 3, arms 7+7, hands 7+7; deliberate rev-alias handling; alias-collision/missing/duplicate/swap all block); initial-state artifact carries full inventory, resolved map, dims, and mapping digest, and passes `UNITREE_G1_SONIC_STATE_DIMS` | `g1_proprioception_map.py` (new), `isaac_runtime_task_backend.py` |
| FABLE-004 | The pinned WBC revision is the authority for the official positional ZMQ output; the executor no longer expects invented joint-name/digest fields the upstream controller does not emit. The reviewed MuJoCo order supplies the 29+7+7 mapping, corrected thumb/index/middle hand order, and digest; FK writes by joint name, the installed git revision must equal the pin, live Isaac requires all 43 DOFs, and real loopback ZMQ tests cover stale replies, timeout, and concurrency. | `gear_sonic_joint_order_contract.py`, `gear_sonic_official_zmq_executor.py`, `gear_sonic_controller_fk_adapter.py`, `gear_sonic_container_smoke.py` |
| FABLE-005 | Before capacity, the gate resolves the exact image digest live with registry tooling rather than trusting sibling JSON; it binds attempt/launch/registry/worker image identity and source commit/dirty hash across checkout, bundle, healthcheck metadata, and canaries. Worker canaries carry run/attempt/nonce. Every attempt artifact is re-hashed again after capacity and immediately before launch, closing the source-file TOCTOU window. | `g1_kitchen_pre_allocation_identity.py`, `groot_oscar_digitalocean_closed_loop_job.py`, `g1_kitchen_worker_image_evidence.py` |
| FABLE-006 | Signed episode baseline captured after settle and before action zero, bound to attempt/nonce/session/stage/prim/contract hash; every transition emits `episode_initial_value`, `step_before`, `step_after`, `step_delta`, `episode_delta`; relative criteria evaluate current minus episode initial in the backend, the executor service, and the host-side transition validator (`oscar_isaac_closed_loop_eval.py`); step-pair deltas are diagnostic only | `task_episode_baseline.py` (new), `isaac_runtime_task_backend.py`, `isaac_persistent_task_executor_service.py`, `oscar_isaac_closed_loop_eval.py` |
| FABLE-007 | Redirects are rejected before credentials can be forwarded, all relevant clients use one bounded host/scheme policy, and Hugging Face loads pass explicit pinned revision, `trust_remote_code=False`, and `local_files_only`. Bandit policy passes with no high, untriaged-medium, or orphaned finding. The Full Test Lane has zero skips. `main` is now protected for admins and requires PR review plus the strict CI/Full-Lane check set, linear history, resolved conversations, and no force push/deletion. A new GitHub-OIDC release workflow signs a SHA/release/run/artifact-hash subject and retention verifies it with `gh attestation verify`. | `safe_outbound_http.py`, `retrieval_index_stage.py`, `.github/workflows/release-signature-verification.yml`, `.github/workflows/release-evidence-retention.yml` |

Supporting fixes: cross-test `MUJOCO_GL` leak in
`tests/test_g1_site_3dgs_mujoco_preview.py`; `pyzmq` added to dev extras (and
`uv.lock`) so real ZMQ tests cannot skip green. The Full Test Lane explicitly
selects `groot-libero-cpu` for Torch and native LeRobot coverage; those heavy
model-worker dependencies are deliberately absent from the generic dev and
capture-orchestrator production images;
quality-ledger artifact digests rebound via
`scripts/rebind_quality_gap_ledger_digests.py` (statuses untouched: still 91
partial / 16 open / 0 closed). The original grandfathered source limits remain
unchanged; new logic was extracted into focused modules instead of raising caps.

## 2. Verification executed locally

- `python scripts/verify_source_governance.py` — passed (371 modules).
- `ruff check src/blueprint_pipeline scripts tests deploy` — clean.
- Focused FABLE regression sets — passed.
- Local Bandit under the exact CI invocation — `passed`, high=0, medium=65,
  with all medium findings reviewed and no orphaned triage.
- `scripts/pytest_full.sh -rs` — **4893 passed, 0 failed, 0 skipped** in 810s.
- Live GitHub branch protection readback — active on `main`, strict required
  checks, admin enforcement, one approving review, stale/last-push approval,
  conversation resolution, linear history, force-push/deletion disabled.

Hosted CI and the Full Test Lane run on PR #68. Per the audit, every required
check must pass on its latest head SHA before any image build or paid attempt;
green results from superseded heads do not count.

## 3. New runtime contracts operators must supply

| Input | Consumed by | Env/flag |
|---|---|---|
| Ed25519 attestation public-key pins (roles: startup, policy, task_transition, controller, scorer, semantic_review, geometry) | closure proof-row validation | The sealed worker publishes `runtime_ephemeral_trust.json` with base64 raw public keys, role mappings, fingerprints, and the full attempt identity. Closure discovers it from the collected archive and rejects cross-attempt reuse. An explicit `BLUEPRINT_G1_ATTESTATION_PUBLIC_KEY_PINS_FILE` or `attestation_pins_file=` remains supported for externally provisioned pins. |
| Live Docker registry access | pre-allocation identity gate | `docker buildx imagetools inspect <digest-ref>` is executed by the gate; caller-authored registry evidence files are ignored/removed |
| Worker-side leaf signing keys | worker emission of attested leaf artifacts | Generated allocation-locally as mode-0600 Ed25519 keys; private material remains under `/run/blueprint-secrets`, while only attempt-bound public pins are collected. |

## 4. Live-run readiness (FABLE-008/009): blocked, with exact blockers

Verified on this machine at 2026-07-11:

1. `hosted_same_sha_checks_required` — PR #68 is published and branch
   protection is active, but image build remains gated until every required
   hosted check passes on the latest PR head and the protected merge completes.
2. `sealed_image_stale` — configured sealed ref
   `blueprint-groot-oscar-eval@sha256:aa7a7727…` is the 2026-07-07 build; it
   contains neither `df030e45` nor this patch series. The local generic image
   (`sha256:c237ce2b…`, healthy) is identity-bound to `d1220f78 + dirty` and
   fails the new identity gate by design.
3. `build_storage_insufficient` — sealed build requires ~120 GiB free; this
   host has ~11 GiB available. Options: free ≥120 GiB locally, or use the
   remote amd64 builder after independently verifying its changed SSH host key
   (do not delete the known-hosts entry blindly), or build on a rented CPU/GPU
   box using the pinned Dockerfile.
4. `total_spend_cap_unset` — provider use (DO or RunPod) is authorized, but
   the stop rules require an explicit total spend cap and one-resource limit
   before any paid call (`--max-spend-usd` is mandatory in the paid lane).
5. `strict_scorer_service_unconfigured` / `semantic_review_command_unconfigured`
   — without them the forward/inverse-consistency and semantic-review rows
   block, so a paid episode today cannot close as `completed`.

Recommended order once an operator supplies the cap and keys: commit/PR →
hosted CI + Full Test Lane green → build + push sealed image from the clean
SHA (120 GiB builder), resolve to `@sha256:` and write registry evidence →
same-allocation fast + review canaries with teardown/zero-inventory proof →
one strict microwave episode against the frozen task
(`/root/Microwave017`, stance `[-1.229635, 1.471274, 0.84]`, yaw `3.141593`,
episode-relative `+0.35 rad`). Provider credentials (DigitalOcean + RunPod
API keys, Spaces, Docker Hub PAT, HF token) are present and mode-0600 on this
host; Vast remains unfunded.

## 5. Scorer/calibration plan (FABLE-010) — separate deliverable

1. Stand up the forward/inverse action-recovery service behind HTTPS with the
   existing strict request/result contracts
   (`wam_action_consistency_contract.py` validators are already fail-closed).
2. Responses must carry numeric recovered actions, per-dimension
   error/uncertainty, units, timing, termination chunk, controller/generated
   state hashes, evidence refs, calibration identity, and an Ed25519 signature
   over the payload with a non-replayed runtime ID (verify against the same
   pin registry as section 3).
3. Calibration set before first live use: held-out positives; action-agnostic
   visual motion; swapped actions; replayed motion; wrong
   dimension/unit/timing; threshold-edge cases. Fail-closed abstention below
   calibrated confidence.
4. Semantic review stays a separate service over the exact ordered frame set
   (request/response hashing already enforced); it must never decide
   articulation success, and strict consistency must never substitute for it.

## 6. SC3 fidelity study plan (FABLE-012) — separate lane

Runs only for a public rank-fidelity claim; it does not gate the sim-only
product lane. Minimum design is unchanged from the audit: ≥7 independent
policies/checkpoints, locked InD/OOD splits, matched replicates, raw per-cell
outputs with failures/abstentions, independent human labels with adjudication,
Pearson/Spearman/MMRV with hierarchical bootstrap CIs, coverage-vs-abstention
curves, frozen digests, and a report that separates paper results from
Blueprint measurements.
