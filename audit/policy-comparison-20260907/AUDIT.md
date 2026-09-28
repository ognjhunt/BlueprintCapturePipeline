# Offline policy-comparison integrity audit — 2026-09-07

This audit reproduced 22 failing adversarial cases across seven producer-boundary findings and implemented bounded local fixes. It also retained passing controls for existing guards. The result is a reviewable code/test deliverable, **not a qualified matrix, policy ranking, current GPU observation check, or physical proof**.

## Scope, authority and identity

- Supports ADP-003/004/005 and ADP-009D's public-scene day-28 development rehearsal. Completion artifacts are this report, the invariant tests, before/after logs, local commits and consumer handoff packet. Existing lifecycle tests proved orchestration but did not exercise the reproduced semantic inconsistencies. The smallest changes strengthen existing readers, reset checks and scorer provenance; no new runtime or campaign was introduced.
- Worktree: `/private/tmp/bcp-policy-comparison-audit-20260907-e524`; branch: `codex/policy-comparison-audit-20260907-e524`.
- Starting fetched main: `c217cf39659d5c8c4988dd463d02b7450b5c6345`. Primary Pipeline stayed at `b7c0249b5cfbe48bd14a00b6b4ca6e2169a50eba`, clean at admission. Final refresh advanced main through `da3ff3ba2` / #1764; none of this patch's six production files changed in that interval. The audit was not rebased onto unrelated restart work.
- Website primary: `c2cef1a6f464847d9cc94a9a2a7d5c82ccb85882`, clean, read-only. Fetched Website main: `5b07fe48e93cabf2a758b7740aab3d6594cbdd10`; the current consumer helper was read with `git show`, because the older primary lacks that file.
- Interpreter: `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/.venv/bin/python`, resolved `/opt/homebrew/Caskroom/miniforge/base/bin/python3.12`, Python 3.12.11. pytest 9.1.1, numpy 2.3.1, Pillow 12.3.0, jsonschema 4.26.0. Checkout-local `src` and root were first on PYTHONPATH; dependencies were reused, not installed or downloaded. Exact identity and lock hash: `evidence/identity.json`.
- Primary status, worktree inventory and open PRs were inspected. Relevant existing persistent-pair and rescore worktrees were clean; no other checkout was edited. Existing owner-binding code was reused. The execution owner retains all deploy, launch, resource, queue and lock ownership. Handoffs are local files, not sent notifications.
- Read current root/nested AGENTS, north-star/backlog, exact-workcell and lifecycle doctrine, product/strategy context, evidence ladder and historical reference. The dated current owner instruction selects SAM3.1/FlashSplat/artifixer3D+ and fresh Raw V3.2. Older Aura/InFusion/ScanNet passages in the north-star, README, ladder and shared doctrine are a recorded documentation conflict, not authority to restart retired methods. No doctrine rewrite was attempted here.

## Authoritative source-to-consumer map

Source paths in this report are relative to the isolated worktree. These are distinct boundaries, not interchangeable hashes.

| Boundary | Real source / authoritative identity | Assertion versus independent observation |
| --- | --- | --- |
| Owner's pair → setup/specs | `task_evaluation_scene_policy_binding.py`: `candidate_map`, `validate_setup_pair`, `validate_execution_specs`, `execution_setup_binding_blockers` | Owner checkpoint inventory digest is compared with setup checkpoint, execution checkpoint and runtime inventory. A model name is not checkpoint identity. |
| Policy-neutral primary matrix | `exact_workcell_variation_inputs.py`, `exact_workcell_variation_matrix.py`, `exact_workcell_variation_runtime.py` | Admitted source dimension/unit/range/measurement authority, canonical object/anchor, seed, partition, condition/reset/matrix digests. Compilation is preparation, not applied physics. |
| Schedule | `compile_evaluation_schedule`, `validate_matrix_and_schedule` | Exactly two frozen identity digests plus both controls on every cell, fixed retries and power/design digests. The default is 100 cells/400 episodes; the ten-cell canary is a different typed plan. |
| Canary dispatch → bundle | `task_evaluation_policy_canary_dispatcher.py` → `validate_runtime_input_manifest` → `execute_paired_session` | Runtime inputs bind plan/configuration/activation/task contract and each resolved scenario. Candidate IDs are validated as a pair. Selected-cell contexts carry the frozen cell/seed and scenario digest. |
| Load identity | `native_task_arena_policy_canary_worker.py:_run_selected_cell` → `native_task_arena_policy_worker.py:_policy_client` | Actual reached clients are `OpenPIWebsocketDroidPolicyClient` and `GrootN17DroidPolicyClient`. OpenPI verifies metadata on each fresh connection; GR00T binds its worker receipt and frozen modality interface, then pings/resets. Offline fakes test protocol behavior, not remote weights. |
| Resolved scene | worker `_resolved_scene_plan` → native environment builder | Object start/yaw and camera/light perturbations enter the copied plan. The canonical plan is not edited. Friction/material-cousin gaps remain explicitly labeled with canonical-material fallback in this diagnostic route. |
| Actual reset | worker `env.reset(seed=cell.seed)` → `build_native_task_episode_environment` → `IsaacEpisodeAdapter.reset` | Factory binds `plan.scenario.seed`; subsequent readiness and episode resets use that seed and reset servo state. The `reset_state` artifact is an environment binding receipt, not a complete independently measured reset. New readiness retains and compares measured task samples and arm joints. |
| Observation | `IsaacEpisodeAdapter.read_policy_inputs` → `adp009d_droid_observation` → `run_policy_episode` | Actual RGB arrays are resized with aspect-preserving centered padding, no crop. pi05 gets 224×224 external/wrist; GR00T gets 180×320 external/wrist. Overview is review-only. Measured source dimensions, ordered composites and raw camera bindings are retained. |
| Transport | OpenPI `infer` / GR00T `infer` → `groot_n17_wire_client` | GR00T adds batch/time dimensions, converts numeric state to float32 and assembles language. A real codec roundtrip test compares final pixel arrays with decoded retained PNG bytes. OpenPI passes the supplied observation to its injected vendor transport; its vendor serializer/real server are not newly qualified here. |
| Action | query tracker → `adp009d_droid_action_execution` → episode adapter/servo | Vendor response, extracted chunk, executable rows, delivery readback and observed joint/task state are separate evidence. Unsafe returned actions are refused before delivery; no policy success assertion controls the score. |
| Deterministic score | `adp_task_scoring.score_task_episode_from_spec` → `adp_rigid_task_scoring.score_rigid_task_episode` → `adp_rigid_retreat_scoring.score_retreat` | Native pose/contact/safety/gripper samples determine lift, containment, support, release, settle and retreat. Scoring-frame transforms and live destination pose matter. |
| Media and receipt | `episode_visual_evidence`, `policy_episode_lifecycle`, `adp_episode_evidence_index` | Writers seal PNG bytes, raw RGB hashes, frame/observation manifests, genuine terminal observation and derived videos. Independent readers rehash those bytes; new checks enforce semantic consistency as well. |
| Child → final session | `_aggregate_isolated_cell_results` → `validate_session_result` | Child seals and resolved fields are now compared against frozen inputs. The reader enforces one checkpoint/runtime identity per candidate and matching resolved science across paired rows. |
| Correction | `task_evaluation_policy_canary_rescore.py` | Original result, score receipts and inventories remain immutable; corrections are new digest-bound overlays. Source identity now includes the split rigid and retreat scorers. |
| Website delivery | `task_evaluation_result_delivery.py` → `task_evaluation_policy_canary_result_projection.py` | Artifact bytes/paths are bound; per-candidate marginal metrics are produced. The canary forbids an official winner/ranking. Website's separate paired display is covered by the handoff below. |
| Prospective decision | `adp_prospective_design.compile_decision` | Uses its frozen schedule and invalid-trial rule: all scheduled trials remain in the denominator. It is not the Website mutually-scorable sign-test cohort and is not invoked by this canary to declare a winner. |

## Reproduced findings and bounded fixes

Each regression uses synthetic state or fake clients. Before logs are retained; an expected-failure regression is counted as reproduced only when the actual function accepted the forbidden input or crossed the wrong boundary.

| ID | Failing invariant, minimal input and observed baseline | Expected result / patch / layering |
| --- | --- | --- |
| F1 | `validate_session_result` accepted a resealed 20-row result with one candidate's second checkpoint or runtime digest changed. It also accepted a changed resolved scenario, with or without updating that scenario's digest, while cell/seed labels matched. | Refuse identity/scenario drift. Reader now rehashes retained scenarios and compares per-candidate checkpoint/runtime tuples and per-cell scientific bindings. Legacy results that omit resolved scenarios entirely remain readable; those results do not thereby prove reset parity. `regressions-before.log`: four identity/scenario failures. |
| F2 | `_aggregate_isolated_cell_results` accepted two candidates agreeing on the same wrong resolved scenario, and accepted a child whose result digest was invalid. Matching pair labels did not bind the pair back to frozen inputs. | Validate child seal and each cell's seed/spec/family/resolved values against the input manifest. Sort children by their retained selected index and rows by candidate, preserving their actual artifact directory binding. Count/membership checks remain. `aggregation-before.log`: two failures. |
| F3 | In the real episode runner, a fake reset restored arm joints but moved the object by 10 cm. Readiness's second reset was accepted. If only the third reset drifted, policy queries occurred before the later scorer refused the start pose. | Compare selected measured task reset fields, including pose, gripper and contact/safety fields when present, using only frozen pose tolerances. Retain initial/restored task samples with their digests. Validate the final reset before `episode_started`. Existing scorer already caught the third-reset wrong pose later: this was a pre-query boundary gap, not an undetected successful score. `reset-before.log`: two failures. |
| F4 | Delete one native sample at step 1, 2, 3 or 4 from a six-step analytic lift/translate/place trajectory. The scorer accepted increasing-but-gapped indices and returned an outcome instead of refusing incomplete evidence. Missing intermediate safety observations cannot prove an uninterrupted trajectory. | Require nonnegative consecutive sample indices. Complete analytic trace remains successful. Existing step-72 collision fixture was made genuinely continuous while preserving the exact step-72 event/time/force expectation. `regressions-before.log`: four sparse-trace failures. Historical receipts are not rewritten; applying a new scorer may now refuse sparse old traces. |
| F5 | Media reader accepted independently resealed wrong width, per-frame timestamp/kind, count or terminal kind. A JPEG with updated file/raw hashes also passed the purported retained-PNG reader. | Verify actual PNG format/RGB/dimensions/byte size, observation/frame index/time/kind joins, counts, terminal placement and calibration dimensions. Existing same-length byte corruption/swapped files/digest failures already refused. `media-before.log`: six failures and six passing controls. |
| F6 | A tiny local Git fixture changed only `adp_rigid_task_scoring.py` or `adp_rigid_retreat_scoring.py`. `resolve_scorer_identity` still returned a clean scorer identity because those files were omitted from its source list. | Include both reached scoring modules in clean-source checks and source hashes. Dirty source now refuses; a new local commit changes the source-files digest. Existing immutable correction overlays remain intact. `rescore-before.log`: two failures. |
| F7 | With actual nested score `undetermined`/null or `not_scored`, `execute_paired_session` marked completed rows interpretable solely from query/delivery/motion and grader labels. | New nested episodes require a scored boolean deterministic outcome for interpretation. Completion remains `completed_unqualified`; no qualified-controls gate was imposed. Legacy digest-only callback summaries retain their existing contract. `interpretability-before.log`: two failures. |

The six production files changed are the session, worker, episode runner, rigid scorer, media reader and rescore identity module. Tests exercise real functions rather than merely assert the added lines exist. No schemas, frozen task thresholds, published profiles or historical outputs were changed.

## Candidate / cell / seed / reset / media / scorer parity matrix

“Passed offline” means a deterministic local test or stated static binding; it never means the GPU emitted correct images. “Runtime-unverified” names the exact missing independent observation.

| Dimension / adversarial case | Evidence / result | Boundary or remaining limit |
| --- | --- | --- |
| Exactly two candidates, duplicates/extra IDs | Existing session + owner-binding tests; new membership mutations | Passed offline; no new permanent candidate restriction. |
| Checkpoint/runtime changed across cells | F1 + F6 | Refused offline. Uniform substitution relative to an owner requires the separate setup/spec binding; pair consistency alone is insufficient. |
| Reordered candidates/rows | `test_episode_reordering_preserves_pair_acceptance` | Accepted as the same pair; digest naturally binds serialization order. No ranking is created. |
| Same cell ID, substituted seed | new membership test + runtime manifest validation | Refused. |
| Same labels, altered resolved parameters | F1/F2 | Refused, including agreement by both candidates on wrong parameters. |
| Missing/duplicate cells; partial matrix | new membership tests; strict controls tests | Full closeout refuses malformed membership; typed strict blocked partial results retain failures and unexecuted counts. |
| Anchor, partitions, pairwise/held-out composition | exact-workcell compilation, tamper and power tests | Deterministic preparation proved; create-only publication refuses changing sealed bundles. No 10→100-cell equivalence claim. |
| Object cousins | primary matrix rejection tests; scenario materialization tests | Primary object identity is fixed. Diagnostic material-cousin fallback is a declared coverage gap, not cousin qualification. |
| Actual seeded simulator reset | native factory reset callback and real-worker fake-Isaac rehearsal | Seed plumbing exists. Full physical state/reset parity on GPU remains unverified. |
| Stale task after readiness/final reset | F3; reset/contact wrapper tests | Refused before query for the compared measured fields. Complete camera/physics/collider readback is not supplied by this new helper. |
| Reset artifact equality | source inspection + new measured samples | Entire artifact hashes are deliberately not compared across candidates. Metadata can legitimately differ; exact samples and byte bindings are retained. |
| Warm candidate state/action horizon | real client preflight/reset tests; 10 isolated-cell lifecycle | Separate candidate clients, GR00T reset, fresh verified OpenPI connections and fixed horizons tested without weights. Remote server internals remain runtime-unverified. |
| Camera selection, shape, padding, channels | DROID observation suite; nonuniform GR00T wire probe | External/wrist only; overview excluded; candidate-specific frozen shapes preserved. No rollout-wide pixel-equality assertion. |
| Policy-visible post-adaptation frames | lossless composite and real GR00T codec equality test | Exact final GR00T image bytes matched. OpenPI injected transport inputs are tested; vendor transport/runtime qualification is separate. |
| Prompt and numeric state | real client tests and source trace | Prompt/state assembly and finite/interface guards inspected. Image manifests do not alone prove every serialized proprioceptive byte or remote preprocessing state. |
| Missing/corrupt/same-length/swapped frame bytes | new media mutation controls | Refused by independent rehashing. |
| JPEG/video-only substitute | F5; existing complete-media/lifecycle tests | JPEG refuses; review video cannot replace authoritative frames. |
| Incorrect manifest hashes/count/dimensions | F5 + existing evidence index | Refused. |
| Camera name/order/time binding | frame-camera IDs/digests plus F5 | Duplicate/renamed source mappings cannot establish a second camera. Real sensor freshness cannot be inferred merely from equal/unequal pixels or asserted timestamps. |
| Terminal observation | F5 and lifecycle early-terminal/complete-media tests | Missing/misclassified terminal evidence refuses. Genuine failure before observation preserves a typed gap. |
| After-observation failure/query/action truth | real lifecycle action rejection and failed-cell tests | Actual queried/returned/delivered facts retained; no invented observation or robot motion. |
| Lift/translation/release thresholds | hand-computed six-step trajectory, exact and +0.1 mm thresholds | Inclusive boundaries pass; just-outside thresholds fail for both candidates' common scorer. |
| Destination containment/frames/units | live rotated destination, missing pose, native scoring-frame transform tests | Existing deterministic frame/pose guards pass offline. No real metrology claim. |
| Settle/cadence/missing samples | F4; episode cadence refusal; original scorer tests | Missing samples refuse, task cadence mismatch refuses, complete settle windows score. |
| Contact/gripper/collision/timeout/retreat | rigid and retreat suites, analytic traces | Native safety/readback omissions abstain; step-72 force event remains exact. Scene/world/local retreat axis and oriented bounds covered by existing tests. |
| Nonfinite values and incomplete readback | new pose/gripper faults; existing rigid/retreat tests | No success from NaN/Inf or missing required native channels. |
| Policy/human/learned override | prospective grader tests; immutable interpretation sidecar tests | Policy self-grading rejected; learned sidecar cannot change score/ranking/qualification. |
| Qualified per-cell controls | exact schedule and strict control-gate tests | Every scored cell requires negative/positive controls under that contract. Canonical-only controls do not qualify a varied matrix. |
| Diagnostic controls gaps | session tests + F7 | Explicit diagnostic gaps remain readable; ranking/qualification remain false. |
| Attempted/completed/interpretable/scorable | F7 + prospective missing-trial test + consumer packet | Kept distinct. Delivery marginal rates and mutually-scorable paired display must not be conflated. |
| Asymmetric missingness/no shared valid pairs | consumer packet; empty/partial prospective decision tests | Marginals can disagree with paired direction. Prospective fixed-denominator design retains missing trials; the canary declares no official winner. |
| Duplicated receipts/subset results/order | membership tests; prospective duplicate/partial/reversal tests | Duplicate trials refuse; missing scheduled trials remain non-success under the preregistered rule; order does not change decision. |
| Late corrections/sealed thresholds/holdout access | rescore overlay + sidecar mutation tests; schedule/matrix tamper tests | Original receipts retained; revised scoring has a new source-bound identity. Software preparation/sealing checks do not prove live holdout-access discipline. |

## Consumer handoff and remaining integration boundaries

`evidence/consumer-handoff.json` is produced by the real Pipeline delivery builder from **synthetic** two-candidate rows and test-owned artifact/closure placeholders. It is not valid media or historical execution evidence. Reproduce with `consumer_probe.py` under the network-denied test environment.

- Ten matched labels, six interpretable rows per candidate. Marginal success: pi05 `2/6`; GR00T `4/6`.
- Only cells 0 and 1 are mutually scorable. pi05 wins both; GR00T wins neither; two-sided exact sign-test p = 0.5.
- Refreshed Website main still derives `leader`/`deltaPoints` from marginal rates while reporting discordant paired counts. This is the previously owned Website mismatch, independently traced to the current source, not a second TypeScript patch. See `website-consumer-snapshot.ts:565` and prior audit's exact synthetic result. The local packet is ready for the Website owner; nothing was sent.

Additional explicit limits:

1. The worker's `reset_state` artifact remains primarily adapter/configuration evidence. Native task/arm restoration is improved, but there is no new all-field cross-candidate independently measured camera/light/friction/mass/collider/contact-reset certificate. Construction's `read_native_task_arena_scenario_parameters` is a separate readback surface, not called by this policy worker. Do not use this audit as that runtime proof.
2. The primary exact-workcell compiler/schedule and diagnostic canary dispatch are distinct routes. Existing `adp009d_franka_evaluation_harness` materializes scenarios; its existence is not evidence that this run executed a qualified full matrix or produced all family/degradation results.
3. The generic session accepts registry-shaped candidate IDs, while the reached native client/projection route retains its current pi05/GR00T compatibility assumptions. This audit did not select candidates or redesign registry dispatch.
4. F3 validates restoration against retained initial samples, not all initial state against owner truth. The deterministic scorer still validates the configured object reset pose. Uniformly wrong camera/physics state and server-side preprocessing/cache behavior require measured runtime evidence before a qualified claim.
5. Published legacy summaries are not automatically reinterpreted by these changes. Rescoring is a separately invoked, immutable overlay path; this session invoked it only over synthetic tests.

## Verification and historical evidence

The baseline focused set passed **261 tests in 29.77 s**. That result did not prevent the 22 newly reproduced failures. Subsequent logs include expected regression failures and two intermediate fixture/error-taxonomy adjustments; they are preserved rather than erased.

Implementation commit: `37eabbf04f9c9e789a4cfc5b470ce93a2bb1b421`. The committed implementation passed all **43 new regressions in 4.55 s** via `VERIFY.sh regressions` under OS network denial. The first verification-script invocation exposed macOS Bash 3 empty-array handling; that audit-only script was fixed and rerun successfully.

Final relevant runs:

| Log | Result | Claim protected |
| --- | --- | --- |
| `final-focused.log` | 439 passed, 69.77 s | Matrix, harness, reset/episode, media/index, scoring/retreat, controls, owner bindings, corrections, sidecars and new regressions before the final two interpretation cases. |
| `final-session-lifecycle.log` | 62 passed, 78.95 s | Final session interpretation change, all integrity regressions, real-worker lifecycle and provider import closure. Includes both mandatory rehearsal files. |
| `sandbox-clients-regressions.log` | 90 passed, 1 deliberately deselected, 5.68 s | Final new regressions and actual OpenPI/GR00T wrappers under OS network denial. The omitted case intentionally opens a loopback server; fake-client/wire-codec tests cover this audit without any server. |
| `lint.log` | All checks passed | Changed Python source/test syntax and lint. |

Counts overlap and must not be summed as unique tests. `VERIFY.sh` gives exact named-file command groups and the interpreter/PYTHONPATH/sandbox setup. No bare pytest, repository full suite, GPU suite, hosted inference, model download or real policy query was run. Earlier focused runs used injected offline seams plus a Python socket tripwire; final client/regression verification additionally used macOS `sandbox-exec` network denial. Initial baseline relied on the existing hermetic fixtures.

The supplied prior audit reopened Scene 839873's 20 learned rollouts, `completed_unqualified` / `internal_policy_canary` / `diagnostic_policy_execution`, with historical billing and provider-zero receipts. This audit uses that report as historical evidence only. It did not reopen the host, rehash those media, fetch an offloaded archive, refresh provider inventory or claim a current qualified result. Physical evidence is outside this offline audit and is not a blocker to completing it.
