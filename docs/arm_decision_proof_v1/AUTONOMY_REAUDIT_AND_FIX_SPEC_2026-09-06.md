# Second independent autonomy audit and remaining fix specifications

**Verdict: not ready for a live hands-off run.** Several previous defects are
fixed, but the statement that all eleven findings are closed is incorrect. The
latest activation correction fails through the actual production CLI, the terminal
index has no production caller, and the revised preflight rejects its own installer's
valid output. Other mode-transition and result-closeout defects remain.

Scope: ADP-009D/day-28, development-only two-policy fixed-arm rehearsal. The goal
is one persistent owner intent traversing preparation, construction, controls,
policy, publication and resource closure without hand-authored intermediate
evidence. This audit makes no physical or qualified-evaluation claim. No production
configuration, paid holds, authority or budgets were changed; no resources were
allocated and no notifications were sent.

Existing components are substantial. The missing work is consumer wiring,
validated state transitions and durable evidence joining. Use the existing
installers, resolvers, queues and validators rather than adding another orchestration
layer. Each specification below names the failing boundary and its acceptance proof.

## Exact baseline and host observations

- Audited Pipeline main: `50df5e4cd8d0ab8565131dad829b335d5c5265aa` (#1736).
  #1729 and #1731–#1736 are merged.
- Controls sub-audit used `357b00aa775bc30a59563cad1df3ff1b86b33007`; its tree
  is identical to the audited main (verified with `git diff`).
- Website readback/forwarder audit: `cdcbee08`.
- During the read-only audit, production moved from `5976c269a` to `50df5e4cd`.
  `deploy-receipts/iteration_50df5e4c.json` reports `deployed`. The scene service
  subsequently wrote an **idle** receipt for `50df5e4cd`. This proves startup in
  unarmed mode, not successful processing of a scene or the authorized branch.
- Live scene config contains `preparation_worker`, public_scene support,
  `activation_enabled=false` and `terminal_result_root`. Owner intent count is zero.
- Controls bootstrap and controls config are still missing. No
  `BLUEPRINT_SCENE_PROJECT_SPEND_CONFIG` is configured in progression, controls
  or capacity; the new `scene-project-spend/current.json` pointer does not exist.
- Final preflight readback: 21:51:36 UTC, **50df5e4cd**, seven blockers.
  Its findings are missing
  current activation intent, missing controls env, unset controls config,
  controls-registry permission denied, launch hold, inactive launch trigger and
  policy hold. The last three are intentional.
- Billing export is fresh (21:44:21 UTC snapshot). Launch execute=false and
  force-dry-run=true remain effective; launch-dispatcher.path remains inactive;
  policy dispatcher retains `ExecCondition=/usr/bin/false`. Scene progression
  timer is active. No paid run was attempted by this audit.

## Disposition of the previous eleven findings

| Previous finding | Independent result |
|---|---|
| A1 conversion re-decode | Fixed in the reproduced production-field, retained-authority case; real progression reaches publication_ready without decoder execution. |
| A2 prefix-reuse inputs | Four machinery roots added; fresh provider-zero input is still missing before prefix adoption. |
| A3 activation producer | Not closed. Actual CLI rejects authorized config; fresh uploads and previously prepared links also fail. |
| A4 owned queue selection | Original selector issue fixed. New activation-registry root mismatch still prevents consumer discovery. |
| A5 consent rewriting | Fixed: mismatched terms refuse and no intent is staged. |
| A6 terminal producer | Not closed. New helper/CLI exists but no production code calls it or persists all its required source files. |
| A7 completion integrity | Execution commit and publication schema/run/seal checks fixed; real index still certifies an unpublished URI. |
| A8 late closeout | Pipeline ordering, historical commits and failed notifications improved; Website polling and terminal-failure visibility still fail. |
| A9 preflight validation | New regression: valid installed content catalogs are rejected as unsealed. |
| A10 supported candidates | Unsupported IDs rejected early; exact supported checkpoint capability is still not checked at intake. |
| A11 dependency publication | Fixed: dependency conflict refuses before publishing owner intent. |

## R1 — Repair the actual production activation entrypoint

Priority: blocking. Source: `task_evaluation_scene_progression.py:550–558`,
`task_evaluation_scene_preparation_service.py:15–16`,
`task_evaluation_scene_preparation_installation.py:209`.

The service invokes the progression module, whose `main()` reads the config and
calls `run_preparation_service` whenever `preparation_worker` exists. The installer
always supplies that field. With activation authorized it also supplies
`activation_enabled=true`; the wrapper requires false. Thus the claim that no
production unit reaches the wrapper is incorrect.

Genuine installer config → actual `main(["--config", path])` reproduces:

```
activation_enabled True preparation_worker_present True
production_main_error ValueError scene_progression_preparation_service_scope_invalid
```

The new test `test_preparation_service_wrapper_refuses_the_authorized_activation_config`
explicitly expects this refusal while its docstring incorrectly calls the wrapper
a test convenience. It never invokes the actual CLI dispatch.

Required fix: define a consistent typed preparation/activation execution path
through `main`, retaining the real owned preparation worker. Do not simply bypass
the wrapper and thereby abandon owned-queue processing. Acceptance: run the actual
CLI with genuinely installed unarmed and authorized configs; both process their
appropriate stages without allocation. Verify the effective systemd entrypoint
against that same command. Configuration-field assertions alone are insufficient.

## R2 — Separate preparation accounting from activation authorization

Priority: blocking for upload-and-go. Source:
`task_evaluation_scene_preparation_installation.py:157`,
`task_evaluation_scene_progression.py:389–406`, `task_evaluation_scene_intake.py:214`.

Completed mesh/splat machinery correctly declares zero preparation spend and
provider `control_plane`. Progression only uses its administrative preparation
attempt API while activation is false. When true, it instead calls the paid
reservation API with that zero-cost attempt, which refuses.

Genuine installed authorized mesh config → real `process_scene_intents` returns
`blocked / preparation / scene_intake_attempt_spend_invalid`. The same issue occurs
when authorizing a previously prepared upload. This is independent of R1: invoking
the processor directly still fails.

Required fix: select administrative versus paid preparation by the source's actual
accounting contract, independently of later activation permission. Reserve actual
construction exposure separately. Acceptance: fresh mesh and splat with authorization
already present, plus unarmed preparation followed by authorization, preserve zero
preparation spend and create exactly the required construction reservation once.

## R3 — Implement an idempotent prepared-to-authorized link transition

Priority: blocking. Source: `task_evaluation_scene_progression.py:187–192,232,459`.

An unarmed preparation link omits `scene_configuration_attempt`. Progression reuses
an existing link after mode changes, while `_activation` unconditionally reads that
field. Actual `_link` followed by `_activation` on the retained link reproduces
`KeyError 'scene_configuration_attempt'`.

Required fix: preserve the original immutable preparation link; under valid current
consent reserve the actual configuration attempt and create a versioned, digest-bound
activation link. Do not edit evidence in place or rerun completed preparation.
Acceptance: prepare → restart → authorize → activate → restart is idempotent,
creates one reservation, and refuses expiry/revocation/overspend. Include both
uploaded and public-scene inputs, not only a fresh authorized fixture.

## R4 — Give activation producer and consumer the same registry

Priority: blocking. Source: `task_evaluation_scene_preparation_installation.py:204,216`,
`deploy/systemd/blueprint-task-evaluation-configured-controls-progression.service:60,96`.

The installer writes new activation intents under
`<inputs>/scene-configuration-activation-intents`. The configured-controls unit
explicitly scans `/etc/blueprint/task-evaluation-scene-configuration-activation-intents`.
The emitted environment does not override the consumer's root. The original A4
owned preparation queue override is fixed, but does not fix this different registry.

Required fix: use one canonical configured registry identity through installer,
producer and unit arguments, with service-readable/writable permissions as required.
Acceptance: generate a real activation intent from installed config, invoke the
consumer with effective unit arguments, and observe exactly one correctly bound
selection. Missing controls bootstrap/config and registry permissions must also be
resolved by the canonical installer before host readiness is claimed.

## R5 — Supply fresh prefix-reconciliation evidence before factory admission

Priority: blocking when a reusable completed prefix exists. Source:
`task_evaluation_public_scene_attempt_factory.py:403–407`,
`task_evaluation_scene_release_binding.py:74`.

The added machinery roots repair the former missing-key defect. However the factory
still requires `release.provider_zero` before adopting a discovered prefix; the
automatic release resolver never emits it and progression has no refresh path.
Reproduction substitutes only discovery, then real progression refuses with
`scene_configuration_submission_public_factory_prefix_reconciliation_required`.
Producing zero at paid-launch time is too late for this earlier boundary.

Required fix: reopen a canonical fresh, scoped provider-zero/ownership receipt
at preparation time and snapshot the validated evidence into the immutable attempt.
Do not treat release metadata as a permanently fresh inventory observation.
Acceptance: real retained candidate discovery and adoption with fresh zero; stale
zero, pending teardown, ambiguous create and changed inputs refuse. Completed
identical GPU stages are reused, with the original failure/evidence retained.

## R6 — Install the project-spend monitor consumed by activation

Priority: blocking host/configuration gap. Source:
`task_evaluation_scene_preparation_installation.py:211`,
`task_evaluation_scene_progression.py:212–225`,
`task_evaluation_scene_spend.py:130–144`.

The authorized config points to `<state>/scene-project-spend/current.json`.
`_activation` calls `refresh_configured_scene_project_spend`, which returns None
without `BLUEPRINT_SCENE_PROJECT_SPEND_CONFIG`, then tries to read that pointer.
The host has neither the configured monitor nor this file. Fresh provider billing
is a different artifact and does not supply the required project-spend reconciliation.

Required fix: canonically provision the sealed monitor configuration, retained
official-source seed, owner reservation roots, output/current paths and effective
unit environment. Bind accepted budget authority; do not invent a seed or zero bill.
Acceptance: real publisher refreshes the exact pointer activation consumes; adding
a reservation changes conservative exposure; stale/missing/incomplete seed refuses.
The complete proposed run must fit its aggregate and per-provider bounds before
paid authorization, independently of the account's displayed credit balance.

## R7 — Make preflight consume the actual installed catalog schema

Priority: blocking readiness regression. Source:
`task_evaluation_production_chain_preflight.py:1172–1185` and
`task_evaluation_controls_autoprovision_installation.py`.

The installer intentionally retains `task_evaluation_controls_robot_content_catalog.v1`;
the real resolver accepts it and binds the active release. The new preflight insists
on resolved `task_evaluation_controls_robot_catalog.v1` instead. Genuine installer
output → real resolver succeeds, but preflight reports
`controls_autoprovision_robot_catalog_unsealed`.

Required fix: call the shared resolver with the active release instead of maintaining
a contradictory schema rule. Retain service-user byte/readability checks.
Acceptance: genuine installer → preflight passes; changed seal, asset bytes,
runtime bytes and invalid schemas independently refuse. Use a production installer
fixture rather than manually constructing the schema the test expects.

## R8 — Connect real terminal producers to the index

Priority: blocking. Source: `task_evaluation_scene_terminal_result_index.py`,
`task_evaluation_policy_canary_dispatcher.py:1139–1170`.

`index_terminal_owner_result` has no production callsite: only its CLI and tests
invoke it. The dispatcher still computes projection and authenticated sync in
memory rather than persisting the source files the new helper requires. Configuring
`terminal_result_root` installs a reader location, not a writer. The new round-trip
test constructs those source receipts using fixture helpers; it does not run their
actual producers.

Required fix: persist the actual producer outputs and invoke an idempotent index
transition from real delivery/reconciliation, with installer-owned roots and run/
owner bindings. Acceptance: actual dispatcher delivery and resource reconciliation
produce indexable evidence, then the real scene timer observes terminal status.
Restart between each write must recover; no operator-created intermediate files.

## R9 — Require real publication evidence before certifying a result URI

Priority: blocking completion-integrity defect. Source:
`task_evaluation_scene_terminal_result_index.py:136–144`.

The index accepts caller-supplied URI and size and seals them alongside the projection
digest without reading a publisher receipt or verifying durable bytes. Real index
→ real reconciler reproduces `completed` for
`https://unpublished.invalid/never-published.json`, size 123456. New schema/run/seal
checks prevent casual tampering but do not establish that anything was published.

Required fix: consume canonical publication and authenticated readback evidence,
bound to exact run/projection identity and published bytes. Keep a canonical
projection digest distinct from the raw published-file digest if they differ.
Acceptance: missing publication, invented URI/size, wrong bytes and wrong run refuse;
genuine retained publication succeeds. This is an integrity defect, not a demonstrated
remote authentication bypass.

## R10 — Validate and atomically publish a complete terminal index

Priority: blocking recovery defect. Source:
`task_evaluation_scene_terminal_result_index.py:127–131`.

The index checks schemas and immediately writes immutable files, before the whole
set has passed consumer validation. A sync receipt with `status=skipped` is accepted
as `terminal_result_indexed`. Correcting the source to `succeeded` then returns
`terminal_result_index_immutable_conflict`, permanently stranding that directory.

Required fix: validate the complete cross-bound receipt set and publication first;
stage it privately, fsync, and publish one atomic immutable index/pointer. Invalid
or incomplete source evidence must not poison the final namespace. Acceptance:
incomplete→complete, skipped→succeeded, interruption at every write, duplicate
completion and conflicting run identities produce truthful, recoverable outcomes.

## R11 — Keep Website closeout active after execution authority ends

Priority: blocking for hands-off result delivery. Sources:
Website `server/utils/taskEvaluationSceneIntake.ts:445–450,498–514`, Pipeline
`task_evaluation_scene_intake.py:320–324`.

Pipeline now permits read-only reconciliation before expiry/revocation/pause and
after a deployment, which is a real fix. But Website's forwarder only polls selected
active states; once it records expired/revoked, it stops polling. A later valid
Pipeline completion therefore remains undiscovered automatically. Explicit revocation
has the same issue.

Separately, the Pipeline status adapter overwrites every non-completed status after
expiry/revocation. Reproduction records `blocked / provider_capacity_unavailable`
in the engine but exposes `expired / scene_intake_authority_expired` to Website.

Required fix: separate authority for future execution from status of an already
authorized attempt. Continue bounded read-only reconciliation until resource and
result closure; preserve terminal failure details. Acceptance: expire/revoke while
billing/readback is pending, then deliver success or failure through the actual
Website polling path without operator refresh and without permitting new spend.

## R12 — Add typed nonexecution failure closeout

Priority: blocking failure-path completeness. Source:
`task_evaluation_policy_canary_dispatcher.py:2109–2154`,
`task_evaluation_launch_reconciler.py:486–490`,
`task_evaluation_scene_terminal_reconciler.py:193–210,298–304`.

Preprovider failures produce `preprovider_blocked.json` and a different sync path,
not the policy projection the index requires. For these failures the existing launch
reconciler explicitly records that post-teardown provider-zero closure is not
applicable. The new terminal reconciler universally requires that closure shape.

Required fix: a typed adapter for proven nonexecution must retain actual boundary
evidence and truthful no-allocation status without fabricating episodes or teardown
receipts. Ambiguous creates still require real ownership/provider reconciliation.
Acceptance: pre-admission refusal, deterministic no-create, ambiguous create,
post-allocation failure and successful execution all reach distinct correct owner
and Website terminal outcomes; zero retry cap stays zero.

## R13 — Check supported checkpoint content before construction spend

Priority: capability-validation improvement; not a proven current-841757 blocker.
Source: `task_evaluation_scene_intake.py:115–119`.

The exact pair IDs are now restricted correctly, but arbitrary well-formed checkpoint
digests remain accepted for those IDs. The frozen downstream setup still has to
reject incompatible content later. Verify the actual 841757 pair against runnable
inventory before any construction spend. For upload-and-go, use a resolver/admission
that binds exact supported artifact digests early; never replace the owner's pair.

## Verification and reproduction evidence

Three independent audit lanes ran focused suites:

- Source/provisioner/factory/installer: 31 passed, 7 deselected, 8.67s; explicit
  installer slow coverage: 7 passed, 3.78s. Protects conversion/consent/dependency
  fixes and configuration construction, but misses the CLI activation branch.
- Controls/preflight/intake/progression: 74 passed, 6.79s. Protects existing helper
  contracts; genuine installer-to-consumer experiments reveal the failures above.
- Terminal reconciler/index: 24 passed, 3.20s. Confirms revised identity/notification
  behavior; real index experiments expose publication and atomicity gaps.

These counts include overlapping installer tests; they are not 136 distinct tests.
No broad suite or paid/GPU run was used as a substitute for boundary verification.

Scripts and retained synthetic outputs:
[audits/2026-09-06-r2](audits/2026-09-06-r2). Run from an isolated checkout of
`50df5e4cd` with its root and src on PYTHONPATH and pytest installed, for example:

```
PYTHONPATH="$PWD:$PWD/src" .venv/bin/python <audit-script-path>
```

They use committed fixtures and allocate no resources. Prefix reproduction stubs
only candidate discovery; exact live 841757 prefix inventory still needs a saved-job
replay. The corrected conversion reproduction uses production receipt fields and
retained disclosure authorities; the previous stale-fixture conversion reproduction
is superseded and is not evidence of a remaining A1 defect.

## Required readiness packet

Do not declare readiness from `publication_ready`, an empty queue, module imports,
a deployment exit code, or an unarmed worker's successful tick. All can coexist
with the failures above.

After fixes, rehearse one retained real input through the actual installed CLI,
unarmed preparation, authorized mode transition, prefix adoption, real activation
selector, controls provisioning, exact policy binding, and preprovider admission.
Use real producer outputs for both successful and failed terminal reconciliation
and authenticated Website polling; external model/GPU boundaries may be faked in
the hermetic rehearsal, but contract producers/consumers may not be replaced with
success constants or hand-authored substitute receipts.

Retain a current-release preflight, matching config/registry identities, fresh
project-spend and provider-zero evidence, exact two-candidate inventory, scenario/
media/scoring boundaries, and the proposed bounded paid action. Preserve all holds
until the user authorizes that action. A Quick-10 canary remains diagnostic; it is
not a full scored evaluation without the reusable scenario harness and controls.
