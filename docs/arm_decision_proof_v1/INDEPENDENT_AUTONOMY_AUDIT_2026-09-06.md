# Independent audit of the seven-item autonomous chain

Verdict: **not ready for the live end-to-end run; not all seven items are
code-complete.** There are reproducible producer/consumer defects as well as
unfinished host configuration. Passing component tests does not establish the
claimed autonomous chain.

Scope: ADP-009D, day-28 development-only fixed-arm rehearsal, exactly two frozen
policies. This audit changed no production configuration, authority or holds,
sent no external notifications, and allocated no resources. Physical trials are
not a blocker for this simulator-only diagnostic claim.

The completion artifact is one owner intent traversing the real preparation,
construction, controls, policy, publication and resource-closure consumers without
hand-authored intermediate evidence. Existing components are substantial, but
their interfaces and production routing remain incomplete. The smallest fixes
are adapters, validated configuration and receipt reconciliation, not a new engine.

## Audited identities and observed host state

- Current main: `4e0a38f4a8c833c06719784f7584e57c559695e4` (#1728).
- Deployed: `5e9ecc3be0996766f0099fa1ee33051206f82a29` (#1727).
- Spec E: PR #1729, **OPEN**, head
  `1e95c27b7d230dd2b72d732bc3036f028ae17e1b`, checks green at readback.
  Review findings for E concern this candidate, not deployed main.
- Host readbacks around 19:34–19:40 UTC: public-scene support enabled, owner-only
  selection and shared owner-store environment installed in controls and both
  dispatchers. The scene service exits successfully but is idle. Owner intent
  count and all owned preparation queue counts are zero.
- Controls bootstrap, controls config and its environment file are absent.
  Configured-controls intent registry is root-owned mode 0750 and the service
  probe reports permission denied. The new installer can address this, but has
  not run with a valid bootstrap.
- Latest preflight is 19:20:57 UTC on the deployed SHA, with seven blockers:
  missing current activation intent, missing controls env, unset controls config,
  controls-registry permission denied, launch execution hold, inactive launch
  trigger, and policy execution hold. The last three are intentional. The old
  argparse probe crashes are gone.
- Billing export timestamp 19:33:12 UTC. Capacity reports approximately $20.60065
  Vast credit, no reservation of that balance, disk utilization 81.46%, and no
  delivered alert. These observations do not establish whole-chain funding.
- Launch dispatcher is execute=false / force-dry-run=true. Policy dispatcher
  has execute configuration but is stopped by its explicit ExecCondition hold.
  Do not describe every service's execute flag as false.

## Plain-English status of all seven items

| Item | Independent status | What remains |
|---|---|---|
| 1. Automatic per-scene setup | Partial implementation | Fix conversion reuse and prefix inputs; connect installed preparation to activation; provision real owner input. |
| 2. Upload-and-go authority | Authentication exists; authority handling has a defect | Preserve exact accepted terms, admit only supported candidate pairs, and prove a real authenticated submission/readback. |
| 3. Remove manual execution holds | Owner selection is installed; holds intentionally remain | Fix queue routing and authority production before reviewing bounded arming. Removing holds alone cannot connect the chain. |
| 4. Automatic controls | Real producer and installer exist; not operational | Install the validated catalog/config with service permissions and route owned results to the actual activation consumer. |
| 5. Recovery and terminal status | Incomplete, with correctness defects | Supply prefix/ownership inputs, connect real terminal producers, correct identity/publication checks and closeout after expiry/deploy. |
| 6. Capacity and funding | Core billing/credit safeguards functioning | Bind fresh project-spend inputs and full-chain limits; alert delivery remains optional operational work. |
| 7. Look-ahead | Argv/git fixes are real; readiness coverage incomplete | Validate the same seals/bytes/queues as production consumers and run actual service-user rehearsals. |

## Findings and implementation acceptance criteria

### A1 — Conversion reuse fix stops at provisioning

The provisioner accepts a conversion produced by another commit
(`task_evaluation_public_scene_intent_provisioner.py:219`), but
`task_evaluation_public_scene_attempt_factory.py:130–146` still invokes local
conversion on commit mismatch, before submission at line 349.

Reproduction: change only the retained conversion's commit, reseal it, provision,
then call real `process_scene_intents`. Provision succeeds; the consumer blocks
with `splat_render_runtime_root_invalid`. The test at
`tests/test_task_evaluation_public_scene_intent_provisioner.py:266` stops before
this consumer. This proves the claimed fix is incomplete; it does not by itself
prove the exact host decoder is absent.

Fix: apply the admitted content-identity rule in the real factory, preserving
source/output byte and semantic checks. If a conversion is genuinely required,
preflight its runtime before scheduling work. Acceptance must exercise factory
and preparation consumer across a compatible release change, with no redundant
conversion for unchanged admitted bytes and rejection of changed bytes.

### A2 — Completed-prefix producer and consumer shapes disagree

Factory lines 382–398 require fresh `release.provider_zero` and machinery keys
`child_queue_root`, `parent_queue_root`, `execution_root`, and
`release_retention_binding_root` when discovery returns a candidate. The new
provisioner (`:197`) omits those machinery fields; canonical release binding
(`task_evaluation_scene_release_binding.py:74`) omits provider-zero evidence.

Reproduction replaces discovery only; the remaining real path blocks with
`scene_configuration_submission_public_factory_prefix_reconciliation_required`.
Supplying zero alone leaves the missing machinery keys. Exact host prefix inventory
was not replayed in this audit, so this is a verified conditional path defect.

Fix: canonically supply execution/queue roots and a fresh immutable reconciliation
snapshot. Do not put a forever-fresh provider-zero assertion into a static release.
Rehearse real discovered completed stages through selection and adoption, then
prove changed inputs and unresolved ownership refuse. No completed GPU stage may
be repeated merely because a deploy changed the source commit.

### A3 — Installed preparation service has no usable activation transition

Installer line 163 sets activation false. The actual installed wrapper
`task_evaluation_scene_preparation_service.py:15–16` **requires false**. Progression
only provisions construction activation when true (`:461–467`); required activation
root and project-spend inputs are absent from the installed preparation config.
Changing the flag alone makes the wrapper refuse.

Fix: connect a separate authorized activation consumer or extend the service with
a typed, separately admitted activation mode. Preserve preparation-only behavior.
Acceptance: real installed-service output reaches an exact owner-bound activation
and launch request under no-spend rehearsal, while absent/expired consent refuses.
Use the existing canonical release resolver rather than hand-building release JSON.

### A4 — Owned preparations are invisible to the activation worker

Host scene preparation writes `task-evaluation-owned-scene-preparations`.
Configured-controls worker CLI points at `task-evaluation-launch-preparations`.
The separate autoprovision config defaults to the owned root, but does not redirect
the onward selector: `task_evaluation_configured_controls_progression_worker.py:1714`
passes context to `task_evaluation_controls_autoprovision.py:462`, which forwards
the CLI queue to `task_evaluation_scene_configuration_activation_automation.py:1278`
and `:1317`. Only that queue's results are scanned.

Fix: explicitly route the owned construction-preparation source through every
relevant selector without confusing later controls/episode preparation queues.
Acceptance: one eligible owner result plus unrelated legacy results yields exactly
one correctly bound activation, using production-configured roots. Installing
controls bootstrap alone is insufficient.

### A5 — Public provisioner rewrites retained provider consent

`task_evaluation_public_scene_intent_provisioner.py:346` replaces the owner's
`consent.provider_terms_reference` with the supplied review file hash. It does not
require that the owner accepted that hash. Downstream owner authority (`:175`)
compares against the already substituted value.

Reproduction: input consent `terms-v1` becomes a different SHA reference and the
real factory reaches `publication_ready`. A file digest identifies terms; it does
not establish acceptance. This is a provenance defect, not a demonstrated remote
authentication bypass.

Fix: validate exact accepted terms or an explicit retained delegation binding them;
refuse mismatch rather than rewriting consent. Test changed terms against unchanged
owner authority through the real consumer. Do not assemble new owner consent from
an agent's description of historical approval.

### A6 — Spec E has no production input writer/configuration

The candidate reconciler expects six files under `terminal_result_root/<intent-id>`:
`policy_canary_result_projection.json`, `policy_canary_webapp_sync.json`,
`provider_zero_closure.json`, `terminal_result_publication.json`, `launch_request.json`
and `launch_profile.json` (E lines 99–105,165,184,205,248–251). No production writer
or installer supplies that structure. Tests manufacture all six (`:282–302`).

Real policy delivery (`task_evaluation_policy_canary_dispatcher.py:1139–1215`)
retains `dispatch_receipt.json`; projection and sync are computed in memory. The
launch reconciler writes `post_teardown_provider_zero_receipt.json` elsewhere.
There is no producer for the new `terminal_result_publication.json`.

Fix: consume existing sealed receipts or add a durable, idempotent index/adapter
written by actual producers, with installer-owned roots and owner/run bindings.
Acceptance must run real delivery, reconciliation and owner status consumers,
then authenticated Website readback. Do not fill the directory by hand.

### A7 — Spec E can mark the wrong execution/publication complete

E lines 120–124 compare request commit to profile commit; lines 155–159 compare
attempt commit to current release. The two pairs are never joined. Changing the
execution request/profile commit together still returns `completed []`.

E lines 204–214 also omit publication schema, run identity, durable producer seal
and referenced-byte validation. A wrong schema, unrelated run, nonexistent HTTPS
URI and omitted `provider_allocated` still return `completed []` in reproduction.

Fix: join owner intent, reserved attempt, source/runtime/input identities, execution
profile, launch request, projection, publication and resource closure through the
canonical validators. Require durable authenticated publication evidence. Negative
tests must alter each link independently and refuse every mismatch. These are
completion-integrity defects, not evidence of a Website authentication exploit.

### A8 — Spec E strands legitimate completed runs

- E line 178 accepts notification `accepted/delivered` only. Real authenticated
  sync (`task_evaluation_run_webapp_sync.py:452–460`) and dispatcher (`:1170–1174`)
  allow failed notification after successful durable readback. Reproduction returns
  perpetual `running / terminal_website_readback_pending` for that valid outcome.
- Expiry/revocation/pause early returns precede the reconciliation hook in
  progression (`:333–351`), preventing read-only closeout of already authorized
  execution. Intake status (`:315–319`) overwrites non-completed status accordingly.
- E's current-release equality (`:157`) silently prevents reconciliation after
  another deploy, even if the historical attempt was legitimately authorized.

Fix: separate permission for new execution from read-only closure of existing
attempts; separately report notification delivery. Resolve historical immutable
execution identities without substituting current code or accepting another run.
Test success after consent expiry, revocation, pause, deploy and failed notification.

### A9 — Owner preflight reports green for invalid consumer inputs

`task_evaluation_production_chain_preflight.py:1154–1177` checks existence/readability
but omits catalog seals and actual asset/runtime hashes. The passing fixture has
no seal and intentionally bogus digests (`tests/test_owner_scope_preflight.py:40–45`).
Reproduction: `preflight_blockers []`, then real consumer refuses
`controls_autoprovision_digest_invalid` on the same config.

Fix: bounded offline validation using the production catalog/config validators,
plus service-user and real-sandbox readback. Include queue-route checks and typed
refusals for missing credentials/spend pointers. A path probe is not full readiness.

### A10 — Accepted candidate scope exceeds implemented capability

Intake/policy binding accepts arbitrary exact two-candidate identities, but handoff
`task_evaluation_policy_canary_handoff.py:597–599` and scene setup (`:950`) use
`pi05_droid` and `groot_n17_droid`; presubmission inputs also name fixed manifests
(`:800–801`). This is not a blocker if 841757's admitted pair matches the exact
supported inventory. It is a late failure for another accepted pair.

Fix: verify actual pair/checkpoint capability before any construction spend, or
resolve supported candidates generically. Reject unsupported pairs at intake;
never silently replace them. Do not broaden policy scope for this audit.

### A11 — Interrupted provisioning exposes an incomplete intent

Provisioner stages the visible intent at line 357, then writes binding/machinery
at 361–362. An immutable conflict or I/O failure leaves an intent despite failed
provisioning. Fix by validating/staging immutable dependencies before publishing
the intent, with idempotent recovery. Test a crash/conflict at each publication step.
This is a hardening improvement after the blocking chain connections above.

## Recovery and funding cautions

Automatic requeue of an empty provider response is not completed by the current
launch reconciler. It intentionally does not allocate/retry (`:849`); stale leases
with fresh provider-zero evidence move to blocked (`:981–993`). Source recovery
recognizes bounded create failures, but needs child/ownership roots absent from
the new preparation bootstrap. Do not replace this with blind retry: preserve
ambiguous creates, reconcile ownership/billing/teardown, obey the owner's retry
cap (including zero), and emit a terminal refusal when no successor is authorized.

Billing and credit collection are working. Controls still needs fresh, valid
project-spend inputs as well as its catalog. Account credit alone does not prove
the combined preparation, reconstruction, placement, controls and policy ceilings
fit. Produce a per-provider bounded exposure packet before asking for paid approval.
Alert delivery is lower priority; it must not become an excuse to bypass hard caps.

## Verification performed and limits

- Source provisioner tests: **14 passed**, 4.86s. Protect factory preparation and
  authority shape; the nominal fixture sets submission_enabled=false and stops at
  publication_ready, so it does not prove installed-service or construction execution.
- Controls installation/autoprovision, policy binding and owner wiring: **63 passed**,
  18.97s. The installer test genuinely invokes the controls producer, with external
  publication/provider-zero reads faked. It does not prove the production queue route.
- Candidate E tests: **16 passed**, 2.42s. Their manually assembled receipts conceal
  the producer gap. The test named authenticated Website projection calls local
  `scene_intent_status`; it does not contact Website.
- Credit admission, billing reconciler and owner preflight: **33 passed**, 1.93s.
  Protect funding refusals, billing parsing and current preflight checks; the
  adversarial preflight experiment demonstrates their missing consumer validation.
- Policy lifecycle and provider import-closure: **18 passed**, 69.34s, using
  `python -m pytest -q -m '' tests/test_native_task_arena_policy_canary_lifecycle_rehearsal.py tests/test_provider_runtime_import_closure.py`.
  This confirms the canary orchestration and sealed runtime imports; it does not
  close the owner queue, source preparation or terminal publication gaps above.
- No paid/GPU execution, live artifact re-publication, or host configuration changes.
  Synthetic reproductions establish code defects, not scientific results. Exact live
  841757 retained inputs have not traversed all downstream consumers in this audit.

Reproducers and recorded outputs are under [audits/2026-09-06](audits/2026-09-06).
Run from an isolated checkout with its `src` and repo root on PYTHONPATH. Source and
preflight scripts use main `4e0a38f4a`; terminal script requires exact #1729 head
`1e95c27b7d230dd2b72d732bc3036f028ae17e1b` (imports its committed test fixtures).
Use the checkout's Python environment with pytest available. They allocate no
provider resources and contact no external services.

## Required next proof before the paid decision

Fix A5/A7's authority/completion integrity, then A1–A4/A6/A8's chain connections;
strengthen A9 and preflight the exact pair in A10. Install validated controls
bootstrap through its canonical installer. Freeze one release and one real owner
intent, then drive the actual timer/queue consumers against retained 841757 inputs
with no model/GPU calls. Keep the original failed jobs and adopted stages visible.

The no-spend evidence must reach actual construction admission, controls intent,
policy binding/dispatch admission and terminal reconciliation using real producer
shapes. Report expected external/GPU boundaries separately from code refusals.
Only after that packet is clean should the user review exact arming and bounded
paid execution. A Quick-10 canary remains diagnostic; a full scored evaluation also
needs the reusable scenario matrix and controls on every scored cell.
