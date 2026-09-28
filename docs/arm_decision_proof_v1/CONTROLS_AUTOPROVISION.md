# Preparation-to-controls worker

ADP-009D/day-28, development-only. The completion artifact is
`autoprovision-receipt.json`, binding the persisted authenticated scene intent,
preparation link, exact robot/runtime catalog binding, actual canonical producer
output, and installed registry digest. This receipt proves provisioning only;
it does not prove GPU execution, policy comparison, or physical outcomes.

The configured-controls progression worker calls `process_config` before scene
activation when `BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG` names a
service-owned JSON configuration. The same operation is available through:

```sh
python -m blueprint_pipeline.task_evaluation_controls_autoprovision \
  --config /etc/blueprint/controls-autoprovision.json \
  --expected-production-commit <exact-40-character-commit>
```

Configuration fields are `scene_root`, `preparation_queue_root`, `controls_root`,
`intent_root`, `profile_dir`, `robot_catalog_path`, `trusted_clients`, and optional
`service_group` (default `blueprint`). Keep generated controls under the existing
retained task-evaluation input store, with registry paths readable by the controls
service. The worker does not create infrastructure or buy storage.

The intake factory uses `build_preparation_link(**fields)` to seal a
`task_evaluation_scene_preparation_link.v1` record. Required fields are `intent_id`,
`intent_digest`, `preparation_id`, `request_digest`, `expected_production_commit`,
`team_namespace`, `scene_id`, `task_id`, and `result_filename`. The filename is
exactly `<preparation_id>-<request_digest without sha256 prefix>.json`. The factory
retains immutable history under the scene intent before updating its current
`preparation-link.json`. The worker reads only the current link, and resolves
the result and envelope under the preparation queue's `results` and `materialized`
directories. The preparation request must carry `scene_intent_digest`; legacy
preparations without that authenticated owner binding cannot enter this worker.

The signed owner task names `robot_binding_id`. Its service-owned catalog uses
schema `task_evaluation_controls_robot_catalog.v1`, a canonical `catalog_digest`,
and `bindings` keyed by that ID. Each binding supplies:

- `robot_asset_usd` and `embodiment_camera_template`, each `{path, digest}`;
- `runtime_source_payload_dir`, `runtime_digest`, `expected_production_commit`;
- `project_spend_current_path`;
- `openai_project_id`, `openai_api_key_id` (identifiers, never secret key values);
- optional `external_layer_bucket` and `phase_hard_cap_usd` (default $2).

`payload_digest(Path(...))` hashes a canonical mapping of every relative runtime
file path to its SHA-256. Symlinks and empty payloads are refused. The catalog is
an admitted embodiment/runtime binding, not a customer-supplied path mapping.

The spend controller refreshes the current pointer only after actual project
reconciliation and inventory verification. Its schema is
`task_evaluation_project_spend_current.v1` with `path`, `digest`,
`observed_at_epoch`, and canonical `receipt_digest`. The worker requires an
observation at most 900 seconds old, verifies referenced bytes, and calls
`validate_project_spend_reconciliation` to reopen all baseline/billing sources.
It retains an immutable per-attempt snapshot. A compatibility catalog may instead
provide `project_spend_reconciliation: {path,digest}` plus its original
`project_spend_observed_at_epoch`; the same freshness checks apply. Copying or
restamping old billing does not constitute a new observation.

Construction, controls, and OpenAI placement each reserve a separate durable
owner attempt before publication. Defaults reserve $2 + $2 + $2.56 and three
attempt slots. These holds supplement earlier construction/preparation holds;
they never refund themselves after failures. Identical retries reuse the same
holds, input snapshot, and authority issue time. Expiry is bounded by original
consent, and the phase retry cap remains zero. The original exactly-two-policy
artifact identities are retained in the completion receipt; this worker does not
run or grade them.

The canonical continuation producer and registry installer are the defaults.
Tests replace external readback/publication only. The worker can publish admitted
runtime metadata and read provider inventory; it never calls a provider create.
Expiry, revocation, mismatched releases, and failed provisioning filter the
affected scene from configuration activation, controls progression, and canary
handoff. Other scenes continue. Corruption with unresolved scene ownership stops
the tick so an older installed intent cannot bypass an unreadable authority.
Queued dispatch can call `owner_authority_blocker(config_path,
scene_intent_digest=...)`; `None` means owner consent remains live, and a string
is the refusal code. This supplements exact attempt and paid-resource admission.

## Deployment: materializing the config before its env pointer

`task_evaluation_controls_autoprovision_installation.install_controls_autoprovision`
recompiles one operator-owned, sealed bootstrap
(`/etc/blueprint/task-evaluation-controls-autoprovision-bootstrap.json`,
schema `task_evaluation_controls_autoprovision_bootstrap.v1`) into three
service-owned files: the sealed **content** catalog
(`task_evaluation_controls_robot_content_catalog.v1`, so the worker re-binds it
to the active release each tick without this step mutating source bytes), the
config JSON, and the `EnvironmentFile`. The bootstrap names
`robot_catalog_bindings` (each binding is validated by running the real
`resolve_robot_catalog`, so a changed robot USD, camera template or runtime
payload is refused at install), `scene_root`, `preparation_queue_root`,
`controls_root`, `intent_root`, `profile_dir`, `trusted_clients`, and the
service account/group.

The order is load-bearing: the config and catalog are written and validated
**before** the `EnvironmentFile` that exports
`BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG`. If that variable ever
named a missing or unparseable file, `progression_owner_scope` would set
`unresolved` and the worker would return a blocked report, stopping every scene.
So the env pointer is only written once a readable, valid config exists.
`deploy_control_plane_commit` runs the installer after the autostart-intent
registry step (so the registry directory's group-writable permission wins) and
before the unit restart; without the operator bootstrap the deploy records
`not_configured` and the worker stays in the legacy no-autoprovision lane. The
`--controls-autoprovision-bootstrap-file` flag overrides the default path.

The autoprovision worker installs its write-once `0440` controls intents into
`/etc/blueprint/task-evaluation-configured-controls-intents` (the registry the
configuration-activation and controls-progression consumer already read), so the
controls-progression unit lists that directory under `ReadWritePaths` and the
deploy makes the directory group-writable while the intent files stay immutable.

## Owner-only dispatch selection

Both dispatchers and controls progression carry
`BLUEPRINT_TASK_EVALUATION_DISPATCH_OWNER_SCOPE=persistent_owner_only` and load
the shared owner-store identity
(`/etc/blueprint/task-evaluation-scene-progression.env`, which supplies
`BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT` and
`…_SCENE_INTAKE_CLIENT_IDS`) so `scene_policy_binding.scene_store()` can resolve
the persistent owner. Selection is not authority: it only narrows which rows a
dispatcher may claim; every owner reservation, frozen profile, standing grant,
release, spend, expiry, revocation and provider-zero gate still runs before any
paid step, and the execute/dry-run holds are untouched. The production chain
preflight's owner-mode checks (`owner_scope_checks`) refuse a default scope, an
unresolvable owner store, a missing/unreadable autoprovision config, and missing
robot/camera/runtime assets before a dispatch selects a wrong or empty row.

When the original task omitted `robot_binding_id`, the authenticated intake
issuer can use `task_evaluation_scene_robot_assignment.assign_scene_robot` with
`ack="assign-scene-robot"`, the exact owner/intent/issuer, `robot_catalog_path`,
`robot_binding_id`, and a private `authorization_reference` JSON path. That file
uses schema `task_evaluation_scene_robot_assignment_authorization.v1` and binds
`authorized=true`, `intent_id`, `intent_digest`, `owner`, `authenticated_issuer`,
`robot_binding_id`, and `catalog_binding_digest`. The digest comes from
`catalog_binding_digest(binding)` and excludes only the dynamically rebound
`expected_production_commit`; robot, camera, runtime, caps, credential identifiers
and spend-pointer metadata remain bound. Preserve the actual human authorization
or delegated decision in that private file; do not invent a new approval.

The API checks current nonrevoked consent and actual robot/camera/runtime bytes,
then writes immutable `<intent>/robot-assignment.json` under the intake lock.
Authorization references are absolute nonsymlink files, mode 0600/0640, whose
exact bytes are retained by reference. Duplicate requests retain the same record;
conflicting choices refuse. A sole catalog entry never becomes an implicit choice,
and an existing task choice cannot be overridden. Controls and terminal adoption
reopen the assignment before provisioning and include `robot_assignment_digest`
in their retained input identity and provisioning receipt. The original task,
source freezes, completed artifacts, policies, spending bounds and holds remain
unchanged. Removing or changing the assignment, authorization or catalog row
refuses further provisioning. Other task/source consumers do not gain a robot
field from this operational assignment.

An explicit request to omit control trials uses the existing diagnostic policy
lane. A private owner directive lives at
`/etc/blueprint/task-evaluation-diagnostic-control-omissions/<intent_id>.json`
(`BLUEPRINT_TASK_EVALUATION_DIAGNOSTIC_CONTROL_OMISSION_ROOT` for another server
registry). Schema `task_evaluation_scene_control_omission_directive.v1` binds the
exact owner/intent/issuer, original task digest, frozen policy pair, current consent
expiry, actual user request and authorization reference. It permits exactly 20
policy episodes, names the omitted zero-action and scripted-positive controls,
and fixes `run_kind=internal_policy_canary`,
`claim_ceiling=diagnostic_policy_execution`, unchanged task-scoring criteria and
`qualified_comparison_permitted=false`. The standard `directive_digest` covers
the record; its file must be nonsymlink and private (0600/0640).

The normal worker preserves robot placement and native construction. Once its
construction launch exists, the directive prevents standalone controls admission;
the diagnostic handoff independently requires the real qualified construction
receipt, retained billing lineage, Website acknowledgement and teardown/provider
zero. It never synthesizes a controls result. The existing `direct_policy_request`
producer derives the control-omission authority and contract. Both policies keep
the same cells, seeds, scoring criteria and lossless-media requirements. The normal
result-delivery path explicitly labels `controls_omitted_by_user`. Missing opt-in
preserves the strict default; tampered, expired or conflicting opt-in refuses.

Omitted-control execution still requires final robot-reset and camera binding.
The handoff reads an immutable kinematic calibration keyed by exact assigned robot
USD SHA under `/etc/blueprint/task-evaluation-policy-camera-calibrations`.
Schema `policy_canary_robot_camera_kinematic_calibration.v1` binds
`source_robot_asset_sha256`, exact `camera_start_binding` and
`native_reference_gate` path/SHA/size references, and `calibration_digest`.
Only the retained joint-frame calibration and measured wrist pose are reused;
an old blocked visibility gate does not become passed evidence. The new binding
uses the current construction's native initial root/joint readback, current task
and cameras, and checks framing for all ten cells. Current native visibility,
collision, composition, billing and teardown gates remain mandatory. Neither
original task/source records nor already admitted controls are rewritten.
