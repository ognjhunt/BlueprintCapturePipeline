# Control-plane capacity response

Use this runbook when a capacity alert pages an operator, a scene waits in
`queued_for_capacity`, or the disk report says growth or reclaim is blocked.
The [storage contract](../CONTROL_PLANE_STORAGE.md) defines the reservation and
retention rules.

## First look

Run `python3 scripts/operator_door.py status`, then
`python3 scripts/operator_door.py usage`. Read the `capacity` section and the
door-readable `capacity/summary.json` and `storage-gc/summary.json` if present.
Compare the affected mount, free bytes, floor, live reservations, refused roles,
forecast, volume growth status, reclaim outlook, and retained reasons. A missing
or stale GC summary means reclaim effectiveness is unknown; it is not proof that
there is nothing to free.

The controller warns at 70% utilization and marks a mount critical at 85%, or
when any stage would be refused. A `page` alert requires a person to act: floor
within three days, admission refused, unreadable mount, blocked volume growth,
an unconfigured alert route, or critical capacity with fresh GC evidence showing
no candidates and no bytes reclaimed. `warn` alerts include utilization, poor
usage attribution, release-retirement attention, and unreported break-glass
notes. A new page fingerprint posts immediately; a persisting page repeats
hourly. The webhook must reach a person, not merely accept a request.

The scene-intake HTTP route accepts a signed intent even if launch preparation
is short of space and returns `capacity.state=queued_for_capacity`. Execution
still waits at the whole-chain admission gate. `capacity_wait` records the
shortfall and an ETA only when a current growth or reclaim plan supports one;
`operator_action_required` and `unknown` carry no promised time.

## Grow or reclaim

For a configured DigitalOcean volume, check `volume_resize.status` and
`reason`. `volume_at_maximum` needs a provider limit increase before raising
`BLUEPRINT_CAPACITY_VOLUME_MAX_GIB`. To permit the approved automatic step, set
`BLUEPRINT_CAPACITY_AUTORESIZE_ACK=grow-control-plane-volume` in the protected
host environment after reviewing the plan. A rejected resize must be checked
against the provider receipt and filesystem size before another action.

Use the door for reclaim: `python3 scripts/operator_door.py unit start
blueprint-control-plane-storage-gc.service`, then inspect its plan and receipt.
Evidence offload requires `BLUEPRINT_CONTROL_PLANE_EVIDENCE_OFFLOAD=1`, verified
remote readback, and a pointer before local bytes may leave. Scene-workspace
retirement is a separate opt-in; use `python3 scripts/operator_door.py
retire-scene-workspace <scene_id> --wait` to review an exact-path plan and
`--apply` only with the owner's approval. Nobody hand-deletes evidence.

Use an owned, expiring door hold instead of stopping a timer or path directly:
`python3 scripts/operator_door.py hold <unit> --owner <name> --reason <why>
--for 2h`. Release it through the door after the reason ends. If a host change
had to bypass the door, record a sealed break-glass note immediately with
`python -m blueprint_pipeline.control_plane_break_glass record` and verify it
appears in the next deploy receipt. A note is an audit record, not cleanup
permission.

## Owner's Phase 0 checklist

1. Review the current `status` and `usage` evidence; approve a root-disk resize
   or exact surveyed cleanup before freeing data.
2. Request a DigitalOcean volume limit above 100 GB. Once granted, raise
   `BLUEPRINT_CAPACITY_VOLUME_MAX_GIB` to the approved limit and set
   `BLUEPRINT_CAPACITY_AUTORESIZE_ACK=grow-control-plane-volume` if automatic
   growth is desired.
3. Point `BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL` to a route that pages a person.
   Send a bounded test alert and confirm delivery to that person. An empty route
   is itself visible as `operator_alert_route_unconfigured` in the capacity
   summary.

## Review retained experiment-folder decisions

### Observe registered lane retention

The optional `lane_scratch` phase observes registered folders beneath explicitly
configured `/mnt/blueprint-work/lanes` and
`/var/lib/blueprint/task-evaluation-inputs/lanes` parents. Configure at most two
parents using repeated `--lane-scratch-root` arguments on the existing GC `run`
command, or colon-separated `BLUEPRINT_CONTROL_PLANE_GC_LANE_SCRATCH_ROOTS`.
There is no default lane-parent scan and no arbitrary-root override.

`BLUEPRINT_CONTROL_PLANE_GC_LANE_SCRATCH` accepts empty/unset, `0` or `false`
as false, and `1` or `true` as true (case-insensitive). The flag records a request
for future activation: **this phase is always report-only**, even when it is
true and the GC tick runs with `--apply` and its acknowledgement. It creates no
lock file, releases no pin and deletes/offloads nothing. Invalid configuration
produces a fixed local alert and an incomplete report.

The private raw report retains sealed lease identities and reasons why every
registered folder is kept. Unregistered folders are counted without traversing
their payload. Measured logical and allocated footprints include the lease,
deduplicate regular file inodes, exclude directory allocation and never predict
freed disk space. Partial observation makes totals unknown. Strict pin matches
can retain a folder; a complete ledger with no match still cannot clear queues,
processes, consumers, owner approval, evidence policy or restore requirements.

The door-readable summary projects typed counts and footprint fields without
owners, folder names, paths, references or digests. Its candidate bytes remain
null, removal bytes remain zero, and count-only lane reasons do not enter byte
rankings or queue ETA. Expected lane incompleteness leaves other phase forecasts
intact; an unexpected internal exception retains the tick's existing global
phase-failure behavior. Enabling actual lane cleanup remains a later reviewed
execution gate.

### Validate retained decisions

A retained census can be annotated and validated locally without inspecting its
current target folders. Prepare UTF-8 JSON with
`schema_version: control_plane_lane_scratch_annotations.v1`, a `census_digest`
of `sha256:` plus the SHA-256 of the **exact retained census file bytes**, and a
`decisions` list. Every inventory row needs one decision with its exact `path`
and an explicit `owner`; owner guesses are not approval.

Example decision shapes (replace identifiers and the absolute expiry):

```json
[
  {"path":"/mnt/blueprint-work/experiment-a","action":"keep","owner":"nijel","expires_at_epoch":1800000000},
  {"path":"/mnt/blueprint-work/experiment-b","action":"register","owner":"nijel","lane":"ops","name":"experiment-b","reason":"retention_review","class_intent":"cache","cleanup":"owner_review","ttl_seconds":86400,"run_ref":"run-1","size_budget_bytes":4096},
  {"path":"/mnt/blueprint-work/experiment-c","action":"offload","owner":"nijel","reason":"retention_review"},
  {"path":"/mnt/blueprint-work/experiment-d","action":"delete","owner":"nijel","reason":"retention_review"}
]
```

Keep expiry must be in the future and no more than 14 days away. Registration
requires explicit TTL no greater than 14 days and exactly one `run_ref` or
`scene_ref`; caches need a positive integer size budget. Existing
`lanes/<lane>/<name>` paths must match registration metadata. Offload and delete
proposals are refused when any retained reference is present. A historical path
outside that lane layout remains a proposal; registration validation does not
relocate it.

```bash
python3 scripts/lane_scratch_census.py \
  --validate-census retained-census.json --annotations annotations.json \
  --json-out validated-decisions.json
```

`--work-root` and `--inputs-root` may supply explicit lexical containment roots.
Scan/reference options are refused in this mode. Inputs must be bounded regular
files with no linked ancestors or parent traversal. Lexical inventory paths and
allowed roots are bounded to 4096 UTF-8 bytes and 64 components; these are parser
resource limits, not current filesystem or `PATH_MAX` proof. The complete census must
have consistent counts, measurements and reference accounting. Malformed,
incomplete, ambiguous or digest-mismatched inputs produce a small typed refusal;
the optional artifact is written only after validation succeeds and cannot alias
either input. Stdout and the artifact contain the same bounded JSON bytes.

The report records `mutations: 0`, `execution_authorized: false` and
`requires_fresh_reference_check: true`. It authenticates neither an annotator nor
current filesystem/reference state. It issues no lease, pointer, cleanup ACK or
reclaimed-byte forecast. Applying decisions, registering old folders, enabling
cleanup and preserving/restoring evidence remain separately reviewed execution
gates after the merged reference/pin proofs; this command frees no bytes.

### Exact-scene metadata and measured KEEP plan

The standalone planner reads bounded retained JSON and measures only paths bound
by the supported historical lineage. It opens no payload files and returns a
KEEP-only report. It does not establish current rights, provider-zero, exclusive
ownership, consumer fences, restore proof or retirement eligibility.

```bash
python -m blueprint_pipeline.task_evaluation_scene_lifecycle_plan \
  --intent-id INTENT_ID --context-file retained-planner-context.json \
  --now UNIX_SECONDS
```

The context is an operator-supplied regular JSON file with these exact fields:
`roots`, `parent_routes`, `retained_metadata_roots`, `acquisition_anchors`,
`retained_metadata_files`, `pins_root`, `primary_queue_contracts`,
`auxiliary_queue_contracts`, `reference_family_contracts`, and `progression_config`.
`roots` names the 18 roots accepted by the retained native-owner inventory;
`parent_routes` contains explicit `queue_root`/`input_root` pairs including the
canonical preparation route. At most four physical acquisition anchors are
allowed, including the context's retained parent unless it coalesces with an
anchor. Explicit metadata selectors contain only a supported `role` and `path`
under a declared retained metadata root. They authorize lookup, not ownership.

Primary queue contracts contain `root_path` and a finite `states` list;
auxiliary contracts contain `family` (`preparation` or `sam`) and `root_path`;
reference contracts contain `family` (`preparation` or `activation`) and
`queue_root`. `progression_config` is a retained JSON path or null. Matching its
sealed root declarations does not prove the running service configuration.

One five-second resource budget covers context acquisition, lineage, pins and
queue observers, measurement and output. Physical acquired bytes are reported
separately from the reference interpreter's additional conservative supplied-input
work charge. Both consume the same allowance. Metadata changes, missing or unknown
history, linked metadata, unsafe output identities and exhausted budgets keep the
report incomplete. Observed counts may remain available after later metadata drift;
current measured totals become null. An absent family is unknown, not zero bytes.
Malformed input and resource/publication failures return bounded typed JSON without
echoing raw input. There is no apply option and this command frees no bytes.
