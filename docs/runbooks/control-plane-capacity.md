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
within three days at the week's trend, floor within six hours at the last hour's
rate (`floor_within_hours`), admission refused, unreadable mount, blocked volume
growth, an unconfigured alert route, or critical capacity with fresh GC evidence
showing no candidates and no bytes reclaimed. The week's trend starts at its
oldest row, so a reclaim inside the week can hide a fast writer. The recent rate
needs at least 30 minutes of ticks, at least 1 GiB lost in each half of the
window, and a decline at least 2 GiB beyond the largest live reservation that
window saw, since admitted writers were budgeted against the floor. It projects
that unreserved rate against the admission headroom. Once paging, it clears only
when the projection reaches twelve hours or the decline stops. `warn` alerts include
utilization, poor usage attribution, a container runtime store over 20 GiB,
release-retirement attention, and unreported break-glass notes. A new page fingerprint posts immediately; a persisting page repeats
hourly. The webhook must reach a person, not merely accept a request.

The scene-intake HTTP route accepts a signed intent even if launch preparation
is short of space and returns `capacity.state=queued_for_capacity`. Execution
still waits at the whole-chain admission gate. `capacity_wait` records the
shortfall and an ETA only when a current growth or reclaim plan supports one;
`operator_action_required` and `unknown` carry no promised time.

The reclaim outlook counts verified applying replay-cache candidates; plan-only
`estimated_candidate_bytes` do not support an ETA. Registry-run residue counts
only when evidence offload and residue offload are both enabled, the phase is
enabled and applied, and its byte totals are complete. Already removed or
offloaded bytes are subtracted from each phase's pre-apply candidates before
promising future space. A residue phase with retained rows or missing reason
accounting remains unknown, since its total does not separate blocked bytes from
actionable bytes. Older summaries that omit the residue switch cannot prove an
enabled phase is actionable. Releasing cache pins frees no bytes itself: any
eligible files are counted by derived-directory cleanup, without adding pin
counts to the forecast. These counters describe logical local eviction, not
measured filesystem free space or a promise that retained runs will fit the next
tick's publication cap. Any deferred or skipped residue leaves its forecast
unknown.

## Grow or reclaim

For a configured DigitalOcean volume, check `volume_resize.status` and
`reason`. `volume_at_maximum` needs a provider limit increase before raising
`BLUEPRINT_CAPACITY_VOLUME_MAX_GIB`. To permit the approved automatic step, set
`BLUEPRINT_CAPACITY_AUTORESIZE_ACK=grow-control-plane-volume` in the protected
host environment after reviewing the plan. A rejected resize must be checked
against the provider receipt and filesystem size before another action.

Use the door for reclaim: `python3 scripts/operator_door.py unit start
blueprint-control-plane-storage-gc.service`, then inspect its plan and receipt.
Every storage GC switch is on by default since 2026-09-30, and `=0` in the
environment file opts one out. The unit reads that file optionally, so a missing
file now means every phase applies. To stop reclaim in an emergency, place a door
hold on `blueprint-control-plane-storage-gc.timer` (below) rather than editing or
removing the file. Evidence offload still requires verified
remote readback and a pointer before local bytes may leave. Scene-workspace
retirement is a separate switch; to retire one scene by hand, use `python3 scripts/operator_door.py
retire-scene-workspace <scene_id> --wait` to review an exact-path plan and
`--apply` only with the owner's approval. Nobody hand-deletes evidence.

Use an owned, expiring door hold instead of stopping a timer or path directly:
`python3 scripts/operator_door.py hold <unit> --owner <name> --reason <why>
--for 2h`. Release it through the door after the reason ends. If a host change
had to bypass the door, record a sealed break-glass note immediately with
`python -m blueprint_pipeline.control_plane_break_glass record` and verify it
appears in the next deploy receipt. A note is an audit record, not cleanup
permission.

## Container image storage on the root disk

Docker on the host keeps image layers and BuildKit cache in the containerd
image store, `/var/lib/containerd`, with metadata in `/var/lib/docker`. Both stay
on the root disk: `deploy/host/mount_work_volume.sh` moves Blueprint roots only,
and no storage-GC phase touches them. With the default `docker` driver and the
containerd image store, as on this host, a local `docker buildx build --push`
still unpacks the image into that store. On 2026-10-01 one worker-image build
grew the store to about 80 GB and took root from 55 to 31 GB free within hours.

When the hourly walk reaches them, the survey names both stores as host roots
and warns `usage_container_runtime_large` above 20 GiB. A truncated walk marks
their totals as lower bounds (`container_runtime_complete: false`). On this host
the root walk has been stopping before `/var/lib`. A fast unreserved decline
pages `floor_within_hours` whatever the writer. The door cannot read these
paths, so the host owner reads `docker system df` and the store's size directly.

Freeing or moving this space is an owner decision. Nothing reclaims it
automatically:

- Once the pushed image's registry digest and revision label are verified, its
  local copy and build cache are reproducible. Removing them still needs the
  owner's approval, like any cleanup.
- To bound the cache, configure the daemon's BuildKit garbage collection
  (`builder.gc` in `/etc/docker/daemon.json`; key names vary by Docker
  version), or build worker images off the host.
- To keep image bytes off the root disk, move the containerd root onto the work
  volume while both Docker and containerd are stopped. Register the new path
  first, with a storage-table row and a survey alias. Otherwise the survey
  counts it as orphan scratch, and that pages. The volume then carries each
  build's peak, so check its headroom first.

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

One scene-specific resource budget is initialized once with a maximum thirty-
second deadline and one million cumulative work values. It covers context
acquisition, lineage, pins and queue observers, measurement and output. The
existing independent observer defaults remain five seconds and 100,000 values;
all other native ceilings and per-pass guards remain unchanged. A used, closed,
failed or partly initialized budget cannot be reset or extended. Physical acquired bytes are reported
separately from the reference interpreter's additional conservative supplied-input
work charge. Both consume the same allowance. Metadata changes, missing or unknown
history, linked metadata, unsafe output identities and exhausted budgets keep the
report incomplete. Observed counts may remain available after later metadata drift;
current measured totals become null. An absent family is unknown, not zero bytes.
Malformed input and resource/publication failures return bounded typed JSON without
echoing raw input. There is no apply option and this command frees no bytes.


### Record authenticated owner intent without changing folders

The owner-consent workflow records root-admin intent against exact retained
census and annotation bytes. It does not register, renew, offload or delete a
folder. A successful report keeps `references_clear=false`,
`consumer_fence_checked=false`, `execution_authorized=false`, mutations zero,
and candidate bytes/ETA contributions null.

Installation provisions an absent-only policy at
`/etc/blueprint-operator-door/lane-owner-policy.json` (root:root,0600), a store
at `/var/lib/blueprint-operator-door/requests/owner-consents` (root:root,0700),
and its stable `.owner-consents.lock` (root:root,0600). Upgrades retain existing
policy bytes, records and lock inode, and refuse unsafe existing paths. The
initial policy is disabled with no principals. The installed config defaults
`owner_census_decisions_enabled` to0. Installing this code enables no cleanup.

Operational issuance requires the reviewed installed door package at
`/opt/blueprint/operator-door`, a root-owned nonwritable door config, an enabled
finite policy, and the explicit config switch set to1. The config may retain
its installed root:blueprint0640 permissions and root:blueprint2750 parent;
policy and private records require root:root0600. Configure each policy row
with a literal `principal`, explicit `owners`, finite `allowed_actions`, and
`max_consent_seconds` between1 and1209600. No wildcard or inferred owner is
accepted. Only put the independently approved principals/owners/actions in
that policy; changing its exact bytes invalidates older consent records.

After checking the retained files' independent raw SHA256 and byte counts,
a root administrator can issue bounded metadata using the active release:

```bash
sudo env PYTHONPATH=/opt/blueprint/task-evaluation-control-plane/src \
  /opt/blueprint/BlueprintCapturePipeline/.venv/bin/python \
  /opt/blueprint/task-evaluation-control-plane/scripts/lane_scratch_census.py \
  --issue-owner-consent --census "$CENSUS" --annotations "$ANNOTATIONS" \
  --expected-census-sha256 "$CENSUS_SHA256" --expected-census-size-bytes "$CENSUS_SIZE" \
  --expected-annotations-sha256 "$ANNOTATIONS_SHA256" --expected-annotations-size-bytes "$ANNOTATIONS_SIZE" \
  --principal "$APPROVED_PRINCIPAL" --selected-path "$EXACT_RETAINED_PATH" \
  --consent-expires-at-epoch "$APPROVED_EXPIRY" \
  --door-config /etc/blueprint-operator-door/door.json
```

Every census row is validated before selected rows are projected. Owner guesses
remain guesses. The consent output supplies `consent_id`, `expected_sha256` and
`expected_size_bytes` for the immutable protected record; an existing record is
never overwritten. Expiry is bounded by policy and the selected keep/register
expiry. At most100 selected rows are accepted. Store capacity and invocation
limits can refuse issuance; nothing is removed to make space.

Use the existing authenticated door POST `/requests` with only this JSON:

```json
{"kind":"owner-census-decision","consent_id":"<32 lowercase hex>","expected_sha256":"sha256:<64 lowercase hex>","expected_size_bytes":1234}
```

The request requires `operate` scope. Request fields cannot supply a principal,
owner, policy, target path or apply flag. Copied spool `requested_by` is
unverified caller context, not administrative consent. The root runner invokes
one fixed active-release report wrapper with a read-only private store/policy
and writes only public result metadata. Both the current config and exact policy
are re-read; expired, revoked, changed or exhausted records refuse.

Inspect the returned request ID with `python3 scripts/operator_door.py request
<id>`. Its completed outcome identifies
`<results>/<id>.owner-census.json` plus that report's independent raw SHA256 and
byte count. Use the existing checked `pull` with those identities to retain the
report:

```bash
python3 scripts/operator_door.py pull "$REPORT_PATH" "$LOCAL_REPORT" \
  --expected-sha256 "$REPORT_SHA256" --expected-size "$REPORT_SIZE"
```

The existing client's `request --wait` success set does not include
`owner_consent_observed`; it can return1 despite this successful report outcome.
Use `request <id>` and inspect the bounded outcome's exit_code0/status instead;
that outcome proves report observation only. Public report/outcome files are regular root0644 under root0755 results
ancestry; protected consent remains0600 under0700 and is never made readable
through the door. Public reports omit policy bytes, tokens and private documents.

### Current historical owner review

After a separate, exact target-generation approval has been recorded, request
the current historical owner census with `python3 scripts/operator_door.py
legacy-owner-census --wait`. This is a distinct read-only request with no path,
owner, action or cleanup option. The outcome names `<results>/<id>.legacy-owner-census.json`
and its raw SHA-256 and byte count; use the checked `pull` command above with
those exact values. Rows include every bounded current candidate's size, age,
references and KEEP reason. A row receives `legacy_owner_review` and an owner
only while its generation, policy, expiry and references still validate.
Unregistered, changed, expired and active folders remain visible without an
owner label. An incomplete or oversized scan publishes zero owner labels.
Every row remains `gc_eligible=false`: this request does not authorize
deletion, offload or reclaim credit.

Leaf CLI refusal is bounded typed JSON; detailed fixed refusal is retained in
the request log when safely available. Wrapper invocation failure has the fixed
outcome `owner_consent_report_refused`, including resource exhaustion; it never
copies arbitrary stderr into the outcome. No report claims a held consumer
fence, a fresh complete reference inventory, restore proof or retirement
admission. Actual target action still requires a separate reviewed execution
contract and explicit operational approval.
