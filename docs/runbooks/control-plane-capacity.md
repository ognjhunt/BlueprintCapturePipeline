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

The reclaim outlook counts verified applying replay-cache candidates; plan-only
`estimated_candidate_bytes` do not support an ETA. Registry-run residue counts
only when evidence offload and residue offload are both enabled, the phase is
enabled and applied, and its byte totals are complete. Older summaries that omit
the residue switch cannot prove an enabled phase is actionable. Releasing cache
pins frees no bytes itself: any eligible files are counted by derived-directory
cleanup, without adding pin counts to the forecast.

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
