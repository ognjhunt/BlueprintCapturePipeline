# Protocol Amendment 4: Vast No-ID Watchdog Settlement

Frozen prospectively: 2026-07-30T11:10:00-0500

## Triggering evidence

The stale-ask launch never received or recorded a Vast instance identifier.
After the unrelated writer's GPU closed, the successor owner proved both its
exact prefix and global Vast inventory were zero and issued the standard
mode-0600 owner cancellation request. The already running watchdog retained the
lease because its Vast early-cancel path required two exact-ID absence probes
even when no ID had ever existed.

No ID may be invented to satisfy that rule. The existing watchdog remains the
owner of its lease and reservation and will perform its ordinary hard-deadline
settlement. It must not be killed, impersonated, or bypassed.

## Generic settlement rule

For future Vast watchdogs:

- when a safe attempt-local instance-ID file exists, early cancellation still
  requires two exact-ID absence probes plus two exact-prefix and two global-zero
  inventories;
- when no attempt-local instance-ID file exists, early cancellation requires
  two exact-prefix and two global-zero inventories from the provider API;
- an unsafe, unreadable, malformed, or scope-mismatched ID file remains terminal
  and may not be treated as no-ID;
- early cancellation performs no provider mutation; and
- pending teardown, transferred lease, and campaign reservation close only
  after those independent zero checks pass.

Focused regression coverage must prove the no-ID path performs no synthetic ID
inspection, no termination, no sleep to deadline, and exactly two prefix plus
two global inventory checks. Existing recorded-ID and live-resource safety tests
remain mandatory.

## Claim boundary

Control-plane settlement proves only that no matching billable resource remains
and the local reservation is closed. It does not retroactively prove that a GPU
ran, that no provider-side request was attempted, or that any policy, WAM,
ranking, transfer, physical, economic, or thesis claim passed.
