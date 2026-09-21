# Controls paused by platform owner — 2026-09-21

ADP-009D, day-21 development policy execution. The owner explicitly requested:
“Skip/pause the controls stage for all future runs.” This supersedes the older
per-scene opt-in requirement for current development runs.

The checked-in control-stage policy pauses zero-action negative and scripted
positive control trials. New website scene intents derive the existing typed
omission automatically, binding their own owner, task, policies, and expiry.
The platform owner authorizes this operating mode; the code does not claim that
each site owner personally requested it. Existing explicit directives remain
validated. Revoked or expired scene authority is not extended.

The placement feedback controller stops after successful construction and closes
its retained GPU without invoking controls. The normal controller then uses the
existing policy-canary handoff. Standalone controls admission, warm continuation,
and per-cell native controls refuse execution while paused. No controls success
receipt is fabricated. Policy results remain diagnostic and cannot claim a
controls-qualified comparison. Task scoring criteria are unchanged.

This pause alone does not remove the separate scripted construction prerequisite.
Separating basic simulation validity from scripted task success remains necessary
before a failed scripted grasp can proceed to a policy trial. Simulation checks,
policy-input media, authority, spending, and teardown remain required.

Completion evidence for this change: focused owner-binding, controller callback,
allocator no-launch, native no-execution, and terminal closeout tests. Live proof
requires deployment and a controller-origin policy run; merge is not that proof.
