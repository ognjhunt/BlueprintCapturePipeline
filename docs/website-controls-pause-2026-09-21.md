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

New website runs proceed from CPU scene compilation to the existing diagnostic
policy runtime without a scripted construction rehearsal. The runtime still checks
asset physics, gripper commands, reset state, and cameras before policy queries.
A failed scripted grasp is not a policy-admission requirement. The camera start is
computed from the sealed robot plan and validated kinematics, explicitly awaiting
native readback; no native construction success is invented. Simulation checks,
policy-input media, authority, spending, and teardown remain required. Existing
started construction runs retain their original closeout requirements.

Completion evidence for this change: focused owner-binding, controller callback,
allocator no-launch, native no-execution, and terminal closeout tests. Live proof
requires deployment and a controller-origin policy run; merge is not that proof.
