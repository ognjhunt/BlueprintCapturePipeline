# Marked-area construction collection failure, 2026-09-09

ADP-009 development-only fixed-arm rehearsal, day-28 execution/evidence gate.
The completion artifact remains a measured native construction result followed
by controls and the frozen two-policy run; this attempt did not qualify it.

Instance 50400126 ran source 4e41f024b1f2674b3a2e9c5f3493bdc4894ab7ae
against the retained marked-area packet. Direct worker observation showed the
scene renderer and gripper-convention check complete, followed by failed pose
arrival phases. The worker wrote a 17,417,753-byte output archive and reported
upload success. The control plane then raised a log-transport failure before
its existing independent output-download block, tore down the instance and
removed the staged output. Its result correctly contains no native result.
Only separately retained camera snapshots survived. Object-store version
inspection found no recoverable prior version. Do not infer joint-error values,
construction qualification, or policy outcomes from the missing result.

The live API diagnosis reproduced HTTP 400 (`Request body must be valid JSON`)
for `PUT /instances/request_logs/50400126` and HTTP 200 for the same payload at
`PUT /instances/request_logs/50400126/`. The official Vast CLI uses the latter.
The endpoint now includes the slash. More fundamentally, an observed output
upload is downloaded before startup classification may raise and trigger
teardown. This preserves diagnostics while leaving missing startup proof and
scientific failures blocked; output presence does not confer qualification.

This is a process defect: prior helper-level log/output tests did not drive the
real adapter through its early heartbeat exception. The new adapter regression
checks download-before-teardown with unreadable logs and a failed native result.
Focused transport tests passed (27), adapter boundary examples passed (3), and
policy lifecycle/import-closure rehearsals passed (18). No additional paid run
was used to discover or validate the collection fix.

## Camera representability refusal

The 2026-09-09 v11 diagnostic attempt (Vast instance 50430972, source
`2aa2c1c26dfd2cde93ba6d155df1ddd5fadd05a7`) stopped before the first policy
query. Replaying the sealed plan through `camera_runtime_parameters` on CPU
reproduced `native_task_arena_camera_intrinsics_not_representable:overview`:
the review camera declared center `(640, 360)` while the existing camera
adapter requires `(639.5, 359.5)` for a 1280 by 720 image. The native receipt
retained the exception class but omitted its typed refusal code.

This was a process defect: a pure adapter predicate was first reached after
allocation, and the lifecycle fixtures used incomplete camera placeholders.
The startup preflight now invokes the real camera adapter for every resolved
cell, the lifecycle fixtures include complete cameras, and native initialization
failures preserve only allowlisted typed codes. Corrected inputs use the
adapter's existing review-camera convention; official policy-camera calibration
and robot placement remain unchanged. The failed run was retained and torn down;
no policy episode or native camera qualification was claimed.
