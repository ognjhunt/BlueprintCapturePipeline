# Camera pose binding incident — 2026-09-09

ADP-009D development-only policy rehearsal / day-14 simulator precursor.
V14 (ca29d708440e, Vast 50434665) disproved the earlier claim that one reset
rerender completed the repair. The exact native gate measured the requested
seven joint positions within float32 precision, while the wrist camera reported
its prior pose and zero task-object pixels. No policy was queried. The allocator
retained the failed output and closed the instance.

The pinned Isaac source has a producer/consumer settings mismatch. PhysxManager
`_load_fabric` enables Fabric, disables USD transform updates, and publishes
`/isaaclab/fabric_enabled`. FabricFrameView reads `/physics/fabricEnabled`, whose
missing value selects USD fallback. Blueprint now binds that compatibility key
to its configured Fabric mode before the camera views are constructed. Official
camera calibration, reset pose, task geometry, and visibility thresholds remain
bound to the existing packet. The native gate records each camera view backend.

The new CPU regression executes the exact pinned producer, view constructor,
and pose-source selection methods. Its external GPU buffers are fakes, so it
proves configuration/routing only. The earlier reset test proves call order,
not real camera freshness. Neither CPU test nor merge is live GPU proof; that
requires a native camera readback and task-pixel gate pass on the corrected run.

Validation: focused camera, runtime, policy-canary lifecycle and isolated provider
import-closure suites. This incident and preserved V14 failure remain evidence.
