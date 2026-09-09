# Native reset camera freshness

ADP-009D, the development-only two-candidate fixed-arm rehearsal, requires exact
policy-visible observations. The observed v13 attempt on instance 50432858
returned the predecessor's wrist image and camera pose after a different joint
reset, then stopped before any policy query. Both world-camera checks passed.

The pinned Isaac Lab `ManagerBasedEnv.reset` writes state and forwards
kinematics, but its default `num_rerenders_on_reset = 0` leaves camera frames
stale. The focused regression retains that upstream reset method and invokes it
through the actual Blueprint configuration callback: zero rerenders returns the
old frame; one rerender returns the frame for the new reset without advancing
physics. Camera-disabled configurations retain zero rerenders.

The production runtime requests one rerender before each camera-enabled reset
returns observations. Policy camera gates also retain requested and measured
joint positions and the configured rerender count beside the camera snapshot.
This is a reusable runtime fix; CPU reproduction does not claim native visibility,
policy success, qualification, or physical truth. The failed attempt and its
media remain retained. Native visibility and exact policy-input retention remain
required for the diagnostic execution.
