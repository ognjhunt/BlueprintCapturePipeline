# Direct camera aim — ADP-009D / day-14 development rehearsal

The operator explicitly requested on 2026-09-09: “POINT THE CAMERA AT THE
OBJECT”. V15 had already demonstrated that selecting Fabric alone left the
initial wrist view unchanged; no candidate policy was queried. Its failure is
retained, and no native camera repair is claimed from a CPU test or merge.

A digest-bound per-cell operator camera instruction now selects a direct
initial look-at orientation toward the task object's known starting position.
The camera keeps its official optical intrinsics and mounting position; its
orientation is explicitly operator-directed, not advertised as the official
DROID mount. That relative orientation remains fixed during the episode. Every
camera pose is composed from the current native PhysX link transform and that
fixed mount, avoiding stale USD transforms and timestamp-cached reset poses.
The camera does not track the object after the reset, and no policy sees object
state through this mechanism. Both candidates receive the same resolved setup.

The unchanged native RGB/semantic visibility check must pass before the
visible-object baseline can begin. Camera matrices, exact observation media,
and the operator instruction remain in the result evidence. These are
internal diagnostic runs; they do not establish physical success or qualify
an official DROID policy comparison.

The operator also proposed an object-acquisition variation. After a valid
baseline, preregister paired visible, partially occluded, and initially
out-of-view starts. That future variation should permit genuine search with a
working camera and record time to first sighting and deterministic task
outcome. It is a proposed follow-up, not an addition to this frozen matrix.
