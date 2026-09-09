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

## V16 root-cause correction

V16 (`ec7a0b1bbb4a84da7461e83ad2021744b4ff9cb5`, GPU 50439577)
failed before any policy query. The book is rendered: external/overview native
semantic AOVs measured 24,573/24,580 object pixels. The wrist measured zero.
External and overview resolve to the same world pose and equivalent 500-pixel
focal length; their nearly identical images are explained by configuration.

The active backend is `IsaacRtxRenderer`, not `OVRTXRenderer`. In the exact
sealed runtime source archive (`b7bbb7ae3064ad581a710d57121131eca776aabcab3ce2593b4ef396c6a3d3f1`),
`isaaclab_physx/renderers/isaac_rtx_renderer.py` makes `update_camera` a no-op
and consumes scene camera prims through Replicator. V16 replaced a pose getter
and updated configuration metadata but never called the scene pose setter.
Its changed pose receipt therefore did not prove a changed rendered camera.
The previous inspection followed OVRTX's implemented update hook; that was the
wrong producer-to-consumer path for this run. The retained wrist image changed
by only 1.51 levels per RGB channel on average from V15, consistent with the
same view plus render noise.

There was a second ordering defect: the mount was frozen at environment
construction, before the complete environment reset applied the configured
joints. V16's reported optical coordinates placed the object at approximately
(0.108, -0.473, -0.329) meters, well below the image bounds, instead of centering
it. The new regression reproduces the missing reset against the exact frozen
V16 module and models a renderer that consumes scene transforms, not metadata.
The earlier pure matrix tests did not cover those runtime boundaries; discovering
these defects on paid compute was a process defect.

The correction completes the setup reset, freezes the target-facing mount from
the measured native body pose, writes the USD mount orientation and actual
Fabric world camera pose, advances the render generation, and returns the scene
view's independent pose readback. A write/readback mismatch refuses the frame.
Both policies still use the same fixed mount, intrinsics, scene, object, cells,
and seeds; coordinates are used only for setup. Native image confirmation is
still required before claiming repair or production success.
