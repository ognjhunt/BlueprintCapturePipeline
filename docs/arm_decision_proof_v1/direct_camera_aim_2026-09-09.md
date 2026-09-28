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

## V17 CUDA readback boundary

V17 (`14caef0729a1539bc3e2026e5a748d1f7c2b9f42`, GPU 50441475)
closed with `TypeError` before retaining a reset frame or querying a policy.
The new scene readback called `ProxyArray.numpy()`. The pinned wrapper forwards
that method to its CUDA torch tensor, whose NumPy conversion requires an
explicit host copy. Replaying the exact pinned wrapper's forwarding behavior
against V17 reproduces the exception at the new pose-readback line. The local
scene-view fake returned plain NumPy-compatible arrays and missed this boundary.

The repair reads through `ProxyArray.warp.numpy()` (the explicit host-copy
path), handles native torch values via `detach().cpu().numpy()`, and retains
only source file/function/line locations for setup exceptions. Exception
messages, local variables, and absolute paths remain excluded. Camera aim,
scene writes, reset order, policy inputs, and the frozen matrix are unchanged.
Native image confirmation is still outstanding.

## Standard production mount

The ordinary wrist-camera path also installs the native scene-pose writer.
It reads the camera's authored local OpenGL pose after Isaac's convention
conversion and keeps that calibration fixed on the measured PhysX body.
It does not reset the environment during installation, use task coordinates,
or change the authored mount. Its receipt is separate from the explicitly
operator-directed aim receipt. Baseline visibility predicates remain required;
correct synchronization alone does not prove that a particular mount sees the
task object. Both paths share the same CUDA-safe scene writer and readback.

## Restored DROID mount and downward arm configuration

The operator rejected further camera re-aiming and requested the original
DROID wrist camera with the gripper pointed downward toward the book. The
next setup removes the operator camera-aim field, preserves the exact DROID
mount position, quaternion and intrinsics, and changes only the sealed arm
joint reset. Its gripper forward axis is vertical down; geometric projection
centers the book, while a peripheral sample can intersect a finger. This is
provisional setup evidence, not rendered visibility.

V18 (`7e292f7cda6657a036d89b9744adbf34e15e1dc9`) retained a nearly black
wrist frame whose semantic AOV labels every pixel as robot. Its reported
world camera centers the book, and the exact source joint chain matches the
measured native body pose within 9.4e-7. Rays using the original DROID mount
on that same arm pose are clear against the retained robot and scene meshes.
Those observations do not establish that the intended lens pose was inside
the housing; the earlier explanation was too strong. The rendered view and
its reported pose still need an independently verified binding.

The camera writer now authors and reads back the USD world transform as well
as the Fabric pose. Only the camera resets its USD transform stack; its
calibrated body-relative mount remains fixed and follows measured PhysX
poses every frame. Real OpenUSD tests with a deliberately stale parent
reproduce the predecessor's divergence between a correct Fabric readback
and a wrong USD world camera. The corrected writer requires both to agree.
Actual native frame visibility remains the acceptance criterion.

## Observed native wrist visibility, 2026-09-10

The frozen `67d50c1253f573111fd4453b84429a8cf90b96e5` run on provider
instance 50448155 retained a wrist RGB image of the open gray book. Independent
visual review identified the same page photographs and binding seam in the
external view. The wrist semantic AOV measured 419,398 task-object pixels
(45.5 percent), zero robot pixels, and a centered centroid. The book is clipped
at the top edge; full-object framing is not claimed. All three camera roles
passed the native observation gate. Joint reset error was below 4.3e-8 radians,
and the receipt binds the original DROID mount, USD and Fabric world readbacks,
and the actual renderer camera and render-product paths.

This establishes visibility for this setup, not an isolated causal comparison
between the transform-writer repair and the changed arm pose. Production reuse
preserves each admitted body-relative camera calibration, synchronizes from
measured native robot poses, and checks rendered visibility for each new site.
The book-specific starting joint angles are a site configuration. Neither policy
acted: a separate setup query-budget rounding defect was refused before the
first episode observation. Scientific outcome and qualified ranking remain
unproven.
