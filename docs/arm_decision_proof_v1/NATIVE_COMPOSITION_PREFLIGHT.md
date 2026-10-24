# Native support composition preflight

This closes an ADP-009 development-only construction gap on the day-14 public
rehearsal path: semantic visibility alone admitted a tray with background
ParticleField pixels breaking through its opaque surface. The completion
artifact is `prepolicy_observation_gate/native_asset_composition_gate/composition_gate.json`.

Before policy actions, the existing warmed scene captures full and native-mesh-only
passes at the exact reset external and overview cameras. It hides only scene
appearance; it never authors or clears visibility on live robot or task colliders.
The standalone attribution diagnostic retains its optional appearance-only pass;
the coverage gate does not need that pass or claim background-distance attribution.
The passes bind camera calibration, native state, renderer settings, lossless RGB,
semantic IDs and metric-distance AOVs. No physics steps or policy queries occur.
Visibility and full-scene sensor buffers are restored before returning.

The gate requires at least 64 support pixels after a two-pixel square erosion,
and blocks on any of those pixels missing from the full-scene support mask.
Silhouette differences remain recorded separately. Missing views, capture errors,
and changed native state fail closed. Native semantic packed-RGBA IDs use the
same decoder as policy-camera observability. Provider bundles include the full
import closure.

This is a coverage check for the recorded reset views. It does not prove that
all future poses are artifact-free, identify which Gaussians caused the pixels,
or distinguish legitimate foreground occlusion from an incorrect composition.
It never authorizes automatic Gaussian deletion. Any removal must use the
admitted SAM3.1/FlashSplat route; an occlusion-ordering repair must be verified
against retained full/mesh/background passes before promotion.

Asset packages also exclude exported lights and cameras from a new derived USD,
retain the original export, and record excluded prim attributes. Geometry beneath
such a prim causes refusal. Blender author exports disable lights, cameras and
world-material conversion. This lighting correction has not itself been proven
to eliminate background intrusion.

Validation: the focused composition/packaging tests cover visibility restoration,
fixed state, stale buffers, native RGBA labels, interior intrusion, and preserved
geometry/physics. The mandatory lifecycle rehearsal and isolated provider import
closure cover the modified pre-policy worker and shipped dependency boundary.

The September 22 website development rehearsal reached both policy servers and
native camera rendering, then failed after diagnostic edits invalidated a Robotiq
PhysX shape view. Removing robot visibility edits preserves the same full-versus-
mesh occlusion predicate. Overview also requests metric depth, and a final camera
refresh failure is retained alongside the original error instead of hiding it.
CPU USD-notice tests prove native colliders receive no visibility edits; the next
native run must still establish that the simulator failure is resolved. This is
ADP-009D day-14/day-28 development evidence, not captured-room integration proof.

For the ADP-009D Day-35 runtime comparison, the existing no-policy preflight
bundle can now bind a standalone three-pass request and an exact resolved cell
plan. This supplies the missing appearance-only view from a retained coverage
refusal without repeating policy dependency/checkpoint/server setup. Both files,
the seed, plan digest and original packet receipt must agree before bundling;
the worker verifies them again. The pinned baseline or GA image is selected from
the verified runtime-source profile, and receipt loading rechecks that image
against the original source receipt inside the immutable archive. The canonical
`native-task-arena-runtime-preflight` allocator, US/Ada selection, spend,
watchdog and teardown rules remain unchanged.

```bash
python -m blueprint_pipeline.native_task_arena_runtime_preflight_bundle \
  --job-dir NEW_BUNDLE --packet-dir RETAINED_PACKET \
  --runtime-source-packet-receipt EXACT_RUNTIME_SOURCE_RECEIPT \
  --implementation-commit IMMUTABLE_COMMIT \
  --composition-request COMPOSITION_REQUEST \
  --composition-scene-plan EXACT_RESOLVED_CELL_PLAN
```

This diagnostic remains no-motion, zero-policy and development-only. A captured
background distance AOV does not establish a Gaussian physical surface or
authorize deletion. The coverage gate and its any-interior-pixel refusal are
unchanged. Tests drive the actual sealed entrypoint and existing three-pass
capture with CPU USD/kernel stand-ins, verify its isolated import closure and
close order, and reject mismatched image, seed, plan, packet, bytes or incomplete
request pairs. Actual native attribution remains unproven until the admitted
diagnostic executes and its retained readbacks are reviewed.
