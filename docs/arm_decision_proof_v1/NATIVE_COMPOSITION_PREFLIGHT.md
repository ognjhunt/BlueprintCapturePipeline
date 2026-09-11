# Native support composition preflight

This closes an ADP-009 development-only construction gap on the day-14 public
rehearsal path: semantic visibility alone admitted a tray with background
ParticleField pixels breaking through its opaque surface. The completion
artifact is `prepolicy_observation_gate/native_asset_composition_gate/composition_gate.json`.

Before policy loading, the existing warmed scene captures full, appearance-only,
and native-mesh-only passes at the exact reset external and overview cameras.
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
