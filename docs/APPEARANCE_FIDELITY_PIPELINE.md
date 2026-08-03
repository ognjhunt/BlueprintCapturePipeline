# Appearance Fidelity After Scaniverse

This lane improves captured-scene appearance without allowing a prettier
derivative to become stronger evidence than the capture. It applies to every
capture/reconstruction route that requests visual review, target discovery, or
policy-ranking support.

## Build on upstream tools

Blueprint does not implement a Gaussian editor, codec, renderer, or neural
enhancer. It orchestrates and qualifies replaceable upstream tools:

| Stage | Preferred upstream capability | Blueprint responsibility |
|---|---|---|
| Inspect, convert, clean | PlayCanvas `@playcanvas/splat-transform` | Pin runtime and input digest; preserve the source; record exact actions and before/after statistics. |
| Manual review | SuperSplat browser editor | Import a human-reviewed edit receipt; never silently reproduce manual edits. |
| Reconstruction retry | Scaniverse export baseline; Postshot or another qualified provider when source frames and poses exist | Route by input availability, held-out quality, rights, cost, and task/site requirements. |
| Faithful appearance render | Qualified native 3DGS renderer | Compare exact-camera output with source/reference frames and preserve SH degree and full anisotropic Gaussians. |
| Dynamics and collision | Qualified simulator selected for the task/site; commonly Isaac for Franka/G1 | Keep physics, collision, and robot state independent from appearance rendering. |
| Robot composite | Exact-camera, depth-aware compositor using the official simulator robot render | Bind frame, camera, asset, and runtime digests. The composite is visual support, not dynamics proof. |
| Optional enhancement | DiFix, ArtiFixer, Fixer, Harmonizer, or a declared successor | Mark output presentation-only and forbid it from evidence, policy observations, target binding, geometry, collision, and routing qualification. |

## Physics composition after a splat reconstruction

A Gaussian reconstruction is appearance, not automatically a SimReady physics
scene. Blueprint follows the same separation used by NVIDIA's smartphone-to-NuRec
reference workflow: preserve the reconstruction for rendering, then compose clean,
explicit physics support around it.

Use the smallest sufficient derived layer:

1. First test the digest-bound scan-derived collision candidate with no manufactured
   ground. This qualification lane answers whether the captured/derived collider
   itself supports contact; a proxy must never make that test pass.
2. For evaluation continuity, add a bounded floor proxy registered to the inferred
   floor band when the source mesh has holes or low outliers. Do not silently add an
   infinite plane. Record its transform, bounds, thickness, source-support samples,
   and exact source/placement digests.
3. Add a disclosed fixed-base mount for Franka. The mount is simulator support, not
   evidence of a physical pedestal, anchoring, load capacity, or installation.
4. For inspection-only tasks, the captured task object may remain appearance-only.
   For contact, articulation, or object-state-transition tasks, replace only the
   interaction zone with a digest-bound, independently validated SimReady asset.
   Qualify scale, site-to-asset transform, support, orientation, penetration,
   reprojection, materials, colliders, joints, limits, and physics properties.
5. If the scan-derived collider intersects the robot or task zone everywhere, do
   not run it simultaneously and call the result realistic. Route to an explicit
   proxy-composed evaluation lane: appearance splat plus clean floor/workcell/task
   assets. Preserve the failed source-collider qualification as separate evidence.

Policy ranks from a proxy-composed lane are valid only for that exact composed
simulation. They do not prove the captured site's floor continuity, task-object
dynamics, metric reach, or physical success.

The pinned SplatTransform CLI is the default headless cleanup adapter. It reads
Scaniverse SPZ and standard/compressed PLY, exposes statistics and qualified
filters, can render lossless images, and can produce voxel/collision candidates.
Its cleanup result remains unqualified until Blueprint's fidelity gate passes.
The SuperSplat application itself remains the browser-first manual editor.

## Required sequence

1. Hash and preserve the full-resolution provider export as immutable appearance
   truth. Never overwrite it.
2. Record representation-specific coordinates, bounds, up axis, transform, splat
   count, and spherical-harmonics degree. Cameras are bound to the exact asset
   digest and coordinate-basis digest; a PLY camera cannot silently be reused for
   a basis-converted SPZ.
3. Inspect before changing anything. A cleanup candidate may remove only
   nonfinite, independently qualified low-opacity, or robust spatial-outlier
   Gaussians. Global count-based decimation is forbidden.
4. Record the pinned upstream implementation digest, exact command/actions,
   source and output digests, source and retained counts, retained fraction,
   bounds, SH degree, and per-reason removal counts.
5. Render the unmodified source and every cleanup candidate with exact cameras at
   the requested output resolution. Require SSIM, PSNR, and LPIPS measurements
   against held-out source/reference frames before the candidate can qualify.
6. Select the best qualified appearance renderer independently from the dynamics
   engine. A simulator need not be the sole renderer.
7. Render the official Franka Panda by default, or the official G1 for humanoid
   tasks, from the selected simulator. Composite using exact camera and depth.
8. Run policy evaluation from simulator observations and qualified scene state,
   never from a generatively enhanced presentation frame.

## Tool routing

- Use SplatTransform first for lossless conversion, statistics, nonfinite
  filtering, reversible robust-bounds/floater candidates, and renderer or voxel
  conformance probes. Its `--decimate` action is prohibited in a qualified lane.
- Use Postshot only when original images/video frames and camera poses can be
  supplied. An exported splat alone does not contain enough source observations
  for a high-quality retraining comparison.
- Treat SuperSplat manual edits as reviewable candidates with an edit receipt.
- Treat DiFix/ArtiFixer/Fixer/Harmonizer outputs as presentation derivatives.
  Even a method that distills edits back into a 3D representation cannot become
  evaluation evidence without a future, separately approved contract change.
- Splat-derived voxel or collision meshes are candidates, not collision truth.
  They must pass metric-scale, coordinate, collider, and simulator-contact gates.
  The default headless generator is the pinned SplatTransform voxel/collision
  path; Blueprint should not implement a replacement splat mesher unless that
  upstream adapter cannot satisfy a measured requirement.

## Fail-closed cases

The route abstains when the source was overwritten, global decimation occurred,
removal counts do not reconcile, the retained fraction is below policy, SH bands
were discarded, the coordinate basis is unbound, the renderer is incompatible,
reference comparison is absent or below threshold, or an enhanced derivative is
present in an evaluation/policy input.

## Upstream references

- [PlayCanvas SplatTransform](https://github.com/playcanvas/splat-transform)
- [PlayCanvas SuperSplat](https://github.com/playcanvas/supersplat)
- [Postshot CLI](https://www.jawset.com/docs/d/Postshot%2BUser%2BGuide/Command-line%2BInterface)
- [NVIDIA DiFix3D+](https://research.nvidia.com/labs/toronto-ai/difix3d/)
- [NVIDIA ArtiFixer](https://research.nvidia.com/labs/sil/projects/artifixer/)
- [NVIDIA reconstruction enhancement workflow](https://developer.nvidia.com/blog/how-to-enhance-3d-gaussian-reconstruction-quality-for-simulation/)
- [NVIDIA smartphone reconstruction to Isaac workflow](https://developer.nvidia.com/blog/reconstruct-a-scene-in-nvidia-isaac-sim-using-only-a-smartphone/)
