# Dynamic 3DGS objects without ghost duplicates (2026-08-02)

## Decision

A captured mug does not stay inside one immutable scene splat if it must move.
Its selected Gaussians are partitioned into a separate object-local PLY, and
the same rows are omitted from a new static-background PLY. At runtime the
background is loaded once and the mug PLY is loaded once. The mug's visual
transform and its SimReady collider both consume one simulator body-pose
channel.

This is an exact mechanical partition, not proof that the perception mask
contains every and only real mug primitive. Object membership remains a
candidate until independently evaluated. Collider, mass, friction, and
articulation remain unvalidated SimReady estimates until their existing gates
pass.

## Plain-English mug example

1. Three or more calibrated camera views mark the mug in 2D.
2. The contribution/visibility step identifies candidate mug Gaussians. A
   footprint-aware depth test rejects wall or counter Gaussians merely seen
   behind the 2D mask.
3. The source PLY is split by exact row ID. If the source has 1,000,000
   Gaussians and the selection has 8,000, the background gets exactly 992,000
   and the object gets exactly 8,000. The sets are disjoint and exhaustive.
4. The mug Gaussians are recentered into a local object frame while preserving
   opacity, scale, rotation, DC color, and all higher spherical-harmonic bands.
5. The mug's local visual frame is bound to its SimReady body/collider frame.
   Every simulator frame supplies one `T_world_body`; the renderer derives
   `T_world_appearance = T_world_body * T_body_appearance`.
6. Spark.js loads one background `SplatMesh` and one mug `SplatMesh`. Moving the
   body moves the latter. The former cannot show the original mug because its
   PLY does not contain the selected rows.

The old scene-binding form without a Gaussian partition is now explicitly
`scene_gaussian_splat_static_only`; it cannot claim duplicate-free dynamic
rendering.

## Current paper and implementation evidence

| Work | What the primary source supports | Blueprint use |
| --- | --- | --- |
| [FlashSplat, ECCV 2024](https://arxiv.org/abs/2409.08270) ([official code](https://github.com/florinshen/FlashSplat)) | Lifts 2D masks into globally optimized Gaussian labels using the renderer's alpha-blending structure; demonstrates object removal/editing. | Preferred replaceable high-quality selection adapter. Blueprint's local CPU baseline is deliberately not labeled FlashSplat. |
| [SAGA / Segment Any 3D Gaussians](https://jumpat.github.io/SAGA/) ([official code](https://github.com/Jumpat/SegAnyGAussians)) | Scale-aware promptable Gaussian segmentation. | Candidate interactive segmentation adapter. |
| [SAGOnline, arXiv 2025](https://arxiv.org/abs/2508.08219) | Online zero-shot Gaussian instance segmentation using SAM2 mask propagation and Gaussian-level instance identities. | Candidate streaming/online adapter. |
| [SAGO, arXiv 2026](https://arxiv.org/abs/2607.01628) | Setup-free, interactive segmentation with virtual views and reported sub-second interaction. | Newest watched interactive segmentation route; not an implemented or qualified dependency. |
| [ReferSplat, ICML 2025](https://proceedings.mlr.press/v267/he25h.html) | Language-referred 3D Gaussian segmentation. | Candidate text-to-object-selection front end. |
| [Splat-MOVER / SEE-Splat](https://splatmover.github.io/) ([paper](https://arxiv.org/abs/2405.04378)) | Robotics pipeline that creates a semantic Gaussian mask, transforms selected Gaussians, infills the revealed scene, and reports Kinova hardware experiments. | Closest established manipulation precedent; supports the split-transform-infill architecture, not Blueprint physical success. |
| [CubifyGS, arXiv June 2026](https://arxiv.org/abs/2606.28720) | Builds object-level Gaussian assets for rigid transforms and uses explicit pruning to suppress ghosting. | Strong direct support for the exact remove-then-reinsert decision. |
| [MRO-GWM, arXiv June 2026](https://arxiv.org/abs/2606.01950) | Uses canonical object Gaussians with rigid-body transforms in an object-centric world model. | Supports object-local canonical frames; world-model predictions still have no physics authority. |
| [R5DGS, arXiv May 2026](https://arxiv.org/abs/2605.25909) | Preserves semantic identity and propagates rigid transforms from object motion. | Supports identity-preserving rigid transform composition. |
| [RiGS, arXiv May 2026](https://arxiv.org/abs/2605.23672) | Separates static, rigidly moving, and transient Gaussian components. | Supports explicit static/dynamic decomposition. |
| [PersistGS, arXiv June 2026](https://arxiv.org/abs/2606.03479) | Represents objects with Gaussians, collision meshes, and SE(3) physics trajectories. | Closest recent precedent for appearance + collision geometry driven by a shared pose. |

## Revealed background and inpainting

Removing the mug can expose a region that no camera ever observed. Blueprint
does not silently fill that region and call it capture truth. The implemented
partition records zero generated Gaussians. If observed neighboring views
already contain the surface, a later deterministic completion adapter may use
them. Otherwise an inpainted patch remains generated support, with provenance
and a lower claim ceiling.

Relevant current methods include [3DGIC, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_3D_Gaussian_Inpainting_with_Depth-Guided_Cross-View_Consistency_CVPR_2025_paper.html),
[Inpaint360GS, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Wang_Inpaint360GS_Efficient_Object-Aware_3D_Inpainting_via_Gaussian_Splatting_for_360deg_WACV_2026_paper.html),
[GPGS, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37515),
and [GOR-IS, arXiv May 2026](https://arxiv.org/abs/2605.00498). They are
candidate completion adapters, not authority upgrades.

## Implemented proof surface

- `gaussian_object_selection.v1` binds object membership to the exact source
  PLY digest and ordered row IDs.
- `gaussian_object_partition.v1` writes and re-verifies the background/object
  PLY digests, counts, disjointness, exhaustiveness, and object frame.
- `dynamic_splat_scene.v1` binds exactly one visual instance and one collider
  instance to one pose channel.
- `dynamic_splat_render_request.v1` resolves a particular simulator frame into
  one transform per object.
- The real local Spark.js compositor renders the initial and moved fixture. A
  pixel test verifies the red mug moves 60 cm and does not remain at its old
  image location.

That rendered test establishes the fixture's composition behavior. It does not
establish semantic segmentation quality on a real capture, collider accuracy,
physical realism, robot-camera parity, grasp success, or sim-to-real transfer.
