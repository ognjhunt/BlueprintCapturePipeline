# Website source frames

ADP-009B / public day 14; beta journey step: concurrent scene preparation.
`website_scene_geometry` decodes a website capture's source video on the CPU
into the frames that SAM tracking, image editing and Atlas posing read. It
needs no model weights and no GPU, and never claims measured scale or physical
evidence. MapAnything, which used to estimate depth on these frames, was
removed on 2026-10-22; poses and depth come from World Labs Atlas
([website-atlas-scene-source.md](website-atlas-scene-source.md)).

An unconfirmed task context must keep its typed `scene_preparation` purpose,
identity and digest. Preparation preserves `confirmed=false` and the customer's
criteria, and grants no spending authority.

## Sampling

Source sampling is independent of provider capacity:

- The preparation CLI accepts `--maximum-frames`; Python callers accept
  `maximum_frames`; the service may set `BLUEPRINT_WEBSITE_SOURCE_MAXIMUM_FRAMES`.
  The default stays 16.
- The requested count is bound into new artifact identity. The unchanged
  default can verify and reuse retained inputs without decoding again,
  including adapter-v1 inputs that still name the model they were first
  prepared for.
- Hidden reconstruction holdouts keep their existing split semantics.
- Source sample count, held-out count, planned provider inputs and actual
  provider inputs are separate fields.

## The pending Atlas route

`run_website_scene_geometry` records `atlas_pose_inputs.json` as **pending**,
with zero API mutation. The admitted producer is
`paid_resource_allocator.run_sponsored_website_atlas_pose`
([website-atlas-pose-operation.md](website-atlas-pose-operation.md)). It needs:

- Atlas access;
- a reviewed Atlas price;
- a provider-retention decision;
- the WebApp-signed spend grant, reserved under its own `world_labs_atlas`
  identity.

A retained `atlas_posed_rgbd_result.json` (`website_atlas_pose_result.v1`) carries
a canonical `digest`, the exact source and input `binding`, the documented pose
`endpoint` and the real completed `operation`. The adapter validates the
returned frame count and RGB/depth/camera bundles and records
`atlas_posed_rgbd_handoff.json`. Explicit `target_cameras` and an optional
`prompt` also prepare `atlas_generation_inputs.json` for `atlasGenerate`, without
resizing pixels, relabelling axes, assuming measured scale or making a request.
Input usage is reported only from the producer's `operation_receipt`, and only
while that receipt still binds these frames and its retained output bytes.

See [images2PosedRGBD](https://atlas-beta.worldlabs.ai/docs/images-to-posed-images),
[cameras and posed images](https://atlas-beta.worldlabs.ai/docs/cameras-and-posed-images)
and the [API reference](https://atlas-beta.worldlabs.ai/docs/api-reference).

## Deploy notes kept from the former MapAnything controller page

- The canonical control-plane deploy installs the agent-run dispatcher service
  and timer, and restores the timer after deployment or reboot. The service
  still requires `BLUEPRINT_AGENT_RUN_DISPATCH_ENABLED=true` and its configured
  capture scope; installing the timer alone does not authorise a run. Its Python
  entrypoint is checked by the production startup guard.
- CAD generation and repair share the pinned `gen_step()` shape-return contract:
  the CAD CLI owns exports, and the agent returns geometry rather than a
  filename or dictionary.
