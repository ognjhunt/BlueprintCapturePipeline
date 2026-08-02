# Interactable captured scenes: approach decision (2026-08-02)

Question: how does a captured 3DGS scene become *interactable* for task/site
evaluation — able to carry contact, grasping, and articulation — without
surrendering measurement truth?

Evidence labels follow the 2026-08-01 routing research (VF verified fact, EC
external claim, INF inference). All sources below were fetched live on
2026-08-02.

## Approaches compared

| Approach | What it is | Evidence | Interaction validity ceiling | Verdict |
| --- | --- | --- | --- | --- |
| SimReady object conversion | Segment the task object, reconstruct/complete geometry, generate colliders + mass/friction/articulation, composite back into the splat scene (appearance stays 3DGS, physics slots swap to assets) | VF: `NVIDIA-Omniverse/usd-content-agents` is Apache-2.0, locally runnable, VLM-driven SimReady generation incl. physics classification and articulation inference; EC: Lightwheel SimReadyGen and Palatial offer the same conversion as cloud services (Palatial: per-part mass/friction/inertia "verified by drop and joint tests", USD/MJCF/URDF export) | Generated values are estimates; full validity after the collider-qualification / articulation-measurement / material-identification gates | **Adopted.** Implemented as `blueprint_pipeline.simready_asset_lane` |
| Learned world models | Predict scene dynamics from the splat/video directly | VF: the routing kernel caps `learned_world_model` at C4 and structurally denies collision/force/safety authority | Comparative policy ranking only; never contact truth | Rejected for interaction; retained for Q-WM ranking research |
| Direct splat physics (PhysGaussian-class) | MPM continuum simulation on the Gaussian kernels themselves | EC: research code; materials fitted, not identified; particle filling noisy-splat sensitive; no robot contact interface | Visual dynamics; fitted attributes are not force truth | Research watch item (cataloged `physgaussian-mpm-on-splats`), not the interaction lane |
| Mesh extraction (SuGaR/2DGS-class) | Extract an editable mesh from the splat, then simulate conventionally | VF: fast splat-to-mesh extraction with editing/rigging workflows | Extracted mesh is a derived geometry candidate, never a validated collider | Adopted **as an input source** for the SimReady lane's segmented-mesh path (cataloged `sugar-2dgs-mesh-extraction`) |
| Hybrid separation (Re3Sim-class) | Splat renders appearance; separately compiled colliders carry physics | VF: already this repository's architecture (collider qualification pipeline, InteriorGS placement) | Scene-level only; task objects still need per-object assets | Retained as the scene substrate the SimReady lane extends to object level |

## Decision

1. The **SimReady object-conversion lane is the interaction path**: the splat
   remains the appearance layer and per-object physics slots are filled by
   generated assets. There is no easy alternative that preserves measurement
   truth — world models are ceiling-capped by our own kernel, and direct splat
   physics carries fitted attributes.
2. **Generated is never validated.** Every SimReady estimate is flagged
   `estimated`, every candidate evidence record enters the site profile
   `validated=False`, and the router demonstrably abstains with
   `collider_validation` as the smallest next action until the existing gates
   pass (pinned by `tests/test_simready_asset_lane.py`).
3. **Generator preference order**: (a) the local geometry pipeline
   (segmented mesh -> watertight repair -> convex colliders -> density-class
   estimates), zero external exposure; (b) `nvidia-usd-content-agents` as the
   primary rich generator (Apache-2.0, locally runnable; VLM backend must be a
   local NIM or the site's privacy gate blocks it); (c) Lightwheel- and
   Palatial-class cloud providers behind the R2 contract/retention gates,
   planned but never called by this repository.
4. **Palatial reclassification**: live verification shows Palatial is a
   SimReady *generation* platform (captures to physics assets with drop/joint
   verification claims), not merely hosted interactive delivery; it is
   cataloged as `palatial-simready-cloud`, provider-gated, EC-labeled.
5. **Test before target-simulator admission.** Every local draft passes through
   a schema-checked structural/dynamics preflight. The exact local probe found
   trimesh, pxr/USD, and MuJoCo available, while Blender and NVIDIA's
   Content-Agent Validation executable are absent. Those optional agentic
   validators are recorded as typed unavailable—no install or provider call
   was attempted—and their absence is not disguised as a successful NVIDIA or
   Blender validation run.
6. **A movable object requires an exact Gaussian partition.** A whole-scene
   3DGS plus a second mug splat would render a duplicate. The implemented
   `gaussian_object_partition` path removes the selected rows from the static
   background, preserves them in an object-local standard-3DGS PLY, and binds
   that appearance to the SimReady body's pose through `dynamic_splat_scene`.
   The research and the 2025-2026 paper comparison are recorded in
   `docs/dynamic_3dgs_object_pipeline_research_2026-08-02.md`.

## Monitoring

All four new candidates (`nvidia-usd-content-agents`,
`palatial-simready-cloud`, `physgaussian-mpm-on-splats`,
`sugar-2dgs-mesh-extraction`) are research-catalog entries, so the release
watcher (`measurement_research_monitoring`) and snapshot monitor
(`measurement_research_monitor`) diff them automatically; version or access
changes propose R1 reverification or requalification through the admission
machinery, humans approving.
