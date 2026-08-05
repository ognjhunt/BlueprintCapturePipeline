# ADP-009 Public Evidence Ladder And SimReady Replacement Test Program

Status: **active execution strategy**
Approved: 2026-08-04
Backlog item: **ADP-009**
Phase gates: public-scene day 7 / 14 / 21 / 28
Observed completion artifact: one digest-bound public-scene qualification matrix
with exact inputs, rights, code revisions, transformations, metrics, failures,
claim ceilings, and replay commands.

Observed tranche state (2026-08-05): the deterministic ten-role index admits
the matched InteriorGS/SAGE-3D `840313` pair, the bounded NVIDIA USD Content
Agents v0.5.2 authoring comparison, and AuraFusion360's unchanged author-data
smoke. Six roles remain explicitly blocked, so ADP-009A is partial. Aura's
publisher workflow completed with unmodified source and retained its produced
point cloud and same-camera comparison frames; that author control is not an
InteriorGS result. Material, Texture, Physics, and static Validation Agents
executed on the parametric replacement control; this does not constitute scene
inpainting, dynamic Isaac qualification, metric truth, or physical evidence.

## Decision

Use the new datasets and test 3DGS object removal, background inpainting, and
replacement by an exact SimReady USD before any fresh-site capture.

The public program is not one dataset and not one monolithic benchmark. It is a
composed evidence ladder because no released public resource supplies all of:

- real capture truth;
- metrically authoritative geometry;
- a high-quality 3DGS;
- matched collision geometry;
- an editable object and known hidden background;
- a physics-authored task object;
- a fixed-arm task and reset;
- two frozen policies;
- matched physical outcomes.

The first falsifiable deliverable is therefore:

> For one exact rights-admitted InteriorGS/SAGE-3D scene, remove one annotated
> source object from both the appearance and collision representations, run the
> exact released InFusion path through a preregistered Blueprint interface
> adapter to complete the background, insert one
> digest-bound SimReady USD with explicit visual mesh, collider, metric size,
> pose, mass, inertia, friction, and restitution, load the result in Isaac Sim,
> and produce visual, frame, collision, contact, dynamics, media, and replay
> receipts. Then repeat the editing path with independent metric-surface scoring
> on one admitted ScanNet++ scene. Inpaint360GS remains an unchanged author-data
> reproducibility control rather than the InteriorGS adapter.

The public outputs remain `development_only`. The final acquisition phase is one
fresh, previously unseen, rights-cleared site with observed clean-background and
object-present evidence. The prospective decision and physical holdout still
follow that capture.

## Why ADP-009 Is Necessary

ADP-008 is complete. It established two-candidate execution, receipts, a
software outcome firebreak, sealing, external outcome joining, and replay on
SIMPLER. It did not establish:

- capture-to-metric-scene transfer;
- authoritative scale or local surface accuracy;
- 3DGS-to-mesh/collision registration;
- object selection and exact partitioning;
- factual or plausible background completion;
- removal of a source collider without leaving ghost physics;
- insertion of a measured SimReady USD;
- contact, grasp, or dynamics fidelity;
- decision stability or abstention under scene-construction variation.

The smallest reversible change is additive: preserve every historical ADP-008
artifact and add a public-scene manifest, thin dataset/method adapters, a hybrid
scene composition contract, and the tests below. Do not reinterpret SIMPLER as
a scene-reconstruction result.

## First-Principles Model

The scene is aligned but independently qualified things. The authoritative
geometry differs by dataset:

```text
InteriorGS 3DGS + metadata -> synthetic appearance and metric frame/OBB oracle
matching SAGE-3D USD       -> static collision proxy
rendered RGB + cameras     -> synthetic method inputs, never independent evidence

ScanNet++ RGB + aligned poses + laser/depth -> independent metric surface authority

registered 3DGS appearance -> remove source-object Gaussians
                                  |
                                  +-> released-code candidate completion

exact SimReady USD task object -> visual mesh + collider + physical properties

all representations -> one digest-bound source-to-simulator metric transform
```

A Gaussian covariance is a rendering parameter, not a physical surface. The
published InteriorGS/SAGE pairing does not include a measurement-authoritative
local surface mesh; its metric OBBs and navigation-oriented collision proxy do
not create one. A SAM mask is semantic support, not a ruler. DA3 depth is a
learned estimate, not qualification truth. A collision mesh is not qualified
because it loads. A generated background is not a recovered fact merely because
it looks plausible.

## Audit Of The Submitted Research Report

The report's central architectural conclusion is correct: use 3DGS for
appearance, explicit meshes/USD for physics, and retain a metric transform.
Several readiness claims need tightening.

| Claim | Audit result | Blueprint consequence |
| --- | --- | --- |
| InteriorGS is metric | Its publisher declares meters, right-handed Z-up coordinates, semantic boxes, occupancy, and floorplans, but the release does not provide a measurement-authoritative local surface mesh. | Use it as a synthetic coordinate/semantic/OBB oracle, not local-surface metrology. |
| SAGE-3D makes the scenes simulation-ready | The project releases matching USDZ appearance and collision assets for Isaac, but the benchmark is navigation-oriented. Static collision bodies do not provide a high-fidelity surface oracle, task-object mass, inertia, materials, reset, or manipulation contact qualification. | Use it for exact scene pairing and static collision control; replace the chosen movable object with our own SimReady USD. |
| ScanNet++ is the strongest real option | Correct among the audited resources: official aligned DSLR poses are metric, laser geometry is available, and iPhone depth is stored in millimeters. It does not guarantee a suitable complete apple or ready-made splat. | Train the splat from the official metric poses; measure and score against laser/depth, never Gaussian extent. |
| SceneSplat gives ready large-scale metric data | It aggregates heterogeneous sources. An average depth error in meters is not a per-scene unit, frame, normalization, rights, or collision certificate. | Do not make SceneSplat-7K/49K an ADP-v1 dependency. Admit a source-specific scene only if a measured variation gap remains. |
| SAM 3.1 plus DA3 provides a metric apple | No. SAM proposes pixels; DA3 predicts geometry. Neither creates observed physical dimensions. | SAM may propose masks; DA3 may cross-check or identify a missing measurement; neither may authorize metric or collision claims. |
| The listed 2026 editing stack is executable | Only some cited methods have released implementations. Several project names have papers/pages but no runnable official code. | Critical-path admission requires exact code revision, dependencies, weights, license, and a smoke receipt. |
| Inpaint360GS is the closest adapter for publisher splats | No. Its released custom-data path reconstructs/trains its own representation and its virtual-view/depth stages are method-internal. | Run its unchanged author-data workflow as a reproducibility control; do not present a Blueprint render adapter as stock Inpaint360GS. |
| InFusion directly solves our interface | It is the closest released interface shape audited: depth, camera-to-world, and intrinsics produce a supplemental PLY that is explicitly composed with an original PLY. It still needs exact format, license, coordinate-frame, mask, and checkpoint admission. | Make a thin InFusion adapter the primary InteriorGS completion candidate, while keeping predicted hidden depth `visual_candidate_only`. |
| AuraFusion360 is drop-in for InteriorGS | Its released 360 pipeline starts by training object-masked Gaussians and then operates on its own checkpoint/config representation. | Admit it as a modern quality challenger only after a custom checkpoint/format adapter passes. |
| Inpainting recovers the hidden counter | Only when the surface is observed elsewhere or in a second state. Otherwise it creates a candidate completion. | Score visual plausibility separately from factual recovery, and obtain clean-background observations in the final capture. |

Primary sources:

- [InteriorGS dataset and terms](https://huggingface.co/datasets/spatialverse/InteriorGS)
- [SAGE-3D project](https://sage-3d.github.io/), [USDZ scenes](https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz), and [collision meshes](https://huggingface.co/datasets/spatialverse/SAGE-3D_Collision_Mesh)
- [ScanNet++ documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation) and [official 3DGS example](https://github.com/scannetpp/3DGS-demo)
- [SceneSplat repository](https://github.com/unique1i/SceneSplat)
- [Inpaint360GS official repository](https://github.com/dfki-av/Inpaint360GS)
- [InFusion official repository](https://github.com/ali-vilab/Infusion)
- [AuraFusion360 official repository](https://github.com/kkennethwu/AuraFusion360_official)

## Exact Dataset Stack

The required dataset order is exact: (1) one rights-admitted InteriorGS scene
plus its matching SAGE-3D scene/collider, then (2) one rights-admitted
ScanNet++ scene. ARKitScenes, WildRGB-D, a heterogeneous aggregate, or an
authored control cannot substitute for either required dataset.

### 0. SIMPLER: retained decision-harness proof

Keep the immutable ADP-008 replay. It qualifies the decision/outcome machinery,
not scene construction. It is not rerun or re-labeled for ADP-009.

### 1. InteriorGS plus SAGE-3D: preferred synthetic editing control after rights

Admit one exact scene ID from all three stores:

- compressed InteriorGS PLY and semantic sidecars;
- matching SAGE-3D USDZ appearance scene;
- matching SAGE-3D collision USD.

This is the default paired-splat/collision control after the exact terms are
accepted and recorded. The current SAGE-3D assets are CC-BY-NC-4.0 and
InteriorGS has custom gated terms, so they are research evidence and cannot
silently become a commercial runtime dependency. If either exact source cannot
be rights-admitted, the required InteriorGS/SAGE gate stops with that typed
rights outcome. An independently authored scene may run as a separate positive
control, but it never substitutes for or completes the public-dataset gate.

Select one rigid object that has:

- a stable semantic/instance ID and metric oriented box;
- a matching collision body that can be identified and removed;
- a supporting surface suitable for contact probes;
- sufficient frozen calibration and test trajectories for boundary and
  occlusion evaluation;
- dimensions within the intended fixed-arm task envelope.

The object need not be an apple. The scientific unit is “one rigid source object
replaced by one exact SimReady USD.” An apple-specific demonstration may follow
only if an admitted scene supports it.

InteriorGS/SAGE provides a metric shared frame, semantic OBBs, and a static
collision proxy. It does **not** provide a measurement-authoritative local
surface mesh. Rendered depth from the splat or SAGE collider is therefore not
surface truth; it may support debugging and collision-aware camera placement.

### 2. Inpaint360GS: unchanged author-data reproducibility control

[Inpaint360GS](https://github.com/dfki-av/Inpaint360GS) released code, data, and
results. Admit one exact commit, every nested dependency and license, the exact
weights, and one unchanged author-dataset smoke as the upstream reproducibility
control.

The repository's top-level license is Apache-2.0, but that does not authorize
every vendored/submodule dependency. The admission record must inventory the
Gaussian-splatting, segmentation, tracking, LaMa, depth, and evaluation
dependencies individually.

Its custom-data preparation, retraining, virtual-camera generation, and
method-derived depth do not match the desired `existing PLY + known cameras +
mask` boundary. Run the exact unchanged author-data workflow only. That smoke
answers whether the pinned upstream code reproduces; it does not complete the
InteriorGS edit and may not be relabeled as a Blueprint interface test.

### 3. InFusion: primary Blueprint interface-adapter candidate

The released [InFusion repository](https://github.com/ali-vilab/Infusion) is the
closest audited code boundary to Blueprint's hybrid scene. Its inference stage
accepts an RGB result, mask, depth, camera-to-world matrix, and intrinsics,
creates a supplemental PLY in the supplied world frame, and its published
`compose.py` explicitly combines that PLY with an original Gaussian PLY before
refinement.

The repository root is MIT, but the released checkpoint currently has no
declared model-card license and bundled Gaussian-renderer code carries separate
noncommercial terms. The author compositor also hardcodes spherical-harmonic
degree zero; it is not safe for a publisher PLY with higher-order SH fields.

That compatibility is not automatic. Before the InteriorGS run, preregister and
test a thin adapter for:

1. publisher PLY versus InFusion Gaussian attributes, including an SH-preserving
   compositor rather than the author's degree-zero-only merge;
2. world/camera convention, handedness, axes, units, and matrix direction;
3. mask polarity, pixel origin, resolution, and depth convention;
4. exact checkpoint and all nested code/model licenses;
5. object-present source-Ply partitioning and explicit composition locality;
6. preservation of the original metric frame after supplemental-Ply insertion.

Freeze collision-aware camera intrinsics and poses, then render object-present
RGB from InteriorGS. Those images and cameras are
`render_derived_synthetic_method_inputs`: they test the interface and
self-consistency, never independent observation evidence. In the first adapter,
use InFusion's method-native incomplete-Gaussian depth as its depth input and
keep any external metric depth separate as a validation oracle. Supplying an
external validation-depth product is outside this default profile; it requires
a separate fork/profile and its own admission rather than a silent substitution.

InFusion predicts the hidden depth and lifts supplemental Gaussians; it cannot
recover an unobserved true surface. Its completion remains
`visual_candidate_only`. Any clean-background truth stays inaccessible until the
result digest is sealed.

### 4. AuraFusion360: modern 360-quality challenger

[AuraFusion360](https://github.com/kkennethwu/AuraFusion360_official) has a
released full-code pipeline for depth-aware unseen masks, Gaussian
initialization, and multiview appearance refinement. It is the primary quality
challenger after InFusion because the selected InteriorGS case is a 360-degree
scene.

It is not a drop-in PLY editor: the released workflow begins by training
object-masked Gaussians, writes custom `is_masked_*` PLY fields, and then
consumes its own configs/checkpoints. Its root is Apache-2.0 while copied core
Gaussian dependencies retain separate noncommercial terms. Run it
only after an exact custom adapter proves representation, camera, mask, frame,
and checkpoint compatibility. Report it as a separate branch, never as the
unchanged author path or an in-place publisher-Ply edit.

### 5. Conditional research ablations

- [3DGIC](https://github.com/peterjohnsonhuang/3dgic): its official repository
  explicitly describes the released path as suboptimal and requires a custom
  representation; the full implementation remains with the sponsor, and the
  released license is noncommercial;
- [GPGS](https://github.com/yongjoon99/GPGS): geometry-aware candidate, but its
  own RGB/COLMAP/Point-MAE workflow and missing root license keep it off the
  default adapter path;
- [GS PatchMatch](https://github.com/adriaanpardoel/gs-patchmatch): directly
  edits an existing PLY using a supporting mesh and is a useful deterministic
  repetitive-countertop control, but it is structure-weak and lacks an
  admissible root license;
- [CoIn](https://github.com/Yuheng2000/CoIn): official code exists, but the
  released removal path retrains from COLMAP/learned depth, object insertion is
  unfinished, and the repository lacks a root license;
- [GOR-IS](https://github.com/applezyh/GOR-IS): specialized intrinsic/light-
  transport representation for a preregistered glossy, reflection, or cast-
  shadow case; separate admission and noncommercial-use review are required.

These are bounded ablations, not silent fallbacks and not substitutes for the
InFusion primary or AuraFusion360 challenger.

### 6. NVIDIA USD Content Agents: candidate SimReady authoring backend

[NVIDIA USD Content Agents](https://github.com/NVIDIA-Omniverse/usd-content-agents)
v0.5.2 is released Apache-2.0 code with Codex/Claude workflow skills, existing
CAD/mesh/URDF/MJCF-to-USD conversion, material/physics agents, and SimReady
validation. It is **not** an image-to-CAD or splat-to-CAD generator. Its physics
predictions and VLM judges are proposals, not measured property authority.

Use it behind a provider-neutral `simready_authoring_backend` contract after
geometry already exists. The first canary is a checked-in parametric rigid
object with analytic dimensions, volume, center of mass, and inertia. Bind the
exact Content Agents, converter, SimReady profile, model/backend, renderer,
solver, prompt/config, source/output, and environment digests. Require
`Prop-Robotics-Isaac@1.0.0` explicitly and run dynamic Isaac checks separately;
static SimReady validation is not task physics.

Compare:

- known-good manually authored USD control;
- deterministic parametric CAD plus NVIDIA Content Agents;
- SimReadyGen only if its exact code, rights, and runtime remain admitted.

Promote the Content Agents path only if it preserves geometry and satisfies the
same metric, collision, dynamics, contact, grasp, repeatability, latency, cost,
and human-correction gates. A Codex- or Claude-authored CAD script is acceptable
only when the script, parameters, CAD kernel, and output are immutable and the
geometry passes independent tests.

### 7. ScanNet++: targeted real measured transfer

After account/access and noncommercial terms are explicitly accepted, select one
train/validation scene with:

- laser mesh/point cloud;
- aligned metric DSLR poses and frozen calibration/test DSLR trajectories;
- usable iPhone RGB-D when available;
- a suitable supporting surface and rigid target region;
- no test-split leakage.

Train the appearance splat with the official aligned poses. Use laser mesh or
rendered source depth for measurements. Do not rerun arbitrary-scale COLMAP and
do not rescale from Gaussian extents.

If the scene has no object/background counterfactual, the removal result can
earn only visual-plausibility evidence. Factual recovery is tested separately by
masking an observed background patch or using a separately admitted controlled
object-present/clean-background pair. The final fresh capture supplies the real
object-present/clean-background pair.

### Excluded from ADP v1

- ARKitScenes: useful consumer-capture breadth, but its mobile geometry is below
  the quality target of this bounded rehearsal; phone-capture behavior moves to
  Blueprint's final Raw V3.2 site capture;
- WildRGB-D: useful object-only research, but it is not a site-level
  splat/collision/editing substrate and does not justify another active lane;
- SceneSplat-7K/49K: too heterogeneous and large for the present question;
- Realsee3D/Argus: useful future panoramic research, no ready splat/collider/task;
- paper-only editing methods: no critical-path integration until official code,
  weights, license, and a reproducible smoke exist;
- Postshot/NuRec: optional proprietary adapters only, never the sole scientific
  authority or required reproduction path.

## Released-Code Rule

A method is “code out” only when the admission packet records:

1. a retrievable official repository and exact commit;
2. all submodule and dependency commits;
3. exact model/weight identities and digests;
4. code, model, and dataset license/allowed-use records;
5. immutable inputs and a reproducible smoke command;
6. expected outputs and a digest-bound smoke receipt;
7. hardware/runtime requirements and a zero-spend preflight;
8. a declared claim ceiling.

The 2026-08-04 interface audit inspected these candidate heads; they are audit
identities, not admission receipts:

| Role | Project | Audited commit | Blocking admission finding |
| --- | --- | --- | --- |
| Reproducibility control | Inpaint360GS | `d54c893285c6cb27788e05cce607e7d3cca6388a` | own training/distillation path; nested dependency licenses |
| Primary interface adapter | InFusion | `788da7f40cad4314831a053b7419df277d7814c4` | checkpoint license, Graphdeco terms, SH0-only author compositor |
| Multiview challenger | AuraFusion360 | `f23b26c44ba84608306ba952510533ebf4c7877d` | custom PLY/checkpoint adapter and nested dependency licenses |

A paper, project page, pseudocode, announced code release, or arXiv revision does
not pass this gate. As of this audit, InFusion, AuraFusion360, 3DGIC, GPGS,
GS PatchMatch, CoIn, and GOR-IS have official repositories, subject to exact
dependency and rights admission.
InFusion is the primary interface-adapter candidate; AuraFusion360 is the
360-quality challenger. The others remain conditional or blocked by interface,
completeness, or license findings. TRACE, LightHarmony3D, ArtisanGS, BEA-GS,
GSCompleter, 3D-GIMP, FocusGS, D3DR, and UniMGS remain excluded because an
official runnable implementation was not verified.

SAM 3.1 and DA3 have released official implementations, but their roles remain
bounded. [SAM 3.1](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
may propose/track masks. [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
may cross-check depth/poses and trigger abstention. Neither is metric authority.

## Step-By-Step Qualification Tests

Every threshold below is preregistered from the task tolerance, sensor
uncertainty, object clearance, gripper geometry, and required contact precision.
Paper-average PSNR or depth scores do not set Blueprint thresholds.

### Gate 0 — Exact admission

Inputs: one component manifest per coherent executable case, plus the
[`public_scene_suite_index.v1`](../schemas/public_scene_suite_index.v1.schema.json)
index that binds the required component admission receipts and their exact
roles. A component
receipt can authorize materialization of that one case; it cannot mark the
ADP-009A matrix complete.

Pass only if:

- allowed use and reviewer/authority are recorded;
- exact source, scene ID, revision, file size, and SHA-256 are bound;
- source units, handedness, up axis, origin, normalization, and inverse transform
  are explicit;
- appearance, semantics/OBBs, collision, measurement-authoritative geometry
  where the dataset actually provides it, and task-object roles are separate;
- the observation bundle binds its exact camera model, render/materialization
  dependency, active appearance and collision artifacts, object-present method
  inputs, separately identified validation oracles, withheld clean-background
  truth, and no unscaled SfM rerun;
- each released method runs with an allowlist-only artifact mount; a manifest
  flag alone does not protect truth if the process can read the whole suite;
- calibration and test trajectory IDs are digest-bound and disjoint;
- exact code/weights/dependencies pass a smoke;
- the reproducibility control binds one exact unchanged Inpaint360GS revision;
- the InteriorGS adapter binds exact InFusion code plus the Blueprint adapter,
  with AuraFusion360 admitted separately as the challenger;
- the replacement is one independently versioned, digest-bound USD package
  whose visual, collision, and physics payloads come from that same SimReady
  object source;
- the claim ceiling is `development_only`.

Negative tests deliberately fail missing rights, changed digests, mismatched
InteriorGS/SAGE IDs, unknown units/up axis, missing inverse normalization,
paper-only code, DA3-as-scale-authority, and attempted claim elevation.

### Gate 1 — Metric frame and cross-modal registration

Tests:

1. source-to-Blueprint-to-Isaac transform round trip;
2. determinant/handedness/up-axis checks;
3. known-distance and room-extent residuals;
4. camera reprojection against test-trajectory landmarks;
5. source mesh/depth versus registered appearance depth only on ScanNet++ or
   another source that actually provides measurement-authoritative surface
   geometry;
6. support-plane height/orientation;
7. object OBB center/size residual against the source oracle;
8. deliberate `cm`/`m`, `x0.1`/`x10`, axis-swap, and reflection mutations.

For InteriorGS/SAGE, support-plane and object tests qualify consistency with the
published OBB/collision proxy only; they do not establish local physical surface
accuracy. ScanNet++ laser/depth supplies that independent surface test.

Silent recentering or scale repair is failure. A mutation must either be rejected
or produce the exact typed missing measurement.

### Gate 2 — Object selection and exact partition

Tests:

1. multiview mask precision/recall and temporal/identity consistency;
2. stem/contact/boundary error review where applicable;
3. mask-to-depth agreement only when the named depth source is independently
   authoritative; otherwise mask-to-OBB and multiview consistency;
4. Gaussian partition conservation: every source Gaussian appears exactly once
   in background, object, or declared uncertain boundary;
5. semantic/OBB containment and boundary contamination;
6. source-collider identity and exact removal binding;
7. no background collider or Gaussian is removed by object ID drift.

SAM output is a proposal. The admitted partition is a digest-bound artifact with
reviewed uncertainty.

### Gate 3 — Removal and background completion

Produce the default outputs:

1. unedited source scene;
2. source object removed without completion;
3. InFusion-completed scene through the preregistered Blueprint adapter;
4. AuraFusion360 completion through its separately verified adapter.

The unchanged Inpaint360GS author-data output is a separate reproducibility
control. GPGS and GOR-IS outputs exist only for preregistered conditional
ablations.

Score three evidence levels separately.

**Frozen test trajectories — visual plausibility and locality only**

- PSNR, SSIM, and LPIPS only outside the removal mask for edit locality;
- boundary-band continuity and cross-view consistency without pretending the
  object-present pixels are the desired completed background;
- metric depth consistency only where observed geometry exists;
- cross-view feature/warp consistency;
- halo, floater, ghost-object, texture-swim, shadow, and reflection residuals;
- edit locality outside the dilated removal volume;
- render completeness from every frozen camera;
- blinded human/no-reference artifact review for truly hidden regions.

**Controlled known-background benchmark — factual recovery**

- insert a synthetic occluder over retained real/authored RGB/depth, then run the
  same selection, removal, and completion path without access to the clean truth;
- run the exact same completion path without access to the withheld truth;
- score RGB, depth, plane/surface residual, boundary continuity, and false
  geometry against the withheld original;
- release truth only after the completion digest is sealed.

**Fresh-site clean background — factual recovery**

- use separately observed clean-background frames registered to the
  object-present capture;
- never call a generated region factual if the clean observation is absent.

### Gate 4 — SimReady USD replacement

The replacement asset must be a distinct `simready_task_object` source and bind:

- exact USD and dependency digests;
- visual mesh and simplified collision mesh;
- prescribed or independently measured width/depth/height and uncertainty;
- source-world and simulator pose;
- support/contact point;
- mass, center of mass, inertia tensor;
- friction, restitution, and material identity;
- semantic/task/reset metadata.

Asset construction is an independently scored branch. Deterministic measured or
parametric geometry is the authority. NVIDIA Content Agents may convert,
materialize, author supported physics fields, and validate the candidate; any
missing COM/inertia/collider fields are supplied by a small deterministic
Blueprint adapter and verified, never guessed into authority by a VLM.

Tests:

1. CAD/mesh-to-USD bounds, units, topology, volume, origin, and transform
   preservation against the immutable source;
2. dimensions and pose against the source object/task prescription;
3. visual-to-collider surface distance and containment;
4. source object appearance count equals zero after removal;
5. source object collision count equals zero after removal;
6. replacement object exists exactly once in appearance and physics;
7. support gap, penetration, and contact-normal residual;
8. deterministic drop/settle, perturbation, slide, and tip checks;
9. gripper approach, contact, close, lift, and release probes;
10. rendered contact agrees with collider contact from frozen test cameras;
11. changing USD digest, scale, mass, inertia, or material creates a new scene
    digest and never edits the original base scene.

### Gate 5 — Hybrid Isaac scene

Compose:

```text
inpainted background 3DGS appearance
+ validated static SAGE/source-derived collision shell
- removed source-object collision
+ exact SimReady USD task object
+ robot/task island
+ one shared metric frame
```

Require stage load, nonblank renders, camera alignment, deterministic reset,
stable static contacts, task success computation, and no collision/render ghosts.
The SAGE collision shell remains a static-scene control; the inserted object owns
its task physics.

### Gate 6 — Deterministic variation matrix

Use fixed seeds and three test types:

- one-factor-at-a-time diagnosis;
- a pairwise covering array for common interactions;
- targeted interactions selected from the task's failure model.

Vary only recorded dimensions:

- camera pose/intrinsics/exposure/lighting;
- source-view count and angular coverage;
- depth dropout, edge noise, and pose drift;
- mask erosion/dilation and boundary contamination;
- inpainting baseline and completion uncertainty;
- object size, pose, contact offset, and support height;
- collider simplification and visual/collider offset;
- mass, center of mass, inertia, friction, and restitution;
- background appearance while holding task physics fixed;
- task geometry while holding appearance fixed.

Also inject impossible faults: `x0.1`, `x10`, centimeter/meter confusion,
handedness reflection, up-axis swap, mismatched scene IDs, missing depth, absent
clean background, and tampered files.

Expected behavior is either:

- the same bounded construction/decision result inside the preregistered
  tolerance; or
- a typed abstention naming the smallest missing measurement.

Many scene variants establish construction robustness inside this matrix. They
do not establish multi-site generalization or replace policy/trial sample size.

### Gate 7 — Public-data full rehearsal

Freeze one admitted hybrid scene, one fixed-arm task island, exactly two genuine
frozen policies/configurations, and a preregistered condition matrix. Run the
complete simulator-side Task Evaluation Run:

```text
scene/method manifests
-> removal/inpainting/replacement receipts
-> qualified hybrid scene
-> two-candidate conditions and resets
-> lossless policy-input frames + manifest + review video
-> independent simulator metric trace
-> sealed development decision or abstention
-> evidence matrix and replay
```

InteriorGS, SAGE-3D, and ScanNet++ do not provide matched physical policy
outcomes. This rehearsal therefore stops at a `development_only` simulator
decision. The immutable SIMPLER replay separately proves external
physical-outcome join mechanics.

### Gate 8 — Fresh, previously unseen site

Fresh capture is the final new construction input, not the literal final proof
event. After the public matrix passes and the partner protocol is frozen:

1. capture one rights-cleared workcell in a clean-background state;
2. retain gravity/up, intrinsics, poses, RGB-D/LiDAR, and a known-length control;
3. obtain close views of support/contact boundaries;
4. place and observe the source task object in the same registered frame;
5. bind the exact replacement SimReady USD and physical measurements;
6. repeat Gates 1–7 without changing thresholds or methods;
7. execute the two frozen candidates;
8. seal selection/elimination/abstention and the predicted failure boundary;
9. release randomized/interleaved physical holdout outcomes;
10. perform the exact Physical Outcome Join and publish the bounded verdict.

This may be one guided capture session with a clean-background segment followed
by an object-present segment. Do not demand a second full walk unless a measured
coverage/registration gate proves it necessary.

## Public-Scene Phase Gates

| Day | Required outcome | Stop condition |
| --- | --- | --- |
| 7 | Component receipts and a matrix index bind the unchanged Inpaint360GS author-data control, exact rights-admitted InteriorGS/SAGE pair, InFusion primary, AuraFusion360 challenger, controlled clean-background case, exact SimReady object, and physics control; ScanNet++ has an admitted receipt or a recorded still-blocking access outcome; conditional ablations and NVIDIA Content Agents have an admitted receipt or explicit outcome | Exact InteriorGS/SAGE rights are unavailable; an authored positive control is offered as a substitute; a required capability has neither an admitted input nor a recorded outcome; a component receipt is mislabeled as matrix completion |
| 14 | The exact InteriorGS/SAGE pair uses frozen render-derived method-input cameras, removes one object from appearance and collision, runs the InFusion adapter and AuraFusion360 challenger, and inserts one exact USD; the standalone object passes frame, collision, contact, and Isaac checks, with NVIDIA Content Agents compared as an authoring candidate | Hidden scale repair, circular calibration/test trajectories, external validation depth leaked into a method without a preregistered interface, ghost collision, unregistered representations, or unreproducible method |
| 21 | The exact rights-admitted ScanNet++ real measured transfer, controlled known-background recovery, and deterministic variation tests are complete; measurement gaps may produce typed abstentions inside the admitted run | ScanNet++ access/rights remain blocked, generated content is promoted to factual truth, or faults pass silently |
| 28 | One command reproduces all admitted public cases, editing receipts, hybrid-scene qualification, variation outcomes, simulator-side two-candidate run, media, claim ceilings, and replay | Any non-site-specific seam still needs hand-authored evidence or a paper-only dependency |

## Claim Boundaries

Passing ADP-009 proves only that Blueprint can construct, edit, physicalize,
stress, execute, and replay the service workflow on admitted public inputs. It
does not prove:

- commercial rights to a noncommercial dataset or method;
- factual recovery of a never-observed hidden surface;
- partner-site capture quality;
- task-specific real friction, compliance, latency, or dynamics;
- policy observation-domain match at the final site;
- sim-to-real decision fidelity;
- customer value, deployment, safety, or multi-site generalization.

Those missing claims remain missing until the fresh-site protocol and physical
outcome join supply them.
