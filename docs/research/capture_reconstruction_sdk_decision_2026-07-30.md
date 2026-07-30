# Capture and Reconstruction Build-versus-SDK Decision

Status: controlled-beta decision, verified against primary sources on 2026-07-30

## Decision

Blueprint should retain `BlueprintCapture`'s native ARKit/AVFoundation/
CoreMotion recorder as the highest-authority iPhone lane and keep Raw Contract
3.2 as the evidence authority. No third-party capture or reconstruction SDK
replaces it.

For the controlled beta:

- adopt the existing hermetic decoded-observation and strict Raw Contract 3.2
  metric-scaffold adapters;
- add COLMAP plus Nerfstudio/gsplat as the preferred next local appearance/SfM
  implementation behind the existing reconstruction interface, with metric
  authority disabled unless the source bundle or an accepted anchor supplies it;
- keep RoomPlan as an optional structural/semantic prior and Object Capture as
  an optional isolated-object method;
- retain original INSV files and use the Insta360 SDK only after application
  approval, exact-version pinning, and license review; stitched output and gyro
  remain derived evidence;
- keep Scaniverse/Niantic, World Labs Marble, and Lightwheel behind replaceable
  import/provider profiles. Do not enable live customer upload until commercial
  rights, data-use, deletion, credentials, and spend authorization pass;
- treat Real2Code/ArtFormer-like completion as generated-only research output;
  it must carry a generated-region mask and cannot establish hidden geometry,
  articulation, physics, or task success;
- use NVIDIA's SimReady specification/validator as a validation aid, not as
  evidence that a visually plausible or provider-generated asset has correct
  physics.

No SDK, subscription, API credit, or provider plan was purchased or activated
for this decision.

## Scoring rubric

`3` is strong and directly usable for the controlled beta; `2` is usable with a
bounded adapter or qualification; `1` is weak, manual, unclear, or materially
restricted; `0` is absent or incompatible. A high total does not grant evidence
authority: claim-level qualification and exact source binding still apply.

Candidate abbreviations:

- `Native`: BlueprintCapture ARKit/AVFoundation/CoreMotion
- `RoomPlan`: Apple RoomPlan
- `ObjectCap`: Apple RealityKit Object Capture
- `Scaniverse`: Niantic Spatial Scaniverse/Reconstruct
- `Insta360`: official Camera/Media SDK workflow
- `Local`: COLMAP plus Nerfstudio/gsplat
- `Marble`: World Labs World API/Marble
- `Lightwheel`: Lightwheel SimReady library/platform
- `Generated`: Real2Code/ArtFormer-like completion

### Access, capture, and synchronization

| Candidate | Programmatic availability | Devices/formats | Retained originals | Poses/frame semantics |
| --- | ---: | ---: | ---: | ---: |
| Native | 3 | 3 | 3 | 3 |
| RoomPlan | 3 | 1 | 1 | 2 |
| ObjectCap | 3 | 2 | 2 | 2 |
| Scaniverse | 2 | 3 | 1 | 1 |
| Insta360 | 2 | 3 | 3 | 1 |
| Local | 3 | 2 | 3 | 2 |
| Marble | 3 | 3 | 1 | 1 |
| Lightwheel | 0 | 1 | 0 | 0 |
| Generated | 2 | 2 | 1 | 0 |

### Calibration and motion evidence

| Candidate | Intrinsics/distortion | Decoded PTS and retained-frame map | Metric depth/confidence | IMU and monotonic time |
| --- | ---: | ---: | ---: | ---: |
| Native | 3 | 3 | 3 | 3 |
| RoomPlan | 2 | 0 | 2 | 0 |
| ObjectCap | 2 | 0 | 1 | 0 |
| Scaniverse | 1 | 0 | 2 | 0 |
| Insta360 | 2 | 1 | 0 | 2 |
| Local | 3 | 2 | 1 | 0 |
| Marble | 1 | 0 | 1 | 0 |
| Lightwheel | 1 | 0 | 1 | 0 |
| Generated | 0 | 0 | 0 | 0 |

### Tracking, outputs, semantics, and privacy

| Candidate | Resets/relocalization | Mesh/splat/point/USD outputs | Semantic/instance data | Offline/private processing |
| --- | ---: | ---: | ---: | ---: |
| Native | 3 | 2 | 2 | 3 |
| RoomPlan | 2 | 2 | 2 | 3 |
| ObjectCap | 0 | 2 | 0 | 3 |
| Scaniverse | 1 | 3 | 2 | 1 |
| Insta360 | 1 | 1 | 0 | 3 |
| Local | 2 | 3 | 1 | 3 |
| Marble | 0 | 3 | 1 | 0 |
| Lightwheel | 0 | 3 | 2 | 1 |
| Generated | 0 | 2 | 1 | 2 |

### Export, reproducibility, license, and provider data use

| Candidate | Raw-data export | Deterministic versioning | Commercial use | Customer-data model training |
| --- | ---: | ---: | ---: | ---: |
| Native | 3 | 3 | 3 | 3 |
| RoomPlan | 2 | 3 | 3 | 3 |
| ObjectCap | 2 | 3 | 3 | 3 |
| Scaniverse | 1 | 2 | 1 | 0 |
| Insta360 | 3 | 2 | 1 | 2 |
| Local | 3 | 3 | 3 | 3 |
| Marble | 1 | 2 | 2 | 1 |
| Lightwheel | 1 | 1 | 1 | 1 |
| Generated | 2 | 2 | 1 | 2 |

### Lifecycle, stability, cost, and Blueprint provenance

| Candidate | Deletion/revocation | API stability | Pricing/lock-in | Preserve hashes/provenance |
| --- | ---: | ---: | ---: | ---: |
| Native | 3 | 3 | 3 | 3 |
| RoomPlan | 3 | 3 | 3 | 3 |
| ObjectCap | 3 | 3 | 3 | 3 |
| Scaniverse | 1 | 2 | 1 | 2 |
| Insta360 | 2 | 2 | 2 | 3 |
| Local | 3 | 3 | 3 | 3 |
| Marble | 1 | 2 | 1 | 2 |
| Lightwheel | 1 | 1 | 0 | 1 |
| Generated | 2 | 1 | 2 | 2 |

## Evidence and rationale

### Native Apple lane

Apple's `ARFrame` exposes the captured image, camera state, and timestamp, while
scene depth is a separately supported frame semantic aligned to the captured
image on capable devices. This supports Blueprint's native lane, but does not by
itself prove that an AR frame was retained by the video encoder; Raw Contract
3.2's independently decoded PTS and encoder-retention map remain necessary.
[ARFrame timestamp](https://developer.apple.com/documentation/arkit/arframe/timestamp),
[scene depth](https://developer.apple.com/documentation/arkit/arconfiguration/framesemantics-swift.struct/scenedepth)

RoomPlan uses camera and LiDAR data to produce a parametric room model and can
export USD/USDZ. Its recognized walls, openings, doors, furniture, and appliance
boxes are useful structural and semantic priors, not contact geometry or task
physics. [RoomPlan](https://developer.apple.com/documentation/roomplan),
[CapturedRoom](https://developer.apple.com/documentation/roomplan/capturedroom),
[ModelProvider](https://developer.apple.com/documentation/roomplan/capturedroom/modelprovider)

RealityKit Object Capture reconstructs selected objects from multi-angle photos
and exposes model, point-cloud, and estimated-pose outputs. It is appropriate
for isolated task objects with independent scale/placement/physics validation,
not whole-site capture authority.
[Object Capture](https://developer.apple.com/documentation/realitykit/realitykit-object-capture),
[Photogrammetry output](https://developer.apple.com/documentation/realitykit/photogrammetrysession/output)

### Insta360

The official SDK requires an application/approval and provides Camera and Media
SDKs across desktop and mobile platforms. 360 video is not generally stitched
in-camera, so Media SDK output is derived; native INSV must be retained. The
official integration guide describes gyro acceleration/angular velocity but not
metric camera translation, and some camera modes require multiple INSV tracks or
files. Therefore Insta360 metadata can strengthen a 360 intake but cannot become
ARKit-equivalent metric authority.
[SDK guide](https://onlinemanual.insta360.com/developer/en-us/resource/sdk),
[integration guide](https://onlinemanual.insta360.com/developer/en-us/resource/integration),
[X3 file transfer](https://onlinemanual.insta360.com/x3/en-us/camera/filetransfer)

### Scaniverse and Niantic Spatial

Current Scaniverse plans advertise mobile and 360 capture, metric scale,
Gaussian splats, meshes, VPS maps, and FBX/SPZ/PLY or USDZ exports. 360 cloud
generation is paid, commercial rights require Pro or Enterprise, and custom API
integration is Enterprise. The published price is $20/month for Plus and
$50/month for Pro; 360 processing consumes 7,200 credits per minute.
[plans and pricing](https://www.nianticspatial.com/en/pricing)

The current Business and Developer Terms cover reconstruction services and
APIs, restrict benchmarking/competitive use, and grant Niantic broad rights to
use User Materials and Output to operate, improve, develop, and train its
technology. Those terms are incompatible with a default confidential-customer
lane without a negotiated agreement. A DPA exists, but it does not by itself
override the product/data-use terms.
[business terms](https://www.nianticspatial.com/ja/terms-business),
[business privacy policy](https://www.nianticspatial.com/privacy-business),
[DPA](https://www.nianticspatial.com/legal/terms/niantic-spatial-platform-sdk/controller-processor)

Decision: support a hash-bound external-reconstruction import and a disabled
provider adapter, but do not purchase or enable live upload under this goal.

### Local COLMAP, Nerfstudio, and gsplat

COLMAP is a scriptable SfM/MVS pipeline. Its documented outputs include camera
intrinsics, image poses, sparse points, dense depth/normal maps, fused PLY point
clouds, and meshes. It is BSD licensed, with a documented caveat that bundled
dependencies carry their own licenses. SfM still has scale ambiguity unless a
trusted anchor or source metric solution is supplied.
[COLMAP](https://colmap.github.io/),
[output format](https://colmap.github.io/format.html),
[license](https://colmap.github.io/license.html)

Nerfstudio can process video/custom image data, train Splatfacto on COLMAP
geometry, and export cameras, point clouds, meshes, or Gaussian-splat PLY files.
Its Splatfacto documentation says direct equirectangular rendering is currently
unsupported, so 360 input needs an explicit perspective projection/normalization
step. Nerfstudio and gsplat are Apache-2.0 projects.
[custom data](https://docs.nerf.studio/quickstart/custom_dataset.html),
[Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html),
[geometry export](https://docs.nerf.studio/quickstart/export_geometry.html),
[gsplat](https://github.com/nerfstudio-project/gsplat)

Decision: this is the replaceable local baseline. Pin exact binaries,
dependencies, container/runtime digest, parameters, source frames, and output
hashes. Keep appearance, SfM, and mesh outputs derived and below raw authority.

### World Labs Marble

The public World API accepts text, image, multi-image, or video and returns SPZ
splats plus a GLB collider mesh. It is a public, programmatic, paid generation
API—not a calibrated capture recorder. API credits cost $1 per 1,250 with a $5
minimum purchase, while current standard world generation costs 1,500 credits
plus any pano step. Commercial API output rights are available subject to the
terms/order form, and the terms allow anonymized/aggregated derivatives of
inputs and outputs for model improvement subject to opt-out rights.
[API quickstart](https://docs.worldlabs.ai/api),
[API pricing](https://docs.worldlabs.ai/api/pricing),
[terms](https://docs.worldlabs.ai/terms-of-service),
[world output](https://docs.worldlabs.ai/api/reference/worlds/get)

Decision: keep a disabled replaceable adapter and import contract. Any output is
generated appearance/debug evidence; its reported scale or collider mesh cannot
self-qualify as observed metric or physics truth. No credits were purchased.

### Lightwheel and SimReady validation

Lightwheel publicly offers a SimReady library and Enterprise platform with
rigid, articulated, and deformable assets, but the current primary pages expose
a demo/custom-sales workflow rather than a documented general asset-generation
API. Commercial licenses are described as part of its Enterprise package.
[Lightwheel](https://www.lightwheel.ai/),
[asset library](https://www.lightwheel.ai/asset-library),
[platform](https://www.lightwheel.ai/lightwheel-platform)

NVIDIA's current SimReady Foundation provides rules for validating OpenUSD
collision, rigid-body, joints, and mass-property schemas. Passing schema rules is
necessary support evidence, but independently measured geometry/material/contact
properties are still required for claim qualification.
[SimReady Foundation](https://nvidia.github.io/simready-foundation/2026.04.0/guides/getting_started.html),
[SimReady specification](https://docs.omniverse.nvidia.com/simready/latest/overview/simready-spec.html)

Decision: no live Lightwheel integration without documented API access,
commercial terms, identity-matched assets, deletion/data-use terms, and exact
independent validation. Adopt the NVIDIA validator only as a local validation
step when the dependency and license review pass.

### Generated articulated completion

Real2Code reconstructs articulated objects from visual observations using part
segmentation/shape completion and LLM-generated joint code; ArtFormer generates
articulated geometry and kinematic relations from learned priors. These are
research methods, not authoritative measurements of a customer's hidden object.
[Real2Code](https://real2code.github.io/),
[ArtFormer paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Su_ArtFormer_Controllable_Generation_of_Diverse_3D_Articulated_Objects_CVPR_2025_paper.pdf)

Decision: generated completion remains masked, versioned, and prohibited from
upgrading existence, metric, collision, articulation, mass, friction, support,
clearance, physical-success, deployment, or safety claims.

## Remaining live-proof requirements

- a rights-cleared physical iPhone Raw Contract 3.2 bundle or bounded 360
  capture submitted through the supported interface;
- exact Insta360 SDK approval/license and sample-export verification before
  enabling native-container processing;
- local COLMAP/Nerfstudio/gsplat execution on that capture with exact runtime,
  cost, output digests, held-out metrics, and explicit scale ceiling;
- a negotiated provider agreement plus user authorization before any
  confidential Scaniverse, Marble, or Lightwheel upload;
- deletion receipts and opt-out/data-use evidence for any external provider;
- independent asset/physics validation before any generated or imported mesh is
  admitted to collision/contact evidence.
