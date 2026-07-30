# Public Indoor Capture Dataset Audit

Status: primary-source and byte-level audit performed 2026-07-30

## Decision

The research list is useful, but it mixes four different test classes:

1. renderer/trainer benchmarks;
2. posed-image or panoramic reconstruction benchmarks;
3. real indoor walkthrough proxies;
4. raw capture-authority tests.

Only the third and fourth classes exercise the controlled-beta product. A
dataset result never upgrades a derived reconstruction into Blueprint raw
capture authority.

MuSHRoom is the best public test available today for an indoor site-walkthrough
proxy. Its iPhone sequences are openly downloadable under CC BY 4.0 and include
RGB, depth, per-frame camera transforms, an independent short trajectory, and a
Polycam mesh/point cloud. It does not retain original video, decoded video PTS,
IMU, encoder-retention events, or Raw Contract 3.2 semantics, so it can test the
external-reconstruction and observed-site paths but cannot close the physical
iPhone capture gate.

## Candidate audit

| Candidate | Verified facts and corrections | Rights/access | Blueprint test | Cannot prove | Disposition |
| --- | --- | --- | --- | --- | --- |
| InteriorGS | 1,000 synthetic indoor scenes reconstructed as 3D Gaussian splats from more than five million rendered images. Each complete scene contains a SuperSplat-compressed `3dgs_compressed.ply`, author `labels.json`, occupancy sidecars, and `structure.json`, using right/back/up axes and meters. It is not a captured indoor walkthrough. | Gated. The dataset terms permit non-commercial research and education, prohibit commercial use and redistribution, and require the application/terms flow. | Packed-Ply admission and rendering; deterministic object/room normalization; rotated-box preservation; semantic/placement fixture tests; appearance regressions. | Real capture QA, privacy/consent, sensor synchronization, external-scene reconstruction, customer-site semantics, physics, or commercial-beta launch proof. | **Use locally for synthetic semantic-splat regression only.** |
| Mip-NeRF 360, Tanks and Temples, Deep Blending | Canonical 3DGS renderer/trainer benchmarks. The reported vanilla-3DGS PSNR ranges are broadly correct, but the timing claim is hardware-dependent. These posed image sets bypass Blueprint upload, frame retention, decoded-PTS synchronization, and capture QA. | Dataset-specific terms apply. The original Graphdeco 3DGS implementation has a research/evaluation license and is not a product dependency to adopt without review. | Pin a qualified trainer/runtime and reproduce held-out PSNR/SSIM/LPIPS within a declared tolerance. | Capture correctness, rights admission, task discovery, metric authority, physics, or product workflow. | Later trainer qualification, not today's launch gate. |
| ScanNet++ iPhone NVS | 1,006 indoor scenes; 60 FPS 1920x1440 iPhone RGB, 256x192 aligned depth, ARKit poses/intrinsics/IMU, COLMAP poses, and a 12-scene iPhone test split. Default download is about 1.5 TB. | Application-gated and governed by non-commercial terms. | Excellent offline research benchmark for iPhone NVS and sensor degradation after access approval. | Commercial-beta evidence or Blueprint Raw Contract 3.2 retention truth. | Do not use as launch proof. |
| MuSHRoom | Ten real rooms. The audited `koivu` iPhone archive contains long and independent short trajectories, processed RGB/depth, camera transforms, held-out frame IDs, mesh, and point cloud. It contains no original retained video, IMU, tracking/relocalization log, or encoder-retention map. | Open Zenodo record, DOI `10.5281/zenodo.10230733`, CC BY 4.0. Paper documents privacy/confidentiality precautions. Treat screens/signage as `restricted_local_only` unless separately cleared. | Content-addressed external-reconstruction intake; source binding; coverage/held-out-view checks; deterministic task-analysis replay; imported-result normalization; immutable testbed compilation. | Raw iPhone authority, decoded-PTS/retained-frame correctness, physical truth, collision/contact, deployment, or policy ranking. | **Use now. Best public indoor proxy.** |
| ARKitScenes | Claude's `1900+` figure is inaccurate. Apple reports 5,047 captures of 1,661 unique indoor scenes, collected with iPad Pro rather than iPhone. Raw assets can include MOV, depth/confidence, intrinsics, trajectory, and mesh. | Apple's current license includes a bounded commercial grant for qualifying licensees under its stated MAU condition; the earlier blanket `non-commercial` ledger statement was incorrect. Exact organizational eligibility still needs legal confirmation. | Potential video/depth/pose import and QA test after selecting one scene and recording exact license eligibility. | iPhone Pro UX, Raw Contract 3.2 retained-frame evidence, rights/consent metadata in Blueprint format. | Candidate, not current launch proof. |
| SceneSplat-49K | Approximately 49K raw scenes, 45K curated splats, 22.4M frames, and 26.12B Gaussians. It is a precomputed reconstruction/training corpus. | Access-gated; dataset card restricts use to non-commercial research/education even though page metadata also displays a CC license. Use the stricter explicit condition. | Visual/reference regression for splat consumers after access approval. | Capture intake, reconstruction independence, commercial launch, or raw authority. | Research only. |
| 360Roam | Ten real indoor panoramic scenes are listed. The authoritative project page still labels the dataset `available soon`; the asserted CC BY-NC-SA license was not confirmed from the primary page. It is posed panorama data rather than native camera video. | Availability and governing terms unresolved. | Equirectangular reconstruction benchmark only after exact files, hashes, and terms are obtained. | Native INSV ingestion, stitching, video PTS, metric scale, or launch rights. | Wait for verified access/terms. |
| FIORD | Five indoor and five outdoor environments captured as tripod dual-fisheye stills with an Insta360 ONE RS 1-Inch; includes calibration, sparse SfM, and Faro ground-truth clouds. It is not a moving walkthrough video. | Zenodo indoor subset is CC BY 4.0; about 37.6 GB for four scenes. | Fisheye calibration, projection, SfM, source-to-ground-truth alignment, and geometric error. | Frame extraction/PTS, walking coverage, native video continuity, or task workflow. | Strong later fisheye geometry benchmark. |
| OmniSplat converted datasets | Repository provides converted OpenMVG/SfM variants of six omnidirectional datasets. Conversion saves setup work but does not change source-dataset licenses or create raw authority. | Code is MIT; each dataset retains its own terms. | Adapter/rasterizer compatibility and deterministic converted-input tests. | Original-capture provenance, rights, native video, or metric truth. | Useful benchmark harness, not launch evidence. |
| OmniBlender and OB3D | Synthetic omnidirectional scenes with exact cameras; OB3D also provides pixel-aligned depth and normals. | Dataset-specific research terms must be checked before redistribution. | Projection/rasterizer unit tests with exact camera and depth truth. | Any real-capture, privacy, rights, or customer-workflow claim. | Good hermetic math fixtures only. |
| Ricoh360 | A real panoramic benchmark, but the primary supplementary material describes 11 large scenes rather than Claude's 12. Posed train/test images do not equal a raw walkthrough. | Exact dataset license must be verified. | Held-out omnidirectional appearance test. | Native capture container, PTS, metric scale, or commercial launch. | Later benchmark after license check. |
| ODGS, OmniGS, PFGS360, Seam360GS | The technical descriptions are broadly accurate: native spherical/equirectangular rasterization, pose-free 360 work, and explicit dual-fisheye seam modeling. They are research implementations, not current Blueprint adapters. | Review each code/dependency/data license before adoption. | Comparative 360 method qualification on the same held-out views and camera representation. | Product qualification merely because a paper reports good results. | Candidate methods behind the existing replaceable interface. |
| ODGS-SLAM | Provides controlled real and synthetic sequences across omnidirectional/fisheye/multicamera configurations with trajectories. It is a SLAM benchmark, not customer capture authority. | Dataset terms still require exact verification. | Pose/trajectory estimation regression. | Product rights, task discovery, or reconstruction physics. | Research benchmark only. |
| Antigravity `.insv` sample | The published codec inspection is plausible and useful for container tooling, but the sample is outdoor drone footage, not an indoor walkthrough. Competition terms do not clearly grant Blueprint redistribution/product rights. | Reuse rights unresolved. | At most a local throwaway INSV probe after explicit rights confirmation. | Indoor site coverage, task discovery, privacy-safe product intake, or launch. | Reject for this goal. |
| XR AI Spotlight intermediates | Paid access, exact files, and reuse rights were not independently verified. | Paid and unresolved. | None under current evidence. | All launch claims. | Do not purchase or use. |

## Real-data test executed today

The smallest MuSHRoom iPhone scene, `koivu_iphone.tar.gz`, was downloaded from
the official Zenodo record and inspected locally.

- publisher MD5: `a359dba714e7829be11747ce5dee141c`;
- verified archive SHA-256:
  `68735cfa0758e1288a006c30dc8b95ffb4caa3392bc9c68c0c3ea6c111966518`;
- size: `146575749` bytes;
- archive safety: 874 members, no traversal, symlinks, or hardlinks;
- long sequence: 309 RGB and 309 depth images, 294 pose frames, 29 declared
  held-out IDs;
- short sequence: 121 RGB, depth, and pose frames;
- selected imported point cloud SHA-256:
  `748af95d385bfedfb2058b28b59a6f431947c13deb34321719fcc6a2b16dac1e`.

The real point cloud was materialized through the current
`precomputed_external_reconstruction` Capture Intake path with the exact archive
digest and DOI as source binding. Admission was content-addressed and
idempotent. All metric, collision, contact, physical-success, deployment,
safety, task-discovery, and comparative-policy-ranking authority stayed false;
the frozen verdict remained `thesis_not_supported`.

The follow-up external-reconstruction lane now admits PLY through the signed
transfer boundary, preserves the source-capture digest in the immutable intake,
requires explicit local-adapter authorization, and emits an inspectable
appearance-only `Reconstruction Result`. The actual 367,960-vertex MuSHRoom PLY
ran through intake, QA, plan, authorization, execution, and compiler-input
loading for zero paid-compute cost. It ended `partial` because decoded source
observations were absent. The transfer and malware scanner were injected test
doubles, so this is not deployed security proof.

The Task Evaluation Supervisor was then run against the same materialized
intake. All six capabilities were registered, but no explicitly supported live
inference credential was configured. It made zero model calls and zero tool
actions, returned an abstention, and replayed successfully without invoking a
model. This is a fail-closed pass, not a live-agent intelligence result.

## InteriorGS compatibility test executed today

The retained scene `0787_841244` was tested against the actual bytes rather
than a renamed or converted fixture. Blueprint recognized the PLY as
`supersplat_compressed_3dgs`: 630,898 Gaussian vertices, 630,898 spherical
harmonic rows, and 2,465 chunks. The same packed splat was converted by the
pinned PlayCanvas transformer and rendered as a nonblank 512x512 Metal frame.
That establishes import/decode/display compatibility only.

The dataset-author sidecars normalized into a deterministic, hash-bound
`object_index.v2` containing 278 valid objects, seven rooms, 37 walls, and 16
holes. Exact eight-corner oriented boxes are retained while robot-placement
consumers receive separately labeled conservative world AABBs. The source
digests are:

- splat: `9a0c451e57edf3623a77026937757efb09e6a1fc33de1f67699a6b076354ccf6`;
- labels: `b6bc08ec84818111ca736da5a746643c36c3d9ebbfd67ab8e4f8b4359ba49856`;
- structure: `4cd397f59ce3c876bbf0ca118cb9e849ef5747e8ead6d9cc8f14d352d902b4ed`.

The retained local scene does not include the dataset's occupancy PNG/JSON,
so occupancy compatibility is not claimed. InteriorGS annotations remain
synthetic dataset-author metadata, not observed customer-capture facts. They do
not establish metric transform validity for an imported customer scene,
collision geometry, physics, physical success, deployment, safety, or policy
ranking. The ranking verdict remains `thesis_not_supported`.

The separate SAGE-3D USDZ derivative was also audited with the public
`839916.usdz` sample (171,136,801 bytes, SHA-256
`fb9e2ac75303e192c7cd2083c23c1c89233bec2304141b38b8f83fae0215560f`).
Its package contains `default.usda`, `gauss.usda`, and a 171 MB NuRec payload.
Local OpenUSD opened the package, resolved `/World`, and identified a NuRec
`Volume` plus two `OmniNuRecFieldAsset` prims. This proves CPU package/schema
inspection, not RTX rendering. The USDZ is an appearance volume; SAGE-3D's
collision bodies are a separate USD dataset and still require exact scene-ID
matching, transform validation, and independent physics qualification.

For Blueprint customer captures, a SAM3-class model can supply 2D masks and
cross-frame tracks, but it cannot by itself create trustworthy metric 3D object
records. The required lane is: decoded retained frames plus poses/intrinsics and
depth; open-vocabulary proposals; 2D masks/tracks; calibrated multi-view fusion;
3D box/coverage estimation; relationship inference kept separate from observed
facts; held-out reprojection/depth checks; and human approval. Missing pose,
depth, scale, or coverage must lower the JSON claim ceiling or trigger recapture.

## Agent and contract tests executed today

Twelve focused adversarial tests passed:

- false observation and unearned-decision rejection;
- untrusted prompt-injection handling;
- metadata and filename injection handling;
- capture-only blocked termination;
- live SDK denial without explicit inference authorization;
- proof, budget, and hidden-label mutation denial;
- no proposer/evaluator self-grading;
- replay-tamper rejection;
- standalone manifest ingress;
- kernel-decision invariance to manager prose;
- external reconstruction source-digest enforcement;
- plan/authorize/execute for the local decoded-observation method.

These tests establish that the new agent architecture cannot promote prose,
provider output, or compromised capability output into scientific truth. They
do not establish that a live model can complete an indoor walkthrough workflow.

## Tests to run next

### Completed: external-reconstruction import and real PLY execution

The hermetic method now accepts only `precomputed_external_reconstruction`,
verifies source and asset digests, parses a size-bounded PLY header without
executing embedded content, records unknown coverage/generated-region/metric
status, and exact-replays from immutable inputs. It emits only an appearance
layer and cannot upgrade raw, captured-observation, task, metric, collision,
physics, physical, deployment, safety, or comparative-ranking claims.

The next bounded product test is to add the processed MuSHRoom RGB/pose material
as a separately source-bound observation representation, then run task-candidate
analysis, human approval, immutable testbed compilation, and a partial Decision
Envelope. This remains a public proxy, not the raw-capture launch gate.

### P0: live-agent, zero-tool mutation smoke

With a separately configured supported OpenAI API credential and a strict
inference budget, run the supervisor on the same MuSHRoom intake. Require:

- only registered read-only capabilities;
- no provider upload;
- no proof-state or budget mutation;
- task candidates tied to observed frames/regions;
- mandatory human approval before a Decision/Evidence Request;
- deterministic kernel result independent of prose;
- exact replay artifacts and bounded token/cost receipts.

The current environment has no supported live inference credential, so the
safe denial path—not live reasoning—was tested today.

### P1: held-out trajectory and coverage test

Use MuSHRoom long capture as source evidence and the independent short capture
plus declared held-out long frames only as evaluation. Verify transform
conventions, reprojection, visible-region coverage, and leakage prevention.
Do not calibrate thresholds on held-out views.

### P1: raw iPhone/video contract test

The user explicitly accepted the ARKitScenes license for local evaluation and
authorized scene `40958756`. The accepted exact-commit proxy run decoded the
MOV, bound original video PTS to timed ARKit timestamps and intrinsics, joined
all 163 official trajectory samples to RGB/depth/confidence/intrinsics, froze
40 observations into disjoint 32-candidate/8-held-out partitions, filtered
depth to confidence 2 plus positive samples, and replayed identically. Its
terminal digest is
`sha256:4c1d69c959ce1df03be4196b4dc2cf6c762c73fd4f474bc2c95d6cf94f64b0f6`.
This remains local public-dataset proxy evidence. Only an actual Raw Contract
3.2 bundle can test retained-frame/encoder-attempt truth, and separate legal
review is still required before any broader organizational/commercial use.

### P1: 360 projection tests

Start with a small synthetic OB3D/OmniBlender scene for exact projection math,
then one verified FIORD indoor scene for real fisheye geometry. Defer 360Roam,
Ricoh360, and native INSV product claims until exact access and licenses are
recorded.

### P2: trainer/rasterizer qualification

Only after a pinned COLMAP plus Nerfstudio/gsplat adapter exists, use one
canonical indoor scene and one real walkthrough proxy to measure held-out
PSNR/SSIM/LPIPS, reprojection, runtime, peak memory, artifact digest
repeatability, and failure behavior. Renderer scores remain appearance evidence
only.

## Primary sources

- [InteriorGS dataset](https://huggingface.co/datasets/spatialverse/InteriorGS)
- [InteriorGS repository](https://github.com/manycore-research/InteriorGS)
- [InteriorGS terms of use](https://kloudsim-usa-cos.kujiale.com/InteriorGS/InteriorGS_Terms_of_Use.pdf)
- [SAGE-3D InteriorGS USDZ](https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz)
- [SAGE-3D collision meshes](https://huggingface.co/datasets/spatialverse/SAGE-3D_Collision_Mesh)
- [PlayCanvas splat-transform](https://github.com/playcanvas/splat-transform)
- [SAM 3 official repository](https://github.com/facebookresearch/sam3)
- [SAM 3D Objects official repository](https://github.com/facebookresearch/sam-3d-objects)
- [ConceptGraphs](https://concept-graphs.github.io/)
- [Open3DIS](https://open3dis.github.io/)
- [OGScene3D](https://arxiv.org/abs/2603.16301)
- [MuSHRoom iPhone dataset](https://zenodo.org/records/10230733)
- [MuSHRoom WACV 2024 paper](https://openaccess.thecvf.com/content/WACV2024/html/Ren_MuSHRoom_Multi-Sensor_Hybrid_Room_Dataset_for_Joint_3D_Reconstruction_and_WACV_2024_paper.html)
- [ScanNet++ documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation)
- [ScanNet++ terms](https://scannetpp.mlsg.cit.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf)
- [ARKitScenes](https://github.com/apple/ARKitScenes)
- [ARKitScenes data layout](https://github.com/apple/ARKitScenes/blob/main/DATA.md)
- [ARKitScenes license](https://github.com/apple/ARKitScenes/blob/main/LICENSE)
- [SceneSplat-49K](https://huggingface.co/datasets/GaussianWorld/scene_splat_49k)
- [360Roam](https://huajianup.github.io/research/360Roam/)
- [FIORD](https://parmisian.github.io/fiord360.github.io/)
- [FIORD indoor Zenodo record](https://zenodo.org/records/15181976)
- [OmniSplat](https://github.com/esw0116/OmniSplat)
- [ODGS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6882dbdc34bcd094e6f858c06ce30edb-Abstract-Conference.html)
- [OmniGS](https://openaccess.thecvf.com/content/WACV2025/html/Li_OmniGS_Fast_Radiance_Field_Reconstruction_Using_Omnidirectional_Gaussian_Splatting_WACV_2025_paper.html)
- [PFGS360](https://openaccess.thecvf.com/content/CVPR2026/html/Zhuang_Pose-Free_Omnidirectional_Gaussian_Splatting_for_360-Degree_Videos_with_Consistent_Depth_CVPR_2026_paper.html)
- [Seam360GS](https://openaccess.thecvf.com/content/ICCV2025/html/Shin_Seam360GS_Seamless_360deg_Gaussian_Splatting_from_Real-World_Omnidirectional_Images_ICCV_2025_paper.html)
- [ODGS-SLAM](https://odgs-slam.github.io/)
- [Original 3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
