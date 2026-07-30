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

The current Capture QA path returned `analysis_required` because it has no
external-reconstruction normalizer. That is accurate but incomplete behavior:
the asset is safely admitted and bound, yet cannot become an inspectable
`Reconstruction Result` or testbed layer.

The Task Evaluation Supervisor was then run against the same materialized
intake. All six capabilities were registered, but no explicitly supported live
inference credential was configured. It made zero model calls and zero tool
actions, returned an abstention, and replayed successfully without invoking a
model. This is a fail-closed pass, not a live-agent intelligence result.

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

### P0: implement and qualify the external-reconstruction import adapter

Add a local hermetic method that:

1. accepts only `precomputed_external_reconstruction` intake;
2. verifies source-capture binding and exact asset digest;
3. parses size-bounded PLY/mesh metadata without executing embedded content;
4. emits an observed-versus-derived `Reconstruction Result` with no metric or
   physics upgrade unless independently qualified;
5. records coordinate system, scale declaration, coverage, uncertainty, rights,
   provider, and claim ceiling;
6. fails stale/mismatched source binding;
7. exact-replays from immutable inputs.

Run MuSHRoom through intake -> QA -> reconstruction plan -> explicit local
authorization -> execution -> task-candidate analysis -> human approval ->
immutable testbed. The expected terminal result is a partial Decision Envelope
or abstention with exact next measurements, not a pass.

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
