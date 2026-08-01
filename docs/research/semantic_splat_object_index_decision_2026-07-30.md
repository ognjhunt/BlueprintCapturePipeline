# Semantic splat object-index decision (2026-07-30)

## Decision

Blueprint will keep SAM 3.1-style video masks and persistent track IDs as 2D
observations, but the production object-index path will be a source-bound
mask-to-Gaussian or mask-to-surface lifting stage. It will not treat a detector
box plus one depth sample as an authoritative metric object.

The current Splat Analyzer adapter remains an immediate executable candidate
generator. Its output is visualization-only and cannot qualify metric placement,
orientation, collision, contact, articulation, physics, or physical success.
The object-index artifact remains an authoritative record of those candidates;
that does not make the candidate geometry canonical capture truth.

## Implemented now

- The exact InteriorGS SuperSplat PLY header is recognized before external
  execution. Direct Splat Analyzer use fails closed with a request for a
  hash-bound standard-3DGS derivative, conversion-runtime digest, and explicit
  world-axis transform. Blueprint's native InteriorGS importer remains the
  direct lane for the compressed source file.
- Splat Analyzer objects are forced to presentation-only rough interaction
  volumes even when a backend payload tries to self-assert metric or physics
  readiness. The enclosing catalog records candidate status without upgrading
  it to canonical object truth.
- Numeric depth cannot drive robot/object placement by default. Metric use now
  requires exact source-capture, retained-frame, retained-file, loaded-payload,
  calibration, pose, and sync-row digests; decoded PTS; declared z-depth meters;
  verified scale; and bounded uncertainty/reprojection evidence. The loaded
  depth payload is hashed independently and compared to the declaration.
- The current multi-view AABB fusion path is retained as a qualified metric-depth
  method or explicit diagnostic. It is not mislabeled as the future
  contribution-weighted Gaussian semantic lifter.
- A provider-neutral contribution-lifting baseline now accepts exact stable
  Gaussian mappings, persistent track registries, full-frame mask probabilities,
  source-frame PTS, calibrated camera records, and the renderer's actual
  front-to-back `transmittance * alpha` rows. It independently verifies the
  canonical mapping, track, camera, mask, contribution, and view payload digests;
  caps input size; accumulates foreground and background evidence; requires
  multi-view and angular support; excludes generated-only Gaussians; keeps
  adjacent same-label track IDs separate; and returns a targeted abstention when
  evidence is insufficient.
- A bounded file stage verifies the exact JSON artifact bytes and sizes before
  executing the pure lifter and atomically writes a provenance-linked terminal
  result. The result ceiling is per-Gaussian semantic support. It explicitly
  remains unready for metric boxes, collision, physics, task success, or physical
  claims.
- A separate Z-up metric OBB stage now consumes only a digest-valid terminal
  lifting result, the exact Gaussian mapping, and hash-bound observed depth,
  verified mesh-surface, or Gaussian-center support points. It rejects
  generated/unknown support, requires verified metric scale, removes bounded MAD
  outliers, fits a deterministic minimum-area horizontal rectangle, estimates
  vertical limits independently, and emits eight ordered corners. Gaussian-center
  fits carry an explicitly weaker approximate ceiling. Every output remains a
  candidate pending separate collision/occupancy validation.
- A separate collision-consistency stage now consumes the exact terminal OBB
  result plus an exact-digest, metric Z-up collision scene produced by a method
  that is explicitly qualified for target overlap, support contact, non-target
  penetration, verified-free-space conflict, coverage, and generated-region
  checks. The collision-scene producer and validator must differ, target
  identity is separately evidence-bound, and generated geometry cannot supply
  the check. The deterministic baseline computes oriented-box versus AABB volume
  overlap, support-plane contact/overlap under declared spatial uncertainty,
  penetration/conflict fractions, corner coverage, and generated-region
  intersections. It returns a precise next experiment on disagreement. A pass
  remains an independently cross-checked semantic candidate; it never sets
  `collision_ready` or `physics_ready`.
- An independent semantic-geometry benchmark now consumes the exact OBB result
  plus a rights-cleared, independently produced and reviewed metric Z-up
  reference set that was withheld from prediction. It uses deterministic
  label-aware optimal assignment rather than greedy instance matching and
  reports object recall, false-positive fraction, center/dimension/yaw error,
  true oriented 3D OBB IoU, adjacent same-label instance recovery, and geometry
  drift under hash-bound view-removal reruns. Cuboid axis swaps and 180-degree
  yaw symmetry are handled explicitly; near-square objects are excluded from
  yaw scoring rather than assigned a misleading error. A bounded CLI stage
  verifies exact artifact bytes before emitting the diagnostic result. A
  complete digest-bound prediction-input manifest is mandatory, and the
  evaluator rejects a reference annotation, source, or alignment artifact that
  appeared in prediction inputs.

Still missing is a real renderer adapter that emits those exact contribution
artifacts from a production analysis splat. The checked-in stage consumes and
qualifies contribution rows but does not synthesize them or claim a render ran.
Large-scene production also needs a bounded chunked/binary transport rather than
one JSON view artifact, followed by graph cleanup, a production surface-point
adapter, production collision-scene/support evidence, and testbed projection.
The metric suite is implemented, but no public-dataset result is claimed until
an independently reviewed reference split and real predictions are supplied.

## Primary-source audit

### Splat Analyzer

Source: <https://github.com/nigelhartman/splat_analyzer>

- MIT licensed and supports configured local execution on Apple Metal or NVIDIA
  CUDA.
- Renders RGB/depth views, runs OWLv2, back-projects detections, clusters them,
  and emits `interactions.json`.
- The current implementation samples a 5x5 depth patch at the 2D box center,
  derives world width/height from the pixel box, invents depth extent as the
  mean of those two values, writes an identity rotation, and uses fixed-anchor
  distance clustering. This is a rough interaction volume, not a validated OBB.
- Camera generation defaults to Y-up. Blueprint customer/testbed coordinates are
  Z-up, so a hash-bound analysis derivative and explicit transform are required.
- Its PLY loader expects conventional `x/y/z`, opacity, scale, quaternion, and SH
  properties. InteriorGS SuperSplat-compressed PLY must first be converted to an
  analysis-grade standard 3DGS derivative; conversion cannot upgrade authority.

### LUDVIG

Sources: <https://github.com/naver/ludvig> and
<https://juliettemarrie.github.io/ludvig/>

LUDVIG demonstrates learning-free aggregation of 2D features or masks onto
Gaussians and graph diffusion over geometry/visual similarity. Its repository
license is explicitly noncommercial, so Blueprint may use the publication as an
algorithmic reference but must not copy its implementation into the commercial
product.

### ZeroSplat

Sources: <https://inkmind-ai.github.io/ZeroSplat/> and
<https://arxiv.org/abs/2607.18801>

ZeroSplat is the closest current research design: semantic parsing, SAM 3 masks,
volume-rendering-based lifting to Gaussians, cross-view verification, and KNN
diffusion. The official page still marks code as forthcoming, so it is not an
executable dependency today.

### TrackRef3D and GaussDet

Sources: <https://arxiv.org/abs/2605.26576> and
<https://arxiv.org/abs/2606.30638>

TrackRef3D supports the track-first/label-later identity strategy. GaussDet
supports 3D instance grouping followed by multi-view detector label aggregation.
Both inform Blueprint's design, but neither is adopted as a qualified runtime in
this decision.

### SpatialLM 1.1 and Spatial Code

Sources: <https://github.com/manycore-research/SpatialLM>,
<https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B>, and
<https://github.com/Beckschen/spatialcode>.

SpatialLM accepts Z-up point clouds and emits indoor layout/object boxes, making
it useful as a research cross-check. Its 1.1 model weights are CC-BY-NC-4.0 and
its category scope is furniture-heavy, so it is not a commercial canonical
parser. Spatial Code describes video-to-3D tracks/OBBs, but the public repository
still contains only the release timeline and no runnable implementation/models.

## Blueprint stage graph

1. `rendered_view_detector`: open-vocabulary candidate boxes/prompts.
2. `source_track_importer`: implemented provider-neutral normalization for
   compact probability-RLE masks and persistent track IDs. Every observation is
   bound to an encoder-retained source frame, decoded PTS, camera record,
   provider profile, model/runtime digest, allowed use, and exact provider
   result; labels remain inferred candidates and never become observed facts or
   geometry authority. The file entrypoint rejects symlinks, input overwrite,
   oversized payloads, and provider-byte hash/size mismatches before emitting a
   terminal compact artifact.
3. `gaussian_contribution_lifter`: accumulate foreground/background evidence
   using renderer contribution weights and exact camera bindings.
4. `instance_fusion`: track-aware, multi-view, disconnected-component cleanup.
5. `oriented_box_fitter`: implemented baseline for robust outlier removal,
   horizontal minimum-area fitting, independent vertical bounds, and eight Z-up
   corners; production surface-point evidence remains incomplete.
6. `collision_validator`: implemented deterministic consistency baseline for
   independently qualified target volumes, support planes, occupied/free-space
   volumes, coverage, and generated regions; production collision-scene and
   support-plane adapters remain incomplete.
7. `confidence_scorer`: view count/diversity, ambiguity, coverage, reprojection,
   scale, support, and held-out validation.

Every stage binds the raw capture digest, reconstruction digest, stable Gaussian
mapping or analysis derivative, camera solution, model/runtime version, and
output digest. Unsupported or generated-only regions remain explicit.

## Evaluation gate

Use a rights-cleared research fixture only within its license envelope. The
implemented benchmark measures object recall/false positives, center error,
dimension error, yaw error, true 3D OBB IoU, adjacent same-category separation,
and stability under removed frames/views. Held-out reprojection remains a
separate renderer gate. No benchmark pass upgrades collision, physics, physical
success, deployment, safety, or the frozen comparative policy-ranking verdict
`thesis_not_supported`.
