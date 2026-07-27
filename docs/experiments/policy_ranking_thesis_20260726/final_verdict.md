# Final verdict

Verdict: `thesis_not_supported`

The frozen pipeline did not satisfy the preregistered conjunction required to
support Blueprint's policy-ranking thesis. This is a verdict on the tested
pipeline and evidence, not a claim that every future Blueprint evaluator or
world model must fail.

## Separately reported components

### 1. Frozen benchmark calibration: not supported

The independently labeled RoboArena calibration partition produced primary
pairwise accuracy 0.672566 with a session-cluster 95% bootstrap interval of
[0.548387, 0.777778], positive Kendall tau-b 0.333333, and the correct aggregate
top policy. Those favorable results are insufficient under the frozen rule:

- selective pairwise coverage was 0.008850, below the required 0.25;
- action-following pass rate was 0.102041, below the required 0.80;
- the registered `abstention_improves` and `action_following` gates both failed;
- a complete held-out result was never obtained because the two frozen attempts
  were rate-limited, so no held-out generalization claim is made.

The benchmark component therefore did not demonstrate a useful, trustworthy
ordering. The failed selective-abstention gate is an explicit
`thesis_not_supported_if_any` condition in the frozen protocol; it is not
relaxed because held-out scoring remained unavailable.

### 2. Captured-site transfer: execution succeeded; useful ranking did not

The exact merged-main image executed four public OpenPI DROID joint-position
checkpoints against two scene lanes and three preregistered can offsets per
scene: 24 contract-valid GPU episodes. The InteriorGS-derived visual layer
remained separate from the local MuJoCo Franka/can/tray interaction layer; the
same policy adapter and deterministic evaluator were used for the NVIDIA
Warehouse control scene. No full-site USD rebuild and no policy-specific scorer
change was required.

The frozen uncertainty rule abstained from a total ordering in both scenes:

| Scene | Mean-score order (diagnostic only) | Confident pairs | Total ranking |
|---|---|---|---|
| InteriorGS `0787_841244` | pi0-FAST 0.03575; pi0 0.02580; pi0.5 0.02472; pi0-100k 0.01693 | pi0-FAST > pi0-100k only | abstained |
| NVIDIA Warehouse | pi0.5 0.03822; pi0-100k 0.02735; pi0 0.02539; pi0-FAST 0.01986 | pi0.5 > pi0-FAST only | abstained |

Across all 24 episodes, 20 had nonzero lift progress, zero had transport
progress, zero achieved containment, and five ended stable. Independent visual
spot checks agreed that the can was displaced or knocked over rather than
placed. The transfer result proves pipeline ingestion, learned-policy execution,
scene separation, and conservative abstention. It does not provide the useful
four-policy prospective ordering required by the thesis and does not prove
site-specific physical success.

## Time, cost, and teardown

- Exact-main CPU image build: 1,319.3 seconds and at most $0.061081.
- Vast learned-policy campaign: 991 charged GPU-seconds and conservatively
  $0.206458 (single RTX 6000 Ada; frozen ceiling $0.75/hour).
- Current release-build plus GPU execution: about 38.5 serial compute minutes
  and $0.267539. The reusable image-build cost can be amortized, but it is part
  of this proof run.
- Conservative paid-provider accounting across judge attempts, exact-main
  build, and GPU execution is approximately $11.60.
- Exact Vast instance, prefix inventory, and global inventory were all proven
  absent; the pending teardown record was closed, the exclusive lease released,
  and the campaign reservation settled.

The run was inexpensive in cloud-provider terms. It did not establish a
substantial speed advantage: the benchmark's attributed OSCAR-generation plus
judge estimate was slower than the independently attributable physical
action-only lower bound, while the new-site campaign has no comparable
site-specific physical campaign. Physical monetary cost was not measured.

## Limitations and contradictions retained

- No Blueprint physical robot was purchased, rented, borrowed, operated, or
  commissioned. RoboArena supplied the independent frozen physical outcomes.
- The new scenes have no independent site-specific outcome labels; their
  results are prospective only.
- The OpenPI runtime dynamically fetched the FAST action-tokenizer helper
  without logging an explicit repository revision. Checkpoint bytes and the
  Blueprint/OpenPI/Menagerie source identities were verified, but this helper is
  a complete-provenance limitation for the run.
- The 3DGS background was a private 300,000-splat derivative. Full 3DGS depth
  occlusion and captured-site collision geometry were not reconstructed.
- Wrist observations were geometrically full-rotation after the pre-execution
  camera fix but remained strongly self-occluded in visual review.
- The four policies were public DROID policies, not policies independently
  proven on this exact can-to-tray simulator setup. Their universal task failure
  may reflect policy/task/domain mismatch as well as policy quality.

The evidence is sufficient to reject the tested capital-constrained pipeline's
claim of a useful, trustworthy policy ordering. It is not sufficient to claim
that the policies would fail physically at either new site.

## Evidence identities

- Frozen protocol: `preregistered_protocol.json`, logical protocol SHA-256
  `8ed8a7b4...` (the immutable decision thresholds are preserved from v1).
- Benchmark result: `calibration_report.json`; labels were joined only after
  its prediction inventory was complete.
- Exact-main source/image: `b28b847c...` / `sha256:f8f4dc01...`.
- Private GPU input: SHA-256 `52140529...`.
- Returned GPU archive and validator: SHA-256 `2dd36776...` and `b751b80f...`.
- Provider-zero recovery receipt: SHA-256 `f76955b1...`.

The 165 MB returned episode archive remains in the goal's private evidence
store rather than Git. Its digest, validator digest, per-episode manifest
digests, and scene-level ranking-manifest digests bind this report to the
preserved output without redistributing the InteriorGS-derived observations or
checkpoint data.
