# SPEC-03: Semantic deduplication stage (embedding clustering + trajectory RMS)

- Status: Proposed
- Priority: **P0 — launch blocker** (for Post-Training Data Package quality)
- Area: new stage; touches `retrieval_index_stage.py`, `synthesis/reference_selection.py`
- Paper: OSCAR (arXiv 2606.04463) §semantic deduplication

## Problem

OSCAR uses a two-stage semantic dedup — SigLIP visual clustering (>0.95 similarity) to
find similar-background clips, then trajectory RMS-distance verification to drop true
duplicates while keeping diverse motions in similar scenes. We have **no content-based
dedup at all**:

- The only dedup mechanisms are geometric/temporal: stationary-pan dropping by pose/time
  stride (`retrieval_index_stage.py:973-980`) and `near_duplicate` selection by pose
  distance `0.12 m` + temporal/frame-index gaps
  (`synthesis/reference_selection.py:20`, `:389-402`).
- DINOv3 cosine similarity exists in the repo (`frame_alignment_stage.py:379`) but is used
  only for cross-session SE(3) alignment, not dedup.

Two visually identical clips captured from different poses, sessions, or devices both
pass. As capture supply scales (the whole point of beta), packages will accumulate
redundant content, inflating apparent coverage while adding no training/eval value.

## Why this blocks beta

Buyers pay per package on the premise of curated, diverse coverage. Redundancy is a
silent quality defect that gets worse with scale and is expensive to retrofit once
packages have shipped (buyers would need re-issues). OSCAR treats dedup as a required
stage, not an optimization.

## Proposed fix

Add a `semantic_dedup_stage` between curation (SPEC-02) and package export:

1. **Stage 1 — visual clustering**: embed clip keyframes with an image-text encoder
   (SigLIP per the paper; the existing DINOv3 embedding path is an acceptable first
   implementation since the infra already exists — keep the encoder swappable per
   WORLD_MODEL_STRATEGY_CONTEXT.md). Cluster clips with cosine similarity above a
   configurable threshold (default 0.95).
2. **Stage 2 — trajectory verification**: within each visual cluster, compute RMS
   distance between (a) camera-pose trajectories for walkthrough clips or (b) action/EE
   trajectories for manipulation episodes. Drop members below an RMS floor as duplicates;
   keep diverse-motion members.
3. Record dedup decisions in the package manifest (cluster id, kept/dropped, similarity
   and RMS values) so provenance survives and decisions are auditable/reversible.
4. Wire the same index into buyer-facing coverage stats so "N clips" always means N
   post-dedup clips.

## Acceptance criteria

- [ ] Duplicated fixture clips (same content, different session ids) collapse to one kept member with the drop recorded in the manifest.
- [ ] Same-scene different-motion fixtures are all kept (trajectory verification works).
- [ ] Encoder is behind an interface with the model name recorded per run (swappable backend).
- [ ] Thresholds config-driven; defaults documented against OSCAR's 0.95.
- [ ] Package coverage counts reflect post-dedup totals.
