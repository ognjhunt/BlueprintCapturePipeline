"""Semantic deduplication stage (SPEC-03, launch-audit-2026-07-02).

Two-stage semantic dedup modeled on OSCAR (arXiv 2606.04463):

1. **Stage 1 — visual clustering.** Embed one keyframe per clip behind a
   swappable :class:`EmbeddingProvider` interface and single-linkage cluster
   clips whose keyframe cosine similarity exceeds a configurable threshold
   (default 0.95, per OSCAR). Production deployments wire a SigLIP or DINOv3
   provider (the DINOv3 embedding path already exists in
   ``frame_alignment_stage``); this module ships a deterministic
   downsampled-pixel provider so tests and offline lanes run without model
   weights. The provider name/version is recorded in every manifest so
   dedup decisions stay auditable per WORLD_MODEL_STRATEGY_CONTEXT.md
   (world-model backends must remain swappable).

2. **Stage 2 — trajectory verification.** Within each visual cluster,
   camera-pose trajectories are resampled to a common length and compared by
   RMS distance. Members whose RMS distance to an already-kept member falls
   below the floor are true duplicates: the first member is kept, the rest
   are dropped. Diverse-motion members in the same scene are all kept.

Outputs a dedup manifest (cluster id, kept/dropped per clip, similarity and
RMS values, provider provenance) and post-dedup coverage counts — buyer
facing "N clips" must always mean N *post-dedup* clips.

Doctrine: raw capture inputs are read-only. This stage reads clip records
and keyframe images and writes derived artifacts to
``derived/semantic_dedup``. Nothing is fabricated: clips whose keyframes
cannot be embedded are left un-clustered (recorded as ``not_embeddable``),
and cluster members whose trajectories cannot be compared are dropped fail
closed (config can opt out).

Usage (library):

    from blueprint_pipeline.semantic_dedup_stage import run_semantic_dedup_stage
    result = run_semantic_dedup_stage(bundle_dir=...)

Usage (CLI):

    python -m blueprint_pipeline.semantic_dedup_stage <bundle_dir> \
        [--config thresholds.yaml] [--output-dir OUT]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from .clip_curation_stage import load_clip_records, load_image_gray
from .camera_geometry_validation import validate_se3_matrix
from .common import PipelineError, utc_now_iso, write_json
from .logging_utils import log_event

logger = logging.getLogger("blueprint.semantic_dedup")

DEDUP_MANIFEST_SCHEMA_VERSION = "semantic_dedup_manifest.v2"
DEFAULT_OUTPUT_SUBDIR = Path("derived") / "semantic_dedup"

# Encoders expected in production. Kept as documentation-of-intent in the
# manifest so downstream consumers can tell a fixture run from a real one.
PRODUCTION_EMBEDDING_BACKENDS = ("siglip", "dinov3")


# ---------------------------------------------------------------------------
# Embedding provider interface (swappable backend)
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Swappable keyframe embedding backend.

    Production wires SigLIP (per the OSCAR paper) or the existing DINOv3
    path from ``frame_alignment_stage``; tests use
    :class:`DownsampledPixelEmbeddingProvider`. Implementations must be
    deterministic for a given input and must expose ``name`` and
    ``version`` so provenance is recorded per run.
    """

    name: str
    version: str

    def embed_image(self, gray: np.ndarray) -> np.ndarray:
        """Return a 1-D float embedding for a grayscale (0..255) image."""
        ...


class DownsampledPixelEmbeddingProvider:
    """Deterministic test/fixture embedding provider (no model weights).

    Block-averages the grayscale image to ``grid x grid``, mean-centers, and
    L2-normalizes. Identical images embed identically (cosine similarity
    1.0) and unrelated structure decorrelates, which is sufficient for
    offline tests and fixture bundles. NOT a production encoder — production
    runs must wire SigLIP/DINOv3 via :class:`EmbeddingProvider`.
    """

    name = "downsampled_pixel_fixture"
    version = "1.0"
    production_ready = False
    model_id = ""
    revision = ""

    def __init__(self, grid: int = 16) -> None:
        self.grid = int(grid)

    def embed_image(self, gray: np.ndarray) -> np.ndarray:
        gray = np.asarray(gray, dtype=np.float64)
        if gray.ndim != 2 or gray.shape[0] < self.grid or gray.shape[1] < self.grid:
            # Nearest-neighbour upsample tiny images to the grid.
            rows = np.linspace(0, gray.shape[0] - 1, self.grid).round().astype(int)
            cols = np.linspace(0, gray.shape[1] - 1, self.grid).round().astype(int)
            small = gray[np.ix_(rows, cols)]
        else:
            h = (gray.shape[0] // self.grid) * self.grid
            w = (gray.shape[1] // self.grid) * self.grid
            cropped = gray[:h, :w]
            small = cropped.reshape(
                self.grid, h // self.grid, self.grid, w // self.grid
            ).mean(axis=(1, 3))
        flat = small.reshape(-1)
        centered = flat - flat.mean()
        norm = float(np.linalg.norm(centered))
        if norm <= 1e-12:
            # Constant image: no visual structure to compare.
            return np.zeros_like(flat)
        return centered / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise PipelineError(f"Embedding dims differ: {a.shape} vs {b.shape}")
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticDedupConfig:
    """Thresholds for the two-stage semantic dedup.

    ``similarity_threshold`` defaults to 0.95, matching OSCAR's SigLIP
    clustering threshold. ``min_trajectory_rms_m`` is the duplicate floor:
    within a visual cluster, a member whose resampled camera-pose trajectory
    is within this RMS distance of an already-kept member is dropped as a
    duplicate. Loadable from YAML/JSON via :meth:`from_file`.
    """

    # Stage 1: keyframe cosine-similarity clustering (OSCAR: >0.95).
    similarity_threshold: float = 0.95
    # Stage 2: trajectories are linearly resampled to this many points
    # before RMS comparison so clips of different lengths are comparable.
    trajectory_resample_points: int = 32
    # RMS floor (metres): below this, two same-cluster trajectories are the
    # same motion -> duplicate. OSCAR verifies trajectory RMS within visual
    # clusters; the metre-scale default is tuned for site walkthroughs.
    min_trajectory_rms_m: float = 0.10
    # Fail-closed: cluster members whose trajectory cannot be compared are
    # dropped (they cannot demonstrate diverse motion). Set true to keep.
    keep_unverifiable_cluster_members: bool = False
    # Production decisions require a real encoder, multiple keyframes, and
    # measurable SE(3)-relative trajectories. Fixture mode remains explicit.
    production_mode: bool = True
    keyframe_samples: int = 3
    min_keyframe_embeddings: int = 3
    # Deterministic multi-table angular LSH bounds similarity candidate work.
    ann_table_count: int = 16
    ann_projection_bits: int = 16
    ann_hamming_probe_radius: int = 1
    ann_bucket_capacity: int = 8
    ann_max_candidates_per_vector: int = 64
    ann_seed: int = 1741
    max_clip_count: int = 100_000
    max_embedding_memory_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise PipelineError("similarity_threshold must be in (0, 1]")
        if self.trajectory_resample_points < 2:
            raise PipelineError("trajectory_resample_points must be >= 2")
        if self.min_trajectory_rms_m < 0.0:
            raise PipelineError("min_trajectory_rms_m must be non-negative")
        if self.keyframe_samples < 1:
            raise PipelineError("keyframe_samples must be >= 1")
        if not 1 <= self.min_keyframe_embeddings <= self.keyframe_samples:
            raise PipelineError(
                "min_keyframe_embeddings must be between 1 and keyframe_samples"
            )
        if not 1 <= self.ann_table_count <= 64:
            raise PipelineError("ann_table_count must be between 1 and 64")
        if not 2 <= self.ann_projection_bits <= 32:
            raise PipelineError("ann_projection_bits must be between 2 and 32")
        if not 0 <= self.ann_hamming_probe_radius <= 1:
            raise PipelineError("ann_hamming_probe_radius must be 0 or 1")
        if not 1 <= self.ann_bucket_capacity <= 256:
            raise PipelineError("ann_bucket_capacity must be between 1 and 256")
        if not 1 <= self.ann_max_candidates_per_vector <= 4096:
            raise PipelineError(
                "ann_max_candidates_per_vector must be between 1 and 4096"
            )
        if self.max_clip_count < 1:
            raise PipelineError("max_clip_count must be positive")
        if self.max_embedding_memory_bytes < 1024 * 1024:
            raise PipelineError("max_embedding_memory_bytes must be at least 1 MiB")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticDedupConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise PipelineError(
                f"Unknown semantic dedup config keys: {unknown}; known keys: {sorted(known)}"
            )
        return cls(**dict(payload))

    @classmethod
    def from_file(cls, path: str | Path) -> "SemanticDedupConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            payload = yaml.safe_load(text) or {}
        else:
            payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise PipelineError(f"Semantic dedup config at {path} must be a mapping")
        return cls.from_dict(payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Keyframes and trajectories
# ---------------------------------------------------------------------------


def _clip_keyframe_images(
    clip: Mapping[str, Any], bundle_dir: Optional[Path], *, sample_count: int
) -> List[np.ndarray]:
    """Load deterministic first/middle/last image-bearing keyframes."""

    frames = clip.get("frames") or []
    candidates = [f for f in frames if isinstance(f, Mapping) and f.get("image_path")]
    if not candidates:
        return []
    sample_count = min(max(1, sample_count), len(candidates))
    indices = sorted(
        {
            int(round(index))
            for index in np.linspace(0, len(candidates) - 1, sample_count)
        }
    )
    images: List[np.ndarray] = []
    for index in indices:
        raw_path = Path(str(candidates[index]["image_path"]))
        path = raw_path if raw_path.is_absolute() or bundle_dir is None else bundle_dir / raw_path
        image = load_image_gray(path)
        if image is not None:
            images.append(image)
    return images


def _clip_trajectory(clip: Mapping[str, Any]) -> Optional[np.ndarray]:
    poses: List[np.ndarray] = []
    for frame in clip.get("frames") or []:
        if not isinstance(frame, Mapping):
            continue
        T = frame.get("T_world_camera")
        if T is None:
            continue
        validation = validate_se3_matrix(T, field="T_world_camera")
        if not validation["valid"]:
            return None
        poses.append(np.asarray(validation["matrix"], dtype=np.float64))
    if len(poses) < 2:
        return None
    origin = poses[0]
    origin_rotation_inverse = origin[:3, :3].T
    origin_translation = origin[:3, 3]
    # Express positions in the first camera frame. This is invariant to a
    # session-wide rigid translation/rotation while preserving relative motion.
    relative = [
        origin_rotation_inverse @ (pose[:3, 3] - origin_translation)
        for pose in poses
    ]
    return np.asarray(relative, dtype=np.float64)


def _aggregate_keyframe_embeddings(
    images: Sequence[np.ndarray],
    *,
    provider: EmbeddingProvider,
    minimum: int,
) -> tuple[Optional[np.ndarray], List[str]]:
    vectors: List[np.ndarray] = []
    hashes: List[str] = []
    for image in images:
        array = np.asarray(image)
        hashes.append(hashlib.sha256(array.tobytes(order="C")).hexdigest())
        vector = np.asarray(provider.embed_image(array), dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.size and np.isfinite(vector).all() and norm > 1e-12:
            vectors.append(vector / norm)
    if len(vectors) < minimum or len({vector.shape for vector in vectors}) != 1:
        return None, hashes
    concatenated = np.concatenate(vectors)
    norm = float(np.linalg.norm(concatenated))
    return (concatenated / norm if norm > 1e-12 else None), hashes


def resample_trajectory(trajectory: np.ndarray, points: int) -> np.ndarray:
    """Linearly resample an (n, 3) trajectory to (points, 3)."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    n = trajectory.shape[0]
    if n == points:
        return trajectory
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, points)
    return np.stack(
        [np.interp(dst, src, trajectory[:, axis]) for axis in range(trajectory.shape[1])],
        axis=1,
    )


def trajectory_rms_distance(a: np.ndarray, b: np.ndarray, *, resample_points: int) -> float:
    """RMS pointwise distance between two trajectories, resampled to a
    common length first so clips of different frame counts are comparable."""
    ra = resample_trajectory(a, resample_points)
    rb = resample_trajectory(b, resample_points)
    deltas = np.linalg.norm(ra - rb, axis=1)
    return float(np.sqrt(np.mean(deltas**2)))


# ---------------------------------------------------------------------------
# Clustering (stage 1)
# ---------------------------------------------------------------------------


def _ann_union_find_clusters(
    embeddings: Sequence[Optional[np.ndarray]],
    threshold: float,
    *,
    table_count: int,
    projection_bits: int,
    hamming_probe_radius: int,
    bucket_capacity: int,
    max_candidates_per_vector: int,
    seed: int,
) -> Tuple[List[Optional[int]], Dict[Tuple[int, int], float], Dict[str, Any]]:
    """Bounded single-linkage clustering with deterministic angular LSH.

    Returns (cluster assignment per index or None when not embeddable,
    sparse candidate similarities, index diagnostics).
    """
    n = len(embeddings)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    dimensions = {
        np.asarray(item).reshape(-1).shape[0]
        for item in embeddings
        if item is not None
    }
    if len(dimensions) > 1:
        raise PipelineError("Embedding dims differ across semantic dedup inputs")
    dimension = next(iter(dimensions), 0)
    rng = np.random.default_rng(int(seed))
    projections = [
        rng.standard_normal((projection_bits, dimension), dtype=np.float32)
        for _ in range(table_count)
    ]
    buckets: List[Dict[int, List[int]]] = [{} for _ in range(table_count)]
    pair_sims: Dict[Tuple[int, int], float] = {}
    candidate_comparison_count = 0

    def signatures(vector: np.ndarray) -> List[int]:
        result: List[int] = []
        for projection in projections:
            bits = projection @ vector
            signature = 0
            for bit_index, enabled in enumerate(bits >= 0.0):
                if bool(enabled):
                    signature |= 1 << bit_index
            result.append(signature)
        return result

    for i in range(n):
        if embeddings[i] is None:
            continue
        vector = np.asarray(embeddings[i], dtype=np.float32).reshape(-1)
        vector_signatures = signatures(vector)
        candidates: set[int] = set()
        # Exact buckets come first so a known duplicate cannot be displaced by
        # multiprobe noise when the global per-vector comparison cap is hit.
        for table_index, signature in enumerate(vector_signatures):
            table = buckets[table_index]
            candidates.update(table.get(signature, ()))
        if hamming_probe_radius == 1 and len(candidates) < max_candidates_per_vector:
            for table_index, signature in enumerate(vector_signatures):
                table = buckets[table_index]
                for bit_index in range(projection_bits):
                    candidates.update(table.get(signature ^ (1 << bit_index), ()))
                    if len(candidates) >= max_candidates_per_vector:
                        break
                if len(candidates) >= max_candidates_per_vector:
                    break
        for j in sorted(candidates)[:max_candidates_per_vector]:
            candidate_comparison_count += 1
            sim = cosine_similarity(embeddings[j], embeddings[i])
            if sim > threshold:
                pair_sims[(j, i)] = sim
                union(j, i)
        for table_index, signature in enumerate(vector_signatures):
            bucket = buckets[table_index].setdefault(signature, [])
            if len(bucket) < bucket_capacity:
                bucket.append(i)

    root_to_cluster: Dict[int, int] = {}
    assignments: List[Optional[int]] = []
    next_cluster = 0
    for i in range(n):
        if embeddings[i] is None:
            assignments.append(None)
            continue
        root = find(i)
        if root not in root_to_cluster:
            root_to_cluster[root] = next_cluster
            next_cluster += 1
        assignments.append(root_to_cluster[root])
    exhaustive_pair_count = n * max(0, n - 1) // 2
    return assignments, pair_sims, {
        "backend": "deterministic_angular_lsh",
        "table_count": table_count,
        "projection_bits": projection_bits,
        "hamming_probe_radius": hamming_probe_radius,
        "bucket_capacity": bucket_capacity,
        "max_candidates_per_vector": max_candidates_per_vector,
        "seed": int(seed),
        "embedding_dimension": dimension,
        "candidate_comparison_count": candidate_comparison_count,
        "exhaustive_pair_count": exhaustive_pair_count,
        "candidate_fraction": (
            round(candidate_comparison_count / exhaustive_pair_count, 8)
            if exhaustive_pair_count
            else 0.0
        ),
        "pairwise_similarity_matrix_materialized": False,
    }


# ---------------------------------------------------------------------------
# Dedup core (pure)
# ---------------------------------------------------------------------------


def dedup_clips(
    clips: Sequence[Mapping[str, Any]],
    *,
    config: Optional[SemanticDedupConfig] = None,
    provider: Optional[EmbeddingProvider] = None,
    bundle_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run two-stage semantic dedup over clip records; returns the manifest.

    Pure with respect to the bundle: reads clip records and keyframe images,
    writes nothing.
    """
    config = config or SemanticDedupConfig()
    provider = provider or DownsampledPixelEmbeddingProvider()
    if len(clips) > config.max_clip_count:
        raise PipelineError(
            f"semantic_dedup_clip_count_exceeds_limit:{len(clips)}>{config.max_clip_count}"
        )

    clip_ids = [
        str(clip.get("clip_id") or clip.get("id") or f"clip_{index}")
        for index, clip in enumerate(clips)
    ]

    minimum_embeddings = (
        config.min_keyframe_embeddings if config.production_mode else 1
    )
    embeddings: List[Optional[np.ndarray]] = []
    keyframe_hashes: List[List[str]] = []
    for clip in clips:
        images = _clip_keyframe_images(
            clip,
            bundle_dir,
            sample_count=config.keyframe_samples,
        )
        embedding, hashes = _aggregate_keyframe_embeddings(
            images,
            provider=provider,
            minimum=minimum_embeddings,
        )
        embeddings.append(embedding)
        keyframe_hashes.append(hashes)

    embedding_memory_bytes = sum(
        int(np.asarray(embedding).nbytes)
        for embedding in embeddings
        if embedding is not None
    )
    if embedding_memory_bytes > config.max_embedding_memory_bytes:
        raise PipelineError(
            "semantic_dedup_embedding_memory_limit_exceeded:"
            f"{embedding_memory_bytes}>{config.max_embedding_memory_bytes}"
        )
    assignments, pair_sims, ann_diagnostics = _ann_union_find_clusters(
        embeddings,
        config.similarity_threshold,
        table_count=config.ann_table_count,
        projection_bits=config.ann_projection_bits,
        hamming_probe_radius=config.ann_hamming_probe_radius,
        bucket_capacity=config.ann_bucket_capacity,
        max_candidates_per_vector=config.ann_max_candidates_per_vector,
        seed=config.ann_seed,
    )

    trajectories = [_clip_trajectory(clip) for clip in clips]

    member_records: List[Dict[str, Any]] = []
    for index, clip_id in enumerate(clip_ids):
        member_records.append(
            {
                "clip_id": clip_id,
                "cluster_id": assignments[index],
                "kept": True,
                "embeddable": embeddings[index] is not None,
                "keyframe_sha256": keyframe_hashes[index],
                "keyframe_count": len(keyframe_hashes[index]),
                "trajectory_verifiable": trajectories[index] is not None,
                "similarity_to_kept": None,
                "trajectory_rms_to_kept_m": None,
                "drop_reason": None,
            }
        )
        if embeddings[index] is None:
            if config.production_mode:
                member_records[index]["kept"] = False
                member_records[index]["drop_reason"] = "keyframe_embedding_unverifiable"
                member_records[index]["note"] = "not_embeddable_excluded_from_production"
            else:
                member_records[index]["note"] = "not_embeddable_kept_fixture_only"
        elif trajectories[index] is None and config.production_mode:
            member_records[index]["kept"] = False
            member_records[index]["drop_reason"] = "trajectory_not_measurable_fail_closed"

    # Stage 2: trajectory verification within each cluster, in input order
    # (keep-first policy: the earliest member of a duplicate group wins).
    clusters: Dict[int, List[int]] = {}
    for index, cluster_id in enumerate(assignments):
        if cluster_id is not None and member_records[index]["kept"]:
            clusters.setdefault(cluster_id, []).append(index)

    def similarity_for(left: int, right: int) -> float:
        pair = (min(left, right), max(left, right))
        cached = pair_sims.get(pair)
        if cached is not None:
            return cached
        left_embedding = embeddings[left]
        right_embedding = embeddings[right]
        if left_embedding is None or right_embedding is None:
            return 0.0
        return cosine_similarity(left_embedding, right_embedding)

    for cluster_id, members in sorted(clusters.items()):
        kept_members: List[int] = []
        for index in members:
            record = member_records[index]
            if not kept_members:
                kept_members.append(index)
                continue
            duplicate_of: Optional[int] = None
            duplicate_rms: Optional[float] = None
            unverifiable_against: Optional[int] = None
            for kept_index in kept_members:
                traj_a = trajectories[index]
                traj_b = trajectories[kept_index]
                if traj_a is None or traj_b is None:
                    unverifiable_against = kept_index
                    continue
                rms = trajectory_rms_distance(
                    traj_a, traj_b, resample_points=config.trajectory_resample_points
                )
                if rms < config.min_trajectory_rms_m:
                    duplicate_of = kept_index
                    duplicate_rms = rms
                    break
            if duplicate_of is not None:
                record["kept"] = False
                record["duplicate_of_clip_id"] = clip_ids[duplicate_of]
                record["similarity_to_kept"] = round(
                    similarity_for(index, duplicate_of), 6
                )
                record["trajectory_rms_to_kept_m"] = round(float(duplicate_rms), 6)
                record["drop_reason"] = "duplicate_trajectory_within_visual_cluster"
            elif unverifiable_against is not None and not config.keep_unverifiable_cluster_members:
                record["kept"] = False
                record["duplicate_of_clip_id"] = clip_ids[unverifiable_against]
                record["similarity_to_kept"] = round(
                    similarity_for(index, unverifiable_against), 6
                )
                record["drop_reason"] = "trajectory_not_measurable_fail_closed"
            else:
                kept_members.append(index)
                if kept_members[0] != index:
                    record["similarity_to_kept"] = round(
                        similarity_for(index, kept_members[0]), 6
                    )
                    traj_a = trajectories[index]
                    traj_b = trajectories[kept_members[0]]
                    if traj_a is not None and traj_b is not None:
                        record["trajectory_rms_to_kept_m"] = round(
                            trajectory_rms_distance(
                                traj_a,
                                traj_b,
                                resample_points=config.trajectory_resample_points,
                            ),
                            6,
                        )

    kept_clip_ids = [r["clip_id"] for r in member_records if r["kept"]]
    dropped = [r for r in member_records if not r["kept"]]

    provider_model_id = str(getattr(provider, "model_id", "") or "").strip()
    provider_revision = str(getattr(provider, "revision", "") or "").strip()
    provider_production_ready = bool(getattr(provider, "production_ready", False))
    provider_is_production = (
        provider.name in PRODUCTION_EMBEDDING_BACKENDS
        and provider_production_ready
        and bool(provider_model_id)
        and len(provider_revision) == 40
        and all(character in "0123456789abcdef" for character in provider_revision.lower())
    )
    production_blockers: List[str] = []
    if config.production_mode and not provider_is_production:
        production_blockers.append("production_embedding_provider_not_approved")
    if config.production_mode and any(not record["embeddable"] for record in member_records):
        production_blockers.append("one_or_more_clips_missing_required_keyframe_embeddings")
    if config.production_mode and any(
        not record["trajectory_verifiable"] for record in member_records
    ):
        production_blockers.append("one_or_more_clips_missing_valid_relative_se3_trajectory")
    production_accepted_clip_ids = (
        kept_clip_ids if config.production_mode and not production_blockers else []
    )

    cluster_summaries = [
        {
            "cluster_id": cluster_id,
            "members": [
                {
                    key: member_records[index][key]
                    for key in (
                        "clip_id",
                        "kept",
                        "similarity_to_kept",
                        "trajectory_rms_to_kept_m",
                        "drop_reason",
                    )
                    if key in member_records[index]
                }
                | (
                    {"duplicate_of_clip_id": member_records[index]["duplicate_of_clip_id"]}
                    if "duplicate_of_clip_id" in member_records[index]
                    else {}
                )
                for index in members
            ],
        }
        for cluster_id, members in sorted(clusters.items())
    ]

    return {
        "schema_version": DEDUP_MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "config": config.to_dict(),
        "embedding_provider": {
            "name": provider.name,
            "version": provider.version,
            "model_id": provider_model_id or None,
            "revision": provider_revision or None,
            "production_ready_declared": provider_production_ready,
            "production_backends": list(PRODUCTION_EMBEDDING_BACKENDS),
            "is_production_backend": provider_is_production,
        },
        "scalability": {
            **ann_diagnostics,
            "input_clip_limit": config.max_clip_count,
            "embedding_memory_bytes": embedding_memory_bytes,
            "embedding_memory_limit_bytes": config.max_embedding_memory_bytes,
            "keyframes_loaded_one_clip_at_a_time": True,
            "bounded_candidate_index": True,
        },
        "clusters": cluster_summaries,
        "clips": member_records,
        "kept_clip_ids": kept_clip_ids,
        "production_accepted_clip_ids": production_accepted_clip_ids,
        "dropped_clip_ids": [r["clip_id"] for r in dropped],
        "production_status": "passed" if config.production_mode and not production_blockers else "blocked",
        "production_blockers": sorted(set(production_blockers)),
        # Coverage counts are POST-dedup by contract: buyer-facing "N clips"
        # must never include duplicates.
        "coverage": {
            "input_clip_count": len(member_records),
            "post_dedup_clip_count": len(kept_clip_ids),
            "dropped_duplicate_count": len(dropped),
        },
    }


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run_semantic_dedup_stage(
    *,
    bundle_dir: str | Path,
    config: Optional[SemanticDedupConfig] = None,
    config_path: Optional[str | Path] = None,
    provider: Optional[EmbeddingProvider] = None,
    output_dir: Optional[str | Path] = None,
    accepted_clip_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Dedup the clips of a bundle directory and write the dedup manifest.

    Raw capture inputs under ``bundle_dir`` are read-only; the dedup
    manifest is written to ``<bundle_dir>/derived/semantic_dedup`` (or
    ``output_dir``).
    """
    bundle_dir = Path(bundle_dir)
    if config is not None and config_path is not None:
        raise PipelineError("Pass either config or config_path, not both")
    if config_path is not None:
        config = SemanticDedupConfig.from_file(config_path)
    config = config or SemanticDedupConfig()

    clips = load_clip_records(bundle_dir)
    if accepted_clip_ids is not None:
        accepted = {str(item).strip() for item in accepted_clip_ids if str(item).strip()}
        clips = [
            clip
            for index, clip in enumerate(clips)
            if str(
                clip.get("clip_id")
                or clip.get("id")
                or f"clip_{index:06d}"
            ).strip()
            in accepted
        ]
    manifest = dedup_clips(clips, config=config, provider=provider, bundle_dir=bundle_dir)
    manifest["canonical_input_filter"] = {
        "applied": accepted_clip_ids is not None,
        "accepted_clip_ids": sorted(
            {str(item).strip() for item in accepted_clip_ids or [] if str(item).strip()}
        ),
        "exact_input_match": accepted_clip_ids is not None
        and len(clips)
        == len({str(item).strip() for item in accepted_clip_ids or [] if str(item).strip()}),
    }

    out_dir = Path(output_dir) if output_dir is not None else bundle_dir / DEFAULT_OUTPUT_SUBDIR
    manifest_path = out_dir / "semantic_dedup_manifest.json"
    write_json(manifest_path, manifest)

    log_event(
        logger,
        logging.INFO,
        "semantic_dedup_stage_complete",
        bundle_dir=bundle_dir,
        input_clip_count=manifest["coverage"]["input_clip_count"],
        post_dedup_clip_count=manifest["coverage"]["post_dedup_clip_count"],
        dropped_duplicate_count=manifest["coverage"]["dropped_duplicate_count"],
        embedding_provider=manifest["embedding_provider"]["name"],
        manifest_path=manifest_path,
    )

    return {
        "status": (
            "blocked"
            if config.production_mode and manifest["production_blockers"]
            else "completed"
            if config.production_mode
            else "completed_fixture_only"
        ),
        "bundle_dir": str(bundle_dir),
        "manifest_path": str(manifest_path),
        "embedding_provider": dict(manifest["embedding_provider"]),
        "input_clip_count": manifest["coverage"]["input_clip_count"],
        "post_dedup_clip_count": manifest["coverage"]["post_dedup_clip_count"],
        "dropped_duplicate_count": manifest["coverage"]["dropped_duplicate_count"],
        "kept_clip_ids": list(manifest["kept_clip_ids"]),
        "production_accepted_clip_ids": list(manifest["production_accepted_clip_ids"]),
        "production_blockers": list(manifest["production_blockers"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m blueprint_pipeline.semantic_dedup_stage",
        description="Two-stage semantic dedup (visual clustering + trajectory RMS).",
    )
    parser.add_argument("bundle_dir", type=Path, help="Bundle directory with clips_manifest.json")
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON threshold overrides")
    parser.add_argument("--output-dir", type=Path, default=None, help="Derived artifact directory")
    args = parser.parse_args(argv)

    result = run_semantic_dedup_stage(
        bundle_dir=args.bundle_dir,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] != "blocked" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
