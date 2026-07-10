from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.semantic_dedup_stage import (
    _ann_union_find_clusters,
    DownsampledPixelEmbeddingProvider,
    SemanticDedupConfig,
    cosine_similarity,
    dedup_clips,
    resample_trajectory,
    run_semantic_dedup_stage,
    trajectory_rms_distance,
)


_FIXTURE_CONFIG = SemanticDedupConfig(
    production_mode=False,
    min_keyframe_embeddings=1,
)


def test_ann_index_bounds_candidates_and_finds_exact_duplicate() -> None:
    rng = np.random.default_rng(17)
    matrix = rng.standard_normal((1000, 32)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix[-1] = matrix[0]

    assignments, _similarities, diagnostics = _ann_union_find_clusters(
        list(matrix),
        0.95,
        table_count=16,
        projection_bits=8,
        hamming_probe_radius=1,
        bucket_capacity=8,
        max_candidates_per_vector=64,
        seed=1741,
    )

    assert assignments[0] == assignments[-1]
    assert diagnostics["pairwise_similarity_matrix_materialized"] is False
    assert diagnostics["bucket_capacity"] == 8
    assert diagnostics["max_candidates_per_vector"] == 64
    assert diagnostics["candidate_comparison_count"] < diagnostics[
        "exhaustive_pair_count"
    ]


def test_semantic_dedup_rejects_workload_over_explicit_clip_limit() -> None:
    with pytest.raises(PipelineError, match="semantic_dedup_clip_count_exceeds_limit"):
        dedup_clips(
            [{"clip_id": "one"}, {"clip_id": "two"}],
            config=SemanticDedupConfig(
                production_mode=False,
                min_keyframe_embeddings=1,
                max_clip_count=1,
            ),
        )


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _pose_matrix(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> List[List[float]]:
    mat = np.eye(4)
    mat[0, 3] = x
    mat[1, 3] = y
    mat[2, 3] = z
    return mat.tolist()


def _clip(
    clip_id: str,
    *,
    image_path: Optional[str],
    positions: List[tuple],
    session_id: str = "session-a",
) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    for i, (x, y, z) in enumerate(positions):
        frame: Dict[str, Any] = {
            "frame_id": f"{i:06d}",
            "timestamp": i / 30.0,
            "T_world_camera": _pose_matrix(x=x, y=y, z=z),
        }
        if image_path is not None:
            frame["image_path"] = image_path
        frames.append(frame)
    return {"clip_id": clip_id, "session_id": session_id, "frames": frames}


def _line(n: int, *, axis: str = "x", step: float = 0.05) -> List[tuple]:
    points = []
    for i in range(n):
        d = step * i
        points.append((d, 0.0, 0.0) if axis == "x" else (0.0, 0.0, d))
    return points


def _scene_a_image() -> np.ndarray:
    # Horizontal ramp.
    return np.tile(np.linspace(10.0, 240.0, 64), (64, 1))


def _scene_b_image() -> np.ndarray:
    # Vertical ramp: decorrelated from the horizontal ramp after centering.
    return np.tile(np.linspace(10.0, 240.0, 64).reshape(-1, 1), (1, 64))


def _scene_c_image() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.integers(0, 256, size=(64, 64)).astype(np.float64)


def _write_image(bundle_dir: Path, rel: str, image: np.ndarray) -> str:
    path = bundle_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, image)
    return rel


def _write_bundle(bundle_dir: Path, clips: List[Dict[str, Any]]) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "clips_manifest.json").write_text(
        json.dumps({"clips": clips}), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Provider + geometry primitives
# ---------------------------------------------------------------------------


def test_fixture_provider_is_deterministic_and_discriminative() -> None:
    provider = DownsampledPixelEmbeddingProvider()
    a1 = provider.embed_image(_scene_a_image())
    a2 = provider.embed_image(_scene_a_image())
    b = provider.embed_image(_scene_b_image())
    assert cosine_similarity(a1, a2) == pytest.approx(1.0)
    assert cosine_similarity(a1, b) < 0.95


def test_trajectory_rms_resamples_to_common_length() -> None:
    # Same straight path sampled at different rates: RMS ~ 0.
    dense = np.asarray(_line(81, step=0.02))
    sparse = np.asarray(_line(41, step=0.04))
    assert dense[-1, 0] == pytest.approx(sparse[-1, 0])
    rms = trajectory_rms_distance(dense, sparse, resample_points=32)
    assert rms < 1e-9
    assert resample_trajectory(dense, 32).shape == (32, 3)

    # Orthogonal paths: RMS well above the duplicate floor.
    other = np.asarray(_line(81, axis="z", step=0.02))
    assert trajectory_rms_distance(dense, other, resample_points=32) > 0.5


# ---------------------------------------------------------------------------
# Dedup behavior
# ---------------------------------------------------------------------------


def test_duplicate_clips_collapse_to_one_kept_member(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/scene_a.npy", _scene_a_image())

    # Same content captured in two sessions at different frame rates.
    clip_1 = _clip("clip_dup_1", image_path=rel, positions=_line(81, step=0.02), session_id="s1")
    clip_2 = _clip("clip_dup_2", image_path=rel, positions=_line(41, step=0.04), session_id="s2")

    manifest = dedup_clips(
        [clip_1, clip_2], config=_FIXTURE_CONFIG, bundle_dir=bundle
    )
    by_id = {c["clip_id"]: c for c in manifest["clips"]}
    assert by_id["clip_dup_1"]["kept"] is True
    assert by_id["clip_dup_2"]["kept"] is False
    dropped = by_id["clip_dup_2"]
    assert dropped["cluster_id"] == by_id["clip_dup_1"]["cluster_id"]
    assert dropped["duplicate_of_clip_id"] == "clip_dup_1"
    assert dropped["drop_reason"] == "duplicate_trajectory_within_visual_cluster"
    assert dropped["similarity_to_kept"] > 0.95
    assert dropped["trajectory_rms_to_kept_m"] < 0.10
    assert manifest["kept_clip_ids"] == ["clip_dup_1"]
    assert manifest["dropped_clip_ids"] == ["clip_dup_2"]


def test_same_scene_different_trajectories_all_kept(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/scene_b.npy", _scene_b_image())

    clip_x = _clip("clip_walk_x", image_path=rel, positions=_line(60, axis="x"))
    clip_z = _clip("clip_walk_z", image_path=rel, positions=_line(60, axis="z"))

    manifest = dedup_clips(
        [clip_x, clip_z], config=_FIXTURE_CONFIG, bundle_dir=bundle
    )
    by_id = {c["clip_id"]: c for c in manifest["clips"]}
    # Visually clustered together...
    assert by_id["clip_walk_x"]["cluster_id"] == by_id["clip_walk_z"]["cluster_id"]
    # ...but trajectory verification keeps both (diverse motion).
    assert by_id["clip_walk_x"]["kept"] is True
    assert by_id["clip_walk_z"]["kept"] is True
    assert by_id["clip_walk_z"]["trajectory_rms_to_kept_m"] > 0.10
    assert sorted(manifest["kept_clip_ids"]) == ["clip_walk_x", "clip_walk_z"]


def test_different_scenes_form_separate_clusters(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel_a = _write_image(bundle, "frames/scene_a.npy", _scene_a_image())
    rel_b = _write_image(bundle, "frames/scene_b.npy", _scene_b_image())
    rel_c = _write_image(bundle, "frames/scene_c.npy", _scene_c_image())

    clips = [
        _clip("clip_a", image_path=rel_a, positions=_line(30)),
        _clip("clip_b", image_path=rel_b, positions=_line(30)),
        _clip("clip_c", image_path=rel_c, positions=_line(30)),
    ]
    manifest = dedup_clips(clips, config=_FIXTURE_CONFIG, bundle_dir=bundle)
    cluster_ids = {c["clip_id"]: c["cluster_id"] for c in manifest["clips"]}
    assert len(set(cluster_ids.values())) == 3
    # Identical trajectories in *different* scenes are not duplicates.
    assert manifest["coverage"]["post_dedup_clip_count"] == 3
    assert manifest["dropped_clip_ids"] == []


def test_clip_without_keyframe_image_is_kept_unclustered(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/scene_a.npy", _scene_a_image())
    clips = [
        _clip("clip_with_image", image_path=rel, positions=_line(30)),
        _clip("clip_no_image", image_path=None, positions=_line(30)),
    ]
    manifest = dedup_clips(clips, config=_FIXTURE_CONFIG, bundle_dir=bundle)
    by_id = {c["clip_id"]: c for c in manifest["clips"]}
    assert by_id["clip_no_image"]["cluster_id"] is None
    assert by_id["clip_no_image"]["embeddable"] is False
    assert by_id["clip_no_image"]["kept"] is True
    assert by_id["clip_no_image"]["note"] == "not_embeddable_kept_fixture_only"


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_thresholds_are_config_driven(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/scene_b.npy", _scene_b_image())
    clip_x = _clip("clip_walk_x", image_path=rel, positions=_line(60, axis="x"))
    clip_z = _clip("clip_walk_z", image_path=rel, positions=_line(60, axis="z"))

    # Default RMS floor keeps both; a huge floor treats them as duplicates.
    strict = SemanticDedupConfig.from_dict(
        {
            "min_trajectory_rms_m": 100.0,
            "production_mode": False,
            "min_keyframe_embeddings": 1,
        }
    )
    manifest = dedup_clips([clip_x, clip_z], config=strict, bundle_dir=bundle)
    assert manifest["kept_clip_ids"] == ["clip_walk_x"]
    assert manifest["dropped_clip_ids"] == ["clip_walk_z"]

    # A stricter similarity threshold splits the visual cluster entirely.
    exact = SemanticDedupConfig(
        similarity_threshold=1.0,
        production_mode=False,
        min_keyframe_embeddings=1,
    )
    manifest = dedup_clips([clip_x, clip_z], config=exact, bundle_dir=bundle)
    assert (
        {c["clip_id"]: c["cluster_id"] for c in manifest["clips"]}["clip_walk_x"]
        != {c["clip_id"]: c["cluster_id"] for c in manifest["clips"]}["clip_walk_z"]
    )
    assert manifest["coverage"]["post_dedup_clip_count"] == 2


def test_config_loadable_from_yaml_and_unknown_keys_rejected(tmp_path: Path) -> None:
    yaml_path = tmp_path / "dedup.yaml"
    yaml_path.write_text(
        "similarity_threshold: 0.9\nmin_trajectory_rms_m: 0.25\n", encoding="utf-8"
    )
    config = SemanticDedupConfig.from_file(yaml_path)
    assert config.similarity_threshold == 0.9
    assert config.min_trajectory_rms_m == 0.25
    # Untouched keys keep the documented OSCAR default.
    assert SemanticDedupConfig().similarity_threshold == 0.95

    with pytest.raises(PipelineError, match="Unknown semantic dedup config"):
        SemanticDedupConfig.from_dict({"similarity": 0.9})


# ---------------------------------------------------------------------------
# Provider provenance + stage entry point
# ---------------------------------------------------------------------------


class _CustomProvider:
    name = "unit_test_provider"
    version = "9.9-test"

    def __init__(self) -> None:
        self._inner = DownsampledPixelEmbeddingProvider()

    def embed_image(self, gray: np.ndarray) -> np.ndarray:
        return self._inner.embed_image(gray)


class _ProductionProvider(_CustomProvider):
    name = "dinov3"
    version = "1.0"
    production_ready = True
    model_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    revision = "ea8dc2863c51be0a264bab82070e3e8836b02d51"


def test_stage_writes_manifest_with_provider_provenance_and_coverage(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/scene_a.npy", _scene_a_image())
    clips = [
        _clip("clip_dup_1", image_path=rel, positions=_line(81, step=0.02), session_id="s1"),
        _clip("clip_dup_2", image_path=rel, positions=_line(41, step=0.04), session_id="s2"),
        _clip("clip_walk_z", image_path=rel, positions=_line(60, axis="z"), session_id="s3"),
    ]
    _write_bundle(bundle, clips)

    result = run_semantic_dedup_stage(
        bundle_dir=bundle,
        provider=_CustomProvider(),
        config=_FIXTURE_CONFIG,
    )
    assert result["status"] == "completed_fixture_only"
    assert result["input_clip_count"] == 3
    assert result["post_dedup_clip_count"] == 2
    assert result["dropped_duplicate_count"] == 1
    assert result["embedding_provider"]["name"] == "unit_test_provider"
    assert result["embedding_provider"]["version"] == "9.9-test"

    manifest_path = Path(result["manifest_path"])
    assert manifest_path.is_file()
    assert "derived" in manifest_path.parts

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "semantic_dedup_manifest.v2"
    provider = manifest["embedding_provider"]
    assert provider["name"] == "unit_test_provider"
    assert provider["is_production_backend"] is False
    assert "siglip" in provider["production_backends"]
    assert "dinov3" in provider["production_backends"]

    # Coverage counts are post-dedup.
    assert manifest["coverage"]["input_clip_count"] == 3
    assert manifest["coverage"]["post_dedup_clip_count"] == 2
    assert len(manifest["kept_clip_ids"]) == 2

    # Cluster records carry similarity + RMS values for audit.
    cluster_members = [m for cluster in manifest["clusters"] for m in cluster["members"]]
    dropped = [m for m in cluster_members if not m["kept"]]
    assert len(dropped) == 1
    assert dropped[0]["clip_id"] == "clip_dup_2"
    assert dropped[0]["similarity_to_kept"] is not None
    assert dropped[0]["trajectory_rms_to_kept_m"] is not None
    assert dropped[0]["duplicate_of_clip_id"] == "clip_dup_1"

    # Default fixture provider provenance is recorded when none is injected.
    default_result = run_semantic_dedup_stage(bundle_dir=bundle, output_dir=tmp_path / "out_b")
    assert default_result["embedding_provider"]["name"] == "downsampled_pixel_fixture"
    assert default_result["status"] == "blocked"
    assert default_result["production_accepted_clip_ids"] == []
    assert "production_embedding_provider_not_approved" in default_result["production_blockers"]

    # Raw inputs untouched.
    original = json.loads((bundle / "clips_manifest.json").read_text(encoding="utf-8"))
    assert len(original["clips"]) == 3


def test_production_provider_accepts_only_fully_verifiable_clips(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/scene_a.npy", _scene_a_image())
    clips = [
        _clip("clip-valid", image_path=rel, positions=_line(20)),
        _clip("clip-no-image", image_path=None, positions=_line(20)),
    ]
    _write_bundle(bundle, clips)

    result = run_semantic_dedup_stage(
        bundle_dir=bundle,
        provider=_ProductionProvider(),
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    assert result["status"] == "blocked"
    assert result["production_accepted_clip_ids"] == []
    assert "one_or_more_clips_missing_required_keyframe_embeddings" in result[
        "production_blockers"
    ]
    by_id = {record["clip_id"]: record for record in manifest["clips"]}
    assert by_id["clip-no-image"]["kept"] is False
    assert by_id["clip-no-image"]["drop_reason"] == "keyframe_embedding_unverifiable"


def test_relative_se3_trajectory_dedup_is_global_transform_invariant(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rel = _write_image(bundle, "frames/scene_a.npy", _scene_a_image())
    base = _clip("base", image_path=rel, positions=_line(30))
    transformed = _clip("transformed", image_path=rel, positions=_line(30))
    angle = np.pi / 2.0
    global_transform = np.eye(4)
    global_transform[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    global_transform[:3, 3] = [5.0, -3.0, 2.0]
    for frame in transformed["frames"]:
        local_pose = np.asarray(frame["T_world_camera"], dtype=np.float64)
        frame["T_world_camera"] = (global_transform @ local_pose).tolist()

    manifest = dedup_clips(
        [base, transformed],
        config=_FIXTURE_CONFIG,
        bundle_dir=bundle,
    )

    assert manifest["kept_clip_ids"] == ["base"]
    assert manifest["dropped_clip_ids"] == ["transformed"]
    assert manifest["clips"][1]["trajectory_rms_to_kept_m"] == pytest.approx(0.0)
