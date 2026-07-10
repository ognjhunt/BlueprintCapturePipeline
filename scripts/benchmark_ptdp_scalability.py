#!/usr/bin/env python3
"""Benchmark bounded PTDP primitives at the declared 100k-clip ceiling."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from blueprint_pipeline.post_training_data_package import _write_jsonl  # type: ignore[import-untyped]
from blueprint_pipeline.semantic_dedup_stage import (  # type: ignore[import-untyped]
    _ann_union_find_clusters,
)


DEFAULT_CLIP_COUNT = 100_000
DEFAULT_DIMENSION = 32
ANN_TIME_SLO_SECONDS = 120.0
JSONL_TIME_SLO_SECONDS = 30.0
PROCESS_RSS_SLO_BYTES = 1024 * 1024 * 1024
JSONL_SIZE_SLO_BYTES = 128 * 1024 * 1024
ANN_CANDIDATE_SLO = 10_000_000


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def run_benchmark(*, clip_count: int, dimension: int) -> dict[str, Any]:
    if not 1 <= clip_count <= DEFAULT_CLIP_COUNT:
        raise ValueError(f"clip_count_must_be_1_to_{DEFAULT_CLIP_COUNT}")
    if not 8 <= dimension <= 256:
        raise ValueError("dimension_must_be_8_to_256")

    rss_before = _max_rss_bytes()
    rng = np.random.default_rng(1741)
    matrix = rng.standard_normal((clip_count, dimension)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    if clip_count > 1:
        matrix[-1] = matrix[0]
    ann_started = time.monotonic()
    assignments, similarities, diagnostics = _ann_union_find_clusters(
        list(matrix),
        0.95,
        table_count=16,
        projection_bits=16,
        hamming_probe_radius=1,
        bucket_capacity=8,
        max_candidates_per_vector=64,
        seed=1741,
    )
    ann_seconds = time.monotonic() - ann_started
    rss_after_ann = _max_rss_bytes()

    with tempfile.TemporaryDirectory(prefix="blueprint-ptdp-scale-") as temp_dir:
        jsonl_path = Path(temp_dir) / "episodes.jsonl"
        jsonl_started = time.monotonic()
        _write_jsonl(
            jsonl_path,
            (
                {
                    "clip_id": f"clip_{index:06d}",
                    "object_sha256": f"{index % 257:064x}",
                    "object_reference": f"objects/sha256/{index % 257:02x}",
                }
                for index in range(clip_count)
            ),
        )
        jsonl_seconds = time.monotonic() - jsonl_started
        jsonl_size_bytes = jsonl_path.stat().st_size
    rss_after_jsonl = _max_rss_bytes()

    duplicate_found = clip_count == 1 or assignments[0] == assignments[-1]
    blockers: list[str] = []
    if ann_seconds > ANN_TIME_SLO_SECONDS:
        blockers.append("ann_time_slo_exceeded")
    if jsonl_seconds > JSONL_TIME_SLO_SECONDS:
        blockers.append("jsonl_time_slo_exceeded")
    if max(rss_after_ann, rss_after_jsonl) - rss_before > PROCESS_RSS_SLO_BYTES:
        blockers.append("process_rss_slo_exceeded")
    if jsonl_size_bytes > JSONL_SIZE_SLO_BYTES:
        blockers.append("jsonl_size_slo_exceeded")
    if int(diagnostics["candidate_comparison_count"]) > ANN_CANDIDATE_SLO:
        blockers.append("ann_candidate_slo_exceeded")
    if not duplicate_found:
        blockers.append("known_duplicate_not_found")
    if diagnostics.get("pairwise_similarity_matrix_materialized") is not False:
        blockers.append("pairwise_similarity_matrix_materialized")

    return {
        "schema_version": "blueprint.ptdp_scalability_benchmark.v1",
        "status": "passed" if not blockers else "failed",
        "workload": {
            "clip_count": clip_count,
            "embedding_dimension": dimension,
            "embedding_bytes": int(matrix.nbytes),
            "known_duplicate_injected": clip_count > 1,
        },
        "slo": {
            "ann_time_seconds_max": ANN_TIME_SLO_SECONDS,
            "jsonl_time_seconds_max": JSONL_TIME_SLO_SECONDS,
            "process_rss_growth_bytes_max": PROCESS_RSS_SLO_BYTES,
            "jsonl_size_bytes_max": JSONL_SIZE_SLO_BYTES,
            "ann_candidate_comparisons_max": ANN_CANDIDATE_SLO,
        },
        "measurements": {
            "ann_seconds": round(ann_seconds, 6),
            "jsonl_seconds": round(jsonl_seconds, 6),
            "process_rss_growth_bytes": max(rss_after_ann, rss_after_jsonl) - rss_before,
            "jsonl_size_bytes": jsonl_size_bytes,
            "ann_candidate_comparison_count": diagnostics[
                "candidate_comparison_count"
            ],
            "ann_candidate_fraction": diagnostics["candidate_fraction"],
            "sparse_similarity_count": len(similarities),
            "known_duplicate_found": duplicate_found,
            "pairwise_similarity_matrix_materialized": diagnostics[
                "pairwise_similarity_matrix_materialized"
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "blockers": blockers,
        "claim_boundary": {
            "synthetic_max_count_bounded_primitives_executed": clip_count
            == DEFAULT_CLIP_COUNT,
            "full_real_media_ptdp_workload_executed": False,
            "buyer_package_quality_proven": False,
        },
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-count", type=int, default=DEFAULT_CLIP_COUNT)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(clip_count=args.clip_count, dimension=args.dimension)
        _write_json_atomic(args.output.resolve(), result)
    except (OSError, ValueError) as exc:
        print(f"[ptdp-scalability] ERROR {exc}", file=sys.stderr)
        return 1
    print(
        f"[ptdp-scalability] {result['status']} clips={args.clip_count} "
        f"ann={result['measurements']['ann_seconds']}s "
        f"jsonl={result['measurements']['jsonl_seconds']}s"
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
