#!/usr/bin/env python3
"""Run the bounded public processed-observation Task Evaluation Run proxy."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.public_processed_task_testbed import (  # noqa: E402
    compile_public_processed_task_testbed_proxy,
)


def _source_commit() -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("source_commit_invalid")
    if dirty:
        raise RuntimeError("source_checkout_not_clean")
    return commit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dataset-manifest", type=Path, required=True)
    parser.add_argument("--candidate-dataset-manifest", type=Path, required=True)
    parser.add_argument("--camera-observation-manifest", type=Path, required=True)
    parser.add_argument("--appearance-proxy-summary", type=Path, required=True)
    parser.add_argument("--appearance-ply", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--operator-identity", required=True)
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args(argv)
    summary = compile_public_processed_task_testbed_proxy(
        processed_dataset_manifest_path=args.processed_dataset_manifest,
        candidate_dataset_manifest_path=args.candidate_dataset_manifest,
        camera_observation_manifest_path=args.camera_observation_manifest,
        appearance_proxy_summary_path=args.appearance_proxy_summary,
        appearance_ply_path=args.appearance_ply,
        output_root=args.output_root,
        operator_identity=args.operator_identity,
        source_commit_sha=_source_commit(),
        timestamp=args.timestamp,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
