#!/usr/bin/env python3
"""Build a live-lane-compatible bundle from a scene-bound articulated probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from blueprint_pipeline.articulated_isaac_bundle import (
    ArticulatedIsaacBundleError,
    build_articulated_isaac_bundle,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--worker-source", required=True)
    parser.add_argument("--source-commit-sha", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = build_articulated_isaac_bundle(
            probe_root=args.probe_root,
            job_dir=args.job_dir,
            worker_source=args.worker_source,
            source_commit_sha=args.source_commit_sha,
        )
    except (OSError, ValueError, ArticulatedIsaacBundleError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    if not receipt.get("predecessor_binding_digest"):
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": ["simready_scene_predecessor_binding_missing"],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "built",
                "scene_id": receipt["scene_id"],
                "bundle_path": str(Path(receipt["bundle_path"])),
                "bundle_sha256": receipt["bundle_sha256"],
                "predecessor_binding_digest": receipt[
                    "predecessor_binding_digest"
                ],
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
