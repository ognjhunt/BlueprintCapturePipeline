#!/usr/bin/env python3
"""Build the retained-scene GPU render bundle from a frozen request.

``build_retained_scene_gpu_render_bundle`` had no caller outside its own test.
The bundle the live lane runs on was therefore produced by an invocation that
existed only in one session's shell history: nothing in the repository said how
it was built, against which scene root, or how to build it again. When the
control-plane host was rebuilt on 2026-08-12, the bundle bytes were staged by
hand and their receipt still pointed at the authoring workstation.

The bundle must be rebuilt whenever the deployed commit moves, because the
allocator refuses a bundle whose ``blueprint_commit`` is not the commit it is
running. That makes "how do I rebuild this" a routine question, not a rare one,
and it deserves an answer in the repository rather than in a memory.

Reads retained scene bytes and writes one job directory; performs no provider
mutation and no network access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from blueprint_pipeline.adp_retained_scene_render_packet import (
    RetainedSceneRenderPacketError,
    build_retained_scene_gpu_render_bundle,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request",
        required=True,
        help="Frozen render request manifest.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Checkout the bundle is sealed at; its HEAD becomes blueprint_commit.",
    )
    parser.add_argument(
        "--scene-input-root",
        help=(
            "Directory the request's relative scene inputs resolve against. "
            "Required for a portable request; unused by one naming absolute paths."
        ),
    )
    parser.add_argument(
        "--job-dir",
        required=True,
        help="Empty directory to write the bundle, its receipt, and the runtime tree into.",
    )
    args = parser.parse_args(argv)

    try:
        receipt = build_retained_scene_gpu_render_bundle(
            request_path=args.request,
            repo_root=args.repo_root,
            job_dir=args.job_dir,
            scene_input_root=args.scene_input_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        codes = (
            list(exc.codes)
            if isinstance(exc, RetainedSceneRenderPacketError)
            else [type(exc).__name__]
        )
        print(
            json.dumps(
                {
                    "schema_version": "adp009d_retained_scene_gpu_render_bundle.v1",
                    "status": "blocked",
                    "blockers": codes,
                    "provider_mutation_performed": False,
                },
                indent=1,
                sort_keys=True,
            )
        )
        return 2

    print(
        json.dumps(
            {
                "status": receipt["status"],
                "blueprint_commit": receipt["blueprint_commit"],
                "bundle_sha256": receipt["bundle_sha256"],
                "bundle_size_bytes": receipt["bundle_size_bytes"],
                "bundle_relative_path": receipt["bundle_relative_path"],
                "receipt_path": str(
                    Path(args.job_dir).expanduser().resolve()
                    / "adp_retained_scene_gpu_render_bundle_receipt.json"
                ),
                "task_lanes": [row["task_id"] for row in receipt["task_lanes"]],
                "blockers": receipt["blockers"],
                "provider_mutation_performed": False,
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "ready" and not receipt["blockers"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
