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
from blueprint_pipeline.retained_scene_sealed_bundle_rebuild import (
    rebuild_retained_scene_bundle_from_sealed_predecessor,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="Frozen render request manifest.")
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
        help="Empty directory to write the bundle, its receipt, and the runtime tree into.",
    )
    parser.add_argument(
        "--sealed-predecessor-bundle",
        help=(
            "Rebuild mode: sealed earlier retained-scene bundle whose scientific "
            "inputs will be reopened without following its authoring-machine paths."
        ),
    )
    parser.add_argument(
        "--source-standard-splat",
        help="Rebuild mode: exact host-resident standard source PLY.",
    )
    parser.add_argument(
        "--output-root",
        help="Rebuild mode: empty collision-free production-host output root.",
    )
    args = parser.parse_args(argv)

    try:
        rebuild_mode = any(
            value
            for value in (
                args.sealed_predecessor_bundle,
                args.source_standard_splat,
                args.output_root,
            )
        )
        if rebuild_mode:
            if (
                not args.sealed_predecessor_bundle
                or not args.source_standard_splat
                or not args.output_root
                or args.request
                or args.job_dir
                or args.scene_input_root
            ):
                raise RetainedSceneRenderPacketError(
                    ["retained_scene_sealed_rebuild_cli_arguments_invalid"]
                )
            receipt = rebuild_retained_scene_bundle_from_sealed_predecessor(
                predecessor_bundle_path=args.sealed_predecessor_bundle,
                source_standard_splat_path=args.source_standard_splat,
                repo_root=args.repo_root,
                output_root=args.output_root,
            )
        else:
            if not args.request or not args.job_dir:
                raise RetainedSceneRenderPacketError(
                    ["retained_scene_render_bundle_cli_arguments_invalid"]
                )
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
                "blueprint_commit": receipt.get("blueprint_commit")
                or receipt.get("source_commit_sha"),
                "bundle_sha256": receipt.get("bundle_sha256")
                or (receipt.get("bundle_receipt") or {}).get("bundle_sha256"),
                "bundle_size_bytes": receipt.get("bundle_size_bytes"),
                "bundle_relative_path": receipt.get("bundle_relative_path"),
                "receipt_path": receipt.get("receipt_path")
                or str(
                    Path(args.job_dir).expanduser().resolve()
                    / "adp_retained_scene_gpu_render_bundle_receipt.json"
                ),
                "task_lanes": [row["task_id"] for row in receipt.get("task_lanes") or []]
                or receipt.get("task_ids"),
                "blockers": receipt.get("blockers", []),
                "provider_mutation_performed": False,
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "ready" and not receipt.get("blockers") else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
