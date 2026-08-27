"""Argument parsing for the ADP Content Agents bundle builder."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


BundleBuilder = Callable[..., Mapping[str, Any]]


def run(argv: list[str] | None, *, bundle_builder: BundleBuilder) -> int:
    """Parse the legacy module CLI and invoke the supplied bundle builder."""

    parser = argparse.ArgumentParser(
        description="Build the immutable ADP-009A Content Agents Vast bundle."
    )
    parser.add_argument(
        "--repo-root", default=str(Path(__file__).resolve().parents[2])
    )
    parser.add_argument("--content-agents-root", required=True)
    parser.add_argument(
        "--reference-image",
        action="append",
        help="Repeat for every exact rights-admitted reference image.",
    )
    parser.add_argument("--job-dir", required=True)
    parser.add_argument(
        "--input-variant",
        choices=(
            "control_v1",
            "match_v2",
            "articulated_v1",
            "agent_cad_v1",
            "paired_target_registered_v1",
        ),
        default="control_v1",
    )
    parser.add_argument("--evidence-root")
    parser.add_argument("--agent-cad-output-manifest")
    parser.add_argument("--agent-mesh-projection-receipt")
    parser.add_argument("--paired-target-construction-bindings")
    parser.add_argument("--paired-target-task-id")
    parser.add_argument("--reference-rights-authority")
    parser.add_argument("--content-agents-execution-route")
    parser.add_argument("--historical-replay-only", action="store_true")
    args = parser.parse_args(argv)
    receipt = bundle_builder(
        repo_root=args.repo_root,
        content_agents_root=args.content_agents_root,
        reference_image_paths=args.reference_image,
        job_dir=args.job_dir,
        input_variant=args.input_variant,
        evidence_root=args.evidence_root,
        agent_cad_output_manifest_path=args.agent_cad_output_manifest,
        agent_mesh_projection_receipt_path=args.agent_mesh_projection_receipt,
        paired_target_construction_bindings_path=(
            args.paired_target_construction_bindings
        ),
        paired_target_task_id=args.paired_target_task_id,
        reference_rights_authority_path=args.reference_rights_authority,
        content_agents_execution_route_path=args.content_agents_execution_route,
        historical_replay_only=args.historical_replay_only,
    )
    print(json.dumps(dict(receipt), indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "ready" else 2


__all__ = ["run"]
