#!/usr/bin/env python3
"""Rebuild exhaustive CAD visual-review evidence from host-imported candidates."""

from __future__ import annotations

import argparse
import grp
import json
from pathlib import Path
import pwd

from blueprint_pipeline.decision_evidence_contracts import canonical_json
from blueprint_pipeline.simready_cad_agent_host_import import (
    SimReadyCadAgentHostImportError,
    materialize_cad_visual_review_host_rematerialization,
)


def _pair(value: str, *, option: str) -> tuple[str, str]:
    left, separator, right = value.partition("=")
    if not separator or not left or not right:
        raise argparse.ArgumentTypeError(f"{option} requires LEFT=RIGHT")
    return left, right


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""The source review is exhaustive: pass all four imported candidates
(Task A/Task B x earthtojake/pan_chera) and both exact source mappings.
Selected candidates alone are intentionally insufficient.
""",
    )
    parser.add_argument(
        "--cad-host-import-receipt",
        action="append",
        required=True,
        type=Path,
        help="Repeat for every candidate covered by the source exhaustive review.",
    )
    parser.add_argument("--source-visual-review", required=True, type=Path)
    parser.add_argument("--expected-source-visual-review-digest", required=True)
    parser.add_argument("--expected-source-visual-review-sha256", required=True)
    parser.add_argument(
        "--expected-source-visual-review-size-bytes", required=True, type=int
    )
    parser.add_argument("--destination-root", required=True, type=Path)
    parser.add_argument("--owner-user", required=True)
    parser.add_argument("--owner-group", required=True)
    parser.add_argument(
        "--expected-candidate",
        action="append",
        required=True,
        metavar="JSON",
        help=(
            "Repeat exact slot/task/asset/backend/source_receipt_digest binding; "
            "Scene 840920 requires four."
        ),
    )
    parser.add_argument(
        "--source-map",
        action="append",
        required=True,
        metavar="OLD_PREFIX=HOST_ROOT",
    )
    parser.add_argument(
        "--source-override",
        action="append",
        default=[],
        metavar="SHA256=PATH",
    )
    args = parser.parse_args()
    source_maps = [
        _pair(value, option="--source-map") for value in args.source_map
    ]
    override_pairs = [
        _pair(value, option="--source-override")
        for value in args.source_override
    ]
    if len({digest for digest, _path in override_pairs}) != len(override_pairs):
        parser.error("duplicate --source-override digest")
    try:
        expected_candidates = [json.loads(value) for value in args.expected_candidate]
    except json.JSONDecodeError as exc:
        parser.error(f"invalid --expected-candidate JSON: {exc}")
    try:
        owner_uid = pwd.getpwnam(args.owner_user).pw_uid
        owner_gid = grp.getgrnam(args.owner_group).gr_gid
    except KeyError as exc:
        parser.error(f"unknown owner user/group: {exc}")
    try:
        receipt = materialize_cad_visual_review_host_rematerialization(
            cad_host_import_receipt_paths=args.cad_host_import_receipt,
            source_visual_review_path=args.source_visual_review,
            destination_root=args.destination_root,
            source_prefix_mappings=source_maps,
            expected_candidates=expected_candidates,
            expected_source_visual_review_digest=(
                args.expected_source_visual_review_digest
            ),
            expected_source_visual_review_sha256=(
                args.expected_source_visual_review_sha256
            ),
            expected_source_visual_review_size_bytes=(
                args.expected_source_visual_review_size_bytes
            ),
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            source_overrides=dict(override_pairs),
        )
    except SimReadyCadAgentHostImportError as exc:
        parser.error(str(exc))
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
