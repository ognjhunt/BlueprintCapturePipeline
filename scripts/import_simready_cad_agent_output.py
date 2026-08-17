#!/usr/bin/env python3
"""Import one historical CAD-agent output onto a host-resident evidence root."""

from __future__ import annotations

import argparse
import grp
from pathlib import Path
import pwd

from blueprint_pipeline.decision_evidence_contracts import canonical_json
from blueprint_pipeline.simready_cad_agent_host_import import (
    SimReadyCadAgentHostImportError,
    materialize_simready_cad_agent_host_import,
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
        epilog="""Example (repeat for all four Scene 840920 candidates):
  python scripts/import_simready_cad_agent_output.py \\
    --source-receipt "$HOST_SOURCE_IMPORT/receipts/task_a_earthtojake.json" \\
    --destination-root "$HOST_OUTPUT/task_a_earthtojake" \\
    --artifact-root "$HOST_OUTPUT/shared-artifacts" \\
    --owner-user blueprint --owner-group blueprint \\
    --source-map "$OLD_USER_PREFIX=$HOST_SOURCE_IMPORT/user" \\
    --source-map "$OLD_TMP_PREFIX=$HOST_SOURCE_IMPORT/tmp"
""",
    )
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--destination-root", required=True, type=Path)
    parser.add_argument("--owner-user", required=True)
    parser.add_argument("--owner-group", required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Shared content-addressed root reused across repeatable candidate imports.",
    )
    parser.add_argument(
        "--source-map",
        action="append",
        default=[],
        metavar="OLD_PREFIX=HOST_ROOT",
        help="Repeatable exact historical-prefix to staged-host-root mapping.",
    )
    parser.add_argument(
        "--source-override",
        action="append",
        default=[],
        metavar="SHA256=PATH",
        help="Optional exact-digest fallback for a missing mapped source file.",
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
        owner_uid = pwd.getpwnam(args.owner_user).pw_uid
        owner_gid = grp.getgrnam(args.owner_group).gr_gid
    except KeyError as exc:
        parser.error(f"unknown owner user/group: {exc}")
    try:
        receipt = materialize_simready_cad_agent_host_import(
            source_receipt_path=args.source_receipt,
            destination_root=args.destination_root,
            source_prefix_mappings=source_maps,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
            source_overrides=dict(override_pairs),
            artifact_root=args.artifact_root,
        )
    except SimReadyCadAgentHostImportError as exc:
        parser.error(str(exc))
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
