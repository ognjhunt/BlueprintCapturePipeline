#!/usr/bin/env python3
"""Build and validate a repo-local same-capture lineage packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blueprint_pipeline.same_capture_lineage import (  # noqa: E402
    build_same_capture_lineage_packet,
    validate_same_capture_lineage_packet,
    write_same_capture_lineage_packet,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True, help="Local capture root containing raw/ and pipeline/ artifacts.")
    parser.add_argument("--paperclip-issue-id", help="Durable Paperclip issue id for this capture-chain proof.")
    parser.add_argument("--paperclip-issue-url", help="Optional Paperclip issue URL.")
    parser.add_argument("--write", help="Write packet to this path. Defaults to stdout only.")
    args = parser.parse_args()

    capture_root = Path(args.capture_root).expanduser().resolve()
    if args.write:
        output_path = write_same_capture_lineage_packet(
            capture_root=capture_root,
            paperclip_issue_id=args.paperclip_issue_id,
            paperclip_issue_url=args.paperclip_issue_url,
            output_path=Path(args.write),
        )
        packet = json.loads(output_path.read_text(encoding="utf-8"))
        print(str(output_path))
    else:
        packet = build_same_capture_lineage_packet(
            capture_root=capture_root,
            paperclip_issue_id=args.paperclip_issue_id,
            paperclip_issue_url=args.paperclip_issue_url,
        )
        print(json.dumps(packet, indent=2, sort_keys=True))

    validation = validate_same_capture_lineage_packet(packet)
    if validation["status"] != "valid" or packet["status"] != "repo_proven":
        print("same-capture lineage blocked:", file=sys.stderr)
        for blocker in [*validation["blockers"], *packet["repo_blockers"]]:
            print(f"- {blocker}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
