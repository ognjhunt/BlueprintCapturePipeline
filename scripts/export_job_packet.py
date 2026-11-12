#!/usr/bin/env python3
"""Write one job's job packet folder from files the operator pulled from the host (offline, local writes only).

Classification: local artifact writer. No network, no door, no provider. See
docs/runbooks/job-packet.md and src/blueprint_pipeline/job_packet_export.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blueprint_pipeline.job_packet_export import (  # noqa: E402
    LEVELS, JobInputs, JobPacketRefused, OwnedCaptureApproval, build_job_packet,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True,
                        help="the pulled scenes/site-<request>/captures/<capture id>/ directory")
    parser.add_argument("--output-root", type=Path, required=True, help="where packet-<capture>-<date>/ is made")
    parser.add_argument("--level", choices=LEVELS, default="summary")
    parser.add_argument("--team-scene-dir", type=Path, help="the team scene packet directory (team_scene_inputs.v1.json)")
    parser.add_argument("--configured-scene-dir", type=Path,
                        help="the configured scene publication directory (configured_scene_bundle.v1.zip, object USDZ)")
    parser.add_argument("--screening-publication", type=Path, help="beta-policy-screening/publications/<campaign>.json")
    parser.add_argument("--review-videos-dir", type=Path, help="pipeline/beta_screening_review_videos/<attempt>/")
    parser.add_argument("--first-look-file", type=Path, help="the website's first look, exported as .md or .txt")
    parser.add_argument("--first-look-checked-by", help="who checked the exported first look names no site details")
    parser.add_argument("--redact-term", action="append", default=[],
                        help="a site detail (name, address) the summary's first look must not contain; repeatable")
    parser.add_argument("--video", type=Path, help="the job's video, when raw/ holds more than one")
    parser.add_argument("--blueprint-owned-capture", action="store_true")
    parser.add_argument("--capture-id", help="with --blueprint-owned-capture: the capture id on the allow-list")
    parser.add_argument("--approved-by")
    parser.add_argument("--approval-evidence", help="the owner's words, quoted")
    args = parser.parse_args(argv)
    owned = None
    if args.blueprint_owned_capture:
        if not (args.capture_id and args.approved_by and args.approval_evidence):
            parser.error("--blueprint-owned-capture needs --capture-id, --approved-by and --approval-evidence")
        owned = OwnedCaptureApproval(capture_id=args.capture_id, approved_by=args.approved_by,
                                     approval_evidence=args.approval_evidence)
    inputs = JobInputs(capture_dir=args.capture_dir, team_scene_dir=args.team_scene_dir,
                       configured_scene_dir=args.configured_scene_dir,
                       screening_publication=args.screening_publication, review_videos_dir=args.review_videos_dir,
                       first_look_file=args.first_look_file, video=args.video)
    try:
        manifest = build_job_packet(inputs, output_root=args.output_root, level=args.level, owned=owned,
                                    first_look_checked_by=args.first_look_checked_by, redact_terms=args.redact_term)
    except JobPacketRefused as exc:
        print(json.dumps({"status": "refused", "code": exc.code, "detail": str(exc)}))
        return 2
    print(json.dumps({"status": "written", "packet_dir": manifest["packet_dir"], "packet_id": manifest["packet_id"],
                      "level": manifest["level"], "files": len(manifest["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
