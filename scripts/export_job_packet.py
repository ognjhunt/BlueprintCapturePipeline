#!/usr/bin/env python3
"""Write one job's job packet folder, and its operator record beside it, from files pulled from the host.

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
    JobInputs, JobPacketRefused, OwnedCaptureApproval, build_job_packet,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull-root", type=Path, required=True,
                        help="the directory pulled into, laid out as the host's site root: <root>[/<bucket>]/scenes/...")
    parser.add_argument("--capture-dir", type=Path, required=True,
                        help="the pulled <root>[/<bucket>]/scenes/site-<request>/captures/<capture id>/ directory")
    parser.add_argument("--operator", required=True, help="who is making this packet (kept in the operator record)")
    parser.add_argument("--output-root", type=Path, required=True, help="where packet-<number>/ is made")
    parser.add_argument("--team-scene-dir", type=Path, help="the team scene packet directory (team_scene_inputs.v1.json)")
    parser.add_argument("--configured-scene-dir", type=Path,
                        help="the configured scene publication directory (configured_scene_bundle.v1.zip, object USDZ)")
    parser.add_argument("--screening-publication", type=Path, help="beta-policy-screening/publications/<campaign>.json")
    parser.add_argument("--review-videos-dir", type=Path, help="pipeline/beta_screening_review_videos/<attempt>/")
    parser.add_argument("--video", type=Path, help="the job's video, when raw/ holds more than one")
    parser.add_argument("--blueprint-owned-capture", action="store_true")
    parser.add_argument("--capture-id", help="with --blueprint-owned-capture: the capture id on the allow-list")
    parser.add_argument("--approved-by")
    parser.add_argument("--approved-on", help="the date of the owner's approval, YYYY-MM-DD")
    parser.add_argument("--approval-evidence", help="the owner's words, quoted")
    args = parser.parse_args(argv)
    owned = None
    if args.blueprint_owned_capture:
        if not (args.capture_id and args.approved_by and args.approved_on and args.approval_evidence):
            parser.error("--blueprint-owned-capture needs --capture-id, --approved-by, --approved-on and "
                         "--approval-evidence")
        owned = OwnedCaptureApproval(capture_id=args.capture_id, approved_by=args.approved_by,
                                     approval_evidence=args.approval_evidence, approved_on=args.approved_on)
    inputs = JobInputs(capture_dir=args.capture_dir, team_scene_dir=args.team_scene_dir,
                       configured_scene_dir=args.configured_scene_dir,
                       screening_publication=args.screening_publication, review_videos_dir=args.review_videos_dir,
                       video=args.video, pull_root=args.pull_root)
    try:
        manifest = build_job_packet(inputs, output_root=args.output_root, owned=owned, operator=args.operator)
    except JobPacketRefused as exc:
        print(json.dumps({"status": "refused", "code": exc.code, "detail": str(exc)}))
        return 2
    print(json.dumps({"status": "written", "packet_dir": manifest["packet_dir"],
                      "operator_record": manifest["operator_record"], "packet_id": manifest["packet_id"],
                      "packet_number": manifest["packet_number"], "files": len(manifest["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
