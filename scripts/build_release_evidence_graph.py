#!/usr/bin/env python3
"""Build a fail-closed release evidence graph for one launch scope."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.artifact_storage import default_evidence_root  # noqa: E402
from blueprint_pipeline.common import write_json  # noqa: E402
from blueprint_pipeline.release_evidence_graph import (  # noqa: E402
    evaluate_release_evidence_graph,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope", choices=("BASE", "SIM", "PTDP", "SC3", "PAID", "LIVE"), required=True
    )
    parser.add_argument("--repository-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=default_evidence_root() / "release_evidence",
        help=(
            "Evidence root containing v2 envelopes and their contained sources/ artifacts. "
            "Every node also needs a trusted per-node Ed25519 verifier attestation."
        ),
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=root / "docs" / "release_evidence_requirements.json",
    )
    parser.add_argument(
        "--output", type=Path, default=default_evidence_root() / "release_evidence_graph.json"
    )
    args = parser.parse_args()
    graph = evaluate_release_evidence_graph(
        scope=args.scope,
        repository_sha=args.repository_sha,
        image_digest=args.image_digest,
        evidence_dir=args.evidence_dir,
        requirements_path=args.requirements,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, graph)
    print(json.dumps({"status": graph["status"], "blockers": graph["blockers"]}))
    return int(graph["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
