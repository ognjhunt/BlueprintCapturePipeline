#!/usr/bin/env python3
"""Build the deterministic local Site Reference Database v1 fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.site_reference_fixture import (  # noqa: E402
    build_site_reference_database_v1_fixture,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a local staged-capture fixture through Site Reference Database v1 summary projection."
    )
    parser.add_argument(
        "--output-root",
        default="output/site-reference-database-v1-fixture",
        help="Directory where the fixture source bundle and local storage tree will be rebuilt.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path for a machine-readable summary of generated fixture paths.",
    )
    args = parser.parse_args(argv)

    try:
        result = build_site_reference_database_v1_fixture(args.output_root)
    except Exception as exc:
        print(f"[site-reference-fixture] FAILED: {exc}")
        return 1

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"[site-reference-fixture] capture_root={result['capture_root']}")
    print(f"[site-reference-fixture] site_reference_index={result['site_reference_index_path']}")
    print(f"[site-reference-fixture] summary_projection={result['summary_projection_path']}")
    print(f"[site-reference-fixture] retrieval_validation={result['retrieval_validation_path']}")
    print(f"[site-reference-fixture] reference_ids={','.join(result['reference_ids'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
