#!/usr/bin/env python3
"""Normalize NeoVerse service outputs into the Stage 1 contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.world_model_contract_adapter import normalize_remote_result


def normalize_backend_manifest(*, result_manifest: dict[str, object], output_dir: Path, backend_report_path: Path) -> dict[str, object]:
    return dict(
        normalize_remote_result(
            backend="neoverse",
            result_manifest=result_manifest,
            output_dir=output_dir,
            backend_report_path=backend_report_path,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize NeoVerse outputs")
    parser.add_argument("--result-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend-report", required=True)
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.result_manifest).read_text(encoding="utf-8"))
    normalize_backend_manifest(
        result_manifest=dict(manifest),
        output_dir=Path(args.output_dir),
        backend_report_path=Path(args.backend_report),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
