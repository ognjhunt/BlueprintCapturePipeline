#!/usr/bin/env python3
"""Materialize and rehearse one sealed, private-derived Aura residual bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_json
from blueprint_pipeline.public_scene_aura_exact_residual_bundle import (
    build_aura_exact_residual_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build only a no-upload Aura exact-residual bundle and invoke its "
            "zero-cost provider-entrypoint rehearsal."
        )
    )
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--aura-source-directory", type=Path, required=True)
    parser.add_argument("--lama-source-directory", type=Path, required=True)
    parser.add_argument("--wonderworld-source-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Checkout whose sealed runner is copied into the provider bundle.",
    )
    args = parser.parse_args()
    receipt = build_aura_exact_residual_bundle(
        preflight_path=args.preflight,
        aura_source_directory=args.aura_source_directory,
        lama_source_directory=args.lama_source_directory,
        wonderworld_source_directory=args.wonderworld_source_directory,
        output_root=args.output_root,
        repo_root=args.repo_root,
    )
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
