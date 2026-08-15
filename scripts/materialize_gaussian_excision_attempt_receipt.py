#!/usr/bin/env python3
"""Seal one retained Gaussian-excision attempt as immutable evidence.

This is a no-provider operator surface over the production validator.  Every
input must live below the declared evidence root, and the underlying
materializer refuses overwrite and claim upgrades.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from blueprint_pipeline.adp_gaussian_excision_vast import (
    materialize_gaussian_excision_attempt_receipt,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--run-result", required=True)
    parser.add_argument("--execution-result", required=True)
    parser.add_argument("--teardown-manifest", required=True)
    parser.add_argument("--watchdog-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = materialize_gaussian_excision_attempt_receipt(
            evidence_root=args.evidence_root,
            bundle_receipt_path=args.bundle_receipt,
            run_result_path=args.run_result,
            execution_result_path=args.execution_result,
            teardown_manifest_path=args.teardown_manifest,
            watchdog_evidence_path=args.watchdog_evidence,
            output_path=args.output,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "sealed",
                "output": str(Path(args.output).expanduser().resolve()),
                "receipt_digest": receipt["receipt_digest"],
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
