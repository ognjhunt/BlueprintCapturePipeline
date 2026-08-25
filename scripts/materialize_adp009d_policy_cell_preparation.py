#!/usr/bin/env python3
"""Seal all frozen task-A policy cells without a provider or policy query."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from blueprint_pipeline.adp009d_policy_cell_preparation import (
    materialize_policy_cell_matrix,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-suite", required=True)
    parser.add_argument("--policy-readiness", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = materialize_policy_cell_matrix(
            scenario_suite_path=args.scenario_suite,
            policy_readiness_path=args.policy_readiness,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                    "paid_resource_allocation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "materialized",
                "cell_count": result["cell_count"],
                "candidate_cell_count": result["candidate_cell_count"],
                "executable_candidate_cells_before_controls": result[
                    "executable_candidate_cells_before_controls"
                ],
                "materialization_digest": result["materialization_digest"],
                "output": args.output,
                "provider_mutation_performed": False,
                "paid_resource_allocation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
