#!/usr/bin/env python3
"""Seal one canonical, non-scoring policy diagnostic execution spec."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from blueprint_pipeline.native_task_arena_policy_diagnostic_bundle import (
    build_policy_diagnostic_execution_spec,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-id",
        required=True,
        choices=("pi05_droid", "groot_n17_droid"),
    )
    parser.add_argument("--scene-plan", required=True)
    parser.add_argument("--construction-result", required=True)
    parser.add_argument("--control-result", required=True)
    parser.add_argument("--scene-policy-readiness")
    parser.add_argument("--scenario-suite")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        optional = {}
        if args.scene_policy_readiness:
            optional["scene_policy_readiness_path"] = args.scene_policy_readiness
        if args.scenario_suite:
            optional["scenario_suite_path"] = args.scenario_suite
        result = build_policy_diagnostic_execution_spec(
            candidate_id=args.candidate_id,
            scene_plan_path=args.scene_plan,
            construction_result_path=args.construction_result,
            control_result_path=args.control_result,
            output_path=args.output,
            **optional,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "sealed_diagnostic",
                "candidate_id": result["candidate_id"],
                "execution_spec_digest": result["execution_spec_digest"],
                "claim_ceiling": result["claim_ceiling"],
                "output": args.output,
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
