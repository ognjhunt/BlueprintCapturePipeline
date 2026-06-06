#!/usr/bin/env python3
"""Stage a raw download bundle onto VM-backed storage and optionally run qualification steps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.local_bundle_workflow import run_local_bundle_workflow  # noqa: E402


def _print_summary(result: dict[str, object]) -> None:
    print(f"[stage-capture-bundle] capture_root={result['capture_root']}")
    preflight = result.get("preflight")
    if isinstance(preflight, dict):
        print(f"[stage-capture-bundle] preflight_status={preflight.get('status')}")
    if result.get("qualification"):
        print("[stage-capture-bundle] qualification=completed")
    if result.get("evaluation_prep"):
        eval_result = result["evaluation_prep"]
        if isinstance(eval_result, dict):
            print(f"[stage-capture-bundle] evaluation_prep_manifest={eval_result.get('manifest_path')}")
            print(f"[stage-capture-bundle] evaluation_prep_status={eval_result.get('status')}")

    commands = result.get("commands")
    if isinstance(commands, dict):
        print("[stage-capture-bundle] next_commands:")
        for name, command in commands.items():
            print(f"  {name}: {command}")

    remaining = result.get("remaining_runtime_requirements")
    if isinstance(remaining, dict):
        print("[stage-capture-bundle] remaining_runtime_requirements:")
        for name, values in remaining.items():
            if isinstance(values, list):
                print(f"  {name}: {', '.join(str(value) for value in values)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage a raw Capture App bundle and optionally run the current "
            "World Labs/package/CPU-preflight pipeline"
        )
    )
    parser.add_argument("--source-bundle", required=True, help="Path to the raw download folder that contains raw/")
    parser.add_argument("--storage-root", required=True, help="Parent directory that will contain the bucket root")
    parser.add_argument("--bucket", default="local-blueprint", help="Bucket directory name under storage-root")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--link", action="store_true", help="Stage raw/ as a symlink (default)")
    mode.add_argument("--copy", action="store_true", help="Copy raw/ into the staged capture root")
    parser.add_argument("--force", action="store_true", help="Replace an existing staged capture root")
    parser.add_argument("--run-qualification", action="store_true", help="Run preflight, materialization, and qualification after staging")
    parser.add_argument("--run-evaluation-prep", action="store_true", help="Run evaluation-prep after qualification")
    parser.add_argument(
        "--pipeline-lane",
        default="current",
        choices=(
            "current",
            "qualification",
            "evaluation_prep",
            "simulation_automation",
            "scene_memory",
            "retrieval_index",
            "frame_alignment",
            "synthesis_coverage_validation",
            "cosmos_single_capture_smoke",
            "all",
        ),
        help=(
            "Pipeline lane to request when --run-qualification is set. "
            "current/all expands to qualification, evaluation_prep, and simulation_automation."
        ),
    )
    parser.add_argument("--json-output", help="Optional path to write the full result payload as JSON")
    args = parser.parse_args(argv)

    try:
        result = run_local_bundle_workflow(
            source_bundle=args.source_bundle,
            storage_root=args.storage_root,
            bucket=args.bucket,
            mode="copy" if args.copy else "link",
            force=bool(args.force),
            run_qualification=bool(args.run_qualification),
            run_evaluation_prep=bool(args.run_evaluation_prep),
            pipeline_lane=args.pipeline_lane,
        )
    except Exception as exc:
        print(f"[stage-capture-bundle] FAILED: {exc}")
        return 1

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
