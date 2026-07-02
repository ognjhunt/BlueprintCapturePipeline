#!/usr/bin/env python3
"""Launch or analyze the G1 textured-robot render-noise audit (spec matrix A-G).

The audit isolates which part of the textured G1 render path breaks in close robot-POV
manipulation frames — missing texture assets, sample starvation, denoiser behavior, PBR
material response, lighting underexposure, or camera/pose clipping — by rendering one raw
PNG per material/render variant on ONE dynamic scene/stance/camera setup. The task string
resolves to a target through the normal scene-placement path; nothing scene-specific is
hardcoded here, so any task/site that can produce a robot POV seed frame can be audited
(kitchen/fridge is only the first regression case).

Subcommands:
  launch    bundle + stage + (optionally) run the audit on a GPU provider via the parity job
            in --render-noise-audit mode. Without --allow-paid it stops at a launchable plan.
  analyze   local re-analysis of a collected run dir: frame stats, gates, interpretation,
            contact sheet, and textured_robot_render_noise_audit_manifest.json.
  plan      write the default variant-plan JSON (A-G) for inspection or a custom run.

Claim boundary: simulator/render-quality audit only — not task success, policy quality,
physical readiness, or WAM rank fidelity evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for candidate in (str(SRC_DIR), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from blueprint_pipeline.g1_render_noise_audit import (  # noqa: E402
    AUDIT_MANIFEST_NAME,
    analyze_render_noise_audit_run,
    build_variant_plan,
)
from blueprint_pipeline.isaac_g1_kitchen_parity_job import (  # noqa: E402
    DEFAULT_G1_USD_RELATIVE,
    run_isaac_g1_kitchen_parity_job,
)


def _audit_scenario(args: argparse.Namespace) -> dict:
    scenario: dict = {
        "scenario_id": args.scenario_id,
        "instruction": args.task,
        "task": args.task,
    }
    if args.target_object_id:
        scenario["target_object_ids"] = [s.strip() for s in args.target_object_id.split(",") if s.strip()]
    if args.affordance_object_id:
        scenario["affordance_object_ids"] = [
            s.strip() for s in args.affordance_object_id.split(",") if s.strip()
        ]
    return scenario


def _cmd_launch(args: argparse.Namespace) -> int:
    scenario = _audit_scenario(args)
    manifest = run_isaac_g1_kitchen_parity_job(
        scenarios=[scenario],
        out_dir=args.out_dir,
        kitchen_asset_dir=args.kitchen_asset_dir,
        kitchen_url=args.kitchen_url,
        g1_usd=args.g1_usd,
        provider=args.provider,
        allow_paid=args.allow_paid,
        allow_dirty_paid_launch=args.allow_dirty_paid_launch,
        cold=args.cold,
        image=args.image,
        max_seconds=args.max_seconds,
        marker_timeout=args.marker_timeout,
        width=args.width,
        height=args.height,
        warm_candidates=tuple(args.warm_candidate or ()),
        warm_only=args.warm_only,
        render_subframes=args.render_subframes,
        render_noise_audit=True,
        audit_high_spp=args.audit_high_spp,
        audit_warmup_frames=args.audit_warmup_frames,
        audit_boost_light_intensity=args.audit_boost_light_intensity,
    )
    out_path = Path(args.out_dir) / "g1_render_noise_audit_job_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "status": manifest.get("status"),
        "blockers": manifest.get("blockers"),
        "render_noise_audit": manifest.get("render_noise_audit"),
        "job_manifest": str(out_path),
    }, indent=2, default=str))
    return 0 if manifest.get("status") in ("completed", "prepared") else 1


def _cmd_analyze(args: argparse.Namespace) -> int:
    manifest = analyze_render_noise_audit_run(args.run_dir, out_dir=args.out_dir)
    interpretation = manifest.get("interpretation") or {}
    print(f"[g1-render-noise-audit] status={manifest.get('status')}")
    print(f"[g1-render-noise-audit] variants={manifest.get('variants_executed')}")
    print(f"[g1-render-noise-audit] primary_diagnosis={interpretation.get('primary_diagnosis')}")
    for finding in interpretation.get("findings") or []:
        print(f"[g1-render-noise-audit] finding: {finding.get('rule')} -> {finding.get('evidence')}")
    out_root = Path(args.out_dir) if args.out_dir else Path(str(manifest.get("audit_dir") or args.run_dir))
    print(f"[g1-render-noise-audit] manifest={out_root / AUDIT_MANIFEST_NAME}")
    return 0 if manifest.get("status") == "completed" else 1


def _cmd_plan(args: argparse.Namespace) -> int:
    plan = build_variant_plan()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"[g1-render-noise-audit] plan={args.out} variants={len(plan['variants'])}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="prepare or run the GPU audit via the parity job")
    launch.add_argument("--task", required=True,
                        help='free-form task string, e.g. "open the fridge door" — resolved '
                             "dynamically against the scene (no coordinates)")
    launch.add_argument("--scenario-id", default="render_noise_audit")
    launch.add_argument("--target-object-id", default="",
                        help="optional comma-separated target object id/alias hints")
    launch.add_argument("--affordance-object-id", default="",
                        help="optional comma-separated affordance object id/alias hints")
    launch.add_argument("--out-dir", required=True)
    launch.add_argument("--kitchen-asset-dir", default=None)
    launch.add_argument("--kitchen-url", default=None,
                        help="previously staged scene asset zip signed URL (skips re-upload)")
    launch.add_argument("--g1-usd", default=DEFAULT_G1_USD_RELATIVE)
    launch.add_argument("--provider", default="runpod")
    launch.add_argument("--allow-paid", action="store_true")
    launch.add_argument("--allow-dirty-paid-launch", action="store_true")
    launch.add_argument("--cold", action="store_true")
    launch.add_argument("--warm-candidate", action="append", default=[],
                        help="RunPod stopped pod id to warm-restart before cold create (repeatable)")
    launch.add_argument("--warm-only", action="store_true")
    launch.add_argument("--image", default=None)
    launch.add_argument("--max-seconds", type=int, default=2400)
    launch.add_argument("--marker-timeout", type=int, default=900)
    launch.add_argument("--width", type=int, default=1280)
    launch.add_argument("--height", type=int, default=960)
    launch.add_argument("--render-subframes", type=int, default=16,
                        help="baseline subframes the 'current default' budget derives from")
    launch.add_argument("--audit-high-spp", type=int, default=0)
    launch.add_argument("--audit-warmup-frames", type=int, default=0)
    launch.add_argument("--audit-boost-light-intensity", type=float, default=0.0)
    launch.set_defaults(func=_cmd_launch)

    analyze = sub.add_parser("analyze", help="analyze a collected audit run dir locally")
    analyze.add_argument("--run-dir", required=True)
    analyze.add_argument("--out-dir", default=None)
    analyze.set_defaults(func=_cmd_analyze)

    plan = sub.add_parser("plan", help="write the default A-G variant plan JSON")
    plan.add_argument("--out", required=True)
    plan.set_defaults(func=_cmd_plan)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
