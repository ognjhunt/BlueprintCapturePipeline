"""Prepare or compare retained-workcell runtime measurements on the control plane."""
from __future__ import annotations
import json
from pathlib import Path
from .decision_evidence_contracts import canonical_digest
from .native_task_runtime_comparison import (
    BASE_ARENA_COMMIT, BASE_COMMIT, BASE_IMAGE, GA_ARENA_COMMIT, GA_COMMIT, GA_IMAGE,
    construction_comparison_binding, validate_construction_comparison_request,
    compare_runtime_measurements,
)

def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-construction")
    prepare.add_argument("--packet-dir", required=True)
    prepare.add_argument("--variant", choices=("baseline", "isaaclab_3_ga"), required=True)
    prepare.add_argument("--output", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("runtime_comparison_output_exists")
    if args.command == "prepare-construction":
        from .native_task_arena_bundle import verify_native_task_arena_packet
        from .native_task_construction_plan import materialize_native_task_construction_phase_plan
        root, receipt, _ = verify_native_task_arena_packet(args.packet_dir)
        plan = json.loads((root / "native_task_arena_scene_plan.v1.json").read_text())
        phase_plan = materialize_native_task_construction_phase_plan(plan)
        ga = args.variant == "isaaclab_3_ga"
        value = {"schema_version": "native_task_runtime_comparison_request.v1",
            "scope": "retained_construction_replay", "variant": args.variant,
            "packet_receipt_digest": receipt["receipt_digest"],
            "binding": construction_comparison_binding(plan, phase_plan),
            "runtime_identity": {"container_image": GA_IMAGE if ga else BASE_IMAGE,
                "isaaclab_commit": GA_COMMIT if ga else BASE_COMMIT,
                "arena_commit": GA_ARENA_COMMIT if ga else BASE_ARENA_COMMIT,
                "physics_backend": "isaacsim_physx", "renderer": "isaac_rtx"},
            "candidate_policy_queried": False, "runtime_promotion_authorized": False}
        value["request_digest"] = canonical_digest(value, digest_field="request_digest")
        validate_construction_comparison_request(value, plan=plan, phase_plan=phase_plan)
    else:
        value = compare_runtime_measurements(json.loads(Path(args.baseline).read_text()),
                                            json.loads(Path(args.candidate).read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "status": value.get("status", "prepared"),
                      "blockers": value.get("blockers", [])}))
    return 2 if value.get("blockers") else 0


if __name__ == "__main__":
    raise SystemExit(main())
