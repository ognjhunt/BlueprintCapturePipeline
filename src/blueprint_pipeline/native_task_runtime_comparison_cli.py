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
    policy = sub.add_parser("prepare-policy")
    policy.add_argument("--packet-dir", required=True)
    policy.add_argument("--runtime-source-receipt", required=True)
    policy.add_argument("--pi05-spec", required=True)
    policy.add_argument("--groot-spec", required=True)
    policy.add_argument("--readiness", required=True)
    policy.add_argument("--construction", required=True)
    policy.add_argument("--control", required=True)
    policy.add_argument("--implementation-commit", required=True)
    policy.add_argument("--variant", choices=("baseline", "isaaclab_3_ga"), required=True)
    policy.add_argument("--confirmed-by", required=True)
    policy.add_argument("--authorization-reference", required=True)
    policy.add_argument("--output", required=True, help="New preparation directory")
    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("runtime_comparison_output_exists")
    if args.command == "prepare-policy":
        from .native_task_runtime_policy_comparison_setup import prepare_retained_policy_comparison
        result = prepare_retained_policy_comparison(packet_dir=args.packet_dir,
            runtime_source_receipt=args.runtime_source_receipt,
            retained_spec_paths={"pi05_droid": args.pi05_spec, "groot_n17_droid": args.groot_spec},
            readiness_path=args.readiness, construction_path=args.construction, control_path=args.control,
            implementation_commit=args.implementation_commit, variant=args.variant,
            confirmed_by=args.confirmed_by, authorization_reference=args.authorization_reference, output_dir=output)
        print(json.dumps(result))
        return 0 if result["status"] == "passed" else 2
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
