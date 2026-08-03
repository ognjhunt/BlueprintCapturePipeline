"""Command-line boundary for the post-capture evidence spine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .post_capture_evidence_spine import (
    PostCaptureEvidenceError,
    run_post_capture_evidence_spine,
)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PostCaptureEvidenceError(["post_capture_input_file_invalid"]) from exc
    if not isinstance(value, dict):
        raise PostCaptureEvidenceError(["post_capture_input_file_invalid"])
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-artifact", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--appearance-candidate", type=Path)
    parser.add_argument("--canonical-registered-appearance", type=Path)
    parser.add_argument("--canonical-registration-measurement", type=Path)
    parser.add_argument("--teleport-run-receipt", type=Path)
    parser.add_argument("--teleport-import-receipt", type=Path)
    parser.add_argument("--depth-surface-result", type=Path)
    parser.add_argument("--depth-surface-root", type=Path)
    parser.add_argument("--geometry-qualification", type=Path)
    parser.add_argument("--registration-qualification", type=Path)
    parser.add_argument("--target-orchestration", type=Path)
    parser.add_argument("--target-pipeline-request", type=Path)
    parser.add_argument("--placement-candidate", type=Path)
    parser.add_argument("--placement-request", type=Path)
    parser.add_argument("--collision-glb", type=Path)
    parser.add_argument("--placement-qualification", type=Path)
    parser.add_argument("--simready-task-zone-qualification", type=Path)
    parser.add_argument("--routing-bundle", type=Path)
    parser.add_argument("--task-metric", type=Path)
    parser.add_argument("--policy-candidate", action="append", default=[], type=Path)
    parser.add_argument("--policy-attempt", action="append", default=[], type=Path)
    parser.add_argument(
        "--authorizer-identity", default="blueprint-post-capture-admission"
    )
    arguments = parser.parse_args(argv)
    def optional(path: Path | None) -> dict[str, Any] | None:
        return _load(path) if path else None
    result = run_post_capture_evidence_spine(
        run_id=arguments.run_id,
        source_artifact=_load(arguments.source_artifact),
        source_root=arguments.source_root,
        output_root=arguments.output_root,
        appearance_candidate=optional(arguments.appearance_candidate),
        canonical_registered_appearance=optional(
            arguments.canonical_registered_appearance
        ),
        canonical_registration_measurement=optional(
            arguments.canonical_registration_measurement
        ),
        teleport_run_receipt=optional(arguments.teleport_run_receipt),
        teleport_import_receipt=optional(arguments.teleport_import_receipt),
        depth_surface_result=optional(arguments.depth_surface_result),
        depth_surface_root=arguments.depth_surface_root,
        geometry_qualification=optional(arguments.geometry_qualification),
        registration_qualification=optional(arguments.registration_qualification),
        target_orchestration=optional(arguments.target_orchestration),
        target_pipeline_request=optional(arguments.target_pipeline_request),
        placement_candidate=optional(arguments.placement_candidate),
        placement_request=optional(arguments.placement_request),
        collision_glb_path=arguments.collision_glb,
        placement_qualification=optional(arguments.placement_qualification),
        simready_task_zone_qualification=optional(
            arguments.simready_task_zone_qualification
        ),
        routing_bundle=optional(arguments.routing_bundle),
        task_metric=optional(arguments.task_metric),
        policy_candidates=[_load(path) for path in arguments.policy_candidate],
        policy_attempts=[_load(path) for path in arguments.policy_attempt],
        authorizer_identity=arguments.authorizer_identity,
    )
    print(
        json.dumps(
            {
                "run_root": result["run_root"],
                "status": result["terminal"]["status"],
                "terminal_stage": result["terminal"]["terminal_stage"],
                "smallest_missing_measurement": result["terminal"][
                    "smallest_missing_measurement"
                ],
                "run_digest": result["manifest"][
                    "post_capture_evidence_run_digest"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if result["terminal"]["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
