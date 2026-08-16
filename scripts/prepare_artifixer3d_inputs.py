#!/usr/bin/env python3
"""Materialize the ArtiFixer3D input chain from a command line.

The appearance path's head needs a candidate-inputs receipt before anything
downstream exists, and the four functions that produce one could be called from
no script and from no module carrying a `main()`. So the chain dead-ended at its
first step: the bundle CLI asks for a candidate receipt, and nothing in a
production path could make one.

That is the same defect as #512 (lanes), #520 (bundle modules) and #523
(authority materializers), in a fourth scope. It is also not rare -- 77 of the
161 public `materialize_*` functions in `src/blueprint_pipeline` are reachable
from nothing, which is why `tests/test_materializer_reachability.py` now
rediscovers that set rather than trusting a list.

The order the steps run in:

    calibrated residual preflight
        -> object-absent reference receipt   (one per task, from edited frames)
        -> candidate inputs                  (the receipt the bundle consumes)
        -> whole-frame semantic teacher      (one per task, from teacher frames)
        -> dual-target inputs                (the paired receipt)

`editor-identity` is a JSON file rather than flags because it records *which
model produced the frames*, and that provenance belongs in one signed object
rather than being reassembled from arguments at each call site.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

from blueprint_pipeline.materializer_cli import Param, Step, run
from blueprint_pipeline.public_scene_artifixer3d_bundle import (
    materialize_artifixer3d_use_attestation,
)
from blueprint_pipeline.public_scene_artifixer3d_candidate_inputs import (
    materialize_artifixer3d_candidate_inputs,
    materialize_object_absent_reference_candidate_receipt,
)
from blueprint_pipeline.public_scene_artifixer3d_dual_target_inputs import (
    materialize_dual_target_artifixer3d_inputs,
    materialize_semantic_teacher_artifixer_handoff,
    materialize_whole_frame_semantic_teacher_receipt,
)


EDITOR_IDENTITY = Param(
    "--editor-identity",
    "JSON file naming the model that produced these frames.",
    required=True,
    json_file=True,
)

STEPS: dict[str, Step] = {
    "object-absent-reference": Step(
        "One task's object-absent reference receipt, from edited frames.",
        materialize_object_absent_reference_candidate_receipt,
        {
            "source_candidate_inputs_receipt_path": Param("--source-candidate-inputs", required=True),
            "task_id": Param("--task-id", required=True),
            "object_absent_frames_root": Param("--object-absent-frames-root", required=True),
            "editor_identity": EDITOR_IDENTITY,
            "prompt_policy": Param("--prompt-policy", required=True),
            "output_path": Param("--output", required=True),
        },
    ),
    "candidate-inputs": Step(
        "The candidate-inputs receipt the ArtiFixer3D bundle consumes.",
        materialize_artifixer3d_candidate_inputs,
        {
            "calibrated_residual_preflight_path": Param(
                "--calibrated-residual-preflight", required=True
            ),
            "output_root": Param("--output-root", required=True),
            "selected_task_ids": Param(
                "--task-id", "Repeatable; omit for every task.", accumulate=True
            ),
            "object_absent_reference_receipt_paths": Param(
                "--object-absent-reference", "Repeatable.", accumulate=True, default=()
            ),
        },
    ),
    "semantic-teacher": Step(
        "One task's whole-frame semantic teacher receipt.",
        materialize_whole_frame_semantic_teacher_receipt,
        {
            "source_candidate_inputs_receipt_path": Param("--source-candidate-inputs", required=True),
            "task_id": Param("--task-id", required=True),
            "semantic_teacher_frames_root": Param("--semantic-teacher-frames-root", required=True),
            "editor_identity": EDITOR_IDENTITY,
            "prompt_policy": Param("--prompt-policy", required=True),
            "output_path": Param("--output", required=True),
        },
    ),
    "dual-target": Step(
        "The paired receipt, from a candidate receipt plus teacher receipts.",
        materialize_dual_target_artifixer3d_inputs,
        {
            "source_candidate_inputs_receipt_path": Param("--source-candidate-inputs", required=True),
            "semantic_teacher_receipt_paths": Param(
                "--semantic-teacher-receipt", "Repeatable; one per task.", accumulate=True
            ),
            "output_root": Param("--output-root", required=True),
            "transition_radius_pixels": Param(
                "--transition-radius-pixels",
                "Width of the repair support band, in pixels.",
                required=True,
                type=int,
            ),
            "selected_task_ids": Param(
                "--task-id", "Repeatable; omit for every task.", accumulate=True
            ),
        },
    ),
    "from-semantic-result": Step(
        "Seal a paid semantic result into per-task receipts and one paired packet.",
        materialize_semantic_teacher_artifixer_handoff,
        {
            "result_import_path": Param("--result-import", required=True),
            "semantic_teacher_packet_path": Param(
                "--semantic-teacher-packet", required=True
            ),
            "source_candidate_inputs_receipt_path": Param(
                "--source-candidate-inputs", required=True
            ),
            "transition_radius_pixels": Param(
                "--transition-radius-pixels", required=True, type=int
            ),
            "output_root": Param("--output-root", required=True),
        },
    ),
    "use-attestation": Step(
        "Bind explicit retained use authority to one candidate receipt.",
        materialize_artifixer3d_use_attestation,
        {
            "candidate_inputs_receipt_path": Param(
                "--candidate-inputs-receipt", required=True
            ),
            "output_path": Param("--output", required=True),
            "authorized_by": Param("--authorized-by", required=True),
            "authorization_kind": Param(
                "--authorization-kind",
                default="explicit_user_direction_in_current_goal",
            ),
        },
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    return run(STEPS, argv, description=__doc__)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
