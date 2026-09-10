#!/usr/bin/env python3
"""Materialize retained completed-scene and Astra recovery inputs.

Every command calls the existing validator/materializer on local retained
inputs. No command allocates a provider, sends a model request, submits a run,
or changes production queue state. Use explicit scratch output paths for
rehearsals; native adoption remains development-only candidate evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

from blueprint_pipeline.materializer_cli import Param, Step, run
from blueprint_pipeline.task_evaluation_completed_placement_adoption import (
    materialize_legacy_checkpoint_alias,
)
from blueprint_pipeline.task_evaluation_completed_scene_attempt_factory import (
    materialize_completed_scene_attempt,
)
from blueprint_pipeline.task_evaluation_completed_scene_inputs import materialize_completed_mesh_inputs
from blueprint_pipeline.task_evaluation_completed_scene_submission import (
    materialize_completed_scene_submission,
)
from blueprint_pipeline.task_evaluation_scene_configuration_astra_phase_adoption import (
    materialize_phase_adoption,
)
from blueprint_pipeline.task_evaluation_scene_configuration_astra_runtime import (
    AstraRuntimePackageError, materialize_packaged_blender_runtime,
)
from blueprint_pipeline.task_evaluation_scene_configuration_repair_support import (
    materialize_repair_support,
)
from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError
from blueprint_pipeline.task_object_astra_native_adoption import materialize_astra_native_adoption


def _retain_phase_adoption(*, prior_runtime: Path, phases: Sequence[str] | None,
                           blender_round: int, output_path: Path) -> dict:
    receipt = materialize_phase_adoption(
        prior_runtime=prior_runtime, phases=list(phases or ()), blender_round=blender_round,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A different adoption descriptor must never replace a retained one.
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _path(flag: str) -> Param:
    return Param(flag, required=True, type=Path)


def _json(flag: str) -> Param:
    return Param(flag, required=True, json_file=True)


STEPS: dict[str, Step] = {
    "legacy-placement-alias": Step(
        "Retain the byte-identical legacy alias for an existing placement checkpoint.",
        materialize_legacy_checkpoint_alias,
        {"intent_path": _path("--intent"), "binding_root": _path("--binding-root")},
    ),
    "astra-phase-adoption": Step(
        "Seal selected completed Astra phases into a new retained descriptor.",
        _retain_phase_adoption,
        {"prior_runtime": _path("--prior-runtime"),
         "phases": Param("--phase", "Repeat in completed-prefix order.", accumulate=True),
         "blender_round": Param("--blender-round", type=int, default=0),
         "output_path": _path("--output")},
    ),
    "packaged-blender-runtime": Step(
        "Validate and unpack the already-sealed local Blender runtime archive.",
        materialize_packaged_blender_runtime,
        {"package_root": _path("--package-root"), "destination_root": _path("--destination-root")},
    ),
    "astra-native-adoption": Step(
        "Bind retained book/tray CAD candidates and independent native observations.",
        materialize_astra_native_adoption,
        {"source_packet_root": _path("--source-packet-root"),
         "book_packaging_path": _path("--book-packaging"), "book_static_path": _path("--book-static"),
         "tray_packaging_path": _path("--tray-packaging"), "tray_static_path": _path("--tray-static"),
         "tray_simready_path": _path("--tray-simready"), "authority": _json("--authority"),
         "support_measurement_path": _path("--support-measurement"),
         "settle_reference_path": _path("--settle-reference"), "output_root": _path("--output-root"),
         "qualification_limits": _json("--qualification-limits")},
    ),
    "completed-scene-attempt": Step(
        "Materialize an owner-bound completed-scene attempt using retained release inputs.",
        materialize_completed_scene_attempt,
        {"intent_path": _path("--intent"), "source_binding_path": _path("--source-binding"),
         "machinery_path": _path("--machinery"), "release_binding_path": _path("--release-binding"),
         "output_root": _path("--output-root"), "attempt_id": Param("--attempt-id", required=True)},
    ),
    "completed-mesh-inputs": Step(
        "Bind normalized completed-mesh inputs to the saved stage configuration.",
        materialize_completed_mesh_inputs,
        {"envelope": _json("--envelope"), "stage_one_configuration": _json("--stage-one-configuration"),
         "output_root": _path("--output-root")},
    ),
    "completed-scene-submission": Step(
        "Prepare a completed-scene submission locally without submitting it.",
        materialize_completed_scene_submission,
        {"binding": _json("--binding"), "task": _json("--task"),
         "task_request_path": _path("--task-request"), "deploy_receipt_path": _path("--deploy-receipt"),
         "release_provenance_path": _path("--release-provenance"),
         "release_environment_path": _path("--release-environment"),
         "runtime_publication_root": _path("--runtime-publication-root"),
         "expected_production_commit": Param("--expected-production-commit", required=True),
         "namespace_timestamp": Param("--namespace-timestamp", required=True),
         "release_admission_mode": Param("--release-admission-mode", required=True),
         "staging_root": _path("--staging-root"),
         "scene_intent_digest": Param("--scene-intent-digest", required=True)},
    ),
    "repair-support": Step(
        "Retain source-frame-bound calibrated repair masks without calling a model.",
        materialize_repair_support,
        {"calibrated_mask_path": _path("--calibrated-mask"), "sam_mask_path": _path("--sam-mask"),
         "source_frame_path": _path("--source-frame"),
         "calibration_digest": Param("--calibration-digest", required=True),
         "output_root": _path("--output-root")},
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(STEPS, argv, description=__doc__)
    except (AssetAuthoringError, AstraRuntimePackageError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)],
                          "provider_mutation_performed": False}, sort_keys=True))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
