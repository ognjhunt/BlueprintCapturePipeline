#!/usr/bin/env python3
"""Prepare retained appearance and website inputs without starting a worker.

ADP-009D / Day 28: replay the local artifact preparation seams independently
of the provider drivers. Candidate selection retains the original request,
result, billing lineage and image bytes. Collision partitioning and object
observations remain development-only; website input binding retains its
existing disclosure and six-stage validation.

Run with PYTHONPATH=src, for example:
  python scripts/prepare_retained_scene_inputs.py retained-selection \
    --source-request request.json --source-result result.json \
    --source-output-root retained-output --camera-id camera_0 \
    --output-root selected-candidates

The multi-source selector takes a JSON list of objects with request_path,
result_path, output_root and camera_ids. All paths refer to local retained
artifacts. Each command writes only its selected output root; it does not call
a provider, allocate capacity, advance a queue or grant execution authority.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from blueprint_pipeline.materializer_cli import Param, Step, run
from blueprint_pipeline.sage_collision_partition import materialize_sage_collision_partition
from blueprint_pipeline.semantic_teacher_candidate_reuse import (
    materialize_retained_selection,
    materialize_retained_selection_from_sources,
)
from blueprint_pipeline.task_evaluation_scene_configuration_adapters import (
    TaskEvaluationSceneConfigurationAdapterError,
)
from blueprint_pipeline.website_native_inputs import materialize_website_inputs
from blueprint_pipeline.website_object_observations import materialize_object_observations


def _selection_summary(path: Path) -> dict:
    """Report the original selection seal and path without rewriting its bytes."""
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": value["schema_version"],
        "status": value["status"],
        "receipt_path": str(path.resolve()),
        "receipt_digest": value["selection_digest"],
    }


def _retain_selection(
    *, source_request_path, source_result_path, source_output_root, camera_ids, output_root
) -> dict:
    return _selection_summary(materialize_retained_selection(
        source_request_path=source_request_path,
        source_result_path=source_result_path,
        source_output_root=source_output_root,
        camera_ids=camera_ids,
        output_root=output_root,
    ))


def _retain_sources(*, sources, output_root) -> dict:
    return _selection_summary(materialize_retained_selection_from_sources(
        sources=sources, output_root=output_root,
    ))


STEPS: dict[str, Step] = {
    "retained-selection": Step(
        "Select exact raw candidates from one retained image-edit run.",
        _retain_selection,
        {
            "source_request_path": Param("--source-request", required=True, type=Path),
            "source_result_path": Param("--source-result", required=True, type=Path),
            "source_output_root": Param("--source-output-root", required=True, type=Path),
            "camera_ids": Param("--camera-id", "Repeatable camera selector.", accumulate=True, default=()),
            "output_root": Param("--output-root", required=True, type=Path),
        },
    ),
    "retained-selection-from-sources": Step(
        "Select exact raw candidates from several retained runs.",
        _retain_sources,
        {
            "sources": Param("--sources", "JSON list of retained runs and camera selectors.",
                             required=True, json_file=True),
            "output_root": Param("--output-root", required=True, type=Path),
        },
    ),
    "sage-collision-partition": Step(
        "Partition two labeled objects while retaining every source face.",
        materialize_sage_collision_partition,
        {
            "source_path": Param("--source", required=True, type=Path),
            "labels_path": Param("--labels", required=True, type=Path),
            "instance_ids": Param("--instance-id", "Repeat exactly twice.", accumulate=True, default=()),
            "output_root": Param("--output-root", required=True, type=Path),
        },
    ),
    "object-observations": Step(
        "Retain bound capture observations for the native authoring stage.",
        materialize_object_observations,
        {
            "preparation": Param("--preparation", required=True, json_file=True),
            "source_geometry": Param("--source-geometry", required=True, json_file=True),
            "task_masks": Param("--task-masks", required=True, json_file=True),
            "output_root": Param("--output-root", required=True, type=Path),
        },
    ),
    "website-inputs": Step(
        "Bind retained website derivatives to the original six-stage envelope.",
        materialize_website_inputs,
        {
            "envelope": Param("--envelope", required=True, json_file=True),
            "stage_one_configuration": Param("--stage-one-configuration", required=True, json_file=True),
            "output_root": Param("--output-root", required=True, type=Path),
        },
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(STEPS, argv, description=__doc__)
    except TaskEvaluationSceneConfigurationAdapterError as exc:
        # Retained-reference failures use the adapters' RuntimeError subtype.
        # Preserve the refusing predicate and the same blocked exit contract.
        print(json.dumps({"status": "blocked", "blockers": [f"{type(exc).__name__}:{exc}"],
                          "provider_mutation_performed": False}, indent=1, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
