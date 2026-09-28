#!/usr/bin/env python3
"""Seal retained ArtiFixer and paired-native spend/teardown evidence.

ADP-009D needs these terminal receipts before adopting completed work. Each
command calls the existing deterministic validator with retained local files;
it does not allocate, query a provider, or perform teardown.
"""
from __future__ import annotations

from collections.abc import Sequence

from blueprint_pipeline.materializer_cli import Param, Step, run
from blueprint_pipeline.paired_target_native_import_vast import (
    materialize_paired_target_native_import_preallocation_provider_zero,
    materialize_paired_target_native_import_provider_zero,
    materialize_paired_target_native_import_supplemental_spend_reconciliation,
)
from blueprint_pipeline.public_scene_artifixer3d_vast import (
    materialize_artifixer3d_postblocked_provider_zero,
    materialize_artifixer3d_supplemental_spend_reconciliation,
)

OUTPUT = Param("--output", required=True)
AUTHORITY = Param("--attempt-authority", required=True)
RESULT = Param("--result", required=True)
STEPS = {
    "artifixer-spend": Step(
        "Bind prior excision and retained-render spend from closed attempts.",
        materialize_artifixer3d_supplemental_spend_reconciliation,
        {
            "gaussian_excision_closeouts": Param("--gaussian-excision-closeouts", required=True, json_file=True),
            "retained_scene_render_attempts": Param("--retained-scene-render-attempts", required=True, json_file=True),
            "output_path": OUTPUT,
        },
    ),
    "artifixer-provider-zero": Step(
        "Bind completed or blocked ArtiFixer teardown and recorded provider zero.",
        materialize_artifixer3d_postblocked_provider_zero,
        {
            "attempt_authority_path": AUTHORITY,
            "result_path": RESULT,
            "adapter_result_path": Param("--adapter-result", required=True),
            "cleanup_path": Param("--cleanup", required=True),
            "watchdog_path": Param("--watchdog", required=True),
            "output_path": OUTPUT,
        },
    ),
    "native-spend": Step(
        "Bind terminal CAD attempts to the paired-native spend chain.",
        materialize_paired_target_native_import_supplemental_spend_reconciliation,
        {
            "content_agents_attempts": Param("--content-agents-attempts", required=True, json_file=True),
            "output_path": OUTPUT,
        },
    ),
    "native-preallocation-zero": Step(
        "Close a consumed native authority that stopped before allocation.",
        materialize_paired_target_native_import_preallocation_provider_zero,
        {
            "attempt_authority_path": AUTHORITY,
            "result_path": RESULT,
            "watchdog_handoff_path": Param("--watchdog-handoff", required=True),
            "cleanup_path": Param("--cleanup", required=True),
            "api_provider_zero_path": Param("--api-provider-zero", required=True),
            "output_path": OUTPUT,
        },
    ),
    "native-provider-zero": Step(
        "Bind the native terminal result and its retained watchdog/cleanup records.",
        materialize_paired_target_native_import_provider_zero,
        {"attempt_authority_path": AUTHORITY, "result_path": RESULT, "output_path": OUTPUT},
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    return run(STEPS, argv, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
