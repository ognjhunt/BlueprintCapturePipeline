from __future__ import annotations

import ast
from pathlib import Path

from blueprint_pipeline import public_scene_artifixer3d_candidate_inputs as candidate_inputs
from blueprint_pipeline.public_scene_repair_preflight_contract import (
    CALIBRATED_RESIDUAL_PREFLIGHT_SCHEMA_VERSION,
)


def test_artifixer_candidate_inputs_import_only_neutral_preflight_contract() -> None:
    source_path = Path(candidate_inputs.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "public_scene_repair_preflight_contract" in imported_modules
    assert "public_scene_aura_exact_residual_preflight" not in imported_modules
    assert (
        candidate_inputs.CALIBRATED_RESIDUAL_PREFLIGHT_SCHEMA
        == CALIBRATED_RESIDUAL_PREFLIGHT_SCHEMA_VERSION
    )
