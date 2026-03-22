"""Helpers for locked autoresearch pytest checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from .common import load_target_manifest, read_json


@dataclass(frozen=True)
class EvalContext:
    manifest: dict[str, Any]
    eval_dir: Path
    candidate_dir: Path

    def case_entry(self, case_id: str) -> dict[str, Any]:
        for entry in self.manifest.get("eval_cases", []):
            if str(entry.get("case_id")) == case_id:
                return dict(entry)
        raise KeyError(f"Unknown eval case: {case_id}")

    def expectations(self, case_id: str) -> dict[str, Any]:
        case = self.case_entry(case_id)
        return read_json(Path(case["fixture_root"]) / "expectations.json")

    def output_path(self, case_id: str) -> Path:
        case = self.case_entry(case_id)
        return self.eval_dir / "cases" / case_id / str(case["expected_output_name"])

    def json_output(self, case_id: str) -> dict[str, Any]:
        return read_json(self.output_path(case_id))

    def text_output(self, case_id: str) -> str:
        return self.output_path(case_id).read_text(encoding="utf-8")


def load_pytest_eval_context() -> EvalContext:
    manifest_path = os.getenv("AUTORESEARCH_TARGET_MANIFEST")
    eval_dir = os.getenv("AUTORESEARCH_EVAL_DIR")
    candidate_dir = os.getenv("AUTORESEARCH_CANDIDATE_DIR")
    if not manifest_path or not eval_dir or not candidate_dir:
        pytest.skip(
            "Locked autoresearch harness tests require AUTORESEARCH_TARGET_MANIFEST, "
            "AUTORESEARCH_EVAL_DIR, and AUTORESEARCH_CANDIDATE_DIR."
        )
    return EvalContext(
        manifest=load_target_manifest(manifest_path),
        eval_dir=Path(eval_dir),
        candidate_dir=Path(candidate_dir),
    )


def require_target(target_skill: str) -> EvalContext:
    context = load_pytest_eval_context()
    if context.manifest["target_skill"] != target_skill:
        pytest.skip(f"Locked tests loaded for {context.manifest['target_skill']}, not {target_skill}")
    return context
