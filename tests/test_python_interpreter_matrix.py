from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_python_interpreter_matrix_validator_passes_on_current_ci_python() -> None:
    root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_python_interpreter_matrix.py",
            "--assert-current",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "[python-matrix] ok" in completed.stdout


def test_python_interpreter_matrix_marks_python_313_noncanonical() -> None:
    root = Path(__file__).resolve().parents[1]
    matrix = json.loads(
        (root / "docs" / "CI_PYTHON_INTERPRETER_MATRIX.json").read_text(encoding="utf-8")
    )

    assert matrix["canonical_launch_evidence_python"] == "3.12"
    assert matrix["package_requires_python"] == ">=3.10,<3.13"
    assert "3.13" not in matrix["package_supported_python"]
    assert matrix["non_canonical_launch_evidence_python"][0]["python_version"] == "3.13"
