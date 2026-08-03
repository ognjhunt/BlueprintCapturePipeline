from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.post_capture_evidence_cli import _load, main
from blueprint_pipeline.post_capture_evidence_spine import PostCaptureEvidenceError


def test_cli_loader_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(PostCaptureEvidenceError, match="post_capture_input_file_invalid"):
        _load(path)


def test_cli_help_is_available_without_loading_inputs() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
