"""Object-index backend preflight helpers (read by ``scripts/setup_environment.py``).

The capture app's object-index stage itself was removed on 2026-11-11.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline import object_index_stage as oi


def test_object_index_backend_preflight_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", "custom {INPUT_JSON}")
    assert oi._command_from_env("OBJECT_INDEX_YOLO_WORLD_COMMAND") == "custom {INPUT_JSON}"
    monkeypatch.delenv("OBJECT_INDEX_YOLO_WORLD_COMMAND")
    assert "object_index_yolo_world_runner.py" in oi._command_from_env("OBJECT_INDEX_YOLO_WORLD_COMMAND")
    assert "object_index_splat_analyzer_runner.py" in oi._command_from_env("OBJECT_INDEX_SPLAT_ANALYZER_COMMAND")
    assert oi._command_from_env("UNKNOWN_COMMAND") == ""
    assert oi._module_available("json") is True
    monkeypatch.setattr(oi.importlib.util, "find_spec", lambda _name: (_ for _ in ()).throw(RuntimeError("boom")))
    assert oi._module_available("json") is False
    monkeypatch.setenv("SAM3_WEIGHTS_PATH", str(tmp_path / "sam.pt"))
    assert oi._default_sam3_weights_path() == tmp_path / "sam.pt"

    monkeypatch.setattr(oi, "_module_available", lambda name: name == "torch")
    requirements = oi._backend_runtime_requirements("yolo_world")
    assert requirements["missing_modules"] == ["ultralytics"]
    sam_requirements = oi._backend_runtime_requirements("sam3")
    assert sam_requirements["support_level"] == "optional"
    assert str(tmp_path / "sam.pt") in sam_requirements["missing_paths"]
    splat_requirements = oi._backend_runtime_requirements("splat_analyzer")
    assert splat_requirements["support_level"] == "optional"
    assert splat_requirements["required_modules"] == []
    assert oi._backend_runtime_requirements("other")["required_modules"] == []

    monkeypatch.setattr(
        oi,
        "_backend_runtime_requirements",
        lambda backend_name: {
            "support_level": "required",
            "required_modules": [],
            "missing_modules": [],
            "required_paths": [],
            "missing_paths": [],
        },
    )
    assert oi._backend_preflight_status(backend_name="yolo_world", command_template="")["reason"] == "command_not_configured"
    assert oi._backend_preflight_status(backend_name="yolo_world", command_template='"unterminated')["status"] == "invalid"
    assert oi._backend_preflight_status(backend_name="yolo_world", command_template="   ")["reason"] == "empty_command"
    assert oi._backend_preflight_status(backend_name="yolo_world", command_template="")["configured"] is False
    missing = oi._backend_preflight_status(backend_name="yolo_world", command_template="/not/a/real/exe {INPUT_JSON}")
    assert missing["status"] == "missing"
    monkeypatch.setattr(
        oi,
        "_backend_runtime_requirements",
        lambda backend_name: {
            "support_level": "optional" if backend_name == "sam3" else "required",
            "required_modules": ["x"],
            "missing_modules": ["x"],
            "required_paths": [],
            "missing_paths": [],
        },
    )
    assert oi._backend_preflight_status(backend_name="sam3", command_template="python -m x")["status"] == "optional_unavailable"
    assert oi._backend_preflight_status(backend_name="yolo_world", command_template="python -m x")["status"] == "runtime_missing"
    monkeypatch.setattr(
        oi,
        "_backend_runtime_requirements",
        lambda backend_name: {
            "support_level": "required",
            "required_modules": [],
            "missing_modules": [],
            "required_paths": ["/missing/weights.pt"],
            "missing_paths": ["/missing/weights.pt"],
        },
    )
    assert oi._backend_preflight_status(backend_name="yolo_world", command_template="python -m x")["reason"] == "missing_paths:/missing/weights.pt"
    monkeypatch.setattr(
        oi,
        "_backend_runtime_requirements",
        lambda backend_name: {
            "support_level": "optional",
            "required_modules": [],
            "missing_modules": [],
            "required_paths": ["/missing/optional.pt"],
            "missing_paths": ["/missing/optional.pt"],
        },
    )
    assert oi._backend_preflight_status(backend_name="sam3", command_template="python -m x")["status"] == "optional_unavailable"
