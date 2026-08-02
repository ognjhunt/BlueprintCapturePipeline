from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/postshot_windows_worker/launch_postshot_worker.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("launch_postshot_worker", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_aws_worker_launch_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(
        module,
        "_aws_session",
        lambda: pytest.fail("disabled launcher must not open an AWS session"),
    )

    with pytest.raises(
        SystemExit,
        match="legacy_postshot_windows_worker_launch_disabled_use_paid_resource_allocator",
    ):
        module.launch(SimpleNamespace())
