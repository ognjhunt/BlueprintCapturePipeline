from __future__ import annotations



from blueprint_pipeline.measurement_adapter_execution import (
    _safe_environment,
)


D = ["sha256:" + char * 64 for char in "abcdef"]


def test_subprocess_environment_isolates_home_and_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/operator-home-must-not-leak")
    monkeypatch.setenv("XDG_CACHE_HOME", "/operator-cache-must-not-leak")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setattr(
        "blueprint_pipeline.measurement_adapter_execution.platform.system",
        lambda: "Linux",
    )

    environment = _safe_environment(tmp_path)

    assert environment["HOME"] == str(tmp_path)
    assert environment["XDG_CACHE_HOME"] == str(tmp_path / ".cache")
    assert environment["TMPDIR"] == str(tmp_path)
    assert environment["MUJOCO_GL"] == "disable"

    monkeypatch.setenv("MUJOCO_GL", "egl")
    assert _safe_environment(tmp_path)["MUJOCO_GL"] == "disable"


