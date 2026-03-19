from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.synthesis import cosmos_inference


def test_try_load_cosmos_official_repo_exposes_subprocess_env(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "cosmos-predict2.5"
    inference_entrypoint = repo_root / "examples" / "inference.py"
    python_bin = repo_root / ".venv" / "bin" / "python"
    inference_entrypoint.parent.mkdir(parents=True)
    python_bin.parent.mkdir(parents=True)
    inference_entrypoint.write_text("print('ok')\n", encoding="utf-8")
    python_bin.write_text("", encoding="utf-8")

    monkeypatch.setenv("COSMOS_OFFICIAL_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("COSMOS_DISABLE_GUARDRAILS", "true")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(cosmos_inference, "_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT", str(repo_root))
    monkeypatch.setattr(cosmos_inference, "_DEFAULT_COSMOS_DISABLE_GUARDRAILS", True)

    model = cosmos_inference._try_load_cosmos_official_repo("nvidia/Cosmos-Predict2.5-2B")

    assert model is not None
    assert model["backend"] == "official_repo_script"
    assert model["model_variant"] == "2B/post-trained"
    env = model["subprocess_env"]
    assert str((repo_root / ".venv" / "bin").resolve()) == env["PATH"].split(":")[0]
    assert str((Path.home() / ".local" / "bin").resolve()) in env["PATH"].split(":")
    assert "VIRTUAL_ENV" not in env
    assert "UV_PYTHON" not in env


def test_prepend_search_paths_deduplicates_entries() -> None:
    merged = cosmos_inference._prepend_search_paths(
        ["/custom/bin", "/usr/bin", "/custom/bin"],
        "/usr/bin:/bin:/usr/bin",
    )

    assert merged == "/custom/bin:/usr/bin:/bin"
