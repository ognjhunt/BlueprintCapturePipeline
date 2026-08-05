from pathlib import Path


def test_aura_lama_runtime_isolates_legacy_opencv_distribution() -> None:
    entrypoint = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_adp_aura_interiorgs_provider_runtime.sh"
    ).read_text(encoding="utf-8")

    assert 'pip uninstall --python "${LAMA_PY}" opencv-python' in entrypoint
    assert '--reinstall --no-deps opencv-python-headless==4.5.5.64' in entrypoint
    assert 'metadata.version("opencv-python-headless") == "4.5.5.64"' in entrypoint
    assert 'numpy.__version__ == "1.21.6"' in entrypoint
    assert "aurafusion360_interiorgs_lama_opencv_validation_failed" in entrypoint
