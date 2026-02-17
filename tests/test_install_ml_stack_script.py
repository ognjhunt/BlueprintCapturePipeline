"""Regression guards for ML stack installer script."""

from pathlib import Path


def _script_text() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "install_ml_stack.sh"
    return script_path.read_text(encoding="utf-8")


def test_install_script_uses_correct_da3_repo_url() -> None:
    text = _script_text()
    assert "https://github.com/ByteDance-Seed/Depth-Anything-3.git" in text
    assert "https://github.com/DepthAnything/Depth-Anything-V3.git" not in text


def test_install_script_keeps_required_ml_stack_steps() -> None:
    text = _script_text()
    required_snippets = [
        "/app/scripts/install_colmap_cuda.sh",
        "https://github.com/nv-tlabs/3DGRUT.git",
        "https://github.com/facebookresearch/sam3.git",
        "depth-anything/DA3Metric-Large",
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
    ]
    for snippet in required_snippets:
        assert snippet in text

