"""ADP-009: the ARTISAN component builder's root-dependency check (2026-10-13 production install)."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_task_evaluation_scene_configuration_artisan_component.py"


def _builder():
    spec = importlib.util.spec_from_file_location("artisan_component_builder_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_index_local_label_satisfies_its_public_pin():
    """The pinned CUDA index installs torch 2.7.1+cu128 for the root torch==2.7.1."""
    builder = _builder()
    installed = {"torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128", "absl-py": "2.4.0"}
    assert builder.missing_root_dependencies(
        ["torch==2.7.1", "torchvision==0.22.1", "absl-py==2.4.0"], installed) == []


def test_a_different_absent_or_unpinned_root_is_still_missing():
    builder = _builder()
    installed = {"numpy": "1.26.4", "torch": "2.7.2+cu128"}
    assert builder.missing_root_dependencies(
        ["numpy==2.0.0", "scipy==1.14.0", "torch==2.7.1", "fastapi>=0.1"], installed) == [
        "numpy==2.0.0", "scipy==1.14.0", "torch==2.7.1", "fastapi>=0.1"]


def test_root_names_compare_normalized():
    builder = _builder()
    assert builder.missing_root_dependencies(["Pillow==11.0.0", "typing_extensions==4.12.2"],
                                             {"pillow": "11.0.0", "typing-extensions": "4.12.2"}) == []
