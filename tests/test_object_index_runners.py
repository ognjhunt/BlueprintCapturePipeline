from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_runner(module_name: str, file_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPTS_DIR / file_name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dino() -> ModuleType:
    return _load_runner("object_index_grounding_dino_runner", "object_index_grounding_dino_runner.py")


@pytest.fixture(scope="module")
def sam3() -> ModuleType:
    return _load_runner("object_index_sam3_runner", "object_index_sam3_runner.py")


# --- Grounding-DINO runner -------------------------------------------------


def test_dino_prompts_honors_prompt_bank_precedence(dino: ModuleType) -> None:
    # task_specific wins over all when both present.
    assert dino._prompts(
        {"prompt_bank": {"task_specific": ["right handle"], "all": ["sink", "faucet"]}}
    ) == ["right handle"]
    # falls back to all when task_specific is empty/blank.
    assert dino._prompts(
        {"prompt_bank": {"task_specific": ["", "   "], "all": ["sink", "faucet"]}}
    ) == ["sink", "faucet"]
    assert dino._prompts({"prompt_bank": {"all": ["sink"]}}) == ["sink"]
    # whitespace is stripped, blanks dropped.
    assert dino._prompts({"prompt_bank": {"task_specific": ["  knob ", ""]}}) == ["knob"]
    # missing/garbage prompt bank yields empty list.
    assert dino._prompts({}) == []
    assert dino._prompts({"prompt_bank": "bad"}) == []
    assert dino._prompts({"prompt_bank": {"task_specific": "bad", "all": "bad"}}) == []


def test_dino_keyframes_filters_non_mapping_entries(dino: ModuleType) -> None:
    keyframes = dino._keyframes(
        {"keyframes": [{"frame_index": 0, "image_path": "a.png"}, "bad", 7, None, {"frame_index": 1}]}
    )
    assert keyframes == [{"frame_index": 0, "image_path": "a.png"}, {"frame_index": 1}]
    assert dino._keyframes({}) == []
    assert dino._keyframes({"keyframes": "bad"}) == []


def test_dino_run_grounding_skips_on_empty_input(dino: ModuleType) -> None:
    result = dino._run_grounding({})
    assert result["backend_status"] == "skipped"
    assert result["reason"] == "missing_prompts_or_keyframes"
    assert result["detections"] == []
    # prompts present but no keyframes -> still skipped for same reason.
    assert (
        dino._run_grounding({"prompt_bank": {"all": ["sink"]}})["reason"]
        == "missing_prompts_or_keyframes"
    )


def test_dino_run_grounding_reports_ultralytics_missing(
    dino: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the ultralytics import inside _run_with_ultralytics to fail.
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    result = dino._run_grounding(
        {
            "prompt_bank": {"all": ["sink", "faucet"]},
            "keyframes": [{"frame_index": 0, "image_path": "a.png"}],
        }
    )
    assert result["backend_status"] == "skipped"
    assert result["reason"].startswith("ultralytics_missing:")
    assert result["detections"] == []


def test_dino_main_arg_validation(dino: ModuleType, tmp_path: Path) -> None:
    assert dino.main([]) == 2
    assert dino.main(["only-one"]) == 2
    assert dino.main(["a", "b", "c"]) == 2
    # Two args with empty-input payload -> writes a skipped report and returns 0.
    input_path = tmp_path / "in.json"
    output_path = tmp_path / "out.json"
    input_path.write_text("{}", encoding="utf-8")
    assert dino.main([str(input_path), str(output_path)]) == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["backend_status"] == "skipped"
    assert written["reason"] == "missing_prompts_or_keyframes"


# --- SAM3 runner -----------------------------------------------------------


def test_sam3_environment_normalization(sam3: ModuleType) -> None:
    assert sam3._environment({"environment": "kitchen"}) == "kitchen"
    assert sam3._environment({"environment": "WAREHOUSE"}) == "warehouse"
    assert sam3._environment({"environment": "  Office  "}) == "office"
    # unknown / missing / non-string normalize to "auto".
    assert sam3._environment({"environment": "spaceship"}) == "auto"
    assert sam3._environment({}) == "auto"
    assert sam3._environment({"environment": ""}) == "auto"
    assert sam3._environment({"environment": 7}) == "auto"


def test_sam3_run_skips_when_video_missing(sam3: ModuleType, tmp_path: Path) -> None:
    result = sam3._run_sam3({"video_path": str(tmp_path / "absent.mov")})
    assert result["backend_status"] == "skipped"
    assert result["reason"] == "video_not_found"
    assert result["objects"] == []
    # blank / missing video_path is also treated as video_not_found.
    assert sam3._run_sam3({})["reason"] == "video_not_found"
    assert sam3._run_sam3({"video_path": "   "})["reason"] == "video_not_found"


def test_sam3_main_arg_validation(sam3: ModuleType, tmp_path: Path) -> None:
    assert sam3.main([]) == 2
    assert sam3.main(["only-one"]) == 2
    assert sam3.main(["a", "b", "c"]) == 2
    input_path = tmp_path / "in.json"
    output_path = tmp_path / "out.json"
    input_path.write_text("{}", encoding="utf-8")
    assert sam3.main([str(input_path), str(output_path)]) == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["backend_status"] == "skipped"
    assert written["reason"] == "video_not_found"
