from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from blueprint_pipeline.content_agents_model_compatibility import (
    apply_content_agents_model_compatibility,
    materialize_content_agents_model_compatibility_plan,
    openai_custom_temperature_supported,
)


def _released_source_fixture(root: Path) -> Path:
    vision = root / "world_understanding/functions/models/vision_language_models.py"
    backend = root / "world_understanding/functions/models/backends/public/openai.py"
    vision.parent.mkdir(parents=True)
    backend.parent.mkdir(parents=True)
    vision.write_text(
        "from world_understanding.telemetry import traced_vlm\n\n"
        "class OpenAIVLM(object):\n"
        "    def generate(self, temperature=None):\n"
        "        invoke_kwargs = {}\n"
        "        if temperature is not None:\n"
        "            invoke_kwargs[\"temperature\"] = temperature\n"
        "    async def agenerate(self, temperature=None):\n"
        "        invoke_kwargs = {}\n"
        "        if temperature is not None:\n"
        "            invoke_kwargs[\"temperature\"] = temperature\n"
        "    def generate_with_image_caption_pairs(self, temperature=None):\n"
        "        invoke_kwargs = {}\n"
        "        if temperature is not None:\n"
        "            invoke_kwargs[\"temperature\"] = temperature\n\n"
        "class AnthropicVLM(object):\n"
        "    pass\n",
        encoding="utf-8",
    )
    backend.write_text(
        "from world_understanding.functions.models.backends.registry import (\n"
        "    register_chat_backend,\n"
        ")\n\n"
        "_DEFAULT_OPENAI_MODEL = \"gpt-5.4\"\n\n"
        "def create_openai_chat(model=None, temperature=None):\n"
        "    chat_kwargs = {}\n"
        "    if temperature is not None:\n"
        "        chat_kwargs[\"temperature\"] = temperature\n"
        "    return chat_kwargs\n",
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize(
    ("model_id", "supported"),
    [
        ("gpt-4.1", True),
        ("gpt-5.6-luna", False),
        ("gpt-5.6-luna-2026-08-01", False),
        ("future-model", True),
    ],
)
def test_openai_temperature_capability_is_model_bound(
    model_id: str, supported: bool
) -> None:
    assert openai_custom_temperature_supported(model_id) is supported


def test_luna_overlay_normalizes_every_released_openai_temperature_seam(
    tmp_path: Path,
) -> None:
    source = _released_source_fixture(tmp_path / "source")
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    materialize_content_agents_model_compatibility_plan(
        model_ids=("gpt-5.6-luna", "gpt-image-2"), destination=plan_path
    )

    receipt = apply_content_agents_model_compatibility(
        source_root=source,
        plan_path=plan_path,
        receipt_path=receipt_path,
    )

    assert receipt["status"] == "applied"
    assert receipt["released_source_archive_remains_unmodified"] is True
    assert sum(
        row["temperature_call_sites_normalized"]
        for row in receipt["modified_files"]
    ) == 4
    vision = (
        source / "world_understanding/functions/models/vision_language_models.py"
    ).read_text(encoding="utf-8")
    backend = (
        source / "world_understanding/functions/models/backends/public/openai.py"
    ).read_text(encoding="utf-8")
    assert vision.count("openai_custom_temperature_supported(self._model_name)") == 3
    assert "openai_custom_temperature_supported(model or _DEFAULT_OPENAI_MODEL)" in backend
    assert json.loads(receipt_path.read_text())["status"] == "applied"


def test_legacy_model_preserves_exact_released_source_bytes(tmp_path: Path) -> None:
    source = _released_source_fixture(tmp_path / "source")
    before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*.py")
    }
    plan_path = tmp_path / "plan.json"
    materialize_content_agents_model_compatibility_plan(
        model_ids=("gpt-4.1",), destination=plan_path
    )

    receipt = apply_content_agents_model_compatibility(
        source_root=source,
        plan_path=plan_path,
        receipt_path=tmp_path / "receipt.json",
    )

    assert receipt["status"] == "not_required"
    assert receipt["modified_files"] == []
    assert {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*.py")
    } == before


def test_provider_copy_script_is_standalone_and_fails_closed_on_changed_seam(
    tmp_path: Path,
) -> None:
    source = _released_source_fixture(tmp_path / "source")
    vision = source / "world_understanding/functions/models/vision_language_models.py"
    vision.write_text(
        vision.read_text().replace(
            "        if temperature is not None:\n",
            "        if temperature is not None and temperature >= 0:\n",
            1,
        ),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"
    materialize_content_agents_model_compatibility_plan(
        model_ids=("gpt-5.6-luna",), destination=plan_path
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "src/blueprint_pipeline/content_agents_model_compatibility.py"
    )
    receipt = tmp_path / "receipt.json"

    completed = subprocess.run(
        [
            "python3",
            str(script),
            "--source-root",
            str(source),
            "--plan",
            str(plan_path),
            "--receipt",
            str(receipt),
        ],
        check=False,
    )

    assert completed.returncode == 2
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "blocked"
    assert payload["blockers"] == [
        "content_agents_openai_vlm_temperature_seam_changed"
    ]
