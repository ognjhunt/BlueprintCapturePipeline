"""The ARTISAN component applies an ordered list of upstream patches and seals each digest.

Owner direction 2026-11-02: ARTISAN authors with gpt-6.1-sol at maximum
reasoning. Upstream refuses unknown or unpriced models, so
``patches/artisan/0002-gpt-6-1-sol.patch`` prices and admits it; the meter
prices every call through the same upstream table.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.openai_prompt_cache import pricing_for_model
from blueprint_pipeline.task_object_authoring_backend import ARTISAN_MODELS
from blueprint_pipeline.task_object_authoring_contracts import file_record

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_task_evaluation_scene_configuration_artisan_component.py"
SOL_PATCH = (ROOT / "patches/artisan/0002-gpt-6-1-sol.patch").read_text()
OPUS_PATCH = (ROOT / "patches/artisan/0003-claude-opus-5-5.patch").read_text()


def _builder():
    spec = importlib.util.spec_from_file_location("artisan_component_builder_patches_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _added(patch: str, path: str) -> str:
    """The lines ``patch`` adds to ``path``."""
    section = patch.split(f"+++ b/{path}\n", 1)[1].split("\n--- a/", 1)[0]
    return "\n".join(line[1:] for line in section.splitlines() if line.startswith("+"))


def test_every_listed_patch_is_tracked_and_confinement_comes_first():
    builder = _builder()
    assert builder.UPSTREAM_PATCHES[0] == "0001-confine-generated-code.patch"
    assert builder.UPSTREAM_PATCHES[1] == "0002-gpt-6-1-sol.patch"
    assert builder.UPSTREAM_PATCHES[2] == "0003-claude-opus-5-5.patch"
    assert sorted(path.name for path in (ROOT / "patches/artisan").glob("*.patch")) == list(builder.UPSTREAM_PATCHES)


def test_patches_apply_in_order_and_each_digest_is_sealed(tmp_path, monkeypatch):
    """The second patch edits a line only the first one writes: order is load-bearing."""
    repository, source, staging = tmp_path / "repository", tmp_path / "source", tmp_path / "staging"
    (repository / "patches/artisan").mkdir(parents=True)
    source.mkdir()
    staging.mkdir()
    (source / "model.py").write_text("MODEL = 'a'\n")
    (repository / "patches/artisan/first.patch").write_text(
        "--- a/model.py\n+++ b/model.py\n@@ -1 +1 @@\n-MODEL = 'a'\n+MODEL = 'b'\n")
    (repository / "patches/artisan/second.patch").write_text(
        "--- a/model.py\n+++ b/model.py\n@@ -1 +1 @@\n-MODEL = 'b'\n+MODEL = 'c'\n")
    builder = _builder()
    monkeypatch.setattr(builder, "UPSTREAM_PATCHES", ("first.patch", "second.patch"))
    applied = builder.apply_upstream_patches(repository, source, staging)
    assert (source / "model.py").read_text() == "MODEL = 'c'\n"
    assert [row["name"] for row in applied] == ["first.patch", "second.patch"]
    for row in applied:
        record = file_record(repository / "patches/artisan" / row["name"])
        assert (row["sha256"], row["size_bytes"]) == (record["sha256"], record["size_bytes"])
        copied = staging / builder.PATCH_DIRECTORY / row["name"]
        assert file_record(copied)["sha256"] == row["sha256"]
    # Out of order, the second patch does not apply and the build stops.
    (source / "model.py").write_text("MODEL = 'a'\n")
    (tmp_path / "other-staging").mkdir()
    monkeypatch.setattr(builder, "UPSTREAM_PATCHES", ("second.patch", "first.patch"))
    with pytest.raises(subprocess.CalledProcessError):
        builder.apply_upstream_patches(repository, source, tmp_path / "other-staging")
    assert (source / "model.py").read_text() == "MODEL = 'a'\n"


def test_the_sol_patch_prices_gpt_6_1_sol_at_the_verified_rates():
    added = _added(SOL_PATCH, "pipeline/cost.py")
    table = ast.literal_eval(re.search(r"OPENAI_GPT_6_1_SOL_PRICING: dict\[str, float\] = (\{.*?\n\})",
                                       added, re.S).group(1))
    verified = pricing_for_model("gpt-6.1-sol")
    assert verified.model_family == "gpt-6.1-sol"
    threshold = verified.long_context_threshold_tokens
    assert table == {
        "input_uncached": verified.uncached_input_per_million_usd,
        "input_cached": verified.cached_read_per_million_usd,
        "input_cache_write": verified.cache_write_per_million_usd,
        "output": verified.output_per_million_usd,
        "prompt_tier_threshold_tokens": threshold,
        "input_uncached_above_threshold": verified.uncached_input_per_million_usd * verified.long_context_input_multiplier,
        "input_cached_above_threshold": verified.cached_read_per_million_usd * verified.long_context_input_multiplier,
        "input_cache_write_above_threshold": verified.cache_write_per_million_usd * verified.long_context_input_multiplier,
        "output_above_threshold": verified.output_per_million_usd * verified.long_context_output_multiplier,
    }
    assert '"gpt-6.1-sol": OPENAI_GPT_6_1_SOL_PRICING,' in added


def test_the_sol_patch_admits_every_configured_text_role_and_the_max_effort():
    registry = _added(SOL_PATCH, "pipeline/llm/model_registry.py")
    assert registry.strip() == '"gpt-6.1-sol": _TEXT_ROLES,'  # text, compaction and image envelope
    for role in ("model", "compaction_model", "image_model"):
        assert ARTISAN_MODELS[role] == "gpt-6.1-sol"
    # The image tool keeps its existing upstream price.
    assert ARTISAN_MODELS["image_tool_model"] == "gpt-image-2.5-sunburst"
    client = _added(SOL_PATCH, "pipeline/llm/openai.py")
    # Upstream already sends low/medium/high/xhigh/max verbatim for the models in this set.
    assert 'EXTENDED_REASONING_MODELS = frozenset({"gpt-6-astra", "gpt-6.1-sol"})' in client
    assert ARTISAN_MODELS["thinking"] == "max"
    # Context window and compaction pressure as for the other 1,050,000-token models.
    assert client.count('"gpt-6-astra", "gpt-6.1-sol"}:') == 2
    providers = _added(SOL_PATCH, "pipeline/providers.py")
    assert '"gpt-6.1-sol",' in providers and "requires the OpenAI gpt-6-astra or gpt-6.1-sol model" in providers


def _hunks(patch: str) -> dict[str, list[str]]:
    """Each file the patch touches, with its changed (``+``/``-``) lines."""
    files: dict[str, list[str]] = {}
    current = None
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current = line[len("+++ b/"):]
            files[current] = []
        elif line.startswith("--- a/"):
            continue
        elif current is not None and line[:1] in {"+", "-"}:
            files[current].append(line)
    return files


def test_the_opus_patch_touches_only_the_anthropic_path():
    hunks = _hunks(OPUS_PATCH)
    assert set(hunks) == {"pipeline/cost.py", "pipeline/llm/anthropic.py", "pipeline/providers.py"}
    # Nothing is removed from the OpenAI code it shares a file with: an OpenAI job is unchanged.
    assert [line for line in hunks["pipeline/cost.py"] if line.startswith("-")] == []
    assert [line for line in hunks["pipeline/providers.py"] if line.startswith("-")] == []


def test_the_opus_patch_prices_claude_opus_5_5_at_the_list_price():
    # The meter bounds and prices every Anthropic call from this row: a wrong price breaks the one cap.
    added = _added(OPUS_PATCH, "pipeline/cost.py")
    table = ast.literal_eval(re.search(r"ANTHROPIC_OPUS_5_5_PRICING: dict\[str, float\] = (\{.*?\n\})",
                                       added, re.S).group(1))
    # Anthropic list price for claude-opus-5-5, read 2026-11-13 at platform.claude.com/docs/en/about-claude/pricing.
    listed = {"input": 4.0, "cache_read": 0.2, "cache_write": 5.0, "cache_write_1h": 8.0, "output": 20.0}
    assert table == {"input_uncached": listed["input"], "input_cached": listed["cache_read"],
                     "input_cache_write": listed["cache_write"], "input_cache_write_1h": listed["cache_write_1h"],
                     "output": listed["output"]}
    assert '"claude-opus-5-5": ANTHROPIC_OPUS_5_5_PRICING,' in added
