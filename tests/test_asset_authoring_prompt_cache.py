"""Hermetic cache routing tests, not provider cache-hit or cost-saving claims."""
import inspect
from pathlib import Path

from pydantic import BaseModel
import pytest

from blueprint_pipeline import asset_authoring_prompt_cache as cache
from blueprint_pipeline.openai_prompt_cache import explicit_cache_input, explicit_cache_request_kwargs


class Candidate(BaseModel):
    source_code: str


class OtherCandidate(BaseModel):
    source_code: str
    explanation: str


@pytest.fixture
def prefix():
    # Useful existing authoring rules as a stable test fixture; no padded provider call.
    return (Path(__file__).parents[1] / "src/blueprint_pipeline/task_object_physical_property_review.py").read_text()


def test_policy_uses_astra_high_canonical_explicit_write_read_options(prefix):
    policy = cache.asset_cache_policy(family="cad", output_type=Candidate, stable_prefix=prefix)
    assert policy.status == "enabled"
    assert policy.model_family == "gpt-6-astra"
    assert policy.reasoning_effort == "high"
    assert policy.ttl == "30m"
    assert policy.expected_reuse_count == 2
    assert policy.economics.stable_prefix_tokens == len(prefix.encode("utf-8"))
    assert "not_measured_tokens" in cache.PREFIX_TOKEN_COUNT_BASIS
    assert explicit_cache_request_kwargs(policy) == {
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        "prompt_cache_key": policy.cache_key,
    }


def test_cache_identity_is_reused_across_runs_objects_rounds_without_caching_frames(prefix):
    first = cache.asset_cache_policy(family="source_analysis", output_type=Candidate, stable_prefix=prefix)
    second = cache.asset_cache_policy(family="source_analysis", output_type=Candidate, stable_prefix=prefix)
    assert first.cache_key == second.cache_key
    assert not {"run_id", "object_id", "round"}.intersection(inspect.signature(cache.asset_cache_policy).parameters)
    for run_id, object_id, round_id in [("run-one", "book", 0), ("run-two", "tray", 2)]:
        dynamic = [{"role": "user", "content": [
            {"type": "input_text", "text": f"{run_id}:{object_id}:{round_id}"},
            {"type": "input_image", "image_url": "data:image/png;base64,fixture"},
        ]}]
        rendered = explicit_cache_input(policy=first, stable_developer_prefix=prefix, dynamic_input=dynamic)
        assert rendered[0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
        assert rendered[1:] == dynamic
        assert "prompt_cache_breakpoint" not in str(rendered[1:])


def test_schema_privacy_prefix_and_stage_each_isolate_cache_identity(prefix):
    original = cache.asset_cache_policy(family="physics", output_type=Candidate, stable_prefix=prefix)
    alternatives = [
        cache.asset_cache_policy(family="physics", output_type=OtherCandidate, stable_prefix=prefix),
        cache.asset_cache_policy(family="physics", output_type=Candidate, stable_prefix=prefix, privacy_scope="partner_a"),
        cache.asset_cache_policy(family="physics", output_type=Candidate, stable_prefix=prefix + "\nNew material rule."),
        cache.asset_cache_policy(family="visual_review", output_type=Candidate, stable_prefix=prefix),
    ]
    assert len({original.cache_key, *(policy.cache_key for policy in alternatives)}) == 5
    families = [cache.asset_cache_policy(family=family, output_type=Candidate, stable_prefix=prefix)
                for family in ("cad", "source_analysis", "blender_author", "physics", "visual_review")]
    assert len({policy.cache_key for policy in families}) == 5


def test_model_and_effort_are_bound_into_canonical_identity(prefix, monkeypatch):
    original = cache.asset_cache_policy(family="cad", output_type=Candidate, stable_prefix=prefix)
    monkeypatch.setattr(cache, "ASSET_CACHE_MODEL", "gpt-5.6-terra")
    assert cache.asset_cache_policy(family="cad", output_type=Candidate, stable_prefix=prefix).cache_key != original.cache_key
    monkeypatch.setattr(cache, "ASSET_CACHE_MODEL", "gpt-6-astra")
    monkeypatch.setattr(cache, "ASSET_CACHE_REASONING_EFFORT", "medium")
    assert cache.asset_cache_policy(family="cad", output_type=Candidate, stable_prefix=prefix).cache_key != original.cache_key


def test_short_prefix_is_not_padded_and_invalid_scope_or_family_refused():
    policy = cache.asset_cache_policy(family="cad", output_type=Candidate, stable_prefix="Concise instruction")
    assert policy.status == "disabled"
    assert policy.economics.stable_prefix_tokens == len("Concise instruction")
    assert "prompt_cache_key" not in explicit_cache_request_kwargs(policy)
    with pytest.raises(ValueError, match="family_invalid"):
        cache.asset_cache_policy(family="run-specific-family", output_type=Candidate, stable_prefix="rules")
    with pytest.raises(ValueError, match="privacy_scope_missing"):
        cache.asset_cache_policy(family="cad", output_type=Candidate, stable_prefix="rules", privacy_scope="")
