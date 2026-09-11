"""Stable Astra authoring cache policies using the canonical explicit-cache seam.

Set the returned policy and the same stable_developer_prefix on the agent spec.
Useful pinned skill/instruction content belongs before the sole breakpoint; all
source frames, object state, IDs, histories and feedback belong in dynamic input.
UTF-8 byte counts are conservative reservation bounds, not measured tokens or
proof of the provider's minimum cacheable prefix. Never pad prefixes to meet it;
only provider write/read usage receipts prove reuse or savings.
"""
from typing import Literal

from pydantic import BaseModel

from .openai_prompt_cache import PromptCachePolicy, create_prompt_cache_policy

AssetCacheFamily = Literal["cad", "source_analysis", "blender_author", "physics", "visual_review"]
ASSET_CACHE_MODEL = "gpt-6-astra"
ASSET_CACHE_REASONING_EFFORT = "high"
ASSET_CACHE_CONTRACT_VERSION = "asset_authoring_prompt_cache.v1"
PREFIX_TOKEN_COUNT_BASIS = "utf8_byte_upper_bound_not_measured_tokens"


def asset_cache_policy(
    *, family: AssetCacheFamily, output_type: type[BaseModel], stable_prefix: str,
    privacy_scope: str = "blueprint_internal",
    reasoning_effort: Literal['medium', 'high'] | None = None,
) -> PromptCachePolicy:
    """Use one write plus two expected reads within the canonical 30-minute TTL.

    Cache identity is independent of runs, objects and rounds because none are
    inputs. The caller remains responsible for keeping them out of stable_prefix.
    """
    if family not in {"cad", "source_analysis", "blender_author", "physics", "visual_review"}:
        raise ValueError("asset_authoring_cache_family_invalid")
    if not privacy_scope.strip():
        raise ValueError("asset_authoring_cache_privacy_scope_missing")
    return create_prompt_cache_policy(
        model=ASSET_CACHE_MODEL, family=f"asset_{family}",
        contract_version=ASSET_CACHE_CONTRACT_VERSION, stable_prefix=stable_prefix,
        stable_prefix_tokens=len(stable_prefix.encode("utf-8")), tool_schema=[],
        output_schema=output_type.model_json_schema(), reasoning_effort=reasoning_effort or ASSET_CACHE_REASONING_EFFORT,
        verbosity="low", privacy_scope=privacy_scope, processing_region="default",
        expected_reuse_count=2, expected_reuse_probability=1.0,
        explicit_breakpoint_available=True, explicit_breakpoints=("stable_developer_prefix",),
        dynamic_suffix_fields=("source_frames", "object_context", "run_id", "object_id", "round", "feedback"),
    )
