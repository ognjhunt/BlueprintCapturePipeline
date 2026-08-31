"""Bounded five-call GPT-5.6 Sol explicit prompt-cache mechanics canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .openai_prompt_cache import (
    create_prompt_cache_policy,
    explicit_cache_input,
    explicit_cache_request_kwargs,
    usage_and_cost_receipt,
    worst_case_reservation_usd,
)


SCHEMA_VERSION = "openai_prompt_cache_mechanics_canary.v1"
DEFAULT_MODEL = "gpt-5.6-sol"
EXACT_REQUEST_COUNT = 5
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class OpenAIPromptCacheCanaryError(RuntimeError):
    """The bounded provider canary was not admissible or did not meet its contract."""


def load_secure_api_key_file(path_value: str | Path | None) -> str:
    raw = str(path_value or os.getenv("OPENAI_API_KEY_FILE") or "").strip()
    if not raw:
        raise OpenAIPromptCacheCanaryError("openai_api_key_file_missing")
    path = Path(raw).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise OpenAIPromptCacheCanaryError("openai_api_key_file_not_secure_0600")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise OpenAIPromptCacheCanaryError("openai_api_key_file_empty")
    return value


def _stable_prefix(contract_version: str) -> str:
    corpus = " ".join(
        ["stable evidence boundary cache reuse"] * 280
    )
    return (
        f"Blueprint rights-safe prompt-cache mechanics contract {contract_version}.\n"
        "Return only the word OK. This synthetic text contains no customer, scene, user, "
        "credential, host, policy, or private artifact data. Its sole purpose is to prove that "
        "an explicit reusable prefix is written once and read when only a suffix changes.\n"
        f"{corpus}"
    )


def _policy(*, contract_version: str, stable_prefix: str, reuse_count: int):
    return create_prompt_cache_policy(
        model=DEFAULT_MODEL,
        family="prompt_cache_mechanics_canary",
        contract_version=contract_version,
        stable_prefix=stable_prefix,
        stable_prefix_tokens=len(stable_prefix.encode("utf-8")) // 5,
        tool_schema=[],
        output_schema={"type": "text", "exact": "OK"},
        reasoning_effort="none",
        verbosity="low",
        privacy_scope="synthetic_rights_safe",
        processing_region="default",
        expected_reuse_count=reuse_count,
        expected_reuse_probability=1.0 if reuse_count else 0.0,
        explicit_breakpoint_available=reuse_count > 0,
        dynamic_suffix_fields=("run_id", "call_name", "dynamic_nonce"),
    )


def _safe_policy(policy) -> dict[str, Any]:
    value = policy.model_dump(mode="json")
    raw_key = str(value.pop("cache_key", "") or "")
    value["cache_key_digest"] = (
        "sha256:" + hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        if raw_key
        else None
    )
    return value


def run_mechanics_canary(
    *,
    output_dir: str | Path,
    api_key_file: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    max_total_cost_usd: float = 0.25,
    timeout_seconds: float = 60.0,
    source_commit: str,
    verify_source_commit: bool = True,
) -> dict[str, Any]:
    if model != DEFAULT_MODEL:
        raise OpenAIPromptCacheCanaryError("canary_model_must_be_gpt_5_6_sol")
    if not 0 < max_total_cost_usd <= 1.0:
        raise OpenAIPromptCacheCanaryError("canary_cost_cap_invalid")
    if _COMMIT.fullmatch(source_commit) is None:
        raise OpenAIPromptCacheCanaryError("canary_source_commit_invalid")
    if verify_source_commit:
        repo_root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.stdout.strip() != source_commit:
            raise OpenAIPromptCacheCanaryError(
                "canary_source_commit_does_not_match_checkout_head"
            )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if status.stdout.strip():
            raise OpenAIPromptCacheCanaryError("canary_checkout_not_clean")
    key = load_secure_api_key_file(api_key_file)
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - production dependency failure
        raise OpenAIPromptCacheCanaryError("openai_sdk_missing") from exc
    client = OpenAI(
        api_key=key,
        max_retries=0,
        timeout=max(1.0, timeout_seconds),
    )
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    report_path = output_root / "openai_prompt_cache_mechanics_canary.v1.json"

    # A canary must prove a cold write for the exact deployed source even when
    # an earlier canary ran inside the provider TTL. Deployment identity is a
    # stable contract dimension, unlike the per-call run IDs kept after the
    # breakpoint, so it deliberately isolates canary generations.
    cache_generation = source_commit[:12]
    contract_v1 = f"mechanics-v1-{cache_generation}"
    contract_v2 = f"mechanics-v2-{cache_generation}"
    prefix_v1 = _stable_prefix(contract_v1)
    prefix_v2 = _stable_prefix(contract_v2)
    policy_v1 = _policy(
        contract_version=contract_v1,
        stable_prefix=prefix_v1,
        reuse_count=2,
    )
    policy_v2 = _policy(
        contract_version=contract_v2,
        stable_prefix=prefix_v2,
        reuse_count=1,
    )
    one_off_policy = create_prompt_cache_policy(
        model=model,
        family="prompt_cache_mechanics_one_off",
        contract_version="mechanics-one-off-v1",
        stable_prefix="One-off explicit-only request with no breakpoint.",
        stable_prefix_tokens=0,
        tool_schema=[],
        output_schema={"type": "text", "exact": "OK"},
        reasoning_effort="none",
        verbosity="low",
        privacy_scope="synthetic_rights_safe",
        processing_region="default",
        expected_reuse_count=0,
        expected_reuse_probability=0.0,
        explicit_breakpoint_available=False,
        dynamic_suffix_fields=("one_off_payload",),
    )
    calls = [
        ("A_reusable_write", policy_v1, prefix_v1, "run-a", "suffix-a"),
        ("B_reusable_read", policy_v1, prefix_v1, "run-a", "suffix-b"),
        ("C_different_run_id_read", policy_v1, prefix_v1, "run-c", "suffix-c"),
        ("D_contract_version_miss", policy_v2, prefix_v2, "run-d", "suffix-d"),
        ("E_one_off_no_write", one_off_policy, None, "run-e", "suffix-e"),
    ]
    records: list[dict[str, Any]] = []
    cumulative_cost = 0.0
    for call_name, policy, stable_prefix, run_id, nonce in calls:
        dynamic_text = json.dumps(
            {
                "call_name": call_name,
                "run_id": run_id,
                "dynamic_nonce": nonce,
                "request": "Return only OK.",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if stable_prefix is None:
            rendered_input: str | list[dict[str, Any]] = dynamic_text
        else:
            rendered_input = explicit_cache_input(
                policy=policy,
                stable_developer_prefix=stable_prefix,
                dynamic_input=dynamic_text,
            )
        input_ceiling = (
            policy.economics.stable_prefix_tokens
            + len(dynamic_text.encode("utf-8"))
        )
        projected = worst_case_reservation_usd(
            model=model,
            input_token_ceiling=input_ceiling,
            max_output_tokens=32,
            cache_policy=policy,
        )
        if projected is None or cumulative_cost + projected > max_total_cost_usd:
            raise OpenAIPromptCacheCanaryError("canary_cost_cap_would_be_exceeded")
        request_shape = {
            "model": model,
            "call_name": call_name,
            "policy_digest": policy.policy_digest,
            "stable_prefix_digest": policy.stable_prefix_digest,
            "dynamic_suffix_digest": canonical_digest({"dynamic_text": dynamic_text}),
            "prompt_cache_options": explicit_cache_request_kwargs(policy).get(
                "prompt_cache_options"
            ),
            "cache_key_present": policy.cache_key is not None,
            "breakpoint_count": 1 if stable_prefix is not None else 0,
            "store": False,
            "max_output_tokens": 32,
            "reasoning_effort": "none",
        }
        started = time.monotonic()
        response = client.responses.create(
            model=model,
            input=rendered_input,
            max_output_tokens=32,
            reasoning={"effort": "none"},
            text={"verbosity": "low"},
            store=False,
            **explicit_cache_request_kwargs(policy),
        )
        usage = usage_and_cost_receipt(response, model=model)
        actual = usage.get("estimated_total_cost_usd")
        if not isinstance(actual, (int, float)):
            raise OpenAIPromptCacheCanaryError("canary_usage_cost_missing")
        cumulative_cost += float(actual)
        records.append(
            {
                "call_name": call_name,
                "request_shape_digest": canonical_digest(request_shape),
                "request_shape": request_shape,
                "cache_policy": _safe_policy(policy),
                "usage": usage,
                "latency_seconds": max(0.0, time.monotonic() - started),
                "provider_response_id": str(getattr(response, "id", "") or ""),
                "provider_response_status": str(
                    getattr(response, "status", "") or ""
                ),
                "raw_prompt_recorded": False,
                "raw_secret_values_recorded": False,
            }
        )

    by_name = {record["call_name"]: record for record in records}
    usage_a = by_name["A_reusable_write"]["usage"]
    usage_b = by_name["B_reusable_read"]["usage"]
    usage_c = by_name["C_different_run_id_read"]["usage"]
    usage_d = by_name["D_contract_version_miss"]["usage"]
    usage_e = by_name["E_one_off_no_write"]["usage"]
    # Provider tokenization is authoritative for the deployed success target.
    # The deterministic byte/5 estimate is intentionally conservative for the
    # economic decision, but can overstate the exact cacheable-token count.
    # Compare reads with the prefix OpenAI actually wrote on call A.
    provider_stable_write_tokens = int(usage_a["cache_write_tokens"])
    blockers: list[str] = []
    if usage_a["cached_tokens"] != 0 or usage_a["cache_write_tokens"] < 1_024:
        blockers.append("call_a_did_not_write_reusable_prefix")
    if usage_b["cached_tokens"] < 0.7 * provider_stable_write_tokens:
        blockers.append("call_b_reusable_prefix_read_below_target")
    if usage_c["cached_tokens"] < 0.7 * provider_stable_write_tokens:
        blockers.append("call_c_different_run_id_did_not_read")
    if usage_d["cached_tokens"] != 0 or usage_d["cache_write_tokens"] < 1_024:
        blockers.append("call_d_contract_version_not_isolated")
    if usage_e["cache_write_tokens"] != 0:
        blockers.append("call_e_one_off_created_cache_write")
    if len(records) != EXACT_REQUEST_COUNT:
        blockers.append("exact_request_count_mismatch")
    reusable = [usage_a, usage_b, usage_c]
    reusable_cost = sum(float(row["estimated_total_cost_usd"]) for row in reusable)
    reusable_no_cache = sum(
        float(row["estimated_cost_without_caching_usd"]) for row in reusable
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not blockers else "failed",
        "model": model,
        "source_commit": source_commit,
        "cache_generation": cache_generation,
        "source_commit_matches_checkout_head": verify_source_commit,
        "request_count": len(records),
        "exact_request_count": EXACT_REQUEST_COUNT,
        "retry_cap": 0,
        "max_total_cost_usd": max_total_cost_usd,
        "estimated_total_cost_usd": cumulative_cost,
        "reusable_family_estimated_cost_usd": reusable_cost,
        "provider_stable_prefix_write_tokens": provider_stable_write_tokens,
        "reusable_family_no_cache_cost_usd": reusable_no_cache,
        "reusable_family_estimated_savings_usd": reusable_no_cache - reusable_cost,
        "reusable_family_estimated_savings_ratio": (
            (reusable_no_cache - reusable_cost) / reusable_no_cache
            if reusable_no_cache
            else 0.0
        ),
        "calls": records,
        "blockers": blockers,
        "raw_prompts_recorded": False,
        "raw_secret_values_recorded": False,
        "api_key_file_path_recorded": False,
        "official_billing_status": "pending_openai_platform_settlement",
        "report_digest": "",
    }
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    write_json(report_path, report)
    if blockers:
        raise OpenAIPromptCacheCanaryError(
            "openai_prompt_cache_canary_failed:" + ",".join(blockers)
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-key-file")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--max-total-cost-usd", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    run_mechanics_canary(
        output_dir=args.output_dir,
        api_key_file=args.api_key_file,
        model=args.model,
        max_total_cost_usd=args.max_total_cost_usd,
        timeout_seconds=args.timeout_seconds,
        source_commit=args.source_commit,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MODEL",
    "EXACT_REQUEST_COUNT",
    "OpenAIPromptCacheCanaryError",
    "SCHEMA_VERSION",
    "run_mechanics_canary",
    "load_secure_api_key_file",
]
