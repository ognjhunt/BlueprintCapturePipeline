from __future__ import annotations

import json
import types

from blueprint_pipeline.openai_prompt_cache_canary import run_mechanics_canary


def test_five_call_canary_retains_write_read_version_and_one_off_proof(
    tmp_path, monkeypatch
) -> None:
    key_file = tmp_path / "openai-key"
    key_file.write_text("test-only-key\n", encoding="utf-8")
    key_file.chmod(0o600)
    requests: list[dict] = []
    usages = [
        (0, 1_464),
        (1_464, 0),
        (1_464, 0),
        (0, 2_500),
        (0, 0),
    ]

    class Responses:
        def create(self, **kwargs):
            requests.append(kwargs)
            cached, written = usages.pop(0)
            dynamic = 800 if cached or written else 120
            input_tokens = cached + written + dynamic
            return types.SimpleNamespace(
                id=f"resp_{len(requests)}",
                status="completed",
                output_text="OK",
                usage=types.SimpleNamespace(
                    input_tokens=input_tokens,
                    output_tokens=2,
                    input_tokens_details=types.SimpleNamespace(
                        cached_tokens=cached,
                        cache_write_tokens=written,
                    ),
                    output_tokens_details=types.SimpleNamespace(reasoning_tokens=0),
                ),
            )

    class Client:
        def __init__(self, *, api_key, max_retries, timeout):
            assert api_key == "test-only-key"
            assert max_retries == 0
            assert timeout == 60.0
            self.responses = Responses()

    import openai

    monkeypatch.setattr(openai, "OpenAI", Client)
    report = run_mechanics_canary(
        output_dir=tmp_path / "output",
        api_key_file=key_file,
        max_total_cost_usd=0.25,
        source_commit="1" * 40,
        verify_source_commit=False,
    )

    assert report["status"] == "passed"
    assert report["request_count"] == 5
    assert report["retry_cap"] == 0
    assert report["provider_stable_prefix_write_tokens"] == 1_464
    assert report["calls"][0]["usage"]["cache_write_tokens"] == 1_464
    assert report["calls"][1]["usage"]["cached_tokens"] == 1_464
    assert report["calls"][2]["usage"]["cached_tokens"] == 1_464
    assert report["calls"][3]["usage"]["cache_write_tokens"] == 2_500
    assert report["calls"][4]["usage"]["cache_write_tokens"] == 0
    assert requests[0]["prompt_cache_key"] == requests[1]["prompt_cache_key"]
    assert requests[1]["prompt_cache_key"] == requests[2]["prompt_cache_key"]
    assert requests[3]["prompt_cache_key"] != requests[0]["prompt_cache_key"]
    assert "prompt_cache_key" not in requests[4]
    assert requests[4]["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert "prompt_cache_breakpoint" not in json.dumps(requests[4]["input"])
    retained = (tmp_path / "output" / "openai_prompt_cache_mechanics_canary.v1.json")
    assert retained.is_file()
    text = retained.read_text(encoding="utf-8")
    assert "test-only-key" not in text
    assert "blueprint:cache:v1:" not in text
