from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline.policy_ranking_evaluator_diagnostic import GPT54_MINI_MODEL
from blueprint_pipeline.policy_ranking_evaluator_diagnostic_openai_batch import (
    collect_shard,
    prepare_shard,
)
from blueprint_pipeline.policy_ranking_roboarena_calibration import canonical_sha256


def test_prepare_batch_shard_contains_no_policy_identity(tmp_path: Path) -> None:
    frames = []
    for index in range(32):
        path = tmp_path / f"{index}.jpg"
        path.write_bytes(str(index).encode())
        frames.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pair = {
        "pair_id": "pair-id",
        "task_instruction": "move cup",
        "episode_a": {"frames": frames, "policy_id_internal_only": "secret-a"},
        "episode_b": {"frames": frames, "policy_id_internal_only": "secret-b"},
    }
    inventory = {"status": "ready", "pair_count": 441, "pairs": [pair] * 441}
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    manifest = prepare_shard(
        inventory,
        model=GPT54_MINI_MODEL,
        offset=0,
        count=1,
        jsonl_path=tmp_path / "batch.jsonl",
        manifest_path=tmp_path / "manifest.json",
        source_commit="a" * 40,
    )
    line = json.loads((tmp_path / "batch.jsonl").read_text())
    encoded = json.dumps(line)
    assert "secret-a" not in encoded
    assert "secret-b" not in encoded
    assert line["custom_id"] == "pair-id"
    assert manifest["policy_identity_in_batch_body"] is False


def test_collect_failed_batch_records_error_and_deletes_input(
    tmp_path: Path, monkeypatch
) -> None:
    deleted: list[str] = []

    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"
            self.batches = SimpleNamespace(
                retrieve=lambda batch_id: SimpleNamespace(
                    id=batch_id,
                    status="failed",
                    errors={"data": [{"code": "invalid_request"}]},
                    request_counts={"completed": 0, "failed": 0, "total": 0},
                    usage={"input_tokens": 0, "output_tokens": 0},
                )
            )
            self.files = SimpleNamespace(delete=lambda file_id: deleted.append(file_id))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient))
    key = tmp_path / "key"
    key.write_text("test-key")
    key.chmod(0o600)
    report = collect_shard(
        {
            "batch_id": "batch-id",
            "input_file_id": "file-id",
            "shard_id": "shard-id",
        },
        {},
        {},
        api_key_file=key,
        output_path=tmp_path / "collection.json",
    )

    assert report["status"] == "failed"
    assert report["terminal"] is True
    assert report["request_counts"]["total"] == 0
    assert report["input_file_deleted"] is True
    assert deleted == ["file-id"]


def test_collect_completed_batch_retains_incomplete_row_and_deletes_input(
    tmp_path: Path, monkeypatch
) -> None:
    deleted: list[str] = []
    raw_line = json.dumps(
        {
            "custom_id": "pair-id",
            "response": {
                "status_code": 200,
                "body": {
                    "id": "response-id",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "input_tokens_details": {"cached_tokens": 0},
                    },
                },
            },
            "error": None,
        }
    )

    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"
            self.batches = SimpleNamespace(
                retrieve=lambda batch_id: SimpleNamespace(
                    id=batch_id,
                    status="completed",
                    output_file_id="output-file-id",
                )
            )
            self.files = SimpleNamespace(
                content=lambda file_id: SimpleNamespace(text=raw_line),
                delete=lambda file_id: deleted.append(file_id),
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient))
    key = tmp_path / "key"
    key.write_text("test-key")
    key.chmod(0o600)
    pair = {"pair_id": "pair-id"}
    inventory = {"status": "ready", "pairs": [pair] * 441}
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    report = collect_shard(
        {
            "batch_id": "batch-id",
            "input_file_id": "input-file-id",
            "shard_id": "shard-id",
        },
        {
            "pair_ids": ["pair-id"],
            "arm_id": "gpt54_mini_challenger",
            "model": GPT54_MINI_MODEL,
        },
        inventory,
        api_key_file=key,
        output_path=tmp_path / "collection.json",
    )

    assert report["status"] == "failed"
    assert report["result_count"] == 0
    assert report["error_count"] == 1
    assert report["errors"][0]["response_status"] == "incomplete"
    assert report["estimated_batch_cost_usd"] > 0
    assert report["input_file_deleted"] is True
    assert deleted == ["input-file-id"]
