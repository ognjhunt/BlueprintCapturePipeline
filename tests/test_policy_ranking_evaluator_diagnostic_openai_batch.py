from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.policy_ranking_evaluator_diagnostic import (
    GPT5_MODEL,
    GPT54_MINI_MODEL,
    complete_graph_diagnostic_protocol,
)
from blueprint_pipeline.policy_ranking_evaluator_diagnostic_openai_batch import (
    OpenAIBatchDiagnosticError,
    TRANSPORT_REPAIR_ARM_ID,
    TRANSPORT_REPAIR_MAX_OUTPUT_TOKENS,
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
    pairs = [{**pair, "pair_id": f"pair-{index}"} for index in range(441)]
    inventory = {
        "status": "ready",
        "pair_count": 441,
        "pairs": pairs,
    }
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
    assert line["custom_id"] == "pair-0"
    assert manifest["policy_identity_in_batch_body"] is False


def test_prepare_complete_graph_shard_binds_medium_gpt5_configuration(
    tmp_path: Path,
) -> None:
    frames = []
    for index in range(32):
        path = tmp_path / f"successor-{index}.jpg"
        path.write_bytes(str(index).encode())
        frames.append(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    pair = {
        "task_instruction": "move cup",
        "episode_a": {"frames": frames, "policy_id_internal_only": "secret-a"},
        "episode_b": {"frames": frames, "policy_id_internal_only": "secret-b"},
    }
    pairs = [{**pair, "pair_id": f"pair-{index}"} for index in range(1323)]
    inventory = {
        "status": "ready",
        "pair_count": 1323,
        "protocol_sha256": complete_graph_diagnostic_protocol()["protocol_sha256"],
        "pairs": pairs,
    }
    inventory["inventory_sha256"] = canonical_sha256(inventory)

    manifest = prepare_shard(
        inventory,
        model=GPT5_MODEL,
        offset=0,
        count=1,
        jsonl_path=tmp_path / "successor.jsonl",
        manifest_path=tmp_path / "successor-manifest.json",
        source_commit="a" * 40,
        reasoning_effort="medium",
        max_output_tokens=4000,
        arm_id="gpt5_complete_graph",
    )
    body = json.loads((tmp_path / "successor.jsonl").read_text())["body"]
    assert body["reasoning"] == {"effort": "medium"}
    assert body["max_output_tokens"] == 4000
    assert manifest["arm_id"] == "gpt5_complete_graph"
    assert manifest["reasoning_effort"] == "medium"


def test_prepare_registered_transport_repair_subset_binds_only_frozen_repair_config(
    tmp_path: Path,
) -> None:
    frames = []
    for index in range(32):
        path = tmp_path / f"repair-{index}.jpg"
        path.write_bytes(str(index).encode())
        frames.append(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    inventory = {
        "status": "ready",
        "comparison_graph_kind": "registered_complete_graph_subset",
        "pair_count": 1,
        "protocol_sha256": complete_graph_diagnostic_protocol()["protocol_sha256"],
        "parent_inventory_sha256": "b" * 64,
        "outcome_labels_accessed_to_build_pairs": False,
        "pairs": [
            {
                "pair_id": "repair-pair",
                "task_instruction": "move cup",
                "episode_a": {"frames": frames, "policy_id_internal_only": "secret-a"},
                "episode_b": {"frames": frames, "policy_id_internal_only": "secret-b"},
            }
        ],
    }
    inventory["inventory_sha256"] = canonical_sha256(inventory)

    manifest = prepare_shard(
        inventory,
        model=GPT5_MODEL,
        offset=0,
        count=1,
        jsonl_path=tmp_path / "repair.jsonl",
        manifest_path=tmp_path / "repair-manifest.json",
        source_commit="a" * 40,
        reasoning_effort="medium",
        max_output_tokens=TRANSPORT_REPAIR_MAX_OUTPUT_TOKENS,
        arm_id=TRANSPORT_REPAIR_ARM_ID,
    )
    body = json.loads((tmp_path / "repair.jsonl").read_text())["body"]
    assert body["max_output_tokens"] == TRANSPORT_REPAIR_MAX_OUTPUT_TOKENS
    assert manifest["arm_id"] == TRANSPORT_REPAIR_ARM_ID
    assert "secret-a" not in (tmp_path / "repair.jsonl").read_text()
    assert "secret-b" not in (tmp_path / "repair.jsonl").read_text()

    with pytest.raises(
        OpenAIBatchDiagnosticError,
        match="complete_graph_transport_repair_config_mismatch",
    ):
        prepare_shard(
            inventory,
            model=GPT5_MODEL,
            offset=0,
            count=1,
            jsonl_path=tmp_path / "invalid-repair.jsonl",
            manifest_path=tmp_path / "invalid-repair-manifest.json",
            source_commit="a" * 40,
            reasoning_effort="medium",
            max_output_tokens=4000,
            arm_id="gpt5_complete_graph",
        )


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
    pairs = [{"pair_id": f"pair-{index}"} for index in range(441)]
    pairs[0]["pair_id"] = "pair-id"
    inventory = {"status": "ready", "pair_count": 441, "pairs": pairs}
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
