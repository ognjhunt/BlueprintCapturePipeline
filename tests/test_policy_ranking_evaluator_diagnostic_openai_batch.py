from __future__ import annotations

import hashlib
import json
from pathlib import Path

from blueprint_pipeline.policy_ranking_evaluator_diagnostic import GPT54_MINI_MODEL
from blueprint_pipeline.policy_ranking_evaluator_diagnostic_openai_batch import prepare_shard
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
