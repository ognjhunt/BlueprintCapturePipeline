from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import native_g1_provider_runtime as provider_runtime

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_development_pair import EPISODE_FILENAME, TRACE_FILENAME
from blueprint_pipeline.native_g1_development_selection import TEMPLATE_FIELDS
from blueprint_pipeline.native_g1_provider_runtime import (
    PAIR_ORDER,
    _query_count,
    _stage_models,
    _template,
    run_g1_provider_campaign,
)


def test_runtime_template_carries_only_worker_fields(tmp_path: Path) -> None:
    source = tmp_path / "publisher-source/source"
    source.mkdir(parents=True)
    models = {
        "sonic": {
            "files": [
                {"role": "encoder", "sha256": "sha256:" + "a" * 64},
                {"role": "decoder", "sha256": "sha256:" + "b" * 64},
            ]
        }
    }
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    path = _template(
        root=runtime,
        output=tmp_path,
        packet=runtime / "inputs/scene_packets/movement",
        objective_name="movement",
        models=models,
        runtime_python=tmp_path / "policy-runtime/bin/python",
        provisioning_path=tmp_path / "runtime-provision.json",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert set(value) == TEMPLATE_FIELDS
    assert value["max_steps"] == 3000
    assert value["device"] == "cuda:0"
    assert value["sonic_encoder_sha256"] == "sha256:" + "a" * 64


def test_model_staging_fetches_distinct_candidates_concurrently_in_pair_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = threading.Barrier(len(PAIR_ORDER))

    def fetch(*, inventory_path: Path, candidate_id: str, output_dir: Path) -> dict:
        barrier.wait(timeout=5)
        return {"candidate_id": candidate_id}

    def sonic(*, inventory_path: Path, output_dir: Path) -> dict:
        return {"files": []}

    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_provider_runtime._load_script",
        lambda path, name: (
            SimpleNamespace(materialize_candidate=fetch)
            if name == "g1_checkpoint_fetcher"
            else SimpleNamespace(stage_sonic_assets=sonic)
        ),
    )
    output = tmp_path / "output"
    output.mkdir()
    result = _stage_models(tmp_path / "runtime", output)

    assert [row["candidate_id"] for row in result["checkpoints"]] == list(PAIR_ORDER)
    assert [
        json.loads((output / "models" / (candidate + ".json")).read_text())["candidate_id"]
        for candidate in PAIR_ORDER
    ] == list(PAIR_ORDER)


def test_query_count_requires_digest_bound_observed_policy_queries(tmp_path: Path) -> None:
    candidate = "humanoidarena_dp_g1_dex3_sonic"
    episode_dir = tmp_path / candidate / "episode"
    episode_dir.mkdir(parents=True)
    trace = {"candidate_id": candidate, "policy_query_count": 1}
    trace["trace_digest"] = canonical_digest(trace, digest_field="trace_digest")
    (episode_dir / TRACE_FILENAME).write_text(json.dumps(trace), encoding="utf-8")
    (episode_dir / EPISODE_FILENAME).write_text(
        json.dumps({"trace_digest": trace["trace_digest"]}), encoding="utf-8"
    )
    assert _query_count(tmp_path, candidate) == 1
    trace["policy_query_count"] = 0
    (episode_dir / TRACE_FILENAME).write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="g1_provider_policy_query_evidence_invalid"):
        _query_count(tmp_path, candidate)


def test_runner_refuses_unpinned_non_isaac_environment(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="g1_provider_runtime_environment_invalid"):
        run_g1_provider_campaign(tmp_path, output)
    assert list(output.iterdir()) == []


def test_runtime_import_preflight_retains_all_failures_before_model_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()

    def probe(name: str) -> SimpleNamespace:
        if name in {"isaaclab_arena_g1", "onnxruntime"}:
            raise ModuleNotFoundError(name)
        return SimpleNamespace(__version__="available")

    monkeypatch.setattr(provider_runtime.importlib, "import_module", probe)
    result = provider_runtime._preflight_g1_runtime_imports(output)
    assert result["status"] == "blocked"
    assert [row["module"] for row in result["imports"] if not row["available"]] == [
        "isaaclab_arena_g1", "onnxruntime",
    ]
    assert json.loads(
        (output / "native_g1_runtime_import_preflight.v1.json").read_text(encoding="utf-8")
    ) == result
    assert result["receipt_digest"] == canonical_digest(result, digest_field="receipt_digest")
