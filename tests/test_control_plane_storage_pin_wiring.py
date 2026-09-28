"""Producers pin the derived directories they create; the terminal dispatch releases them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_launch_activation_worker as activation_worker
from blueprint_pipeline import task_evaluation_launch_preparation_worker as preparation_worker
from blueprint_pipeline import task_evaluation_policy_canary_dispatcher as dispatcher
from blueprint_pipeline.control_plane_storage_pins import (
    PINS_ROOT_ENV,
    load_storage_pins,
    pin_activation_best_effort,
    pin_path,
    write_storage_pin,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_episode_compilation_worker import (
    COMPILER_OUTPUT_SCHEMA_VERSION,
    process_episode_compilation_queue,
)
from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
    stage_launch_preparation_request,
)
from tests.test_task_evaluation_episode_compilation_worker import _stage
from tests.test_task_evaluation_launch_preparation_worker import (
    SERVICE_ACCOUNT,
    fake_adapter,
    fetcher,
    request_with_fetchable_bytes,
)


def test_preparation_pins_its_directory_after_materializing(tmp_path: Path) -> None:
    value, payloads = request_with_fetchable_bytes()
    queue = tmp_path / "queue"
    stage_launch_preparation_request(value=value, queue_root=queue, submitted_by="blueprint-webapp")

    run = preparation_worker.process_launch_preparation_queue(
        queue_root=queue,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        source_commit=value["expected_production_commit"],
        fetcher=fetcher(payloads),
        adapter_materializer=fake_adapter,
        episode_compilation_queue_root=tmp_path / "episode-compilation",
        storage_pins_root=tmp_path / "pins",
    )

    assert run["results"][0]["status"] != "blocked", run["results"][0]
    pin = json.loads(
        pin_path(tmp_path / "pins", "preparation", value["preparation_id"]).read_text(
            encoding="utf-8"
        )
    )
    assert pin["paths"] == [str(tmp_path / "inputs" / value["preparation_id"])]
    assert pin["depends_on"] == []
    assert pin["released_at_epoch"] is None


def test_compilation_pins_its_output_and_depends_on_its_preparation(tmp_path: Path) -> None:
    queue, inputs, envelope = _stage(tmp_path)

    def compile_episode(*, envelope, materialized_references, output_root):
        packet = output_root / "native-task-arena-bundle.zip"
        packet.write_bytes(b"production-compiled-episode-packet")
        adapter_path = output_root / "adapter-result.json"
        adapter = {
            "schema_version": "task_evaluation_native_arena_adapter_result.v1",
            "status": "native_arena_adapter_materialized",
            "packet_receipt_digest": "sha256:" + "a" * 64,
            "runtime_source_receipt_digest": "sha256:" + "b" * 64,
            "result_digest": "",
        }
        adapter["result_digest"] = canonical_digest(adapter, digest_field="result_digest")
        adapter_path.write_text(json.dumps(adapter) + "\n", encoding="utf-8")
        result = {
            "schema_version": COMPILER_OUTPUT_SCHEMA_VERSION,
            "status": "completed",
            "run_id": envelope["run_id"],
            "configured_scene_revision_digest": envelope["configured_scene_revision_digest"],
            "compiled_episode_packet": {
                "format": "native_task_arena_bundle_zip",
                "path": str(packet),
                "digest": "sha256:" + hashlib.sha256(packet.read_bytes()).hexdigest(),
                "size_bytes": packet.stat().st_size,
            },
            "adapter_result": {
                "path": str(adapter_path),
                "digest": adapter["result_digest"],
                "packet_receipt_digest": adapter["packet_receipt_digest"],
                "runtime_source_receipt_digest": adapter["runtime_source_receipt_digest"],
            },
            "compiled_by_production": True,
            "customer_supplied_prebuilt_episode_packet": False,
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
            "raw_secret_values_recorded": False,
            "compiler_output_digest": "",
        }
        result["compiler_output_digest"] = canonical_digest(
            result, digest_field="compiler_output_digest"
        )
        return result

    run = process_episode_compilation_queue(
        queue_root=queue,
        input_root=inputs,
        output_root=tmp_path / "outputs",
        source_commit=envelope["expected_production_commit"],
        episode_compiler=compile_episode,
        storage_pins_root=tmp_path / "pins",
    )

    assert run["results"][0]["status"] == "compiled_for_production_launch"
    pin = json.loads(
        pin_path(tmp_path / "pins", "compilation", envelope["compilation_id"]).read_text(
            encoding="utf-8"
        )
    )
    assert pin["paths"] == [str(tmp_path / "outputs" / envelope["compilation_id"])]
    assert pin["depends_on"] == [{"kind": "preparation", "owner_id": envelope["preparation_id"]}]

    # A refused compile leaves no pin behind.
    (tmp_path / "second").mkdir()
    queue_b, inputs_b, envelope_b = _stage(tmp_path / "second")
    refused = process_episode_compilation_queue(
        queue_root=queue_b,
        input_root=inputs_b,
        output_root=tmp_path / "second" / "outputs",
        source_commit=envelope_b["expected_production_commit"],
        episode_compiler=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        storage_pins_root=tmp_path / "second" / "pins",
    )
    assert refused["results"][0]["status"] == "blocked"
    assert not (tmp_path / "second" / "pins").exists()


def test_activation_pin_depends_on_preparation_and_compilation_and_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = {"activation_id": "activation-1", "preparation": {"preparation_id": "prep-1"}}

    pin = pin_activation_best_effort(request, tmp_path / "launch-activations", pins_root=tmp_path / "pins")

    assert pin["paths"] == [str(tmp_path / "launch-activations" / "activation-1")]
    assert pin["depends_on"] == [
        {"kind": "compilation", "owner_id": "prep-1"},
        {"kind": "preparation", "owner_id": "prep-1"},
    ]
    assert pin_activation_best_effort({"activation_id": "x"}, tmp_path, pins_root=tmp_path / "pins") is None
    assert pin_activation_best_effort("not-a-request", tmp_path, pins_root=tmp_path / "pins") is None
    monkeypatch.delenv(PINS_ROOT_ENV, raising=False)
    assert pin_activation_best_effort(request, tmp_path) is None
    monkeypatch.setenv(PINS_ROOT_ENV, str(tmp_path / "env-pins"))
    assert pin_activation_best_effort(request, tmp_path)["owner_id"] == "activation-1"

    # The worker pins only activations that reached the prepared terminal state.
    source = Path(activation_worker.__file__).read_text(encoding="utf-8")
    assert (
        'if terminal_state == "prepared":\n'
        "            storage_pins.pin_activation_best_effort(request, activation_base)"
    ) in source


def test_terminal_dispatch_receipt_releases_the_activation_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pins = tmp_path / "pins"
    write_storage_pin(pins_root=pins, kind="preparation", owner_id="prep-1", paths=[tmp_path / "p"])
    write_storage_pin(
        pins_root=pins,
        kind="activation",
        owner_id="activation-1",
        paths=[tmp_path / "a"],
        depends_on=[{"kind": "preparation", "owner_id": "prep-1"}],
    )
    monkeypatch.setenv(PINS_ROOT_ENV, str(pins))

    dispatcher._release_activation_storage_pin(tmp_path / "dispatches" / "activation-1")

    statuses = {(pin["kind"], pin["owner_id"]): pin["status"] for pin in load_storage_pins(pins)}
    assert statuses == {
        ("preparation", "prep-1"): "released",
        ("activation", "activation-1"): "released",
    }
    monkeypatch.delenv(PINS_ROOT_ENV)
    dispatcher._release_activation_storage_pin(tmp_path / "dispatches" / "nothing")

    # The release follows the terminal receipt of an executed dispatch, and only that one.
    source = Path(dispatcher.__file__).read_text(encoding="utf-8")
    assert source.count('_write_exclusive(root / "dispatch_receipt.json", receipt)') == 2
    assert (
        '_write_exclusive(root / "dispatch_receipt.json", receipt)\n'
        "    _release_activation_storage_pin(root)\n"
        "    _event_and_sync("
    ) in source
