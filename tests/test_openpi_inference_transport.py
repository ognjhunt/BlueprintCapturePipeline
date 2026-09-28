"""Real loopback WebSockets with the pinned OpenPI wire sequence; no policy model."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from functools import partial
import json
import queue
import sys
import threading
import time
import types

import msgpack
import numpy as np
import pytest
import websockets.asyncio.server
import websockets.sync.client

from blueprint_pipeline import openpi_droid_policy_runtime as runtime
from blueprint_pipeline import native_task_arena_policy_canary_worker as worker
from blueprint_pipeline.policy_request_evidence import persist_request_evidence
from tests.test_openpi_droid_policy_runtime import _cohort, _runtime_metadata


class _PinnedWireClient:
    """The pinned client's ctor hook and pack/send/recv/unpack wire sequence."""

    def __init__(self, host, port, api_key=None):
        self._uri, self._api_key = f"ws://{host}:{port}", api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self):
        raise AssertionError("The upstream infinite startup loop must be overridden")

    def get_server_metadata(self):
        return self._server_metadata

    def infer(self, observation):
        self._ws.send(msgpack.packb(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack.unpackb(response)


@pytest.fixture
def bounded_default_client(monkeypatch):
    vendor = types.ModuleType("openpi_client")
    vendor.websocket_client_policy = types.SimpleNamespace(WebsocketClientPolicy=_PinnedWireClient)
    vendor.msgpack_numpy = types.SimpleNamespace(unpackb=msgpack.unpackb)
    monkeypatch.setitem(sys.modules, "openpi_client", vendor)
    monkeypatch.setattr(runtime, "_bounded_openpi_client_type", partial(
        runtime._bounded_openpi_client_type,
        startup_timeout_seconds=0.12, inference_timeout_seconds=0.8,
        close_timeout_seconds=0.03,
    ))
    calls = []
    connect = websockets.sync.client.connect

    def short_heartbeat(*args, **kwargs):
        kwargs.setdefault("ping_interval", 0.03)
        kwargs.setdefault("ping_timeout", 0.03)
        calls.append(dict(kwargs))
        return connect(*args, **kwargs)

    monkeypatch.setattr(websockets.sync.client, "connect", short_heartbeat)
    return calls


@contextmanager
def _server(metadata, *, inference_delay=0.0, metadata_delay=0.0, handshake_delay=0.0):
    ready, stop, requests = queue.Queue(), threading.Event(), []

    async def handler(connection):
        try:
            await asyncio.sleep(metadata_delay)
            await connection.send(msgpack.packb(metadata))
            observation = msgpack.unpackb(await connection.recv())
            requests.append(observation)
            # Like the pinned OpenPI handler, synchronous inference blocks the
            # event loop and its pong handling while the client thread remains live.
            time.sleep(inference_delay)
            await connection.send(msgpack.packb({"actions": [[0.0] * 8] * 10}))
            await connection.wait_closed()
        except websockets.ConnectionClosed:
            pass

    async def before_handshake(_connection, _request):
        await asyncio.sleep(handshake_delay)

    async def serve():
        async with websockets.asyncio.server.serve(
            handler, "127.0.0.1", 0, ping_interval=None, close_timeout=0.03,
            process_request=before_handshake,
        ) as server:
            ready.put(server.sockets[0].getsockname()[1])
            await asyncio.to_thread(stop.wait)

    thread = threading.Thread(target=lambda: asyncio.run(serve()), daemon=True)
    thread.start()
    try:
        yield ready.get(timeout=3), requests
    finally:
        stop.set()
        thread.join(timeout=3)
        assert not thread.is_alive(), "Loopback server cleanup must be bounded"


def _client(tmp_path, port):
    spec = runtime.load_policy_spec(_cohort(tmp_path), policy_id="pi0_fast_droid_jointpos_polaris")
    return runtime.OpenPIWebsocketDroidPolicyClient(spec=spec, host="127.0.0.1", port=port)


def test_cold_inference_outlasts_heartbeat_without_losing_or_repeating_request(
    tmp_path, bounded_default_client,
):
    spec = runtime.load_policy_spec(_cohort(tmp_path), policy_id="pi0_fast_droid_jointpos_polaris")
    with _server(_runtime_metadata(spec), inference_delay=0.25) as (port, requests):
        client = _client(tmp_path, port)
        assert client.preflight_readiness()["candidate_inference_performed"] is False
        assert requests == []
        observation = {"prompt": "fixture pick", "state": [0.1, 0.2]}
        received = client.infer(observation)
        assert np.asarray(received).shape == (10, 8)
        assert requests == [observation]
        assert client.last_request_evidence()["serialization_verified"] is True
        assert client.last_inference_evidence()["server_response_received"] is True
        assert client.candidate_policy_queried is True
        assert client._client is None
    assert len(bounded_default_client) == 2  # Metadata-only readiness, then one request.
    assert all(call["ping_timeout"] is None for call in bounded_default_client)
    assert all(call["open_timeout"] == 0.12 for call in bounded_default_client)


def test_inference_deadline_preserves_one_attempt_and_unproven_response(
    tmp_path, bounded_default_client, monkeypatch,
):
    monkeypatch.setattr(runtime, "_bounded_openpi_client_type", partial(
        runtime._bounded_openpi_client_type, inference_timeout_seconds=0.08,
    ))
    spec = runtime.load_policy_spec(_cohort(tmp_path), policy_id="pi0_fast_droid_jointpos_polaris")
    retained = []
    episode_id = "run--cell--pi05_droid"
    with _server(_runtime_metadata(spec), inference_delay=0.35) as (port, requests):
        client = _client(tmp_path, port)
        client.bind_request_evidence_sink(lambda receipt: retained.append(persist_request_evidence(
            receipt, root=tmp_path / "episodes", episode_id=episode_id, query_index=0,
        )))
        started = time.monotonic()
        with pytest.raises(TimeoutError) as caught:
            client.infer({"prompt": "fixture pick"})
        assert time.monotonic() - started < 0.3
        assert len(requests) == len(retained) == len(bounded_default_client) == 1
        assert client.last_request_evidence()["serialization_verified"] is True
        assert client.candidate_policy_queried is False
        assert client._client is None
        progress = {
            "first_observation_retained": True,
            "candidate_policy_query_attempted": True,
            "candidate_policy_queried": False,
            "policy_request_artifacts": retained,
        }
        path = worker._write_episode_failure_gap(
            output_root=tmp_path, run_id="run", context={"cell_id": "cell", "candidate_id": "pi05_droid"},
            failure=caught.value, progress=progress,
        )
    evidence = json.loads(path.read_text())
    assert evidence["failure_type"] == "TimeoutError"
    assert evidence["candidate_policy_query_attempted"] is True
    assert evidence["candidate_policy_queried"] is False
    assert evidence["policy_response_status"] == "unproven"
    assert evidence["candidate_action_returned"] is False
    assert evidence["actions_reached_robot"] is False
    assert evidence["candidate_policy_action_queries"] == []
    receipt = json.loads((tmp_path / evidence["evidence_artifacts"]["policy_query_receipt"]["relative_path"]).read_text())
    assert receipt["candidate_policy_query_attempted"] is True
    assert receipt["candidate_policy_queried"] is False
    assert receipt["policy_response_status"] == "unproven"
    assert receipt["policy_request_artifacts"] == retained
    assert receipt["policy_queries"] == []


@pytest.mark.parametrize("delays", [
    {"metadata_delay": 0.35},
    {"handshake_delay": 0.35},
    {"handshake_delay": 0.08, "metadata_delay": 0.08},
])
def test_startup_deadline_is_bounded_without_attempting_inference(
    tmp_path, bounded_default_client, delays,
):
    with _server({}, **delays) as (port, requests):
        client = _client(tmp_path, port)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            client.preflight_readiness()
        assert time.monotonic() - started < 0.3
        assert requests == []
        assert client._client is None
        assert client.candidate_policy_queried is False
        assert len(bounded_default_client) == 1


def test_bounded_metadata_handshake_still_refuses_the_wrong_policy_before_send(
    tmp_path, bounded_default_client,
):
    spec = runtime.load_policy_spec(_cohort(tmp_path), policy_id="pi0_fast_droid_jointpos_polaris")
    metadata = {**_runtime_metadata(spec), "policy_id": "wrong-policy"}
    with _server(metadata) as (port, requests):
        client = _client(tmp_path, port)
        with pytest.raises(ValueError, match="policy_server_identity_mismatch:policy_id"):
            client.infer({"prompt": "fixture pick"})
        assert requests == []
        assert client._client is None
        assert len(bounded_default_client) == 1


def test_connection_refusal_is_one_attempt_and_preserves_original_failure(tmp_path, bounded_default_client, monkeypatch):
    calls = []
    failure = ConnectionRefusedError("offline refused")

    def refuse(*args, **kwargs):
        calls.append(kwargs)
        raise failure

    monkeypatch.setattr(websockets.sync.client, "connect", refuse)
    client = _client(tmp_path, 8000)
    with pytest.raises(ConnectionRefusedError) as caught:
        client.infer({"prompt": "fixture pick"})
    assert caught.value is failure
    assert len(calls) == 1
    assert client._client is None
    with pytest.raises(ValueError, match="openpi_policy_request_evidence_missing"):
        client.last_request_evidence()


def test_failure_before_query_is_explicitly_not_attempted(tmp_path):
    path = worker._write_episode_failure_gap(
        output_root=tmp_path, run_id="run", context={"cell_id": "cell", "candidate_id": "pi05_droid"},
        failure=RuntimeError("no input"), progress={},
    )
    evidence = json.loads(path.read_text())
    assert evidence["candidate_policy_query_attempted"] is False
    assert evidence["policy_response_status"] == "not_attempted"
    assert "policy_query_receipt" not in evidence["evidence_artifacts"]


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_unbounded_transport_deadlines_are_refused(seconds):
    with pytest.raises(ValueError, match="openpi_policy_transport_timeout_invalid"):
        runtime._bounded_openpi_client_type(_PinnedWireClient, msgpack.unpackb, inference_timeout_seconds=seconds)
