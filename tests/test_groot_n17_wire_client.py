from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from importlib import metadata as importlib_metadata
from pathlib import Path

import msgpack
import numpy as np
import pytest
import zmq

from blueprint_pipeline.groot_n17_wire_client import (
    SELF_CHECK_MARKER,
    UPSTREAM_SERVER_CLIENT_SHA256,
    WIRE_DEPENDENCY_VERSIONS,
    GrootN17WirePolicyClient,
    decode_wire_message,
    encode_wire_message,
    run_wire_codec_self_check,
)


def test_staged_wire_deps_sibling_directory_wins_sys_path(
    tmp_path, monkeypatch
) -> None:
    """The staged pinned codec must precede Isaac's bundled msgpack/pyzmq.

    Pins installed into the kit environment never win against Isaac's own
    newer copies (live 20260825T134736Z self-check refusal), so the module
    prepends its staged sibling directory before importing the codec.
    """

    import sys

    from blueprint_pipeline import groot_n17_wire_client as client

    module_file = tmp_path / "groot_n17_wire_client.py"
    module_file.write_text("# staged copy\n", encoding="utf-8")
    staged = tmp_path / client.STAGED_WIRE_DEPS_DIRNAME
    staged.mkdir()
    # Simulate the staged dir already present later in the path: the prepend
    # must deduplicate and still land it first.
    monkeypatch.setattr(
        sys, "path", ["/isaac/bundled", str(staged), *sys.path]
    )

    location = client._prepend_staged_wire_deps(str(module_file))

    assert location == str(staged)
    assert sys.path[0] == str(staged)
    assert sys.path.count(str(staged)) == 1


def test_fresh_import_resolves_staged_modules_and_metadata_over_conflicts(
    tmp_path,
) -> None:
    """Use wrapper numpy while staged protocol wheels beat Isaac conflicts."""

    source_root = Path(__file__).resolve().parents[1] / "src" / "blueprint_pipeline"
    runtime = tmp_path / "provider_runtime"
    staged = runtime / "groot_wire_deps"
    conflict = tmp_path / "isaac_bundled"
    staged.mkdir(parents=True)
    conflict.mkdir()
    (runtime / "groot_n17_wire_client.py").write_bytes(
        (source_root / "groot_n17_wire_client.py").read_bytes()
    )

    # Copy the installed, pinned protocol distributions exactly as uv's staged
    # target would. This includes pyzmq's native shared libraries and lets the
    # subprocess execute the complete strict codec self-check.
    for distribution in WIRE_DEPENDENCY_VERSIONS:
        installed = importlib_metadata.distribution(distribution)
        for relative in installed.files or ():
            if any(part.endswith(".dist-info") for part in Path(relative).parts):
                continue
            source = Path(installed.locate_file(relative))
            if not source.is_file():
                continue
            destination = staged / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        version = WIRE_DEPENDENCY_VERSIONS[distribution]
        metadata_root = staged / (
            f"{distribution.replace('-', '_')}-{version}.dist-info"
        )
        metadata_root.mkdir()
        (metadata_root / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n",
            encoding="utf-8",
        )

    def write_conflicting_distribution(
        *, import_name: str, distribution: str, version: str
    ) -> None:
        root = conflict
        (root / f"{import_name}.py").write_text(
            f"ORIGIN = {str(root)!r}\n", encoding="utf-8"
        )
        metadata_root = root / f"{distribution.replace('-', '_')}-{version}.dist-info"
        metadata_root.mkdir()
        (metadata_root / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n",
            encoding="utf-8",
        )

    write_conflicting_distribution(
        import_name="msgpack", distribution="msgpack", version="1.2.1"
    )
    write_conflicting_distribution(
        import_name="msgpack_numpy", distribution="msgpack-numpy", version="0.4.9"
    )
    write_conflicting_distribution(
        import_name="zmq", distribution="pyzmq", version="27.1.0"
    )

    code = """
import json
from importlib import metadata
import sys
import numpy as wrapper_numpy
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import groot_n17_wire_client as client
client.run_wire_codec_self_check(require_distribution_versions=True)
print(json.dumps({
    "msgpack_file": client.msgpack.__file__,
    "msgpack_numpy_file": client.mnp.__file__,
    "numpy_file": client.np.__file__,
    "numpy_preloaded_from_wrapper": client.np is wrapper_numpy,
    "zmq_file": client.zmq.__file__,
    "versions": {
        name: metadata.version(name)
        for name in ("msgpack", "msgpack-numpy", "pyzmq")
    },
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(conflict), str(runtime)],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed["versions"] == WIRE_DEPENDENCY_VERSIONS
    for field in ("msgpack_file", "msgpack_numpy_file", "zmq_file"):
        assert Path(observed[field]).is_relative_to(staged)
    assert observed["numpy_preloaded_from_wrapper"] is True
    assert not Path(observed["numpy_file"]).is_relative_to(staged)


def test_missing_staged_wire_deps_directory_is_a_no_op(tmp_path) -> None:
    from blueprint_pipeline import groot_n17_wire_client as client

    module_file = tmp_path / "groot_n17_wire_client.py"
    module_file.write_text("# staged copy\n", encoding="utf-8")

    assert client._prepend_staged_wire_deps(str(module_file)) is None


def test_wire_client_imports_without_gr00t_transformers_or_torch() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    code = r'''
import importlib.abc
import sys

class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", 1)[0] in {"gr00t", "transformers", "torch"}:
            raise AssertionError(f"forbidden_import:{fullname}")
        return None

sys.meta_path.insert(0, Guard())
from blueprint_pipeline.groot_n17_wire_client import run_wire_codec_self_check
run_wire_codec_self_check(require_distribution_versions=False)
print("wire-only-import-ok")
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(source_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "wire-only-import-ok"


def test_codec_matches_pinned_upstream_golden_control_frames() -> None:
    assert UPSTREAM_SERVER_CLIENT_SHA256 == (
        "d29d2e68e97ceae1762243315f23f22e696538da60fecd87d9a1391d20f7f1b9"
    )
    assert encode_wire_message({"endpoint": "ping"}).hex() == (
        "81a8656e64706f696e74a470696e67"
    )
    assert encode_wire_message(
        {"endpoint": "reset", "data": {"options": None}}
    ).hex() == (
        "82a8656e64706f696e74a57265736574a46461746181a76f7074696f6e73c0"
    )
    run_wire_codec_self_check(require_distribution_versions=False)


@pytest.mark.parametrize(
    "marker",
    [
        "__ModalityConfig__",
        b"__ModalityConfig__",
        "__ModalityConfig_class__",
        b"__ModalityConfig_class__",
    ],
)
@pytest.mark.parametrize(
    "payload",
    [
        {"modality_keys": ["joint_position"], "delta_indices": [0, 1]},
        '{"modality_keys":["joint_position"],"delta_indices":[0,1]}',
        b'{"modality_keys":["joint_position"],"delta_indices":[0,1]}',
    ],
)
def test_modality_config_decodes_without_nvidia_types(marker, payload) -> None:
    decoded = decode_wire_message(
        msgpack.packb({"config": {marker: True, "as_json": payload}})
    )
    assert decoded["config"] == {
        "modality_keys": ["joint_position"],
        "delta_indices": [0, 1],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"__ModalityConfig__": True},
        {"__ModalityConfig__": True, "as_json": []},
        {"__ModalityConfig__": True, "as_json": {"delta_indices": [0]}},
    ],
)
def test_malformed_modality_markers_fail_closed(payload) -> None:
    with pytest.raises(ValueError, match="groot_wire_modality"):
        decode_wire_message(msgpack.packb(payload))


def test_empty_delta_indices_match_pinned_upstream_modality_contract() -> None:
    decoded = decode_wire_message(
        msgpack.packb(
            {
                "__ModalityConfig__": True,
                "as_json": {
                    "modality_keys": ["joint_position"],
                    "delta_indices": [],
                },
            }
        )
    )
    assert decoded == {"modality_keys": ["joint_position"], "delta_indices": []}


def test_checked_in_frames_match_pinned_upstream_protocol() -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures/groot_n17_wire_protocol_b9955401.json"
        ).read_text(encoding="utf-8")
    )
    assert fixture["source_sha256"] == UPSTREAM_SERVER_CLIENT_SHA256
    assert fixture["dependency_versions"] == WIRE_DEPENDENCY_VERSIONS
    frames = fixture["fixtures"]

    request = {
        "endpoint": "get_action",
        "data": {
            "observation": {
                "state": np.asarray([[1.0, -2.0]], dtype=np.float32)
            },
            "options": None,
        },
    }
    assert encode_wire_message(request).hex() == frames["numeric_observation_request"]

    actions, info = decode_wire_message(bytes.fromhex(frames["action_response"]))
    assert np.array_equal(
        actions["joint_position"],
        np.arange(14, dtype=np.float32).reshape(1, 2, 7),
    )
    assert info == {"latency_s": 0.01}

    modality = decode_wire_message(
        bytes.fromhex(frames["modality_response_empty_delta_indices"])
    )
    assert modality["action"] == {
        "modality_keys": ["joint_position"],
        "delta_indices": [],
    }


def test_object_dtype_is_rejected_before_pickle_on_both_boundaries() -> None:
    with pytest.raises(TypeError, match="object_dtype_encode_forbidden"):
        encode_wire_message(np.asarray([object()], dtype=object))
    forged = msgpack.packb({b"nd": 1, b"kind": b"O", b"data": b"pickle"})
    with pytest.raises(ValueError, match="object_dtype_decode_forbidden"):
        decode_wire_message(forged)


class _FakeSocket:
    def __init__(self, *, response: bytes | BaseException) -> None:
        self.response = response
        self.sent: list[bytes] = []
        self.options: list[tuple[int, int]] = []
        self.connected: list[str] = []
        self.closed: list[int] = []

    def setsockopt(self, option: int, value: int) -> None:
        self.options.append((option, value))

    def connect(self, endpoint: str) -> None:
        self.connected.append(endpoint)

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv(self) -> bytes:
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    def close(self, *, linger: int) -> None:
        self.closed.append(linger)


class _FakeContext:
    def __init__(self, sockets: list[_FakeSocket]) -> None:
        self.sockets = sockets
        self.requested_types: list[int] = []
        self.terminated = 0

    def socket(self, socket_type: int) -> _FakeSocket:
        self.requested_types.append(socket_type)
        return self.sockets[len(self.requested_types) - 1]

    def term(self) -> None:
        self.terminated += 1


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "policy.example", "10.0.0.7", " localhost "]
)
def test_external_policy_endpoints_are_refused(host: str) -> None:
    with pytest.raises(ValueError, match="groot_wire_endpoint_invalid"):
        GrootN17WirePolicyClient(host=host)


def test_ipv6_loopback_is_bracketed_for_zmq() -> None:
    socket = _FakeSocket(response=encode_wire_message({"status": "ok"}))
    context = _FakeContext([socket])
    client = GrootN17WirePolicyClient(host="::1", context_factory=lambda: context)

    assert socket.connected == ["tcp://[::1]:5555"]
    client.close()


def test_timeout_replaces_socket_without_resending() -> None:
    first = _FakeSocket(response=zmq.Again())
    replacement = _FakeSocket(response=encode_wire_message({"status": "ok"}))
    context = _FakeContext([first, replacement])
    client = GrootN17WirePolicyClient(
        host="127.0.0.1",
        timeout_ms=1234,
        context_factory=lambda: context,
    )

    with pytest.raises(zmq.Again):
        client.call_endpoint("ping", requires_input=False)

    assert len(first.sent) == 1
    assert replacement.sent == []
    assert first.closed == [0]
    assert replacement.options == [(zmq.RCVTIMEO, 1234), (zmq.SNDTIMEO, 1234)]
    client.close()
    client.close()
    assert replacement.closed == [0]
    assert context.terminated == 1


def test_ping_times_out_once_and_returns_false() -> None:
    first = _FakeSocket(response=zmq.Again())
    replacement = _FakeSocket(response=encode_wire_message({"status": "ok"}))
    context = _FakeContext([first, replacement])
    client = GrootN17WirePolicyClient(context_factory=lambda: context)

    assert client.ping() is False
    assert len(first.sent) == 1
    assert replacement.sent == []
    client.close()


@pytest.mark.parametrize(
    "response",
    [b"ERROR", encode_wire_message({"error": "refused"})],
)
def test_error_frame_and_mapping_fail_closed(response: bytes) -> None:
    first = _FakeSocket(response=response)
    context = _FakeContext([first])
    client = GrootN17WirePolicyClient(context_factory=lambda: context)
    with pytest.raises(RuntimeError, match="groot_wire_server_error"):
        client.call_endpoint("ping", requires_input=False)
    client.close()


def _unused_tcp_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_loopback_server_exercises_exact_endpoint_envelopes() -> None:
    port = _unused_tcp_port()
    ready = threading.Event()
    received: list[dict] = []

    def server() -> None:
        context = zmq.Context()
        reply = context.socket(zmq.REP)
        reply.bind(f"tcp://127.0.0.1:{port}")
        ready.set()
        try:
            for _ in range(4):
                request = decode_wire_message(reply.recv())
                received.append(request)
                endpoint = request["endpoint"]
                if endpoint == "ping":
                    response = {"status": "ok"}
                elif endpoint == "get_modality_config":
                    response = {
                        "action": {
                            "__ModalityConfig__": True,
                            "as_json": {
                                "modality_keys": ["joint_position"],
                                "delta_indices": [0, 1],
                            },
                        }
                    }
                elif endpoint == "get_action":
                    response = [
                        {"joint_position": np.zeros((1, 2, 7), dtype=np.float32)},
                        {"latency_s": 0.01},
                    ]
                else:
                    response = {"status": "reset"}
                reply.send(encode_wire_message(response))
        finally:
            reply.close(linger=0)
            context.term()

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    assert ready.wait(timeout=2.0)
    with GrootN17WirePolicyClient(host="127.0.0.1", port=port) as client:
        assert client.ping() is True
        assert client.get_modality_config()["action"]["delta_indices"] == [0, 1]
        actions, info = client.get_action({"state": np.asarray([1.0])})
        assert actions["joint_position"].shape == (1, 2, 7)
        assert info == {"latency_s": 0.01}
        assert client.reset() == {"status": "reset"}
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert [request["endpoint"] for request in received] == [
        "ping",
        "get_modality_config",
        "get_action",
        "reset",
    ]
    assert set(received[0]) == {"endpoint"}
    assert set(received[1]) == {"endpoint"}
    assert received[2]["data"]["options"] is None
    assert np.array_equal(
        received[2]["data"]["observation"]["state"], np.asarray([1.0])
    )
    assert received[3] == {"endpoint": "reset", "data": {"options": None}}
    assert SELF_CHECK_MARKER == "GROOT_N17_WIRE_CLIENT_SELF_CHECK_OK"
