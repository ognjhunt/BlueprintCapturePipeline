"""Real ZMQ loopback tests for the official GEAR-SONIC PUB/SUB roundtrip.

These tests exercise the real socket topology (executor binds the action PUB
endpoint and connects a SUB to the controller state endpoint), the slow-joiner
grace behaviour, token matching against stale replies, timeouts, and concurrent
attempts. They run against real pyzmq sockets over loopback; when pyzmq is not
installed in the venv they skip with a precise reason.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

try:
    import msgpack
except ImportError:  # pragma: no cover - msgpack is a repo dependency
    msgpack = None
try:
    import zmq
except ImportError:
    zmq = None

from blueprint_pipeline import gear_sonic_joint_order_contract as contract
from blueprint_pipeline import gear_sonic_official_zmq_executor as executor

pytestmark = pytest.mark.skipif(
    zmq is None or msgpack is None,
    reason="pyzmq_not_installed_in_venv",
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _pose_message(token: list[float]) -> bytes:
    return b"pose" + msgpack.packb({"token_state": token}, use_single_float=False)


def _command_message() -> bytes:
    return b"command" + msgpack.packb({"start": True, "stop": False, "planner": False})


def _state_payload(token: list[float]) -> dict:
    return {
        "token_state": token,
        "body_q_target": [0.1] * 29,
        "body_q_measured": [0.0] * 29,
        "base_quat_measured": [1.0, 0.0, 0.0, 0.0],
        "ros_timestamp": 123,
        "joint_order_schema_version": contract.JOINT_ORDER_SCHEMA_VERSION,
        "body_joint_names": list(contract.PROTOCOL_V4_BODY_JOINT_NAMES),
        "left_hand_joint_names": list(contract.PROTOCOL_V4_LEFT_HAND_JOINT_NAMES),
        "right_hand_joint_names": list(contract.PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES),
        "mapping_digest": contract.PROTOCOL_V4_MAPPING_DIGEST,
    }


class _FakeController(threading.Thread):
    """Loopback stand-in for the official controller process.

    Connects a SUB socket to the executor-bound action endpoint (proving the
    executor binds), binds a PUB socket for ``g1_debug`` states (proving the
    executor connects), and echoes each received pose token back, optionally
    prefixed with stale states carrying a foreign token.
    """

    def __init__(
        self,
        *,
        action_endpoint: str,
        state_port: int,
        join_delay_seconds: float = 0.0,
        stale_tokens: list[list[float]] | None = None,
        mutate_state=None,
    ) -> None:
        super().__init__(daemon=True)
        self.action_endpoint = action_endpoint
        self.state_port = state_port
        self.join_delay_seconds = join_delay_seconds
        self.stale_tokens = stale_tokens or []
        self.mutate_state = mutate_state
        self.received_poses: list[dict] = []
        self.stop_event = threading.Event()

    def run(self) -> None:  # pragma: no cover - thread body exercised via tests
        time.sleep(self.join_delay_seconds)
        context = zmq.Context()
        sub = context.socket(zmq.SUB)
        pub = context.socket(zmq.PUB)
        sub.setsockopt(zmq.LINGER, 0)
        pub.setsockopt(zmq.LINGER, 0)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.connect(self.action_endpoint)
        pub.bind(f"tcp://127.0.0.1:{self.state_port}")
        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)
        try:
            while not self.stop_event.is_set():
                events = dict(poller.poll(50))
                if sub not in events:
                    continue
                raw = sub.recv()
                if not raw.startswith(b"pose"):
                    continue
                pose = msgpack.unpackb(raw[len(b"pose") :], raw=False)
                self.received_poses.append(pose)
                for stale in self.stale_tokens:
                    pub.send(b"g1_debug" + msgpack.packb(_state_payload(stale)))
                state = _state_payload(list(pose["token_state"]))
                if self.mutate_state is not None:
                    state = self.mutate_state(state)
                pub.send(b"g1_debug" + msgpack.packb(state))
        finally:
            sub.close()
            pub.close()
            context.term()

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=5)


def _roundtrip(token: list[float], *, action_port: int, state_port: int, timeout: float = 10.0):
    return executor._zmq_pubsub_roundtrip(
        pose_message=_pose_message(token),
        command_message=_command_message(),
        motion_token=token,
        timeout_seconds=timeout,
        action_endpoint=f"tcp://127.0.0.1:{action_port}",
        state_endpoint=f"tcp://127.0.0.1:{state_port}",
        slow_joiner_grace_seconds=0.05,
    )


def test_roundtrip_binds_action_pub_and_connects_state_sub() -> None:
    action_port, state_port = _free_port(), _free_port()
    controller = _FakeController(
        action_endpoint=f"tcp://127.0.0.1:{action_port}", state_port=state_port
    )
    controller.start()
    try:
        token = [0.25] * 64
        state = _roundtrip(token, action_port=action_port, state_port=state_port)
        assert state["token_state"] == pytest.approx(token)
        assert state["body_q_target"] == [0.1] * 29
        # The controller received the pose through the executor-bound endpoint.
        assert controller.received_poses
        assert controller.received_poses[0]["token_state"] == pytest.approx(token)
    finally:
        controller.stop()


def test_roundtrip_survives_slow_joiner_controller() -> None:
    action_port, state_port = _free_port(), _free_port()
    controller = _FakeController(
        action_endpoint=f"tcp://127.0.0.1:{action_port}",
        state_port=state_port,
        join_delay_seconds=0.5,
    )
    controller.start()
    try:
        token = [0.5] * 64
        state = _roundtrip(token, action_port=action_port, state_port=state_port)
        assert state["token_state"] == pytest.approx(token)
    finally:
        controller.stop()


def test_roundtrip_ignores_stale_replies_with_foreign_tokens() -> None:
    action_port, state_port = _free_port(), _free_port()
    stale = [[9.0] * 64, [7.5] * 64]

    def mark(state: dict) -> dict:
        state["ros_timestamp"] = 999
        return state

    controller = _FakeController(
        action_endpoint=f"tcp://127.0.0.1:{action_port}",
        state_port=state_port,
        stale_tokens=stale,
        mutate_state=mark,
    )
    controller.start()
    try:
        token = [0.125] * 64
        state = _roundtrip(token, action_port=action_port, state_port=state_port)
        assert state["token_state"] == pytest.approx(token)
        assert state["ros_timestamp"] == 999
    finally:
        controller.stop()


@pytest.mark.slow
def test_roundtrip_times_out_without_matching_controller_state() -> None:
    action_port, state_port = _free_port(), _free_port()
    token = [0.75] * 64
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="official_gear_sonic_matching_controller_state_timeout"):
        _roundtrip(token, action_port=action_port, state_port=state_port, timeout=1.0)
    assert time.monotonic() - started < 30.0


def test_concurrent_attempts_each_match_their_own_token() -> None:
    state_port = _free_port()
    action_ports = [_free_port(), _free_port()]
    controllers = [
        _FakeController(
            action_endpoint=f"tcp://127.0.0.1:{port}",
            state_port=state_port if index == 0 else _free_port(),
        )
        for index, port in enumerate(action_ports)
    ]
    # Give the second controller its own state channel too; each attempt pairs
    # one action endpoint with one state endpoint but both run concurrently.
    for controller in controllers:
        controller.start()
    tokens = [[0.1] * 64, [0.9] * 64]
    results: dict[int, dict] = {}
    errors: list[Exception] = []

    def attempt(index: int) -> None:
        try:
            results[index] = _roundtrip(
                tokens[index],
                action_port=action_ports[index],
                state_port=controllers[index].state_port,
            )
        except Exception as error:  # pragma: no cover - failure path
            errors.append(error)

    threads = [threading.Thread(target=attempt, args=(index,)) for index in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert not errors
        assert results[0]["token_state"] == pytest.approx(tokens[0])
        assert results[1]["token_state"] == pytest.approx(tokens[1])
    finally:
        for controller in controllers:
            controller.stop()
