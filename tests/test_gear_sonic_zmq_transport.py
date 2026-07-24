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


def _planner_start_message() -> bytes:
    return b"command" + msgpack.packb({"start": True, "stop": False, "planner": True})


def _stream_message() -> bytes:
    return b"command" + msgpack.packb({"start": False, "stop": False, "planner": False})


def _state_payload(
    token: list[float],
    *,
    state_index: int = 0,
    left_hand: list[float] | None = None,
    right_hand: list[float] | None = None,
) -> dict:
    return {
        "token_state": token,
        "body_q_target": [0.1] * 29,
        "body_q_measured": [0.0] * 29,
        "base_quat_measured": [1.0, 0.0, 0.0, 0.0],
        "index": state_index,
        # Mirror the pinned controller's simulator behavior when ROS 2
        # wall-clock is unavailable.
        "ros_timestamp": 0.0,
        "last_left_hand_action": list(left_hand or [0.0] * 7),
        "last_right_hand_action": list(right_hand or [0.0] * 7),
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
        require_planner_start: bool = False,
    ) -> None:
        super().__init__(daemon=True)
        self.action_endpoint = action_endpoint
        self.state_port = state_port
        self.join_delay_seconds = join_delay_seconds
        self.stale_tokens = stale_tokens or []
        self.mutate_state = mutate_state
        self.require_planner_start = require_planner_start
        self.received_poses: list[dict] = []
        self.received_commands: list[dict] = []
        self.stop_event = threading.Event()
        self.state_index = 0

    def _state(
        self,
        token: list[float],
        *,
        left_hand: list[float] | None = None,
        right_hand: list[float] | None = None,
    ) -> dict:
        self.state_index += 1
        return _state_payload(
            token,
            state_index=self.state_index,
            left_hand=left_hand,
            right_hand=right_hand,
        )

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
                if raw.startswith(b"command"):
                    command = msgpack.unpackb(raw[len(b"command") :], raw=False)
                    self.received_commands.append(command)
                    if command.get("start") and command.get("planner"):
                        # The real pinned controller exposes g1_debug only
                        # after the PLANNER start command enters CONTROL.
                        pub.send(
                            b"g1_debug"
                            + msgpack.packb(self._state([0.0] * 64))
                        )
                    continue
                if not raw.startswith(b"pose"):
                    continue
                if self.require_planner_start and not any(
                    command.get("start") and command.get("planner")
                    for command in self.received_commands
                ):
                    continue
                if self.require_planner_start and not any(
                    not command.get("start") and not command.get("planner")
                    for command in self.received_commands
                ):
                    continue
                pose = msgpack.unpackb(raw[len(b"pose") :], raw=False)
                self.received_poses.append(pose)
                for stale in self.stale_tokens:
                    pub.send(b"g1_debug" + msgpack.packb(self._state(stale)))
                state = self._state(
                    list(pose["token_state"]),
                    left_hand=pose.get("left_hand_joints"),
                    right_hand=pose.get("right_hand_joints"),
                )
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


def _sequenced_roundtrip(
    token: list[float], *, action_port: int, state_port: int, timeout: float = 10.0
):
    return executor._zmq_pubsub_roundtrip(
        pose_message=_pose_message(token),
        planner_start_command_message=_planner_start_message(),
        stream_command_message=_stream_message(),
        motion_token=token,
        timeout_seconds=timeout,
        action_endpoint=f"tcp://127.0.0.1:{action_port}",
        state_endpoint=f"tcp://127.0.0.1:{state_port}",
        slow_joiner_grace_seconds=0.05,
    )


def _horizon_pose_message(
    token: list[float], left_hand: list[float], right_hand: list[float]
) -> bytes:
    return b"pose" + msgpack.packb(
        {
            "token_state": token,
            "left_hand_joints": left_hand,
            "right_hand_joints": right_hand,
        },
        use_single_float=False,
    )


def _horizon_roundtrip(
    tokens: list[list[float]],
    *,
    action_port: int,
    state_port: int,
    timeout: float = 10.0,
):
    left_hands = [[index / 10.0] * 7 for index in range(len(tokens))]
    right_hands = [[-index / 10.0] * 7 for index in range(len(tokens))]
    return executor._zmq_pubsub_horizon_roundtrip(
        pose_messages=[
            _horizon_pose_message(token, left_hands[index], right_hands[index])
            for index, token in enumerate(tokens)
        ],
        motion_tokens=tokens,
        left_hands=left_hands,
        right_hands=right_hands,
        frame_indices=[81 + index for index in range(len(tokens))],
        planner_start_command_message=_planner_start_message(),
        stream_command_message=_stream_message(),
        control_hz=50.0,
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


def test_roundtrip_enters_planner_control_before_streaming_pose() -> None:
    action_port, state_port = _free_port(), _free_port()
    controller = _FakeController(
        action_endpoint=f"tcp://127.0.0.1:{action_port}",
        state_port=state_port,
        require_planner_start=True,
    )
    controller.start()
    try:
        token = [0.375] * 64
        state = _sequenced_roundtrip(
            token, action_port=action_port, state_port=state_port
        )
        assert state["token_state"] == pytest.approx(token)
        planner_index = next(
            index
            for index, command in enumerate(controller.received_commands)
            if command["start"] and command["planner"]
        )
        stream_index = next(
            index
            for index, command in enumerate(controller.received_commands)
            if not command["start"] and not command["planner"]
        )
        assert planner_index < stream_index
        assert controller.received_poses
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


def test_horizon_uses_monotonic_index_with_fixed_sim_timestamp() -> None:
    action_port, state_port = _free_port(), _free_port()
    controller = _FakeController(
        action_endpoint=f"tcp://127.0.0.1:{action_port}",
        state_port=state_port,
        require_planner_start=True,
    )
    controller.start()
    tokens = [[0.1] * 64, [0.2] * 64, [0.3] * 64]
    try:
        states = _horizon_roundtrip(
            tokens,
            action_port=action_port,
            state_port=state_port,
        )
        assert all(
            state["token_state"] == pytest.approx(tokens[index])
            for index, state in enumerate(states)
        )
        assert [state["ros_timestamp"] for state in states] == [0.0, 0.0, 0.0]
        assert [state["index"] for state in states] == sorted(
            state["index"] for state in states
        )
        assert all(
            state["_blueprint_controller_frame_match_mode"]
            == "strict_monotonic_state_index_unique_action_without_reported_frame"
            for state in states
        )
    finally:
        controller.stop()
