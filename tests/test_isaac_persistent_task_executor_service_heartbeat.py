from __future__ import annotations

import pytest

from blueprint_pipeline import isaac_persistent_task_executor_service as service


class _Server:
    timeout = None

    def __init__(self) -> None:
        self.requests = 0

    def handle_request(self) -> None:
        self.requests += 1


class _Backend:
    def __init__(self) -> None:
        self.refreshes = 0

    def refresh_live_state_snapshot(self) -> dict:
        self.refreshes += 1
        return {"heartbeat_sequence": self.refreshes}


def test_service_refreshes_live_isaac_state_from_serial_request_thread() -> None:
    server = _Server()
    backend = _Backend()
    service._serve_with_live_state_heartbeat(
        server=server,
        backend=backend,
        max_iterations=4,
    )
    assert server.timeout == 0.02
    assert server.requests == 4
    assert backend.refreshes == 4


def test_service_fails_closed_without_live_snapshot_refresh_method() -> None:
    with pytest.raises(
        RuntimeError, match="persistent_isaac_live_state_heartbeat_method_missing"
    ):
        service._serve_with_live_state_heartbeat(
            server=_Server(),
            backend=object(),
            max_iterations=1,
        )
