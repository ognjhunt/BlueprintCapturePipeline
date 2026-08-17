"""A Windows-only trainer must never be routed onto a Linux container.

Every generic reconstruction adapter is a Vast Linux container. Postshot ships
only postshot-cli.exe. Silently falling back would allocate a paid GPU and then
fail on a binary that cannot run there, so the fallback has to be refused.
"""

from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.reconstruction_gpu_admission import (
    CANONICAL_POSTSHOT_AWS_WINDOWS_ADAPTER_ID,
    CANONICAL_SPLATFACTO_VAST_ADAPTER_ID,
    EXECUTION_ADAPTER_IDS,
    GENERIC_VAST_OPERATION_ADAPTER_ID,
    WINDOWS_TRAINER_ADAPTER_IDS,
    select_reconstruction_execution_adapter_id,
)


def _request(tmp_path: Path, adapter_id: str | None) -> Path:
    path = tmp_path / "request.json"
    body: dict = {"schema_version": "reconstruction_launch_request.v1"}
    if adapter_id is not None:
        body["requested_execution_adapter_id"] = adapter_id
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_postshot_windows_adapter_is_a_known_execution_adapter() -> None:
    assert CANONICAL_POSTSHOT_AWS_WINDOWS_ADAPTER_ID in EXECUTION_ADAPTER_IDS
    assert CANONICAL_POSTSHOT_AWS_WINDOWS_ADAPTER_ID in WINDOWS_TRAINER_ADAPTER_IDS


def test_a_windows_request_is_not_rewritten_to_the_linux_adapter(
    tmp_path: Path,
) -> None:
    selected = select_reconstruction_execution_adapter_id(
        _request(tmp_path, CANONICAL_POSTSHOT_AWS_WINDOWS_ADAPTER_ID), execute=True
    )
    assert selected == CANONICAL_POSTSHOT_AWS_WINDOWS_ADAPTER_ID
    assert selected != GENERIC_VAST_OPERATION_ADAPTER_ID


def test_splatfacto_routing_is_unchanged(tmp_path: Path) -> None:
    assert (
        select_reconstruction_execution_adapter_id(
            _request(tmp_path, CANONICAL_SPLATFACTO_VAST_ADAPTER_ID), execute=True
        )
        == CANONICAL_SPLATFACTO_VAST_ADAPTER_ID
    )


def test_unnamed_adapter_still_defaults_to_the_generic_linux_operation(
    tmp_path: Path,
) -> None:
    assert (
        select_reconstruction_execution_adapter_id(_request(tmp_path, None), execute=True)
        == GENERIC_VAST_OPERATION_ADAPTER_ID
    )


def test_selection_grants_nothing_without_execute(tmp_path: Path) -> None:
    assert (
        select_reconstruction_execution_adapter_id(
            _request(tmp_path, CANONICAL_POSTSHOT_AWS_WINDOWS_ADAPTER_ID), execute=False
        )
        is None
    )


def test_allocator_blocks_a_windows_trainer_instead_of_running_it_on_linux() -> None:
    """The executor is not qualified yet; blocking must cost nothing."""
    import inspect

    from blueprint_pipeline import paid_resource_allocator

    source = inspect.getsource(paid_resource_allocator)
    guard = source.split("WINDOWS_TRAINER_ADAPTER_IDS:", 1)[1][:900]
    assert "windows_trainer_executor_not_qualified" in guard
    assert '"status": "blocked"' in guard
    assert '"provider_mutations_performed": 0' in guard
    assert '"cost_usd": 0.0' in guard
    # The guard must sit ahead of the Vast branch, not after it.
    assert source.index("WINDOWS_TRAINER_ADAPTER_IDS:") < source.index(
        "run_reconstruction_vast_operation("
    )
