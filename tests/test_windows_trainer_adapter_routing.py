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


def test_allocator_routes_a_windows_trainer_to_its_own_executor() -> None:
    """It must reach the Windows executor, never the Linux Vast operation."""
    import inspect

    from blueprint_pipeline import paid_resource_allocator

    source = inspect.getsource(paid_resource_allocator)
    # Slice exactly the Windows branch: from its guard to the next elif.
    after = source.split("WINDOWS_TRAINER_ADAPTER_IDS:", 1)[1]
    branch = after.split("elif operation in", 1)[0]
    assert "run_reconstruction_aws_windows_operation(" in branch
    assert "run_reconstruction_vast_operation(" not in branch
    # The paid grant must reach the provider or every launch comes back blocked.
    assert "paid_resource_admission_grant=grant" in branch
    # The Windows branch must sit ahead of the Vast branch, not after it.
    assert source.index("WINDOWS_TRAINER_ADAPTER_IDS:") < source.index(
        "run_reconstruction_vast_operation("
    )
