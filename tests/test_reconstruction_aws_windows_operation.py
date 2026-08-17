"""Hostile coverage for the Windows trainer operation.

Every test here is about one of two things: not losing money, and not letting a
finished process pass for a good result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline.reconstruction_aws_windows_operation import (
    ReconstructionAwsWindowsError,
    run_reconstruction_aws_windows_operation,
)


class _Provider:
    """Fake EC2 provider with scriptable behaviour."""

    def __init__(
        self,
        *,
        states: list[str] | None = None,
        before_count: int = 0,
        after_count: int = 0,
        terminate_raises: bool = False,
        inventory_raises_after: bool = False,
        launch_instance_id: str | None = "i-0abc",
    ) -> None:
        self.states = states or ["running", "terminated"]
        self.before_count = before_count
        self.after_count = after_count
        self.terminate_raises = terminate_raises
        self.inventory_raises_after = inventory_raises_after
        self.launch_instance_id = launch_instance_id
        self.launched = 0
        self.terminated: list[str] = []
        self._inventory_calls = 0

    def billable_inventory(self, *, name_prefix: str) -> dict:
        self._inventory_calls += 1
        if self._inventory_calls == 1:
            return {"billable_instance_count": self.before_count}
        if self.inventory_raises_after:
            raise RuntimeError("describe failed")
        return {"billable_instance_count": self.after_count}

    def launch(self, job_dir, request, *, cold=False,
               paid_resource_admission_grant=None) -> dict:
        self.launched += 1
        # Mirror the real provider: no grant means no allocation.
        if paid_resource_admission_grant is None:
            return {"status": "blocked",
                    "blockers": ["legacy_gpu_render_provider_launch_disabled"],
                    "allocation_created": False}
        return {"instance_id": self.launch_instance_id, "allocation_created": True}

    def inspect(self, instance_id: str) -> dict:
        return {"state": self.states.pop(0) if self.states else "running"}

    def terminate(self, instance_id: str) -> dict:
        if self.terminate_raises:
            raise RuntimeError("terminate failed")
        self.terminated.append(instance_id)
        return {"instance_id": instance_id, "state": "shutting-down"}


ADMISSION = {"admission_digest": "sha256:" + "a" * 64, "retry_cap": 0}
REQUEST = {"bound_request_digest": "sha256:" + "b" * 64}
PREFLIGHT = {"verified": True}
GRANT = object()  # opaque: the executor forwards it, it does not inspect it


def _fetch_ok(_url: str, destination: Path):
    destination.write_bytes(b"bundle")
    return type("T", (), {"sha256": "sha256:" + "c" * 64})()


def _validate_ok(**_kwargs) -> dict:
    return {"output_bundle_receipt_digest": "sha256:" + "d" * 64}


def _run(tmp_path: Path, provider: _Provider, **overrides) -> dict:
    kwargs: dict[str, Any] = dict(
        bound_request=REQUEST,
        preflight=PREFLIGHT,
        job_dir=tmp_path,
        output_bundle_get_url="https://example.invalid/out.zip",
        provider=provider,
        allocator_admission=ADMISSION,
        paid_resource_admission_grant=GRANT,
        name_prefix="blueprint-postshot",
        hard_ttl_seconds=5400,
        output_fetcher=_fetch_ok,
        output_validator=_validate_ok,
        sleeper=lambda _s: None,
        clock=iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0] * 40).__next__,
    )
    kwargs.update(overrides)
    return run_reconstruction_aws_windows_operation(**kwargs)


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_successful_run_collects_output_then_tears_down(tmp_path: Path) -> None:
    provider = _Provider()
    result = _run(tmp_path, provider)
    assert result["status"] == "completed"
    assert result["output_retrieved_before_teardown"] is True
    assert provider.terminated == ["i-0abc"]
    assert result["teardown"]["confirmed"] is True
    assert result["blockers"] == []


def test_a_completed_run_is_never_a_quality_claim(tmp_path: Path) -> None:
    result = _run(tmp_path, _Provider())
    assert result["operation_scientific_success_inferred"] is False


# --------------------------------------------------------------------------
# Money safety
# --------------------------------------------------------------------------


def test_preexisting_billable_instance_aborts_before_spending(tmp_path: Path) -> None:
    provider = _Provider(before_count=1)
    result = _run(tmp_path, provider)
    assert result["status"] == "blocked"
    assert "aws_windows_provider_not_zero_before_allocation" in result["blockers"]
    assert provider.launched == 0
    assert result["provider_mutations_performed"] == 0


def test_teardown_still_runs_when_the_output_is_missing(tmp_path: Path) -> None:
    def exploding(_url, _destination):
        raise OSError("404")

    provider = _Provider()
    result = _run(tmp_path, provider, output_fetcher=exploding)
    assert result["status"] == "failed"
    assert provider.terminated == ["i-0abc"]
    assert any("output_unavailable" in b for b in result["blockers"])


def test_teardown_still_runs_when_the_worker_dies(tmp_path: Path) -> None:
    provider = _Provider(states=["terminated"])
    result = _run(tmp_path, provider)
    assert provider.terminated == ["i-0abc"]


def test_ttl_expiry_stops_polling_and_tears_down(tmp_path: Path) -> None:
    provider = _Provider(states=["running"] * 50)
    result = _run(tmp_path, provider, hard_ttl_seconds=2)
    assert "aws_windows_hard_ttl_exceeded" in result["blockers"]
    assert provider.terminated == ["i-0abc"]


def test_instance_still_billable_after_teardown_is_a_blocker(tmp_path: Path) -> None:
    """Asking it to stop is not the same as it having stopped."""
    provider = _Provider(after_count=1)
    result = _run(tmp_path, provider)
    assert result["status"] == "failed"
    assert "aws_windows_provider_not_zero_after_teardown" in result["blockers"]


def test_failed_termination_is_reported_not_swallowed(tmp_path: Path) -> None:
    provider = _Provider(terminate_raises=True)
    result = _run(tmp_path, provider)
    assert any("teardown_failed" in b for b in result["blockers"])
    assert result["teardown"]["confirmed"] is False


def test_unverifiable_inventory_is_not_treated_as_zero(tmp_path: Path) -> None:
    """A failed describe must never be recorded as proof nothing is running."""
    provider = _Provider(inventory_raises_after=True)
    result = _run(tmp_path, provider)
    assert any("provider_zero_after_unverifiable" in b for b in result["blockers"])
    assert result["status"] == "failed"


def test_an_unreadable_inventory_shape_is_not_treated_as_zero(tmp_path: Path) -> None:
    """A response we cannot parse is unknown, not empty.

    This is the quieter half of the same rule: the describe call *succeeded*,
    but returned nothing we recognise as a count. Reading that as zero would
    let a still-running instance pass as torn down.
    """

    class _Opaque(_Provider):
        def billable_inventory(self, *, name_prefix: str) -> dict:
            self._inventory_calls += 1
            if self._inventory_calls == 1:
                return {"billable_instance_count": 0}
            return {"unexpected_shape": "from a provider API change"}

    result = _run(tmp_path, _Opaque())
    assert any("provider_zero_after_unverifiable" in b for b in result["blockers"])
    assert result["status"] == "failed"


def test_a_pre_launch_inventory_we_cannot_read_still_blocks(tmp_path: Path) -> None:
    class _Opaque(_Provider):
        def billable_inventory(self, *, name_prefix: str) -> dict:
            return {"unexpected_shape": "from a provider API change"}

    result = _run(tmp_path, _Opaque())
    assert "aws_windows_provider_zero_before_unverifiable" in result["blockers"]


def test_launch_without_an_instance_id_still_reports_the_mutation(
    tmp_path: Path,
) -> None:
    """An ambiguous allocation may still be billing; it must not read as free."""
    provider = _Provider(launch_instance_id=None)
    with pytest.raises(ReconstructionAwsWindowsError):
        _run(tmp_path, provider)
    assert provider.launched == 1


# --------------------------------------------------------------------------
# Authority
# --------------------------------------------------------------------------


def test_missing_allocator_admission_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionAwsWindowsError) as excinfo:
        _run(tmp_path, _Provider(), allocator_admission={})
    assert "requires_allocator_admission" in str(excinfo.value)


def test_nonzero_retry_cap_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionAwsWindowsError) as excinfo:
        _run(
            tmp_path,
            _Provider(),
            allocator_admission={"admission_digest": "sha256:" + "a" * 64, "retry_cap": 1},
        )
    assert "retry_cap_must_be_zero" in str(excinfo.value)


def test_missing_preflight_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionAwsWindowsError) as excinfo:
        _run(tmp_path, _Provider(), preflight={})
    assert "preflight_missing" in str(excinfo.value)


def test_a_tampered_output_bundle_fails_the_run(tmp_path: Path) -> None:
    def rejecting(**_kwargs):
        raise ValueError("output_bundle_digest_mismatch")

    provider = _Provider()
    result = _run(tmp_path, provider, output_validator=rejecting)
    assert result["status"] == "failed"
    assert result["output_retrieved_before_teardown"] is False
    assert provider.terminated == ["i-0abc"]


def test_launch_without_a_paid_grant_is_blocked_not_silently_run(
    tmp_path: Path,
) -> None:
    """The provider fails closed without a grant; the executor must forward it.

    Omitting it is not a safety hole but it is a functional one: every launch
    would come back blocked and the lane could never run.
    """
    provider = _Provider()
    with pytest.raises(ReconstructionAwsWindowsError) as excinfo:
        _run(tmp_path, provider, paid_resource_admission_grant=None)
    assert "launch_blocked" in str(excinfo.value)
    assert provider.terminated == []  # nothing was allocated, nothing to tear down
