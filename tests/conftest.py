from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _find_contracts_src_dir() -> Path | None:
    for parent in REPO_ROOT.parents:
        candidate = parent / "BlueprintContracts" / "src"
        if candidate.is_dir():
            return candidate
    return None

contract_src_dir = _find_contracts_src_dir()

for candidate in (REPO_ROOT, SRC_DIR, contract_src_dir):
    if candidate is None:
        continue
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


@pytest.fixture(autouse=True)
def _isolated_pending_teardown_registry(tmp_path_factory, monkeypatch):
    """Paid lanes persist pending_teardown.v1 records; keep tests out of ~/."""
    registry = tmp_path_factory.mktemp("pending-teardowns")
    monkeypatch.setenv("BLUEPRINT_PENDING_TEARDOWN_DIR", str(registry))
    return registry


@pytest.fixture(autouse=True)
def _isolated_paid_provider_lane_lease_dir(tmp_path_factory, monkeypatch):
    """Paid lanes acquire an exclusive lane lease; keep tests out of ~/."""
    lease_dir = tmp_path_factory.mktemp("paid-lane-leases")
    monkeypatch.setenv("BLUEPRINT_PAID_PROVIDER_LANE_LEASE_DIR", str(lease_dir))
    return lease_dir
