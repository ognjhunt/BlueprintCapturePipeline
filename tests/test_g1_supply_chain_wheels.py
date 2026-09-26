"""G1 provider pins remain visible to the supply-chain license gate."""

import json
from pathlib import Path

from scripts.build_supply_chain_evidence import reviewed_g1_runtime_wheel_keys


def test_pinned_g1_provider_wheels_have_exact_current_license_reviews() -> None:
    root = Path(__file__).resolve().parents[1]
    keys = reviewed_g1_runtime_wheel_keys(root)
    approvals = json.loads(
        (root / "docs/runtime_dependency_license_policy.json").read_text()
    )["components"]
    assert len(keys) == 16
    assert all(approvals.get(key, {}).get("approved") is True for key in keys)


def test_exact_g1_provider_pins_are_exempt_and_dynamic_pins_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    lock = root / "src/blueprint_pipeline/native_task_g1_runtime_lock.py"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        'G1_RUNTIME_DEPENDENCY_WHEELS = ({"package": "pin", "version": "4.1.0"},)\n'
    )
    assert reviewed_g1_runtime_wheel_keys(root) == frozenset({"pin==4.1.0"})

    lock.write_text('G1_RUNTIME_DEPENDENCY_WHEELS = load_wheels()\n')
    assert not reviewed_g1_runtime_wheel_keys(root)
    lock.write_text(
        'G1_RUNTIME_DEPENDENCY_WHEELS = ({"package": "pin", "version": "4.1.0"},)\n'
        'G1_RUNTIME_DEPENDENCY_WHEELS = ({"package": "other", "version": "1"},)\n'
    )
    assert not reviewed_g1_runtime_wheel_keys(root)
