from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_paid_resource_allocator.py"
SPEC = importlib.util.spec_from_file_location("verify_paid_resource_allocator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def test_request_dict_runpod_create_is_discovered_and_unclassified() -> None:
    source = '''
RUNPOD_REST_API_BASE = "https://rest.runpod.io/v1"
request = {"url": f"{RUNPOD_REST_API_BASE}/pods", "method": "POST"}
'''
    assert verifier._direct_paid_mutation_signals(source) == {"runpod_pod_create"}
    assert verifier._unclassified_direct_mutators(
        {"src/blueprint_pipeline/new_bypass.py": source}, set()
    ) == {"src/blueprint_pipeline/new_bypass.py"}


def test_request_dict_runpod_create_is_accepted_only_when_manifested() -> None:
    path = "src/blueprint_pipeline/canonical_adapter.py"
    source = '''
RUNPOD_REST_API_BASE = "https://rest.runpod.io/v1"
request = {"url": f"{RUNPOD_REST_API_BASE}/pods", "method": "POST"}
'''
    assert verifier._unclassified_direct_mutators({path: source}, {path}) == set()


def test_model_volume_watchdog_handoff_is_machine_enforced() -> None:
    blockers = set(verifier.verify())
    assert "model_volume_watchdog_handoff_schema_missing" not in blockers
    assert "model_volume_watchdog_process_handoff_missing" not in blockers
    assert "model_volume_missing_key_terminal_evidence_missing" not in blockers
    assert "model_volume_ready_handoff_liveness_guard_missing" not in blockers
    assert "gpu_preflight_model_volume_watchdog_handoff_guard_missing" not in blockers
    assert "gpu_launch_refresh_drops_model_volume_watchdog_handoff" not in blockers
    assert "runbook_model_volume_watchdog_handoff_missing" not in blockers
    assert "remote_build_final_tag_promotion_guard_missing" not in blockers
    assert "remote_build_final_tag_promotion_order_invalid" not in blockers
    assert "remote_build_pushes_unvalidated_final_release_tag" not in blockers
