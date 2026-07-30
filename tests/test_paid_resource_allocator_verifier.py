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


def test_s3_write_or_delete_is_discovered_and_unclassified() -> None:
    source = "client.upload_file(path, bucket, key)\nclient.delete_object(Bucket=bucket, Key=key)"
    assert verifier._direct_paid_mutation_signals(source) == {
        "s3_object_write_or_delete"
    }
    assert verifier._unclassified_direct_mutators(
        {"src/blueprint_pipeline/new_s3_bypass.py": source}, set()
    ) == {"src/blueprint_pipeline/new_s3_bypass.py"}


def test_provider_hostname_signals_use_exact_regex_matching() -> None:
    source = '''
request = {"url": "https://api.runpod.io/v2/pods", "method": "POST"}
'''
    assert verifier._direct_paid_mutation_signals(source) == {"runpod_pod_create"}


def test_unmanifested_script_mutator_is_rejected() -> None:
    path = "scripts/new_paid_bypass.py"
    source = 'client.upload_file("source", "bucket", "key")'
    assert verifier._unclassified_direct_mutators({path: source}, set()) == {path}


def test_third_s3_capability_or_transport_caller_is_rejected() -> None:
    sources = {
        "src/blueprint_pipeline/groot_oscar_runpod_s3_model_cache.py": """
def upload_and_verify_model_cache():
    _issue_transport_execution_capability()
    _upload_and_verify_model_cache_impl()
""",
        "src/blueprint_pipeline/groot_oscar_model_cache_s3_remote_executor.py": """
def execute_remote_packet():
    _issue_transport_execution_capability()
    _upload_and_verify_model_cache_impl()
""",
        "scripts/new_bypass.py": """
from blueprint_pipeline.groot_oscar_runpod_s3_model_cache import (
    _issue_transport_execution_capability as mint,
    _upload_and_verify_model_cache_impl as mutate,
)
def bypass():
    mint()
    mutate()
""",
    }
    assert verifier._s3_transport_capability_callers(sources) != (
        verifier.APPROVED_S3_TRANSPORT_CAPABILITY_CALLERS
    )


def test_production_scan_recurses_through_source_and_scripts(tmp_path: Path) -> None:
    source = tmp_path / "src/blueprint_pipeline/nested/adapter.py"
    script = tmp_path / "scripts/nested/executor.py"
    source.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    script.write_text("", encoding="utf-8")
    assert verifier._production_python_paths(tmp_path) == [script, source]


def test_operator_docs_reject_legacy_paid_commands_and_allow_canonical() -> None:
    forbidden = """
blueprint-run-runpod-provider-adapter --request x --mode on-demand-pod
"""
    assert verifier._forbidden_operator_doc_commands(forbidden) == {
        "legacy_runpod_adapter_paid_mode"
    }
    canonical = "python -m blueprint_pipeline.paid_resource_allocator gpu-canary --execute"
    assert verifier._forbidden_operator_doc_commands(canonical) == set()


def test_model_volume_watchdog_handoff_is_machine_enforced() -> None:
    blockers = set(verifier.verify())
    assert "paid_resource_admission_issuer_set_mismatch" not in blockers
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
    assert "remote_build_pushes_unvalidated_final_foundation_tag" not in blockers
    assert "lambda_termination_shared_admission_guard_missing" not in blockers
