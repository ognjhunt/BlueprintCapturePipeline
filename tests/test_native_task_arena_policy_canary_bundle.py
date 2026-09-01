from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import native_task_arena_policy_canary_bundle as bundle
from blueprint_pipeline import native_task_arena_policy_canary_worker as worker
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import policy_canary_allocator_lane as allocator_lane


def _spec(candidate: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "native_task_arena_policy_canary_execution_spec.v1",
        "candidate_id": candidate,
        "execution_authority": "internal_policy_canary_unqualified",
        "claim_ceiling": "diagnostic_policy_execution",
        "ranking_permitted": False,
        "qualification_permitted": False,
        "scene_promotion_permitted": False,
        "policy_endpoint": {"host": "127.0.0.1", "port": 8000},
        "policy_spec": {"candidate_id": candidate},
        "candidate_rights_binding": {"status": "admitted"},
        "checkpoint_digest": "sha256:" + "1" * 64,
        "runtime_identity_digest": "sha256:" + "2" * 64,
        "prompt": "Move the object",
        "max_policy_queries": 10,
        "open_loop_horizon": 8,
        "execution_spec_digest": "",
    }
    value["execution_spec_digest"] = canonical_digest(
        value, digest_field="execution_spec_digest"
    )
    return value


@pytest.mark.parametrize("candidate", ["pi05_droid", "groot_n17_droid"])
def test_canary_execution_spec_preserves_unqualified_boundary(candidate: str) -> None:
    assert bundle._validate_spec(_spec(candidate), candidate=candidate)[
        "ranking_permitted"
    ] is False


def test_canary_execution_spec_rejects_ranking_permission() -> None:
    spec = _spec("pi05_droid")
    spec["ranking_permitted"] = True
    spec["execution_spec_digest"] = canonical_digest(
        spec, digest_field="execution_spec_digest"
    )

    with pytest.raises(ValueError, match="policy_canary_execution_spec_invalid"):
        bundle._validate_spec(spec, candidate="pi05_droid")


def test_bundle_cli_exposes_every_immutable_input(monkeypatch, capsys) -> None:
    observed = {}

    def build(**kwargs):
        observed.update(kwargs)
        return {"status": "ready", "bundle_sha256": "sha256:" + "f" * 64}

    monkeypatch.setattr(bundle, "build_policy_canary_session_bundle", build)
    args = [
        "--job-dir",
        "job",
        "--packet-dir",
        "packet",
        "--runtime-source-packet-receipt",
        "runtime.json",
        "--runtime-input-manifest-path",
        "inputs.json",
        "--session-authority-path",
        "authority.json",
        "--pi05-execution-spec-path",
        "pi05.json",
        "--groot-execution-spec-path",
        "groot.json",
        "--pi05-checkpoint-inventory-path",
        "inventory.json",
        "--implementation-commit",
        "a" * 40,
    ]

    assert bundle.main(args) == 0

    assert observed["implementation_commit"] == "a" * 40
    assert observed["generated_at"] is None
    assert observed["runtime_input_manifest_path"] == "inputs.json"
    assert '"status": "ready"' in capsys.readouterr().out


def test_provider_entrypoint_provisions_both_servers_once_and_runs_one_worker() -> None:
    script = bundle._entrypoint()

    assert script.count("adp009d_policy_provisioning.pi05_droid.sh") == 1
    assert script.count("adp009d_policy_provisioning.groot_n17_droid.sh") == 1
    assert script.count('"$RUNTIME_DIR/adp_arena_provider_runner.py"') == 1
    assert "for candidate in pi05_droid groot_n17_droid" in script
    assert "native_task_arena_policy_canary_session_result.v1.json" in script
    pi_copy = script.index("policy_execution_spec.pi05_droid.json")
    pi_provision = script.index("adp009d_policy_provisioning.pi05_droid.sh")
    groot_copy = script.index("policy_execution_spec.groot_n17_droid.json")
    groot_provision = script.index("adp009d_policy_provisioning.groot_n17_droid.sh")
    worker = script.index('"$RUNTIME_DIR/adp_arena_provider_runner.py"')
    assert pi_copy < pi_provision < groot_copy < groot_provision < worker


def test_candidate_servers_have_distinct_transports_ports_and_receipts() -> None:
    from blueprint_pipeline.adp009d_policy_server_worker import (
        CANDIDATE_DEFAULT_PORTS,
        transport_for,
    )

    assert transport_for("pi05_droid") == "openpi_websocket"
    assert transport_for("groot_n17_droid") == "groot_zmq"
    assert CANDIDATE_DEFAULT_PORTS[transport_for("pi05_droid")] == 8000
    assert CANDIDATE_DEFAULT_PORTS[transport_for("groot_n17_droid")] == 5555
    script = bundle._entrypoint()
    assert "adp009d_policy_server_receipt.$candidate.json" in script


def test_provider_worker_has_one_simulation_launch_outside_episode_loop() -> None:
    source = (
        Path(bundle.__file__).with_name("native_task_arena_policy_canary_worker.py")
    ).read_text(encoding="utf-8")

    assert source.count("launch_native_task_isaaclab(") == 1
    assert source.count("run_policy_episode(") == 1
    assert "execute_paired_session(" in source
    assert "provider_closeout_pending=True" in source
    assert "policy_canary_telemetry.jsonl" in source
    assert "from mcap.writer import Writer" in source
    assert "mcap_unavailable:" in source


def test_provider_result_is_durable_before_isaac_close_can_exit(
    tmp_path: Path,
) -> None:
    class ExitOnClose:
        def close(self) -> None:
            raise SystemExit(0)

    result = {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "status": "runtime_completed_unqualified_pending_closeout",
        "episodes": [],
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    result_path = tmp_path / "provider-result.json"

    with pytest.raises(SystemExit) as exited:
        worker._seal_result_before_simulation_close(
            result_path=result_path,
            result=result,
            simulation_app=ExitOnClose(),
        )

    assert exited.value.code == 0
    sealed = json.loads(result_path.read_text(encoding="utf-8"))
    assert sealed == result
    assert sealed["result_digest"] == canonical_digest(
        sealed, digest_field="result_digest"
    )


def test_episode_failure_gap_retains_safe_diagnostic_without_host_path(
    tmp_path: Path,
) -> None:
    path = worker._write_episode_failure_gap(
        output_root=tmp_path,
        run_id="run-1",
        context={"candidate_id": "pi05_droid", "cell_id": "anchor-1", "seed": 7},
        failure=RuntimeError("camera failed at /workspace/private/runtime.py"),
    )

    gap = json.loads(path.read_text(encoding="utf-8"))
    assert gap["failure_type"] == "RuntimeError"
    assert gap["failure_message"] == "camera failed at <path>"
    assert "/workspace" not in json.dumps(gap)
    assert gap["gap_digest"] == canonical_digest(gap, digest_field="gap_digest")


def test_provider_canary_package_imports_from_its_shipped_module_closure(
    tmp_path: Path,
) -> None:
    package = Path(bundle.__file__).resolve().parent
    staged = tmp_path / "blueprint_pipeline"
    staged.mkdir()
    (staged / "__init__.py").write_text("", encoding="utf-8")
    module_names = {
        *bundle.POLICY_RUNTIME_MODULE_NAMES,
        "native_task_arena_policy_worker.py",
        "native_task_arena_policy_canary_session.py",
        "native_task_arena_policy_canary_worker.py",
    }
    for name in module_names:
        shutil.copy2(package / name, staged / name)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import blueprint_pipeline.native_task_arena_policy_canary_session; "
                "import blueprint_pipeline.native_task_arena_policy_canary_worker"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_paid_allocator_routes_canary_only_through_one_session_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority_path = tmp_path / "authority.json"
    receipt_path = tmp_path / "bundle.json"
    authority_path.write_text("{}\n", encoding="utf-8")
    receipt_path.write_text("{}\n", encoding="utf-8")
    authority = {
        "hard_cap_usd": 4.0,
        "hard_ttl_seconds": 14_400,
        "authority_digest": "sha256:" + "a" * 64,
    }
    receipt = {
        "bundle_sha256": "sha256:" + "b" * 64,
        "runtime_inputs_digest": "sha256:" + "c" * 64,
    }
    observed = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "d" * 40}),
    )
    monkeypatch.setattr(
        allocator_lane, "validate_session_authority", lambda _value: authority
    )
    monkeypatch.setattr(
        allocator_lane,
        "validate_provider_bundle",
        lambda _value, **_kwargs: receipt,
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "dry_run_ready", "provider_mutations_performed": 0}

    monkeypatch.setattr(
        allocator_lane, "run_native_task_arena_policy_canary_session_vast", fake_run
    )
    exit_code = allocator.main(
        [
            "gpu-canary",
            "--provider",
            "vast",
            "--probe-kind",
            allocator_lane.PROBE_KIND,
            "--admission-out",
            str(tmp_path / "admission.json"),
            "--adapter-output",
            str(tmp_path / "adapter.json"),
            "--adp-job-dir",
            str(tmp_path / "job"),
            "--adp-max-spend-usd",
            "4.0",
            "--adp-hard-ttl-seconds",
            "14400",
            "--native-task-arena-policy-canary-session-authority",
            str(authority_path),
            "--native-task-arena-policy-canary-session-bundle-receipt",
            str(receipt_path),
        ]
    )

    assert exit_code == 0
    assert observed["prepared_bundle"] == receipt
    assert observed["session_authority"] == authority
    assert observed["execute"] is False
