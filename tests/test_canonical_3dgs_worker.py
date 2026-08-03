from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import struct

import pytest

from blueprint_pipeline.canonical_3dgs_admission import (
    Canonical3DGSAdmissionError,
    build_canonical_3dgs_worker_admission,
    require_canonical_3dgs_worker_admission,
)
from blueprint_pipeline.canonical_3dgs_cli import COMMANDS, main as canonical_cli_main
from blueprint_pipeline.canonical_3dgs_pipeline import (
    POSTSHOT_METHOD,
    SPLATFACTO_METHOD,
    Canonical3DGSPipelineError,
)
from blueprint_pipeline.canonical_3dgs_worker import (
    run_postshot_arm,
    run_splatfacto_arm,
    validate_trainer_runtime_binding,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _write_standard_splat(path: Path) -> None:
    properties = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        + "".join(f"property float {name}\n" for name in properties)
        + "end_header\n"
    )
    path.write_bytes(
        header.encode("ascii")
        + struct.pack("<14f", 0, 0, 1, 0, 0, 0, 1, -3, -3, -3, 1, 0, 0, 0)
    )


def _transport_receipt() -> dict:
    value = {
        "schema_version": "canonical_3dgs_transport_bundle.v1",
        "status": "compiled",
        "transport_bundle_digest": "sha256:" + "1" * 64,
        "transport_manifest_digest": "sha256:" + "2" * 64,
        "canonical_3dgs_execution_plan_digest": "sha256:" + "3" * 64,
        "worker_python_package_digest": "sha256:" + "8" * 64,
        "colmap_training_dataset_digest": "sha256:" + "4" * 64,
        "source_capture_digest": "sha256:" + "5" * 64,
        "frozen_split_digest": "sha256:" + "6" * 64,
        "source_commit_sha": "a" * 40,
        "dataset_members": [
            {
                "relative_path": "images/frame.png",
                "archive_path": "campaign/dataset/images/frame.png",
                "digest": "sha256:" + "7" * 64,
                "bytes": 1,
            }
        ],
        "dataset_member_count": 1,
        "hidden_heldout_pixels_included": False,
        "raw_secret_values_included": False,
        "provider_allocation_performed": False,
        "paid_execution_authorized_by_bundle": False,
        "proof_effect": "none",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def _allocator_admission(
    transport: dict,
    *,
    worker_image: str,
    authority: str = "operator-approved-run-1",
    arm_id: str = "postshot-primary",
) -> dict:
    value = {
        "schema_version": "reconstruction_gpu_canary_admission.v1",
        "status": "execute_ready",
        "blockers": [],
        "operation": "trainer_canary",
        "operation_request_digest": transport["canonical_3dgs_execution_plan_digest"],
        "operation_input_bundle_digest": transport["transport_bundle_digest"],
        "reconstruction_dataset_digest": transport["colmap_training_dataset_digest"],
        "frozen_split_digest": transport["frozen_split_digest"],
        "source_commit_sha": transport["source_commit_sha"],
        "worker_image_digest": worker_image,
        "max_spend_usd": 15.0,
        "hard_ttl_seconds": 7200,
        "retry_cap": 0,
        "authority_id": authority,
        "watchdog_armed": True,
        "provider_zero_verified": True,
        "provider_mutations_performed": 0,
        "paid_execution_started": False,
        "execution_adapter_qualified": True,
        "execution_adapter_id": (
            "canonical_postshot_windows_v1"
            if arm_id == "postshot-primary"
            else "reconstruction_vast_operation_v1"
        ),
        "worker_platform": "windows" if arm_id == "postshot-primary" else "linux",
    }
    value["admission_digest"] = canonical_digest(value, digest_field="admission_digest")
    return value


def test_postshot_worker_runs_full_resolution_and_never_returns_secrets(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "output"
    seen: list[list[str]] = []

    def execute(arguments: list[str], cwd: Path, log: Path) -> int:
        seen.append(list(arguments))
        log.write_text(
            "login operator@example.invalid password=super-secret "
            "https://worker.invalid/presigned?token=value\n",
            encoding="utf-8",
        )
        Path(arguments[arguments.index("--output") + 1]).write_bytes(b"project")
        _write_standard_splat(Path(arguments[arguments.index("--export-splat") + 1]))
        return 0

    receipt = run_postshot_arm(
        {
            "arm_id": "postshot-primary",
            "method_id": POSTSHOT_METHOD,
        },
        dataset,
        output,
        environment={
            "POSTSHOT_LOGIN_EMAIL": "operator@example.invalid",
            "POSTSHOT_LOGIN_PASSWORD": "super-secret",
            "POSTSHOT_CLI_PATH": "postshot-cli.exe",
        },
        executor=execute,
    )

    assert receipt["exit_code"] == 0
    assert set(row["kind"] for row in receipt["artifacts"]) == {
        "standard_3dgs_ply",
        "postshot_project",
        "training_log",
    }
    rendered = str(receipt)
    assert "operator@example.invalid" not in rendered
    assert "super-secret" not in rendered
    sanitized_log = (output / "training.log").read_text(encoding="utf-8")
    assert "operator@example.invalid" not in sanitized_log
    assert "super-secret" not in sanitized_log
    assert "worker.invalid" not in sanitized_log
    assert "[REDACTED" in sanitized_log
    assert seen[0].index("--login") < seen[0].index("train")
    assert seen[0][seen[0].index("--max-image-size") + 1] == "0"
    assert "--no-recenter-points" in seen[0]


def test_postshot_runtime_binding_hashes_actual_executable_bytes(tmp_path: Path) -> None:
    executable = tmp_path / "postshot-cli.exe"
    executable.write_bytes(b"exact-postshot-cli")
    digest = "sha256:" + hashlib.sha256(executable.read_bytes()).hexdigest()
    admission = {
        "trainer_runtime_digest": digest,
        "trainer_runtime_version": "Postshot fixture 1.0",
    }

    binding = validate_trainer_runtime_binding(
        "postshot-primary",
        admission,
        {"POSTSHOT_CLI_PATH": str(executable)},
    )
    assert binding["trainer_runtime_digest"] == digest

    executable.write_bytes(b"drifted-postshot-cli")
    with pytest.raises(Canonical3DGSPipelineError, match="trainer_runtime_digest_mismatch"):
        validate_trainer_runtime_binding(
            "postshot-primary",
            admission,
            {"POSTSHOT_CLI_PATH": str(executable)},
        )


def test_postshot_worker_fails_closed_without_runtime_credentials(tmp_path: Path) -> None:
    with pytest.raises(Canonical3DGSPipelineError, match="credentials_missing"):
        run_postshot_arm(
            {"arm_id": "postshot-primary", "method_id": POSTSHOT_METHOD},
            tmp_path / "dataset",
            tmp_path / "output",
            environment={},
        )


def test_splatfacto_worker_trains_then_exports_standard_ply(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "output"
    seen: list[list[str]] = []

    def execute(arguments: list[str], cwd: Path, log: Path) -> int:
        seen.append(list(arguments))
        with log.open("a", encoding="utf-8") as stream:
            stream.write("command complete\n")
        if arguments[0] == "ns-train-test":
            config = cwd / "outputs/capture/splatfacto/run/config.yml"
            config.parent.mkdir(parents=True)
            config.write_text("method: splatfacto\n", encoding="utf-8")
        else:
            export = Path(arguments[arguments.index("--output-dir") + 1])
            export.mkdir(parents=True, exist_ok=True)
            _write_standard_splat(export / "splat.ply")
        return 0

    receipt = run_splatfacto_arm(
        {
            "arm_id": "splatfacto-comparison",
            "method_id": SPLATFACTO_METHOD,
        },
        dataset,
        output,
        environment={
            "NS_TRAIN_PATH": "ns-train-test",
            "NS_EXPORT_PATH": "ns-export-test",
        },
        executor=execute,
        runtime_versions={"nerfstudio": "1.1.5", "gsplat": "1.4.0"},
    )

    assert receipt["exit_code"] == 0
    assert [command[0] for command in seen] == ["ns-train-test", "ns-export-test"]
    assert seen[0][1:4] == ["splatfacto", "--vis", "tensorboard"]
    assert "colmap" in seen[0]
    assert seen[0][seen[0].index("--colmap-path") + 1] == "sparse/0"
    assert seen[0][seen[0].index("--orientation-method") + 1] == "none"
    assert seen[0][seen[0].index("--auto-scale-poses") + 1] == "False"
    assert seen[0][seen[0].index("--eval-mode") + 1] == "all"
    assert seen[0][seen[0].index("--pipeline.model.stop_split_at") + 1] == "15000"
    assert not any("continue_cull_post_densification" in value for value in seen[0])
    assert set(row["kind"] for row in receipt["artifacts"]) == {
        "standard_3dgs_ply",
        "nerfstudio_config",
        "training_log",
    }


def test_splatfacto_worker_fails_when_training_produces_no_unique_config(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    def execute(arguments: list[str], cwd: Path, log: Path) -> int:
        log.write_text("training returned success without config\n", encoding="utf-8")
        return 0

    receipt = run_splatfacto_arm(
        {
            "arm_id": "splatfacto-comparison",
            "method_id": SPLATFACTO_METHOD,
        },
        dataset,
        tmp_path / "output",
        environment={"NS_TRAIN_PATH": "ns-train-test"},
        executor=execute,
        runtime_versions={"nerfstudio": "1.1.5", "gsplat": "1.4.0"},
    )

    assert receipt["exit_code"] == 70
    assert [row["kind"] for row in receipt["artifacts"]] == ["training_log"]


def test_splatfacto_worker_refuses_package_pin_drift_before_execution(tmp_path: Path) -> None:
    called = False

    def execute(arguments: list[str], cwd: Path, log: Path) -> int:
        nonlocal called
        called = True
        return 0

    with pytest.raises(Canonical3DGSPipelineError, match="runtime_version_mismatch"):
        run_splatfacto_arm(
            {
                "arm_id": "splatfacto-comparison",
                "method_id": SPLATFACTO_METHOD,
            },
            tmp_path / "dataset",
            tmp_path / "output",
            executor=execute,
            runtime_versions={"nerfstudio": "1.1.6", "gsplat": "1.4.0"},
        )
    assert called is False


def test_splatfacto_setup_installs_and_smokes_blueprint_worker_entrypoint() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/setup_splatfacto_venv.sh").read_text(encoding="utf-8")

    assert 'pip" install --no-deps -e "${REPO_ROOT}"' in script
    assert "canonical_3dgs_cli run-arm --help" in script


def test_canonical_module_cli_consolidates_all_operations_without_console_scripts() -> None:
    assert set(COMMANDS) == {
        "prepare",
        "run-arm",
        "finalize",
        "transport",
        "admit-worker",
        "request-execution",
        "evaluate",
        "register",
    }
    assert canonical_cli_main(["--help"]) == 0
    assert canonical_cli_main(["unknown-operation"]) == 64


def test_worker_admission_binds_authority_watchdog_spend_and_exact_transport() -> None:
    transport = _transport_receipt()
    worker_image = "blueprint/postshot-worker@sha256:" + "a" * 64
    admission = build_canonical_3dgs_worker_admission(
        transport_receipt=transport,
        arm_id="postshot-primary",
        worker_platform="windows",
        paid_allocator_admission=_allocator_admission(transport, worker_image=worker_image),
        worker_image_digest=worker_image,
        trainer_runtime_digest="sha256:" + "9" * 64,
        trainer_runtime_version="fixture-trainer-1.0",
        authority_id="operator-approved-run-1",
        max_spend_usd=15.0,
        hard_ttl_seconds=7200,
        provider_upload_authorized=True,
        paid_compute_authorized=True,
        watchdog_armed=True,
        provider_zero_before_allocation=True,
        timestamp="2026-08-03T12:00:00Z",
    )

    assert admission["status"] == "admitted"
    accepted = require_canonical_3dgs_worker_admission(
        admission,
        arm_id="postshot-primary",
        plan_digest=transport["canonical_3dgs_execution_plan_digest"],
        dataset_digest=transport["colmap_training_dataset_digest"],
        transport_bundle_digest=transport["transport_bundle_digest"],
        worker_package_digest=transport["worker_python_package_digest"],
        observed_now=datetime(2026, 8, 3, 12, 30, tzinfo=timezone.utc),
    )
    assert accepted["retry_cap"] == 0
    assert accepted["provider_zero_required_after_execution"] is True


def test_worker_admission_blocks_missing_authority_and_cannot_cross_arm() -> None:
    transport = _transport_receipt()
    worker_image = "blueprint/postshot-worker@sha256:" + "a" * 64
    blocked = build_canonical_3dgs_worker_admission(
        transport_receipt=transport,
        arm_id="postshot-primary",
        worker_platform="windows",
        paid_allocator_admission=_allocator_admission(transport, worker_image=worker_image),
        worker_image_digest=worker_image,
        trainer_runtime_digest="invalid",
        trainer_runtime_version="",
        authority_id="",
        max_spend_usd=15.0,
        hard_ttl_seconds=7200,
        provider_upload_authorized=False,
        paid_compute_authorized=False,
        watchdog_armed=False,
        provider_zero_before_allocation=False,
        timestamp="2026-08-03T12:00:00Z",
    )
    assert blocked["status"] == "blocked"
    with pytest.raises(Canonical3DGSAdmissionError, match="not_admitted"):
        require_canonical_3dgs_worker_admission(
            blocked,
            arm_id="postshot-primary",
            plan_digest=transport["canonical_3dgs_execution_plan_digest"],
            dataset_digest=transport["colmap_training_dataset_digest"],
            transport_bundle_digest=transport["transport_bundle_digest"],
            worker_package_digest=transport["worker_python_package_digest"],
        )

    admitted = build_canonical_3dgs_worker_admission(
        transport_receipt=transport,
        arm_id="postshot-primary",
        worker_platform="windows",
        paid_allocator_admission=_allocator_admission(transport, worker_image=worker_image),
        worker_image_digest=worker_image,
        trainer_runtime_digest="sha256:" + "9" * 64,
        trainer_runtime_version="fixture-trainer-1.0",
        authority_id="operator-approved-run-1",
        max_spend_usd=15.0,
        hard_ttl_seconds=7200,
        provider_upload_authorized=True,
        paid_compute_authorized=True,
        watchdog_armed=True,
        provider_zero_before_allocation=True,
        timestamp="2026-08-03T12:00:00Z",
    )
    with pytest.raises(Canonical3DGSAdmissionError, match="arm_id"):
        require_canonical_3dgs_worker_admission(
            admitted,
            arm_id="splatfacto-comparison",
            plan_digest=transport["canonical_3dgs_execution_plan_digest"],
            dataset_digest=transport["colmap_training_dataset_digest"],
            transport_bundle_digest=transport["transport_bundle_digest"],
            worker_package_digest=transport["worker_python_package_digest"],
            observed_now=datetime(2026, 8, 3, 12, 30, tzinfo=timezone.utc),
        )


def test_worker_admission_rejects_allocator_tamper_and_expiry() -> None:
    transport = _transport_receipt()
    worker_image = "blueprint/splatfacto-worker@sha256:" + "b" * 64
    allocator = _allocator_admission(
        transport, worker_image=worker_image, arm_id="splatfacto-comparison"
    )
    allocator["retry_cap"] = 1
    admission = build_canonical_3dgs_worker_admission(
        transport_receipt=transport,
        arm_id="splatfacto-comparison",
        worker_platform="linux",
        paid_allocator_admission=allocator,
        worker_image_digest=worker_image,
        trainer_runtime_digest="sha256:" + "9" * 64,
        trainer_runtime_version="nerfstudio=1.1.5;gsplat=1.4.0",
        authority_id="operator-approved-run-1",
        max_spend_usd=15.0,
        hard_ttl_seconds=7200,
        provider_upload_authorized=True,
        paid_compute_authorized=True,
        watchdog_armed=True,
        provider_zero_before_allocation=True,
        timestamp="2026-08-03T12:00:00Z",
    )
    assert admission["status"] == "blocked"
    assert any("retry_cap" in code for code in admission["blockers"])

    valid_allocator = _allocator_admission(
        transport, worker_image=worker_image, arm_id="splatfacto-comparison"
    )
    admitted = build_canonical_3dgs_worker_admission(
        transport_receipt=transport,
        arm_id="splatfacto-comparison",
        worker_platform="linux",
        paid_allocator_admission=valid_allocator,
        worker_image_digest=worker_image,
        trainer_runtime_digest="sha256:" + "9" * 64,
        trainer_runtime_version="nerfstudio=1.1.5;gsplat=1.4.0",
        authority_id="operator-approved-run-1",
        max_spend_usd=15.0,
        hard_ttl_seconds=7200,
        provider_upload_authorized=True,
        paid_compute_authorized=True,
        watchdog_armed=True,
        provider_zero_before_allocation=True,
        timestamp="2026-08-03T12:00:00Z",
    )
    with pytest.raises(Canonical3DGSAdmissionError, match="expired"):
        require_canonical_3dgs_worker_admission(
            admitted,
            arm_id="splatfacto-comparison",
            plan_digest=transport["canonical_3dgs_execution_plan_digest"],
            dataset_digest=transport["colmap_training_dataset_digest"],
            transport_bundle_digest=transport["transport_bundle_digest"],
            worker_package_digest=transport["worker_python_package_digest"],
            observed_now=datetime(2026, 8, 3, 14, 0, 1, tzinfo=timezone.utc),
        )


def test_linux_vast_allocator_cannot_authorize_postshot_windows_arm() -> None:
    transport = _transport_receipt()
    worker_image = "blueprint/postshot-worker@sha256:" + "a" * 64
    linux_allocator = _allocator_admission(
        transport,
        worker_image=worker_image,
        arm_id="splatfacto-comparison",
    )
    admission = build_canonical_3dgs_worker_admission(
        transport_receipt=transport,
        arm_id="postshot-primary",
        worker_platform="windows",
        paid_allocator_admission=linux_allocator,
        worker_image_digest=worker_image,
        trainer_runtime_digest="sha256:" + "9" * 64,
        trainer_runtime_version="fixture-postshot-1.0",
        authority_id="operator-approved-run-1",
        max_spend_usd=15.0,
        hard_ttl_seconds=7200,
        provider_upload_authorized=True,
        paid_compute_authorized=True,
        watchdog_armed=True,
        provider_zero_before_allocation=True,
        timestamp="2026-08-03T12:00:00Z",
    )
    assert admission["status"] == "blocked"
    assert "canonical_3dgs_allocator_adapter_not_platform_qualified" in admission[
        "blockers"
    ]
    assert "canonical_3dgs_allocator_worker_platform_mismatch" in admission["blockers"]
