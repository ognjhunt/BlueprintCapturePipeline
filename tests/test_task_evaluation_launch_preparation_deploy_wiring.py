from __future__ import annotations

import json
import re
import os
from pathlib import Path

import scripts.deploy_control_plane_commit as deploy
from blueprint_pipeline import live_pipeline_intake_service as intake


ROOT = Path(__file__).resolve().parents[1]


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_no_spend_preparation_worker_has_hardened_service_and_path_unit() -> None:
    service = text(
        "deploy/systemd/blueprint-task-evaluation-launch-preparation.service"
    )
    path = text("deploy/systemd/blueprint-task-evaluation-launch-preparation.path")
    assert "User=blueprint" in service
    assert "NoNewPrivileges=true" in service
    assert "ProtectSystem=strict" in service
    assert "ProtectHome=true" in service
    assert "task_evaluation_launch_preparation_worker" in service
    assert "paid_resource_allocator" not in service
    assert "provider_adapter" not in service
    assert "EnvironmentFile=-/etc/blueprint/pipeline-control-plane.env" in service
    assert "task-evaluation-launch-preparations/pending" in path
    assert "blueprint-task-evaluation-launch-preparation.service" in path
    assert "BLUEPRINT_TASK_EVALUATION_SCENE_CONSTRUCTION_QUEUE_ROOT=" in service
    assert "task-evaluation-scene-constructions" in service
    assert "BLUEPRINT_TASK_EVALUATION_EPISODE_COMPILATION_QUEUE_ROOT=" in service
    assert "task-evaluation-episode-compilations" in service
    assert (
        "Environment='BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_ALLOWED_URI_PREFIXES_JSON="
        '["s3://blueprint/task-evaluation/production-inputs/",'
        '"s3://blueprint-task-evaluation-artifacts-prod/blueprint/arm-decision-proof-v1/'
        'configured-scenes/artifacts/",'
        '"s3://blueprint/blueprint/arm-decision-proof-v1/configured-scenes/"]\''
    ) in service
    assert service.index(
        "Environment='BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_ALLOWED_URI_PREFIXES_JSON="
    ) > service.index(
        "EnvironmentFile=-/etc/blueprint/task-evaluation-scene-configuration-release.env"
    )
    for binding in (
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ACCESS_KEY_ID_FILE="
        "/etc/blueprint/provider-secrets/backblaze_b2_key_id",
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_SECRET_ACCESS_KEY_FILE="
        "/etc/blueprint/provider-secrets/backblaze_b2_application_key",
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ENDPOINT_URL_FILE="
        "/etc/blueprint/provider-secrets/backblaze_b2_s3_endpoint_url",
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_BUCKET_FILE="
        "/etc/blueprint/provider-secrets/backblaze_b2_bucket",
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_REGION_FILE="
        "/etc/blueprint/provider-secrets/backblaze_b2_region",
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET="
        "blueprint-task-evaluation-artifacts-prod",
    ):
        assert f"Environment={binding}" in service
        assert service.index(f"Environment={binding}") > service.index(
            "EnvironmentFile=-/etc/blueprint/task-evaluation-scene-configuration-release.env"
        )


def test_canonical_environment_documents_bounded_input_prefixes() -> None:
    environment = text("deploy/systemd/pipeline-control-plane.env.example")
    assert (
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_ALLOWED_URI_PREFIXES_JSON="
        '["s3://blueprint/task-evaluation/production-inputs/",'
        '"s3://blueprint-task-evaluation-artifacts-prod/blueprint/arm-decision-proof-v1/'
        'configured-scenes/artifacts/",'
        '"s3://blueprint/blueprint/arm-decision-proof-v1/configured-scenes/"]'
    ) in environment
    assert "BLUEPRINT_TASK_EVALUATION_SPLAT_RENDER_RUNTIME_ROOT=" in environment
    assert "BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE=" in environment
    assert "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE=" in environment
    assert "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ACCESS_KEY_ID_FILE=" in environment
    assert "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_SECRET_ACCESS_KEY_FILE=" in environment
    assert "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ENDPOINT_URL_FILE=" in environment
    assert "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_BUCKET_FILE=" in environment
    assert "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_REGION_FILE=" in environment
    assert (
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET="
        "blueprint-task-evaluation-artifacts-prod"
    ) in environment
    assert "OPENAI_PROJECT_ID=" in environment
    assert "OPENAI_API_KEY_ID=" in environment
    assert "OPENAI_ADMIN_API_KEY_FILE=" in environment
    assert "BLUEPRINT_OPENAI_COST_SCOPE_ATTESTATION_FILE=" in environment
    assert "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID=" in environment
    assert "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID=" in environment
    assert "OPENAI_CONTENT_AGENTS_API_KEY_ID=" in environment


def test_preparation_loads_exact_release_scene_runtime_after_shared_environment() -> None:
    service = text(
        "deploy/systemd/blueprint-task-evaluation-launch-preparation.service"
    )
    assert service.index("EnvironmentFile=-/etc/blueprint/pipeline-control-plane.env") < (
        service.index(
            "EnvironmentFile=-/etc/blueprint/task-evaluation-scene-configuration-release.env"
        )
    )


def test_installer_creates_both_automatic_progression_queues() -> None:
    installer = text("scripts/install_live_pipeline_control_plane.sh")
    for queue in (
        "task-evaluation-scene-constructions",
        "task-evaluation-episode-compilations",
    ):
        for state in ("pending", "processing", "completed", "blocked", "results"):
            assert f'"${{STATE_DIR}}/{queue}/{state}"' in installer


def test_canonical_installer_installs_and_enables_preparation_pair() -> None:
    installer = text("scripts/install_live_pipeline_control_plane.sh")
    for unit in (
        "blueprint-task-evaluation-launch-preparation.service",
        "blueprint-task-evaluation-launch-preparation.path",
    ):
        assert f'"${{REPO_ROOT}}/deploy/systemd/{unit}"' in installer
        assert f'"${{SYSTEMD_DIR}}/{unit}"' in installer
    assert (
        "systemctl enable --now "
        "blueprint-task-evaluation-launch-preparation.path"
    ) in installer


def test_episode_compiler_has_hardened_no_network_service_and_path_unit() -> None:
    service = text(
        "deploy/systemd/blueprint-task-evaluation-episode-compilation.service"
    )
    path = text(
        "deploy/systemd/blueprint-task-evaluation-episode-compilation.path"
    )
    assert "User=blueprint" in service
    assert "NoNewPrivileges=true" in service
    assert "ProtectSystem=strict" in service
    assert "RestrictAddressFamilies=AF_UNIX" in service
    assert "task_evaluation_episode_compilation_worker" in service
    assert "git rev-parse --verify HEAD^{commit}" in service
    assert "git status --porcelain --untracked-files=no" in service
    assert '--source-commit "$${source_commit}"' in service
    assert "BLUEPRINT_SOURCE_COMMIT" not in service
    assert "paid_resource_allocator" not in service
    assert "provider_adapter" not in service
    assert "task-evaluation-episode-compilations/pending" in path
    assert "blueprint-task-evaluation-episode-compilation.service" in path
    assert "compiled-episodes" in service


def test_every_disk_reservation_worker_has_exact_systemd_write_access() -> None:
    reservation_root = "/var/lib/blueprint/pipeline-control-plane/disk-reservations"
    workers = {
        "blueprint-task-evaluation-launch-preparation.service": (
            "src/blueprint_pipeline/task_evaluation_launch_preparation_worker.py"
        ),
        "blueprint-task-evaluation-episode-compilation.service": (
            "src/blueprint_pipeline/task_evaluation_episode_compilation_worker.py"
        ),
        "blueprint-task-evaluation-launch-activation.service": (
            "src/blueprint_pipeline/task_evaluation_launch_activation_worker.py"
        ),
        "blueprint-task-evaluation-policy-canary-dispatcher.service": (
            "src/blueprint_pipeline/task_evaluation_policy_canary_dispatcher.py"
        ),
    }
    for unit, module in workers.items():
        assert "reserve_control_plane_disk(" in text(module)
        service = text(f"deploy/systemd/{unit}")
        assert (
            f"Environment=BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT="
            f"{reservation_root}"
        ) in service
        write_paths = next(
            line.split("=", 1)[1].split()
            for line in service.splitlines()
            if line.startswith("ReadWritePaths=")
        )
        assert reservation_root in write_paths, (
            f"{unit} calls disk reservation but its strict filesystem sandbox "
            "does not expose the ledger as an exact writable path"
        )


def test_canonical_installer_installs_and_enables_episode_compilation_pair() -> None:
    installer = text("scripts/install_live_pipeline_control_plane.sh")
    for unit in (
        "blueprint-task-evaluation-episode-compilation.service",
        "blueprint-task-evaluation-episode-compilation.path",
    ):
        assert f'"${{REPO_ROOT}}/deploy/systemd/{unit}"' in installer
        assert f'"${{SYSTEMD_DIR}}/{unit}"' in installer
    assert (
        "systemctl enable --now "
        "blueprint-task-evaluation-episode-compilation.path"
    ) in installer


def test_exact_sha_deployer_installs_and_arms_all_no_spend_intake_paths() -> None:
    assert "blueprint-task-evaluation-launch-preparation.service" in (
        deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    )
    assert "blueprint-task-evaluation-launch-preparation.path" in (
        deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    )
    assert deploy.DEFAULT_ALWAYS_ARM_PATH_UNITS == (
        "blueprint-task-evaluation-launch-preparation.path",
        "blueprint-task-evaluation-episode-compilation.path",
        "blueprint-task-evaluation-launch-activation.path",
        "blueprint-scene-object-discovery.path",
    )
    assert "blueprint-scene-object-discovery.service" in (
        deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    )
    assert "blueprint-scene-object-discovery.path" in (
        deploy.DEFAULT_DEPLOYED_SYSTEMD_UNITS
    )
    assert "blueprint-task-evaluation-launch-dispatcher.path" not in (
        deploy.DEFAULT_ALWAYS_ARM_PATH_UNITS
    )
    assert deploy.DEFAULT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT in (
        deploy.DEFAULT_SCENE_OBJECT_DISCOVERY_RUNTIME_DIRECTORIES
    )
    assert (
        f"{deploy.DEFAULT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT}/pending"
        in deploy.DEFAULT_SCENE_OBJECT_DISCOVERY_RUNTIME_DIRECTORIES
    )


def test_exact_sha_deployer_binds_discovery_queue_into_intake_runtime() -> None:
    source = text("scripts/deploy_control_plane_commit.py")
    assert "BLUEPRINT_SCENE_OBJECT_DISCOVERY_QUEUE_ROOT=" in source
    assert "_install_scene_object_discovery_runtime_directories()" in source


def test_exact_sha_deployer_materializes_discovery_runtime_directories(
    tmp_path: Path, monkeypatch
) -> None:
    owner = (os.getuid(), os.getgid())
    monkeypatch.setattr(deploy, "_service_account_ids", lambda _account: owner)
    directories = (
        str(tmp_path / "queue"),
        str(tmp_path / "queue" / "pending"),
        str(tmp_path / "inputs"),
    )

    receipts = deploy._install_scene_object_discovery_runtime_directories(
        directories=directories,
        account="test-account",
    )

    assert [receipt["path"] for receipt in receipts] == list(directories)
    assert all(Path(path).is_dir() for path in directories)
    assert all(Path(path).stat().st_mode & 0o777 == 0o750 for path in directories)


def test_canonical_environment_declares_every_intake_queue_root() -> None:
    """Every queue root the intake reads must exist in the canonical env.

    The intake answers a Website-started preparation or activation with
    ``..._queue_not_configured`` when its queue-root variable is unset, and it
    reads those names only from the environment.  Two of the four were absent
    from the canonical file, so every host provisioned from it rejected
    Website-started scene configuration at the first request.  Discover the
    names from the service module rather than listing them here: a hand-kept
    list is exactly what let these two slip.
    """

    source = text("src/blueprint_pipeline/live_pipeline_intake_service.py")
    read_constants = set(
        re.findall(r"os\.getenv\(([A-Z][A-Z0-9_]*QUEUE_ROOT_ENV)\)", source)
    )
    assert read_constants, "no intake queue-root environment reads discovered"
    environment = text("deploy/systemd/pipeline-control-plane.env.example")
    declared = {
        line.split("=", 1)[0].strip()
        for line in environment.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    missing = set()
    for constant in sorted(read_constants):
        name = getattr(intake, constant)
        if name not in declared:
            missing.add(name)
    assert not missing, f"intake queue roots absent from canonical env: {sorted(missing)}"


def test_every_control_plane_unit_can_read_its_own_git_identity() -> None:
    """A unit running from the release worktree must declare safe.directory.

    The deployer stages release worktrees root-owned so a service account
    cannot rewrite the code it runs, and the workers run as ``blueprint``.
    Git then refuses those worktrees for "dubious ownership", so any
    identity read inside the worker fails -- production surfaced this as
    ``splat_render_runtime_repository_identity_unavailable`` blocking every
    Website-started scene configuration. The dispatcher already exported
    safe.directory; the preparation, activation, and compilation workers did
    not. Discover the units from the deploy tree so a new worker cannot be
    added without the same export.
    """

    units = sorted(
        path
        for path in (ROOT / "deploy/systemd").glob("*.service")
        if "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO" in path.read_text(
            encoding="utf-8"
        )
    )
    assert units, "no control-plane units discovered"
    missing = []
    for path in units:
        source = path.read_text(encoding="utf-8")
        if (
            "GIT_CONFIG_COUNT=1" not in source
            or "GIT_CONFIG_KEY_0=safe.directory" not in source
            or 'GIT_CONFIG_VALUE_0="$${PWD}"' not in source
        ):
            missing.append(path.name)
    assert not missing, f"control-plane units without git safe.directory: {missing}"


def test_preparation_and_activation_share_one_allowed_uri_prefix_list() -> None:
    """Both gates validate the same request references, so a divergent allowlist
    silently blocks a launch at the second gate after passing the first."""
    key = "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_ALLOWED_URI_PREFIXES_JSON="

    def prefixes(path: str) -> list[str]:
        for line in text(path).splitlines():
            marker = line.find(key)
            if marker == -1:
                continue
            value = line[marker + len(key) :].rstrip().rstrip("'")
            return json.loads(value)
        raise AssertionError(f"{key} missing from {path}")

    preparation = prefixes(
        "deploy/systemd/blueprint-task-evaluation-launch-preparation.service"
    )
    activation = prefixes(
        "deploy/systemd/blueprint-task-evaluation-launch-activation.service"
    )
    canonical = prefixes("deploy/systemd/pipeline-control-plane.env.example")
    assert preparation == activation == canonical
    assert "s3://blueprint/blueprint/arm-decision-proof-v1/configured-scenes/" in (
        preparation
    )
    assert all(prefix.endswith("/") for prefix in preparation)



def test_every_storage_pin_writer_has_exact_systemd_write_access() -> None:
    pins_root = "/var/lib/blueprint/pipeline-control-plane/storage-pins"
    writers = {
        "blueprint-task-evaluation-launch-preparation.service": (
            "src/blueprint_pipeline/task_evaluation_launch_preparation_worker.py",
            "write_storage_pin(",
        ),
        "blueprint-task-evaluation-episode-compilation.service": (
            "src/blueprint_pipeline/task_evaluation_episode_compilation_worker.py",
            "write_storage_pin(",
        ),
        "blueprint-task-evaluation-launch-activation.service": (
            "src/blueprint_pipeline/task_evaluation_launch_activation_worker.py",
            "pin_activation_best_effort(",
        ),
        "blueprint-task-evaluation-policy-canary-dispatcher.service": (
            "src/blueprint_pipeline/task_evaluation_policy_canary_dispatcher.py",
            "release_storage_pin(",
        ),
    }
    for unit, (module, call) in writers.items():
        assert call in text(module), (unit, call)
        service = text(f"deploy/systemd/{unit}")
        assert f"Environment=BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT={pins_root}" in service
        write_paths = next(
            line.split("=", 1)[1].split()
            for line in service.splitlines()
            if line.startswith("ReadWritePaths=")
        )
        assert pins_root in write_paths, (
            f"{unit} touches storage pins but its strict filesystem sandbox does not "
            "expose the pins ledger as an exact writable path"
        )
    deploy_source = text("scripts/deploy_control_plane_commit.py")
    assert "_install_storage_pins_runtime_root()" in deploy_source


def test_storage_gc_timer_pair_is_deployed_armed_and_scoped_by_storage_class() -> None:
    from blueprint_pipeline.control_plane_storage_gc import (
        CONTENT_STORE_ROOTS_ENV,
        DERIVED_ROOTS_ENV,
        EVIDENCE_ROOTS_ENV,
        QUEUE_ROOTS_ENV,
        RUN_ACK,
    )
    from blueprint_pipeline.control_plane_storage_pins import PINS_ROOT_ENV
    from blueprint_pipeline.control_plane_storage_roots import classify_path

    service = text("deploy/systemd/blueprint-control-plane-storage-gc.service")
    timer = text("deploy/systemd/blueprint-control-plane-storage-gc.timer")
    assert (
        f"-m blueprint_pipeline.control_plane_storage_gc run --apply --ack {RUN_ACK}"
    ) in service
    assert "User=root" in service and "ProtectSystem=strict" in service
    assert "CapabilityBoundingSet=CAP_DAC_OVERRIDE" in service
    assert "AmbientCapabilities=CAP_DAC_OVERRIDE" in service
    assert "ReadWritePaths=/var/lib/blueprint " not in service
    assert "/var/lib/blueprint/pipeline-control-plane/policy-canary-presubmission" in service

    def roots(env: str) -> list[str]:
        line = next(row for row in service.splitlines() if row.startswith(f"Environment={env}="))
        return [item for item in line.split("=", 2)[2].split(":") if item]

    for root in roots(DERIVED_ROOTS_ENV):
        assert classify_path(root).storage_class == "cache", root
    for root in roots(CONTENT_STORE_ROOTS_ENV):
        assert root.endswith("/sha256") and classify_path(root).storage_class == "cache", root
    for root in roots(EVIDENCE_ROOTS_ENV):
        assert classify_path(root).storage_class == "evidence_cold", root
    for root in roots(QUEUE_ROOTS_ENV):
        assert classify_path(root).storage_class == "work", root
    assert roots(PINS_ROOT_ENV) == ["/var/lib/blueprint/pipeline-control-plane/storage-pins"]
    assert not any(
        row.startswith("Environment=BLUEPRINT_CONTROL_PLANE_EVIDENCE_OFFLOAD=")
        for row in service.splitlines()
    ), "offload must stay an operator opt-in from the environment file"
    assert "Unit=blueprint-control-plane-storage-gc.service" in timer
    assert "OnUnitInactiveSec=" in timer and "Persistent=true" in timer

    deploy_source = text("scripts/deploy_control_plane_commit.py")
    assert '"blueprint-control-plane-storage-gc.timer",' in deploy_source
    assert '"blueprint-control-plane-storage-gc.service",' in deploy_source
    installer = text("scripts/install_live_pipeline_control_plane.sh")
    assert "blueprint-control-plane-storage-gc.service" in installer
    assert "systemctl enable --now blueprint-control-plane-storage-gc.timer" in installer
