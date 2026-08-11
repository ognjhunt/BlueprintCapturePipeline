from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _text(relative: str) -> str:
    return (REPO / relative).read_text(encoding="utf-8")


def test_canonical_allocator_dependencies_are_in_the_production_base() -> None:
    pyproject = _text("pyproject.toml")
    base_dependencies = pyproject.split("[project.optional-dependencies]", 1)[0]
    frozen_base = _text("requirements.txt")

    for requirement in (
        '"packaging>=24.0"',
        '"usd-core>=24.0"',
        '"boto3>=1.34.0"',
        '"botocore>=1.34.0"',
    ):
        assert requirement in base_dependencies
    for requirement in ("packaging==", "usd-core==", "boto3==", "botocore=="):
        assert requirement in frozen_base


def test_production_launch_units_preserve_four_layer_control_boundary() -> None:
    dispatcher = _text("deploy/systemd/blueprint-task-evaluation-launch-dispatcher.service")
    path_unit = _text("deploy/systemd/blueprint-task-evaluation-launch-dispatcher.path")
    reconciler = _text("deploy/systemd/blueprint-task-evaluation-launch-reconciler.service")
    supervisor = _text("deploy/systemd/blueprint-task-evaluation-launch-supervisor.service")

    assert "task_evaluation_launch_dispatcher" in dispatcher
    assert "--execute" in dispatcher
    assert "--execute-launch-id" in dispatcher
    assert "BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE_ID" in dispatcher
    assert (
        'ARGS+=(--execute --execute-launch-id '
        '"$${BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE_ID}")'
    ) in dispatcher
    assert (
        '"$${BLUEPRINT_TASK_EVALUATION_LAUNCH_FORCE_DRY_RUN:-}" != true ] '
        '&& [ "$${BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE:-}" = true'
    ) in dispatcher
    assert "EnvironmentFile values override Environment=" in dispatcher
    assert "blueprint-gpu-spend-guard.service" in dispatcher
    assert "GIT_CONFIG_KEY_0=safe.directory" in dispatcher
    for unit in (dispatcher, reconciler, supervisor):
        assert "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO=" in unit
        assert "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_PYTHON=" in unit
        assert "BLUEPRINT_PIPELINE_REPO" not in unit
        assert (
            'GIT_CONFIG_VALUE_0="$$(realpath '
            '"$${BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO}")"'
        ) in unit
        assert 'PYTHONPATH=src "$${BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_PYTHON}"' in unit
    for binding in (
        "VAST_API_KEY_FILE=/etc/blueprint/provider-secrets/vast_api_key",
        "VAST_LAUNCH_LOCK_FILE=/var/lib/blueprint/pipeline-control-plane/provider-locks/vast_paid_launch.lock",
        "NGC_API_KEY_FILE=/etc/blueprint/provider-secrets/ngc_api_key",
        "DOCKER_USERNAME_FILE=/etc/blueprint/provider-secrets/docker_username",
        "DOCKER_PAT_FILE=/etc/blueprint/provider-secrets/docker_pat",
        "HF_TOKEN_FILE=/etc/blueprint/provider-secrets/huggingface_token",
        "BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE=/etc/blueprint/provider-secrets/digitalocean_spaces_access_key_id",
        "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE=/etc/blueprint/provider-secrets/digitalocean_spaces_secret_access_key",
        "BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL_FILE=/etc/blueprint/provider-secrets/digitalocean_spaces_endpoint_url",
        "BLUEPRINT_WAM_OBJECT_STORE_BUCKET_FILE=/etc/blueprint/provider-secrets/digitalocean_spaces_bucket",
        "BLUEPRINT_WAM_OBJECT_STORE_REGION_FILE=/etc/blueprint/provider-secrets/digitalocean_spaces_region",
    ):
        assert binding in dispatcher
    assert "PathExistsGlob=" in path_unit
    assert "task-evaluation-launches/pending/*.json" in path_unit

    assert "task_evaluation_launch_reconciler" in reconciler
    assert "blueprint-gpu-spend-guard.service" in reconciler
    assert "--guard-report" in reconciler

    assert "task_evaluation_launch_supervisor" in supervisor
    assert "ReadOnlyPaths=" in supervisor
    assert "task-evaluation-launches" in supervisor
    assert "task-evaluation-launch-runs" in supervisor
    assert "paid_resource_allocator" not in supervisor
    assert "provider-secrets" not in supervisor


def test_provider_zero_inputs_share_the_immutable_task_evaluation_release() -> None:
    guard = _text("deploy/systemd/blueprint-gpu-spend-guard.service")
    billing = _text("deploy/systemd/blueprint-provider-billing-reconciler.service")

    for unit in (guard, billing):
        assert (
            "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO="
            "/opt/blueprint/task-evaluation-control-plane"
        ) in unit
        assert (
            "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_PYTHON="
            "/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python"
        ) in unit
        assert "BLUEPRINT_PIPELINE_REPO" not in unit
        assert (
            'GIT_CONFIG_VALUE_0="$$(realpath '
            '"$${BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO}")"'
        ) in unit
        assert 'PYTHONPATH=src "$${BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_PYTHON}"' in unit

    assert "scripts/gpu_spend_guard.py" in guard
    assert "blueprint_pipeline.provider_billing_reconciler" in billing


def test_installer_and_environment_enable_durable_queue_and_independent_recovery() -> None:
    installer = _text("scripts/install_live_pipeline_control_plane.sh")
    environment = _text("deploy/systemd/pipeline-control-plane.env.example")

    for unit in (
        "blueprint-task-evaluation-launch-dispatcher.path",
        "blueprint-task-evaluation-launch-reconciler.timer",
        "blueprint-task-evaluation-launch-supervisor.timer",
        "blueprint-gpu-spend-guard.timer",
    ):
        assert unit in installer
    for directory in (
        "task-evaluation-launches/pending",
        "task-evaluation-launches/processing",
        "task-evaluation-launches/completed",
        "task-evaluation-launches/blocked",
        "task-evaluation-launch-runs",
        "task-evaluation-control-plane-releases",
        "task-evaluation-launch-reconciliation",
        "provider-locks",
    ):
        assert directory in installer

    assert "BLUEPRINT_TASK_EVALUATION_LAUNCH_TRIGGER_MODE=systemd_path" in environment
    assert "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO=/opt/blueprint/task-evaluation-control-plane" in environment
    assert "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_PYTHON=/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python" in environment
    assert "# BLUEPRINT_TASK_EVALUATION_SECRET_PROFILE_ID=canonical-vast-adp" in environment
    assert "# BLUEPRINT_ALLOW_TASK_EVALUATION_LAUNCH_TRIGGER=true" in environment
    assert "# BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE=true" in environment
    assert "# BLUEPRINT_TASK_EVALUATION_AGENT_SUPERVISOR_ENABLED=true" in environment
    for binding in (
        "VAST_API_KEY_FILE=/etc/blueprint/provider-secrets/vast_api_key",
        "VAST_LAUNCH_LOCK_FILE=/var/lib/blueprint/pipeline-control-plane/provider-locks/vast_paid_launch.lock",
        "NGC_API_KEY_FILE=/etc/blueprint/provider-secrets/ngc_api_key",
        "DOCKER_USERNAME_FILE=/etc/blueprint/provider-secrets/docker_username",
        "DOCKER_PAT_FILE=/etc/blueprint/provider-secrets/docker_pat",
        "HF_TOKEN_FILE=/etc/blueprint/provider-secrets/huggingface_token",
        "BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE=/etc/blueprint/provider-secrets/digitalocean_spaces_access_key_id",
        "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE=/etc/blueprint/provider-secrets/digitalocean_spaces_secret_access_key",
        "BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL_FILE=/etc/blueprint/provider-secrets/digitalocean_spaces_endpoint_url",
        "BLUEPRINT_WAM_OBJECT_STORE_BUCKET_FILE=/etc/blueprint/provider-secrets/digitalocean_spaces_bucket",
        "BLUEPRINT_WAM_OBJECT_STORE_REGION_FILE=/etc/blueprint/provider-secrets/digitalocean_spaces_region",
    ):
        assert binding in environment


def test_installer_gives_the_service_account_ownership_of_the_checkout() -> None:
    """A root-owned checkout makes git refuse with "detected dubious ownership"
    when the service account probes the allocator's source identity, which
    rejects a paid launch at admission. Observed in production after a root
    deploy."""
    from pathlib import Path as _Path

    script = _Path("scripts/install_live_pipeline_control_plane.sh").read_text(
        encoding="utf-8"
    )

    assert 'chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${REPO_ROOT}"' in script
    assert 'cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P' in script
    assert "dubious ownership" in script


def test_dispatcher_exit_code_separates_outcome_from_dispatch_failure() -> None:
    """Exiting non-zero on a blocked launch made systemd mark the unit failed and
    deactivate the path unit watching the queue, so one blocked run silently
    stopped every later website trigger from dispatching."""
    from pathlib import Path as _Path

    source = _Path(
        "src/blueprint_pipeline/task_evaluation_launch_dispatcher.py"
    ).read_text(encoding="utf-8")

    assert 'return 0 if result["status"] == "completed" else 2' not in source
    assert (
        'return 0 if result.get("schema_version") == QUEUE_RUN_SCHEMA_VERSION else 2'
        in source
    )
