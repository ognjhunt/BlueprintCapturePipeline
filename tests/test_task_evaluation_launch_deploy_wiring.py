import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _text(relative: str) -> str:
    return (REPO / relative).read_text(encoding="utf-8")


def _shell_script(unit: str, directive: str) -> str:
    prefix = f"{directive}=/bin/bash -lc '"
    line = next(line for line in unit.splitlines() if line.startswith(prefix))
    assert line.endswith("'")
    return line.removeprefix(prefix).removesuffix("'")


def test_shell_wrapped_production_unit_commands_parse_before_deploy() -> None:
    units_and_directives = (
        ("deploy/systemd/blueprint-task-evaluation-launch-dispatcher.service", "ExecStart"),
        ("deploy/systemd/blueprint-task-evaluation-launch-reconciler.service", "ExecStart"),
        ("deploy/systemd/blueprint-provider-billing-reconciler.service", "ExecStart"),
        ("deploy/systemd/blueprint-gpu-spend-guard.service", "ExecStart"),
        ("deploy/systemd/blueprint-gpu-spend-guard.service", "ExecStopPost"),
    )

    for relative, directive in units_and_directives:
        completed = subprocess.run(
            ("bash", "-n", "-c", _shell_script(_text(relative), directive)),
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, f"{relative} {directive}: {completed.stderr}"


def test_production_launch_units_preserve_four_layer_control_boundary() -> None:
    dispatcher = _text("deploy/systemd/blueprint-task-evaluation-launch-dispatcher.service")
    path_unit = _text("deploy/systemd/blueprint-task-evaluation-launch-dispatcher.path")
    reconciler = _text("deploy/systemd/blueprint-task-evaluation-launch-reconciler.service")

    assert "task_evaluation_launch_dispatcher" in dispatcher
    assert "--execute" in dispatcher
    assert "--execute-launch-id" in dispatcher
    assert "--public-catalog" in dispatcher
    assert "BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE_ID" in dispatcher
    assert "Environment=BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE=true" in dispatcher
    assert "Environment=BLUEPRINT_TASK_EVALUATION_LAUNCH_MAX_CONCURRENCY=1" in dispatcher
    assert (
        '--max-concurrency "$${BLUEPRINT_TASK_EVALUATION_LAUNCH_MAX_CONCURRENCY:-1}"'
        in dispatcher
    )
    assert "BLUEPRINT_TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH" in dispatcher
    assert (
        "BLUEPRINT_TASK_EVALUATION_SCENE_CONSTRUCTION_QUEUE_ROOT="
        "/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-constructions"
    ) in dispatcher
    assert "task-evaluation-launch-profile-catalog.json" in dispatcher
    # Execution is armed by the execute flag; the launch id narrows the window
    # when one is set. Coupling them meant a standing per-profile authorization
    # -- digest-bound, expiring, count- and spend-bounded -- could never admit
    # anything, because the unit never passed `--execute` without a hand-edited
    # per-run id.
    assert "ARGS+=(--execute); if [ -n " in dispatcher
    assert (
        'ARGS+=(--execute-launch-id "$${BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE_ID}")'
    ) in dispatcher
    assert (
        '"$${BLUEPRINT_TASK_EVALUATION_LAUNCH_FORCE_DRY_RUN:-}" != true ] '
        '&& [ "$${BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE:-}" = true'
    ) in dispatcher
    # Every credential this unit needs is named on the unit. It runs with
    # ProtectHome=true and home /nonexistent, so a secret resolved from a
    # developer home is unreadable here -- which is how the Content Agents
    # paid preflight came to be unproducible on the deployed host.
    assert "OPENAI_API_KEY_FILE=/etc/blueprint/provider-secrets/openai_api_key" in dispatcher
    assert (
        "OPENAI_ADMIN_API_KEY_FILE="
        "/etc/blueprint/provider-secrets/openai_admin_api_key"
    ) in dispatcher
    assert (
        "BLUEPRINT_OPENAI_COST_SCOPE_ATTESTATION_FILE="
        "/etc/blueprint/provider-secrets/openai_cost_scope_attestation.json"
    ) in dispatcher
    # Scene-configuration OpenAI stages each need an exclusive key scope; a
    # shared key/attestation cannot pass the per-stage official-cost gate.
    for stage in (
        "artifixer_semantic_teacher",
        "artifixer_visual_review",
        "content_agents",
    ):
        upper = stage.upper()
        assert (
            f"OPENAI_{upper}_API_KEY_FILE="
            f"/etc/blueprint/provider-secrets/openai_api_key_{stage}"
        ) in dispatcher
        assert (
            f"BLUEPRINT_OPENAI_{upper}_COST_SCOPE_ATTESTATION_FILE="
            f"/etc/blueprint/provider-secrets/openai_cost_scope_attestation_{stage}.json"
        ) in dispatcher
    assert (
        "BLUEPRINT_GPU_PROVIDER_SECRETS_DIR=/etc/blueprint/provider-secrets"
    ) in dispatcher
    assert "EnvironmentFile values override Environment=" in dispatcher
    assert "blueprint-gpu-spend-guard.service" in dispatcher
    assert "GIT_CONFIG_KEY_0=safe.directory" in dispatcher
    for unit in (dispatcher, reconciler):
        assert "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO=" in unit
        assert "BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_PYTHON=" in unit
        assert "BLUEPRINT_PIPELINE_REPO" not in unit
        assert (
            'cd -P "$${BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO}" '
            '&& export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory '
            'GIT_CONFIG_VALUE_0="$${PWD}"'
        ) in unit
        assert "$$(realpath" not in unit
        assert 'PYTHONPATH=src "$${BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_PYTHON}"' in unit
    for binding in (
        "VAST_API_KEY_FILE=/etc/blueprint/provider-secrets/vast_api_key",
        "BLUEPRINT_VAST_SSH_IDENTITY_FILE=/etc/blueprint/provider-secrets/vast_ssh_id_ed25519",
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
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ACCESS_KEY_ID_FILE=/etc/blueprint/provider-secrets/backblaze_b2_key_id",
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_SECRET_ACCESS_KEY_FILE=/etc/blueprint/provider-secrets/backblaze_b2_application_key",
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ENDPOINT_URL_FILE=/etc/blueprint/provider-secrets/backblaze_b2_s3_endpoint_url",
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_BUCKET_FILE=/etc/blueprint/provider-secrets/backblaze_b2_bucket",
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_REGION_FILE=/etc/blueprint/provider-secrets/backblaze_b2_region",
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET=blueprint-task-evaluation-artifacts-prod",
    ):
        assert binding in dispatcher
    assert (
        "PathChanged=/var/lib/blueprint/pipeline-control-plane/"
        "task-evaluation-launches/pending"
    ) in path_unit
    assert "PathExistsGlob=" in path_unit
    assert "task-evaluation-launches/pending/*.json" in path_unit

    assert "task_evaluation_launch_reconciler" in reconciler
    assert "blueprint-gpu-spend-guard.service" in reconciler
    assert "--guard-report" in reconciler
    # R8: the reconciler tick files owner terminal receipts; the unit must forward
    # the canary dispatch root, the terminal result root and the owner intent
    # store, and each must be the path the producing/consuming unit uses.
    for flag, variable, path in (
        ("--policy-canary-dispatch-root", "BLUEPRINT_TASK_EVALUATION_POLICY_CANARY_DISPATCH_ROOT",
         "/var/lib/blueprint/pipeline-control-plane/task-evaluation-policy-canaries"),
        ("--terminal-result-root", "BLUEPRINT_TASK_EVALUATION_TERMINAL_RESULT_ROOT",
         "/var/lib/blueprint/task-evaluation-inputs/task-evaluation-terminal-results"),
        ("--scene-intent-root", "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT",
         "/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents"),
    ):
        assert f'{flag} "$${{{variable}}}"' in reconciler
        assert f"Environment={variable}={path}" in reconciler


def test_canonical_environment_binds_scene_configuration_runtime_secret_paths() -> None:
    """A rebuilt host must retain every non-secret runtime path binding."""

    environment = _text("deploy/systemd/pipeline-control-plane.env.example")
    expected = {
        "OPENAI_API_KEY_FILE": "/etc/blueprint/provider-secrets/openai_api_key",
        "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE": (
            "/etc/blueprint/provider-secrets/openai_api_key_artifixer_semantic_teacher"
        ),
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE": (
            "/etc/blueprint/provider-secrets/openai_api_key_artifixer_visual_review"
        ),
        "OPENAI_CONTENT_AGENTS_API_KEY_FILE": (
            "/etc/blueprint/provider-secrets/openai_api_key_content_agents"
        ),
        "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE": (
            "/etc/blueprint/provider-secrets/"
            "openai_cost_scope_attestation_artifixer_semantic_teacher.json"
        ),
        "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE": (
            "/etc/blueprint/provider-secrets/"
            "openai_cost_scope_attestation_artifixer_visual_review.json"
        ),
        "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE": (
            "/etc/blueprint/provider-secrets/"
            "openai_cost_scope_attestation_content_agents.json"
        ),
    }
    declared = {
        name: value
        for line in environment.splitlines()
        if line and not line.lstrip().startswith("#") and "=" in line
        for name, value in [line.split("=", 1)]
    }
    assert {name: declared.get(name) for name in expected} == expected


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
            'cd -P "$${BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_REPO}" '
            '&& export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory '
            'GIT_CONFIG_VALUE_0="$${PWD}"'
        ) in unit
        assert "$$(realpath" not in unit
        assert 'PYTHONPATH=src "$${BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_PYTHON}"' in unit

    assert "scripts/gpu_spend_guard.py" in guard
    assert "BLUEPRINT_GPU_SPEND_GUARD_OUTPUT_ROOT=/var/lib/blueprint/pipeline-control-plane" in guard
    assert '--output-root "$${BLUEPRINT_GPU_SPEND_GUARD_OUTPUT_ROOT}"' in guard
    assert "blueprint_pipeline.provider_billing_reconciler" in billing


def test_provider_billing_reconciler_timer_is_deployed_for_accounting_closure() -> None:
    """The provider billing reconciler must be part of the deployed unit set.

    Its evidence is the accounting-closure feed the capacity/credit guard reads;
    an enabled-but-unmanaged timer drifts inactive after a deploy or reboot (it
    last fired weeks before its omission was noticed), leaving spend accounting
    stale on a control plane that is meant to run hands-off with real money.
    Deploying the service + timer makes every deploy install, enable, and start
    them, so the ten-minute reconciliation keeps running.
    """
    deployer = _text("scripts/deploy_control_plane_commit.py")
    assert '"blueprint-provider-billing-reconciler.service",' in deployer
    assert '"blueprint-provider-billing-reconciler.timer",' in deployer
