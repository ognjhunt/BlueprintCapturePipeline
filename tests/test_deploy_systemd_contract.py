import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
TERRAFORM_MAIN = REPO_ROOT / "deploy" / "terraform" / "main.tf"
ROOT_DOCKERFILE = REPO_ROOT / "Dockerfile"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_live_pipeline_control_plane.sh"
ACTIVE_LAUNCH_DOCS_AND_COMMAND_SOURCES = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "PAID_MARKETPLACE_BETA_LAUNCH_GATE.md",
    REPO_ROOT / "docs" / "FIRST_GPU_E2E_RUNBOOK.md",
    REPO_ROOT / "src" / "blueprint_pipeline" / "first_gpu_run_packet.py",
    REPO_ROOT / "src" / "blueprint_pipeline" / "g1_controlled_proof_setup.py",
    REPO_ROOT / "src" / "blueprint_pipeline" / "wam_vla_policy_endpoint_setup.py",
)


def _read(name: str) -> str:
    return (SYSTEMD_DIR / name).read_text(encoding="utf-8")


def test_active_launch_docs_and_generated_commands_do_not_embed_laptop_paths():
    forbidden = (
        "/Users/nijelhunt_1/workspace/BlueprintCapturePipeline",
        "/Users/nijelhunt_1/workspace/Blueprint-WebApp",
        "/Users/nijelhunt_1/workspace/BlueprintCapture",
    )
    for path in ACTIVE_LAUNCH_DOCS_AND_COMMAND_SOURCES:
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{needle} leaked into {path.relative_to(REPO_ROOT)}"


def test_production_systemd_units_set_fail_closed_runtime_posture():
    for unit in (
        "blueprint-pipeline-intake.service",
        "blueprint-pipeline-control-plane.service",
        "blueprint-pubsub-handoff-listener.service",
    ):
        text = _read(unit)

        assert "/Users/nijelhunt_1/" not in text
        assert "BLUEPRINT_PIPELINE_REPO=/opt/blueprint/BlueprintCapturePipeline" in text
        assert "BLUEPRINT_LAUNCH_PROOF_MODE=production" in text
        assert "PRIVACY_PIPELINE_ENABLED=true" in text
        assert "PRIVACY_FAIL_CLOSED=true" in text
        assert "PIPELINE_SYNC_REQUIRED=true" in text
        assert "RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO=true" in text
        assert "blueprint_pipeline.production_runtime_env_guard" in text
        if unit == "blueprint-pipeline-control-plane.service":
            assert "BLUEPRINT_OPERATOR_ALERT_REQUIRE_WEBHOOK=true" in text


def test_pubsub_handoff_listener_has_repeated_deployed_runner():
    service = _read("blueprint-pubsub-handoff-listener.service")
    timer = _read("blueprint-pubsub-handoff-listener.timer")
    env_example = _read("pipeline-control-plane.env.example")
    installer = INSTALL_SCRIPT.read_text(encoding="utf-8")

    assert "blueprint_pipeline.pubsub_handoff_listener" in service
    assert "--subscription" in service
    assert "BLUEPRINT_PUBSUB_HANDOFF_STAGE_CONTROL_PLANE=true" in service
    assert "BLUEPRINT_PUBSUB_HANDOFF_SKIP_RUN_E2E=true" in service
    assert "BLUEPRINT_PUBSUB_HANDOFF_SUBSCRIPTION=blueprint-pipeline-handoff-listener" in env_example
    assert "BLUEPRINT_PUBSUB_HANDOFF_STAGE_CONTROL_PLANE=true" in env_example
    assert "BLUEPRINT_PUBSUB_HANDOFF_SKIP_RUN_E2E=true" in env_example
    assert (
        "BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX="
        "/var/lib/blueprint/pipeline-control-plane/robot-eval-job-requests"
    ) in env_example
    assert (
        "BLUEPRINT_LIVE_PIPELINE_INTAKE_WORK_DIR="
        "/var/lib/blueprint/pipeline-control-plane/incoming_webapp_job_requests"
    ) in env_example
    assert "OnUnitActiveSec=1min" in timer
    assert "Unit=blueprint-pubsub-handoff-listener.service" in timer
    assert "blueprint-pubsub-handoff-listener.service" in installer
    assert "blueprint-pubsub-handoff-listener.timer" in installer
    assert "systemctl enable --now blueprint-pubsub-handoff-listener.timer" in installer
    assert "${STATE_DIR}/robot-eval-job-requests" in installer
    assert "${STATE_DIR}/incoming_webapp_job_requests" in installer
    assert "${HANDOFF_DIR}" in installer


def test_control_plane_postcheck_pages_or_fails_blocked_manifests():
    service = _read("blueprint-pipeline-control-plane.service")
    postcheck = _read("blueprint-control-plane-postchecks.sh")
    env_example = _read("pipeline-control-plane.env.example")

    assert "deploy/systemd/blueprint-control-plane-postchecks.sh" in service
    assert "blueprint_pipeline.live_pipeline_proof_audit" in postcheck
    assert "blueprint_pipeline.live_pipeline_manifest_alert" in postcheck
    assert "BLUEPRINT_OPERATOR_ALERT_REQUIRE_WEBHOOK=true" in env_example
    assert "BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL" in env_example


def test_terraform_alert_policies_require_notification_channels_or_explicit_waiver():
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")

    assert 'variable "allow_empty_monitoring_notification_channels"' in text
    assert "notification_channels = var.monitoring_notification_channels" in text
    assert text.count(
        "var.allow_empty_monitoring_notification_channels || length(var.monitoring_notification_channels) > 0"
    ) >= 3
    assert "Set allow_empty_monitoring_notification_channels=true only for dry-run plans." in text


def test_terraform_privacy_runners_are_private_and_invoked_by_named_principals():
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")

    assert "privacy_runner_public_invoker" not in text
    assert 'member   = "allUsers"' not in text
    assert 'member   = "allAuthenticatedUsers"' not in text
    assert 'resource "google_cloud_run_service_iam_member" "privacy_runner_invoker"' in text
    assert "serviceAccount:${google_service_account.pipeline_runner.email}" in text
    assert 'variable "additional_privacy_runner_invoker_members"' in text
    assert 'name  = "BLUEPRINT_CLOUD_RUN_IAM_AUTH_ENABLED"' in text
    assert 'value = "true"' in text


def test_main_deploy_image_includes_ffmpeg_for_clip_and_keyframe_lanes():
    text = ROOT_DOCKERFILE.read_text(encoding="utf-8")

    assert "ffmpeg" in text


def test_ci_workflows_checkout_the_pinned_blueprint_contracts_revision():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"BlueprintContracts\.git@([0-9a-f]{40})", pyproject)
    assert match is not None
    pinned_ref = match.group(1)

    for workflow in (
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / ".github" / "workflows" / "full-test-lane.yml",
        REPO_ROOT / ".github" / "workflows" / "sim-only-local-gate.yml",
    ):
        text = workflow.read_text(encoding="utf-8")
        assert f"CONTRACTS_REF: {pinned_ref}" in text
        assert "CONTRACTS_REF: ${{ github.head_ref || github.ref_name }}" not in text
        assert "CONTRACTS_REF: ${{ github.ref_name }}" not in text


def test_sim_only_gate_uses_explicit_webapp_ref_with_launch_fixes():
    text = (REPO_ROOT / ".github" / "workflows" / "sim-only-local-gate.yml").read_text(
        encoding="utf-8"
    )

    assert "WEBAPP_REF: ${{ inputs.webapp_ref || 'main' }}" in text
    assert "WEBAPP_REF: ${{ inputs.webapp_ref || github.head_ref || github.ref_name }}" not in text
    assert "launch-robot-eval-delivery-forwarding-fixes" not in text


def test_sim_only_gate_uses_headless_linux_mujoco_rendering():
    text = (REPO_ROOT / ".github" / "workflows" / "sim-only-local-gate.yml").read_text(
        encoding="utf-8"
    )

    assert "runs-on: ubuntu-latest" in text
    assert "MUJOCO_GL: osmesa" in text
    assert "libosmesa6" in text
    assert "runs-on: macos-latest" not in text
    assert "MUJOCO_GL: glfw" not in text
