import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
TERRAFORM_MAIN = REPO_ROOT / "deploy" / "terraform" / "main.tf"
TERRAFORM_TFVARS_EXAMPLE = REPO_ROOT / "deploy" / "terraform" / "terraform.tfvars.example"
ROOT_DOCKERFILE = REPO_ROOT / "Dockerfile"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_live_pipeline_control_plane.sh"
PIPELINE_DEPLOY_SCRIPT = REPO_ROOT / "deploy" / "scripts" / "deploy.sh"
FULL_TEST_LANE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "full-test-lane.yml"
CI_REQUIRED_CHECKS_DOC = REPO_ROOT / "docs" / "CI_REQUIRED_CHECKS.md"
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


def _terraform_resource_body(text: str, resource_type: str, name: str) -> str:
    needle = f'resource "{resource_type}" "{name}" {{'
    start = text.index(needle)
    body_start = start + len(needle)
    depth = 1
    index = body_start
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[body_start:index]
        index += 1
    raise AssertionError(f"Unclosed Terraform resource block: {resource_type}.{name}")


def _terraform_variable_body(text: str, name: str) -> str:
    needle = f'variable "{name}" {{'
    start = text.index(needle)
    body_start = start + len(needle)
    depth = 1
    index = body_start
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[body_start:index]
        index += 1
    raise AssertionError(f"Unclosed Terraform variable block: {name}")


def test_active_launch_docs_and_generated_commands_do_not_embed_laptop_paths():
    for path in ACTIVE_LAUNCH_DOCS_AND_COMMAND_SOURCES:
        text = path.read_text(encoding="utf-8")
        assert re.search(r"/Users/[^/]+/", text) is None, (
            f"personal absolute path leaked into {path.relative_to(REPO_ROOT)}"
        )


def test_production_systemd_units_set_fail_closed_runtime_posture():
    for unit in (
        "blueprint-pipeline-intake.service",
        "blueprint-pipeline-control-plane.service",
        "blueprint-pubsub-handoff-listener.service",
    ):
        text = _read(unit)

        assert re.search(r"/Users/[^/]+/", text) is None
        assert "BLUEPRINT_PIPELINE_REPO=/opt/blueprint/BlueprintCapturePipeline" in text
        assert "BLUEPRINT_LAUNCH_PROOF_MODE=production" in text
        assert "PRIVACY_PIPELINE_ENABLED=true" in text
        assert "PRIVACY_FAIL_CLOSED=true" in text
        assert "PIPELINE_SYNC_REQUIRED=true" in text
        assert "RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO=true" in text
        assert "blueprint_pipeline.production_runtime_env_guard" in text
        if unit == "blueprint-pipeline-control-plane.service":
            assert "BLUEPRINT_OPERATOR_ALERT_REQUIRE_WEBHOOK=true" in text


def test_production_systemd_units_run_nonroot_with_strict_resource_isolation() -> None:
    required_controls = (
        "User=blueprint",
        "Group=blueprint",
        "UMask=0077",
        "NoNewPrivileges=true",
        "PrivateTmp=true",
        "PrivateDevices=true",
        "ProtectSystem=strict",
        "ProtectHome=true",
        "ProtectKernelTunables=true",
        "ProtectKernelModules=true",
        "ProtectKernelLogs=true",
        "ProtectControlGroups=true",
        "RestrictSUIDSGID=true",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "SystemCallFilter=@system-service",
        "ReadWritePaths=/var/lib/blueprint",
        "TasksMax=512",
        "MemoryMax=8G",
        "CPUQuota=200%",
    )
    for unit in SYSTEMD_DIR.glob("*.service"):
        text = unit.read_text(encoding="utf-8")
        for control in required_controls:
            assert control in text, (unit.name, control)

    installer = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert 'SERVICE_USER="${SERVICE_USER:-blueprint}"' in installer
    assert 'SERVICE_GROUP="${SERVICE_GROUP:-blueprint}"' in installer
    assert '"${HANDOFF_DIR}"' in installer
    assert '-m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}"' in installer
    assert 'install -o root -g "${SERVICE_GROUP}" -m 0640' in installer
    assert 'install -d -m 0755 "${SYSTEMD_DIR}" "${ENV_DIR}" "${HANDOFF_DIR}"' not in installer

    baseline = (REPO_ROOT / "docs" / "SYSTEMD_SECURITY_BASELINE.md").read_text(encoding="utf-8")
    assert "systemd-analyze security" in baseline
    assert "threshold=40" in baseline


def test_pubsub_handoff_listener_has_repeated_deployed_runner():
    service = _read("blueprint-pubsub-handoff-listener.service")
    timer = _read("blueprint-pubsub-handoff-listener.timer")
    env_example = _read("pipeline-control-plane.env.example")
    installer = INSTALL_SCRIPT.read_text(encoding="utf-8")

    assert "blueprint_pipeline.pubsub_handoff_listener" in service
    assert "--subscription" in service
    assert "BLUEPRINT_PUBSUB_HANDOFF_STAGE_CONTROL_PLANE=true" in service
    assert "BLUEPRINT_PUBSUB_HANDOFF_SKIP_RUN_E2E=true" in service
    assert (
        "BLUEPRINT_PUBSUB_HANDOFF_SUBSCRIPTION=blueprint-pipeline-handoff-listener" in env_example
    )
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
    assert (
        text.count(
            "var.allow_empty_monitoring_notification_channels || length(var.monitoring_notification_channels) > 0"
        )
        >= 4
    )
    assert "Set allow_empty_monitoring_notification_channels=true only for dry-run plans." in text


def test_terraform_queue_depth_alert_uses_beta_backpressure_thresholds():
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")
    body = _terraform_resource_body(text, "google_monitoring_alert_policy", "queue_depth")

    assert 'variable "pipeline_queue_depth_alert_threshold"' in text
    assert "default     = 50" in text
    assert 'variable "pipeline_queue_depth_alert_duration"' in text
    assert 'default     = "300s"' in text
    assert "duration        = var.pipeline_queue_depth_alert_duration" in body
    assert "threshold_value = var.pipeline_queue_depth_alert_threshold" in body
    assert "threshold_value = 100" not in body
    assert 'duration        = "600s"' not in body
    assert "per-user intake pressure" in body


def test_terraform_keeps_firestore_latency_alert_without_phantom_captures_indexes():
    # SCALE2-07: the four `captures_*` index resources targeted a literal
    # `captures` collection that no repo has ever written. The real
    # capture-record collection is creatorCaptures, whose composites (including
    # the createdAtShard scale-up companions) are owned by
    # Blueprint-WebApp/firestore.indexes.json. This repo keeps only the shared
    # database and the latency alerting.
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")
    latency_alert = _terraform_resource_body(
        text,
        "google_monitoring_alert_policy",
        "firestore_request_latency",
    )

    assert 'resource "google_firestore_index" "captures_status"' not in text
    assert 'resource "google_firestore_index" "captures_user"' not in text
    assert 'resource "google_firestore_index" "captures_status_created_at_shard"' not in text
    assert 'resource "google_firestore_index" "captures_user_created_at_shard"' not in text
    assert 'collection = "captures"' not in text

    assert "serviceruntime.googleapis.com/api/request_latencies" in latency_alert
    assert "resource.service == 'firestore.googleapis.com'" in latency_alert
    assert "| condition val() > 0.25 's'" in latency_alert
    assert "creatorCaptures created_at composite indexes" in latency_alert
    assert "Blueprint-WebApp firestore.indexes.json" in latency_alert


def test_terraform_gpu_warm_pool_defaults_stay_scale_to_zero():
    # SCALE2-06: warm pools are a deliberate owner decision, not a default.
    # One warm GPU instance is ~$1,825/month at the $2.5/GPU-hr planning rate
    # (73% of the $2,500 spend-review threshold); current volume is >10x
    # below the ~20 invocations/hour breakeven. See
    # docs/GPU_WARM_POOL_ECONOMICS_2026-07-20.md.
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")

    for variable in (
        "privacy_sam3_min_instances",
        "privacy_vip_min_instances",
        "privacy_deepprivacy2_min_instances",
        "video_to_world_min_instances",
    ):
        block = _terraform_resource_body_for_variable(text, variable)
        assert "default     = 0" in block, f"{variable} must default to scale-to-zero"
        assert "<= 2" in block, f"{variable} must stay capped at 2 warm instances"

    for key in ("sam3", "vip", "deepprivacy2", "video_to_world"):
        assert f"min_instance_count = local.privacy_runner_min_instances.{key}" in text
    # No GPU service may hardcode an always-warm floor.
    assert "min_instance_count = 1" not in text
    assert "min_instance_count = 2" not in text


def _terraform_resource_body_for_variable(text: str, name: str) -> str:
    marker = f'variable "{name}"'
    start = text.index(marker)
    end = text.index("\nvariable ", start + 1) if "\nvariable " in text[start + 1 :] else len(text)
    return text[start:end]


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


def test_privacy_runner_auth_tokens_fail_closed_before_deploy() -> None:
    terraform = TERRAFORM_MAIN.read_text(encoding="utf-8")
    deploy_script = PIPELINE_DEPLOY_SCRIPT.read_text(encoding="utf-8")
    privacy_secret = _terraform_variable_body(terraform, "privacy_runner_token_secret_name")
    video_secret = _terraform_variable_body(terraform, "video_to_world_runner_token_secret_name")

    assert 'default     = ""' not in privacy_secret
    assert "^[A-Za-z0-9_-]{1,255}$" in privacy_secret
    assert "video_to_world_runner_token_secret_name" in video_secret
    assert 'variable "privacy_runner_token"' not in terraform
    assert 'variable "pipeline_sync_token"' not in terraform
    assert "value = var.privacy_runner_token" not in terraform
    assert "value = var.pipeline_sync_token" not in terraform
    assert "value = local.video_to_world_runner_token" not in terraform
    assert 'name = "WORLDLABS_API_KEY"' in terraform
    assert "data.google_secret_manager_secret.worldlabs_api_key.secret_id" in terraform
    assert "validate_runtime_secret_references" in deploy_script
    assert "PRIVACY_RUNNER_TOKEN_SECRET_NAME" in deploy_script
    assert "VIDEO_TO_WORLD_RUNNER_TOKEN_SECRET_NAME" in deploy_script
    assert "PIPELINE_SYNC_TOKEN_SECRET_NAME" in deploy_script
    assert "WORLDLABS_API_KEY_SECRET_NAME" in deploy_script
    assert "TF_VAR_privacy_runner_token_secret_name" in deploy_script
    assert "TF_VAR_pipeline_sync_token_secret_name" in deploy_script
    assert "PRIVACY_RUNNER_TOKEN=" not in deploy_script
    assert "PIPELINE_SYNC_TOKEN=" not in deploy_script
    assert "WORLDLABS_API_KEY=" not in deploy_script
    assert "Creating terraform.tfvars" not in deploy_script


def test_terraform_state_is_remote_locked_retained_and_cmek_validated() -> None:
    terraform = TERRAFORM_MAIN.read_text(encoding="utf-8")
    deploy_script = PIPELINE_DEPLOY_SCRIPT.read_text(encoding="utf-8")
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert 'backend "gcs" {}' in terraform
    assert "validate_terraform_state_backend.py" in deploy_script
    assert "TERRAFORM_STATE_BUCKET" in deploy_script
    assert "TERRAFORM_STATE_PREFIX" in deploy_script
    assert "TERRAFORM_STATE_KMS_KEY" in deploy_script
    assert "-backend-config=kms_encryption_key=" in deploy_script
    assert "-reconfigure" in deploy_script
    assert "*.tfstate" in gitignore
    assert "*.tfstate.*" in gitignore
    assert "*.tfvars" in gitignore
    assert "*.tfplan" in gitignore


def test_full_deploy_requires_authenticated_readback_service_canaries() -> None:
    terraform = TERRAFORM_MAIN.read_text(encoding="utf-8")
    deploy_script = PIPELINE_DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert 'output "deployed_image_digests"' in terraform
    assert "run_deployment_service_canaries.py" in deploy_script
    assert "--topology-evidence" in deploy_script
    assert "--privacy-secret-name" in deploy_script
    assert "--video-secret-name" in deploy_script
    assert "Authenticated deployment service canaries did not pass" in deploy_script
    assert "authenticated_service_canaries" in deploy_script
    assert "model task success remains separately required" in deploy_script


def test_pipeline_failure_alert_counts_failures_over_five_minutes() -> None:
    terraform = TERRAFORM_MAIN.read_text(encoding="utf-8")
    body = _terraform_resource_body(
        terraform, "google_monitoring_alert_policy", "pipeline_failures"
    )

    assert 'alignment_period     = "300s"' in body
    assert 'per_series_aligner   = "ALIGN_SUM"' in body
    assert 'cross_series_reducer = "REDUCE_SUM"' in body
    assert "threshold_value = 5" in body
    assert 'comparison      = "COMPARISON_GT"' in body
    assert 'duration        = "0s"' in body
    assert "ALIGN_RATE" not in body


def test_us_only_beta_residency_is_enforced_as_code() -> None:
    terraform = TERRAFORM_MAIN.read_text(encoding="utf-8")
    tfvars_example = TERRAFORM_TFVARS_EXAMPLE.read_text(encoding="utf-8")
    deploy_script = PIPELINE_DEPLOY_SCRIPT.read_text(encoding="utf-8")
    primary = _terraform_variable_body(terraform, "primary_region")
    secondary = _terraform_variable_body(terraform, "secondary_regions")

    assert 'startswith(var.primary_region, "us-")' in primary
    assert 'startswith(region, "us-")' in secondary
    assert "europe-west1" not in secondary
    assert "europe-west1" not in tfvars_example
    assert "europe-west1" not in deploy_script
    assert "validate_beta_data_residency" in deploy_script
    assert '"$PRIMARY_REGION" != us-*' in deploy_script


def test_terraform_is_the_single_deployment_topology_owner() -> None:
    deploy_script = PIPELINE_DEPLOY_SCRIPT.read_text(encoding="utf-8")
    full_deploy = deploy_script[deploy_script.index("    # Full deployment") :]

    assert 'topology_owner: "terraform"' in deploy_script
    assert "terraform plan -input=false -detailed-exitcode" in deploy_script
    assert "Post-apply Terraform drift detected" in deploy_script
    assert "terraform output -json" in deploy_script
    assert "blueprint.terraform_topology_evidence.v1" in deploy_script
    assert "--function-only was removed" in deploy_script
    assert "    apply_terraform" in full_deploy
    assert "    setup_iam" not in full_deploy
    assert "    create_pubsub_topics" not in full_deploy
    assert "    create_cloud_tasks_queues" not in full_deploy
    assert "    create_cloud_run_jobs" not in full_deploy
    assert "    deploy_cloud_function" not in full_deploy


def test_terraform_privacy_runners_have_per_service_gpu_caps_and_spend_alert():
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")

    for variable in (
        "privacy_sam3_max_instances",
        "privacy_vip_max_instances",
        "privacy_deepprivacy2_max_instances",
        "video_to_world_max_instances",
        "gpu_runner_billable_instance_time_alert_threshold",
    ):
        assert f'variable "{variable}"' in text

    service_caps = {
        "privacy_sam3": "local.privacy_runner_max_instances.sam3",
        "privacy_vip": "local.privacy_runner_max_instances.vip",
        "privacy_deepprivacy2": "local.privacy_runner_max_instances.deepprivacy2",
        "video_to_world": "local.privacy_runner_max_instances.video_to_world",
    }
    for service, expected_cap in service_caps.items():
        body = _terraform_resource_body(text, "google_cloud_run_v2_service", service)
        assert f"max_instance_count = {expected_cap}" in body
        assert "max_instance_count = var.max_concurrent_jobs" not in body

    assert 'resource "google_monitoring_alert_policy" "gpu_runner_billable_instance_time"' in text
    assert "run.googleapis.com/container/billable_instance_time" in text
    assert "cloud_run_revision" in text
    assert "local.privacy_runner_monitoring_service_filter" in text
    assert "Sustained GPU runner billable instance time" in text


def test_terraform_declares_private_large_video_ingest_topic_for_extract_frames():
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")

    assert 'variable "capture_extract_frames_service_account_email"' in text
    assert 'resource "google_pubsub_topic" "large_video_ingest"' in text
    assert 'name   = "blueprint-large-video-ingest"' in text
    assert (
        'resource "google_pubsub_topic_iam_member" "capture_extract_frames_large_video_ingest_publisher"'
        in text
    )
    assert 'member  = "serviceAccount:${var.capture_extract_frames_service_account_email}"' in text
    assert 'output "large_video_ingest_topic"' in text


def test_terraform_declares_optional_gcp_billing_budget_for_gpu_fleet():
    text = TERRAFORM_MAIN.read_text(encoding="utf-8")
    body = _terraform_resource_body(text, "google_billing_budget", "gpu_fleet_beta")

    assert '"billingbudgets.googleapis.com"' in text
    assert 'variable "billing_account_id"' in text
    assert 'variable "gpu_fleet_billing_budget_usd"' in text
    assert 'variable "gpu_fleet_billing_budget_thresholds"' in text
    assert 'data "google_project" "current"' in text
    assert 'count = var.billing_account_id != "" ? 1 : 0' in body
    assert "billing_account = var.billing_account_id" in body
    assert 'display_name    = "Blueprint GPU Fleet Beta Budget"' in body
    assert 'projects = ["projects/${data.google_project.current.number}"]' in body
    assert 'currency_code = "USD"' in body
    assert "units         = tostring(var.gpu_fleet_billing_budget_usd)" in body
    assert "threshold_percent = threshold_rules.value" in body
    assert 'output "gpu_fleet_billing_budget"' in text


def test_systemd_spend_guard_enforces_5000_admission_lock_and_page_path():
    service = _read("blueprint-gpu-spend-guard.service")
    env_example = _read("pipeline-control-plane.env.example")
    installer = INSTALL_SCRIPT.read_text(encoding="utf-8")

    assert "BLUEPRINT_GPU_FLEET_MAX_TOTAL_SPEND_USD=5000.0" in service
    assert "BLUEPRINT_GPU_FLEET_MAX_TOTAL_SPEND_USD=500.0" not in service
    assert "--require-billing-reconciliation" in service
    assert "--admission-lock-report" in service
    assert "--admission-override" in service
    assert "ExecStopPost=" in service
    assert "blueprint_pipeline.live_pipeline_manifest_alert" in service
    assert "BLUEPRINT_GPU_SPEND_GUARD_PAGE_AUDIT" in service
    assert "BLUEPRINT_GPU_PROVIDER_SECRETS_DIR=/etc/blueprint/provider-secrets" in service
    assert "BLUEPRINT_REQUIRE_PAID_SPEND_ADMISSION_LOCK=true" in env_example
    assert "BLUEPRINT_GPU_BILLING_EXPORT=" in env_example
    assert "blueprint-gpu-spend-guard.service" in installer
    assert "blueprint-gpu-spend-guard.timer" in installer
    assert "systemctl enable --now blueprint-gpu-spend-guard.timer" in installer
    assert '"${STATE_DIR}/gpu_spend_guard"' in installer
    assert '"${PROVIDER_SECRETS_DIR}"' in installer


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


def test_full_test_lane_gates_pull_requests_and_deploy_contract():
    workflow = FULL_TEST_LANE_WORKFLOW.read_text(encoding="utf-8")
    deploy_script = PIPELINE_DEPLOY_SCRIPT.read_text(encoding="utf-8")
    required_doc = CI_REQUIRED_CHECKS_DOC.read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert 'branches: ["main"]' in workflow
    assert "schedule:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "uv run scripts/pytest_full.sh" in workflow
    assert '--junitxml="${{ runner.temp }}/blueprint-ci/full-test-lane-junit.xml"' in workflow
    assert 'FULL_TEST_LANE_REQUIRED="${FULL_TEST_LANE_REQUIRED:-true}"' in deploy_script
    assert "check_full_test_lane_deploy_gate" in deploy_script
    assert "FULL_TEST_LANE_COMMIT" in deploy_script
    assert "FULL_TEST_LANE_EVIDENCE_URI" in deploy_script
    assert "verify_deploy_release_provenance.py" in deploy_script
    assert "verify_clean_release_source" in deploy_script
    assert "status --porcelain=v1 --untracked-files=all" in deploy_script
    assert "fetch --quiet origin main" in deploy_script
    assert "refs/remotes/origin/main" in deploy_script
    assert "FULL_TEST_LANE_BYPASS_REASON" not in deploy_script
    assert "this deploy path has no text-only CI bypass" in deploy_script
    assert (
        "Full Test Lane / Full pytest lane on CPU runner passed for this exact commit"
        in deploy_script
    )
    assert "Full Test Lane / Full pytest lane on CPU runner" in required_doc
    assert "passed for that exact commit SHA" in required_doc
    assert "deploy/scripts/deploy.sh" in required_doc
    assert "weekly scheduled run is supplementary health evidence only" in required_doc


def test_release_images_are_versioned_manifested_and_rejected_if_latest():
    terraform = TERRAFORM_MAIN.read_text(encoding="utf-8")
    tfvars_example = TERRAFORM_TFVARS_EXAMPLE.read_text(encoding="utf-8")
    deploy_script = PIPELINE_DEPLOY_SCRIPT.read_text(encoding="utf-8")
    required_doc = CI_REQUIRED_CHECKS_DOC.read_text(encoding="utf-8")

    image_variables = (
        "docker_image",
        "privacy_sam3_image",
        "privacy_vip_image",
        "privacy_deepprivacy2_image",
        "video_to_world_image",
    )
    image_names = (
        "blueprint-pipeline",
        "sam3-privacy",
        "vip-privacy",
        "deepprivacy2-privacy",
        "video-to-world",
    )

    for variable in image_variables:
        body = _terraform_variable_body(terraform, variable)
        assert "default" not in body
        assert "nullable    = false" in body
        assert "^.+@sha256:[0-9a-f]{64}$" in body
        assert "non-latest versioned release tag" not in body
        assert f"var.{variable}" in body

    for image_name in image_names:
        assert f"{image_name}:latest" not in terraform
        assert f"{image_name}:latest" not in tfvars_example
        assert f"{image_name}@sha256:" in tfvars_example

    assert 'IMAGE_TAG="${IMAGE_TAG:-}"' in deploy_script
    assert 'IMAGE_TAG="${GIT_SHA:0:12}"' in deploy_script
    assert "validate_release_image_tag" in deploy_script
    assert "pin_pushed_image_digests" in deploy_script
    assert "gcloud container images describe" in deploy_script
    assert "image_summary.fully_qualified_digest" in deploy_script
    assert "require_resolved_image_matches" in deploy_script
    assert "tag/digest mismatch" in deploy_script
    assert "pin_pushed_image_digests" in deploy_script
    assert "DEPLOYMENT_MANIFEST_PATH" in deploy_script
    assert "blueprint.pipeline_deployment_manifest.v1" in deploy_script
    assert "deploy/scripts/deploy.sh --rollback --rollback-image-tag " in deploy_script
    assert "latest`/`dev`/`test`" in required_doc
    assert "pipeline-deployment-manifest.json" in required_doc
    assert "Terraform image variables require immutable `@sha256` digests" in required_doc


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
