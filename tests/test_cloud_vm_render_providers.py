"""No-network contract tests for GCP/AWS paid GPU render adapters."""
from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.cloud_vm_render_providers import AWSRenderProvider, GCPRenderProvider
from blueprint_pipeline.gpu_render_providers import RenderLaunchSpec
from blueprint_pipeline import provider_closure_audit
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)


_ORIGINAL_CLOUD_LAUNCHES = {
    AWSRenderProvider: AWSRenderProvider.launch,
    GCPRenderProvider: GCPRenderProvider.launch,
}

@pytest.fixture(autouse=True)
def _issue_test_only_provider_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    admission = build_paid_lane_admission(resource_class="gpu_render", blockers=[])
    grant = require_paid_resource_admission(
        admission,
        resource_class="gpu_render",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
    for provider_class, original in _ORIGINAL_CLOUD_LAUNCHES.items():
        def granted_launch(self, *args, _original=original, **kwargs):
            kwargs.setdefault("paid_resource_admission_grant", grant)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(provider_class, "launch", granted_launch)


def _spec() -> RenderLaunchSpec:
    return RenderLaunchSpec(
        name="blueprint-gpu-test",
        image="registry.example/worker@sha256:" + "a" * 64,
        env={"BLUEPRINT_EVAL_MANIFEST_URI": "https://input.invalid/signed"},
        bootstrap_argv=["-lc", "run-worker"],
    )


def _gcp_env(monkeypatch) -> None:
    values = {
        "BLUEPRINT_GCP_PROJECT": "blueprint-test",
        "BLUEPRINT_GCP_ZONE": "us-central1-a",
        "BLUEPRINT_GCP_MACHINE_TYPE": "g2-standard-8",
        "BLUEPRINT_GCP_SOURCE_IMAGE": "projects/test/global/images/gpu-worker-v1",
        "BLUEPRINT_GCP_NETWORK": "blueprint-workers",
        "BLUEPRINT_GCP_SUBNETWORK": "projects/blueprint-test/regions/us-central1/subnetworks/workers",
        "BLUEPRINT_GCP_SERVICE_ACCOUNT": "gpu-worker@blueprint-test.iam.gserviceaccount.com",
        "BLUEPRINT_GCP_GPU_QUOTA_METRIC": "NVIDIA_L4_GPUS",
        "BLUEPRINT_GCP_HOURLY_RATE_USD": "0.71",
        "BLUEPRINT_GCP_MAX_HOURLY_RATE_USD": "1.00",
        "BLUEPRINT_GCP_PRIVATE_EGRESS_READY": "true",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _aws_env(monkeypatch) -> None:
    values = {
        "BLUEPRINT_AWS_ACCOUNT_ID": "123456789012",
        "BLUEPRINT_AWS_REGION": "us-east-1",
        "BLUEPRINT_AWS_INSTANCE_TYPE": "g6e.2xlarge",
        "BLUEPRINT_AWS_AMI_ID": "ami-0123456789abcdef0",
        "BLUEPRINT_AWS_SUBNET_ID": "subnet-1234",
        "BLUEPRINT_AWS_SECURITY_GROUP_IDS": "sg-1234",
        "BLUEPRINT_AWS_IAM_INSTANCE_PROFILE_ARN": "arn:aws:iam::123456789012:instance-profile/blueprint-gpu-worker",
        "BLUEPRINT_AWS_HOURLY_RATE_USD": "1.86",
        "BLUEPRINT_AWS_MAX_HOURLY_RATE_USD": "2.00",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_gcp_build_request_is_explicit_private_and_auto_deleting(tmp_path: Path, monkeypatch) -> None:
    _gcp_env(monkeypatch)
    request = GCPRenderProvider().build_request(_spec(), tmp_path)
    assert request["configuration_blockers"] == []
    body = request["instance_body"]
    assert body["machineType"].endswith("/g2-standard-8")
    assert body["networkInterfaces"][0].get("accessConfigs") is None
    assert body["disks"][0]["autoDelete"] is True
    assert body["deletionProtection"] is False
    assert body["serviceAccounts"][0]["email"].startswith("gpu-worker@")
    assert body["metadata"]["items"][0]["key"] == "startup-script"
    startup = body["metadata"]["items"][0]["value"]
    assert "docker pull" not in startup
    assert "docker login" not in startup
    assert "test -f /etc/blueprint/worker-image-ref" in startup
    assert "docker image inspect" in startup
    assert body["disks"][0]["initializeParams"]["diskType"].endswith("/pd-balanced")


def test_gcp_gcloud_cli_auth_mode_uses_active_cli_token_without_recording_it(
    monkeypatch,
) -> None:
    _gcp_env(monkeypatch)
    monkeypatch.setenv("BLUEPRINT_GCP_AUTH_MODE", "gcloud_cli")

    class Result:
        stdout = "access-token-value\n"

    monkeypatch.setattr(
        "blueprint_pipeline.cloud_vm_render_providers.subprocess.run",
        lambda *args, **kwargs: Result(),
    )
    available = GCPRenderProvider().available()
    assert available["available"] is True, available
    assert available["credentials_source"] == "gcloud_cli"
    assert "access-token-value" not in str(available)


def test_gcp_fractional_g4_requires_vgpu_driver_and_uses_hyperdisk(
    monkeypatch, tmp_path: Path
) -> None:
    _gcp_env(monkeypatch)
    monkeypatch.setenv("BLUEPRINT_GCP_MACHINE_TYPE", "g4-standard-24")
    monkeypatch.setenv("BLUEPRINT_GCP_GPU_QUOTA_METRIC", "NVIDIA_RTX_PRO_6000_GPUS")
    monkeypatch.setenv("BLUEPRINT_GCP_GPU_QUOTA_UNITS", "1")
    request = GCPRenderProvider().build_request(_spec(), tmp_path)
    assert "gcp_fractional_vgpu_driver_unverified" in request["configuration_blockers"]
    assert request["gpu_quota_units"] == 1.0
    assert request["instance_body"]["disks"][0]["initializeParams"]["diskType"].endswith(
        "/hyperdisk-balanced"
    )

    monkeypatch.setenv("BLUEPRINT_GCP_FRACTIONAL_VGPU_DRIVER_READY", "true")
    request = GCPRenderProvider().build_request(_spec(), tmp_path)
    assert request["configuration_blockers"] == []


def test_gcp_preflight_verifies_named_gpu_quota(monkeypatch, tmp_path: Path) -> None:
    _gcp_env(monkeypatch)
    provider = GCPRenderProvider()
    request = provider.build_request(_spec(), tmp_path)

    def fake_call(method, path, body=None, *, timeout=90):
        if "/regions/" in path:
            return 200, {"quotas": [{"metric": "NVIDIA_L4_GPUS", "limit": 2, "usage": 1}]}
        return 200, {"name": "ok"}

    monkeypatch.setattr(provider, "_call", fake_call)
    result = provider.capacity_preflight(request)
    assert result["status"] == "available"
    assert result["quota_verified"] is True
    assert result["checks"]["regional_quota"]["required_gpu_count"] == 1.0


def test_gcp_spot_uses_service_usage_quota_when_legacy_region_row_absent(
    monkeypatch, tmp_path: Path
) -> None:
    _gcp_env(monkeypatch)
    monkeypatch.setenv("BLUEPRINT_GCP_PROVISIONING_MODEL", "SPOT")
    monkeypatch.setenv(
        "BLUEPRINT_GCP_GPU_QUOTA_METRIC",
        "compute.googleapis.com/preemptible_nvidia_rtx_pro_6000_gpus",
    )
    provider = GCPRenderProvider()
    request = provider.build_request(_spec(), tmp_path)

    def fake_call(method, path, body=None, *, timeout=90):
        if "/regions/" in path:
            return 200, {"quotas": []}
        return 200, {"name": "ok"}

    monkeypatch.setattr(provider, "_call", fake_call)
    monkeypatch.setattr(
        provider,
        "_service_usage_call",
        lambda *args, **kwargs: (
            200,
            {
                "consumerQuotaLimits": [
                    {
                        "quotaBuckets": [
                            {
                                "effectiveLimit": "1",
                                "dimensions": {"region": "us-central1"},
                            }
                        ]
                    }
                ]
            },
        ),
    )
    result = provider.capacity_preflight(request)
    assert result["status"] == "available"
    assert result["checks"]["regional_quota"]["source"] == "service_usage"
    assert request["instance_body"]["scheduling"]["provisioningModel"] == "SPOT"
    assert request["instance_body"]["scheduling"]["instanceTerminationAction"] == "DELETE"


def test_gcp_launch_never_mutates_without_guard(monkeypatch, tmp_path: Path) -> None:
    _gcp_env(monkeypatch)
    provider = GCPRenderProvider()
    request = provider.build_request(_spec(), tmp_path)
    monkeypatch.setattr(
        provider,
        "_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no API call")),
    )
    result = provider.launch(tmp_path, request)
    assert result["status"] == "blocked"
    assert "gcp_render_prelaunch_spend_guard_missing" in result["blockers"]


def test_gcp_empty_inventory_response_is_confirmed_zero(monkeypatch) -> None:
    _gcp_env(monkeypatch)
    provider = GCPRenderProvider()
    monkeypatch.setattr(provider, "_call", lambda *args, **kwargs: (200, {}))
    result = provider.billable_inventory(name_prefix="blueprint-gpu-")
    assert result["api_confirmed"] is True
    assert result["live_resource_count"] == 0


class _FakeEC2:
    def describe_instance_types(self, **kwargs):
        return {"InstanceTypes": [{"InstanceType": "g6e.2xlarge", "VCpuInfo": {"DefaultVCpus": 8}}]}

    def describe_instance_type_offerings(self, **kwargs):
        return {"InstanceTypeOfferings": [{"InstanceType": "g6e.2xlarge", "Location": "us-east-1"}]}

    def describe_images(self, **kwargs):
        return {"Images": [{"ImageId": kwargs["ImageIds"][0]}]}

    def describe_subnets(self, **kwargs):
        return {"Subnets": [{"SubnetId": kwargs["SubnetIds"][0]}]}

    def describe_security_groups(self, **kwargs):
        return {"SecurityGroups": [{"GroupId": value} for value in kwargs["GroupIds"]]}

    def run_instances(self, **kwargs):
        return {"Instances": [{"InstanceId": "i-0123456789abcdef0"}]}

    def describe_instances(self, **kwargs):
        return {"Reservations": []}

    def terminate_instances(self, **kwargs):
        return {"TerminatingInstances": []}


class _FakeQuota:
    def get_service_quota(self, **kwargs):
        return {"Quota": {"Value": 16.0}}


class _FakeSTS:
    def get_caller_identity(self):
        return {"Account": "123456789012"}


class _FakeIAM:
    def get_instance_profile(self, **kwargs):
        return {
            "InstanceProfile": {
                "Arn": "arn:aws:iam::123456789012:instance-profile/blueprint-gpu-worker"
            }
        }


def test_aws_preflight_binds_account_network_image_and_quota(monkeypatch, tmp_path: Path) -> None:
    _aws_env(monkeypatch)
    provider = AWSRenderProvider()
    monkeypatch.setattr(provider, "_ec2", lambda: _FakeEC2())
    monkeypatch.setattr(provider, "_service_quotas", lambda: _FakeQuota())
    monkeypatch.setattr(provider, "_sts", lambda: _FakeSTS())
    monkeypatch.setattr(provider, "_iam", lambda: _FakeIAM())
    request = provider.build_request(_spec(), tmp_path)
    result = provider.capacity_preflight(request)
    assert result["status"] == "available"
    assert result["checks"]["account"] is True
    assert result["checks"]["quota"]["required_vcpus"] == 8
    assert request["run_instances"]["MetadataOptions"]["HttpTokens"] == "required"
    assert request["run_instances"]["ClientToken"]
    assert request["run_instances"]["BlockDeviceMappings"][0]["Ebs"]["DeleteOnTermination"] is True
    startup = request["run_instances"]["UserData"]
    assert "docker pull" not in startup
    assert "docker login" not in startup
    assert "test -f /etc/blueprint/worker-image-ref" in startup


def test_aws_launch_and_inventory_contract(monkeypatch, tmp_path: Path) -> None:
    _aws_env(monkeypatch)
    provider = AWSRenderProvider()
    monkeypatch.setattr(provider, "_ec2", lambda: _FakeEC2())
    monkeypatch.setattr(provider, "capacity_preflight", lambda request: {"status": "available", "blockers": []})
    request = provider.build_request(_spec(), tmp_path)
    request["prelaunch_spend_guard"] = {
        "required_before_provider_launch": True,
        "can_launch": True,
    }
    result = provider.launch(tmp_path, request)
    assert result["status"] == "launched"
    assert result["instance_id"] == "i-0123456789abcdef0"
    assert (tmp_path / "started_aws_instance_id.txt").read_text() == result["instance_id"]
    inventory = provider.billable_inventory(name_prefix="blueprint-gpu-test")
    assert inventory["api_confirmed"] is True
    assert inventory["live_resource_count"] == 0


def test_cloud_adapters_fail_closed_when_account_configuration_missing(monkeypatch, tmp_path: Path) -> None:
    names = (
        "BLUEPRINT_GCP_PROJECT",
        "BLUEPRINT_GCP_AUTH_MODE",
        "BLUEPRINT_GCP_ZONE",
        "BLUEPRINT_GCP_MACHINE_TYPE",
        "BLUEPRINT_GCP_SOURCE_IMAGE",
        "BLUEPRINT_GCP_NETWORK",
        "BLUEPRINT_GCP_SUBNETWORK",
        "BLUEPRINT_GCP_SERVICE_ACCOUNT",
        "BLUEPRINT_GCP_GPU_QUOTA_METRIC",
        "BLUEPRINT_GCP_HOURLY_RATE_USD",
        "BLUEPRINT_GCP_MAX_HOURLY_RATE_USD",
        "BLUEPRINT_GCP_PRIVATE_EGRESS_READY",
        "BLUEPRINT_GCP_FRACTIONAL_VGPU_DRIVER_READY",
        "BLUEPRINT_GCP_PROVISIONING_MODEL",
        "BLUEPRINT_AWS_ACCOUNT_ID",
        "BLUEPRINT_AWS_REGION",
        "BLUEPRINT_AWS_INSTANCE_TYPE",
        "BLUEPRINT_AWS_AMI_ID",
        "BLUEPRINT_AWS_SUBNET_ID",
        "BLUEPRINT_AWS_SECURITY_GROUP_IDS",
        "BLUEPRINT_AWS_IAM_INSTANCE_PROFILE_ARN",
        "BLUEPRINT_AWS_HOURLY_RATE_USD",
        "BLUEPRINT_AWS_MAX_HOURLY_RATE_USD",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    gcp = GCPRenderProvider().build_request(_spec(), tmp_path)
    aws = AWSRenderProvider().build_request(_spec(), tmp_path)
    assert "gcp_project_missing" in gcp["configuration_blockers"]
    assert "aws_account_id_missing" in aws["configuration_blockers"]


def test_provider_closure_reports_gcp_and_aws_credentials_without_reading_values(
    monkeypatch, tmp_path: Path
) -> None:
    gcp_file = tmp_path / "gcp.json"
    gcp_file.write_text("credential-material", encoding="utf-8")
    aws_file = tmp_path / "aws-credentials"
    aws_file.write_text("credential-material", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(gcp_file))
    monkeypatch.setenv("BLUEPRINT_GCP_PROJECT", "blueprint-test")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(aws_file))
    monkeypatch.setenv("BLUEPRINT_AWS_ACCOUNT_ID", "123456789012")
    gcp = provider_closure_audit._credential_audit("gcp")
    aws = provider_closure_audit._credential_audit("aws")
    assert gcp["credential_configured"] is True
    assert aws["credential_configured"] is True
    assert gcp["raw_secret_values_read"] is False
    assert aws["raw_secret_values_recorded"] is False
