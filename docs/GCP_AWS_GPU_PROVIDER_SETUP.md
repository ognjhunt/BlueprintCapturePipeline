# GCP and AWS paid-GPU provider setup

GCP Compute Engine and AWS EC2 implement the same `GpuRenderProvider` contract
as RunPod, Vast, and DigitalOcean: request construction, read-only quota/capacity
preflight, guarded launch, inspection, scoped inventory, permanent teardown,
fleet spend coverage, and closure credential reporting.

No account resources are inferred or created. An operator must explicitly choose
the account/project, location, VM shape, image, private network, worker identity,
registry mode, storage behavior, and hourly price. Missing or unverifiable values
block before a mutating API call.

## Shared safety contract

- Use a digest-pinned worker image.
- Signed input/output URLs remain the provider-neutral data transport.
- A passing prelaunch spend guard is mandatory for `launch()`.
- GCP disks and AWS root EBS are encrypted/auto-deleted with the VM.
- GCP receives no external IP. AWS uses only the named subnet/security groups.
- `terminate()` plus fresh API-confirmed zero inventory is spend closure; `stop()` is not.
- Production host images already contain the exact worker digest. VM startup
  verifies `/etc/blueprint/worker-image-ref` and the local Docker cache; it does
  not log in to a registry or pull image layers.
- Capacity preflight is advisory; only create proves allocation and post-delete
  inventory proves absence.

## GCP Compute Engine

The launcher identity needs scoped read access to project/zone/region, quota,
machine/accelerator, image and network resources; instance and disk lifecycle
permissions; network use; and `iam.serviceAccounts.actAs` on the selected worker.
For Artifact Registry, give the worker service account repository-scoped Reader.

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.blueprint-secrets/gcp_gpu_launcher.json"
export BLUEPRINT_GCP_PROJECT="your-project-id"
export BLUEPRINT_GCP_ZONE="us-central1-a"
export BLUEPRINT_GCP_MACHINE_TYPE="g2-standard-8"
export BLUEPRINT_GCP_SOURCE_IMAGE="projects/your-project/global/images/blueprint-gpu-host-v1"
export BLUEPRINT_GCP_NETWORK="blueprint-workers"
export BLUEPRINT_GCP_SUBNETWORK="projects/your-project/regions/us-central1/subnetworks/blueprint-gpu-workers"
export BLUEPRINT_GCP_SERVICE_ACCOUNT="blueprint-gpu-worker@your-project.iam.gserviceaccount.com"
export BLUEPRINT_GCP_GPU_QUOTA_METRIC="NVIDIA_L4_GPUS"
export BLUEPRINT_GCP_GPU_QUOTA_UNITS="1"
export BLUEPRINT_GCP_PRIVATE_EGRESS_READY="true"
export BLUEPRINT_GCP_HOURLY_RATE_USD="<verified-current-all-in-rate>"
export BLUEPRINT_GCP_MAX_HOURLY_RATE_USD="<authorized-ceiling>"
```

When the workstation's active `gcloud` user is the intentionally selected
launcher identity, use `BLUEPRINT_GCP_AUTH_MODE=gcloud_cli` instead of exporting
a service-account file. The adapter obtains a short-lived access token for each
API operation and never records it. Application Default Credentials remain the
default for unattended runners.

For an attached rather than integrated GPU, also set
`BLUEPRINT_GCP_ACCELERATOR_TYPE` and `BLUEPRINT_GCP_ACCELERATOR_COUNT`.
`BLUEPRINT_GCP_PRIVATE_EGRESS_READY=true` is an operator assertion that the
selected private subnet can reach the signed artifact endpoints without an
external VM address. Worker image pulls happen during the host-image bake, not
VM startup. The adapter fails closed if the egress assertion is absent.

For fractional G4 shapes (`g4-standard-6`, `g4-standard-12`, and
`g4-standard-24`), use the provider-reported accelerator quota count (currently
one integrated accelerator for `g4-standard-24`) and only assert driver
readiness after the host image has been verified with Google's required vGPU
driver. For the lower-cost Spot path, bind the preemptible quota metric and
provisioning model explicitly:

```bash
export BLUEPRINT_GCP_GPU_QUOTA_METRIC="compute.googleapis.com/preemptible_nvidia_rtx_pro_6000_gpus"
export BLUEPRINT_GCP_GPU_QUOTA_UNITS="1"
export BLUEPRINT_GCP_PROVISIONING_MODEL="SPOT"
export BLUEPRINT_GCP_FRACTIONAL_VGPU_DRIVER_READY="true"
```

G4 requests default to `hyperdisk-balanced`; G4 does not support Persistent
Disk. Other machine families default to `pd-balanced`. Override explicitly with
`BLUEPRINT_GCP_BOOT_DISK_TYPE` only when the selected machine supports it.
Registry settings are retained for the separate host-image builder identity;
the production VM startup script never logs in or pulls. For private Artifact
Registry images during baking:

```bash
export BLUEPRINT_GCP_REGISTRY_AUTH="gcp_artifact_registry"
export BLUEPRINT_GCP_REGISTRY_HOST="us-central1-docker.pkg.dev"
```

`public` is the default registry mode.

## AWS EC2

The launcher uses the standard boto3 credential chain. Prefer a named profile,
web-identity role, or container role. `BLUEPRINT_AWS_ACCOUNT_ID` is mandatory;
STS caller identity must match it. The launcher principal needs scoped EC2 read
and lifecycle permissions, Service Quotas read, `ec2:CreateTags`, and
`iam:PassRole` only for the selected worker profile.

```bash
export AWS_PROFILE="blueprint-gpu-launcher"
export BLUEPRINT_AWS_ACCOUNT_ID="123456789012"
export BLUEPRINT_AWS_REGION="us-east-1"
export BLUEPRINT_AWS_INSTANCE_TYPE="g6e.2xlarge"
export BLUEPRINT_AWS_AMI_ID="ami-..."
export BLUEPRINT_AWS_SUBNET_ID="subnet-..."
export BLUEPRINT_AWS_SECURITY_GROUP_IDS="sg-..." # comma-separated
export BLUEPRINT_AWS_IAM_INSTANCE_PROFILE_ARN="arn:aws:iam::123456789012:instance-profile/blueprint-gpu-worker"
export BLUEPRINT_AWS_HOURLY_RATE_USD="<verified-current-all-in-rate>"
export BLUEPRINT_AWS_MAX_HOURLY_RATE_USD="<authorized-ceiling>"
```

`BLUEPRINT_AWS_KEY_NAME` is optional. For a private ECR image:

```bash
export BLUEPRINT_AWS_REGISTRY_AUTH="aws_ecr"
export BLUEPRINT_AWS_REGISTRY_HOST="123456789012.dkr.ecr.us-east-1.amazonaws.com"
```

The selected AMI must contain the NVIDIA driver, Docker plus NVIDIA Container
Toolkit, Python 3, the exact digest-pinned worker image, and matching
`/etc/blueprint/worker-image-ref`. Customer VM startup performs no ECR login or
image pull.

The production warm-pool architecture, Packer host-image build, readiness
contract, and promotion gate are in
[`docs/runbooks/production-gpu-startup-and-warm-pool.md`](runbooks/production-gpu-startup-and-warm-pool.md).

## Read-only verification and launch wiring

```bash
python -m blueprint_pipeline.isaac_particlefield_render_job --list-providers
python scripts/gpu_spend_guard.py --require-provider gcp --require-provider aws \
  --json-report /tmp/gpu_spend_guard.json
```

Both providers are accepted by the render job:

```bash
python -m blueprint_pipeline.isaac_particlefield_render_job --provider gcp <required-args>
python -m blueprint_pipeline.isaac_particlefield_render_job --provider aws <required-args>
```

After an attempt, terminate and rerun the guard. Closure requires `succeeded`
inventory with zero matching resources. A submitted delete alone is insufficient.

## Proof boundary

Local tests prove request shape, boundary validation, quota/account binding,
guard enforcement, inventory normalization, and teardown behavior against fake
APIs. They do not prove live capacity, image boot, registry IAM, valid runtime
artifacts, or real-account teardown. Those require fresh provider evidence.
