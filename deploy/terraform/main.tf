# Blueprint Capture Pipeline - GCP Infrastructure
#
# This Terraform configuration creates all necessary GCP resources for the
# video-to-3D capture pipeline with multi-region GPU support.
#
# Usage:
#   cd deploy/terraform
#   terraform init
#   terraform plan -var="project_id=blueprint-8c1ca"
#   terraform apply -var="project_id=blueprint-8c1ca"

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }

  # Required partial backend configuration. deploy/scripts/deploy.sh supplies
  # the approved US bucket, prefix, and CMEK key after validating bucket
  # retention/versioning/public-access controls. The GCS backend provides
  # remote state locking; local state is not an allowed production path.
  backend "gcs" {}
}

# =============================================================================
# Variables
# =============================================================================

variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "blueprint-8c1ca"
}

variable "primary_region" {
  description = "US-only beta primary deployment region"
  type        = string
  default     = "us-central1"

  validation {
    condition     = startswith(var.primary_region, "us-")
    error_message = "The beta data-residency policy requires a US GCP primary_region."
  }
}

variable "secondary_regions" {
  description = "US-only beta secondary regions for overflow"
  type        = list(string)
  default     = ["us-east1"]

  validation {
    condition = alltrue([
      for region in var.secondary_regions : startswith(region, "us-")
    ])
    error_message = "The beta data-residency policy forbids non-US secondary regions."
  }
}

variable "storage_bucket" {
  description = "Firebase Storage bucket name"
  type        = string
  default     = "blueprint-8c1ca.appspot.com"
}

variable "docker_image" {
  description = "Immutable digest-pinned Docker image for the pipeline."
  type        = string
  nullable    = false

  validation {
    condition     = can(regex("^.+@sha256:[0-9a-f]{64}$", var.docker_image))
    error_message = "docker_image must be pinned to an immutable sha256 digest."
  }
}

variable "privacy_sam3_image" {
  description = "Immutable digest-pinned Docker image for the SAM3 privacy service."
  type        = string
  nullable    = false

  validation {
    condition     = can(regex("^.+@sha256:[0-9a-f]{64}$", var.privacy_sam3_image))
    error_message = "privacy_sam3_image must be pinned to an immutable sha256 digest."
  }
}

variable "privacy_vip_image" {
  description = "Immutable digest-pinned Docker image for the VIP privacy service."
  type        = string
  nullable    = false

  validation {
    condition     = can(regex("^.+@sha256:[0-9a-f]{64}$", var.privacy_vip_image))
    error_message = "privacy_vip_image must be pinned to an immutable sha256 digest."
  }
}

variable "privacy_deepprivacy2_image" {
  description = "Immutable digest-pinned Docker image for the DeepPrivacy2 service."
  type        = string
  nullable    = false

  validation {
    condition     = can(regex("^.+@sha256:[0-9a-f]{64}$", var.privacy_deepprivacy2_image))
    error_message = "privacy_deepprivacy2_image must be pinned to an immutable sha256 digest."
  }
}

variable "video_to_world_image" {
  description = "Immutable digest-pinned Docker image for the video_to_world geometry service."
  type        = string
  nullable    = false

  validation {
    condition     = can(regex("^.+@sha256:[0-9a-f]{64}$", var.video_to_world_image))
    error_message = "video_to_world_image must be pinned to an immutable sha256 digest."
  }
}

variable "max_concurrent_jobs" {
  description = "Maximum concurrent pipeline dispatches. External beta target requires at least 25. GPU privacy/video-to-world services use the narrower per-service caps below."
  type        = number
  default     = 25

  validation {
    condition     = var.max_concurrent_jobs >= 25
    error_message = "max_concurrent_jobs must be at least 25 for the external beta capacity target."
  }
}

variable "pipeline_queue_depth_alert_threshold" {
  description = "Cloud Tasks queue-depth alert threshold for beta tail-latency backpressure."
  type        = number
  default     = 50

  validation {
    condition     = var.pipeline_queue_depth_alert_threshold >= 25 && var.pipeline_queue_depth_alert_threshold <= 100
    error_message = "pipeline_queue_depth_alert_threshold must stay between the 25-concurrency beta target and the old 100-task delayed alert."
  }
}

variable "pipeline_queue_depth_alert_duration" {
  description = "How long Cloud Tasks queue depth may exceed the beta threshold before alerting."
  type        = string
  default     = "300s"

  validation {
    condition     = can(regex("^[0-9]+s$", var.pipeline_queue_depth_alert_duration))
    error_message = "pipeline_queue_depth_alert_duration must be a seconds value such as 300s."
  }
}

variable "privacy_sam3_max_instances" {
  description = "Maximum sam3-detect Cloud Run GPU instances. Keep this lower than pipeline dispatch concurrency to bound privacy-runner spend."
  type        = number
  default     = 3

  validation {
    condition     = var.privacy_sam3_max_instances >= 1 && var.privacy_sam3_max_instances <= 25
    error_message = "privacy_sam3_max_instances must be between 1 and 25."
  }
}

variable "privacy_vip_max_instances" {
  description = "Maximum vip-inpaint Cloud Run GPU instances. Keep this lower than pipeline dispatch concurrency to bound privacy-runner spend."
  type        = number
  default     = 2

  validation {
    condition     = var.privacy_vip_max_instances >= 1 && var.privacy_vip_max_instances <= 25
    error_message = "privacy_vip_max_instances must be between 1 and 25."
  }
}

variable "privacy_deepprivacy2_max_instances" {
  description = "Maximum deepprivacy2-anonymize Cloud Run GPU instances. Keep this lower than pipeline dispatch concurrency to bound privacy-runner spend."
  type        = number
  default     = 2

  validation {
    condition     = var.privacy_deepprivacy2_max_instances >= 1 && var.privacy_deepprivacy2_max_instances <= 25
    error_message = "privacy_deepprivacy2_max_instances must be between 1 and 25."
  }
}

variable "video_to_world_max_instances" {
  description = "Maximum video-to-world Cloud Run GPU instances. Keep this lower than pipeline dispatch concurrency to bound geometry-runner spend."
  type        = number
  default     = 2

  validation {
    condition     = var.video_to_world_max_instances >= 1 && var.video_to_world_max_instances <= 25
    error_message = "video_to_world_max_instances must be between 1 and 25."
  }
}

variable "gpu_runner_billable_instance_time_alert_threshold" {
  description = "Per-runner Cloud Run billable instance time rate threshold in instance-seconds per second before alerting."
  type        = number
  default     = 1.0

  validation {
    condition     = var.gpu_runner_billable_instance_time_alert_threshold > 0
    error_message = "gpu_runner_billable_instance_time_alert_threshold must be greater than zero."
  }
}

variable "pipeline_job_timeout_seconds" {
  description = "Cloud Run Job timeout for pipeline execution"
  type        = number
  default     = 14400
}

variable "blueprint_preview_provider" {
  description = "Preview provider used when captures request preview artifacts"
  type        = string
  default     = "world_labs"
}

variable "worldlabs_default_model" {
  description = "Default World Labs model used for request manifests"
  type        = string
  default     = "Marble 0.1-mini"
}

variable "worldlabs_api_key_secret_name" {
  description = "Existing Secret Manager secret containing WORLDLABS_API_KEY"
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9_-]{1,255}$", var.worldlabs_api_key_secret_name))
    error_message = "worldlabs_api_key_secret_name must name an existing Secret Manager secret."
  }
}

variable "privacy_pipeline_enabled" {
  description = "Enable privacy-safe walkthrough post-processing in the production runtime"
  type        = bool
  default     = true
}

variable "privacy_fail_closed" {
  description = "Fail closed when privacy processing cannot safely complete"
  type        = bool
  default     = true
}

variable "sam3_weights_path" {
  description = "SAM3 checkpoint path or URI for the sam3-detect service"
  type        = string
  default     = ""
}

variable "vip_model_path" {
  description = "Optional VIP model path or URI for custom inpainting backends"
  type        = string
  default     = ""
}

variable "deepprivacy2_model_path" {
  description = "DeepPrivacy2 model cache path or URI for the deepprivacy2-anonymize service"
  type        = string
  default     = ""
}

variable "depth_anything_model_path" {
  description = "Depth Anything model path or URI for the vip-inpaint service"
  type        = string
  default     = ""
}

variable "huggingface_token_secret_name" {
  description = "Optional Secret Manager secret name used for HF_TOKEN/HUGGING_FACE_HUB_TOKEN on privacy services"
  type        = string
  default     = ""
}

variable "pipeline_sync_webapp_url" {
  description = "Blueprint-WebApp pipeline sync endpoint"
  type        = string

  validation {
    condition     = can(regex("^https://", var.pipeline_sync_webapp_url))
    error_message = "pipeline_sync_webapp_url is required and must use HTTPS."
  }
}

variable "pipeline_sync_token_secret_name" {
  description = "Existing Secret Manager secret containing PIPELINE_SYNC_TOKEN"
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9_-]{1,255}$", var.pipeline_sync_token_secret_name))
    error_message = "pipeline_sync_token_secret_name must name an existing Secret Manager secret."
  }
}

variable "capture_extract_frames_service_account_email" {
  description = "Optional service account email for the BlueprintCapture extractFrames Cloud Function; when set, it can publish large-video ingest requests."
  type        = string
  default     = ""
}

variable "billing_account_id" {
  description = "Optional Cloud Billing account id. When set, Terraform creates a project-scoped GPU fleet beta billing budget."
  type        = string
  default     = ""
}

variable "gpu_fleet_billing_budget_usd" {
  description = "Project-scoped GCP billing budget amount for the beta GPU/provider fleet."
  type        = number
  default     = 5000

  validation {
    condition     = var.gpu_fleet_billing_budget_usd > 0 && floor(var.gpu_fleet_billing_budget_usd) == var.gpu_fleet_billing_budget_usd
    error_message = "gpu_fleet_billing_budget_usd must be a positive whole-dollar amount."
  }
}

variable "gpu_fleet_billing_budget_thresholds" {
  description = "Alert thresholds for the optional GCP billing budget."
  type        = list(number)
  default     = [0.5, 0.8, 1.0]

  validation {
    condition = alltrue([
      for threshold in var.gpu_fleet_billing_budget_thresholds :
      threshold > 0 && threshold <= 1.5
    ])
    error_message = "gpu_fleet_billing_budget_thresholds values must be > 0 and <= 1.5."
  }
}

variable "privacy_runner_token_secret_name" {
  description = "Existing Secret Manager secret containing PRIVACY_RUNNER_TOKEN"
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9_-]{1,255}$", var.privacy_runner_token_secret_name))
    error_message = "privacy_runner_token_secret_name must name an existing Secret Manager secret."
  }
}

variable "additional_privacy_runner_invoker_members" {
  description = "Additional non-public IAM members allowed to invoke GPU privacy runner Cloud Run services, for example serviceAccount:runner@example.iam.gserviceaccount.com."
  type        = list(string)
  default     = []

  validation {
    condition = alltrue([
      for member in var.additional_privacy_runner_invoker_members :
      !contains(["allUsers", "allAuthenticatedUsers"], member)
    ])
    error_message = "Privacy runner invokers must be named principals, not allUsers or allAuthenticatedUsers."
  }
}

variable "video_to_world_runner_token_secret_name" {
  description = "Optional distinct Secret Manager secret for VIDEO_TO_WORLD_RUNNER_TOKEN; defaults to privacy_runner_token_secret_name"
  type        = string
  default     = ""

  validation {
    condition = (
      var.video_to_world_runner_token_secret_name == "" ||
      can(regex("^[A-Za-z0-9_-]{1,255}$", var.video_to_world_runner_token_secret_name))
    )
    error_message = "video_to_world_runner_token_secret_name must be empty to inherit the privacy token secret or name an existing secret."
  }
}

variable "video_to_world_pipeline_preset" {
  description = "Default upstream execution preset for video_to_world: preprocess_only, preprocess_plus_alignment, full_fast, or full_extensive"
  type        = string
  default     = "preprocess_plus_alignment"
}

variable "video_to_world_command_template" {
  description = "Optional explicit shell command template for video_to_world service execution"
  type        = string
  default     = ""
}

variable "enable_notifications" {
  description = "Enable push notifications via FCM"
  type        = bool
  default     = true
}

variable "monitoring_notification_channels" {
  description = "Google Monitoring notification channel resource names for production pipeline alerts. Production applies should pass at least one channel."
  type        = list(string)
  default     = []
}

variable "allow_empty_monitoring_notification_channels" {
  description = "Explicit waiver for dry-run Terraform plans without alert receivers. Keep false for production applies."
  type        = bool
  default     = false
}

# =============================================================================
# Providers
# =============================================================================

provider "google" {
  project = var.project_id
}

provider "google-beta" {
  project = var.project_id
}

# =============================================================================
# Locals
# =============================================================================

locals {
  all_regions                             = concat([var.primary_region], var.secondary_regions)
  video_to_world_runner_token_secret_name = var.video_to_world_runner_token_secret_name != "" ? var.video_to_world_runner_token_secret_name : var.privacy_runner_token_secret_name
  privacy_runner_service_names = {
    sam3           = google_cloud_run_v2_service.privacy_sam3.name
    vip            = google_cloud_run_v2_service.privacy_vip.name
    deepprivacy2   = google_cloud_run_v2_service.privacy_deepprivacy2.name
    video_to_world = google_cloud_run_v2_service.video_to_world.name
  }
  privacy_runner_max_instances = {
    sam3           = min(var.privacy_sam3_max_instances, var.max_concurrent_jobs)
    vip            = min(var.privacy_vip_max_instances, var.max_concurrent_jobs)
    deepprivacy2   = min(var.privacy_deepprivacy2_max_instances, var.max_concurrent_jobs)
    video_to_world = min(var.video_to_world_max_instances, var.max_concurrent_jobs)
  }
  privacy_runner_monitoring_service_filter = join(" OR ", [
    for service in values(local.privacy_runner_service_names) :
    "resource.labels.service_name=\"${service}\""
  ])
  privacy_runner_invoker_members = toset(concat(
    ["serviceAccount:${google_service_account.pipeline_runner.email}"],
    var.additional_privacy_runner_invoker_members,
  ))
  privacy_runner_invoker_bindings = {
    for pair in setproduct(keys(local.privacy_runner_service_names), local.privacy_runner_invoker_members) :
    "${pair[0]}-${substr(sha1(pair[1]), 0, 12)}" => {
      service = local.privacy_runner_service_names[pair[0]]
      member  = pair[1]
    }
  }

  # Common labels for all resources
  common_labels = {
    project     = "blueprint-capture"
    environment = "production"
    managed-by  = "terraform"
  }
}

# =============================================================================
# Enable Required APIs
# =============================================================================

resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "cloudfunctions.googleapis.com",
    "cloudtasks.googleapis.com",
    "pubsub.googleapis.com",
    "firestore.googleapis.com",
    "storage.googleapis.com",
    "cloudbuild.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "fcm.googleapis.com",
    "billingbudgets.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

data "google_project" "current" {
  project_id = var.project_id
}

# =============================================================================
# Service Accounts
# =============================================================================

# Pipeline service account - runs the CPU-only Cloud Run job
resource "google_service_account" "pipeline_runner" {
  account_id   = "pipeline-runner"
  display_name = "Blueprint Pipeline Runner"
  description  = "Service account for the CPU-only qualification and World Labs pipeline job"
}

# Pipeline invoker - invokes Cloud Run Jobs from Cloud Tasks
resource "google_service_account" "pipeline_invoker" {
  account_id   = "pipeline-invoker"
  display_name = "Blueprint Pipeline Invoker"
  description  = "Service account for invoking pipeline jobs from Cloud Tasks"
}

# Storage trigger - Cloud Function service account
resource "google_service_account" "storage_trigger" {
  account_id   = "storage-trigger"
  display_name = "Blueprint Storage Trigger"
  description  = "Service account for storage trigger Cloud Function"
}

resource "google_service_account" "privacy_sam3_service" {
  account_id   = "privacy-sam3-service"
  display_name = "Blueprint SAM3 Detect Service"
  description  = "Service account for the sam3-detect privacy HTTP service"
}

resource "google_service_account" "privacy_vip_service" {
  account_id   = "privacy-vip-service"
  display_name = "Blueprint VIP Inpaint Service"
  description  = "Service account for the vip-inpaint privacy HTTP service"
}

resource "google_service_account" "privacy_deepprivacy2_service" {
  account_id   = "privacy-deepprivacy2-service"
  display_name = "Blueprint DeepPrivacy2 Anonymize Service"
  description  = "Service account for the deepprivacy2-anonymize privacy HTTP service"
}

resource "google_service_account" "video_to_world_service" {
  account_id   = "video-to-world-service"
  display_name = "Blueprint Video To World Service"
  description  = "Service account for the video-to-world geometry HTTP service"
}

# Existing secret payloads are never accepted as Terraform variables. Terraform
# stores only Secret Manager resource references in state and Cloud Run specs.
data "google_secret_manager_secret" "privacy_runner_token" {
  project   = var.project_id
  secret_id = var.privacy_runner_token_secret_name

  depends_on = [google_project_service.required_apis]
}

data "google_secret_manager_secret" "video_to_world_runner_token" {
  project   = var.project_id
  secret_id = local.video_to_world_runner_token_secret_name

  depends_on = [google_project_service.required_apis]
}

data "google_secret_manager_secret" "pipeline_sync_token" {
  project   = var.project_id
  secret_id = var.pipeline_sync_token_secret_name

  depends_on = [google_project_service.required_apis]
}

data "google_secret_manager_secret" "worldlabs_api_key" {
  project   = var.project_id
  secret_id = var.worldlabs_api_key_secret_name

  depends_on = [google_project_service.required_apis]
}

data "google_secret_manager_secret" "huggingface_token" {
  count = var.huggingface_token_secret_name != "" ? 1 : 0

  project   = var.project_id
  secret_id = var.huggingface_token_secret_name

  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret_iam_member" "pipeline_runner_secrets" {
  for_each = toset([
    data.google_secret_manager_secret.privacy_runner_token.secret_id,
    data.google_secret_manager_secret.video_to_world_runner_token.secret_id,
    data.google_secret_manager_secret.pipeline_sync_token.secret_id,
    data.google_secret_manager_secret.worldlabs_api_key.secret_id,
  ])

  project   = var.project_id
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

resource "google_secret_manager_secret_iam_member" "privacy_service_runner_token" {
  for_each = {
    sam3         = google_service_account.privacy_sam3_service.email
    vip          = google_service_account.privacy_vip_service.email
    deepprivacy2 = google_service_account.privacy_deepprivacy2_service.email
  }

  project   = var.project_id
  secret_id = data.google_secret_manager_secret.privacy_runner_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${each.value}"
}

resource "google_secret_manager_secret_iam_member" "video_to_world_runner_token" {
  project   = var.project_id
  secret_id = data.google_secret_manager_secret.video_to_world_runner_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.video_to_world_service.email}"
}

resource "google_secret_manager_secret_iam_member" "privacy_service_huggingface" {
  for_each = var.huggingface_token_secret_name != "" ? {
    sam3         = google_service_account.privacy_sam3_service.email
    vip          = google_service_account.privacy_vip_service.email
    deepprivacy2 = google_service_account.privacy_deepprivacy2_service.email
  } : {}

  project   = var.project_id
  secret_id = data.google_secret_manager_secret.huggingface_token[0].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${each.value}"
}

# =============================================================================
# IAM Bindings for Pipeline Runner
# =============================================================================

# Storage access (read raw uploads, write processed outputs)
resource "google_project_iam_member" "pipeline_runner_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

resource "google_project_iam_member" "privacy_services_storage" {
  for_each = {
    sam3           = google_service_account.privacy_sam3_service.email
    vip            = google_service_account.privacy_vip_service.email
    deepprivacy2   = google_service_account.privacy_deepprivacy2_service.email
    video_to_world = google_service_account.video_to_world_service.email
  }

  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${each.value}"
}

resource "google_project_iam_member" "privacy_services_logging" {
  for_each = {
    sam3           = google_service_account.privacy_sam3_service.email
    vip            = google_service_account.privacy_vip_service.email
    deepprivacy2   = google_service_account.privacy_deepprivacy2_service.email
    video_to_world = google_service_account.video_to_world_service.email
  }

  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${each.value}"
}

# Firestore access (update job status)
resource "google_project_iam_member" "pipeline_runner_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

# Logging
resource "google_project_iam_member" "pipeline_runner_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

# Metrics
resource "google_project_iam_member" "pipeline_runner_metrics" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

# =============================================================================
# IAM Bindings for Pipeline Invoker
# =============================================================================

# Allow invoking Cloud Run Jobs
resource "google_project_iam_member" "pipeline_invoker_run" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.pipeline_invoker.email}"
}

# Allow creating Cloud Tasks
resource "google_project_iam_member" "pipeline_invoker_tasks" {
  project = var.project_id
  role    = "roles/cloudtasks.enqueuer"
  member  = "serviceAccount:${google_service_account.pipeline_invoker.email}"
}

# =============================================================================
# IAM Bindings for Storage Trigger
# =============================================================================

# Storage access (read manifests)
resource "google_project_iam_member" "storage_trigger_storage" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.storage_trigger.email}"
}

# Firestore access (create capture records)
resource "google_project_iam_member" "storage_trigger_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.storage_trigger.email}"
}

# Pub/Sub publish
resource "google_project_iam_member" "storage_trigger_pubsub" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.storage_trigger.email}"
}

resource "google_project_iam_member" "pipeline_runner_pubsub_subscriber" {
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

# Cloud Tasks enqueue
resource "google_project_iam_member" "storage_trigger_tasks" {
  project = var.project_id
  role    = "roles/cloudtasks.enqueuer"
  member  = "serviceAccount:${google_service_account.storage_trigger.email}"
}

# Cloud Run invoke (for direct job invocation)
resource "google_project_iam_member" "storage_trigger_run" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.storage_trigger.email}"
}

resource "google_project_iam_member" "storage_trigger_run_jobs" {
  project = var.project_id
  role    = "roles/run.jobsExecutorWithOverrides"
  member  = "serviceAccount:${google_service_account.storage_trigger.email}"
}

# Logging
resource "google_project_iam_member" "storage_trigger_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.storage_trigger.email}"
}

# =============================================================================
# Pub/Sub Topics
# =============================================================================

# Main pipeline trigger topic — carries the DESCRIPTOR dispatch payload consumed by the
# on_swap_dispatch worker (event-triggered on this topic). Do NOT attach the pull-based capture
# bridge listener here: its schema differs (XR-04). The listener has its own topic below.
resource "google_pubsub_topic" "pipeline_trigger" {
  name   = "blueprint-capture-pipeline-handoff"
  labels = local.common_labels

  message_retention_duration = "86400s" # 24 hours
}

# Dedicated capture-bridge handoff topic (XR-04) — carries ONLY the canonical handoff schema
# (bucket, scene_id, capture_id, raw_prefix_uri, pipeline_handoff_uri) emitted by the
# storage-trigger raw-upload-complete branch and consumed by the pull listener subscription.
resource "google_pubsub_topic" "capture_bridge_handoff" {
  name   = "blueprint-capture-bridge-handoff"
  labels = local.common_labels

  message_retention_duration = "86400s" # 24 hours
}

resource "google_pubsub_topic" "large_video_ingest" {
  name   = "blueprint-large-video-ingest"
  labels = local.common_labels

  message_retention_duration = "86400s" # 24 hours
}

resource "google_pubsub_topic_iam_member" "capture_extract_frames_large_video_ingest_publisher" {
  count = var.capture_extract_frames_service_account_email != "" ? 1 : 0

  project = var.project_id
  topic   = google_pubsub_topic.large_video_ingest.name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${var.capture_extract_frames_service_account_email}"
}

resource "google_billing_budget" "gpu_fleet_beta" {
  count = var.billing_account_id != "" ? 1 : 0

  billing_account = var.billing_account_id
  display_name    = "Blueprint GPU Fleet Beta Budget"

  budget_filter {
    projects = ["projects/${data.google_project.current.number}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.gpu_fleet_billing_budget_usd)
    }
  }

  dynamic "threshold_rules" {
    for_each = var.gpu_fleet_billing_budget_thresholds

    content {
      threshold_percent = threshold_rules.value
      spend_basis       = "CURRENT_SPEND"
    }
  }

  depends_on = [
    google_project_service.required_apis["billingbudgets.googleapis.com"],
  ]
}

# Dead letter topic for failed messages
resource "google_pubsub_topic" "pipeline_dlq" {
  name   = "pipeline-trigger-dlq"
  labels = local.common_labels
}

resource "google_pubsub_subscription" "pipeline_handoff_listener" {
  name  = "blueprint-pipeline-handoff-listener"
  topic = google_pubsub_topic.capture_bridge_handoff.id

  ack_deadline_seconds       = 600
  message_retention_duration = "604800s"

  retry_policy {
    minimum_backoff = "60s"
    maximum_backoff = "600s"
  }

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.pipeline_dlq.id
    max_delivery_attempts = 5
  }

  labels = local.common_labels
}

# =============================================================================
# Cloud Tasks Queues (per region)
# =============================================================================

resource "google_cloud_tasks_queue" "pipeline_queue" {
  for_each = toset(local.all_regions)

  name     = "blueprint-pipeline-queue"
  location = each.value

  rate_limits {
    max_dispatches_per_second = 10
    max_concurrent_dispatches = var.max_concurrent_jobs
  }

  retry_config {
    max_attempts       = 5
    min_backoff        = "60s"
    max_backoff        = "3600s"
    max_doublings      = 3
    max_retry_duration = "86400s" # 24 hours
  }

  stackdriver_logging_config {
    sampling_ratio = 1.0
  }
}

# Dead letter queue (primary region only)
resource "google_cloud_tasks_queue" "pipeline_dlq" {
  name     = "blueprint-pipeline-dlq"
  location = var.primary_region

  rate_limits {
    max_dispatches_per_second = 1
    max_concurrent_dispatches = 1
  }

  retry_config {
    max_attempts = 1
  }
}

# =============================================================================
# Cloud Run Jobs (per region)
# =============================================================================

resource "google_cloud_run_v2_job" "pipeline" {
  provider = google-beta
  for_each = toset(local.all_regions)

  name     = "blueprint-pipeline"
  location = each.value
  labels   = local.common_labels

  template {
    parallelism = 1
    task_count  = 1

    template {
      execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
      max_retries           = 3
      timeout               = "${var.pipeline_job_timeout_seconds}s"

      service_account = google_service_account.pipeline_runner.email

      containers {
        image   = var.docker_image
        command = ["python"]
        args    = ["-m", "blueprint_pipeline.capture_orchestrator"]

        resources {
          limits = {
            cpu    = "4"
            memory = "16Gi"
          }
        }

        env {
          name  = "PIPELINE_PROJECT_ID"
          value = var.project_id
        }

        env {
          name  = "PIPELINE_REGION"
          value = each.value
        }

        env {
          name  = "PIPELINE_BUCKET"
          value = var.storage_bucket
        }

        env {
          name  = "GCS_ROOT"
          value = "/mnt/gcs"
        }

        env {
          name  = "BLUEPRINT_ENV"
          value = "production"
        }

        env {
          name  = "ENABLE_NOTIFICATIONS"
          value = var.enable_notifications ? "true" : "false"
        }

        env {
          name  = "BLUEPRINT_PREVIEW_PROVIDER"
          value = var.blueprint_preview_provider
        }

        env {
          name  = "WORLDLABS_DEFAULT_MODEL"
          value = var.worldlabs_default_model
        }

        env {
          name = "WORLDLABS_API_KEY"

          value_source {
            secret_key_ref {
              secret  = data.google_secret_manager_secret.worldlabs_api_key.secret_id
              version = "latest"
            }
          }
        }

        env {
          name  = "PRIVACY_PIPELINE_ENABLED"
          value = var.privacy_pipeline_enabled ? "true" : "false"
        }

        env {
          name  = "PRIVACY_FAIL_CLOSED"
          value = var.privacy_fail_closed ? "true" : "false"
        }

        env {
          name  = "BLUEPRINT_CLOUD_RUN_IAM_AUTH_ENABLED"
          value = "true"
        }

        env {
          name  = "PRIVACY_SAM3_URL"
          value = google_cloud_run_v2_service.privacy_sam3.uri
        }

        env {
          name  = "PRIVACY_VIP_URL"
          value = google_cloud_run_v2_service.privacy_vip.uri
        }

        env {
          name  = "PRIVACY_DEEPPRIVACY2_URL"
          value = google_cloud_run_v2_service.privacy_deepprivacy2.uri
        }

        env {
          name = "PRIVACY_RUNNER_TOKEN"

          value_source {
            secret_key_ref {
              secret  = data.google_secret_manager_secret.privacy_runner_token.secret_id
              version = "latest"
            }
          }
        }

        env {
          name  = "VIDEO_TO_WORLD_URL"
          value = google_cloud_run_v2_service.video_to_world.uri
        }

        env {
          name = "VIDEO_TO_WORLD_RUNNER_TOKEN"

          value_source {
            secret_key_ref {
              secret  = data.google_secret_manager_secret.video_to_world_runner_token.secret_id
              version = "latest"
            }
          }
        }

        env {
          name  = "RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO"
          value = "true"
        }

        env {
          name  = "PIPELINE_SYNC_WEBAPP_URL"
          value = var.pipeline_sync_webapp_url
        }

        env {
          name = "PIPELINE_SYNC_TOKEN"

          value_source {
            secret_key_ref {
              secret  = data.google_secret_manager_secret.pipeline_sync_token.secret_id
              version = "latest"
            }
          }
        }

        env {
          name  = "PIPELINE_SYNC_REQUIRED"
          value = "true"
        }

        env {
          name  = "PIPELINE_SYNC_MAX_ATTEMPTS"
          value = "5"
        }

        env {
          name  = "PIPELINE_SYNC_RETRY_DELAY_MS"
          value = "1000"
        }

        volume_mounts {
          name       = "capture-storage"
          mount_path = "/mnt/gcs"
        }
      }

      volumes {
        name = "capture-storage"

        gcs {
          bucket    = var.storage_bucket
          read_only = false
        }
      }
    }
  }

  depends_on = [
    google_project_service.required_apis["run.googleapis.com"],
  ]
}

# =============================================================================
# Cloud Run Services - Privacy Runners
# =============================================================================

resource "google_cloud_run_v2_service" "privacy_sam3" {
  provider = google-beta

  name     = "sam3-detect"
  location = var.primary_region
  labels   = local.common_labels

  template {
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"
    max_instance_request_concurrency = 1
    service_account                  = google_service_account.privacy_sam3_service.email
    timeout                          = "3600s"

    scaling {
      min_instance_count = 0
      max_instance_count = local.privacy_runner_max_instances.sam3
    }

    containers {
      image = var.privacy_sam3_image

      resources {
        limits = {
          cpu              = "4"
          memory           = "16Gi"
          "nvidia.com/gpu" = "1"
        }
      }

      env {
        name  = "PRIVACY_RUNNER_KIND"
        value = "sam3"
      }

      env {
        name  = "GCS_ROOT"
        value = "/mnt/gcs"
      }

      env {
        name = "PRIVACY_RUNNER_TOKEN"

        value_source {
          secret_key_ref {
            secret  = data.google_secret_manager_secret.privacy_runner_token.secret_id
            version = "latest"
          }
        }
      }

      env {
        name  = "SAM3_WEIGHTS_PATH"
        value = var.sam3_weights_path
      }

      dynamic "env" {
        for_each = var.huggingface_token_secret_name != "" ? {
          HF_TOKEN               = "HF_TOKEN"
          HUGGING_FACE_HUB_TOKEN = "HUGGING_FACE_HUB_TOKEN"
        } : {}

        content {
          name = env.value

          value_source {
            secret_key_ref {
              secret  = data.google_secret_manager_secret.huggingface_token[0].secret_id
              version = "latest"
            }
          }
        }
      }

      volume_mounts {
        name       = "capture-storage"
        mount_path = "/mnt/gcs"
      }
    }

    volumes {
      name = "capture-storage"

      gcs {
        bucket    = var.storage_bucket
        read_only = false
      }
    }
  }
}

resource "google_cloud_run_v2_service" "privacy_vip" {
  provider = google-beta

  name     = "vip-inpaint"
  location = var.primary_region
  labels   = local.common_labels

  template {
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"
    max_instance_request_concurrency = 1
    service_account                  = google_service_account.privacy_vip_service.email
    timeout                          = "7200s"

    scaling {
      min_instance_count = 0
      max_instance_count = local.privacy_runner_max_instances.vip
    }

    containers {
      image = var.privacy_vip_image

      resources {
        limits = {
          cpu              = "4"
          memory           = "16Gi"
          "nvidia.com/gpu" = "1"
        }
      }

      env {
        name  = "PRIVACY_RUNNER_KIND"
        value = "vip"
      }

      env {
        name  = "GCS_ROOT"
        value = "/mnt/gcs"
      }

      env {
        name = "PRIVACY_RUNNER_TOKEN"

        value_source {
          secret_key_ref {
            secret  = data.google_secret_manager_secret.privacy_runner_token.secret_id
            version = "latest"
          }
        }
      }

      env {
        name  = "VIP_MODEL_PATH"
        value = var.vip_model_path
      }

      env {
        name  = "DEPTH_ANYTHING_MODEL_PATH"
        value = var.depth_anything_model_path
      }

      dynamic "env" {
        for_each = var.huggingface_token_secret_name != "" ? {
          HF_TOKEN               = "HF_TOKEN"
          HUGGING_FACE_HUB_TOKEN = "HUGGING_FACE_HUB_TOKEN"
        } : {}

        content {
          name = env.value

          value_source {
            secret_key_ref {
              secret  = data.google_secret_manager_secret.huggingface_token[0].secret_id
              version = "latest"
            }
          }
        }
      }

      volume_mounts {
        name       = "capture-storage"
        mount_path = "/mnt/gcs"
      }
    }

    volumes {
      name = "capture-storage"

      gcs {
        bucket    = var.storage_bucket
        read_only = false
      }
    }
  }
}

resource "google_cloud_run_v2_service" "privacy_deepprivacy2" {
  provider = google-beta

  name     = "deepprivacy2-anonymize"
  location = var.primary_region
  labels   = local.common_labels

  template {
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"
    max_instance_request_concurrency = 1
    service_account                  = google_service_account.privacy_deepprivacy2_service.email
    timeout                          = "7200s"

    scaling {
      min_instance_count = 0
      max_instance_count = local.privacy_runner_max_instances.deepprivacy2
    }

    containers {
      image = var.privacy_deepprivacy2_image

      resources {
        limits = {
          cpu              = "4"
          memory           = "16Gi"
          "nvidia.com/gpu" = "1"
        }
      }

      env {
        name  = "PRIVACY_RUNNER_KIND"
        value = "deepprivacy2"
      }

      env {
        name  = "GCS_ROOT"
        value = "/mnt/gcs"
      }

      env {
        name = "PRIVACY_RUNNER_TOKEN"

        value_source {
          secret_key_ref {
            secret  = data.google_secret_manager_secret.privacy_runner_token.secret_id
            version = "latest"
          }
        }
      }

      env {
        name  = "DEEPPRIVACY2_MODEL_PATH"
        value = var.deepprivacy2_model_path
      }

      dynamic "env" {
        for_each = var.huggingface_token_secret_name != "" ? {
          HF_TOKEN               = "HF_TOKEN"
          HUGGING_FACE_HUB_TOKEN = "HUGGING_FACE_HUB_TOKEN"
        } : {}

        content {
          name = env.value

          value_source {
            secret_key_ref {
              secret  = data.google_secret_manager_secret.huggingface_token[0].secret_id
              version = "latest"
            }
          }
        }
      }

      volume_mounts {
        name       = "capture-storage"
        mount_path = "/mnt/gcs"
      }
    }

    volumes {
      name = "capture-storage"

      gcs {
        bucket    = var.storage_bucket
        read_only = false
      }
    }
  }
}

resource "google_cloud_run_v2_service" "video_to_world" {
  provider = google-beta

  name     = "video-to-world"
  location = var.primary_region
  labels   = local.common_labels

  template {
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"
    max_instance_request_concurrency = 1
    service_account                  = google_service_account.video_to_world_service.email
    timeout                          = "7200s"

    scaling {
      min_instance_count = 0
      max_instance_count = local.privacy_runner_max_instances.video_to_world
    }

    containers {
      image = var.video_to_world_image

      resources {
        limits = {
          cpu              = "4"
          memory           = "16Gi"
          "nvidia.com/gpu" = "1"
        }
      }

      env {
        name  = "GCS_ROOT"
        value = "/mnt/gcs"
      }

      env {
        name = "VIDEO_TO_WORLD_RUNNER_TOKEN"

        value_source {
          secret_key_ref {
            secret  = data.google_secret_manager_secret.video_to_world_runner_token.secret_id
            version = "latest"
          }
        }
      }

      env {
        name  = "VIDEO_TO_WORLD_PIPELINE_PRESET"
        value = var.video_to_world_pipeline_preset
      }

      env {
        name  = "VIDEO_TO_WORLD_COMMAND_TEMPLATE"
        value = var.video_to_world_command_template
      }

      volume_mounts {
        name       = "capture-storage"
        mount_path = "/mnt/gcs"
      }
    }

    volumes {
      name = "capture-storage"

      gcs {
        bucket    = var.storage_bucket
        read_only = false
      }
    }
  }
}

resource "google_cloud_run_service_iam_member" "privacy_runner_invoker" {
  for_each = local.privacy_runner_invoker_bindings

  location = var.primary_region
  project  = var.project_id
  service  = each.value.service
  role     = "roles/run.invoker"
  member   = each.value.member
}

# =============================================================================
# Cloud Function - Storage Trigger
# =============================================================================

# Source code bucket
resource "google_storage_bucket" "function_source" {
  name     = "${var.project_id}-function-source"
  location = var.primary_region

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# Cloud Function (Gen2)
resource "google_cloudfunctions2_function" "storage_trigger" {
  name     = "storage-trigger"
  location = var.primary_region
  labels   = local.common_labels

  build_config {
    runtime     = "python311"
    entry_point = "on_storage_finalize"

    source {
      storage_source {
        bucket = google_storage_bucket.function_source.name
        object = google_storage_bucket_object.function_source.name
      }
    }
  }

  service_config {
    max_instance_count    = 100
    min_instance_count    = 0
    available_memory      = "512M"
    timeout_seconds       = 60
    service_account_email = google_service_account.storage_trigger.email

    environment_variables = {
      PIPELINE_PROJECT_ID                     = var.project_id
      PIPELINE_REGION                         = var.primary_region
      PIPELINE_BUCKET                         = var.storage_bucket
      REGIONS                                 = join(",", local.all_regions)
      SWAP_TRIGGER_DISPATCH_MODE              = "pubsub"
      SWAP_TRIGGER_PUBSUB_TOPIC               = google_pubsub_topic.pipeline_trigger.name
      SWAP_TRIGGER_HANDOFF_PUBSUB_TOPIC       = google_pubsub_topic.capture_bridge_handoff.name
      SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF = "true"
    }
  }

  event_trigger {
    trigger_region = var.primary_region
    event_type     = "google.cloud.storage.object.v1.finalized"

    event_filters {
      attribute = "bucket"
      value     = var.storage_bucket
    }

    retry_policy = "RETRY_POLICY_RETRY"
  }

  depends_on = [
    google_project_service.required_apis["cloudfunctions.googleapis.com"],
    google_storage_bucket_object.function_source,
  ]
}

# Cloud Function (Gen2) - async orchestration worker (Pub/Sub consumer)
resource "google_cloudfunctions2_function" "swap_dispatch_worker" {
  name     = "swap-dispatch-worker"
  location = var.primary_region
  labels   = local.common_labels

  build_config {
    runtime     = "python311"
    entry_point = "on_swap_dispatch"

    source {
      storage_source {
        bucket = google_storage_bucket.function_source.name
        object = google_storage_bucket_object.function_source.name
      }
    }
  }

  service_config {
    max_instance_count    = 100
    min_instance_count    = 0
    available_memory      = "4096M"
    timeout_seconds       = 3600
    service_account_email = google_service_account.storage_trigger.email

    environment_variables = {
      PIPELINE_PROJECT_ID     = var.project_id
      PIPELINE_REGION         = var.primary_region
      PIPELINE_BUCKET         = var.storage_bucket
      REGIONS                 = join(",", local.all_regions)
      PIPELINE_EXECUTION_MODE = "cloud_run_job"
      PIPELINE_RUN_JOB_NAME   = google_cloud_run_v2_job.pipeline[var.primary_region].name
      PIPELINE_RUN_JOB_REGION = var.primary_region
    }
  }

  event_trigger {
    trigger_region = var.primary_region
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"

    event_filters {
      attribute = "topic"
      value     = google_pubsub_topic.pipeline_trigger.id
    }

    retry_policy = "RETRY_POLICY_RETRY"
  }

  depends_on = [
    google_project_service.required_apis["cloudfunctions.googleapis.com"],
    google_storage_bucket_object.function_source,
    google_pubsub_topic.pipeline_trigger,
  ]
}

# Package function source code
data "archive_file" "function_source" {
  type        = "zip"
  source_dir  = "${path.module}/../.."
  output_path = "${path.module}/function-source.zip"
  excludes = [
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "deploy",
    "dist",
    "docs",
    "local_runs_worldlabs",
    "skillpacks",
    "tests",
  ]
}

resource "google_storage_bucket_object" "function_source" {
  name   = "storage-trigger-${data.archive_file.function_source.output_md5}.zip"
  bucket = google_storage_bucket.function_source.name
  source = data.archive_file.function_source.output_path
}

# =============================================================================
# Firestore Database
# =============================================================================

resource "google_firestore_database" "default" {
  name        = "(default)"
  location_id = var.primary_region
  type        = "FIRESTORE_NATIVE"

  depends_on = [
    google_project_service.required_apis["firestore.googleapis.com"],
  ]
}

# Firestore indexes for captures collection. The legacy createdAt composites
# stay for current readers; sharded companions are the scale-up path once
# writers populate createdAtShard and readers aggregate per-shard queries.
resource "google_firestore_index" "captures_status" {
  collection = "captures"
  database   = google_firestore_database.default.name

  fields {
    field_path = "status"
    order      = "ASCENDING"
  }

  fields {
    field_path = "createdAt"
    order      = "ASCENDING"
  }
}

resource "google_firestore_index" "captures_user" {
  collection = "captures"
  database   = google_firestore_database.default.name

  fields {
    field_path = "creatorId"
    order      = "ASCENDING"
  }

  fields {
    field_path = "createdAt"
    order      = "DESCENDING"
  }
}

resource "google_firestore_index" "captures_status_created_at_shard" {
  collection = "captures"
  database   = google_firestore_database.default.name

  fields {
    field_path = "status"
    order      = "ASCENDING"
  }

  fields {
    field_path = "createdAtShard"
    order      = "ASCENDING"
  }

  fields {
    field_path = "createdAt"
    order      = "ASCENDING"
  }
}

resource "google_firestore_index" "captures_user_created_at_shard" {
  collection = "captures"
  database   = google_firestore_database.default.name

  fields {
    field_path = "creatorId"
    order      = "ASCENDING"
  }

  fields {
    field_path = "createdAtShard"
    order      = "ASCENDING"
  }

  fields {
    field_path = "createdAt"
    order      = "DESCENDING"
  }
}

# =============================================================================
# Monitoring and Alerting
# =============================================================================

# Alert policy for failed jobs
resource "google_monitoring_alert_policy" "pipeline_failures" {
  display_name = "Blueprint Pipeline Failures"
  combiner     = "OR"

  conditions {
    display_name = "Job Failure Rate"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_job\" AND metric.type=\"run.googleapis.com/job/completed_task_attempt_count\" AND metric.labels.result=\"failed\""
      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  notification_channels = var.monitoring_notification_channels

  lifecycle {
    precondition {
      condition     = var.allow_empty_monitoring_notification_channels || length(var.monitoring_notification_channels) > 0
      error_message = "monitoring_notification_channels must include at least one channel for production alert policies. Set allow_empty_monitoring_notification_channels=true only for dry-run plans."
    }
  }

  documentation {
    content   = "More than 5 pipeline job failures in 5 minutes. Check Cloud Run Job logs for details."
    mime_type = "text/markdown"
  }
}

# Alert policy for queue depth
resource "google_monitoring_alert_policy" "queue_depth" {
  display_name = "Blueprint Pipeline Queue Depth"
  combiner     = "OR"

  conditions {
    display_name = "High Queue Depth"

    condition_threshold {
      filter          = "resource.type=\"cloud_tasks_queue\" AND metric.type=\"cloudtasks.googleapis.com/queue/depth\""
      duration        = var.pipeline_queue_depth_alert_duration
      comparison      = "COMPARISON_GT"
      threshold_value = var.pipeline_queue_depth_alert_threshold

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = var.monitoring_notification_channels

  lifecycle {
    precondition {
      condition     = var.allow_empty_monitoring_notification_channels || length(var.monitoring_notification_channels) > 0
      error_message = "monitoring_notification_channels must include at least one channel for production alert policies. Set allow_empty_monitoring_notification_channels=true only for dry-run plans."
    }
  }

  documentation {
    content   = "Pipeline queue depth exceeds the beta backpressure threshold for the configured duration. Check dispatch latency, per-user intake pressure, and worker health before admitting more captures."
    mime_type = "text/markdown"
  }
}

# Alert policy for Firestore request latency. This is the runtime monitor for
# possible createdAt index hotspotting during soak/load tests; Key Visualizer
# remains the source for confirming a specific index-key hotspot.
resource "google_monitoring_alert_policy" "firestore_request_latency" {
  display_name = "Blueprint Firestore Request Latency"
  combiner     = "OR"

  conditions {
    display_name = "Firestore p99 API latency"

    condition_monitoring_query_language {
      duration = "300s"
      query    = <<-EOT
        fetch consumed_api
        | metric 'serviceruntime.googleapis.com/api/request_latencies'
        | filter (resource.service == 'firestore.googleapis.com')
        | group_by 5m,
            [value_request_latencies_percentile:
              percentile(value.request_latencies, 99)]
        | every 5m
        | condition val() > 0.25 's'
      EOT
    }
  }

  notification_channels = var.monitoring_notification_channels

  lifecycle {
    precondition {
      condition     = var.allow_empty_monitoring_notification_channels || length(var.monitoring_notification_channels) > 0
      error_message = "monitoring_notification_channels must include at least one channel for production alert policies. Set allow_empty_monitoring_notification_channels=true only for dry-run plans."
    }
  }

  documentation {
    content   = "Firestore p99 request latency exceeds 250ms for 5 minutes. During beta soak or scale-up, inspect captures.createdAt composite indexes, Key Visualizer index heatmaps, write rate, and sharded createdAtShard migration readiness before admitting more capture traffic."
    mime_type = "text/markdown"
  }
}

# Alert when capture-bridge handoffs are not being drained by the deployed
# listener. This catches the failure mode where uploads publish successfully but
# the pull subscription ages until Pub/Sub dead-letters the message.
resource "google_monitoring_alert_policy" "capture_handoff_listener_lag" {
  display_name = "Blueprint Capture Handoff Listener Lag"
  combiner     = "OR"

  conditions {
    display_name = "Oldest unacked capture handoff age"

    condition_threshold {
      filter          = "resource.type=\"pubsub_subscription\" AND metric.type=\"pubsub.googleapis.com/subscription/oldest_unacked_message_age\" AND resource.labels.subscription_id=\"${google_pubsub_subscription.pipeline_handoff_listener.name}\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 300

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MAX"
      }
    }
  }

  notification_channels = var.monitoring_notification_channels

  lifecycle {
    precondition {
      condition     = var.allow_empty_monitoring_notification_channels || length(var.monitoring_notification_channels) > 0
      error_message = "monitoring_notification_channels must include at least one channel for production alert policies. Set allow_empty_monitoring_notification_channels=true only for dry-run plans."
    }
  }

  documentation {
    content   = "Capture handoff messages are aging on blueprint-pipeline-handoff-listener. Check the deployed listener timer/service before messages hit the dead-letter threshold."
    mime_type = "text/markdown"
  }
}

resource "google_monitoring_alert_policy" "gpu_runner_billable_instance_time" {
  display_name = "Blueprint GPU Runner Billable Instance Time"
  combiner     = "OR"

  conditions {
    display_name = "Sustained GPU runner billable instance time"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/container/billable_instance_time\" AND (${local.privacy_runner_monitoring_service_filter})"
      duration        = "900s"
      comparison      = "COMPARISON_GT"
      threshold_value = var.gpu_runner_billable_instance_time_alert_threshold

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = var.monitoring_notification_channels

  lifecycle {
    precondition {
      condition     = var.allow_empty_monitoring_notification_channels || length(var.monitoring_notification_channels) > 0
      error_message = "monitoring_notification_channels must include at least one channel for production alert policies. Set allow_empty_monitoring_notification_channels=true only for dry-run plans."
    }
  }

  documentation {
    content   = "GPU privacy/video-to-world Cloud Run billable instance time is sustained above the configured threshold. Confirm the jobs are operator-authorized and not retrying or over-scaling."
    mime_type = "text/markdown"
  }
}

# =============================================================================
# Outputs
# =============================================================================

output "pipeline_service_account" {
  description = "Service account email for pipeline runner"
  value       = google_service_account.pipeline_runner.email
}

output "invoker_service_account" {
  description = "Service account email for pipeline invoker"
  value       = google_service_account.pipeline_invoker.email
}

output "trigger_service_account" {
  description = "Service account email for storage trigger"
  value       = google_service_account.storage_trigger.email
}

output "cloud_run_jobs" {
  description = "Cloud Run Job URLs by region"
  value       = { for k, v in google_cloud_run_v2_job.pipeline : k => v.name }
}

output "privacy_runner_services" {
  description = "GPU runner service URLs"
  value = {
    sam3           = google_cloud_run_v2_service.privacy_sam3.uri
    vip            = google_cloud_run_v2_service.privacy_vip.uri
    deepprivacy2   = google_cloud_run_v2_service.privacy_deepprivacy2.uri
    video_to_world = google_cloud_run_v2_service.video_to_world.uri
  }
}

output "deployed_image_digests" {
  description = "Immutable image digests bound into the refreshed deployment topology"
  value = {
    pipeline             = var.docker_image
    privacy_sam3         = var.privacy_sam3_image
    privacy_vip          = var.privacy_vip_image
    privacy_deepprivacy2 = var.privacy_deepprivacy2_image
    video_to_world       = var.video_to_world_image
  }
}

output "privacy_runner_max_instances" {
  description = "Per-service GPU runner max instances after applying the global pipeline concurrency ceiling."
  value       = local.privacy_runner_max_instances
}

output "cloud_tasks_queues" {
  description = "Cloud Tasks Queue paths by region"
  value       = { for k, v in google_cloud_tasks_queue.pipeline_queue : k => v.id }
}

output "pubsub_topic" {
  description = "Pub/Sub topic for pipeline triggers"
  value       = google_pubsub_topic.pipeline_trigger.id
}

output "pubsub_handoff_listener_subscription" {
  description = "Pull subscription consumed by blueprint-pubsub-handoff-listener"
  value       = google_pubsub_subscription.pipeline_handoff_listener.id
}

output "large_video_ingest_topic" {
  description = "Pub/Sub topic that receives large-video ingest requests from BlueprintCapture extractFrames."
  value       = google_pubsub_topic.large_video_ingest.id
}

output "gpu_fleet_billing_budget" {
  description = "Optional GCP billing budget resource for the beta GPU/provider fleet."
  value       = var.billing_account_id != "" ? google_billing_budget.gpu_fleet_beta[0].name : null
}

output "storage_trigger_function" {
  description = "Storage trigger Cloud Function URL"
  value       = google_cloudfunctions2_function.storage_trigger.service_config[0].uri
}

output "swap_dispatch_worker_function" {
  description = "Swap dispatch worker Cloud Function name"
  value       = google_cloudfunctions2_function.swap_dispatch_worker.name
}
