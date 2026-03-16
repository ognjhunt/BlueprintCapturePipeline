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

  # Optional: Use GCS backend for state storage
  # backend "gcs" {
  #   bucket = "blueprint-terraform-state"
  #   prefix = "capture-pipeline"
  # }
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
  description = "Primary deployment region"
  type        = string
  default     = "us-central1"
}

variable "secondary_regions" {
  description = "Secondary regions for overflow and geographic distribution"
  type        = list(string)
  default     = ["us-east1", "europe-west1"]
}

variable "storage_bucket" {
  description = "Firebase Storage bucket name"
  type        = string
  default     = "blueprint-8c1ca.appspot.com"
}

variable "docker_image" {
  description = "Docker image for the pipeline"
  type        = string
  default     = "gcr.io/blueprint-8c1ca/blueprint-pipeline:latest"
}

variable "privacy_sam3_image" {
  description = "Docker image for the SAM3 privacy service"
  type        = string
  default     = "gcr.io/blueprint-8c1ca/sam3-privacy:latest"
}

variable "privacy_vip_image" {
  description = "Docker image for the VIP privacy service"
  type        = string
  default     = "gcr.io/blueprint-8c1ca/vip-privacy:latest"
}

variable "privacy_deepprivacy2_image" {
  description = "Docker image for the DeepPrivacy2 service"
  type        = string
  default     = "gcr.io/blueprint-8c1ca/deepprivacy2-privacy:latest"
}

variable "max_concurrent_jobs" {
  description = "Maximum concurrent privacy service instances"
  type        = number
  default     = 10
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
  default     = ""
}

variable "pipeline_sync_token" {
  description = "Shared auth token for Blueprint-WebApp pipeline sync"
  type        = string
  default     = ""
  sensitive   = true
}

variable "privacy_runner_token" {
  description = "Shared auth token for privacy runner HTTP services"
  type        = string
  default     = ""
  sensitive   = true
}

variable "enable_notifications" {
  description = "Enable push notifications via FCM"
  type        = bool
  default     = true
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
  all_regions = concat([var.primary_region], var.secondary_regions)

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
  ])

  service            = each.value
  disable_on_destroy = false
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
    sam3         = google_service_account.privacy_sam3_service.email
    vip          = google_service_account.privacy_vip_service.email
    deepprivacy2 = google_service_account.privacy_deepprivacy2_service.email
  }

  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${each.value}"
}

resource "google_project_iam_member" "privacy_services_logging" {
  for_each = {
    sam3         = google_service_account.privacy_sam3_service.email
    vip          = google_service_account.privacy_vip_service.email
    deepprivacy2 = google_service_account.privacy_deepprivacy2_service.email
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

# Main pipeline trigger topic
resource "google_pubsub_topic" "pipeline_trigger" {
  name   = "pipeline-trigger"
  labels = local.common_labels

  message_retention_duration = "86400s" # 24 hours
}

# Dead letter topic for failed messages
resource "google_pubsub_topic" "pipeline_dlq" {
  name   = "pipeline-trigger-dlq"
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
          name  = "PRIVACY_PIPELINE_ENABLED"
          value = var.privacy_pipeline_enabled ? "true" : "false"
        }

        env {
          name  = "PRIVACY_FAIL_CLOSED"
          value = var.privacy_fail_closed ? "true" : "false"
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
          name  = "PRIVACY_RUNNER_TOKEN"
          value = var.privacy_runner_token
        }

        env {
          name  = "PIPELINE_SYNC_WEBAPP_URL"
          value = var.pipeline_sync_webapp_url
        }

        env {
          name  = "PIPELINE_SYNC_TOKEN"
          value = var.pipeline_sync_token
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
      max_instance_count = var.max_concurrent_jobs
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
        name  = "PRIVACY_RUNNER_TOKEN"
        value = var.privacy_runner_token
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
              secret  = var.huggingface_token_secret_name
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
      max_instance_count = var.max_concurrent_jobs
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
        name  = "PRIVACY_RUNNER_TOKEN"
        value = var.privacy_runner_token
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
              secret  = var.huggingface_token_secret_name
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
      max_instance_count = var.max_concurrent_jobs
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
        name  = "PRIVACY_RUNNER_TOKEN"
        value = var.privacy_runner_token
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
              secret  = var.huggingface_token_secret_name
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

resource "google_cloud_run_service_iam_member" "privacy_runner_public_invoker" {
  for_each = {
    sam3         = google_cloud_run_v2_service.privacy_sam3.name
    vip          = google_cloud_run_v2_service.privacy_vip.name
    deepprivacy2 = google_cloud_run_v2_service.privacy_deepprivacy2.name
  }

  location = var.primary_region
  project  = var.project_id
  service  = each.value
  role     = "roles/run.invoker"
  member   = "allUsers"
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
      PIPELINE_PROJECT_ID        = var.project_id
      PIPELINE_REGION            = var.primary_region
      PIPELINE_BUCKET            = var.storage_bucket
      REGIONS                    = join(",", local.all_regions)
      SWAP_TRIGGER_DISPATCH_MODE = "pubsub"
      SWAP_TRIGGER_PUBSUB_TOPIC  = google_pubsub_topic.pipeline_trigger.name
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

# Firestore indexes for captures collection
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
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [] # Add notification channels as needed

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
      duration        = "600s"
      comparison      = "COMPARISON_GT"
      threshold_value = 100

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = []

  documentation {
    content   = "Pipeline queue depth exceeds 100 tasks for 10+ minutes. Consider scaling up or investigating processing issues."
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
  description = "Privacy runner service URLs"
  value = {
    sam3         = google_cloud_run_v2_service.privacy_sam3.uri
    vip          = google_cloud_run_v2_service.privacy_vip.uri
    deepprivacy2 = google_cloud_run_v2_service.privacy_deepprivacy2.uri
  }
}

output "cloud_tasks_queues" {
  description = "Cloud Tasks Queue paths by region"
  value       = { for k, v in google_cloud_tasks_queue.pipeline_queue : k => v.id }
}

output "pubsub_topic" {
  description = "Pub/Sub topic for pipeline triggers"
  value       = google_pubsub_topic.pipeline_trigger.id
}

output "storage_trigger_function" {
  description = "Storage trigger Cloud Function URL"
  value       = google_cloudfunctions2_function.storage_trigger.service_config[0].uri
}

output "swap_dispatch_worker_function" {
  description = "Swap dispatch worker Cloud Function name"
  value       = google_cloudfunctions2_function.swap_dispatch_worker.name
}
