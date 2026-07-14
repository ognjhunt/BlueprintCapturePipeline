packer {
  required_plugins {
    googlecompute = {
      version = ">= 1.1.6"
      source  = "github.com/hashicorp/googlecompute"
    }
  }
}

variable "project_id" {
  type = string
}

variable "zone" {
  type = string
}

variable "image_storage_location" {
  type        = string
  description = "Explicit GCP multi-region or region in which to store the image."
}

variable "source_image" {
  type        = string
  description = "Immutable GPU-driver-ready source image selected and verified by the operator."
}

variable "image_name" {
  type = string
}

variable "worker_image_ref" {
  type        = string
  description = "Exact worker OCI ref in repository@sha256:<64 lowercase hex> form."
  validation {
    condition     = can(regex("^[^[:space:]@]+@sha256:[0-9a-f]{64}$", var.worker_image_ref))
    error_message = "worker_image_ref must be digest pinned"
  }
}

variable "network" {
  type = string
}

variable "subnetwork" {
  type = string
}

variable "service_account_email" {
  type = string
}

variable "builder_machine_type" {
  type    = string
  default = "e2-standard-8"
}

variable "disk_size_gb" {
  type    = number
  default = 250
}

source "googlecompute" "gpu_worker_host" {
  project_id              = var.project_id
  zone                    = var.zone
  source_image            = var.source_image
  image_name              = var.image_name
  image_description       = "Blueprint GPU host: driver, Docker, NVIDIA runtime, and digest-pinned worker cache"
  machine_type            = var.builder_machine_type
  disk_size               = var.disk_size_gb
  disk_type               = "pd-ssd"
  network                 = var.network
  subnetwork              = var.subnetwork
  service_account_email   = var.service_account_email
  scopes                  = ["https://www.googleapis.com/auth/cloud-platform"]
  use_internal_ip         = true
  omit_external_ip        = true
  use_os_login            = true
  image_storage_locations = [var.image_storage_location]
  image_labels = {
    "blueprint-managed" = "true"
    "startup-class"     = "prebaked-warm-worker"
  }
}

build {
  name    = "blueprint-gpu-worker-host"
  sources = ["source.googlecompute.gpu_worker_host"]

  provisioner "shell" {
    script          = "deploy/packer/scripts/bake_gpu_worker_host.sh"
    execute_command = "sudo -E bash '{{ .Path }}'"
    environment_vars = [
      "BLUEPRINT_WORKER_IMAGE_REF=${var.worker_image_ref}",
    ]
  }
}
