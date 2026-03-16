#!/bin/bash
#
# Blueprint Capture Pipeline - Deployment Script
#
# This script deploys the complete infrastructure for the video-to-3D pipeline.
#
# Usage:
#   ./deploy.sh                    # Deploy everything
#   ./deploy.sh --docker-only      # Only build and push Docker image
#   ./deploy.sh --terraform-only   # Only apply Terraform
#   ./deploy.sh --function-only    # Only deploy Cloud Function
#   ./deploy.sh --dry-run          # Show what would be done
#
# Prerequisites:
#   - gcloud CLI authenticated with appropriate permissions
#   - Docker installed and running
#   - Terraform >= 1.5.0 installed
#   - jq installed for JSON parsing
#

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Default values (override with environment variables)
PROJECT_ID="${PROJECT_ID:-blueprint-8c1ca}"
PRIMARY_REGION="${PRIMARY_REGION:-us-central1}"
SECONDARY_REGIONS="${SECONDARY_REGIONS:-us-east1,europe-west1}"
STORAGE_BUCKET="${STORAGE_BUCKET:-${PROJECT_ID}.appspot.com}"
IMAGE_NAME="${IMAGE_NAME:-blueprint-pipeline}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SAM3_IMAGE_NAME="${SAM3_IMAGE_NAME:-sam3-privacy}"
VIP_IMAGE_NAME="${VIP_IMAGE_NAME:-vip-privacy}"
DEEPPRIVACY2_IMAGE_NAME="${DEEPPRIVACY2_IMAGE_NAME:-deepprivacy2-privacy}"
SWAP_TOPIC="${SWAP_TOPIC:-pipeline-trigger}"
BLUEPRINT_PREVIEW_PROVIDER="${BLUEPRINT_PREVIEW_PROVIDER:-world_labs}"
WORLDLABS_DEFAULT_MODEL="${WORLDLABS_DEFAULT_MODEL:-Marble 0.1-mini}"
PRIVACY_PIPELINE_ENABLED="${PRIVACY_PIPELINE_ENABLED:-true}"
PRIVACY_FAIL_CLOSED="${PRIVACY_FAIL_CLOSED:-true}"
PRIVACY_SAM3_URL="${PRIVACY_SAM3_URL:-}"
PRIVACY_VIP_URL="${PRIVACY_VIP_URL:-}"
PRIVACY_DEEPPRIVACY2_URL="${PRIVACY_DEEPPRIVACY2_URL:-}"
SAM3_WEIGHTS_PATH="${SAM3_WEIGHTS_PATH:-}"
VIP_MODEL_PATH="${VIP_MODEL_PATH:-}"
DEEPPRIVACY2_MODEL_PATH="${DEEPPRIVACY2_MODEL_PATH:-}"
DEPTH_ANYTHING_MODEL_PATH="${DEPTH_ANYTHING_MODEL_PATH:-}"
HUGGINGFACE_TOKEN_SECRET_NAME="${HUGGINGFACE_TOKEN_SECRET_NAME:-}"
PRIVACY_RUNNER_TOKEN="${PRIVACY_RUNNER_TOKEN:-}"
PIPELINE_SYNC_WEBAPP_URL="${PIPELINE_SYNC_WEBAPP_URL:-}"
PIPELINE_SYNC_TOKEN="${PIPELINE_SYNC_TOKEN:-}"
WORLDLABS_API_KEY="${WORLDLABS_API_KEY:-}"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DEPLOY_DIR")"
TERRAFORM_DIR="$DEPLOY_DIR/terraform"
FUNCTIONS_DIR="$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is required but not installed."
        exit 1
    fi
}

confirm() {
    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    read -p "$1 [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

# =============================================================================
# Deployment Functions
# =============================================================================

check_prerequisites() {
    log_info "Checking prerequisites..."

    check_command gcloud
    check_command docker
    check_command terraform
    check_command jq

    # Check gcloud authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null 2>&1; then
        log_error "gcloud is not authenticated. Run: gcloud auth login"
        exit 1
    fi

    # Check project
    CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
    if [[ "$CURRENT_PROJECT" != "$PROJECT_ID" ]]; then
        log_warning "Current gcloud project ($CURRENT_PROJECT) differs from target ($PROJECT_ID)"
        if confirm "Set gcloud project to $PROJECT_ID?"; then
            gcloud config set project "$PROJECT_ID"
        fi
    fi

    log_success "Prerequisites check passed"
}

enable_apis() {
    log_info "Enabling required GCP APIs..."

    APIS=(
        "run.googleapis.com"
        "cloudfunctions.googleapis.com"
        "cloudtasks.googleapis.com"
        "pubsub.googleapis.com"
        "firestore.googleapis.com"
        "storage.googleapis.com"
        "cloudbuild.googleapis.com"
        "secretmanager.googleapis.com"
        "logging.googleapis.com"
        "monitoring.googleapis.com"
    )

    for api in "${APIS[@]}"; do
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY-RUN] Would enable: $api"
        else
            gcloud services enable "$api" --quiet || true
        fi
    done

    log_success "APIs enabled"
}

build_docker_image() {
    log_info "Building Docker images..."

    cd "$PROJECT_ROOT"

    IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
    SAM3_IMAGE_URI="gcr.io/${PROJECT_ID}/${SAM3_IMAGE_NAME}:${IMAGE_TAG}"
    VIP_IMAGE_URI="gcr.io/${PROJECT_ID}/${VIP_IMAGE_NAME}:${IMAGE_TAG}"
    DEEPPRIVACY2_IMAGE_URI="gcr.io/${PROJECT_ID}/${DEEPPRIVACY2_IMAGE_NAME}:${IMAGE_TAG}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would build: $IMAGE_URI"
        log_info "[DRY-RUN] Would build: $SAM3_IMAGE_URI"
        log_info "[DRY-RUN] Would build: $VIP_IMAGE_URI"
        log_info "[DRY-RUN] Would build: $DEEPPRIVACY2_IMAGE_URI"
        return
    fi

    docker build \
        --target production \
        -t "$IMAGE_URI" \
        -f Dockerfile \
        .

    docker build \
        -t "$SAM3_IMAGE_URI" \
        -f deploy/docker/sam3/Dockerfile \
        .

    docker build \
        -t "$VIP_IMAGE_URI" \
        -f deploy/docker/vip/Dockerfile \
        .

    docker build \
        -t "$DEEPPRIVACY2_IMAGE_URI" \
        -f deploy/docker/deepprivacy2/Dockerfile \
        .

    log_success "Docker images built"
}

push_docker_image() {
    log_info "Pushing Docker images to GCR..."

    IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
    SAM3_IMAGE_URI="gcr.io/${PROJECT_ID}/${SAM3_IMAGE_NAME}:${IMAGE_TAG}"
    VIP_IMAGE_URI="gcr.io/${PROJECT_ID}/${VIP_IMAGE_NAME}:${IMAGE_TAG}"
    DEEPPRIVACY2_IMAGE_URI="gcr.io/${PROJECT_ID}/${DEEPPRIVACY2_IMAGE_NAME}:${IMAGE_TAG}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would push: $IMAGE_URI"
        log_info "[DRY-RUN] Would push: $SAM3_IMAGE_URI"
        log_info "[DRY-RUN] Would push: $VIP_IMAGE_URI"
        log_info "[DRY-RUN] Would push: $DEEPPRIVACY2_IMAGE_URI"
        return
    fi

    # Configure Docker for GCR
    gcloud auth configure-docker gcr.io --quiet

    docker push "$IMAGE_URI"
    docker push "$SAM3_IMAGE_URI"
    docker push "$VIP_IMAGE_URI"
    docker push "$DEEPPRIVACY2_IMAGE_URI"

    log_success "Docker images pushed"
}

apply_terraform() {
    log_info "Applying Terraform configuration..."

    cd "$TERRAFORM_DIR"

    # Create terraform.tfvars if it doesn't exist
    if [[ ! -f "terraform.tfvars" ]]; then
        log_info "Creating terraform.tfvars from example..."
        cat > terraform.tfvars << EOF
project_id         = "${PROJECT_ID}"
primary_region     = "${PRIMARY_REGION}"
secondary_regions  = [$(echo "$SECONDARY_REGIONS" | tr ',' '\n' | sed 's/.*/"&"/' | tr '\n' ',' | sed 's/,$//')]
storage_bucket     = "${STORAGE_BUCKET}"
docker_image       = "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
privacy_sam3_image = "gcr.io/${PROJECT_ID}/${SAM3_IMAGE_NAME}:${IMAGE_TAG}"
privacy_vip_image  = "gcr.io/${PROJECT_ID}/${VIP_IMAGE_NAME}:${IMAGE_TAG}"
privacy_deepprivacy2_image = "gcr.io/${PROJECT_ID}/${DEEPPRIVACY2_IMAGE_NAME}:${IMAGE_TAG}"
privacy_runner_token = "${PRIVACY_RUNNER_TOKEN}"
sam3_weights_path = "${SAM3_WEIGHTS_PATH}"
vip_model_path = "${VIP_MODEL_PATH}"
deepprivacy2_model_path = "${DEEPPRIVACY2_MODEL_PATH}"
depth_anything_model_path = "${DEPTH_ANYTHING_MODEL_PATH}"
huggingface_token_secret_name = "${HUGGINGFACE_TOKEN_SECRET_NAME}"
pipeline_sync_webapp_url = "${PIPELINE_SYNC_WEBAPP_URL}"
pipeline_sync_token = "${PIPELINE_SYNC_TOKEN}"
EOF
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would run: terraform init && terraform plan"
        terraform init -input=false
        terraform plan -input=false
        return
    fi

    # Initialize Terraform
    terraform init -input=false

    # Plan and apply
    terraform plan -input=false -out=tfplan

    if confirm "Apply Terraform changes?"; then
        terraform apply -input=false tfplan
        rm -f tfplan
    fi

    log_success "Terraform applied"
}

deploy_cloud_function() {
    log_info "Deploying Cloud Function..."

    cd "$FUNCTIONS_DIR"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would deploy Cloud Function: storage-trigger"
        return
    fi

    # Deploy the function
    gcloud functions deploy storage-trigger \
        --gen2 \
        --runtime python311 \
        --region "$PRIMARY_REGION" \
        --source . \
        --entry-point on_storage_finalize \
        --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
        --trigger-event-filters="bucket=${STORAGE_BUCKET}" \
        --memory 512M \
        --timeout 60s \
        --set-env-vars "PIPELINE_PROJECT_ID=${PROJECT_ID},PIPELINE_REGION=${PRIMARY_REGION},REGIONS=${SECONDARY_REGIONS},SWAP_TRIGGER_DISPATCH_MODE=pubsub,SWAP_TRIGGER_PUBSUB_TOPIC=${SWAP_TOPIC}"

    gcloud functions deploy swap-dispatch-worker \
        --gen2 \
        --runtime python311 \
        --region "$PRIMARY_REGION" \
        --source . \
        --entry-point on_swap_dispatch \
        --trigger-topic "${SWAP_TOPIC}" \
        --memory 4096M \
        --timeout 3600s \
        --set-env-vars "PIPELINE_PROJECT_ID=${PROJECT_ID},PIPELINE_REGION=${PRIMARY_REGION},REGIONS=${SECONDARY_REGIONS},PIPELINE_EXECUTION_MODE=cloud_run_job,PIPELINE_RUN_JOB_NAME=blueprint-pipeline,PIPELINE_RUN_JOB_REGION=${PRIMARY_REGION}"

    log_success "Cloud Function deployed"
}

create_cloud_run_jobs() {
    log_info "Creating Cloud Run Jobs..."

    IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

    # Split regions
    IFS=',' read -ra REGIONS <<< "${PRIMARY_REGION},${SECONDARY_REGIONS}"

    for region in "${REGIONS[@]}"; do
        log_info "Creating Cloud Run Job in $region..."

        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY-RUN] Would create job: blueprint-pipeline in $region"
            continue
        fi

        # Check if job exists
        if gcloud run jobs describe blueprint-pipeline --region "$region" &>/dev/null; then
            log_info "Job exists in $region, updating..."
            gcloud run jobs update blueprint-pipeline \
                --image "$IMAGE_URI" \
                --region "$region" \
                --cpu 4 \
                --memory 16Gi \
                --max-retries 3 \
                --task-timeout 3600s \
                --command python \
                --args "-m,blueprint_pipeline.capture_orchestrator" \
                --add-volume name=capture-storage,type=cloud-storage,bucket=${STORAGE_BUCKET} \
                --add-volume-mount volume=capture-storage,mount-path=/mnt/gcs \
                --set-env-vars "PIPELINE_PROJECT_ID=${PROJECT_ID},PIPELINE_REGION=${region},PIPELINE_BUCKET=${STORAGE_BUCKET},GCS_ROOT=/mnt/gcs,BLUEPRINT_PREVIEW_PROVIDER=${BLUEPRINT_PREVIEW_PROVIDER},WORLDLABS_API_KEY=${WORLDLABS_API_KEY},WORLDLABS_DEFAULT_MODEL=${WORLDLABS_DEFAULT_MODEL},PRIVACY_PIPELINE_ENABLED=${PRIVACY_PIPELINE_ENABLED},PRIVACY_FAIL_CLOSED=${PRIVACY_FAIL_CLOSED},PRIVACY_SAM3_URL=${PRIVACY_SAM3_URL},PRIVACY_VIP_URL=${PRIVACY_VIP_URL},PRIVACY_DEEPPRIVACY2_URL=${PRIVACY_DEEPPRIVACY2_URL},PRIVACY_RUNNER_TOKEN=${PRIVACY_RUNNER_TOKEN},PIPELINE_SYNC_WEBAPP_URL=${PIPELINE_SYNC_WEBAPP_URL},PIPELINE_SYNC_TOKEN=${PIPELINE_SYNC_TOKEN}" \
                --quiet
        else
            log_info "Creating new job in $region..."
            gcloud run jobs create blueprint-pipeline \
                --image "$IMAGE_URI" \
                --region "$region" \
                --cpu 4 \
                --memory 16Gi \
                --max-retries 3 \
                --task-timeout 3600s \
                --command python \
                --args "-m,blueprint_pipeline.capture_orchestrator" \
                --add-volume name=capture-storage,type=cloud-storage,bucket=${STORAGE_BUCKET} \
                --add-volume-mount volume=capture-storage,mount-path=/mnt/gcs \
                --set-env-vars "PIPELINE_PROJECT_ID=${PROJECT_ID},PIPELINE_REGION=${region},PIPELINE_BUCKET=${STORAGE_BUCKET},GCS_ROOT=/mnt/gcs,BLUEPRINT_PREVIEW_PROVIDER=${BLUEPRINT_PREVIEW_PROVIDER},WORLDLABS_API_KEY=${WORLDLABS_API_KEY},WORLDLABS_DEFAULT_MODEL=${WORLDLABS_DEFAULT_MODEL},PRIVACY_PIPELINE_ENABLED=${PRIVACY_PIPELINE_ENABLED},PRIVACY_FAIL_CLOSED=${PRIVACY_FAIL_CLOSED},PRIVACY_SAM3_URL=${PRIVACY_SAM3_URL},PRIVACY_VIP_URL=${PRIVACY_VIP_URL},PRIVACY_DEEPPRIVACY2_URL=${PRIVACY_DEEPPRIVACY2_URL},PRIVACY_RUNNER_TOKEN=${PRIVACY_RUNNER_TOKEN},PIPELINE_SYNC_WEBAPP_URL=${PIPELINE_SYNC_WEBAPP_URL},PIPELINE_SYNC_TOKEN=${PIPELINE_SYNC_TOKEN}" \
                --quiet
        fi
    done

    log_success "Cloud Run Jobs created"
}

create_cloud_tasks_queues() {
    log_info "Creating Cloud Tasks queues..."

    # Split regions
    IFS=',' read -ra REGIONS <<< "${PRIMARY_REGION},${SECONDARY_REGIONS}"

    QUEUES=("blueprint-pipeline-queue" "blueprint-pipeline-queue-low" "blueprint-pipeline-queue-high" "blueprint-pipeline-queue-urgent" "blueprint-pipeline-dlq")

    for region in "${REGIONS[@]}"; do
        for queue in "${QUEUES[@]}"; do
            log_info "Creating queue $queue in $region..."

            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "[DRY-RUN] Would create queue: $queue in $region"
                continue
            fi

            # Check if queue exists
            if gcloud tasks queues describe "$queue" --location "$region" &>/dev/null; then
                log_info "Queue $queue already exists in $region"
                continue
            fi

            # Create queue
            if [[ "$queue" == *"-dlq" ]]; then
                # Dead letter queue with low rate
                gcloud tasks queues create "$queue" \
                    --location "$region" \
                    --max-dispatches-per-second 1 \
                    --max-concurrent-dispatches 1 \
                    --quiet
            else
                # Normal queue
                gcloud tasks queues create "$queue" \
                    --location "$region" \
                    --max-dispatches-per-second 10 \
                    --max-concurrent-dispatches 32 \
                    --max-attempts 5 \
                    --min-backoff 60s \
                    --max-backoff 3600s \
                    --quiet
            fi
        done
    done

    log_success "Cloud Tasks queues created"
}

create_pubsub_topics() {
    log_info "Creating Pub/Sub topics..."

    TOPICS=("pipeline-trigger" "pipeline-trigger-dlq")

    for topic in "${TOPICS[@]}"; do
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY-RUN] Would create topic: $topic"
            continue
        fi

        if gcloud pubsub topics describe "$topic" &>/dev/null; then
            log_info "Topic $topic already exists"
        else
            gcloud pubsub topics create "$topic" --quiet
            log_info "Created topic: $topic"
        fi
    done

    log_success "Pub/Sub topics created"
}

setup_iam() {
    log_info "Setting up IAM permissions..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would configure IAM permissions"
        return
    fi

    # Create service accounts if they don't exist
    for sa in "pipeline-runner" "pipeline-invoker" "storage-trigger"; do
        SA_EMAIL="${sa}@${PROJECT_ID}.iam.gserviceaccount.com"

        if gcloud iam service-accounts describe "$SA_EMAIL" &>/dev/null; then
            log_info "Service account $sa already exists"
        else
            gcloud iam service-accounts create "$sa" \
                --display-name "Blueprint Pipeline - ${sa}" \
                --quiet
            log_info "Created service account: $sa"
        fi
    done

    # Grant roles to pipeline-runner
    RUNNER_EMAIL="pipeline-runner@${PROJECT_ID}.iam.gserviceaccount.com"
    for role in "roles/storage.objectAdmin" "roles/datastore.user" "roles/logging.logWriter" "roles/monitoring.metricWriter"; do
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member "serviceAccount:${RUNNER_EMAIL}" \
            --role "$role" \
            --quiet --no-user-output-enabled
    done

    # Grant roles to pipeline-invoker
    INVOKER_EMAIL="pipeline-invoker@${PROJECT_ID}.iam.gserviceaccount.com"
    for role in "roles/run.invoker" "roles/cloudtasks.enqueuer"; do
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member "serviceAccount:${INVOKER_EMAIL}" \
            --role "$role" \
            --quiet --no-user-output-enabled
    done

    # Grant roles to storage-trigger
    TRIGGER_EMAIL="storage-trigger@${PROJECT_ID}.iam.gserviceaccount.com"
    for role in "roles/storage.objectViewer" "roles/datastore.user" "roles/pubsub.publisher" "roles/cloudtasks.enqueuer" "roles/run.invoker" "roles/run.jobsExecutorWithOverrides" "roles/logging.logWriter"; do
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member "serviceAccount:${TRIGGER_EMAIL}" \
            --role "$role" \
            --quiet --no-user-output-enabled
    done

    log_success "IAM permissions configured"
}

print_summary() {
    echo ""
    echo "=============================================="
    echo "  Blueprint Capture Pipeline - Deployed"
    echo "=============================================="
    echo ""
    echo "Project: $PROJECT_ID"
    echo "Primary Region: $PRIMARY_REGION"
    echo "Secondary Regions: $SECONDARY_REGIONS"
    echo ""
    echo "Resources:"
    echo "  - Docker Image: gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "  - Privacy Service Image (SAM3): gcr.io/${PROJECT_ID}/${SAM3_IMAGE_NAME}:${IMAGE_TAG}"
    echo "  - Privacy Service Image (VIP): gcr.io/${PROJECT_ID}/${VIP_IMAGE_NAME}:${IMAGE_TAG}"
    echo "  - Privacy Service Image (DeepPrivacy2): gcr.io/${PROJECT_ID}/${DEEPPRIVACY2_IMAGE_NAME}:${IMAGE_TAG}"
    echo "  - Cloud Run Job (CPU): blueprint-pipeline"
    echo "  - Cloud Run Services (GPU): sam3-detect, vip-inpaint, deepprivacy2-anonymize"
    echo "  - Cloud Function: storage-trigger"
    echo "  - Cloud Tasks Queue: blueprint-pipeline-queue"
    echo "  - Pub/Sub Topic: pipeline-trigger"
    echo ""
    echo "Service Accounts:"
    echo "  - pipeline-runner@${PROJECT_ID}.iam.gserviceaccount.com"
    echo "  - pipeline-invoker@${PROJECT_ID}.iam.gserviceaccount.com"
    echo "  - storage-trigger@${PROJECT_ID}.iam.gserviceaccount.com"
    echo ""
    echo "Next Steps:"
    echo "  1. Upload a video to gs://${STORAGE_BUCKET}/scenes/{scene_id}/..."
    echo "  2. Monitor the Cloud Function logs for trigger"
    echo "  3. Check Cloud Run Jobs for processing status"
    echo "  4. Query Firestore 'captures' collection for results"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

main() {
    local DOCKER_ONLY=false
    local TERRAFORM_ONLY=false
    local FUNCTION_ONLY=false
    local DRY_RUN=false
    local SKIP_DOCKER=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --docker-only)
                DOCKER_ONLY=true
                shift
                ;;
            --terraform-only)
                TERRAFORM_ONLY=true
                shift
                ;;
            --function-only)
                FUNCTION_ONLY=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --docker-only     Only build and push Docker image"
                echo "  --terraform-only  Only apply Terraform configuration"
                echo "  --function-only   Only deploy Cloud Function"
                echo "  --skip-docker     Skip Docker build (use existing image)"
                echo "  --dry-run         Show what would be done without making changes"
                echo "  --help            Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Export for child functions
    export DRY_RUN

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "DRY-RUN mode - no changes will be made"
    fi

    # Run deployment steps
    check_prerequisites

    if [[ "$DOCKER_ONLY" == "true" ]]; then
        build_docker_image
        push_docker_image
        exit 0
    fi

    if [[ "$TERRAFORM_ONLY" == "true" ]]; then
        apply_terraform
        exit 0
    fi

    if [[ "$FUNCTION_ONLY" == "true" ]]; then
        deploy_cloud_function
        exit 0
    fi

    # Full deployment
    enable_apis

    if [[ "$SKIP_DOCKER" != "true" ]]; then
        build_docker_image
        push_docker_image
    fi

    setup_iam
    create_pubsub_topics
    create_cloud_tasks_queues
    create_cloud_run_jobs
    deploy_cloud_function

    # Optionally apply Terraform (alternative to manual resource creation)
    # apply_terraform

    print_summary

    log_success "Deployment complete!"
}

main "$@"
