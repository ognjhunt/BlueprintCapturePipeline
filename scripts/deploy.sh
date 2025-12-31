#!/bin/bash
# Blueprint Capture Pipeline - Full Deployment Script
#
# This script deploys the complete pipeline infrastructure:
# 1. Validates prerequisites
# 2. Builds and pushes Docker image
# 3. Deploys Terraform infrastructure
# 4. Deploys Cloud Function trigger
# 5. Validates deployment
#
# Usage:
#   ./scripts/deploy.sh [--dry-run] [--skip-docker] [--skip-terraform] [--skip-function]
#
# Environment variables:
#   GCP_PROJECT_ID - GCP project ID (default: blueprint-8c1ca)
#   GCP_REGION - Primary region (default: us-central1)
#   DOCKER_TAG - Docker image tag (default: latest)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-blueprint-8c1ca}"
REGION="${GCP_REGION:-us-central1}"
DOCKER_TAG="${DOCKER_TAG:-latest}"
BUCKET_NAME="${PROJECT_ID}.appspot.com"
IMAGE_NAME="gcr.io/${PROJECT_ID}/blueprint-pipeline:${DOCKER_TAG}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Flags
DRY_RUN=false
SKIP_DOCKER=false
SKIP_TERRAFORM=false
SKIP_FUNCTION=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-docker)
            SKIP_DOCKER=true
            shift
            ;;
        --skip-terraform)
            SKIP_TERRAFORM=true
            shift
            ;;
        --skip-function)
            SKIP_FUNCTION=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Logging functions
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

# Run command with dry-run support
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
    else
        "$@"
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing=()

    # Check required tools
    if ! command -v gcloud &> /dev/null; then
        missing+=("gcloud (Google Cloud SDK)")
    fi

    if ! command -v docker &> /dev/null; then
        missing+=("docker")
    fi

    if ! command -v terraform &> /dev/null; then
        missing+=("terraform")
    fi

    if ! command -v jq &> /dev/null; then
        missing+=("jq")
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing required tools:"
        for tool in "${missing[@]}"; do
            echo "  - $tool"
        done
        exit 1
    fi

    # Check gcloud authentication
    if ! gcloud auth print-access-token &> /dev/null; then
        log_error "Not authenticated with gcloud. Run: gcloud auth login"
        exit 1
    fi

    # Check project exists and is accessible
    if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
        log_error "Cannot access project: $PROJECT_ID"
        exit 1
    fi

    log_success "All prerequisites met"
}

# Enable required APIs
enable_apis() {
    log_info "Enabling required GCP APIs..."

    local apis=(
        "run.googleapis.com"
        "cloudfunctions.googleapis.com"
        "cloudbuild.googleapis.com"
        "cloudtasks.googleapis.com"
        "pubsub.googleapis.com"
        "firestore.googleapis.com"
        "storage.googleapis.com"
        "secretmanager.googleapis.com"
        "logging.googleapis.com"
        "monitoring.googleapis.com"
    )

    for api in "${apis[@]}"; do
        log_info "  Enabling $api..."
        run_cmd gcloud services enable "$api" --project="$PROJECT_ID" --quiet
    done

    log_success "All APIs enabled"
}

# Build and push Docker image
build_docker() {
    if [ "$SKIP_DOCKER" = true ]; then
        log_warning "Skipping Docker build (--skip-docker)"
        return
    fi

    log_info "Building Docker image..."
    cd "$ROOT_DIR"

    # Configure Docker for GCR
    run_cmd gcloud auth configure-docker --quiet

    # Build the image
    log_info "  Building: $IMAGE_NAME"
    run_cmd docker build \
        --target production \
        -t "$IMAGE_NAME" \
        -f Dockerfile \
        .

    # Push to GCR
    log_info "  Pushing to GCR..."
    run_cmd docker push "$IMAGE_NAME"

    log_success "Docker image built and pushed: $IMAGE_NAME"
}

# Deploy Terraform infrastructure
deploy_terraform() {
    if [ "$SKIP_TERRAFORM" = true ]; then
        log_warning "Skipping Terraform deployment (--skip-terraform)"
        return
    fi

    log_info "Deploying Terraform infrastructure..."
    cd "$ROOT_DIR/deploy/terraform"

    # Initialize Terraform
    log_info "  Initializing Terraform..."
    run_cmd terraform init -upgrade

    # Validate configuration
    log_info "  Validating configuration..."
    run_cmd terraform validate

    # Plan deployment
    log_info "  Planning deployment..."
    run_cmd terraform plan \
        -var="project_id=$PROJECT_ID" \
        -var="docker_image=$IMAGE_NAME" \
        -var="primary_region=$REGION" \
        -out=tfplan

    # Apply deployment
    if [ "$DRY_RUN" = false ]; then
        log_info "  Applying deployment..."
        run_cmd terraform apply -auto-approve tfplan

        # Clean up plan file
        rm -f tfplan
    fi

    log_success "Terraform infrastructure deployed"
}

# Deploy Cloud Function
deploy_function() {
    if [ "$SKIP_FUNCTION" = true ]; then
        log_warning "Skipping Cloud Function deployment (--skip-function)"
        return
    fi

    log_info "Deploying Cloud Function (storage_trigger)..."
    cd "$ROOT_DIR/functions"

    # Create requirements.txt if it doesn't exist
    if [ ! -f requirements.txt ]; then
        log_info "  Creating requirements.txt..."
        cat > requirements.txt << 'EOF'
google-cloud-storage>=2.10.0
google-cloud-firestore>=2.14.0
google-cloud-pubsub>=2.18.0
google-cloud-run>=0.9.0
EOF
    fi

    # Deploy the function
    run_cmd gcloud functions deploy storage-trigger \
        --gen2 \
        --runtime=python311 \
        --region="$REGION" \
        --source=. \
        --entry-point=on_storage_finalize \
        --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
        --trigger-event-filters="bucket=$BUCKET_NAME" \
        --memory=512MB \
        --timeout=60s \
        --set-env-vars="PIPELINE_PROJECT_ID=$PROJECT_ID,PIPELINE_REGION=$REGION,PIPELINE_BUCKET=$BUCKET_NAME" \
        --service-account="storage-trigger@${PROJECT_ID}.iam.gserviceaccount.com" \
        --project="$PROJECT_ID" \
        --quiet

    log_success "Cloud Function deployed"
}

# Validate deployment
validate_deployment() {
    log_info "Validating deployment..."

    local errors=()

    # Check Cloud Run Job exists
    if ! gcloud run jobs describe blueprint-pipeline \
        --region="$REGION" \
        --project="$PROJECT_ID" &> /dev/null; then
        errors+=("Cloud Run Job 'blueprint-pipeline' not found in $REGION")
    else
        log_success "  Cloud Run Job: OK"
    fi

    # Check Cloud Function exists
    if ! gcloud functions describe storage-trigger \
        --gen2 \
        --region="$REGION" \
        --project="$PROJECT_ID" &> /dev/null; then
        errors+=("Cloud Function 'storage-trigger' not found in $REGION")
    else
        log_success "  Cloud Function: OK"
    fi

    # Check Cloud Tasks queue exists
    if ! gcloud tasks queues describe blueprint-pipeline-queue \
        --location="$REGION" \
        --project="$PROJECT_ID" &> /dev/null; then
        errors+=("Cloud Tasks queue 'blueprint-pipeline-queue' not found in $REGION")
    else
        log_success "  Cloud Tasks Queue: OK"
    fi

    # Check Pub/Sub topic exists
    if ! gcloud pubsub topics describe pipeline-trigger \
        --project="$PROJECT_ID" &> /dev/null; then
        errors+=("Pub/Sub topic 'pipeline-trigger' not found")
    else
        log_success "  Pub/Sub Topic: OK"
    fi

    # Check Docker image exists
    if ! gcloud container images describe "$IMAGE_NAME" &> /dev/null; then
        errors+=("Docker image '$IMAGE_NAME' not found in GCR")
    else
        log_success "  Docker Image: OK"
    fi

    if [ ${#errors[@]} -ne 0 ]; then
        log_error "Validation failed:"
        for error in "${errors[@]}"; do
            echo "  - $error"
        done
        return 1
    fi

    log_success "All components validated successfully"
}

# Print deployment summary
print_summary() {
    echo ""
    echo "=============================================="
    echo "  DEPLOYMENT SUMMARY"
    echo "=============================================="
    echo ""
    echo "  Project:      $PROJECT_ID"
    echo "  Region:       $REGION"
    echo "  Docker Image: $IMAGE_NAME"
    echo "  Bucket:       gs://$BUCKET_NAME"
    echo ""
    echo "  Components:"
    echo "    - Cloud Run Job: blueprint-pipeline"
    echo "    - Cloud Function: storage-trigger"
    echo "    - Cloud Tasks Queue: blueprint-pipeline-queue"
    echo "    - Pub/Sub Topic: pipeline-trigger"
    echo ""
    echo "  To test the deployment, upload a capture to:"
    echo "    gs://$BUCKET_NAME/scenes/{scene_id}/iphone/{timestamp}/raw/"
    echo ""
    echo "  Monitor logs:"
    echo "    gcloud logging read 'resource.type=cloud_run_job' --project=$PROJECT_ID"
    echo ""
    echo "=============================================="
}

# Main execution
main() {
    echo ""
    echo "=============================================="
    echo "  Blueprint Capture Pipeline - Deployment"
    echo "=============================================="
    echo ""

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN MODE - No changes will be made"
        echo ""
    fi

    log_info "Configuration:"
    log_info "  Project ID: $PROJECT_ID"
    log_info "  Region: $REGION"
    log_info "  Docker Tag: $DOCKER_TAG"
    echo ""

    # Execute deployment steps
    check_prerequisites
    enable_apis
    build_docker
    deploy_terraform
    deploy_function

    if [ "$DRY_RUN" = false ]; then
        validate_deployment
    fi

    print_summary

    log_success "Deployment complete!"
}

main "$@"
