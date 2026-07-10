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
#   ./deploy.sh --rollback --rollback-image-tag <tag>
#                                  # Roll Cloud Run jobs back to a known-good image tag
#   ./deploy.sh --dry-run          # Show what would be done
#
# Deploy gate:
#   FULL_TEST_LANE_COMMIT="$(git rev-parse HEAD)" \
#   FULL_TEST_LANE_EVIDENCE_URI="https://github.com/.../actions/runs/..." \
#     ./deploy.sh
#
# Release tags default to the current git SHA prefix. Do not deploy `latest`.
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
SECONDARY_REGIONS="${SECONDARY_REGIONS:-us-east1}"
STORAGE_BUCKET="${STORAGE_BUCKET:-${PROJECT_ID}.appspot.com}"
IMAGE_NAME="${IMAGE_NAME:-blueprint-pipeline}"
IMAGE_TAG="${IMAGE_TAG:-}"
SAM3_IMAGE_NAME="${SAM3_IMAGE_NAME:-sam3-privacy}"
VIP_IMAGE_NAME="${VIP_IMAGE_NAME:-vip-privacy}"
DEEPPRIVACY2_IMAGE_NAME="${DEEPPRIVACY2_IMAGE_NAME:-deepprivacy2-privacy}"
VIDEO_TO_WORLD_IMAGE_NAME="${VIDEO_TO_WORLD_IMAGE_NAME:-video-to-world}"
SWAP_TOPIC="${SWAP_TOPIC:-blueprint-capture-pipeline-handoff}"
# Dedicated capture-bridge handoff topic (XR-04): pull listener consumes ONLY canonical handoff
# payloads here, distinct from the descriptor topic consumed by on_swap_dispatch.
HANDOFF_TOPIC="${HANDOFF_TOPIC:-blueprint-capture-bridge-handoff}"
BLUEPRINT_PREVIEW_PROVIDER="${BLUEPRINT_PREVIEW_PROVIDER:-world_labs}"
WORLDLABS_DEFAULT_MODEL="${WORLDLABS_DEFAULT_MODEL:-Marble 0.1-mini}"
BLUEPRINT_LAUNCH_PROOF_MODE="${BLUEPRINT_LAUNCH_PROOF_MODE:-production}"
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
PRIVACY_RUNNER_TOKEN_SECRET_NAME="${PRIVACY_RUNNER_TOKEN_SECRET_NAME:-}"
VIDEO_TO_WORLD_URL="${VIDEO_TO_WORLD_URL:-}"
VIDEO_TO_WORLD_RUNNER_TOKEN_SECRET_NAME="${VIDEO_TO_WORLD_RUNNER_TOKEN_SECRET_NAME:-$PRIVACY_RUNNER_TOKEN_SECRET_NAME}"
VIDEO_TO_WORLD_PIPELINE_PRESET="${VIDEO_TO_WORLD_PIPELINE_PRESET:-preprocess_plus_alignment}"
VIDEO_TO_WORLD_COMMAND_TEMPLATE="${VIDEO_TO_WORLD_COMMAND_TEMPLATE:-}"
PIPELINE_SYNC_WEBAPP_URL="${PIPELINE_SYNC_WEBAPP_URL:-}"
PIPELINE_SYNC_TOKEN_SECRET_NAME="${PIPELINE_SYNC_TOKEN_SECRET_NAME:-}"
WORLDLABS_API_KEY_SECRET_NAME="${WORLDLABS_API_KEY_SECRET_NAME:-}"
TERRAFORM_STATE_BUCKET="${TERRAFORM_STATE_BUCKET:-}"
TERRAFORM_STATE_PREFIX="${TERRAFORM_STATE_PREFIX:-capture-pipeline}"
TERRAFORM_STATE_KMS_KEY="${TERRAFORM_STATE_KMS_KEY:-}"
ROLLBACK_IMAGE_TAG="${ROLLBACK_IMAGE_TAG:-}"
ROLLBACK_VERIFY_COMMAND="${ROLLBACK_VERIFY_COMMAND:-python -m pytest tests/test_deploy_systemd_contract.py tests/test_launch_readiness_packet.py}"
ROLLBACK_HEALTH_CHECK="${ROLLBACK_HEALTH_CHECK:-true}"
FULL_TEST_LANE_REQUIRED="${FULL_TEST_LANE_REQUIRED:-true}"
FULL_TEST_LANE_COMMIT="${FULL_TEST_LANE_COMMIT:-}"
FULL_TEST_LANE_EVIDENCE_URI="${FULL_TEST_LANE_EVIDENCE_URI:-}"
FULL_TEST_LANE_PROVENANCE_PATH="${FULL_TEST_LANE_PROVENANCE_PATH:-}"
GIT_SHA="${GIT_SHA:-}"
RELEASE_ID="${RELEASE_ID:-}"
DEPLOYMENT_MANIFEST_PATH="${DEPLOYMENT_MANIFEST_PATH:-}"
TOPOLOGY_EVIDENCE_PATH="${TOPOLOGY_EVIDENCE_PATH:-}"
DEPLOYMENT_CANARY_PATH="${DEPLOYMENT_CANARY_PATH:-}"
IMAGE_DIGEST_URI="${IMAGE_DIGEST_URI:-}"
SAM3_IMAGE_DIGEST_URI="${SAM3_IMAGE_DIGEST_URI:-}"
VIP_IMAGE_DIGEST_URI="${VIP_IMAGE_DIGEST_URI:-}"
DEEPPRIVACY2_IMAGE_DIGEST_URI="${DEEPPRIVACY2_IMAGE_DIGEST_URI:-}"
VIDEO_TO_WORLD_IMAGE_DIGEST_URI="${VIDEO_TO_WORLD_IMAGE_DIGEST_URI:-}"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DEPLOY_DIR")"
TERRAFORM_DIR="$DEPLOY_DIR/terraform"
FUNCTIONS_DIR="$PROJECT_ROOT/functions"

GIT_SHA="${GIT_SHA:-$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || true)}"
if [[ -z "$IMAGE_TAG" ]]; then
    if [[ -n "$GIT_SHA" ]]; then
        IMAGE_TAG="${GIT_SHA:0:12}"
    else
        IMAGE_TAG="manual-$(date -u +%Y%m%d%H%M%S)"
    fi
fi
RELEASE_ID="${RELEASE_ID:-$IMAGE_TAG}"
DEPLOYMENT_MANIFEST_PATH="${DEPLOYMENT_MANIFEST_PATH:-$PROJECT_ROOT/output/deployments/pipeline-deployment-manifest.json}"
TOPOLOGY_EVIDENCE_PATH="${TOPOLOGY_EVIDENCE_PATH:-$PROJECT_ROOT/output/deployments/terraform-topology-evidence.json}"
DEPLOYMENT_CANARY_PATH="${DEPLOYMENT_CANARY_PATH:-$PROJECT_ROOT/output/deployments/deployment-service-canaries.json}"
FULL_TEST_LANE_PROVENANCE_PATH="${FULL_TEST_LANE_PROVENANCE_PATH:-$PROJECT_ROOT/output/deployments/full-test-lane-provenance.json}"

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

current_git_sha() {
    git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || true
}

validate_release_image_tag() {
    if [[ -z "$IMAGE_TAG" ]]; then
        log_error "IMAGE_TAG must be non-empty."
        exit 2
    fi
    case "$IMAGE_TAG" in
        latest|dev|test|local)
            log_error "IMAGE_TAG must be a release tag or git SHA, not '${IMAGE_TAG}'."
            exit 2
            ;;
        *[!A-Za-z0-9_.-]*)
            log_error "IMAGE_TAG contains unsupported characters: ${IMAGE_TAG}"
            exit 2
            ;;
    esac
}

validate_secret_name() {
    local variable_name="$1"
    local secret_name="$2"
    if [[ -z "$secret_name" || ! "$secret_name" =~ ^[A-Za-z0-9_-]{1,255}$ ]]; then
        log_error "${variable_name} must name an existing Secret Manager secret."
        exit 2
    fi
}

validate_runtime_secret_references() {
    validate_secret_name "PRIVACY_RUNNER_TOKEN_SECRET_NAME" "$PRIVACY_RUNNER_TOKEN_SECRET_NAME"
    validate_secret_name "VIDEO_TO_WORLD_RUNNER_TOKEN_SECRET_NAME" "$VIDEO_TO_WORLD_RUNNER_TOKEN_SECRET_NAME"
    validate_secret_name "PIPELINE_SYNC_TOKEN_SECRET_NAME" "$PIPELINE_SYNC_TOKEN_SECRET_NAME"
    validate_secret_name "WORLDLABS_API_KEY_SECRET_NAME" "$WORLDLABS_API_KEY_SECRET_NAME"
    if [[ "$PIPELINE_SYNC_WEBAPP_URL" != https://* ]]; then
        log_error "PIPELINE_SYNC_WEBAPP_URL is required and must use https://."
        exit 2
    fi
    if [[ -n "$HUGGINGFACE_TOKEN_SECRET_NAME" && ! "$HUGGINGFACE_TOKEN_SECRET_NAME" =~ ^[A-Za-z0-9_-]{1,255}$ ]]; then
        log_error "HUGGINGFACE_TOKEN_SECRET_NAME must be empty or name an existing Secret Manager secret."
        exit 2
    fi
}

validate_beta_data_residency() {
    if [[ "$PRIMARY_REGION" != us-* ]]; then
        log_error "US-only beta policy forbids primary region ${PRIMARY_REGION}."
        exit 2
    fi
    local region
    local -a configured_secondary_regions
    IFS=',' read -r -a configured_secondary_regions <<< "$SECONDARY_REGIONS"
    for region in "${configured_secondary_regions[@]}"; do
        if [[ -n "$region" && "$region" != us-* ]]; then
            log_error "US-only beta policy forbids secondary region ${region}."
            exit 2
        fi
    done
}

resolve_image_digest_uri() {
    local tagged_uri="$1"
    local repo="${tagged_uri%:*}"
    local digest_uri
    local digest

    digest_uri="$(gcloud container images describe "$tagged_uri" \
        --format='get(image_summary.fully_qualified_digest)' 2>/dev/null || true)"
    if [[ -n "$digest_uri" && "$digest_uri" != "None" ]]; then
        echo "$digest_uri"
        return 0
    fi

    digest="$(gcloud container images describe "$tagged_uri" \
        --format='get(image_summary.digest)' 2>/dev/null || true)"
    if [[ "$digest" == sha256:* ]]; then
        echo "${repo}@${digest}"
        return 0
    fi

    return 1
}

set_release_image_uris() {
    IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
    SAM3_IMAGE_URI="gcr.io/${PROJECT_ID}/${SAM3_IMAGE_NAME}:${IMAGE_TAG}"
    VIP_IMAGE_URI="gcr.io/${PROJECT_ID}/${VIP_IMAGE_NAME}:${IMAGE_TAG}"
    DEEPPRIVACY2_IMAGE_URI="gcr.io/${PROJECT_ID}/${DEEPPRIVACY2_IMAGE_NAME}:${IMAGE_TAG}"
    VIDEO_TO_WORLD_IMAGE_URI="gcr.io/${PROJECT_ID}/${VIDEO_TO_WORLD_IMAGE_NAME}:${IMAGE_TAG}"
}

require_resolved_image_matches() {
    local image_name="$1"
    local expected_digest_uri="$2"
    local resolved_digest_uri="$3"
    if [[ ! "$resolved_digest_uri" =~ @sha256:[0-9a-f]{64}$ ]]; then
        log_error "${image_name} did not resolve to an immutable sha256 image digest."
        exit 2
    fi
    if [[ -n "$expected_digest_uri" && "$expected_digest_uri" != "$resolved_digest_uri" ]]; then
        log_error "${image_name} tag/digest mismatch: expected ${expected_digest_uri}, registry returned ${resolved_digest_uri}."
        exit 2
    fi
}

pin_pushed_image_digests() {
    log_info "Resolving immutable image digests for release ${IMAGE_TAG}..."

    local expected_pipeline="$IMAGE_DIGEST_URI"
    local expected_sam3="$SAM3_IMAGE_DIGEST_URI"
    local expected_vip="$VIP_IMAGE_DIGEST_URI"
    local expected_deepprivacy2="$DEEPPRIVACY2_IMAGE_DIGEST_URI"
    local expected_video_to_world="$VIDEO_TO_WORLD_IMAGE_DIGEST_URI"
    local resolved_pipeline
    local resolved_sam3
    local resolved_vip
    local resolved_deepprivacy2
    local resolved_video_to_world

    resolved_pipeline="$(resolve_image_digest_uri "$IMAGE_URI")" || {
        log_error "Could not resolve digest for $IMAGE_URI"
        exit 1
    }
    resolved_sam3="$(resolve_image_digest_uri "$SAM3_IMAGE_URI")" || {
        log_error "Could not resolve digest for $SAM3_IMAGE_URI"
        exit 1
    }
    resolved_vip="$(resolve_image_digest_uri "$VIP_IMAGE_URI")" || {
        log_error "Could not resolve digest for $VIP_IMAGE_URI"
        exit 1
    }
    resolved_deepprivacy2="$(resolve_image_digest_uri "$DEEPPRIVACY2_IMAGE_URI")" || {
        log_error "Could not resolve digest for $DEEPPRIVACY2_IMAGE_URI"
        exit 1
    }
    resolved_video_to_world="$(resolve_image_digest_uri "$VIDEO_TO_WORLD_IMAGE_URI")" || {
        log_error "Could not resolve digest for $VIDEO_TO_WORLD_IMAGE_URI"
        exit 1
    }
    require_resolved_image_matches "pipeline image" "$expected_pipeline" "$resolved_pipeline"
    require_resolved_image_matches "SAM3 image" "$expected_sam3" "$resolved_sam3"
    require_resolved_image_matches "VIP image" "$expected_vip" "$resolved_vip"
    require_resolved_image_matches "DeepPrivacy2 image" "$expected_deepprivacy2" "$resolved_deepprivacy2"
    require_resolved_image_matches "video-to-world image" "$expected_video_to_world" "$resolved_video_to_world"
    IMAGE_DIGEST_URI="$resolved_pipeline"
    SAM3_IMAGE_DIGEST_URI="$resolved_sam3"
    VIP_IMAGE_DIGEST_URI="$resolved_vip"
    DEEPPRIVACY2_IMAGE_DIGEST_URI="$resolved_deepprivacy2"
    VIDEO_TO_WORLD_IMAGE_DIGEST_URI="$resolved_video_to_world"
}

validate_terraform_state_backend() {
    if [[ -z "$TERRAFORM_STATE_BUCKET" || ! "$TERRAFORM_STATE_BUCKET" =~ ^[A-Za-z0-9._-]+$ ]]; then
        log_error "TERRAFORM_STATE_BUCKET must name the approved remote GCS state bucket."
        exit 2
    fi
    if [[ -z "$TERRAFORM_STATE_PREFIX" || "$TERRAFORM_STATE_PREFIX" == /* || "$TERRAFORM_STATE_PREFIX" == *..* ]]; then
        log_error "TERRAFORM_STATE_PREFIX must be a contained nonempty GCS prefix."
        exit 2
    fi
    if [[ ! "$TERRAFORM_STATE_KMS_KEY" =~ ^projects/[^/]+/locations/us[^/]*/keyRings/[^/]+/cryptoKeys/[^/]+$ ]]; then
        log_error "TERRAFORM_STATE_KMS_KEY must name an approved US Cloud KMS key."
        exit 2
    fi
    local bucket_metadata
    bucket_metadata="$(gcloud storage buckets describe "gs://${TERRAFORM_STATE_BUCKET}" --format=json)" || {
        log_error "Unable to read the configured Terraform state bucket."
        exit 2
    }
    uv run --frozen python "$PROJECT_ROOT/scripts/validate_terraform_state_backend.py" \
        --expected-bucket "$TERRAFORM_STATE_BUCKET" \
        --expected-kms-key "$TERRAFORM_STATE_KMS_KEY" \
        <<< "$bucket_metadata"
}

check_full_test_lane_deploy_gate() {
    local current_sha
    current_sha="$(current_git_sha)"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would require Full Test Lane success for ${current_sha:-<unknown git sha>} before deploy"
        return
    fi

    if [[ "$FULL_TEST_LANE_REQUIRED" != "true" ]]; then
        log_error "FULL_TEST_LANE_REQUIRED must remain true; this deploy path has no text-only CI bypass."
        exit 2
    fi

    if [[ -z "$current_sha" ]]; then
        log_error "Could not determine the current git SHA; refusing deploy without Full Test Lane commit evidence."
        exit 2
    fi
    if [[ -n "$FULL_TEST_LANE_COMMIT" && "$FULL_TEST_LANE_COMMIT" != "$current_sha" ]]; then
        log_error "Full Test Lane deploy gate failed. Set FULL_TEST_LANE_COMMIT=${current_sha} only after Full Test Lane / Full pytest lane on CPU runner passed for this exact commit."
        exit 2
    fi
    if [[ -z "$FULL_TEST_LANE_EVIDENCE_URI" ]]; then
        log_error "FULL_TEST_LANE_EVIDENCE_URI must point to the successful Full Test Lane run or archived artifact for ${current_sha}."
        exit 2
    fi

    uv run --frozen python "$PROJECT_ROOT/scripts/verify_deploy_release_provenance.py" \
        --root "$PROJECT_ROOT" \
        --expected-sha "$current_sha" \
        --run-url "$FULL_TEST_LANE_EVIDENCE_URI" \
        --output "$FULL_TEST_LANE_PROVENANCE_PATH"
    FULL_TEST_LANE_COMMIT="$current_sha"
    log_success "Canonical Full Test Lane provenance verified for ${current_sha}: ${FULL_TEST_LANE_EVIDENCE_URI}"
}

verify_clean_release_source() {
    local current_sha
    local origin_main_sha
    local source_status
    current_sha="$(current_git_sha)"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would require a clean main checkout at origin/main before deployment"
        return
    fi
    if [[ -z "$current_sha" || ! "$current_sha" =~ ^[0-9a-f]{40}$ ]]; then
        log_error "Current source SHA is unavailable or malformed."
        exit 2
    fi
    if [[ -n "$GIT_SHA" && "$GIT_SHA" != "$current_sha" ]]; then
        log_error "GIT_SHA (${GIT_SHA}) does not match checked-out source (${current_sha})."
        exit 2
    fi
    source_status="$(git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=all)"
    if [[ -n "$source_status" ]]; then
        log_error "Deployment requires a clean checkout; tracked or untracked source changes are present."
        exit 2
    fi
    git -C "$PROJECT_ROOT" fetch --quiet origin main
    origin_main_sha="$(git -C "$PROJECT_ROOT" rev-parse refs/remotes/origin/main)"
    if [[ "$current_sha" != "$origin_main_sha" ]]; then
        log_error "Checked-out source ${current_sha} is not in exact parity with origin/main ${origin_main_sha}."
        exit 2
    fi
    GIT_SHA="$current_sha"
    log_success "Clean source and origin/main parity verified at ${current_sha}"
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
    check_command python3
    check_command gh
    check_command uv

    python3 "$PROJECT_ROOT/scripts/validate_pubsub_handoff_infra.py"
    verify_clean_release_source
    check_full_test_lane_deploy_gate
    validate_release_image_tag
    validate_runtime_secret_references
    validate_beta_data_residency

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

check_rollback_prerequisites() {
    log_info "Checking rollback prerequisites..."

    check_command gcloud
    check_command jq
    if [[ -n "$ROLLBACK_VERIFY_COMMAND" ]]; then
        check_command bash
    fi

    log_success "Rollback prerequisites check passed"
}

rollback_deployment() {
    if [[ -z "$ROLLBACK_IMAGE_TAG" ]]; then
        log_error "Rollback requires --rollback-image-tag <tag> or ROLLBACK_IMAGE_TAG."
        exit 2
    fi

    check_rollback_prerequisites

    local rollback_image_uri="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${ROLLBACK_IMAGE_TAG}"
    IFS=',' read -ra REGIONS <<< "${PRIMARY_REGION},${SECONDARY_REGIONS}"

    log_warning "Rolling blueprint-pipeline Cloud Run jobs back to ${rollback_image_uri}"
    for region in "${REGIONS[@]}"; do
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY-RUN] Would update Cloud Run Job blueprint-pipeline in ${region} to ${rollback_image_uri}"
            continue
        fi

        gcloud run jobs update blueprint-pipeline \
            --image "$rollback_image_uri" \
            --region "$region" \
            --quiet
    done

    if [[ -n "$ROLLBACK_VERIFY_COMMAND" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY-RUN] Would run rollback verification: ${ROLLBACK_VERIFY_COMMAND}"
        else
            log_info "Running rollback verification: ${ROLLBACK_VERIFY_COMMAND}"
            (cd "$PROJECT_ROOT" && bash -lc "$ROLLBACK_VERIFY_COMMAND")
        fi
    fi

    if [[ "$ROLLBACK_HEALTH_CHECK" == "true" ]]; then
        for region in "${REGIONS[@]}"; do
            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "[DRY-RUN] Would verify deployed image for blueprint-pipeline in ${region}"
                continue
            fi

            local actual_image
            actual_image="$(gcloud run jobs describe blueprint-pipeline \
                --region "$region" \
                --format=json | jq -r '.template.template.containers[0].image // ""')"
            if [[ "$actual_image" != "$rollback_image_uri" ]]; then
                log_error "Rollback health check failed in ${region}: expected ${rollback_image_uri}, got ${actual_image:-<empty>}"
                exit 1
            fi
            log_success "Rollback health check passed in ${region}: ${actual_image}"
        done
    fi

    log_success "Rollback complete. Record the image tag, verification output, and incident id before closing."
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

    set_release_image_uris

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would build: $IMAGE_URI"
        log_info "[DRY-RUN] Would build: $SAM3_IMAGE_URI"
        log_info "[DRY-RUN] Would build: $VIP_IMAGE_URI"
        log_info "[DRY-RUN] Would build: $DEEPPRIVACY2_IMAGE_URI"
        log_info "[DRY-RUN] Would build: $VIDEO_TO_WORLD_IMAGE_URI"
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

    docker build \
        -t "$VIDEO_TO_WORLD_IMAGE_URI" \
        -f deploy/docker/video_to_world/Dockerfile \
        .

    log_success "Docker images built"
}

push_docker_image() {
    log_info "Pushing Docker images to GCR..."

    set_release_image_uris

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would push: $IMAGE_URI"
        log_info "[DRY-RUN] Would push: $SAM3_IMAGE_URI"
        log_info "[DRY-RUN] Would push: $VIP_IMAGE_URI"
        log_info "[DRY-RUN] Would push: $DEEPPRIVACY2_IMAGE_URI"
        log_info "[DRY-RUN] Would push: $VIDEO_TO_WORLD_IMAGE_URI"
        return
    fi

    # Configure Docker for GCR
    gcloud auth configure-docker gcr.io --quiet

    docker push "$IMAGE_URI"
    docker push "$SAM3_IMAGE_URI"
    docker push "$VIP_IMAGE_URI"
    docker push "$DEEPPRIVACY2_IMAGE_URI"
    docker push "$VIDEO_TO_WORLD_IMAGE_URI"
    pin_pushed_image_digests

    log_success "Docker images pushed"
}

apply_terraform() {
    log_info "Applying Terraform configuration..."

    cd "$TERRAFORM_DIR"

    local pipeline_image="${IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}}"
    local sam3_image="${SAM3_IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${SAM3_IMAGE_NAME}:${IMAGE_TAG}}"
    local vip_image="${VIP_IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${VIP_IMAGE_NAME}:${IMAGE_TAG}}"
    local deepprivacy2_image="${DEEPPRIVACY2_IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${DEEPPRIVACY2_IMAGE_NAME}:${IMAGE_TAG}}"
    local video_to_world_image_ref="${VIDEO_TO_WORLD_IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${VIDEO_TO_WORLD_IMAGE_NAME}:${IMAGE_TAG}}"

    if [[ "$DRY_RUN" != "true" ]]; then
        require_resolved_image_matches "pipeline image" "$pipeline_image" "$pipeline_image"
        require_resolved_image_matches "SAM3 image" "$sam3_image" "$sam3_image"
        require_resolved_image_matches "VIP image" "$vip_image" "$vip_image"
        require_resolved_image_matches "DeepPrivacy2 image" "$deepprivacy2_image" "$deepprivacy2_image"
        require_resolved_image_matches "video-to-world image" "$video_to_world_image_ref" "$video_to_world_image_ref"
    fi

    # Terraform receives only non-secret configuration and Secret Manager
    # resource names. Secret payloads never enter tfvars, argv, environment, or
    # state. Existing secrets are resolved by Terraform data sources.
    export TF_VAR_project_id="$PROJECT_ID"
    export TF_VAR_primary_region="$PRIMARY_REGION"
    export TF_VAR_secondary_regions
    TF_VAR_secondary_regions="$(jq -cn --arg regions "$SECONDARY_REGIONS" '$regions | split(",") | map(select(length > 0))')"
    export TF_VAR_storage_bucket="$STORAGE_BUCKET"
    export TF_VAR_docker_image="$pipeline_image"
    export TF_VAR_privacy_sam3_image="$sam3_image"
    export TF_VAR_privacy_vip_image="$vip_image"
    export TF_VAR_privacy_deepprivacy2_image="$deepprivacy2_image"
    export TF_VAR_video_to_world_image="$video_to_world_image_ref"
    export TF_VAR_privacy_runner_token_secret_name="$PRIVACY_RUNNER_TOKEN_SECRET_NAME"
    export TF_VAR_video_to_world_runner_token_secret_name="$VIDEO_TO_WORLD_RUNNER_TOKEN_SECRET_NAME"
    export TF_VAR_worldlabs_api_key_secret_name="$WORLDLABS_API_KEY_SECRET_NAME"
    export TF_VAR_pipeline_sync_token_secret_name="$PIPELINE_SYNC_TOKEN_SECRET_NAME"
    export TF_VAR_pipeline_sync_webapp_url="$PIPELINE_SYNC_WEBAPP_URL"
    export TF_VAR_huggingface_token_secret_name="$HUGGINGFACE_TOKEN_SECRET_NAME"
    export TF_VAR_video_to_world_pipeline_preset="$VIDEO_TO_WORLD_PIPELINE_PRESET"
    export TF_VAR_video_to_world_command_template="$VIDEO_TO_WORLD_COMMAND_TEMPLATE"
    export TF_VAR_sam3_weights_path="$SAM3_WEIGHTS_PATH"
    export TF_VAR_vip_model_path="$VIP_MODEL_PATH"
    export TF_VAR_deepprivacy2_model_path="$DEEPPRIVACY2_MODEL_PATH"
    export TF_VAR_depth_anything_model_path="$DEPTH_ANYTHING_MODEL_PATH"
    export TF_VAR_blueprint_preview_provider="$BLUEPRINT_PREVIEW_PROVIDER"
    export TF_VAR_worldlabs_default_model="$WORLDLABS_DEFAULT_MODEL"
    export TF_VAR_privacy_pipeline_enabled="$PRIVACY_PIPELINE_ENABLED"
    export TF_VAR_privacy_fail_closed="$PRIVACY_FAIL_CLOSED"

    validate_terraform_state_backend
    local -a terraform_init_args=(
        init
        -input=false
        -reconfigure
        "-backend-config=bucket=${TERRAFORM_STATE_BUCKET}"
        "-backend-config=prefix=${TERRAFORM_STATE_PREFIX}"
        "-backend-config=kms_encryption_key=${TERRAFORM_STATE_KMS_KEY}"
    )

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would run: terraform init && terraform plan"
        terraform "${terraform_init_args[@]}"
        terraform plan -input=false
        return
    fi

    # Initialize Terraform
    terraform "${terraform_init_args[@]}"

    # Plan and apply
    terraform plan -input=false -out=tfplan

    if confirm "Apply Terraform changes?"; then
        terraform apply -input=false tfplan
        rm -f tfplan
    else
        log_error "Terraform apply was not approved; no deployment was completed."
        exit 2
    fi

    # Terraform is the sole declared topology owner. Refreshing and requiring a
    # zero-change plan makes provider-read drift a hard deployment failure.
    set +e
    terraform plan -input=false -detailed-exitcode -out=postapply-drift-check.tfplan
    local drift_exit=$?
    set -e
    rm -f postapply-drift-check.tfplan
    if [[ $drift_exit -eq 2 ]]; then
        log_error "Post-apply Terraform drift detected; refusing launch evidence."
        exit 2
    fi
    if [[ $drift_exit -ne 0 ]]; then
        log_error "Post-apply Terraform drift check failed."
        exit 2
    fi

    mkdir -p "$(dirname "$TOPOLOGY_EVIDENCE_PATH")"
    local terraform_outputs
    terraform_outputs="$(terraform output -json)"
    jq -n \
        --arg schema_version "blueprint.terraform_topology_evidence.v1" \
        --arg generated_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg release_id "$RELEASE_ID" \
        --arg git_sha "${GIT_SHA:-}" \
        --arg terraform_workspace "$(terraform workspace show)" \
        --argjson terraform_outputs "$terraform_outputs" \
        '{
          schema_version: $schema_version,
          generated_at: $generated_at,
          release_id: $release_id,
          git_sha: $git_sha,
          topology_owner: "terraform",
          provider_refresh_zero_drift: true,
          terraform_workspace: $terraform_workspace,
          terraform_outputs: $terraform_outputs,
          blockers: [],
          claim_boundary: "Provider-refreshed Terraform state; separate service canaries are still required for live behavior proof."
        }' > "$TOPOLOGY_EVIDENCE_PATH"

    uv run --frozen python "$PROJECT_ROOT/scripts/run_deployment_service_canaries.py" \
        --topology-evidence "$TOPOLOGY_EVIDENCE_PATH" \
        --project-id "$PROJECT_ID" \
        --privacy-secret-name "$PRIVACY_RUNNER_TOKEN_SECRET_NAME" \
        --video-secret-name "$VIDEO_TO_WORLD_RUNNER_TOKEN_SECRET_NAME" \
        --output "$DEPLOYMENT_CANARY_PATH"
    if [[ "$(jq -r '.status // ""' "$DEPLOYMENT_CANARY_PATH")" != "passed" ]]; then
        log_error "Authenticated deployment service canaries did not pass."
        exit 2
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
        --trigger-location us \
        --memory 512M \
        --timeout 60s \
        --set-env-vars "^~^PIPELINE_PROJECT_ID=${PROJECT_ID}~PIPELINE_REGION=${PRIMARY_REGION}~REGIONS=${SECONDARY_REGIONS}~SWAP_TRIGGER_DISPATCH_MODE=pubsub~SWAP_TRIGGER_PUBSUB_TOPIC=${SWAP_TOPIC}~SWAP_TRIGGER_HANDOFF_PUBSUB_TOPIC=${HANDOFF_TOPIC}~SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF=true"

    gcloud functions deploy swap-dispatch-worker \
        --gen2 \
        --runtime python311 \
        --region "$PRIMARY_REGION" \
        --source . \
        --entry-point on_swap_dispatch \
        --trigger-topic "${SWAP_TOPIC}" \
        --memory 4096M \
        --timeout 3600s \
        --set-env-vars "^~^PIPELINE_PROJECT_ID=${PROJECT_ID}~PIPELINE_REGION=${PRIMARY_REGION}~REGIONS=${SECONDARY_REGIONS}~PIPELINE_EXECUTION_MODE=cloud_run_job~PIPELINE_RUN_JOB_NAME=blueprint-pipeline~PIPELINE_RUN_JOB_REGION=${PRIMARY_REGION}"

    log_success "Cloud Function deployed"
}

create_cloud_run_jobs() {
    log_error "Manual Cloud Run mutation is disabled; Terraform is the sole topology and Secret Manager reference owner."
    return 2
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

    TOPICS=("$SWAP_TOPIC" "$HANDOFF_TOPIC" "pipeline-trigger-dlq")

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

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would create pull subscription: blueprint-pipeline-handoff-listener"
    elif gcloud pubsub subscriptions describe blueprint-pipeline-handoff-listener &>/dev/null; then
        log_info "Subscription blueprint-pipeline-handoff-listener already exists"
    else
        gcloud pubsub subscriptions create blueprint-pipeline-handoff-listener \
            --topic "$HANDOFF_TOPIC" \
            --ack-deadline 600 \
            --message-retention-duration 7d \
            --min-retry-delay 60s \
            --max-retry-delay 600s \
            --dead-letter-topic pipeline-trigger-dlq \
            --max-delivery-attempts 5 \
            --quiet
        log_info "Created subscription: blueprint-pipeline-handoff-listener"
    fi

    log_success "Pub/Sub topics and subscriptions created"
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
    echo "Release ID: $RELEASE_ID"
    echo "Git SHA: ${GIT_SHA:-unknown}"
    echo "Primary Region: $PRIMARY_REGION"
    echo "Secondary Regions: $SECONDARY_REGIONS"
    echo ""
    echo "Resources:"
    echo "  - Docker Image: ${IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}}"
    echo "  - Privacy Service Image (SAM3): ${SAM3_IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${SAM3_IMAGE_NAME}:${IMAGE_TAG}}"
    echo "  - Privacy Service Image (VIP): ${VIP_IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${VIP_IMAGE_NAME}:${IMAGE_TAG}}"
    echo "  - Privacy Service Image (DeepPrivacy2): ${DEEPPRIVACY2_IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${DEEPPRIVACY2_IMAGE_NAME}:${IMAGE_TAG}}"
    echo "  - Geometry Service Image (video_to_world): ${VIDEO_TO_WORLD_IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${VIDEO_TO_WORLD_IMAGE_NAME}:${IMAGE_TAG}}"
    echo "  - Cloud Run Job (CPU): blueprint-pipeline"
    echo "  - Cloud Run Services (GPU): sam3-detect, vip-inpaint, deepprivacy2-anonymize, video-to-world"
    echo "  - Cloud Function: storage-trigger"
    echo "  - Cloud Tasks Queue: blueprint-pipeline-queue"
    echo "  - Pub/Sub Topic: ${SWAP_TOPIC}"
    echo "  - Pub/Sub Topic (capture-bridge handoff): ${HANDOFF_TOPIC}"
    echo "  - Pub/Sub Subscription: blueprint-pipeline-handoff-listener"
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

write_deployment_manifest() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would write deployment manifest to ${DEPLOYMENT_MANIFEST_PATH}"
        return
    fi

    mkdir -p "$(dirname "$DEPLOYMENT_MANIFEST_PATH")"
    jq -n \
        --arg schema_version "blueprint.pipeline_deployment_manifest.v1" \
        --arg created_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg release_id "$RELEASE_ID" \
        --arg git_sha "${GIT_SHA:-}" \
        --arg image_tag "$IMAGE_TAG" \
        --arg pipeline_image "${IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}}" \
        --arg sam3_image "${SAM3_IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${SAM3_IMAGE_NAME}:${IMAGE_TAG}}" \
        --arg vip_image "${VIP_IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${VIP_IMAGE_NAME}:${IMAGE_TAG}}" \
        --arg deepprivacy2_image "${DEEPPRIVACY2_IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${DEEPPRIVACY2_IMAGE_NAME}:${IMAGE_TAG}}" \
        --arg video_to_world_image "${VIDEO_TO_WORLD_IMAGE_DIGEST_URI:-gcr.io/${PROJECT_ID}/${VIDEO_TO_WORLD_IMAGE_NAME}:${IMAGE_TAG}}" \
        --arg full_test_lane_commit "$FULL_TEST_LANE_COMMIT" \
        --arg full_test_lane_evidence_uri "$FULL_TEST_LANE_EVIDENCE_URI" \
        --arg full_test_lane_provenance_path "$FULL_TEST_LANE_PROVENANCE_PATH" \
        --arg topology_evidence_path "$TOPOLOGY_EVIDENCE_PATH" \
        --arg deployment_canary_path "$DEPLOYMENT_CANARY_PATH" \
        --argjson full_test_lane_required "$([[ "$FULL_TEST_LANE_REQUIRED" == "true" ]] && echo true || echo false)" \
        '{
          schema_version: $schema_version,
          created_at_utc: $created_at_utc,
          release_id: $release_id,
          git_sha: $git_sha,
          image_tag: $image_tag,
          images: {
            pipeline: $pipeline_image,
            privacy_sam3: $sam3_image,
            privacy_vip: $vip_image,
            privacy_deepprivacy2: $deepprivacy2_image,
            video_to_world: $video_to_world_image
          },
          full_test_lane: {
            required: $full_test_lane_required,
            commit: $full_test_lane_commit,
            evidence_uri: $full_test_lane_evidence_uri,
            verified_provenance_path: $full_test_lane_provenance_path
          },
          rollback: {
            command_template: "deploy/scripts/deploy.sh --rollback --rollback-image-tag " + $image_tag,
            rollback_image_tag: $image_tag
          },
          topology: {
            owner: "terraform",
            evidence_path: $topology_evidence_path,
            provider_refresh_zero_drift_required: true
          },
          authenticated_service_canaries: {
            required: true,
            evidence_path: $deployment_canary_path
          },
          claim_boundary: "Provider-refreshed Terraform topology plus authenticated no-op service canaries; provider/model task success remains separately required."
        }' > "$DEPLOYMENT_MANIFEST_PATH"
    log_success "Deployment manifest written: ${DEPLOYMENT_MANIFEST_PATH}"
}

# =============================================================================
# Main
# =============================================================================

main() {
    local DOCKER_ONLY=false
    local TERRAFORM_ONLY=false
    local DRY_RUN=false
    local SKIP_DOCKER=false
    local ROLLBACK=false

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
                log_error "--function-only was removed: Terraform is the sole deployment topology owner."
                exit 2
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --rollback-image-tag|--rollback-tag)
                if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                    log_error "$1 requires a non-empty image tag."
                    exit 2
                fi
                ROLLBACK_IMAGE_TAG="${2:-}"
                shift 2
                ;;
            --verify-command)
                if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                    log_error "$1 requires a non-empty command."
                    exit 2
                fi
                ROLLBACK_VERIFY_COMMAND="${2:-}"
                shift 2
                ;;
            --skip-rollback-verify)
                ROLLBACK_VERIFY_COMMAND=""
                shift
                ;;
            --skip-rollback-health)
                ROLLBACK_HEALTH_CHECK=false
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
                echo "  --function-only   Removed; Terraform owns Cloud Function topology"
                echo "  --skip-docker     Skip Docker build (use existing image)"
                echo "  --rollback        Roll Cloud Run jobs back to --rollback-image-tag"
                echo "  --rollback-image-tag <tag>"
                echo "                    Known-good ${IMAGE_NAME} image tag to restore"
                echo "  --verify-command <cmd>"
                echo "                    Local rollback verification command"
                echo "  --skip-rollback-verify"
                echo "                    Do not run the local rollback verification command"
                echo "  --skip-rollback-health"
                echo "                    Do not verify the deployed Cloud Run job image"
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

    if [[ "$ROLLBACK" == "true" ]]; then
        rollback_deployment
        exit 0
    fi

    # Run deployment steps
    check_prerequisites

    if [[ "$DOCKER_ONLY" == "true" ]]; then
        build_docker_image
        push_docker_image
        exit 0
    fi

    if [[ "$TERRAFORM_ONLY" == "true" ]]; then
        set_release_image_uris
        if [[ "$DRY_RUN" != "true" ]]; then
            pin_pushed_image_digests
        fi
        apply_terraform
        write_deployment_manifest
        exit 0
    fi

    # Full deployment
    if [[ "$SKIP_DOCKER" != "true" ]]; then
        build_docker_image
        push_docker_image
    elif [[ "$DRY_RUN" != "true" ]]; then
        set_release_image_uris
        pin_pushed_image_digests
    fi

    apply_terraform

    write_deployment_manifest

    print_summary

    log_success "Deployment complete!"
}

main "$@"
