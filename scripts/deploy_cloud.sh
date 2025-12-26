#!/bin/bash
# Deploy the complete cloud infrastructure for BlueprintCapturePipeline
# This script sets up:
# 1. Cloud Function for storage triggers
# 2. Pub/Sub topics and subscriptions
# 3. Cloud Run Jobs with GPU

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-blueprint-8c1ca}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-blueprint-8c1ca.appspot.com}"
CLOUD_RUN_JOB="${CLOUD_RUN_JOB:-blueprint-pipeline-job}"

echo "🚀 Deploying BlueprintCapturePipeline Cloud Infrastructure"
echo "   Project: $PROJECT_ID"
echo "   Region: $REGION"
echo "   Bucket: $BUCKET"
echo ""

# Check gcloud is authenticated
if ! gcloud auth list 2>&1 | grep -q "ACTIVE"; then
    echo "❌ Not authenticated with gcloud. Run: gcloud auth login"
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID

# ============================================================================
# Step 1: Deploy Cloud Function
# ============================================================================
echo "📦 Step 1: Deploying Cloud Function..."

cd "$(dirname "$0")/../functions"

# Install dependencies for function
pip install -r requirements.txt --target=.package --quiet

# Deploy the function
gcloud functions deploy storage_trigger \
    --gen2 \
    --runtime python311 \
    --trigger-resource $BUCKET \
    --trigger-event google.storage.object.finalize \
    --entry-point on_storage_finalize \
    --region $REGION \
    --memory 512MB \
    --timeout 60s \
    --set-env-vars PIPELINE_PROJECT_ID=$PROJECT_ID,PIPELINE_REGION=$REGION \
    --quiet

echo "✅ Cloud Function deployed"

# ============================================================================
# Step 2: Create Pub/Sub Topic and Subscription
# ============================================================================
echo "📦 Step 2: Creating Pub/Sub infrastructure..."

# Create topic (ignore if exists)
gcloud pubsub topics create pipeline-trigger 2>/dev/null || echo "   Topic already exists"

# Create subscription for Cloud Run Jobs
gcloud pubsub subscriptions create pipeline-job-trigger \
    --topic=pipeline-trigger \
    --push-endpoint="https://${CLOUD_RUN_JOB}-${PROJECT_ID}.${REGION}.run.app/trigger" \
    --ack-deadline=600 \
    --message-retention-duration=1h \
    2>/dev/null || echo "   Subscription already exists"

echo "✅ Pub/Sub infrastructure created"

# ============================================================================
# Step 3: Build and Deploy Cloud Run Job
# ============================================================================
echo "📦 Step 3: Building Cloud Run Job container..."

cd "$(dirname "$0")/.."

# Build the container
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/$CLOUD_RUN_JOB:latest \
    --timeout=30m \
    --quiet

# Create/update the Cloud Run Job with GPU
echo "   Deploying Cloud Run Job with GPU..."

gcloud run jobs create $CLOUD_RUN_JOB \
    --image gcr.io/$PROJECT_ID/$CLOUD_RUN_JOB:latest \
    --region $REGION \
    --cpu 8 \
    --memory 32Gi \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --task-timeout 60m \
    --max-retries 1 \
    --set-env-vars PIPELINE_PROJECT_ID=$PROJECT_ID,PIPELINE_REGION=$REGION \
    2>/dev/null || \
gcloud run jobs update $CLOUD_RUN_JOB \
    --image gcr.io/$PROJECT_ID/$CLOUD_RUN_JOB:latest \
    --region $REGION \
    --cpu 8 \
    --memory 32Gi \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --task-timeout 60m \
    --max-retries 1 \
    --set-env-vars PIPELINE_PROJECT_ID=$PROJECT_ID,PIPELINE_REGION=$REGION

echo "✅ Cloud Run Job deployed"

# ============================================================================
# Step 4: Set up IAM permissions
# ============================================================================
echo "📦 Step 4: Configuring IAM permissions..."

# Get the Cloud Function service account
FUNCTION_SA="$PROJECT_ID@appspot.gserviceaccount.com"

# Grant Cloud Run invoker role
gcloud run jobs add-iam-policy-binding $CLOUD_RUN_JOB \
    --region $REGION \
    --member="serviceAccount:$FUNCTION_SA" \
    --role="roles/run.invoker" \
    --quiet

# Grant storage access
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$FUNCTION_SA" \
    --role="roles/storage.objectViewer" \
    --quiet

echo "✅ IAM permissions configured"

# ============================================================================
# Done!
# ============================================================================
echo ""
echo "🎉 Deployment complete!"
echo ""
echo "Pipeline flow:"
echo "  1. Upload video to: gs://$BUCKET/scenes/{scene_id}/{source}/{timestamp}/raw/"
echo "  2. Cloud Function triggers when manifest.json + walkthrough.mov are uploaded"
echo "  3. Function publishes to Pub/Sub topic: pipeline-trigger"
echo "  4. Cloud Run Job executes with GPU for 3DGS reconstruction"
echo "  5. Results saved to: gs://$BUCKET/sessions/{session_id}/"
echo ""
echo "Test with:"
echo "  gsutil cp -r /path/to/capture gs://$BUCKET/scenes/test-scene/iphone/$(date +%Y-%m-%dT%H:%M:%S)-test/raw/"
