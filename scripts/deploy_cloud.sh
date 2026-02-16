#!/bin/bash
# Deploy the complete cloud infrastructure for BlueprintCapturePipeline
# This script sets up:
# 1. Cloud Function for storage triggers
# 2. Cloud Tasks queue for job management
# 3. Cloud Run Jobs with GPU
# 4. Firestore indexes for status tracking
# 5. IAM permissions
# 6. Monitoring dashboards

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-blueprint-8c1ca}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-blueprint-8c1ca.appspot.com}"
CLOUD_RUN_JOB="${CLOUD_RUN_JOB:-blueprint-pipeline}"
TASK_QUEUE="${TASK_QUEUE:-blueprint-pipeline-queue}"
DLQ_NAME="${DLQ_NAME:-blueprint-pipeline-dlq}"
SWAP_TOPIC="${SWAP_TOPIC:-pipeline-trigger}"

echo "🚀 Deploying BlueprintCapturePipeline Cloud Infrastructure"
echo "   Project: $PROJECT_ID"
echo "   Region: $REGION"
echo "   Bucket: $BUCKET"
echo "   Cloud Run Job: $CLOUD_RUN_JOB"
echo "   Task Queue: $TASK_QUEUE"
echo ""

# Check gcloud is authenticated
if ! gcloud auth list 2>&1 | grep -q "ACTIVE"; then
    echo "❌ Not authenticated with gcloud. Run: gcloud auth login"
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📦 Enabling required APIs..."
gcloud services enable \
    cloudfunctions.googleapis.com \
    cloudtasks.googleapis.com \
    run.googleapis.com \
    firestore.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com \
    --quiet

echo "✅ APIs enabled"

# ============================================================================
# Step 1: Create Cloud Tasks Queues
# ============================================================================
echo ""
echo "📦 Step 1: Creating Cloud Tasks queues..."

# Create main processing queue
gcloud tasks queues create $TASK_QUEUE \
    --location=$REGION \
    --max-dispatches-per-second=10 \
    --max-concurrent-dispatches=32 \
    --max-attempts=3 \
    --min-backoff=60s \
    --max-backoff=3600s \
    --max-doublings=3 \
    2>/dev/null || echo "   Queue $TASK_QUEUE already exists"

# Create dead letter queue
gcloud tasks queues create $DLQ_NAME \
    --location=$REGION \
    --max-dispatches-per-second=1 \
    --max-concurrent-dispatches=1 \
    2>/dev/null || echo "   Dead letter queue $DLQ_NAME already exists"

echo "✅ Cloud Tasks queues created"

# ============================================================================
# Step 2: Create Firestore Indexes
# ============================================================================
echo ""
echo "📦 Step 2: Creating Firestore indexes..."

# Create firestore.indexes.json if it doesn't exist
cat > /tmp/firestore.indexes.json << 'EOF'
{
  "indexes": [
    {
      "collectionGroup": "captures",
      "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "status", "order": "ASCENDING" },
        { "fieldPath": "createdAt", "order": "ASCENDING" }
      ]
    },
    {
      "collectionGroup": "captures",
      "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "sceneId", "order": "ASCENDING" },
        { "fieldPath": "createdAt", "order": "DESCENDING" }
      ]
    },
    {
      "collectionGroup": "captures",
      "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "creatorId", "order": "ASCENDING" },
        { "fieldPath": "createdAt", "order": "DESCENDING" }
      ]
    }
  ],
  "fieldOverrides": []
}
EOF

# Deploy indexes (this may take a while)
gcloud firestore indexes composite create \
    --collection-group=captures \
    --field-config=field-path=status,order=ascending \
    --field-config=field-path=createdAt,order=ascending \
    2>/dev/null || echo "   Index for status+createdAt already exists"

gcloud firestore indexes composite create \
    --collection-group=captures \
    --field-config=field-path=sceneId,order=ascending \
    --field-config=field-path=createdAt,order=descending \
    2>/dev/null || echo "   Index for sceneId+createdAt already exists"

echo "✅ Firestore indexes created"

# ============================================================================
# Step 3: Deploy Cloud Function
# ============================================================================
echo ""
echo "📦 Step 3: Deploying Cloud Function..."

cd "$(dirname "$0")/../functions"

# Deploy the function
gcloud functions deploy storage_trigger \
    --gen2 \
    --runtime python311 \
    --trigger-resource $BUCKET \
    --trigger-event google.storage.object.finalize \
    --entry-point on_storage_finalize \
    --region $REGION \
    --memory 512MB \
    --timeout 120s \
    --set-env-vars PIPELINE_PROJECT_ID=$PROJECT_ID,PIPELINE_REGION=$REGION,TASK_QUEUE=$TASK_QUEUE,SWAP_TRIGGER_DISPATCH_MODE=pubsub,SWAP_TRIGGER_PUBSUB_TOPIC=$SWAP_TOPIC \
    --quiet

echo "✅ Cloud Function deployed"

gcloud functions deploy swap_dispatch_worker \
    --gen2 \
    --runtime python311 \
    --trigger-topic $SWAP_TOPIC \
    --entry-point on_swap_dispatch \
    --region $REGION \
    --memory 4096MB \
    --timeout 3600s \
    --set-env-vars PIPELINE_PROJECT_ID=$PROJECT_ID,PIPELINE_REGION=$REGION,TASK_QUEUE=$TASK_QUEUE \
    --quiet

echo "✅ Dispatch worker deployed"

# ============================================================================
# Step 4: Build and Deploy Cloud Run Job
# ============================================================================
echo ""
echo "📦 Step 4: Building Cloud Run Job container..."

cd "$(dirname "$0")/.."

# Create Artifact Registry repository if it doesn't exist
gcloud artifacts repositories create blueprint-pipeline \
    --repository-format=docker \
    --location=$REGION \
    2>/dev/null || echo "   Artifact Registry repository already exists"

# Configure Docker for Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Build the container using Cloud Build
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/blueprint-pipeline/${CLOUD_RUN_JOB}:latest"

gcloud builds submit \
    --tag $IMAGE_URI \
    --timeout=45m \
    --machine-type=e2-highcpu-32 \
    --quiet

echo "   Container built: $IMAGE_URI"

# Create/update the Cloud Run Job with GPU
echo "   Deploying Cloud Run Job with GPU..."

gcloud run jobs create $CLOUD_RUN_JOB \
    --image $IMAGE_URI \
    --region $REGION \
    --cpu 8 \
    --memory 32Gi \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --task-timeout 90m \
    --max-retries 2 \
    --set-env-vars PIPELINE_PROJECT_ID=$PROJECT_ID,PIPELINE_REGION=$REGION,TASK_QUEUE=$TASK_QUEUE \
    2>/dev/null || \
gcloud run jobs update $CLOUD_RUN_JOB \
    --image $IMAGE_URI \
    --region $REGION \
    --cpu 8 \
    --memory 32Gi \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --task-timeout 90m \
    --max-retries 2 \
    --set-env-vars PIPELINE_PROJECT_ID=$PROJECT_ID,PIPELINE_REGION=$REGION,TASK_QUEUE=$TASK_QUEUE

echo "✅ Cloud Run Job deployed"

# ============================================================================
# Step 5: Create Service Account for Pipeline Invoker
# ============================================================================
echo ""
echo "📦 Step 5: Configuring IAM and Service Accounts..."

# Create pipeline invoker service account
INVOKER_SA="pipeline-invoker@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create pipeline-invoker \
    --description="Service account for invoking pipeline Cloud Run Jobs" \
    --display-name="Pipeline Invoker" \
    2>/dev/null || echo "   Service account already exists"

# Grant required roles to the service account
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$INVOKER_SA" \
    --role="roles/run.invoker" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$INVOKER_SA" \
    --role="roles/storage.objectAdmin" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$INVOKER_SA" \
    --role="roles/datastore.user" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$INVOKER_SA" \
    --role="roles/cloudtasks.enqueuer" \
    --quiet

# Grant Cloud Function default SA permissions
FUNCTION_SA="${PROJECT_ID}@appspot.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$FUNCTION_SA" \
    --role="roles/run.invoker" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$FUNCTION_SA" \
    --role="roles/datastore.user" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$FUNCTION_SA" \
    --role="roles/cloudtasks.enqueuer" \
    --quiet

echo "✅ IAM permissions configured"

# ============================================================================
# Step 6: Create Monitoring Dashboard
# ============================================================================
echo ""
echo "📦 Step 6: Creating monitoring dashboard..."

cat > /tmp/dashboard.json << EOF
{
  "displayName": "Blueprint Pipeline Dashboard",
  "gridLayout": {
    "columns": "2",
    "widgets": [
      {
        "title": "Cloud Run Job Executions",
        "xyChart": {
          "dataSets": [{
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "resource.type=\"cloud_run_job\" resource.labels.job_name=\"$CLOUD_RUN_JOB\"",
                "aggregation": {
                  "alignmentPeriod": "60s",
                  "perSeriesAligner": "ALIGN_RATE"
                }
              }
            }
          }]
        }
      },
      {
        "title": "Captures by Status",
        "xyChart": {
          "dataSets": [{
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "resource.type=\"firestore_instance\" metric.type=\"firestore.googleapis.com/document/read_count\"",
                "aggregation": {
                  "alignmentPeriod": "300s",
                  "perSeriesAligner": "ALIGN_SUM"
                }
              }
            }
          }]
        }
      },
      {
        "title": "Task Queue Depth",
        "xyChart": {
          "dataSets": [{
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "resource.type=\"cloud_tasks_queue\" resource.labels.queue_id=\"$TASK_QUEUE\"",
                "aggregation": {
                  "alignmentPeriod": "60s",
                  "perSeriesAligner": "ALIGN_MEAN"
                }
              }
            }
          }]
        }
      },
      {
        "title": "Cloud Function Invocations",
        "xyChart": {
          "dataSets": [{
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "resource.type=\"cloud_function\" resource.labels.function_name=\"storage_trigger\"",
                "aggregation": {
                  "alignmentPeriod": "60s",
                  "perSeriesAligner": "ALIGN_RATE"
                }
              }
            }
          }]
        }
      }
    ]
  }
}
EOF

gcloud monitoring dashboards create --config-from-file=/tmp/dashboard.json 2>/dev/null || echo "   Dashboard may already exist"

echo "✅ Monitoring dashboard created"

# ============================================================================
# Step 7: Create Alert Policies
# ============================================================================
echo ""
echo "📦 Step 7: Creating alert policies..."

# Create alert for failed jobs
cat > /tmp/alert-failed-jobs.json << EOF
{
  "displayName": "Pipeline Job Failures",
  "conditions": [{
    "displayName": "Cloud Run Job failed",
    "conditionThreshold": {
      "filter": "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"$CLOUD_RUN_JOB\" AND metric.type=\"run.googleapis.com/job/completed_execution_count\" AND metric.labels.result=\"failed\"",
      "aggregations": [{
        "alignmentPeriod": "300s",
        "perSeriesAligner": "ALIGN_SUM"
      }],
      "comparison": "COMPARISON_GT",
      "thresholdValue": 0,
      "duration": "0s"
    }
  }],
  "alertStrategy": {
    "autoClose": "604800s"
  },
  "combiner": "OR",
  "enabled": true
}
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/alert-failed-jobs.json 2>/dev/null || echo "   Alert policy may already exist"

echo "✅ Alert policies created"

# Cleanup temp files
rm -f /tmp/firestore.indexes.json /tmp/dashboard.json /tmp/alert-failed-jobs.json

# ============================================================================
# Done!
# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🎉 Deployment complete!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Infrastructure deployed:"
echo "  ✅ Cloud Function: storage_trigger"
echo "  ✅ Cloud Tasks Queue: $TASK_QUEUE"
echo "  ✅ Dead Letter Queue: $DLQ_NAME"
echo "  ✅ Cloud Run Job: $CLOUD_RUN_JOB (with NVIDIA L4 GPU)"
echo "  ✅ Firestore indexes for captures collection"
echo "  ✅ Monitoring dashboard and alerts"
echo ""
echo "Pipeline flow:"
echo "  1. iOS app uploads video to: gs://$BUCKET/scenes/{scene_id}/{source}/{timestamp}/raw/"
echo "  2. Cloud Function triggers when manifest.json + walkthrough.mov are uploaded"
echo "  3. Function creates Firestore status document in 'captures' collection"
echo "  4. Cloud Run Job executes with GPU for 3DGS reconstruction"
echo "  5. Status updated in Firestore, push notification sent to user"
echo "  6. Results saved to: gs://$BUCKET/sessions/{session_id}/"
echo ""
echo "Firestore status tracking:"
echo "  Collection: captures"
echo "  Fields: id, sceneId, creatorId, status, stage, progress, outputs, metrics"
echo "  Statuses: queued → processing → completed/failed"
echo ""
echo "Scaling for 1000+ scans/day:"
echo "  - Task queue handles rate limiting and retries"
echo "  - Dead letter queue captures persistent failures"
echo "  - Monitoring dashboard shows queue depth and job status"
echo ""
echo "Test with:"
echo "  gsutil cp -r /path/to/capture gs://$BUCKET/scenes/test-scene/iphone/\$(date +%Y-%m-%dT%H:%M:%S)-test/raw/"
echo ""
