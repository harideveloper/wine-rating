#!/bin/bash

# Fix Vertex AI Pipeline Permissions
# Run this script to grant the necessary permissions

# Your configuration
PROJECT_NUMBER="1035860259529"
SERVICE_ACCOUNT="1035860259529-compute@developer.gserviceaccount.com"
PROJECT_ID_DEV1="dev1-bfa7"
PROJECT_ID_DEV2="dev2-ea8f"


# 1. Grant Vertex AI permissions to the service account in both projects
echo "Adding Vertex AI roles to service account..."

# For target project (dev1-bfa7)
gcloud projects add-iam-policy-binding $PROJECT_ID_DEV1 \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID_DEV1 \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/aiplatform.serviceAgent"

# For source project (dev2-ea8f)  
gcloud projects add-iam-policy-binding $PROJECT_ID_DEV2 \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID_DEV2 \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/aiplatform.serviceAgent"

# 2. Create/enable Vertex AI Metadata Store (if it doesn't exist)
echo "Ensuring Vertex AI Metadata Store exists..."

# Enable in target project
gcloud config set project $PROJECT_ID_DEV1
gcloud services enable aiplatform.googleapis.com

# Enable in source project
gcloud config set project $PROJECT_ID_DEV2
gcloud services enable aiplatform.googleapis.com

# 3. Grant additional storage permissions
echo "Adding storage permissions..."

gcloud projects add-iam-policy-binding $PROJECT_ID_DEV1 \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID_DEV2 \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/storage.objectViewer"

# 4. Grant logging permissions
echo "Adding logging permissions..."

gcloud projects add-iam-policy-binding $PROJECT_ID_DEV1 \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding $PROJECT_ID_DEV2 \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/logging.logWriter"

echo "✅ Permissions setup complete!"
echo ""
echo "🔍 Checking current permissions for service account:"
echo "Target project ($PROJECT_ID_DEV1):"
gcloud projects get-iam-policy $PROJECT_ID_DEV1 \
    --flatten="bindings[].members" \
    --format="table(bindings.role)" \
    --filter="bindings.members:$SERVICE_ACCOUNT"

echo ""
echo "Source project ($PROJECT_ID_DEV2):"
gcloud projects get-iam-policy $PROJECT_ID_DEV2 \
    --flatten="bindings[].members" \
    --format="table(bindings.role)" \
    --filter="bindings.members:$SERVICE_ACCOUNT"

echo ""
echo "🚀 Try running your pipeline again with: make promote"