#!/bin/bash
# set_env.sh - Set environment variables for local development

echo "🔧 Setting environment variables..."

# Google Cloud Configuration
export PROJECT_ID=dev2-ea8f
export REGION=europe-west2
export GCS_BUCKET=model-build-wine-dev2-ea8f

# Model Configuration
export MODEL_GCS_PATH=gs://model-build-wine-dev2-ea8f/models/rating-prediction-model/model.pkl
export AIP_STORAGE_URI=gs://model-build-wine-dev2-ea8f/models/rating-prediction-model/

# Google Cloud Authentication
export GOOGLE_APPLICATION_CREDENTIALS="${HOME}/.config/gcloud/application_default_credentials.json"

# Server Configuration
export HOST=0.0.0.0
export PORT=8080
export LOG_LEVEL=INFO

echo "✅ Environment variables set:"
echo "   PROJECT_ID: $PROJECT_ID"
echo "   MODEL_GCS_PATH: $MODEL_GCS_PATH"
echo "   AIP_STORAGE_URI: $AIP_STORAGE_URI"
echo "   GOOGLE_APPLICATION_CREDENTIALS: $GOOGLE_APPLICATION_CREDENTIALS"

echo ""
echo "🚀 Now run your server:"
echo "   python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload"