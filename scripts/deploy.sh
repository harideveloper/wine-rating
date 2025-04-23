#!/bin/bash

# Wine Recommendation System Deployment Script
# This script compiles and deploys the pipeline to Vertex AI

set -e  # Exit on any error

# Parse command line arguments
PROJECT_ID=""
BUCKET_NAME=""
REGION="europe-west2"
MODEL_NAME="wine-recommender"

print_usage() {
  echo "Usage: $0 --project-id=<project-id> --bucket=<bucket-name> [--region=<region>] [--model-name=<model-name>]"
}

for i in "$@"; do
  case $i in
    --project-id=*)
      PROJECT_ID="${i#*=}"
      shift
      ;;
    --bucket=*)
      BUCKET_NAME="${i#*=}"
      shift
      ;;
    --region=*)
      REGION="${i#*=}"
      shift
      ;;
    --model-name=*)
      MODEL_NAME="${i#*=}"
      shift
      ;;
    --help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $i"
      print_usage
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$PROJECT_ID" ] || [ -z "$BUCKET_NAME" ]; then
  echo "Error: project-id and bucket are required parameters."
  print_usage
  exit 1
fi

echo "=========================================="
echo "Deploying Wine Recommendation System"
echo "=========================================="
echo "Project ID: $PROJECT_ID"
echo "Bucket: $BUCKET_NAME"
echo "Region: $REGION"
echo "Model Name: $MODEL_NAME"
echo "=========================================="

# Ensure we're in the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR/.."

# Check for config directory and create if it doesn't exist
if [ ! -d "config" ]; then
  mkdir -p config
  echo "Created config directory."
fi

# Update configuration file with provided parameters
cat > config/pipeline_config.yaml << EOF
# Pipeline Configuration

# Google Cloud Platform Settings
gcp:
  project_id: "$PROJECT_ID"
  region: "$REGION"
  gcs_bucket: "$BUCKET_NAME"

# Model Settings
model:
  name: "$MODEL_NAME"
  version: "v1"
  framework: "sklearn"

# Pipeline Settings
pipeline:
  name: "wine-recommendation-pipeline"
  description: "Wine recommendation system pipeline"
  enable_caching: true
  pipeline_root: "gs://$BUCKET_NAME/pipeline_root/wine_recommendation"
EOF

echo "Configuration file updated."

# Ensure gcloud is authenticated and set to the correct project
echo "Setting gcloud project to $PROJECT_ID..."
gcloud config set project "$PROJECT_ID"

# Ensure Vertex AI API is enabled
echo "Enabling required APIs..."
gcloud services enable aiplatform.googleapis.com 

# Compile the pipeline
echo "Compiling pipeline..."
cd pipeline
python -c "from wine_recommendation_pipeline import compile_pipeline; compile_pipeline()"
echo "Pipeline compiled successfully."

# Execute the deployment
echo "Executing pipeline on Vertex AI..."
python -c "from wine_recommendation_pipeline import run_pipeline; run_pipeline(project_id='$PROJECT_ID', gcs_bucket='$BUCKET_NAME', model_name='$MODEL_NAME', region='$REGION')"

echo "=========================================="
echo "Deployment initiated!"
echo "Check your GCP Console to monitor the pipeline execution."
echo "=========================================="