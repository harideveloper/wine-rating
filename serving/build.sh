#!/bin/bash
# build_generic_serving.sh - Build generic serving container

# Configuration
PROJECT_ID="dev2-ea8f"  # Your GCP project ID
REGION="europe-west2"   # Your region
IMAGE_NAME="generic-ml-serving"
IMAGE_TAG="latest"
REPOSITORY="vertex-pipelines"

# Full image URI
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "🐳 Building generic ML serving container..."
echo "📍 Project: ${PROJECT_ID}"
echo "🏷️ Image URI: ${IMAGE_URI}"

# Enable APIs and create repository
echo "🔧 Setting up infrastructure..."
gcloud services enable artifactregistry.googleapis.com --project=${PROJECT_ID}
gcloud artifacts repositories create ${REPOSITORY} \
    --repository-format=docker \
    --location=${REGION} \
    --project=${PROJECT_ID} || echo "Repository already exists"

# Configure Docker
echo "🔐 Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build image
echo "🔨 Building Docker image..."
docker build -t ${IMAGE_URI} -f Dockerfile .

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

# Push image
echo "📤 Pushing to registry..."
docker push ${IMAGE_URI}

if [ $? -ne 0 ]; then
    echo "❌ Push failed!"
    exit 1
fi

echo "✅ Generic serving container ready!"
echo ""
echo "🎯 Container URI: ${IMAGE_URI}"
echo ""
echo "🔧 Usage in Vertex AI Model registration:"
echo "model = aiplatform.Model.upload("
echo "    display_name='my-model',"
echo "    artifact_uri='gs://bucket/models/my-model/',"
echo "    serving_container_image_uri='${IMAGE_URI}',"
echo "    serving_container_predict_route='/predict',"
echo "    serving_container_health_route='/health'"
echo ")"
echo ""
echo "📋 Supported model formats:"
echo "  - Scikit-learn (.pkl, .joblib)"
echo "  - XGBoost (.pkl, .joblib)"  
echo "  - LightGBM (.pkl, .joblib)"
echo "  - Any pickled Python model"