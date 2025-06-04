"""
Constants for the wine rating prediction pipeline.
"""

# Project configuration
PROJECT_ID = "dev2-ea8f"
REGION = "europe-west2"

# Storage configuration
GCS_BUCKET = "model-output-wine-dev2-ea8f"
DATA_PATH = f"gs://{GCS_BUCKET}/wine_data.csv"

# Pipeline configuration
PIPELINE_FILE = "wine_rating_pipeline.json"
PIPELINE_JOB_DISPLAY_NAME = "wine-rating-job"
PIPELINE_ROOT_SUFFIX = "pipeline_root/wine_rating"

# Model configuration
MODEL_DISPLAY_NAME = "wine-rating-model"
ENDPOINT_DISPLAY_NAME = "wine-rating-endpoint"
EVALUATION_THRESHOLD = 0.6

# ML Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
SERVING_CONTAINER_IMAGE_URI = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"

# Deployment configuration
MACHINE_TYPE = "n1-standard-2"
MIN_REPLICA_COUNT = 1
MAX_REPLICA_COUNT = 1

BATCH_INPUT_PATH = f"gs://{GCS_BUCKET}/batch.jsonl"
BATCH_OUTPUT_PATH = f"gs://{GCS_BUCKET}/predictions/"
PIPELINE_ROOT = f"gs://{GCS_BUCKET}/pipeline_root"
