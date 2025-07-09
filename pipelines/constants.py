"""
Constants for the Wine Quality Predictor Pipeline.
"""

# General
PROJECT_ID = "dev2-ea8f"
REGION = "europe-west2"
APPLICATION = "wine-quality"
TYPE = "online-prediction"

# Storage Paths
MODEL_BUCKET = "model-build-wine-dev2-ea8f"
DATA_PATH = f"gs://{MODEL_BUCKET}/dataset/wine_data.csv"


# Pipeline Configuration
PIPELINE_FILE = f"{APPLICATION}-{TYPE}.json"
PIPELINE_JOB_DISPLAY_NAME = f"{APPLICATION}-{TYPE}-pipeline"
PIPELINE_JOB_DISPLAY_DESC = "E2E Wine Rating Demo Pipeline"
PIPELINE_ROOT_SUFFIX = f"pipeline/{APPLICATION}-{TYPE}"


# Model Configuration
MODEL_DISPLAY_NAME = f"{APPLICATION}-{TYPE}-model"
MODEL_ENDPOINT_DISPLAY_NAME = f"{APPLICATION}-{TYPE}-endpoint"
EVALUATION_THRESHOLD = 0.6

# Model Parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 50

# Container Configuration
BASE_CONTAINER_IMAGE = "python:3.9"
MODEL_SERVING_IMAGE = (
    "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
)

# Deployment Confi
MACHINE_TYPE = "n1-standard-2"
MIN_REPLICA_COUNT = 1
MAX_REPLICA_COUNT = 1

# Logging
ENABLE_DETAILED_LOGGING = False
