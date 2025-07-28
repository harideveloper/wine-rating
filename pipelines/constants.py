"""
Constants for the wine quality online prediction pipeline.
"""

import os

# Auth Token
AUTH_TOKEN = os.getenv("CLOUDSDK_AUTH_ACCESS_TOKEN", "local")
# if not AUTH_TOKEN:
#     raise ValueError("CLOUDSDK_AUTH_ACCESS_TOKEN environment variable is not set")
# General configuration
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = "europe-west2"
APPLICATION = os.getenv("APPLICATION", "wine-quality")
TYPE = os.getenv("TYPE", "online-prediction")
PIPELINE_SA = os.getenv("PIPELINE_SA")
# Storage configuration
DATA_BUCKET = os.getenv("DATA_BUCKET")
DATA_PATH = os.getenv("DATA_PATH", f"gs://{DATA_BUCKET}/dataset/wine_data.csv")
PIPELINE_BUCKET = os.getenv("PIPELINE_BUCKET")
# Pipeline configuration
BUILD_NUMBER = os.getenv("HARNESS_BUILD_ID", "local")
PIPELINE_FILE = f"{APPLICATION}-{TYPE}-{BUILD_NUMBER}.json"
PIPELINE_JOB_DISPLAY_NAME = f"{APPLICATION}-{TYPE}-pipeline"
PIPELINE_JOB_DISPLAY_DESC = "E2E model training & deployment demo pipeline"
PIPELINE_ROOT_SUFFIX = f"pipeline/{APPLICATION}-{TYPE}"
# Model configuration
MODEL_DISPLAY_NAME = f"{APPLICATION}-{TYPE}-model"
MODEL_ENDPOINT_DISPLAY_NAME = f"{APPLICATION}-{TYPE}-endpoint"
EVALUATION_THRESHOLD = os.getenv("EVALUATION_THRESHOLD", "0.6")
# Model parameters
TEST_SIZE = os.getenv("TEST_SIZE", "0.2")
RANDOM_STATE = os.getenv("RANDOM_STATE", "42")
N_ESTIMATORS = os.getenv("N_ESTIMATORS", "50")
# Base Images
# noqa: E501
# pylint: disable=line-too-long
BASE_CONTAINER_IMAGE = os.getenv(
    "BASE_CONTAINER_IMAGE",
    "europe-docker.pkg.dev/dv-sts-psv-tool-svc-bak-08870/staging/hxkxmfkerhkxktic/chhelffddhzzrvzs/null/images/vertex-ai:7e6bdf0",
)
# noqa: E501
# pylint: disable=line-too-long
MODEL_SERVING_IMAGE = os.getenv(
    "MODEL_SERVING_IMAGE",
    "europe-docker.pkg.dev/dv-sts-psv-tool-svc-bak-08870/release/hxkxmfkerhkxktic/chhelffddhzzrvzs/null/images/vai-predictions-image:v1.0.0-VAI_PREDS_DEV",
)
# Deployment configuration
MACHINE_TYPE = os.getenv("MACHINE_TYPE", "n1-standard-2")
MIN_REPLICA_COUNT = os.getenv("MIN_REPLICA_COUNT", "1")
MAX_REPLICA_COUNT = os.getenv("MAX_REPLICA_COUNT", "1")
# Logging
ENABLE_DETAILED_LOGGING = os.getenv("ENABLE_DETAILED_LOGGING", "False")
