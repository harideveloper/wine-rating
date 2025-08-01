"""
Constants for the wine quality online prediction pipeline.
"""
import os
# Auth Token
AUTH_TOKEN = os.getenv("CLOUDSDK_AUTH_ACCESS_TOKEN", "local")
# if not AUTH_TOKEN:
#     raise ValueError("CLOUDSDK_AUTH_ACCESS_TOKEN environment variable is not set")
# General configuration
PROJECT_ID = os.getenv("PROJECT_ID","dev2-ea8f")
REGION = "europe-west2"
APPLICATION = os.getenv("APPLICATION", "wine-quality")
TYPE = os.getenv("TYPE", "online-prediction")
PIPELINE_SA = os.getenv("PIPELINE_SA","212373574046-compute@developer.gserviceaccount.com")
# Storage configuration
DATA_BUCKET = os.getenv("DATA_BUCKET","model-build-wine-dev2-ea8f")
DATA_PATH = os.getenv("DATA_PATH", f"gs://{DATA_BUCKET}/dataset/wine_data.csv")
PIPELINE_BUCKET = os.getenv("PIPELINE_BUCKET","model-build-wine-dev2-ea8f")
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

BASE_CONTAINER_IMAGE = os.getenv("BASE_CONTAINER_IMAGE","python:3.9")
MODEL_SERVING_IMAGE = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
# Deployment configuration
MACHINE_TYPE = os.getenv("MACHINE_TYPE", "n1-standard-2")
MIN_REPLICA_COUNT = os.getenv("MIN_REPLICA_COUNT", "1")
MAX_REPLICA_COUNT = os.getenv("MAX_REPLICA_COUNT", "1")
# Logging
ENABLE_DETAILED_LOGGING = os.getenv("ENABLE_DETAILED_LOGGING", "True")
IS_LOCAL= os.getenv("IS_LOCAL", "True")