"""Constants for model promotion pipeline."""

import os

# Auth Token
AUTH_TOKEN = os.getenv("CLOUDSDK_AUTH_ACCESS_TOKEN", "local")
# General configuration
SOURCE_PROJECT_ID = os.getenv("SOURCE_PROJECT_ID", "dev2-ea8f")
TARGET_PROJECT_ID = os.getenv("TARGET_PROJECT_ID", "dev1-bfa7")
REGION = os.getenv("REGION", "europe-west2")
APPLICATION = os.getenv("APPLICATION", "wine-quality")
TYPE = os.getenv("TYPE", "promotion")
PIPELINE_SA = os.getenv(
    "PIPELINE_SA", "1035860259529-compute@developer.gserviceaccount.com"
)  # pylint: disable=line-too-long
# Pipeline configuration
BUILD_NUMBER = os.getenv("HARNESS_BUILD_ID", "local")
PIPELINE_FILE = f"{APPLICATION}-{TYPE}-{BUILD_NUMBER}.json"
PROMOTION_JOB_DISPLAY_NAME = f"{APPLICATION}-{TYPE}-pipeline"
PROMOTION_JOB_DISPLAY_DESC = "Model promotion pipeline"
MODEL_ENDPOINT_DISPLAY_NAME = f"{APPLICATION}-{TYPE}-endpoint"
# Model Configuration
MODEL_DISPLAY_NAME = "wine-quality-online-prediction-model"
# Base Images
# noqa: E501
# pylint: disable=line-too-long
BASE_CONTAINER_IMAGE = os.getenv("BASE_CONTAINER_IMAGE", "python:3.9")
# noqa: E501
# pylint: disable=line-too-long
MODEL_SERVING_IMAGE = (
    "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
)
# Storage configuration
PIPELINE_BUCKET = os.getenv("PIPELINE_BUCKET", "wine-pipeline-dev1-bfa7")
PIPELINE_ROOT_SUFFIX = f"pipeline/{APPLICATION}-{TYPE}"
# Deployment configuration
MACHINE_TYPE = os.getenv("MACHINE_TYPE", "n1-standard-2")
MIN_REPLICA_COUNT = os.getenv("MIN_REPLICA_COUNT", "1")
MAX_REPLICA_COUNT = os.getenv("MAX_REPLICA_COUNT", "1")

IS_LOCAL = os.getenv("IS_LOCAL", "true").lower() == "true"
PROMOTION_THRESHOLD = os.getenv("PROMOTION_THRESHOLD", "0.9")

MODEL_GCS_URI = os.getenv("MODEL_GCS_URI", f"gs://{PIPELINE_BUCKET}/promoted_models/{BUILD_NUMBER}/")
