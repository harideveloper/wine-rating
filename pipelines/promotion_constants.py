"""Constants for model promotion pipeline."""

import os

# Auth Token
AUTH_TOKEN = os.getenv("CLOUDSDK_AUTH_ACCESS_TOKEN", "local")

# General configuration
SOURCE_PROJECT_ID = os.getenv("SOURCE_PROJECT_ID", "dev2-ea8f")
TARGET_PROJECT_ID = os.getenv("TARGET_PROJECT_ID", "dev1-bfa7")
LOCATION = os.getenv("LOCATION", "europe-west2")
APPLICATION = os.getenv("APPLICATION", "wine-quality")
TYPE = os.getenv("TYPE", "promotion")
PIPELINE_SA = os.getenv("PIPELINE_SA","1035860259529-compute@developer.gserviceaccount.com")

# Pipeline configuration
BUILD_NUMBER = os.getenv("HARNESS_BUILD_ID", "local")
PIPELINE_FILE = f"{APPLICATION}-{TYPE}-{BUILD_NUMBER}.json"
PROMOTION_PIPELINE_NAME = "model-promotion-pipeline"
PROMOTION_JOB_DISPLAY_NAME = f"{APPLICATION}-{TYPE}-pipeline"
PROMOTION_JOB_DISPLAY_DESC = "Model promotion pipeline between registries"

# Model selection configuration
MODEL_DISPLAY_NAME_FILTER = os.getenv("MODEL_DISPLAY_NAME_FILTER", "")
MODEL_ID = os.getenv("MODEL_ID", "")

# Promotion settings
ADD_PROD_SUFFIX = os.getenv("ADD_PROD_SUFFIX", "True").lower() == "true"
OVERRIDE_EXISTING = os.getenv("OVERRIDE_EXISTING", "False").lower() == "true"

# Validation thresholds
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.0"))
REQUIRED_LABELS = os.getenv("REQUIRED_LABELS", "")

# Base Images
BASE_CONTAINER_IMAGE = os.getenv("BASE_CONTAINER_IMAGE","python:3.9")


# Notification settings
NOTIFICATION_CHANNEL = os.getenv("NOTIFICATION_CHANNEL", "console")

# Storage configuration
PIPELINE_BUCKET = os.getenv("PIPELINE_BUCKET","wine-pipeline-dev1-bfa7")
PIPELINE_ROOT_SUFFIX = f"pipeline/{APPLICATION}-{TYPE}"