# wine_batch_constants.py
"""
Constants for the wine batch prediction pipeline.
Update these values according to your GCP project and data locations.
"""

# GCP Project Configuration
PROJECT_ID = "dev2-ea8f"  # Your GCP project ID
REGION = "europe-west2"   # Your preferred region
GCS_BUCKET = "model-output-wine-dev2-ea8f"  # Your GCS bucket name

# Data Paths
DATA_PATH = f"gs://{GCS_BUCKET}/wine_data.csv"  # Training data
BATCH_INPUT_PATH = f"gs://{GCS_BUCKET}/batch.jsonl"  # Batch input data
BATCH_OUTPUT_PATH = f"gs://{GCS_BUCKET}/predictions/"  # Prediction results

# Pipeline Configuration
PIPELINE_ROOT = f"gs://{GCS_BUCKET}/pipeline_root"  # Pipeline artifacts storage
PIPELINE_FILE = "wine_batch_pipeline.json"  # Compiled pipeline file
PIPELINE_JOB_DISPLAY_NAME = "wine-batch-prediction-pipeline"

# Model Configuration
MODEL_DISPLAY_NAME = "wine-rating-batch-model"
MACHINE_TYPE = "n1-standard-4"  # Machine type for batch prediction

# Pipeline Parameters
RANDOM_STATE = 42
N_ESTIMATORS = 10  # Reduced for faster training

# Serving Configuration
SERVING_CONTAINER_IMAGE_URI = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"

# File Formats
SUPPORTED_BATCH_FORMATS = ["jsonl", "csv"]
EXPECTED_FEATURES = ["price_numeric", "Country", "Type", "Grape", "Style"]

# Logging
ENABLE_DETAILED_LOGGING = True