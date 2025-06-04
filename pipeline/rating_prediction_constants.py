# rating_prediction_constants.py
"""
Constants for the rating prediction batch pipeline.
Update these values according to your GCP project and data locations.
"""

PROJECT_ID = "dev2-ea8f" 
REGION = "europe-west2"   
GCS_BUCKET = "model-build-wine-dev2-ea8f"  

# Data Paths
DATA_PATH = f"gs://{GCS_BUCKET}/wine_data.csv"  
BATCH_INPUT_PATH = f"gs://{GCS_BUCKET}/batch.jsonl"  
BATCH_OUTPUT_PATH = f"gs://{GCS_BUCKET}/predictions/"  

# Pipeline Configuration
PIPELINE_ROOT = f"gs://{GCS_BUCKET}/pipeline_root" 
PIPELINE_FILE = "rating_prediction_pipeline.json"  
PIPELINE_JOB_DISPLAY_NAME = "rating-prediction-batch-pipeline"

# Model Configuration
MODEL_DISPLAY_NAME = "rating-prediction-model"
MACHINE_TYPE = "n1-standard-4" 

# Pipeline Parameters
RANDOM_STATE = 42
N_ESTIMATORS = 10 

# Serving Configuration
SERVING_CONTAINER_IMAGE_URI = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
BASE_IMAGE = "python:3.9"

# Logging
ENABLE_DETAILED_LOGGING = True