# wine_quality_constants.py
"""
Constants for the Wine Quality Batch Predictor Pipeline.
Update these values according to your GCP project and data locations.
"""

PROJECT_ID = "dev2-ea8f" 
REGION = "europe-west2"   
GCS_BUCKET = "model-build-wine-dev2-ea8f"  

# Data Paths
DATA_PATH = f"gs://{GCS_BUCKET}/dataset/wine_data.csv"  
BATCH_INPUT_PATH = f"gs://{GCS_BUCKET}/batch/input/batch_v2.jsonl"  
BATCH_OUTPUT_PATH = f"gs://{GCS_BUCKET}/batch/output"

# Pipeline Configuration
PIPELINE_ROOT = f"gs://{GCS_BUCKET}/pipeline_root" 
PIPELINE_FILE = "wine_quality_batch_predictor.json"
PIPELINE_JOB_DISPLAY_NAME = "Wine Quality Batch Predictor"

# Model Configuration
MODEL_DISPLAY_NAME = "wine-quality-predictor-model"

# Pipeline Parameters
RANDOM_STATE = 42
N_ESTIMATORS = 50

# Container Configuration
BASE_IMAGE = "python:3.9"

# Logging
ENABLE_DETAILED_LOGGING = True