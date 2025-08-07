#!/usr/bin/env python3
"""Fetch model and save to JSON."""

import json
import logging
import os
from google.cloud import aiplatform
from constants import SOURCE_PROJECT_ID, REGION, MODEL_DISPLAY_NAME
from utils import init_vertex_ai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Fetching model: %s", MODEL_DISPLAY_NAME)
    logger.info("From project: %s", SOURCE_PROJECT_ID)
    
    # Initialize Vertex AI
    init_vertex_ai(SOURCE_PROJECT_ID, REGION)
    
    # Find models
    models = aiplatform.Model.list(filter=f'display_name="{MODEL_DISPLAY_NAME}"')
    if not models:
        logger.error("No models found")
        exit(1)
    
    # Get latest model with ready-for-promotion=true
    selected_model = None
    for model in sorted(models, key=lambda m: m.create_time, reverse=True):
        labels = model.labels or {}
        if labels.get("ready-for-promotion", "").lower() == "true":
            selected_model = model
            break
    
    if not selected_model:
        logger.error("No models ready for promotion")
        exit(1)
    
    # Extract model info
    labels = selected_model.labels or {}
    model_data = {
        "model_uri": selected_model.resource_name,
        "display_name": selected_model.display_name,
        "quality_score": float(labels.get("quality-score", "0").replace("-", ".")),
        "eval_status": labels.get("eval-status", "unknown"),
        "ready_for_promotion": labels.get("ready-for-promotion", "false")
    }
    
    # Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)
    
    # Save to JSON
    with open("artifacts/model.json", "w") as f:
        json.dump(model_data, f, indent=2)
    
    logger.info("Model fetched successfully")
    logger.info("URI: %s", model_data['model_uri'])
    logger.info("Quality Score: %s", model_data['quality_score'])
    logger.info("Eval Status: %s", model_data['eval_status'])


if __name__ == "__main__":
    main()