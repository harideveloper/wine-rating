#!/usr/bin/env python3
"""Register/promote model to target project."""

import json
import logging
from google.cloud import aiplatform
from constants import (
    SOURCE_PROJECT_ID, 
    TARGET_PROJECT_ID, 
    REGION, 
    MODEL_SERVING_IMAGE
)
from utils import init_vertex_ai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Registering model")
    
    # Load gate result (contains model info)
    with open("artifacts/gate.json", "r") as f:
        gate_data = json.load(f)
    
    if not gate_data["gate_passed"]:
        logger.error("Gate failed - skipping registration")
        exit(1)
    
    model_uri = gate_data["model_uri"]
    display_name = gate_data["display_name"]
    
    logger.info("Model: %s", display_name)
    logger.info("From: %s", SOURCE_PROJECT_ID)
    logger.info("To: %s", TARGET_PROJECT_ID)
    
    # Get source model
    logger.info("Accessing source model")
    init_vertex_ai(SOURCE_PROJECT_ID, REGION)
    source_model = aiplatform.Model(model_uri)
    
    # Switch to target project and register model
    logger.info("Switching to target project for registration")
    init_vertex_ai(TARGET_PROJECT_ID, REGION)
    
    target_model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=source_model.uri,
        serving_container_image_uri=MODEL_SERVING_IMAGE,
        sync=True
    )
    
    # Save registration result
    registration_result = {
        "source_model_uri": model_uri,
        "target_model_uri": target_model.resource_name,
        "display_name": target_model.display_name,
        "source_project": SOURCE_PROJECT_ID,
        "target_project": TARGET_PROJECT_ID
    }
    
    with open("artifacts/registration.json", "w") as f:
        json.dump(registration_result, f, indent=2)
    
    logger.info("Model registered successfully")
    logger.info("Target URI: %s", target_model.resource_name)


if __name__ == "__main__":
    main()