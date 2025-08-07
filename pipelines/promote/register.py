#!/usr/bin/env python3
"""Register/promote model to target project."""

import json
import logging
import datetime
from google.cloud import aiplatform
from constants import (
    SOURCE_PROJECT_ID, 
    TARGET_PROJECT_ID, 
    REGION, 
    MODEL_SERVING_IMAGE,
    BUILD_NUMBER
)
from utils import init_vertex_ai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logging.info("Starting model promotion")
    
    try:
        # Load gate result (contains model info)
        with open("artifacts/gate.json", "r") as f:
            gate_data = json.load(f)

        if not gate_data["gate_passed"]:
            logging.error("Gate failed - skipping registration")
            exit(1)

        # Extract model information
        model_uri = gate_data["model_uri"]
        display_name = gate_data["display_name"]
        original_metadata = gate_data  # Keep original metadata

        logging.info("Promoting model: %s", model_uri)
        logging.info(
            "Source project: %s, Target project: %s", SOURCE_PROJECT_ID, TARGET_PROJECT_ID
        )

        # Get source model
        init_vertex_ai(SOURCE_PROJECT_ID, REGION)
        source_model = aiplatform.Model(model_uri)

        logging.info("Retrieved source model: %s", source_model.display_name)

        # Prepare labels for target model (prod workload)
        production_labels = {
            "promoted-from": SOURCE_PROJECT_ID,
            "promotion-date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
            "harness-build-id": BUILD_NUMBER,
        }

        # Copy existing labels from source model (prod test)
        if source_model.labels:
            production_labels.update(source_model.labels)
            
        # Update environment label from prd-test to prd-workload
        production_labels["env"] = "prd-workload"

        # Switch to target project
        init_vertex_ai(TARGET_PROJECT_ID, REGION)

        # Check if model already exists in target project
        existing_models = []
        try:
            existing_models = aiplatform.Model.list(
                filter=f'display_name="{source_model.display_name}"'
            )
            logging.info(
                "Found %d existing models with same display name", len(existing_models)
            )
        except Exception as e:
            logging.warning(
                "Failed to list existing models: %s. Treating as new model creation.", e
            )
            existing_models = []

        # Get container image
        container_image = MODEL_SERVING_IMAGE
        if hasattr(source_model, 'container_spec') and source_model.container_spec:
            container_image = source_model.container_spec.image_uri

        if existing_models:
            # Model exists - create new version
            logging.info("Model exists in target project, creating new version")
            target_model = aiplatform.Model.upload(
                display_name=source_model.display_name,
                artifact_uri=source_model.uri,
                serving_container_image_uri=container_image,
                description=f"Promoted from {SOURCE_PROJECT_ID}",
                labels=production_labels,
                parent_model=existing_models[0].resource_name,
                is_default_version=True,
                sync=True,
            )
            logging.info("Created new model version: %s", target_model.resource_name)
        else:
            # Model doesn't exist - create new model
            logging.info("Model doesn't exist in target project, creating new model")
            target_model = aiplatform.Model.upload(
                display_name=source_model.display_name,
                artifact_uri=source_model.uri,
                serving_container_image_uri=container_image,
                description=f"Promoted from {SOURCE_PROJECT_ID}",
                labels=production_labels,
                is_default_version=True,
                sync=True,
            )
            logging.info("Created new model: %s", target_model.resource_name)

        logging.info("Model promoted successfully to: %s", target_model.resource_name)

        # Preserve original metadata and add promotion details
        registration_result = {
            "source_model_uri": model_uri,
            "target_model_uri": target_model.resource_name,
            "display_name": target_model.display_name,
            "source_project": SOURCE_PROJECT_ID,
            "target_project": TARGET_PROJECT_ID,
            "promotion_date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
            "promotion_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_version": (
                target_model.version_id
                if hasattr(target_model, "version_id")
                else "1"
            ),
            "is_new_model": len(existing_models) == 0,
            "promotion_status": "completed",
            "harness_build_id": BUILD_NUMBER,
            # Preserve original quality metrics
            "quality_score": original_metadata.get("quality_score", 0.0),
            "eval_status": original_metadata.get("eval_status", "unknown"),
            "container_image": container_image,
        }

        with open("artifacts/registration.json", "w") as f:
            json.dump(registration_result, f, indent=2)

        logging.info("Model promotion completed successfully")

    except Exception as e:
        logging.error("Model promotion failed: %s", e)

        # Set error metadata
        error_registration_result = {
            "source_model_uri": gate_data.get("model_uri", "") if 'gate_data' in locals() else "",
            "target_model_uri": "",
            "display_name": gate_data.get("display_name", "unknown") if 'gate_data' in locals() else "unknown",
            "source_project": SOURCE_PROJECT_ID,
            "target_project": TARGET_PROJECT_ID,
            "promotion_status": "failed",
            "error_message": str(e),
            "promotion_date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
            "harness_build_id": BUILD_NUMBER,
        }

        with open("artifacts/registration.json", "w") as f:
            json.dump(error_registration_result, f, indent=2)

        raise


if __name__ == "__main__":
    main()