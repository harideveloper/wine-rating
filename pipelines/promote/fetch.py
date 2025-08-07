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


def parse_metric(value_str: str) -> float:
    """Convert dashed format to float: '0-9992' -> 0.9992"""
    try:
        return (
            float(value_str.replace("-", "."))
            if "-" in value_str
            else float(value_str)
        )
    except (ValueError, TypeError):
        return 0.0


def main():
    logging.info(
        "Fetching model: %s from project: %s, location: %s",
        MODEL_DISPLAY_NAME,
        SOURCE_PROJECT_ID,
        REGION,
    )

    try:
        # Initialize Vertex AI
        init_vertex_ai(SOURCE_PROJECT_ID, REGION)
        
        models = aiplatform.Model.list(filter=f'display_name="{MODEL_DISPLAY_NAME}"')

        if not models:
            raise ValueError(f"No models found with display name: {MODEL_DISPLAY_NAME}")

        # Get latest model with ready-for-promotion = true
        sorted_models = sorted(models, key=lambda m: m.create_time, reverse=True)

        selected_model = None
        for model in sorted_models:
            labels = model.labels or {}
            if labels.get("ready-for-promotion", "").lower() == "true":
                selected_model = model
                break

        if not selected_model:
            raise ValueError(
                f"No models with 'ready-for-promotion=true' found among {len(sorted_models)} models"
            )

        # Extract labels and convert dashed format to float
        labels = selected_model.labels or {}

        # Essential metadata only
        model_data = {
            "model_uri": selected_model.resource_name,
            "display_name": selected_model.display_name,
            "resource_name": selected_model.resource_name,
            "harness_build_id": labels.get("harness-build-id", "unknown"),
            "eval_status": labels.get("eval-status", "unknown"),
            "ready_for_promotion": labels.get("ready-for-promotion", "false"),
            "quality_score": parse_metric(labels.get("quality-score", "0")),
            "mae": parse_metric(labels.get("mae", "0")),
            "mse": parse_metric(labels.get("mse", "0")),
            "r2_score": parse_metric(labels.get("r2-score", "0")),
            "rmse": parse_metric(labels.get("rmse", "0")),
        }
        
        # Create artifacts directory
        os.makedirs("artifacts", exist_ok=True)
        
        # Save to JSON
        with open("artifacts/model.json", "w") as f:
            json.dump(model_data, f, indent=2)

        logging.info("Model fetched successfully: %s", selected_model.display_name)
        logging.info(
            "Key metrics - Quality: %s, Eval status: %s, Ready: %s",
            model_data["quality_score"],
            model_data["eval_status"],
            model_data["ready_for_promotion"],
        )
        
    except Exception as e:
        logging.error("Model fetch failed: %s", e)

        # Create artifacts directory
        os.makedirs("artifacts", exist_ok=True)
        
        # Minimal error metadata
        error_model_data = {
            "model_uri": "",
            "display_name": MODEL_DISPLAY_NAME,
            "resource_name": "",
            "harness_build_id": "unknown",
            "eval_status": "failed",
            "ready_for_promotion": "false",
            "quality_score": 0.0,
            "mae": 0.0,
            "mse": 0.0,
            "r2_score": 0.0,
            "rmse": 0.0,
            "error_message": str(e),
        }
        
        # Save error info
        with open("artifacts/model.json", "w") as f:
            json.dump(error_model_data, f, indent=2)
        
        raise


if __name__ == "__main__":
    main()