#!/usr/bin/env python3
"""Check promotion gate and save result to JSON."""

import json
import logging
from constants import PROMOTION_THRESHOLD

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Checking promotion gate")
    
    # Load model data
    with open("artifacts/model.json", "r") as f:
        model_data = json.load(f)
    
    quality_score = model_data["quality_score"]
    eval_status = model_data["eval_status"]
    ready = model_data["ready_for_promotion"]
    
    logger.info("Model: %s", model_data['display_name'])
    logger.info("Quality Score: %s (threshold: %s)", quality_score, PROMOTION_THRESHOLD)
    logger.info("Eval Status: %s", eval_status)
    logger.info("Ready: %s", ready)
    
    # Gate logic
    gate_passed = (
        quality_score >= PROMOTION_THRESHOLD and
        eval_status == "completed" and
        ready.lower() == "true"
    )
    
    # Save gate result
    gate_result = {
        "gate_passed": gate_passed,
        "quality_score": quality_score,
        "threshold": PROMOTION_THRESHOLD,
        "eval_status": eval_status,
        "ready_for_promotion": ready,
        "model_uri": model_data["model_uri"],
        "display_name": model_data["display_name"]
    }
    
    with open("artifacts/gate.json", "w") as f:
        json.dump(gate_result, f, indent=2)
    
    if gate_passed:
        logger.info("Gate PASSED")
    else:
        logger.error("Gate FAILED")
        if quality_score < PROMOTION_THRESHOLD:
            logger.error("Quality too low: %s < %s", quality_score, PROMOTION_THRESHOLD)
        if eval_status != "completed":
            logger.error("Eval not completed: %s", eval_status)
        if ready.lower() != "true":
            logger.error("Not ready: %s", ready)
        exit(1)


if __name__ == "__main__":
    main()