#!/usr/bin/env python3
"""Check promotion gate and save result to JSON."""

import json
import logging
from constants import PROMOTION_THRESHOLD

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logging.info("Starting promotion gate")
    
    try:
        # Load model data
        with open("artifacts/model.json", "r") as f:
            model_data = json.load(f)
        
        # Extract metadata like KFP component
        model_name = model_data.get("display_name", "unknown")
        eval_status = model_data.get("eval_status", "")
        quality_score = float(model_data.get("quality_score", "0.0"))
        ready_for_promotion = model_data.get("ready_for_promotion", "false")
        
        # Convert threshold to float for comparison
        promotion_threshold = float(PROMOTION_THRESHOLD)

        logging.info("Processing model: %s", model_name)
        logging.info(
            "Quality score: %.4f, Threshold: %.4f", quality_score, promotion_threshold
        )

        # Check all gate criteria
        gate_passed = True
        failure_reasons = []
        
        if eval_status != "completed":
            logging.warning(
                "Gate failed: eval_status is %s (required: completed)", eval_status
            )
            gate_passed = False
            failure_reasons.append(f"eval_status is {eval_status} (required: completed)")

        if quality_score < promotion_threshold:
            logging.warning(
                "Gate failed: quality_score %.4f < %.4f",
                quality_score,
                promotion_threshold,
            )
            gate_passed = False
            failure_reasons.append(f"quality_score {quality_score:.4f} < {promotion_threshold:.4f}")

        if ready_for_promotion.lower() != "true":
            logging.warning(
                "Gate failed: ready_for_promotion is %s (required: true)",
                ready_for_promotion,
            )
            gate_passed = False
            failure_reasons.append(f"ready_for_promotion is {ready_for_promotion} (required: true)")

        # Save gate result with detailed information
        gate_result = {
            "gate_passed": gate_passed,
            "model_name": model_name,
            "quality_score": quality_score,
            "promotion_threshold": promotion_threshold,
            "eval_status": eval_status,
            "ready_for_promotion": ready_for_promotion,
            "model_uri": model_data.get("model_uri", ""),
            "display_name": model_name,
            "failure_reasons": failure_reasons,
            "harness_build_id": model_data.get("harness_build_id", "unknown")
        }
        
        with open("artifacts/gate.json", "w") as f:
            json.dump(gate_result, f, indent=2)

        if gate_passed:
            logging.info("Promotion gate passed - model approved for promotion")
        else:
            logging.error("Promotion gate failed")
            for reason in failure_reasons:
                logging.error("  - %s", reason)
            exit(1)

    except Exception as e:
        logging.error("Promotion gate failed: %s", e)
        
        # Save error gate result
        error_gate_result = {
            "gate_passed": False,
            "model_name": "unknown",
            "quality_score": 0.0,
            "promotion_threshold": float(PROMOTION_THRESHOLD),
            "eval_status": "error",
            "ready_for_promotion": "false",
            "model_uri": "",
            "display_name": "unknown",
            "failure_reasons": [f"Gate evaluation error: {str(e)}"],
            "error_message": str(e)
        }
        
        with open("artifacts/gate.json", "w") as f:
            json.dump(error_gate_result, f, indent=2)
        
        exit(1)


if __name__ == "__main__":
    main()