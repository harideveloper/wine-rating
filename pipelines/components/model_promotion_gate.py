"""Promotion gate component - validates labels before promotion."""

from kfp.v2.dsl import Model, Input, component
from pipelines.components.constants import BASE_CONTAINER_IMAGE


# pylint: disable=import-outside-toplevel, too-many-arguments, broad-exception-caught, too-many-positional-arguments, too-many-locals
@component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image=BASE_CONTAINER_IMAGE,
)
def promotion_gate(
    fetched_model: Input[Model],
    promotion_threshold: float = 0.99,
) -> bool:
    """
    Promotion gate component - gates model promotion based on criteria.

    Args:
        fetched_model: Input KFP Model with extracted metadata from fetch_model
        promotion_threshold: Minimum quality score for promotion (default: 0.99)

    Returns:
        bool: True if promotion gate passes
    """
    import logging

    logging.info("Starting promotion gate")
    try:
        metadata = fetched_model.metadata
        model_name = metadata.get("display_name", "unknown")
        eval_status = metadata.get("eval_status", "")
        quality_score = float(metadata.get("quality_score", "0.0"))
        ready_for_promotion = metadata.get("ready_for_promotion", "false")

        logging.info("Processing model: %s", model_name)
        logging.info(
            "Quality score: %.4f, Threshold: %.4f", quality_score, promotion_threshold
        )

        # Check all gate criteria
        if eval_status != "completed":
            logging.warning(
                "Gate failed: eval_status is %s (required: completed)", eval_status
            )
            return False

        if quality_score < promotion_threshold:
            logging.warning(
                "Gate failed: quality_score %.4f < %.4f",
                quality_score,
                promotion_threshold,
            )
            return False

        if ready_for_promotion.lower() != "true":
            logging.warning(
                "Gate failed: ready_for_promotion is %s (required: true)",
                ready_for_promotion,
            )
            return False

        # All checks passed
        logging.info("Promotion gate passed - model approved for promotion")
        return True

    except Exception as e:
        logging.error("Promotion gate failed: %s", e)
        return False
