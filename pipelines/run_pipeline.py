"""Main pipeline runner - runs training or promotion pipeline based on input."""

import argparse
import sys
from pathlib import Path

from pipelines.training.run_pipeline import main as run_training
from pipelines.promotion.run_pipeline import main as run_promotion
from pipelines.shared.log_utils import get_logger

logger = get_logger(__name__)


def run_training_pipeline():
    """Run the training pipeline."""
    logger.info("Starting Wine Quality Training Pipeline")
    logger.info("=" * 60)

    try:
        return run_training()
    except ImportError as e:
        logger.error("Failed to import training pipeline: %s", e)
        return False
    except (RuntimeError, ValueError, TypeError) as e:
        logger.error("Training pipeline execution failed: %s", e)
        return False


def run_promotion_pipeline():
    """Run the promotion pipeline."""
    logger.info("Starting Wine Quality Promotion Pipeline")
    logger.info("=" * 60)

    try:
        return run_promotion()
    except ImportError as e:
        logger.error("Failed to import promotion pipeline: %s", e)
        return False
    except (RuntimeError, ValueError, TypeError) as e:
        logger.error("Promotion pipeline execution failed: %s", e)
        return False


def main():
    """Main entry point for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Wine Quality ML Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --pipeline training     # Run training pipeline
  python run_pipeline.py --pipeline promotion    # Run promotion pipeline
  python run_pipeline.py -p training             # Short form
        """,
    )

    parser.add_argument(
        "--pipeline",
        "-p",
        choices=["training", "promotion"],
        required=True,
        help="Pipeline to run: training or promotion",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel("DEBUG")

    logger.debug("Running '%s' pipeline", args.pipeline)
    logger.debug("Working directory: %s", Path.cwd())

    # Route to appropriate pipeline
    pipeline_success = (
        run_training_pipeline()
        if args.pipeline == "training"
        else run_promotion_pipeline()
    )

    if pipeline_success:
        logger.info("%s pipeline completed successfully", args.pipeline.title())
    else:
        logger.error("%s pipeline failed", args.pipeline.title())

    return pipeline_success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
