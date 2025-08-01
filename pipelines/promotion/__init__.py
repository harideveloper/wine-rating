"""promotion pipeline package"""

__version__ = "1.0.0"
__author__ = "golden path data capabilities"
__description__ = "promotion pipeline for demo model"
__url__ = "https://github.com/company/wine-pipelines"

from .wine_quality_model_promotion import (
    model_promotion_pipeline,
    compile_pipeline,
)

__all__ = ["compile_pipeline", "model_promotion_pipeline"]
