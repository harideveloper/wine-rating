"""online package init"""

__version__ = "1.0.0"
__author__ = "golden path data capabilities"
__description__ = "training pipeline for demo model"
__url__ = "https://github.com/company/wine-pipelines"

from .wine_quality_online_predictor import (
    wine_quality_online_predictor_pipeline,
    compile_pipeline,
)


__all__ = ["compile_pipeline", "wine_quality_online_predictor_pipeline"]
