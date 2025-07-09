"""online package init"""

from .wine_quality_online_predictor import (
    wine_quality_online_predictor_pipeline,
    compile_pipeline,
)

__all__ = ["compile_pipeline", "wine_quality_online_predictor_pipeline"]
