"""online package init"""

from .wine_quality_model_promotion import (
    model_promotion_pipeline,
    compile_pipeline,
)

__all__ = ["compile_pipeline", "model_promotion_pipeline"]
