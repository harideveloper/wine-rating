"""shared package init"""

__version__ = "1.0.0"
__author__ = "golden path data capabilities"
__description__ = "Shared packages for sample ml pipelines"
__url__ = "https://github.com/company/wine-pipelines"

from .log_utils import get_logger
from .pipeline_base_utils import (
    PipelineConfig,
    run_pipeline_with_error_handling,
)

__all__ = [
    "get_logger",
    "PipelineConfig",
    "run_pipeline_with_error_handling",
]
