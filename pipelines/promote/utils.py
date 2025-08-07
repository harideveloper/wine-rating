"""Utility functions for model promotion pipeline."""

import logging
from google.cloud import aiplatform
from google.oauth2.credentials import Credentials
from constants import IS_LOCAL, AUTH_TOKEN, PIPELINE_SA

logger = logging.getLogger(__name__)


def init_vertex_ai(project_id, region):
    """
    Initialize Vertex AI with appropriate credentials based on environment.
    
    Args:
        project_id: GCP project ID
        region: GCP region
    """
    if IS_LOCAL:
        logger.info("Initializing AI Platform for local environment")
        aiplatform.init(project=project_id, location=region)
    else:
        logger.info("Initializing AI Platform for CI/production environment")
        if not AUTH_TOKEN:
            raise ValueError("AUTH_TOKEN is required for non-local environment")
        if not PIPELINE_SA:
            raise ValueError("Service account is required for non-local environment")

        credentials = Credentials(AUTH_TOKEN)
        aiplatform.init(
            project=project_id,
            location=region,
            credentials=credentials,
            service_account=PIPELINE_SA,
        )
    
    logger.info("AI Platform initialized for project: %s, region: %s", project_id, region)