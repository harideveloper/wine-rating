#!/usr/bin/env python3
"""Utility functions for interacting with GCS."""

import logging
from google.cloud import storage
from google.oauth2.credentials import Credentials
import constants

def get_storage_client(project_id):
    """Create a GCS client using AUTH_TOKEN in CI, or ADC locally."""
    if constants.IS_LOCAL:
        logging.info("Running in local mode - using Application Default Credentials")
        return storage.Client(project=project_id)
    else:
        if not constants.AUTH_TOKEN:
            raise ValueError("AUTH_TOKEN environment variable is not set in CI mode")
        creds = Credentials(token=constants.AUTH_TOKEN)
        return storage.Client(project=project_id, credentials=creds)
