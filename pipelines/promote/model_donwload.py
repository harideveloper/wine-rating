#!/usr/bin/env python3
"""Download an artifact from GCS."""

import logging
import sys
from gcs_utils import get_storage_client
import constants

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting GCS download")

    try:
        bucket_name = constants.GCS_SOURCE_BUCKET
        blob_name = constants.GCS_SOURCE_BLOB
        local_path = constants.LOCAL_DOWNLOAD_PATH

        storage_client = get_storage_client(constants.SOURCE_PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.download_to_filename(local_path)
        logging.info(
            "Downloaded blob '%s' from bucket '%s' to local file '%s'",
            blob_name,
            bucket_name,
            local_path
        )

    except Exception as e:
        logging.error("Failed to download from GCS: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
