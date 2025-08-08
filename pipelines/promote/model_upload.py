#!/usr/bin/env python3
"""Upload an artifact to GCS."""

import logging
import sys
from gcs_utils import get_storage_client
import constants


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting GCS upload")

    try:
        bucket_name = constants.GCS_DEST_BUCKET
        blob_name = constants.GCS_DEST_BLOB
        local_path = constants.LOCAL_UPLOAD_PATH

        storage_client = get_storage_client(constants.TARGET_PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.upload_from_filename(local_path)
        logging.info(
            "Uploaded local file '%s' to bucket '%s' as blob '%s'",
            local_path,
            bucket_name,
            blob_name
        )

    except Exception as e:
        logging.error("Failed to upload to GCS: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
