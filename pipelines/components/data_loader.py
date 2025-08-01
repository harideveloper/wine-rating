"""Data loader component for wine quality pipeline."""

from kfp.v2.dsl import component, Output, Dataset
from pipelines.components.constants import BASE_CONTAINER_IMAGE


# pylint: disable=import-outside-toplevel,broad-exception-caught
@component(
    packages_to_install=["pandas", "google-cloud-storage"],
    base_image=BASE_CONTAINER_IMAGE,
)
def load_data(data_path: str, output_data: Output[Dataset]) -> None:
    """
    Load wine quality data from Google Cloud Storage.

    """
    import pandas as pd
    from google.cloud import storage
    import io
    import logging

    try:
        if not data_path or not data_path.startswith("gs://"):
            raise ValueError("data_path must be a valid GCS URL starting with 'gs://'")
        logging.info("Loading data from: %s", data_path)

        gcs_path_parts = data_path.replace("gs://", "").split("/")
        bucket_name = gcs_path_parts[0]
        blob_name = "/".join(gcs_path_parts[1:])

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_text()
        df = pd.read_csv(io.StringIO(content))

        if df.empty:
            raise ValueError("Loaded dataset is empty")
        logging.info("Data loaded: %s rows, %s columns", df.shape[0], df.shape[1])

        if "id" not in df.columns:
            df["id"] = range(1, len(df) + 1)

        df.to_csv(output_data.path, index=False)
        logging.info("Data loading completed successfully")
    except Exception as e:
        logging.error("Data loading failed: %s", e)
        raise
