"""Model saver component for wine quality pipeline."""

from kfp.v2.dsl import Model, Input, Output, component
from constants import BASE_CONTAINER_IMAGE


@component(
    packages_to_install=["google-cloud-storage"], base_image=BASE_CONTAINER_IMAGE
)
def save_model(model_artifact: Input[Model], uploaded_model_artifact: Output[Model]):
    """Uploads the wine rating model artifact."""
    # pylint: disable=import-outside-toplevel
    import os
    import logging

    try:
        logging.info("Starting model save process")
        source_path = model_artifact.path + ".joblib"
        model_dir = os.path.dirname(uploaded_model_artifact.path)
        model_file_path = os.path.join(model_dir, "model.joblib")
        logging.info("Preparing model directory")
        os.makedirs(model_dir, exist_ok=True)
        logging.info("Copying model file")
        with open(source_path, "rb") as source_file:
            model_data = source_file.read()
        with open(model_file_path, "wb") as target_file:
            target_file.write(model_data)
        logging.info("Model file copied successfully")
        uploaded_model_artifact.uri = model_dir
        metadata_count = 0
        for key, value in model_artifact.metadata.items():
            try:
                uploaded_model_artifact.metadata[key] = value
                metadata_count += 1
            except (TypeError, ValueError) as metadata_error:
                logging.warning(
                    "Failed to copy metadata for key '%s': %s", key, metadata_error
                )
        logging.info("Copied %s metadata items", metadata_count)
        logging.info("Model save completed successfully")
    except Exception as e:
        logging.error("Model save failed: %s", e)
        raise
