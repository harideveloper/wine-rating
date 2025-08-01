"""Model promoter component with robust error handling."""

from kfp.v2.dsl import Model, Input, Output, component
from pipelines.components.constants import BASE_CONTAINER_IMAGE


# pylint: disable=import-outside-toplevel, too-many-arguments, broad-exception-caught, too-many-positional-arguments, too-many-locals
@component(
    packages_to_install=["google-cloud-aiplatform"], base_image=BASE_CONTAINER_IMAGE
)
def promote_model(
    fetched_model: Input[Model],
    promoted_model: Output[Model],
    source_project: str,
    target_project: str,
    location: str,
):
    """Promotes model from source to target project with robust error handling."""
    from google.cloud import aiplatform
    import logging
    import datetime

    try:
        logging.info("Starting model promotion")

        # fetch model from prod test by model resource name
        model_resource_name = fetched_model.uri
        original_metadata = fetched_model.metadata.copy()

        logging.info("Promoting model: %s", model_resource_name)
        logging.info(
            "Source project: %s, Target project: %s", source_project, target_project
        )

        aiplatform.init(project=source_project, location=location)
        source_model = aiplatform.Model(model_resource_name)

        logging.info("Retrieved source model: %s", source_model.display_name)

        # Prepare labels for target model (prod workload)
        production_labels = {
            "environment": "prd-workload",
            "promoted-from": source_project,
            "promotion-date": datetime.datetime.utcnow().date().isoformat(),
        }

        # Copy existing labels from source model (prod test)
        if source_model.labels:
            production_labels.update(source_model.labels)

        aiplatform.init(project=target_project, location=location)

        # Check if model already exists in target project
        existing_models = []
        try:
            existing_models = aiplatform.Model.list(
                filter=f'display_name="{source_model.display_name}"'
            )
            logging.info(
                "Found %d existing models with same display name", len(existing_models)
            )
        except Exception as e:
            logging.warning(
                "Failed to list existing models: %s. Treating as new model creation.", e
            )
            existing_models = []

        if existing_models:
            # Model exists - create new version
            logging.info("Model exists in target project, creating new version")
            target_model = aiplatform.Model.upload(
                display_name=source_model.display_name,
                artifact_uri=source_model.uri,
                serving_container_image_uri=source_model.container_spec.image_uri,
                description=f"Promoted from {source_project}",
                labels=production_labels,
                parent_model=existing_models[0].resource_name,
                is_default_version=True,
                sync=True,
            )
            logging.info("Created new model version: %s", target_model.resource_name)
        else:
            # Model doesn't exist - create new model
            logging.info("Model doesn't exist in target project, creating new model")
            target_model = aiplatform.Model.upload(
                display_name=source_model.display_name,
                artifact_uri=source_model.uri,
                serving_container_image_uri=source_model.container_spec.image_uri,
                description=f"Promoted from {source_project}",
                labels=production_labels,
                is_default_version=True,
                sync=True,
            )
            logging.info("Created new model: %s", target_model.resource_name)

        logging.info("Model promoted successfully to: %s", target_model.resource_name)

        # Preserve orginal metadata and add promotion details to metadata
        promoted_model.uri = target_model.resource_name
        promoted_model.metadata = original_metadata.copy()
        promoted_model.metadata.update(
            {
                "display_name": target_model.display_name,
                "resource_name": target_model.resource_name,
                "source_project": source_project,
                "target_project": target_project,
                "promotion_date": datetime.datetime.utcnow().date().isoformat(),
                "model_version": (
                    target_model.version_id
                    if hasattr(target_model, "version_id")
                    else "1"
                ),
                "is_new_model": len(existing_models) == 0,
            }
        )

        logging.info("Model promotion completed successfully")
    except Exception as e:
        logging.error("Model promotion failed: %s", e)

        # Set error metadata
        promoted_model.uri = fetched_model.uri
        promoted_model.metadata = fetched_model.metadata.copy()
        promoted_model.metadata.update(
            {
                "promotion_status": "failed",
                "error_message": str(e),
                "source_project": source_project,
                "target_project": target_project,
            }
        )

        raise
