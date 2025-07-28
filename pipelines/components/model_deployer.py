"""Model deployer component for wine quality pipeline."""

from kfp.v2.dsl import Model, Input, Output, component
from constants import BASE_CONTAINER_IMAGE


# pylint: disable=too-many-arguments, too-many-locals
@component(
    packages_to_install=["google-cloud-aiplatform"], base_image=BASE_CONTAINER_IMAGE
)
def deploy_model(
    model_registry_name: Input[Model],
    endpoint: Output[Model],
    endpoint_display_name: str,
    project: str,
    region: str,
    machine_type: str,
    min_replica_count: int,
    max_replica_count: int,
):
    """Deploys a model to Vertex AI Endpoint."""
    # pylint: disable=import-outside-toplevel
    from google.cloud import aiplatform
    import logging

    aiplatform.init(project=project, location=region)
    model_name = model_registry_name.metadata.get("resource_name")
    if not model_name:
        logging.info("Model resource_name not found in metadata: %s", model_name)
        raise ValueError("ERROR: Model resource_name is missing from metadata.")
    try:
        model = aiplatform.Model(model_name)
        logging.info("Retrieved model : %s", model.resource_name)
    except Exception as e:
        print(f"ERROR: retrieving model: {e}")
        raise e
    deployed_model_display_name = f"{model_registry_name.metadata.get('display_name', 'model')}-deployed"  # pylint: disable=line-too-long
    try:
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_display_name}"',
            order_by="create_time desc",
            project=project,
            location=region,
        )
        if endpoints:
            endpoint_to_use = endpoints[0]
            logging.info("Using existing endpoint: %s", endpoint_to_use.resource_name)
        else:
            endpoint_to_use = aiplatform.Endpoint.create(
                display_name=endpoint_display_name, project=project, location=region
            )  # pylint: disable=line-too-long
            logging.info("Using new endpoint: %s", endpoint_to_use.resource_name)
        endpoint_to_use.deploy(
            model=model,
            deployed_model_display_name=deployed_model_display_name,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            traffic_split={"0": 100},
        )
        logging.info("Model deployed to endpoint: %s", endpoint_to_use.resource_name)
        endpoint.uri = endpoint_to_use.resource_name
    except Exception as e:
        logging.error("Model deployment model : %s", e)
        if "endpoint_to_use" in locals():
            endpoint.uri = endpoint_to_use.resource_name
        raise e
